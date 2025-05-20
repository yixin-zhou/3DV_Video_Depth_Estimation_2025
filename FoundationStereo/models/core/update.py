"""
更新模块 - 提供迭代式视差/光流更新的网络组件

这个模块实现了StereoAnyVideo模型中的关键更新组件，包括：
1. 基础和高级GRU单元，用于递归更新特征
2. 各种流头(FlowHead)，用于从特征生成视差/光流
3. 运动编码器，用于编码相关性和当前流估计
4. 时空注意力模块，用于增强特征的时间和空间一致性
5. 顶层更新块，协调整个更新过程
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from models.core.attention import LoFTREncoderLayer


def pool2x(x):
    """2倍下采样池化函数
    
    Args:
        x (torch.Tensor): 输入特征图
        
    Returns:
        torch.Tensor: 2倍下采样后的特征图
    """
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    """4倍下采样池化函数
    
    Args:
        x (torch.Tensor): 输入特征图
        
    Returns:
        torch.Tensor: 4倍下采样后的特征图
    """
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    """插值函数，将特征图调整为目标尺寸
    
    Args:
        x (torch.Tensor): 输入特征图
        dest (torch.Tensor): 目标特征图，提供目标尺寸
        
    Returns:
        torch.Tensor: 插值调整后的特征图
    """
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class FlowHead(nn.Module):
    """流头模块，用于从特征生成2D光流/视差
    
    一个简单的两层卷积网络，将特征映射为2通道的流场（x和y方向）
    
    Args:
        input_dim (int, optional): 输入特征维度. 默认: 128
        hidden_dim (int, optional): 隐藏层维度. 默认: 256
        output_dim (int, optional): 输出维度. 默认: 2 (x和y方向)
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, C, H, W]
            
        Returns:
            torch.Tensor: 预测的流场 [B, 2, H, W]
        """
        return self.conv2(self.relu(self.conv1(x)))


class FlowHead3D(nn.Module):
    """3D流头模块，用于从3D特征生成光流/视差
    
    与2D流头类似，但使用3D卷积处理时空特征
    
    Args:
        input_dim (int, optional): 输入特征维度. 默认: 128
        hidden_dim (int, optional): 隐藏层维度. 默认: 256
    """
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入3D特征 [B, C, T, H, W]
            
        Returns:
            torch.Tensor: 预测的3D流场 [B, 2, T, H, W]
        """
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    """卷积GRU单元
    
    实现标准的卷积GRU，用于在保持空间维度的同时递归更新特征
    
    Args:
        hidden_dim (int): 隐藏状态维度
        input_dim (int): 输入特征维度
        kernel_size (int, optional): 卷积核大小. 默认: 3
    """
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        """前向传播
        
        Args:
            h (torch.Tensor): 当前隐藏状态
            cz (torch.Tensor): 更新门偏置
            cr (torch.Tensor): 重置门偏置
            cq (torch.Tensor): 候选隐藏状态偏置
            *x_list: 输入特征列表，将被连接
            
        Returns:
            torch.Tensor: 更新后的隐藏状态
        """
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)  # 更新门
        r = torch.sigmoid(self.convr(hx) + cr)  # 重置门
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)  # 候选隐藏状态

        h = (1-z) * h + z * q  # 更新隐藏状态
        return h


class SepConvGRU(nn.Module):
    """分离卷积GRU单元
    
    使用水平和垂直分离卷积实现的GRU，可以在较低的计算成本下捕获较大的感受野
    
    Args:
        hidden_dim (int, optional): 隐藏状态维度. 默认: 128
        input_dim (int, optional): 输入特征维度. 默认: 192+128
    """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        # 水平方向的门控卷积
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        # 垂直方向的门控卷积
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        """前向传播
        
        先进行水平方向的GRU更新，再进行垂直方向的GRU更新
        
        Args:
            h (torch.Tensor): 当前隐藏状态
            *x: 输入特征，将被连接
            
        Returns:
            torch.Tensor: 更新后的隐藏状态
        """
        # 水平方向处理
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # 垂直方向处理
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    """基础运动编码器
    
    将相关性体积和当前流场编码为运动特征
    
    Args:
        cor_planes (int): 相关性平面数量
    """
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()

        # 相关性体积编码器
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        # 流场编码器
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        # 融合编码器
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """前向传播
        
        Args:
            flow (torch.Tensor): 当前流场估计 [B, 2, H, W]
            corr (torch.Tensor): 相关性体积 [B, cor_planes, H, W]
            
        Returns:
            torch.Tensor: 编码后的运动特征 [B, 128, H, W]，包括原始流场
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # 连接相关性和流特征
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        # 保留原始流场信息
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder3D(nn.Module):
    """3D基础运动编码器
    
    3D版本的运动编码器，用于处理时空数据
    
    Args:
        cor_planes (int): 相关性平面数量
    """
    def __init__(self, cor_planes):
        super(BasicMotionEncoder3D, self).__init__()

        # 使用3D卷积处理时空数据
        self.convc1 = nn.Conv3d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv3d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv3d(2, 128, 5, padding=2)
        self.convf2 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv = nn.Conv3d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """前向传播
        
        Args:
            flow (torch.Tensor): 当前3D流场估计 [B, 2, T, H, W]
            corr (torch.Tensor): 3D相关性体积 [B, cor_planes, T, H, W]
            
        Returns:
            torch.Tensor: 编码后的3D运动特征 [B, 128, T, H, W]
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU3D(nn.Module):
    """3D分离卷积GRU
    
    扩展至3D的分离卷积GRU，分别在三个维度上应用门控机制
    
    Args:
        hidden_dim (int, optional): 隐藏状态维度. 默认: 128
        input_dim (int, optional): 输入特征维度. 默认: 192+128
    """
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU3D, self).__init__()
        # 深度(Z)方向卷积
        self.convz1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convr1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        # 高度(Y)方向卷积
        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        # 时间(T)方向卷积
        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        """前向传播
        
        依次在Z(深度)、Y(高度)和T(时间)三个维度上应用GRU更新
        
        Args:
            h (torch.Tensor): 当前隐藏状态 [B, C, T, H, W]
            x (torch.Tensor): 输入特征 [B, D, T, H, W]
            
        Returns:
            torch.Tensor: 更新后的隐藏状态 [B, C, T, H, W]
        """
        # Z方向更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Y方向更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # T方向更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SKSepConvGRU3D(nn.Module):
    """SK分离卷积GRU3D
    
    带有选择性核(Selective Kernel)设计的3D分离卷积GRU，
    使用大核和小核的组合来增强感受野和特征提取能力
    
    Args:
        hidden_dim (int, optional): 隐藏状态维度. 默认: 128
        input_dim (int, optional): 输入特征维度. 默认: 192+128
    """
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SKSepConvGRU3D, self).__init__()
        # Z方向使用大核(15)+小核(5)的组合
        self.convz1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        # Y方向卷积
        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        # T方向卷积
        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        """前向传播
        
        依次在Z、Y和T三个维度上应用GRU更新，Z方向使用增强的选择性核设计
        
        Args:
            h (torch.Tensor): 当前隐藏状态 [B, C, T, H, W]
            x (torch.Tensor): 输入特征 [B, D, T, H, W]
            
        Returns:
            torch.Tensor: 更新后的隐藏状态 [B, C, T, H, W]
        """
        # Z方向更新，使用选择性核增强的卷积
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Y方向更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # T方向更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicUpdateBlock(nn.Module):
    """基本更新块
    
    实现单帧(2D)更新块，将当前状态和相关性特征更新为新的流场估计
    
    Args:
        hidden_dim (int): 隐藏状态维度
        cor_planes (int): 相关性平面数量
        mask_size (int, optional): 上采样掩码尺寸. 默认: 8
        attention_type (str, optional): 注意力类型. 默认: None
    """
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(BasicUpdateBlock, self).__init__()
        self.attention_type = attention_type
        # 根据注意力类型初始化相应的注意力模块
        if attention_type is not None:
            if "update_time" in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)

            if "update_space" in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)

        # 核心组件
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # 上采样掩码生成网络
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size ** 2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True, t=1):
        """前向传播
        
        Args:
            net (torch.Tensor): 当前隐藏状态
            inp (torch.Tensor): 上下文特征
            corr (torch.Tensor): 相关性特征
            flow (torch.Tensor): 当前流场估计
            upsample (bool, optional): 是否生成上采样掩码. 默认: True
            t (int, optional): 时间步数. 默认: 1
            
        Returns:
            tuple: 
                - net (torch.Tensor): 更新后的隐藏状态
                - mask (torch.Tensor): 上采样掩码
                - delta_flow (torch.Tensor): 流场更新量
        """
        # 编码运动特征
        motion_features = self.encoder(flow, corr)
        inp = torch.cat((inp, motion_features), dim=1)
        
        # 应用注意力机制
        if self.attention_type is not None:
            if "update_time" in self.attention_type:
                inp = self.time_attn(inp, T=t)
            if "update_space" in self.attention_type:
                inp = self.space_attn(inp, T=t)
                
        # GRU更新
        net = self.gru(net, inp)
        # 预测流场更新量
        delta_flow = self.flow_head(net)

        # 生成上采样掩码（缩放以平衡梯度）
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class Attention(nn.Module):
    """自注意力机制
    
    标准的多头自注意力实现
    
    Args:
        dim (int): 特征维度
        num_heads (int, optional): 注意力头数. 默认: 8
        qkv_bias (bool, optional): 是否使用QKV偏置. 默认: False
        qk_scale (float, optional): 缩放因子. 默认: None
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # 简化实现，使用同一投影矩阵生成QKV
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, N, C]
            
        Returns:
            torch.Tensor: 注意力处理后的特征 [B, N, C]
        """
        B, N, C = x.shape
        # 简化的QKV分离，实际上这里未真正分离QKV（使用同一特征）
        qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv, qkv, qkv

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 注意力加权
        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        return x


class TimeAttnBlock(nn.Module):
    """时间注意力块
    
    在时间维度上应用自注意力，增强时间一致性
    
    Args:
        dim (int, optional): 特征维度. 默认: 256
        num_heads (int, optional): 注意力头数. 默认: 8
    """
    def __init__(self, dim=256, num_heads=8):
        super(TimeAttnBlock, self).__init__()
        self.temporal_attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None)
        self.temporal_fc = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)

        # 初始化为零，确保刚开始不会改变特征
        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)

    def forward(self, x, T=1):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [(B*T), C, H, W]
            T (int, optional): 时间步数. 默认: 1
            
        Returns:
            torch.Tensor: 时间注意力处理后的特征 [(B*T), C, H, W]
        """
        _, _, h, w = x.shape

        # 重排列为时间序列形式 [(B*H*W), T, C]
        x = rearrange(x, "(b t) m h w -> (b h w) t m", h=h, w=w, t=T)
        # 应用自注意力
        res_temporal1 = self.temporal_attn(self.temporal_norm1(x))
        # 重排列并应用线性层
        res_temporal1 = rearrange(
            res_temporal1, "(b h w) t m -> b (h w t) m", h=h, w=w, t=T
        )
        res_temporal1 = self.temporal_fc(res_temporal1)
        # 重排列为原始形状
        res_temporal1 = rearrange(
            res_temporal1, " b (h w t) m -> b t m h w", h=h, w=w, t=T
        )
        x = rearrange(x, "(b h w) t m -> b t m h w", h=h, w=w, t=T)
        # 残差连接
        x = x + res_temporal1
        # 重排列回批次形式
        x = rearrange(x, "b t m h w -> (b t) m h w", h=h, w=w, t=T)
        return x


class SpaceAttnBlock(nn.Module):
    """空间注意力块
    
    在空间维度上应用自注意力，增强空间信息整合
    
    Args:
        dim (int, optional): 特征维度. 默认: 256
        num_heads (int, optional): 注意力头数. 默认: 8
    """
    def __init__(self, dim=256, num_heads=8):
        super(SpaceAttnBlock, self).__init__()
        # 使用LoFTR编码器层作为空间注意力机制
        self.encoder_layer = LoFTREncoderLayer(dim, nhead=num_heads, attention="linear")

    def forward(self, x, T=1):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [(B*T), C, H, W]
            T (int, optional): 时间步数. 默认: 1
            
        Returns:
            torch.Tensor: 空间注意力处理后的特征 [(B*T), C, H, W]
        """
        _, _, h, w = x.shape
        # 重排列为序列形式 [(B*T), (H*W), C]
        x = rearrange(x, "(b t) m h w -> (b t) (h w) m", h=h, w=w, t=T)
        # 应用空间注意力
        x = self.encoder_layer(x, x)
        # 重排列回原始形状
        x = rearrange(x, "(b t) (h w) m -> (b t) m h w", h=h, w=w, t=T)
        return x


class SequenceUpdateBlock3D(nn.Module):
    """序列更新块3D
    
    视频序列的3D更新块，整合时空信息进行视差/流场估计
    
    Args:
        hidden_dim (int): 隐藏状态维度
        cor_planes (int): 相关性平面数量
        mask_size (int, optional): 上采样掩码尺寸. 默认: 8
    """
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(SequenceUpdateBlock3D, self).__init__()

        # 核心组件
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        # 3D上采样掩码生成网络
        self.mask3d = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim + 128, (mask_size ** 2) * (3 * 3 * 3), 1, padding=0),
        )
        # 时空注意力模块
        self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
        self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)

    def forward(self, net, inp, corrs, flows, t):
        """前向传播
        
        Args:
            net (torch.Tensor): 当前隐藏状态 [(B*T), C, H, W]
            inp (torch.Tensor): 上下文特征 [(B*T), C, H, W]
            corrs (torch.Tensor): 相关性特征 [(B*T), C, H, W]
            flows (torch.Tensor): 当前流场估计 [(B*T), 2, H, W]
            t (int): 时间步数
            
        Returns:
            tuple:
                - net (torch.Tensor): 更新后的隐藏状态 [(B*T), C, H, W]
                - mask (torch.Tensor): 上采样掩码 [(B*T), C, H, W]
                - delta_flow (torch.Tensor): 流场更新量 [(B*T), 2, H, W]
        """
        # 编码运动特征
        motion_features = self.encoder(flows, corrs)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        # 应用时空注意力
        inp_tensor = self.time_attn(inp_tensor, T=t)
        inp_tensor = self.space_attn(inp_tensor, T=t)

        # 重排列为3D格式
        net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
        inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)

        # 3D GRU更新
        net = self.gru(net, inp_tensor)

        # 生成流场更新量
        delta_flow = self.flow_head(net)

        # 生成上采样掩码
        mask = 0.25 * self.mask3d(net)
        # 重排列回2D批次格式
        net = rearrange(net, " b c t h w -> (b t) c h w")
        mask = rearrange(mask, " b c t h w -> (b t) c h w")
        delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
        return net, mask, delta_flow
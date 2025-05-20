# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
基础子模块集合 (Basic Submodules Collection)
该模块包含FoundationStereo中使用的各种基础网络组件和函数。
涵盖了卷积模块、注意力机制、代价体构建、特征变换等多种组件，
这些组件共同构成了立体匹配系统的基础构建块。
"""

import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def _is_contiguous(tensor: torch.Tensor) -> bool:
    """
    检查张量是否连续存储
    
    在某些操作前确保张量内存连续，提高计算效率
    
    参数:
        tensor: 待检查的张量
        
    返回:
        布尔值，表示张量是否连续存储
    """
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    """
    2D特征图的层归一化
    
    专门为通道优先(NCHW)格式的2D特征图设计的层归一化:
    - 保持与常规卷积网络的兼容性
    - 在通道维度上进行归一化
    - 支持两种实现方式，根据张量是否连续选择
    
    与BatchNorm不同，LayerNorm对每个样本独立归一化，
    减少了对批量大小的依赖，并在训练和推理阶段保持一致行为。
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        层归一化前向传播
        
        参数:
            x: 输入特征图，形状为(B,C,H,W)
            
        返回:
            归一化后的特征图，形状与输入相同
        """
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x



class BasicConv(nn.Module):
    """
    基础卷积模块
    
    实现灵活的卷积操作，支持多种配置:
    - 2D或3D卷积/反卷积
    - 多种归一化方式(批归一化、实例归一化等)
    - 可选的ReLU激活
    
    作为网络中的基本构建块，用于特征提取和转换。
    支持不同维度和归一化需求，增强模型的灵活性和适应性。
    """
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm3d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm2d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        """
        基础卷积模块前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            经过卷积、归一化和激活处理后的特征图
        """
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv3dNormActReduced(nn.Module):
    """
    分解的3D卷积模块
    
    通过分解3D卷积为空间卷积和视差卷积，降低计算复杂度:
    - 首先在空间维度(H,W)进行2D卷积，保持视差维度不变
    - 然后在视差维度(D)进行1D卷积，处理视差信息
    
    这种设计显著减少了参数量和计算量，同时保持了对3D代价体的有效处理能力。
    特别适用于代价体过滤等需要3D上下文但计算量大的场景。
    """
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
          kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1,kernel_size,kernel_size), padding=(0, kernel_size//2, kernel_size//2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        分解3D卷积前向传播
        
        参数:
            x: 输入3D代价体，形状为(B,C,D,H,W)
            
        返回:
            经过空间和视差维度卷积处理的特征体
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x




class ResnetBasicBlock(nn.Module):
  """
  2D残差基本块
  
  实现ResNet风格的2D残差连接:
  - 包含两个带归一化的卷积层
  - 支持可选的下采样路径
  - 使用残差连接改善梯度流动
  
  残差连接允许网络更有效地学习特征的增量变化，
  而不是强制每层学习完整的特征表示，有助于构建更深层网络。
  """
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    """
    2D残差块前向传播
    
    参数:
        x: 输入特征图
        
    返回:
        经过残差连接处理的特征图
    """
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class ResnetBasicBlock3D(nn.Module):
  """
  3D残差基本块
  
  实现ResNet风格的3D残差连接:
  - 结构与2D残差块类似，但使用3D卷积
  - 适用于处理代价体等3D数据
  - 保持残差学习的优势，提升3D特征学习能力
  
  在代价体过滤等任务中，3D残差块有助于提取更复杂的特征模式，
  同时保持良好的梯度传播和训练稳定性。
  """
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    """
    3D残差块前向传播
    
    参数:
        x: 输入3D特征体
        
    返回:
        经过残差连接处理的3D特征体
    """
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = flash_attn_func(Q, K, V, window_size=window_size)  # Replace with actual FlashAttention function

        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)

        return output



class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src



class UpsampleConv(nn.Module):
    def __init__(self, C_in, C_out, is_3d=False, kernel_size=3, bias=True, stride=1, padding=1):
        super().__init__()
        self.is_3d = is_3d
        if is_3d:
          self.conv = nn.Conv3d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        else:
          self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        if self.is_3d:
          mode = 'trilinear'
        else:
          mode = 'bilinear'
        x = F.interpolate(x, size=None, scale_factor=2, align_corners=False, mode=mode)
        x = self.conv(x)
        return x



class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(out_channels*2, out_channels*mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    """
    分组相关性计算
    
    将特征通道分组并计算组内相关性:
    - 特征图按通道分为num_groups组
    - 每组内计算归一化特征的点积
    - 生成表示相似度的相关性图
    
    分组相关性比全通道相关性计算效率更高，同时保持足够的表示能力。
    通过在组内归一化，增强了特征匹配的鲁棒性。
    
    参数:
        fea1: 第一个特征图，通常是左图特征
        fea2: 第二个特征图，通常是右图特征
        num_groups: 分组数量
        
    返回:
        分组相关性特征，形状为(B,num_groups,H,W)
    """
    B, C, H, W = fea1.shape
    assert C % num_groups == 0, f"C:{C}, num_groups:{num_groups}"
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    with torch.cuda.amp.autocast(enabled=False):
      cost = (F.normalize(fea1.float(), dim=2) * F.normalize(fea2.float(), dim=2)).sum(dim=2)  #!NOTE Divide first for numerical stability
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """
    构建分组相关性代价体(GWC)
    
    通过相对偏移采样构建视差代价体:
    - 遍历可能的视差范围(0到maxdisp)
    - 对每个视差，计算左图特征与右图偏移特征的相关性
    - 组合成3D代价体用于后续视差估计
    
    GWC代价体相比简单连接代价体，能更有效地表达匹配关系，
    同时比全通道相关性代价体更加计算高效。
    
    参数:
        refimg_fea: 参考图像(左图)特征
        targetimg_fea: 目标图像(右图)特征
        maxdisp: 最大视差值
        num_groups: 分组数量
        stride: 视差步长
        
    返回:
        分组相关性代价体，形状为(B,num_groups,maxdisp,H,W)
    """
    """
    @refimg_fea: left image feature
    @targetimg_fea: right image feature
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    """
    构建连接代价体
    
    通过特征连接构建视差代价体:
    - 遍历可能的视差范围(0到maxdisp)
    - 对每个视差，直接连接左图特征和偏移的右图特征
    - 组合成3D代价体提供给后续处理
    
    连接代价体保留了原始特征的所有信息，允许网络自行学习匹配关系，
    同时也增加了内存消耗。与GWC代价体互补，提供不同的特征表示。
    
    参数:
        refimg_fea: 参考图像(左图)特征
        targetimg_fea: 目标图像(右图)特征
        maxdisp: 最大视差值
        
    返回:
        连接特征代价体，形状为(B,2C,maxdisp,H,W)
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume



def disparity_regression(x, maxdisp):
    """
    视差回归函数
    
    将视差概率体转换为视差估计:
    - 构建视差索引值(0到maxdisp-1)
    - 计算概率加权平均，得到每个像素的视差估计
    
    这是一种"软判决"方法，允许亚像素级的视差估计，
    比直接选择最大概率的"硬判决"方法更精确。
    
    参数:
        x: 视差概率体，形状为(B,maxdisp,H,W)
        maxdisp: 最大视差值
        
    返回:
        回归得到的视差图，形状为(B,1,H,W)
    """
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    """
    特征注意力模块
    
    将2D图像特征转换为3D代价体的通道注意力:
    - 压缩并投影图像特征到代价体通道空间
    - 使用sigmoid激活产生注意力权重
    - 对代价体进行逐通道加权
    
    这种设计将2D图像特征中的语义信息引入到3D代价体处理中，
    帮助代价体聚焦于更有意义的匹配特征。
    
    参数:
        cv_chan: 代价体通道数
        feat_chan: 图像特征通道数
    """
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1)
            )

    def forward(self, cv, feat):
        '''
        特征注意力前向传播
        
        参数:
            cv: 代价体(B,C,D,H,W)
            feat: 2D图像特征(B,C',H,W)
            
        返回:
            注意力增强的代价体，与输入形状相同
        '''
        att = torch.sigmoid(self.feat_att(feat))
        att = att.unsqueeze(2).repeat(1, 1, cv.shape[2], 1, 1)
        cv = cv * att
        return cv

def context_upsample(disp_low, up_weights):
    """
    上下文感知的视差上采样
    
    使用学习的上采样权重上采样低分辨率视差:
    - 基于九个上采样位置的权重分布
    - 相比简单的双线性插值提供更精确的边缘保持
    - 利用高分辨率特征指导上采样过程
    
    这种上采样方法能更好地保持物体边缘和细节，
    避免了简单插值导致的边缘模糊和混合视差。
    
    参数:
        disp_low: 低分辨率视差图，形状为(B,1,h,w)
        up_weights: 上采样权重，形状为(B,9,2h,2w)
        
    返回:
        上采样后的视差图，形状为(B,2h,2w)
    """
    N, _, H, W = disp_low.shape
    N, _, h, w = up_weights.shape
    assert h == 2*H and w == P2*W
    disp_k = torch.stack([disp_low, disp_low, disp_low, disp_low,
                           disp_low, disp_low, disp_low, disp_low,
                           disp_low], dim=2)
    disp_k = disp_k.view(N, 1, 9, H, W)
    disp_upsample = F.unfold(disp_k, kernel_size=(3, 3), stride=1, padding=1)
    disp_upsample = disp_upsample.reshape(N, 9, H, W)
    
    # upsample via interpolation and learned weights
    disp_upsample = F.interpolate(disp_upsample, scale_factor=2, mode='bilinear', align_corners=True)
    disp_upsample = torch.sum(disp_upsample * up_weights, dim=1)
    return disp_upsample



class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe
    # self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]



class CostVolumeDisparityAttention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)


  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B,C,D,H,W = x.shape
    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)

    return x



class ChannelAttentionEnhancement(nn.Module):
    """
    通道注意力增强模块
    
    实现基于压缩和激励(Squeeze-and-Excitation)的通道注意力:
    - 全局平均池化提取每个通道的统计信息
    - 通过瓶颈MLP学习通道间的依赖关系
    - 生成通道注意力权重进行自适应特征重标定
    
    通道注意力能有效增强信息丰富的通道并抑制不重要的通道，
    提高特征表示的效率和有效性。
    
    参数:
        in_planes: 输入特征通道数
        ratio: 瓶颈压缩比例，控制MLP中间层的大小
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        通道注意力前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            通道注意力权重，用于特征重标定
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    """
    空间注意力提取器
    
    生成突出显示重要空间区域的注意力图:
    - 提取通道维度上的统计信息(最大值和平均值)
    - 使用卷积层学习空间注意力权重
    - 生成空间注意力掩码以突出关键区域
    
    空间注意力能帮助模型聚焦于图像中的重要区域，
    如物体边缘和纹理丰富的区域，这对于精确的视差估计至关重要。
    
    参数:
        kernel_size: 卷积核大小，控制感受野范围
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        空间注意力前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            空间注意力掩码，突出显示重要区域
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm=='layer':
          self.norm = LayerNorm2d(dim, eps=1e-6)
        else:
          self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
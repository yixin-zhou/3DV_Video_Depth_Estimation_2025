"""
特征提取模块 - 提供用于深度估计和特征匹配的多种特征提取器

这个模块包含了三种关键的特征提取器：
1. BasicEncoder: 用于提取上下文和相关性特征的通用编码器
2. MultiBasicEncoder: 多尺度特征提取器（未在当前模型中使用）
3. DepthExtractor: 利用Video-Depth-Anything预训练模型提取深度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import importlib
import timm
from einops import rearrange


class ResidualBlock(nn.Module):
    """残差块，用于构建特征提取网络
    
    实现了带有残差连接的标准卷积块，支持多种规范化方法。
    
    Args:
        in_planes (int): 输入通道数
        planes (int): 输出通道数
        norm_fn (str, optional): 规范化方法，可选 "group", "batch", "instance", "none". 默认: "group"
        stride (int, optional): 卷积步长，用于下采样. 默认: 1
    """
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个3x3卷积，可能有下采样（stride > 1）
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        # 第二个3x3卷积，保持空间维度
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 确定每个组的通道数（用于分组规范化）
        num_groups = planes // 8

        # 根据规范化方法选择相应的层
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        # 残差连接的下采样路径，用于匹配空间和通道维度
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
        )

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, in_planes, H, W]
            
        Returns:
            torch.Tensor: 输出特征图 [B, planes, H/stride, W/stride]
        """
        # 主路径
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        # 残差路径
        x = self.downsample(x)

        # 残差连接和ReLU激活
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """基础特征编码器
    
    用于提取图像的上下文特征和匹配特征，使用残差块构建的编码器网络。
    在StereoAnyVideo中用作cnet和fnet特征提取器。
    
    Args:
        input_dim (int, optional): 输入通道数. 默认: 3 (RGB图像)
        output_dim (int, optional): 输出通道数. 默认: 128
        norm_fn (str, optional): 规范化方法. 默认: "batch"
        dropout (float, optional): 退出概率，用于正则化. 默认: 0.0
    """
    def __init__(self, input_dim=3, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        # 选择规范化方法
        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64, affine=False)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        # 初始下采样卷积
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 创建多个残差块层
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)  # 保持分辨率
        self.layer2 = self._make_layer(96, stride=2)  # 降采样到原始分辨率的1/4
        self.layer3 = self._make_layer(128, stride=1) # 保持分辨率

        # 最终输出卷积，调整通道数
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # 可选的dropout层
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """创建由两个残差块组成的层
        
        Args:
            dim (int): 输出通道数
            stride (int, optional): 第一个残差块的步长. 默认: 1
            
        Returns:
            nn.Sequential: 包含两个残差块的序列模块
        """
        # 第一个残差块可能有下采样（stride > 1）
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        # 第二个残差块保持空间维度
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor or tuple/list): 输入图像/特征图 [B, C, H, W] 或者列表/元组
            
        Returns:
            torch.Tensor or tuple: 输出特征 [B, output_dim, H/4, W/4] 或者分割的元组
        """
        # 如果输入是列表或元组，合并批次维度（用于处理成对图像）
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # 主干网络前向传播
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        # 可选的dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # 如果输入是列表，将输出分割回列表
        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x


class MultiBasicEncoder(nn.Module):
    """多尺度特征编码器
    
    支持提取多个尺度的特征，可以输出1/8、1/16和1/32分辨率的特征图。
    在当前StereoAnyVideo实现中未使用，但保留为扩展功能。
    
    Args:
        output_dim (list, optional): 各尺度的输出通道数列表. 默认: [128]
        norm_fn (str, optional): 规范化方法. 默认: 'batch'
        dropout (float, optional): 退出概率. 默认: 0.0
        downsample (int, optional): 下采样级别控制. 默认: 3
    """
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        # 选择规范化方法
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # 初始卷积层，步长根据downsample参数调整
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 创建多个残差块层，步长根据downsample参数调整
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        # 1/8尺度的输出层
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        # 1/16尺度的输出层
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)

        # 1/32尺度的输出层
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)
        self.outputs32 = nn.ModuleList(output_list)

        # 可选的dropout层
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """创建由两个残差块组成的层
        
        Args:
            dim (int): 输出通道数
            stride (int, optional): 第一个残差块的步长. 默认: 1
            
        Returns:
            nn.Sequential: 包含两个残差块的序列模块
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像/特征图 [B, C, H, W]
            dual_inp (bool, optional): 是否有双输入. 默认: False
            num_layers (int, optional): 要使用的编码器层数. 默认: 3
            
        Returns:
            tuple: 多尺度特征的元组，取决于num_layers参数
        """
        # 主干网络前向传播
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 处理双输入情况
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        # 1/8尺度输出
        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        # 1/16尺度特征和输出
        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]
        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        # 1/32尺度特征和输出
        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)


class DepthExtractor(nn.Module):
    """深度特征提取器
    
    使用预训练的Video-Depth-Anything模型提取视频序列的深度特征。
    这是StereoAnyVideo模型的关键组件，提供强大的单目深度先验。
    """
    def __init__(self):
        """初始化DepthExtractor
        
        加载预训练的Video-Depth-Anything模型并设置额外的下采样卷积。
        """
        super(DepthExtractor, self).__init__()

        # 设置Video-Depth-Anything模型路径和导入模块
        thirdparty_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./models/Video-Depth-Anything"))
        sys.path.append(thirdparty_path)
        videodepthanything_ppl = importlib.import_module(
            "stereoanyvideo.models.Video-Depth-Anything.video_depth_anything.video_depth"
        )
        
        # 模型配置，支持两种编码器尺寸
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        encoder = 'vits'  # 选择vits编码器（小型）

        # 实例化Video-Depth-Anything模型并加载预训练权重
        self.depthanything = videodepthanything_ppl.VideoDepthAnything(**model_configs[encoder])
        self.depthanything.load_state_dict(torch.load(f'./models/Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth'))
        self.depthanything.eval()
        
        # 额外的4x4卷积，用于进一步下采样和处理特征
        self.conv = nn.Conv2d(32, 32, kernel_size=4, stride=4)

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入视频序列 [B, T, C, H, W]
            
        Returns:
            torch.Tensor: 提取的深度特征 [B, T, 32, H//4, W//4]
        """
        # 保存原始高度和宽度
        B, T, C, orig_h, orig_w = x.shape

        # 计算能被14整除的新高度和宽度（ViT模型需要）
        new_h = (orig_h // 14) * 14
        new_w = (orig_w // 14) * 14

        # 调整输入尺寸以适配ViT模型要求
        resized_input = F.interpolate(
            x.flatten(0, 1),  # [B*T, C, H, W]
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).unflatten(0, (B, T))  # 恢复 [B, T, C, new_h, new_w]

        # 通过Video-Depth-Anything模型提取深度特征
        depth_features_resized = self.depthanything(resized_input).contiguous()

        # 将深度特征调整回原始分辨率
        depth_features = F.interpolate(
            depth_features_resized,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )

        # 应用4x4卷积下采样深度特征
        depth_features = self.conv(depth_features).unflatten(0, (B, T))

        return depth_features

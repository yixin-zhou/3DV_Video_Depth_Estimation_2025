"""
基础构建模块 (Basic Building Blocks)
该模块定义了DepthAnything模型的基础构建组件，包括特征提取、特征融合和残差块。
这些组件被用于构建DPT (Dense Prediction Transformer) 架构，
实现从视觉Transformer特征到密集深度图的转换。
"""

import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """
    创建特征转换网络
    
    将ViT的多层次特征转换为统一通道数:
    - 为每个层级的特征创建独立的卷积层
    - 可选地扩展更深层的通道数
    - 统一所有特征的通道数格式，便于后续融合
    
    参数:
        in_shape: 输入特征的通道数列表
        out_shape: 输出特征的基础通道数
        groups: 卷积分组数，用于分组卷积
        expand: 是否扩展深层特征的通道数
        
    返回:
        包含特征转换层的模块
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """
    残差卷积单元
    
    实现带有残差连接的双卷积结构:
    - 包含两个3x3卷积层
    - 可选的批量归一化
    - 残差连接跳过主路径
    
    此模块提高了网络的收敛性能和特征表示能力，
    能够学习残差特征而不改变输入维度。
    """

    def __init__(self, features, activation, bn):
        """
        初始化残差卷积单元
        
        参数:
            features: 特征通道数
            activation: 激活函数
            bn: 是否使用批量归一化
        """
        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图
            
        返回:
            经过残差连接处理的特征图
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    特征融合块
    
    融合不同分辨率的特征:
    - 处理不同层级的特征，如浅层和深层特征
    - 使用残差卷积单元增强特征表示
    - 通过上采样实现特征融合
    
    在DPT架构中，该模块负责自顶向下融合多尺度特征，
    逐步恢复空间分辨率并保留语义信息。
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """
        初始化特征融合块
        
        参数:
            features: 特征通道数
            activation: 激活函数
            deconv: 是否使用转置卷积而非插值
            bn: 是否使用批量归一化
            expand: 是否扩展特征通道数
            align_corners: 插值时是否对齐角点
            size: 指定输出大小
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """
        前向传播
        
        参数:
            *xs: 输入特征图列表，通常包含主特征和跳跃连接特征
            size: 可选的输出尺寸
            
        返回:
            融合后的特征图
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

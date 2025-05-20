# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
特征提取模块 (Feature Extraction Module)
该模块负责从输入图像中提取多尺度特征表示，为立体匹配提供基础特征。
整合了多种特征提取机制，包括CNN编码器和基于DepthAnything的特征提取器。
主要包含以下组件：
- ResidualBlock: 构建特征提取器的基本残差单元
- MultiBasicEncoder: 基础多尺度特征提取器
- ContextNetDino: 基于DINOv2的上下文特征提取网络
- DepthAnythingFeature: 利用DepthAnything模型提取深度敏感特征
- Feature: 整合多种特征提取能力的主要特征提取模块
"""

import torch,logging,os,sys,urllib,warnings
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import *
from Utils import *
import timm


class ResidualBlock(nn.Module):
    """
    残差块实现，用于构建特征提取网络的基本单元
    
    支持多种归一化方式：
    - group: 组归一化，平衡计算效率和性能
    - batch: 批归一化，适用于较大批次
    - instance: 实例归一化，减少样本间的风格差异
    - layer: 层归一化，对每个样本独立归一化
    - none: 无归一化，用于特殊情况
    
    每个残差块包含两个卷积层和可选的下采样路径，通过残差连接提高梯度传播效率。
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn=='layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        """
        残差块的前向传播
        
        实现残差连接的"捷径"，让梯度能够更有效地流动，缓解深层网络的梯度消失问题
        
        参数:
            x: 输入特征图
            
        返回:
            添加残差连接后的特征图
        """
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class MultiBasicEncoder(nn.Module):
    """
    多尺度特征提取的基础编码器
    
    构建包含多个分辨率级别的特征金字塔:
    - 1/4分辨率特征 (outputs04)
    - 1/8分辨率特征 (outputs08)
    - 1/16分辨率特征 (outputs16)
    
    特点:
    - 由多个残差块堆叠构成
    - 支持可配置的下采样级别
    - 可选的dropout机制增强泛化能力
    - 灵活的输出维度配置
    - 支持不同级别的特征输出，可根据需要提取不同深度的特征
    
    通过_make_layer方法构建残差块序列，形成网络的主干部分。
    forward方法中可选择性返回不同深度的特征层次。
    """
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # 第一个卷积层，根据downsample参数决定是否下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 构建编码器的多个层次
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        # 不同尺度的输出层
        output_list = []

        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        # 可选的dropout层
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # 初始化各层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        构建残差层
        
        参数:
            dim: 输出通道数
            stride: 卷积步长
            
        返回:
            包含两个残差块的Sequential模块
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):
        """
        网络前向传播，提取多尺度特征
        
        参数:
            x: 输入图像或特征
            dual_inp: 是否处理双输入（如立体图像对）
            num_layers: 输出特征的层数，可为1、2或3
            
        返回:
            不同尺度的特征列表，根据num_layers参数决定返回层数
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        # 1/4分辨率特征
        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        # 1/8分辨率特征
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        # 1/16分辨率特征
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)



class ContextNetDino(MultiBasicEncoder):
    """
    基于DINOv2的上下文特征提取网络
    
    扩展自MultiBasicEncoder，整合ViT特征和CNN特征:
    - 继承基础编码器的特征提取能力
    - 增强对语义信息的理解
    - 支持多尺度特征融合
    
    主要用途:
    - 提取上下文信息用于视差更新
    - 为GRU更新模块提供精细的特征表示
    - 生成不同分辨率的特征表示
    
    核心机制:
    - 在每个尺度上集成ViT特征
    - 通过conv2层融合多模态特征
    - 保持与原始MultiBasicEncoder的层次结构
    - 支持不同层次的特征输出
    
    由于不使用传统的继承方式，需要自行实现特征提取的各个组件。
    """
    def __init__(self, args, output_dim=[128], norm_fn='batch', downsample=2):
        nn.Module.__init__(self)
        self.args = args
        self.patch_size = 14
        self.image_size = 518
        self.vit_feat_dim = 384
        code_dir = os.path.dirname(os.path.realpath(__file__))

        self.out_dims = output_dim

        self.norm_fn = norm_fn

        # 选择归一化方法
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn=='layer':
            self.norm1 = LayerNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 构建多层特征提取网络
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.down = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0),
          nn.BatchNorm2d(128),
        )
        
        # 获取ViT特征维度并构建融合层
        vit_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2
        self.conv2 = BasicConv(128+vit_dim, 128, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(256)

        # 构建三种尺度的输出层
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

    def _make_layer(self, dim, stride=1):
        """
        构建残差层
        
        参数:
            dim: 输出通道数
            stride: 卷积步长
            
        返回:
            包含两个残差块的Sequential模块
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x_in, vit_feat, dual_inp=False, num_layers=3):
        """
        网络前向传播，整合CNN特征和ViT特征
        
        参数:
            x_in: 输入图像
            vit_feat: ViT特征，来自DepthAnything模型
            dual_inp: 是否双输入处理
            num_layers: 输出特征的层数
            
        返回:
            多尺度特征元组，包含不同分辨率的特征
        """
        B,C,H,W = x_in.shape
        x = self.conv1(x_in)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 计算合适的图像尺寸，兼容ViT模型
        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        
        # 融合CNN特征和ViT特征
        x = torch.cat([x, vit_feat], dim=1)
        x = self.conv2(x)
        outputs04 = [f(x) for f in self.outputs04]

        # 生成1/8分辨率特征
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        # 生成1/16分辨率特征
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16)


class DepthAnythingFeature(nn.Module):
    """
    DepthAnything特征提取器
    
    利用预训练的深度估计模型提取特征:
    - 支持多种ViT模型大小(small, base, large)
    - 提取对深度敏感的特征表示
    - 作为整体特征提取的骨干网络
    
    模型配置:
    - vitl: ViT-Large, 高精度但计算量大，用于高质量的深度估计
    - vitb: ViT-Base, 平衡精度与效率
    - vits: ViT-Small, 轻量级模型，适合资源受限场景
    
    特点:
    - 利用预训练的DepthAnything模型，无需从头训练
    - 提取多级特征，每级对应不同的语义层次
    - 通过中间层索引获取不同深度的特征
    - 生成多尺度的特征表示，用于不同分辨率的视差估计
    """
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    def __init__(self, encoder='vits'):
        super().__init__()
        from depth_anything.dpt import DepthAnything
        self.encoder = encoder
        depth_anything = DepthAnything(self.model_configs[encoder])
        self.depth_anything = depth_anything

        # 定义不同模型大小的中间层索引
        self.intermediate_layer_idx = {   #!NOTE For V2
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }


    def forward(self, x):
        """
        前向传播提取DepthAnything特征
        
        参数:
            x: (B,C,H,W) 输入图像
            
        返回:
            包含多个特征层和视差预测的字典:
            - out: 输出特征
            - path_1到path_4: 不同尺度的特征路径
            - features: 原始ViT特征
            - disp: 预测的视差
        """
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)


        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(features, patch_h, patch_w, return_intermediate=True)

        return {'out':out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4, 'features':features, 'disp':disp}  # path_1 is 1/2; path_2 is 1/4


class Feature(nn.Module):
    """
    FoundationStereo的主要特征提取模块
    
    集成多种特征提取能力:
    - EdgeNext骨干网络提取CNN特征
    - DepthAnything提取深度敏感的ViT特征
    - 多级特征融合形成强大的表示
    
    特征处理流程:
    1. 输入左右图像
    2. 通过EdgeNext提取多尺度CNN特征
    3. 通过DepthAnything提取ViT特征
    4. 特征解码并上采样
    5. 融合CNN与ViT特征
    
    网络组成:
    - stem和stages: EdgeNext主干网络
    - dino: DepthAnything特征提取器
    - deconv层: 特征上采样和融合
    - conv4: 最终特征精炼
    
    输出是多尺度特征列表[x4, x8, x16, x32]和ViT特征，用于后续代价体构建和上下文处理。
    """
    def __init__(self, args):
        super(Feature, self).__init__()
        self.args = args
        
        # 加载预训练的EdgeNext模型作为基础特征提取器
        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        
        # 初始化DepthAnything特征提取器并冻结参数
        self.dino = DepthAnythingFeature(encoder=self.args.vit_size)
        self.dino = freeze_model(self.dino)
        vit_feat_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2

        # 特征解码和上采样网络
        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        
        # 最终特征融合层
        self.conv4 = nn.Sequential(
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
        )

        self.patch_size = 14
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]]

    def forward(self, x):
        """
        前向传播，提取并融合CNN和ViT特征
        
        参数:
            x: 输入图像，可以是单张图像或拼接的左右图像对
            
        返回:
            多尺度特征列表[x4, x8, x16, x32]和ViT特征
        """
        B,C,H,W = x.shape
        
        # 计算适合ViT处理的图像尺寸
        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H,W, divider=divider, max_H=1344, max_W=1344)
        x_in_ = F.interpolate(x, size=(H_resize, W_resize), mode='bicubic', align_corners=False)
        
        # 提取DepthAnything特征
        self.dino = self.dino.eval()
        with torch.no_grad():
          output = self.dino(x_in_)
        vit_feat = output['out']
        vit_feat = F.interpolate(vit_feat, size=(H//4,W//4), mode='bilinear', align_corners=True)
        
        # 提取EdgeNext特征
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)

        # 特征上采样和融合
        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.conv4(x4)
        
        return [x4, x8, x16, x32], vit_feat



# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
FoundationStereo核心实现模块
这是FoundationStereo立体匹配模型的主要实现文件，整合所有子模块，
实现从立体图像对到深度/视差图的端到端处理流程。
主要包括特征提取、代价体构建、视差估计和细化优化等关键步骤。
"""

import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.update import *
from core.extractor import *
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.utils.utils import *
from Utils import *
import time,huggingface_hub

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from einops import rearrange
import collections
from collections import defaultdict
from itertools import repeat
import unfoldNd

from models.core.update import SequenceUpdateBlock3D
from models.core.extractor import BasicEncoder, MultiBasicEncoder, DepthExtractor
from models.core.corr import AAPC
from models.core.utils.utils import InputPadder, interp


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def _ntuple(n):
    """将单个值或可迭代对象转换为n元组
    
    Args:
        n (int): 元组长度
        
    Returns:
        function: 转换函数
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    """检查值是否存在（不为None）
    
    Args:
        val: 待检查的值
        
    Returns:
        bool: 如果值不为None则为True
    """
    return val is not None


def default(val, d):
    """返回值如果存在，否则返回默认值
    
    Args:
        val: 首选值
        d: 默认值
        
    Returns:
        val如果存在，否则为d
    """
    return val if exists(val) else d


# 生成2元组的辅助函数
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    """多层感知机模块
    
    用于处理相关性特征的MLP模块，可选择使用卷积或线性层实现。
    
    Args:
        in_features (int): 输入特征维度
        hidden_features (int, optional): 隐藏层特征维度. 默认: None (与in_features相同)
        out_features (int, optional): 输出特征维度. 默认: None (与in_features相同)
        act_layer (nn.Module, optional): 激活函数. 默认: nn.GELU
        norm_layer (nn.Module, optional): 规范化层. 默认: None
        bias (bool or tuple, optional): 是否使用偏置. 默认: True
        drop (float or tuple, optional): 丢弃率. 默认: 0.0
        use_conv (bool, optional): 是否使用卷积代替线性层. 默认: False
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        # 根据use_conv参数决定使用卷积或线性层
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        # 第一层转换
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        # 可选的规范化层
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        # 第二层转换
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征
            
        Returns:
            torch.Tensor: 处理后的特征
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def normalize_image(img):
    '''
    图像归一化函数
    
    将RGB图像从0-255范围转换为标准化的网络输入格式。
    使用ImageNet预训练模型标准的均值和标准差进行归一化处理。
    
    参数:
        img: (B,C,H,W) 范围为0-255的RGB图像张量
        
    返回:
        归一化后的图像张量，适合模型输入，范围大约在[-2,2]
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    """
    沙漏网络结构，用于代价体的过滤和增强
    
    实现编码器-解码器架构，用于处理视差代价体:
    - 编码阶段: 三层下采样，扩大感受野和增加通道数
    - 解码阶段: 三层上采样，恢复空间分辨率
    - 跳跃连接: 连接对应的编码和解码层
    
    网络结构:
    - conv1, conv2, conv3: 三级下采样编码器
    - conv3_up, conv2_up, conv1_up: 三级上采样解码器
    - agg_0, agg_1: 特征聚合模块
    - atts: 视差注意力机制
    - feature_att系列: 特征注意力增强
    
    通过多级卷积操作和特征增强，沙漏网络能够有效滤除匹配噪声，
    增强正确匹配的置信度，生成更精确的代价体表示。
    """
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        
        # 第一层下采样编码器，将输入通道扩展为2倍
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        # 第二层下采样编码器，将通道扩展为4倍
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        # 第三层下采样编码器，将通道扩展为6倍
        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))

        # 第一层上采样解码器，通道恢复为4倍
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        # 第二层上采样解码器，通道恢复为2倍
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        # 第三层上采样解码器，通道恢复为原始通道数
        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        
        # 输出精细化模块
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        # 特征聚合模块，整合从编码器跳跃连接过来的特征
        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        
        # 视差维度上的注意力机制，增强代价体表示
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        
        # 特征下采样模块，用于多分辨率处理
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        # 特征注意力增强模块，用于融合来自特征提取器的多尺度特征
        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        """
        沙漏网络前向传播
        
        参数:
            x: 输入代价体
            features: 来自特征提取器的多尺度特征列表
            
        返回:
            过滤后的代价体
        """
        # 下采样编码阶段，并通过feature_att融合不同尺度的特征
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        # 上采样解码阶段，带有跳跃连接
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)  # 跳跃连接
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)  # 跳跃连接
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        # 恢复到原始分辨率
        conv = self.conv1_up(conv1)
        
        # 视差注意力处理
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        
        # 残差连接
        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    FoundationStereo主模型实现
    
    集成所有子模块，实现完整的立体匹配流程:
    1. 特征提取：从左右图像提取特征
    2. 代价体构建：基于提取的特征构建匹配代价体
    3. 代价体过滤：通过沙漏网络增强代价体表示
    4. 视差估计：初始视差估计和迭代细化
    5. 视差上采样：生成全分辨率视差输出
    
    核心组件:
    - feature: 特征提取网络，基于EdgeNext和DepthAnything
    - cnet: 上下文网络，提取上下文特征用于视差更新
    - update_block: GRU更新模块，迭代优化视差估计
    - cost_agg: 代价体聚合，基于沙漏网络
    - sam/cam: 空间和通道注意力模块
    - corr_stem/corr_feature_att: 代价体特征处理
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 定义上下文特征维度和代价体参数
        context_dims = args.hidden_dims
        self.cv_group = 8  # 组分组相关性体的组数
        volume_dim = 28    # 代价体通道数

        # 初始化上下文网络，用于GRU更新
        self.cnet = ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        
        # 视差更新模块，基于GRU结构迭代更新视差
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        
        # 空间和通道注意力模块，增强特征表示
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        # 上下文特征转换卷积
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        # 特征提取网络
        self.feature = Feature(args)
        
        # 特征投影层，用于代价体构建
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        # 图像下采样网络，生成1/2和1/4分辨率的特征
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        # 视差上采样网络
        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )

        # 代价体处理模块
        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        
        # 代价体特征注意力
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        
        # 代价体聚合网络，基于沙漏架构
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        
        # 视差分类器，将代价体转换为概率体
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        # 初始化几何编码相关参数
        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx

        # # 相关性特征处理MLP，将4*9*9维度的相关性特征转换为128维度,dqr
        # # self.corr_mlp = Mlp(in_features=4 * 9 * 9, hidden_features=256, out_features=128)
        # self.corr_mlp = Mlp(in_features=28 * 104, hidden_features=256, out_features=128)
        # # 序列更新块，用于迭代优化视差估计,dqr
        # self.update_block = SequenceUpdateBlock3D(hidden_dim=self.args.hidden_dims[0], cor_planes=128, mask_size=4)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        """ 
        模型前向传播主函数
        
        完整的立体匹配处理流程:
        1. 图像归一化和特征提取
        2. 构建代价体(GWC和连接体)
        3. 代价体过滤和初始视差估计
        4. 提取上下文特征
        5. 几何编码和迭代GRU更新
        6. 视差上采样和后处理
        
        参数:
            image1: 左图像，形状为[B,t,3,H,W]
            image2: 右图像，形状为[B,t,3,H,W]
            iters: GRU更新迭代次数，默认12
            flow_init: 可选的初始视差，默认为None
            test_mode: 测试模式标志，影响中间结果输出
            low_memory: 低内存模式，减少内存使用但可能降低速度
            init_disp: 可选的外部提供初始视差
            
        返回:
            如果test_mode为True，返回最终视差预测
            否则返回(初始视差, 所有视差预测列表)
        """
        B, T, c, h, w = image1.shape
        low_memory = low_memory or (self.args.get('low_memory', False))
        image1 = image1.flatten(0, 1)  # [B*T, C, H, W]
        image2 = image2.flatten(0, 1)
        B = B * T
        # torch.set_grad_enabled(False)
        
        # 图像归一化
        image1 = normalize_image(image1)
        image2 = normalize_image(image2)
        
        with autocast(enabled=self.args.mixed_precision):
            # 特征提取，处理左右图像
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]  # 只取左图像的ViT特征
            
            # 分离左右图像特征
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            
            # 生成1/2分辨率的左图特征，用于后续上采样
            stem_2x = self.stem_2(image1)

            # 构建组分组相关性(GWC)代价体
            gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
            
            # 构建连接代价体
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
            del left_tmp, right_tmp
            
            # 合并两种代价体并进行处理
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left)

            # 从代价体生成初始视差估计
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  #(B, max_disp, H, W)
            if init_disp is None:
                init_disp = disparity_regression(prob, self.args.max_disp//4)  # 视差回归，加权平均

            # 提取上下文特征，用于GRU更新
            cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list(cnet_list)
            cnet_dqr_used = cnet_list[0][0]  # 1/4分辨率的上下文特征
            net = torch.tanh(cnet_dqr_used)  # 作为GRU的隐藏状态
            inp = torch.relu(cnet_dqr_used)  # 作为GRU的输入


            # 级联细化策略 (1/16 + 1/8 + 1/4)
            flow = None  # 当前分辨率的流场
            flow_up = None  # 上采样后的流场
            flow_init = init_disp.cuda()
            # 调整尺度以匹配当前特征图
            scale = cnet_dqr_used.shape[2] / flow_init.shape[2]
            flow = scale * interp(flow_init, size=(cnet_dqr_used.shape[2], cnet_dqr_used.shape[3]))
            
            flow_hori = torch.zeros_like(flow)
            flow = torch.cat([flow, flow_hori], dim=1)
        
            comb_volume = comb_volume.flatten(1, 2).float()
            
            return flow, comb_volume, net, inp

class OurStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        self.foundation_stereo = FoundationStereo(args)
        
        # 冻结 foundation_stereo 的所有参数
        for param in self.foundation_stereo.parameters():
            param.requires_grad = False    
        
        # 相关性特征处理MLP，将4*9*9维度的相关性特征转换为128维度,dqr
        # self.corr_mlp = Mlp(in_features=4 * 9 * 9, hidden_features=256, out_features=128)
        self.corr_mlp = Mlp(in_features=28 * 104, hidden_features=256, out_features=128)
        # 序列更新块，用于迭代优化视差估计,dqr
        self.update_block = SequenceUpdateBlock3D(hidden_dim=self.args.hidden_dims[0], cor_planes=128, mask_size=4)
    
    def convex_upsample_3D(self, flow, mask, b, T, rate=4):
        """使用卷积组合上采样3D流场
        
        对于视频序列，在保持时间维度的同时上采样空间维度。
        使用unfoldNd库进行3D卷积组合。
        
        Args:
            flow (torch.Tensor): 输入流场 [N*T, 2, H/rate, W/rate]
            mask (torch.Tensor): 用于上采样的掩码 [N*T, rate*rate*27, H/rate, W/rate]
            b (int): 批次大小
            T (int): 时间步长
            rate (int, optional): 上采样率. 默认: 4
            
        Returns:
            torch.Tensor: 上采样后的流场 [N*T, 2, H, W]
        """
        # 重排列批次和时间维度
        flow = rearrange(flow, "(b t) c h w -> b c t h w", b=b, t=T)
        mask = rearrange(mask, "(b t) c h w -> b c t h w", b=b, t=T)

        N, _, T, H, W = flow.shape

        # 将掩码重塑为适合3D卷积组合的形状
        mask = mask.view(N, 1, 27, 1, rate, rate, T, H, W)  # (N, 1, 27, rate, rate, rate, T, H, W) if upsample T
        # 在第3维上应用softmax，确保27个位置的权重和为1
        mask = torch.softmax(mask, dim=2)

        # 使用unfoldNd库执行3D滑动窗口操作
        upsample = unfoldNd.UnfoldNd([3, 3, 3], padding=1)
        # 将流场放大rate倍
        flow_upsampled = upsample(rate * flow)
        # 重塑为[N, 2, 27, 1, 1, 1, T, H, W]
        flow_upsampled = flow_upsampled.view(N, 2, 27, 1, 1, 1, T, H, W)
        # 使用掩码加权求和
        flow_upsampled = torch.sum(mask * flow_upsampled, dim=2)
        # 调整维度顺序
        flow_upsampled = flow_upsampled.permute(0, 1, 5, 2, 6, 3, 7, 4)
        # 重塑为[N, 2, T, rate*H, rate*W]
        flow_upsampled = flow_upsampled.reshape(N, 2, T, rate * H,
                                                rate * W)  # (N, 2, rate*T, rate*H, rate*W) if upsample T
        # 重排列回[N*T, 2, rate*H, rate*W]
        up_flow = rearrange(flow_upsampled, "b c t h w -> (b t) c h w")

        return up_flow
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        
        B, T, c, h, w = image1.shape
        
        flow, comb_volume, net, inp = self.foundation_stereo.forward(image1, image2, iters, flow_init, test_mode, low_memory, init_disp)
        
        flow_predictions = []  # 存储所有尺度所有迭代的预测
        
        # 1/4分辨率迭代（最终尺度）
        for itr in range(iters):

            flow = flow.detach()
            print("flow:", flow.shape)
            # out_corrs = self.corr_mlp(comb_volume.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            out_corrs = self.corr_mlp(comb_volume.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow, t=T)
            # delta_flow.detach()
            # out_corrs.detach()
            # up_mask.detach()
                
            flow = flow + delta_flow
            flow_up = self.convex_upsample_3D(flow, up_mask, B, T, rate=4)  # [b*T, 2, h, w]
            # flow_up.detach()
            flow_predictions.append(flow_up[:, :1])  # 只保留水平分量（视差）
            
        print("flow_predictions:", len(flow_predictions))
        for i in flow_predictions:
            print(i.shape)
        
        # 整理所有预测结果
        predictions = torch.stack(flow_predictions)  # [num_predictions, b*T, 1, h, w]
        # 重排列为[num_predictions, T, b, 1, h, w]以便于处理时间维度
        predictions = rearrange(predictions, "d (b t) c h w -> b d t c h w", b=B, t=T)
        predictions = predictions.squeeze(3) # [b, num_predictions, T, h, w]
        # 获取最终预测
        flow_up = predictions[:,-1]  # [b, T, h, w]

        if test_mode:
            return flow_up  # 测试模式只返回最终预测

        return predictions  # 训练模式返回所有预测

if __name__ == '__main__':
    
    import os,sys
    code_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(f'{code_dir}/../')
    from omegaconf import OmegaConf
    from core.utils.utils import InputPadder
    from Utils import *
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    path = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/cfg.yaml"
    cfg = OmegaConf.load(path)
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    # for k in args.__dict__:
    #     cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    args.valid_iters = 32
    
    model = OurStereo(args)
    model.cuda()
    ckpt_dir = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
    ckpt = torch.load(ckpt_dir, weights_only=False)
    model.foundation_stereo.load_state_dict(ckpt['model'])
    print("Loading pretrained model done.")
    
    B = 1
    T = 2
    C = 3
    H, W =  512, 256
    
    img0, img1 = torch.randn(B, T, C, H, W).half().cuda(), torch.randn(B, T, C, H, W).half().cuda()
    
    model.eval()
    x = model.forward(img0, img1, iters=10, test_mode=True)
    
    print(x.shape)
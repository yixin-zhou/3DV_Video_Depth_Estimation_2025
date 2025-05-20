# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
视差更新模块 (Disparity Update Module)
该模块负责迭代更新和细化视差估计，是FoundationStereo的核心组件之一。
主要通过GRU（门控循环单元）结构和多层次特征融合实现视差的逐步优化。
包含多种GRU变体和视差头网络，支持选择性更新机制以提高效率和精度。
"""

import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import *
from core.extractor import *

class DispHead(nn.Module):
    """
    视差头网络
    
    从特征图预测视差增量的网络结构:
    - 利用EdgeNext编码器增强特征表示
    - 通过多层卷积处理输入特征
    - 最终产生视差增量预测
    
    主要用于GRU更新模块中，预测当前迭代步骤的视差调整量。
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
          nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        """
        前向传播，预测视差增量
        
        参数:
            x: 输入特征图
            
        返回:
            视差增量预测
        """
        return self.conv(x)

class ConvGRU(nn.Module):
    """
    卷积GRU单元
    
    传统GRU的卷积变体，用于处理特征图序列:
    - 三个主要门控机制: 更新门(z)、重置门(r)和候选激活值(q)
    - 使用卷积操作代替原始GRU中的线性变换
    - 支持外部提供的上下文特征
    
    在视差更新过程中，保持特征的空间结构，同时实现时序记忆功能。
    """
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        """
        GRU单元前向传播
        
        参数:
            h: 当前隐藏状态
            cz: 更新门的上下文特征
            cr: 重置门的上下文特征
            cq: 候选激活的上下文特征
            x_list: 输入特征列表
            
        返回:
            更新后的隐藏状态
        """
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    """
    基础运动编码器
    
    将视差和相关性特征编码为运动特征:
    - 处理几何编码特征
    - 编码当前视差图
    - 融合相关性和视差信息
    
    生成用于GRU更新的运动特征，捕捉视差变化和匹配关系。
    """
    def __init__(self, args, ngroup=8):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (ngroup+1)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+256, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        """
        编码运动特征
        
        参数:
            disp: 当前视差估计
            corr: 几何相关性特征
            
        返回:
            编码后的运动特征，与原始视差连接
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    """
    2倍下采样函数
    
    使用平均池化实现特征下采样，保持更多上下文信息
    
    参数:
        x: 输入特征图
        
    返回:
        下采样后的特征图
    """
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    """
    4倍下采样函数
    
    使用平均池化实现更大尺度的特征下采样
    
    参数:
        x: 输入特征图
        
    返回:
        下采样后的特征图
    """
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    """
    特征插值函数
    
    将特征图插值到目标尺寸，用于特征融合
    
    参数:
        x: 输入特征图
        dest: 目标特征图，提供目标尺寸
        
    返回:
        插值后的特征图
    """
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class RaftConvGRU(nn.Module):
    """
    RAFT风格的卷积GRU单元
    
    基于RAFT架构的GRU变体:
    - 简化的参数结构
    - 更适合视差估计任务
    - 无需外部上下文特征
    
    用于SelectiveConvGRU中作为基础GRU单元。
    """
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        """
        RAFT GRU前向传播
        
        参数:
            h: 当前隐藏状态
            x: 输入特征
            hx: 预计算的h和x的连接特征
            
        返回:
            更新后的隐藏状态
        """
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h


class SelectiveConvGRU(nn.Module):
    """
    选择性卷积GRU
    
    结合小卷积核和大卷积核的GRU单元:
    - 使用注意力机制动态融合两种GRU的输出
    - 小卷积核(1x1)关注局部细节
    - 大卷积核(3x3)关注更广泛的上下文
    
    通过注意力选择性地组合不同感受野的特征，提高视差更新的准确性。
    """
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        super(SelectiveConvGRU, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim+hidden_dim, input_dim+hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        """
        选择性GRU前向传播
        
        参数:
            att: 注意力权重
            h: 当前隐藏状态
            x: 输入特征列表
            
        返回:
            更新后的隐藏状态
        """
        x = torch.cat(x, dim=1)
        x = self.conv0(x)
        hx = torch.cat([x, h], dim=1)
        hx = self.conv1(hx)
        h = self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)

        return h


class BasicSelectiveMultiUpdateBlock(nn.Module):
    """
    基础选择性多层更新模块
    
    实现多尺度GRU更新架构:
    - 支持1到3层GRU单元
    - 较低分辨率GRU关注全局结构
    - 较高分辨率GRU关注局部细节
    - 不同尺度间的特征交互
    
    是FoundationStereo中视差迭代更新的核心组件，通过多层次更新提高视差估计的精度。
    """
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, volume_dim)

        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, 256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, net, inp, corr, disp, att):
        """
        多层GRU更新前向传播
        
        自顶向下更新GRU隐藏状态，先处理低分辨率，再处理高分辨率:
        
        参数:
            net: GRU隐藏状态列表
            inp: 上下文特征列表
            corr: 几何相关性特征
            disp: 当前视差估计
            att: 注意力权重列表
            
        返回:
            更新后的隐藏状态、特征掩码和视差增量
        """
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], pool2x(net[0]))

        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, interp(net[1], net[0]))

        delta_disp = self.disp_head(net[0])

        # 缩放掩码以平衡梯度
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp

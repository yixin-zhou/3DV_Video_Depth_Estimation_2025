# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
几何编码模块 (Geometry Encoding Module)
该模块负责根据当前视差估计生成几何编码特征，为视差更新提供几何约束信息。
主要包含Combined_Geo_Encoding_Volume类，它将特征相关性和代价体信息结合，
生成更丰富的几何表示。这些几何特征对于GRU更新视差估计至关重要。
"""

import torch,pdb,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *

class Combined_Geo_Encoding_Volume:
    """
    组合几何编码体类
    
    将特征相关性和现有代价体结合，根据当前视差估计生成几何编码特征。
    这些特征为视差迭代更新提供了重要的几何约束信息。
    
    主要功能：
    - 构建特征相关性金字塔
    - 构建代价体金字塔
    - 根据当前视差对几何特征进行采样
    - 生成多尺度几何特征
    
    几何特征的多尺度表示有助于捕获不同尺度的视差信息，提高视差估计的准确性。
    """
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        """
        初始化组合几何编码体
        
        参数：
            init_fmap1: 左图像特征，来自特征提取网络
            init_fmap2: 右图像特征，来自特征提取网络
            geo_volume: 初始代价体，经过沙漏网络处理的代价体
            num_levels: 几何特征金字塔的层数
            dx: 视差偏移张量，用于生成视差采样点
        """
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx

        # 构建特征相关性矩阵
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        
        # 调整代价体形状以便于后续处理
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d).contiguous()

        # 调整相关性矩阵形状
        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        
        # 构建多尺度金字塔
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        
        # 构建代价体金字塔，逐级下采样
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        # 构建相关性金字塔，逐级下采样
        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)


    def __call__(self, disp, coords, low_memory=False):
        """
        根据当前视差生成几何编码特征
        
        对每一层金字塔，根据当前视差估计对代价体和相关性矩阵进行采样，
        然后将所有层的特征连接起来形成最终的几何编码特征。
        
        参数：
            disp: 当前视差估计，形状为[B,1,H,W]
            coords: 像素坐标，形状为[B,H,W,1]
            low_memory: 是否使用低内存模式
            
        返回：
            多尺度几何特征，形状为[B,C,H,W]
        """
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []
        
        # 对每个金字塔层级进行处理
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            
            # 计算采样坐标
            x0 = self.dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            
            # 根据当前视差对代价体进行采样
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)

            # 计算右图对应坐标
            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + self.dx   # X on right image
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            
            # 根据右图坐标对相关性矩阵进行采样
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)

            # 收集每一层的几何特征
            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
            
        # 将所有层的特征连接成最终的几何编码特征
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2):
        """
        计算特征图之间的相关性
        
        使用einsum计算归一化后的特征之间的点积，生成相关性矩阵。
        该相关性矩阵表示左图特征与右图特征之间的匹配程度。
        
        参数：
            fmap1: 左图特征，形状为[B,D,H,W1]
            fmap2: 右图特征，形状为[B,D,H,W2]
            
        返回：
            相关性矩阵，形状为[B,H,W1,1,W2]
        """
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.reshape(B, D, H, W1)
        fmap2 = fmap2.reshape(B, D, H, W2)
        
        # 使用float32精度计算相关性，以确保数值稳定性
        with torch.cuda.amp.autocast(enabled=False):
          corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        
        corr = corr.reshape(B, H, W1, 1, W2)
        return corr
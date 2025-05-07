# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2(nn.Module):
    def __init__(self, model_name='vits'):
        super(DINOv2, self).__init__()
        
        assert model_name in ['vits', 'vitb', 'vitl']
        
        # 使用本地的 dinov2 模型
        from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
        
        if model_name == 'vits':
            self.model = vit_small(patch_size=14)
            self.embed_dim = 384
        elif model_name == 'vitb':
            self.model = vit_base(patch_size=14)
            self.embed_dim = 768
        elif model_name == 'vitl':
            self.model = vit_large(patch_size=14)
            self.embed_dim = 1024
        
        # 将 model 设置为评估模式
        self.model.eval()
        
        # patch_size 设置为 14，与原始 dinov2 一致
        self.patch_size = 14
    
    def get_intermediate_layers(self, x, n, return_class_token=False):
        # 将视频序列重塑为批次图像
        if len(x.shape) == 5:  # 如果是视频序列 [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
        
        # 调用底层模型的 get_intermediate_layers
        return self.model.get_intermediate_layers(x, n, return_class_token=return_class_token)
    
    def forward(self, x):
        # 将视频序列重塑为批次图像
        if len(x.shape) == 5:  # 如果是视频序列 [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)
        
        # 获取 DINOv2 特征
        features = self.model.forward_features(x)
        return features 
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    """
    时序注意力块
    
    对输入特征在时间维度上应用多头自注意力机制
    用于捕捉不同时间帧之间的依赖关系
    """
    def __init__(
        self,
        d_model,
        num_heads,
        zero_initialize=True,
    ):
        """
        初始化时序注意力块
        
        参数:
            d_model: 特征维度
            num_heads: 注意力头数量
            zero_initialize: 是否将输出投影层初始化为零，有助于稳定训练
        """
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        
        self.norm = nn.LayerNorm(
            d_model, elementwise_affine=True
        )
        
        if zero_initialize:
            nn.init.zeros_(self.attn.out_proj.weight.data)
            nn.init.zeros_(self.attn.out_proj.bias.data)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征，形状为 [B, T, N, C]
               B - 批量大小
               T - 时间长度
               N - 空间位置数 (H*W)
               C - 通道数
               
        返回:
            时序注意力增强后的特征，形状与输入相同
        """
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        
        # 重塑为 [B*N, T, C] 以进行时序注意力计算
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B*N, T, C)
        
        # 应用层归一化和自注意力
        x_norm = self.norm(x_reshaped)
        
        # 计算注意力并添加残差连接
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + attn_out
        
        # 重塑回原始形状 [B, T, N, C]
        x = x_reshaped.reshape(B, N, T, C).permute(0, 2, 1, 3)
        
        return x


class TemporalTransformerBlock(nn.Module):
    """
    时序Transformer块
    
    包含多个时序注意力块的序列
    用于逐步增强特征的时序表示
    """
    def __init__(
        self,
        d_model,
        num_heads,
        num_attention_blocks=1,
        zero_initialize=True,
    ):
        """
        初始化时序Transformer块
        
        参数:
            d_model: 特征维度
            num_heads: 注意力头数量
            num_attention_blocks: 注意力块数量
            zero_initialize: 是否将最后一个注意力块的输出投影层初始化为零
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TemporalAttentionBlock(
                d_model,
                num_heads,
                zero_initialize=True if (i == num_attention_blocks - 1) and zero_initialize else False
            )
            for i in range(num_attention_blocks)
        ])
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征，形状为 [B, T, N, C]
               
        返回:
            经过多个注意力块处理后的特征，形状与输入相同
        """
        # x: [B, T, N, C]
        for block in self.blocks:
            x = block(x)
        return x


class TemporalModule(nn.Module):
    """
    时序模块: 将时序自注意力应用于空间特征
    
    该模块负责处理视频序列中的时间依赖关系
    通过时序Transformer块增强特征表示的时间一致性
    """
    
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=1,
        num_attention_blocks=2,
        temporal_max_len=32,
        zero_initialize=True,
        pos_embedding_type="ape",
    ):
        """
        初始化时序模块
        
        参数:
            in_channels: 输入通道数
            num_attention_heads: 注意力头数量
            num_transformer_block: Transformer块数量
            num_attention_blocks: 每个Transformer块中的注意力块数量
            temporal_max_len: 最大时间长度
            zero_initialize: 是否将最后一个Transformer块的输出初始化为零
            pos_embedding_type: 位置编码类型，默认为'ape'(绝对位置编码)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_transformer_block = num_transformer_block
        self.pos_embedding_type = pos_embedding_type
        self.temporal_max_len = temporal_max_len
        
        # 绝对位置编码
        if self.pos_embedding_type == "ape":
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, temporal_max_len, 1, in_channels)
            )
            nn.init.normal_(self.pos_embedding, std=0.02)
        
        # 创建Transformer块列表
        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                in_channels,
                num_attention_heads,
                num_attention_blocks=num_attention_blocks,
                zero_initialize=True if (i == num_transformer_block - 1) and zero_initialize else False
            )
            for i in range(num_transformer_block)
        ])
        
    def forward(self, x, encoder_hidden_states=None, encoder_attention_mask=None, local_attn_mask=None):
        """
        前向传播函数
        
        参数:
            x: 输入特征，形状为 [B, C, T, H, W]
               B - 批量大小
               C - 通道数
               T - 时间长度
               H, W - 空间尺寸
            encoder_hidden_states: 编码器隐藏状态，未使用
            encoder_attention_mask: 编码器注意力掩码，未使用
            local_attn_mask: 局部注意力掩码，未使用
            
        返回:
            时序增强后的特征，形状与输入相同
        """
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # 确保时间长度不超过最大长度
        assert T <= self.temporal_max_len, f"输入序列长度 {T} 超过最大长度 {self.temporal_max_len}"
        
        # 重塑为 [B, T, H*W, C]
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T, H*W, C)
        
        # 添加位置编码
        if self.pos_embedding_type == "ape":
            x = x + self.pos_embedding[:, :T, :, :]
        
        # 应用Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 重塑回原始形状 [B, C, T, H, W]
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        return x 
"""
注意力机制模块 - 提供用于空间特征增强的各种注意力机制实现

这个模块包含了几种注意力机制实现：
1. LinearAttention: 线性注意力，计算复杂度为O(N)，适用于高分辨率特征图
2. FullAttention: 标准的缩放点积注意力，计算复杂度为O(N²)
3. LoFTREncoderLayer: LoFTR架构中使用的Transformer编码器层
4. LocalFeatureTransformer: 用于局部特征变换的Transformer模块

这些注意力机制用于StereoAnyVideo模型中的空间特征增强，帮助捕获远距离空间依赖关系。
"""

import math
import copy
import torch
import torch.nn as nn
from torch.nn import Module, Dropout

"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""


def elu_feature_map(x):
    """ELU特征映射函数，用于线性注意力
    
    使用ELU激活函数+1作为特征映射，确保所有值都为正，允许使用线性注意力近似
    
    Args:
        x (torch.Tensor): 输入特征
        
    Returns:
        torch.Tensor: 应用ELU+1后的特征
    """
    return torch.nn.functional.elu(x) + 1


class PositionEncodingSine(nn.Module):
    """正弦位置编码
    
    实现2D图像的正弦位置编码，为注意力机制提供位置信息
    
    Args:
        d_model (int): 模型维度/通道数
        max_shape (tuple, optional): 特征图的最大尺寸. 默认: (256, 256)
        temp_bug_fix (bool, optional): 是否使用修复bug的实现. 默认: True
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): 对于1/8特征图，最大长度256对应原图2048像素
            temp_bug_fix (bool): 如[issue](https://github.com/zju3dv/LoFTR/issues/41)所述，
                原始LoFTR实现中位置编码有一个bug，但对最终性能影响很小。
                为保持向后兼容，我们保留两种实现。
        """
        super().__init__()
        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float()
                * (-math.log(10000.0) / (d_model // 2))
            )
        else:  # 有bug的实现（仅用于向后兼容）
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float()
                * (-math.log(10000.0) / d_model // 2)
            )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """将位置编码添加到输入特征
        
        Args:
            x (torch.Tensor): 输入特征 [N, C, H, W]
            
        Returns:
            torch.Tensor: 添加位置编码后的特征 [N, C, H, W]
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)].to(x.device)


class LinearAttention(Module):
    """线性注意力机制
    
    实现计算复杂度为O(N)的线性注意力，适用于处理大尺寸特征图
    
    Args:
        eps (float, optional): 避免除零的小常数. 默认: 1e-6
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """多头线性注意力计算，来自"Transformers are RNNs"
        
        Args:
            queries (torch.Tensor): 查询张量 [N, L, H, D]
            keys (torch.Tensor): 键张量 [N, S, H, D]
            values (torch.Tensor): 值张量 [N, S, H, D]
            q_mask (torch.Tensor, optional): 查询掩码 [N, L]
            kv_mask (torch.Tensor, optional): 键值掩码 [N, S]
            
        Returns:
            torch.Tensor: 注意力加权后的值 (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # 将掩码位置设为零
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # 防止fp16溢出
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    """完整的点积注意力机制
    
    实现标准的缩放点积注意力，计算复杂度为O(N²)
    
    Args:
        use_dropout (bool, optional): 是否使用dropout. 默认: False
        attention_dropout (float, optional): Dropout率. 默认: 0.1
    """
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """多头缩放点积注意力，即完整注意力
        
        Args:
            queries (torch.Tensor): 查询张量 [N, L, H, D]
            keys (torch.Tensor): 键张量 [N, S, H, D]
            values (torch.Tensor): 值张量 [N, S, H, D]
            q_mask (torch.Tensor, optional): 查询掩码 [N, L]
            kv_mask (torch.Tensor, optional): 键值掩码 [N, S]
            
        Returns:
            torch.Tensor: 注意力加权后的值 (N, L, H, D)
        """

        # 计算未归一化的注意力并应用掩码
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf")
            )

        # 计算注意力和加权平均
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


# Ref: https://github.com/zju3dv/LoFTR/blob/master/src/loftr/loftr_module/transformer.py
class LoFTREncoderLayer(nn.Module):
    """LoFTR Transformer编码器层
    
    基于LoFTR论文的Transformer编码器层实现，用于处理局部特征
    
    Args:
        d_model (int): 模型维度
        nhead (int): 注意力头数
        attention (str, optional): 注意力类型，"linear"或"full". 默认: "linear"
    """
    def __init__(self, d_model, nhead, attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # 多头注意力
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # 规范化和dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """前向传播
        
        Args:
            x (torch.Tensor): 查询特征 [N, L, C]
            source (torch.Tensor): 源特征 [N, S, C]
            x_mask (torch.Tensor, optional): 查询掩码 [N, L]
            source_mask (torch.Tensor, optional): 源掩码 [N, S]
            
        Returns:
            torch.Tensor: 更新后的特征 [N, L, C]
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # 多头注意力
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # 前馈网络
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """局部特征变换器
    
    基于LoFTR的局部特征变换模块，用于增强特征表示
    
    Args:
        d_model (int): 模型维度
        nhead (int): 注意力头数
        layer_names (list): 层名称列表，可以是"self"或"cross"
        attention (str): 注意力类型，"linear"或"full"
    """

    def __init__(self, d_model, nhead, layer_names, attention):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """重置模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """前向传播
        
        Args:
            feat0 (torch.Tensor): 第一组特征 [N, L, C]
            feat1 (torch.Tensor): 第二组特征 [N, S, C]
            mask0 (torch.Tensor, optional): 第一组特征掩码 [N, L]
            mask1 (torch.Tensor, optional): 第二组特征掩码 [N, S]
            
        Returns:
            tuple: 更新后的特征对 (feat0, feat1)
        """
        assert self.d_model == feat0.size(
            2
        ), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):

            if name == "self":
                # 自注意力：特征与自身交互
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                # 交叉注意力：特征与另一组特征交互
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1

"""
StereoAnyVideo模型的核心实现 - 立体视频深度估计网络

这个模块包含StereoAnyVideo模型的主要实现，该模型能够从立体视频中估计深度/视差。
模型采用多尺度渐进细化策略，从粗到细处理特征，并利用时空注意力机制保持时间一致性。
"""

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

# 用于混合精度训练/推理的自动转换器
autocast = torch.cuda.amp.autocast

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


class StereoAnyVideo(nn.Module):
    """StereoAnyVideo模型的主类
    
    该模型结合了深度特征提取、上下文特征提取和相关性匹配，通过多尺度细化估计视差。
    
    Args:
        mixed_precision (bool, optional): 是否使用混合精度训练/推理. 默认: False
    """
    def __init__(self, mixed_precision=False):
        super(StereoAnyVideo, self).__init__()

        # 是否使用混合精度
        self.mixed_precision = mixed_precision

        # 模型超参数
        self.hidden_dim = 128  # 隐藏层特征维度
        self.context_dim = 128  # 上下文特征维度
        self.dropout = 0  # dropout率

        # 特征提取网络和更新模块
        # 上下文特征提取器，输出96通道特征图
        self.cnet = BasicEncoder(output_dim=96, norm_fn='instance', dropout=self.dropout)
        # 匹配特征提取器，输出96通道特征图
        self.fnet = BasicEncoder(output_dim=96, norm_fn='instance', dropout=self.dropout)
        # 深度特征提取器，使用预训练Video-Depth-Anything模型
        self.depthnet = DepthExtractor()
        # 相关性特征处理MLP，将4*9*9维度的相关性特征转换为128维度
        self.corr_mlp = Mlp(in_features=4 * 9 * 9, hidden_features=256, out_features=128)
        # 序列更新块，用于迭代优化视差估计
        self.update_block = SequenceUpdateBlock3D(hidden_dim=self.hidden_dim, cor_planes=128, mask_size=4)

    @torch.jit.ignore
    def no_weight_decay(self):
        """指定不应用权重衰减的参数
        
        Returns:
            set: 不需要权重衰减的参数名称集合
        """
        return {"time_embed"}

    def freeze_bn(self):
        """冻结所有BatchNorm层
        
        在测试/推理时使用，防止BatchNorm统计信息更新
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate=4):
        """使用卷积组合上采样2D流场
        
        Args:
            flow (torch.Tensor): 输入流场 [N, 2, H/rate, W/rate]
            mask (torch.Tensor): 用于上采样的掩码 [N, rate*rate*9, H/rate, W/rate]
            rate (int, optional): 上采样率. 默认: 4
            
        Returns:
            torch.Tensor: 上采样后的流场 [N, 2, H, W]
        """
        N, _, H, W = flow.shape
        # 将掩码重塑为[N, 1, 9, rate, rate, H, W]，以适用于卷积组合
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        # 在掩码的第3维上应用softmax，确保9个位置的权重和为1
        mask = torch.softmax(mask, dim=2)

        # 将流场放大rate倍，并使用滑动窗口展开成[N, 2*9, H*W]
        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        # 重塑为[N, 2, 9, 1, 1, H, W]
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        # 使用掩码加权求和，实现卷积组合
        up_flow = torch.sum(mask * up_flow, dim=2)
        # 调整维度顺序
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # 重塑为最终形状[N, 2, rate*H, rate*W]
        return up_flow.reshape(N, 2, rate * H, rate * W)

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

    def zero_init(self, fmap):
        """初始化全零流场
        
        创建与输入特征图形状匹配的全零流场（视差场）。
        
        Args:
            fmap (torch.Tensor): 参考特征图 [N, C, H, W]
            
        Returns:
            torch.Tensor: 初始化的全零流场 [N, 2, H, W]
        """
        N, C, H, W = fmap.shape
        # 创建水平方向的全零流场（用于视差）
        flow_u = torch.zeros([N, 1, H, W], dtype=torch.float)
        # 创建垂直方向的全零流场（在立体匹配中通常为0）
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)
        # 合并为[N, 2, H, W]形状的流场并移至正确设备
        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def forward_batch_test(
        self, batch_dict, iters = 24, flow_init=None,
    ):
        """批量测试前向传播，用于处理超过GPU内存的长视频序列
        
        将长视频序列分成重叠的小块，每块独立处理后再拼接。
        使用滑动窗口策略确保时间上的平滑过渡。
        
        Args:
            batch_dict (dict): 包含"stereo_video"键的输入字典
            iters (int, optional): 每个尺度的迭代次数. 默认: 24
            flow_init (torch.Tensor, optional): 初始流场. 默认: None
            
        Returns:
            dict: 包含"disparity"键的预测字典
        """
        # 滑动窗口参数
        kernel_size = 20  # 每个批次处理的帧数
        stride = kernel_size // 2  # 重叠的帧数
        predictions = defaultdict(list)  # 存储预测结果

        disp_preds = []  # 视差预测列表
        video = batch_dict["stereo_video"]  # 输入立体视频

        num_ims = len(video)  # 视频总帧数
        print("video", video.shape)

        # 使用滑动窗口处理视频
        for i in range(0, num_ims, stride):
            # 提取当前批次的左右视图
            left_ims = video[i : min(i + kernel_size, num_ims), 0]
            padder = InputPadder(left_ims.shape, divis_by=32)  # 确保尺寸可被32整除
            right_ims = video[i : min(i + kernel_size, num_ims), 1]
            left_ims, right_ims = padder.pad(left_ims, right_ims)  # 填充图像
            
            # 如果提供了初始流场，使用它初始化
            if flow_init is not None:
                flow_init_ims = flow_init[i: min(i + kernel_size, num_ims)]
                flow_init_ims = padder.pad(flow_init_ims)[0]
                with autocast(enabled=self.mixed_precision):
                    disparities_forw = self.forward(
                        left_ims[None].cuda(),
                        right_ims[None].cuda(),
                        flow_init=flow_init_ims,
                        iters=iters,
                        test_mode=True,
                    )
            else:
                # 否则从零开始估计视差
                with autocast(enabled=self.mixed_precision):
                    disparities_forw = self.forward(
                        left_ims[None].cuda(),
                        right_ims[None].cuda(),
                        iters=iters,
                        test_mode=True,
                    )

            # 恢复原始分辨率并移至CPU
            disparities_forw = padder.unpad(disparities_forw[:, 0])[:, None].cpu()

            # 处理重叠区域的预测结果
            if len(disp_preds) > 0 and len(disparities_forw) >= stride:
                # 只保留当前批次中后半部分的预测，前半部分与前一批次重叠
                if len(disparities_forw) < kernel_size:
                    disp_preds.append(disparities_forw[stride // 2 :])
                else:
                    disp_preds.append(disparities_forw[stride // 2 : -stride // 2])

            elif len(disp_preds) == 0:
                # 第一批次，保留除最后stride//2帧外的所有预测
                disp_preds.append(disparities_forw[: -stride // 2])

        # 合并所有预测并返回绝对值（视差总是正的）
        predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]

        return predictions

    def forward(self, image1, image2, flow_init=None, iters=12, test_mode=False):
        """模型主前向传播函数
        
        处理左右视图，提取特征，在多个尺度上迭代优化视差估计。
        
        Args:
            image1 (torch.Tensor): 左视图序列 [b, T, c, h, w]
            image2 (torch.Tensor): 右视图序列 [b, T, c, h, w]
            flow_init (torch.Tensor, optional): 初始视差估计. 默认: None
            iters (int, optional): 最细尺度的迭代次数. 默认: 12
            test_mode (bool, optional): 是否在测试模式下运行. 默认: False
            
        Returns:
            torch.Tensor: 如果test_mode为True，返回最终视差；否则返回所有尺度所有迭代的视差预测
        """
        # 获取输入形状
        b, T, c, h, w = image1.shape

        # 图像预处理：归一化到[0,1]区间
        image1 = image1 / 255.0
        image2 = image2 / 255.0

        # 使用ImageNet预训练模型的均值和标准差进行标准化
        mean = torch.tensor([0.485, 0.456, 0.406], device=image1.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image1.device).view(1, 1, 3, 1, 1)

        image1 = (image1 - mean) / std
        image2 = (image2 - mean) / std
        image1 = image1.float()
        image2 = image2.float()

        # 特征提取网络
        with autocast(enabled=self.mixed_precision):
            # 提取深度特征
            fmap1_depth_feature = self.depthnet(image1)  # [b, T, 32, h/4, w/4]
            fmap2_depth_feature = self.depthnet(image2)  # [b, T, 32, h/4, w/4]
            # 提取上下文特征（只用于左视图）
            fmap1_cnet_feature = self.cnet(image1.flatten(0, 1)).unflatten(0, (b, T))  # [b, T, 96, h/4, w/4]
            # 提取匹配特征（左右视图都需要）
            fmap1_fnet_feature = self.fnet(image1.flatten(0, 1)).unflatten(0, (b, T))  # [b, T, 96, h/4, w/4]
            fmap2_fnet_feature = self.fnet(image2.flatten(0, 1)).unflatten(0, (b, T))  # [b, T, 96, h/4, w/4]

        # 合并深度和匹配特征，作为主要特征表示
        fmap1 = torch.cat((fmap1_depth_feature, fmap1_fnet_feature), dim=2).flatten(0, 1)  # [b*T, 32+96, h/4, w/4]
        fmap2 = torch.cat((fmap2_depth_feature, fmap2_fnet_feature), dim=2).flatten(0, 1)  # [b*T, 32+96, h/4, w/4]

        # 合并深度和上下文特征，作为网络更新的输入
        context = torch.cat((fmap1_depth_feature, fmap1_cnet_feature), dim=2).flatten(0, 1)  # [b*T, 32+96, h/4, w/4]

        with autocast(enabled=self.mixed_precision):
            # 准备更新网络的输入
            net = torch.tanh(context)  # 作为GRU的隐藏状态
            inp = torch.relu(context)  # 作为GRU的输入

            # 创建1/8分辨率的特征
            s_net = F.avg_pool2d(net, 2, stride=2)  # [b*T, 128, h/8, w/8]
            s_inp = F.avg_pool2d(inp, 2, stride=2)  # [b*T, 128, h/8, w/8]

            # 创建1/8分辨率的匹配特征
            s_fmap1 = F.avg_pool2d(fmap1, 2, stride=2)  # [b*T, 128, h/8, w/8]
            s_fmap2 = F.avg_pool2d(fmap2, 2, stride=2)  # [b*T, 128, h/8, w/8]

            # 创建1/16分辨率的匹配特征
            ss_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)  # [b*T, 128, h/16, w/16]
            ss_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)  # [b*T, 128, h/16, w/16]

            # 创建1/16分辨率的网络状态
            ss_net = F.avg_pool2d(net, 4, stride=4)  # [b*T, 128, h/16, w/16]
            ss_inp = F.avg_pool2d(inp, 4, stride=4)  # [b*T, 128, h/16, w/16]

        # 创建各尺度的相关性计算函数
        corr_fn = AAPC(fmap1, fmap2)  # 1/4分辨率
        s_corr_fn = AAPC(s_fmap1, s_fmap2)  # 1/8分辨率
        ss_corr_fn = AAPC(ss_fmap1, ss_fmap2)  # 1/16分辨率

        # 级联细化策略 (1/16 + 1/8 + 1/4)
        flow_predictions = []  # 存储所有尺度所有迭代的预测
        flow = None  # 当前分辨率的流场
        flow_up = None  # 上采样后的流场

        # 如果提供了初始流场，使用它初始化
        if flow_init is not None:
            flow_init = flow_init.cuda()
            # 调整尺度以匹配当前特征图
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = scale * interp(flow_init, size=(fmap1.shape[2], fmap1.shape[3]))
        else:
            # 否则从粗到细逐步细化流场
            
            # 在1/16分辨率初始化全零流场
            ss_flow = self.zero_init(ss_fmap1)  # [b*T, 2, h/16, w/16]

            # 1/16分辨率迭代
            for itr in range(iters // 2):
                # 每隔一次迭代切换相关性窗口大小
                if itr % 2 == 0:
                    small_patch = False  # 使用1x9窗口
                else:
                    small_patch = True  # 使用3x3窗口

                # 分离流场以防止梯度累积
                ss_flow = ss_flow.detach()
                # 计算相关性
                out_corrs = ss_corr_fn(ss_flow, None, small_patch=small_patch)  # [b*T, 4*9*9, h/16, w/16]
                # 通过MLP处理相关性特征
                out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [b*T, 128, h/16, w/16]
                
                with autocast(enabled=self.mixed_precision):
                    # 更新网络状态并预测流场更新
                    ss_net, up_mask, delta_flow = self.update_block(ss_net, ss_inp, out_corrs, ss_flow, t=T)

                # 更新流场
                ss_flow = ss_flow + delta_flow
                # 上采样到1/4分辨率
                flow = self.convex_upsample_3D(ss_flow, up_mask, b, T, rate=4)  # [b*T, 2, h/4, w/4]
                # 插值到原始分辨率（用于可视化和评估）
                flow_up = 4 * F.interpolate(flow, size=(4 * flow.shape[2], 4 * flow.shape[3]), mode='bilinear',
                                          align_corners=True)  # [b*T, 2, h, w]
                # 只保留水平分量（视差）
                flow_predictions.append(flow_up[:, :1])

            # 准备1/8分辨率的流场
            scale = s_fmap1.shape[2] / flow.shape[2]
            s_flow = scale * interp(flow, size=(s_fmap1.shape[2], s_fmap1.shape[3]))  # [b*T, 2, h/8, w/8]

            # 1/8分辨率迭代
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                s_flow = s_flow.detach()
                out_corrs = s_corr_fn(s_flow, None, small_patch=small_patch)
                out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                with autocast(enabled=self.mixed_precision):
                    s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow, t=T)

                s_flow = s_flow + delta_flow
                flow = self.convex_upsample_3D(s_flow, up_mask, b, T, rate=4)
                flow_up = 2 * F.interpolate(flow, size=(2 * flow.shape[2], 2 * flow.shape[3]), mode='bilinear',
                                          align_corners=True)
                flow_predictions.append(flow_up[:, :1])

            # 准备1/4分辨率的流场
            scale = fmap1.shape[2] / flow.shape[2]
            flow = scale * interp(flow, size=(fmap1.shape[2], fmap1.shape[3]))  # [b*T, 2, h/4, w/4]

        # 1/4分辨率迭代（最终尺度）
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch)
            out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow, t=T)

            flow = flow + delta_flow
            flow_up = self.convex_upsample_3D(flow, up_mask, b, T, rate=4)  # [b*T, 2, h, w]
            flow_predictions.append(flow_up[:, :1])  # 只保留水平分量（视差）

        # 整理所有预测结果
        predictions = torch.stack(flow_predictions)  # [num_predictions, b*T, 1, h, w]
        # 重排列为[num_predictions, T, b, 1, h, w]以便于处理时间维度
        predictions = rearrange(predictions, "d (b t) c h w -> d t b c h w", b=b, t=T)
        # 获取最终预测
        flow_up = predictions[-1]  # [T, b, 1, h, w]

        if test_mode:
            return flow_up  # 测试模式只返回最终预测

        return predictions  # 训练模式返回所有预测



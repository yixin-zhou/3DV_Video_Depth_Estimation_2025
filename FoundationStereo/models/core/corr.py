"""
相关性计算模块 - 用于计算立体匹配中左右视图特征之间的相关性

这个模块实现了立体匹配中的核心操作 - 特征相关性计算。
主要实现了All-in-All-Pair Correlation (AAPC)，一种高效的全局相关性计算方法。
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """双线性采样器，基于像素坐标从图像中采样
    
    这个函数使用网格采样从特征图中采样值，将坐标系从像素坐标转换为归一化坐标。
    
    Args:
        img (torch.Tensor): 输入图像或特征图 [B, C, H, W]
        coords (torch.Tensor): 像素坐标 [B, H, W, 2]，最后一维是(x, y)坐标
        mode (str, optional): 插值模式. 默认: "bilinear"
        mask (bool, optional): 是否返回有效采样位置的掩码. 默认: False
    
    Returns:
        torch.Tensor: 采样后的图像 [B, C, H, W]
        torch.Tensor (可选): 有效采样位置的掩码，当mask=True时返回
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # 将像素坐标转换为[-1, 1]范围内的归一化坐标
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid/(H - 1) - 1
    img = img.contiguous()
    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    # 使用grid_sample进行双线性插值采样
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        # 创建有效采样位置的掩码（坐标在[-1, 1]范围内的位置）
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    """创建网格坐标张量
    
    为批次图像创建标准网格坐标，用于流和位移计算。
    
    Args:
        batch (int): 批次大小
        ht (int): 图像高度
        wd (int): 图像宽度
        device (torch.device): 计算设备
        
    Returns:
        torch.Tensor: 坐标网格 [B, 2, H, W]，其中最后两维是(x, y)坐标
    """
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    # 反转坐标顺序，使其符合(x, y)约定
    coords = torch.stack(coords[::-1], dim=0).float()
    # 扩展到批次维度
    return coords[None].repeat(batch, 1, 1, 1)


class AAPC:
    """全像素对相关性计算类 (All-in-All-Pair Correlation)
    
    这个类实现了立体匹配中的核心相关性计算操作，它分析了左右特征图中所有像素对之间的相关性，
    比简单的1D视差搜索更灵活，可以处理更复杂的场景和变形。
    
    该方法将特征图分割为多个通道组，并对每组计算局部邻域内的相关性，然后将结果合并。
    """
    def __init__(self, fmap1, fmap2, att=None):
        """初始化AAPC相关性计算器
        
        Args:
            fmap1 (torch.Tensor): 第一个特征图（通常是左视图）[B, C, H, W]
            fmap2 (torch.Tensor): 第二个特征图（通常是右视图）[B, C, H, W]
            att (torch.Tensor, optional): 注意力权重, 未使用. 默认: None
        """
        self.fmap1 = fmap1  # 左视图特征
        self.fmap2 = fmap2  # 右视图特征

        self.att = att  # 注意力权重（预留但未使用）
        # 创建标准网格坐标，用于后续变形
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False):
        """对给定流/视差计算相关性
        
        Args:
            flow (torch.Tensor): 当前估计的流/视差 [B, 2, H, W]
            extra_offset (torch.Tensor): 额外偏移，未使用
            small_patch (bool, optional): 是否使用小型相关窗口. 默认: False
                - True: 使用3x3窗口进行相关性计算
                - False: 使用1x9窗口进行相关性计算（细长窗口，适合视差搜索）
        
        Returns:
            torch.Tensor: 计算的相关性特征 [B, N, H, W]，其中N是相关性通道数
        """
        # 调用主相关性计算函数
        corr = self.correlation(self.fmap1, self.fmap2, flow, small_patch)
        return corr

    def correlation(self, left_feature, right_feature, flow, small_patch):
        """计算左右特征图之间的相关性
        
        Args:
            left_feature (torch.Tensor): 左视图特征 [B, C, H, W]
            right_feature (torch.Tensor): 右视图特征 [B, C, H, W]
            flow (torch.Tensor): 当前估计的流/视差 [B, 2, H, W]
            small_patch (bool): 是否使用小型相关窗口
            
        Returns:
            torch.Tensor: 计算的相关性特征 [B, N, H, W]
        """
        # 在立体匹配中，我们只关心水平视差，因此垂直分量设为0
        flow[:, 1:] = 0
        # 计算变形坐标：原坐标减去流/视差
        coords = self.coords - flow
        coords = coords.permute(0, 2, 3, 1)
        # 根据当前流/视差对右视图特征进行扭曲采样
        right_feature = bilinear_sampler(right_feature, coords)

        # 根据small_patch参数决定使用的相关窗口尺寸和膨胀率
        if small_patch:
            # 小型相关窗口：3x3
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            # 细长相关窗口：1x9 (更适合视差搜索)
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        # 将特征图分为4个通道组，分别计算相关性
        N, C, H, W = left_feature.size()
        lefts = torch.split(left_feature, [C // 4] * 4, dim=1)
        rights = torch.split(right_feature, [C // 4] * 4, dim=1)
        corrs = []
        for i in range(len(psize_list)):
            # 对每组通道计算相关性
            corr = self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        # 沿通道维度合并所有相关性结果
        final_corr = torch.cat(corrs, dim=1)
        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
        """计算给定窗口大小和膨胀率的相关性
        
        对于左特征图的每个位置，在右特征图的局部窗口内计算相关性。
        
        Args:
            left_feature (torch.Tensor): 左视图特征 [B, C, H, W]
            right_feature (torch.Tensor): 右视图特征 [B, C, H, W]
            psize (tuple, optional): 相关窗口大小 (高度, 宽度). 默认: (3, 3)
            dilate (tuple, optional): 膨胀率 (垂直, 水平). 默认: (1, 1)
            
        Returns:
            torch.Tensor: 计算的相关性特征 [B, N, H, W]，其中N为窗口内像素对的数量
        """
        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]  # 垂直和水平膨胀率
        # 计算需要填充的像素数
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        # 使用复制填充特征图的边缘
        left_pad = F.pad(left_feature, [padx, padx, pady, pady], mode='replicate')
        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        # 存储所有相关性结果
        corr_list = []
        # 对左特征图的每个偏移位置
        for dy1 in range(0, pady * 2 + 1, di_y):
            for dx1 in range(0, padx * 2 + 1, di_x):
                # 截取左特征图偏移区域
                left_crop = left_pad[:, :, dy1:dy1 + H, dx1:dx1 + W]

                # 对右特征图的每个偏移位置
                for dy2 in range(0, pady * 2 + 1, di_y):
                    for dx2 in range(0, padx * 2 + 1, di_x):
                        # 截取右特征图偏移区域
                        right_crop = right_pad[:, :, dy2:dy2 + H, dx2:dx2 + W]
                        assert right_crop.size() == left_crop.size()
                        # 计算内积相关性（求和而不是平均，保持信号幅度）
                        corr = (left_crop * right_crop).sum(dim=1, keepdim=True)  # 在通道维度上求和
                        corr_list.append(corr)

        # 沿通道维度合并所有相关性结果
        corr_final = torch.cat(corr_list, dim=1)
        return corr_final
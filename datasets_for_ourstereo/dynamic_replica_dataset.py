"""
Dynamic Replica Dataset for Video Stereo Training

完全按照 VideoSintelDataset 的思路，但适配 Dynamic Replica 的文件夹结构
"""

import os
import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset

from .datasets import VideoSeqDataset
from .augmentor import VideoSeqAugmentor


def read_dynamic_replica_depth(depth_path):
    """读取 Dynamic Replica 深度图
    
    Args:
        depth_path: 深度图路径 (.geometric.png)
        
    Returns:
        numpy.ndarray: 深度图，形状为 [H, W]
    """
    # 读取深度图 (16位PNG)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if depth is None:
        raise ValueError(f"无法读取深度图: {depth_path}")
    
    # 转换为浮点数深度值 (单位: 米)
    # Dynamic Replica 的深度图通常需要缩放
    depth = depth.astype(np.float32) / 1000.0  # 假设原始单位是毫米
    
    return depth


def read_dynamic_replica_disparity(depth_path, focal_length=None, baseline=None):
    """直接返回深度值作为target (不转换为视差)

    Args:
        depth_path: 深度图路径
        focal_length: 焦距 (像素) - 保留参数兼容性，但不使用
        baseline: 基线距离 (米) - 保留参数兼容性，但不使用

    Returns:
        numpy.ndarray: 深度图，形状为 [H, W]
    """
    # 直接返回深度值，不转换为视差
    return read_dynamic_replica_depth(depth_path)


class DynamicReplicaDataset(VideoSeqDataset):
    """Dynamic Replica 视频立体数据集
    
    完全按照 VideoSintelDataset 的思路实现，但适配 Dynamic Replica 的文件结构
    """
    
    def __init__(self,
                 base_dir="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted",
                 aug_params={},
                 crop_size=None,
                 preload_data=True,
                 max_sequences=None):
        """初始化 Dynamic Replica 数据集

        Args:
            base_dir: 数据根目录 (data/extracted)
            aug_params: 数据增强参数
            crop_size: 裁剪尺寸 [H, W]
            preload_data: 是否预加载所有数据到内存
            max_sequences: 限制加载的序列数量 (用于调试)
        """
        super().__init__(crop_size,
                        aug_params,
                        read_dynamic_replica_depth,
                        read_dynamic_replica_disparity,
                        )
        
        self.base_dir = base_dir
        self.preload_data = preload_data
        self.max_sequences = max_sequences
        
        # 初始化数据集并预处理
        self._init_dataset()
    
    def _init_dataset(self):
        """初始化数据集，扫描所有序列并预处理"""
        print(f"初始化 Dynamic Replica 数据集...")
        print(f"数据根目录: {self.base_dir}")
        
        # 常量定义 (与 VideoSintelDataset 保持一致)
        STANDARD_FRAMES = 50  # 标准化帧数
        BATCHS_PER_SEQ = 2    # 每个序列分割成的批次数
        
        # 获取所有训练序列目录
        train_dirs = glob(os.path.join(self.base_dir, "train_*"))
        train_dirs.sort()
        
        # 限制序列数量 (用于调试)
        if self.max_sequences is not None:
            train_dirs = train_dirs[:self.max_sequences]
            print(f"限制序列数量为: {self.max_sequences}")
        
        print(f"找到 {len(train_dirs)} 个训练序列")
        
        # 初始化样本列表
        self.sample_list = []
        
        # 处理每个训练序列
        for train_dir in train_dirs:
            train_name = os.path.basename(train_dir)
            print(f"处理序列: {train_name}")
            
            # 获取该序列下的所有立体对
            stereo_pairs = self._get_stereo_pairs(train_dir)
            
            # 处理每个立体对
            for pair_name in stereo_pairs:
                try:
                    # 处理单个立体对序列
                    self._process_stereo_pair(train_dir, pair_name, STANDARD_FRAMES, BATCHS_PER_SEQ)
                except Exception as e:
                    print(f"处理立体对 {pair_name} 失败: {e}")
                    continue
        
        # 计算内存使用情况
        total_samples = len(self.sample_list)
        frames_per_sample = STANDARD_FRAMES // BATCHS_PER_SEQ
        
        # 估算内存使用
        if hasattr(self, 'augmentor') and self.augmentor and hasattr(self.augmentor, 'crop_size'):
            h, w = self.augmentor.crop_size
        else:
            h, w = 256, 256  # 默认尺寸
        
        memory_per_sample = frames_per_sample * (2 * 3 * h * w + h * w) * 4  # 左右图像+视差图
        total_memory_mb = (total_samples * memory_per_sample) / (1024 * 1024)
        
        print(f"Dynamic Replica 数据集初始化完成:")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 每个样本帧数: {frames_per_sample}")
        print(f"  - 图像尺寸: {h}x{w}")
        print(f"  - 估算内存使用: {total_memory_mb:.1f} MB")
    
    def _get_stereo_pairs(self, train_dir):
        """获取训练序列中的所有立体对
        
        Args:
            train_dir: 训练序列目录路径
            
        Returns:
            list: 立体对名称列表 (不包含 _left/_right 后缀)
        """
        # 获取所有左目录
        left_dirs = glob(os.path.join(train_dir, "*_left"))
        
        # 提取立体对名称 (去掉 _left 后缀)
        pair_names = []
        for left_dir in left_dirs:
            left_name = os.path.basename(left_dir)
            if left_name.endswith("_left"):
                pair_name = left_name[:-5]  # 去掉 "_left"
                
                # 检查对应的右目录是否存在
                right_dir = os.path.join(train_dir, pair_name + "_right")
                if os.path.exists(right_dir):
                    pair_names.append(pair_name)
        
        return sorted(pair_names)
    
    def _process_stereo_pair(self, train_dir, pair_name, standard_frames, batchs_per_seq):
        """处理单个立体对序列
        
        Args:
            train_dir: 训练序列目录
            pair_name: 立体对名称
            standard_frames: 标准化帧数
            batchs_per_seq: 每个序列分割的批次数
        """
        # 构建左右目录路径
        left_dir = os.path.join(train_dir, pair_name + "_left")
        right_dir = os.path.join(train_dir, pair_name + "_right")
        
        # 获取图像和深度文件列表
        left_images = sorted(glob(os.path.join(left_dir, "images", "*.png")))
        right_images = sorted(glob(os.path.join(right_dir, "images", "*.png")))
        left_depths = sorted(glob(os.path.join(left_dir, "depths", "*.geometric.png")))
        
        # 检查文件数量是否匹配
        if len(left_images) != len(right_images) or len(left_images) != len(left_depths):
            print(f"警告: {pair_name} 文件数量不匹配 - "
                  f"左图:{len(left_images)}, 右图:{len(right_images)}, 深度:{len(left_depths)}")
            return
        
        if len(left_images) == 0:
            print(f"警告: {pair_name} 没有找到图像文件")
            return
        
        # 标准化序列长度
        num_frames = len(left_images)
        if num_frames < standard_frames:
            # 重复帧以达到标准长度
            repeat_factor = standard_frames // num_frames + 1
            left_images = (left_images * repeat_factor)[:standard_frames]
            right_images = (right_images * repeat_factor)[:standard_frames]
            left_depths = (left_depths * repeat_factor)[:standard_frames]
        else:
            # 截取前 standard_frames 帧
            left_images = left_images[:standard_frames]
            right_images = right_images[:standard_frames]
            left_depths = left_depths[:standard_frames]
        
        # 分割成小批次
        frames_per_batch = standard_frames // batchs_per_seq
        
        for i in range(batchs_per_seq):
            start_idx = i * frames_per_batch
            end_idx = start_idx + frames_per_batch
            
            batch_left = left_images[start_idx:end_idx]
            batch_right = right_images[start_idx:end_idx]
            batch_depths = left_depths[start_idx:end_idx]
            
            # 添加到样本列表
            self.sample_list.append([batch_left, batch_right, batch_depths])
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sample_list)
    
    def __getitem__(self, index):
        """获取单个样本
        
        Args:
            index: 样本索引
            
        Returns:
            tuple: (left_seq, right_seq, depth_seq)
                - left_seq: 左图像序列 [T, C, H, W]
                - right_seq: 右图像序列 [T, C, H, W]
                - depth_seq: 深度序列 [T, H, W]
        """
        left_paths, right_paths, depth_paths = self.sample_list[index]
        
        # 读取图像序列 - 按照 VideoSeqAugmentor 期望的格式
        sequence = []  # 将存储 [left, right] 对
        depths = []    # 将存储深度图

        for left_path, right_path, depth_path in zip(left_paths, right_paths, depth_paths):
            # 读取左右图像
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            if left_img is None or right_img is None:
                raise ValueError(f"无法读取图像: {left_path} 或 {right_path}")

            # BGR -> RGB
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            # 直接使用深度值作为target
            depth = read_dynamic_replica_disparity(depth_path)

            # 检查深度值是否有效
            if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                print(f"警告: 发现无效深度值 in {depth_path}")
                # 将无效值替换为有效范围内的值
                depth = np.nan_to_num(depth, nan=20.0, posinf=50.0, neginf=1.0)

            # 确保深度值在合理范围内
            depth = np.clip(depth, 0.1, 100.0)

            # 按照 augmentor 期望的格式添加
            sequence.append([left_img, right_img])
            depths.append(depth)

        # 应用数据增强
        if self.augmentor:
            sequence, depths = self.augmentor(sequence, depths)

        # 转换为所需的格式
        left_seq = np.stack([seq[0] for seq in sequence])   # [T, H, W, 3]
        right_seq = np.stack([seq[1] for seq in sequence])  # [T, H, W, 3]
        disp_seq = np.stack(depths)                         # [T, H, W]
        
        # 转换为torch张量并调整维度顺序
        left_seq = torch.from_numpy(left_seq).permute(0, 3, 1, 2).float()  # [T, C, H, W]
        right_seq = torch.from_numpy(right_seq).permute(0, 3, 1, 2).float()  # [T, C, H, W]
        disp_seq = torch.from_numpy(disp_seq).float()  # [T, H, W]
        
        return left_seq, right_seq, disp_seq

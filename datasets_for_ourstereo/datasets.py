import os
import sys

# 1) 找到 project/ 根目录：
root = os.path.dirname(os.path.dirname(__file__))
# 2) 把它插到搜索路径最前面：
sys.path.insert(0, root)

from torch.utils.data import Dataset
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from datasets_for_ourstereo.augmentor import VideoSeqAugmentor
from utils.utils_read import read_sintel_depth, read_sintel_disparity
from abc import ABC, abstractmethod


class VideoSeqDataset(Dataset, ABC):
    def __init__(self, crop_size, aug_params, depth_reader, disp_reader):
        self.augmentor = VideoSeqAugmentor(crop_size, **aug_params)
        self.depth_reader = depth_reader
        self.disp_reader = disp_reader
        self.sample_list = []

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass



class VideoSintelDataset(VideoSeqDataset):
    def __init__(self,
                 dstype,
                 base_dir="/home/shizl/3DV_Video_Depth_Estimation_2025/data/MPI-Sintel-stereo-training-20150305/training",
                 aug_params={},
                 crop_size=None):
        super().__init__(crop_size,
                        aug_params,
                        read_sintel_depth,
                        read_sintel_disparity,
                        )
        self.dstype = dstype  # 'clean' or 'final'
        self.base_dir = base_dir
        
        # 初始化数据集并预处理
        self._init_dataset()
        
    def _init_dataset(self):
        """初始化数据集，读取并预处理所有视频序列，统一每个序列为50帧"""
        import os
        import cv2
        import numpy as np
        from glob import glob
        from tqdm import tqdm  # 进度条
        
        STANDARD_FRAMES = 50  # 统一的帧数
        
        print(f"正在初始化 {self.dstype} 数据集...")
        
        # 获取所有序列名称
        left_path = os.path.join(self.base_dir, f"{self.dstype}_left")
        sequence_paths = glob(os.path.join(left_path, "*"))
        sequence_names = [os.path.basename(p) for p in sequence_paths]
        
        # 对每个序列，读取并预处理所有帧
        for seq_name in tqdm(sequence_names, desc="处理序列"):
            left_seq_path = os.path.join(self.base_dir, f"{self.dstype}_left", seq_name)
            right_seq_path = os.path.join(self.base_dir, f"{self.dstype}_right", seq_name)
            disp_seq_path = os.path.join(self.base_dir, "disparities", seq_name)
            
            # 获取该序列的所有帧
            left_frames = sorted(glob(os.path.join(left_seq_path, "*.png")))
            right_frames = sorted(glob(os.path.join(right_seq_path, "*.png")))
            disp_frames = sorted(glob(os.path.join(disp_seq_path, "*.png")))
            
            # 确保左右视图和视差图的数量匹配
            assert len(left_frames) == len(right_frames) == len(disp_frames), \
                f"Sequence {seq_name} has mismatched number of frames: " \
                f"left={len(left_frames)}, right={len(right_frames)}, disp={len(disp_frames)}"
            
            actual_frames = len(left_frames)
            print(f"序列 {seq_name} 实际帧数: {actual_frames}, 目标帧数: {STANDARD_FRAMES}")
            
            # 初始化序列容器
            sequence = []  # 将存储 [left, right] 对
            disparities = []  # 将存储视差图
            
            # 读取所有可用的帧（不限制为STANDARD_FRAMES）
            for i in range(actual_frames):
                # 读取左右图像
                left_img = cv2.imread(left_frames[i])
                right_img = cv2.imread(right_frames[i])
                
                # BGR to RGB
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                
                # 读取视差图 (使用专用的 reader 函数)
                disp = self.disp_reader(disp_frames[i])
                
                # 添加到序列中
                sequence.append([left_img, right_img])
                disparities.append(disp)
            
            # 应用数据增强
            if self.augmentor is not None:
                # print("sequence[0].shape, disparities[0].shape(before):", sequence[0][0].shape, disparities[0].shape)
                sequence, disparities = self.augmentor(sequence, disparities)
                # print("sequence[0].shape, disparities[0].shape(after):", sequence[0][0].shape, disparities[0].shape)
            
            # 转换为所需的格式
            left_seq = np.stack([seq[0] for seq in sequence])   # [T, H, W, 3]
            right_seq = np.stack([seq[1] for seq in sequence])  # [T, H, W, 3]
            disp_seq = np.stack(disparities)                    # [T, H, W]
            
            # 现在处理帧数标准化（在增强之后、添加到sample_list之前）
            
            # 如果帧数超过STANDARD_FRAMES，截断到STANDARD_FRAMES
            if actual_frames > STANDARD_FRAMES:
                left_seq = left_seq[:STANDARD_FRAMES]
                right_seq = right_seq[:STANDARD_FRAMES]
                disp_seq = disp_seq[:STANDARD_FRAMES]
                print(f"序列 {seq_name} 帧数过多，已截断至前 {STANDARD_FRAMES} 帧")
            
            # 如果帧数不足STANDARD_FRAMES，复制最后一帧填充
            elif actual_frames < STANDARD_FRAMES:
                # 获取最后一帧
                last_left = left_seq[-1:].repeat(STANDARD_FRAMES - actual_frames, axis=0)
                last_right = right_seq[-1:].repeat(STANDARD_FRAMES - actual_frames, axis=0)
                last_disp = disp_seq[-1:].repeat(STANDARD_FRAMES - actual_frames, axis=0)
                
                # 连接原始序列和填充帧
                left_seq = np.concatenate([left_seq, last_left], axis=0)
                right_seq = np.concatenate([right_seq, last_right], axis=0)
                disp_seq = np.concatenate([disp_seq, last_disp], axis=0)
                print(f"序列 {seq_name} 帧数不足，已复制最后一帧填充至 {STANDARD_FRAMES} 帧")
            
            # 转换为 torch tensor 并调整维度顺序
            left_seq = torch.from_numpy(left_seq).permute(0, 3, 1, 2).float()   # [T, 3, H, W]
            right_seq = torch.from_numpy(right_seq).permute(0, 3, 1, 2).float() # [T, 3, H, W]
            disp_seq = torch.from_numpy(disp_seq).float()                       # [T, H, W]
            
            # 确保所有序列都是STANDARD_FRAMES帧
            assert left_seq.shape[0] == right_seq.shape[0] == disp_seq.shape[0] == STANDARD_FRAMES, \
                f"处理后的序列帧数不正确: {left_seq.shape[0]}/{right_seq.shape[0]}/{disp_seq.shape[0]} != {STANDARD_FRAMES}"
            
            # 将预处理好的张量添加到样本列表中
            self.sample_list.append([left_seq, right_seq, disp_seq])
            
        print(f"数据集初始化完成，共 {len(self.sample_list)} 个序列，每个序列 {STANDARD_FRAMES} 帧。")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        """直接返回预处理好的张量数据"""
        return self.sample_list[index]



if __name__ == '__main__':
    ds = VideoSintelDataset(dstype='clean')

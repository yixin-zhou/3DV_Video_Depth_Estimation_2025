# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from video_depth_anything.dinov2 import DINOv2
from video_depth_anything.dpt_temporal import DPTHeadTemporal
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from core.utils.utils import *

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32  # 推理时的视频片段长度
OVERLAP = 10    # 相邻视频片段的重叠帧数
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]  # 关键帧索引
INTERP_LEN = 8  # 插值帧数量

class VideoDepthAnything(nn.Module):
    """
    VideoDepthAnything 模型
    
    基于DINOv2和时序DPT头(DPTHeadTemporal)的视频深度估计模型
    能够从视频序列中提取时序相关的深度特征
    """
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        """
        初始化VideoDepthAnything模型
        
        参数:
            encoder: ViT模型版本，可选 'vits', 'vitb', 'vitl'
            features: 特征维度
            out_channels: 各层输出通道数
            use_bn: 是否使用批量归一化
            use_clstoken: 是否使用类别标记
            num_frames: 视频序列帧数
            pe: 位置编码类型，默认为'ape'(绝对位置编码)
        """
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        # 创建DINOv2主干网络
        self.pretrained = DINOv2(model_name=encoder)

        # 创建时序DPT头，用于时空特征处理
        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x, return_intermediate=False):
        """
        前向传播函数
        
        参数:
            x: 视频序列 [B,T,C,H,W]
            return_intermediate: 是否返回中间特征
            
        返回:
            如果return_intermediate=True: 
                返回(out, path_1, path_2, path_3, path_4, disp)，包含各层特征和视差图
            否则: 
                返回深度图，形状为[B,T,H,W]
        """
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        # 从DINOv2获取中间层特征
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        if return_intermediate:
            # 返回中间特征，用于特征融合
            out, path_1, path_2, path_3, path_4, disp = self.head(features, patch_h, patch_w, T, return_intermediate=True)
            return out, path_1, path_2, path_3, path_4, disp
        else:
            # 只返回深度估计结果
            depth = self.head(features, patch_h, patch_w, T)
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
            depth = F.relu(depth)
            return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        """
        推理整个视频的深度
        
        参数:
            frames: 视频帧，形状为[N,H,W,C]
            target_fps: 目标帧率
            input_size: 输入大小，默认518
            device: 计算设备
            fp32: 是否使用FP32精度
            
        返回:
            depth_list: 深度图列表
            target_fps: 目标帧率
        """
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        # 图像预处理转换
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # 准备视频帧
        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        # 填充视频帧，确保长度合适
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None
        # 分段处理视频
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                # 使用之前的关键帧，保持时间一致性
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        # 对深度图进行时间对齐和平滑处理
        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                # 第一段直接添加
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                # 后续段需要与前一段对齐
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                # 计算尺度和偏移，对齐深度图
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                # 处理重叠区域，使用插值平滑过渡
                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                # 添加当前段的非重叠部分
                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                # 更新参考帧
                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
        
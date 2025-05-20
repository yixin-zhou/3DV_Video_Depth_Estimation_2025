"""
数据变换工具 (Data Transformation Utilities)
该模块提供了用于深度估计数据处理的变换函数和类。
包含调整图像尺寸、归一化和数据格式转换等基础操作，
这些工具用于预处理输入图像和深度图，使其符合网络的输入要求。
"""

import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import cv2
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """
    应用最小尺寸约束
    
    保持纵横比调整样本尺寸，确保满足最小尺寸要求:
    - 如果样本尺寸已经满足要求，则不做调整
    - 计算缩放比例，保持原始宽高比
    - 同时调整图像、视差图和掩码
    
    参数:
        sample: 包含图像和深度信息的样本字典
        size: 目标最小尺寸(高,宽)
        image_interpolation_method: 图像插值方法
        
    返回:
        调整后的样本尺寸元组
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """
    尺寸调整变换
    
    灵活的图像尺寸调整类，支持多种调整策略:
    - 可调整整个样本或仅调整图像
    - 支持保持纵横比或强制调整到指定尺寸
    - 提供多种调整方法，如下界约束、上界约束和最小变化
    - 可指定输出尺寸为特定数字的倍数
    
    该类用于预处理输入数据，使其符合模型的输入要求，
    同时提供灵活的配置选项以满足不同场景的需求。
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """
        初始化尺寸调整变换
        
        参数:
            width: 目标宽度
            height: 目标高度
            resize_target: 是否调整目标(深度图等)
            keep_aspect_ratio: 是否保持纵横比
            ensure_multiple_of: 确保输出尺寸是该数的倍数
            resize_method: 调整方法，可选"lower_bound"、"upper_bound"或"minimal"
            image_interpolation_method: 图像插值方法
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """
        约束值为指定数的倍数
        
        将给定值调整为最接近的倍数值:
        - 默认进行四舍五入
        - 如果超过最大值，则向下取整
        - 如果低于最小值，则向上取整
        
        参数:
            x: 输入值
            min_val: 最小允许值
            max_val: 最大允许值
            
        返回:
            调整后的值
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        """
        计算调整后的尺寸
        
        根据输入尺寸和调整策略，计算输出尺寸:
        - 计算宽高缩放比例
        - 根据调整方法和约束条件确定最终尺寸
        
        参数:
            width: 输入宽度
            height: 输入高度
            
        返回:
            调整后的尺寸(宽,高)元组
        """
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        """
        应用尺寸调整变换
        
        参数:
            sample: 输入样本字典，包含图像和可选的深度信息
            
        返回:
            调整尺寸后的样本
        """
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage(object):
    """
    图像归一化变换
    
    使用指定的均值和标准差对图像进行归一化:
    - 应用公式 (image - mean) / std
    - 使图像值分布适合深度学习模型处理
    
    这是深度学习预处理的标准步骤，使模型收敛更快更稳定。
    """

    def __init__(self, mean, std):
        """
        初始化归一化变换
        
        参数:
            mean: 均值，用于减法操作
            std: 标准差，用于除法操作
        """
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        """
        应用归一化变换
        
        参数:
            sample: 输入样本字典
            
        返回:
            归一化后的样本
        """
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """
    网络输入准备变换
    
    将样本准备为神经网络输入格式:
    - 转换图像通道顺序为(C,H,W)
    - 确保数据为连续存储的float32类型
    - 对深度图和掩码进行相应处理
    
    这是模型推理前的最后处理步骤，确保数据格式符合PyTorch要求。
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        应用网络输入准备变换
        
        参数:
            sample: 输入样本字典
            
        返回:
            准备好的网络输入样本
        """
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample

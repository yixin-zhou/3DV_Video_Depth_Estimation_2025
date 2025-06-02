#!/usr/bin/env python3
"""
使用 DynamicReplicaDataset 训练模型的示例脚本

这个脚本展示了如何在训练中使用 DynamicReplicaDataset
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
root = os.path.dirname(__file__)
sys.path.insert(0, root)

from datasets_for_ourstereo.dynamic_replica_dataset import DynamicReplicaDataset
from datasets_for_ourstereo.datasets import VideoSintelDataset


def create_dynamic_replica_dataloader(args, is_train=True):
    """创建 DynamicReplicaDataset 数据加载器
    
    Args:
        args: 命令行参数
        is_train: 是否为训练数据集
        
    Returns:
        torch.utils.data.DataLoader: 数据加载器
    """
    # 设置数据增强参数 (仅对训练集应用增强)
    aug_params = {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': [0.8, 1.2],
        'hue': 0.5/3.14,
        'gamma_params': [0.9, 1.3, 1.0, 1.2]
    } if is_train else {}
    
    # 数据集路径
    base_dir = args.dynamic_replica_path
    
    print(f"{'训练' if is_train else '验证'} 数据集路径: {base_dir}")
    print(f"裁剪尺寸: {args.crop_size}")
    
    # 创建 DynamicReplica 数据集
    dataset = DynamicReplicaDataset(
        base_dir=base_dir,
        aug_params=aug_params,
        crop_size=args.crop_size,
        preload_data=True,
        max_sequences=args.max_sequences if hasattr(args, 'max_sequences') else None,
        focal_length=args.focal_length if hasattr(args, 'focal_length') else 1050.0,
        baseline=args.baseline if hasattr(args, 'baseline') else 0.54
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=is_train,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    
    return loader


def create_mixed_dataloader(args, is_train=True):
    """创建混合数据集 (Sintel + DynamicReplica) 的数据加载器
    
    Args:
        args: 命令行参数
        is_train: 是否为训练数据集
        
    Returns:
        torch.utils.data.DataLoader: 数据加载器
    """
    from torch.utils.data import ConcatDataset
    
    # 设置数据增强参数
    aug_params = {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': [0.8, 1.2],
        'hue': 0.5/3.14,
        'gamma_params': [0.9, 1.3, 1.0, 1.2]
    } if is_train else {}
    
    datasets = []
    
    # 添加 Sintel 数据集
    if hasattr(args, 'sintel_path') and args.sintel_path:
        print(f"添加 Sintel 数据集: {args.sintel_path}")
        sintel_dataset = VideoSintelDataset(
            dstype='clean',
            base_dir=args.sintel_path,
            aug_params=aug_params,
            crop_size=args.crop_size
        )
        datasets.append(sintel_dataset)
    
    # 添加 DynamicReplica 数据集
    if hasattr(args, 'dynamic_replica_path') and args.dynamic_replica_path:
        print(f"添加 DynamicReplica 数据集: {args.dynamic_replica_path}")
        replica_dataset = DynamicReplicaDataset(
            base_dir=args.dynamic_replica_path,
            aug_params=aug_params,
            crop_size=args.crop_size,
            max_sequences=args.max_sequences if hasattr(args, 'max_sequences') else None
        )
        datasets.append(replica_dataset)
    
    if not datasets:
        raise ValueError("至少需要指定一个数据集路径")
    
    # 合并数据集
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)
        print(f"合并数据集，总样本数: {len(combined_dataset)}")
    
    # 创建数据加载器
    loader = DataLoader(
        combined_dataset, 
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=is_train,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    
    return loader


def test_dataloader(dataloader, name="数据加载器"):
    """测试数据加载器"""
    print(f"\n测试 {name}...")
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    # 测试第一个批次
    for batch_idx, (left_seq, right_seq, disp_seq) in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  - 左图像序列: {left_seq.shape}, 类型: {left_seq.dtype}")
        print(f"  - 右图像序列: {right_seq.shape}, 类型: {right_seq.dtype}")
        print(f"  - 视差序列: {disp_seq.shape}, 类型: {disp_seq.dtype}")
        print(f"  - 左图像值范围: [{left_seq.min():.3f}, {left_seq.max():.3f}]")
        print(f"  - 右图像值范围: [{right_seq.min():.3f}, {right_seq.max():.3f}]")
        print(f"  - 视差值范围: [{disp_seq.min():.3f}, {disp_seq.max():.3f}]")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print(f"✅ {name} 测试完成")


def main():
    parser = argparse.ArgumentParser(description="DynamicReplicaDataset 训练示例")
    
    # 数据集路径
    parser.add_argument("--dynamic_replica_path", type=str, 
                       default="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted",
                       help="DynamicReplica 数据集路径")
    parser.add_argument("--sintel_path", type=str, 
                       default="/home/shizl/3DV_Video_Depth_Estimation_2025/data/MPI-Sintel-stereo-training-20150305/training",
                       help="Sintel 数据集路径")
    
    # 数据参数
    parser.add_argument("--crop_size", nargs=2, type=int, default=[256, 256], help="裁剪尺寸 [H, W]")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--max_sequences", type=int, default=None, help="限制序列数量 (用于调试)")
    
    # DynamicReplica 特定参数
    parser.add_argument("--focal_length", type=float, default=1050.0, help="相机焦距")
    parser.add_argument("--baseline", type=float, default=0.54, help="双目基线距离")
    
    # 测试选项
    parser.add_argument("--test_mode", choices=["dynamic_replica", "sintel", "mixed"], 
                       default="dynamic_replica", help="测试模式")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DynamicReplicaDataset 训练示例")
    print("=" * 60)
    print(f"测试模式: {args.test_mode}")
    print(f"批次大小: {args.batch_size}")
    print(f"裁剪尺寸: {args.crop_size}")
    print(f"工作进程数: {args.num_workers}")
    
    try:
        if args.test_mode == "dynamic_replica":
            # 测试 DynamicReplica 数据集
            train_loader = create_dynamic_replica_dataloader(args, is_train=True)
            test_dataloader(train_loader, "DynamicReplica 训练数据加载器")
            
        elif args.test_mode == "sintel":
            # 测试 Sintel 数据集 (作为对比)
            from FoundationStereo.Train_our_model import fetch_dataloader
            train_loader = fetch_dataloader(args, is_train=True)
            test_dataloader(train_loader, "Sintel 训练数据加载器")
            
        elif args.test_mode == "mixed":
            # 测试混合数据集
            train_loader = create_mixed_dataloader(args, is_train=True)
            test_dataloader(train_loader, "混合训练数据加载器")
        
        print("\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

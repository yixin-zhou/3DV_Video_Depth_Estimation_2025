import os
import sys
import argparse
import logging
import json
import heapq
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from glob import glob

import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from munch import DefaultMunch
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf
from tqdm import tqdm

# 1) 找到 project/ 根目录：
root = os.path.dirname(os.path.dirname(__file__))
# 2) 把它插到搜索路径最前面：
sys.path.insert(0, root)

from datasets_for_ourstereo.dynamic_replica_dataset import DynamicReplicaDataset
from FoundationStereo.core.our_stereo import OurStereo
from FoundationStereo.train_utils.losses import sequence_loss_video
from FoundationStereo.train_utils.logger import Logger
from FoundationStereo.train_utils.topk_checkpoints import TopKCheckpoints


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """查找最新的检查点文件

    Args:
        ckpt_dir: 检查点目录

    Returns:
        str: 最新检查点路径，如果没有找到则返回None
    """
    if not os.path.exists(ckpt_dir):
        return None

    # 查找所有checkpoint_epoch_*.pth文件
    checkpoint_pattern = os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # 按epoch数字排序，找到最新的
    def extract_epoch(path):
        filename = os.path.basename(path)
        # 从 checkpoint_epoch_XX.pth 中提取 XX
        try:
            epoch_str = filename.split('_')[2].split('.')[0]
            return int(epoch_str)
        except:
            return -1

    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return latest_checkpoint


def find_best_checkpoint(ckpt_dir: str) -> Optional[str]:
    """查找最佳检查点文件

    Args:
        ckpt_dir: 检查点目录

    Returns:
        str: 最佳检查点路径，如果没有找到则返回None
    """
    if not os.path.exists(ckpt_dir):
        return None

    # 首先尝试从metric_history.json中读取最佳检查点
    metric_history_path = os.path.join(ckpt_dir, "metric_history.json")
    if os.path.exists(metric_history_path):
        try:
            import json
            with open(metric_history_path, 'r') as f:
                data = json.load(f)

            best_checkpoints = data.get('best_checkpoints', [])
            if best_checkpoints:
                # 返回第一个（最佳的）检查点
                best_path = best_checkpoints[0]['path']
                if os.path.exists(best_path):
                    return best_path
        except Exception as e:
            logging.warning(f"无法读取metric_history.json: {e}")

    # 如果没有metric_history.json，查找best_epoch_*.pth文件
    best_pattern = os.path.join(ckpt_dir, "best_epoch_*.pth")
    best_files = glob(best_pattern)

    if not best_files:
        return None

    # 按文件修改时间排序，返回最新的
    latest_best = max(best_files, key=os.path.getmtime)
    return latest_best


def auto_find_checkpoint(args) -> Optional[str]:
    """自动查找检查点

    Args:
        args: 命令行参数

    Returns:
        str: 检查点路径，如果没有找到则返回None
    """
    if args.restore_ckpt:
        # 如果明确指定了检查点路径，直接使用
        return args.restore_ckpt

    if args.resume_from_best:
        # 查找最佳检查点
        best_ckpt = find_best_checkpoint(args.ckpt_path)
        if best_ckpt:
            logging.info(f"找到最佳检查点: {best_ckpt}")
            return best_ckpt
        else:
            logging.warning("未找到最佳检查点")

    if args.auto_resume:
        # 查找最新检查点
        latest_ckpt = find_latest_checkpoint(args.ckpt_path)
        if latest_ckpt:
            logging.info(f"找到最新检查点: {latest_ckpt}")
            return latest_ckpt
        else:
            logging.warning("未找到最新检查点")

    return None


def fetch_dataloader(args, is_train: bool = True) -> torch.utils.data.DataLoader:
    """获取 Dynamic Replica 数据加载器
    
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
    
    # 选择训练集或验证集的路径
    base_dir = args.train_dataset_path if is_train else args.val_dataset_path
    
    logging.info(f"{'训练' if is_train else '验证'} 数据集路径: {base_dir}")
    logging.info(f"裁剪尺寸: {args.crop_size}")
    
    # 创建 Dynamic Replica 数据集
    dataset = DynamicReplicaDataset(
        base_dir=base_dir,
        aug_params=aug_params,
        crop_size=args.crop_size,
        preload_data=True,
        max_sequences=getattr(args, 'max_sequences', None)
    )
    
    # 创建数据加载器，添加pin_memory和persistent_workers优化
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=is_train,
        pin_memory=True,  # 加速GPU传输
        persistent_workers=args.num_workers > 0  # 保持worker进程活跃
    )
    
    return loader


def fetch_optimizer(args, model) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """创建优化器和学习率调度器
    
    Args:
        args: 命令行参数
        model: 待优化的模型
        
    Returns:
        tuple: (optimizer, scheduler) - 优化器和学习率调度器
    """
    # 分组参数：对不同类型的参数使用不同的学习率和权重衰减
    param_groups = []
    
    # 基础参数组
    base_params = []
    # 预训练backbone参数（如果需要较小学习率）
    backbone_params = []
    
    for name, param in model.named_parameters():
        if 'foundation_stereo' in name:
            backbone_params.append(param)
        else:
            base_params.append(param)
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': args.lr * 0.1,  # backbone使用较小学习率
            'weight_decay': args.wdecay
        })
    
    if base_params:
        param_groups.append({
            'params': base_params,
            'lr': args.lr,
            'weight_decay': args.wdecay
        })
    
    # 如果没有分组，使用所有参数
    if not param_groups:
        param_groups = model.parameters()

    # 创建AdamW优化器
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    
    # 选择学习率调度器
    scheduler_type = getattr(args, 'scheduler', 'multistep')
    
    if scheduler_type == 'multistep':
        milestones = [int(args.num_epochs * 0.6), int(args.num_epochs * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif scheduler_type == 'warmup_cosine':
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_cosine_schedule(step):
            warmup_steps = args.num_epochs * 0.1
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (args.num_epochs - warmup_steps)))
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    else:
        # 默认使用MultiStepLR
        milestones = [int(args.num_epochs * 0.6), int(args.num_epochs * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    logging.info(f"使用优化器: AdamW with lr={args.lr}, weight_decay={args.wdecay}")
    logging.info(f"使用调度器: {scheduler_type}")
    
    return optimizer, scheduler


def compute_loss(predictions: torch.Tensor, depth_seq: torch.Tensor, 
                loss_gamma: float = 0.9, max_depth: float = 100.0) -> Dict[str, Any]:
    """计算深度预测的损失
    
    Args:
        predictions: 模型预测的深度，形状为 [b, N, T, h, w]
        depth_seq: 真实深度序列，形状为 [b, T, h, w]
        loss_gamma: 损失衰减因子
        max_depth: 最大深度阈值
    
    Returns:
        dict: 包含损失和度量的字典
    """
    # 检查输入形状
    assert len(predictions.shape) == 5, f"Expected predictions shape [b, N, T, h, w], got {predictions.shape}"
    assert len(depth_seq.shape) == 4, f"Expected depth_seq shape [b, T, h, w], got {depth_seq.shape}"
    
    # 计算加权损失 (使用深度损失而不是视差损失)
    loss, metrics = sequence_loss_video(predictions, depth_seq, loss_gamma=loss_gamma, max_flow=max_depth)
    
    # 添加额外的统计信息
    with torch.no_grad():
        final_pred = predictions[:, -1]  # 最终预测
        abs_error = (final_pred - depth_seq).abs()
        
        # 添加更多评估指标
        metrics.update({
            'mean_abs_error': abs_error.mean().item(),
            'median_abs_error': abs_error.median().item(),
            'std_abs_error': abs_error.std().item(),
            'max_abs_error': abs_error.max().item(),
        })
    
    return {
        'loss': loss,
        'metrics': metrics
    }


def forward_batch(left_seq: torch.Tensor, right_seq: torch.Tensor, 
                 depth_seq: torch.Tensor, model: torch.nn.Module,
                 loss_gamma: float = 0.9) -> Dict[str, Any]:
    """处理一个批次数据的前向传播
    
    将批次数据送入模型，计算损失和指标
    
    Args:
        left_seq: 左序列，形状为 [B, T, C, H, W]
        right_seq: 右序列，形状为 [B, T, C, H, W]
        depth_seq: 深度序列，形状为 [B, T, H, W]
        model: OurStereo模型
        loss_gamma: 损失衰减因子
    
    Returns:
        dict: 包含模型输出、损失和度量的字典
    """
    # 验证输入形状
    assert left_seq.shape == right_seq.shape, f"Left and right sequences must have same shape"
    assert left_seq.shape[:2] == depth_seq.shape[:2], f"Sequence batch and time dimensions must match"
    
    # 模型前向传播获取深度预测
    # 输出 depths 形状为 [B, N, T, H, W]，N 是迭代次数
    try:
        depths = model(left_seq, right_seq)
    except Exception as e:
        logging.error(f"Model forward pass failed: {e}")
        raise
    
    # 计算损失
    loss_dict = compute_loss(depths, depth_seq, loss_gamma=loss_gamma)
    
    return {
        'stereo': {
            'predictions': depths,
            'loss': loss_dict['loss'],
            'metrics': loss_dict['metrics']
        }
    }


class DynamicReplicaTrainer(LightningLite):
    def run(self, args):
        """主训练循环"""
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logging.info("开始 Dynamic Replica 训练...")

        # 创建数据加载器
        train_loader = fetch_dataloader(args, is_train=True)
        val_loader = fetch_dataloader(args, is_train=False) if args.val_dataset_path else None

        # 加载模型配置文件
        from omegaconf import OmegaConf
        config_path = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/cfg.yaml"
        cfg = OmegaConf.load(config_path)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        model_args = OmegaConf.create(cfg)
        model_args.valid_iters = 32

        # 创建模型
        model = OurStereo(model_args)

        # 加载预训练的 foundation_stereo 权重
        foundationstereo_ckpt_dir = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
        foundationstereo_ckpt = torch.load(foundationstereo_ckpt_dir, weights_only=False)
        model.foundation_stereo.load_state_dict(foundationstereo_ckpt['model'])
        logging.info("已加载预训练的 foundation_stereo 权重")

        # 创建优化器和调度器
        optimizer, scheduler = fetch_optimizer(args, model)

        # 设置模型、优化器和数据加载器
        model, optimizer = self.setup(model, optimizer)
        train_loader = self.setup_dataloaders(train_loader)
        if val_loader:
            val_loader = self.setup_dataloaders(val_loader)

        # 初始化混合精度缩放器
        scaler = GradScaler(enabled=args.mixed_precision)

        # 初始化记录器和检查点追踪器
        logger = Logger(model, scheduler, args.ckpt_path)
        topk_tracker = TopKCheckpoints(k=3, metric_name='depth_error', larger_is_better=False)

        # 尝试加载历史记录
        if os.path.exists(args.ckpt_path):
            topk_tracker.load_metric_history(args.ckpt_path)

        # 初始化训练状态变量
        start_epoch = 0
        total_steps = 0
        best_metric = float('inf')
        patience_counter = 0
        patience = getattr(args, 'patience', 20)  # 早停耐心值

        # 自动查找并恢复检查点
        checkpoint_path = auto_find_checkpoint(args)
        if checkpoint_path:
            logging.info(f"恢复检查点: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # 加载模型状态
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logging.warning("检查点中未找到模型状态，跳过模型加载")

                # 加载优化器状态（如果存在）
                if 'optimizer' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        logging.info("已恢复优化器状态")
                    except Exception as e:
                        logging.warning(f"无法恢复优化器状态: {e}")

                # 加载调度器状态（如果存在）
                if 'scheduler' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        logging.info("已恢复调度器状态")
                    except Exception as e:
                        logging.warning(f"无法恢复调度器状态: {e}")

                # 恢复训练状态
                start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
                total_steps = checkpoint.get('total_steps', 0)

                # 如果是最佳检查点，可能没有训练状态信息
                if 'epoch' not in checkpoint:
                    logging.info("这是一个最佳检查点，从epoch 0开始重新训练")
                    start_epoch = 0
                    total_steps = 0

                logging.info(f"从 epoch {start_epoch} 恢复训练，总步数: {total_steps}")

            except Exception as e:
                logging.error(f"加载检查点失败: {e}")
                logging.info("将从头开始训练")
                start_epoch = 0
                total_steps = 0

        # 开始训练前验证
        if args.validate_at_start and val_loader:
            logging.info("开始训练前验证...")
            val_metrics = self.validate(model, val_loader)
            logging.info(f"初始验证指标: {val_metrics}")

        # 主训练循环
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            # 训练一个epoch
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
            for i_batch, batch in enumerate(progress_bar):
                if batch is None:
                    logging.warning(f"Batch {i_batch} is None, skipping")
                    continue

                left_seq, right_seq, depth_seq = batch

                # 将数据移至GPU (非阻塞传输)
                left_seq = left_seq.cuda(non_blocking=True)
                right_seq = right_seq.cuda(non_blocking=True)
                depth_seq = depth_seq.cuda(non_blocking=True)

                assert model.training

                # 检查输入数据是否有效
                if torch.any(torch.isnan(left_seq)) or torch.any(torch.isnan(right_seq)) or torch.any(torch.isnan(depth_seq)):
                    logging.warning(f"发现NaN输入数据，跳过batch {i_batch}")
                    continue

                if torch.any(torch.isinf(left_seq)) or torch.any(torch.isinf(right_seq)) or torch.any(torch.isinf(depth_seq)):
                    logging.warning(f"发现Inf输入数据，跳过batch {i_batch}")
                    continue

                # 清除梯度
                optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效

                # 启用混合精度
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    # 前向传播
                    output = forward_batch(left_seq, right_seq, depth_seq, model,
                                         loss_gamma=getattr(args, 'loss_gamma', 0.9))

                # 计算总损失
                loss = 0
                logger.update()
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]
                        logger.writer.add_scalar(
                            f"live_{k}_loss", v["loss"].item(), total_steps
                        )
                    if "metrics" in v:
                        logger.push(v["metrics"], k)

                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"发现无效损失值: {loss.item()}，跳过batch {i_batch}")
                    continue

                # 累计epoch损失
                epoch_loss += loss.item()
                num_batches += 1

                # 记录训练指标
                if self.global_rank == 0:
                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )

                self.barrier()  # 同步分布式进程

                # 反向传播 (支持梯度累积)
                loss = loss / getattr(args, 'accumulate_grad_batches', 1)  # 缩放损失
                self.backward(scaler.scale(loss))

                # 梯度累积
                if (i_batch + 1) % getattr(args, 'accumulate_grad_batches', 1) == 0:
                    # 检查梯度是否有效（在unscale之前）
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                logging.warning(f"发现无效梯度在参数 {name}")
                                has_nan_grad = True
                                break

                    if has_nan_grad:
                        logging.warning(f"跳过梯度更新 batch {i_batch}")
                        # 简单地清除梯度，不调用scaler.update()
                        optimizer.zero_grad()
                        continue

                    # 只有在梯度有效时才进行unscale
                    scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'gradient_clip', 1.0))  # 梯度裁剪

                    # 检查梯度范数
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(f"无效梯度范数: {grad_norm}，跳过更新")
                        # 简单地清除梯度，不调用scaler.update()
                        optimizer.zero_grad()
                        continue

                    # 优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    total_steps += 1

                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * getattr(args, "accumulate_grad_batches", 1):.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })

                    # 定期清理GPU内存缓存
                    if total_steps % 50 == 0:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()  # 同时进行Python垃圾回收

                # 每个batch结束后清理中间变量
                if 'output' in locals():
                    del output
                if 'left_seq' in locals():
                    del left_seq, right_seq, depth_seq

                # 更频繁的内存清理
                if i_batch % 20 == 0:
                    torch.cuda.empty_cache()

            # 更新学习率
            scheduler.step()

            # 计算平均epoch损失
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logging.info(f"Epoch {epoch}: 平均损失 = {avg_epoch_loss:.6f}")

            # 记录epoch指标
            if self.global_rank == 0:
                logger.writer.add_scalar("epoch_loss", avg_epoch_loss, epoch)
                logger.writer.add_scalar("learning_rate_epoch", optimizer.param_groups[0]["lr"], epoch)

            # 验证
            if val_loader and (epoch + 1) % args.eval_epochs == 0:
                logging.info(f"开始验证 (Epoch {epoch})...")
                val_metrics = self.validate(model, val_loader)
                logging.info(f"验证指标: {val_metrics}")

                # 根据验证结果更新最佳检查点
                topk_tracker.update(epoch, val_metrics, model, args.ckpt_path)

                # 早停检查
                current_metric = val_metrics.get('depth_error', float('inf'))
                if current_metric < best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    logging.info(f"新的最佳验证指标: {best_metric:.4f}")
                else:
                    patience_counter += 1
                    logging.info(f"验证指标未改善，耐心计数: {patience_counter}/{patience}")

                # 早停检查
                if patience_counter >= patience and getattr(args, 'early_stopping', False):
                    logging.info(f"验证指标连续{patience}次未改善，触发早停")
                    break

                # 保存验证历史
                topk_tracker.save_metric_history(args.ckpt_path)

            # 定期保存检查点
            if args.save_regular_epochs and (epoch + 1) % args.save_epochs == 0:
                if self.global_rank == 0:
                    checkpoint_path = os.path.join(args.ckpt_path, f"checkpoint_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'total_steps': total_steps,
                        'loss': avg_epoch_loss
                    }, checkpoint_path)
                    logging.info(f"保存检查点: {checkpoint_path}")

        logging.info("训练完成!")

    def validate(self, model: torch.nn.Module, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        在验证集上评估模型

        Args:
            model: 要评估的模型
            val_loader: 验证数据加载器

        Returns:
            dict: 验证指标
        """
        model.eval()

        # 初始化累积指标
        metrics_accumulator = {
            'depth_error': 0.0,
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'rmse_log': 0.0,
            'a1': 0.0,  # δ < 1.25
            'a2': 0.0,  # δ < 1.25²
            'a3': 0.0,  # δ < 1.25³
        }
        num_samples = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    logging.warning(f"Validation batch {batch_idx} is None, skipping")
                    continue

                left_seq, right_seq, depth_seq = batch

                # 将数据移至GPU (非阻塞传输)
                left_seq = left_seq.cuda(non_blocking=True)
                right_seq = right_seq.cuda(non_blocking=True)
                depth_seq = depth_seq.cuda(non_blocking=True)

                try:
                    # 前向传播，设置test_mode=True获取最终预测
                    # 返回形状为 [b, T, h, w]
                    final_depths = model(left_seq, right_seq, test_mode=True)

                    # 计算各种误差指标
                    abs_error = (final_depths - depth_seq).abs()

                    batch_size = left_seq.size(0)

                    # 计算深度评估指标
                    valid_mask = (depth_seq > 0.1) & (depth_seq < 100.0)  # 有效深度范围

                    if valid_mask.sum() > 0:
                        pred_valid = final_depths[valid_mask]
                        gt_valid = depth_seq[valid_mask]

                        # 绝对相对误差
                        abs_rel = (abs_error[valid_mask] / gt_valid).mean()

                        # 平方相对误差
                        sq_rel = ((final_depths[valid_mask] - gt_valid) ** 2 / gt_valid).mean()

                        # RMSE
                        rmse = torch.sqrt(((final_depths[valid_mask] - gt_valid) ** 2).mean())

                        # RMSE log
                        rmse_log = torch.sqrt(((torch.log(pred_valid) - torch.log(gt_valid)) ** 2).mean())

                        # 阈值准确率
                        thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
                        a1 = (thresh < 1.25).float().mean()
                        a2 = (thresh < 1.25 ** 2).float().mean()
                        a3 = (thresh < 1.25 ** 3).float().mean()

                        # 累积各种指标
                        metrics_accumulator['depth_error'] += abs_error.mean().item() * batch_size
                        metrics_accumulator['abs_rel'] += abs_rel.item() * batch_size
                        metrics_accumulator['sq_rel'] += sq_rel.item() * batch_size
                        metrics_accumulator['rmse'] += rmse.item() * batch_size
                        metrics_accumulator['rmse_log'] += rmse_log.item() * batch_size
                        metrics_accumulator['a1'] += a1.item() * batch_size
                        metrics_accumulator['a2'] += a2.item() * batch_size
                        metrics_accumulator['a3'] += a3.item() * batch_size

                    num_samples += batch_size

                    # 更新进度条
                    current_error = metrics_accumulator['depth_error'] / num_samples
                    progress_bar.set_postfix({'Depth Error': f'{current_error:.4f}'})

                    # 显式删除中间变量释放内存
                    del final_depths, abs_error
                    if 'pred_valid' in locals():
                        del pred_valid, gt_valid, thresh
                    if 'abs_rel' in locals():
                        del abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

                    # 每10个batch清理一次GPU缓存
                    if batch_idx % 10 == 0:
                        torch.cuda.synchronize()  # 确保所有操作完成
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()  # Python垃圾回收

                except Exception as e:
                    logging.error(f"Validation batch {batch_idx} failed: {e}")
                    continue
                finally:
                    # 确保清理batch数据
                    if 'left_seq' in locals():
                        del left_seq, right_seq, depth_seq

        # 计算平均指标
        if num_samples > 0:
            for key in metrics_accumulator:
                metrics_accumulator[key] /= num_samples
        else:
            logging.warning("No valid validation samples found")

        # 验证结束后彻底清理内存
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # 打印验证后的内存使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"验证完成后GPU内存: {memory_allocated:.2f} GB")

        return metrics_accumulator


def main():
    parser = argparse.ArgumentParser(description="Dynamic Replica 深度估计训练")

    # 数据集路径
    parser.add_argument("--train_dataset_path", type=str,
                       default="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted",
                       help="训练数据集路径")
    parser.add_argument("--val_dataset_path", type=str,
                       default="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted_val",
                       help="验证数据集路径")

    # 数据参数
    parser.add_argument("--crop_size", nargs=2, type=int, default=[256, 256], help="裁剪尺寸 [H, W]")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载器工作进程数")
    parser.add_argument("--max_sequences", type=int, default=None, help="限制序列数量 (用于调试)")

    # 训练参数
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--wdecay", type=float, default=0.00001, help="权重衰减")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练的总epoch数")
    parser.add_argument("--save_epochs", type=int, default=10, help="每隔多少个epoch保存一次普通检查点")
    parser.add_argument("--save_regular_epochs", action="store_true", help="是否定期保存普通检查点，无论验证结果")
    parser.add_argument("--eval_epochs", type=int, default=5, help="每隔多少个epoch验证一次")
    parser.add_argument("--mixed_precision", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--validate_at_start", action="store_true", help="是否在训练开始前验证")
    parser.add_argument("--scheduler", type=str, default="multistep",
                       choices=["multistep", "cosine", "warmup_cosine"],
                       help="学习率调度器类型")
    parser.add_argument("--loss_gamma", type=float, default=0.9, help="损失衰减因子")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="梯度累积批次数")

    # 模型参数
    parser.add_argument("--restore_ckpt", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--auto_resume", action="store_true", help="自动从最新检查点恢复训练")
    parser.add_argument("--resume_from_best", action="store_true", help="从最佳检查点恢复训练")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/dynamic_replica", help="检查点保存路径")
    parser.add_argument("--freeze_encoder", action="store_true", help="是否冻结编码器")

    # OurStereo 模型特定参数
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[128, 128, 128], help="隐藏层维度")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="GRU层数")
    parser.add_argument("--n_downsample", type=int, default=2, help="下采样层数")
    parser.add_argument("--max_disp", type=int, default=416, help="最大视差")
    parser.add_argument("--corr_levels", type=int, default=2, help="相关性层数")
    parser.add_argument("--corr_radius", type=int, default=4, help="相关性半径")
    parser.add_argument("--corr_implementation", type=str, default="reg", help="相关性实现方式")
    parser.add_argument("--slow_fast_gru", action="store_true", help="是否使用慢快GRU")
    parser.add_argument("--train_iters", type=int, default=22, help="训练迭代次数")
    parser.add_argument("--valid_iters", type=int, default=32, help="验证迭代次数")

    # 早停和监控参数
    parser.add_argument("--early_stopping", action="store_true", help="是否启用早停")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")

    # 调试和监控参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_images", action="store_true", help="是否保存预测图像")
    parser.add_argument("--tensorboard", action="store_true", help="是否启用TensorBoard日志")

    args = parser.parse_args()

    # 创建检查点目录
    os.makedirs(args.ckpt_path, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.ckpt_path, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # 打印配置
    logging.info("=" * 60)
    logging.info("Dynamic Replica 深度估计训练")
    logging.info("=" * 60)
    logging.info(f"训练数据集: {args.train_dataset_path}")
    logging.info(f"验证数据集: {args.val_dataset_path}")
    logging.info(f"检查点路径: {args.ckpt_path}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"学习率: {args.lr}")
    logging.info(f"训练epochs: {args.num_epochs}")
    logging.info(f"裁剪尺寸: {args.crop_size}")
    logging.info(f"混合精度: {args.mixed_precision}")

    # 启动训练
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    device_list = [int(d) for d in visible_devices.split(',') if d.strip()]

    if len(device_list) > 1:
        # 多GPU训练
        from pytorch_lightning.strategies import DDPStrategy
        from datetime import timedelta
        ddp_strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=7200),  # 增加超时时间到2小时
            process_group_backend="nccl"
        )
        trainer = DynamicReplicaTrainer(devices=len(device_list),
                    accelerator="cuda",
                    precision=16 if args.mixed_precision else 32,
                    strategy=ddp_strategy,
                    )
        logging.info(f"使用 {len(device_list)} 张GPU进行训练: {device_list}")
    else:
        # 单GPU训练
        trainer = DynamicReplicaTrainer(devices=1,
                    accelerator="cuda",
                    precision=16 if args.mixed_precision else 32,
                    )
        logging.info(f"使用单GPU进行训练: {device_list[0] if device_list else 0}")

    # 开始训练
    trainer.run(args)


if __name__ == "__main__":
    main()

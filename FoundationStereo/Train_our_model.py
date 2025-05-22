"""
our_stereo模型训练脚本

这个脚本实现了our_stereo立体视频深度估计模型的训练流程，包括：
1. 数据加载和预处理
2. 模型训练和优化
3. 损失计算和评估
4. 模型保存和恢复
"""

import os, sys

# 1) 找到 project/ 根目录：
root = os.path.dirname(os.path.dirname(__file__))
# 2) 把它插到搜索路径最前面：
sys.path.insert(0, root)

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from munch import DefaultMunch
import json
import heapq
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf

from FoundationStereo.train_utils.utils import (
    # run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from train_utils.logger import Logger

from FoundationStereo.core.our_stereo import OurStereo
# from FoundationStereo.evaluation.core.evaluator import Evaluator
from FoundationStereo.train_utils.losses import sequence_loss, temporal_loss, sequence_loss_video
from datasets_for_ourstereo.datasets import VideoSintelDataset


def fetch_dataloader(args, is_train=True):
    """获取数据加载器
    
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
    
    # 根据是否是训练集，选择不同的数据
    dstype = 'clean'  # 默认使用clean数据
    
    # 选择训练集或验证集的路径
    if is_train:
        base_dir = args.train_dataset_path
    else:
        base_dir = args.val_dataset_path if args.val_dataset_path else args.train_dataset_path
    print("args.crop_size:", args.crop_size)
    # 创建Sintel数据集
    dataset = VideoSintelDataset(
        dstype=dstype,
        base_dir=base_dir,
        aug_params=aug_params,
        crop_size=args.crop_size if is_train else None
    )
    
    # 创建数据加载器
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        drop_last=is_train
    )
    
    return loader


def fetch_optimizer(args, model):
    """
    Args:
        args: 命令行参数
        model: 待优化的模型
        
    Returns:
        tuple: (optimizer, scheduler) - 优化器和学习率调度器
    """

    # 创建AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    
    # 使用多步学习率衰减调度器，每训练一定数量的epoch后降低学习率
    milestones = [int(args.num_epochs * 0.6), int(args.num_epochs * 0.8)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    return optimizer, scheduler


def compute_loss(predictions, disp_seq):
    """计算视差预测的损失
    
    Args:
        predictions: 模型预测的视差，形状为 [b, N, T, h, w]
        disp_seq: 真实视差序列，形状为 [b, T, h, w]
    
    Returns:
        dict: 包含损失和度量的字典
    """
    
    # 计算加权L1损失
    loss, metrics = sequence_loss_video(predictions, disp_seq, loss_gamma=0.8)
    
    return {
        'loss': loss,
        'metrics': metrics
    }


def forward_batch(left_seq, right_seq, disp_seq, model):
    """处理一个批次数据的前向传播
    
    将批次数据送入模型，计算损失和指标
    
    Args:
        left_seq: 左序列，形状为 [B, T, C, H, W]
        right_seq: 右序列，形状为 [B, T, C, H, W]
        disp_seq: 视差序列，形状为 [B, T, H, W]
        model: OurStereo模型
    
    Returns:
        dict: 包含模型输出、损失和度量的字典
    """
    
    # 模型前向传播获取视差预测
    # 输出 disparities 形状为 [B, N, T, H, W]，N 是迭代次数
    print("left_seq, right_seq:", left_seq.shape, right_seq.shape)
    disparities = model(left_seq, right_seq)
    
    # 计算损失
    loss_dict = compute_loss(disparities, disp_seq)
    
    return {
        'stereo': {
            'predictions': disparities,
            'loss': loss_dict['loss'],
            'metrics': loss_dict['metrics']
        }
    }


class TopKCheckpoints:
    """保持K个最佳检查点的辅助类"""
    
    def __init__(self, k=3, metric_name='epe', larger_is_better=False):
        """
        初始化最佳检查点追踪器
        
        Args:
            k: 保存的最佳检查点数量
            metric_name: 用于比较的指标名称
            larger_is_better: 指标值越大越好（如准确率），默认为False（如误差）
        """
        self.k = k
        self.metric_name = metric_name
        self.larger_is_better = larger_is_better
        # 使用最小/最大堆来跟踪最佳检查点
        self.best_checkpoints = []
        self.metric_history = []
        
    def update(self, epoch, metrics, model, save_path):
        """
        更新最佳检查点列表
        
        Args:
            epoch: 当前训练轮次
            metrics: 评估指标字典
            model: 当前模型
            save_path: 检查点保存路径
            
        Returns:
            bool: 如果此检查点是前K个最佳之一则为True
        """
        # 获取关键指标值
        metric_value = metrics.get(self.metric_name, float('inf'))
        # 记录所有的验证结果
        self.metric_history.append((epoch, metric_value))
        
        # 如果堆中少于k个元素，或者这个检查点比堆中最差的要好
        if len(self.best_checkpoints) < self.k:
            # 保存检查点
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_path)
            
            # 添加到堆中（取反值以实现最小堆）
            comparison_value = metric_value if not self.larger_is_better else -metric_value
            heapq.heappush(self.best_checkpoints, (comparison_value, epoch, checkpoint_path))
            logging.info(f"保存新的Top-{self.k}检查点：Epoch {epoch}, {self.metric_name} = {metric_value}")
            return True
        else:
            # 检查是否比当前最差的检查点更好
            worst_value, worst_epoch, worst_path = self.best_checkpoints[0]
            comparison_value = metric_value if not self.larger_is_better else -metric_value
            
            if comparison_value < worst_value:  # 更好的检查点
                # 删除最差的检查点文件
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    
                # 保存新的检查点
                checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_path)
                
                # 更新堆
                heapq.heapreplace(self.best_checkpoints, (comparison_value, epoch, checkpoint_path))
                logging.info(f"替换检查点: Epoch {worst_epoch} -> Epoch {epoch}, {self.metric_name} 从 {worst_value if not self.larger_is_better else -worst_value} 改善到 {metric_value}")
                return True
        return False

    def save_metric_history(self, save_path):
        """保存所有验证指标的历史记录"""
        history_path = os.path.join(save_path, "validation_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.metric_history, f)
        
    def get_best_checkpoint(self):
        """获取最佳检查点的路径"""
        if not self.best_checkpoints:
            return None
        
        # 找到最佳的检查点（堆中的最好的那个）
        best_value, best_epoch, best_path = max(self.best_checkpoints) if self.larger_is_better else min(self.best_checkpoints)
        return best_path


class Lite(LightningLite):
    """Lightning Lite训练器
    
    使用Lightning Lite框架实现的训练器，支持分布式训练
    """
    def run(self, args, args_1):
        """执行训练流程
        
        Args:
            args: 命令行参数
        """
        self.seed_everything(0)  # 固定随机种子

        # 初始化评估器 dqr here to redefine evaluator
        # evaluator = Evaluator()

        # 设置可视化
        eval_vis_cfg = {
            "visualize_interval": 0,  # 0表示不可视化
            "exp_dir": args.ckpt_path,
        }
        eval_vis_cfg = DefaultMunch.fromDict(eval_vis_cfg, object())
        # evaluator.setup_visualization(eval_vis_cfg)

        # 创建模型并移至GPU dqr 这里要加并行训练
        model = OurStereo(args_1)
        model.cuda()
        foundationstereo_ckpt_dir = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
        foundationstereo_ckpt = torch.load(foundationstereo_ckpt_dir, weights_only=False)
        model.foundation_stereo.load_state_dict(foundationstereo_ckpt['model'])

        # 保存训练配置
        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        # 获取数据加载器
        train_loader = fetch_dataloader(args, is_train=True)
        val_loader = fetch_dataloader(args, is_train=False)
        
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        val_loader = self.setup_dataloaders(val_loader, move_to_device=False)

        logging.info(f"Train loader size: {len(train_loader)}")
        logging.info(f"Val loader size: {len(val_loader)}")

        # 获取优化器和调度器
        optimizer, scheduler = fetch_optimizer(args, model)
        print("Parameter Count:", count_parameters(model))
        logging.info(f"Parameter Count: {count_parameters(model)}")
        
        # 初始化记录器和检查点追踪器
        logger = Logger(model, scheduler, args.ckpt_path)
        topk_tracker = TopKCheckpoints(k=3, metric_name='epe', larger_is_better=False)
        
        # 初始化训练状态变量
        start_epoch = 0
        total_steps = 0
        
        # 尝试从目录中加载最新检查点 dqr 这里要根据实际的 checkpoint 路径修改
        if os.path.exists(args.ckpt_path):
            folder_ckpts = [
                f
                for f in os.listdir(args.ckpt_path)
                if not os.path.isdir(os.path.join(args.ckpt_path, f)) and f.endswith(".pth") and not "final" in f
            ]
            if len(folder_ckpts) > 0:
                ckpt_path = sorted(folder_ckpts)[-1]
                ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
                logging.info(f"Loading checkpoint {ckpt_path}")
                if "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                if "epoch" in ckpt:
                    start_epoch = ckpt["epoch"] + 1
                    logging.info(f"恢复训练，从epoch {start_epoch}开始")

        # 如果指定了恢复检查点，从指定路径加载
        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
            logging.info("Loading checkpoint...")
            strict = True

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=strict)
            logging.info(f"Done loading checkpoint")
            
        # 设置模型和优化器
        model, optimizer = self.setup(model, optimizer)
        model.train()
        
        # 初始化混合精度训练的缩放器
        scaler = GradScaler(enabled=args.mixed_precision)

        # 训练循环
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # 训练一个epoch
            for i_batch, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")):
                optimizer.zero_grad()  # 清除梯度
                if batch is None:
                    print("batch is None")
                    continue
                
                left_seq, right_seq, disp_seq = batch
                # 将数据移至GPU
                left_seq = left_seq.cuda()
                right_seq = right_seq.cuda()
                disp_seq = disp_seq.cuda()

                assert model.training

                # 启用混合精度
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    # 前向传播
                    output = forward_batch(left_seq, right_seq, disp_seq, model)

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
                
                # 反向传播
                self.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
                total_steps += 1

            # 每个epoch结束后的处理
            epoch_loss /= max(num_batches, 1)  # 计算平均epoch损失
            logging.info(f"Epoch {epoch}/{args.num_epochs} - 平均损失: {epoch_loss:.4f}")
            logger.writer.add_scalar("epoch_loss", epoch_loss, epoch)
            
            # 更新学习率
            scheduler.step()
            
            # 每隔固定epoch进行验证
            if (epoch + 1) % args.eval_epochs == 0 or epoch == 0:
                logging.info(f"Epoch {epoch}: 开始验证...")
                model.eval()
                val_metrics = self.validate(model, val_loader)
                
                # 记录验证指标
                for metric_name, metric_value in val_metrics.items():
                    logger.writer.add_scalar(f"val_{metric_name}", metric_value, epoch)
                
                logging.info(f"验证指标: {val_metrics}")
                
                # 根据验证结果更新最佳检查点
                topk_tracker.update(epoch, val_metrics, model, args.ckpt_path)
                
                # 保存验证历史
                topk_tracker.save_metric_history(args.ckpt_path)
            
            # 每隔固定epoch保存普通检查点
            if args.save_regular_epochs and (epoch + 1) % args.save_epochs == 0:
                if self.global_rank == 0:
                    ckpt_path = Path(args.ckpt_path) / f"model_epoch_{epoch}_regular.pth"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )
        
        # 保存最终模型
        if self.global_rank == 0:
            ckpt_path = Path(args.ckpt_path) / f"model_final.pth"
            torch.save({"model": model.state_dict(), "epoch": args.num_epochs - 1}, ckpt_path)
            
            # 记录最佳检查点信息
            best_ckpt = topk_tracker.get_best_checkpoint()
            if best_ckpt:
                logging.info(f"最佳检查点: {best_ckpt}")
                with open(os.path.join(args.ckpt_path, "best_checkpoint.txt"), 'w') as f:
                    f.write(f"Best checkpoint: {best_ckpt}\n")
    
    def validate(self, model, val_loader):
        """
        在验证集上评估模型
        
        Args:
            model: 要评估的模型
            val_loader: 验证数据加载器
            
        Returns:
            dict: 验证指标
        """
        model.eval()
        val_loss = 0.0
        epe_sum = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch is None:
                    continue
                    
                left_seq, right_seq, disp_seq = batch
                # 将数据移至GPU
                left_seq = left_seq.cuda()
                right_seq = right_seq.cuda()
                disp_seq = disp_seq.cuda()
                
                # 前向传播，设置test_mode=True获取最终预测
                # 返回形状为 [b, T, h, w]
                final_disparities = model(left_seq, right_seq, test_mode=True)
                
                # 计算整个批次的平均EPE (End-Point-Error)
                epe = (final_disparities - disp_seq).abs().mean().item()
                epe_sum += epe * left_seq.size(0)
                num_samples += left_seq.size(0)
        
        # 计算平均指标
        avg_epe = epe_sum / max(num_samples, 1)
        
        return {'epe': avg_epe}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据集参数
    parser.add_argument("--train_dataset_path", type=str, default="/Users/dengqinrui/Downloads/Stereo Video Depth/MPI-Sintel-stereo-training-20150305/training", help="训练数据集路径")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="验证数据集路径，如不指定则使用训练集路径")
    parser.add_argument("--crop_size", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--wdecay", type=float, default=0.00001)
    parser.add_argument("--num_epochs", type=int, default=100, help="训练的总epoch数")
    parser.add_argument("--save_epochs", type=int, default=10, help="每隔多少个epoch保存一次普通检查点")
    parser.add_argument("--save_regular_epochs", action="store_true", help="是否定期保存普通检查点，无论验证结果")
    parser.add_argument("--eval_epochs", type=int, default=5, help="每隔多少个epoch验证一次")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--validate_at_start", action="store_true")
    
    # 模型参数
    parser.add_argument("--restore_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/our_stereo")
    parser.add_argument("--freeze_encoder", action="store_true")
    
    args = parser.parse_args()
    
    print("args.crop_size:", args.crop_size)
    
    # 创建检查点目录
    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.ckpt_path, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    # 启动训练 dqr 这里要加并行训练
    # lite = Lite(devices="auto", accelerator="auto", precision=16 if args.mixed_precision else 32)
    lite = Lite(devices=[0], accelerator="cuda", precision=16 if args.mixed_precision else 32)
    if True:
        path = "/home/shizl/3DV_Video_Depth_Estimation_2025/FoundationStereo/pretrained_models/23-51-11/cfg.yaml"
        cfg = OmegaConf.load(path)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        args_1 = OmegaConf.create(cfg)
        args_1.valid_iters = 32

    lite.run(args, args_1)

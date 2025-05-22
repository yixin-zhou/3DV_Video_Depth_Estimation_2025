"""
StereoAnyVideo模型训练脚本

这个脚本实现了StereoAnyVideo立体视频深度估计模型的训练流程，包括：
1. 数据加载和预处理
2. 模型训练和优化
3. 损失计算和评估
4. 模型保存和恢复
"""

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
from pytorch_lightning.lite import LightningLite
from torch.cuda.amp import GradScaler

from stereoanyvideo.train_utils.utils import (
    # run_test_eval,
    save_ims_to_tb,
    count_parameters,
)
from stereoanyvideo.train_utils.logger import Logger

from stereoanyvideo.evaluation.core.evaluator import Evaluator
from stereoanyvideo.train_utils.losses import sequence_loss, temporal_loss
import stereoanyvideo.datasets.video_datasets as datasets
from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo


def fetch_optimizer(args, model):
    """创建优化器和学习率调度器
    
    注意：冻结VDA特征提取器的参数，只训练模型的其他部分
    
    Args:
        args: 命令行参数
        model: 待优化的模型
        
    Returns:
        tuple: (optimizer, scheduler) - 优化器和学习率调度器
    """
    # 冻结Video-Depth-Anything特征提取器
    for name, param in model.named_parameters():
        if any([key in name for key in ['depthanything']]):
            param.requires_grad_(False)
    # 创建AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    # 使用OneCycleLR学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


def forward_batch(batch, model, args):
    """处理一个批次数据的前向传播
    
    将批次数据送入模型，计算损失和指标
    
    Args:
        batch (dict): 包含图像和视差标签的批次数据
        model (nn.Module): StereoAnyVideo模型
        args: 命令行参数
        
    Returns:
        dict: 包含损失值、指标和预测结果的字典
    """
    output = {}
    # 模型前向传播获取视差预测
    disparities = model(
        batch["img"][:, :, 0],  # 左视图
        batch["img"][:, :, 1],  # 右视图
        iters=args.train_iters,
        test_mode=False,
    )
    # 处理每条轨迹的损失
    num_traj = len(batch["disp"][0])
    for i in range(num_traj):
        # 计算序列损失
        seq_loss, metrics = sequence_loss(
            disparities[:, i], -batch["disp"][:, i, 0], batch["valid_disp"][:, i, 0])
        output[f"disp_{i}"] = {"loss": seq_loss / num_traj, "metrics": metrics}

    # 保存最终预测结果
    output["disparity"] = {
        "predictions": torch.cat(
            [disparities[-1, i] for i in range(num_traj)], dim=1).detach(),
    }
    return output

class Lite(LightningLite):
    """Lightning Lite训练器
    
    使用Lightning Lite框架实现的训练器，支持分布式训练
    """
    def run(self, args):
        """执行训练流程
        
        Args:
            args: 命令行参数
        """
        self.seed_everything(0)  # 固定随机种子

        # 初始化评估器
        evaluator = Evaluator()

        # 设置可视化
        eval_vis_cfg = {
            "visualize_interval": 0,  # 0表示不可视化
            "exp_dir": args.ckpt_path,
        }
        eval_vis_cfg = DefaultMunch.fromDict(eval_vis_cfg, object())
        evaluator.setup_visualization(eval_vis_cfg)

        # 创建模型并移至GPU
        model = StereoAnyVideo()
        model.cuda()

        # 保存训练配置
        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        # 获取数据加载器
        train_loader = datasets.fetch_dataloader(args)
        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)

        logging.info(f"Train loader size:  {len(train_loader)}")

        # 获取优化器和调度器
        optimizer, scheduler = fetch_optimizer(args, model)
        print("Parameter Count:", {count_parameters(model)})
        logging.info(f"Parameter Count:  {count_parameters(model)}")
        total_steps = 0
        logger = Logger(model, scheduler, args.ckpt_path)

        # 尝试从目录中加载最新检查点
        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        # 如果指定了恢复检查点，从指定路径加载
        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")
            strict = True

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=strict)

            logging.info(f"Done loading checkpoint")
            
        # 设置模型和优化器
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        model.cuda()
        model.train()
        model.module.module.freeze_bn()  # 冻结BatchNorm层

        # 初始化混合精度训练的缩放器
        scaler = GradScaler(enabled=args.mixed_precision)

        # 训练循环
        should_keep_training = True
        global_batch_num = 0
        epoch = -1
        while should_keep_training:
            epoch += 1

            for i_batch, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()  # 清除梯度
                if batch is None:
                    print("batch is None")
                    continue
                # 将数据移至GPU
                for k, v in batch.items():
                    batch[k] = v.cuda()

                assert model.training

                # 前向传播
                output = forward_batch(batch, model, args)

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

                # 记录训练指标
                if self.global_rank == 0:
                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1
                    
                self.barrier()  # 同步分布式进程
                
                # 反向传播
                self.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

                # 优化器步骤
                scaler.step(optimizer)
                if total_steps < args.num_steps:
                    scheduler.step()
                scaler.update()
                total_steps += 1

                # 保存检查点
                if self.global_rank == 0:
                    if (total_steps % args.save_steps == 0) or (total_steps == 1 and args.validate_at_start):
                        ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                        save_path = Path(
                            f"{args.ckpt_path}/model_{args.name}_{ckpt_iter}.pth"
                        )

                        save_dict = {
                            "model": model.module.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "total_steps": total_steps,
                        }

                        logging.info(f"Saving file {save_path}")
                        self.save(save_dict, save_path)

                self.barrier()

                # 检查是否应该停止训练
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        # 保存最终模型
        logger.close()
        PATH = f"{args.ckpt_path}/{args.name}_final.pth"
        torch.save(model.module.module.state_dict(), PATH)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="StereoAnyVideo", help="name your experiment")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    # 训练参数
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size used during training."
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=["things", "monkaa", "driving"],
        help="training datasets.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="max learning rate.")

    parser.add_argument(
        "--num_steps", type=int, default=80000, help="length of training schedule."
    )
    parser.add_argument(
        "--save_steps", type=int, default=3000, help="length of training schedule."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        default=[320, 720],
        help="size of the random image crops used during training.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=12,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )

    parser.add_argument(
        "--sample_len", type=int, default=5, help="length of training video samples"
    )
    parser.add_argument(
        "--validate_at_start", action="store_true", help="validate the model at start"
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate every n epoch",
    )

    parser.add_argument(
        "--num_workers", type=int, default=6, help="number of dataloader workers."
    )
    # 验证参数
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of updates to the disparity field in each forward pass during validation.",
    )
    # 数据增强参数
    parser.add_argument(
        "--img_gamma", type=float, nargs="+", default=None, help="gamma range"
    )
    parser.add_argument(
        "--saturation_range",
        type=float,
        nargs="+",
        default=None,
        help="color saturation",
    )
    parser.add_argument(
        "--do_flip",
        default=False,
        choices=["h", "v"],
        help="flip the images horizontally or vertically",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        nargs="+",
        default=[0, 0],
        help="re-scale the images randomly",
    )
    parser.add_argument(
        "--noyjitter",
        action="store_true",
        help="don't simulate imperfect rectification",
    )
    args = parser.parse_args()

    # 创建检查点目录
    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        filename=args.ckpt_path + '/' + args.name + '.log',
        filemode='a',
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    # 启动分布式训练
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        accelerator="gpu",
        precision=32,
    ).run(args)

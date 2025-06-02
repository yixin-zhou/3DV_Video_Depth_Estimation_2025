#!/bin/bash

# 首先清理所有训练进程
echo "清理所有训练进程..."
pkill -f "Train_dynamic_replica"
sleep 10

# 清理CUDA缓存
python -c "import torch; torch.cuda.empty_cache(); print('CUDA缓存已清理')"

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# 设置离线模式环境变量，避免网络下载
export TORCH_HOME="/home/shizl/.cache/torch"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置数据路径
TRAIN_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted"
VAL_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted_val"

# 使用之前的检查点目录
CKPT_PATH="checkpoints/dynamic_replica_20250601_085103"

echo "从最佳检查点恢复训练"
echo "训练数据路径: ${TRAIN_DATA_PATH}"
echo "验证数据路径: ${VAL_DATA_PATH}"
echo "检查点目录: ${CKPT_PATH}"

# 检查检查点目录是否存在
if [ ! -d "${CKPT_PATH}" ]; then
    echo "错误: 检查点目录不存在: ${CKPT_PATH}"
    exit 1
fi

# 列出可用的检查点
echo "可用的检查点:"
ls -la ${CKPT_PATH}/*.pth

# 设置GPU可见设备（使用相邻的GPU减少通信延迟）
export CUDA_VISIBLE_DEVICES=3,4,5,6

# 设置NCCL环境变量解决超时问题
export NCCL_TIMEOUT=7200  # 增加超时时间到2小时
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=1  # 禁用P2P通信，使用PCIe
export NCCL_SHM_DISABLE=1  # 禁用共享内存
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环接口
export NCCL_DEBUG=WARN  # 启用警告级别调试信息
export NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理

# 内存优化配置 (移除expandable_segments避免内部错误)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# PyTorch分布式设置
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "使用2个相邻GPU进行训练: GPU 3, 4"
echo "NCCL配置: 超时7200s, 禁用P2P/IB/SHM, 使用PCIe通信"

# 从最佳检查点恢复训练，使用更保守的参数
python FoundationStereo/Train_dynamic_replica.py \
  --train_dataset_path ${TRAIN_DATA_PATH} \
  --val_dataset_path ${VAL_DATA_PATH} \
  --crop_size 256 256 \
  --batch_size 1 \
  --accumulate_grad_batches 2 \
  --num_workers 2 \
  --lr 0.000005 \
  --wdecay 0.00001 \
  --num_epochs 200 \
  --save_epochs 200 \
  --eval_epochs 3 \
  --mixed_precision \
  --scheduler cosine \
  --gradient_clip 0.1 \
  --early_stopping \
  --patience 20 \
  --ckpt_path ${CKPT_PATH} \
  --resume_from_best \
  --save_regular_epochs \
  --validate_at_start \
  --tensorboard

echo "训练完成！检查点保存在: ${CKPT_PATH}"

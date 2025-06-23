#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# 设置离线模式环境变量，避免网络下载
export TORCH_HOME="/home/shizl/.cache/torch"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置数据路径
TRAIN_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted"
VAL_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/extracted_val"

# 设置检查点保存路径
CKPT_PATH="checkpoints/dynamic_replica_$(date +%Y%m%d_%H%M%S)"

# 创建检查点目录
mkdir -p ${CKPT_PATH}

echo "训练数据路径: ${TRAIN_DATA_PATH}"
echo "验证数据路径: ${VAL_DATA_PATH}"
echo "检查点保存路径: ${CKPT_PATH}"

# 设置GPU可见设备（使用2张空闲GPU进行并行训练）
export CUDA_VISIBLE_DEVICES=3,4,5

# 训练参数
python FoundationStereo/Train_dynamic_replica.py \
  --train_dataset_path ${TRAIN_DATA_PATH} \
  --val_dataset_path ${VAL_DATA_PATH} \
  --crop_size 256 256 \
  --batch_size 1 \
  --accumulate_grad_batches 6 \
  --num_workers 8 \
  --lr 0.00005 \
  --wdecay 0.00001 \
  --num_epochs 200 \
  --save_epochs 25 \
  --eval_epochs 5 \
  --mixed_precision \
  --scheduler cosine \
  --gradient_clip 0.5 \
  --early_stopping \
  --patience 20 \
  --ckpt_path ${CKPT_PATH} \
  --save_regular_epochs \
  --validate_at_start \
  --tensorboard

echo "训练完成！检查点保存在: ${CKPT_PATH}"

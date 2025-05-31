#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# 设置训练数据集路径
TRAIN_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/MPI-Sintel-stereo-training-20150305/training"
VAL_DATA_PATH="/home/shizl/3DV_Video_Depth_Estimation_2025/data/MPI-Sintel-stereo-training-20150305/val"  # 如果不指定，将使用训练数据集路径

# 设置模型检查点保存路径
CKPT_PATH="checkpoints/our_stereo"

# 设置GPU可见设备（如果需要指定GPU）
export CUDA_VISIBLE_DEVICES=3

# 训练参数
python FoundationStereo/Train_our_model.py \
  --train_dataset_path ${TRAIN_DATA_PATH} \
  --val_dataset_path ${VAL_DATA_PATH} \
  --crop_size 256 256 \
  --batch_size 1 \
  --num_workers 16 \
  --lr 0.00005 \
  --wdecay 0.002 \
  --num_epochs 1000 \
  --save_epochs 10 \
  --eval_epochs 5 \
  --mixed_precision \
  --ckpt_path ${CKPT_PATH} \
  # --save_regular_epochs \

# 取消注释下面的行以恢复训练
# --restore_ckpt ${CKPT_PATH}/model_epoch_X.pth \

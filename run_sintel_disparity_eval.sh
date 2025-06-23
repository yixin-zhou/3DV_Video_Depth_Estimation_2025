#!/bin/bash

# Sintel视差评估脚本
# 计算TEPE和EPE指标

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# 设置离线模式
export TORCH_HOME="/home/shizl/.cache/torch"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置GPU
export CUDA_VISIBLE_DEVICES=3

echo "=" * 60
echo "Sintel视差评估 - market_5和mountain_1"
echo "=" * 60

# 检查点路径
CHECKPOINT="checkpoints/dynamic_replica_20250601_085103/best_epoch_044_depth_error_0.9860.pth"

# 检查检查点是否存在
if [ ! -f "${CHECKPOINT}" ]; then
    echo "错误: 检查点文件不存在: ${CHECKPOINT}"
    echo "可用的检查点:"
    ls -la checkpoints/dynamic_replica_20250601_085103/*.pth
    exit 1
fi

echo "使用检查点: ${CHECKPOINT}"

# 检查Sintel数据集
SINTEL_ROOT="data/MPI-Sintel-stereo-training-20150305"
if [ ! -d "${SINTEL_ROOT}" ]; then
    echo "错误: Sintel数据集不存在: ${SINTEL_ROOT}"
    exit 1
fi

echo "Sintel数据集: ${SINTEL_ROOT}"

# 检查目标场景
SCENES=("market_5" "mountain_1")
for scene in "${SCENES[@]}"; do
    left_dir="${SINTEL_ROOT}/training/clean_left/${scene}"
    right_dir="${SINTEL_ROOT}/training/clean_right/${scene}"
    disp_dir="${SINTEL_ROOT}/training/disparities/${scene}"
    
    if [ ! -d "${left_dir}" ] || [ ! -d "${right_dir}" ] || [ ! -d "${disp_dir}" ]; then
        echo "错误: 场景 ${scene} 的数据不完整"
        echo "  左目: ${left_dir}"
        echo "  右目: ${right_dir}"
        echo "  视差: ${disp_dir}"
        exit 1
    fi
    
    # 检查帧数
    frame_count=$(ls "${left_dir}"/*.png 2>/dev/null | wc -l)
    echo "场景 ${scene}: ${frame_count} 帧"
done

# 输出文件
OUTPUT_FILE="sintel_disparity_eval_results_$(date +%Y%m%d_%H%M%S).json"

echo "输出文件: ${OUTPUT_FILE}"
echo "指标计算: EPE和TEPE"
echo ""

# 运行评估
/home/shizl/anaconda3/envs/langchain/bin/python evaluation/evaluate_sintel.py \
    --checkpoint "${CHECKPOINT}" \
    --device cuda \
    --batch_size 5 \
    --output_file "${OUTPUT_FILE}"

echo ""
echo "评估完成！"
echo "结果保存在: ${OUTPUT_FILE}"

# 显示结果摘要
if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "结果摘要:"
    /home/shizl/anaconda3/envs/langchain/bin/python -c "
import json
with open('${OUTPUT_FILE}', 'r') as f:
    results = json.load(f)
print(f'总场景数: {len(results[\"scenes\"])}')
print(f'成功场景: {results[\"successful_scenes\"]}')
print(f'失败场景: {results[\"failed_scenes\"]}')
print()
for result in results['scene_results']:
    if 'error' not in result:
        scene = result['scene_name']
        epe = result.get('avg_epe', 0)
        tepe = result.get('avg_tepe', 0)
        frames = result.get('total_frames', 0)
        time_taken = result.get('processing_time', 0)
        print(f'{scene}:')
        print(f'  处理帧数: {frames}')
        print(f'  平均EPE: {epe:.6f}')
        print(f'  平均TEPE: {tepe:.6f}')
        print(f'  处理时间: {time_taken:.2f}秒')
        print()
"
fi

#!/usr/bin/env python3
"""
Sintel视差评估脚本
专门评估market_5和mountain_1场景
计算TEPE和EPE指标
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import logging
from tqdm import tqdm
import time
import json

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'FoundationStereo'))

from FoundationStereo.core.our_stereo import OurStereo


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    logging.info(f"加载模型检查点: {checkpoint_path}")
    
    # 创建模型参数对象
    from omegaconf import OmegaConf
    
    # 使用与训练时相同的配置
    args = OmegaConf.create({
        'hidden_dims': [128, 128, 128],
        'n_gru_layers': 3,
        'n_downsample': 2,
        'max_disp': 416,
        'corr_levels': 2,
        'corr_radius': 4,
        'corr_implementation': "reg",
        'slow_fast_gru': False,
        'train_iters': 22,
        'valid_iters': 32,
        'mixed_precision': True,
        'vit_size': 'vitl'
    })
    
    # 创建模型
    model = OurStereo(args)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除DDP包装的前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    logging.info("模型加载完成")
    return model


def load_camera_params(scene_name, frame_idx):
    """加载相机参数"""
    cam_file = f"data/MPI-Sintel-stereo-training-20150305/camdata_left/{scene_name}/frame_{frame_idx:04d}.cam"
    
    if not os.path.exists(cam_file):
        logging.warning(f"相机参数文件不存在: {cam_file}")
        # 返回默认参数
        return {'focal_length': 1050.0, 'baseline': 0.1}
    
    try:
        # 读取相机参数文件
        with open(cam_file, 'r') as f:
            lines = f.readlines()
        
        # 解析相机参数 (简化版本，根据实际格式调整)
        # 这里使用默认值，实际应该解析文件内容
        focal_length = 1050.0  # 默认焦距
        baseline = 0.1  # 默认基线 (米)
        
        return {'focal_length': focal_length, 'baseline': baseline}
        
    except Exception as e:
        logging.warning(f"解析相机参数失败: {e}")
        return {'focal_length': 1050.0, 'baseline': 0.1}


def crop_center(image, target_size=(512, 512)):
    """中心裁剪图像到目标尺寸"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 计算裁剪区域
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    # 确保不超出边界
    start_h = max(0, start_h)
    start_w = max(0, start_w)
    end_h = min(h, start_h + target_h)
    end_w = min(w, start_w + target_w)
    
    # 裁剪
    if len(image.shape) == 3:
        cropped = image[start_h:end_h, start_w:end_w, :]
    else:
        cropped = image[start_h:end_h, start_w:end_w]
    
    # 如果裁剪后尺寸不足，进行填充
    if cropped.shape[:2] != target_size:
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            padded[:cropped.shape[0], :cropped.shape[1], :] = cropped
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
        cropped = padded
    
    return cropped


def load_and_crop_scene_batch(scene_name, start_frame=0, batch_size=10, target_size=(512, 512)):
    """加载并裁剪场景批次数据"""
    
    # 数据路径
    left_dir = f"data/MPI-Sintel-stereo-training-20150305/training/clean_left/{scene_name}"
    right_dir = f"data/MPI-Sintel-stereo-training-20150305/training/clean_right/{scene_name}"
    disp_dir = f"data/MPI-Sintel-stereo-training-20150305/training/disparities/{scene_name}"
    
    left_frames = []
    right_frames = []
    gt_disparities = []
    
    logging.info(f"加载场景 {scene_name} 的帧 {start_frame}-{start_frame+batch_size-1}")
    
    for i in range(batch_size):
        frame_idx = start_frame + i + 1  # Sintel帧编号从1开始
        
        # 加载图像
        left_file = os.path.join(left_dir, f"frame_{frame_idx:04d}.png")
        right_file = os.path.join(right_dir, f"frame_{frame_idx:04d}.png")
        disp_file = os.path.join(disp_dir, f"frame_{frame_idx:04d}.png")
        
        if not all(os.path.exists(f) for f in [left_file, right_file, disp_file]):
            logging.warning(f"文件不存在，跳过帧 {frame_idx}")
            break
        
        # 读取图像
        left_img = cv2.imread(left_file)
        right_img = cv2.imread(right_file)
        disp_img = cv2.imread(disp_file, cv2.IMREAD_ANYDEPTH)
        
        if any(img is None for img in [left_img, right_img, disp_img]):
            logging.warning(f"无法读取图像，跳过帧 {frame_idx}")
            break
        
        # 转换为RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # 视差图缩放
        disp_img = disp_img.astype(np.float32) / 256.0
        
        # 中心裁剪到相同区域
        left_cropped = crop_center(left_img, target_size)
        right_cropped = crop_center(right_img, target_size)
        disp_cropped = crop_center(disp_img, target_size)
        
        left_frames.append(left_cropped)
        right_frames.append(right_cropped)
        gt_disparities.append(disp_cropped)
    
    if len(left_frames) == 0:
        return None, None, None
    
    return left_frames, right_frames, gt_disparities


def preprocess_frames(left_frames, right_frames):
    """预处理图像序列"""
    def normalize_frames(frames):
        processed = []
        for frame in frames:
            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0
            # 转换为 CHW 格式
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)
        return np.stack(processed, axis=0)  # [T, 3, H, W]
    
    left_tensor = normalize_frames(left_frames)
    right_tensor = normalize_frames(right_frames)
    
    # 添加batch维度: [1, T, 3, H, W]
    left_tensor = torch.from_numpy(left_tensor).unsqueeze(0)
    right_tensor = torch.from_numpy(right_tensor).unsqueeze(0)
    
    return left_tensor, right_tensor


def run_inference(model, left_tensor, right_tensor, device='cuda'):
    """运行推理"""
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)
    
    with torch.no_grad():
        # 运行推理
        depth_maps = model(left_tensor, right_tensor, test_mode=True)
    
    return depth_maps


def depth_to_disparity(depth_maps, focal_length, baseline):
    """将深度转换为视差"""
    # 避免除零
    depth_maps = np.maximum(depth_maps, 1e-6)
    disparity = (focal_length * baseline) / depth_maps
    return disparity


def calculate_epe(pred_disp, gt_disp):
    """计算EPE (End-Point Error)"""
    # 计算像素级误差
    error = np.abs(pred_disp - gt_disp)
    epe = np.mean(error)
    return epe


def calculate_tepe(pred_disp, gt_disp):
    """
    计算TEPE (Temporal End-Point Error)
    
    Args:
        pred_disp: 预测视差序列 [T, H, W]
        gt_disp: ground truth视差序列 [T, H, W]
    
    Returns:
        tepe: TEPE值
    """
    T = pred_disp.shape[0]
    if T < 2:
        return 0.0
    
    tepe_sum = 0.0
    valid_pairs = 0
    
    for t in range(T - 1):
        # 计算预测视差的时序变化
        pred_diff = pred_disp[t] - pred_disp[t + 1]  # [H, W]
        
        # 计算ground truth视差的时序变化
        gt_diff = gt_disp[t] - gt_disp[t + 1]  # [H, W]
        
        # 计算差异
        temporal_error = pred_diff - gt_diff  # [H, W]
        
        # 平方误差
        squared_error = temporal_error ** 2  # [H, W]
        
        # 累积误差
        tepe_sum += squared_error.sum()
        valid_pairs += squared_error.size
    
    # 计算TEPE
    if valid_pairs > 0:
        tepe = np.sqrt(tepe_sum / valid_pairs)
    else:
        tepe = 0.0
    
    return tepe


def save_depth_and_disparity_maps(depth_maps, pred_disparities, gt_disparities,
                                  scene_name, batch_num, frame_range, output_dir="sintel_results"):
    """保存深度图和视差图"""

    # 创建输出目录
    scene_output_dir = os.path.join(output_dir, f"{scene_name}_batch_{batch_num:02d}")
    os.makedirs(scene_output_dir, exist_ok=True)

    logging.info(f"保存结果到: {scene_output_dir}")

    for i in range(depth_maps.shape[0]):
        frame_idx = frame_range[0] + i

        # 保存深度图
        depth_map = depth_maps[i]

        # 原始深度值 (.npy)
        np.save(os.path.join(scene_output_dir, f"depth_frame_{frame_idx:04d}.npy"), depth_map)

        # 深度图可视化 (.png)
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(scene_output_dir, f"depth_frame_{frame_idx:04d}.png"), depth_vis)

        # 保存预测视差图
        pred_disp = pred_disparities[i]

        # 原始视差值 (.npy)
        np.save(os.path.join(scene_output_dir, f"pred_disp_frame_{frame_idx:04d}.npy"), pred_disp)

        # 视差图可视化 (.png)
        disp_normalized = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
        disp_vis = (disp_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(scene_output_dir, f"pred_disp_frame_{frame_idx:04d}.png"), disp_vis)

        # 保存ground truth视差图 (参考)
        gt_disp = gt_disparities[i]
        np.save(os.path.join(scene_output_dir, f"gt_disp_frame_{frame_idx:04d}.npy"), gt_disp)

        # GT视差图可视化
        gt_disp_normalized = (gt_disp - gt_disp.min()) / (gt_disp.max() - gt_disp.min())
        gt_disp_vis = (gt_disp_normalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(scene_output_dir, f"gt_disp_frame_{frame_idx:04d}.png"), gt_disp_vis)

    logging.info(f"保存了 {depth_maps.shape[0]} 帧的深度图和视差图")
    return scene_output_dir


def evaluate_scene(model, scene_name, device='cuda', batch_size=10, save_results=True):
    """评估单个场景"""
    logging.info("=" * 60)
    logging.info(f"评估场景: {scene_name}")
    logging.info("=" * 60)
    
    start_time = time.time()
    all_batch_results = []
    
    try:
        current_frame = 0
        batch_num = 0
        total_frames = 50  # 已知这两个场景都是50帧
        
        while current_frame < total_frames:
            batch_num += 1
            logging.info(f"\n--- 批次 {batch_num} ---")
            
            # 加载并裁剪当前批次
            left_frames, right_frames, gt_disparities = load_and_crop_scene_batch(
                scene_name, current_frame, batch_size
            )
            
            if left_frames is None:
                break
            
            # 预处理
            left_tensor, right_tensor = preprocess_frames(left_frames, right_frames)
            
            logging.info(f"处理张量尺寸: {left_tensor.shape}")
            
            # 推理
            depth_maps = run_inference(model, left_tensor, right_tensor, device)
            depth_maps_np = depth_maps.cpu().numpy()[0]  # [T, H, W]
            
            # 转换为视差
            # 使用第一帧的相机参数（简化处理）
            cam_params = load_camera_params(scene_name, current_frame + 1)
            focal_length = cam_params['focal_length']
            baseline = cam_params['baseline']
            
            pred_disparities = depth_to_disparity(depth_maps_np, focal_length, baseline)
            gt_disparities_np = np.stack(gt_disparities, axis=0)
            
            # 计算指标
            epe = calculate_epe(pred_disparities, gt_disparities_np)
            tepe = calculate_tepe(pred_disparities, gt_disparities_np)

            # 保存深度图和视差图
            if save_results:
                frame_range = [current_frame, current_frame + len(left_frames) - 1]
                output_dir = save_depth_and_disparity_maps(
                    depth_maps_np, pred_disparities, gt_disparities_np,
                    scene_name, batch_num, frame_range
                )

            batch_result = {
                'batch_num': batch_num,
                'frame_range': [current_frame, current_frame + len(left_frames) - 1],
                'num_frames': len(left_frames),
                'epe': float(epe),
                'tepe': float(tepe),
                'focal_length': focal_length,
                'baseline': baseline,
                'output_dir': output_dir if save_results else None
            }
            
            all_batch_results.append(batch_result)
            
            logging.info(f"批次 {batch_num} 结果:")
            logging.info(f"  EPE: {epe:.6f}")
            logging.info(f"  TEPE: {tepe:.6f}")
            
            # 清理GPU内存
            del left_tensor, right_tensor, depth_maps, depth_maps_np
            torch.cuda.empty_cache()
            
            current_frame += len(left_frames)
        
        # 计算整个场景的平均指标
        if all_batch_results:
            avg_epe = np.mean([r['epe'] for r in all_batch_results])
            avg_tepe = np.mean([r['tepe'] for r in all_batch_results])
        else:
            avg_epe = avg_tepe = 0.0
        
        scene_result = {
            'scene_name': scene_name,
            'total_frames': current_frame,
            'num_batches': len(all_batch_results),
            'avg_epe': float(avg_epe),
            'avg_tepe': float(avg_tepe),
            'batch_results': all_batch_results,
            'processing_time': time.time() - start_time
        }
        
        logging.info(f"\n场景 {scene_name} 评估完成")
        logging.info(f"总处理时间: {scene_result['processing_time']:.2f} 秒")
        logging.info(f"平均EPE: {avg_epe:.6f}")
        logging.info(f"平均TEPE: {avg_tepe:.6f}")
        
        return scene_result, True
        
    except Exception as e:
        logging.error(f"场景 {scene_name} 评估失败: {e}")
        return {'scene_name': scene_name, 'error': str(e)}, False


def main():
    parser = argparse.ArgumentParser(description="Sintel视差评估 - market_5和mountain_1")
    
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/dynamic_replica_20250601_085103/best_epoch_044_depth_error_0.9860.pth",
                       help="模型检查点路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="批次大小")
    parser.add_argument("--output_file", type=str, default="sintel_disparity_eval_results.json",
                       help="结果输出文件")
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="是否保存深度图和视差图")
    parser.add_argument("--results_dir", type=str, default="sintel_results",
                       help="结果保存目录")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("Sintel视差评估 - market_5和mountain_1")
    logging.info("=" * 60)
    logging.info(f"检查点: {args.checkpoint}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"设备: {args.device}")
    
    # 加载模型
    model = load_model(args.checkpoint, args.device)
    
    # 评估场景
    scenes = ['market_5', 'mountain_1']
    all_results = []
    successful_scenes = []
    failed_scenes = []
    
    for i, scene_name in enumerate(scenes, 1):
        logging.info(f"\n进度: {i}/{len(scenes)}")
        
        result, success = evaluate_scene(model, scene_name, args.device, args.batch_size, args.save_results)
        all_results.append(result)
        
        if success:
            successful_scenes.append(scene_name)
        else:
            failed_scenes.append(scene_name)
        
        # 场景间休息
        if i < len(scenes):
            logging.info("等待5秒后评估下一个场景...")
            torch.cuda.empty_cache()
            time.sleep(5)
    
    # 保存结果
    final_results = {
        'checkpoint': args.checkpoint,
        'scenes': scenes,
        'successful_scenes': len(successful_scenes),
        'failed_scenes': len(failed_scenes),
        'scene_results': all_results,
        'successful_scene_names': successful_scenes,
        'failed_scene_names': failed_scenes
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 最终报告
    logging.info("\n" + "=" * 60)
    logging.info("Sintel视差评估完成")
    logging.info("=" * 60)
    logging.info(f"成功评估: {len(successful_scenes)} 个场景")
    logging.info(f"失败场景: {len(failed_scenes)} 个场景")
    
    if successful_scenes:
        logging.info("成功的场景:")
        for scene in successful_scenes:
            scene_result = next(r for r in all_results if r.get('scene_name') == scene)
            logging.info(f"  ✅ {scene}: EPE={scene_result.get('avg_epe', 0):.6f}, TEPE={scene_result.get('avg_tepe', 0):.6f}")
    
    if failed_scenes:
        logging.info("失败的场景:")
        for scene in failed_scenes:
            logging.info(f"  ❌ {scene}")
    
    logging.info(f"详细结果保存在: {args.output_file}")


if __name__ == "__main__":
    main()

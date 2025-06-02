"""
Top-K 检查点管理器

用于管理训练过程中的最佳检查点，保留性能最好的K个模型
"""

import os
import json
import heapq
import torch
import logging
from typing import Dict, Any, List, Tuple


class TopKCheckpoints:
    """管理Top-K最佳检查点的类"""
    
    def __init__(self, k: int = 3, metric_name: str = 'loss', larger_is_better: bool = False):
        """
        初始化Top-K检查点管理器
        
        Args:
            k: 保留的最佳检查点数量
            metric_name: 用于比较的指标名称
            larger_is_better: True表示指标越大越好，False表示越小越好
        """
        self.k = k
        self.metric_name = metric_name
        self.larger_is_better = larger_is_better
        
        # 使用堆来维护top-k检查点
        # 对于larger_is_better=True，使用最小堆（保留最大的k个）
        # 对于larger_is_better=False，使用最大堆（保留最小的k个）
        self.checkpoints = []  # [(metric_value, epoch, checkpoint_path), ...]
        
        # 记录所有历史指标
        self.metric_history = []  # [(epoch, metrics_dict), ...]
        
        logging.info(f"初始化Top-{k}检查点管理器，指标: {metric_name}, "
                    f"{'越大越好' if larger_is_better else '越小越好'}")
    
    def _get_comparable_value(self, metric_value: float) -> float:
        """获取用于比较的值（处理larger_is_better逻辑）"""
        if self.larger_is_better:
            return metric_value  # 最小堆，直接使用原值
        else:
            return -metric_value  # 最大堆，使用负值
    
    def update(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, 
               checkpoint_dir: str) -> bool:
        """
        更新检查点
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标字典
            model: 要保存的模型
            checkpoint_dir: 检查点保存目录
            
        Returns:
            bool: 是否保存了新的最佳检查点
        """
        if self.metric_name not in metrics:
            logging.warning(f"指标 '{self.metric_name}' 不在验证结果中: {list(metrics.keys())}")
            return False
        
        metric_value = metrics[self.metric_name]
        comparable_value = self._get_comparable_value(metric_value)
        
        # 记录历史
        self.metric_history.append((epoch, metrics.copy()))
        
        # 检查是否需要保存这个检查点
        should_save = False
        
        if len(self.checkpoints) < self.k:
            # 还没有达到k个检查点，直接保存
            should_save = True
        else:
            # 已经有k个检查点，检查是否比最差的好
            worst_comparable_value = self.checkpoints[0][0]  # 堆顶是最差的
            if comparable_value > worst_comparable_value:
                should_save = True
        
        if should_save:
            # 保存检查点
            checkpoint_path = os.path.join(checkpoint_dir, f"best_epoch_{epoch:03d}_{self.metric_name}_{metric_value:.4f}.pth")
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'metric_value': metric_value
                }, checkpoint_path)
                
                logging.info(f"保存最佳检查点: {checkpoint_path} ({self.metric_name}={metric_value:.4f})")
                
                # 更新堆
                if len(self.checkpoints) < self.k:
                    heapq.heappush(self.checkpoints, (comparable_value, epoch, checkpoint_path))
                else:
                    # 移除最差的检查点
                    old_comparable_value, old_epoch, old_path = heapq.heappop(self.checkpoints)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        logging.info(f"删除旧检查点: {old_path}")
                    
                    # 添加新的检查点
                    heapq.heappush(self.checkpoints, (comparable_value, epoch, checkpoint_path))
                
                return True
                
            except Exception as e:
                logging.error(f"保存检查点失败: {e}")
                return False
        
        return False
    
    def get_best_checkpoint(self) -> Tuple[str, float]:
        """
        获取最佳检查点路径和对应的指标值
        
        Returns:
            tuple: (checkpoint_path, metric_value)
        """
        if not self.checkpoints:
            return None, None
        
        # 找到最佳的检查点
        if self.larger_is_better:
            # 对于larger_is_better=True，找最大值
            best_comparable_value, best_epoch, best_path = max(self.checkpoints)
            best_metric_value = best_comparable_value
        else:
            # 对于larger_is_better=False，找最小值
            best_comparable_value, best_epoch, best_path = max(self.checkpoints)
            best_metric_value = -best_comparable_value
        
        return best_path, best_metric_value
    
    def get_all_checkpoints(self) -> List[Tuple[str, float, int]]:
        """
        获取所有保存的检查点信息
        
        Returns:
            list: [(checkpoint_path, metric_value, epoch), ...] 按指标值排序
        """
        result = []
        for comparable_value, epoch, path in self.checkpoints:
            if self.larger_is_better:
                metric_value = comparable_value
            else:
                metric_value = -comparable_value
            result.append((path, metric_value, epoch))
        
        # 按指标值排序（最佳的在前）
        result.sort(key=lambda x: x[1], reverse=self.larger_is_better)
        return result
    
    def save_metric_history(self, checkpoint_dir: str):
        """保存指标历史到JSON文件"""
        history_path = os.path.join(checkpoint_dir, "metric_history.json")
        
        try:
            # 准备保存的数据
            save_data = {
                'metric_name': self.metric_name,
                'larger_is_better': self.larger_is_better,
                'k': self.k,
                'history': [
                    {
                        'epoch': epoch,
                        'metrics': metrics
                    }
                    for epoch, metrics in self.metric_history
                ],
                'best_checkpoints': [
                    {
                        'path': path,
                        'metric_value': metric_value,
                        'epoch': epoch
                    }
                    for path, metric_value, epoch in self.get_all_checkpoints()
                ]
            }
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"保存指标历史: {history_path}")
            
        except Exception as e:
            logging.error(f"保存指标历史失败: {e}")
    
    def load_metric_history(self, checkpoint_dir: str) -> bool:
        """从JSON文件加载指标历史"""
        history_path = os.path.join(checkpoint_dir, "metric_history.json")
        
        if not os.path.exists(history_path):
            logging.info(f"指标历史文件不存在: {history_path}")
            return False
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复历史数据
            self.metric_history = [
                (item['epoch'], item['metrics'])
                for item in data['history']
            ]
            
            # 恢复检查点信息（只恢复仍然存在的文件）
            self.checkpoints = []
            for item in data.get('best_checkpoints', []):
                path = item['path']
                if os.path.exists(path):
                    metric_value = item['metric_value']
                    epoch = item['epoch']
                    comparable_value = self._get_comparable_value(metric_value)
                    self.checkpoints.append((comparable_value, epoch, path))
            
            # 重新构建堆
            heapq.heapify(self.checkpoints)
            
            logging.info(f"加载指标历史: {len(self.metric_history)} 条记录, "
                        f"{len(self.checkpoints)} 个有效检查点")
            return True
            
        except Exception as e:
            logging.error(f"加载指标历史失败: {e}")
            return False
    
    def print_summary(self):
        """打印检查点管理器的摘要信息"""
        print(f"\n=== Top-{self.k} 检查点摘要 ===")
        print(f"指标: {self.metric_name} ({'越大越好' if self.larger_is_better else '越小越好'})")
        print(f"历史记录: {len(self.metric_history)} 个epoch")
        print(f"保存的检查点: {len(self.checkpoints)} 个")
        
        if self.checkpoints:
            print("\n最佳检查点:")
            for i, (path, metric_value, epoch) in enumerate(self.get_all_checkpoints()):
                print(f"  {i+1}. Epoch {epoch:3d}: {self.metric_name}={metric_value:.4f} -> {os.path.basename(path)}")
        
        print("=" * 40)

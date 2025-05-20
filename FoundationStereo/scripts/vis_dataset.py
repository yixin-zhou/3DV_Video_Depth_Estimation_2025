"""
数据集可视化工具 (Dataset Visualization Tool)
该脚本用于可视化立体匹配数据集的样本，包括左右视图图像和视差图。
通过matplotlib创建三窗口并排显示，帮助用户直观了解数据集内容，
用于数据集质量检查、结果分析和演示目的。
"""

import os,sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *


if __name__ == "__main__":
  # 命令行参数解析
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_path", type=str, default="./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/")
  args = parser.parse_args()

  # 读取数据集中的图像和视差图
  root = args.dataset_path
  left = imageio.imread(f'{root}/left/rgb/0000.jpg')
  right = imageio.imread(f'{root}/right/rgb/0000.jpg')
  
  # 解码并可视化视差图
  disp = depth_uint8_decoding(imageio.imread(f'{root}/left/disparity/0000.png'))
  vis = vis_disparity(disp)

  import matplotlib.pyplot as plt

  # 创建三窗口布局
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  # 显示左图
  axes[0].imshow(left)
  axes[0].set_title('Left Image')
  axes[0].axis('off')

  # 显示右图
  axes[1].imshow(right)
  axes[1].set_title('Right Image')
  axes[1].axis('off')

  # 显示视差图可视化结果
  axes[2].imshow(vis)
  axes[2].set_title('Disparity Visualization')
  axes[2].axis('off')

  # 调整布局并显示
  plt.tight_layout()
  plt.show()

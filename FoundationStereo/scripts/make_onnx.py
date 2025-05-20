"""
ONNX模型导出工具 (ONNX Model Export Tool)
该脚本用于将FoundationStereo模型导出为ONNX格式，便于跨平台部署和推理。
通过简化模型的前向传播过程，移除训练相关组件，
创建一个专注于推理的优化版本，适用于各种推理框架和硬件。
"""

import warnings, argparse, logging, os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import omegaconf, yaml, torch,pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo


class FoundationStereoOnnx(FoundationStereo):
    """
    ONNX导出专用FoundationStereo模型
    
    简化的FoundationStereo模型变体，专为ONNX导出设计:
    - 重写前向传播函数，去除额外输出
    - 固定推理参数，简化接口
    - 移除训练相关组件，专注于高效推理
    
    该变体确保导出的ONNX模型具有清晰的输入输出接口，
    便于在各种推理框架中使用。
    """
    def __init__(self, args):
        """
        初始化ONNX导出模型
        
        参数:
            args: 模型配置参数
        """
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """
        简化的前向传播函数
        
        移除额外输出和超参数选择，专注于核心推理功能:
        - 启用自动混合精度计算
        - 使用固定迭代次数
        - 仅返回最终视差图
        
        参数:
            left: 左图像
            right: 右图像
            
        返回:
            视差预测结果
        """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp



if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output/foundation_stereo.onnx', help='Path to save results.')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 禁用梯度计算
    torch.autograd.set_grad_enabled(False)

    # 加载模型配置
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    for k in args.__dict__:
      cfg[k] = args.__dict__[k]
    if 'vit_size' not in cfg:
      cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    
    # 初始化模型并加载权重
    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # 创建示例输入张量
    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()

    # 导出ONNX模型
    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        opset_version=16,
        input_names = ['left', 'right'],
        output_names = ['disp'],
        dynamic_axes={
            'left': {0 : 'batch_size'},
            'right': {0 : 'batch_size'},
            'disp': {0 : 'batch_size'}
        },
    )


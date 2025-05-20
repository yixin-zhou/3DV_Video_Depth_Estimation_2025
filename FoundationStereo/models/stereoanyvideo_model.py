from typing import ClassVar

import torch
import torch.nn.functional as F
from pytorch3d.implicitron.tools.config import Configurable
from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo


class StereoAnyVideoModel(Configurable, torch.nn.Module):
    """StereoAnyVideo模型的包装类，提供模型加载和推理接口
    
    这个类是整个StereoAnyVideo系统的主入口点，负责：
    1. 加载预训练的模型权重
    2. 处理权重字典中可能的不同格式
    3. 将模型移至GPU
    4. 提供简单的推理接口
    
    继承:
        Configurable: 提供配置功能
        torch.nn.Module: PyTorch模型基类
    """

    MODEL_CONFIG_NAME: ClassVar[str] = "StereoAnyVideoModel"  # 配置名称常量
    model_weights: str = "./checkpoints/StereoAnyVideo_MIX.pth"  # 默认预训练权重路径

    def __post_init__(self):
        """初始化方法，在Configurable.__init__之后调用
        
        完成模型实例化、权重加载和设备分配
        """
        super().__init__()

        # 设置是否使用混合精度训练/推理
        self.mixed_precision = False
        # 实例化核心StereoAnyVideo模型
        model = StereoAnyVideo(mixed_precision=self.mixed_precision)

        # 加载预训练权重
        state_dict = torch.load(self.model_weights, map_location="cpu")
        # 处理不同格式的权重字典
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {"module." + k: v for k, v in state_dict.items()}  # 处理分布式训练保存的权重
        # 严格加载权重，确保所有参数都匹配
        model.load_state_dict(state_dict, strict=True)

        # 保存模型实例，移至GPU并设为评估模式
        self.model = model
        self.model.to("cuda")
        self.model.eval()

    def forward(self, batch_dict, iters=20):
        """模型前向推理接口
        
        Args:
            batch_dict (dict): 输入数据字典，必须包含"stereo_video"键，值为双目视频张量
            iters (int, optional): 迭代次数，影响视差估计的精度. 默认: 20
            
        Returns:
            dict: 包含"disparity"键的字典，保存估计的视差图
        """
        return self.model.forward_batch_test(batch_dict, iters=iters)
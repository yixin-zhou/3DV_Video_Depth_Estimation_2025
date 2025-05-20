"""
模型库模块 - 提供StereoAnyVideo模型的配置和实例化功能

这个模块实现了模型动态加载和配置管理的功能，包括：
1. 提供统一的模型访问接口
2. 管理模型默认配置
3. 支持根据参数动态实例化模型
"""

import copy
from pytorch3d.implicitron.tools.config import get_default_args
from stereoanyvideo.models.stereoanyvideo_model import StereoAnyVideoModel

# 支持的模型列表
MODELS = [StereoAnyVideoModel]

# 模型名称到模型类的映射字典
_MODEL_NAME_TO_MODEL = {model_cls.__name__: model_cls for model_cls in MODELS}
# 模型配置名称到默认配置的映射字典
_MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG = {}
for model_cls in MODELS:
    _MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG[
        model_cls.MODEL_CONFIG_NAME
    ] = get_default_args(model_cls)

# 表示不使用任何模型的常量
MODEL_NAME_NONE = "NONE"


def model_zoo(model_name: str, **kwargs):
    """模型库函数，根据名称和参数实例化模型
    
    根据提供的模型名称和配置参数，创建并返回相应的模型实例。
    
    Args:
        model_name (str): 模型名称，应与模型类名匹配
        **kwargs: 模型配置参数
        
    Returns:
        object: 实例化的模型对象，如果model_name为"NONE"则返回None
        
    Raises:
        ValueError: 如果提供的模型名称不存在
    """
    if model_name.upper() == MODEL_NAME_NONE:
        return None

    model_cls = _MODEL_NAME_TO_MODEL.get(model_name)

    if model_cls is None:
        raise ValueError(f"No such model name: {model_name}")

    # 准备模型参数
    model_cls_params = {}
    if "model_zoo" in getattr(model_cls, "__dataclass_fields__", []):
        model_cls_params["model_zoo"] = model_zoo
    print(
        f"{model_cls.MODEL_CONFIG_NAME} model configs:",
        kwargs.get(model_cls.MODEL_CONFIG_NAME),
    )
    # 实例化并返回模型
    return model_cls(**model_cls_params, **kwargs.get(model_cls.MODEL_CONFIG_NAME, {}))


def get_all_model_default_configs():
    """获取所有模型的默认配置
    
    返回所有已注册模型的默认配置参数的深拷贝。
    
    Returns:
        dict: 模型配置名称到默认配置的映射字典
    """
    return copy.deepcopy(_MODEL_CONFIG_NAME_TO_DEFAULT_CONFIG)

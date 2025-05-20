"""
DPT深度预测模型 (Dense Prediction Transformer)
该模块实现了结合DINOv2视觉特征的DPT架构，用于单目深度估计。
基于Transformer编码器提取的多尺度特征，通过特征融合网络生成密集深度图。
DepthAnything模型在此基础上进行了优化，实现了高精度的零样本深度估计。
"""

import argparse
import torch,os,sys,pdb
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from dinov2.models.vision_transformer import vit_small,vit_base,vit_large
from depth_anything.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    """
    创建特征融合块
    
    包装FeatureFusionBlock的工厂函数，用于创建具有标准配置的特征融合模块
    
    参数:
        features: 特征通道数
        use_bn: 是否使用批量归一化
        size: 可选的输出尺寸
        
    返回:
        配置好的特征融合块实例
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    """
    DPT解码头
    
    将Transformer特征转换为深度预测的解码头:
    - 将Transformer特征映射到多尺度特征图
    - 自顶向下的特征融合路径
    - 支持单通道深度预测或多类别分割
    
    该模块是DPT架构的核心部分，负责从Transformer特征恢复空间分辨率
    并生成高质量的密集预测结果。
    """
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        """
        初始化DPT解码头
        
        参数:
            nclass: 输出通道数，深度预测为1
            in_channels: Transformer特征维度
            features: 融合模块的特征通道数
            use_bn: 是否使用批量归一化
            out_channels: 多尺度特征的通道配置
            use_clstoken: 是否使用分类标记增强特征
        """
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken

        # 投影层，将Transformer特征映射到指定通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        # 调整不同尺度特征的分辨率
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        # 分类标记读取投影
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        # 创建特征处理网络
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        # 创建多层次的特征融合网络
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        # 输出头，根据任务类型创建不同结构
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

    def forward(self, out_features, patch_h, patch_w, return_intermediate=False, patch_size=14):
        """
        前向传播
        
        参数:
            out_features: Transformer提取的特征，包含多个层级
            patch_h: 特征图高度(patch数量)
            patch_w: 特征图宽度(patch数量)
            return_intermediate: 是否返回中间层特征
            patch_size: Transformer的patch大小
            
        返回:
            如果return_intermediate为True，返回最终输出和中间层特征
            否则仅返回最终深度预测
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            # 将序列特征重塑为空间特征图
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # 投影和调整特征尺寸
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        # 提取不同层级的特征
        layer_1, layer_2, layer_3, layer_4 = out

        # 应用特征变换
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 自顶向下融合特征
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 输出预测并上采样到原始分辨率
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * patch_size), int(patch_w * patch_size)), mode="bilinear", align_corners=True)
        
        if return_intermediate:
          depth = self.scratch.output_conv2(out)
          depth = F.relu(depth)
          disp = 1/depth
          disp[depth==0] = 0
          disp = disp/disp.max()
          return out, path_1, path_2, path_3, path_4, disp

        else:
          out = self.scratch.output_conv2(out)
          return out


class DPT_DINOv2(nn.Module):
    """
    基于DINOv2的DPT模型
    
    集成DINOv2视觉Transformer和DPT解码头:
    - 使用预训练的DINOv2作为特征提取骨干网络
    - 支持不同规模的ViT模型(小型、基础、大型)
    - 通过DPT解码头生成深度预测
    
    该模型结合了DINOv2强大的视觉表示能力和DPT的密集预测能力，
    为深度估计任务提供了强大的基础架构。
    """
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, pretrained_dino=False):
        """
        初始化DPT_DINOv2模型
        
        参数:
            encoder: ViT编码器类型，可选'vits'(小型)、'vitb'(基础)、'vitl'(大型)
            features: 解码头特征通道数
            out_channels: 多尺度特征的通道配置
            use_bn: 是否使用批量归一化
            use_clstoken: 是否使用分类标记增强特征
            pretrained_dino: 是否加载预训练的DINOv2权重
        """
        super(DPT_DINOv2, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl']

        # 加载预训练的DINOv2模型
        self.pretrained = torch.hub.load(
            'facebookresearch/dinov2', 
            'dinov2_{:}14'.format(encoder), 
            pretrained=pretrained_dino,
            # source='local',            # 强制使用本地
            skip_validation=True,     # 关键！跳过仓库验证
            )
        
        
        # 获取Transformer隐藏维度
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        # 创建DPT解码头
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        """
        模型前向传播
        
        参数:
            x: 输入图像
            
        返回:
            中间特征和视差预测
        """
        h, w = x.shape[-2:]

        # 提取多层Transformer特征
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_size = self.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        output = self.depth_head(features, patch_h, patch_w, patch_size=patch_size, return_intermediate=True)
        return output


class DepthAnything(DPT_DINOv2):
    """
    DepthAnything深度估计模型
    
    基于DPT_DINOv2的深度估计专用模型:
    - 继承DPT_DINOv2的特征提取和解码能力
    - 针对单目深度估计任务进行了优化
    - 输出表示场景深度的单通道图像
    
    该模型是FoundationStereo中用于提取深度特征的核心组件之一，
    利用其强大的深度感知能力增强立体匹配效果。
    """
    def __init__(self, config):
        """
        初始化DepthAnything模型
        
        参数:
            config: 模型配置字典，包含DPT_DINOv2需要的参数
        """
        super().__init__(**config)

    def forward(self, x):
        """
        模型前向传播
        
        参数:
            x: 输入图像
            
        返回:
            深度预测，形状为(B,H,W)
        """
        h, w = x.shape[-2:]

        # 提取Transformer特征
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_size = self.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        
        # 生成深度预测并上采样
        depth = self.depth_head(features, patch_h, patch_w, patch_size=patch_size)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()

    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))

    print(model)

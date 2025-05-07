# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.update import *
from core.extractor import *
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.utils.utils import *
from Utils import *
import time,huggingface_hub
from video_depth_anything.video_depth import VideoDepthAnything


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def normalize_image(img):
    '''
    归一化图像到 ImageNet 均值和标准差
    
    参数:
        img: (B,C,H,W) 范围在 0-255 之间的RGB图像
    
    返回:
        归一化后的图像，形状不变
    '''
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img/255.0).contiguous()


class hourglass(nn.Module):
    """
    沙漏网络结构，用于代价体积聚合
    
    该模块采用U型网络结构，通过下采样-上采样路径和跳跃连接来处理3D代价体积
    同时结合EdgeNeXt特征以增强代价体积表示
    """
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        # 下采样路径
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))

        # 上采样路径
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        # 跳跃连接的特征聚合
        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
                                   
        # 代价体积注意力模块                           
        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        # 特征注意力层，用于整合EdgeNeXt特征
        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        """
        前向传播函数
        
        参数:
            x: 初始代价体积
            features: EdgeNeXt特征列表
            
        返回:
            聚合后的代价体积
        """
        # 下采样路径，同时融合EdgeNeXt特征
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        # 上采样路径，同时融合下采样特征
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        # 最终输出
        conv = self.conv1_up(conv1)
        x = self.conv_patch(x)
        x = self.atts["4"](x)
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
        conv = conv + x
        conv = self.conv_out(conv)

        return conv



class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        """
        初始化FoundationStereo模型
        
        参数:
            args: 配置参数，包括模型维度、迭代次数等
        """
        super().__init__()
        self.args = args

        # 添加默认参数，如果未提供
        if not hasattr(args, 'num_frames'):
            args.num_frames = 32  # 默认帧数

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        # 上下文网络，提取上下文特征
        self.cnet = ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        # 更新模块，用于迭代优化视差
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        # 空间和通道注意力
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        # 上下文特征处理
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])

        # 特征提取器，集成了VideoDepthAnything和EdgeNeXt
        self.feature = Feature(args)
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        # 特征下采样网络
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        # 视差上采样模块
        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )

        # 代价体积处理模块
        self.corr_stem = nn.Sequential(
            nn.Conv3d(32, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
            )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(
          BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
          ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
          nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
        )

        # 相关性搜索半径
        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2*r+1, requires_grad=False).reshape(1, 1, 2*r+1, 1)
        self.dx = dx


    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        """
        将低分辨率视差上采样到原始分辨率
        
        参数:
            disp: 低分辨率视差图
            mask_feat_4: 遮罩特征
            stem_2x: 2倍下采样的特征
            
        返回:
            上采样后的视差图
        """
        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()


    def forward(self, image1_seq, image2_seq, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        """ 
        估计视频序列的视差
        
        参数:
            image1_seq: 左视图视频序列 [B,T,C,H,W]
            image2_seq: 右视图视频序列 [B,T,C,H,W]
            iters: GRU迭代次数
            flow_init: 初始光流，通常不使用
            test_mode: 是否为测试模式
            low_memory: 是否启用低内存模式
            init_disp: 初始视差，如果为None则自动生成
            
        返回:
            test_mode下: 最终视差图
            train_mode下: (初始视差图, 视差预测列表)
        """
        B, T, C, H, W = image1_seq.shape
        low_memory = low_memory or (self.args.get('low_memory', False))
        
        # 归一化
        image1_seq_norm = torch.stack([normalize_image(image1_seq[:, t]) for t in range(T)], dim=1)
        image2_seq_norm = torch.stack([normalize_image(image2_seq[:, t]) for t in range(T)], dim=1)
        
        with autocast(enabled=self.args.mixed_precision):
            # 这里连接左右视频序列
            combined_seq = torch.cat([image1_seq_norm, image2_seq_norm], dim=0)  # [2B, T, C, H, W]
            out, vit_feat = self.feature(combined_seq)
            
            # 分离左右特征
            vit_feat = vit_feat[:B]  # 只使用左视图特征
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            
            # 使用最后一帧进行处理
            stem_2x = self.stem_2(image1_seq_norm[:, -1])

            # 构建代价体积
            gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)  # Group-wise correlation volume (B, N_group, max_disp, H, W)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
            del left_tmp, right_tmp
            
            # 代价体积处理
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            comb_volume = self.corr_stem(comb_volume)
            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left)

            # 从几何编码体积初始化视差
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)  #(B, max_disp, H, W)
            if init_disp is None:
              init_disp = disparity_regression(prob, self.args.max_disp//4)  # Weighted sum of disparity

            # 上下文特征提取
            cnet_list = self.cnet(image1_seq_norm[:, -1], vit_feat=vit_feat, num_layers=self.args.n_gru_layers)   #(1/4, 1/8, 1/16)
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]   # Hidden information
            inp_list = [torch.relu(x[1]) for x in cnet_list]   # Context information list of pyramid levels
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        # 构建几何编码体积
        geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)  # (B,H,W,1) Horizontal only
        disp = init_disp.float()
        disp_preds = []

        # GRUs迭代更新视差 (1/4分辨率)
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)
            with autocast(enabled=self.args.mixed_precision):
              net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)

            disp = disp + delta_disp.float()
            if test_mode and itr < iters-1:
                continue

            # 上采样预测
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)


        if test_mode:
            return disp_up

        return init_disp, disp_preds


    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
        """
        分层处理视频序列中的图像对，先进行小分辨率处理再细化
        
        参数:
            image1: 左视图图像
            image2: 右视图图像
            iters: GRU迭代次数
            test_mode: 是否为测试模式
            low_memory: 是否启用低内存模式
            small_ratio: 小分辨率的比例
            
        返回:
            最终视差图
        """
        B,_,H,W = image1.shape
        img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
        img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
        padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
        img1_small, img2_small = padder.pad(img1_small, img2_small)
        disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
        disp_small = padder.unpad(disp_small.float())
        disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
        disp_small_up = disp_small_up.clip(0, None)

        padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
        image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
        disp_small_up += padder._pad[0]
        init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
        disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
        disp = padder.unpad(disp.float())
        return disp


class VideoDepthAnythingFeature(nn.Module):
    """
    VideoDepthAnythingFeature: 视频深度特征提取器
    
    使用VideoDepthAnything从视频序列中提取深度相关特征
    支持不同的ViT模型配置，包括small, base和large版本
    """
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'num_frames': 32, 'pe': 'ape'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'num_frames': 32, 'pe': 'ape'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'num_frames': 32, 'pe': 'ape'}
    }

    def __init__(self, encoder='vits', num_frames=32):
        """
        初始化视频深度特征提取器
        
        参数:
            encoder: 使用的ViT模型版本，可选 'vits', 'vitb', 'vitl'
            num_frames: 处理的视频序列帧数
        """
        super().__init__()
        self.encoder = encoder
        self.num_frames = num_frames
        config = self.model_configs[encoder].copy()
        config['num_frames'] = num_frames
        self.video_depth_anything = VideoDepthAnything(**config)

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

    def forward(self, x_seq):
        """
        前向传播函数
        
        参数:
            x_seq: 视频序列 (B,T,C,H,W)
            
        返回:
            包含VideoDepthAnything特征的字典
        """
        B, T, C, H, W = x_seq.shape
        
        # 为视频数据准备
        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = get_resize_keep_aspect_ratio(H, W, divider=divider, max_H=1344, max_W=1344)
        x_in_ = F.interpolate(x_seq.reshape(B*T, C, H, W), size=(H_resize, W_resize), mode='bicubic', align_corners=False)
        x_in_ = x_in_.reshape(B, T, C, H_resize, W_resize)
        
        self.dino = self.video_depth_anything.eval()
        with torch.no_grad():
            output = self.video_depth_anything(x_in_)
        
        # 获取特征并调整大小
        vit_feat = output['out']
        vit_feat = F.interpolate(vit_feat, size=(H//4, W//4), mode='bilinear', align_corners=True)
        
        # 处理 EdgeNeXt 特征 - 这里假设我们只处理当前帧
        x = self.video_depth_anything.stem(x_seq[:, -1])  # 只处理最后一帧
        x4 = self.video_depth_anything.stages[0](x)
        x8 = self.video_depth_anything.stages[1](x4)
        x16 = self.video_depth_anything.stages[2](x8)
        x32 = self.video_depth_anything.stages[3](x16)

        x16 = self.video_depth_anything.deconv32_16(x32, x16)
        x8 = self.video_depth_anything.deconv16_8(x16, x8)
        x4 = self.video_depth_anything.deconv8_4(x8, x4)
        
        # 拼接 VideoDepthAnything 特征与 EdgeNeXt 特征
        x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.video_depth_anything.conv4(x4)
        
        return [x4, x8, x16, x32], vit_feat


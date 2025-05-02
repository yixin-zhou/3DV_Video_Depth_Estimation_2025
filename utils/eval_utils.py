import numpy as np
import random
import os
import torch


# Copy from Sintel Depth Dataset official I/O Python codes
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


# Copy from https://github.com/qhd1996/seed-everything
def seed(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# class Eval_Depth_Metrics:
#     def __init__(self, mode, **kwargs):
#         assert mode in ['image', 'video'], "mode must be 'image' or 'video'"
#         self.mode = mode
#
#         if mode == 'image':
#             self.gt_depth = kwargs.get('gt_depth')
#             self.pred_depth = kwargs.get('pred_depth')
#             assert len(self.pred_depth.shape) == 3, "The length of pred_depth should be 3, [B, H, W]"
#             assert self.gt_depth.shape == self.pred_depth.shape, "The shape of gt_depth and pred_depth is different."
#
#         if mode == 'video':
#             self.pred_depth_seq = kwargs.get('pred_depth_seq')
#             self.gt_depth_seq = kwargs.get('gt_depth_seq')
#             self.cam_poses = kwargs.get('cam_poses')
#             self.intrinsics = kwargs.get('intrinsics')
#             self.baseline = kwargs.get('baseline')
#             assert len(self.pred_depth_seq.shape) == 4, "The length of pred_depth_seq should be 4, [Batch_size, Frames, H, W]"
#             assert self.pred_depth_seq.shape == self.gt_depth_seq.shape, "The shape of gt_depth_seq and pred_depth_seq is different."
#
#     def abs_relative_difference(self, valid_mask=None):
#         '''
#         Function:
#             Compute the Absolute Relative Error (AbsRel) metric.
#             Supports both image mode ([B, H, W]) and video mode ([B, N, H, W]).
#
#         :param valid_mask: (optional) A boolean mask indicating valid pixels.
#
#         :return:
#             torch.Tensor: Scalar mean AbsRel over all valid pixels in the batch or sequence.
#         '''
#         if self.mode == 'image':
#             pred = self.pred_depth
#             gt = self.gt_depth
#         else:
#             pred = self.pred_depth_seq
#             gt = self.gt_depth_seq
#
#         abs_relative_diff = torch.abs(pred - gt) / gt
#         if valid_mask is not None:
#             abs_relative_diff[~valid_mask] = 0
#             n = valid_mask.sum(dim=(-1, -2))
#         else:
#             n = torch.tensor(pred.shape[-1] * pred.shape[-2], device=pred.device)
#
#         abs_relative_diff = torch.sum(abs_relative_diff, dim=(-1, -2)) / n
#         return abs_relative_diff.mean()
#
#
#     def root_mean_squared_error(self, valid_mask=None):
#         '''
#         Function:
#             Compute the Root Mean Squared Error (RMSE) metric.
#             Supports both image mode ([B, H, W]) and video mode ([B, N, H, W]).
#
#         :param valid_mask: (optional) A boolean mask indicating valid pixels.
#
#         :return:
#             torch.Tensor: Scalar mean RMSE over all valid pixels in the batch or sequence.
#         '''
#         if self.mode == 'image':
#             pred = self.pred_depth
#             gt = self.gt_depth
#         else:
#             pred = self.pred_depth_seq
#             gt = self.gt_depth_seq
#
#         squared_error = (pred - gt) ** 2
#
#         if valid_mask is not None:
#             squared_error[~valid_mask] = 0
#             n = valid_mask.sum(dim=(-1, -2))  # Shape: [B] or [B, N]
#         else:
#             n = torch.tensor(pred.shape[-1] * pred.shape[-2], device=pred.device)
#
#         mse = torch.sum(squared_error, dim=(-1, -2)) / n  # Mean squared error per image/frame
#         rmse = torch.sqrt(mse)  # Take square root for RMSE
#
#         return rmse.mean()
#
#     def delta(self, valid_mask=None, threshold=1.25):
#         '''
#         Function:
#             Compute the δ accuracy metric (δ1, δ2, or δ3 depending on the threshold).
#
#         :param valid_mask: (optional) A boolean mask indicating valid pixels.
#         :param threshold: threshold value for δ metric. Default is 1.25 (δ1).
#         :return:
#             Scalar accuracy (between 0 and 1), indicating the proportion of pixels
#         '''
#         if self.mode == 'image':
#             pred = self.pred_depth  # [B, H, W]
#             gt = self.gt_depth
#         else:
#             pred = self.pred_depth_seq  # [B, N, H, W]
#             gt = self.gt_depth_seq
#
#         ratio = torch.max(pred / gt, gt / pred)
#         if valid_mask is not None:
#             ratio = ratio[valid_mask]
#         delta1 = (ratio < threshold).float().mean()
#
#         return delta1
#
#     def tae(self):
#         return 0


class Eval_Disparity_Metrics:
    def __init__(self, mode, **kwargs):
        assert mode in ['image', 'video'], "mode must be 'image' or 'video'"
        self.mode = mode

        if mode == 'image':
            self.gt_disparity = kwargs.get('gt_disparity')
            self.pred_disparity = kwargs.get('pred_disparity')
            assert len(self.pred_disparity.shape) == 3, "The length of pred_disparity should be 3, [B, H, W]"
            assert self.gt_disparity.shape == self.pred_disparity.shape, "The shape of gt_disparity and pred_disparity is different."

        if mode == 'video':
            self.pred_disparity_seq = kwargs.get('pred_disparity_seq')
            self.gt_disparity_seq = kwargs.get('gt_disparity_seq')
            self.cam_poses = kwargs.get('cam_poses')
            self.intrinsics = kwargs.get('intrinsics')
            self.baseline = kwargs.get('baseline')
            assert len(
                self.pred_disparity_seq.shape) == 4, "The length of pred_disparity_seq should be 4, [Batch_size, Frames, H, W]"
            assert self.pred_disparity_seq.shape == self.gt_disparity_seq.shape, "The shape of gt_disparity_seq and pred_disparity_seq is different."

    def end_point_error(self, valid_mask=None):
        if self.mode == 'image':
            pred = self.pred_disparity  # [B, H, W]
            gt = self.gt_disparity
        else:
            pred = self.pred_disparity_seq  # [B, N, H, W]
            gt = self.gt_disparity_seq

        diff = torch.abs(pred -gt)

        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum(dim=(-1, -2))  # Shape: [B] or [B, N]
        else:
            n = torch.tensor(pred.shape[-1] * pred.shape[-2], device=pred.device)

        epe = (torch.sum(diff, dim=(-1, -2)) / n).mean()
        return epe

    def D1(self, valid_mask=None):
        if self.mode == 'image':
            pred = self.pred_disparity  # [B, H, W]
            gt = self.gt_disparity
        else:
            pred = self.pred_disparity_seq  # [B, N, H, W]
            gt = self.gt_disparity_seq

        diff = torch.abs(pred -gt)
        bad = (diff > 3.0) & (diff > 0.05 * gt)

        if valid_mask is not None:
            bad = bad & valid_mask

        d1 = bad.float().mean()
        return d1

    def bad_pixel_x(self, threshold, valid_mask=None):
        if self.mode == 'image':
            pred = self.pred_disparity  # [B, H, W]
            gt = self.gt_disparity
        else:
            pred = self.pred_disparity_seq  # [B, N, H, W]
            gt = self.gt_disparity_seq

        diff = torch.abs(pred - gt)
        bad_pixel = diff > threshold

        if valid_mask is not None:
            bad_pixel = bad_pixel & valid_mask

        bpx = bad_pixel.float().mean()
        return bpx

    def tepe(self, valid_mask=None):
        pred = self.pred_disparity_seq
        gt = self.gt_disparity_seq
        T = len(self.pred_disparity_seq)
        assert valid_mask.shape == pred.shape, "The shape of valid mask is different from pred_disparity_seq."

        pred_diff = pred[:-1] - pred[1:] # shape: [T-1, H, W]
        gt_diff = gt[:-1] - gt[1:]
        diff = (pred_diff - gt_diff) ** 2

        if valid_mask is not None:
            valid = valid_mask[:-1] & valid_mask[1:]
            diff = diff * valid.float()
            num_valid = valid.float().sum()
        else:
            num_valid = diff.numel()
        tepe = torch.sqrt(diff.sum() / num_valid)
        return tepe


import numpy as np
import cv2
from matplotlib import pyplot as plt


# Adapt from https://github.com/facebookresearch/consistent_depth/blob/main/utils/visualization.py
CM_MAGMA = (np.array([plt.get_cmap('magma').colors]).
            transpose([1, 0, 2]) * 255)[..., ::-1].astype(np.uint8)


def viz_disparity(disparity, disparity_min=None, disparity_max=None, title=None, show=False, save_path=None):
    if disparity_min is None:
        disparity_min = np.min(disparity)
    if disparity_max is None:
        disparity_max = np.max(disparity)

    disparity_norm = np.clip((disparity - disparity_min) / (disparity_max - disparity_min), 0, 1)
    disparity_inverted = 1.0 - disparity_norm
    disparity_uint8 = (disparity_inverted * 255).astype(np.uint8)

    color = ((cv2.applyColorMap(disparity_uint8, CM_MAGMA) / 255) ** 2.2 * 255).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, color) # save as BGR
    plt.imshow(color[..., ::-1])  # BGR → RGB
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == '__main__':
    depth = np.load('../FoundationStereo/datasets/demo/temporal_consistency_demo/output/frame_5/depth_meter.npy')
    viz_disparity(depth, show=False, save_path='test.png')
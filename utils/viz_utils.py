import numpy as np
import cv2
from matplotlib import pyplot as plt


# Adapt from https://github.com/facebookresearch/consistent_depth/blob/main/utils/visualization.py
CM_MAGMA = (np.array([plt.get_cmap('magma').colors]).
            transpose([1, 0, 2]) * 255)[..., ::-1].astype(np.uint8)


def visualize_depth(depth, depth_min=None, depth_max=None, title=None, show=False, save_path=None):
    if depth_min is None:
        depth_min = np.min(depth)
    if depth_max is None:
        depth_max = np.max(depth)

    depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
    depth_inverted = 1.0 - depth_norm
    depth_uint8 = (depth_inverted * 255).astype(np.uint8)

    color = ((cv2.applyColorMap(depth_uint8, CM_MAGMA) / 255) ** 2.2 * 255).astype(np.uint8)
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
    visualize_depth(depth, show=True, save_path='test.png')
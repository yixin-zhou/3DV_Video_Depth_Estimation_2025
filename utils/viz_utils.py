import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from natsort import natsorted

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
        cv2.imwrite(save_path, color)  # save as BGR
    plt.imshow(color[..., ::-1])  # BGR → RGB
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()


def images2videos(images_dir, video_savepath, fps=15, verbose=False):
    images = natsorted([os.path.join(images_dir, image_dir) for image_dir in os.listdir(images_dir)])
    _, ext = os.path.splitext(video_savepath)
    if ext == '.mp4':
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_savepath, fourcc, fps, (width, height))
        for img in images:
            frame = cv2.imread(img)
            video.write(frame)
        video.release()
        if verbose:
            print(f'Mp4 video is saved to {video_savepath}')

    elif ext == '.gif':
        frames = [Image.open(img).convert("RGB") for img in images]
        frames[0].save(
            video_savepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000/fps,
            loop=0
        )
        if verbose:
            print(f'GIF video is saved to {video_savepath}')
    else:
        raise ValueError('Only support transferring images to .mp4 or .gif video')


if __name__ == '__main__':
    depth = np.load('../FoundationStereo/datasets/demo/temporal_consistency_demo/output/frame_5/depth_meter.npy')
    viz_disparity(depth, show=False, save_path='test.png')
    # path = '../data/raw_kitti/val/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data'
    # images2videos(images_dir=path, video_savepath='../test.gif')

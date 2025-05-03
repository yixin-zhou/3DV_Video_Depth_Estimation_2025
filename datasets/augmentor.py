import numpy as np
import random
from PIL import Image
from torchvision.transforms import ColorJitter, functional, Compose
import cv2

cv2.setNumThreads(0)  # Disable OpenCV multithreading to avoid conflicts with PyTorch's internal CPU multithreading.
cv2.ocl.setUseOpenCL(False)  # Disable OpenCV from using OpenCL (GPU acceleration) to avoid conflicts with CUDA.


# Adapt from https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/augmentor.py#L37
class AdjustGamma:
    def __init__(self, gamma, gain):
        self.gamma, self.gain = (
            gamma,
            gain
        )

    def __call__(self, img):
        adjusted_img = functional.adjust_gamma(img, gamma=self.gamma, gain=self.gain)
        return adjusted_img

    def __repr__(self):
        return f"Lat used gamma adjust Params: gamma={self.gamma}, gain={self.gain}"


class VideoSeqAugmentor:
    def __init__(self, crop_size, saturation_range=[0.6, 1.4], gamma_params=[0.8, 1.4, 1.0, 1.2]):
        # Probability of different augmentation method
        self.prob_eraser_aug = 0.4
        self.prob_vflip_aug = 0.1
        self.prob_hflip_aug = 0.1
        self.prob_asymmetric_color_aug = 0.2

        # Params for augmentation method
        self.crop_size = crop_size
        self.gamma_params = gamma_params
        self.saturation_range = saturation_range

    def color_transform(self, seq):
        def generate_photo_aug():
            gamma = random.uniform(self.gamma_params[0], self.gamma_params[1])
            gain = random.uniform(self.gamma_params[2], self.gamma_params[3])
            photo_aug = Compose([
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=self.saturation_range,
                    hue=0.5 / 3.14,
                ),
                AdjustGamma(gamma, gain)
            ])
            return photo_aug

        if np.random.rand() < self.prob_asymmetric_color_aug:
            for cam in [0, 1]:
                photo_aug = generate_photo_aug()
                for frame in range(len(seq)):
                    adjusted_img = np.array(photo_aug(Image.fromarray(seq[frame][cam])), dtype=np.uint8)
                    seq[frame][cam] = adjusted_img
        else:
            photo_aug = generate_photo_aug()
            for cam in [0, 1]:
                for frame in range(len(seq)):
                    adjusted_img = np.array(photo_aug(Image.fromarray(seq[frame][cam])), dtype=np.uint8)
                    seq[frame][cam] = adjusted_img
        return seq

    def eraser_transform(self, seq, bounds=[30, 60]):
        w, h = seq[0][0].shape[:2]
        for cam in [0, 1]:
            for frame in range(len(seq)):
                if np.random.rand() < self.prob_eraser_aug:
                    mean_color = seq[frame][cam].reshape(-1, 3).mean(axis=0)
                    x0 = np.random.randint(0, w - bounds[1])
                    y0 = np.random.randint(0, h - bounds[1])
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    seq[frame][cam][y0: y0 + dy, x0: x0 + dx, :] = mean_color
        return seq

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
    def __init__(self,
                 crop_size,
                 saturation_range=[0.6, 1.4],
                 gamma_params=[0.8, 1.4, 1.0, 1.2],
                 bounds=[20, 60],
                 max_scale=0.5,
                 min_scale=-0.2):

        # Probability of different augmentation method
        self.prob_eraser_aug = 0.3
        self.prob_flip_aug = 0.5
        self.prob_resize = 0.7
        self.prob_asymmetric_color_aug = 0.2

        # Params for augmentation method
        self.crop_size = crop_size
        self.gamma_params = gamma_params
        self.saturation_range = saturation_range
        self.bounds = bounds
        self.max_scale = max_scale
        self.min_scale = min_scale

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

    def eraser_transform(self, seq):
        h, w = seq[0][0].shape[:2]
        for cam in [0, 1]:
            for frame in range(len(seq)):
                if np.random.rand() < self.prob_eraser_aug:
                    mean_color = seq[frame][cam].reshape(-1, 3).mean(axis=0)
                    x0 = np.random.randint(0, w - self.bounds[1] - 1)
                    y0 = np.random.randint(0, h - self.bounds[1] - 1)
                    dx = np.random.randint(self.bounds[0], self.bounds[1])
                    dy = np.random.randint(self.bounds[0], self.bounds[1])
                    seq[frame][cam][y0: y0 + dy, x0: x0 + dx, :] = mean_color
        return seq

    def flip_transform(self, seq, disp):
        if np.random.rand() < self.prob_flip_aug:
            for frame in range(len(seq)):
                flip_type = random.choice([0, 1, -1])
                flip_disp = cv2.flip(disp[frame], flip_type)
                disp[frame] = flip_disp

                for cam in [0, 1]:
                    flip_img = cv2.flip(seq[frame][cam], flip_type)
                    seq[frame][cam] = flip_img
        return seq, disp

    def spatial_transform(self, seq, disp):
        h, w = seq[0][0].shape[:2]
        crop_w, crop_h = self.crop_size

        if np.random.rand() >= self.prob_resize:
            return seq, disp

        min_x_scale = np.log2(crop_w / w)
        min_y_scale = np.log2(crop_h / h)

        x_scale = 2 ** np.random.uniform(
            max(min_x_scale, self.min_scale), self.max_scale
        )
        y_scale = 2 ** np.random.uniform(
            max(min_y_scale, self.min_scale), self.max_scale
        )

        new_h = int(h * y_scale) + 8
        new_w = int(w * x_scale) + 8

        for frame in range(len(seq)):
            resized_disp = cv2.resize(
                disp[frame], dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            for cam in [0, 1]:
                resized_img = cv2.resize(
                    seq[frame][cam], dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                seq[frame][cam] = resized_img

            disp[frame] = resized_disp

        y0 = np.random.randint(0, new_h - crop_h + 1)
        x0 = np.random.randint(0, new_w - crop_w + 1)

        for frame in range(len(seq)):
            disp[frame] = disp[frame][y0: y0 + crop_h, x0: x0 + crop_w]

            for cam in [0, 1]:
                seq[frame][cam] = seq[frame][cam][y0: y0 + crop_h,
                                  x0: x0 + crop_w, :]

        return seq, disp

    def __call__(self, seq, disp):
        seq = self.color_transform(seq)
        seq, disp = self.flip_transform(seq, disp)
        seq, disp = self.spatial_transform(seq, disp)
        seq = self.eraser_transform(seq)

        # Make sure that the sequence images and disparities are continuous in memory
        for frame in range(len(seq)):
            disp[frame] = np.ascontiguousarray(disp[frame])
            for cam in [0, 1]:
                seq[frame][cam] = np.ascontiguousarray(seq[frame][cam])

        return seq, disp

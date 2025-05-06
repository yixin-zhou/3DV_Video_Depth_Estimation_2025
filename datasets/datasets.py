from torch.utils.data import Dataset
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from augmentor import VideoSeqAugmentor

class VideoSeqDataset:
    def __init__(self, crop_size, aug_params, reader):
        self.augmentor = VideoSeqAugmentor(crop_size, **aug_params)








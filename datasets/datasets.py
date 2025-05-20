from torch.utils.data import Dataset
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from augmentor import VideoSeqAugmentor
from utils.utils_read import read_sintel_depth, read_sintel_disparity


class VideoSeqDataset:
    def __init__(self, crop_size, aug_params, depth_reader, disp_reader):
        self.augmentor = VideoSeqAugmentor(crop_size, **aug_params)
        self.depth_reader = depth_reader
        self.disp_reader = disp_reader
        self.sample_list = []

    # def _get_output_tensor(self):
    #     for frame in range(len(self.sample_list)):



class VideoSintelDataset(VideoSeqDataset):
    def __init__(self,
                 dstype,
                 base_dir="data/",
                 aug_params={},
                 crop_size=None):
        super().__init__(crop_size,
                         aug_params,
                         read_sintel_disparity,
                         read_sintel_depth)
        self.dstype = dstype
        self.base_dir = base_dir



if __name__ == '__main__':
    ds = VideoSintelDataset(dstype='clean')
    print(0)

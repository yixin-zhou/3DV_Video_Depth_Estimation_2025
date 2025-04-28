from torch.utils.data import Dataset
import torch

class MyStereoMixDataset(Dataset):
    def __init__(self,
                 dataset_name:str,
                 left_images_path:str,
                 right_images_path:str,
                 mode:str,
                 depth_maps=None,
                 camera_params_path=None):

        if mode not in ('train', 'test', 'infer'):
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: 'train', 'test', 'infer'.")

        self.dataset_name = dataset_name
        self.left_images_path = left_images_path
        self.right_images_path = right_images_path
        if camera_params_path is not None:
            self.camera_params_path = camera_params_path

        if self.dataset_name == 'Virtual_KITTI_2':
            self.depth_maps = depth_maps






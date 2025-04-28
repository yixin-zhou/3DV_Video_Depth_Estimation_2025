import os
import numpy as np
import requests
from tqdm import tqdm
import tarfile
import shutil
from natsort import natsorted
import cv2
from tqdm import tqdm

BASELINE = 0.532725
fx = 725.0087


def download_url(url, save_path, desc):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)

    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        raise ValueError("WARNING: Downloaded size does not match expected size!")


def extract_tar(tar_filepath, extract_path):
    if not os.path.isfile(tar_filepath):
        raise ValueError(f'The .tar file "{tar_filepath}" does not exeist.')
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(tar_filepath) as tar:
        tar.extractall(path=extract_path)
    print(f'{tar_filepath} is successfully extracted to {extract_path}')


def download_extract_delete(url, save_path, desc):
    download_url(url, save_path, desc=desc)
    extract_tar(save_path, save_path.replace('.tar', ''))
    if os.path.isfile(save_path):
        os.remove(save_path)


if __name__ == '__main__':
    dataset_path = '../datasets/'

    scene_name = ['clone', '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'fog', 'morning', 'overcast',
                  'rain', 'sunset']
    scene_list = {'01': {'test': ['15-deg-left'], 'train': [x for x in scene_name if x != '15-deg-left']},
                  '02': {'test': ['30-deg-right'], 'train': [x for x in scene_name if x != '30-deg-right']},
                  '06': {'test': ['fog'], 'train': [x for x in scene_name if x != 'fog']},
                  '18': {'test': ['morning'], 'train': [x for x in scene_name if x != 'morning']},
                  '20': {'test': ['rain'], 'train': [x for x in scene_name if x != 'rain']}
                  }

    # Download Virtual KITTI2 RGB and depth dataset and extract them
    for content_type in ['textgt']:
        vktti_url = f'https://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_{content_type}.tar'
        if content_type == 'textgt':
            vktti_url = vktti_url + '.gz'
        download_extract_delete(url=vktti_url,
                                save_path=os.path.join(dataset_path, f'vkitti_2.0.3_{content_type}.tar'),
                                desc=f'Downloading vkitti_2.0.3_{content_type}.tar')

    # Merge Virtual KITTI2 RGB and depth dataset
    os.makedirs(os.path.join(dataset_path, 'vktti_2.0.3/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'vktti_2.0.3/test'), exist_ok=True)

    for key, item in scene_list.items():
        for split in ['train', 'test']:
            scenes = item[split]
            for scene in scenes:
                target_dir = os.path.join(dataset_path, 'vktti_2.0.3', split, 'Scene' + key, scene)
                os.makedirs(target_dir, exist_ok=True)
                src_dir_depth = os.path.join(dataset_path, 'vkitti_2.0.3_depth', 'Scene' + key, scene, 'frames/depth')
                src_dir_rgb = os.path.join(dataset_path, 'vkitti_2.0.3_rgb', 'Scene' + key, scene, 'frames/rgb')

                src_intrinsic = os.path.join(dataset_path, 'vkitti_2.0.3_textgt', 'Scene' + key, scene, 'intrinsic.txt')
                src_extrinsic = os.path.join(dataset_path, 'vkitti_2.0.3_textgt', 'Scene' + key, scene, 'extrinsic.txt')

                shutil.move(src_dir_depth, target_dir)
                shutil.move(src_dir_rgb, target_dir)
                shutil.move(src_intrinsic, os.path.join(target_dir, 'intrinsic.txt'))
                shutil.move(src_extrinsic, os.path.join(target_dir, 'extrinsic.txt'))

    for content_type in ['rgb', 'depth', 'textgt']:
        shutil.rmtree(os.path.join(dataset_path, f'vkitti_2.0.3_{content_type}'))

    vktti_dir = os.path.join(dataset_path, 'vktti_2.0.3')
    print(f'Merge rgb, depth and textgt(intrinsic.txt and extrinsic.txt) to {vktti_dir}')

    with tqdm(total=120, desc='Transferring depth maps to disparities') as pbar:
        for key, item in scene_list.items():
            for split in ['train', 'test']:
                scenes = item[split]
                for scene in scenes:
                    for cam in ['Camera_0', 'Camera_1']:
                        depth_dir = os.path.join(vktti_dir, split, 'Scene' + key, scene, 'depth', cam)
                        disp_dir = os.path.join(vktti_dir, split, 'Scene' + key, scene, 'disparity', cam)
                        os.makedirs(disp_dir, exist_ok=True)
                        depth_maps_path = [os.path.join(depth_dir, depth_map_path)
                                           for depth_map_path in natsorted(os.listdir(depth_dir))]
                        for path in depth_maps_path:
                            depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                            depth_meter = depth / 100
                            disparity = (fx * BASELINE) / depth_meter

                            filename = os.path.basename(path).replace('depth', 'disparity').replace('.png', '.npy')
                            np.savez_compressed(os.path.join(disp_dir, filename), disparity=disparity)
                        pbar.update(1)

    print('Depth maps transferring done')

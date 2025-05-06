import numpy as np
import os
import requests
from tqdm import tqdm


# Copy from Sintel Depth Dataset official I/O Python codes
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


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


def cam2txt(cam_filepath, output_path, baseline=0.1, verbose=True):
    '''
    Function:
        Transform the Sintel .cam format camera parameters to .txt format,
        which is used for Foundation Stereo

    :param cam_filepath: the filepath of Sintel .cam format camera parameters file
    :param output_path: the output filepath of .txt format camera parameters file
    :param baseline: the length of baseline in the unit of meter
    :param verbose: whether to print the information of transformation

    :return:
        a dict which contains the intrinsic and extrinsic matrix and baseline
    '''

    f = open(cam_filepath, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
    N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))

    with open(output_path, 'w') as K:
        K.write(' '.join(map(str, M.flatten())) + '\n')
        K.write(str(baseline) + '\n')
        K.write(' '.join(map(str, M.flatten())) + '\n')

    if verbose:
        print(f"The intrinsic and extrinsic matrix have been saved to {output_path}")

    return {'M': M, 'N': N, 'baseline': baseline}


# Copy from https://github.com/NVlabs/FoundationStereo/blob/master/Utils.py
def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def depth2disparity(depth, fx, baseline):
    assert len(depth.shape) == 2, f"The channel of input depth should be 2, instead of {len(depth.shape)}"
    disparity = fx * baseline / depth
    return disparity

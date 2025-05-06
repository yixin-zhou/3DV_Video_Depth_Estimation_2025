import numpy as np
import os
import pandas as pd
import pykitti


# Copy from Sintel Depth Dataset official I/O Python codes
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def read_sintel_camdata(camdata_path):
    f = open(camdata_path, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
    N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
    return {'intrinsic': M, 'extrinsic': N}


def read_vktti_camdata(intrinsic, extrinsic, frame_id, camera_id=0):
    intr = pd.read_csv(intrinsic, sep='\s+')
    row = intr[(intr['frame'] == frame_id) & (intr['cameraID'] == camera_id)]

    if row.empty:
        raise ValueError(f"No intrinsics found for frame {frame_id} and camera {camera_id}")

    fx = row['K[0,0]'].values[0]
    fy = row['K[1,1]'].values[0]
    cx = row['K[0,2]'].values[0]
    cy = row['K[1,2]'].values[0]

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    extr = pd.read_csv(extrinsic, sep='\s+')
    row = extr[(intr['frame'] == frame_id) & (intr['cameraID'] == camera_id)]

    if row.empty:
        raise ValueError(f"No extrinsics found for frame {frame_id}, camera {camera_id}")

    R = row[['r1,1', 'r1,2', 'r1,3',
             'r2,1', 'r2,2', 'r2,3',
             'r3,1', 'r3,2', 'r3,3']].values.reshape(3, 3)
    T = row[['t1', 't2', 't3']].values.reshape(3, 1)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = T

    return {'intrinsic': K, 'extrinsic': extrinsic}


def read_kitti_camdata(calib_dic, oxts_data):



if __name__ == '__main__':
    camdata = read_sintel_camdata("../data/MPI-Sintel-depth-training-20150305/training/camdata_left/alley_1/frame_0001.cam")
    camdata2 = read_vktti_camdata(extrinsic="../data/vktti_2.0.3/train/Scene01/15-deg-right/extrinsic.txt", intrinsic="../data/vktti_2.0.3/train/Scene01/15-deg-right/intrinsic.txt", frame_id=32)
    print(camdata2)
import zipfile
import os
from utils.Utils import download_url
from natsort import natsorted

test_sequences = ["2011_09_26_drive_0002",
                  "2011_09_26_drive_0005",
                  "2011_09_26_drive_0013",
                  "2011_09_26_drive_0020",
                  "2011_09_26_drive_0023",
                  "2011_09_26_drive_0036",
                  "2011_09_26_drive_0079",
                  "2011_09_26_drive_0095",
                  "2011_09_26_drive_0113",
                  "2011_09_28_drive_0037",
                  "2011_09_29_drive_0026",
                  "2011_09_30_drive_0016",
                  "2011_10_03_drive_0047"
                  ]

save_dir = '../datasets/'
depth_annotated_dir = os.path.join(save_dir, 'data_depth_annotated.zip')

base_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'
depth_annotated_url = base_url + "data_depth_annotated.zip"
# download_url(depth_annotated_url, save_path=depth_annotated_dir, desc='Downloading KITTI Depth Prediction Dataset')

# print(f"Unzipping {depth_annotated_dir}")
# with zipfile.ZipFile(depth_annotated_dir, 'r') as zip_ref:
#     zip_ref.extractall(depth_annotated_dir.replace('.zip', ''))
#
# os.remove(depth_annotated_dir)

sequences = {}
for split in ['train', 'val']:
    split_dir = os.path.join(depth_annotated_dir.replace('.zip', ''), split)
    sequences[split] = natsorted(os.listdir(split_dir))

raw_kitti_dir = os.path.join(save_dir, 'raw_kitti')
for key, items in sequences.items():
    for item in items:
        sequence_url = base_url + 'raw_data/' + item.replace('_sync', '') + '/' + item + '.zip'
        if item.replace('_sync', '') in test_sequences:
            sequence_save_dir = os.path.join(raw_kitti_dir, 'test', item + '.zip')
        else:
            sequence_save_dir = os.path.join(raw_kitti_dir, split, item + '.zip')
        download_url(sequence_url, save_path=sequence_save_dir, desc=f'Downloading {item}')

        with zipfile.ZipFile(os.path.join(sequence_save_dir), 'r') as zip_ref:
            zip_ref.extractall(sequence_save_dir.replace('.zip', ''))
        os.remove(sequence_save_dir)


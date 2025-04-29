import zipfile
import os
from utils.Utils import download_url

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

depth_annotated_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
download_url(depth_annotated_url, save_path=depth_annotated_dir, desc='Downloading KITTI Depth Prediction Dataset')

print(f"Unzipping {depth_annotated_dir}")
with zipfile.ZipFile(depth_annotated_dir, 'r') as zip_ref:
    zip_ref.extractall(depth_annotated_dir.replace('.zip', ''))

os.remove(depth_annotated_dir)

sequences = {}
for split in ['train', 'val']:
    split_dir = os.path.join(depth_annotated_dir.replace('.zip', ''), split)
    sequences[split] = os.listdir(split_dir)

# for key, items in sequences:

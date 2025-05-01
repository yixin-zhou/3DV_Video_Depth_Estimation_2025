from utils.Utils import download_url
import zipfile
import os

Sintel_url = 'http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip'
save_dir = '../datasets'
save_path = os.path.join(save_dir, 'MPI-Sintel-stereo-training-20150305.zip')
extract_path = os.path.join(save_dir, 'MPI-Sintel-stereo-training-20150305')

download_url(Sintel_url, save_path=save_path, desc='Downloading Sintel Stereo Dataset')

with zipfile.ZipFile(save_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

os.remove(save_path)

print(f'Download Sintel Dataset Done')

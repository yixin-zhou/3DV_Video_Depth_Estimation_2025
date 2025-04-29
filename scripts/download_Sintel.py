from utils.Utils import download_url
import zipfile
import os

Sintel_url = 'http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip'

download_url(Sintel_url, save_path='../datasets/MPI-Sintel-stereo-training-20150305.zip', desc='Downloading Sintel Stereo Dataset')


<div align="center">
  <h1>VDE: Video Depth Estimation</h1>

  <a href="#"><img src="https://img.shields.io/badge/Paper-ComingSoon-lightgrey" alt="Paper PDF"></a>
  <a href="https://github.com/yixin-zhou/3DV_Video_Depth_Estimation_2025"><img src="https://img.shields.io/badge/Code-GitHub-green" alt="GitHub Code"></a>
  <a href="https://bazziprincess.github.io/3DV_Video_Depth_Estimation_2025/"><img src="https://img.shields.io/badge/Project_Page-Live-blue" alt="Project Page"></a>
  <a href="https://drive.google.com/file/d/1bQj5nzu0udYlUbuWDGPCmjOIJWrI9yXo/view?usp=sharing"><img src="https://img.shields.io/badge/Poster-GoogleSlides-orange" alt="Poster"></a>

  <br><br>
  <strong>Qinrui Deng, Tong Su, Hepeng Fan, Yixin Zhou</strong><br>
  <em>ETH Zurich</em>
</div>

---

## Abstract

Depth estimation has long been a fundamental problem in computer vision, with numerous monocular and stereo-based methods successfully applied in fields such as robotics and autonomous driving. While monocular depth estimation methods have achieved impressive results on various real-world image and video datasets, their lack of absolute scale information continues to limit their practical use. In contrast, stereo-based approaches can readily produce depth maps with scale information. However, when applied to consecutive video frames, these methods often suffer from poor temporal consistency. To address this challenge, we propose a stereo-based model tailored for video data, offering strong zero-shot inference capabilities and robust temporal coherence. The model does not require full retraining; fine-tuning on a small dataset is sufficient to significantly enhance spatiotemporal consistency. Our experiments on the Sintel dataset demonstrate the effectiveness of the proposed approach.

<p align="center">
  <img src="preview_image.png" width="100%">
</p>

---


## Quick Start


### Dataset

You can use the following scripts to automatically download and extract datasets:

```bash
# Download Sintel dataset
python scripts/download_sintel.py

# Download KITTI Depth dataset
python scripts/download_kitti_depth.py

# Download Virtual KITTI2 dataset
python scripts/download_virtual_kitti2.py
###  Setup

```bash
git clone https://github.com/yixin-zhou/3DV_Video_Depth_Estimation_2025.git
cd 3DV_Video_Depth_Estimation_2025
conda create -n vde python=3.10
conda activate vde
pip install -r requirements.txt
```

###  Training

```bash
bash Train_our_model.sh
```

### To resume training:

```bash
bash resume_training_from_best.sh
```

### Inference

To evaluate the model on different benchmarks, run one of the following:

```bash
# For KITTI Depth
python evaluation/evaluate_kitti_depth.py

# For Sintel
python evaluation/evaluate_sintel.py

# For Virtual KITTI2
python evaluation/evaluate_virtual_kitti2.py
```

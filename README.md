# Dual Image Super-Resolution for High-Resolution Optical Satellite Imagery

This project was developed as part of the ISRO Hackathon 2025. It addresses the challenge of enhancing the resolution and visual clarity of satellite imagery using a pipeline that includes pre-processing and deep learning-based super-resolution.


# Problem Statement

Satellite images often suffer from resolution limitations due to sensor constraints or transmission limitations. This project proposes a method to enhance low-resolution satellite images through:

- Shadow detection

- Brightness normalization

- Deep learning-based image super-resolution


# Approach

The pipeline includes the following major stages:

1. Shadow Detection: Adaptive thresholding is applied to grayscale images to isolate shadow regions.

2. Brightness Normalization: Histogram equalization is performed to reduce lighting inconsistencies.

3. Super-Resolution: An ESRGAN-based model (via TensorFlow Hub) is applied to enhance image resolution.

4. Quality Analysis: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) are computed to evaluate output quality.


# Directory Structure

ISRO_Hackathon/
├── data/
│   └── raw/                     # Original satellite image(s)
├── outputs/
│   ├── shadow_masks/           # Binary masks from shadow detection
│   ├── brightness_fixed/       # Histogram equalized grayscale images
│   └── super_resolved/         # Images after super-resolution
├── src/
│   ├── download_image.py       # Image downloader script
│   ├── preprocess.py           # Preprocessing step (shadow + brightness)
│   ├── edsr_predict.py         # Super-resolution using ESRGAN
│   ├── compare_quality.py      # PSNR and SSIM evaluation
│   ├── visualize_outputs.py    # Comparison grid generation
│   └── pipeline.py             # Full end-to-end pipeline
├── README.md
└── requirements.txt


# Output Image Samples:

outputs/super_resolved/image1_bicubic.png
outputs/super_resolved/image1_edsr.png
outputs/super_resolved/image1_pipeline_edsr.png


## 📊 Quality Evaluation

We computed PSNR and SSIM to compare the image quality between different super-resolution techniques:

| Method             | PSNR (dB) | SSIM   |
|--------------------|-----------|--------|
| Bicubic            | 21.43     | 0.7708 |
| EDSR (ESRGAN)      | 11.14     | 0.3334 |
| Full Pipeline (Ours) | 9.84    | 0.2753 |


# Requirements:

Install the dependencies using:
pip install -r requirements.txt


# Key packages include:

TensorFlow 2.15
TensorFlow Hub
OpenCV
NumPy
Matplotlib
Scikit-Image


# Author:

Arham Ali
B.Tech CSE (5th Semester)
ISRO Hackathon 2025 Participant

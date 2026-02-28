# DpSSeg: Deeply Supervised Multi-Scale Fusion Network for Seismic Salt Segmentation

## 1. Description

This repository provides the PyTorch implementation of:

"DpSSeg: Deeply Supervised Multi-Scale Fusion Network for Seismic Salt Segmentation"

The model is designed to balance segmentation accuracy and inference efficiency.
It achieves competitive segmentation performance while maintaining real-time inference speed (~135 FPS under our hardware conditions).

---

## 2. Dataset Information

Dataset used:

TGS Salt Identification Challenge  
https://www.kaggle.com/competitions/tgs-salt-identification-challenge

### Dataset Characteristics

- Image size: 101×101
- Grayscale PNG images
- Binary masks (0 / 255)
- Seismic salt segmentation

### Preprocessing 

- Convert grayscale to 3-channel input
- Resize to 256×256
- Bilinear interpolation for images
- Nearest-neighbor interpolation for masks
- Normalize to [0, 1]

---

## 3. Code Structure
```
DpSSeg_code/
├── configs/
│   └── dpseg_tgs.yaml        # training configuration
├── datasets/
│   └── tgs_dataset.py        # TGS dataset loader
├── engine/
│   └── trainer.py            # training loop
├── metrics/
│   └── iou.py                # IoU metric
├── models/
│   ├── backbone/
│   │   └── resnet.py
│   ├── fusion/
│   │   └── multiscale_fusion.py
│   ├── heads/
│   │   └── segmentation_head.py
│   ├── dpseg.py
│   └── __init__.py
├── tools/
│   ├── benchmark_fps.py
│   ├── init_project.py
│   ├── test_dataset.py
│   ├── test_metric.py
│   └── test_model_forward.py
├── utils/
│   ├── env.py
│   ├── logger.py
│   ├── path.py
│   └── seed.py
├── README.md
├── requirements.txt
└── train.py
```
### Key Components

- ResNet34 backbone
- Multi-scale fusion module
- Deep supervision auxiliary heads
- mIoU evaluation
- FPS benchmark tool

---

## 4. Requirements

Tested Environment:

- OS: Ubuntu 20.04 LTS
- Python: 3.10
- PyTorch: 2.1.0
- CUDA: 11.8
- GPU: NVIDIA RTX 3090 (24GB)

Install dependencies:
pip install -r requirements.txt

---

## 5. Usage Instructions

### 5.1 Prepare Dataset

Download TGS dataset and organize as:
```
data/
└── train/
    ├── images/
    └── masks/
```
Update paths in:

configs/dpseg_tgs.yaml

---

### 5.2 Train the Model
python train.py 

Checkpoints will be saved to:
outputs/best_model.pth

---

### 5.3 Evaluate / Benchmark
To measure inference speed:
python tools/benchmark_fps.py

---

### 5.4 Tests & Checks
For debugging purposes:

Dataset check:
python tools/test_dataset.py

Model forward check:
python tools/test_model_forward.py

Metric check:
python tools/test_metric.py

---



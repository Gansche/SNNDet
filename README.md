# SNNDet

## Setup

Create an environment and install dependencies:

```bash
conda create -n snndet python=3.10.19
conda activate snndet
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python python-dateutil pyyaml requests psutil torchinfo spikingjelly timm einops pandas
```

## Model Weights

This repo does **not** ship the 0.8B pretrained weights by default. Please download the weight file from **[this link](https://drive.google.com/drive/folders/1ikknSXe7PYFBBbFdqHPU0kkSB8MlHqdI?usp=sharing)** and place it under `model/`, for example:

```text
model/snndet_0.8B.pt
```

## Run

```bash
python main.py --model model/snndet_0.8B.pt --target <picture>

# example
# python main.py --model model/snndet_0.8B.pt --target assets/demo.jpg
```

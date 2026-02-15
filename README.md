# SNNDet

## Setup

Create an environment and install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Model Weights

This repo does **not** ship the 0.8B pretrained weights by default. Please download the weight file from **[this link](https://drive.google.com/drive/folders/1ikknSXe7PYFBBbFdqHPU0kkSB8MlHqdI?usp=sharing)** and place it under `models/`, for example:

```text
models/snndet_0.8B.pt
```

## Run

```bash
python main.py --model models/snndet_0.8B.pt --target <picture>
```
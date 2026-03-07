# Nuke 17 ML: Vision Transformer (DA3) Integration Pipeline

## Abstract
This repository contains the  port of the **Depth Anything 3 (Small)** Vision Transformer model into the Foundry Nuke 17 Machine Learning (ML) inference pipeline. By utilizing a custom PyTorch TorchScript wrapper.

## Setup Instructions

This implementation utilizes an isolated Mamba environment

```powershell
# 1. Create the isolated environment for Python 3.11 required by PyTorch 2.2
mamba create --prefix D:\Nuke_Scripts\nuke17_ml_env python=3.11 -y

# 2. Install the Torch CPU inference backend to match Nuke 17's internal specs
mamba install --prefix D:\Nuke_Scripts\nuke17_ml_env pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch -y

# 3. Install integration libraries
mamba install --prefix D:\Nuke_Scripts\nuke17_ml_env huggingface_hub timm -c conda-forge -y
pip install transformers
```

## Git LFS Initialization
For storing compiled `.cat` Nuke node configurations locally:

```powershell
git init
git lfs install
git lfs track "*.cat"
git add .gitattributes
```

*Note: `.pt` files are no longer hosted in the repository to optimize bandwidth and LFS usage.*

## Build Instructions

Before using the Nuke CatFileCreator, you must generate the PyTorch `.pt` file locally. Run the following command:

```powershell
python src/build_da3_small.py
```

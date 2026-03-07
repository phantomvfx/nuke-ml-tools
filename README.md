# Nuke 17 ML: Vision Transformer (DA3) Integration Pipeline

## Abstract
This repository contains the  port of the **Depth Anything 3 (Small)** Vision Transformer model into the Foundry Nuke 17 Machine Learning (ML) inference pipeline. By utilizing a custom PyTorch TorchScript wrapper.

## Setup Instructions

This implementation is purpose-built for **Nuke 17**, which utilizes **Python 3.11** and **Torch 2.2**. This environment ensures native compatibility with Nuke's internal specifications.

### Quick Start

```powershell
# 1. Create the isolated environment
mamba create --prefix ./nuke17_ml_env python=3.11 -y

# 2. Install all dependencies via the manifest
mamba run --prefix ./nuke17_ml_env pip install -r requirements.txt
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

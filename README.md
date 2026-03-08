# Nuke 17 ML Tools Pipeline

## Abstract
This repository contains production-ready integrations of two advanced ML models into the Foundry Nuke 17 pipeline:
1. **Depth Anything 3 (Small)**: A Vision Transformer port into Nuke's native `.cat` Inference pipeline.
2. **NormalCrafter**: A Video Diffusion Model for temporally consistent normal sequences, integrated via a seamless Python Gizmo wrapper.

## Setup Instructions
This implementation is purpose-built for **Nuke 17**, which utilizes **Python 3.11** and **Torch 2.2**. This environment ensures native compatibility with Nuke's internal specifications.

### Quick Start
1. **Create the isolated environment:**
   ```powershell
   mamba create --prefix ./nuke17_ml_env python=3.11 -y
   ```
2. **Install all dependencies via the manifest:**
   ```powershell
   mamba run --prefix ./nuke17_ml_env pip install -r requirements.txt
   ```

### Repository Structure
```
nuke-ml-tools/
├── nuke17_ml_env/           # Shared isolated mamba environment
├── requirements.txt         # Shared Python dependencies
├── tools/
│   ├── DepthAnything3/      # ViT DA3 Monocular Depth .cat Export
│   └── NormalCrafter/       # Video Diffusion Normal Map Generator
```

## Build Instructions (Depth Anything 3 .cat)
Before using the Nuke CatFileCreator, you must generate the PyTorch `.pt` file locally. Run the following command:
```powershell
mamba run --prefix ./nuke17_ml_env python tools/DepthAnything3/src/build_da3_small.py
```

## NormalCrafter Usage
Because NormalCrafter is an iterative video diffusion model, it runs externally rather than through a single CatFile node.
To use **NormalCrafter**:
1. Copy `tools/NormalCrafter/nuke/NormalCrafter.gizmo` and `tools/NormalCrafter/nuke/normalcrafter_nuke_ui.py` to your `.nuke` directory or anywhere on your Nuke plugin path (`NUKE_PATH`).
2. Inside Nuke, create a **NormalCrafter** node and plug in your sequence.
3. Click **Generate Normal Map Sequence**.
   *This automatically renders the input to a temporary EXR sequence, triggers the ML generation via subprocess using the `nuke17_ml_env`, and creates a Read node containing the temporally stable results once finished.*

## Git LFS Initialization
Binary assets and compiled `.cat` Nuke node configurations are managed via Git LFS to ensure repository performance.

```powershell
git init
git lfs install
git lfs track "*.cat"
git add .gitattributes
```

> [!NOTE]
> Intermediate `.pt` files are excluded from the repository via `.gitignore` to optimize bandwidth and LFS storage quota.

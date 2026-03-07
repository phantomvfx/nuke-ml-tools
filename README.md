# Nuke 17 ML: Vision Transformer (DA3) Integration Pipeline

## Abstract
This repository contains a production-ready port of the **Depth Anything 3 (Small)** Vision Transformer model into the Foundry Nuke 17 Machine Learning (ML) inference pipeline. By utilizing a custom PyTorch TorchScript wrapper, this tool enables high-quality monocular depth estimation directly within the Nuke node graph.

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

### Model Handling
This pipeline uses the `huggingface_hub` API to manage model weights.

*   **Automatic Acquisition:** The build script is configured to automatically download the **DA3-SMALL** weights from Hugging Face if they are not detected locally.
*   **Storage Configuration:** To define a custom storage location for models (e.g., a shared drive), set your environment variable:
    ```powershell
    set HF_HOME=C:\Path\To\Your\AI_Assets
    ```

## Build Instructions
Before using the Nuke CatFileCreator, you must generate the PyTorch `.pt` file locally. Run the following command:
```powershell
mamba run --prefix ./nuke17_ml_env python src/build_da3_small.py
```


# SSL-Satellite
Lightweight pipeline for preparing, publishing and training segmentation models on Sentinel‑2 RGB patches with aligned masks (ESA WorldCover / other sources). Designed for reproducible extraction, preprocessing, and Hugging Face dataset publishing. I recommend starting with the notebooks and then looking at the source code. 

## Key features
- Download Sentinel‑2 + WorldCover via Microsoft Planetary Computer (STAC).
- Reproject/resample categorical masks to Sentinel grid (nearest).
- Tile imagery into 256×256 image–mask pairs for training.
- Utilities to fix BGR→RGB, push patched datasets to Hugging Face, and simple training notebooks (UNet).

## Requirements
See requirements.txt. Install PyTorch separately per https://pytorch.org (choose correct CUDA/CPU wheel).

Recommended extras:
- huggingface-cli (huggingface-hub) for dataset publishing
- planetary-computer, pystac-client, rasterio, torchgeo, timm, opencv-python, shapely

## Quickstart
1. Create virtual env and install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   # install torch separately from https://pytorch.org
   ```
2. Open notebooks:
   ```bash
   jupyter lab
   ```
3. Configure bbox/year/paths in the notebooks and run cells.
4. To publish datasets:
   ```bash
   huggingface-cli login
   # run upload_images.ipynb cells
   ```

## Notebooks
- notebooks/upload_images.ipynb — build & push patched datasets; BGR→RGB fix.
- notebooks/satelite_segmentation.ipynb — download, align masks, tile into 256×256 pairs.
- notebooks/segmentation_training.ipynb — example training/visualization using UNet.

## Project layout
- notebooks/ — reproducible pipelines and examples
- src/data — patch extraction, alignment, IO utilities
- src/model — model definitions and training utilities
- requirements.txt — Python dependencies

import argparse
import glob
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

import numpy as np
import rasterio

from src.data.utils import _preprocess_sentinel2

def visualize_full(path, rgb_bands, max_display=16312, stretch=(2, 98)):
    """_summary_
    Visualize the satellite raster (requires a lot of memory for full size). 
    """
    with rasterio.open(path) as src:
        raw = src.read(rgb_bands, masked=True)   # shape: (B, H, W)
    numpy_img = _preprocess_sentinel2(raw)         # expected shape: (H, W, B) and floats in [0,1]

    H, W, C = numpy_img.shape

    # Downsample for display if huge (use simple striding to keep dependency-free)
    if max(H, W) > max_display:
        step = int(np.ceil(max(H, W) / max_display))
        numpy_scaled = numpy_img[::step, ::step, :]
        H, W, C = numpy_img.shape
    else:
        numpy_scaled = numpy_img

    lo, hi = stretch
    if C > 1:
        p_low = np.nanpercentile(numpy_scaled, lo, axis=(0, 1))
        p_high = np.nanpercentile(numpy_scaled, hi, axis=(0, 1))
        denom = np.where((p_high - p_low) == 0, 1.0, (p_high - p_low))
        img_stretched = (numpy_img - p_low) / denom
        img_stretched = np.clip(img_stretched, 0.0, 1.0)
    else:
        p_low = np.nanpercentile(numpy_scaled, lo)
        p_high = np.nanpercentile(numpy_scaled, hi)
        denom = (p_high - p_low) if (p_high - p_low) != 0 else 1.0
        img_stretched = np.clip((numpy_img - p_low) / denom, 0.0, 1.0)

    rgb = np.stack([img_stretched[..., k] for k in range(3)], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    plt.figure(figsize=(max(6, W/150), max(4, H/150)))
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Load .tiff images to images")
    parser.add_argument("--directory", type=str, default = "")
    parser.add_argument("--bands", type=str, default="3,2,1", 
                        help="comma-separated band indices, e.g. 3,2,1")
    parser.add_argument("--restrict_img", type=int, default=16312)
    parser.add_argument("--pattern", type=str, default="**/*.tif",
                        help="glob pattern relative to directory (supports ** for recursion)")
    args = parser.parse_args()

    bands = tuple(int(b.strip()) for b in args.bands.split(",") if b.strip())

    files = sorted(glob.glob(
        os.path.join(args.directory, args.pattern), 
        recursive=True)
        )
    
    print(f"Found {len(files)} file(s) matching pattern")

    for idx, f in enumerate(files):
        print(f"[{idx}/{len(files)}] Visualizing file: {f}")
        visualize_full(f, bands, args.restrict_img)

if __name__ == "__main__":
    main()

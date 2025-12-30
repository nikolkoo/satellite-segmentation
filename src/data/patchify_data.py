import argparse
import glob
from itertools import zip_longest
import json
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import numpy as np
import rasterio
from pathlib import Path

from src.data.utils import _make_json_serializable, _numpy_to_image, _patch_to_pil_mask, _preprocess_sentinel2

def _mask_within_threshold(patch, min_threshold = 0.95):
    """_summary_
    To assure that the image is acceptable we validate the mask. 
    The mask is used to assure that we are within the covered image. 
    """
    combined_mask = np.any(np.ma.getmaskarray(patch), axis=0)  # shape (H, W)
    valid = ~combined_mask

    # Get proportion of image over the whole image
    valid_frac = float(valid.sum()) / (valid.size)

    return valid_frac > min_threshold

def _mask_has_significant_classes(patch, background_value=0, min_non_bg_frac=0.1):
    """_summary_
    Return True if the mask patch contains at least `min_non_bg_frac`
    pixels that are not equal to `background_value`.
    `patch` is the masked-array returned from rasterio (bands, H, W).
    """
    # support single-band and multi-band masks (take first band)
    arr = patch[0] if patch.ndim == 3 else patch
    # get unmasked values
    vals = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
    if vals.size == 0:
        return False
    non_bg = np.count_nonzero(vals != background_value)
    return (non_bg / float(vals.size)) >= float(min_non_bg_frac)

def _black_within_threshold(img, max_threshold = 0.2):
    """_summary_
    Assure that the images contains less black than max_threshold
    """
    patch = np.asarray(img)
    black_value = (5.0 / 255.0) if patch.dtype.kind == "f" else 5
    channels = range(patch.shape[2])

    # Assume that patch is of shape [H,W,B]
    masks = [(patch[..., c] < black_value) for c in channels]
    black_mask = np.logical_and.reduce(masks)
    black_frac = float(np.count_nonzero(black_mask)) / black_mask.size

    return black_frac < max_threshold

def _is_texturally_interesting(img, std_thresh=0.04, grad_thresh=0.01):
    """_summary_
    Return True if patch is visually 'boring' (flat color / low texture).
    - img_patch: float ndarray in [0,1], shape (H,W,B)
    - thresholds tuned for normalized data; adjust if your data range differs.
    """
    img_patch = np.asarray(img) / 255
    # make grayscale luminance
    if img_patch.ndim == 3 and img_patch.shape[2] >= 3:
        r, g, b = img_patch[..., 0], img_patch[..., 1], img_patch[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = img_patch[..., 0] if img_patch.ndim == 3 else img_patch

    # 1) low standard deviation -> flat
    std = float(np.nanstd(gray))
    if std < std_thresh:
        return False

    # 2) low gradient magnitude mean -> few edges / texture
    gy, gx = np.gradient(gray)
    grad_mean = float(np.nanmean(np.sqrt(gx * gx + gy * gy)))
    if grad_mean < grad_thresh:
        return False

    return True

def useful_image(img_patch):
    """_summary_
    Helper function to define the criteria for an accepted image.
    """
    return (_black_within_threshold(img_patch) and 
        _is_texturally_interesting(img_patch))

def _iterate_patches(path, bands, tile_size = 512, logger = None, threshold = _mask_within_threshold):
    """_summary_
    Iterate through a large image after loading the image with Rasterio. 
    """
    with rasterio.open(path) as src:
        n_cols = src.width  // tile_size
        n_rows = src.height // tile_size
        for i in range(n_rows):
            row_off = i * tile_size
            for j in range(n_cols):
                col_off = j * tile_size

                # handle edge tiles by clipping window size
                read_w = min(tile_size, max(0, src.width  - col_off))
                read_h = min(tile_size, max(0, src.height - row_off))
                if read_w == 0 or read_h == 0:
                    continue

                win = Window(col_off, row_off, read_w, read_h)
                patch = src.read(bands, window=win, masked=True)  # shape (bands, H, W)

                # Only yield masks that are within threshold
                if threshold(patch):
                    yield row_off, col_off, patch

        # --- CALCULATE STATISTICS ---
        total_area = src.width * src.height
        area_covered = n_cols * n_rows * tile_size**2
        percentage_covered = area_covered / total_area

        logger["percentage covered"] = percentage_covered
        print(f"{percentage_covered*100:.2f}% of the patch iterated")

def save_as_mask_pairs(path_mask: str, path_img: str, bands_mask = (1,), bands_img = (1,2,3), img_size = 256, img_prefix = "", out_directory = "images", logger = None):
    """_summary_
    Save satellite mask-image pairs.
    """
    # --- DIRECTORY FOR STORED IMAGES ---
    out_directory = Path(out_directory) 
    out_directory.mkdir(parents = True, exist_ok = True)

    mask_path = out_directory / "mask"
    mask_path.mkdir(parents = True, exist_ok = True)

    img_path = out_directory / "img"
    img_path.mkdir(parents = True, exist_ok = True)

    # --- ADD LOGGING ---
    if logger is None:
        logger = {}
    stats = logger.setdefault(str(out_directory), {})

    # --- CREATE AND ITERATE PATCHES ---
    _always_true = lambda patch, **kwargs: True
    img_patches = _iterate_patches(path_img, bands_img, img_size, stats, threshold=_always_true)
    masked_patches = _iterate_patches(path_mask, bands_mask, img_size, stats, threshold=_always_true)
    
    total, kept = 0, 0
    for img_item, mask_item in zip_longest(img_patches, masked_patches):
        if img_item is None or mask_item is None:
            # mismatch in yielded windows — skip to keep pairs aligned
            continue

        r_img, c_img, img_patch = img_item
        r_mask, c_mask, mask_patch = mask_item

        # skip if windows differ
        if (r_img, c_img) != (r_mask, c_mask):
            # windows not aligned — skip this pair 
            continue

        total += 1

        # preprocess and check image quality
        img_patch = _preprocess_sentinel2(img_patch)
        img = _numpy_to_image(img_patch)
        img_ok = useful_image(img)

        # check mask has useful (non-background) content
        mask_ok = _mask_has_significant_classes(mask_patch, background_value=0)

        # only save pair when both are acceptable
        if img_ok and mask_ok:
            img_name = f"{img_prefix}_{r_img}_{c_img}.png"
            img.save(img_path / img_name)

            pil_mask = _patch_to_pil_mask(mask_patch)
            name = f"{img_prefix}_{r_img}_{c_img}_mask.png"
            pil_mask.save(mask_path / name)

            kept += 1

    print(f"[{kept}/{total}] images saved")

    stats["kept"] = kept
    stats["total"] = total

    return logger

def save_as_images(path: str, bands = (1,2,3), img_size = 256, img_prefix = "", out_directory = "images", logger = None):
    """_summary_
    Divide images into single images. Similar to save_as_mask_pairs, but for single images.
    """
    # --- DIRECTORY FOR STORED IMAGES ---
    out_directory = Path(out_directory)
    out_directory.mkdir(parents = True, exist_ok = True)

    filtered_path = out_directory / img_prefix
    filtered_path.mkdir(parents = True, exist_ok = True)

    # --- ADD LOGGING ---
    if logger is None:
        logger = {}
    stats = logger.setdefault(str(filtered_path), {})

    # --- CREATE AND ITERATE PATCHES ---
    all_patches = _iterate_patches(path, bands, img_size, stats)
    total, kept = 0, 0
    for (r, c, img_patch) in all_patches:
        total += 1
        img_patch = _preprocess_sentinel2(img_patch)
        img = _numpy_to_image(img_patch) 

        # Save the image if it contains useful data
        if useful_image(img):
            img_name = f"{img_prefix}_{r}_{c}.png"
            img.save(out_directory / img_name)
            kept += 1
        else:
            # TODO: implement such that no useful images are stored separately
            img_name = f"{img_prefix}_{r}_{c}_filtered.png"
            #img.save(filtered_path / img_name)
    print(f"[{kept}/{total}] images saved")

    stats["kept"] = kept
    stats["total"] = total

    return logger
    
def main():
    parser = argparse.ArgumentParser(description="Load .tiff images to images")
    parser.add_argument("--directory", type=str, default = "")
    parser.add_argument("--bands", type=str, default="3,2,1", 
                        help="comma-separated band indices, e.g. 1,2,3")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_prefix", type=str, default="patch")
    parser.add_argument("--out_directory", type=str, default="image_patches")
    parser.add_argument("--pattern", type=str, default="**/*.tif",
                        help="glob pattern relative to directory (supports ** for recursion)")
    args = parser.parse_args()

    bands = tuple(int(b.strip()) for b in args.bands.split(",") if b.strip())

    files = sorted(glob.glob(
        os.path.join(args.directory, args.pattern), 
        recursive=True)
        )
    
    print(f"Found {len(files)} file(s) matching pattern")

    # TODO: add the files into different folders
    logger = {}
    for idx, f in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing file: {f}")
        prefix = args.img_prefix + str(idx)
        save_as_images(f, bands, args.img_size, prefix, args.out_directory, logger)

    # --- SAVE LOGGER ---
    logger_path = Path(args.out_directory) / "logging"
    with open(logger_path, "w", encoding = "utf-8") as f:
        json.dump(_make_json_serializable(logger), f, indent = 2)

if __name__ == "__main__":
    main()

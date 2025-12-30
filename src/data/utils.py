import numpy as np
from PIL import Image

def _preprocess_sentinel2(img_patch):
    """_summary_
    A simple function to scale the sentinel2 image from Kartverket.
    """
    # Scale patch to float if type is integer or unsigned integer
    if img_patch.dtype in ("int16", "uint16"):
        img_patch = img_patch.astype(np.float32) / 10000.0
    
    # Mask out the values within the mask
    if np.ma.isMaskedArray(img_patch):
        img_patch = img_patch.filled(0.0)

    # Reshape from [C,H,W] -> [H,W,C]
    img_patch = np.transpose(img_patch, (1,2,0)) 

    return img_patch

def _numpy_to_image(img_patch):
    """_summary_
    Convert the raw float image to an numpy format. 
    """
    # Scale to 255 integers for screen viewing
    if img_patch.dtype.kind == "f":
        gamma = 2.2
        img_scaled = img_patch ** (1.0 / gamma)
        img_scaled = np.clip(img_scaled, 0.0, 1.0)
        img_patch = (img_scaled * 255.0).round().astype(np.uint8)

    img = Image.fromarray(img_patch, mode = "RGB")
    return img

def _patch_to_pil_mask(patch):
    # Code written with Chat-GPT5
    """_summary_
    Convert rasterio patch (bands, H, W) for a single-band categorical mask
    into a PIL image (mode 'L'). Handles masked arrays by filling with 0.
    """
    # patch may be shape (1, H, W) or (H, W)
    if patch.ndim == 3:
        arr = patch[0]
    else:
        arr = patch

    # if masked array, fill nodata with 0
    if np.ma.is_masked(arr):
        arr = arr.filled(0)

    # ensure integer type and in uint8 range
    arr = arr.astype(np.int32)  # keep class codes safe
    # if codes < 256 convert to uint8, otherwise map/rescale as needed
    if arr.max() < 256:
        arr = arr.astype(np.uint8)
        mode = "L"
    else:
        # keep as 16-bit if classes exceed 255 (PIL supports 'I;16')
        arr = arr.astype(np.uint16)
        mode = "I;16"

    return Image.fromarray(arr, mode=mode)

def _make_json_serializable(obj):
    """_summary_
    Recursively convert numpy/scalar types to plain Python for JSON.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj) 
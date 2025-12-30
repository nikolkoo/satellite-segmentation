import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from model.get_pretrained import torchgeo_res50
from model.load_masks import load_satelite_masks
from model.res50_unet import build_unet_from_resnet50

def get_colormap(pred):
    """_summary_
    Get the colormap used by Worldcover.
    """
    # default colormap for 11 classes
    # https://collections.sentinel-hub.com/worldcover/readme.html
    if class_colors is None:
        default_colors = np.array([
                [  0,   0,   0],  # 0  No data
                [  0, 100,   0],  # 1  0x006400  Tree cover (10)
                [255, 187,  34],  # 2  0xffbb22  Shrubland (20)
                [255, 255,  76],  # 3  0xffff4c  Grassland (30)
                [240, 150, 255],  # 4  0xf096ff  Cropland (40)
                [250,   0,   0],  # 5  0xfa0000  Built up (50)
                [180, 180, 180],  # 6  0xb4b4b4  Bare / sparse veg (60)
                [240, 240, 240],  # 7  0xf0f0f0  Snow and Ice (70)
                [  0, 100, 200],  # 8  0x0064c8  Permanent water bodies (80)
                [  0, 150, 160],  # 9  0x0096a0  Herbaceous wetland (90)
                [250, 230, 160],  #10  0xfae6a0  Moss and lichen (100)
            ], dtype=np.uint8)
        class_colors = default_colors

    # ensure class_colors covers all predicted classes
    maxc = pred.max()
    if maxc >= len(class_colors):
        # extend by repeating a colormap
        rng = np.random.RandomState(0)
        extra = rng.randint(0, 255, size=(maxc - len(class_colors) + 1, 3), dtype=np.uint8)
        class_colors = np.vstack([class_colors, extra])
    return class_colors

def preprocess_pairs(img, mask):
    """_summary_
    Adjust the images and mask to make them suitable for visualization. 
    """
    # prepare numpy image
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
    else:
        arr = np.array(img)
    
    # assert that the images are of the correct shape
    if arr.ndim == 2:  # single channel -> convert to 3-channel gray
        arr = np.stack([arr]*3, axis=-1)
    h, w, c = arr.shape
    assert c >= 3, "expected 3-channel image"

    # normalize to [0,1] float32
    if arr.dtype == np.uint8:
        img_f = arr.astype(np.float32) / 255.0
    else:
        img_f = arr.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

    return img_f, mask

def visualize_segmentation(model, img, mask, device=None, class_colors=None, alpha=0.5):
    """_summary_
    Visualize the segmentation image-mask pair using the predicted result from model. 
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare numpy image
    img_f, mask = preprocess_pairs(img, mask)

    # tensor shape (1, C, H, W)
    inp = torch.from_numpy(img_f[..., :3]).permute(2,0,1).unsqueeze(0).float().to(device)

    # predict logit for classes
    model.eval()
    with torch.no_grad():
        logits = model(inp)                 
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32) 

    class_colors = get_colormap(pred)

    color_mask = class_colors[pred]  # (H, W, 3) uint8
    truth_mask = class_colors[mask.squeeze(-1).cpu().numpy().astype(np.int32)]

    # plot original + overlay
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.imshow(img_f if img_f.max() <= 1.0 else img_f/255.0)
    ax.imshow(color_mask, alpha=alpha)
    ax.axis("off")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.imshow(img_f if img_f.max() <= 1.0 else img_f/255.0)
    ax.imshow(truth_mask, alpha=alpha)
    ax.axis("off")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.imshow(img_f if img_f.max() <= 1.0 else img_f/255.0)
    ax.axis("off")
    plt.show()

    return pred

def main():
    encoder = torchgeo_res50()

    NUM_CLASSES = 11
    unet = build_unet_from_resnet50(encoder, num_classes = NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    ds = load_satelite_masks(wrap_dataloader = False)
    # Load image
    img_idx = 500

    img = ds["image"][img_idx]
    mask = ds["mask"][img_idx]

    ckpt_path = Path("model") / f"checkpoint_epoch_50.pt"  # adjust filename
    ckpt = torch.load(ckpt_path, map_location=device)

    # handle possible DataParallel 'module.' prefix
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    unet.load_state_dict(state)
    visualize_segmentation(unet, img, mask, alpha = 1)
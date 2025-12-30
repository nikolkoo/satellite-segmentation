from pathlib import Path
import random
import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.model.get_pretrained import torchgeo_res50
from src.model.load_masks import load_satelite_masks
from src.model.res50_unet import build_unet_from_resnet50

def dice_loss_logits(pred_logits, target, eps=1e-6):
    # https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    pred = F.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    inter = (pred * target_onehot).sum(dim=(0,2,3))
    denom = (pred + target_onehot).sum(dim=(0,2,3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def augment_batch(imgs, masks, p_hflip=0.5, p_vflip=0.5, p_rot=0.5, p_bright=0.2):
    """_summary_
    Augments the image by flipping them randomly and adjusting the brightness.
    """
    B = imgs.shape[0]
    for i in range(B):
        # horizontal flip
        if random.random() < p_hflip:
            imgs[i] = imgs[i].flip(-1)
            masks[i] = masks[i].flip(-1)
        # vertical flip
        if random.random() < p_vflip:
            imgs[i] = imgs[i].flip(-2)
            masks[i] = masks[i].flip(-2)
        # random 90-degree rotation
        if random.random() < p_rot:
            k = random.choice([1, 2, 3])  # multiples of 90 deg
            imgs[i] = torch.rot90(imgs[i], k, dims=(1,2))
            masks[i] = torch.rot90(masks[i], k, dims=(0,1))
        # brightness jitter 
        if random.random() < p_bright:
            factor = 0.8 + 0.4 * random.random()  # 0.8 .. 1.2
            imgs[i] = (imgs[i] * factor).clamp(0.0, 1.0)
    return imgs, masks

def train_one_epoch(model, loader, optimizer, device, accum_steps=1, lambda_dice=0.5):
    """_summary_
    One iteration through the training epoch. 
    """
    model.train()
    running_loss = 0.0
    seen = 0
    for step, batch in enumerate(loader):
        imgs = batch['image'].to(device) # shape: (B, C, H, W)
        if imgs.dtype != torch.float32:
            imgs = imgs.float() / 255.0
        masks = batch['mask'].to(device).long() # shape: (B, H, W)
        masks = masks.squeeze(1)

        imgs, masks = augment_batch(imgs, masks)

        optimizer.zero_grad()

        logits = model(imgs)
        ce = F.cross_entropy(logits, masks)
        d = dice_loss_logits(logits, masks)
        loss = ce + lambda_dice * d
        loss = loss / accum_steps

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * imgs.size(0) * accum_steps
        seen += imgs.size(0)

    avg_loss = running_loss / max(1, seen)
    return avg_loss

def get_scheduler(warmup_epochs, epochs, optimizer):
    """_summary_
    Load the default scheduler for the training. 
    """
    opt = getattr(optimizer, "optim", optimizer)
    def _lr_schedule(curr_epoch):
        if curr_epoch < warmup_epochs:
            # Linear warmup
            return float(curr_epoch + 1) / warmup_epochs
        
        # Cosine decay
        return 0.5 * (1 + math.cos(math.pi * (curr_epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = LambdaLR(opt, lr_lambda=_lr_schedule)
    return scheduler


def run_training(model, train_loader, epochs=70, lr=1e-4, num_classes=9, device=None, save_path=None):
    """_summary_
    Train the model for segmentation. 
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = get_scheduler(10, epochs, optimizer)
    save_path = Path(save_path)
    writer_path =save_path / "logging"
    writer_path.mkdir(parents=True, exist_ok = True)

    writer = SummaryWriter(log_dir=writer_path)

    save_epochs = [1, 50, 70]
    head_epochs = 30
    for epoch in range(1, epochs+1):
        # switch to fine-tune when head stage finished
        if epoch == head_epochs:
            # unfreeze last block if present
            for p in model.encoder.layer4.parameters():
                p.requires_grad = True
            # recreate optimizer with a lower lr for unfrozen params
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
            scheduler = get_scheduler(10, epochs - head_epochs + 1, optimizer)  # restart schedule for remaining epochs

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        msg = (f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f}")
        print(msg)

        if epoch in save_epochs and save_path is not None:
            ckpt_path = save_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_path)

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("LR/epoch", optimizer.param_groups[0]['lr'], epoch)
    writer.close()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchgeo_res50()

    NUM_CLASSES = 11
    unet = build_unet_from_resnet50(model, num_classes = NUM_CLASSES)
    unet.to(device)
    
    batch_size = 8
    num_workers = 4
    loader = load_satelite_masks(batch_size, num_workers)

    save_path = "model"
    run_training(unet, loader, device = device, save_path = save_path)
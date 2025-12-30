from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

def make_random_train_val_loaders(dataset, val_frac=0.15, batch_size=8, seed=42, num_workers=4):
    """_summary_
    Split the dataset into training and validation datasets. 
    """
    n = len(dataset)
    val_n = int(n * val_frac)
    train_n = n - val_n
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def load_satelite_masks(url = "nikolkoo/SatelliteSegmentation", batch_size = 8, num_workers = 8, wrap_dataloader = True):
    """_summary_
    Load the satellite image-mask pairs from the url. 
    """
    ds = load_dataset(url, split = "train")
    
    #train_loader, val_loader = make_random_train_val_loaders(ds, batch_size = batch_size)
    ds = ds.map(lambda ex: {"mask": (np.array(ex["mask"], dtype = np.uint8) // 10).astype(np.uint8)}, batched=True)
    ds = ds.with_format("torch")
    if wrap_dataloader:
        ds = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    return ds
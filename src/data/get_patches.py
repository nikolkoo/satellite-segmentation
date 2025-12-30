#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURE THESE ---
HF_DATASET = "nikolkoo/Sentinel2RGBNorway"   # your HF dataset name
IMAGE_EXT = ".png"                           # choose extension for saved images

def main():
    """_summary_
    Code loads dataset from huggingface and saves them as images. 
    """
    parser = argparse.ArgumentParser(description="Load sentinel2 images")
    parser.add_argument("--out_directory", type=str, default="patched_data")
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.out_directory, exist_ok=True)

    # Load the HuggingFace dataset
    print(f"Loading HF dataset '{HF_DATASET}' ...")
    ds = load_dataset(HF_DATASET, split="train")  # assuming single 'train' split

    print(f"Dataset has {len(ds)} images. Saving to {args.out_directory} ...")

    # Iterate and save images
    for idx, row in enumerate(tqdm(ds, desc="Saving images")):
        img = row["image"]  # this is a PIL Image
        save_path = os.path.join(args.out_directory, f"{idx}{IMAGE_EXT}")
        img.save(save_path)

    print("All images saved. Ready for SimCLR training!")

if __name__ == "__main__":
    main()

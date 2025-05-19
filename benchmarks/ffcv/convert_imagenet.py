"""Convert an ImageNet dataset (with synset folders) to PyTorch ImageFolder format (class index folders).

Usage:
    python convert_imagenet.py --data_dir /path/to/imagenet-raw/train

This script helps you:
- Change the folder structure from synset names (like n01440764) to class index numbers (0, 1, 2, ...)
- Move all images into their new class index folders
- Clean up by removing the old synset folders

After running this script, your dataset will be ready to use with torchvision.datasets.ImageFolder or
any PyTorch dataloader expecting class index folders.
"""

import json
import os
from argparse import ArgumentParser
from functools import lru_cache
from glob import glob

import requests
from torchvision.datasets import ImageFolder
from tqdm import tqdm


@lru_cache(maxsize=1)
def load_imagenet_class_index():
    """Load the ImageNet class index mapping from class names to class indices."""
    class_index_url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    try:
        response = requests.get(class_index_url, timeout=10)
        response.raise_for_status()
        class_index_data = response.json()
        return {v[0]: int(k) for k, v in class_index_data.items()}
    except (requests.RequestException, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load ImageNet class index: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert ImageNet synset folders to PyTorch ImageFolder style.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the ImageNet dataset directory (containing synset folders).",
    )
    args = parser.parse_args()
    imagenet_dir = args.data_dir
    if not os.path.exists(imagenet_dir):
        raise FileNotFoundError(f"The specified directory does not exist: {imagenet_dir}")

    print("[INFO] Loading ImageNet class index mapping...")
    class_index_mapping = load_imagenet_class_index()

    # Create class index folders if not present
    print("[INFO] Creating class index folders...")
    for _, class_index in class_index_mapping.items():
        folder_path = os.path.join(imagenet_dir, str(class_index))
        os.makedirs(folder_path, exist_ok=True)

    # Move images from synset folders to class index folders
    print("[INFO] Moving images to class index folders...")
    for file_path in tqdm(glob(f"{imagenet_dir}/*/*.*"), desc="Moving files"):
        dirname = os.path.basename(os.path.dirname(file_path))
        if not dirname.startswith("n"):
            continue
        class_index = class_index_mapping[dirname]
        destination_path = os.path.join(imagenet_dir, str(class_index), os.path.basename(file_path))
        os.rename(file_path, destination_path)

    # Remove old synset folders
    print("[INFO] Removing old synset folders...")
    for folder in tqdm(glob(f"{imagenet_dir}/*"), desc="Removing old folders"):
        if os.path.basename(folder).startswith("n") and os.path.isdir(folder):
            try:
                os.rmdir(folder)
            except OSError:
                print(f"[WARNING] Could not remove folder (not empty?): {folder}")

    print("[SUCCESS] Conversion complete.")
    print("[INFO] All images are now organized in class index folders.")
    print("[INFO] You can now use the dataset with torchvision.datasets.ImageFolder.")

    # Show a sample from the resulting dataset
    dataset = ImageFolder(root=imagenet_dir, transform=None)
    print(f"[INFO] Number of classes: {len(dataset.classes)}")
    print(f"[INFO] Example class indices: {dataset.classes[:5]}")
    if len(dataset) > 0:
        print(f"[INFO] Sample image path: {dataset.samples[0][0]}, class index: {dataset.samples[0][1]}")
    else:
        print("[INFO] No images found in the converted dataset.")

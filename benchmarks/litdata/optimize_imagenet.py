"""Optimize ImageNet dataset for benchmarking using litdata.

This script provides functionality to optimize ImageNet images for benchmarking purposes by applying resizing,
format conversion, and other optimizations.
"""

import argparse
import io
import json
import os
import time
from functools import lru_cache, partial
from typing import Union

import numpy as np
import requests
from lightning import seed_everything
from PIL import Image
from tqdm import tqdm

from litdata import optimize, walk


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


def get_class_from_filepath(filepath: str, classes: dict) -> int:
    """Extract class index from file path."""
    class_name = os.path.basename(os.path.dirname(filepath))
    return classes[class_name]


def get_inputs(input_dir: str):
    """Get inputs for optimization: file paths and class indices."""
    classes = load_imagenet_class_index()
    filepaths = np.random.permutation(
        [
            os.path.join(root, filename)
            for root, _, filenames in tqdm(walk(input_dir), smoothing=0)
            for filename in filenames
        ]
    )
    return [(filepath, get_class_from_filepath(filepath, classes)) for filepath in filepaths]


def optimize_fn(data: tuple[str, int], args: dict) -> tuple[Image.Image, int]:
    """Optimization function for each image."""
    filepath, class_index = data
    img = Image.open(filepath)
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize if requested
    if args.get("resize") and args.get("resize_size") is not None:
        resize_size = args["resize_size"]
        # If int, scale max dimension to resize_size, preserving aspect ratio
        if isinstance(resize_size, int):
            max_dim = max(img.size)
            scale = resize_size / max_dim
            new_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(new_size)
        # If tuple, resize to exact (width, height)
        elif isinstance(resize_size, tuple) and len(resize_size) == 2:
            img = img.resize(resize_size)
    # Format conversion
    if args.get("write_mode") == "jpeg":
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=args.get("quality", 90))
        buff.seek(0)
        img = Image.open(buff)
    elif args.get("write_mode") == "pil":
        img = Image.frombytes(img.mode, img.size, img.tobytes())
    return img, class_index


def main():
    """Main function to parse arguments and optimize the dataset."""
    parser = argparse.ArgumentParser(description="Optimize ImageNet dataset for benchmarking using litdata.")
    parser.add_argument("--input_dir", required=True, help="Input directory for raw dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for optimized dataset")
    parser.add_argument("--write_mode", choices=["jpeg", "pil"], default=None, help="Store images in specified format.")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality if JPEG is selected")
    parser.add_argument(
        "--resize",
        action="store_true",
        help=(
            "Resize images. If --resize_size is an int, the largest dimension will be scaled to this value,"
            ", preserving aspect ratio. If a tuple is provided, the image will be resized to the exact"
            "(width, height) specified."
        ),
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        nargs="+",
        default=None,
        help="Resize size: int for max dimension (aspect ratio preserved), or two ints for (width height)",
    )
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for optimization")
    parser.add_argument("--chunk_bytes", type=str, default="64MB", help="Chunk size for optimization")
    parser.add_argument(
        "--reorder_files",
        action="store_true",
        help="Whether to reorder files for optimal storage/performance.",
    )
    parser.add_argument("--num_downloaders", type=int, default=10, help="Number of downloaders to use for optimize fn.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    print(f"[INFO] Running optimize with arguments: {args}")

    seed_everything(args.seed)

    # Handle resize_size: if two ints are given, treat as tuple, else int or None
    resize_size: Union[int, tuple[int, int], None] = None
    if args.resize_size is not None:
        if isinstance(args.resize_size, list):
            if len(args.resize_size) == 1:
                resize_size = args.resize_size[0]
            elif len(args.resize_size) == 2:
                resize_size = tuple(args.resize_size)
        else:
            resize_size = args.resize_size

    optimize_args = {
        "resize": args.resize,
        "resize_size": resize_size,
        "write_mode": args.write_mode,
        "quality": args.quality,
    }

    is_train = "train" in args.input_dir.lower()
    if not is_train:
        raise ValueError("Only training dataset optimization is supported. Please provide a 'train' directory.")

    inputs = get_inputs(args.input_dir)

    print(f"Optimizing {len(inputs)} images from {args.input_dir}...")
    start_time = time.perf_counter()
    optimize(
        fn=partial(optimize_fn, args=optimize_args),
        inputs=inputs,
        output_dir=args.output_dir,
        chunk_bytes=args.chunk_bytes,
        reorder_files=args.reorder_files,
        num_downloaders=args.num_downloaders,
        num_workers=args.num_workers,
    )
    end_time = time.perf_counter()
    print(f"Time taken to optimize dataset: {end_time - start_time:.2f} seconds")
    print("Done!")


if __name__ == "__main__":
    main()

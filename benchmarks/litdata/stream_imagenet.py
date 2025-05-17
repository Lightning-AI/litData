"""Benchmark ImageNet streaming dataset with litdata. Supports JPEG and PIL formats."""

import argparse
import os
import shutil
import time
from contextlib import suppress

import lightning as L
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

from litdata import StreamingDataLoader, StreamingDataset
from litdata.utilities.dataset_utilities import get_default_cache_dir


def clear_cache_dir(cache_dir: str) -> None:
    """Clear the cache directory."""
    with suppress(Exception):
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)


def to_rgb(img):
    """Convert image to RGB format."""
    if isinstance(img, torch.Tensor):
        if img.shape[0] == 1:
            img = img.repeat((3, 1, 1))
        if img.shape[0] == 4:
            img = img[:3]
    else:
        if img.mode == "L":
            img = img.convert("RGB")
    return img


def main():
    """Benchmark ImageNet streaming dataset with litdata. Supports JPEG and PIL formats."""
    parser = argparse.ArgumentParser(description="Streaming ImageNet benchmark with litdata.")
    parser.add_argument("--input_dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--cache_dir", default=get_default_cache_dir(), help="Path to the cache directory")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16"], help="Data type: float32 or float16"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for benchmarking")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for dataloader")
    parser.add_argument(
        "--drop_last", dest="drop_last", action="store_true", help="Drop the last incomplete batch (default)"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to run benchmark")
    parser.add_argument("--max_cache_size", default="200GB", help="Max cache size for streaming dataset")
    parser.add_argument("--use_pil", action="store_true", help="If set, applies T.ToImage() as the first transform.")
    parser.add_argument(
        "--clear_cache",
        dest="clear_cache",
        action="store_true",
        help="Clear the cache directory before and after running the benchmark (default)",
    )
    parser.add_argument("--no_clear_cache", dest="clear_cache", action="store_false", help="Do not clear cache")
    parser.set_defaults(clear_cache=True)
    args = parser.parse_args()
    print(f"[INFO] Running streaming benchmark with arguments: {args}")

    L.seed_everything(42)

    if args.clear_cache:
        print(f"[INFO] Clearing cache directory: {args.cache_dir}")
        clear_cache_dir(args.cache_dir)

    # Compose transforms based on format
    transforms = []
    if args.use_pil:
        transforms.append(T.ToImage())
    transforms.extend(
        [
            T.RandomResizedCrop(224, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToDtype(torch.float32 if args.dtype == "float32" else torch.float16, scale=True),
        ]
    )
    transform = T.Compose(transforms)

    class ImageNetStreamingDataset(StreamingDataset):
        def __init__(self, transform, *a, **kw):
            self.transform = transform
            super().__init__(*a, **kw)

        def __getitem__(self, index):
            img, class_index = super().__getitem__(index)
            if self.transform is not None:
                img = self.transform(img)
            return to_rgb(img), int(class_index)

    print(f"[INFO] Initializing streaming dataset from: {args.input_dir}")
    dataloader = StreamingDataLoader(
        ImageNetStreamingDataset(
            transform=transform,
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            max_cache_size=args.max_cache_size,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    print(
        f"[INFO] Starting benchmark for {args.epochs} epoch(s) with batch size {args.batch_size} "
        f"and {args.num_workers} workers."
    )
    for epoch in range(args.epochs):
        num_samples = 0
        t0 = time.perf_counter()
        for data in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", smoothing=0, mininterval=1):
            num_samples += data[0].shape[0]
        elapsed = time.perf_counter() - t0
        print(
            f"[RESULT] Epoch {epoch + 1}: Streamed {num_samples} samples in {elapsed:.2f}s "
            f"({num_samples / elapsed:.2f} images/sec)"
        )

    if args.clear_cache:
        print(f"[INFO] Clearing cache directory after benchmark: {args.cache_dir}")
        clear_cache_dir(args.cache_dir)
    print("[INFO] Finished streaming benchmark.")


if __name__ == "__main__":
    main()

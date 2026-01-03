"""Benchmark streaming raw dataset with litdata."""

import argparse
import os
import shutil
import time
from contextlib import suppress

import torch
import torchvision.transforms.v2 as T
from litdata.streaming.raw_dataset import StreamingRawDataset
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, decode_image, decode_jpeg
from tqdm import tqdm


def clear_cache_dir(cache_dir: str) -> None:
    """Clear the cache directory."""
    with suppress(Exception):
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)


def deserialize_jpeg(data: bytes) -> torch.Tensor:
    """Deserialize JPEG bytes to a tensor."""
    arr = torch.frombuffer(data, dtype=torch.uint8)
    with suppress(RuntimeError):
        return decode_jpeg(arr, mode=ImageReadMode.RGB)

    return decode_image(arr, mode=ImageReadMode.RGB)


class StreamingRawImageDataset(StreamingRawDataset):
    """Streaming raw dataset for ImageNet with JPEG deserialization."""

    def __init__(self, *args, **kwargs):
        """Initialize the dataset with optional transform."""
        super().__init__(*args, **kwargs)


def get_transform(dtype=torch.float32):
    """Get the transform for the dataset."""
    return T.Compose(
        [
            T.Lambda(deserialize_jpeg),
            T.Resize((224, 224)),
            T.ToDtype(dtype, scale=True),
        ]
    )


def main():
    """Benchmark streaming raw dataset with litdata."""
    parser = argparse.ArgumentParser(description="Streaming raw dataset benchmark with litdata")
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="If set, dataset yields raw bytes without transform (default is False)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Data type for tensor mode (ignored if --bytes set)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="s3://imagenet-1m-template/raw/train",
        help="Input directory or S3 path for the dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="imagenet_cache",
        help="Local cache directory",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Number of workers")
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear cache directory before and after running",
    )
    parser.set_defaults(clear_cache=False)
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to run")

    args = parser.parse_args()

    if args.clear_cache:
        print(f"[INFO] Clearing cache directory before run: {args.cache_dir}")
        clear_cache_dir(args.cache_dir)

    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    if args.bytes:
        print("[INFO] Running in BYTES mode (raw bytes, no transform)")
    else:
        print(f"[INFO] Running in TENSOR mode with dtype={dtype}")

    print("[INFO] Indexing dataset...")
    start_idx = time.perf_counter()
    dataset = StreamingRawImageDataset(
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        transform=get_transform(dtype) if not args.bytes else None,
    )

    length = len(dataset)
    print(f"[INFO] Indexed {length} samples in {time.perf_counter() - start_idx:.2f}s")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        total = 0
        t0 = time.perf_counter()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=1):
            total += len(batch)
        elapsed = time.perf_counter() - t0
        print(
            f"[RESULT] Epoch {epoch + 1}: Processed {total} samples in {elapsed:.2f}s "
            f"({total / elapsed:.2f} samples/sec)"
        )

    if args.clear_cache:
        print(f"[INFO] Clearing cache directory after run: {args.cache_dir}")
        clear_cache_dir(args.cache_dir)

    print("[INFO] Finished benchmark.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full 2-epoch ImageNet S3 bench matching the historic Studio script.

No batch limit — streams the entire dataset each epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm

from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset

INPUT = "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search"
CACHE = "/cache/chunks"


def to_rgb(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    elif img.shape[0] == 4:
        img = img[:3]
    return img


class ImageNetStreamingDataset(StreamingDataset):
    def __init__(self, *args, **kwargs):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float16, scale=True),
            ]
        )
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):  # type: ignore[override]
        item = super().__getitem__(index)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            img, label = item
            if not isinstance(label, int):
                label = abs(hash(str(label))) % 1000
        else:
            img, label = item, 0
        return self.transform(to_rgb(img)), int(label)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-pre-download", type=int, default=2)
    p.add_argument("--max-cache-size", default="200GB")
    p.add_argument("--input-dir", default=INPUT)
    p.add_argument("--cache-dir", default=CACHE)
    p.add_argument(
        "--async-prefetch",
        action="store_true",
        help="Force LITDATA_ASYNC_CHUNK_PREFETCH=1 (raises max_pre_download floor to 4).",
    )
    p.add_argument(
        "--no-async-prefetch",
        action="store_true",
        help="Force LITDATA_ASYNC_CHUNK_PREFETCH=0 (remote defaults to on).",
    )
    args = p.parse_args()

    import litdata

    if args.async_prefetch:
        os.environ["LITDATA_ASYNC_CHUNK_PREFETCH"] = "1"
        os.environ.pop("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", None)
    elif args.no_async_prefetch:
        os.environ["LITDATA_ASYNC_CHUNK_PREFETCH"] = "0"
        os.environ["LITDATA_ASYNC_MIN_PRE_DOWNLOAD"] = "0"

    print(
        f"label={args.label} litdata={litdata.__file__} "
        f"python={sys.version.split()[0]} gil_disabled={not sys._is_gil_enabled()}",
        flush=True,
    )
    print(
        f"input={args.input_dir} workers={args.workers} batch_size={args.batch_size} "
        f"epochs={args.epochs} max_pre_download={args.max_pre_download} "
        f"async_prefetch={os.environ.get('LITDATA_ASYNC_CHUNK_PREFETCH', 'default')}",
        flush=True,
    )

    def _clear_cache(path: str) -> None:
        """Wipe cache contents at start (and end). Never remove the mountpoint itself."""
        os.makedirs(path, exist_ok=True)
        # Same as historic Studio benches: `rm -rf /cache/chunks/*`
        os.system(f'rm -rf "{path}"/* "{path}"/.[!.]* "{path}"/..?* 2>/dev/null')

    print(f"clearing cache {args.cache_dir}", flush=True)
    _clear_cache(args.cache_dir)

    ds = ImageNetStreamingDataset(
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        shuffle=True,
        max_pre_download=args.max_pre_download,
        max_cache_size=args.max_cache_size,
    )
    print(f"dataset_len={len(ds)}", flush=True)

    loader = StreamingDataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    epoch_rows = []
    for epoch in range(args.epochs):
        num_samples = 0
        t_first_batch = None
        t0 = time.time()
        for data in tqdm(loader, smoothing=0, mininterval=1, desc=f"{args.label} epoch{epoch}"):
            if t_first_batch is None:
                t_first_batch = time.time() - t0
            num_samples += int(data[0].shape[0])
        elapsed = time.time() - t0
        ips = num_samples / elapsed if elapsed else float("nan")
        row = {
            "label": args.label,
            "epoch": epoch,
            "samples": num_samples,
            "elapsed_s": elapsed,
            "images_per_s": ips,
            "t_first_batch_s": t_first_batch,
        }
        epoch_rows.append(row)
        print(
            f"For {args.label} on epoch {epoch}, streamed over {num_samples} samples "
            f"in {elapsed:.3f}s or {ips:.1f} images/sec "
            f"(t_first_batch={t_first_batch:.3f}s).",
            flush=True,
        )

    total_s = sum(r["samples"] for r in epoch_rows)
    total_t = sum(r["elapsed_s"] for r in epoch_rows)
    summary = {
        "label": args.label,
        "litdata_file": litdata.__file__,
        "epochs": epoch_rows,
        "total_samples": total_s,
        "total_elapsed_s": total_t,
        "mean_images_per_s": total_s / total_t if total_t else float("nan"),
        "epoch0_images_per_s": epoch_rows[0]["images_per_s"] if epoch_rows else None,
        "epoch1_images_per_s": epoch_rows[1]["images_per_s"] if len(epoch_rows) > 1 else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)

    _clear_cache(args.cache_dir)


if __name__ == "__main__":
    main()

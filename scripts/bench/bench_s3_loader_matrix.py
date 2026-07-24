#!/usr/bin/env python3
"""Real-S3 ImageNet matrix: process loader × sync/async chunk download.

Mirrors the old ``stream/lightning_data.py`` loop:
  * ImageNet-optimized chunks from Studio S3 connections
  * ``RandomResizedCrop(224)`` + flip + float16 (torchvision v2)
  * ``batch_size=256``, ``num_workers=cpu_count()``
  * cold cache under ``/cache/chunks`` (or temp)
  * report images/sec

Modes:
  * process + sync
  * process + async chunk prefetch

Example::

    python scripts/bench/bench_s3_loader_matrix.py
    python scripts/bench/bench_s3_loader_matrix.py --epochs 1 --limit-batches 200
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402

from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.resolver import _resolve_dir  # noqa: E402

DEFAULT_INPUT = "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search"
DEFAULT_CACHE = "/cache/chunks"


def _gil_disabled() -> bool:
    fn = getattr(sys, "_is_gil_enabled", None)
    if fn is None:
        return False
    return not fn()


def to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Match historic ``stream/utils.py``."""
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    elif img.shape[0] == 4:
        img = img[:3]
    return img


class ImageNetStreamingDataset(StreamingDataset):
    """Historic ImageNet streaming dataset with training transforms."""

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
        # lightning_data_imagenet: (img, class_index)
        # lightning_data_search:   (img, filepath str)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            img, label = item
            if not isinstance(label, int):
                label = abs(hash(str(label))) % 1000
        else:
            img, label = item, 0
        return self.transform(to_rgb(img)), int(label)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime


def _prepare_cache(cache_dir: str, clean: bool) -> None:
    """Wipe cache contents; never remove mountpoints like ``/cache/chunks``."""
    os.makedirs(cache_dir, exist_ok=True)
    if clean:
        # Historic Studio benches: `rm -rf /cache/chunks/*`
        os.system(f'rm -rf "{cache_dir}"/* "{cache_dir}"/.[!.]* "{cache_dir}"/..?* 2>/dev/null')


def run_once(
    *,
    input_dir: str,
    cache_dir: str,
    async_prefetch: bool,
    workers: int,
    batch_size: int,
    epochs: int,
    limit_batches: int | None,
    max_pre_download: int,
    max_cache_size: str,
    clean_cache: bool,
) -> dict:
    _prepare_cache(cache_dir, clean=clean_cache)
    os.environ["LITDATA_ASYNC_CHUNK_PREFETCH"] = "1" if async_prefetch else "0"
    os.environ["LITDATA_ASYNC_MIN_PRE_DOWNLOAD"] = "0"

    rss_before = _rss_mb()
    cpu_before = _cpu_seconds()
    epoch_stats: list[dict] = []
    err: str | None = None

    try:
        ds = ImageNetStreamingDataset(
            input_dir=input_dir,
            cache_dir=cache_dir,
            shuffle=True,
            max_pre_download=max_pre_download,
            max_cache_size=max_cache_size,
        )

        loader = StreamingDataLoader(
            ds,
            batch_size=batch_size,
            num_workers=workers,
            prefetch_factor=2 if workers > 0 else None,
        )

        for epoch in range(epochs):
            t0 = time.perf_counter()
            num_samples = 0
            for i, data in enumerate(loader):
                num_samples += int(data[0].shape[0])
                if limit_batches is not None and (i + 1) >= limit_batches:
                    break
            elapsed = time.perf_counter() - t0
            epoch_stats.append(
                {
                    "epoch": epoch,
                    "samples": num_samples,
                    "elapsed_s": elapsed,
                    "images_per_s": num_samples / elapsed if elapsed else float("nan"),
                }
            )
            print(
                f"  epoch={epoch} samples={num_samples} "
                f"elapsed={elapsed:.2f}s images/s={epoch_stats[-1]['images_per_s']:.1f}"
            )

        if hasattr(loader, "_iterator") and loader._iterator is not None:
            with_context = getattr(loader._iterator, "_shutdown_workers", None)
            if callable(with_context):
                try:
                    with_context()
                except Exception:
                    pass
        del loader, ds
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    wall = sum(e["elapsed_s"] for e in epoch_stats)
    samples = sum(e["samples"] for e in epoch_stats)
    return {
        "loader": "process",
        "download": "async" if async_prefetch else "sync",
        "workers": workers,
        "batch_size": batch_size,
        "epochs": epochs,
        "limit_batches": limit_batches,
        "max_pre_download": max_pre_download,
        "samples": samples,
        "elapsed_s": wall,
        "images_per_s": samples / wall if wall and samples else float("nan"),
        "cpu_s": _cpu_seconds() - cpu_before,
        "cpu_per_wall": ((_cpu_seconds() - cpu_before) / wall) if wall else float("nan"),
        "rss_delta_mb": _rss_mb() - rss_before,
        "gil_disabled": _gil_disabled(),
        "epochs_detail": epoch_stats,
        "error": err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=100,
        help="Stop after N batches per epoch (None = full epoch). Default 100 for iteration speed.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-pre-download", type=int, default=4)
    parser.add_argument("--max-cache-size", default="200GB")
    parser.add_argument("--keep-cache", action="store_true", help="Do not wipe cache between modes")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["process-sync", "process-async"],
        choices=["process-sync", "process-async"],
    )
    args = parser.parse_args()

    # tempfile fallback when /cache is missing
    cache_dir = args.cache_dir
    if not os.path.isdir(os.path.dirname(cache_dir.rstrip("/")) or "/"):
        cache_dir = tempfile.mkdtemp(prefix="litdata-cache-")
    if cache_dir == DEFAULT_CACHE and not os.path.isdir("/cache"):
        cache_dir = tempfile.mkdtemp(prefix="litdata-cache-")
        print(f"/cache missing — using {cache_dir}")

    resolved = _resolve_dir(args.input_dir)
    print(f"python={sys.version.split()[0]} gil_disabled={_gil_disabled()}")
    print(f"input_dir={args.input_dir}")
    print(f"resolved.url={resolved.url}")
    print(
        f"workers={args.workers} batch_size={args.batch_size} "
        f"epochs={args.epochs} limit_batches={args.limit_batches} "
        f"max_pre_download={args.max_pre_download}"
    )
    print()

    rows: list[dict] = []
    for mode in args.modes:
        async_prefetch = mode.endswith("async")
        print(f"=== {mode} ===")
        row = run_once(
            input_dir=args.input_dir,
            cache_dir=cache_dir,
            async_prefetch=async_prefetch,
            workers=args.workers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            limit_batches=args.limit_batches,
            max_pre_download=args.max_pre_download,
            max_cache_size=args.max_cache_size,
            clean_cache=not args.keep_cache,
        )
        rows.append(row)
        if row.get("error"):
            print(f"FAIL {row['error']}\n")
        else:
            print(
                f"TOTAL images/s={row['images_per_s']:.1f} "
                f"cpu/wall={row['cpu_per_wall']:.2f} "
                f"rssΔ={row['rss_delta_mb']:.0f}MB\n"
            )

    print("=== summary (JSON lines) ===")
    for row in rows:
        slim = {k: v for k, v in row.items() if k != "epochs_detail"}
        print(json.dumps(slim, default=str))


if __name__ == "__main__":
    main()

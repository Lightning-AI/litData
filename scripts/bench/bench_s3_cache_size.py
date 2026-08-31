#!/usr/bin/env python3
"""Sweep ``max_cache_size`` on real S3 ImageNet streaming.

Answers: can we stream a large dataset on limited disk?

Reports **peak on-disk cache bytes** via a background sampler (``du``-style
walk every 200ms) — not a sparse in-loop estimate.

Cache is wiped at the start of each size (``rm -rf /cache/chunks/*``).

Example::

    PYTHON_GIL=0 python scripts/bench/bench_s3_cache_size.py
    PYTHON_GIL=0 python scripts/bench/bench_s3_cache_size.py --sizes 25GB --epochs 1

With ~58–64MB ImageNet chunks and 48 workers × ``max_pre_download=2–4``, the
steady-state working set is already ~5–12GB. Default sweep is 25GB only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402
from tqdm import tqdm  # noqa: E402

from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.utilities.format import _convert_bytes_to_int  # noqa: E402

DEFAULT_INPUT = "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search"
DEFAULT_CACHE = "/cache/chunks"


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


def _clear_cache(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.system(f'rm -rf "{path}"/* "{path}"/.[!.]* "{path}"/..?* 2>/dev/null')


def _cache_bytes(path: str) -> int:
    """Total bytes of regular files under ``path`` (true on-disk usage)."""
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _cache_bytes(entry.path)
            except OSError:
                continue
    return total


class PeakDiskMonitor:
    """Background sampler of max on-disk cache size."""

    def __init__(self, path: str, interval_s: float = 0.2) -> None:
        self.path = path
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="peak-disk", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=5)
        # Final sample
        try:
            self.peak_bytes = max(self.peak_bytes, _cache_bytes(self.path))
        except OSError:
            pass
        return self.peak_bytes

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, _cache_bytes(self.path))
            except OSError:
                pass
            self._stop.wait(self.interval_s)


def run_one(
    *,
    input_dir: str,
    cache_dir: str,
    max_cache_size: str,
    workers: int,
    batch_size: int,
    epochs: int,
    max_pre_download: int,
) -> dict:
    _clear_cache(cache_dir)
    limit_bytes = _convert_bytes_to_int(max_cache_size)
    err: str | None = None
    epoch_rows: list[dict] = []
    monitor = PeakDiskMonitor(cache_dir, interval_s=0.2)
    monitor.start()

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
            n = 0
            t0 = time.time()
            for data in tqdm(
                loader,
                smoothing=0,
                mininterval=1,
                desc=f"cache={max_cache_size} ep{epoch}",
            ):
                n += int(data[0].shape[0])
            elapsed = time.time() - t0
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "samples": n,
                    "elapsed_s": elapsed,
                    "images_per_s": n / elapsed if elapsed else float("nan"),
                }
            )
            print(
                f"  max_cache_size={max_cache_size} epoch={epoch} "
                f"samples={n} elapsed={elapsed:.2f}s "
                f"images/s={epoch_rows[-1]['images_per_s']:.1f}",
                flush=True,
            )
        if getattr(loader, "_iterator", None) is not None:
            shutdown = getattr(loader._iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        del loader, ds
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"  FAIL max_cache_size={max_cache_size}: {err}", flush=True)

    peak_bytes = monitor.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    limit_mb = limit_bytes / (1024 * 1024)
    # ~5% overshoot is acceptable; flag only "multiple times higher" failures.
    within_budget = peak_bytes <= limit_bytes * 1.05

    total_s = sum(r["samples"] for r in epoch_rows)
    total_t = sum(r["elapsed_s"] for r in epoch_rows)
    return {
        "max_cache_size": max_cache_size,
        "limit_mb": limit_mb,
        "ok": err is None and total_s > 0,
        "within_disk_budget": within_budget,
        "error": err,
        "epochs": epoch_rows,
        "total_samples": total_s,
        "total_elapsed_s": total_t,
        "mean_images_per_s": total_s / total_t if total_t else float("nan"),
        "peak_cache_mb": peak_mb,
        "peak_over_limit_x": (peak_bytes / limit_bytes) if limit_bytes else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument(
        "--sizes",
        nargs="+",
        default=["25GB"],
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-pre-download", type=int, default=2)
    args = p.parse_args()

    gil_disabled = not getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        f"python={sys.version.split()[0]} gil_disabled={gil_disabled} "
        f"workers={args.workers} batch_size={args.batch_size} epochs={args.epochs}",
        flush=True,
    )
    print(f"input={args.input_dir}", flush=True)
    print(f"sizes={args.sizes}", flush=True)
    print("metric: peak_cache_mb = max on-disk bytes under cache_dir (200ms sampler)", flush=True)

    rows = []
    for size in args.sizes:
        print(f"\n=== max_cache_size={size} ===", flush=True)
        row = run_one(
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            max_cache_size=size,
            workers=args.workers,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_pre_download=args.max_pre_download,
        )
        rows.append(row)
        print(
            f"  peak_cache_mb={row['peak_cache_mb']:.0f} "
            f"limit_mb={row['limit_mb']:.0f} "
            f"over_limit_x={row['peak_over_limit_x']:.2f} "
            f"within_budget={row['within_disk_budget']} "
            f"images/s={row['mean_images_per_s']:.1f}",
            flush=True,
        )

    print("\n=== summary ===", flush=True)
    print(f"{'size':>8} {'ok':>4} {'diskOK':>6} {'peak_MB':>10} {'limit_MB':>10} {'over_x':>7} {'img/s':>10}")
    for r in rows:
        print(
            f"{r['max_cache_size']:>8} {str(r['ok']):>4} {str(r['within_disk_budget']):>6} "
            f"{r['peak_cache_mb']:>10.0f} {r['limit_mb']:>10.0f} "
            f"{r['peak_over_limit_x']:>7.2f} {r['mean_images_per_s']:>10.1f}"
        )
    print("\n=== summary (JSON lines) ===", flush=True)
    for r in rows:
        slim = {k: v for k, v in r.items() if k != "epochs"}
        print(json.dumps(slim, default=str))


if __name__ == "__main__":
    main()

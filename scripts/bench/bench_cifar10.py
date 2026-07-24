#!/usr/bin/env python3
"""CIFAR-10-shaped decode bench: LitData process workers vs raw PyTorch.

Builds the **same** synthetic 32×32 RGB PIL samples for both paths:

  1. ``raw`` — in-memory ``torch.utils.data.Dataset`` + ``DataLoader`` (no LitData)
  2. ``litdata-process`` — ``StreamingDataLoader`` with process workers

Pass ``--real`` to use official torchvision CIFAR-10 for the LitData optimize
step and a matching in-memory raw baseline.

Example:
  .venv/bin/python scripts/bench/bench_cifar10.py --limit 2000 --workers 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("LITDATA_TIMING", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from litdata import optimize  # noqa: E402
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.timing import StreamingTimingStats  # noqa: E402


def _make_pil(index: int) -> Image.Image:
    rng = np.random.RandomState(index)
    arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _synthetic_sample(index: int) -> dict:
    return {"image": _make_pil(index), "class": int(index % 10)}


def _pil_to_tensor(image: Image.Image | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(image):
        return image
    return torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy())


def _to_batch_item(sample: dict) -> tuple[torch.Tensor, int]:
    return _pil_to_tensor(sample["image"]), int(sample["class"])


class RawCifarDataset(Dataset):
    """Map-style baseline: same PIL samples as LitData, held in memory."""

    def __init__(self, images: list[Image.Image], labels: list[int]) -> None:
        """Store PIL images and integer labels for DataLoader workers."""
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """Return number of in-memory samples."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Decode one PIL image to a CHW uint8 tensor + label."""
        return _pil_to_tensor(self.images[index]), int(self.labels[index])


def _build_raw_synthetic(limit: int) -> RawCifarDataset:
    images = [_make_pil(i) for i in range(limit)]
    labels = [i % 10 for i in range(limit)]
    return RawCifarDataset(images, labels)


def _build_raw_real(raw_root: str, limit: int) -> RawCifarDataset:
    from torchvision.datasets import CIFAR10

    base = CIFAR10(root=raw_root, train=True, download=True)
    n = min(limit, len(base))
    images: list[Image.Image] = []
    labels: list[int] = []
    for i in range(n):
        image, label = base[i]
        images.append(image)
        labels.append(int(label))
    return RawCifarDataset(images, labels)


def _optimize_synthetic(out_dir: str, limit: int, optimize_workers: int) -> None:
    if os.path.exists(os.path.join(out_dir, "index.json")):
        print(f"Reusing optimized dataset at {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"Optimizing synthetic CIFAR-shaped train[:{limit}] → {out_dir}")
    t0 = time.perf_counter()
    optimize(
        fn=_synthetic_sample,
        inputs=list(range(limit)),
        output_dir=out_dir,
        num_workers=optimize_workers,
        chunk_bytes="64MB",
        mode="overwrite",
        verbose=False,
    )
    print(f"Optimize done in {time.perf_counter() - t0:.1f}s")


_REAL_CIFAR = None


def _real_cifar_sample(index: int) -> dict:
    image, label = _REAL_CIFAR[index]
    return {"image": image, "class": int(label)}


def _optimize_real_cifar(raw_root: str, out_dir: str, limit: int, optimize_workers: int) -> None:
    global _REAL_CIFAR
    if os.path.exists(os.path.join(out_dir, "index.json")):
        print(f"Reusing optimized dataset at {out_dir}")
        return
    from torchvision.datasets import CIFAR10

    os.makedirs(out_dir, exist_ok=True)
    _REAL_CIFAR = CIFAR10(root=raw_root, train=True, download=True)
    n = min(limit, len(_REAL_CIFAR))
    print(f"Optimizing CIFAR-10 train[:{n}] → {out_dir}")
    t0 = time.perf_counter()
    optimize(
        fn=_real_cifar_sample,
        inputs=list(range(n)),
        output_dir=out_dir,
        num_workers=optimize_workers,
        chunk_bytes="64MB",
        mode="overwrite",
        verbose=False,
    )
    print(f"Optimize done in {time.perf_counter() - t0:.1f}s")


def _consume(loader) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    n_items = 0
    n_batches = 0
    for batch in loader:
        images, _labels = batch
        n_batches += 1
        n_items += images.shape[0] if torch.is_tensor(images) else len(images)
    return n_items, n_batches, time.perf_counter() - t0


def _run_raw(dataset: Dataset, *, num_workers: int, batch_size: int, shuffle: bool) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )
    # Warm
    _consume(loader)
    n_items, n_batches, elapsed = _consume(loader)
    return {
        "mode": "raw",
        "num_workers": num_workers,
        "items": n_items,
        "batches": n_batches,
        "elapsed_s": elapsed,
        "items_per_s": n_items / elapsed if elapsed else float("nan"),
    }


def _run_litdata(
    data_dir: str,
    *,
    num_workers: int,
    batch_size: int,
    shuffle: bool,
) -> dict:
    StreamingTimingStats.reset_instance()
    cache_dir = tempfile.mkdtemp(prefix="litdata-cifar-cache-")
    try:
        ds = StreamingDataset(
            input_dir=data_dir,
            cache_dir=cache_dir,
            shuffle=shuffle,
            max_pre_download=2,
            transform=_to_batch_item,
        )
        loader = StreamingDataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        _consume(loader)
        StreamingTimingStats.reset_instance()
        n_items, n_batches, elapsed = _consume(loader)
        return {
            "mode": "litdata-process",
            "num_workers": num_workers,
            "items": n_items,
            "batches": n_batches,
            "elapsed_s": elapsed,
            "items_per_s": n_items / elapsed if elapsed else float("nan"),
            "timing": StreamingTimingStats.instance().snapshot(),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _print_row(row: dict, note: str = "") -> None:
    suffix = f"  ({note})" if note else ""
    print(
        f"{row['mode']:<18} workers={row['num_workers']} "
        f"elapsed={row['elapsed_s']:.3f}s items/s={row['items_per_s']:.1f} "
        f"items={row['items']} batches={row['batches']}{suffix}"
    )


def main() -> None:
    """Compare raw PyTorch vs LitData process workers on CIFAR-shaped data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2_000, help="Samples to build / optimize")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimize-workers", type=int, default=2)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / ".benchmarks" / "cifar10"),
        help="Root for raw/optimized data",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use official torchvision CIFAR-10 instead of synthetic 32×32 images",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tag = "real" if args.real else "synthetic"
    opt_dir = str(data_root / f"optimized_{tag}_n{args.limit}")
    os.makedirs(data_root, exist_ok=True)

    print(f"Building raw in-memory baseline ({tag}, n={args.limit})...")
    t0 = time.perf_counter()
    if args.real:
        raw_ds = _build_raw_real(str(data_root / "raw"), args.limit)
        _optimize_real_cifar(str(data_root / "raw"), opt_dir, args.limit, args.optimize_workers)
    else:
        raw_ds = _build_raw_synthetic(args.limit)
        _optimize_synthetic(opt_dir, args.limit, args.optimize_workers)
    print(f"Raw baseline ready in {time.perf_counter() - t0:.1f}s ({len(raw_ds)} samples)")

    print(f"\n=== LitData vs raw ({tag} CIFAR-shaped PIL→tensor) ===")
    print(f"workers={args.workers} batch_size={args.batch_size} limit={args.limit}")

    results: list[dict] = []

    raw_row = _run_raw(
        raw_ds,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
    results.append(raw_row)
    _print_row(raw_row, note="torch Dataset+DataLoader, images in RAM")

    process_row = _run_litdata(
        opt_dir,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
    results.append(process_row)
    _print_row(process_row, note="StreamingDataLoader process workers")

    print("\n=== Relative to raw ===")
    raw_elapsed = results[0]["elapsed_s"]
    for row in results[1:]:
        if not row["elapsed_s"] or not raw_elapsed:
            print(f"{row['mode']:<18} n/a")
            continue
        slower = row["elapsed_s"] / raw_elapsed
        faster = raw_elapsed / row["elapsed_s"]
        print(f"{row['mode']:<18} {slower:.3f}x raw wall time ({faster:.2f}x faster than raw)")


if __name__ == "__main__":
    main()

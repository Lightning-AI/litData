#!/usr/bin/env python3
"""End-to-end StreamingDataset bench: write, local read, and remote→local copy.

The ``local:`` scheme forces PrepareChunksThread + LocalDownloader copies into a
fresh cache, which is the same control flow as ``s3://`` without needing a bucket.

  PYTHONPATH=src python scripts/bench/bench_streaming_e2e.py
  PYTHONPATH=src python scripts/bench/bench_streaming_e2e.py --profile
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import shutil
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("LITDATA_TIMING", "1")

from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.timing import StreamingTimingStats  # noqa: E402
from litdata.streaming.writer import BinaryWriter  # noqa: E402


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _write_mixed(out_dir: str, n: int, chunk_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    writer = BinaryWriter(out_dir, chunk_size=chunk_size)
    for i in range(n):
        writer[i] = {
            "id": i,
            "flag": i % 2 == 0,
            "x": float(i),
            "coords": [float(i), float(i + 1), float(i + 2)],
            "label": f"cls-{i % 10}",
        }
    writer.done()
    writer.merge()


def _write_jpegs(out_dir: str, n: int, chunk_size: int, size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    frames = []
    for _ in range(8):
        arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
        buf = BytesIO()
        PILImage.fromarray(arr).save(buf, format="JPEG", quality=90)
        frames.append(buf.getvalue())
    writer = BinaryWriter(out_dir, chunk_size=chunk_size)
    for i in range(n):
        writer[i] = {"image": frames[i % len(frames)], "id": i}
    writer.done()
    writer.merge()


def _iterate(input_dir: str, cache_dir: str | None, *, workers: int, batch_size: int, max_pre: int) -> dict:
    StreamingTimingStats.reset_instance()
    ds = StreamingDataset(
        input_dir=input_dir,
        cache_dir=cache_dir,
        shuffle=False,
        max_pre_download=max_pre,
        max_cache_size="20GB",
    )
    loader = StreamingDataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None,
    )
    t0 = time.perf_counter()
    n = 0
    for batch in loader:
        first = next(iter(batch.values())) if isinstance(batch, dict) else batch
        n += int(first.shape[0]) if hasattr(first, "shape") else len(first)
    elapsed = time.perf_counter() - t0
    snap = StreamingTimingStats.instance().snapshot()
    return {
        "items": n,
        "elapsed_s": elapsed,
        "items_per_s": n / elapsed if elapsed else float("nan"),
        "timing": snap,
    }


def _print_row(name: str, row: dict) -> None:
    dl = row["timing"].get("chunk_download_s", {})
    dec = row["timing"].get("item_decode_s", {})
    print(
        f"  {name:42s}  {row['elapsed_s'] * 1e3:8.1f} ms  "
        f"{row['items_per_s']:8.0f} items/s  items={row['items']}  "
        f"download_n={dl.get('count', 0)} download_mean={dl.get('mean_s', 0):.4f}s  "
        f"decode_n={dec.get('count', 0)} decode_mean={dec.get('mean_s', 0) * 1e3:.3f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed", type=int, default=20000)
    parser.add_argument("--jpegs", type=int, default=2000)
    parser.add_argument("--jpeg-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="litdata-e2e-")
    mixed_dir = os.path.join(tmp, "mixed")
    jpeg_dir = os.path.join(tmp, "jpegs")
    try:
        t0 = time.perf_counter()
        _write_mixed(mixed_dir, args.mixed, chunk_size=256)
        write_mixed = time.perf_counter() - t0
        t0 = time.perf_counter()
        _write_jpegs(jpeg_dir, args.jpegs, chunk_size=32, size=args.jpeg_size)
        write_jpeg = time.perf_counter() - t0
        print(f"write mixed {args.mixed}        {write_mixed * 1e3:8.1f} ms")
        print(f"write jpeg  {args.jpegs}x{args.jpeg_size}  {write_jpeg * 1e3:8.1f} ms")

        rows: list[tuple[str, dict]] = []

        def add(name: str, fn) -> None:
            times = []
            last = None
            for _ in range(args.repeats):
                last = fn()
                times.append(last["elapsed_s"])
            last = dict(last)
            last["elapsed_s"] = _median(times)
            last["items_per_s"] = last["items"] / last["elapsed_s"] if last["elapsed_s"] else float("nan")
            rows.append((name, last))
            _print_row(name, last)

        add("local mixed w0", lambda: _iterate(mixed_dir, None, workers=0, batch_size=64, max_pre=4))
        add("local mixed w2", lambda: _iterate(mixed_dir, None, workers=2, batch_size=64, max_pre=4))
        add("local jpeg w0", lambda: _iterate(jpeg_dir, None, workers=0, batch_size=32, max_pre=4))

        def remote_mixed() -> dict:
            cache = tempfile.mkdtemp(prefix="litdata-e2e-cache-")
            try:
                return _iterate(f"local:{mixed_dir}", cache, workers=0, batch_size=64, max_pre=4)
            finally:
                shutil.rmtree(cache, ignore_errors=True)

        def remote_mixed_w2() -> dict:
            cache = tempfile.mkdtemp(prefix="litdata-e2e-cache-")
            try:
                return _iterate(f"local:{mixed_dir}", cache, workers=2, batch_size=64, max_pre=8)
            finally:
                shutil.rmtree(cache, ignore_errors=True)

        add("remote→local mixed w0", remote_mixed)
        add("remote→local mixed w2", remote_mixed_w2)

        if args.profile:
            cache = tempfile.mkdtemp(prefix="litdata-e2e-prof-")
            prof = cProfile.Profile()
            prof.enable()
            _iterate(f"local:{mixed_dir}", cache, workers=0, batch_size=64, max_pre=4)
            prof.disable()
            shutil.rmtree(cache, ignore_errors=True)
            stats = pstats.Stats(prof).sort_stats("tottime")
            print("\n=== cProfile remote→local mixed w0 (top 25) ===")
            stats.print_stats(25)

        print("\n=== summary ===")
        for name, row in rows:
            print(f"{name:42s}  {row['elapsed_s'] * 1e3:8.1f} ms  {row['items_per_s']:8.0f} items/s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matrix over ``max_pre_download`` × ``prefetch_factor`` for local StreamingDataLoader.

Enable timing with ``LITDATA_TIMING=1``. Prints wall time and timing snapshot means.

Example:
  LITDATA_TIMING=1 .venv/bin/python scripts/bench_prefetch_matrix.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("LITDATA_TIMING", "1")

from litdata.streaming import Cache  # noqa: E402
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.timing import StreamingTimingStats  # noqa: E402


def _build(tmpdir: str, n_items: int = 2000, chunk_size: int = 50) -> str:
    cache_dir = os.path.join(tmpdir, "data")
    os.makedirs(cache_dir)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_items):
        cache[i] = {"x": i, "y": float(i)}
    cache.done()
    cache.merge(1)
    return cache_dir


def run_once(data_dir: str, max_pre_download: int, prefetch_factor: int, num_workers: int = 2) -> dict:
    """Run one cold-cache StreamingDataLoader pass and return timing stats."""
    StreamingTimingStats.reset_instance()
    cache_dir = tempfile.mkdtemp(prefix="litdata-prefetch-cache-")
    try:
        ds = StreamingDataset(
            input_dir=f"local:{data_dir}",
            cache_dir=cache_dir,
            shuffle=False,
            max_pre_download=max_pre_download,
            max_cache_size="10GB",
        )
        loader = StreamingDataLoader(
            ds,
            batch_size=16,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        t0 = time.perf_counter()
        n = 0
        for batch in loader:
            n += len(batch["x"]) if isinstance(batch, dict) else len(batch)
        elapsed = time.perf_counter() - t0
        snap = StreamingTimingStats.instance().snapshot()
        return {
            "max_pre_download": max_pre_download,
            "prefetch_factor": prefetch_factor,
            "items": n,
            "elapsed_s": elapsed,
            "items_per_s": n / elapsed if elapsed else float("nan"),
            "timing": snap,
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    """Build a tiny local dataset and print the prefetch matrix."""
    tmp = tempfile.mkdtemp(prefix="litdata-prefetch-data-")
    try:
        data_dir = _build(tmp)
        rows = []
        for max_pre in (1, 2, 4, 8):
            for prefetch in (1, 2, 4):
                row = run_once(data_dir, max_pre, prefetch)
                rows.append(row)
                print(
                    f"max_pre_download={max_pre:<2} prefetch_factor={prefetch:<2} "
                    f"elapsed={row['elapsed_s']:.3f}s items/s={row['items_per_s']:.1f}"
                )
                decode = row["timing"].get("item_decode_s", {})
                download = row["timing"].get("chunk_download_s", {})
                if decode or download:
                    print(
                        f"  decode_mean={decode.get('mean_s', float('nan')):.6f}s "
                        f"download_mean={download.get('mean_s', float('nan')):.6f}s"
                    )
        best = min(rows, key=lambda r: r["elapsed_s"])
        print(
            "\nFastest:",
            f"max_pre_download={best['max_pre_download']} prefetch_factor={best['prefetch_factor']} "
            f"({best['elapsed_s']:.3f}s)",
        )
        print(
            "Note: defaults remain max_pre_download=2 / prefetch_factor=2 until a broader "
            "remote-latency matrix justifies a change."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

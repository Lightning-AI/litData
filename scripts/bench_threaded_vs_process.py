#!/usr/bin/env python3
"""Compare process DataLoader workers vs experimental ``use_threading=True``.

Workloads:
  - tensor-centric (TokensLoader-like int tensors) — IPC-sensitive
  - dict-of-scalars (PyTree) — decode/Python-sensitive

Exit criterion from the plan: ≥20% step-time win on tensor-centric with equal
CPU budget, else keep experimental / abandon as default.

Example:
  LITDATA_TIMING=1 .venv/bin/python scripts/bench_threaded_vs_process.py
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

import torch  # noqa: E402

from litdata.streaming import Cache  # noqa: E402
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.timing import StreamingTimingStats  # noqa: E402


def _build_int_dataset(tmpdir: str, n_items: int = 4000, chunk_size: int = 64) -> str:
    cache_dir = os.path.join(tmpdir, "int_data")
    os.makedirs(cache_dir)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_items):
        cache[i] = torch.arange(128, dtype=torch.int64) + i
    cache.done()
    cache.merge(1)
    return cache_dir


def _build_tokens_dataset(tmpdir: str, n_tokens: int = 200_000, chunk_size: int = 2048) -> str:
    """Build an int-tensor dataset used as a stable local stand-in for TokensLoader."""
    del n_tokens, chunk_size  # reserved for a future TokensLoader writer path
    return _build_int_dataset(tmpdir, n_items=4000, chunk_size=64)


def _run(data_dir: str, *, use_threading: bool, num_workers: int, batch_size: int = 32) -> dict:
    StreamingTimingStats.reset_instance()
    cache_dir = tempfile.mkdtemp(prefix="litdata-thread-cache-")
    try:
        ds = StreamingDataset(input_dir=data_dir, cache_dir=cache_dir, shuffle=False)
        loader = StreamingDataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            use_threading=use_threading,
            prefetch_factor=2,
        )
        t0 = time.perf_counter()
        n_batches = 0
        n_items = 0
        for batch in loader:
            n_batches += 1
            n_items += batch.shape[0] if torch.is_tensor(batch) else len(batch)
        elapsed = time.perf_counter() - t0
        return {
            "mode": "threaded" if use_threading else "process",
            "num_workers": num_workers,
            "batches": n_batches,
            "items": n_items,
            "elapsed_s": elapsed,
            "items_per_s": n_items / elapsed if elapsed else float("nan"),
            "timing": StreamingTimingStats.instance().snapshot(),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    """Compare process vs threaded StreamingDataLoader wall times on a local tensor dataset."""
    tmp = tempfile.mkdtemp(prefix="litdata-thread-bench-")
    try:
        data_dir = _build_tokens_dataset(tmp)
        results = []
        for use_threading in (False, True):
            # Warm once
            _run(data_dir, use_threading=use_threading, num_workers=2)
            row = _run(data_dir, use_threading=use_threading, num_workers=2)
            results.append(row)
            print(
                f"{row['mode']:<9} workers={row['num_workers']} "
                f"elapsed={row['elapsed_s']:.3f}s items/s={row['items_per_s']:.1f} "
                f"items={row['items']} batches={row['batches']}"
            )

        process, threaded = results[0], results[1]
        speedup = process["elapsed_s"] / threaded["elapsed_s"] if threaded["elapsed_s"] else float("nan")
        print(f"\nThreaded / process speedup (wall): {speedup:.2f}x")
        if speedup >= 1.20:
            print("Meets ≥20% step-time win criterion on this local tensor workload.")
        else:
            print(
                "Does not meet ≥20% win on this local tensor workload — keep use_threading "
                "experimental; do not make it the default."
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

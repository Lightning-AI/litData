#!/usr/bin/env python3
"""Microbench for PrepareChunksThread ↔ ItemLoader handoff latency.

Compares the current working tree against a baseline (default: origin/main) using
local ``local:`` streaming so results are PR-ready without cloud credentials.

Metrics:
1. Prefetch side-poll timeouts when download work can proceed (should be 0.0 after).
2. ItemLoader wake latency after a delayed chunk publish (ready Event vs 0.1s poll).
3. End-to-end cold-cache iteration over many tiny remote chunks (handoff-dominated).
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo src/ is importable when run from a worktree or checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    ordered = sorted(samples)
    n = len(ordered)
    p50 = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    idx95 = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
    return p50, ordered[idx95]


def bench_prefetch_side_timeouts() -> dict[str, float]:
    """Return force/delete poll timeouts observed while the prefetch buffer can accept work."""
    from litdata.streaming.config import ChunksConfig
    from litdata.streaming.reader import _DEFAULT_TIMEOUT, PrepareChunksThread
    from litdata.utilities.env import _DistributedEnv

    cache_dir = tempfile.mkdtemp(prefix="litdata-bench-prefetch-")
    try:
        config = MagicMock(spec=ChunksConfig)
        config.num_bytes = 1024
        config._cache_dir = cache_dir
        config.download_chunk_from_index = MagicMock()
        item_loader = MagicMock()
        env = _DistributedEnv(1, 0, 1)
        thread = PrepareChunksThread(config, item_loader, env, max_cache_size=10_000, max_pre_download=2)
        observed: list[float] = []

        def fake_force(timeout: float = _DEFAULT_TIMEOUT) -> None:
            observed.append(timeout)

        def fake_delete(timeout: float = _DEFAULT_TIMEOUT) -> None:
            observed.append(timeout)
            thread._force_stop_event.set()

        thread._force_download = fake_force  # type: ignore[method-assign]
        thread._maybe_delete_chunks = fake_delete  # type: ignore[method-assign]
        thread.run()
        return {
            "force_timeout_s": observed[0] if observed else float("nan"),
            "delete_timeout_s": observed[1] if len(observed) > 1 else float("nan"),
        }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def bench_item_loader_wake(n_trials: int = 20, delay_s: float = 0.05) -> dict[str, float]:
    """Measure how long load_item_from_chunk waits for a chunk published after ``delay_s``."""
    from litdata.streaming import Cache
    from litdata.streaming.dataset import StreamingDataset
    from litdata.streaming.item_loader import PyTreeLoader
    from litdata.streaming.sampler import ChunkedIndex

    root = tempfile.mkdtemp(prefix="litdata-bench-wake-")
    try:
        cache = Cache(root, chunk_size=2)
        for i in range(4):
            cache[i] = i
        cache.done()
        cache.merge()

        dataset = StreamingDataset(root)
        _ = dataset[0]
        loader = dataset.cache._reader._item_loader
        assert isinstance(loader, PyTreeLoader)
        chunk_filepath, begin, filesize = dataset.cache._reader.config[ChunkedIndex(2, chunk_index=1)]

        samples: list[float] = []
        for _ in range(n_trials):
            # Reset open-chunk state so the wait path runs every trial.
            loader._close_open_chunk()
            loader._chunk_filepath = None

            hidden = chunk_filepath + ".hidden"
            os.rename(chunk_filepath, hidden)
            ready = threading.Event()
            # Prefer the ready-provider path when available (this branch); otherwise plain poll.
            if hasattr(loader, "set_chunk_ready_provider"):
                loader.set_chunk_ready_provider(lambda _idx: ready)

            def restore_and_signal() -> None:
                time.sleep(delay_s)
                os.rename(hidden, chunk_filepath)
                ready.set()

            threading.Thread(target=restore_and_signal, daemon=True).start()
            t0 = time.perf_counter()
            item = loader.load_item_from_chunk(2, 1, chunk_filepath, begin, filesize)
            elapsed = time.perf_counter() - t0
            assert item == 2
            samples.append(elapsed)

        p50, p95 = _p50_p95(samples)
        return {
            "trials": float(n_trials),
            "delay_s": delay_s,
            "wake_p50_s": p50,
            "wake_p95_s": p95,
            "wake_mean_s": statistics.fmean(samples),
            "has_ready_provider": float(hasattr(loader, "set_chunk_ready_provider")),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def bench_e2e_local_remote(num_items: int = 400, chunk_size: int = 4, repeats: int = 3) -> dict[str, float]:
    """Cold-cache StreamingDataset iteration over a local: remote with many small chunks."""
    from litdata.streaming import Cache
    from litdata.streaming.dataset import StreamingDataset

    samples: list[float] = []
    for _ in range(repeats):
        root = tempfile.mkdtemp(prefix="litdata-bench-e2e-")
        try:
            data_dir = os.path.join(root, "data")
            cache_dir = os.path.join(root, "cache")
            os.makedirs(data_dir)
            os.makedirs(cache_dir)
            cache = Cache(data_dir, chunk_size=chunk_size)
            for i in range(num_items):
                cache[i] = i
            cache.done()
            cache.merge()

            dataset = StreamingDataset(f"local:{data_dir}", cache_dir=cache_dir, shuffle=False, drop_last=False)
            t0 = time.perf_counter()
            total = 0
            for item in dataset:
                total += int(item)
            elapsed = time.perf_counter() - t0
            assert total == sum(range(num_items))
            samples.append(elapsed)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    p50, p95 = _p50_p95(samples)
    return {
        "num_items": float(num_items),
        "chunk_size": float(chunk_size),
        "num_chunks": float((num_items + chunk_size - 1) // chunk_size),
        "repeats": float(repeats),
        "e2e_p50_s": p50,
        "e2e_p95_s": p95,
        "e2e_mean_s": statistics.fmean(samples),
        "items_per_s": num_items / statistics.fmean(samples),
    }


def main() -> int:
    """Run the microbench suite and print labeled results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="Label printed with the results")
    parser.add_argument("--wake-trials", type=int, default=20)
    parser.add_argument("--wake-delay", type=float, default=0.05)
    parser.add_argument("--e2e-items", type=int, default=400)
    parser.add_argument("--e2e-chunk-size", type=int, default=4)
    parser.add_argument("--e2e-repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"=== bench_download_reader_overlap ({args.label}) ===")
    print(f"repo: {REPO_ROOT}")
    print(f"python: {sys.executable}")

    side = bench_prefetch_side_timeouts()
    print(f"prefetch_side_timeouts: force={side['force_timeout_s']:.3f}s delete={side['delete_timeout_s']:.3f}s")

    wake = bench_item_loader_wake(n_trials=args.wake_trials, delay_s=args.wake_delay)
    print(
        "item_loader_wake: "
        f"p50={wake['wake_p50_s'] * 1000:.1f}ms p95={wake['wake_p95_s'] * 1000:.1f}ms "
        f"mean={wake['wake_mean_s'] * 1000:.1f}ms "
        f"(delay={wake['delay_s'] * 1000:.0f}ms, ready_provider={bool(wake['has_ready_provider'])})"
    )

    e2e = bench_e2e_local_remote(num_items=args.e2e_items, chunk_size=args.e2e_chunk_size, repeats=args.e2e_repeats)
    print(
        "e2e_local_remote: "
        f"p50={e2e['e2e_p50_s']:.3f}s p95={e2e['e2e_p95_s']:.3f}s mean={e2e['e2e_mean_s']:.3f}s "
        f"({e2e['items_per_s']:.0f} items/s, {int(e2e['num_chunks'])} chunks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

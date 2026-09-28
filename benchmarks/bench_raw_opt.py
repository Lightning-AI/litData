"""A/B microbench for StreamingRawDataset optimizations on ImageNet val."""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch.utils.data import DataLoader
from tqdm import tqdm
from uvloop_status import log_loop_runner_backend, uvloop_package_status


def clear_dir(path: str) -> None:
    """Remove ``path`` if it is an existing directory."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def run_once(
    *,
    label: str,
    input_dir: str,
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    max_prefetch: int,
    clear_cache: bool,
) -> dict:
    """Run one StreamingRawDataset throughput trial and return timing stats."""
    from litdata import StreamingRawDataset

    if clear_cache:
        clear_dir(cache_dir)

    kwargs: dict = {
        "input_dir": input_dir,
        "cache_dir": cache_dir,
        "cache_files": False,
        "recompute_index": False,
        "transform": None,
    }
    sig = inspect.signature(StreamingRawDataset.__init__)
    if "max_prefetch" in sig.parameters:
        kwargs["max_prefetch"] = max_prefetch
    if "max_concurrent_downloads" in sig.parameters:
        kwargs["max_concurrent_downloads"] = 64

    t_index = time.perf_counter()
    ds = StreamingRawDataset(**kwargs)
    index_s = time.perf_counter() - t_index
    n = len(ds)
    log_loop_runner_backend(print, prefix=f"[{label}]")

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    it = iter(loader)

    warm_t0 = time.perf_counter()
    warm = next(it)
    warm_s = time.perf_counter() - warm_t0

    samples = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(tqdm(it, total=num_batches, desc=label, leave=False)):
        samples += len(batch)
        if i + 1 >= num_batches:
            break
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed > 0 else 0.0

    storage = getattr(ds, "_storage_path", None) or getattr(ds.cache_manager, "_input_dir_path", "?")
    print(
        f"[{label}] index={index_s:.2f}s ({n} files) storage={storage!r}\n"
        f"         warm1={warm_s:.2f}s ({len(warm)} samples) | "
        f"{num_batches} batches / {samples} samples in {elapsed:.2f}s "
        f"→ {ips:.1f} samples/s (prefetch={max_prefetch}, workers={num_workers}, bs={batch_size})"
    )
    return {
        "label": label,
        "storage_path": storage,
        "indexed": n,
        "index_s": index_s,
        "warm_batch_s": warm_s,
        "batches": num_batches,
        "samples": samples,
        "elapsed_s": elapsed,
        "samples_per_s": ips,
        "max_prefetch": max_prefetch,
    }


def main() -> None:
    """CLI entrypoint for prefetch A/B microbenchmarks."""
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="/teamspace/s3_connections/imagenet-1m-template/raw/val")
    p.add_argument("--cache_root", default=str(Path(tempfile.gettempdir()) / "litdata-raw-bench"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    print(f"uvloop package: {uvloop_package_status()}")
    results = []

    common = {
        "input_dir": args.input_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_batches": args.num_batches,
        "clear_cache": True,
    }

    results.append(
        run_once(
            label="new-studio-path-prefetch0",
            cache_dir=str(Path(args.cache_root) / "new0"),
            max_prefetch=0,
            **common,
        )
    )
    results.append(
        run_once(
            label="new-studio-path-prefetch128",
            cache_dir=str(Path(args.cache_root) / "new128"),
            max_prefetch=max(128, 2 * args.batch_size),
            **common,
        )
    )

    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['label']:32s}  {r['samples_per_s']:8.1f} samples/s  "
            f"index={r['index_s']:.2f}s  warm={r['warm_batch_s']:.2f}s  storage={r['storage_path']}"
        )
    if len(results) >= 2 and results[0]["samples_per_s"] > 0:
        print(f"\nPrefetch speedup: {results[1]['samples_per_s'] / results[0]['samples_per_s']:.2f}x")


if __name__ == "__main__":
    main()

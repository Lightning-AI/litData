"""Minimal spawn DataLoader smoke for StreamingRawDataset (must be a .py file)."""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

from torch.utils.data import DataLoader

from litdata import StreamingRawDataset

INPUT = "/teamspace/s3_connections/imagenet-1m-template/raw/val"
CACHE = Path(tempfile.gettempdir()) / "litdata-spawn-smoke-cache"
SEED = Path(tempfile.gettempdir()) / "litdata-raw-ranged-vs-whole" / "seed"


def main() -> int:
    """Run a short spawn DataLoader smoke against ImageNet val."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if (SEED / "index.json.zstd").exists():
        shutil.copy2(SEED / "index.json.zstd", CACHE / "index.json.zstd")
        print("copied index from seed", flush=True)

    t0 = time.perf_counter()
    ds = StreamingRawDataset(INPUT, cache_dir=str(CACHE), cache_files=False, max_prefetch=0, hedge_delay=0)
    print(f"indexed n={len(ds)} in {time.perf_counter() - t0:.2f}s", flush=True)

    # Warm runtime clients then ensure pickle still works.
    _ = ds[0]
    blob = pickle.dumps(ds, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"pickle size={len(blob):,} bytes after warm", flush=True)
    assert pickle.loads(blob).cache_manager._downloader is None  # noqa: S301

    print("starting spawn DataLoader num_workers=2 ...", flush=True)
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        shuffle=False,
        multiprocessing_context="spawn",
        persistent_workers=False,
    )
    it = iter(loader)
    for i in range(2):
        batch = next(it)
        print(f"batch {i}: n={len(batch)} nbytes0={len(batch[0])}", flush=True)
    del it, loader
    print("SPAWN_SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

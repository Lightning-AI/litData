#!/usr/bin/env python3
"""Real-S3 / Studio-connection benchmarks for next-streaming optimizations.

Measures what local CIFAR microbenches cannot:
  1. Serial vs asyncio.gather chunk downloads against a live bucket.
  2. End-to-end StreamingDataLoader cold-cache throughput with/without
     ``LITDATA_ASYNC_CHUNK_PREFETCH=1`` and a ``max_pre_download`` sweep.

Defaults to the Studio-mounted ImageNet template validation set::

    s3://imagenet-1m-template/optimized/val

Examples::

    python scripts/bench/bench_s3_remote.py
    python scripts/bench/bench_s3_remote.py --input-dir /teamspace/s3_connections/imagenet-1m-template/optimized/val
    python scripts/bench/bench_s3_remote.py --mode download --chunks 6
    python scripts/bench/bench_s3_remote.py --mode e2e --batches 40 --workers 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from litdata.streaming.async_prefetch import (  # noqa: E402
    adownload_chunk_indexes,
    downloader_supports_adownload,
)
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.downloader import get_downloader  # noqa: E402
from litdata.streaming.resolver import _resolve_dir  # noqa: E402
from litdata.streaming.timing import StreamingTimingStats  # noqa: E402

DEFAULT_INPUT = "s3://imagenet-1m-template/optimized/val"


def _load_index(input_dir: str) -> tuple[str, list[dict]]:
    """Return (remote_url, chunks) for an S3 or Studio connection path."""
    resolved = _resolve_dir(input_dir)
    remote = resolved.url or resolved.path
    assert remote is not None

    # Prefer local FUSE/index when available; else download index via downloader.
    local_index = None
    if resolved.path and os.path.exists(os.path.join(resolved.path, "index.json")):
        local_index = os.path.join(resolved.path, "index.json")
    if local_index is None and input_dir.startswith("s3://"):
        # Try Studio mount mirror for known buckets.
        # s3://bucket/key -> /teamspace/s3_connections/bucket/key/index.json
        parts = input_dir[len("s3://") :].split("/", 1)
        if len(parts) == 2:
            candidate = f"/teamspace/s3_connections/{parts[0]}/{parts[1]}/index.json"
            if os.path.exists(candidate):
                local_index = candidate

    if local_index is None:
        raise FileNotFoundError(
            f"Could not locate index.json for {input_dir}. "
            "Pass a /teamspace/s3_connections/... path or ensure the mount is present."
        )

    with open(local_index, encoding="utf-8") as f:
        payload = json.load(f)
    return remote.rstrip("/"), payload["chunks"]


def _fresh_cache(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def bench_download(input_dir: str, n_chunks: int, drop_page_cache: bool = False) -> dict:
    """Compare serial sync downloads vs asyncio.gather on real remote chunks."""
    remote, chunks = _load_index(input_dir)
    resolved = _resolve_dir(input_dir)
    storage_options = {}
    if getattr(resolved, "data_connection_id", None):
        storage_options["data_connection_id"] = resolved.data_connection_id
    n_chunks = min(n_chunks, len(chunks))
    indexes = list(range(n_chunks))
    selected = chunks[:n_chunks]
    total_mb = sum(c["chunk_bytes"] for c in selected) / 1e6

    print(f"\n=== chunk download: {remote} ===")
    print(f"chunks={n_chunks} total≈{total_mb:.1f}MB mean≈{total_mb / n_chunks:.1f}MB")

    serial_cache = _fresh_cache("litdata-s3-serial-")
    async_cache = _fresh_cache("litdata-s3-async-")
    try:
        serial_dl = get_downloader(remote + "/", serial_cache, selected, storage_options, {})
        async_dl = get_downloader(remote + "/", async_cache, selected, storage_options, {})
        print(f"downloader={type(serial_dl).__name__} native_adownload={downloader_supports_adownload(serial_dl)}")

        t0 = time.perf_counter()
        for idx in indexes:
            serial_dl.download_chunk_from_index(idx)
        serial_s = time.perf_counter() - t0

        # Lightweight ChunksConfig stand-in for adownload_chunk_indexes.
        class _Cfg:
            def __init__(self, cache_dir: str, chunks: list, downloader) -> None:
                self._cache_dir = cache_dir
                self._chunks = chunks
                self._downloader = downloader
                self._shared_chunk_indexes: set[int] = set()
                self._compressor_name = None

            def try_decompress(self, local_chunkpath: str) -> None:
                return

            def download_chunk_from_index(self, chunk_index: int) -> None:
                self._downloader.download_chunk_from_index(chunk_index)

            def __getitem__(self, index):  # pragma: no cover - unused here
                raise NotImplementedError

        cfg = _Cfg(async_cache, selected, async_dl)
        t0 = time.perf_counter()
        asyncio.run(adownload_chunk_indexes(cfg, indexes))  # type: ignore[arg-type]
        async_s = time.perf_counter() - t0

        result = {
            "mode": "download",
            "remote": remote,
            "chunks": n_chunks,
            "total_mb": total_mb,
            "serial_s": serial_s,
            "async_s": async_s,
            "speedup": (serial_s / async_s) if async_s else float("nan"),
            "serial_mb_s": total_mb / serial_s if serial_s else float("nan"),
            "async_mb_s": total_mb / async_s if async_s else float("nan"),
            "native_adownload": downloader_supports_adownload(serial_dl),
        }
        print(
            f"serial_sync     {serial_s:.3f}s  ({result['serial_mb_s']:.1f} MB/s)\n"
            f"asyncio_gather  {async_s:.3f}s  ({result['async_mb_s']:.1f} MB/s)\n"
            f"speedup         {result['speedup']:.2f}x"
        )
        return result
    finally:
        shutil.rmtree(serial_cache, ignore_errors=True)
        shutil.rmtree(async_cache, ignore_errors=True)
        del drop_page_cache  # reserved


def _consume(loader: StreamingDataLoader, max_batches: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    n = 0
    for i, batch in enumerate(loader):
        # Custom collate returns (count, nbytes); fallback for other loaders.
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], int):
            n += batch[0]
        elif isinstance(batch, (list, tuple)):
            first = batch[0]
            n += int(first.shape[0]) if hasattr(first, "shape") else len(first)
        elif isinstance(batch, dict):
            first = next(iter(batch.values()))
            n += int(first.shape[0]) if hasattr(first, "shape") else len(first)
        else:
            n += int(batch.shape[0]) if hasattr(batch, "shape") else 1
        if i + 1 >= max_batches:
            break
    return n, time.perf_counter() - t0


def _collate_nbytes(batch: list):
    """Collate without stacking: ImageNet JPEGs vary in shape / use mmap storage.

    Returns ``(count, approx_bytes)`` so the loader still exercises fetch/decode
    while avoiding ``default_collate`` resize failures on mmap tensors.
    """
    import torch

    nbytes = 0
    for item in batch:
        tensors = []
        if isinstance(item, torch.Tensor):
            tensors = [item]
        elif isinstance(item, (list, tuple)):
            tensors = [x for x in item if isinstance(x, torch.Tensor)]
        elif isinstance(item, dict):
            tensors = [x for x in item.values() if isinstance(x, torch.Tensor)]
        for t in tensors:
            nbytes += int(t.numel() * t.element_size())
    return len(batch), nbytes


def bench_e2e(
    input_dir: str,
    *,
    batches: int,
    workers: int,
    batch_size: int,
    max_pre_downloads: list[int],
    async_modes: list[bool],
) -> list[dict]:
    """Cold-cache end-to-end StreamingDataLoader matrix on real S3."""
    resolved = _resolve_dir(input_dir)
    remote = resolved.url or input_dir
    print(f"\n=== e2e StreamingDataLoader: {remote} ===")
    print(f"batches={batches} batch_size={batch_size} workers={workers}")

    rows: list[dict] = []
    for async_on in async_modes:
        for max_pre in max_pre_downloads:
            cache_dir = _fresh_cache("litdata-s3-e2e-")
            os.environ["LITDATA_ASYNC_CHUNK_PREFETCH"] = "1" if async_on else "0"
            os.environ["LITDATA_TIMING"] = "1"
            StreamingTimingStats.reset_instance()
            try:
                ds = StreamingDataset(
                    input_dir=input_dir,
                    cache_dir=cache_dir,
                    shuffle=False,
                    max_pre_download=max_pre,
                    max_cache_size="50GB",
                )
                loader = StreamingDataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=workers,
                    prefetch_factor=2 if workers > 0 else None,
                    collate_fn=_collate_nbytes,
                )
                # Drop first construction costs already paid; measure cold stream.
                n, elapsed = _consume(loader, batches)
                snap = StreamingTimingStats.instance().snapshot()
                row = {
                    "mode": "e2e",
                    "async_prefetch": async_on,
                    "max_pre_download": max_pre,
                    "workers": workers,
                    "batch_size": batch_size,
                    "batches": batches,
                    "items": n,
                    "elapsed_s": elapsed,
                    "items_per_s": n / elapsed if elapsed else float("nan"),
                    "timing": snap,
                }
                rows.append(row)
                dl = snap.get("chunk_download_s", {})
                print(
                    f"async={int(async_on)} max_pre={max_pre:<2} "
                    f"elapsed={elapsed:.2f}s items/s={row['items_per_s']:.1f} items={n} "
                    f"download_mean={dl.get('mean_s', float('nan')):.3f}s "
                    f"download_count={dl.get('count', 0)}"
                )
            finally:
                # Ensure workers release files before rmtree.
                del loader, ds
                shutil.rmtree(cache_dir, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--mode", choices=("download", "e2e", "all"), default="all")
    parser.add_argument("--chunks", type=int, default=6, help="Chunks for download microbench")
    parser.add_argument("--batches", type=int, default=30, help="Batches for e2e cold stream")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-pre-download",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="max_pre_download values for e2e matrix",
    )
    args = parser.parse_args()

    print(f"input_dir={args.input_dir}")
    resolved = _resolve_dir(args.input_dir)
    print(f"resolved.path={resolved.path}")
    print(f"resolved.url={resolved.url}")

    results: list[dict] = []
    if args.mode in ("download", "all"):
        results.append(bench_download(args.input_dir, args.chunks))
    if args.mode in ("e2e", "all"):
        results.extend(
            bench_e2e(
                args.input_dir,
                batches=args.batches,
                workers=args.workers,
                batch_size=args.batch_size,
                max_pre_downloads=list(args.max_pre_download),
                async_modes=[False, True],
            )
        )

    # Compact summary for PR updates.
    print("\n=== summary (JSON lines) ===")
    for row in results:
        slim = {k: v for k, v in row.items() if k != "timing"}
        print(json.dumps(slim, default=str))


if __name__ == "__main__":
    main()

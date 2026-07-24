#!/usr/bin/env python3
"""Microbench: serial vs asyncio.gather chunk downloads (remote-IO spike).

This measures **chunk prefetch IO overlap**, not an async StreamingDataLoader.
Use a downloader with simulated latency (default) or point at real remote data.

Example:
  .venv/bin/python scripts/bench/bench_async_chunk_prefetch.py
  LITDATA_ASYNC_CHUNK_PREFETCH=1 .venv/bin/python scripts/bench/bench_async_chunk_prefetch.py --chunks 8 --delay-ms 100
"""

from __future__ import annotations

import argparse
import asyncio
import os
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
from litdata.streaming.downloader import Downloader  # noqa: E402


class _LatencyDownloader(Downloader):
    """Fake remote downloader with fixed per-object latency."""

    def __init__(self, remote_dir: str, cache_dir: str, chunks: list, delay_s: float) -> None:
        super().__init__(remote_dir, cache_dir, chunks, {})
        self.delay_s = delay_s

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        time.sleep(self.delay_s)
        os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
        with open(local_filepath, "wb") as f:
            f.write(b"0" * 1024)

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        await asyncio.sleep(self.delay_s)
        return b"0" * 1024


class _ConfigShim:
    """Minimal config surface for ``adownload_chunk_indexes``."""

    def __init__(self, cache_dir: str, chunks: list, downloader: Downloader) -> None:
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._downloader = downloader
        self._shared_chunk_indexes: set[int] = set()
        self._compressor_name = None

    def try_decompress(self, local_chunkpath: str) -> None:
        """No-op: synthetic payloads are uncompressed."""
        return


def main() -> None:
    """Compare serial sync downloads vs asyncio.gather overlapping the same latency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=50, help="Per-chunk simulated RTT")
    args = parser.parse_args()

    delay_s = args.delay_ms / 1000.0
    chunks = [{"filename": f"chunk-{i:08d}.bin", "chunk_bytes": 1024} for i in range(args.chunks)]
    indexes = list(range(args.chunks))

    with tempfile.TemporaryDirectory(prefix="litdata-async-prefetch-") as tmp:
        serial_dir = os.path.join(tmp, "serial")
        async_dir = os.path.join(tmp, "async")
        os.makedirs(serial_dir)
        os.makedirs(async_dir)

        serial_dl = _LatencyDownloader("remote://bucket", serial_dir, chunks, delay_s)
        assert downloader_supports_adownload(serial_dl)

        t0 = time.perf_counter()
        for idx in indexes:
            serial_dl.download_chunk_from_index(idx)
        serial_s = time.perf_counter() - t0

        async_dl = _LatencyDownloader("remote://bucket", async_dir, chunks, delay_s)
        async_cfg = _ConfigShim(async_dir, chunks, async_dl)
        t0 = time.perf_counter()
        asyncio.run(adownload_chunk_indexes(async_cfg, indexes))  # type: ignore[arg-type]
        async_s = time.perf_counter() - t0

    print(f"chunks={args.chunks} delay_ms={args.delay_ms}")
    print(f"serial_sync     {serial_s:.3f}s  ({args.chunks * delay_s:.3f}s expected lower bound)")
    print(f"asyncio_gather  {async_s:.3f}s  (~{delay_s:.3f}s expected if fully overlapped)")
    print(f"speedup         {serial_s / async_s:.2f}x" if async_s else "speedup n/a")
    print(
        "\nNote: enable in PrepareChunksThread with LITDATA_ASYNC_CHUNK_PREFETCH=1. "
        "For real S3/R2 (Studio connections), use scripts/bench/bench_s3_remote.py — "
        "local CIFAR-in-RAM is the wrong workload."
    )


if __name__ == "__main__":
    main()

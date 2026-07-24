#!/usr/bin/env python3
"""Microbench: boto3 vs obstore for real S3 chunk downloads.

Compares LitData's sync path (boto3 ``download_file``) against
`obstore <https://github.com/developmentseed/obstore>`_ sync/async GETs on
the same ImageNet-sized chunks.

Example::

    PYTHON_GIL=0 python scripts/bench/bench_obstore_vs_boto3.py
    PYTHON_GIL=0 python scripts/bench/bench_obstore_vs_boto3.py --chunks 8 --concurrency 4
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT = "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search"


def _load_remote_and_chunks(input_dir: str) -> tuple[str, list[dict]]:
    from litdata.streaming.resolver import _resolve_dir

    resolved = _resolve_dir(input_dir)
    remote = (resolved.url or resolved.path or "").rstrip("/")
    if not remote.startswith("s3://"):
        # Studio FUSE path → resolve sibling index for remote URL via mount.
        index_path = os.path.join(resolved.path or input_dir, "index.json")
    else:
        # Prefer mount mirror for index.json
        parts = remote[len("s3://") :].split("/", 1)
        index_path = f"/teamspace/s3_connections/{parts[0]}/{parts[1]}/index.json" if len(parts) == 2 else ""
        if not os.path.exists(index_path):
            index_path = os.path.join(resolved.path or "", "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found for {input_dir}")

    with open(index_path, encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    # Prefer true s3 URL when resolver provides it.
    if resolved.url and resolved.url.startswith("s3://"):
        remote = resolved.url.rstrip("/")
    elif not remote.startswith("s3://"):
        # Infer from mount: /teamspace/s3_connections/<bucket>/<key>
        marker = "/teamspace/s3_connections/"
        path = resolved.path or input_dir
        if marker in path:
            rest = path.split(marker, 1)[1].strip("/")
            remote = "s3://" + rest
        else:
            raise ValueError(f"Could not resolve s3:// URL from {input_dir}")
    return remote, chunks


def _keys(remote: str, chunks: list[dict], n: int) -> list[tuple[str, str, int]]:
    """Return list of (bucket, key, bytes)."""
    parsed = urlparse(remote)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    out = []
    for c in chunks[:n]:
        key = f"{prefix.rstrip('/')}/{c['filename']}" if prefix else c["filename"]
        out.append((bucket, key, int(c["chunk_bytes"])))
    return out


def _boto3_client():
    import boto3

    return boto3.client("s3")


def _obstore_store(bucket: str):
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    session = boto3.Session()
    return S3Store(bucket, credential_provider=Boto3CredentialProvider(session))


def _wipe(dirpath: str) -> None:
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok=True)


def bench_boto3_serial(client, items: list[tuple[str, str, int]], out_dir: str) -> float:
    from boto3.s3.transfer import TransferConfig

    cfg = TransferConfig(use_threads=False)
    t0 = time.perf_counter()
    for i, (bucket, key, _) in enumerate(items):
        dest = os.path.join(out_dir, f"boto3-{i}.bin")
        client.download_file(bucket, key, dest, Config=cfg)
    return time.perf_counter() - t0


def bench_boto3_threaded(client, items: list[tuple[str, str, int]], out_dir: str, workers: int) -> float:
    from boto3.s3.transfer import TransferConfig

    cfg = TransferConfig(use_threads=False)

    def _one(i_bucket_key: tuple[int, str, str]) -> None:
        i, bucket, key = i_bucket_key
        dest = os.path.join(out_dir, f"boto3t-{i}.bin")
        client.download_file(bucket, key, dest, Config=cfg)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, [(i, b, k) for i, (b, k, _) in enumerate(items)]))
    return time.perf_counter() - t0


def bench_obstore_sync(store, items: list[tuple[str, str, int]], out_dir: str) -> float:
    import obstore as obs

    t0 = time.perf_counter()
    for i, (_, key, _) in enumerate(items):
        dest = os.path.join(out_dir, f"obs-sync-{i}.bin")
        resp = obs.get(store, key)
        with open(dest, "wb", buffering=1024 * 1024) as f:
            for chunk in resp.stream():
                f.write(chunk)
    return time.perf_counter() - t0


async def _obstore_async_one(store, key: str, dest: str) -> None:
    import obstore as obs

    resp = await obs.get_async(store, key)
    with open(dest, "wb", buffering=1024 * 1024) as f:
        async for chunk in resp.stream():
            f.write(chunk)


def bench_obstore_async_gather(store, items: list[tuple[str, str, int]], out_dir: str) -> float:
    async def _run() -> None:
        await asyncio.gather(
            *[
                _obstore_async_one(store, key, os.path.join(out_dir, f"obs-async-{i}.bin"))
                for i, (_, key, _) in enumerate(items)
            ]
        )

    t0 = time.perf_counter()
    asyncio.run(_run())
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--chunks", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=4, help="Thread pool size for boto3 concurrent")
    args = p.parse_args()

    remote, chunks = _load_remote_and_chunks(args.input_dir)
    n = min(args.chunks, len(chunks))
    items = _keys(remote, chunks, n)
    total_mb = sum(b for _, _, b in items) / (1024 * 1024)

    print(f"python={sys.version.split()[0]}")
    print(f"remote={remote}")
    print(f"chunks={n} total_mb={total_mb:.1f} mean_mb={total_mb / n:.1f}")
    print(f"keys0={items[0][0]}/{items[0][1]}")

    client = _boto3_client()
    store = _obstore_store(items[0][0])

    results = []
    for name, fn in [
        ("boto3_serial", lambda d: bench_boto3_serial(client, items, d)),
        ("boto3_threaded", lambda d: bench_boto3_threaded(client, items, d, args.concurrency)),
        ("obstore_sync_serial", lambda d: bench_obstore_sync(store, items, d)),
        ("obstore_async_gather", lambda d: bench_obstore_async_gather(store, items, d)),
    ]:
        out = tempfile.mkdtemp(prefix=f"litdata-{name}-")
        try:
            # Warm credentials / connections once before timed run when useful.
            if name == "boto3_serial":
                warm = tempfile.mkdtemp(prefix="warm-")
                try:
                    bench_boto3_serial(client, items[:1], warm)
                finally:
                    shutil.rmtree(warm, ignore_errors=True)
            elapsed = fn(out)
            mb_s = total_mb / elapsed if elapsed else float("nan")
            row = {"name": name, "elapsed_s": elapsed, "MB_s": mb_s, "chunks": n, "total_mb": total_mb}
            results.append(row)
            print(f"{name:22s} {elapsed:7.3f}s  {mb_s:7.1f} MB/s")
        finally:
            shutil.rmtree(out, ignore_errors=True)

    print("\n=== summary JSON ===")
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

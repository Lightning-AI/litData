#!/usr/bin/env python3
r"""Grid-search obstore ``min_chunk_size`` vs download concurrency.

Cold-epoch bottleneck is time until the first few chunk files land so workers
can start reading — not peak MB/s after the pipeline is full.

Metrics per cell (asyncio gather of ``concurrency`` downloads):
  * t_first — wall time until the first chunk file is fully written
  * t_ready — wall time until ``ready_count`` chunks are written (default 4,
    matching async ``max_pre_download`` floor)
  * t_all   — wall time until all in-flight downloads finish

Example::

    PYTHON_GIL=0 python -Xgil=0 scripts/bench/bench_obstore_chunksize_grid.py
    PYTHON_GIL=0 python -Xgil=0 scripts/bench/bench_obstore_chunksize_grid.py \\
        --concurrency 4,8,16,32,48 --chunk-sizes 1,2,4,8,10,16 --reps 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_INPUT = "/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_search"


def _load_keys(input_dir: str, n: int) -> tuple[str, list[str]]:
    from litdata.streaming.resolver import _resolve_dir

    resolved = _resolve_dir(input_dir)
    remote = (resolved.url or "").rstrip("/")
    if not remote.startswith("s3://"):
        marker = "/teamspace/s3_connections/"
        path = resolved.path or input_dir
        if marker in path:
            remote = "s3://" + path.split(marker, 1)[1].strip("/")
        else:
            raise ValueError(f"Could not resolve s3:// URL from {input_dir}")

    parts = remote[len("s3://") :].split("/", 1)
    index_path = f"/teamspace/s3_connections/{parts[0]}/{parts[1]}/index.json"
    with open(index_path, encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    parsed = urlparse(remote)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    keys = [f"{prefix}/{c['filename']}" if prefix else c["filename"] for c in chunks[:n]]
    return bucket, keys


def _store(bucket: str):
    import boto3
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    return S3Store(
        bucket,
        credential_provider=Boto3CredentialProvider(boto3.Session()),
        client_options={"timeout": "200s"},
    )


async def _download_one(
    store,
    key: str,
    dest: str,
    min_chunk_size: int,
    done_times: list[float],
    t0: float,
) -> None:
    import obstore as obs

    resp = await obs.get_async(store, key)
    tmp = dest + ".tmp"
    with open(tmp, "wb", buffering=1024 * 1024) as f:
        async for chunk in resp.stream(min_chunk_size=min_chunk_size):
            f.write(chunk)
    os.replace(tmp, dest)
    done_times.append(time.perf_counter() - t0)


async def _run_cell(
    store,
    keys: list[str],
    concurrency: int,
    min_chunk_size: int,
    ready_count: int,
) -> dict[str, float]:
    root = tempfile.mkdtemp(prefix="litdata-grid-")
    try:
        # Take ``concurrency`` distinct keys (wrap if needed).
        batch = [keys[i % len(keys)] for i in range(concurrency)]
        done_times: list[float] = []
        t0 = time.perf_counter()
        await asyncio.gather(
            *[
                _download_one(
                    store,
                    key,
                    os.path.join(root, f"{i}.bin"),
                    min_chunk_size,
                    done_times,
                    t0,
                )
                for i, key in enumerate(batch)
            ]
        )
        done_times.sort()
        k = min(ready_count, len(done_times))
        return {
            "t_first": done_times[0],
            "t_ready": done_times[k - 1],
            "t_all": done_times[-1],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument(
        "--concurrency",
        default="4,8,16,32,48",
        help="In-flight downloads (proxy for workers × pre-download pressure)",
    )
    p.add_argument(
        "--chunk-sizes",
        default="1,2,4,8,10,16",
        help="obstore stream min_chunk_size in MiB",
    )
    p.add_argument("--ready-count", type=int, default=4, help="Chunks to unlock read")
    p.add_argument("--pool-keys", type=int, default=64, help="Distinct S3 keys to draw from")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", action="store_true", default=True)
    args = p.parse_args()

    concs = _parse_int_list(args.concurrency)
    sizes_mib = _parse_int_list(args.chunk_sizes)
    bucket, keys = _load_keys(args.input_dir, max(args.pool_keys, max(concs)))
    store = _store(bucket)

    print(f"python={sys.version.split()[0]}")
    print(f"bucket={bucket} keys={len(keys)} ready_count={args.ready_count} reps={args.reps}")
    print(f"concurrency={concs}")
    print(f"chunk_sizes_mib={sizes_mib}")
    print()

    if args.warmup:
        asyncio.run(_run_cell(store, keys, concurrency=2, min_chunk_size=8 << 20, ready_count=1))

    rows: list[dict] = []
    # Header: emphasize t_ready (unblock) then t_first.
    print(f"{'conc':>5} {'mcs_MiB':>8} {'t_first':>8} {'t_ready':>8} {'t_all':>8}  (median of {args.reps})")
    print("-" * 52)

    for conc in concs:
        for mib in sizes_mib:
            samples = []
            for _ in range(args.reps):
                samples.append(
                    asyncio.run(
                        _run_cell(
                            store,
                            keys,
                            concurrency=conc,
                            min_chunk_size=mib << 20,
                            ready_count=args.ready_count,
                        )
                    )
                )
            med = {k: statistics.median([s[k] for s in samples]) for k in ("t_first", "t_ready", "t_all")}
            row = {
                "concurrency": conc,
                "min_chunk_mib": mib,
                **med,
                "reps": args.reps,
            }
            rows.append(row)
            print(f"{conc:5d} {mib:8d} {med['t_first']:8.3f} {med['t_ready']:8.3f} {med['t_all']:8.3f}")

    # Best per concurrency by t_ready, then overall.
    print("\n=== best min_chunk_size per concurrency (by t_ready) ===")
    for conc in concs:
        cell = min((r for r in rows if r["concurrency"] == conc), key=lambda r: r["t_ready"])
        print(
            f"conc={conc:3d}  mcs={cell['min_chunk_mib']:2d}MiB  "
            f"t_ready={cell['t_ready']:.3f}s  t_first={cell['t_first']:.3f}s"
        )

    best = min(rows, key=lambda r: r["t_ready"])
    print(
        f"\n=== overall best t_ready ===\n"
        f"conc={best['concurrency']} mcs={best['min_chunk_mib']}MiB "
        f"t_ready={best['t_ready']:.3f}s t_first={best['t_first']:.3f}s "
        f"t_all={best['t_all']:.3f}s"
    )

    print("\n=== JSON ===")
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()

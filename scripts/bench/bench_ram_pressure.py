#!/usr/bin/env python3
"""Reproduce RAM pressure from remote LitData streaming (SSH hang near 100% RAM).

Builds a synthetic 64MB-chunk dataset on Studio ``lightning_storage`` (direct R2,
never FUSE), then streams it while sampling ``MemAvailable`` / RSS / CPU.

Safety: stop when used RAM exceeds ``--ram-ceiling`` (default 0.98 of MemTotal),
slightly above LitData's ``LITDATA_RAM_CEILING`` (default 0.95) so we can see the
library hold the line. Use ``--unsafe-full-ram`` only on a dedicated repro box.

Examples::

    python scripts/bench/bench_ram_pressure.py --prepare
    python scripts/bench/bench_ram_pressure.py --arm aggressive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(os.environ.get("LITDATA_BENCH_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from litdata import Jpeg  # noqa: E402
from litdata.processing.functions import optimize  # noqa: E402
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402

DEFAULT_INPUT = "/teamspace/lightning_storage/litdata-r2"
DEFAULT_CACHE = "/cache/chunks"
DEFAULT_IMAGENET_INPUT = "/teamspace/lightning_storage/litdata-r2/synthetic-imagenet1m"
# 64 MiB samples → one sample per 64MB chunk. ~2000 items ≈ 125GB, enough to
# exceed 110GB host RAM if WILLNEED/page cache are uncapped.
_DEFAULT_PAYLOAD = 64 * 1024 * 1024
_DEFAULT_TARGET_GB = 125.0
_DEFAULT_IMAGE_COUNT = 1_000_000
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_IMAGE_QUALITY = 90


def _synthetic_sample(index: int) -> dict:
    """Deterministic 1-D blob so optimize fills 64MB chunks without RNG cost."""
    nbytes = int(os.environ.get("LITDATA_RAM_BENCH_PAYLOAD", str(_DEFAULT_PAYLOAD)))
    return {"x": np.full(nbytes, index % 256, dtype=np.uint8), "y": int(index % 1000)}


def _synthetic_imagenet_sample(index: int) -> dict:
    """Deterministic ImageNet-shaped RGB JPEG with an ImageNet-style class label."""
    size = int(os.environ["LITDATA_IMAGENET_BENCH_IMAGE_SIZE"])
    quality = int(os.environ["LITDATA_IMAGENET_BENCH_IMAGE_QUALITY"])
    image = np.random.default_rng(index).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return {"image": Jpeg(array=image, quality=quality), "y": int(index % 1000)}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _meminfo() -> dict[str, int]:
    """Return MemTotal / MemAvailable / MemFree / AnonPages in bytes."""
    total = avail = free = anon = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                kib = int(parts[1]) * 1024
                if line.startswith("MemTotal:"):
                    total = kib
                elif line.startswith("MemAvailable:"):
                    avail = kib
                elif line.startswith("MemFree:"):
                    free = kib
                elif line.startswith("AnonPages:"):
                    anon = kib
    except OSError:
        pass
    return {"mem_total_b": total, "mem_available_b": avail, "mem_free_b": free, "anon_b": anon}


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _cpu_pct(prev: tuple[float, float] | None) -> tuple[float, tuple[float, float]]:
    """Process CPU% since ``prev`` (utime+stime, wall)."""
    now = time.perf_counter()
    try:
        with open("/proc/self/stat", encoding="utf-8") as fh:
            fields = fh.read().split()
        ticks = (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        ticks = 0.0
    if prev is None:
        return 0.0, (ticks, now)
    dt = max(1e-6, now - prev[1])
    return max(0.0, (ticks - prev[0]) / dt * 100.0), (ticks, now)


def _wipe_cache(cache_dir: str) -> None:
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)


def prepare_dataset(output_dir: str, target_gb: float, payload_bytes: int, num_workers: int) -> None:
    """Optimize synthetic blobs onto ``lightning_storage`` (direct R2 upload)."""
    os.environ["LITDATA_RAM_BENCH_PAYLOAD"] = str(payload_bytes)
    n_items = max(1, int(target_gb * (1024**3) / max(1, payload_bytes)))
    print(
        f"[prepare] {n_items} samples × {payload_bytes / 1024**2:.0f}MiB → {output_dir} "
        f"(~{n_items * payload_bytes / 1024**3:.1f}GiB, workers={num_workers})",
        flush=True,
    )
    optimize(
        fn=_synthetic_sample,
        inputs=list(range(n_items)),
        output_dir=output_dir,
        chunk_bytes="64MB",
        num_workers=num_workers,
        mode="overwrite",
        reorder_files=False,
    )
    print("[prepare] done", flush=True)


def prepare_synthetic_imagenet(
    output_dir: str,
    num_images: int,
    image_size: int,
    image_quality: int,
    num_workers: int,
) -> None:
    """Optimize deterministic ImageNet-shaped JPEG samples onto direct object storage."""
    os.environ["LITDATA_IMAGENET_BENCH_IMAGE_SIZE"] = str(image_size)
    os.environ["LITDATA_IMAGENET_BENCH_IMAGE_QUALITY"] = str(image_quality)
    print(
        f"[prepare] {num_images:,} synthetic {image_size}×{image_size} JPEG images "
        f"(quality={image_quality}) → {output_dir} (workers={num_workers})",
        flush=True,
    )
    optimize(
        fn=_synthetic_imagenet_sample,
        inputs=list(range(num_images)),
        output_dir=output_dir,
        chunk_bytes="64MB",
        num_workers=num_workers,
        mode="overwrite",
        reorder_files=False,
    )
    print("[prepare] done", flush=True)


def _build_loader(
    *,
    input_dir: str,
    cache_dir: str,
    workers: int,
    batch_size: int,
    prefetch_factor: int | None,
    max_pre_download: int,
) -> StreamingDataLoader:
    ds = StreamingDataset(
        input_dir,
        cache_dir=cache_dir,
        shuffle=True,
        drop_last=True,
        max_pre_download=max_pre_download,
        max_cache_size="200GB",
    )
    return StreamingDataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        pin_memory=True,
    )


def run_stream(args: argparse.Namespace) -> dict:
    """Stream until time/batch floors or the RAM ceiling."""
    mem0 = _meminfo()
    total = mem0["mem_total_b"] or 1
    ceiling = 1.0 if args.unsafe_full_ram else args.ram_ceiling
    extra = []
    if args.extra_rss_gb > 0:
        extra.append(bytearray(int(args.extra_rss_gb * 1024**3)))
        print(f"[bench] allocated extra RSS ≈ {args.extra_rss_gb} GiB", flush=True)

    if args.clear_cache:
        _wipe_cache(args.cache_dir)

    workers = args.workers if args.workers is not None else (os.cpu_count() or 8)
    prefetch = args.prefetch_factor
    max_pre = args.max_pre_download
    hold_n = args.hold_batches if args.hold_batches is not None else max(8, workers * (prefetch or 2))
    held: deque = deque(maxlen=hold_n)
    print(
        f"[bench] arm={args.arm} workers={workers} batch_size={args.batch_size} "
        f"prefetch_factor={prefetch} max_pre_download={max_pre} hold_batches={hold_n} "
        f"ram_ceiling={ceiling}",
        flush=True,
    )

    loader = _build_loader(
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        workers=workers,
        batch_size=args.batch_size,
        prefetch_factor=prefetch,
        max_pre_download=max_pre,
    )
    actual_workers = loader.num_workers
    samples = 0
    batches = 0
    min_avail = mem0["mem_available_b"]
    max_anon = mem0["anon_b"]
    samples_log: list[dict] = []
    cpu_state = None
    hit_ceiling = False

    t0 = time.perf_counter()
    while True:
        it = iter(loader)
        epoch_empty = True
        while True:
            try:
                batch = next(it)
            except StopIteration:
                break
            epoch_empty = False
            batches += 1
            samples += len(batch["y"]) if isinstance(batch, dict) else len(batch)
            payload = batch.get("x", batch.get("image")) if isinstance(batch, dict) else batch
            if payload is None:
                raise RuntimeError("Expected benchmark samples to contain an 'x' or 'image' payload.")
            held.append(payload.detach().clone() if hasattr(payload, "detach") else np.array(payload, copy=True))
            cpu_pct, cpu_state = _cpu_pct(cpu_state)
            mem = _meminfo()
            min_avail = min(min_avail, mem["mem_available_b"] or min_avail)
            max_anon = max(max_anon, mem["anon_b"])
            used_frac = 1.0 - (mem["mem_available_b"] / total)
            anon_frac = mem["anon_b"] / total
            if batches == 1 or batches % args.sample_every == 0:
                row = {
                    "batch": batches,
                    "samples": samples,
                    "elapsed_s": time.perf_counter() - t0,
                    "mem_available_b": mem["mem_available_b"],
                    "anon_b": mem["anon_b"],
                    "rss_b": _rss_bytes(),
                    "cpu_pct": round(cpu_pct, 1),
                    "used_frac": round(used_frac, 4),
                    "anon_frac": round(anon_frac, 4),
                }
                samples_log.append(row)
                print(
                    f"[bench] batch={batches} samples={samples} "
                    f"avail={mem['mem_available_b'] / 1024**3:.1f}GiB "
                    f"used={used_frac:.1%} anon={anon_frac:.1%} "
                    f"rss={row['rss_b'] / 1024**3:.2f}GiB cpu={cpu_pct:.0f}%",
                    flush=True,
                )
            if used_frac >= ceiling or anon_frac >= ceiling:
                hit_ceiling = True
                print(f"[bench] RAM ceiling {ceiling:.0%} hit — stopping", flush=True)
                break
            elapsed = time.perf_counter() - t0
            if batches >= args.min_batches and elapsed >= args.min_seconds:
                break
        if hit_ceiling or epoch_empty:
            break
        elapsed = time.perf_counter() - t0
        if batches >= args.min_batches and elapsed >= args.min_seconds:
            break

    elapsed = time.perf_counter() - t0
    result = {
        "arm": args.arm,
        "git_sha": _git_sha(),
        "input_dir": args.input_dir,
        "requested_workers": workers,
        "actual_workers": actual_workers,
        "prefetch_factor": prefetch,
        "max_pre_download": max_pre,
        "batch_size": args.batch_size,
        "batches": batches,
        "samples": samples,
        "elapsed_s": elapsed,
        "samples_per_s": samples / elapsed if elapsed else float("nan"),
        "mem_total_b": total,
        "mem_available_start_b": mem0["mem_available_b"],
        "mem_available_min_b": min_avail,
        "anon_max_b": max_anon,
        "used_frac_peak": round(1.0 - (min_avail / total), 4),
        "anon_frac_peak": round(max_anon / total, 4),
        "ram_ceiling": ceiling,
        "hit_ceiling": hit_ceiling,
        "extra_rss_gb": args.extra_rss_gb,
        "nproc": os.cpu_count(),
        "samples_log": samples_log,
    }
    del extra
    return result


def main() -> None:
    """CLI: --prepare synthetic R2 dataset, then stream with RAM sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="optimize synthetic data onto --input-dir")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument(
        "--target-gb",
        type=float,
        default=_DEFAULT_TARGET_GB,
        help="Uncompressed payload to write (default 125GiB so uncapped streaming can exceed 110GiB RAM)",
    )
    parser.add_argument("--payload-bytes", type=int, default=_DEFAULT_PAYLOAD)
    parser.add_argument(
        "--synthetic-imagenet",
        action="store_true",
        help="Use synthetic 224×224 JPEG ImageNet samples; --prepare writes --num-images samples.",
    )
    parser.add_argument("--num-images", type=int, default=_DEFAULT_IMAGE_COUNT)
    parser.add_argument("--image-size", type=int, default=_DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-quality", type=int, default=_DEFAULT_IMAGE_QUALITY)
    parser.add_argument("--prepare-workers", type=int, default=16)
    parser.add_argument("--arm", choices=("aggressive", "tuned"), default="aggressive")
    parser.add_argument("--workers", type=int, default=None, help="default: cpu_count (aggressive) or 4 (tuned)")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--max-pre-download", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-batches", type=int, default=80)
    parser.add_argument("--min-seconds", type=float, default=45.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument(
        "--ram-ceiling",
        type=float,
        default=0.98,
        help="Bench abort fraction of MemTotal (above LITDATA_RAM_CEILING so the library is what holds).",
    )
    parser.add_argument("--unsafe-full-ram", action="store_true")
    parser.add_argument(
        "--hold-batches",
        type=int,
        default=None,
        help="Keep this many copied batches in RAM (default workers×prefetch). Simulates a slow GPU consumer.",
    )
    parser.add_argument("--extra-rss-gb", type=float, default=0.0, help="stand-in for training + Cursor RSS")
    parser.add_argument("--no-clear-cache", dest="clear_cache", action="store_false")
    parser.set_defaults(clear_cache=True)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "scripts" / "bench" / "results"))
    args = parser.parse_args()

    if args.synthetic_imagenet and args.input_dir == DEFAULT_INPUT:
        args.input_dir = DEFAULT_IMAGENET_INPUT
    if args.prepare:
        if args.synthetic_imagenet:
            prepare_synthetic_imagenet(
                args.input_dir,
                args.num_images,
                args.image_size,
                args.image_quality,
                args.prepare_workers,
            )
        else:
            prepare_dataset(args.input_dir, args.target_gb, args.payload_bytes, args.prepare_workers)
        return

    if args.arm == "tuned":
        if args.workers is None:
            args.workers = 4
        if args.prefetch_factor is None:
            args.prefetch_factor = 2
        if args.max_pre_download is None:
            args.max_pre_download = 2
    else:
        if args.prefetch_factor is None:
            args.prefetch_factor = 8
        if args.max_pre_download is None:
            args.max_pre_download = 32

    result = run_stream(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ram_pressure.{result['git_sha']}.{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_log"}, indent=2))
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()

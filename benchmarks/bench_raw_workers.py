"""Exhaustive worker × prefetch sweep for StreamingRawDataset.

Uses spawn + persistent_workers after a warm index. Writes a JSON summary for docs.

Protocol: timed window is max(BATCHES, MIN_SECONDS) — both floors required.
High-worker cells (w≥16) always run ≥30s. Artifacts use SHA/ts-suffixed paths.

Ranged vs whole-object compare (optional):
  LITDATA_RAW_RANGE_PARALLEL_THRESHOLD=0         # whole-object GETs (also the dataset default)
  LITDATA_RAW_RANGE_PARALLEL_THRESHOLD=1         # force ranged for any sized object
  LITDATA_RAW_RANGE_PARALLEL_THRESHOLD=33554432  # opt in at 32MiB
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_raw_before_vs_after import effective_min_seconds, git_sha, unique_result_path
from torch.utils.data import DataLoader
from uvloop_status import log_loop_runner_backend, uvloop_package_status

from litdata import StreamingRawDataset

INPUT = "/teamspace/s3_connections/imagenet-1m-template/raw/val"
ROOT = Path(tempfile.gettempdir()) / "litdata-raw-worker-sweep"
OUT_DIR = Path(__file__).resolve().parent / "results"
BS = 64
BATCHES = 300
MIN_SECONDS = 30.0
# Up to host vCPUs (4×L4 Studio = 48).
WORKERS = [0, 1, 2, 4, 8, 16, 24, 32, 48]
PREFETCH = [0, 16, 32, 64, 96, 128]
TIMEOUT = 600.0
OLD_FUSE = 75.2

# Optional override for ranged-vs-whole compare; None → dataset default (0 = opt-in off).
_RANGE_ENV = os.getenv("LITDATA_RAW_RANGE_PARALLEL_THRESHOLD")
RANGE_PARALLEL_THRESHOLD: int | None = int(_RANGE_ENV) if _RANGE_ENV is not None else None


def log(msg: str) -> None:
    """Print a timestamped benchmark log line."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class HangWatchdog:
    """Kill the process if a step exceeds ``timeout_s`` without heartbeat."""

    def __init__(self, timeout_s: float) -> None:
        """Initialize the watchdog with a hang timeout in seconds."""
        self.timeout_s = timeout_s
        self._label = "init"
        self._beat = time.monotonic()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the background watchdog thread."""
        self._t.start()

    def beat(self, label: str) -> None:
        """Record progress so the watchdog does not abort."""
        self._label = label
        self._beat = time.monotonic()

    def stop(self) -> None:
        """Stop the background watchdog thread."""
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            idle = time.monotonic() - self._beat
            if idle > self.timeout_s:
                log(f"HANG at '{self._label}' after {idle:.1f}s — abort")
                os._exit(124)


def copy_index(src: Path, dst: Path) -> None:
    """Copy a cached index tree from ``src`` to ``dst``."""
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_dir():
            shutil.copytree(p, dst / p.name)
        else:
            shutil.copy2(p, dst / p.name)


def run(label: str, *, num_workers: int, max_prefetch: int, seed: Path, wd: HangWatchdog) -> dict:
    """Run one worker/prefetch trial and return timing stats."""
    cache = ROOT / label
    wd.beat(f"{label}: setup")
    copy_index(seed, cache)
    ds_kwargs: dict = {
        "cache_dir": str(cache),
        "cache_files": False,
        "max_prefetch": max_prefetch,
        "max_concurrent_downloads": 64,
    }
    if RANGE_PARALLEL_THRESHOLD is not None:
        ds_kwargs["range_parallel_threshold"] = RANGE_PARALLEL_THRESHOLD
    ds = StreamingRawDataset(INPUT, **ds_kwargs)
    kwargs: dict = {"batch_size": BS, "num_workers": num_workers, "shuffle": False}
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    loader = DataLoader(ds, **kwargs)
    it = iter(loader)
    wd.beat(f"{label}: warm")
    t0 = time.perf_counter()
    next(it)
    warm_s = time.perf_counter() - t0

    min_s = effective_min_seconds(num_workers, MIN_SECONDS)
    samples = 0
    timed_batches = 0
    wd.beat(f"{label}: timed")
    t0 = time.perf_counter()
    while True:
        batch = next(it)
        samples += len(batch)
        timed_batches += 1
        wd.beat(f"{label}: batch {timed_batches}")
        elapsed = time.perf_counter() - t0
        if timed_batches >= BATCHES and elapsed >= min_s:
            break
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed else 0.0
    log(
        f"[{label}] w={num_workers} pf={max_prefetch} "
        f"warm={warm_s:.2f}s | {timed_batches}×{samples // max(timed_batches, 1)} in {elapsed:.2f}s "
        f"(need ≥{BATCHES} batches & ≥{min_s:.0f}s) → {ips:.1f} samples/s ({ips / OLD_FUSE:.1f}x FUSE)"
    )
    del it, loader, ds
    return {
        "label": label,
        "workers": num_workers,
        "prefetch": max_prefetch,
        "batches": timed_batches,
        "min_seconds_effective": min_s,
        "ips": ips,
        "warm_s": warm_s,
        "elapsed": elapsed,
        "samples": samples,
    }


def print_matrix(results: list[dict]) -> None:
    """Print a workers × prefetch samples/s matrix."""
    by_key = {(r["workers"], r["prefetch"]): r["ips"] for r in results}
    header = f"{'w/pf':>6}" + "".join(f"{p:>10}" for p in PREFETCH)
    log("\n=== Matrix (samples/s) ===")
    log(header)
    for w in WORKERS:
        row = f"{w:>6}"
        for p in PREFETCH:
            ips = by_key.get((w, p))
            row += f"{ips:>10.1f}" if ips is not None else f"{'—':>10}"
        log(row)


def main() -> None:
    """CLI entrypoint for the exhaustive worker × prefetch sweep."""
    if ROOT.exists():
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    ncpu = os.cpu_count() or 0
    n_configs = len(WORKERS) * len(PREFETCH)
    log(f"uvloop package: {uvloop_package_status()}")
    log(
        f"Exhaustive sweep input={INPUT} bs={BS} batches={BATCHES} "
        f"mp=spawn persistent_workers cpus={ncpu} configs={n_configs}"
    )
    log(f"WORKERS={WORKERS}")
    log(f"PREFETCH={PREFETCH}")
    log(
        f"range_parallel_threshold="
        f"{RANGE_PARALLEL_THRESHOLD if RANGE_PARALLEL_THRESHOLD is not None else 'default(32MiB)'}"
    )

    wd.beat("index seed")
    seed = ROOT / "seed"
    t0 = time.perf_counter()
    seed_kwargs: dict = {"cache_dir": str(seed), "cache_files": False, "max_prefetch": 0}
    if RANGE_PARALLEL_THRESHOLD is not None:
        seed_kwargs["range_parallel_threshold"] = RANGE_PARALLEL_THRESHOLD
    ds = StreamingRawDataset(INPUT, **seed_kwargs)
    n_files = len(ds)
    storage = ds._storage_path
    log(f"Indexed {n_files} files in {time.perf_counter() - t0:.2f}s storage={storage}")
    log_loop_runner_backend(log, prefix="after index seed")
    del ds

    results: list[dict] = []
    try:
        for w in WORKERS:
            for pf in PREFETCH:
                label = f"w{w}_p{pf}"
                results.append(run(label, num_workers=w, max_prefetch=pf, seed=seed, wd=wd))

        print_matrix(results)
        best = max(results, key=lambda r: r["ips"])
        log(
            f"\nBest: {best['label']} → {best['ips']:.1f} samples/s ({best['ips'] / OLD_FUSE:.1f}x vs FUSE ~{OLD_FUSE})"
        )

        payload = {
            "meta": {
                "input": INPUT,
                "storage": storage,
                "n_files": n_files,
                "batch_size": BS,
                "batches": BATCHES,
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "max_concurrent_downloads": 64,
                "cpus": ncpu,
                "fuse_baseline_samples_per_s": OLD_FUSE,
                "workers": WORKERS,
                "prefetch": PREFETCH,
                "range_parallel_threshold": RANGE_PARALLEL_THRESHOLD,
            },
            "results": results,
            "best": best,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = unique_result_path("raw_worker_prefetch_sweep", sha=git_sha())
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"Wrote {path}")
    finally:
        wd.stop()


if __name__ == "__main__":
    main()

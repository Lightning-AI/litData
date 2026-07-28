r"""A/B: stock StreamingRawDataset (main) vs optimized (feature branch).

Run twice with different PYTHONPATH / --side, then merge:

  PYTHONPATH=/tmp/litdata-raw-before/src \
    python benchmarks/bench_raw_before_vs_after.py --side before

  PYTHONPATH=src \
    python benchmarks/bench_raw_before_vs_after.py --side after

  python benchmarks/bench_raw_before_vs_after.py --merge

Defaults aim for trustworthy windows: >=300 batches (or use --min-seconds),
and warm ``num_workers * prefetch_factor`` batches before timing starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from torch.utils.data import DataLoader

# Same mount path as prior sweeps. After remaps to s3:// via _storage_path.
# Before (main) prefers path→LocalDownloader, which has no adownload_fileobj (returns
# None). For a fair cloud A/B we therefore feed before the resolved s3:// URL.
MOUNT_INPUT = "/teamspace/s3_connections/imagenet-1m-template/raw/val"
S3_INPUT = "s3://imagenet-1m-template/raw/val"
ROOT = Path(tempfile.gettempdir()) / "litdata-raw-before-vs-after"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT = OUT_DIR / "raw_before_vs_after.json"
BS = 64
DEFAULT_BATCHES = 300
DEFAULT_MIN_SECONDS = 10.0
DEFAULT_PREFETCH_FACTOR = 2
WORKERS = [0, 1, 2, 4, 8, 16, 24, 32]
TRUST_WORKERS = [0, 2, 4, 8, 16]
TIMEOUT = 600.0
OLD_FUSE = 75.2


def git_sha() -> str:
    """Return short git SHA for the repo containing this script, or empty."""
    env = os.environ.get("LITDATA_BENCH_GIT", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],

            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def input_for(side: str) -> str:
    """Return dataset input path for ``before`` (s3 URL) or ``after`` (mount)."""
    return S3_INPUT if side == "before" else MOUNT_INPUT


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


def detect_side_capabilities() -> dict:
    """Inspect imported litdata for before/after feature markers."""
    import inspect

    from litdata import StreamingRawDataset

    params = set(inspect.signature(StreamingRawDataset.__init__).parameters)
    has_prefetch = "max_prefetch" in params
    has_range = "range_parallel_threshold" in params
    has_loop = False
    uvloop_status = "n/a (before / no LoopRunner)"
    try:
        from litdata.raw.dataset import _loop_backend_name

        has_loop = True
        try:
            import uvloop

            uvloop_status = f"available (uvloop {getattr(uvloop, '__version__', '?')}; create→{_loop_backend_name()})"
        except ImportError:
            uvloop_status = "not installed (stdlib asyncio fallback)"
    except ImportError:
        pass
    return {
        "has_max_prefetch": has_prefetch,
        "has_range_parallel_threshold": has_range,
        "has_loop_runner": has_loop,
        "uvloop": uvloop_status,
        "params": sorted(params - {"self"}),
    }


def storage_path_of(ds) -> str:
    """Best-effort storage path string for JSON meta."""
    if hasattr(ds, "_storage_path"):
        return str(ds._storage_path)
    cm = getattr(ds, "cache_manager", None)
    if cm is not None and hasattr(cm, "_input_dir_path"):
        return str(cm._input_dir_path)
    indir = getattr(ds, "input_dir", None)
    if indir is not None:
        return str(getattr(indir, "url", None) or getattr(indir, "path", None) or indir)
    return MOUNT_INPUT


def make_dataset(
    cache: str,
    *,
    side: str,
    max_prefetch: int,
    hedge_delay: float | None = None,
    download_timeout: float | None = None,
):
    """Construct StreamingRawDataset with side-appropriate kwargs."""
    from litdata import StreamingRawDataset

    kwargs: dict = {"cache_dir": cache, "cache_files": False}
    if side == "after":
        kwargs["max_prefetch"] = max_prefetch
        kwargs["max_concurrent_downloads"] = 64
        kwargs["range_parallel_threshold"] = 0
        # Match new defaults: hedging opt-in (0). Explicit for older trees / clarity.
        kwargs["hedge_delay"] = 0.0 if hedge_delay is None else hedge_delay
        if download_timeout is not None:
            kwargs["download_timeout"] = download_timeout
    elif hedge_delay is not None or download_timeout is not None:
        # before tree has no these knobs — ignore for stock main.
        pass
    return StreamingRawDataset(input_for(side), **kwargs)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def run_one(
    label: str,
    *,
    side: str,
    num_workers: int,
    max_prefetch: int,
    seed: Path,
    wd: HangWatchdog,
    batches: int,
    min_seconds: float,
    prefetch_factor: int,
    hedge_delay: float | None = None,
    download_timeout: float | None = None,
    sha: str = "",
    jsonl: Path | None = None,
) -> dict:
    """Run one worker/prefetch trial and return timing stats."""
    cache = ROOT / side / label
    wd.beat(f"{label}: setup")
    copy_index(seed, cache)
    ds = make_dataset(
        str(cache),
        side=side,
        max_prefetch=max_prefetch,
        hedge_delay=hedge_delay,
        download_timeout=download_timeout,
    )
    kwargs: dict = {"batch_size": BS, "num_workers": num_workers, "shuffle": False}
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **kwargs)
    it = iter(loader)

    # Drain pipeline buffer before timing: 1 warm + workers×prefetch_factor.
    warm_batches = 1 + (num_workers * prefetch_factor if num_workers > 0 else 0)
    wd.beat(f"{label}: warm({warm_batches})")
    t0 = time.perf_counter()
    for i in range(warm_batches):
        next(it)
        wd.beat(f"{label}: warm {i + 1}/{warm_batches}")
    warm_s = time.perf_counter() - t0

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
        # Stop once either floor is met (recommend ≥300 batches OR ≥10s).
        if timed_batches >= batches or elapsed >= min_seconds:
            break
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed else 0.0
    log(
        f"[{side}/{label}] w={num_workers} pf={max_prefetch} "
        f"warm={warm_batches}@{warm_s:.2f}s | {timed_batches}×{samples // max(timed_batches, 1)} "
        f"in {elapsed:.2f}s → {ips:.1f} samples/s"
    )
    result = {
        "side": side,
        "label": label,
        "workers": num_workers,
        "prefetch": max_prefetch,
        "ips": ips,
        "warm_s": warm_s,
        "warm_batches": warm_batches,
        "elapsed": elapsed,
        "samples": samples,
        "batches": timed_batches,
        "hedge_delay": hedge_delay if side == "after" else None,
        "download_timeout": download_timeout if side == "after" else None,
        "git_sha": sha,
        "ts": time.time(),
    }
    if jsonl is not None:
        append_jsonl(jsonl, result)
    del it, loader, ds
    return result


def configs_for(side: str, workers: list[int], *, safety_grid: bool) -> list[tuple]:
    """Return trial configs.

    Normal: (workers, prefetch, hedge_delay|None, download_timeout|None)
    safety_grid (after only): hedge_delay × download_timeout at p0 for w∈{2,4,8}.
    """
    if safety_grid:
        if side != "after":
            raise SystemExit("--safety-grid is only meaningful with --side after")
        out = []
        for w in (2, 4, 8):
            for hd in (0.0, 1.0):
                for dt in (0.0, 120.0):
                    out.append((w, 0, hd, dt))
        return out
    if side == "before":
        return [(w, 0, None, None) for w in workers]
    return [(w, pf, 0.0, None) for w in workers for pf in (0, 16)]


def partial_path(side: str) -> Path:
    """Return path for a side's partial JSON payload."""
    return OUT_DIR / f"raw_before_vs_after.{side}.json"


def jsonl_path(side: str) -> Path:
    """Return path for a side's incremental JSONL log."""
    return OUT_DIR / f"raw_before_vs_after.{side}.jsonl"


def run_side(
    side: str,
    *,
    workers: list[int],
    batches: int,
    min_seconds: float,
    prefetch_factor: int,
    safety_grid: bool,
) -> None:
    """Index once and sweep configs for ``before`` or ``after``."""
    caps = detect_side_capabilities()
    if side == "after" and not caps["has_max_prefetch"]:
        raise SystemExit("PYTHONPATH points at main tree but --side after requested")
    if side == "before" and caps["has_max_prefetch"]:
        raise SystemExit(f"PYTHONPATH points at optimized tree but --side before requested (params={caps['params']})")

    side_root = ROOT / side
    if side_root.exists():
        shutil.rmtree(side_root, ignore_errors=True)
    side_root.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    jpath = jsonl_path(side)
    if jpath.exists():
        jpath.unlink()

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    ncpu = os.cpu_count() or 0
    cfgs = configs_for(side, workers, safety_grid=safety_grid)
    inp = input_for(side)
    log(f"=== side={side} ===")
    log(f"capabilities: {json.dumps(caps)}")
    log(
        f"input={inp} (mount={MOUNT_INPUT}) bs={BS} batches>={batches} "
        f"min_seconds>={min_seconds} warm=1+w*{prefetch_factor} cpus={ncpu} "
        f"configs={len(cfgs)} sha={sha or '?'}"
    )
    log(f"PYTHONPATH[0]={sys.path[0]!r}")

    try:
        wd.beat("index seed")
        seed = side_root / "seed"
        t0 = time.perf_counter()
        ds = make_dataset(str(seed), side=side, max_prefetch=0)
        n_files = len(ds)
        storage = storage_path_of(ds)
        index_s = time.perf_counter() - t0
        log(f"Indexed {n_files} files in {index_s:.2f}s storage={storage}")
        if caps["has_loop_runner"]:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from uvloop_status import log_loop_runner_backend

                log_loop_runner_backend(log, prefix="after index seed")
            except Exception as e:
                log(f"LoopRunner log skipped: {e}")
        else:
            log("LoopRunner: not present on this tree (asyncio.run per batch)")
        del ds

        results: list[dict] = []
        for w, pf, hd, dt in cfgs:
            label = f"w{w}_p{pf}_h{hd}_t{dt}" if safety_grid else f"w{w}_p{pf}"
            results.append(
                run_one(
                    label,
                    side=side,
                    num_workers=w,
                    max_prefetch=pf,
                    seed=seed,
                    wd=wd,
                    batches=batches,
                    min_seconds=min_seconds,
                    prefetch_factor=prefetch_factor,
                    hedge_delay=hd,
                    download_timeout=dt,
                    sha=sha,
                    jsonl=jpath,
                )
            )

        payload = {
            "side": side,
            "meta": {
                "input": inp,
                "mount_input": MOUNT_INPUT,
                "storage": storage,
                "n_files": n_files,
                "index_s": index_s,
                "batch_size": BS,
                "batches": batches,
                "min_seconds": min_seconds,
                "prefetch_factor": prefetch_factor,
                "warm_batches_formula": "1 + num_workers * prefetch_factor",
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "cpus": ncpu,
                "fuse_baseline_samples_per_s": OLD_FUSE,
                "workers": workers,
                "prefetch": [0] if side == "before" else [0, 16],
                "range_parallel_threshold": 0 if side == "after" else None,
                "max_concurrent_downloads": 64 if side == "after" else None,
                "hedge_delay": 0.0 if side == "after" else None,
                "safety_grid": safety_grid,
                "capabilities": caps,
                "git_sha": sha,
                "git_hint": os.environ.get("LITDATA_BENCH_GIT", ""),
                "jsonl": str(jpath),
                "input_note": (
                    "before uses s3:// directly: main prefers FUSE path→LocalDownloader "
                    "which lacks adownload_fileobj; after uses mount and remaps to s3://"
                    if side == "before"
                    else "after uses mount path; _storage_path prefers cloud URL; hedge_delay=0"
                ),
                "caveat": (
                    "Short windows and high-worker cells can be noisy (~2× run-to-run). "
                    "Trust systematic patterns (e.g. prefetch helps), not fine Δ%."
                ),
            },
            "results": results,
        }
        out = partial_path(side)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"Wrote {out}")
        log(f"JSONL {jpath}")
    finally:
        wd.stop()


def merge() -> None:
    """Merge before/after partial JSON into the comparison artifact."""
    before = json.loads(partial_path("before").read_text())
    after = json.loads(partial_path("after").read_text())

    workers = sorted({r["workers"] for r in before["results"]} | {r["workers"] for r in after["results"]})
    before_by_w = {r["workers"]: r for r in before["results"] if r["prefetch"] == 0}
    after_p0 = {r["workers"]: r for r in after["results"] if r["prefetch"] == 0}
    after_p16 = {r["workers"]: r for r in after["results"] if r["prefetch"] == 16}

    rows = []
    cells = []
    for w in workers:
        b = before_by_w.get(w)
        a0 = after_p0.get(w)
        a16 = after_p16.get(w)
        row = {
            "workers": w,
            "before_ips": b["ips"] if b else None,
            "after_prefetch0_ips": a0["ips"] if a0 else None,
            "after_prefetch16_ips": a16["ips"] if a16 else None,
        }
        if b and a0:
            row["speedup_prefetch0"] = a0["ips"] / b["ips"] if b["ips"] else None
            row["delta_pct_prefetch0"] = ((a0["ips"] - b["ips"]) / b["ips"]) * 100.0
        if b and a16:
            row["speedup_prefetch16"] = a16["ips"] / b["ips"] if b["ips"] else None
            row["delta_pct_prefetch16"] = ((a16["ips"] - b["ips"]) / b["ips"]) * 100.0
        after_best = None
        for cand in (a0, a16):
            if cand and (after_best is None or cand["ips"] > after_best["ips"]):
                after_best = cand
        if b and after_best:
            row["after_best_ips"] = after_best["ips"]
            row["after_best_prefetch"] = after_best["prefetch"]
            row["speedup_best"] = after_best["ips"] / b["ips"] if b["ips"] else None
        rows.append(row)
        if not b:
            continue
        for pf, a in ((0, a0), (16, a16)):
            if a is None:
                continue  # omit missing/crashed
            cells.append(
                {
                    "workers": w,
                    "prefetch": pf,
                    "before_ips": b["ips"],
                    "after_ips": a["ips"],
                    "delta_pct": ((a["ips"] - b["ips"]) / b["ips"]) * 100.0 if b["ips"] else None,
                    "speedup": a["ips"] / b["ips"] if b["ips"] else None,
                    "before_elapsed": b.get("elapsed"),
                    "after_elapsed": a.get("elapsed"),
                    "before_batches": b.get("batches"),
                    "after_batches": a.get("batches"),
                }
            )

    best_after = max(cells, key=lambda c: c["after_ips"]) if cells else None
    payload = {
        "meta": {
            "mount_input": MOUNT_INPUT,
            "batch_size": BS,
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "workers": workers,
            "before": before["meta"],
            "after": after["meta"],
            "delta_definition": (
                "delta_pct = ((after - before) / before) * 100; before is stock main "
                "(no max_prefetch API, measured at prefetch=0)"
            ),
            "note": (
                "before = stock StreamingRawDataset on main via s3:// (no max_prefetch / "
                "LoopRunner; FUSE mount path on main selects LocalDownloader and is broken "
                "for async reads); after = feature/raw-streaming-perf defaults "
                "(range_parallel_threshold=0, hedge_delay=0, mount→s3://)"
            ),
            "caveat": (
                "Run-to-run variance can be ~2× on short/high-worker windows. "
                "Prefer systematic patterns over fine Δ%. High-worker cells need long windows."
            ),
        },
        "cells": cells,
        "best_after": best_after,
        "comparison": rows,
        "before_results": before["results"],
        "after_results": after["results"],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {OUT}")

    print()
    print(f"{'w':>4}  {'pf':>4}  {'before':>10}  {'after':>10}  {'Δ%':>8}  {'×':>6}  {'after_s':>8}")
    print("-" * 62)
    for c in cells:
        ae = c.get("after_elapsed")
        ae_s = f"{ae:.2f}" if isinstance(ae, (int, float)) else "?"
        print(
            f"{c['workers']:>4}  {c['prefetch']:>4}  {c['before_ips']:>10.1f}  "
            f"{c['after_ips']:>10.1f}  {c['delta_pct']:>+7.1f}%  {c['speedup']:>5.2f}x  {ae_s:>8}"
        )
    print()
    if best_after:
        print(
            f"Best after (noisy; treat as indicative): w={best_after['workers']} "
            f"prefetch={best_after['prefetch']} → {best_after['after_ips']:.1f} samples/s "
            f"(~{best_after['delta_pct']:+.0f}% / {best_after['speedup']:.2f}x vs before)"
        )


def main() -> None:
    """CLI entrypoint for before/after sweeps and merge."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=("before", "after"))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--batches",
        type=int,
        default=DEFAULT_BATCHES,
        help=f"minimum timed batches (default {DEFAULT_BATCHES}; recommend ≥300)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help=f"minimum timed window seconds (default {DEFAULT_MIN_SECONDS})",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help="DataLoader prefetch_factor; warm batches = 1 + workers * this (default 2)",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default="",
        help="comma-separated worker counts (default: full matrix; use 0,2,4,8,16 for trust A/B)",
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help=f"shorthand for --workers {','.join(map(str, TRUST_WORKERS))} (recommended A/B)",
    )
    parser.add_argument(
        "--safety-grid",
        action="store_true",
        help="after-only 2×2: hedge_delay∈{0,1} × download_timeout∈{0,120} at w∈{2,4,8} p0",
    )
    args = parser.parse_args()
    if args.merge:
        merge()
        return
    if not args.side:
        parser.error("pass --side before|after or --merge")
    if args.trust:
        workers = TRUST_WORKERS
    elif args.workers.strip():
        workers = [int(x) for x in args.workers.split(",") if x.strip()]
    else:
        workers = WORKERS
    run_side(
        args.side,
        workers=workers,
        batches=args.batches,
        min_seconds=args.min_seconds,
        prefetch_factor=args.prefetch_factor,
        safety_grid=args.safety_grid,
    )


if __name__ == "__main__":
    main()

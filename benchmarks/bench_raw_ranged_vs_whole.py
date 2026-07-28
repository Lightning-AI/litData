"""Focused ranged vs whole-object compare on fixed StreamingRawDataset tree.

Protocol: timed window is max(BATCHES, MIN_SECONDS) — both floors required.
Artifacts use SHA/ts-suffixed paths (never overwrite).
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
ROOT = Path(tempfile.gettempdir()) / "litdata-raw-ranged-vs-whole"
OUT_DIR = Path(__file__).resolve().parent / "results"
BS = 64
BATCHES = 300
MIN_SECONDS = 30.0
TIMEOUT = 600.0
CONFIGS = [(4, 0), (4, 128), (8, 0), (8, 128)]
MODES = [
    ("whole_object", 0),
    ("default_32MiB", 33_554_432),
    ("force_ranged", 1),
]


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


def run(label: str, *, num_workers: int, max_prefetch: int, threshold: int, seed: Path, wd: HangWatchdog) -> dict:
    """Run one ranged-vs-whole trial and return timing stats."""
    cache = ROOT / label
    wd.beat(f"{label}: setup")
    copy_index(seed, cache)
    ds = StreamingRawDataset(
        INPUT,
        cache_dir=str(cache),
        cache_files=False,
        max_prefetch=max_prefetch,
        max_concurrent_downloads=64,
        range_parallel_threshold=threshold,
    )
    loader = DataLoader(
        ds,
        batch_size=BS,
        num_workers=num_workers,
        shuffle=False,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    it = iter(loader)
    wd.beat(f"{label}: warm")
    t0 = time.perf_counter()
    _ = next(it)
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
        f"[{label}] thr={threshold} w={num_workers} pf={max_prefetch} "
        f"warm={warm_s:.2f}s | {timed_batches} batches/{samples} in {elapsed:.2f}s "
        f"(need ≥{BATCHES} & ≥{min_s:.0f}s) → {ips:.1f} samples/s"
    )
    del it, loader, ds
    return {
        "label": label,
        "mode": label.rsplit("_w", 1)[0],
        "range_parallel_threshold": threshold,
        "workers": num_workers,
        "prefetch": max_prefetch,
        "batches": timed_batches,
        "min_seconds_effective": min_s,
        "ips": ips,
        "warm_s": warm_s,
        "elapsed": elapsed,
        "samples": samples,
    }


def main() -> None:
    """CLI entrypoint for ranged vs whole-object comparisons."""
    if ROOT.exists():
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    try:
        log(f"uvloop package: {uvloop_package_status()}")
        log(
            f"ranged-vs-whole input={INPUT} bs={BS} batches={BATCHES} "
            f"mp=spawn persistent_workers configs={len(CONFIGS) * len(MODES)}"
        )

        wd.beat("index seed")
        seed = ROOT / "seed"
        t0 = time.perf_counter()
        ds = StreamingRawDataset(INPUT, cache_dir=str(seed), cache_files=False, max_prefetch=0)
        n_files = len(ds)
        storage = ds._storage_path
        sizes: list[int] = []
        try:
            files = getattr(getattr(ds, "_index", None), "files", None)
            if files:
                for i in range(min(200, len(files))):
                    meta = files[i]
                    sz = getattr(meta, "size", None) or (meta.get("size") if isinstance(meta, dict) else None)
                    if sz is not None:
                        sizes.append(int(sz))
        except Exception as e:
            log(f"size probe skipped: {e}")
        log(f"Indexed {n_files} files in {time.perf_counter() - t0:.2f}s storage={storage}")
        if sizes:
            log(
                f"sample sizes (n={len(sizes)}): "
                f"min={min(sizes)} avg={sum(sizes) // len(sizes)} max={max(sizes)} "
                f"(<< 32MiB → default threshold uses whole-object)"
            )
        log_loop_runner_backend(log, prefix="after index seed")
        del ds

        results = []
        for mode_name, thr in MODES:
            for w, pf in CONFIGS:
                label = f"{mode_name}_w{w}_p{pf}"
                results.append(run(label, num_workers=w, max_prefetch=pf, threshold=thr, seed=seed, wd=wd))

        log("\n=== Comparison table (samples/s) ===")
        header = (
            f"{'workers':>8} {'prefetch':>8} | "
            f"{'whole(thr=0)':>14} {'default(32MiB)':>14} {'force_ranged(1)':>16} | "
            f"{'ranged/whole':>12}"
        )
        log(header)
        log("-" * len(header))
        by = {(r["range_parallel_threshold"], r["workers"], r["prefetch"]): r for r in results}
        for w, pf in CONFIGS:
            a = by[(0, w, pf)]["ips"]
            b = by[(33_554_432, w, pf)]["ips"]
            c = by[(1, w, pf)]["ips"]
            ratio = c / a if a else float("nan")
            log(f"{w:>8} {pf:>8} | {a:>14.1f} {b:>14.1f} {c:>16.1f} | {ratio:>11.2f}x")

        winners = []
        for w, pf in CONFIGS:
            rows = [by[(thr, w, pf)] for _, thr in MODES]
            best = max(rows, key=lambda r: r["ips"])
            winners.append(
                {
                    "workers": w,
                    "prefetch": pf,
                    "best_mode": best["label"].rsplit("_w", 1)[0],
                    "ips": best["ips"],
                }
            )

        mode_means = {}
        for mode_name, thr in MODES:
            vals = [r["ips"] for r in results if r["range_parallel_threshold"] == thr]
            mode_means[mode_name] = sum(vals) / len(vals)

        overall_winner = max(mode_means.items(), key=lambda kv: kv[1])[0]
        log("\nMode mean samples/s: " + ", ".join(f"{k}={v:.1f}" for k, v in mode_means.items()))
        log(f"Overall winner (mean across configs): {overall_winner}")

        for w, pf in [(8, 0), (8, 128)]:
            a = by[(0, w, pf)]["ips"]
            c = by[(1, w, pf)]["ips"]
            winner = "force_ranged" if c > a else ("whole_object" if a > c else "tie")
            log(f"w={w} pf={pf}: whole={a:.1f} vs force_ranged={c:.1f} → {winner} ({max(a, c) / min(a, c):.2f}x)")

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
                "uvloop": uvloop_package_status(),
                "note": (
                    "ImageNet val JPEGs are ~50-200KiB; default 32MiB threshold never engages "
                    "ranged GETs. force_ranged uses threshold=1 to exercise the ranged path."
                ),
                "sample_sizes": (
                    {"n": len(sizes), "min": min(sizes), "avg": sum(sizes) // len(sizes), "max": max(sizes)}
                    if sizes
                    else None
                ),
                "old_sweep_killed": True,
                "old_sweep_json": None,
                "old_sweep_log": "benchmarks/results/raw_worker_prefetch_sweep.log",
                "old_sweep_used_fixed_downloaders": False,
            },
            "modes": dict(MODES),
            "results": results,
            "mode_means": mode_means,
            "winners_per_config": winners,
            "overall_winner": overall_winner,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = unique_result_path("raw_ranged_vs_whole", sha=git_sha())
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"Wrote {path}")
    finally:
        wd.stop()


if __name__ == "__main__":
    main()

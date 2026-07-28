"""Decisive cell: after w=24 p=0 download_timeout=0 (true fast path)."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_raw_before_vs_after import (
    OUT_DIR,
    ROOT,
    TIMEOUT,
    HangWatchdog,
    git_sha,
    make_dataset,
    run_one,
    unique_result_path,
)


def main() -> None:
    """Run one long-window after cell with download_timeout=0."""
    side = "after"
    w, pf, dt = 24, 0, 0.0
    side_root = ROOT / "decisive_timeout0"
    if side_root.exists():
        shutil.rmtree(side_root, ignore_errors=True)
    side_root.mkdir(parents=True)
    seed = side_root / "seed"
    sha = git_sha()
    print(f"python={sys.version}", flush=True)
    print(f"sha={sha}", flush=True)
    t0 = time.perf_counter()
    ds = make_dataset(str(seed), side=side, max_prefetch=0, hedge_delay=0.0, download_timeout=dt)
    n = len(ds)
    print(
        f"indexed n={n} in {time.perf_counter() - t0:.2f}s "
        f"timeout={ds.download_timeout!r} hedge={ds.hedge_delay} "
        f"cm.download_timeout={ds.cache_manager.download_timeout!r}",
        flush=True,
    )
    del ds

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    try:
        result = run_one(
            f"w{w}_p{pf}_t{dt}",
            side=side,
            num_workers=w,
            max_prefetch=pf,
            seed=seed,
            wd=wd,
            batches=300,
            min_seconds=30.0,
            prefetch_factor=2,
            hedge_delay=0.0,
            download_timeout=dt,
            sha=sha,
            jsonl=OUT_DIR / "raw_decisive_timeout0.jsonl",  # append-only
        )
    finally:
        wd.stop()

    out = {
        "python": sys.version,
        "git_sha": sha,
        "cell": {"workers": w, "prefetch": pf, "download_timeout": dt, "hedge_delay": 0.0},
        "ips": result["ips"],
        "elapsed": result["elapsed"],
        "batches": result["batches"],
        "samples": result["samples"],
        "compare": {"after_p0_default_timeout120": 4404.219, "main_w24": 6927.318},
        "delta_vs_after_p0_pct": (result["ips"] - 4404.219) / 4404.219 * 100,
        "delta_vs_main_pct": (result["ips"] - 6927.318) / 6927.318 * 100,
        "result": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = unique_result_path("raw_decisive_timeout0", sha=sha, ts=result.get("ts"))
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "result"}, indent=2), flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()

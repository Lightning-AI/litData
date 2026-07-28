"""Focused post-timeout-fix remeasure: after w∈{16,24,32} × p∈{0,16} at defaults."""

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

CELLS = [(16, 0), (16, 16), (24, 0), (24, 16), (32, 0), (32, 16)]
# Shipped defaults: omit download_timeout kwarg → dataset default 120 (batch-level).
DT_DEFAULT = 120.0


def main() -> None:
    """Run high-w after cells with current batch-level timeout defaults."""
    side = "after"
    side_root = ROOT / "highw_post_timeout"
    if side_root.exists():
        shutil.rmtree(side_root, ignore_errors=True)
    side_root.mkdir(parents=True)
    seed = side_root / "seed"
    sha = git_sha()
    run_ts = time.time()
    print(f"python={sys.version}", flush=True)
    print(f"sha={sha}", flush=True)
    t0 = time.perf_counter()
    ds = make_dataset(str(seed), side=side, max_prefetch=0, hedge_delay=0.0)
    print(
        f"indexed n={len(ds)} in {time.perf_counter() - t0:.2f}s "
        f"ds.timeout={ds.download_timeout!r} cm.timeout={ds.cache_manager.download_timeout!r}",
        flush=True,
    )
    del ds

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    results = []
    # Unique JSONL per run (never truncate a prior high-w log).
    jsonl = OUT_DIR / f"raw_highw_post_timeout.{sha or 'unknown'}.{int(run_ts)}.jsonl"
    try:
        for w, pf in CELLS:
            label = f"w{w}_p{pf}_defaults"
            r = run_one(
                label,
                side=side,
                num_workers=w,
                max_prefetch=pf,
                seed=seed,
                wd=wd,
                batches=300,
                min_seconds=30.0,
                prefetch_factor=2,
                hedge_delay=0.0,
                download_timeout=None,  # use shipped default
                sha=sha,
                jsonl=jsonl,
            )
            # Record effective default for clarity in JSON.
            r = {**r, "download_timeout": DT_DEFAULT, "download_timeout_note": "shipped default (batch-level)"}
            results.append(r)
            print(f"DONE {label} ips={r['ips']:.1f}", flush=True)
    finally:
        wd.stop()

    out = {
        "python": sys.version,
        "git_sha": sha,
        "note": "Focused remeasure after batch-level timeout fix; not a full grid resweep.",
        "defaults": {"hedge_delay": 0.0, "download_timeout": DT_DEFAULT, "max_prefetch_cells": [0, 16]},
        "cells": results,
        "by_key": {f"w{r['workers']}_p{r['prefetch']}": round(r["ips"], 1) for r in results},
        "jsonl": str(jsonl),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = unique_result_path("raw_highw_post_timeout", sha=sha, ts=run_ts)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["by_key"], indent=2), flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()

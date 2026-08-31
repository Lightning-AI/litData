#!/usr/bin/env python3
r"""Grid-search obstore stream chunk size × workers on ImageNet epoch 0.

Uses the real ``bench_s3_full_epochs`` cold epoch (wipe cache each cell).
Ranks by epoch-0 images/s and ``t_first_batch`` (unblock read).

Example::

    PYTHON_GIL=0 python -Xgil=0 scripts/bench/bench_obstore_imagenet_grid.py
    PYTHON_GIL=0 python -Xgil=0 scripts/bench/bench_obstore_imagenet_grid.py \\
        --workers 32,48 --chunk-sizes 2,4,8,16 --max-pre-download 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts" / "bench" / "bench_s3_full_epochs.py"


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run_cell(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    """Run bench, stream stdout/stderr live, return (rc, full stdout text)."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    rc = proc.wait()
    return rc, "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", default="32,48")
    p.add_argument(
        "--chunk-sizes",
        default="2,4,8,16",
        help="MiB values for LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB",
    )
    p.add_argument("--max-pre-download", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-cache-size", default="200GB")
    args = p.parse_args()

    workers = _parse_ints(args.workers)
    sizes = _parse_ints(args.chunk_sizes)
    rows: list[dict] = []

    print(
        f"grid workers={workers} chunk_sizes_mib={sizes} max_pre={args.max_pre_download} batch={args.batch_size}",
        flush=True,
    )

    for w in workers:
        for mib in sizes:
            label = f"w{w}-mcs{mib}"
            env = os.environ.copy()
            env["LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB"] = str(mib)
            env["PYTHON_GIL"] = "0"
            cmd = [
                sys.executable,
                "-Xgil=0",
                str(BENCH),
                "--label",
                label,
                "--epochs",
                "1",
                "--workers",
                str(w),
                "--batch-size",
                str(args.batch_size),
                "--max-pre-download",
                str(args.max_pre_download),
                "--max-cache-size",
                args.max_cache_size,
            ]
            print(f"\n===== RUN {label} =====", flush=True)
            rc, out = _run_cell(cmd, env)
            if rc != 0:
                print(f"FAIL {label} exit={rc}", flush=True)
                rows.append({"workers": w, "min_chunk_mib": mib, "ok": False, "returncode": rc})
                continue

            summary = None
            for line in out.splitlines():
                if line.startswith("SUMMARY "):
                    summary = json.loads(line[len("SUMMARY ") :])
            if not summary:
                print(f"FAIL {label}: no SUMMARY line", flush=True)
                rows.append({"workers": w, "min_chunk_mib": mib, "ok": False})
                continue

            ep0 = summary["epochs"][0]
            row = {
                "workers": w,
                "min_chunk_mib": mib,
                "ok": True,
                "images_per_s": ep0["images_per_s"],
                "elapsed_s": ep0["elapsed_s"],
                "t_first_batch_s": ep0.get("t_first_batch_s"),
                "label": label,
            }
            rows.append(row)
            print(
                f"CELL {label}: ips={row['images_per_s']:.1f} "
                f"t_first_batch={row['t_first_batch_s']:.3f}s "
                f"elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )

    ok = [r for r in rows if r.get("ok")]
    print("\n===== GRID TABLE (epoch 0) =====", flush=True)
    print(f"{'workers':>8} {'mcs_MiB':>8} {'ips':>10} {'t_first':>9} {'elapsed':>9}", flush=True)
    for r in ok:
        print(
            f"{r['workers']:8d} {r['min_chunk_mib']:8d} {r['images_per_s']:10.1f} "
            f"{r['t_first_batch_s']:9.3f} {r['elapsed_s']:9.1f}",
            flush=True,
        )

    if ok:
        by_ips = max(ok, key=lambda r: r["images_per_s"])
        by_ttfb = min(ok, key=lambda r: r["t_first_batch_s"])
        print(
            f"\nbest_ips: w={by_ips['workers']} mcs={by_ips['min_chunk_mib']}MiB "
            f"ips={by_ips['images_per_s']:.1f} t_first={by_ips['t_first_batch_s']:.3f}s",
            flush=True,
        )
        print(
            f"best_t_first_batch: w={by_ttfb['workers']} mcs={by_ttfb['min_chunk_mib']}MiB "
            f"ips={by_ttfb['images_per_s']:.1f} t_first={by_ttfb['t_first_batch_s']:.3f}s",
            flush=True,
        )

    print("\n=== JSON ===", flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

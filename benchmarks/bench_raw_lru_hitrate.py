"""One-shot LRU hit/miss at w=8 p16 with LITDATA_RAW_DEBUG=1.

Parses worker ``raw-debug: _download_batch done ... total_hit=... total_miss=...``
lines (spawn workers log via the root logger to stderr).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["LITDATA_RAW_DEBUG"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_raw_before_vs_after import git_sha, unique_result_path
from torch.utils.data import DataLoader

from litdata import StreamingRawDataset
from litdata.raw import dataset as raw_dataset

raw_dataset._RAW_DEBUG = True

INPUT = "/teamspace/s3_connections/imagenet-1m-template/raw/val"
OUT_DIR = Path(__file__).resolve().parent / "results"
ROOT = Path(tempfile.gettempdir()) / "litdata-raw-lru-hitrate"
LOG = OUT_DIR / "raw_lru_hitrate.log"
DONE_RE = re.compile(
    r"raw-debug: _download_batch done pid=(\d+) n=(\d+) inflight=(\d+) "
    r"batch_hit=(\d+) batch_miss=(\d+) total_hit=(\d+) total_miss=(\d+)"
)


class _Tee(logging.Handler):
    """Append warning+ records to a shared log file."""

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.WARNING)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def main() -> None:
    """Run w=8 p16 and report per-worker prefetch hit rates from debug logs."""
    if ROOT.exists():
        shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    if LOG.exists():
        LOG.unlink()

    # File handler so spawn workers that reconfigure logging still write somewhere
    # when LITDATA_RAW_DEBUG is on; also tee parent stderr.
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh = _Tee(LOG)
    fh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    # Ensure litdata.raw.dataset logger propagates.
    logging.getLogger("litdata.raw.dataset").setLevel(logging.WARNING)

    print(f"python={sys.version}", flush=True)
    cache = ROOT / "cache"
    t0 = time.perf_counter()
    ds = StreamingRawDataset(
        INPUT,
        cache_dir=str(cache),
        cache_files=False,
        max_prefetch=16,
        hedge_delay=0,
        download_timeout=120,
        max_concurrent_downloads=64,
        range_parallel_threshold=0,
    )
    print(f"indexed n={len(ds)} in {time.perf_counter() - t0:.2f}s", flush=True)
    # Force dataset module debug flag in children via env (already set).
    loader = DataLoader(
        ds,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        multiprocessing_context="spawn",
        persistent_workers=True,
        prefetch_factor=2,
    )
    it = iter(loader)
    warm = max(1, 8 * 2)
    for i in range(warm):
        next(it)
        print(f"warm {i + 1}/{warm}", flush=True)
    t0 = time.perf_counter()
    samples = 0
    batches = 0
    for _ in range(80):
        batch = next(it)
        samples += len(batch)
        batches += 1
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed else 0.0
    print(f"timed {batches} batches → {ips:.1f} samples/s", flush=True)
    del it, loader

    # Allow workers a moment to flush final debug lines.
    time.sleep(1.0)

    # Also scrape stderr-captured file; workers may only print to their stderr
    # which DataLoader forwards — capture by re-reading if we teed via a wrapper.
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    # Fallback: some environments only forward worker stderr to our process stderr,
    # which we did not tee. Re-run a tiny in-process sequential prefetch check too.
    per_pid: dict[str, dict[str, int]] = {}
    for m in DONE_RE.finditer(text):
        pid, _n, _inf, _bh, _bm, th, tm = m.groups()
        per_pid[pid] = {"total_hit": int(th), "total_miss": int(tm)}

    # In-process sequential: worker-aware schedule uses num_workers=1 → full max_prefetch.
    ds_seq = StreamingRawDataset(
        INPUT,
        cache_dir=str(ROOT / "seq"),
        cache_files=False,
        max_prefetch=16,
        hedge_delay=0,
        download_timeout=120,
        max_concurrent_downloads=64,
        range_parallel_threshold=0,
    )
    idx = cache / "index.json.zstd"
    if idx.exists():
        (ROOT / "seq").mkdir(parents=True, exist_ok=True)
        shutil.copy2(idx, ROOT / "seq" / "index.json.zstd")
        ds_seq = StreamingRawDataset(
            INPUT,
            cache_dir=str(ROOT / "seq"),
            cache_files=False,
            max_prefetch=16,
            hedge_delay=0,
            download_timeout=120,
            max_concurrent_downloads=64,
            range_parallel_threshold=0,
        )
    bs = 64
    for start in range(0, bs * 30, bs):
        ds_seq.__getitems__(list(range(start, start + bs)))
    seq_hits, seq_misses = ds_seq._prefetch_hits, ds_seq._prefetch_misses
    seq_total = seq_hits + seq_misses

    # Simulate w=8 stride in-process: each "worker" advances by 8 batches.
    ds_w = StreamingRawDataset(
        INPUT,
        cache_dir=str(ROOT / "w8sim"),
        cache_files=False,
        max_prefetch=16,
        hedge_delay=0,
        download_timeout=120,
        max_concurrent_downloads=64,
        range_parallel_threshold=0,
    )
    if idx.exists():
        (ROOT / "w8sim").mkdir(parents=True, exist_ok=True)
        shutil.copy2(idx, ROOT / "w8sim" / "index.json.zstd")
        ds_w = StreamingRawDataset(
            INPUT,
            cache_dir=str(ROOT / "w8sim"),
            cache_files=False,
            max_prefetch=16,
            hedge_delay=0,
            download_timeout=120,
            max_concurrent_downloads=64,
            range_parallel_threshold=0,
        )

    # Monkeypatch get_worker_info so _schedule_prefetch thinks num_workers=8.
    class _Info:
        num_workers = 8
        id = 0

    real_schedule = ds_w._schedule_prefetch

    def schedule_as_w8(indices: list[int]) -> None:
        import torch.utils.data

        real_get = torch.utils.data.get_worker_info
        torch.utils.data.get_worker_info = lambda: _Info()  # type: ignore[assignment]
        try:
            real_schedule(indices)
        finally:
            torch.utils.data.get_worker_info = real_get

    ds_w._schedule_prefetch = schedule_as_w8  # type: ignore[method-assign]
    # Worker 0 batches: 0, 8, 16, ... in batch-index space → sample starts 0, 512, 1024, ...
    for bi in range(0, 40, 8):
        start = bi * bs
        ds_w.__getitems__(list(range(start, start + bs)))
    w_hits, w_misses = ds_w._prefetch_hits, ds_w._prefetch_misses
    w_total = w_hits + w_misses

    worker_hits = sum(v["total_hit"] for v in per_pid.values())
    worker_misses = sum(v["total_miss"] for v in per_pid.values())
    worker_total = worker_hits + worker_misses

    out = {
        "python": sys.version,
        "cell": {"workers": 8, "max_prefetch": 16, "effective_prefetch": min(16, 64 // 8)},
        "timed": {"batches": batches, "ips": ips, "elapsed": elapsed},
        "spawn_workers_from_log": {
            "pids": len(per_pid),
            "total_hit": worker_hits,
            "total_miss": worker_misses,
            "hit_rate": (worker_hits / worker_total) if worker_total else None,
            "per_pid": per_pid,
            "log_lines": text.count("raw-debug: _download_batch done"),
            "note": "None hit_rate means spawn workers did not share the parent log file",
        },
        "in_process_sequential_w1": {
            "prefetch_hits": seq_hits,
            "prefetch_misses": seq_misses,
            "hit_rate": (seq_hits / seq_total) if seq_total else 0.0,
        },
        "in_process_simulated_w8": {
            "note": "stride every 8th batch; effective look-ahead=8",
            "prefetch_hits": w_hits,
            "prefetch_misses": w_misses,
            "hit_rate": (w_hits / w_total) if w_total else 0.0,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out["git_sha"] = git_sha()
    path = unique_result_path("raw_lru_hitrate", sha=out["git_sha"])
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()

r"""A/B: stock StreamingRawDataset (main) vs optimized (feature branch).

Run twice with different PYTHONPATH / --side, then merge:

  PYTHONPATH=/tmp/litdata-raw-before/src \
    python benchmarks/bench_raw_before_vs_after.py --side before

  PYTHONPATH=src \
    python benchmarks/bench_raw_before_vs_after.py --side after

  python benchmarks/bench_raw_before_vs_after.py --merge

Protocol (trustworthy windows)
------------------------------
- Timed window: continue until **both** ``--batches`` **and** ``--min-seconds``
  are met (``max(N batches, T seconds)``). Defaults: 300 batches and 30s.
  High-worker cells (``num_workers >= 16``) always enforce ≥30s.
- Repeats: ``--repeats N`` (use ≥5 for publish). Each cell stores all runs;
  merge reports **median** ips + min/max spread.
- Interleave: ``--interleave --before-pythonpath PATH`` alternates
  before/after per cell (main, head, main, head, …) via subprocesses.
- Artifacts: append-only SHA/ts JSON (+ JSONL); never overwrite prior results.

After measures prefetch in ``[0, 16, 32]`` (publish ≥16; p0 kept in JSON).
Warm ``max(1, num_workers * prefetch_factor)`` batches before timing starts.
"""

from __future__ import annotations

import argparse
import contextlib
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
OUT = OUT_DIR / "raw_before_vs_after.json"  # legacy fixed name; writers use unique_result_path
BS = 64
DEFAULT_BATCHES = 300
DEFAULT_MIN_SECONDS = 30.0
HIGH_WORKER_MIN_SECONDS = 30.0
HIGH_WORKER_THRESHOLD = 16
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_REPEATS = 1
WORKERS = [0, 1, 2, 4, 8, 16, 24, 32]
TRUST_WORKERS = [0, 2, 4, 8, 16]
AFTER_PREFETCH = [0, 16, 32]
TIMEOUT = 600.0
OLD_FUSE = 75.2
# Overridable via --after-prefetch (also used for before when it has max_prefetch).
_PREFETCH_LEVELS: list[int] = list(AFTER_PREFETCH)


def effective_min_seconds(num_workers: int, min_seconds: float) -> float:
    """Enforce ≥30s timed windows at high worker counts."""
    if num_workers >= HIGH_WORKER_THRESHOLD:
        return max(min_seconds, HIGH_WORKER_MIN_SECONDS)
    return min_seconds


def summarize_ips(values: list[float]) -> dict:
    """Return median + spread stats for a list of samples/s measurements."""
    if not values:
        return {"ips_median": None, "ips_min": None, "ips_max": None, "ips_spread_pct": None, "n": 0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    lo, hi = ordered[0], ordered[-1]
    spread = ((hi - lo) / median) * 100.0 if median else None
    return {
        "ips_median": median,
        "ips_min": lo,
        "ips_max": hi,
        "ips_spread_pct": spread,
        "n": len(values),
    }


def git_sha(*, cwd: Path | None = None) -> str:
    """Return short git SHA for ``cwd`` (default: repo containing this script).

    ``LITDATA_BENCH_GIT`` overrides only when ``cwd`` is omitted (runner/artifact
    naming). Prefer :func:`tree_git_sha` for before/after PYTHONPATH provenance.
    """
    if cwd is None:
        env = os.environ.get("LITDATA_BENCH_GIT", "").strip()
        if env:
            return env
        cwd = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def tree_git_sha(pythonpath: str) -> str:
    """Return short SHA for the git checkout that owns a PYTHONPATH entry."""
    raw = (pythonpath or "").strip()
    if not raw:
        return ""
    # First path entry wins (same as import resolution for litdata).
    entry = Path(raw.split(os.pathsep)[0]).resolve()
    # .../src → repo root; already-repo-root → itself.
    candidates = [entry, entry.parent if entry.name == "src" else entry]
    for cand in candidates:
        sha = git_sha(cwd=cand)
        if sha:
            return sha
    return ""


def pythonpath_tree_sha() -> str:
    """SHA of the litdata tree currently first on ``sys.path`` / PYTHONPATH."""
    env_pp = os.environ.get("PYTHONPATH", "")
    if env_pp.strip():
        return tree_git_sha(env_pp)
    # Fall back: package file location.
    try:
        import litdata

        pkg = Path(litdata.__file__).resolve()
        # litdata/__init__.py → src/litdata → src → repo
        return git_sha(cwd=pkg.parents[2]) or git_sha(cwd=pkg.parents[1])
    except Exception:
        return ""


def unique_result_path(stem: str, *, sha: str | None = None, ts: float | None = None) -> Path:
    """Return ``OUT_DIR/{stem}.{sha}.{ts}.json`` — never overwrites a prior result file."""
    sha_part = (sha if sha is not None else git_sha()) or "unknown"
    ts_part = int(ts if ts is not None else time.time())
    path = OUT_DIR / f"{stem}.{sha_part}.{ts_part}.json"
    if path.exists():
        # Same-second collision: bump until free.
        n = 1
        while True:
            alt = OUT_DIR / f"{stem}.{sha_part}.{ts_part}.{n}.json"
            if not alt.exists():
                return alt
            n += 1
    return path


def input_for(side: str, *, before_cloud_native: bool = False) -> str:
    """Return dataset input path.

    Stock main ``before`` needs ``s3://`` (FUSE path→LocalDownloader lacks
    ``adownload_fileobj``). Pre-Stage-1 / cloud-native ``before`` trees share
    mount→s3 remapping with ``after``, so both use the mount for a fair A/B.
    """
    if side == "after" or before_cloud_native:
        return MOUNT_INPUT
    return S3_INPUT


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
    before_cloud_native: bool = False,
):
    """Construct StreamingRawDataset with side-appropriate kwargs."""
    import inspect

    from litdata import StreamingRawDataset

    params = set(inspect.signature(StreamingRawDataset.__init__).parameters)
    kwargs: dict = {
        "cache_dir": cache,
        "cache_files": False,
        "input_dir": input_for(side, before_cloud_native=before_cloud_native),
    }
    if "max_prefetch" in params:
        kwargs["max_prefetch"] = max_prefetch
    if side == "after":
        # Default None → Stage 1 adaptive permits (do not pass 64: that bypasses clamp).
        if "range_parallel_threshold" in params:
            kwargs["range_parallel_threshold"] = 0
        # Match new defaults: hedging opt-in (0). Explicit for older trees / clarity.
        if "hedge_delay" in params:
            kwargs["hedge_delay"] = 0.0 if hedge_delay is None else hedge_delay
        if download_timeout is not None and "download_timeout" in params:
            kwargs["download_timeout"] = download_timeout
    elif before_cloud_native:
        # Pre-Stage-1 baseline: fixed 64 permits/worker (no adaptive clamp).
        if "max_concurrent_downloads" in params:
            kwargs["max_concurrent_downloads"] = 64
        if "range_parallel_threshold" in params:
            kwargs["range_parallel_threshold"] = 0
        if "hedge_delay" in params:
            kwargs["hedge_delay"] = 0.0 if hedge_delay is None else hedge_delay
        if download_timeout is not None and "download_timeout" in params:
            kwargs["download_timeout"] = download_timeout
    return StreamingRawDataset(**kwargs)


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
    repeat: int = 0,
    before_cloud_native: bool = False,
) -> dict:
    """Run one worker/prefetch trial and return timing stats.

    Timing stops only when **both** ``batches`` and the effective min-seconds
    floor are met (``max(N batches, T seconds)``).
    """
    cache = ROOT / side / f"{label}_r{repeat}"
    wd.beat(f"{label}:r{repeat} setup")
    copy_index(seed, cache)
    ds = make_dataset(
        str(cache),
        side=side,
        max_prefetch=max_prefetch,
        hedge_delay=hedge_delay,
        download_timeout=download_timeout,
        before_cloud_native=before_cloud_native,
    )
    kwargs: dict = {"batch_size": BS, "num_workers": num_workers, "shuffle": False}
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **kwargs)
    it = iter(loader)

    # Drain pipeline buffer before timing (≥ workers×prefetch_factor).
    warm_batches = max(1, num_workers * prefetch_factor if num_workers > 0 else 1)
    wd.beat(f"{label}:r{repeat} warm({warm_batches})")
    t0 = time.perf_counter()
    for i in range(warm_batches):
        try:
            next(it)
        except StopIteration:
            # High ips × min_seconds can exceed one epoch (50k/64 ≈ 782 batches).
            it = iter(loader)
            next(it)
        wd.beat(f"{label}:r{repeat} warm {i + 1}/{warm_batches}")
    warm_s = time.perf_counter() - t0

    min_s = effective_min_seconds(num_workers, min_seconds)
    samples = 0
    timed_batches = 0
    wd.beat(f"{label}:r{repeat} timed")
    t0 = time.perf_counter()
    while True:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        samples += len(batch)
        timed_batches += 1
        wd.beat(f"{label}:r{repeat} batch {timed_batches}")
        elapsed = time.perf_counter() - t0
        # max(N batches, T seconds): require both floors (not either/or).
        if timed_batches >= batches and elapsed >= min_s:
            break
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed else 0.0
    log(
        f"[{side}/{label}] r={repeat} w={num_workers} pf={max_prefetch} "
        f"warm={warm_batches}@{warm_s:.2f}s | {timed_batches}×{samples // max(timed_batches, 1)} "
        f"in {elapsed:.2f}s (need ≥{batches} batches & ≥{min_s:.0f}s) → {ips:.1f} samples/s"
    )
    tree_sha = pythonpath_tree_sha()
    result = {
        "side": side,
        "label": label,
        "repeat": repeat,
        "workers": num_workers,
        "prefetch": max_prefetch,
        "ips": ips,
        "warm_s": warm_s,
        "warm_batches": warm_batches,
        "elapsed": elapsed,
        "samples": samples,
        "batches": timed_batches,
        "min_seconds_effective": min_s,
        "hedge_delay": hedge_delay if side == "after" else None,
        "download_timeout": download_timeout if side == "after" else None,
        "git_sha": sha,  # runner / artifact naming (may be LITDATA_BENCH_GIT)
        "tree_sha": tree_sha,  # actual litdata checkout on PYTHONPATH
        "before_sha": tree_sha if side == "before" else None,
        "after_sha": tree_sha if side == "after" else None,
        "ts": time.time(),
    }
    if jsonl is not None:
        append_jsonl(jsonl, result)
    del it, loader, ds
    _reap_zombie_children()
    return result


def _reap_zombie_children() -> None:
    """Best-effort reap of leftover DataLoader worker zombies."""
    with contextlib.suppress(Exception):
        import multiprocessing as mp

        for p in mp.active_children():
            with contextlib.suppress(Exception):
                p.join(timeout=2.0)
            if p.is_alive():
                with contextlib.suppress(Exception):
                    p.kill()
                with contextlib.suppress(Exception):
                    p.join(timeout=1.0)
    # Non-blocking waitpid sweep for any unreaped children.
    with contextlib.suppress(ChildProcessError, Exception):
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid <= 0:
                break


def configs_for(
    side: str,
    workers: list[int],
    *,
    safety_grid: bool,
    before_cloud_native: bool = False,
    prefetch_levels: list[int] | None = None,
) -> list[tuple]:
    """Return trial configs.

    Normal: (workers, prefetch, hedge_delay|None, download_timeout|None)
    safety_grid (after only): hedge_delay × download_timeout at p0 for w∈{2,4,8}.
    Pre-Stage-1 ``before`` (cloud-native) sweeps the same prefetch levels as after.
    """
    levels = list(prefetch_levels if prefetch_levels is not None else _PREFETCH_LEVELS)
    if safety_grid:
        if side != "after":
            raise SystemExit("--safety-grid is only meaningful with --side after")
        out = []
        for w in (2, 4, 8):
            for hd in (0.0, 1.0):
                for dt in (0.0, 120.0):
                    out.append((w, 0, hd, dt))
        return out
    if side == "before" and not before_cloud_native:
        return [(w, 0, None, None) for w in workers]
    return [(w, pf, 0.0, None) for w in workers for pf in levels]


def partial_path(side: str, *, sha: str | None = None, ts: float | None = None) -> Path:
    """Return a unique path for a side's partial JSON payload (never overwrites)."""
    return unique_result_path(f"raw_before_vs_after.{side}", sha=sha, ts=ts)


def jsonl_path(side: str, *, sha: str | None = None, ts: float | None = None) -> Path:
    """Return a unique path for a side's incremental JSONL log (never overwrites)."""
    sha_part = (sha if sha is not None else git_sha()) or "unknown"
    ts_part = int(ts if ts is not None else time.time())
    return OUT_DIR / f"raw_before_vs_after.{side}.{sha_part}.{ts_part}.jsonl"


def latest_partial(side: str) -> Path:
    """Resolve the newest unique partial for ``side``, falling back to the legacy fixed name."""
    matches = sorted(OUT_DIR.glob(f"raw_before_vs_after.{side}.*.json"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    legacy = OUT_DIR / f"raw_before_vs_after.{side}.json"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"no raw_before_vs_after.{side}.* result under {OUT_DIR}")


def cell_summaries(results: list[dict]) -> dict[str, dict]:
    """Group raw runs by label and attach median/spread ips."""
    by_label: dict[str, list[dict]] = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)
    out: dict[str, dict] = {}
    for label, runs in by_label.items():
        stats = summarize_ips([float(r["ips"]) for r in runs])
        head = runs[0]
        out[label] = {
            "label": label,
            "workers": head["workers"],
            "prefetch": head["prefetch"],
            "ips": stats["ips_median"],
            **stats,
            "runs": runs,
        }
    return out


def run_side(
    side: str,
    *,
    workers: list[int],
    batches: int,
    min_seconds: float,
    prefetch_factor: int,
    safety_grid: bool,
    repeats: int = 1,
    prefetch_levels: list[int] | None = None,
) -> None:
    """Index once and sweep configs for ``before`` or ``after``."""
    caps = detect_side_capabilities()
    if side == "after" and not caps["has_max_prefetch"]:
        raise SystemExit("PYTHONPATH points at main tree but --side after requested")
    # Stock main has no max_prefetch. Pre-Stage-1 feature trees do — allow them as
    # a cloud-native baseline for Stage 1 clamp A/B (fixed 64 vs adaptive).
    before_cloud_native = bool(side == "before" and caps["has_max_prefetch"])
    if before_cloud_native:
        log(
            "before tree has max_prefetch/LoopRunner — treating as pre-Stage-1 "
            "baseline (fixed max_concurrent_downloads=64), not stock main"
        )

    side_root = ROOT / side
    if side_root.exists():
        shutil.rmtree(side_root, ignore_errors=True)
    side_root.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha = git_sha()  # runner / artifact filename (LITDATA_BENCH_GIT or script repo)
    tree_sha = pythonpath_tree_sha()
    run_ts = time.time()
    jpath = jsonl_path(side, sha=sha, ts=run_ts)
    levels = list(prefetch_levels if prefetch_levels is not None else _PREFETCH_LEVELS)

    wd = HangWatchdog(TIMEOUT)
    wd.start()
    ncpu = os.cpu_count() or 0
    cfgs = configs_for(
        side,
        workers,
        safety_grid=safety_grid,
        before_cloud_native=before_cloud_native,
        prefetch_levels=levels,
    )
    inp = input_for(side, before_cloud_native=before_cloud_native)
    log(f"=== side={side} ===")
    log(f"capabilities: {json.dumps(caps)}")
    log(
        f"provenance: side={side} tree_sha={tree_sha or '?'} "
        f"runner_sha={sha or '?'} PYTHONPATH={os.environ.get('PYTHONPATH', '')!r}"
    )
    log(
        f"input={inp} (mount={MOUNT_INPUT}) bs={BS} batches>={batches} "
        f"min_seconds>={min_seconds} (high-w≥{HIGH_WORKER_THRESHOLD} → "
        f"≥{HIGH_WORKER_MIN_SECONDS}s) warm=max(1,w*{prefetch_factor}) "
        f"repeats={repeats} cpus={ncpu} configs={len(cfgs)} "
        f"prefetch_levels={levels if side == 'after' or before_cloud_native else [0]}"
    )
    log("protocol: stop when batches AND min_seconds both met (max window)")
    log(f"PYTHONPATH[0]={sys.path[0]!r}")

    try:
        wd.beat("index seed")
        seed = side_root / "seed"
        t0 = time.perf_counter()
        ds = make_dataset(
            str(seed),
            side=side,
            max_prefetch=0,
            before_cloud_native=before_cloud_native,
        )
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
        # Interleave repeats across configs when repeats>1 so A/B noise is
        # comparable; for a single side this is config-major then repeat.
        for rep in range(max(1, repeats)):
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
                        repeat=rep,
                        before_cloud_native=before_cloud_native,
                    )
                )

        accum = os.environ.get("LITDATA_BENCH_ACCUM_OUT", "").strip()
        if accum:
            out = Path(accum)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                prev = json.loads(out.read_text())
                results = list(prev.get("results") or []) + results
                # Renumber repeats so summaries see a contiguous series.
                by_label: dict[str, int] = {}
                for r in results:
                    lab = r["label"]
                    r["repeat"] = by_label.get(lab, 0)
                    by_label[lab] = r["repeat"] + 1
        else:
            out = partial_path(side, sha=sha, ts=run_ts)

        summaries = cell_summaries(results)
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
                "high_worker_min_seconds": HIGH_WORKER_MIN_SECONDS,
                "high_worker_threshold": HIGH_WORKER_THRESHOLD,
                "timing_window": "max(batches, min_seconds) — both floors required",
                "repeats": max(r.get("repeat", 0) for r in results) + 1 if results else max(1, repeats),
                "prefetch_factor": prefetch_factor,
                "warm_batches_formula": "max(1, num_workers * prefetch_factor)",
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "cpus": ncpu,
                "fuse_baseline_samples_per_s": OLD_FUSE,
                "workers": workers,
                "prefetch": (list(levels) if side == "after" or before_cloud_native else [0]),
                "range_parallel_threshold": (0 if side == "after" or before_cloud_native else None),
                "max_concurrent_downloads": (None if side == "after" else (64 if before_cloud_native else None)),
                "hedge_delay": (0.0 if side == "after" or before_cloud_native else None),
                "before_cloud_native": before_cloud_native,
                "safety_grid": safety_grid,
                "capabilities": caps,
                "git_sha": sha,
                "tree_sha": tree_sha,
                "before_sha": tree_sha if side == "before" else None,
                "after_sha": tree_sha if side == "after" else None,
                "git_hint": os.environ.get("LITDATA_BENCH_GIT", ""),
                "jsonl": str(jpath),
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "sys_path0": sys.path[0] if sys.path else "",
                "input_note": (
                    "pre-Stage-1 before: mount→s3:// like after; fixed max_concurrent_downloads=64"
                    if before_cloud_native
                    else (
                        "before uses s3:// directly: main prefers FUSE path→LocalDownloader "
                        "which lacks adownload_fileobj; after uses mount and remaps to s3://"
                        if side == "before"
                        else "after uses mount path; Stage 1 adaptive concurrency (None); hedge_delay=0"
                    )
                ),
                "caveat": (
                    "Use --repeats ≥5 and medians for publish claims. "
                    "Trust systematic patterns, not single-run fine Δ%."
                ),
            },
            "results": results,
            "summaries": summaries,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"Wrote {out}")
        log(f"JSONL {jpath}")
        for label, s in summaries.items():
            spread = s.get("ips_spread_pct")
            spread_s = f" spread={spread:.1f}%" if isinstance(spread, (int, float)) else ""
            log(f"  summary {label}: median={s['ips']:.1f} ips n={s['n']}{spread_s}")
    finally:
        wd.stop()


def run_interleaved(
    *,
    before_pythonpath: str,
    workers: list[int],
    batches: int,
    min_seconds: float,
    prefetch_factor: int,
    repeats: int,
    prefetch_levels: list[int] | None = None,
) -> None:
    """Alternate before/after subprocesses (main, head, main, head, …) into one partial each."""
    script = str(Path(__file__).resolve())
    after_pythonpath = os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    n_rep = max(1, repeats)
    sha = git_sha()  # runner / artifact filenames
    before_sha = tree_git_sha(before_pythonpath)
    after_sha = tree_git_sha(after_pythonpath)
    run_ts = time.time()
    levels = list(prefetch_levels if prefetch_levels is not None else _PREFETCH_LEVELS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs = {
        "before": partial_path("before", sha=sha, ts=run_ts),
        "after": partial_path("after", sha=sha, ts=run_ts),
    }
    log(f"=== interleaved A/B repeats={n_rep} workers={workers} batches>={batches} min_seconds>={min_seconds} ===")
    log(f"provenance: before_sha={before_sha or '?'} after_sha={after_sha or '?'} runner_sha={sha or '?'}")
    log(f"before PYTHONPATH={before_pythonpath} → {outs['before'].name}")
    log(f"after  PYTHONPATH={after_pythonpath} → {outs['after'].name}")
    log(f"prefetch_levels={levels}")
    if not before_sha or not after_sha:
        log("WARNING: could not resolve before_sha/after_sha — do not publish without provenance")
    for rep in range(n_rep):
        for side, pypath in (("before", before_pythonpath), ("after", after_pythonpath)):
            env = os.environ.copy()
            env["PYTHONPATH"] = pypath
            env["LITDATA_BENCH_GIT"] = sha or env.get("LITDATA_BENCH_GIT", "")
            env["LITDATA_BENCH_ACCUM_OUT"] = str(outs[side])
            cmd = [
                sys.executable,
                script,
                "--side",
                side,
                "--workers",
                ",".join(str(w) for w in workers),
                "--batches",
                str(batches),
                "--min-seconds",
                str(min_seconds),
                "--prefetch-factor",
                str(prefetch_factor),
                "--repeats",
                "1",
                "--after-prefetch",
                ",".join(str(p) for p in levels),
            ]
            log(f"interleave rep={rep} side={side}: {' '.join(cmd)}")
            subprocess.check_call(cmd, env=env)  # noqa: S603
    log(f"interleave complete — merge with: python {script} --merge")
    log(f"  before={outs['before']}")
    log(f"  after={outs['after']}")


def _representative_runs(payload: dict) -> list[dict]:
    """Prefer per-cell median summaries; fall back to raw single runs."""
    summaries = payload.get("summaries") or {}
    if summaries:
        return list(summaries.values())
    # Build summaries from raw results when older partials lack them.
    return list(cell_summaries(payload.get("results") or {}).values())


def merge() -> None:
    """Merge before/after partial JSON into the comparison artifact."""
    before_path = latest_partial("before")
    after_path = latest_partial("after")
    log(f"merge before={before_path.name} after={after_path.name}")
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

    before_reps = _representative_runs(before)
    after_reps = _representative_runs(after)

    workers = sorted({r["workers"] for r in before_reps} | {r["workers"] for r in after_reps})
    # Pair same-(w, prefetch) when before swept prefetch (pre-Stage-1 A/B);
    # else fall back to before@p0 vs each after prefetch (stock-main A/B).
    before_by_wp: dict[tuple[int, int], dict] = {}
    for r in before_reps:
        before_by_wp[(r["workers"], r["prefetch"])] = r
    before_by_w_p0 = {r["workers"]: r for r in before_reps if r["prefetch"] == 0}
    after_by_pf: dict[int, dict[int, dict]] = {}
    for r in after_reps:
        after_by_pf.setdefault(r["prefetch"], {})[r["workers"]] = r
    prefetch_levels = sorted(after_by_pf)
    paired_prefetch = any(pf != 0 for _, pf in before_by_wp)

    rows = []
    cells = []
    for w in workers:
        b0 = before_by_w_p0.get(w)
        row: dict = {
            "workers": w,
            "before_ips": b0["ips"] if b0 else None,
            "before_n": b0.get("n") if b0 else None,
            "before_spread_pct": b0.get("ips_spread_pct") if b0 else None,
        }
        after_best = None
        for pf in prefetch_levels:
            a = after_by_pf.get(pf, {}).get(w)
            b = before_by_wp.get((w, pf)) if paired_prefetch else b0
            row[f"after_prefetch{pf}_ips"] = a["ips"] if a else None
            row[f"after_prefetch{pf}_spread_pct"] = a.get("ips_spread_pct") if a else None
            if b and a and b["ips"] and a["ips"]:
                row[f"speedup_prefetch{pf}"] = a["ips"] / b["ips"]
                row[f"delta_pct_prefetch{pf}"] = ((a["ips"] - b["ips"]) / b["ips"]) * 100.0
            if a and (after_best is None or (a["ips"] or 0) > (after_best["ips"] or 0)):
                after_best = a
        b_best = before_by_wp.get((w, after_best["prefetch"])) if after_best and paired_prefetch else b0
        if b_best and after_best and b_best["ips"] and after_best["ips"]:
            row["after_best_ips"] = after_best["ips"]
            row["after_best_prefetch"] = after_best["prefetch"]
            row["speedup_best"] = after_best["ips"] / b_best["ips"]
            row["delta_pct_best"] = ((after_best["ips"] - b_best["ips"]) / b_best["ips"]) * 100.0
        rows.append(row)
        for pf in prefetch_levels:
            a = after_by_pf.get(pf, {}).get(w)
            b = before_by_wp.get((w, pf)) if paired_prefetch else b0
            if a is None or b is None:
                continue  # omit missing/crashed
            cells.append(
                {
                    "workers": w,
                    "prefetch": pf,
                    "before_ips": b["ips"],
                    "after_ips": a["ips"],
                    "before_n": b.get("n"),
                    "after_n": a.get("n"),
                    "before_spread_pct": b.get("ips_spread_pct"),
                    "after_spread_pct": a.get("ips_spread_pct"),
                    "delta_pct": ((a["ips"] - b["ips"]) / b["ips"]) * 100.0 if b["ips"] and a["ips"] else None,
                    "speedup": a["ips"] / b["ips"] if b["ips"] and a["ips"] else None,
                    "before_elapsed": b.get("elapsed"),
                    "after_elapsed": a.get("elapsed"),
                    "before_batches": b.get("batches"),
                    "after_batches": a.get("batches"),
                }
            )

    best_after = max(cells, key=lambda c: c["after_ips"] or 0) if cells else None
    before_sha = (before.get("meta") or {}).get("before_sha") or (before.get("meta") or {}).get("tree_sha")
    after_sha = (after.get("meta") or {}).get("after_sha") or (after.get("meta") or {}).get("tree_sha")
    payload = {
        "meta": {
            "mount_input": MOUNT_INPUT,
            "batch_size": BS,
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "workers": workers,
            "before": before["meta"],
            "after": after["meta"],
            "before_sha": before_sha,
            "after_sha": after_sha,
            "delta_definition": (
                "delta_pct = ((after_median - before_median) / before_median) * 100; "
                "paired same-(w,prefetch) when before swept prefetch (pre-Stage-1 A/B); "
                "else before@p0 vs each after prefetch (stock-main A/B)"
            ),
            "paired_prefetch": paired_prefetch,
            "note": (
                "before = pre-Stage-1 feature tree (fixed max_concurrent_downloads=64) when "
                "before_cloud_native; else stock main via s3://. after = Stage 1 adaptive "
                "(max_concurrent_downloads=None). Both cloud-native arms use mount→s3://. "
                "Publish with proven before_sha/after_sha from tree rev-parse — not runner SHA alone."
            ),
            "caveat": (
                "Protocol: max(≥300 batches, ≥30s) timed window; prefer --repeats ≥5 with "
                "median ips + spread. Trust systematic patterns over single-run fine Δ%."
            ),
            "default_max_prefetch": 16,
            "publish_prefetch": [pf for pf in prefetch_levels if pf >= 16],
            "ips_aggregation": "median across repeats when summaries present",
        },
        "cells": cells,
        "best_after": best_after,
        "comparison": rows,
        "before_results": before["results"],
        "after_results": after["results"],
        "before_summaries": before.get("summaries"),
        "after_summaries": after.get("summaries"),
        "sources": {"before": str(before_path), "after": str(after_path)},
    }
    out = unique_result_path("raw_before_vs_after")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {out}")

    publish = [c for c in cells if c["prefetch"] >= 16]
    print()
    print("Published matrix (prefetch ≥ 16; before paired by prefetch when available):")
    print(
        f"{'w':>4}  {'before@16':>10}  {'after@16':>10}  {'Δ%@16':>8}  "
        f"{'before@best':>11}  {'after@best':>10}  {'best Δ%':>8}"
    )
    print("-" * 80)
    for w in workers:
        a16 = after_by_pf.get(16, {}).get(w)
        b16 = before_by_wp.get((w, 16)) if paired_prefetch else before_by_w_p0.get(w)
        if not a16 and not before_by_w_p0.get(w):
            continue
        ips_b16 = b16["ips"] if b16 else float("nan")
        ips_a16 = a16["ips"] if a16 else float("nan")
        d16 = ((ips_a16 - ips_b16) / ips_b16) * 100.0 if b16 and a16 and ips_b16 else float("nan")
        after_candidates = [after_by_pf.get(pf, {}).get(w) for pf in prefetch_levels]
        after_candidates = [x for x in after_candidates if x is not None]
        best_a = max(after_candidates, key=lambda x: x["ips"] or 0) if after_candidates else None
        best_b = before_by_wp.get((w, best_a["prefetch"])) if best_a and paired_prefetch else before_by_w_p0.get(w)
        db = (
            ((best_a["ips"] - best_b["ips"]) / best_b["ips"]) * 100.0
            if best_a and best_b and best_b["ips"]
            else float("nan")
        )
        ips_bb = best_b["ips"] if best_b else float("nan")
        ips_ba = best_a["ips"] if best_a else float("nan")
        print(
            f"{w:>4}  {ips_b16:>10.1f}  {ips_a16:>10.1f}  {d16:>+7.1f}%  {ips_bb:>11.1f}  {ips_ba:>10.1f}  {db:>+7.1f}%"
        )
    print()
    print("Full cells (includes prefetch=0):")
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
            f"Best after: w={best_after['workers']} "
            f"prefetch={best_after['prefetch']} → {best_after['after_ips']:.1f} samples/s "
            f"(~{best_after['delta_pct']:+.0f}% / {best_after['speedup']:.2f}x vs before)"
        )
    if publish:
        best_pub = max(publish, key=lambda c: c["after_ips"])
        print(
            f"Best published (pf≥16): w={best_pub['workers']} "
            f"prefetch={best_pub['prefetch']} → {best_pub['after_ips']:.1f} samples/s "
            f"(~{best_pub['delta_pct']:+.0f}% / {best_pub['speedup']:.2f}x vs before)"
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
        help=f"minimum timed batches (default {DEFAULT_BATCHES}); window is max(batches, min_seconds)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help=(
            f"minimum timed window seconds (default {DEFAULT_MIN_SECONDS}); "
            f"num_workers≥{HIGH_WORKER_THRESHOLD} always uses ≥{HIGH_WORKER_MIN_SECONDS}s"
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help="DataLoader prefetch_factor; warm batches = max(1, workers * this) (default 2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="repeat each cell N times; report median+spread (default 1; use ≥5 for publish)",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="alternate before/after subprocesses (requires --before-pythonpath); then --merge",
    )
    parser.add_argument(
        "--before-pythonpath",
        type=str,
        default="",
        help="PYTHONPATH for stock main or pre-Stage-1 tree when using --interleave",
    )
    parser.add_argument(
        "--after-prefetch",
        type=str,
        default="",
        help=f"comma-separated prefetch levels for after (and cloud-native before); default {AFTER_PREFETCH}",
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
    if args.trust:
        workers = TRUST_WORKERS
    elif args.workers.strip():
        workers = [int(x) for x in args.workers.split(",") if x.strip()]
    else:
        workers = WORKERS
    if args.after_prefetch.strip():
        prefetch_levels = [int(x) for x in args.after_prefetch.split(",") if x.strip()]
    else:
        prefetch_levels = None
    if args.interleave:
        if not args.before_pythonpath.strip():
            parser.error("--interleave requires --before-pythonpath")
        run_interleaved(
            before_pythonpath=args.before_pythonpath.strip(),
            workers=workers,
            batches=args.batches,
            min_seconds=args.min_seconds,
            prefetch_factor=args.prefetch_factor,
            repeats=args.repeats,
            prefetch_levels=prefetch_levels,
        )
        return
    if not args.side:
        parser.error("pass --side before|after, --interleave, or --merge")
    run_side(
        args.side,
        workers=workers,
        batches=args.batches,
        min_seconds=args.min_seconds,
        prefetch_factor=args.prefetch_factor,
        safety_grid=args.safety_grid,
        repeats=args.repeats,
        prefetch_levels=prefetch_levels,
    )


if __name__ == "__main__":
    main()

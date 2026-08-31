"""Debug / confirm fork-safety with an explicit hang watchdog.

If any step stalls longer than --timeout seconds, abort with a clear message.
Multi-worker steps after parent I/O use spawn by default (OpenSSL-after-fork
can still hang even with fresh downloaders).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("LITDATA_RAW_DEBUG", "1")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d %(process)d %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

from torch.utils.data import DataLoader  # noqa: E402
from uvloop_status import log_loop_runner_backend, uvloop_package_status  # noqa: E402

from litdata import StreamingRawDataset  # noqa: E402

INPUT = "/teamspace/s3_connections/imagenet-1m-template/raw/val"
ROOT = Path(tempfile.gettempdir()) / "litdata-raw-bench-debug"
BS = 32
BATCHES = 5


def log(msg: str) -> None:
    """Print a timestamped benchmark log line."""
    print(f"{time.strftime('%H:%M:%S')} [bench] {msg}", flush=True)


class HangWatchdog:
    """Kill the process if a step exceeds ``timeout_s`` without heartbeat."""

    def __init__(self, timeout_s: float) -> None:
        """Initialize the watchdog with a hang timeout in seconds."""
        self.timeout_s = timeout_s
        self._label = "init"
        self._beat = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hang-watchdog", daemon=True)

    def start(self) -> None:
        """Start the background watchdog thread."""
        self._thread.start()

    def heartbeat(self, label: str) -> None:
        """Record progress so the watchdog does not abort."""
        self._label = label
        self._beat = time.monotonic()
        log(f"watchdog heartbeat: {label}")

    def stop(self) -> None:
        """Stop the background watchdog thread."""
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            idle = time.monotonic() - self._beat
            if idle > self.timeout_s:
                log(
                    f"HANG DETECTED: no progress for {idle:.1f}s at '{self._label}' "
                    f"(timeout={self.timeout_s}s). Aborting."
                )
                os._exit(124)


def copy_index(src: Path, dst: Path) -> None:
    """Copy a cached index tree from ``src`` to ``dst``."""
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    for d in src.iterdir():
        if d.is_dir():
            shutil.copytree(d, dst / d.name)
        else:
            shutil.copy2(d, dst / d.name)


def run(
    label: str,
    *,
    max_prefetch: int,
    num_workers: int,
    watchdog: HangWatchdog,
    reuse: Path | None = None,
    mp_context: str | None = None,
) -> Path:
    """Run one debug trial and return the cache directory used."""
    cache = ROOT / label
    watchdog.heartbeat(f"{label}: begin")
    log(f"=== {label}: workers={num_workers} prefetch={max_prefetch} mp={mp_context}")
    if reuse is not None:
        copy_index(reuse, cache)
    elif cache.exists():
        shutil.rmtree(cache, ignore_errors=True)

    watchdog.heartbeat(f"{label}: construct dataset")
    t0 = time.perf_counter()
    ds = StreamingRawDataset(
        input_dir=INPUT,
        cache_dir=str(cache),
        cache_files=False,
        transform=None,
        max_prefetch=max_prefetch,
        max_concurrent_downloads=64,
    )
    log(f"{label}: dataset ready {time.perf_counter() - t0:.2f}s len={len(ds)}")
    log_loop_runner_backend(log, prefix=f"{label}:")

    kwargs: dict = {"batch_size": BS, "num_workers": num_workers, "shuffle": False}
    if mp_context and num_workers > 0:
        kwargs["multiprocessing_context"] = mp_context

    watchdog.heartbeat(f"{label}: create DataLoader")
    loader = DataLoader(ds, **kwargs)
    it = iter(loader)

    watchdog.heartbeat(f"{label}: warm next()")
    t0 = time.perf_counter()
    batch = next(it)
    log(f"{label}: warm ok {time.perf_counter() - t0:.2f}s n={len(batch)}")

    samples = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(it):
        watchdog.heartbeat(f"{label}: batch {i + 1}")
        samples += len(batch)
        if i + 1 >= BATCHES:
            break
    elapsed = time.perf_counter() - t0
    ips = samples / elapsed if elapsed else 0.0
    log(f"{label}: DONE {samples} samples in {elapsed:.2f}s → {ips:.1f} samples/s")
    del it, loader, ds
    return cache


def main() -> None:
    """CLI entrypoint for fork-safety debug steps with a hang watchdog."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=45.0, help="Hang timeout seconds per step")
    parser.add_argument(
        "--try-fork",
        action="store_true",
        help="Also try num_workers>0 with default fork after parent I/O (may hang on OpenSSL)",
    )
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    wd = HangWatchdog(args.timeout)
    wd.start()
    log(f"start pid={os.getpid()} timeout={args.timeout}s")
    log(f"uvloop package: {uvloop_package_status()}")

    try:
        log("STEP1: workers=0 seed")
        seed = run("seed_w0", max_prefetch=0, num_workers=0, watchdog=wd)

        log("STEP2: workers=0 prefetch (dirty parent)")
        run("w0_p64", max_prefetch=64, num_workers=0, watchdog=wd, reuse=seed)

        log("STEP3: workers=4 spawn (safe after parent I/O)")
        run("w4_p0_spawn", max_prefetch=0, num_workers=4, watchdog=wd, reuse=seed, mp_context="spawn")

        log("STEP4: workers=4 spawn + prefetch")
        run("w4_p128_spawn", max_prefetch=128, num_workers=4, watchdog=wd, reuse=seed, mp_context="spawn")

        if args.try_fork:
            log("STEP5: workers=4 fork after parent I/O (known OpenSSL risk)")
            run("w4_p0_fork", max_prefetch=0, num_workers=4, watchdog=wd, reuse=seed, mp_context=None)

        log("ALL STEPS COMPLETE")
    finally:
        wd.stop()


if __name__ == "__main__":
    main()

# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming raw files with async downloads and optional look-ahead prefetch.

Concurrency model
-----------------
A per-process ``_LoopRunner`` owns a dedicated event-loop thread. All dataset I/O
is dispatched with ``run_coroutine_threadsafe``, so prefetch continues between
``__getitems__`` calls (including ``num_workers=0`` and notebook nested loops).

Runtime clients (downloader, semaphore, path-dedupe, prefetch tasks) are keyed by
``(pid, event loop)`` and recreated when either changes. Fork clears the runner
(threads do not survive ``os.fork``); the next call builds a fresh one.

Cache publishes are atomic via temp file + ``os.replace``. Temp names include
``pid`` and thread id. Cross-process cache writers coordinate with ``O_EXCL``
lock files (``*.litdata-raw.lock``). ``cache_files=True`` + ``item_type="bytes"``
uses write-through: bytes return from the network path while a background thread
publishes the cache.

Known limitations (accepted)
----------------------------
- ``_close_downloader_best_effort`` cannot deterministically await async ``close()``
  when the bound loop is dead; pooled connections rely on GC in that case.
- A failed ``asyncio.gather`` in ``_download_batch`` may leave sibling resolve
  coroutines to finish on the next call or at loop close.
- ``LoopRunner.run(...).result()`` blocks the caller thread (DataLoader worker /
  main); cancellation from ``KeyboardInterrupt`` is best-effort.
- ThreadPoolExecutor worker threads (range downloads / write-behind) may still be
  alive at ``os.fork``; the child reinitializes locks and drops the runner, but
  inherited executor threads are not joined.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import logging
import os
import statistics
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar
from uuid import uuid4

from torch.utils.data import Dataset

from litdata.raw.indexer import BaseIndexer, FileIndexer, FileMetadata
from litdata.streaming.downloader import Downloader, get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)

T = TypeVar("T")
_MISS = object()
_RAW_DEBUG = bool(os.getenv("LITDATA_RAW_DEBUG"))

# Parallel ranged GETs for large objects (S3/GCS/R2 downloaders with real Range support).
# Opt-in: default 0 disables; pass a positive byte threshold to enable.
_RANGE_PARALLEL_THRESHOLD = 0
_RANGE_CHUNK_SIZE = 8 * 1024 * 1024
_TINY_FILE_MEDIAN_BYTES = 100_000
_LOCK_STALE_SECONDS = 300.0
_LOCK_SUFFIX = ".litdata-raw.lock"
# Hedge only small / unknown objects; large whole-object GETs must not 2× egress.
_HEDGE_MAX_BYTES = 8 * 1024 * 1024
# Per-stream floor for hedge delay / batch-timeout sizing (healthy single GET).
# Distinct from ``_ASSUMED_AGGREGATE_BANDWIDTH_BPS`` below (NIC/prefix share used to
# size Stage 1 aggregate concurrency). Do not conflate the two constants.
_HEDGE_ASSUMED_BANDWIDTH_BPS = 25 * 1024 * 1024  # ~25 MB/s per-stream floor
# Cap aggregate sequential look-ahead across DataLoader workers (items total).
# Per-worker effective = min(max_prefetch, budget // num_workers) when num_workers > 1.
_AGGREGATE_PREFETCH_BUDGET = 64
# Stage 1 static concurrency: aggregate in-flight budget = max(bandwidth model,
# Little's-law / latency model), then split across DataLoader workers.
_ASSUMED_AGGREGATE_BANDWIDTH_BPS = 100 * 1024 * 1024  # ~100 MB/s NIC / prefix share
_CONCURRENCY_PIPELINE_SECONDS = 0.5  # target aggregate bytes ≈ bandwidth × this
_ASSUMED_REQUEST_RATE = 6000.0  # tiny-object target req/s for Little's-law arm
_ASSUMED_REQUEST_LATENCY_S = 0.040  # assumed RTT+TTFB (seconds)
_DEFAULT_MEDIAN_FILE_BYTES = 256 * 1024
_AGGREGATE_CONCURRENCY_BUDGET_FLOOR = 32
# Cap keeps high-w stampede (was N×64 → 1536) in check without crushing mid-w
# ImageNet cells to ~128 aggregate (which capped winning w=4/w=8 configs).
_AGGREGATE_CONCURRENCY_BUDGET_CAP = 512
_MIN_CONCURRENCY_PER_WORKER = 8
# Unbenchmarked single-process adaptive path: do not open the full ~512 budget.
_SINGLE_PROCESS_CONCURRENCY_CAP = 128
# Little's-law arm only for sub-MiB objects (request-overhead bound). At ≥1 MiB
# the bandwidth arm alone sizes the budget. Distinct from ``_HEDGE_MAX_BYTES``
# (8 MiB duplicate-GET hedge policy).
_LATENCY_MODEL_MAX_MEDIAN_BYTES = 1024 * 1024

_RUNNER_LOCK = threading.Lock()
_RUNNER: _LoopRunner | None = None

_WRITE_BEHIND_LOCK = threading.Lock()
_WRITE_BEHIND_FUTURES: set[asyncio.Future[Any]] = set()


def _loop_backend_name() -> str:
    """Return ``uvloop`` when the package is importable, else ``asyncio``."""
    try:
        import uvloop  # noqa: F401
    except ImportError:
        return "asyncio"
    return "uvloop"


def _create_event_loop() -> asyncio.AbstractEventLoop:
    """Create a new event loop, preferring uvloop when available."""
    try:
        import uvloop
    except ImportError:
        return asyncio.new_event_loop()
    return uvloop.new_event_loop()


def _consume_task_exception(task: asyncio.Task) -> None:
    """Mark task exceptions as retrieved so asyncio does not warn at GC time."""
    if task.cancelled():
        return
    with contextlib.suppress(asyncio.InvalidStateError, Exception):
        task.exception()


def _close_unawaited(coro: Awaitable[Any]) -> None:
    """Close a coroutine that will not be awaited (avoids 'was never awaited')."""
    close = getattr(coro, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _track_write_behind(fut: asyncio.Future[Any]) -> None:
    """Track a ``run_in_executor`` future until the write-behind finishes."""
    with _WRITE_BEHIND_LOCK:
        _WRITE_BEHIND_FUTURES.add(fut)

    def _done(f: asyncio.Future[Any]) -> None:
        with _WRITE_BEHIND_LOCK:
            _WRITE_BEHIND_FUTURES.discard(f)
        with contextlib.suppress(Exception):
            f.result()

    fut.add_done_callback(_done)


def _drain_write_behind_futures() -> None:
    """Best-effort wait for in-flight write-behind publishes (shared ~0.5s deadline)."""
    with _WRITE_BEHIND_LOCK:
        pending = list(_WRITE_BEHIND_FUTURES)
        _WRITE_BEHIND_FUTURES.clear()
    if not pending:
        return
    deadline = time.monotonic() + 0.5
    for fut in pending:
        while not fut.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.01, remaining))
        with contextlib.suppress(Exception):
            fut.result()


atexit.register(_drain_write_behind_futures)


class _LoopRunner:
    """Process-local asyncio loop running forever on a daemon thread."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self.loop: asyncio.AbstractEventLoop = _create_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="litdata-raw-pool")
        self.loop.set_default_executor(self._executor)
        if _RAW_DEBUG:
            logger.warning(
                "raw-debug: LoopRunner backend=%s pid=%s",
                _loop_backend_name(),
                self._pid,
            )
        self._thread = threading.Thread(target=self._main, name="litdata-raw-aio", daemon=True)
        self._started = threading.Event()
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise RuntimeError("Failed to start StreamingRawDataset event-loop thread")

    def _main(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        with contextlib.suppress(Exception):
            self.loop.close()

    @property
    def pid(self) -> int:
        return self._pid

    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self.loop.is_closed()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is self._thread:
            _close_unawaited(coro)
            raise RuntimeError(
                "LoopRunner.run() called from the event-loop thread; this would deadlock. "
                "Await the coroutine directly instead of calling run()."
            )
        if os.getpid() != self._pid:
            _close_unawaited(coro)
            raise RuntimeError("LoopRunner used after fork; call _get_loop_runner() to recreate")
        if not self.is_alive():
            _close_unawaited(coro)
            raise RuntimeError("StreamingRawDataset event-loop thread is not running")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def shutdown_best_effort(self) -> None:
        if self.loop.is_closed():
            with contextlib.suppress(Exception):
                self._executor.shutdown(wait=False, cancel_futures=True)
            return

        def _stop() -> None:
            try:
                with contextlib.suppress(Exception):
                    self._executor.shutdown(wait=False, cancel_futures=True)
            finally:
                # Must always stop the loop — uvloop has no assignable `_default_executor`.
                self.loop.stop()

        with contextlib.suppress(Exception):
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(_stop)
            else:
                _stop()
        self._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=False, cancel_futures=True)


def _get_loop_runner() -> _LoopRunner:
    """Return the process-local loop runner, creating it if needed."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None or _RUNNER.pid != os.getpid() or not _RUNNER.is_alive():
            if _RUNNER is not None:
                _RUNNER.shutdown_best_effort()
            if _RAW_DEBUG:
                logger.warning("raw-debug: creating LoopRunner pid=%s", os.getpid())
            _RUNNER = _LoopRunner()
        return _RUNNER


def _shutdown_runner_before_fork() -> None:
    """Stop the loop thread before fork (threads do not survive fork safely)."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is not None:
            if _RAW_DEBUG:
                logger.warning("raw-debug: shutting down LoopRunner before fork pid=%s", os.getpid())
            _RUNNER.shutdown_best_effort()
        _RUNNER = None


def _reinit_after_fork() -> None:
    """Drop inherited runner/futures and reinit module-level locks in the child."""
    global _RUNNER, _RUNNER_LOCK, _WRITE_BEHIND_LOCK, _WRITE_BEHIND_FUTURES
    _RUNNER = None
    _RUNNER_LOCK = threading.Lock()
    _WRITE_BEHIND_LOCK = threading.Lock()
    _WRITE_BEHIND_FUTURES = set()


# Backward-compatible alias for tests / callers that still import the old name.
_clear_runner_after_fork = _reinit_after_fork


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_shutdown_runner_before_fork, after_in_child=_reinit_after_fork)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Dispatch ``coro`` onto the process-local loop thread and wait for the result."""
    return _get_loop_runner().run(coro)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _unlink_if_dead_pid(path: Path, pid: int) -> None:
    if not _pid_alive(pid):
        with contextlib.suppress(OSError):
            path.unlink()


def _sweep_orphan_tmp_files(cache_dir: str) -> None:
    """Best-effort cleanup of ``*.tmp.<pid>*`` and ``.range-scratch.<pid>*`` orphans."""
    root = Path(cache_dir)
    if not root.is_dir():
        return
    for tmp in root.rglob("*.tmp.*"):
        marker = ".tmp."
        idx = tmp.name.rfind(marker)
        if idx < 0:
            continue
        suffix = tmp.name[idx + len(marker) :]
        pid_str = suffix.split(".", 1)[0]
        try:
            pid = int(pid_str)
        except ValueError:
            # Non-pid suffix: drop if mtime is stale.
            try:
                if time.time() - tmp.stat().st_mtime > _LOCK_STALE_SECONDS:
                    tmp.unlink()
            except OSError:
                pass
            continue
        _unlink_if_dead_pid(tmp, pid)
    # Orphan ranged-GET scratch files: ``.range-scratch.<pid>.<tid>...``
    for scratch in root.rglob(".range-scratch.*"):
        parts = scratch.name.split(".")
        # "", "range-scratch", "<pid>", ...
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[2])
        except ValueError:
            try:
                if time.time() - scratch.stat().st_mtime > _LOCK_STALE_SECONDS:
                    scratch.unlink()
            except OSError:
                pass
            continue
        _unlink_if_dead_pid(scratch, pid)
    # Stale cross-process lock files from dead pids.
    for lock in root.rglob(f"*{_LOCK_SUFFIX}"):
        try:
            text = lock.read_text().strip()
            pid = int(text.split()[0])
        except (OSError, ValueError, IndexError):
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                    lock.unlink()
            except OSError:
                pass
            continue
        _unlink_if_dead_pid(lock, pid)


def _looks_sequential(indices: list[int]) -> bool:
    """Return True if indices are a contiguous ascending range (typical DataLoader batch)."""
    if len(indices) <= 1:
        return True
    return all(indices[i] == indices[i - 1] + 1 for i in range(1, len(indices)))


def _effective_prefetch(max_prefetch: int, num_workers: int) -> int:
    """Per-worker look-ahead capped so aggregate stays near ``_AGGREGATE_PREFETCH_BUDGET``.

    Constructor default ``max_prefetch=16`` stays ergonomic at low workers (w≤4 keeps 16).
    At higher worker counts each worker gets a smaller share so total in-flight look-ahead
    does not scale as ``num_workers × max_prefetch``.

    TODO(Stage 2): hit-rate controller — if windowed hit rate <30% → halve effective
    look-ahead (floor 0); >90% and batch waited on a miss → +1–2. Not implemented yet.
    """
    if max_prefetch <= 0:
        return 0
    if num_workers <= 1:
        return max_prefetch
    # TODO(open): if effective < 8, return 0 — pending repeats; priority below downloader conformance.
    return min(max_prefetch, max(0, _AGGREGATE_PREFETCH_BUDGET // num_workers))


def _median_file_bytes(files: Sequence[FileMetadata]) -> int | None:
    """Return median positive file size from index metadata, or ``None`` if unknown."""
    sizes = [f.size for f in files if f.size > 0]
    if not sizes:
        return None
    return int(statistics.median(sizes))


def _aggregate_concurrency_budget(median_file_bytes: int | None) -> int:
    """Aggregate in-flight download slots across all workers (size-aware, clamped).

    Takes the max of two models then clamps to ``[floor, cap]``:

    - **bandwidth**: ``(aggregate_bps × pipeline_s) // median_file_bytes`` — keep
      ~50 MiB moving for large objects.
    - **latency / Little's law**: ``target_rate × assumed_latency`` (~6000×0.040 ≈
      240) **only when** ``median < _LATENCY_MODEL_MAX_MEDIAN_BYTES`` (1 MiB) so
      tiny-object paths are not request-starved. Medians ≥1 MiB stay
      bandwidth-bounded (avoids pinning at 240 slots → multi-GB in flight).

    Per-worker floor of 8 means realized aggregate is ``max(budget, 8 × num_workers)``.
    """
    median = median_file_bytes if median_file_bytes and median_file_bytes > 0 else _DEFAULT_MEDIAN_FILE_BYTES
    target_bytes = int(_ASSUMED_AGGREGATE_BANDWIDTH_BPS * _CONCURRENCY_PIPELINE_SECONDS)
    bandwidth_model = max(1, target_bytes // median)
    # Size-gate: Little's-law arm is for request-overhead-bound tiny objects only.
    if median < _LATENCY_MODEL_MAX_MEDIAN_BYTES:
        latency_model = max(1, int(_ASSUMED_REQUEST_RATE * _ASSUMED_REQUEST_LATENCY_S))
    else:
        latency_model = 0
    raw = max(bandwidth_model, latency_model)
    return max(_AGGREGATE_CONCURRENCY_BUDGET_FLOOR, min(_AGGREGATE_CONCURRENCY_BUDGET_CAP, raw))


def _effective_concurrency(
    max_concurrent_downloads: int | None,
    num_workers: int,
    median_file_bytes: int | None = None,
) -> int:
    """Per-worker download permits for the Stage 1 static clamp.

    - ``max_concurrent_downloads is None`` (default): adaptive —
      ``max(floor, budget // num_workers)`` with ``budget`` from
      :func:`_aggregate_concurrency_budget`. When ``num_workers <= 1``, returns
      ``min(budget, _SINGLE_PROCESS_CONCURRENCY_CAP)`` (unbenchmarked path).
    - Explicit ``int``: **exactly** that many permits (no silent clamp). ``<= 0``
      collapses to 1.
    """
    if max_concurrent_downloads is not None:
        return 1 if max_concurrent_downloads <= 0 else max_concurrent_downloads
    budget = _aggregate_concurrency_budget(median_file_bytes)
    if num_workers <= 1:
        return min(budget, _SINGLE_PROCESS_CONCURRENCY_CAP)
    return max(_MIN_CONCURRENCY_PER_WORKER, budget // num_workers)


def _num_dataloader_workers() -> int:
    """Return DataLoader ``num_workers``, or ``1`` when called outside a worker."""
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return 1
    info = get_worker_info()
    if info is None:
        return 1
    return max(1, int(info.num_workers))


def _consume_prefetch_exception(task: asyncio.Task) -> None:
    """Mark prefetch task exceptions as retrieved so asyncio does not warn at GC time."""
    if task.cancelled():
        return
    with contextlib.suppress(asyncio.InvalidStateError):
        exc = task.exception()
        if exc is not None:
            logger.debug("prefetch failed; will retry on demand", exc_info=exc)


def _effective_hedge_delay(hedge_delay: float, size: int | None) -> float | None:
    """Return hedge wait seconds, or ``None`` when hedging should be skipped.

    Unknown / non-positive sizes never hedge (cannot bound duplicate egress). Large
    whole-object GETs (``>= _HEDGE_MAX_BYTES``) never hedge either — callers should
    hedge per ranged chunk instead. Delay is at least ``3 * size / 25MB/s`` so a
    healthy transfer is not spuriously duplicated.
    """
    if hedge_delay <= 0:
        return None
    if size is None or size <= 0:
        return None
    if size >= _HEDGE_MAX_BYTES:
        return None
    expected = size / _HEDGE_ASSUMED_BANDWIDTH_BPS
    return max(hedge_delay, 3.0 * expected)


class _LRUCache:
    """Simple ordered LRU cache keyed by dataset index."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(0, maxsize)
        self._data: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int) -> Any:
        if self.maxsize <= 0 or key not in self._data:
            return _MISS
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: int, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, int) and key in self._data


class CacheManager:
    """Manages file caching for remote datasets, preserving directory structure."""

    def __init__(
        self,
        input_dir: str | Dir,
        cache_dir: str | None = None,
        storage_options: dict | None = None,
        cache_files: bool = False,
        max_concurrent_downloads: int | None = None,
        hedge_delay: float = 0.0,
        download_timeout: float = 120.0,
        range_parallel_threshold: int = _RANGE_PARALLEL_THRESHOLD,
        range_chunk_size: int = _RANGE_CHUNK_SIZE,
    ):
        self.input_dir = _resolve_dir(input_dir)
        self._input_dir_path = _storage_path(self.input_dir)
        self.cache_files = cache_files
        self.max_concurrent_downloads = max_concurrent_downloads
        self.hedge_delay = max(0.0, hedge_delay)
        self.download_timeout = max(0.0, download_timeout) or None
        self.range_parallel_threshold = max(0, range_parallel_threshold)
        self.range_chunk_size = max(1, range_chunk_size)
        self.lock_wait_timeout = _LOCK_STALE_SECONDS

        self.cache_dir = self._create_cache_dir(self._input_dir_path, cache_dir)
        _sweep_orphan_tmp_files(self.cache_dir)

        self.storage_options = storage_options or {}
        # Index median size (bytes); set by StreamingRawDataset after discovery.
        self._median_file_bytes: int | None = None
        self._downloader: Downloader | None = None
        self._downloader_pid: int | None = None
        self._downloader_loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._semaphore_permits: int | None = None
        # Pid-guarded cache of Stage 1 permit count (avoid hot-path get_worker_info).
        self._cached_permits: int | None = None
        self._cached_permits_pid: int | None = None
        self._path_inflight: dict[str, asyncio.Task] = {}
        self._path_inflight_loop: asyncio.AbstractEventLoop | None = None
        # Presence hint only: membership does not skip exists checks (stale marks self-heal).
        self._present_paths: set[str] = set()
        self._range_executor: ThreadPoolExecutor | None = None
        self._range_executor_pid: int | None = None
        self._hedge_fired = 0

    def reset_runtime_state(self) -> None:
        """Drop process/loop-bound clients (call after fork or when pickling)."""
        self._close_downloader_best_effort(self._downloader)
        self._downloader = None
        self._downloader_pid = None
        self._downloader_loop = None
        self._semaphore = None
        self._semaphore_loop = None
        self._semaphore_permits = None
        self._cached_permits = None
        self._cached_permits_pid = None
        self._path_inflight = {}
        self._path_inflight_loop = None
        self._shutdown_range_executor()
        self._hedge_fired = 0
        # Keep _present_paths / _median_file_bytes — index metadata survives fork/spawn.

    def __getstate__(self) -> dict[str, Any]:
        """Serialize config only — never downloader/loop/executor/inflight state.

        Allowlisted keys avoid accidental instance attrs (locks, futures) breaking
        ``multiprocessing_context='spawn'`` pickling.
        """
        return {
            "input_dir": self.input_dir,
            "_input_dir_path": self._input_dir_path,
            "cache_files": self.cache_files,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "hedge_delay": self.hedge_delay,
            "download_timeout": self.download_timeout,
            "range_parallel_threshold": self.range_parallel_threshold,
            "range_chunk_size": self.range_chunk_size,
            "lock_wait_timeout": self.lock_wait_timeout,
            "cache_dir": self.cache_dir,
            "storage_options": self.storage_options,
            "_median_file_bytes": self._median_file_bytes,
            # Runtime — always fresh in the child.
            "_downloader": None,
            "_downloader_pid": None,
            "_downloader_loop": None,
            "_semaphore": None,
            "_semaphore_loop": None,
            "_semaphore_permits": None,
            "_cached_permits": None,
            "_cached_permits_pid": None,
            "_path_inflight": {},
            "_path_inflight_loop": None,
            "_present_paths": set(),
            "_range_executor": None,
            "_range_executor_pid": None,
            "_hedge_fired": 0,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Belt-and-suspenders: never revive process-bound clients after unpickle.
        self._downloader = None
        self._downloader_pid = None
        self._downloader_loop = None
        self._semaphore = None
        self._semaphore_loop = None
        self._semaphore_permits = None
        self._cached_permits = None
        self._cached_permits_pid = None
        self._path_inflight = {}
        self._path_inflight_loop = None
        self._present_paths = set(state.get("_present_paths") or ())
        self._range_executor = None
        self._range_executor_pid = None
        self._hedge_fired = 0
        self._median_file_bytes = state.get("_median_file_bytes")

    def _shutdown_range_executor(self) -> None:
        if self._range_executor is not None:
            with contextlib.suppress(Exception):
                self._range_executor.shutdown(wait=False, cancel_futures=True)
        self._range_executor = None
        self._range_executor_pid = None

    def _get_range_executor(self) -> ThreadPoolExecutor:
        pid = os.getpid()
        if self._range_executor is None or self._range_executor_pid != pid:
            self._shutdown_range_executor()
            # Explicit permit cap when set; otherwise a modest default (adaptive
            # Stage 1 budget is applied on the download semaphore, not here).
            cap = self.max_concurrent_downloads if self.max_concurrent_downloads is not None else 32
            workers = max(4, min(32, cap))
            self._range_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="litdata-raw-range",
            )
            self._range_executor_pid = pid
        return self._range_executor

    @staticmethod
    def _close_downloader_best_effort(downloader: Downloader | None) -> None:
        """Best-effort downloader teardown (sync or async ``close`` / ``aclose``).

        Async close coroutines cannot be awaited safely when the bound loop is already
        dead (fork reset), so the coroutine object is closed and pooled connections
        fall back to GC rather than a deterministic drain.
        """
        if downloader is None:
            return
        for name in ("close", "aclose"):
            fn = getattr(downloader, name, None)
            if not callable(fn):
                continue
            with contextlib.suppress(Exception):
                result = fn()
                if asyncio.iscoroutine(result):
                    with contextlib.suppress(Exception):
                        result.close()
            return

    @property
    def downloader(self) -> Downloader:
        """Lazily initialize the downloader.

        Recreate when the process id **or** running event loop changes. Clients bound to a
        closed parent loop hang both in forked workers and on later main-process access.
        """
        pid = os.getpid()
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._downloader is None or self._downloader_pid != pid or self._downloader_loop is not loop:
            if _RAW_DEBUG and self._downloader is not None:
                logger.warning(
                    "raw-debug: recreating downloader pid=%s->%s loop_changed=%s",
                    self._downloader_pid,
                    pid,
                    self._downloader_loop is not loop,
                )
            self._close_downloader_best_effort(self._downloader)
            self._downloader = get_downloader(
                remote_dir=self._input_dir_path,
                cache_dir=self.cache_dir,
                chunks=[],
                storage_options=self.storage_options,
            )
            self._downloader_pid = pid
            self._downloader_loop = loop
        return self._downloader

    def _effective_download_permits(self) -> int:
        """Worker-aware permit count for the download semaphore (Stage 1 static clamp).

        Computed once per process (pid-guarded cache). Cleared by
        ``reset_runtime_state`` / pickle so forked workers recompute.
        """
        pid = os.getpid()
        if self._cached_permits is not None and self._cached_permits_pid == pid:
            return self._cached_permits
        permits = _effective_concurrency(
            self.max_concurrent_downloads,
            _num_dataloader_workers(),
            self._median_file_bytes,
        )
        self._cached_permits = permits
        self._cached_permits_pid = pid
        return permits

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return a semaphore bound to the current event loop with effective permits.

        Permit count comes from :meth:`_effective_download_permits` (cached per
        process). Loop-keyed like other runtime clients; cleared by
        ``reset_runtime_state``.
        """
        loop = asyncio.get_running_loop()
        permits = self._effective_download_permits()
        if self._semaphore is None or self._semaphore_loop is not loop or self._semaphore_permits != permits:
            n_workers = _num_dataloader_workers()
            budget = (
                _aggregate_concurrency_budget(self._median_file_bytes)
                if self.max_concurrent_downloads is None
                else None
            )
            logger.info(
                "adaptive concurrency: median=%s budget=%s workers=%s permits=%s",
                self._median_file_bytes,
                budget,
                n_workers,
                permits,
            )
            self._semaphore = asyncio.Semaphore(permits)
            self._semaphore_loop = loop
            self._semaphore_permits = permits
        return self._semaphore

    @asynccontextmanager
    async def _permit(self, gated: bool = True) -> AsyncIterator[None]:
        if gated:
            async with self._get_semaphore():
                yield
        else:
            yield

    def _create_cache_dir(self, input_dir: str, cache_dir: str | None = None) -> str:
        """Create cache directory if it doesn't exist."""
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        cache_path = os.path.join(cache_dir, generate_md5_hash(input_dir))
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def get_local_path(self, remote_file_path: str) -> str:
        """Convert remote file path to its local cache location."""
        remote_base_path = self._input_dir_path.rstrip("/") + "/"
        if not remote_file_path.startswith(remote_base_path):
            raise ValueError(f"File path {remote_file_path} does not start with input dir {remote_base_path}")

        relative_path = remote_file_path[len(remote_base_path) :]
        local_path = Path(self.cache_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return str(local_path)

    def _path_is_cached(self, local_path: str) -> bool:
        if local_path in self._present_paths:
            if os.path.exists(local_path):
                return True
            # Stale presence mark (file removed under us) — discard and recheck disk.
            self._present_paths.discard(local_path)
        if os.path.exists(local_path):
            self._present_paths.add(local_path)
            return True
        return False

    def _lock_path(self, local_path: str) -> str:
        return f"{local_path}{_LOCK_SUFFIX}"

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return _pid_alive(pid)

    def _lock_owner_alive(self, lock_path: str) -> bool:
        """Return True if the lock looks held by a live process (or is still being written)."""
        try:
            text = Path(lock_path).read_text().strip()
        except OSError:
            return True
        if not text:
            # Claim race: lock created but pid not written yet — treat as alive unless stale.
            try:
                return time.time() - os.path.getmtime(lock_path) <= _LOCK_STALE_SECONDS
            except OSError:
                return True
        try:
            pid = int(text.split()[0])
        except (ValueError, IndexError):
            try:
                return time.time() - os.path.getmtime(lock_path) <= _LOCK_STALE_SECONDS
            except OSError:
                return True
        return self._pid_alive(pid)

    def _try_claim_lock(self, local_path: str) -> bool:
        """Claim ``local_path.litdata-raw.lock`` with ``O_EXCL``. Return True if we own the download."""
        lock_path = self._lock_path(local_path)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not self._lock_owner_alive(lock_path):
                with contextlib.suppress(OSError):
                    os.remove(lock_path)
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    return False
            else:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                except OSError:
                    age = 0.0
                if age > _LOCK_STALE_SECONDS:
                    with contextlib.suppress(OSError):
                        os.remove(lock_path)
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    except FileExistsError:
                        return False
                else:
                    return False
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)
        return True

    def _release_lock(self, local_path: str) -> None:
        with contextlib.suppress(OSError):
            os.remove(self._lock_path(local_path))

    async def _wait_for_cached_file(self, local_path: str, timeout: float | None = None) -> str:
        """Poll until another process publishes ``local_path`` or the lock is released/stale."""
        if timeout is None:
            timeout = self.lock_wait_timeout
        deadline = time.monotonic() + timeout
        lock_path = self._lock_path(local_path)
        while time.monotonic() < deadline:
            if self._path_is_cached(local_path):
                return local_path
            if not os.path.exists(lock_path):
                break
            if not self._lock_owner_alive(lock_path):
                with contextlib.suppress(OSError):
                    os.remove(lock_path)
                break
            await asyncio.sleep(0.05)
        if self._path_is_cached(local_path):
            return local_path
        raise TimeoutError(f"Timed out waiting for cache file {local_path}")

    @staticmethod
    def _is_remote_object(file_path: str) -> bool:
        return "://" in file_path and not file_path.startswith("file://")

    @staticmethod
    def _is_non_retryable_download_error(exc: BaseException) -> bool:
        return isinstance(
            exc,
            (
                asyncio.CancelledError,
                NotImplementedError,
                ValueError,
                TypeError,
                PermissionError,
                FileNotFoundError,
                IsADirectoryError,
                TimeoutError,
                concurrent.futures.TimeoutError,
            ),
        )

    async def _hedged(self, factory: Callable[[], Coroutine[Any, Any, T]], delay: float) -> T:
        """Run ``factory``; if slow, start a second request and prefer a non-exception winner.

        Each ``factory`` call is expected to acquire its own semaphore permit. The hedge is
        skipped when the semaphore is already exhausted so hedges cannot starve primary work.

        Note: cancelling a losing hedged task that is blocked in ``run_in_executor`` does
        not abort the worker thread — the full chunk transfer may still complete and pay
        bandwidth even after the asyncio task is cancelled.
        """
        first: asyncio.Task[T] = asyncio.create_task(factory())
        if delay <= 0:
            return await first
        done, _ = await asyncio.wait({first}, timeout=delay)
        if done:
            return first.result()

        # Only hedge when a concurrency permit is immediately available.
        if self._get_semaphore().locked():
            return await first

        second: asyncio.Task[T] = asyncio.create_task(factory())
        self._hedge_fired += 1
        logger.debug("hedge fired count=%s delay=%.3fs", self._hedge_fired, delay)
        pending: set[asyncio.Task[T]] = {first, second}
        try:
            while pending:
                finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                success = [t for t in finished if not t.cancelled() and t.exception() is None]
                if success:
                    for task in pending:
                        task.cancel()
                        task.add_done_callback(_consume_task_exception)
                    for task in finished:
                        if task not in success:
                            _consume_task_exception(task)
                    return success[0].result()
                for task in finished:
                    _consume_task_exception(task)
            # Both failed — re-raise from the first attempt.
            return first.result()
        except BaseException:
            for task in (first, second):
                if not task.done():
                    task.cancel()
                    task.add_done_callback(_consume_task_exception)
            raise

    def _download_budget(self, size: int | None = None, timeout: float | None = None) -> float | None:
        """Return a size-aware timeout floor in seconds, or ``None`` when disabled.

        ``download_timeout`` is a floor for sized objects: when ``size`` is known,
        the budget is ``max(download_timeout, size / assumed_bandwidth * 3)`` so large
        transfers are not cut off by a fixed wall-clock cap. Used by batch-level
        hang protection (``max`` over pending indices). Pass an explicit ``timeout``
        to override.
        """
        if timeout is not None:
            return timeout
        base = self.download_timeout
        if base is None:
            return None
        if size is not None and size > 0:
            size_floor = size / _HEDGE_ASSUMED_BANDWIDTH_BPS * 3.0
            return max(base, size_floor)
        return base

    def _supports_range(self, file_path: str) -> bool:
        return file_path.startswith(("s3://", "gs://", "r2://"))

    async def _ranged_download_bytes(self, file_path: str, size: int, *, gated: bool = True) -> bytes:
        """Parallel ranged GETs via the sync ``download_bytes`` API (per-chunk hedge + validate)."""
        chunk = self.range_chunk_size
        ranges = [(start, min(chunk, size - start)) for start in range(0, size, chunk)]
        downloader = self.downloader
        executor = self._get_range_executor()
        base_scratch = os.path.join(
            self.cache_dir,
            f".range-scratch.{os.getpid()}.{threading.get_ident()}",
        )
        chunk_delay = _effective_hedge_delay(self.hedge_delay, chunk)

        async def one(offset: int, length: int) -> tuple[int, bytes]:
            async def fetch() -> bytes:
                # Unique scratch per attempt so first/hedge never share a path.
                scratch = f"{base_scratch}.{offset}.{uuid4().hex}"
                try:
                    async with self._permit(gated):
                        data = await asyncio.get_running_loop().run_in_executor(
                            executor,
                            downloader.download_bytes,
                            file_path,
                            offset,
                            length,
                            scratch,
                        )
                    if len(data) != length:
                        raise RuntimeError(
                            f"Ranged GET short read for {file_path}: offset={offset} expected={length} got={len(data)}"
                        )
                    return data
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(scratch)

            if chunk_delay is not None and self._is_remote_object(file_path):
                data = await self._hedged(fetch, chunk_delay)
            else:
                data = await fetch()
            return offset, data

        parts = await asyncio.gather(*(one(o, n) for o, n in ranges))
        parts.sort(key=lambda x: x[0])
        joined = b"".join(data for _, data in parts)
        if len(joined) != size:
            raise RuntimeError(f"Ranged download size mismatch for {file_path}: expected={size} got={len(joined)}")
        return joined

    async def _fetch_bytes(self, file_path: str, size: int | None = None, *, gated: bool = True) -> bytes:
        """Download object bytes (optional range-parallel + size-gated hedging).

        Hang protection is enforced once per batch in
        ``StreamingRawDataset._download_batch`` (single ``wait_for`` around the
        gather). Defaults (``hedge_delay=0``, ``download_timeout=120``) therefore
        take this bare fast path — no per-item ``asyncio.wait_for``.
        """
        # Per-chunk hedging happens inside ranged downloads; never hedge the whole object.
        if (
            size is not None
            and self.range_parallel_threshold > 0
            and size >= self.range_parallel_threshold
            and self._supports_range(file_path)
        ):
            return await self._ranged_download_bytes(file_path, size, gated=gated)

        delay = _effective_hedge_delay(self.hedge_delay, size) if self._is_remote_object(file_path) else None
        # Pay-per-use: hedging off/ineligible → bare permit + download (batch enforces timeout).
        if delay is None:
            async with self._permit(gated):
                return await self.downloader.adownload_fileobj(file_path)

        async def once() -> bytes:
            async with self._permit(gated):
                return await self.downloader.adownload_fileobj(file_path)

        return await self._hedged(once, delay)

    def _schedule_write_behind(self, local_path: str, data: bytes) -> None:
        """Atomically publish ``data`` to ``local_path`` on a worker thread."""
        if self._path_is_cached(local_path):
            return

        def _write() -> None:
            if os.path.exists(local_path):
                self._present_paths.add(local_path)
                return
            tmp_path = f"{local_path}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, local_path)
                self._present_paths.add(local_path)
            except Exception:
                logger.debug("write-behind cache publish failed for %s", local_path, exc_info=True)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, _write)
        _track_write_behind(fut)

    def _verify_tmp_size(self, tmp_path: str, size: int | None) -> None:
        if size is None or size <= 0:
            return
        actual = os.path.getsize(tmp_path)
        if actual < size:
            raise RuntimeError(f"Downloaded file truncated: expected>={size} got={actual} path={tmp_path}")

    async def _download_owned(self, file_path: str, local_path: str, size: int | None = None) -> str:
        """Download while holding the cross-process lock (caller owns the lock)."""
        tmp_path = f"{local_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        try:
            if self._path_is_cached(local_path):
                return local_path
            try:
                # Hang protection is batch-level; keep the owned path bare.
                await self.downloader.adownload_file(file_path, tmp_path)
            except Exception as first_exc:
                if self._is_non_retryable_download_error(first_exc):
                    raise
                logger.warning(
                    "adownload_file failed for %s; falling back to bytes path",
                    file_path,
                    exc_info=first_exc,
                )
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
                # Caller already holds the download semaphore — avoid nested acquire.
                data = await self._fetch_bytes(file_path, size=size, gated=False)
                await asyncio.to_thread(Path(tmp_path).write_bytes, data)
            self._verify_tmp_size(tmp_path, size)
            os.replace(tmp_path, local_path)
            self._present_paths.add(local_path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
        return local_path

    async def _download_to_cache(self, file_path: str, local_path: str, size: int | None = None) -> str:
        """Download ``file_path`` into ``local_path`` with lock + atomic publish.

        Claim the cross-process lock *after* acquiring a semaphore permit so waiters
        do not convoy behind a lock holder that has not started downloading yet.
        """
        if self._path_is_cached(local_path):
            return local_path

        deadline = time.monotonic() + self.lock_wait_timeout
        while time.monotonic() < deadline:
            if self._path_is_cached(local_path):
                return local_path

            async with self._get_semaphore():
                if self._path_is_cached(local_path):
                    return local_path
                claimed = self._try_claim_lock(local_path)
                if claimed:
                    try:
                        return await self._download_owned(file_path, local_path, size=size)
                    finally:
                        # Sync release — to_thread can be cancelled and leak the lock.
                        self._release_lock(local_path)

            # Another process holds the lock; wait outside the semaphore.
            try:
                remaining = max(0.05, deadline - time.monotonic())
                return await self._wait_for_cached_file(local_path, timeout=remaining)
            except TimeoutError:
                continue

        if self._path_is_cached(local_path):
            return local_path
        raise TimeoutError(f"Timed out waiting for cache file {local_path}")

    async def _dedupe_path(self, key: str, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Coalesce concurrent work for the same cache key on this event loop."""
        loop = asyncio.get_running_loop()
        if self._path_inflight_loop is not loop:
            self._path_inflight = {}
            self._path_inflight_loop = loop

        task = self._path_inflight.get(key)
        if task is None:
            task = asyncio.create_task(factory())
            self._path_inflight[key] = task

            def _clear(_t: asyncio.Task, k: str = key) -> None:
                self._path_inflight.pop(k, None)

            task.add_done_callback(_clear)
        return await task

    async def _ensure_cached_file(self, file_path: str, size: int | None = None) -> str:
        """Ensure ``file_path`` is on disk; dedupe concurrent downloads of the same key."""
        local_path = self.get_local_path(file_path)
        if self._path_is_cached(local_path):
            return local_path
        return await self._dedupe_path(
            file_path,
            lambda: self._download_to_cache(file_path, local_path, size=size),
        )

    async def _download_bytes_write_through(self, file_path: str, size: int | None = None) -> bytes:
        local_path = self.get_local_path(file_path)
        if self._path_is_cached(local_path):
            try:
                return await asyncio.to_thread(Path(local_path).read_bytes)
            except FileNotFoundError:
                # Stale presence mark (e.g. file removed under us) — discard and retry once.
                self._present_paths.discard(local_path)
                if self._path_is_cached(local_path):
                    return await asyncio.to_thread(Path(local_path).read_bytes)
        data = await self._fetch_bytes(file_path, size=size)
        self._schedule_write_behind(local_path, data)
        return data

    async def download_file_async(self, file_path: str, size: int | None = None) -> bytes:
        """Asynchronously download and return file content.

        With ``cache_files=True``, uses write-through: network bytes are returned
        immediately and the cache file is published atomically in the background.
        Concurrent callers for the same path share one in-flight fetch.
        """
        try:
            if self.cache_files:
                return await self._dedupe_path(
                    f"bytes:{file_path}",
                    lambda: self._download_bytes_write_through(file_path, size=size),
                )
            return await self._fetch_bytes(file_path, size=size)
        except Exception as e:
            raise RuntimeError(f"Error downloading file {file_path}: {e}") from e

    async def ensure_file_async(self, file_path: str, size: int | None = None) -> str:
        """Download to the mirrored cache path and return the local path (no full RAM buffer)."""
        if not self.cache_files:
            raise ValueError("ensure_file_async requires cache_files=True")
        try:
            return await self._ensure_cached_file(file_path, size=size)
        except Exception as e:
            raise RuntimeError(f"Error downloading file {file_path}: {e}") from e


def _storage_path(input_dir: Dir) -> str:
    """Prefer cloud URL over FUSE/local path so downloads hit object storage directly."""
    return str(input_dir.url or input_dir.path)


class StreamingRawDataset(Dataset):
    """Base class for streaming raw datasets.

    This class provides the core functionality for streaming raw data from a remote or local source,
    including file discovery, caching, and asynchronous downloading.

    To create a custom dataset, subclass this class and override the `setup` method
    to define the structure of your dataset items from the list of all discovered files.
    """

    def __init__(
        self,
        input_dir: str,
        cache_dir: str | None = None,
        indexer: BaseIndexer | None = None,
        storage_options: dict | None = None,
        cache_files: bool = False,
        recompute_index: bool = False,
        transform: Callable[[Any], Any] | None = None,
        max_concurrent_downloads: int | None = None,
        max_prefetch: int = 16,
        prefetch_cache_size: int | None = None,
        item_type: Literal["bytes", "path"] = "bytes",
        hedge_delay: float = 0.0,
        download_timeout: float = 120.0,
        range_parallel_threshold: int = _RANGE_PARALLEL_THRESHOLD,
        range_chunk_size: int = _RANGE_CHUNK_SIZE,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g., 's3://bucket/dataset/' or Studio connection path).
            cache_dir: Directory for caching files (optional).
            indexer: Custom file indexer (default: FileIndexer).
            storage_options: Cloud storage options.
            cache_files: Whether to cache files locally (default: False).
            recompute_index: Whether to recompute the index (default: False).
                If True, forces a re-scan of the input directory and rebuilds the index,
                ignoring any cached index files. This is useful when the dataset
                structure or files on the remote storage have changed.
            transform: A function to apply to each item. It receives ``bytes`` / ``list[bytes]``
                when ``item_type="bytes"``, or ``str`` / ``list[str]`` paths when ``item_type="path"``.
                Prefer C-level / GIL-releasing transforms, or decode in ``collate_fn``.
            max_concurrent_downloads: Per-worker in-flight download permits.
                ``None`` (default) applies the Stage 1 adaptive formula (size-aware
                aggregate budget from median file size; Little's-law arm only for
                medians below 1 MiB, split across workers with a per-worker floor
                of 8; single-process capped at 128). An explicit ``int`` sets
                **exactly** that many permits with no silent clamp — pass ``64``
                to keep the historical fixed cap.
            max_prefetch: Best-effort sequential look-ahead after each batch (default: 16;
                roughly ``2×`` a typical batch). Pass ``0`` to disable. Look-ahead is per
                DataLoader worker, but when ``num_workers > 1`` the scheduled amount is
                capped to ``min(max_prefetch, 64 // num_workers)`` so aggregate look-ahead
                stays near 64 items (e.g. w=2→16, w=8→8, w=16→4, w=32→2). Raise
                ``max_prefetch`` only helps when it is below that per-worker share.
            prefetch_cache_size: LRU entry cap for prefetched items. Defaults to
                ``max(max_prefetch * 2, max_prefetch)`` when prefetch is enabled.
            item_type: ``"bytes"`` (default) buffers each object in RAM; ``"path"`` downloads to
                the cache and returns local path(s). ``item_type="path"`` requires ``cache_files=True``.
            hedge_delay: Seconds before starting a hedged duplicate request for a slow GET
                (``0`` = off, default). Opt in with a positive delay for object-store p99
                stragglers. Only applied to small objects (~<8MB); large objects use per-chunk
                hedging for ranged downloads.
            download_timeout: Hang-protection floor in seconds for each batch gather
                (``0`` / disabled → no timeout). Defaults to ``120`` and coexists with the
                per-item fast path: individual GETs are never wrapped in ``wait_for``;
                ``_download_batch`` applies one ``wait_for`` around the gather using
                ``max`` of the per-item size-aware floors
                (``max(download_timeout, size / ~25MB/s * 3)``).
            range_parallel_threshold: Objects at least this large use parallel ranged GETs
                when the backend supports Range (``0`` disables; opt in with a positive
                byte threshold via the constructor).
            range_chunk_size: Part size for ranged parallel downloads.
        """
        if item_type not in ("bytes", "path"):
            raise ValueError(f"item_type must be 'bytes' or 'path', got {item_type!r}")
        if item_type == "path" and not cache_files:
            raise ValueError("item_type='path' requires cache_files=True")

        self.input_dir = _resolve_dir(input_dir)
        self._storage_path = _storage_path(self.input_dir)
        self.cache_files = cache_files
        self.item_type = item_type
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_prefetch = max(0, max_prefetch)
        self.hedge_delay = max(0.0, hedge_delay)
        self.download_timeout = max(0.0, download_timeout)
        if prefetch_cache_size is None:
            prefetch_cache_size = max(self.max_prefetch * 2, self.max_prefetch) if self.max_prefetch > 0 else 0
        self.prefetch_cache_size = max(0, prefetch_cache_size)

        self.cache_manager = CacheManager(
            self.input_dir,
            cache_dir,
            storage_options,
            cache_files,
            max_concurrent_downloads=max_concurrent_downloads,
            hedge_delay=self.hedge_delay,
            download_timeout=self.download_timeout,
            range_parallel_threshold=range_parallel_threshold,
            range_chunk_size=range_chunk_size,
        )
        self.indexer = indexer or FileIndexer()
        self.storage_options = storage_options or {}
        self.transform = transform

        self._prefetch_cache = _LRUCache(self.prefetch_cache_size)
        self._inflight: dict[int, asyncio.Task] = {}
        self._inflight_loop: asyncio.AbstractEventLoop | None = None
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._owner_pid = os.getpid()

        # Discover all files — prefer cloud URL over FUSE mount.
        self.files: list[FileMetadata] = self.indexer.build_or_load_index(
            self._storage_path,
            self.cache_manager.cache_dir,
            storage_options,
            recompute_index,
        )
        logger.info("Discovered %s files.", len(self.files))
        median = _median_file_bytes(self.files)
        self.cache_manager._median_file_bytes = median
        self._maybe_warn_tiny_files(median)

        # Transform the flat list of files into the desired item structure.
        self.items: list[FileMetadata] | list[list[FileMetadata]] = self.setup(self.files)
        if not isinstance(self.items, list):
            raise TypeError(f"The setup method must return a list, but returned {type(self.items)}")
        logger.info("Dataset setup with %s items.", len(self.items))

    def _maybe_warn_tiny_files(self, median: int | None = None) -> None:
        """Warn when index median size is tiny (request-overhead bound).

        ``median`` should be the value already computed for Stage 1 concurrency
        sizing so we do not rescan sizes just for the warning.
        """
        if median is None:
            median = self.cache_manager._median_file_bytes
        if median is None:
            return
        n_sized = sum(1 for f in self.files if f.size > 0)
        if n_sized < 8:
            return
        if median < _TINY_FILE_MEDIAN_BYTES:
            logger.warning(
                "Median file size is %.0f bytes. StreamingRawDataset is often request-overhead "
                "bound for tiny objects; consider litdata.optimize() + StreamingDataset for "
                "higher sustained throughput.",
                median,
            )

    def setup(self, files: list[FileMetadata]) -> list[FileMetadata] | list[list[FileMetadata]]:
        """Define the structure of the dataset from the list of discovered files.

        Override this method in a subclass to group or filter files into final dataset items.

        Args:
            files: A list of all `FileMetadata` objects discovered in the `input_dir`.

        Returns:
            The final structure of the dataset, which can be:
            - `List[FileMetadata]`: Each `FileMetadata` object is treated as a single item.
            - `List[List[FileMetadata]]`: Each inner list of `FileMetadata` objects is treated as a single item.
        """
        return files

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.items)

    def _ensure_post_fork_state(self) -> None:
        """Drop parent-process asyncio/prefetch state after DataLoader fork."""
        pid = os.getpid()
        if self._owner_pid == pid:
            return
        if _RAW_DEBUG:
            logger.warning(
                "raw-debug: reset dataset state after fork old_pid=%s new_pid=%s",
                self._owner_pid,
                pid,
            )
        self._inflight = {}
        self._inflight_loop = None
        self._prefetch_cache = _LRUCache(self.prefetch_cache_size)
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._owner_pid = pid
        self.cache_manager.reset_runtime_state()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize dataset config + index; strip loop/task/prefetch runtime.

        Allowlisted keys so accidental instance attrs (locks, write-behind refs)
        cannot enter the spawn pickle payload.
        """
        return {
            "input_dir": self.input_dir,
            "_storage_path": self._storage_path,
            "cache_files": self.cache_files,
            "item_type": self.item_type,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "max_prefetch": self.max_prefetch,
            "hedge_delay": self.hedge_delay,
            "download_timeout": self.download_timeout,
            "prefetch_cache_size": self.prefetch_cache_size,
            "cache_manager": self.cache_manager,
            "indexer": self.indexer,
            "storage_options": self.storage_options,
            "transform": self.transform,
            "files": self.files,
            "items": self.items,
            # Runtime — always fresh in the child (empty cache, no tasks/loops).
            "_prefetch_cache": _LRUCache(self.prefetch_cache_size),
            "_inflight": {},
            "_inflight_loop": None,
            "_prefetch_hits": 0,
            "_prefetch_misses": 0,
            "_owner_pid": None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._owner_pid = os.getpid()
        self._inflight = {}
        self._inflight_loop = None
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        if not isinstance(self._prefetch_cache, _LRUCache):
            self._prefetch_cache = _LRUCache(self.prefetch_cache_size)
        # CacheManager may be restored without its __setstate__ on some paths.
        cm = getattr(self, "cache_manager", None)
        if isinstance(cm, CacheManager):
            cm._downloader = None
            cm._downloader_pid = None
            cm._downloader_loop = None
            cm._semaphore = None
            cm._semaphore_loop = None
            cm._semaphore_permits = None
            cm._cached_permits = None
            cm._cached_permits_pid = None
            cm._path_inflight = {}
            cm._path_inflight_loop = None
            cm._range_executor = None
            cm._range_executor_pid = None

    def __getitem__(self, index: int) -> Any:
        """Get a single item by index."""
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of range for dataset with length {len(self)}")
        self._ensure_post_fork_state()
        if _RAW_DEBUG:
            logger.warning("raw-debug: __getitem__ pid=%s index=%s", os.getpid(), index)
        return _run_async(self._download_batch([index]))[0]

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Asynchronously download a batch of items by indices."""
        self._ensure_post_fork_state()
        if _RAW_DEBUG:
            logger.warning(
                "raw-debug: __getitems__ pid=%s n=%s head=%s",
                os.getpid(),
                len(indices),
                indices[:4],
            )
        return _run_async(self._download_batch(indices))

    def _item_size(self, index: int) -> int:
        item = self.items[index]
        if isinstance(item, FileMetadata):
            return item.size
        if isinstance(item, list):
            return sum(fm.size for fm in item)
        return 0

    def _batch_download_budget(self, indices: list[int]) -> float | None:
        """Return one hang-protection budget for a pending batch, or ``None`` if disabled.

        Uses the max per-item size-aware floor so large objects are not cut off, while
        keeping a single ``wait_for`` around the gather (not per object).

        With a download semaphore smaller than the batch (e.g. ``batch_size=64`` and
        fewer Stage 1 permits), downloads span multiple waves — the single-wave
        ``max(per-item)`` assumption is no longer the default. ``sum(sizes) /
        bandwidth * 3`` (using the per-stream ``_HEDGE_ASSUMED_BANDWIDTH_BPS`` floor)
        raises the timeout when aggregate transfer time exceeds that single-item
        budget.
        """
        base = self.cache_manager.download_timeout
        if base is None:
            return None
        budgets = [self.cache_manager._download_budget(self._item_size(i)) for i in indices]
        sized = [b for b in budgets if b is not None]
        per_item = max(sized) if sized else base
        total_size = sum(self._item_size(i) for i in indices)
        if total_size > 0:
            return max(per_item, total_size / _HEDGE_ASSUMED_BANDWIDTH_BPS * 3.0)
        return per_item

    async def _resolve_index(self, index: int) -> Any:
        """Return a cached/inflight/materialized item for ``index``."""
        task = self._inflight.get(index)
        if task is not None:
            if task.cancelled():
                self._inflight.pop(index, None)
            else:
                value = await task
                self._prefetch_cache.put(index, value)
                self._inflight.pop(index, None)
                return value
        value = await self._materialize_index(index)
        self._prefetch_cache.put(index, value)
        return value

    async def _download_batch(self, indices: list[int]) -> list[Any]:
        """Download/process indices, serving from prefetch cache when possible."""
        if _RAW_DEBUG:
            logger.warning("raw-debug: _download_batch start pid=%s n=%s", os.getpid(), len(indices))
        for index in indices:
            if not (0 <= index < len(self)):
                raise IndexError(f"Index {index} out of range for dataset with length {len(self)}")

        running_loop = asyncio.get_running_loop()
        if self._inflight_loop is not running_loop:
            self._inflight = {}
            self._inflight_loop = running_loop

        results: list[Any] = [None] * len(indices)
        pending_positions: dict[int, list[int]] = {}  # index -> [pos, ...]
        unique_pending: list[int] = []
        batch_hits = 0
        batch_misses = 0
        for pos, index in enumerate(indices):
            cached = self._prefetch_cache.get(index)
            if cached is not _MISS:
                results[pos] = cached
                batch_hits += 1
            else:
                batch_misses += 1
                if index not in pending_positions:
                    pending_positions[index] = []
                    unique_pending.append(index)
                pending_positions[index].append(pos)

        if _RAW_DEBUG:
            self._prefetch_hits += batch_hits
            self._prefetch_misses += batch_misses

        if unique_pending:
            # Largest-first so big objects overlap with smaller ones (LPT).
            unique_pending.sort(key=self._item_size, reverse=True)
            tasks = [asyncio.create_task(self._resolve_index(index)) for index in unique_pending]
            budget = self._batch_download_budget(unique_pending)
            try:
                if budget is None:
                    fetched = await asyncio.gather(*tasks)
                else:
                    # One wait_for for the whole batch — hang protection without per-item tax.
                    fetched = await asyncio.wait_for(asyncio.gather(*tasks), timeout=budget)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                # Plain gather (download_timeout=0): item-level TimeoutError — do not rewrite.
                if budget is None:
                    raise
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Cancelling _resolve_index wrappers does not cancel awaited _inflight
                # download tasks; hung prefetch entries would otherwise poison retries.
                # Prefetch tasks outside unique_pending survive here and self-heal when
                # a later batch needs them (one extra budget delay).
                for idx in unique_pending:
                    inflight = self._inflight.pop(idx, None)
                    if inflight is not None and not inflight.done():
                        inflight.cancel()
                        inflight.add_done_callback(_consume_task_exception)
                await asyncio.gather(*tasks, return_exceptions=True)
                raise TimeoutError(
                    f"Batch download timed out after {budget:.1f}s ({len(unique_pending)} pending indices)"
                ) from exc
            for index, value in zip(unique_pending, fetched):
                for pos in pending_positions[index]:
                    results[pos] = value

        if self.max_prefetch > 0:
            self._schedule_prefetch(indices)
        if _RAW_DEBUG:
            logger.warning(
                "raw-debug: _download_batch done pid=%s n=%s inflight=%s "
                "batch_hit=%s batch_miss=%s total_hit=%s total_miss=%s",
                os.getpid(),
                len(indices),
                len(self._inflight),
                batch_hits,
                batch_misses,
                self._prefetch_hits,
                self._prefetch_misses,
            )
        return results

    def _schedule_prefetch(self, indices: list[int]) -> None:
        """Best-effort sequential look-ahead into the LRU cache.

        With ``DataLoader(num_workers>1)``, each worker receives every N-th batch, so the
        next indices for *this* worker start at ``indices[0] + num_workers * batch_len``.
        Scheduled look-ahead uses :func:`_effective_prefetch` so aggregate in-flight cost
        stays near ``_AGGREGATE_PREFETCH_BUDGET`` rather than ``num_workers × max_prefetch``.
        """
        if self.max_prefetch <= 0 or not indices or not _looks_sequential(indices):
            return

        try:
            from torch.utils.data import get_worker_info

            info = get_worker_info()
            num_workers = info.num_workers if info is not None else 1
        except Exception:
            num_workers = 1

        effective = _effective_prefetch(self.max_prefetch, num_workers)
        if effective <= 0:
            return

        batch_len = len(indices)
        start = indices[0] + num_workers * batch_len
        end = min(start + effective, len(self.items))
        for index in range(start, end):
            if index in self._prefetch_cache or index in self._inflight:
                continue
            task = asyncio.create_task(self._prefetch_index(index))
            task.add_done_callback(_consume_prefetch_exception)
            self._inflight[index] = task

    async def _prefetch_index(self, index: int) -> Any:
        try:
            value = await self._materialize_index(index)
            self._prefetch_cache.put(index, value)
            return value
        finally:
            self._inflight.pop(index, None)

    async def _materialize_index(self, index: int) -> Any:
        item = self.items[index]
        if isinstance(item, FileMetadata):
            return await self._download_and_process_item(item.path, size=item.size)
        if isinstance(item, list):
            file_paths = [fm.path for fm in item]
            sizes = [fm.size for fm in item]
            return await self._download_and_process_group(file_paths, sizes=sizes)
        raise TypeError(f"Dataset items must be of type FileMetadata or List[FileMetadata], but found {type(item)}")

    async def _download_and_process_group(
        self, file_paths: list[str], sizes: Sequence[int | None] | None = None
    ) -> Any:
        """Download all files in a group, then apply the transform."""
        resolved_sizes: list[int | None] = list(sizes) if sizes is not None else [None] * len(file_paths)
        if self.item_type == "path":
            group_data: list[Any] = await asyncio.gather(
                *[self.cache_manager.ensure_file_async(path, size=sz) for path, sz in zip(file_paths, resolved_sizes)]
            )
        else:
            group_data = await asyncio.gather(
                *[self.cache_manager.download_file_async(path, size=sz) for path, sz in zip(file_paths, resolved_sizes)]
            )

        if self.transform:
            return await asyncio.to_thread(self.transform, group_data)
        return group_data

    async def _download_and_process_item(self, file_path: str, size: int | None = None) -> Any:
        """Download a single file and apply the transform."""
        if self.item_type == "path":
            data: Any = await self.cache_manager.ensure_file_async(file_path, size=size)
        else:
            data = await self.cache_manager.download_file_async(file_path, size=size)
        if self.transform:
            return await asyncio.to_thread(self.transform, data)
        return data

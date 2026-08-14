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

"""Experimental asyncio helpers for overlapping **remote chunk downloads**.

This is intentionally **not** an async ``StreamingDataLoader``. Training stays on
the sync ``for batch in loader`` API; decode stays on process workers. Asyncio is
only useful where we wait on network IO.

On by default for remote datasets; force off with ``LITDATA_ASYNC_CHUNK_PREFETCH=0``,
or force on locally with ``LITDATA_ASYNC_CHUNK_PREFETCH=1``.

Strategy:
  * Prefer ``Downloader.adownload_file`` (streaming to disk) when overridden.
  * Else ``Downloader.adownload_fileobj`` + atomic write.
  * Otherwise run sync ``download_chunk_from_index`` in ``asyncio.to_thread`` and
    ``gather`` several chunk indexes — still overlaps latency for blocking cloud SDKs.

Real-S3 note (Studio benches on ~67MB ImageNet chunks): gather concurrency needs
``max_pre_download >= 4`` to matter; the prepare thread raises the floor when
async prefetch is enabled (see :func:`async_prefetch_min_pre_download`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from typing import TYPE_CHECKING

from litdata.streaming.downloader import Downloader, obstore_usable

if TYPE_CHECKING:
    from litdata.streaming.config import ChunksConfig

logger = logging.getLogger("litdata.streaming.async_prefetch")

# Thread-local event loop so PrepareChunksThread does not pay asyncio.run()
# startup on every drain batch.
_THREAD_LOOPS = threading.local()

# Empirically, async gather on real S3 is bottlenecked when max_pre_download==2
# (only 1–2 in-flight). Floor to 4 when the feature is enabled unless overridden.
_DEFAULT_ASYNC_MIN_PRE_DOWNLOAD = 4


def async_chunk_prefetch_enabled(remote_dir: str | None = None) -> bool:
    """Return True when async chunk prefetch should run.

    Env ``LITDATA_ASYNC_CHUNK_PREFETCH`` wins when set (``1``/``0``). When unset,
    default **on** for remote datasets (real-S3: process+async is the fast path)
    and **off** for local-only caches.
    """
    raw = os.getenv("LITDATA_ASYNC_CHUNK_PREFETCH")
    if raw is not None:
        return bool(int(raw))
    return bool(remote_dir)


def async_prefetch_min_pre_download() -> int:
    """Minimum ``max_pre_download`` applied when async chunk prefetch is on.

    Override with ``LITDATA_ASYNC_MIN_PRE_DOWNLOAD`` (default 4). Set to ``0`` to
    disable the floor and keep the caller's ``max_pre_download`` unchanged.
    """
    raw = os.getenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD")
    if raw is None:
        return _DEFAULT_ASYNC_MIN_PRE_DOWNLOAD
    return max(0, int(raw))


def apply_async_pre_download_floor(max_pre_download: int, remote_dir: str | None = None) -> int:
    """Raise ``max_pre_download`` when async prefetch needs gather width."""
    if not async_chunk_prefetch_enabled(remote_dir):
        return max_pre_download
    floor = async_prefetch_min_pre_download()
    if floor <= 0 or max_pre_download >= floor:
        return max_pre_download
    logger.info(
        "Async chunk prefetch: raising max_pre_download %s → %s so "
        "asyncio.gather can overlap remote chunk downloads. "
        "Set LITDATA_ASYNC_MIN_PRE_DOWNLOAD=0 to keep the original value.",
        max_pre_download,
        floor,
    )
    return floor


def downloader_supports_adownload(downloader: Downloader | None) -> bool:
    """True when an async download path is available on ``downloader``.

    Native async methods are obstore-backed. If this process forked after the
    parent started tokio, report False and overlap via ``to_thread`` (boto3).
    When the parent never started obstore (index via boto3), workers keep the
    native path and lazy-init a fresh store.
    """
    if downloader is None:
        return False
    if hasattr(downloader, "_get_store") and not obstore_usable():
        return False
    cls = type(downloader)
    return (
        cls.adownload_file is not Downloader.adownload_file or cls.adownload_fileobj is not Downloader.adownload_fileobj
    )


def _remote_join(remote_dir: str, filename: str) -> str:
    """Join cloud URLs without ``os.path.join`` quirks on schemes."""
    return remote_dir.rstrip("/") + "/" + filename.lstrip("/")


async def _adownload_file_to_path(downloader: Downloader, remote_filepath: str, local_filepath: str) -> None:
    """Fetch ``remote_filepath`` asynchronously and publish atomically."""
    if os.path.exists(local_filepath):
        return
    # Prefer streaming-to-disk when the backend overrides adownload_file.
    if type(downloader).adownload_file is not Downloader.adownload_file:
        await downloader.adownload_file(remote_filepath, local_filepath)
        return
    data = await downloader.adownload_fileobj(remote_filepath)
    if data is None:
        raise NotImplementedError(
            f"{type(downloader).__name__}.adownload_fileobj returned None; "
            "cannot use async chunk prefetch for this backend."
        )
    tmp_path = downloader._temp_download_path(local_filepath)
    try:
        os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(data)
        downloader._atomic_replace(tmp_path, local_filepath)
    except Exception:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.remove(tmp_path)
        raise


async def _adownload_chunk_index(config: ChunksConfig, chunk_index: int) -> None:
    """Async download + decompress for one chunk index (mirrors sync config path)."""
    assert config._chunks is not None
    downloader = config._downloader
    if downloader is None:
        return

    chunk_filename = config._chunks[chunk_index]["filename"]
    local_chunkpath = os.path.join(config._cache_dir, chunk_filename)
    remote_chunkpath = _remote_join(downloader._remote_dir, chunk_filename)
    lazily_ref_counted = chunk_index not in config._shared_chunk_indexes
    lock_path = (
        local_chunkpath.replace(f".{config._compressor_name}", "") if config._compressor_name else local_chunkpath
    )

    if os.path.exists(local_chunkpath):
        config.try_decompress(local_chunkpath)
        if lazily_ref_counted:
            downloader._increment_local_lock(lock_path, chunk_index)
        return

    if lazily_ref_counted:
        downloader._increment_local_lock(lock_path, chunk_index)

    if downloader_supports_adownload(downloader):
        await _adownload_file_to_path(downloader, remote_chunkpath, local_chunkpath)
    else:
        # Overlap blocking SDK calls across threads when native async is unavailable.
        await asyncio.to_thread(downloader.download_chunk_from_index, chunk_index)

    config.try_decompress(local_chunkpath)


async def adownload_chunk_indexes(config: ChunksConfig, chunk_indexes: list[int]) -> None:
    """Download several chunk indexes concurrently (gather)."""
    if not chunk_indexes:
        return
    if len(chunk_indexes) == 1:
        await _adownload_chunk_index(config, chunk_indexes[0])
        return
    await asyncio.gather(*[_adownload_chunk_index(config, idx) for idx in chunk_indexes])


def _thread_event_loop() -> asyncio.AbstractEventLoop:
    """Return a reusable event loop for the current thread."""
    loop = getattr(_THREAD_LOOPS, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _THREAD_LOOPS.loop = loop
    return loop


def close_thread_event_loop() -> None:
    """Shut down the thread-local loop and its default executor.

    ``asyncio.to_thread`` (used when a downloader has no native async path) parks
    workers named ``asyncio_N`` on the loop's default executor. Leaving the loop
    open leaks those threads and trips the session thread-police on Windows.

    Executor shutdown is non-blocking (``wait=False``) so prepare-thread ``finally``
    never joins forever if a download worker is stuck. Threads are marked daemon and
    futures cancelled so orphans cannot keep the process alive or poison later
    DataLoader forks under pytest-xdist.
    """
    loop = getattr(_THREAD_LOOPS, "loop", None)
    _THREAD_LOOPS.loop = None
    if loop is None or loop.is_closed():
        return
    try:
        if not loop.is_running():
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            # Prefer non-blocking teardown over ``shutdown_default_executor()``,
            # which joins executor threads and can hang prepare-thread exit.
            executor = getattr(loop, "_default_executor", None)
            if executor is not None:
                with contextlib.suppress(Exception):
                    loop._default_executor = None
                    for thread in getattr(executor, "_threads", set()):
                        thread.daemon = True
                    # cancel_futures is 3.9+; litdata already requires newer Python.
                    executor.shutdown(wait=False, cancel_futures=True)
    finally:
        with contextlib.suppress(Exception):
            loop.close()


def download_chunk_indexes_concurrently(config: ChunksConfig, chunk_indexes: list[int]) -> None:
    """Sync entry point for ``PrepareChunksThread``: run ``adownload_chunk_indexes``."""
    if not chunk_indexes:
        return
    if len(chunk_indexes) == 1:
        config.download_chunk_from_index(chunk_indexes[0])
        return
    # Reuse a per-thread loop instead of asyncio.run() (which creates+closes a loop
    # on every prefetch batch — measurable overhead under high chunk churn).
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        # Already inside an event loop (tests / nested callers): block via a bridge.
        import concurrent.futures

        def _run() -> None:
            try:
                _thread_event_loop().run_until_complete(adownload_chunk_indexes(config, chunk_indexes))
            finally:
                close_thread_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_run).result()
        return
    loop = _thread_event_loop()
    loop.run_until_complete(adownload_chunk_indexes(config, chunk_indexes))

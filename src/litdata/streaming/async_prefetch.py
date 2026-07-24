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
the sync ``for batch in loader`` API; decode stays on process workers or
``use_threading``. Asyncio is only useful where we wait on network IO.

Enable in ``PrepareChunksThread`` with ``LITDATA_ASYNC_CHUNK_PREFETCH=1``.

Strategy:
  * Prefer ``Downloader.adownload_fileobj`` when a subclass overrides it (obstore).
  * Otherwise run sync ``ChunksConfig.download_chunk_from_index`` in
    ``asyncio.to_thread`` and ``gather`` several chunk indexes — still overlaps
    latency for blocking cloud SDKs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING

from litdata.streaming.downloader import Downloader

if TYPE_CHECKING:
    from litdata.streaming.config import ChunksConfig


def async_chunk_prefetch_enabled() -> bool:
    """Return True when experimental async chunk prefetch is enabled."""
    return bool(int(os.getenv("LITDATA_ASYNC_CHUNK_PREFETCH", "0")))


def downloader_supports_adownload(downloader: Downloader | None) -> bool:
    """True when ``adownload_fileobj`` is overridden (base impl is a no-op)."""
    if downloader is None:
        return False
    return type(downloader).adownload_fileobj is not Downloader.adownload_fileobj


async def _adownload_file_to_path(downloader: Downloader, remote_filepath: str, local_filepath: str) -> None:
    """Fetch ``remote_filepath`` via ``adownload_fileobj`` and publish atomically."""
    if os.path.exists(local_filepath):
        return
    data = await downloader.adownload_fileobj(remote_filepath)
    if data is None:
        raise NotImplementedError(
            f"{type(downloader).__name__}.adownload_fileobj returned None; "
            "cannot use async chunk prefetch for this backend."
        )
    tmp_path = downloader._temp_download_path(local_filepath)
    try:
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
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
    remote_chunkpath = os.path.join(downloader._remote_dir, chunk_filename)
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


def download_chunk_indexes_concurrently(config: ChunksConfig, chunk_indexes: list[int]) -> None:
    """Sync entry point for ``PrepareChunksThread``: run ``adownload_chunk_indexes``."""
    if not chunk_indexes:
        return
    if len(chunk_indexes) == 1:
        config.download_chunk_from_index(chunk_indexes[0])
        return
    asyncio.run(adownload_chunk_indexes(config, chunk_indexes))

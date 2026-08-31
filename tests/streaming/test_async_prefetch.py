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

"""Tests for experimental asyncio chunk-prefetch helpers (not an async DataLoader)."""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

from litdata.streaming import Cache
from litdata.streaming.async_prefetch import (
    adownload_chunk_indexes,
    apply_async_pre_download_floor,
    async_chunk_prefetch_enabled,
    async_download_concurrency,
    async_prefetch_min_pre_download,
    download_chunk_indexes_concurrently,
    downloader_supports_adownload,
    downloader_supports_aupload,
)
from litdata.streaming.config import ChunksConfig
from litdata.streaming.downloader import Downloader, S3Downloader
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.reader import PrepareChunksThread
from litdata.streaming.serializers import _get_serializers
from litdata.utilities.env import _DistributedEnv


class _FakeAsyncDownloader(Downloader):
    """Downloader with simulated network latency and a real ``adownload_fileobj``."""

    def __init__(self, remote_dir: str, cache_dir: str, chunks: list, delay_s: float = 0.05) -> None:
        super().__init__(remote_dir, cache_dir, chunks, {})
        self.delay_s = delay_s
        self.calls: list[str] = []

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        time.sleep(self.delay_s)
        self.calls.append(remote_filepath)
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        with open(local_filepath, "wb") as f:
            f.write(b"x" * 16)

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        await asyncio.sleep(self.delay_s)
        self.calls.append(remote_filepath)
        return b"x" * 16


def _seed_local_chunks(tmpdir, n_chunks: int = 4, chunk_size: int = 4) -> str:
    cache_dir = os.path.join(tmpdir, "local")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_chunks * chunk_size):
        cache[i] = i
    cache.done()
    cache.merge(1)
    return cache_dir


def test_async_chunk_prefetch_disabled_by_default_for_local(monkeypatch):
    monkeypatch.delenv("LITDATA_ASYNC_CHUNK_PREFETCH", raising=False)
    assert async_chunk_prefetch_enabled() is False
    assert async_chunk_prefetch_enabled(remote_dir=None) is False


def test_async_chunk_prefetch_default_on_for_remote(monkeypatch):
    monkeypatch.delenv("LITDATA_ASYNC_CHUNK_PREFETCH", raising=False)
    assert async_chunk_prefetch_enabled(remote_dir="s3://bucket/data") is True


def test_async_chunk_prefetch_env_overrides_remote_default(monkeypatch):
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "0")
    assert async_chunk_prefetch_enabled(remote_dir="s3://bucket/data") is False
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    assert async_chunk_prefetch_enabled(remote_dir=None) is True


def test_apply_async_pre_download_floor(monkeypatch):
    monkeypatch.delenv("LITDATA_ASYNC_CHUNK_PREFETCH", raising=False)
    assert apply_async_pre_download_floor(2) == 2
    assert apply_async_pre_download_floor(2, remote_dir="s3://b/x") == 4

    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    monkeypatch.delenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", raising=False)
    assert async_prefetch_min_pre_download() == 4
    assert apply_async_pre_download_floor(2) == 4
    assert apply_async_pre_download_floor(8) == 8

    monkeypatch.setenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", "0")
    assert apply_async_pre_download_floor(2) == 2


def test_downloader_supports_adownload_detects_override():
    assert downloader_supports_adownload(None) is False
    base = Downloader("remote", "cache", [], {})
    assert downloader_supports_adownload(base) is False
    fake = _FakeAsyncDownloader("remote", "cache", [])
    assert downloader_supports_adownload(fake) is True
    assert downloader_supports_aupload(None) is False
    assert downloader_supports_aupload(base) is False
    assert downloader_supports_aupload(S3Downloader("s3://bucket", "cache", [])) is True


def test_downloader_supports_adownload_false_after_obstore_fork(monkeypatch, tmpdir):
    from litdata.streaming import downloader as downloader_mod

    monkeypatch.setattr(downloader_mod, "_OBSTORE_AVAILABLE", True)
    monkeypatch.setattr(downloader_mod, "_OBSTORE_INIT_PID", os.getpid() + 1)
    dl = S3Downloader("s3://bucket", str(tmpdir), [])
    assert downloader_supports_adownload(dl) is False


class _ConfigShim:
    """Minimal config surface for ``adownload_chunk_indexes`` overlap test."""

    def __init__(self, cache_dir: str, chunks: list, downloader: Downloader) -> None:
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._downloader = downloader
        self._shared_chunk_indexes: set[int] = set()
        self._compressor_name = None

    def try_decompress(self, local_chunkpath: str) -> None:
        return None


def test_adownload_chunk_indexes_overlap_latency(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir)
    chunks = [{"filename": f"chunk-{i:08d}.bin", "chunk_bytes": 16} for i in range(3)]
    fake = _FakeAsyncDownloader("remote://bucket", cache_dir, chunks, delay_s=0.05)
    cfg = _ConfigShim(cache_dir, chunks, fake)

    indexes = [0, 1, 2]
    t0 = time.perf_counter()
    asyncio.run(adownload_chunk_indexes(cfg, indexes))  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0

    # Three 50ms downloads overlapped should finish well under serial 150ms.
    assert elapsed < 0.12
    assert len(fake.calls) == 3
    for idx in indexes:
        assert os.path.exists(os.path.join(cache_dir, chunks[idx]["filename"]))


def test_async_download_concurrency_caps_gather(monkeypatch):
    monkeypatch.delenv("LITDATA_ASYNC_DOWNLOAD_CONCURRENCY", raising=False)
    assert async_download_concurrency(3) == 3
    assert async_download_concurrency(32) == 8
    monkeypatch.setenv("LITDATA_ASYNC_DOWNLOAD_CONCURRENCY", "2")
    assert async_download_concurrency(32) == 2
    monkeypatch.setenv("LITDATA_ASYNC_DOWNLOAD_CONCURRENCY", "1")
    assert async_download_concurrency(8) == 1


def test_download_chunk_indexes_concurrently_single_uses_sync(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    cache_dir = _seed_local_chunks(tmpdir, n_chunks=2, chunk_size=4)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    # Already local — no downloader needed.
    download_chunk_indexes_concurrently(cfg, [0])


def test_prepare_chunks_thread_applies_async_floor(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    monkeypatch.delenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", raising=False)
    cache_dir = _seed_local_chunks(tmpdir, n_chunks=2, chunk_size=2)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    thread = PrepareChunksThread(cfg, MagicMock(), _DistributedEnv(1, 0, 1), max_pre_download=2)
    assert thread._max_pre_download == 4


def test_prepare_chunks_thread_batches_when_async_enabled(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    # Keep the caller's value when already above the floor.
    monkeypatch.setenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", "0")
    cache_dir = _seed_local_chunks(tmpdir, n_chunks=4, chunk_size=2)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    thread = PrepareChunksThread(cfg, MagicMock(), _DistributedEnv(1, 0, 1), max_pre_download=4)
    called: list[list[int]] = []

    def _capture(indexes: list[int]) -> None:
        called.append(list(indexes))
        for idx in indexes:
            # Simulate ready files already present.
            thread._finalize_downloaded_chunk(idx, existed=True)

    monkeypatch.setattr(thread, "_download_chunk_indexes", _capture)
    thread.download([0, 1, 2])
    thread.stop()
    thread.run()
    assert called
    # First batch should include multiple indexes when async prefetch is on.
    assert len(called[0]) >= 2


def test_should_start_download_refills_in_gather_batches(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    monkeypatch.setenv("LITDATA_ASYNC_MIN_PRE_DOWNLOAD", "0")
    cache_dir = _seed_local_chunks(tmpdir, n_chunks=4, chunk_size=2)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    cfg._remote_dir = "r2://bucket/data"
    thread = PrepareChunksThread(cfg, MagicMock(), _DistributedEnv(1, 0, 1), max_pre_download=8)
    assert thread._should_start_download(over_budget=False) is True
    thread._pre_download_counter = 1
    assert thread._should_start_download(over_budget=False) is True
    thread._pre_download_counter = 7
    assert thread._should_start_download(over_budget=False) is True
    thread._pre_download_counter = 8
    assert thread._should_start_download(over_budget=False) is False
    thread._pre_download_counter = 4
    assert thread._should_start_download(over_budget=False) is True


def test_no_use_asyncio_on_streaming_dataloader():
    """Guardrail: asyncio is not a StreamingDataLoader public mode."""
    import inspect

    from litdata.streaming.dataloader import StreamingDataLoader

    sig = inspect.signature(StreamingDataLoader.__init__)
    assert "use_asyncio" not in sig.parameters
    assert "use_threading" not in sig.parameters

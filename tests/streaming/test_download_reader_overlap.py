"""Tests for PrepareChunksThread ↔ ItemLoader handoff optimizations."""

from __future__ import annotations

import os
import threading
from time import perf_counter, sleep
from unittest.mock import MagicMock

import pytest

from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming import Cache
from litdata.streaming.config import ChunksConfig
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.downloader import S3Downloader
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.reader import _DEFAULT_TIMEOUT, PrepareChunksThread
from litdata.utilities.env import _DistributedEnv


def test_prefetch_side_polls_are_nonblocking_when_download_work_available(monkeypatch, tmpdir):
    """Force/delete polls must use timeout=0 while the prefetch buffer can still accept downloads."""
    cache_dir = str(tmpdir / "cache")
    os.makedirs(cache_dir)
    config = MagicMock(spec=ChunksConfig)
    config.num_bytes = 1024
    config._cache_dir = cache_dir
    config.download_chunk_from_index = MagicMock()
    item_loader = MagicMock()
    env = _DistributedEnv(1, 0, 1)

    thread = PrepareChunksThread(config, item_loader, env, max_cache_size=10_000, max_pre_download=2)
    observed: list[tuple[str, float]] = []

    def fake_force(timeout: float = _DEFAULT_TIMEOUT) -> None:
        observed.append(("force", timeout))

    def fake_delete(timeout: float = _DEFAULT_TIMEOUT) -> None:
        observed.append(("delete", timeout))
        # Stop after first loop body so the test does not spin forever.
        thread._force_stop_event.set()

    monkeypatch.setattr(thread, "_force_download", fake_force)
    monkeypatch.setattr(thread, "_maybe_delete_chunks", fake_delete)
    # Keep download queue empty so `_get_from_queue` returns quickly with default timeout,
    # but counter is still below max so side polls should be non-blocking.
    thread.run()

    assert ("force", 0.0) in observed
    assert ("delete", 0.0) in observed


def test_prefetch_side_polls_use_short_timeout_when_buffer_full(monkeypatch, tmpdir):
    cache_dir = str(tmpdir / "cache")
    os.makedirs(cache_dir)
    config = MagicMock(spec=ChunksConfig)
    config.num_bytes = 1024
    config._cache_dir = cache_dir
    item_loader = MagicMock()
    env = _DistributedEnv(1, 0, 1)

    thread = PrepareChunksThread(config, item_loader, env, max_cache_size=10_000, max_pre_download=2)
    thread._pre_download_counter = 2  # buffer full
    observed: list[tuple[str, float]] = []

    def fake_force(timeout: float = _DEFAULT_TIMEOUT) -> None:
        observed.append(("force", timeout))

    def fake_delete(timeout: float = _DEFAULT_TIMEOUT) -> None:
        observed.append(("delete", timeout))
        thread._force_stop_event.set()

    monkeypatch.setattr(thread, "_force_download", fake_force)
    monkeypatch.setattr(thread, "_maybe_delete_chunks", fake_delete)
    thread.run()

    assert ("force", _DEFAULT_TIMEOUT) in observed
    assert ("delete", _DEFAULT_TIMEOUT) in observed
    assert all(timeout != 5 for _, timeout in observed)


def test_item_loader_unblocks_on_ready_event(tmpdir):
    """load_item_from_chunk should wake from the ready Event without a long busy-wait."""
    cache = Cache(str(tmpdir), chunk_size=2)
    for i in range(4):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir))
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)

    from litdata.streaming.sampler import ChunkedIndex

    chunk_filepath, begin, filesize = dataset.cache._reader.config[ChunkedIndex(2, chunk_index=1)]
    # Hide the file and install a readiness Event that becomes ready shortly.
    hidden = chunk_filepath + ".hidden"
    os.rename(chunk_filepath, hidden)
    ready = threading.Event()
    loader.set_chunk_ready_provider(lambda _idx: ready)

    def restore_and_signal() -> None:
        sleep(0.05)
        os.rename(hidden, chunk_filepath)
        ready.set()

    threading.Thread(target=restore_and_signal, daemon=True).start()
    t0 = perf_counter()
    item = loader.load_item_from_chunk(2, 1, chunk_filepath, begin, filesize)
    elapsed = perf_counter() - t0
    assert item == 2
    # Should return well under the old 0.1s poll granularity * multiple iterations.
    assert elapsed < 1.0


def test_item_loader_clears_stale_ready_event(tmpdir):
    """A readiness Event left set after deletion must not busy-spin the wait loop."""
    cache = Cache(str(tmpdir), chunk_size=2)
    for i in range(4):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir))
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)

    from litdata.streaming.sampler import ChunkedIndex

    chunk_filepath, begin, filesize = dataset.cache._reader.config[ChunkedIndex(2, chunk_index=1)]
    hidden = chunk_filepath + ".hidden"
    os.rename(chunk_filepath, hidden)

    ready = threading.Event()
    ready.set()  # stale: previously ready, file now missing
    loader.set_chunk_ready_provider(lambda _idx: ready)

    def restore_after_clear() -> None:
        # Wait until the loader clears the stale signal, then restore + re-signal.
        for _ in range(50):
            if not ready.is_set():
                break
            sleep(0.02)
        os.rename(hidden, chunk_filepath)
        ready.set()

    threading.Thread(target=restore_after_clear, daemon=True).start()
    item = loader.load_item_from_chunk(2, 1, chunk_filepath, begin, filesize)
    assert item == 2
    assert ready.is_set()


def test_s3_downloader_does_not_publish_partial_final_path(monkeypatch, tmpdir):
    """Atomic rename means readers never see a truncated final path mid-download."""
    from litdata.streaming import downloader as downloader_mod

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.client = self

        def download_file(self, bucket, key, filename, ExtraArgs=None, Config=None):
            # Simulate a slow write: create a partial file, pause, then finish.
            with open(filename, "wb") as f:
                f.write(b"partial")
                f.flush()
                sleep(0.05)
                f.write(b"-complete")

    monkeypatch.setattr(downloader_mod, "S3Client", FakeClient)
    local_filepath = os.path.join(tmpdir, "chunk.bin")
    seen_final_sizes: list[int] = []
    stop = threading.Event()

    def watcher() -> None:
        while not stop.is_set():
            if os.path.exists(local_filepath):
                seen_final_sizes.append(os.stat(local_filepath).st_size)
            sleep(0.005)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    try:
        dl = S3Downloader("s3://bucket", str(tmpdir), [])
        dl.download_file("s3://bucket/chunk.bin", local_filepath)
    finally:
        stop.set()
        t.join(timeout=1)

    assert os.path.exists(local_filepath)
    with open(local_filepath, "rb") as f:
        assert f.read() == b"partial-complete"
    # Final path should only ever appear at the completed size (never "partial" alone).
    assert all(size == len(b"partial-complete") for size in seen_final_sizes)


@pytest.mark.skipif(not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")
def test_try_decompress_publishes_complete_bin(tmpdir):
    """try_decompress must publish a .bin that satisfies chunk_bytes after atomic replace."""
    import shutil

    from litdata.streaming.serializers import _get_serializers

    cache_dir = str(tmpdir / "cache")
    os.makedirs(cache_dir)
    cache = Cache(cache_dir, chunk_size=10, compression="zstd")
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    compressed = [f for f in os.listdir(cache_dir) if f.endswith(".zstd.bin")][0]
    compressed_path = os.path.join(cache_dir, compressed)
    target_path = compressed_path.replace(".zstd.bin", ".bin")
    backup_path = compressed_path + ".bak"
    shutil.copy(compressed_path, backup_path)
    if os.path.exists(target_path):
        os.remove(target_path)

    config = ChunksConfig.load(cache_dir, _get_serializers({}), None)
    # try_decompress may delete the compressed source; restore from backup for a clean call.
    if not os.path.exists(compressed_path):
        shutil.copy(backup_path, compressed_path)
    config.try_decompress(compressed_path)
    assert os.path.exists(target_path)
    chunk_index = config._get_chunk_index_from_filename(compressed)
    assert os.stat(target_path).st_size >= int(config._chunks[chunk_index]["chunk_bytes"])


def test_tokens_loader_accepts_exact_filesize(tmpdir):
    """Exact-size chunk files must be considered ready (not strictly greater)."""
    from litdata.streaming.item_loader import BaseItemLoader

    class _Loader(BaseItemLoader):
        def generate_intervals(self):
            return []

        def pre_load_chunk(self, chunk_index, chunk_filepath):
            return None

        def load_item_from_chunk(self, index, chunk_index, chunk_filepath, begin, filesize_bytes):
            return None

        def delete(self, chunk_index, chunk_filepath):
            return None

        def encode_data(self, data, sizes, flattened):
            return b"", None

    loader = _Loader()
    path = os.path.join(tmpdir, "chunk.bin")
    payload = b"\x00" * 16
    with open(path, "wb") as f:
        f.write(payload)
    loader._wait_until_chunk_ready(0, path, filesize_bytes=len(payload))

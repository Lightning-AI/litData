"""POSIX temporal windows retain ownership, concurrency and lifecycle contracts."""

import asyncio
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from litdata import StreamingDataset, TemporalArrayLoader
from litdata.streaming import Cache
from litdata.streaming.window_mmap import _advise_window_ranges, _WindowMMapCache


@pytest.fixture(autouse=True)
def shutdown_cloud_runner():
    yield
    from litdata.raw import dataset as raw_dataset

    runner = raw_dataset._RUNNER
    raw_dataset._shutdown_runner_before_fork()
    if runner is not None:
        runner._executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def temporal(tmp_path, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "1")
    records = [{"x": np.arange(80, dtype=np.float32).reshape(20, 4) + i * 100} for i in range(6)]
    writer = Cache(str(tmp_path), chunk_size=1, item_loader=TemporalArrayLoader())
    for i, record in enumerate(records):
        writer[i] = record
    writer.done()
    writer.merge()
    dataset = StreamingDataset(str(tmp_path), item_loader=TemporalArrayLoader())
    dataset.frame_counts
    yield dataset, records, tmp_path
    dataset.cache._reader._window_reader.close()


def test_sync_posix_windows_avoid_cloud_runner_and_keep_owned_outputs(temporal, monkeypatch):
    dataset, records, _ = temporal

    def unexpected_runner():
        pytest.fail("A synchronous POSIX window should not use the cloud event loop")

    monkeypatch.setattr("litdata.raw.dataset._get_loop_runner", unexpected_runner)
    reader = dataset.cache._reader._window_reader
    reader._maps.keep = 1
    first = dataset.read_window(0, 2, 7)["x"]
    for i in range(1, len(records)):
        np.testing.assert_array_equal(dataset.read_window(i, 3, 4)["x"], records[i]["x"][3:7])
    reader.close()
    np.testing.assert_array_equal(first, records[0]["x"][2:9])
    assert first.flags.owndata
    assert first.flags.writeable
    first[:] = -1


def test_concurrent_windows_survive_mapping_eviction(temporal):
    dataset, records, _ = temporal
    reader = dataset.cache._reader._window_reader
    reader._maps.keep = 1

    def fetch(n):
        i = n % len(records)
        value = dataset.read_window(i, n % 10, 5)["x"]
        np.testing.assert_array_equal(value, records[i]["x"][n % 10 : n % 10 + 5])
        value[:] = -1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, range(96)))
    assert len(reader._maps._maps) <= 1
    np.testing.assert_array_equal(dataset.read_window(0, 0, 20)["x"], records[0]["x"])


@pytest.mark.asyncio
async def test_async_cancellation_does_not_unmap_an_active_decoder(temporal, monkeypatch):
    dataset, _, _ = temporal
    reader = dataset.cache._reader._window_reader
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    decode = reader.temporal.decode_groups

    def slow_decode(*args):
        started.set()
        assert release.wait(timeout=10)
        try:
            return decode(*args)
        finally:
            finished.set()

    monkeypatch.setattr(reader.temporal, "decode_groups", slow_decode)
    task = asyncio.create_task(dataset.aread_window(0, 1, 3))
    try:
        for _ in range(1000):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()  # the event loop remained responsive
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        reader.close()  # leases must protect the still-running worker thread
        assert reader._maps._maps
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        # The thread releases its mapping lease immediately after decoding.
        for _ in range(1000):
            if not reader._maps._maps:
                break
            await asyncio.sleep(0.001)
    assert not reader._maps._maps


def test_invalid_or_empty_projection_does_not_map_payload(temporal):
    dataset, _, _ = temporal
    reader = dataset.cache._reader._window_reader
    assert dataset.read_window(0, 0, 3, []) == {}
    with pytest.raises(KeyError):
        dataset.read_window(0, 0, 3, ["missing"])
    with pytest.raises(IndexError):
        dataset.read_window(0, 19, 3, [])
    assert not reader._maps._maps


def test_posix_escape_hatch_keeps_the_buffered_fallback(temporal, monkeypatch):
    _, records, root = temporal
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    dataset = StreamingDataset(str(root), item_loader=TemporalArrayLoader())
    np.testing.assert_array_equal(dataset.read_window(0, 1, 3)["x"], records[0]["x"][1:4])
    assert dataset.posix_fast is None
    assert not dataset.cache._reader._window_reader.posix_windows


@pytest.mark.skipif(not hasattr(mmap, "MADV_WILLNEED"), reason="madvise is unavailable")
def test_window_prefetch_is_page_bounded_not_whole_chunk():
    class RecordingMapping:
        def __init__(self):
            self.calls = []

        def __len__(self):
            return 1024 * mmap.PAGESIZE + 17

        def madvise(self, *args):
            self.calls.append(args)

    mapping = RecordingMapping()
    _advise_window_ranges(mapping, [(16 * mmap.PAGESIZE + 3, 100), (len(mapping) - 10, 10)])
    assert mapping.calls == [
        (mmap.MADV_WILLNEED, 16 * mmap.PAGESIZE, mmap.PAGESIZE),
        (mmap.MADV_WILLNEED, 1024 * mmap.PAGESIZE, 17),
    ]


def test_window_prefetch_respects_reader_memory_policy(temporal, monkeypatch):
    dataset, _, _ = temporal
    reader = dataset.cache._reader._window_reader
    reader._posix_willneed = False

    def unexpected_advice(*args):
        pytest.fail("Disabled POSIX prefetch must not advise temporal ranges")

    monkeypatch.setattr("litdata.streaming.window._advise_window_ranges", unexpected_advice)
    dataset.read_window(0, 0, 3)


def test_truncated_chunk_fails_before_mapping(temporal):
    dataset, _, root = temporal
    config = dataset.cache._reader.config
    path = root / config._chunks[0]["filename"]
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 1)
    with pytest.raises(OSError, match="Chunk size changed"):
        dataset.read_window(0, 0, 3)


def test_mapping_leases_defer_close_and_release_descriptors(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"other")
    cache = _WindowMMapCache(keep=1)
    with cache.acquire(str(first), 5) as a:
        with cache.acquire(str(second), 5) as b:
            cache.close()
            assert a[:] == b"first"
            assert b[:] == b"other"
        assert b.closed
        assert not a.closed
    assert a.closed
    assert not cache._maps


def test_mapping_cache_resets_after_process_change(tmp_path, monkeypatch):
    path = tmp_path / "chunk"
    path.write_bytes(b"hello")
    cache = _WindowMMapCache()
    with cache.acquire(str(path), 5) as old:
        assert old[:] == b"hello"
    parent = os.getpid()
    monkeypatch.setattr("litdata.streaming.window_mmap.os.getpid", lambda: parent + 1)
    with cache.acquire(str(path), 5) as current:
        assert old.closed
        assert current[:] == b"hello"
    cache.close()

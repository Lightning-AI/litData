import logging
import os
import shutil
from time import sleep
from unittest import mock

import pytest

from litdata.streaming import reader
from litdata.streaming.cache import Cache
from litdata.streaming.config import ChunkedIndex
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.reader import _END_TOKEN, PrepareChunksThread, _get_folder_size
from litdata.streaming.resolver import Dir
from litdata.utilities.env import _DistributedEnv
from tests.streaming.utils import filter_lock_files, get_lock_files


def test_reader_chunk_removal(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    # we don't care about the max cache size here (so very large number)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=28020, compression="zstd")

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    shutil.copytree(cache_dir, remote_dir)

    shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        index = ChunkedIndex(*cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(filter_lock_files(os.listdir(cache_dir))) == 14
    assert len(get_lock_files(os.listdir(cache_dir))) == 0

    # Let's test if cache actually respects the max cache size
    # each chunk is 40 bytes if it has 2 items
    # a chunk with only 1 item is 24 bytes (values determined by checking actual chunk sizes)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=90)

    shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        # we expect at max 3 files to be present (2 chunks and 1 index file)
        # why 2 chunks? Bcoz max cache size is 90 bytes and each chunk is 40 bytes or 24 bytes (1 item)
        # So any additional chunk will go over the max cache size
        assert len(filter_lock_files(os.listdir(cache_dir))) <= 3
        index = ChunkedIndex(*cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(filter_lock_files(os.listdir(cache_dir))) in [1, 2, 3]


def test_reader_chunk_removal_compressed(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    # we don't care about the max cache size here (so very large number)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=28020, compression="zstd")

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    shutil.copytree(cache_dir, remote_dir)

    shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        index = ChunkedIndex(*cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(filter_lock_files(os.listdir(cache_dir))) == 14
    assert len(get_lock_files(os.listdir(cache_dir))) == 0
    # Let's test if cache actually respects the max cache size
    # each chunk is 40 bytes if it has 2 items
    # a chunk with only 1 item is 24 bytes (values determined by checking actual chunk sizes)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=90, compression="zstd")

    shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        # we expect at max 3 files to be present (2 chunks and 1 index file)
        # why 2 chunks? Bcoz max cache size is 90 bytes and each chunk is 40 bytes or 24 bytes (1 item)
        # So any additional chunk will go over the max cache size
        assert len(filter_lock_files(os.listdir(cache_dir))) <= 3
        index = ChunkedIndex(*cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(filter_lock_files(os.listdir(cache_dir))) in [1, 2, 3]


@mock.patch("litdata.streaming.reader._DEBUG", True)
def test_get_folder_size(tmpdir, caplog):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    cache = Cache(cache_dir, chunk_size=10)
    for i in range(100):
        cache[i] = i

    cache.done()
    cache.merge()

    config = cache._reader._try_load_config()

    with caplog.at_level(logging.WARNING):
        cache_size = _get_folder_size(cache_dir, config)

    actual_cache_size = 0
    for file_name in filter_lock_files(os.listdir(cache_dir)):
        if file_name in config.filename_to_size_map:
            actual_cache_size += os.path.getsize(os.path.join(cache_dir, file_name))

    assert cache_size == actual_cache_size
    assert len(caplog.messages) == 0

    # add some extra files to the cache directory
    file_names = ["sample.txt", "sample.bin", "sample.bin.tmp", "sample.bin.ABCD", "sample.binEFGH"]
    for file_name in file_names:
        with open(os.path.join(cache_dir, file_name), "w") as f:
            f.write("sample")
        if file_name != "sample.txt":
            actual_cache_size += os.path.getsize(os.path.join(cache_dir, file_name))

    with caplog.at_level(logging.WARNING):
        cache_size = _get_folder_size(cache_dir, config)

    assert cache_size == actual_cache_size
    assert len(caplog.messages) == 1

    # Assert that a warning was logged
    assert any(
        "Ignoring 'sample.txt': This file doesn't appear to be a valid chunk file" in record.message
        for record in caplog.records
    ), "Expected warning about an invalid chunk file was not logged"


def test_prepare_chunks_thread_eviction(tmpdir, monkeypatch):
    monkeypatch.setattr(reader, "_LONG_DEFAULT_TIMEOUT", 0.1)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    cache._reader._try_load_config()

    assert len(os.listdir(cache_dir)) == 14

    thread = PrepareChunksThread(
        cache._reader.config, item_loader=PyTreeLoader(), distributed_env=_DistributedEnv(1, 1, 1), max_cache_size=10000
    )
    assert not thread._delete_chunks_when_processed

    thread = PrepareChunksThread(
        cache._reader.config, item_loader=PyTreeLoader(), distributed_env=_DistributedEnv(1, 1, 1), max_cache_size=1
    )
    assert thread._delete_chunks_when_processed

    thread.start()

    assert thread._pre_download_counter == 0

    thread.download([0, 1, 2, 3, 4, 5, _END_TOKEN])

    while thread._pre_download_counter == 0:
        sleep(0.01)

    assert not thread._has_exited

    for i in range(5):
        thread.delete([i])
        while len(os.listdir(cache_dir)) != 14 - (i + 1):
            sleep(0.01)

    assert thread._pre_download_counter <= 2

    assert len(os.listdir(cache_dir)) == 9

    thread.join()
    sleep(0.1)
    assert thread._has_exited


@pytest.mark.parametrize("on_demand_bytes", [True, False])
def test_reader_read_bytes(tmpdir, monkeypatch, on_demand_bytes):
    monkeypatch.setattr(reader, "_LONG_DEFAULT_TIMEOUT", 0.1)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=2, max_cache_size=28020, on_demand_bytes=on_demand_bytes)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    for i in range(25):
        idx = ChunkedIndex(*cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        item = cache._reader.read(idx)
        assert item == i

import os
from unittest.mock import patch

import pytest

from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming import Cache
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.posix_fast import (
    advise_willneed,
    available_ram_bytes,
    detect_posix_fast,
    parse_proc_mounts,
    posix_max_data_workers,
    posix_page_bytes,
    posix_prefetch_fits_ram,
    posix_safe_keep,
)
from litdata.streaming.shuffle import FullShuffle, WindowShuffle, posix_shuffle_window
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from tests.streaming.test_item_loader import _write_int_dataset


def test_parse_proc_mounts():
    text = "/dev/sda1 / ext4 rw 0 0\nvast-nfs /mnt/vast nfs4 rw,addr=10.0.0.5 0 0\n10.1.1.1:/export /data nfs rw 0 0\n"
    rows = parse_proc_mounts(text)
    assert rows[1][0] == "/mnt/vast"
    assert rows[1][1] == "nfs4"
    assert "vast" in rows[1][2]


def test_detect_local_path_is_automatic():
    profile = detect_posix_fast("/data/imagenet", mounts_text="/dev/sda1 / ext4 rw 0 0\n")
    assert profile is not None
    assert profile.kind == "posix"
    assert profile.in_place is True


def test_detect_vast_from_mounts():
    mounts = "/dev/sda1 / ext4 rw 0 0\nvast-1 /datasets nfs4 rw 0 0\n"
    profile = detect_posix_fast("/datasets/imagenet", mounts_text=mounts)
    assert profile is not None
    assert profile.kind == "vast"


def test_detect_nfs_from_mounts():
    mounts = "filer:/vol /nfs/data nfs rw 0 0\n"
    profile = detect_posix_fast("/nfs/data/ds", mounts_text=mounts)
    assert profile is not None
    assert profile.kind == "nfs"


def test_detect_skips_object_urls_even_with_local_path():
    assert detect_posix_fast("s3://bucket/key") is None
    assert detect_posix_fast("/teamspace/s3_connections/x", remote_url="s3://bucket/key") is None


def test_detect_env_disable(monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    assert detect_posix_fast("/mnt/vast/data", mounts_text="x /mnt/vast nfs4 rw 0 0\n") is None


def test_posix_fast_deserializes_strings(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=5)
    for i in range(12):
        cache[i] = f"row-{i}"
    cache.done()
    cache.merge()
    dataset = StreamingDataset(str(tmpdir))
    assert [dataset[i] for i in range(12)] == [f"row-{i}" for i in range(12)]


def test_posix_fast_mmap_all_chunks_including_shared(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=40, chunk_size=7)
    dataset = StreamingDataset(data_dir)
    assert dataset.posix_fast is not None
    items = [dataset[i] for i in range(len(dataset))]
    assert items == list(range(40))
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert loader._posix_fast is True
    assert dataset.cache._reader._posix_fast is True
    assert loader._mmap_allowed_chunks
    assert loader._mmap is not None or loader._mapped


def test_posix_fast_does_not_delete_source_chunks(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=20, chunk_size=5)
    dataset = StreamingDataset(data_dir)
    _ = [dataset[i] for i in range(len(dataset))]
    files = [name for name in os.listdir(data_dir) if name.endswith(".bin")]
    assert files
    loader = dataset.cache._reader._item_loader
    for name in files:
        loader.delete(0, os.path.join(data_dir, name))
        assert os.path.exists(os.path.join(data_dir, name))


def test_posix_fast_shuffle_uses_window_shuffle(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "1")
    data_dir = _write_int_dataset(tmpdir, num_items=80, chunk_size=5)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    list(iter(dataset))
    assert isinstance(dataset.shuffler, WindowShuffle)
    items = list(iter(dataset))
    assert sorted(items) == list(range(80))
    assert items != list(range(80))


def test_local_posix_keeps_full_shuffle(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=40, chunk_size=5)
    dataset = StreamingDataset(data_dir, shuffle=True)
    list(iter(dataset))
    assert dataset.posix_fast is not None
    assert dataset.posix_fast.kind == "posix"
    assert isinstance(dataset.shuffler, FullShuffle)


def test_posix_fast_loads_a_page_of_items(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=80, chunk_size=40)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    items = list(iter(dataset))
    assert sorted(items) == list(range(80))
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert loader._page is not None
    assert loader._page_end > loader._page_start
    assert loader._page_bytes > 0


def test_posix_fast_page_bytes_zero_still_reads(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_PAGE_BYTES", "0")
    data_dir = _write_int_dataset(tmpdir, num_items=30, chunk_size=10)
    dataset = StreamingDataset(data_dir, shuffle=True)
    items = list(iter(dataset))
    assert sorted(items) == list(range(30))
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert loader._page is None


def test_posix_fast_disabled_keeps_full_shuffle(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "0")
    data_dir = _write_int_dataset(tmpdir, num_items=40, chunk_size=5)
    dataset = StreamingDataset(data_dir, shuffle=True)
    list(iter(dataset))
    assert isinstance(dataset.shuffler, FullShuffle)


def test_detect_force_on_and_object_url_still_skipped(monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "1")
    profile = detect_posix_fast("/data/ds", mounts_text="")
    assert profile is not None
    assert profile.kind == "forced"
    assert detect_posix_fast("gs://bucket/key") is None
    assert detect_posix_fast("r2://bucket/key") is None
    assert detect_posix_fast(None) is None
    assert detect_posix_fast("") is None


def test_detect_lustre_and_path_vast():
    profile = detect_posix_fast("/scratch/ds", mounts_text="foo /scratch lustre rw 0 0\n")
    assert profile is not None
    assert profile.kind == "lustre"
    profile = detect_posix_fast("/mnt/vastdata/imagenet", mounts_text="/dev/sda1 / ext4 rw 0 0\n")
    assert profile is not None
    assert profile.kind == "vast"


def test_posix_env_helpers(monkeypatch):
    monkeypatch.delenv("LITDATA_POSIX_SHUFFLE_WINDOW", raising=False)
    assert posix_shuffle_window() == 16
    monkeypatch.setenv("LITDATA_POSIX_SHUFFLE_WINDOW", "32")
    assert posix_shuffle_window() == 32
    monkeypatch.setenv("LITDATA_POSIX_SHUFFLE_WINDOW", "nope")
    assert posix_shuffle_window() == 16
    monkeypatch.setenv("LITDATA_POSIX_PAGE_BYTES", "4096")
    assert posix_page_bytes() == 4096
    monkeypatch.setenv("LITDATA_POSIX_PAGE_BYTES", "x")
    assert posix_page_bytes() == 256 * 1024


def test_advise_willneed_missing_file(tmp_path):
    advise_willneed(str(tmp_path / "missing.bin"))


def test_available_ram_bytes_parses_meminfo():
    text = "MemTotal:       1000000 kB\nMemFree:          1000 kB\nMemAvailable:    500000 kB\nCached: 0 kB\n"
    assert available_ram_bytes(text) == 500000 * 1024


def test_posix_prefetch_fits_ram_skips_when_window_crowds_memory(monkeypatch):
    monkeypatch.delenv("LITDATA_POSIX_WILLNEED", raising=False)
    monkeypatch.delenv("LITDATA_POSIX_RAM_FRACTION", raising=False)
    chunk = 64 * 1024 * 1024
    # 208 workers × 4 keep × 64MiB ≈ 53GiB vs 40GiB available
    assert not posix_prefetch_fits_ram(keep=4, chunk_bytes=chunk, num_readers=208, ram_bytes=40 * 1024**3)
    assert posix_prefetch_fits_ram(keep=4, chunk_bytes=chunk, num_readers=8, ram_bytes=40 * 1024**3)
    monkeypatch.setenv("LITDATA_POSIX_WILLNEED", "0")
    assert not posix_prefetch_fits_ram(keep=1, chunk_bytes=1, num_readers=1, ram_bytes=10**12)
    monkeypatch.setenv("LITDATA_POSIX_WILLNEED", "1")
    assert posix_prefetch_fits_ram(keep=4, chunk_bytes=chunk, num_readers=208, ram_bytes=40 * 1024**3)


def test_posix_max_data_workers_caps_all_cores(monkeypatch):
    monkeypatch.delenv("LITDATA_POSIX_MAX_WORKERS", raising=False)
    monkeypatch.delenv("LITDATA_POSIX_RAM_FRACTION", raising=False)
    ram = 50 * 1024**3
    rss = 256 * 1024 * 1024
    capped = posix_max_data_workers(requested=208, ram_bytes=ram, rss_bytes=rss)
    assert 1 <= capped < 208
    monkeypatch.setenv("LITDATA_POSIX_MAX_WORKERS", "0")
    assert posix_max_data_workers(requested=208, ram_bytes=ram, rss_bytes=rss) == 208
    monkeypatch.setenv("LITDATA_POSIX_MAX_WORKERS", "12")
    assert posix_max_data_workers(requested=208, ram_bytes=ram, rss_bytes=rss) == 12


def test_posix_safe_keep_drops_to_one_when_crowded(monkeypatch):
    monkeypatch.delenv("LITDATA_POSIX_WILLNEED", raising=False)
    keep = posix_safe_keep(keep=4, chunk_bytes=64 * 1024 * 1024, num_readers=208, ram_bytes=40 * 1024**3)
    assert keep == 1


def test_window_shuffle_does_not_share_chunks(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_POSIX_FAST", "1")
    data_dir = _write_int_dataset(tmpdir, num_items=80, chunk_size=10)
    dataset = StreamingDataset(data_dir, shuffle=True, drop_last=True)
    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    dataset.worker_env = _WorkerEnv.detect()
    cache = dataset._create_cache(worker_env=dataset.worker_env)
    shuffler = dataset._create_shuffler(cache)
    assert isinstance(shuffler, WindowShuffle)
    workers_chunks, _ = shuffler.get_chunks_and_intervals_per_workers(dataset.distributed_env, 1, 1, 1)
    flat = [c for chunks in workers_chunks for c in chunks]
    assert len(flat) == len(set(flat))


def test_posix_prefetch_slides_along_stripe(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=60, chunk_size=5)
    dataset = StreamingDataset(data_dir, shuffle=False, max_pre_download=2)
    with patch("litdata.streaming.reader.advise_willneed") as mocked:
        list(iter(dataset))
    assert mocked.call_count >= 2


def test_posix_page_is_memoryview(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=40, chunk_size=20)
    dataset = StreamingDataset(data_dir, shuffle=True)
    list(iter(dataset))
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert isinstance(loader._page, memoryview) or loader._page is None
    if loader._mmap_view is not None:
        assert isinstance(loader._mmap_view, memoryview)


@pytest.mark.skipif(os.name != "posix", reason="resource.setrlimit is posix")
def test_posix_fast_tokens_do_not_exhaust_fds(tmpdir):
    import resource

    import torch

    from litdata.streaming.item_loader import TokensLoader

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    cap = min(256, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (cap, hard))
    block_size = 64
    cache = Cache(str(tmpdir), chunk_bytes="2KB", item_loader=TokensLoader(block_size))
    for i in range(400):
        cache._add_item(i, torch.randint(0, 100, (80,)).numpy())
    cache.done()
    cache.merge()
    dataset = StreamingDataset(str(tmpdir), item_loader=TokensLoader(block_size), shuffle=True)
    n = 0
    for _ in dataset:
        n += 1
    assert n > 0
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, TokensLoader)
    assert len(loader._mmaps) <= loader._mmap_keep + 1


@pytest.mark.skipif(not _ZSTD_AVAILABLE, reason="zstd required")
def test_compressed_chunks_do_not_use_posix_mmap(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=10, compression="zstd")
    for i in range(20):
        cache[i] = i
    cache.done()
    cache.merge()
    dataset = StreamingDataset(str(tmpdir))
    assert [dataset[i] for i in range(20)] == list(range(20))
    assert dataset.posix_fast is None
    assert dataset.cache._reader._posix_fast is False

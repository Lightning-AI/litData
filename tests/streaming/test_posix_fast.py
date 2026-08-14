import os

from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.posix_fast import detect_posix_fast, parse_proc_mounts
from litdata.streaming.shuffle import FullShuffle, WindowShuffle
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


def test_posix_fast_shuffle_uses_window_shuffle(tmpdir):
    data_dir = _write_int_dataset(tmpdir, num_items=80, chunk_size=5)
    dataset = StreamingDataset(data_dir, shuffle=True, seed=42)
    list(iter(dataset))
    assert isinstance(dataset.shuffler, WindowShuffle)
    items = list(iter(dataset))
    assert sorted(items) == list(range(80))
    assert items != list(range(80))


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

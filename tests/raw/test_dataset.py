import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from litdata import StreamingRawDataset
from litdata.raw.dataset import CacheManager
from litdata.raw.indexer import FileMetadata


def test_cache_manager_init_with_caching(tmp_path):
    """Test CacheManager initialization with caching enabled."""
    input_dir = "s3://bucket/dataset"
    cache_dir = str(tmp_path / "cache")

    manager = CacheManager(input_dir=input_dir, cache_dir=cache_dir, cache_files=True)

    assert manager.cache_files is True
    assert manager.cache_dir is not None
    assert os.path.exists(manager.cache_dir)
    assert manager.downloader is not None


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_get_local_path(tmp_path):
    """Test local path generation."""
    input_dir = "s3://bucket/dataset"
    cache_dir = str(tmp_path / "cache")

    manager = CacheManager(input_dir=input_dir, cache_dir=cache_dir, cache_files=True)

    file_path = "s3://bucket/dataset/subdir/file.jpg"
    local_path = manager.get_local_path(file_path)

    assert "subdir/file.jpg" in local_path
    assert local_path.startswith(manager.cache_dir)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_default_max_prefetch(tmp_path):
    """Default max_prefetch is a positive look-ahead (16)."""
    (tmp_path / "file1.jpg").write_bytes(b"x")
    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)
    assert dataset.max_prefetch == 16
    assert dataset.prefetch_cache_size == 32


@pytest.mark.parametrize(
    ("num_workers", "max_prefetch", "expected"),
    [
        (1, 16, 16),
        (0, 16, 16),  # treated as single-process (≤1)
        (2, 16, 16),  # min(16, 64//2) = 16
        (4, 16, 16),  # min(16, 64//4) = 16
        (8, 16, 8),  # min(16, 64//8) = 8
        (16, 16, 4),  # min(16, 64//16) = 4
        (24, 16, 2),  # min(16, 64//24) = 2
        (32, 16, 2),  # min(16, 64//32) = 2
        (32, 32, 2),  # still capped by aggregate budget
        (2, 32, 32),  # min(32, 64//2) = 32
        (8, 0, 0),
    ],
)
def test_effective_prefetch_vs_num_workers(num_workers, max_prefetch, expected):
    from litdata.raw.dataset import _effective_prefetch

    assert _effective_prefetch(max_prefetch, num_workers) == expected


@pytest.mark.parametrize(
    ("num_workers", "max_concurrent", "median_bytes", "expected"),
    [
        # Explicit int → exactly that many permits (no silent clamp), any worker count
        (0, 64, 100_000, 64),
        (1, 64, 100_000, 64),
        (24, 64, 100_000, 64),
        (32, 64, 100_000, 64),
        (2, 4, 100_000, 4),
        # Adaptive (None): ~100KB JPEG → bandwidth≈524 → cap 512
        (0, None, 100_000, 128),  # single-process cap
        (1, None, 100_000, 128),
        (2, None, 100_000, 256),  # 512//2
        (4, None, 100_000, 128),  # 512//4
        (8, None, 100_000, 64),  # 512//8
        (16, None, 100_000, 32),  # 512//16
        (24, None, 100_000, 21),  # 512//24
        (32, None, 100_000, 16),  # 512//32
        # Large objects (≥1 MiB): bandwidth-only (no Little's-law pin at 240)
        (4, None, 10 * 1024 * 1024, 8),  # budget=floor 32, 32//4=8
        (16, None, 10 * 1024 * 1024, 2),  # budget=floor 32, 32//16=2
        # Unknown size uses default median (256KiB) → latency arm (240)
        (8, None, None, 30),  # 240//8
    ],
)
def test_effective_concurrency_vs_num_workers(num_workers, max_concurrent, median_bytes, expected):
    from litdata.raw.dataset import _effective_concurrency

    assert _effective_concurrency(max_concurrent, num_workers, median_bytes) == expected


def test_aggregate_concurrency_budget_clamps():
    from litdata.raw.dataset import (
        _AGGREGATE_CONCURRENCY_BUDGET_CAP,
        _AGGREGATE_CONCURRENCY_BUDGET_FLOOR,
        _ASSUMED_AGGREGATE_BANDWIDTH_BPS,
        _ASSUMED_REQUEST_LATENCY_S,
        _ASSUMED_REQUEST_RATE,
        _CONCURRENCY_PIPELINE_SECONDS,
        _aggregate_concurrency_budget,
    )

    latency = int(_ASSUMED_REQUEST_RATE * _ASSUMED_REQUEST_LATENCY_S)  # ~240
    target_bytes = int(_ASSUMED_AGGREGATE_BANDWIDTH_BPS * _CONCURRENCY_PIPELINE_SECONDS)
    assert _aggregate_concurrency_budget(1) == _AGGREGATE_CONCURRENCY_BUDGET_CAP
    # Tiny ImageNet-like: bandwidth wins over latency, then hits cap
    assert _aggregate_concurrency_budget(100_000) == _AGGREGATE_CONCURRENCY_BUDGET_CAP
    assert (
        _AGGREGATE_CONCURRENCY_BUDGET_FLOOR <= _aggregate_concurrency_budget(None) <= _AGGREGATE_CONCURRENCY_BUDGET_CAP
    )
    # Sub-MiB default path still uses Little's-law floor
    assert _aggregate_concurrency_budget(256 * 1024) == max(
        _AGGREGATE_CONCURRENCY_BUDGET_FLOOR,
        min(_AGGREGATE_CONCURRENCY_BUDGET_CAP, max(target_bytes // (256 * 1024), latency)),
    )


@pytest.mark.parametrize(
    "median_bytes",
    [1 * 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024],
)
def test_aggregate_budget_large_median_bandwidth_bounded(median_bytes):
    """Medians ≥1 MiB must not be pinned by the Little's-law arm (~240)."""
    from litdata.raw.dataset import (
        _AGGREGATE_CONCURRENCY_BUDGET_FLOOR,
        _ASSUMED_AGGREGATE_BANDWIDTH_BPS,
        _ASSUMED_REQUEST_LATENCY_S,
        _ASSUMED_REQUEST_RATE,
        _CONCURRENCY_PIPELINE_SECONDS,
        _aggregate_concurrency_budget,
    )

    target_bytes = int(_ASSUMED_AGGREGATE_BANDWIDTH_BPS * _CONCURRENCY_PIPELINE_SECONDS)
    bandwidth = max(1, target_bytes // median_bytes)
    expected = max(_AGGREGATE_CONCURRENCY_BUDGET_FLOOR, min(512, bandwidth))
    got = _aggregate_concurrency_budget(median_bytes)
    assert got == expected
    latency = int(_ASSUMED_REQUEST_RATE * _ASSUMED_REQUEST_LATENCY_S)
    assert got != latency or bandwidth >= latency  # not latency-pinned when bandwidth is smaller


def test_effective_download_permits_cached_per_pid(tmp_path):
    """Permit math runs once per process, not on every semaphore acquire."""
    (tmp_path / "a.jpg").write_bytes(b"x" * 100_000)
    from unittest.mock import patch

    from litdata.raw.dataset import StreamingRawDataset

    ds = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)
    cm = ds.cache_manager
    with patch("litdata.raw.dataset._num_dataloader_workers", side_effect=[8, 16]) as mock_w:
        assert cm._effective_download_permits() == 64  # adaptive: 512//8
        assert cm._effective_download_permits() == 64  # cached — ignores worker change
        assert mock_w.call_count == 1
    cm.reset_runtime_state()
    with patch("litdata.raw.dataset._num_dataloader_workers", return_value=16):
        assert cm._effective_download_permits() == 32  # recomputed: 512//16


def test_effective_download_permits_reset_on_pickle(tmp_path):
    """Pickle/spawn clears the pid-guarded permit cache."""
    import pickle

    (tmp_path / "a.jpg").write_bytes(b"x" * 100_000)
    from unittest.mock import patch

    from litdata.raw.dataset import StreamingRawDataset

    ds = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)
    cm = ds.cache_manager
    with patch("litdata.raw.dataset._num_dataloader_workers", return_value=8):
        assert cm._effective_download_permits() == 64
    blob = pickle.dumps(cm)
    restored = pickle.loads(blob)  # noqa: S301
    assert restored._cached_permits is None
    assert restored._cached_permits_pid is None
    with patch("litdata.raw.dataset._num_dataloader_workers", return_value=16):
        assert restored._effective_download_permits() == 32


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_schedule_prefetch_uses_effective_budget(tmp_path):
    """_schedule_prefetch schedules only the worker-aware effective look-ahead."""
    for i in range(200):
        (tmp_path / f"file{i:03d}.jpg").write_bytes(b"x")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=16)

    class _Info:
        def __init__(self, num_workers: int):
            self.num_workers = num_workers

    def _count_scheduled(num_workers: int) -> int:
        call_count = {"n": 0}

        def counting_create_task(coro):
            call_count["n"] += 1
            coro.close()

            class _Task:
                def add_done_callback(self, cb):
                    return None

            return _Task()

        with (
            patch("torch.utils.data.get_worker_info", return_value=_Info(num_workers)),
            patch("asyncio.create_task", side_effect=counting_create_task),
        ):
            # Sequential batch of 4; start = 0 + num_workers * 4
            dataset._schedule_prefetch([0, 1, 2, 3])
        return call_count["n"]

    # w=16 → effective = min(16, 64//16) = 4
    assert _count_scheduled(16) == 4
    # w=2 → effective = min(16, 64//2) = 16
    assert _count_scheduled(2) == 16


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitem(tmp_path):
    """Test single item access."""
    test_content = b"test image content"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)

    # Patch async download to return test_content
    async def mock_download_file_async(file_path, size=None):
        return test_content

    with patch.object(dataset.cache_manager, "download_file_async", side_effect=mock_download_file_async):
        item = dataset[0]
        assert item == test_content


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitem_index_error(tmp_path):
    """Test index error for out of range access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=0)

    with pytest.raises(IndexError, match="Index 1 out of range"):
        dataset[1]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_setup(tmp_path):
    """Test the setup method for default and custom grouping."""
    # Create test files
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.jpg").write_text("content2")
    (tmp_path / "file3.jpg").write_text("content3")

    # Default setup: returns flat list
    dataset = StreamingRawDataset(input_dir=str(tmp_path))
    assert isinstance(dataset.items, list)
    assert all(isinstance(item, FileMetadata) for item in dataset.items)
    assert len(dataset.items) == 3

    # Custom setup: group files in pairs
    class GroupedDataset(StreamingRawDataset):
        def setup(self, files):
            # Group every two files together
            return [files[i : i + 2] for i in range(0, len(files), 2)]

    grouped_dataset = GroupedDataset(input_dir=str(tmp_path))
    assert isinstance(grouped_dataset.items, list)
    assert all(isinstance(item, list) for item in grouped_dataset.items)
    # Should be 2 groups: [[file1, file2], [file3]]
    assert len(grouped_dataset.items) == 2
    assert all(isinstance(f, FileMetadata) for group in grouped_dataset.items for f in group)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitems(tmp_path):
    """Test synchronous batch item access."""
    test_contents = [b"content1", b"content2", b"content3"]
    for i, content in enumerate(test_contents):
        (tmp_path / f"file{i}.jpg").write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=0)

    # Mock _download_batch to return test contents
    async def mock_download_batch(indices):
        return [test_contents[i] for i in indices]

    with patch.object(dataset, "_download_batch", side_effect=mock_download_batch):
        items = dataset.__getitems__([0, 2])
        assert items == [test_contents[0], test_contents[2]]


@pytest.mark.asyncio
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
async def test_download_batch_flat(tmp_path):
    """Test async batch download for empty and flat indices (default setup)."""
    test_contents = {
        str(tmp_path / "file0.jpg"): b"content1",
        str(tmp_path / "file1.jpg"): b"content2",
        str(tmp_path / "file2.jpg"): b"content3",
    }
    for file_path, content in test_contents.items():
        Path(file_path).write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)

    async def mock_download_and_process_item(file_path, size=None):
        return test_contents[file_path]

    with (
        patch.object(dataset, "_download_and_process_item", side_effect=mock_download_and_process_item),
    ):
        # Test empty indices
        items = await dataset._download_batch([])
        assert items == []

        indices = [0, 2, 1]
        items = await dataset._download_batch(indices)
        file_paths = [f.path for f in dataset.items]
        expected = [test_contents[file_paths[i]] for i in indices]
        assert items == expected


@pytest.mark.asyncio
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
async def test_download_batch_grouped(tmp_path):
    """Test async batch download for grouped indices (custom setup)."""
    test_contents = {
        str(tmp_path / "file0.jpg"): b"content1",
        str(tmp_path / "file1.jpg"): b"content2",
        str(tmp_path / "file2.jpg"): b"content3",
    }
    for file_path, content in test_contents.items():
        Path(file_path).write_bytes(content)

    class GroupedDataset(StreamingRawDataset):
        def setup(self, files):
            return [files[i : i + 2] for i in range(0, len(files), 2)]

    grouped_dataset = GroupedDataset(input_dir=str(tmp_path), max_prefetch=0)

    async def mock_download_and_process_group(file_paths, sizes=None):
        return [test_contents[fp] for fp in file_paths]

    print(grouped_dataset.items)

    with (
        patch.object(grouped_dataset, "_download_and_process_group", side_effect=mock_download_and_process_group),
    ):
        group_indices = list(range(len(grouped_dataset.items)))
        expected = [[test_contents[f.path] for f in group] for group in grouped_dataset.items]

        items = await grouped_dataset._download_batch(group_indices)
        assert items == expected


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_thread_safety(tmp_path):
    """Test thread safety in multi-threaded environments."""
    test_contents = [b"content1", b"content2", b"content3"]
    for i, content in enumerate(test_contents):
        (tmp_path / f"file{i}.jpg").write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=0)

    # Mock _download_batch to return test contents
    async def mock_download_batch(indices):
        return [test_contents[i] for i in indices]

    with patch.object(dataset, "_download_batch", side_effect=mock_download_batch):

        def worker():
            items = dataset.__getitems__([0, 2])
            assert items == [test_contents[0], test_contents[2]]

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitems_type_error(tmp_path):
    """Test type error for invalid indices type."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=0)

    with pytest.raises(TypeError):
        dataset.__getitems__(0)  # Should be a list


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitems_index_error(tmp_path):
    """Test index error for out of range batch access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False, max_prefetch=0)

    with pytest.raises(IndexError, match="out of range"):
        dataset.__getitems__([0, 1])


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_transform(tmp_path):
    """Test transform support in StreamingRawDataset."""
    test_content = b"raw"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    def transform(x):
        return x.decode() + "_transformed"

    dataset = StreamingRawDataset(input_dir=str(tmp_path), transform=transform, max_prefetch=0)

    # Patch async download to return test_content
    async def mock_download_file_async(file_path, size=None):
        return test_content

    with patch.object(dataset.cache_manager, "download_file_async", side_effect=mock_download_file_async):
        item = dataset[0]
        assert item == "raw_transformed"


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_with_dataloader(tmp_path):
    """Test dataset integration with PyTorch DataLoader."""
    test_contents = [b"content1", b"content2", b"content3", b"content4"]
    for i, content in enumerate(test_contents):
        (tmp_path / f"file{i}.jpg").write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path))

    # Mock async download to return test content
    async def mock_download_async(file_path, size=None):
        index = int(file_path.split("file")[1].split(".")[0])
        return test_contents[index]

    with patch.object(dataset.cache_manager, "download_file_async", side_effect=mock_download_async):
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        batches = list(dataloader)
        assert len(batches) == 2  # 4 items / batch_size 2
        assert len(batches[0]) == 2  # First batch has 2 items
        assert len(batches[1]) == 2  # Second batch has 2 items


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_no_files_error(tmp_path):
    """Test error when no files are found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No files found"):
        StreamingRawDataset(input_dir=str(empty_dir), cache_files=False)


# Additional coverage tests
def test_cache_manager_get_local_path_invalid():
    cm = CacheManager(input_dir="s3://bucket/data", cache_dir=None, cache_files=True)
    # Path that does not start with input_dir
    with pytest.raises(ValueError, match="does not start with input dir"):
        cm.get_local_path("s3://bucket/other/file.jpg")


def test_cache_manager_download_file_async_error():
    cm = CacheManager(input_dir="s3://bucket/data", cache_dir=None, cache_files=False)

    async def fail_download(file_path, *args, **kwargs):
        raise Exception("fail")

    cm._downloader = type("Downloader", (), {"adownload_fileobj": fail_download})()
    # Should raise RuntimeError
    import asyncio

    with pytest.raises(RuntimeError, match="Error downloading file"):
        asyncio.run(cm.download_file_async("s3://bucket/data/file.jpg"))


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_invalid_item_type(tmp_path):
    class BadDataset(StreamingRawDataset):
        def setup(self, files):
            print("files:", files)
            return [123]  # Invalid type

    (tmp_path / "file1.jpg").write_text("content1")
    ds = BadDataset(input_dir=str(tmp_path))
    with pytest.raises(TypeError, match="Dataset items must be of type FileMetadata"):
        ds[0]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_invalid_setup(tmp_path):
    class BadDataset(StreamingRawDataset):
        def setup(self, files):
            return files[0]

    (tmp_path / "file1.jpg").write_text("content1")
    with pytest.raises(TypeError, match="The setup method must return a list"):
        BadDataset(input_dir=str(tmp_path))


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_transform_none_and_group(tmp_path):
    # Single item, no transform
    (tmp_path / "file1.jpg").write_bytes(b"abc")
    ds = StreamingRawDataset(input_dir=str(tmp_path))

    # Patch download to return bytes
    async def mock_download_file_async(file_path, size=None):
        return b"abc"

    ds.cache_manager.download_file_async = mock_download_file_async
    assert ds[0] == b"abc"

    # Grouped item, with transform
    class GroupedDS(StreamingRawDataset):
        def setup(self, files):
            return [files]  # One group

    def transform(data):
        return b"-".join(data)

    gds = GroupedDS(input_dir=str(tmp_path), transform=transform)
    gds.cache_manager.download_file_async = mock_download_file_async
    assert gds[0] == b"abc"


def test_bandwidth_tracker_basic():
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=0.2)
    assert tracker.sample_count == 0

    # Tiny GET (< 64 KB) updates latency EMA only
    tracker.record_observation(50_000, 0.010)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 1
    assert bps_count == 0
    assert lat_count == 1
    assert bps is None
    assert pytest.approx(lat, abs=1e-6) == 0.010

    # Small/Medium GET (64 KB <= size < 256 KB) updates BOTH latency & bandwidth EMA
    tracker.record_observation(100_000, 0.020)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 2
    assert bps_count == 1
    assert lat_count == 2
    assert bps is not None
    assert pytest.approx(bps, abs=1.0) == 5_000_000.0
    # EMA for latency: 0.2 * 0.020 + 0.8 * 0.010 = 0.012
    assert pytest.approx(lat, abs=1e-6) == 0.012

    # Large GET (>= 256 KB) updates bandwidth EMA
    tracker.record_observation(10 * 1024 * 1024, 0.118)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 3
    assert bps_count == 2
    assert lat_count == 2
    assert bps is not None


def test_bandwidth_tracker_pickle():
    import pickle

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker()
    tracker.record_observation(100_000, 0.020)
    tracker.record_observation(10 * 1024 * 1024, 0.200)
    blob = pickle.dumps(tracker)
    restored = pickle.loads(blob)  # noqa: S301
    assert restored.sample_count == tracker.sample_count
    assert restored.bps_sample_count == tracker.bps_sample_count
    assert restored.lat_sample_count == tracker.lat_sample_count
    assert restored.bandwidth_bps_ema == tracker.bandwidth_bps_ema
    assert restored.request_latency_s_ema == tracker.request_latency_s_ema


def test_concurrency_budget_warmup_gating():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    tracker = BandwidthTracker()
    # 4 observations (under threshold of 5) -> uses default static budget
    for _ in range(4):
        tracker.record_observation(10 * 1024 * 1024, 0.001)

    # Median 10MB -> default budget: (100MB/s * 0.5s) // 10MB = 5 -> floor 32
    assert _aggregate_concurrency_budget(10 * 1024 * 1024, tracker=tracker) == 32

    # 5th observation -> warm-up gate unlocks empirical EMA
    tracker.record_observation(10 * 1024 * 1024, 0.001)
    # Measured bandwidth is huge -> dynamic budget scales up from 32 to 500
    assert _aggregate_concurrency_budget(10 * 1024 * 1024, tracker=tracker) == 500


def test_concurrency_budget_high_and_low_bandwidth_adaptation():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    # High bandwidth scenario
    high_tracker = BandwidthTracker()
    for _ in range(5):
        high_tracker.record_observation(10 * 1024 * 1024, 0.002)
    budget_high = _aggregate_concurrency_budget(1 * 1024 * 1024, tracker=high_tracker)
    assert budget_high == 512

    # Low bandwidth scenario
    low_tracker = BandwidthTracker()
    for _ in range(5):
        low_tracker.record_observation(10 * 1024 * 1024, 5.0)
    budget_low = _aggregate_concurrency_budget(10 * 1024 * 1024, tracker=low_tracker)
    assert budget_low == 32


def test_class_gated_observation_isolation():
    from litdata.raw.dataset import (
        BandwidthTracker,
        _aggregate_concurrency_budget,
    )

    # Scenario 1: 5 small GETs (< 64 KB) -> lat_sample_count = 5, bps_sample_count = 0
    tracker_small = BandwidthTracker()
    for _ in range(5):
        tracker_small.record_observation(10_000, 0.010)

    bps, lat, bps_cnt, lat_cnt = tracker_small.get_metrics()
    assert bps_cnt == 0
    assert lat_cnt == 5
    assert bps is None

    # Budget for large 10 MB objects must STILL use static default aggregate bandwidth because bps_sample_count < 5
    # Clamped to floor 32
    assert _aggregate_concurrency_budget(10 * 1024 * 1024, tracker=tracker_small) == 32

    # Scenario 2: 5 large GETs (>= 256 KB) -> bps_sample_count = 5, lat_sample_count = 0
    tracker_large = BandwidthTracker()
    for _ in range(5):
        tracker_large.record_observation(10 * 1024 * 1024, 0.001)

    bps, lat, bps_cnt, lat_cnt = tracker_large.get_metrics()
    assert bps_cnt == 5
    assert lat_cnt == 0
    assert lat is None


def test_environment_variable_overrides(monkeypatch):
    from litdata.raw.dataset import (
        _aggregate_concurrency_budget,
        _effective_concurrency,
    )

    monkeypatch.setenv("LITDATA_ASSUMED_BANDWIDTH_BPS", str(500 * 1024 * 1024))
    monkeypatch.setenv("LITDATA_AGGREGATE_CONCURRENCY_BUDGET_CAP", "1024")
    monkeypatch.setenv("LITDATA_AGGREGATE_CONCURRENCY_BUDGET_FLOOR", "16")
    monkeypatch.setenv("LITDATA_SINGLE_PROCESS_CONCURRENCY_CAP", "256")

    assert _aggregate_concurrency_budget(100_000) == 1024
    assert _effective_concurrency(None, num_workers=1, median_file_bytes=100_000) == 256


def test_no_latency_concurrency_inflation():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    tracker = BandwidthTracker()
    # Record 5 latency observations with high latency (200ms vs assumed 40ms)
    for _ in range(5):
        tracker.record_observation(100_000, 0.200)

    # Budget with high latency must not inflate above baseline (240 for sub-1MB)
    budget = _aggregate_concurrency_budget(100_000, tracker=tracker)
    assert budget <= 240


def test_concurrency_latency_monotonicity():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    latencies = [0.040, 0.080, 0.200, 0.500]
    budgets = []

    for lat in latencies:
        tracker = BandwidthTracker()
        for _ in range(5):
            tracker.record_observation(100_000, lat)
        budgets.append(_aggregate_concurrency_budget(100_000, tracker=tracker))

    # For latencies above target (40ms), increasing latency must not increase computed budget
    for i in range(len(budgets) - 1):
        assert budgets[i + 1] <= budgets[i]


def test_concurrency_latency_recovery():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    tracker = BandwidthTracker()
    # Inject high latency
    for _ in range(5):
        tracker.record_observation(100_000, 0.200)
    budget_degraded = _aggregate_concurrency_budget(100_000, tracker=tracker)

    # Now inject healthy latency
    for _ in range(10):
        tracker.record_observation(100_000, 0.040)
    budget_recovered = _aggregate_concurrency_budget(100_000, tracker=tracker)

    assert budget_recovered > budget_degraded


def test_imagenet_bandwidth_observation():
    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker()
    # ImageNet JPEG size ~150 KB
    imagenet_file_size = 150 * 1024
    for _ in range(5):
        tracker.record_observation(imagenet_file_size, 0.015)

    bps, lat, bps_cnt, lat_cnt = tracker.get_metrics()
    assert bps_cnt == 5
    assert lat_cnt == 5
    assert lat is not None
    assert abs(lat - 0.015) < 1e-4
    # ImageNet files (150 KB) MUST record bandwidth observation (bps != None)
    assert bps is not None
    assert bps > 0


def test_guarded_adaptive_floor_reduction():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget

    tracker = BandwidthTracker()
    # Low bandwidth (<10 MiB/s) AND high latency (>100 ms)
    for _ in range(5):
        tracker.record_observation(128 * 1024, 0.500)

    # Under severe evidence (low bandwidth + high latency), budget can drop below default 32 floor
    budget = _aggregate_concurrency_budget(10 * 1024 * 1024, tracker=tracker)
    assert budget < 32


def test_worker_allocation_respects_aggregate_budget():
    from litdata.raw.dataset import BandwidthTracker, _aggregate_concurrency_budget, _effective_concurrency

    test_cases = [
        (512, 8, 100_000),
        (32, 8, 10_000_000),
        (32, 16, 10_000_000),
        (32, 64, 10_000_000),
    ]

    for expected_budget_cap, workers, median_bytes in test_cases:
        tracker = BandwidthTracker()
        budget = _aggregate_concurrency_budget(median_bytes, tracker=tracker)
        total_permits = sum(
            _effective_concurrency(
                None, num_workers=workers, median_file_bytes=median_bytes, tracker=tracker, worker_id=w
            )
            for w in range(workers)
        )
        # Sum of per-worker permits across all workers must not exceed aggregate budget
        assert total_permits <= budget

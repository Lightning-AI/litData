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

    from litdata.raw.dataset import (
        _ASSUMED_AGGREGATE_BANDWIDTH_BPS,
        _LATENCY_RTT_EPSILON,
        BandwidthTracker,
    )

    assumed_bps = float(_ASSUMED_AGGREGATE_BANDWIDTH_BPS)
    tracker = BandwidthTracker(alpha=0.2)
    assert tracker.sample_count == 0

    # --- Tiny GET (50 KB < 64 KB) ---
    # Only updates latency EMA; no bandwidth observation recorded.
    # prev_bps_ema = None → fallback to assumed bandwidth for transfer estimate.
    tracker.record_observation(50_000, 0.010)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 1
    assert bps_count == 0
    assert lat_count == 1
    assert bps is None
    transfer1 = 50_000 / assumed_bps
    expected_lat1 = max(_LATENCY_RTT_EPSILON, 0.010 - transfer1)
    assert pytest.approx(lat, abs=1e-6) == expected_lat1
    # Sanity: transfer-subtracted latency must be strictly less than raw duration.
    assert lat < 0.010

    # --- Small/Medium GET (100 KB: 64 KB <= size < 256 KB) ---
    # Updates BOTH latency and bandwidth EMAs.
    # prev_bps_ema is still None (previous obs was below bandwidth threshold) → fallback.
    tracker.record_observation(100_000, 0.020)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 2
    assert bps_count == 1
    assert lat_count == 2
    assert bps is not None
    assert pytest.approx(bps, abs=1.0) == 5_000_000.0
    transfer2 = 100_000 / assumed_bps  # prev_bps_ema still None before this obs
    est_lat2 = max(_LATENCY_RTT_EPSILON, 0.020 - transfer2)
    expected_lat2 = 0.2 * est_lat2 + 0.8 * expected_lat1
    assert pytest.approx(lat, abs=1e-6) == expected_lat2

    # --- Large GET (10 MiB >= 256 KB) ---
    # Updates bandwidth EMA only; size >= _LATENCY_OBSERVATION_MAX_BYTES.
    tracker.record_observation(10 * 1024 * 1024, 0.118)
    bps, lat, bps_count, lat_count = tracker.get_metrics()
    assert tracker.sample_count == 3
    assert bps_count == 2
    assert lat_count == 2  # unchanged — large GETs do not update latency EMA
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
    # ImageNet JPEG size ~150 KB — straddles both latency (<256 KB) and bandwidth (>=64 KB) thresholds.
    imagenet_file_size = 150 * 1024
    for _ in range(5):
        tracker.record_observation(imagenet_file_size, 0.015)

    bps, lat, bps_cnt, lat_cnt = tracker.get_metrics()
    assert bps_cnt == 5
    assert lat_cnt == 5

    # ImageNet files MUST record both EMA estimates.
    assert bps is not None
    assert bps > 0
    assert lat is not None
    assert lat > 0

    # Transfer-subtracted latency must be strictly less than raw wall-clock duration.
    # (transfer time is non-zero for a 150 KB file)
    assert lat < 0.015

    # Epsilon floor must hold: estimated latency must not go below _LATENCY_RTT_EPSILON.
    assert lat >= 0.001


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


def test_transfer_subtracted_latency():
    """Pre-seed the tracker with 5 large GETs to reach sample threshold.

    Establishes a known bandwidth EMA (20 MB/s), then records a 200 KB request taking 30 ms.
    Verify latency EMA is approximately 30 ms - transfer_time (10 ms) = 20 ms.
    """
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    prior_bps = 20 * 1024 * 1024  # 20 MB/s
    large_size = 10 * 1024 * 1024  # 10 MiB (bandwidth-only GET)

    # Record 5 observations to pass the _MIN_EMPIRICAL_SAMPLES = 5 threshold
    for _ in range(5):
        tracker.record_observation(large_size, large_size / prior_bps)

    bps, lat, bps_cnt, lat_cnt = tracker.get_metrics()
    assert bps_cnt == 5
    assert lat_cnt == 0
    assert pytest.approx(bps, rel=1e-5) == prior_bps

    # Now record 200 KB GET taking 30 ms
    size = 200 * 1024
    duration = 0.030
    tracker.record_observation(size, duration)

    _, lat_after, _, _ = tracker.get_metrics()
    expected_lat = duration - (size / prior_bps)  # 30 ms - 10 ms = 20 ms
    assert lat_after is not None
    assert pytest.approx(lat_after, abs=1e-4) == expected_lat


def test_transfer_subtracted_latency_bootstrap():
    """When no empirical bandwidth exists or bps_sample_count < 5, verify default bandwidth is used."""
    import pytest

    from litdata.raw.dataset import _ASSUMED_AGGREGATE_BANDWIDTH_BPS, BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    assert tracker.bandwidth_bps_ema is None

    # Record a 50 KB GET taking 10 ms (sample 1, below 5 threshold)
    size = 50 * 1024
    duration = 0.010
    tracker.record_observation(size, duration)

    _, lat, _, _ = tracker.get_metrics()
    expected_transfer = size / float(_ASSUMED_AGGREGATE_BANDWIDTH_BPS)
    expected_lat = max(0.001, duration - expected_transfer)
    assert lat is not None
    assert pytest.approx(lat, abs=1e-6) == expected_lat


def test_transfer_subtracted_latency_clamped():
    """Provide an observation where estimated transfer time > observed duration.

    Verify resulting latency is clamped to epsilon (0.001s) rather than becoming zero or negative.
    """
    import pytest

    from litdata.raw.dataset import _LATENCY_EPSILON_S, BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    # Seed 5 large GETs at a slow prior BPS (1 MB/s)
    slow_bps = 1 * 1024 * 1024
    large_size = 10 * 1024 * 1024
    for _ in range(5):
        tracker.record_observation(large_size, large_size / slow_bps)

    # Now record a 100 KB GET arriving in only 5 ms (faster than 100 ms transfer estimate)
    tracker.record_observation(100 * 1024, 0.005)
    _, lat, _, _ = tracker.get_metrics()

    assert lat is not None
    assert pytest.approx(lat, abs=1e-9) == _LATENCY_EPSILON_S


def test_current_bandwidth_sample_does_not_explain_itself():
    """Verify bandwidth calculated from current observation is NOT used to calculate transfer time."""
    from litdata.raw.dataset import _LATENCY_EPSILON_S, BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    # Record a single 200 KB GET taking 30 ms (bps_sample_count = 0 before this sample).
    # If circular, it would use 200 KB / 30 ms to calculate transfer = 30 ms -> lat = 0 -> clamped to epsilon (1 ms).
    # Since it correctly uses fallback default (100 MB/s), transfer = 2 ms -> lat = 28 ms.
    tracker.record_observation(200 * 1024, 0.030)
    _, lat, _, _ = tracker.get_metrics()

    assert lat is not None
    assert lat > 10 * _LATENCY_EPSILON_S  # 28 ms is >> 1 ms epsilon


def test_256k_to_1m_bandwidth_observation():
    """Verify 256 KiB - 1 MiB objects update BPS EMA without updating latency EMA."""
    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker()
    size_500k = 500 * 1024  # 500 KiB
    tracker.record_observation(size_500k, 0.050)

    bps, lat, bps_cnt, lat_cnt = tracker.get_metrics()
    assert bps_cnt == 1
    assert lat_cnt == 0
    assert bps is not None
    assert bps > 0
    assert lat is None


def test_dynamic_semaphore_scale_down_target():
    """Verify _DynamicSemaphore updates target permits on downscale and retains permit bounds."""
    import asyncio

    import pytest

    from litdata.raw.dataset import _DynamicSemaphore

    @pytest.mark.asyncio
    async def _run():
        dyn_sem = _DynamicSemaphore(32)
        assert dyn_sem.target_permits == 32

        # Downscale to 16
        dyn_sem.update_target(16)
        assert dyn_sem.target_permits == 16

        # Acquire 16 permits
        for _ in range(16):
            await dyn_sem.acquire()

        # Releasing permits works cleanly
        for _ in range(16):
            dyn_sem.release()

    asyncio.run(_run())


def test_ranged_gather_individual_chunk_observations(tmp_path):
    """Verify that ranged downloads record observations per chunk rather than one aggregate blob."""
    import os
    from unittest.mock import MagicMock

    from litdata.raw.dataset import StreamingRawDataset

    (tmp_path / "sample.bin").write_bytes(b"x" * 100)
    ds = StreamingRawDataset(
        input_dir=str(tmp_path),
        cache_dir=str(tmp_path),
        range_parallel_threshold=100 * 1024,
        range_chunk_size=100 * 1024,
    )

    import asyncio

    async def _run():
        loop = asyncio.get_running_loop()
        mock_dl = MagicMock()

        def _mock_write(f_path, off, length, scratch):
            from pathlib import Path

            Path(scratch).write_bytes(b"x" * length)

        mock_dl.download_bytes.side_effect = _mock_write
        ds.cache_manager._downloader = mock_dl
        ds.cache_manager._downloader_pid = os.getpid()
        ds.cache_manager._downloader_loop = loop

        data = await ds.cache_manager._ranged_download_bytes("s3://mock-bucket/file.bin", size=200 * 1024)
        assert len(data) == 200 * 1024

    asyncio.run(_run())
    # 2 chunks of 100 KiB (>= 64 KiB min) -> 2 bps observations recorded
    _, _, bps_cnt, _ = ds.cache_manager._bandwidth_tracker.get_metrics()
    ds.cache_manager.reset_runtime_state()
    assert bps_cnt == 2


def test_hedged_get_excludes_hedge_delay(tmp_path):
    """Verify that hedged GET timing is recorded inside the winning attempt."""
    import os
    from unittest.mock import AsyncMock, MagicMock

    from litdata.raw.dataset import StreamingRawDataset

    (tmp_path / "sample.bin").write_bytes(b"y" * 100_000)
    ds = StreamingRawDataset(
        input_dir=str(tmp_path),
        cache_dir=str(tmp_path),
        hedge_delay=0.1,
    )

    import asyncio

    async def _run():
        loop = asyncio.get_running_loop()
        mock_dl = MagicMock()
        mock_dl.adownload_fileobj = AsyncMock(return_value=b"y" * 100_000)
        ds.cache_manager._downloader = mock_dl
        ds.cache_manager._downloader_pid = os.getpid()
        ds.cache_manager._downloader_loop = loop

        data = await ds.cache_manager._fetch_bytes("s3://mock-bucket/file.bin", size=100_000)
        assert len(data) == 100_000

    asyncio.run(_run())
    bps, _, bps_cnt, _ = ds.cache_manager._bandwidth_tracker.get_metrics()
    ds.cache_manager.reset_runtime_state()
    assert bps_cnt == 1
    assert bps is not None


def test_backoff_is_immediate():
    """Verify high latency (>1.1 * target) triggers immediate backoff factor reduction."""
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    # 5 samples with 200 ms latency (target = 40 ms -> factor = 0.20)
    for _ in range(5):
        tracker.record_observation(10 * 1024, 0.200)

    factor = tracker.get_backoff_factor(0.040)
    assert pytest.approx(factor, abs=1e-3) == 0.20


def test_no_recovery_in_deadband():
    """Verify latency inside deadband (target < L <= 1.1 * target) holds current backoff factor."""
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    # Trigger backoff down to 0.50 (80 ms latency)
    for _ in range(5):
        tracker.record_observation(10 * 1024, 0.080)
    assert pytest.approx(tracker.get_backoff_factor(0.040), abs=1e-3) == 0.50

    # Deadband sample (42 ms: 40 ms < 42 ms <= 44 ms)
    tracker.record_observation(10 * 1024, 0.042)
    factor = tracker.get_backoff_factor(0.040)
    assert pytest.approx(factor, abs=1e-3) == 0.50


def test_gradual_recovery():
    """Verify healthy latency (<= target) triggers gradual recovery towards 1.0."""
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    for _ in range(5):
        tracker.record_observation(10 * 1024, 0.080)
    factor_initial = tracker.get_backoff_factor(0.040)
    assert pytest.approx(factor_initial, abs=1e-3) == 0.50

    # Healthy sample (20 ms <= 40 ms target) -> recovers by alpha (0.1 * (1.0 - 0.5) = +0.05)
    tracker.record_observation(10 * 1024, 0.020)
    factor_recovered = tracker.get_backoff_factor(0.040)
    assert factor_recovered > factor_initial
    assert pytest.approx(factor_recovered, abs=1e-3) == 0.55


def test_congestion_interrupts_recovery():
    """Verify new congestion immediately interrupts gradual recovery."""
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    for _ in range(5):
        tracker.record_observation(10 * 1024, 0.080)
    tracker.get_backoff_factor(0.040)  # 0.50

    # 1 healthy sample -> recovers to 0.55
    tracker.record_observation(10 * 1024, 0.020)
    tracker.get_backoff_factor(0.040)

    # Congestion sample (200 ms) -> immediate drop to 0.20
    tracker.record_observation(10 * 1024, 0.200)
    factor = tracker.get_backoff_factor(0.040)
    assert pytest.approx(factor, abs=1e-3) == 0.20


def test_recovery_reaches_one():
    """Verify sustained healthy observations restore backoff factor to 1.0."""
    import pytest

    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker(alpha=1.0)
    for _ in range(5):
        tracker.record_observation(10 * 1024, 0.200)
    tracker.get_backoff_factor(0.040)

    # Repeated healthy samples
    for _ in range(100):
        tracker.record_observation(10 * 1024, 0.010)
        tracker.get_backoff_factor(0.040)

    factor = tracker.get_backoff_factor(0.040)
    assert pytest.approx(factor, abs=1e-2) == 1.0


def test_env_invalid_value_falls_back(monkeypatch):
    """Verify invalid string in environment variable falls back to safe default."""
    from litdata.raw.dataset import _get_assumed_aggregate_bandwidth_bps

    monkeypatch.setenv("LITDATA_ASSUMED_BANDWIDTH_BPS", "not_a_number")
    assert _get_assumed_aggregate_bandwidth_bps() > 0


def test_env_zero_value_falls_back(monkeypatch):
    """Verify zero value in float environment variable falls back to safe default."""
    from litdata.raw.dataset import _get_assumed_request_latency_s

    monkeypatch.setenv("LITDATA_ASSUMED_REQUEST_LATENCY_S", "0.0")
    assert _get_assumed_request_latency_s() > 0.0


def test_env_negative_value_falls_back(monkeypatch):
    """Verify negative integer in environment variable falls back to default."""
    from litdata.raw.dataset import _get_single_process_concurrency_cap

    monkeypatch.setenv("LITDATA_SINGLE_PROCESS_CONCURRENCY_CAP", "-10")
    assert _get_single_process_concurrency_cap() > 0


def test_env_floor_cannot_exceed_cap(monkeypatch):
    """Verify aggregate floor is clamped to cap when floor > cap."""
    from litdata.raw.dataset import _get_aggregate_concurrency_budget_cap, _get_aggregate_concurrency_budget_floor

    monkeypatch.setenv("LITDATA_AGGREGATE_CONCURRENCY_BUDGET_FLOOR", "1000")
    monkeypatch.setenv("LITDATA_AGGREGATE_CONCURRENCY_BUDGET_CAP", "512")
    cap = _get_aggregate_concurrency_budget_cap()
    floor = _get_aggregate_concurrency_budget_floor()
    assert floor <= cap


def test_permit_refresh_respects_reduced_budget_for_new_acquisitions():
    """Verify _DynamicSemaphore target reduction prevents additional permits beyond reduced target."""
    import asyncio

    import pytest

    from litdata.raw.dataset import _DynamicSemaphore

    async def _run():
        sem = _DynamicSemaphore(32)
        sem.update_target(16)
        assert sem.target_permits == 16

        # Acquire 16 permits cleanly
        for _ in range(16):
            await sem.acquire()

        # Target 16 is reached — next acquire without release must fail non-blocking
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)

        for _ in range(16):
            sem.release()

    asyncio.run(_run())


def test_record_observation_returns_atomic_sample_counts():
    """Verify record_observation returns (prev_sample_count, new_sample_count) tuple atomically."""
    from litdata.raw.dataset import BandwidthTracker

    tracker = BandwidthTracker()
    assert tracker.record_observation(-1, 0.1) is None
    assert tracker.record_observation(100, -0.5) is None

    res1 = tracker.record_observation(100_000, 0.05)
    assert res1 == (0, 1)

    res2 = tracker.record_observation(200_000, 0.04)
    assert res2 == (1, 2)

    assert tracker.sample_count == 2


def test_record_download_observation_uses_configured_sample_and_refresh_thresholds(tmp_path, monkeypatch):
    """Verify permit cache invalidation uses min_empirical_samples and permit_refresh_interval configs."""
    monkeypatch.setenv("LITDATA_MIN_EMPIRICAL_SAMPLES", "3")
    monkeypatch.setenv("LITDATA_PERMIT_REFRESH_INTERVAL", "4")

    (tmp_path / "file1.jpg").write_bytes(b"x")
    from litdata.raw.dataset import StreamingRawDataset

    ds = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)
    cm = ds.cache_manager

    cm._cached_permits = 64
    cm._cached_permits_pid = os.getpid()

    # Sample 1 & 2 (< min_samples 3) -> should not invalidate permits
    cm._record_download_observation(100_000, 0.01)
    assert cm._cached_permits == 64
    cm._record_download_observation(100_000, 0.01)
    assert cm._cached_permits == 64

    # Sample 3 (== min_samples 3) -> should invalidate permits
    cm._record_download_observation(100_000, 0.01)
    assert cm._cached_permits is None

    cm._cached_permits = 64
    cm._cached_permits_pid = os.getpid()

    # Sample 4 (% refresh 4 == 0) -> should invalidate permits
    cm._record_download_observation(100_000, 0.01)
    assert cm._cached_permits is None


@pytest.mark.asyncio
async def test_telemetry_duration_excludes_permit_queue_wait(tmp_path):
    """Verify semaphore queue wait time is excluded from recorded telemetry duration."""
    import asyncio
    from unittest.mock import AsyncMock

    from litdata.raw.dataset import StreamingRawDataset

    (tmp_path / "file.jpg").write_bytes(b"12345")
    ds = StreamingRawDataset(input_dir=str(tmp_path), max_prefetch=0)
    cm = ds.cache_manager

    # Force 1 permit max
    cm.max_concurrent_downloads = 1
    cm._downloader = AsyncMock()
    cm._downloader_pid = os.getpid()
    cm._downloader_loop = asyncio.get_running_loop()

    async def mock_download(file_path):
        await asyncio.sleep(0.02)  # actual download takes 20ms
        return b"12345"

    cm.downloader.adownload_fileobj.side_effect = mock_download

    recorded_durations = []

    def mock_record(size, dur):
        recorded_durations.append(dur)

    cm._record_download_observation = mock_record

    # Task 1 holds permit for 0.1s
    async def task1():
        async with cm._permit(True):
            await asyncio.sleep(0.1)

    # Task 2 waits for permit (>= 0.1s wait) then performs fetch
    async def task2():
        await asyncio.sleep(0.01)  # start after task1 holds permit
        await cm._fetch_bytes(str(tmp_path / "file.jpg"), size=5, gated=True)

    await asyncio.gather(task1(), task2())

    assert len(recorded_durations) == 1
    # Duration recorded must measure only task2's actual fetch (20ms), excluding the ~100ms permit queue wait
    assert recorded_durations[0] < 0.08

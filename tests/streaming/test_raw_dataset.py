import os
import sys
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from torch.utils.data import DataLoader

from litdata.streaming.raw_dataset import (
    CacheManager,
    FileIndexer,
    FileMetadata,
    StreamingRawDataset,
)


def test_file_metadata():
    """Test FileMetadata creation and serialization."""
    data = {
        "path": "/path/to/file.jpg",
        "size": 1024,
    }
    metadata = FileMetadata(**data)

    # Basic attribute checks
    assert metadata.path == data["path"]
    assert metadata.size == data["size"]

    # Serialization round-trip
    dict_repr = metadata.to_dict()
    assert dict_repr == data
    metadata2 = FileMetadata.from_dict(dict_repr)
    assert metadata2 == metadata


def test_file_indexer_init():
    """Test FileIndexer initialization."""
    indexer = FileIndexer()
    assert indexer.max_depth == 5
    assert indexer.extensions == []

    indexer = FileIndexer(max_depth=3, extensions=[".jpg", ".png"])
    assert indexer.max_depth == 3
    assert indexer.extensions == [".jpg", ".png"]


def test_should_include_file_no_extensions():
    """Test file inclusion when no extensions filter is set."""
    indexer = FileIndexer()

    assert indexer._should_include_file("/path/to/file.jpg") is True
    assert indexer._should_include_file("/path/to/file.txt") is True
    assert indexer._should_include_file("/path/to/file") is True


def test_should_include_file_with_extensions():
    """Test file inclusion with extensions filter."""
    indexer = FileIndexer(extensions=[".jpg", ".png"])

    assert indexer._should_include_file("/path/to/file.jpg") is True
    assert indexer._should_include_file("/path/to/file.JPG") is True  # Case insensitive
    assert indexer._should_include_file("/path/to/file.png") is True
    assert indexer._should_include_file("/path/to/file.txt") is False
    assert indexer._should_include_file("/path/to/file") is False


def test_discover_local_files(tmp_path):
    """Test local file discovery."""
    # Create test directory structure
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.png").write_text("content2")
    (tmp_path / "file3.txt").write_text("content3")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file4.jpg").write_text("content4")

    indexer = FileIndexer(extensions=[".jpg", ".png"])
    files = indexer._discover_local_files(str(tmp_path))

    # Should find 3 files (.jpg and .png files)
    assert len(files) == 3

    # Check that all returned files are FileMetadata objects
    for file_metadata in files:
        assert isinstance(file_metadata, FileMetadata)
        assert file_metadata.size > 0


@patch("fsspec.filesystem")
def test_discover_cloud_files_s3(mock_filesystem):
    """Test cloud file discovery for S3."""
    # Mock fsspec filesystem
    mock_fs = Mock()
    mock_filesystem.return_value = mock_fs

    # Mock file discovery result
    mock_files = {
        "s3://bucket/file1.jpg": {
            "type": "file",
            "name": "bucket/file1.jpg",
            "size": 1024,
            "LastModified": None,
            "ETag": "abc123",
        },
        "s3://bucket/file2.png": {
            "type": "file",
            "name": "bucket/file2.png",
            "size": 2048,
            "LastModified": None,
            "ETag": "def456",
        },
        "s3://bucket/subdir/": {
            "type": "directory",
            "name": "bucket/subdir/",
        },
    }
    mock_fs.find.return_value = mock_files

    indexer = FileIndexer(extensions=[".jpg", ".png"])
    files = indexer._discover_cloud_files("s3://bucket/", {})

    # Should find 2 files (excluding directory)
    assert len(files) == 2
    assert all(isinstance(f, FileMetadata) for f in files)
    assert all(f.path.startswith("s3://") for f in files)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_build_or_load_index_creates_new(tmp_path):
    """Test that build_or_load_index creates a new index when none exists."""
    # Create test files
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.jpg").write_text("content2")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    indexer = FileIndexer(extensions=[".jpg"])
    files = indexer.build_or_load_index(str(tmp_path), str(cache_dir), {})

    assert len(files) == 2

    # Check that index file was created
    index_file = cache_dir / "index.json.zstd"
    assert index_file.exists()


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_build_or_load_index_loads_existing(tmp_path):
    """Test that build_or_load_index loads existing index when available."""
    # Create test files
    (tmp_path / "file1.jpg").write_text("content1")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    indexer = FileIndexer(extensions=[".jpg"])

    # Build index first time
    files1 = indexer.build_or_load_index(str(tmp_path), str(cache_dir), {})

    # Load index second time (should load from cache)
    with patch.object(indexer, "discover_files") as mock_discover:
        files2 = indexer.build_or_load_index(str(tmp_path), str(cache_dir), {})
        # discover_files should not be called if loading from cache
        mock_discover.assert_not_called()

    assert len(files1) == len(files2)
    assert files1[0].path == files2[0].path


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
def test_streaming_raw_dataset_getitem(tmp_path):
    """Test single item access."""
    test_content = b"test image content"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path))

    # Patch async download to return test_content
    async def mock_download_file_async(file_path):
        return test_content

    with patch.object(dataset.cache_manager, "download_file_async", side_effect=mock_download_file_async):
        item = dataset[0]
        assert item == test_content


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitem_index_error(tmp_path):
    """Test index error for out of range access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

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

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

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

    dataset = StreamingRawDataset(input_dir=str(tmp_path))

    async def mock_download_and_process_item(file_path):
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

    grouped_dataset = GroupedDataset(input_dir=str(tmp_path))

    async def mock_download_and_process_group(file_paths):
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

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

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

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    with pytest.raises(TypeError):
        dataset.__getitems__(0)  # Should be a list


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_getitems_index_error(tmp_path):
    """Test index error for out of range batch access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    with pytest.raises(IndexError, match="list index out of range"):
        dataset.__getitems__([0, 1])


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_with_custom_indexer(tmp_path):
    """Test dataset with custom indexer."""
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.png").write_text("content2")
    (tmp_path / "file3.txt").write_text("content3")

    custom_indexer = FileIndexer(extensions=[".jpg"])

    dataset = StreamingRawDataset(input_dir=str(tmp_path), indexer=custom_indexer, cache_files=False)

    assert len(dataset) == 1  # Only .jpg file should be indexed


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_transform(tmp_path):
    """Test transform support in StreamingRawDataset."""
    test_content = b"raw"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    def transform(x):
        return x.decode() + "_transformed"

    dataset = StreamingRawDataset(input_dir=str(tmp_path), transform=transform)

    # Patch async download to return test_content
    async def mock_download_file_async(file_path):
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
    async def mock_download_async(file_path):
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

    async def fail_download(file_path):
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


def test_file_indexer_should_include_file_edge():
    idx = FileIndexer(extensions=None)
    assert idx._should_include_file("foo.bar") is True
    idx2 = FileIndexer(extensions=[".jpg"])
    assert idx2._should_include_file("foo.txt") is False


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_transform_none_and_group(tmp_path):
    # Single item, no transform
    (tmp_path / "file1.jpg").write_bytes(b"abc")
    ds = StreamingRawDataset(input_dir=str(tmp_path))

    # Patch download to return bytes
    async def mock_download_file_async(file_path):
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

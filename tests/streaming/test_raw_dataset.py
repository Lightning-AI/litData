import os
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


def test_get_local_path(tmp_path):
    """Test local path generation."""
    input_dir = "s3://bucket/dataset"
    cache_dir = str(tmp_path / "cache")

    manager = CacheManager(input_dir=input_dir, cache_dir=cache_dir, cache_files=True)

    file_path = "s3://bucket/dataset/subdir/file.jpg"
    local_path = manager.get_local_path(file_path)

    assert "subdir/file.jpg" in local_path
    assert local_path.startswith(manager.cache_dir)


# @patch("litdata.streaming.raw_dataset.get_downloader")
# def test_download_file_sync_with_caching(mock_get_downloader, tmp_path):
#     """Test synchronous file download with caching."""
#     # Setup mock downloader
#     mock_downloader = Mock()
#     mock_get_downloader.return_value = mock_downloader

#     def mock_download_fileobj(file_path, file_obj):
#         file_obj.write(b"test content")

#     mock_downloader.download_fileobj.side_effect = mock_download_fileobj

#     input_dir = "s3://bucket/dataset"
#     cache_dir = str(tmp_path / "cache")

#     manager = CacheManager(input_dir=input_dir, cache_dir=cache_dir, cache_files=True)

#     file_path = "s3://bucket/dataset/file.jpg"
#     content = manager.download_file_sync(file_path)

#     assert content == b"test content"

#     # Check that file was cached
#     local_path = manager.get_local_path(file_path)
#     assert os.path.exists(local_path)
#     with open(local_path, "rb") as f:
#         assert f.read() == b"test content"


# @patch("litdata.streaming.raw_dataset.get_downloader")
# def test_download_file_sync_without_caching(mock_get_downloader, tmp_path):
#     """Test synchronous file download without caching."""
#     # Setup mock downloader
#     mock_downloader = Mock()
#     mock_get_downloader.return_value = mock_downloader

#     def mock_download_fileobj(file_path, file_obj):
#         file_obj.write(b"test content")

#     mock_downloader.download_fileobj.side_effect = mock_download_fileobj

#     input_dir = "s3://bucket/dataset"

#     manager = CacheManager(input_dir=input_dir, cache_files=False)

#     file_path = "s3://bucket/dataset/file.jpg"
#     content = manager.download_file_sync(file_path)

#     assert content == b"test content"


# @patch("litdata.streaming.raw_dataset.get_downloader")
# def test_download_file_sync_from_cache(mock_get_downloader, tmp_path):
#     """Test that cached files are loaded from disk."""
#     # Setup mock downloader (should not be called)
#     mock_downloader = Mock()
#     mock_get_downloader.return_value = mock_downloader

#     input_dir = "s3://bucket/dataset"
#     cache_dir = str(tmp_path / "cache")

#     manager = CacheManager(input_dir=input_dir, cache_dir=cache_dir, cache_files=True)

#     # Pre-create cached file
#     file_path = "s3://bucket/dataset/file.jpg"
#     local_path = manager.get_local_path(file_path)
#     with open(local_path, "wb") as f:
#         f.write(b"cached content")

#     content = manager.download_file_sync(file_path)

#     assert content == b"cached content"
#     # Downloader should not have been called
#     mock_downloader.download_fileobj.assert_not_called()


# def test_streaming_raw_dataset_init(tmp_path):
#     """Test StreamingRawDataset initialization."""
#     # Create test files
#     (tmp_path / "file1.jpg").write_text("content1")
#     (tmp_path / "file2.jpg").write_text("content2")

#     dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

#     assert len(dataset) == 2
#     assert dataset.cache_manager.cache_files is False


# def test_streaming_raw_dataset_init_with_caching(tmp_path):
#     """Test StreamingRawDataset initialization with caching enabled."""
#     # Create test files
#     (tmp_path / "file1.jpg").write_text("content1")

#     cache_dir = tmp_path / "cache"

#     dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_dir=str(cache_dir), cache_files=True)

#     assert len(dataset) == 1
#     assert dataset.cache_manager.cache_files is True
#     assert os.path.exists(dataset.cache_manager.cache_dir)


# def test_streaming_raw_dataset_len(tmp_path):
#     """Test dataset length."""
#     # Create test files
#     for i in range(5):
#         (tmp_path / f"file{i}.jpg").write_text(f"content{i}")

#     dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

#     assert len(dataset) == 5


def test_streaming_raw_dataset_getitem(tmp_path):
    """Test single item access."""
    test_content = b"test image content"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path))

    with patch.object(dataset.cache_manager, "download_file_sync", return_value=test_content):
        item = dataset[0]
        assert item == test_content


def test_streaming_raw_dataset_getitem_index_error(tmp_path):
    """Test index error for out of range access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    with pytest.raises(IndexError, match="Index 1 out of range"):
        dataset[1]


def test_streaming_raw_dataset_getitems(tmp_path):
    """Test batch item access."""
    test_contents = [b"content1", b"content2", b"content3"]
    for i, content in enumerate(test_contents):
        (tmp_path / f"file{i}.jpg").write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    async def mock_download_batch(indices):
        return [test_contents[i] for i in indices]

    with patch.object(dataset, "_download_batch", side_effect=mock_download_batch):
        items = dataset.__getitems__([0, 2])
        assert items == [test_contents[0], test_contents[2]]


def test_streaming_raw_dataset_getitems_type_error(tmp_path):
    """Test type error for invalid indices type."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    with pytest.raises(TypeError):
        dataset.__getitems__(0)  # Should be a list


def test_streaming_raw_dataset_getitems_index_error(tmp_path):
    """Test index error for out of range batch access."""
    (tmp_path / "file1.jpg").write_text("content1")

    dataset = StreamingRawDataset(input_dir=str(tmp_path), cache_files=False)

    with pytest.raises(IndexError, match="list index out of range"):
        dataset.__getitems__([0, 1])


def test_streaming_raw_dataset_with_custom_indexer(tmp_path):
    """Test dataset with custom indexer."""
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.png").write_text("content2")
    (tmp_path / "file3.txt").write_text("content3")

    custom_indexer = FileIndexer(extensions=[".jpg"])

    dataset = StreamingRawDataset(input_dir=str(tmp_path), indexer=custom_indexer, cache_files=False)

    assert len(dataset) == 1  # Only .jpg file should be indexed


def test_streaming_raw_dataset_transform(tmp_path):
    """Test transform support in StreamingRawDataset."""
    test_content = b"raw"
    (tmp_path / "file1.jpg").write_bytes(test_content)

    def transform(x):
        return x.decode() + "_transformed"

    dataset = StreamingRawDataset(input_dir=str(tmp_path), transform=transform)

    with patch.object(dataset.cache_manager, "download_file_sync", return_value=test_content):
        item = dataset[0]
        assert item == "raw_transformed"


def test_discover_local_files_max_depth(tmp_path):
    """Test local file discovery with max_depth limit (note: not enforced)."""
    (tmp_path / "file1.jpg").write_text("content1")
    level1 = tmp_path / "level1"
    level1.mkdir()
    (level1 / "file2.jpg").write_text("content2")
    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "file3.jpg").write_text("content3")

    # Note: max_depth is not enforced in local file discovery.
    indexer = FileIndexer(max_depth=1, extensions=[".jpg"])
    files = indexer._discover_local_files(str(tmp_path))
    assert len(files) == 3  # All three files

    indexer = FileIndexer(max_depth=2, extensions=[".jpg"])
    files = indexer._discover_local_files(str(tmp_path))
    assert len(files) == 3  # All three files


def test_streaming_raw_dataset_with_dataloader(tmp_path):
    """Test dataset integration with PyTorch DataLoader."""
    # Create test files
    test_contents = [b"content1", b"content2", b"content3", b"content4"]
    for i, content in enumerate(test_contents):
        (tmp_path / f"file{i}.jpg").write_bytes(content)

    dataset = StreamingRawDataset(input_dir=str(tmp_path))

    # Mock download to return test content
    def mock_download_sync(file_path):
        index = int(file_path.split("file")[1].split(".")[0])
        return test_contents[index]

    with patch.object(dataset.cache_manager, "download_file_sync", side_effect=mock_download_sync):
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        batches = list(dataloader)
        assert len(batches) == 2  # 4 items / batch_size 2
        assert len(batches[0]) == 2  # First batch has 2 items
        assert len(batches[1]) == 2  # Second batch has 2 items


def test_streaming_raw_dataset_no_files_error(tmp_path):
    """Test error when no files are found."""
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No files found"):
        StreamingRawDataset(input_dir=str(empty_dir), cache_files=False)


def test_end_to_end_local_files(tmp_path):
    """Test end-to-end functionality with local files."""
    # Create test dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create various file types
    files_data = {
        "image1.jpg": b"fake jpeg data 1",
        "image2.png": b"fake png data 2",
        "document.txt": b"text content",
        "subdir/image3.jpg": b"fake jpeg data 3",
    }

    for file_path, content in files_data.items():
        full_path = dataset_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

    # Test with extension filtering
    dataset = StreamingRawDataset(
        input_dir=str(dataset_dir), indexer=FileIndexer(extensions=[".jpg", ".png"]), cache_files=False
    )

    assert len(dataset) == 3  # 3 image files

    # Test single item access
    item = dataset[0]
    assert isinstance(item, bytes)

    # Test batch access
    batch = dataset.__getitems__([0, 1])
    assert len(batch) == 2
    assert all(isinstance(item, bytes) for item in batch)


def test_dataloader_integration(tmp_path):
    """Test integration with PyTorch DataLoader."""
    # Create test dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    for i in range(10):
        (dataset_dir / f"file{i:02d}.jpg").write_bytes(f"content {i}".encode())

    dataset = StreamingRawDataset(input_dir=str(dataset_dir), cache_files=False)

    # Test with DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,  # Use single process for testing
        shuffle=False,
    )

    batches = list(dataloader)

    # Should have 4 batches: [3, 3, 3, 1]
    assert len(batches) == 4
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1

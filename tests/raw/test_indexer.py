import sys
from unittest.mock import Mock, patch

import pytest

from litdata import StreamingRawDataset
from litdata.raw.indexer import FileIndexer, FileMetadata


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


def test_file_indexer_should_include_file_edge():
    idx = FileIndexer(extensions=None)
    assert idx._should_include_file("foo.bar") is True
    idx2 = FileIndexer(extensions=[".jpg"])
    assert idx2._should_include_file("foo.txt") is False


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


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_streaming_raw_dataset_with_custom_indexer(tmp_path):
    """Test dataset with custom indexer."""
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.png").write_text("content2")
    (tmp_path / "file3.txt").write_text("content3")

    custom_indexer = FileIndexer(extensions=[".jpg"])

    dataset = StreamingRawDataset(input_dir=str(tmp_path), indexer=custom_indexer, cache_files=False)

    assert len(dataset) == 1  # Only .jpg file should be indexed

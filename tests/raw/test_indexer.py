import sys
from unittest.mock import Mock, patch

import pytest

from litdata import StreamingRawDataset
from litdata.constants import _PYTHON_GREATER_EQUAL_3_14
from litdata.raw.indexer import _INDEX_FILENAME, FileIndexer, FileMetadata


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


def test_build_or_load_index_unsupported_scheme(tmp_path):
    """Test that build_or_load_index raises ValueError for unsupported schemes."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    indexer = FileIndexer()
    with pytest.raises(ValueError, match="Unsupported input directory scheme: `ftp`"):
        indexer.build_or_load_index("ftp://unsupported/path", str(cache_dir), {})


def test_discover_files_unsupported_scheme():
    """Test that discover_files raises ValueError for unsupported schemes."""
    indexer = FileIndexer()
    with pytest.raises(ValueError, match="Unsupported input directory scheme: `http`"):
        indexer.discover_files("http://unsupported/path", {})


@patch("litdata.raw.indexer.BaseIndexer._upload_to_cloud")
@patch("litdata.raw.indexer.BaseIndexer._download_from_cloud", side_effect=FileNotFoundError)
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_build_and_cache_remote_index(mock_download, mock_upload, tmp_path):
    """Test that a new index is built, cached locally, and uploaded to remote."""
    input_dir = "s3://my-bucket/data"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    indexer = FileIndexer()
    dummy_files = [FileMetadata("s3://my-bucket/data/file1.txt", 100)]

    with patch.object(indexer, "discover_files", return_value=dummy_files) as mock_discover:
        files = indexer.build_or_load_index(input_dir, str(cache_dir), {})

        assert files == dummy_files
        mock_discover.assert_called_once_with(input_dir, {})

        # Check local cache
        local_index_path = cache_dir / _INDEX_FILENAME
        assert local_index_path.exists()

        # Check remote upload
        remote_index_path = f"{input_dir.rstrip('/')}/{_INDEX_FILENAME}"
        mock_download.assert_called_once_with(remote_index_path, str(local_index_path), {})
        mock_upload.assert_called_once_with(str(local_index_path), remote_index_path, {})


@patch("litdata.raw.indexer.BaseIndexer._upload_to_cloud")
@patch("litdata.raw.indexer.BaseIndexer._download_from_cloud")
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_load_remote_index_from_cache(mock_download, mock_upload, tmp_path):
    """Test loading an index from remote cache when local is empty."""
    input_dir = "s3://my-bucket/data"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    local_index_path = cache_dir / _INDEX_FILENAME

    indexer = FileIndexer()
    dummy_files = [FileMetadata("s3://my-bucket/data/file1.txt", 100)]

    # Simulate successful download by having the mock create the local file
    def fake_download(remote_path, local_path, storage_options):
        import json

        if _PYTHON_GREATER_EQUAL_3_14:
            from compression import zstd
        else:
            import zstd

        metadata = {
            "source": input_dir,
            "files": [f.to_dict() for f in dummy_files],
            "created_at": 0,
        }
        with open(local_path, "wb") as f:
            f.write(zstd.compress(json.dumps(metadata).encode("utf-8")))

    mock_download.side_effect = fake_download

    with patch.object(indexer, "discover_files") as mock_discover:
        files = indexer.build_or_load_index(input_dir, str(cache_dir), {})

        assert files == dummy_files
        mock_discover.assert_not_called()
        remote_index_path = f"{input_dir.rstrip('/')}/{_INDEX_FILENAME}"
        mock_download.assert_called_once_with(remote_index_path, str(local_index_path), {})
        mock_upload.assert_not_called()


@patch("litdata.raw.indexer.BaseIndexer._upload_to_cloud")
@patch("litdata.raw.indexer.BaseIndexer._download_from_cloud")
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_recompute_index_flag_with_cache(mock_download, mock_upload, tmp_path):
    """Test that `recompute_index=True` forces a rebuild even if a cache exists."""
    input_dir = "s3://my-bucket/data"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    local_index_path = cache_dir / _INDEX_FILENAME

    # Create a dummy local cache to ensure it's ignored
    with open(local_index_path, "w") as f:
        f.write("old_index_data")

    indexer = FileIndexer()
    new_dummy_files = [FileMetadata("s3://my-bucket/data/new_file.txt", 200)]

    with patch.object(indexer, "discover_files", return_value=new_dummy_files) as mock_discover:
        files = indexer.build_or_load_index(input_dir, str(cache_dir), {}, recompute_index=True)

        assert files == new_dummy_files
        mock_discover.assert_called_once_with(input_dir, {})
        mock_download.assert_not_called()  # Should not attempt to load from cache
        mock_upload.assert_called_once()


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_recompute_index_excludes_index_file(tmp_path):
    """Test that recomputing the index does not include the index file itself if it exists."""
    # Create test files
    (tmp_path / "file1.jpg").write_text("content1")
    (tmp_path / "file2.jpg").write_text("content2")
    # Create a dummy index file that should be ignored
    (tmp_path / _INDEX_FILENAME).write_text("dummy index content")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Include .zstd to ensure we are not just filtering by extension
    indexer = FileIndexer(extensions=[".jpg", ".zstd"])
    files = indexer.build_or_load_index(str(tmp_path), str(cache_dir), {}, recompute_index=True)

    assert len(files) == 2
    for f in files:
        assert _INDEX_FILENAME not in f.path

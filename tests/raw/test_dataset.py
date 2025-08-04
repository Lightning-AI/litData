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

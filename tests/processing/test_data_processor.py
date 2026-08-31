import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import tempfile
from contextlib import suppress
from functools import partial
from io import BytesIO
from queue import Empty
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning_utilities.core.imports import RequirementCache

from litdata.constants import _INDEX_FILENAME, _ZSTD_AVAILABLE
from litdata.processing import data_processor as data_processor_module
from litdata.processing import functions
from litdata.processing.data_processor import (
    BaseWorker,
    DataChunkRecipe,
    DataProcessor,
    DataRecipe,
    FakeQueue,
    MapRecipe,
    _adaptive_download_concurrency,
    _cache_local_path,
    _chunks_dir,
    _download_data_target,
    _get_item_filesizes,
    _is_local_write_through,
    _is_path,
    _is_remote_path,
    _is_studio_fuse_path,
    _map_items_to_workers_sequentially,
    _map_items_to_workers_weighted,
    _prefetch_maxsize,
    _prepare_items_and_paths,
    _remove_target,
    _to_path,
    _upload_fn,
    _wait_for_disk_usage_higher_than_threshold,
    _wait_for_file_to_exist,
    resolve_keep_data_ordered,
)
from litdata.processing.functions import LambdaDataChunkRecipe, LambdaMapRecipe, _get_input_dir, map, optimize
from litdata.streaming import StreamingDataLoader, StreamingDataset, resolver
from litdata.streaming.cache import Cache, Dir
from litdata.streaming.serializers import _torchcodec_usable


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


_PIL_AVAILABLE = RequirementCache("PIL")


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn(tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    remote_output_dir = os.path.join(tmpdir, "remote_output_dir")
    os.makedirs(remote_output_dir, exist_ok=True)

    filepath = os.path.join(input_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    upload_queue = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return value

    upload_queue.get = fn

    remove_queue = mock.MagicMock()

    assert os.listdir(remote_output_dir) == []

    _upload_fn(upload_queue, remove_queue, cache_dir, Dir(path=remote_output_dir, url=remote_output_dir))

    assert os.listdir(remote_output_dir) == ["a.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_s3_fn(tmpdir, monkeypatch):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    remote_output_dir = os.path.join(tmpdir, "remote_output_dir")
    os.makedirs(remote_output_dir, exist_ok=True)

    filepath = os.path.join(input_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    upload_queue = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return value

    upload_queue.get = fn

    remove_queue = mock.MagicMock()

    fs_provider = mock.MagicMock()

    called = False

    def copy_file(local_filepath, *args):
        nonlocal called
        called = True
        from shutil import copyfile

        copyfile(local_filepath, os.path.join(remote_output_dir, os.path.basename(local_filepath)))

    fs_provider.upload_file = copy_file

    monkeypatch.setattr(data_processor_module, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=mock.MagicMock()))
    monkeypatch.setattr(data_processor_module, "downloader_supports_aupload", lambda _d: False)

    assert os.listdir(remote_output_dir) == []

    assert not called

    _upload_fn(upload_queue, remove_queue, cache_dir, Dir(path=remote_output_dir, url="s3://url"))

    assert called

    assert len(paths) == 0

    assert os.listdir(remote_output_dir) == ["a.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn_reraises_cloud_errors(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    test_file = os.path.join(cache_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("x")

    upload_queue = mock.MagicMock()
    upload_queue.get = mock.Mock(side_effect=[test_file, None])
    fs_provider = mock.MagicMock()
    fs_provider.upload_file.side_effect = RuntimeError("access denied")
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=mock.MagicMock()))
    monkeypatch.setattr(data_processor_module, "downloader_supports_aupload", lambda _d: False)

    with pytest.raises(RuntimeError, match="access denied"):
        _upload_fn(upload_queue, mock.MagicMock(), cache_dir, Dir(path=None, url="s3://bucket/out"))


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn_async_batches(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    files = []
    for name in ("a.bin", "b.bin"):
        path = os.path.join(cache_dir, name)
        with open(path, "w") as handle:
            handle.write(name)
        files.append(path)

    upload_queue = mock.MagicMock()
    upload_queue.get = mock.Mock(side_effect=[*files, None])
    uploaded: list[tuple[str, str]] = []

    class _AsyncUploader:
        async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
            uploaded.append((os.path.basename(local_filepath), remote_filepath))

    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=_AsyncUploader()))
    monkeypatch.setattr(data_processor_module, "downloader_supports_aupload", lambda _d: True)
    monkeypatch.setenv("LITDATA_OPTIMIZE_UPLOAD_BATCH", "2")

    _upload_fn(upload_queue, mock.MagicMock(), cache_dir, Dir(path=None, url="s3://bucket/out"))

    assert {(local, remote) for local, remote in uploaded} == {
        ("a.bin", "s3://bucket/out/a.bin"),
        ("b.bin", "s3://bucket/out/b.bin"),
    }


def test_assert_supported_write_url_rejects_azure(tmpdir):
    from litdata.processing.functions import _assert_supported_write_url

    _assert_supported_write_url(Dir(path=str(tmpdir), url=None))
    _assert_supported_write_url(Dir(path=None, url="s3://bucket/out"))
    with pytest.raises(ValueError, match="azure"):
        _assert_supported_write_url(Dir(path=None, url="azure://container/path"))


def test_resume_fields_from_checkpoint_old_and_new():
    from litdata.processing.data_processor import (
        _resume_fields_from_checkpoint,
        _writer_chunk_index_from_checkpoint,
    )

    chunks = [{"chunk_size": 2}, {"chunk_size": 3}]
    _, inputs_done, next_chunk = _resume_fields_from_checkpoint({"chunks": chunks, "done_till_index": 5})
    assert inputs_done == 5
    assert next_chunk is None
    _, inputs_done, next_chunk = _resume_fields_from_checkpoint(
        {"chunks": chunks, "inputs_done": 4, "samples_written": 5, "next_chunk_index": 7, "done_till_index": 4}
    )
    assert inputs_done == 4
    assert next_chunk == 7

    # Append offset is preserved when a worker has no checkpoint, or only an old one.
    assert _writer_chunk_index_from_checkpoint(2, [], None) == 2
    assert _writer_chunk_index_from_checkpoint(2, chunks, None) == 4
    # New checkpoints store an absolute next_chunk_index — do not add the append offset again.
    assert _writer_chunk_index_from_checkpoint(2, chunks, 7) == 7
    assert _writer_chunk_index_from_checkpoint(0, chunks, 1) == 1


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_remove_target(tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, "a.txt")

    with open(filepath, "w") as f:
        f.write("HERE")

    filepath = os.path.join(input_dir, "a.txt")

    queue_in = mock.MagicMock()

    paths = [filepath, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return [value]

    queue_in.get = fn

    assert os.listdir(cache_dir) == ["a.txt"]

    _remove_target(Dir(path=input_dir), cache_dir, queue_in)

    assert os.listdir(cache_dir) == []


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
@mock.patch("litdata.processing.data_processor._wait_for_disk_usage_higher_than_threshold")
def test_download_data_target(wait_for_disk_usage_higher_than_threshold_mock, tmpdir):
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    remote_input_dir = os.path.join(tmpdir, "remote_input_dir")
    os.makedirs(remote_input_dir, exist_ok=True)

    with open(os.path.join(remote_input_dir, "a.txt"), "w") as f:
        f.write("HERE")

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    queue_in = mock.MagicMock()

    items = [10]
    paths = [os.path.join(input_dir, "a.txt"), None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, items.pop(0), [value])

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty

    queue_out = mock.MagicMock()
    _download_data_target(Dir(input_dir, remote_input_dir), cache_dir, queue_in, queue_out)

    assert queue_out.put._mock_call_args_list[0].args == ((0, 10, [os.path.join(input_dir, "a.txt")]),)
    assert queue_out.put._mock_call_args_list[1].args == (None,)

    assert os.listdir(cache_dir) == ["a.txt"]

    wait_for_disk_usage_higher_than_threshold_mock.assert_called()


def test_download_data_target_preserves_batch_order_on_cache_hit(tmpdir, monkeypatch):
    """A later item must not emit before an earlier one just because the file is already cached."""
    monkeypatch.setenv("LITDATA_OPTIMIZE_DOWNLOAD_BATCH", "8")
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    src = os.path.join(input_dir, "a.txt")
    with open(src, "w") as f:
        f.write("HERE")

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    payloads = [(i, (i * 20, src), [src]) for i in range(3)]
    payloads.append(None)

    class _Queue:
        def __init__(self) -> None:
            self._items = list(payloads)

        def get(self, *_, **__):
            return self._items.pop(0)

        def get_nowait(self):
            if not self._items:
                raise Empty
            return self._items.pop(0)

    emitted: list[Any] = []
    queue_out = mock.MagicMock()
    queue_out.put.side_effect = lambda value: emitted.append(value)

    _download_data_target(Dir(input_dir), cache_dir, _Queue(), queue_out, emit_done=True)

    indexes = [row[0] for row in emitted if row is not None]
    assert indexes == [0, 1, 2]


def test_wait_for_disk_usage_higher_than_threshold():
    disk_usage_mock = mock.Mock(side_effect=[mock.Mock(free=10e9), mock.Mock(free=10e9), mock.Mock(free=10e11)])
    with mock.patch("litdata.processing.data_processor.shutil.disk_usage", disk_usage_mock):
        _wait_for_disk_usage_higher_than_threshold("/", 10, sleep_time=0)
    assert disk_usage_mock.call_count == 3


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_wait_for_file_to_exist(monkeypatch):
    url = "s3://url"
    fs_provider = mock.MagicMock()
    raise_error = [False, False, True]

    def fn(*_, **__):
        return raise_error.pop(0)

    fs_provider.exists = fn
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    _wait_for_file_to_exist(url, sleep_time=0.01)

    assert len(raise_error) == 0

    def fn(*_, **__):
        raise ValueError("HERE")

    fs_provider.exists = fn
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(ValueError, match="HERE"):
        _wait_for_file_to_exist(url, sleep_time=0.01)


def test_cache_dir_cleanup(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "chunks")
    cache_data_dir = os.path.join(tmpdir, "data")

    os.makedirs(cache_dir)

    with open(os.path.join(cache_dir, "a.txt"), "w") as f:
        f.write("Hello World !")

    assert os.listdir(cache_dir) == ["a.txt"]

    data_processor = DataProcessor(input_dir=str(tmpdir))
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(cache_dir))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(cache_data_dir))
    data_processor._cleanup_cache()

    assert os.listdir(cache_dir) == []


def test_map_items_to_workers_weighted(monkeypatch):
    seed_everything(42)

    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[1, 4, 2, 0, 3]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[2, 4, 0], [3, 1]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(5)))
    assert workers_user_items == [[0, 3], [4, 1], [2]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(5)))
    assert workers_user_items == [[4, 0], [1], [2], [3]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[2, 0, 4]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[0, 4], [1]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(5)))
    assert workers_user_items == [[3, 1]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)))
    assert workers_user_items == [[2], [3]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(32)))
    assert workers_user_items == [[0, 24, 28, 4, 16, 20, 8, 12]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(32)))
    assert workers_user_items == [[24, 16, 0, 8], [1, 17, 9, 25]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(32)))
    assert workers_user_items == [[24, 12, 0], [13, 25, 1], [14, 2, 26]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(32)))
    assert workers_user_items == [[16, 0], [1, 17], [2, 18], [3, 19]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "3")
    workers_user_items = _map_items_to_workers_weighted(1, list(range(32)))
    assert workers_user_items == [[3, 7, 19, 31, 11, 23, 27, 15]]
    workers_user_items = _map_items_to_workers_weighted(2, list(range(32)))
    assert workers_user_items == [[14, 22, 6, 30], [15, 31, 23, 7]]
    workers_user_items = _map_items_to_workers_weighted(3, list(range(32)))
    assert workers_user_items == [[21, 9], [22, 10], [23, 11]]
    workers_user_items = _map_items_to_workers_weighted(4, list(range(32)))
    assert workers_user_items == [[12, 28], [13, 29], [30, 14], [15, 31]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "1")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_weighted(2, list(range(5)), weights=[1, 2, 3, 4, 5])
    assert workers_user_items == [[4, 0, 1], [3, 2]]


def test_map_items_per_node_for_shared_queue(monkeypatch):
    """Unordered packing uses one bin per node (`num_workers=1` in the mapper)."""
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    node0 = _map_items_to_workers_sequentially(1, list(range(8)))
    assert node0 == [[0, 1, 2, 3]]
    assert len(node0) == 1

    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    node1 = _map_items_to_workers_sequentially(1, list(range(8)))
    assert node1 == [[4, 5, 6, 7]]
    assert not set(node0[0]) & set(node1[0])


def test_map_items_to_workers_sequentially(monkeypatch):
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [list(range(5))]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[0, 1], [2, 3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(5)))
    assert workers_user_items == [[0], [1, 2], [3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(5)))
    assert workers_user_items == [[0], [1], [2], [3, 4]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [[0, 1]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[0], [1]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)))
    assert workers_user_items == [[2, 3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    assert workers_user_items == [[2], [3, 4]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(32)))
    assert workers_user_items == [[0, 1, 2, 3, 4, 5, 6, 7]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(32)))
    assert workers_user_items == [[0, 1, 2, 3], [4, 5, 6, 7]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(32)))
    assert workers_user_items == [[0, 1], [2, 3], [4, 5]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(32)))
    assert workers_user_items == [[0, 1], [2, 3], [4, 5], [6, 7]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "4")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "3")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(32)))
    assert workers_user_items == [[24, 25, 26, 27, 28, 29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(32)))
    assert workers_user_items == [[24, 25, 26, 27], [28, 29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(3, list(range(32)))
    assert workers_user_items == [[23, 24, 25], [26, 27, 28], [29, 30, 31]]
    workers_user_items = _map_items_to_workers_sequentially(4, list(range(32)))
    assert workers_user_items == [[24, 25], [26, 27], [28, 29], [30, 31]]


def test_map_items_to_workers_sequentially_align_chunking(monkeypatch):
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)), chunk_size=2)
    assert workers_user_items == [list(range(5))]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)), chunk_size=2)
    assert workers_user_items == [[0, 1], [2, 3, 4]]
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(6)), chunk_size=2)
    assert workers_user_items == [[0, 1], [2, 3, 4, 5]]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)), chunk_size=2)
    assert workers_user_items == [[0, 1]]

    # 2 nodes, 2 workers per node, chunk_size=2.
    # Total items = 5 => only the final worker should receive them,
    # because no worker except the last can form even one full chunk. (5/ (2*2*2) = 0.625 ~ 0)
    with pytest.warns(UserWarning, match="Consider reducing chunk_size or using fewer workers"):
        workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)), chunk_size=2)
    assert workers_user_items == [[], []]

    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    workers_user_items = _map_items_to_workers_sequentially(1, list(range(5)), chunk_size=2)
    assert workers_user_items == [[2, 3, 4]]

    # On node 1 (rank 1), last worker should receive all items.
    workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)), chunk_size=2)
    assert workers_user_items == [[], [0, 1, 2, 3, 4]]


def test_fake_queue():
    q = FakeQueue()
    index = [1, 2]
    items = ["a", "b"]
    paths = ["p1", "p2"]

    q.add_items(index, items, paths)

    assert q.get() == (1, "a", "p1")
    assert q.get() == (2, "b", "p2")

    with pytest.raises(Empty):
        q.get()


class CustomDataChunkRecipe(DataChunkRecipe):
    is_generator = False

    def prepare_structure(self, input_dir: str) -> list[Any]:
        filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        assert len(filepaths) == 30
        return filepaths

    def prepare_item(self, item):
        return item


class DummyDataChunkRecipe(DataChunkRecipe):
    is_generator = False

    def prepare_structure(self, input_dir: str) -> list[Any]:
        return []

    def prepare_item(self, item):
        return item


@pytest.mark.parametrize("delete_cached_files", [True])
@pytest.mark.parametrize("fast_dev_run", [10])
@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processsor(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir)

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    cache_data_dir = os.path.join(tmpdir, "cache", "data")
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_data_dir)

    data_processor = DataProcessor(
        input_dir=input_dir,
        num_workers=2,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        keep_data_ordered=True,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

    fast_dev_run_enabled_chunks = [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-0-4.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
        "chunk-1-4.bin",
        "index.json",
    ]

    fast_dev_run_disabled_chunks = [
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-0-4.bin",
        "chunk-0-5.bin",
        "chunk-0-6.bin",
        "chunk-0-7.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
        "chunk-1-4.bin",
        "chunk-1-5.bin",
        "chunk-1-6.bin",
        "chunk-1-7.bin",
        "index.json",
    ]

    chunks = fast_dev_run_enabled_chunks if fast_dev_run == 10 else fast_dev_run_disabled_chunks

    assert sorted(os.listdir(cache_dir)) == chunks

    files = []
    for _, _, filenames in os.walk(os.path.join(cache_dir, "data")):
        files.extend(filenames)

    expected = (0 if delete_cached_files else 20) if fast_dev_run == 10 else (0 if delete_cached_files else 30)
    assert len(files) == expected


def test_data_processor_align_chunking_requires_chunk_size(tmpdir):
    output_dir = str(tmpdir / "output_dir")
    data_processor = DataProcessor(input_dir=Dir(), output_dir=output_dir, num_workers=1, align_chunking=True)
    with pytest.raises(ValueError, match="`chunk_size` is not defined in the data recipe"):
        data_processor.run(
            DummyDataChunkRecipe(
                chunk_bytes="10MB"  # chunk_size is not defined here to trigger the error
            )
        )


class TestDataProcessor(DataProcessor):
    def _broadcast_object(self, obj: Any) -> Any:
        return obj


@pytest.mark.parametrize("delete_cached_files", [False])
@pytest.mark.parametrize("fast_dev_run", [False])
@pytest.mark.skipif(
    condition=(not _PIL_AVAILABLE or sys.platform == "win32" or sys.platform == "linux"), reason="Requires: ['pil']"
)
def test_data_processsor_distributed(fast_dev_run, delete_cached_files, tmpdir, monkeypatch):
    """Ensures the data optimizer works in a fully distributed settings."""
    seed_everything(42)

    monkeypatch.setattr(data_processor_module.os, "_exit", mock.MagicMock())

    from PIL import Image

    input_dir = os.path.join(tmpdir, "dataset")
    os.makedirs(input_dir)

    imgs = []
    for i in range(30):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)

    remote_output_dir = os.path.join(tmpdir, "dst")
    os.makedirs(remote_output_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    data_cache_dir = os.path.join(tmpdir, "data_cache_1")
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    data_processor = TestDataProcessor(
        input_dir=input_dir,
        num_workers=2,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        output_dir=remote_output_dir,
        num_uploaders=1,
        num_downloaders=1,
        keep_data_ordered=True,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

    fast_dev_run_disabled_chunks_0 = [
        "0-index.json",
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-0-2.bin",
        "chunk-0-3.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-1-2.bin",
        "chunk-1-3.bin",
    ]

    assert sorted(os.listdir(remote_output_dir)) == fast_dev_run_disabled_chunks_0

    cache_dir = os.path.join(tmpdir, "cache_2")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "1")
    data_processor = TestDataProcessor(
        input_dir=input_dir,
        num_workers=2,
        num_uploaders=1,
        num_downloaders=1,
        delete_cached_files=delete_cached_files,
        fast_dev_run=fast_dev_run,
        output_dir=remote_output_dir,
        keep_data_ordered=True,
    )
    data_processor.run(CustomDataChunkRecipe(chunk_size=2))

    fast_dev_run_disabled_chunks_1 = [
        "chunk-2-0.bin",
        "chunk-2-1.bin",
        "chunk-2-2.bin",
        "chunk-2-3.bin",
        "chunk-3-0.bin",
        "chunk-3-1.bin",
        "chunk-3-2.bin",
        "chunk-3-3.bin",
        "index.json",
    ]

    expected = sorted(fast_dev_run_disabled_chunks_0 + fast_dev_run_disabled_chunks_1 + ["1-index.json"])

    assert sorted(os.listdir(remote_output_dir)) == expected


class TextTokenizeRecipe(DataChunkRecipe):
    is_generator = True

    def prepare_structure(self, input_dir: str) -> list[Any]:
        return [os.path.join(input_dir, "dummy.txt")]

    def prepare_item(self, filepath):
        for _ in range(100):
            yield torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processsor_nlp(tmpdir, monkeypatch):
    seed_everything(42)

    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", os.path.join(tmpdir, "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", os.path.join(tmpdir, "data"))

    with open(os.path.join(tmpdir, "dummy.txt"), "w") as f:
        f.write("Hello World !")

    data_processor = DataProcessor(input_dir=str(tmpdir), num_workers=1, num_downloaders=1)
    data_processor.run(TextTokenizeRecipe(chunk_size=1024 * 11))

    data_processor_more_wokers = DataProcessor(input_dir=str(tmpdir), num_workers=2, num_downloaders=1)
    data_processor_more_wokers.run(TextTokenizeRecipe(chunk_size=1024 * 11))


class ImageResizeRecipe(MapRecipe):
    def prepare_structure(self, input_dir: str):
        filepaths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
        return [filepath for filepath in filepaths if os.path.isfile(filepath)]

    def prepare_item(self, filepath: Any, output_dir: str, is_last) -> None:
        from PIL import Image

        img = Image.open(filepath)
        img = img.resize((12, 12))
        assert os.path.exists(output_dir)
        img.save(os.path.join(output_dir, os.path.basename(filepath)))


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_process_transform(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir)

    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    data_processor = DataProcessor(
        input_dir=input_dir,
        num_workers=1,
        output_dir=output_dir,
        fast_dev_run=False,
    )
    data_processor.run(ImageResizeRecipe())

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]

    from PIL import Image

    img = Image.open(os.path.join(output_dir, "0.JPEG"))
    assert img.size == (12, 12)


def map_fn(filepath, output_dir):
    from PIL import Image

    img = Image.open(filepath)
    img = img.resize((12, 12))
    assert os.path.exists(output_dir)
    img.save(os.path.join(output_dir, os.path.basename(filepath)))


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    map(map_fn, inputs, output_dir=output_dir, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]

    from PIL import Image

    img = Image.open(os.path.join(output_dir, "0.JPEG"))
    assert img.size == (12, 12)


def optimize_fn(filepath):
    from PIL import Image

    return [Image.open(filepath), os.path.basename(filepath)]


def _key_from_sample(sample):
    return sample["id"]


def _sample_optimize_fn(index):
    return {"id": f"item-{index}", "value": index}


def _identity_optimize_fn(index):
    return index


def _map_copy_fn(path, output_dir):
    name = os.path.basename(path)
    with open(path) as src, open(os.path.join(output_dir, name), "w") as dst:
        dst.write(src.read())


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(optimize_fn, inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_optimize_keep_data_ordered_false_shared_node_queue(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    for i in range(6):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        Image.fromarray(np_data).convert("L").save(os.path.join(input_dir, f"{i}.JPEG"))

    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)
    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    inputs = [os.path.join(input_dir, f"{i}.JPEG") for i in range(6)]
    optimize(
        optimize_fn,
        inputs,
        output_dir=output_dir,
        chunk_size=2,
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
    )

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 6


def test_resolve_keep_data_ordered_default_and_guards():
    assert resolve_keep_data_ordered(None) is False
    assert resolve_keep_data_ordered(None, use_checkpoint=True) is True
    assert resolve_keep_data_ordered(None, align_chunking=True) is True
    assert resolve_keep_data_ordered(True) is True
    assert resolve_keep_data_ordered(False) is False
    with pytest.raises(ValueError, match="Checkpoint feature is not supported"):
        resolve_keep_data_ordered(False, use_checkpoint=True)
    with pytest.raises(ValueError, match="align_chunking requires keep_data_ordered=True"):
        resolve_keep_data_ordered(False, align_chunking=True)


def test_prefetch_maxsize_byte_budget(tmp_path, monkeypatch):
    monkeypatch.delenv("LITDATA_PREFETCH_BYTES", raising=False)
    assert _prefetch_maxsize(4) == 8
    paths = []
    for i in range(8):
        path = tmp_path / f"{i}.bin"
        path.write_bytes(b"x" * (64 * 1024 * 1024))
        paths.append(str(path))
    monkeypatch.setenv("LITDATA_PREFETCH_BYTES", str(128 * 1024 * 1024))
    assert _prefetch_maxsize(8, paths) == 2


def test_data_processor_default_is_unordered():
    processor = DataProcessor(input_dir=Dir(), num_workers=1)
    assert processor.keep_data_ordered is False
    processor = DataProcessor(input_dir=Dir(), num_workers=1, use_checkpoint=True)
    assert processor.keep_data_ordered is True
    processor = DataProcessor(input_dir=Dir(), num_workers=1, align_chunking=True)
    assert processor.keep_data_ordered is True


def test_is_local_write_through_and_chunks_dir(tmp_path):
    local = Dir(path=str(tmp_path / "out"), url=None)
    remote = Dir(path=str(tmp_path / "out"), url="s3://bucket/out")
    lightning = Dir(
        path="/teamspace/lightning_storage/testing/out",
        url="r2://bucket/out",
        data_connection_id="conn-1",
    )
    empty = Dir(path=None, url=None)
    assert _is_local_write_through(local) is True
    assert _is_local_write_through(remote) is False
    assert _is_local_write_through(lightning) is False
    assert _is_local_write_through(empty) is False
    assert _is_local_write_through(None) is False
    assert _chunks_dir(local) == local.path
    assert _chunks_dir(remote) != remote.path
    assert _chunks_dir(lightning) != lightning.path
    assert os.path.isdir(local.path)


def test_adaptive_download_concurrency_env_and_jobs(monkeypatch):
    monkeypatch.delenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY", raising=False)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_WORKERS", "4")
    assert _adaptive_download_concurrency(3) == 3
    monkeypatch.setenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY", "2")
    assert _adaptive_download_concurrency(10) == 2
    monkeypatch.setenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY", "32")
    assert _adaptive_download_concurrency(5) == 5


def test_adaptive_download_concurrency_disk_backoff(monkeypatch):
    monkeypatch.delenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY", raising=False)
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_WORKERS", "16")

    class _Usage:
        def __init__(self, free, total):
            self.free = free
            self.total = total

    monkeypatch.setattr(data_processor_module.shutil, "disk_usage", lambda _path: _Usage(5, 100))
    assert _adaptive_download_concurrency(32, num_workers=16) == 4
    monkeypatch.setattr(data_processor_module.shutil, "disk_usage", lambda _path: _Usage(20, 100))
    assert _adaptive_download_concurrency(32, num_workers=16) == 8


def test_n_chunk_writers_and_upload_threads_write_through(tmp_path, monkeypatch):
    monkeypatch.delenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", raising=False)
    local = Dir(path=str(tmp_path / "out"), url=None)
    processor = DataProcessor(input_dir=Dir(), output_dir=local, num_workers=8, verbose=False)
    assert processor._n_chunk_writers() == 0
    monkeypatch.setenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", "1")
    processor = DataProcessor(input_dir=Dir(), output_dir=local, num_workers=8, verbose=False)
    assert processor._n_chunk_writers() == 2
    assert processor._n_upload_threads() == 0
    processor = DataProcessor(input_dir=Dir(), output_dir=local, num_workers=2, verbose=False)
    assert processor._n_chunk_writers() == 1
    processor = DataProcessor(input_dir=Dir(), output_dir=local, num_workers=8, keep_data_ordered=True, verbose=False)
    assert processor._n_chunk_writers() == 0
    processor = DataProcessor(input_dir=Dir(), output_dir=local, num_workers=8, use_checkpoint=True, verbose=False)
    assert processor._n_chunk_writers() == 0
    remote = Dir(path=str(tmp_path / "out"), url="s3://bucket/out")
    processor = DataProcessor(input_dir=Dir(), output_dir=remote, num_workers=8, verbose=False)
    assert processor._n_upload_threads() >= 2
    r2 = Dir(path=str(tmp_path / "fuse"), url="r2://bucket/out", data_connection_id="conn-1")
    processor = DataProcessor(input_dir=Dir(), output_dir=r2, num_workers=8, verbose=False)
    assert processor._n_upload_threads() >= 2
    assert processor.storage_options["data_connection_id"] == "conn-1"


def test_node_removers_start_when_input_dir_empty(tmp_path, monkeypatch):
    """HF optimize has no input Dir; removers must still delete uploaded cache chunks."""
    remote = Dir(path=str(tmp_path / "fuse"), url="r2://bucket/out", data_connection_id="conn-1")
    processor = DataProcessor(
        input_dir=Dir(), output_dir=remote, num_workers=1, verbose=False, delete_cached_files=True
    )
    started: list = []

    def _fake_start(fn, *args):
        started.append(fn)
        return mock.Mock(is_alive=lambda: False)

    monkeypatch.setattr(processor, "_start_io_thread", _fake_start)
    monkeypatch.setattr(processor, "_n_upload_threads", lambda: 0)
    processor._start_node_io_pools()
    assert data_processor_module._remove_target in started
    assert processor.shared_remove_queue is not None


def test_done_merges_index_despite_leftover_cache_bins_for_lightning_storage(tmp_path, monkeypatch):
    """Dir with FUSE path + R2 url must not abort merge on leftover ``.bin`` in the shared cache."""
    cache_dir = tmp_path / "chunks"
    cache_dir.mkdir()
    (cache_dir / "stale-from-other-job.bin").write_bytes(b"stale")
    monkeypatch.setattr(data_processor_module, "_get_cache_dir", lambda name=None: str(cache_dir))
    monkeypatch.setattr(data_processor_module, "_get_num_nodes", lambda: 1)
    monkeypatch.setattr(data_processor_module, "_get_node_rank", lambda: 0)
    monkeypatch.setattr(data_processor_module, "_put_files_remote", lambda *a, **k: (None, None))

    _write_worker_index(str(cache_dir), "C.bin")
    recipe = _AppendChunkRecipe()
    dest = Dir(path=str(tmp_path / "fuse"), url="r2://bucket/out", data_connection_id="conn-1")
    result = recipe._done(size=None, delete_cached_files=True, output_dir=dest)

    index_path = os.path.join(str(cache_dir), _INDEX_FILENAME)
    assert os.path.isfile(index_path)
    assert _chunk_names(index_path) == ["C.bin"]
    assert result.num_chunks == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_optimize_local_write_through_skips_cache_copy(tmpdir, monkeypatch):
    cache_dir = str(tmpdir / "chunks")
    data_cache = str(tmpdir / "data")
    output_dir = str(tmpdir / "out")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache)
    optimize(
        fn=_identity_optimize_fn,
        inputs=list(range(8)),
        output_dir=output_dir,
        chunk_size=2,
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
        verbose=False,
    )
    assert os.path.isfile(os.path.join(output_dir, "index.json"))
    bins = [name for name in os.listdir(output_dir) if name.endswith(".bin")]
    assert bins
    cache_bins = [name for name in os.listdir(cache_dir) if name.endswith(".bin")] if os.path.isdir(cache_dir) else []
    assert cache_bins == []
    ds = StreamingDataset(output_dir)
    assert sorted(ds[:]) == list(range(8))


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_optimize_writer_split_two_workers(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", "1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    output_dir = str(tmpdir / "out")
    optimize(
        fn=_identity_optimize_fn,
        inputs=list(range(12)),
        output_dir=output_dir,
        chunk_size=3,
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
        verbose=False,
    )
    ds = StreamingDataset(output_dir)
    assert sorted(ds[:]) == list(range(12))


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_optimize_writer_split_four_workers_two_writers(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", "1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    output_dir = str(tmpdir / "out")
    optimize(
        fn=_identity_optimize_fn,
        inputs=list(range(16)),
        output_dir=output_dir,
        chunk_size=2,
        num_workers=4,
        keep_data_ordered=False,
        reorder_files=False,
        verbose=False,
    )
    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 16
    assert {cache[i] for i in range(16)} == set(range(16))
    ranks = {name.split("-")[1] for name in os.listdir(output_dir) if name.startswith("chunk-")}
    assert ranks <= {"0", "1"}


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
@pytest.mark.skipif(not RequirementCache("polars"), reason="Requires polars")
def test_optimize_writer_split_with_key_fn(tmpdir, monkeypatch):
    monkeypatch.setenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", "1")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    output_dir = str(tmpdir / "out")

    optimize(
        fn=_sample_optimize_fn,
        inputs=list(range(8)),
        output_dir=output_dir,
        chunk_size=2,
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
        key_fn=_key_from_sample,
        verbose=False,
    )
    ds = StreamingDataset(output_dir, shuffle=False)
    assert ds["item-3"]["value"] == 3
    assert len(ds) == 8


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_optimize_unordered_checkpoint_not_supported(tmpdir, monkeypatch):
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    with pytest.raises(ValueError, match="Checkpoint feature is not supported"):
        optimize(
            fn=lambda x: x,
            inputs=list(range(4)),
            output_dir=str(tmpdir / "out"),
            chunk_size=2,
            num_workers=2,
            keep_data_ordered=False,
            use_checkpoint=True,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_optimize_unordered_multi_node_env_splits_items(tmpdir, monkeypatch):
    monkeypatch.setenv("DATA_OPTIMIZER_NUM_NODES", "2")
    monkeypatch.setenv("DATA_OPTIMIZER_NODE_RANK", "0")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    output_dir = str(tmpdir / "out")
    optimize(
        fn=_identity_optimize_fn,
        inputs=list(range(8)),
        output_dir=output_dir,
        chunk_size=2,
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
        verbose=False,
    )
    # Rank 0 of 2 writes ``0-index.json`` only; the last node merges ``index.json``.
    node_index = os.path.join(output_dir, "0-index.json")
    assert os.path.isfile(node_index)
    with open(node_index) as handle:
        chunks = json.load(handle)["chunks"]
    n_samples = sum(c.get("dim") or c["chunk_size"] for c in chunks)
    assert n_samples == 4


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_map_keep_data_ordered_false_two_workers(tmpdir, monkeypatch):
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", str(tmpdir / "chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", str(tmpdir / "data"))
    input_dir = tmpdir / "in"
    output_dir = tmpdir / "out"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    inputs = []
    for i in range(6):
        path = str(input_dir / f"{i}.txt")
        with open(path, "w") as handle:
            handle.write(str(i))
        inputs.append(path)

    map(
        fn=_map_copy_fn,
        inputs=inputs,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        num_workers=2,
        keep_data_ordered=False,
        reorder_files=False,
    )
    assert sorted(os.listdir(output_dir)) == [f"{i}.txt" for i in range(6)]


def generate_data(index, shift=None):
    yield from range(index + shift if shift else 0)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_yield(monkeypatch, tmpdir):
    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "output_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    optimize(partial(generate_data, shift=2), [0, 1], output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]


class Optimize:
    def __call__(self, filepath):
        from PIL import Image

        return [Image.open(filepath), os.path.basename(filepath)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_class(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(Optimize(), inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


class OptimizeYield:
    def __call__(self, filepath):
        from PIL import Image

        for _ in range(1):
            yield [Image.open(filepath), os.path.basename(filepath)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_optimize_class_yield(monkeypatch, tmpdir):
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)
    imgs = []
    for i in range(5):
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
        img = Image.fromarray(np_data).convert("L")
        imgs.append(img)
        img.save(os.path.join(input_dir, f"{i}.JPEG"))

    home_dir = os.path.join(tmpdir, "home")
    cache_dir = os.path.join(tmpdir, "cache", "chunks")
    data_cache_dir = os.path.join(tmpdir, "cache", "data")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_HOME_FOLDER", home_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    inputs = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir)]
    inputs = [filepath for filepath in inputs if os.path.isfile(filepath)]

    monkeypatch.setattr(functions, "_get_input_dir", lambda x: input_dir)

    optimize(OptimizeYield(), inputs, output_dir=output_dir, chunk_size=2, num_workers=1)

    assert sorted(os.listdir(output_dir)) == ["chunk-0-0.bin", "chunk-0-1.bin", "chunk-0-2.bin", "index.json"]

    cache = Cache(output_dir, chunk_size=1)
    assert len(cache) == 5


def _noop_map_item(item: Any, output_dir: str) -> None:
    return None


def test_lambda_recipe_pickle_drops_inputs():
    inputs = [bytearray(1024)]
    chunk_recipe = LambdaDataChunkRecipe(str, inputs, 1, None, None)
    map_recipe = LambdaMapRecipe(_noop_map_item, inputs)
    pickled_chunk = pickle.loads(pickle.dumps(chunk_recipe))  # noqa: S301
    pickled_map = pickle.loads(pickle.dumps(map_recipe))  # noqa: S301

    assert chunk_recipe.prepare_structure(None) is inputs
    assert map_recipe.prepare_structure(None) is inputs
    assert pickled_chunk.prepare_structure(None) is None
    assert pickled_map.prepare_structure(None) is None
    assert pickled_chunk.prepare_item(123) == "123"


def test_lambda_transform_recipe(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.cuda.device_count.return_value = 3

    monkeypatch.setattr(functions, "torch", torch_mock)
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", 2)

    called = False

    def fn(output_dir, item, device):
        nonlocal called
        assert device == "cuda:2"
        called = True

    data_recipe = LambdaMapRecipe(fn, range(1))

    data_recipe.prepare_item(1, "", False)
    assert called


def test_lambda_transform_recipe_class(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.cuda.device_count.return_value = 3

    monkeypatch.setattr(functions, "torch", torch_mock)
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", 2)

    called = False

    class Transform:
        def __call__(self, item, output_dir, device):
            nonlocal called
            assert device == "cuda:2"
            called = True

    data_recipe = LambdaMapRecipe(Transform(), range(1))
    data_recipe.prepare_item(1, "", False)
    assert called


def _generate_file_with_size(file_path, num_bytes):
    assert num_bytes % 8 == 0
    content = bytearray(random.getrandbits(8) for _ in range(num_bytes))
    with open(file_path, "wb") as file:
        file.write(content)


def test_get_item_filesizes(tmp_path):
    _generate_file_with_size(tmp_path / "file1", 32)
    _generate_file_with_size(tmp_path / "file2", 64)
    _generate_file_with_size(tmp_path / "file3", 128)
    _generate_file_with_size(tmp_path / "file4", 256)

    items = [
        # not a path
        "not a path",
        # single file path
        str(tmp_path / "file1"),
        # tuple: one file path
        (1, 2, str(tmp_path / "file2")),
        # list: two file paths
        [str(tmp_path / "file2"), None, str(tmp_path / "file3")],
        # list: one file path exists, one does not
        [str(tmp_path / "other" / "other"), None, str(tmp_path / "file4")],
        # dict: with file path
        {"file": str(tmp_path / "file4"), "data": "not file"},
    ]
    num_bytes = _get_item_filesizes(items, base_path=str(tmp_path))
    assert num_bytes == [0, 32, 64, 64 + 128, 256, 256]

    with open(tmp_path / "empty_file", "w"):
        pass
    assert os.path.getsize(tmp_path / "empty_file") == 0
    with pytest.raises(RuntimeError, match="has 0 bytes!"):
        _get_item_filesizes([str(tmp_path / "empty_file")])


def map_fn_index(index, output_dir):
    with open(os.path.join(output_dir, f"{index}.JPEG"), "w") as f:
        f.write("Hello")


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_without_input_dir_local(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    map(
        map_fn_index,
        list(range(5)),
        output_dir=output_dir,
        num_workers=1,
        reorder_files=True,
        weights=[1 for _ in range(5)],
    )

    assert sorted(os.listdir(output_dir)) == ["0.JPEG", "1.JPEG", "2.JPEG", "3.JPEG", "4.JPEG"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows not supported")
def test_data_processing_map_without_input_dir_remote(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    resolved_output = tmpdir / "output"
    os.makedirs(resolved_output, exist_ok=True)
    output_dir = os.path.join("/teamspace", "datasets", "target_dir")

    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "1")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "2")
    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "3")
    monkeypatch.setattr("litdata.processing.data_processor._IS_IN_STUDIO", True)
    monkeypatch.setattr(
        "litdata.streaming.resolver._resolve_datasets",
        Mock(return_value=Dir(path=str(resolved_output), url="url")),
    )

    map(
        map_fn_index,
        list(range(5)),
        output_dir=output_dir,
        num_workers=1,
    )

    assert sorted(os.listdir(resolved_output)) == [f"{i}.JPEG" for i in range(5)]


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_weights_mismatch(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    with pytest.raises(ValueError, match="The provided weights length"):
        map(map_fn_index, list(range(5)), output_dir=output_dir, num_workers=1, reorder_files=True, weights=[1])


def map_fn_index_folder(index, output_dir):
    os.makedirs(os.path.join(output_dir, str(index)))
    with open(os.path.join(output_dir, str(index), f"{index}.JPEG"), "w") as f:
        f.write("Hello")


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "win32", reason="Requires: ['pil']")
def test_data_processing_map_without_input_dir_and_folder(monkeypatch, tmpdir):
    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    map(map_fn_index_folder, list(range(5)), output_dir=output_dir, num_workers=1, reorder_files=True)

    assert sorted(os.listdir(output_dir)) == ["0", "1", "2", "3", "4"]
    assert os.path.exists(os.path.join(output_dir, "0", "0.JPEG"))


def map_fn_map_non_absolute(path, output_dir):
    with open(os.path.join(output_dir, os.path.basename(path)), "w") as f:
        f.write("Hello World")


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows")
def test_data_processing_map_non_absolute_path(monkeypatch, tmpdir):
    monkeypatch.chdir(str(tmpdir))

    for i in range(5):
        with open(f"./{i}.txt", "w") as f:
            f.write("Hello World")

    assert sorted(os.listdir(tmpdir)) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt"]

    map(
        map_fn_map_non_absolute,
        [f"{i}.txt" for i in range(5)],
        output_dir="./output_dir",
        num_workers=1,
        reorder_files=True,
    )

    assert sorted(os.listdir(tmpdir)) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt", "output_dir"]
    assert sorted(os.listdir(os.path.join(tmpdir, "output_dir"))) == ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt"]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_map_error_when_not_empty(monkeypatch):
    fs_provider = mock.MagicMock()
    fs_provider.is_empty = mock.MagicMock(return_value=False)
    monkeypatch.setattr(resolver, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(RuntimeError, match="data and datasets are meant to be immutable"):
        map(
            map_fn,
            [0, 1],
            output_dir=Dir(path=None, url="s3://bucket"),
            error_when_not_empty=True,
        )


def map_fn_is_last(index, output_dir, is_last):
    with open(os.path.join(output_dir, f"{index}_{is_last}.txt"), "w") as f:
        f.write("here")


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
@pytest.mark.parametrize(
    ("num_workers", "expected"),
    [
        (1, ["0_False.txt", "1_False.txt", "2_False.txt", "3_False.txt", "4_True.txt"]),
        (2, ["0_False.txt", "1_True.txt", "2_False.txt", "3_False.txt", "4_True.txt"]),
    ],
)
def test_map_is_last(num_workers, expected, tmpdir):
    map(
        map_fn_is_last,
        list(range(5)),
        output_dir=str(tmpdir),
        error_when_not_empty=False,
        num_workers=num_workers,
        keep_data_ordered=True,
    )

    assert sorted(os.listdir(tmpdir)) == expected


def map_batch_size_fn(indexes, output_dir):
    path = os.path.join(output_dir, str(indexes))
    with open(path, "w") as f:
        f.write("hello world")


def test_map_batch_size(tmpdir):
    map(
        map_batch_size_fn,
        list(range(5)),
        output_dir=str(tmpdir),
        error_when_not_empty=False,
        num_workers=1,
        batch_size=2,
    )

    assert sorted(os.listdir(tmpdir)) == ["[0, 1]", "[2, 3]", "[4]"]


def no_op(index):
    pass


@pytest.mark.parametrize("inputs", [[1], [1, 2], [1, 2, 3]])
def test_empty_optimize(tmpdir, inputs):
    optimize(
        no_op,
        inputs,
        output_dir=str(tmpdir),
        chunk_bytes="64MB",
        num_workers=1,
        optimize_dns=False,
    )

    assert os.listdir(tmpdir) == ["index.json"]


def create_synthetic_audio_bytes(index) -> dict:
    import soundfile as sf

    # load dummy audio as bytes
    data = torch.randn((1, 16000)).numpy().squeeze()  # shape (16000,)

    # convert array to bytes
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, data, 16000, format="WAV")
        with open(tmp.name, "rb") as f:
            data = f.read()

    return {"content": data}


@pytest.mark.skipif(
    condition=not _ZSTD_AVAILABLE or sys.platform == "win32", reason="Requires: ['zstd'] or Windows not supported"
)
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_load_audio_bytes_optimize_and_stream(tmpdir, compression):
    seed_everything(42)

    import soundfile as sf

    optimize(
        fn=create_synthetic_audio_bytes,
        inputs=list(range(100)),
        output_dir=str(tmpdir),
        num_workers=1,
        chunk_bytes="64MB",
        compression=compression,
    )

    dataset = StreamingDataset(input_dir=str(tmpdir))
    sample = dataset[0]
    buffer = BytesIO(sample["content"])
    buffer.seek(0)
    data, sample_rate = sf.read(buffer)
    tensor = torch.from_numpy(data).unsqueeze(0)
    assert tensor.shape == torch.Size([1, 16000])
    assert sample_rate == 16000


def create_synthetic_audio_file(filepath) -> dict:
    import soundfile as sf

    # load dummy audio as bytes
    data = torch.randn((1, 16000)).numpy().squeeze()

    # convert array to bytes
    sf.write(filepath, data, 16000, format="WAV")

    return filepath


@pytest.mark.skipif(
    condition=not _ZSTD_AVAILABLE or sys.platform == "win32" or not _torchcodec_usable(),
    reason="Requires zstd, torchcodec; Windows not supported",
)
@pytest.mark.parametrize("compression", [None])
def test_load_audio_file_optimize_and_stream(tmpdir, compression):
    seed_everything(42)

    optimize(
        fn=create_synthetic_audio_file,
        inputs=[os.path.join(tmpdir, f"{i}.wav") for i in range(5)],
        output_dir=str(tmpdir),
        num_workers=1,
        chunk_bytes="64MB",
        compression=compression,
    )

    dataset = StreamingDataset(input_dir=str(tmpdir))
    sample = dataset[0]
    # Bare .wav paths are Audio: stream a decoder (array + sampling_rate), not a path.
    data = sample["array"]
    sample_rate = sample["sampling_rate"]
    tensor = torch.from_numpy(np.asarray(data)).unsqueeze(0)
    assert tensor.shape[0] == 1
    assert tensor.shape[-1] > 0
    assert int(sample_rate) == 16000


def test_is_path_valid_in_studio(monkeypatch, tmpdir):
    filepath = os.path.join(tmpdir, "a.png")
    with open(filepath, "w") as f:
        f.write("Hello World")

    monkeypatch.setattr(data_processor_module, "_IS_IN_STUDIO", True)

    assert _is_path("/teamspace/studios/this_studio", "/teamspace/studios/this_studio/a.png")
    assert _is_path("/teamspace/studios/this_studio", filepath)


@pytest.mark.skipif(sys.platform == "win32", reason="skip windows")
def test_to_path(tmpdir):
    filepath = os.path.join(tmpdir, "a.png")
    with open(filepath, "w") as f:
        f.write("Hello World")

    assert _to_path("/teamspace/studios/this_studio/a.png") == "/teamspace/studios/this_studio/a.png"
    assert _to_path(filepath) == filepath
    assert _is_remote_path("s3://bucket/a.jpg")
    assert _is_path(None, "s3://bucket/a.jpg")
    assert _is_path("/data", "s3://bucket/a.jpg")
    assert _is_path("/data", "gs://bucket/a.jpg", "gs://bucket")
    assert _to_path("s3://bucket/a.jpg") == "s3://bucket/a.jpg"
    assert _to_path("r2://acc/bucket/a.jpg") == "r2://acc/bucket/a.jpg"
    assert _cache_local_path("s3://bucket/train/a.jpg", Dir(path=None, url="s3://bucket"), "/cache") == os.path.join(
        "/cache", "train/a.jpg"
    )


def test_prepare_items_rewrites_remote_paths_when_input_dir_has_fuse_path():
    input_dir = Dir(path="/teamspace/lightning_storage/testing/in", url="r2://bucket/in")
    items, paths = _prepare_items_and_paths(
        ["r2://bucket/in/a.bin"],
        input_dir,
        "/cache/data",
    )[0]
    assert items == os.path.join("/cache/data", "a.bin")
    assert paths == ["r2://bucket/in/a.bin"]


def test_get_input_dir_lightning_storage_does_not_stat(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("must not stat Studio FUSE mounts")

    monkeypatch.setattr(os.path, "exists", _boom)
    assert _get_input_dir(["/teamspace/lightning_storage/testing/a.bin"]) == "/teamspace/lightning_storage/testing"
    assert _get_input_dir([r"\teamspace\lightning_storage\testing\a.bin"]) == "/teamspace/lightning_storage/testing"


def test_is_path_does_not_stat_studio_fuse(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("must not stat Studio FUSE mounts")

    monkeypatch.setattr(os.path, "isfile", _boom)
    monkeypatch.setattr(os.path, "exists", _boom)
    fuse = "/teamspace/lightning_storage/testing/a.bin"
    assert _is_studio_fuse_path(fuse)
    assert _is_path("/teamspace/lightning_storage/testing", fuse)
    assert _to_path(fuse) == fuse
    assert _get_item_filesizes([fuse]) == [0]


def fetch_from_dataset(batch, output_dir):
    for index in batch.numpy().tolist():
        filepath = os.path.join(output_dir, f"{index}.txt")
        with open(filepath, "w") as f:
            f.write("Hello World!")


#! TODO: fix this test
@pytest.mark.skipif(
    sys.platform == "win32" or sys.platform == "darwin" or sys.platform == "linux", reason="skip windows"
)
def test_streaming_dataset_in_map(tmpdir):
    seed_everything(42)

    output_dir = os.path.join(tmpdir, "output_dir")

    cache = Cache(input_dir=str(tmpdir), chunk_size=10)
    for i in range(107):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))

    map(
        fn=fetch_from_dataset,
        inputs=StreamingDataLoader(dataset, num_workers=1, batch_size=2),
        output_dir=output_dir,
        num_workers=2,
    )

    assert sorted(os.listdir(output_dir)) == sorted([f"{i}.txt" for i in range(107)])


def test_data_chunk_recipe():
    data_recipe = DataChunkRecipe()
    assert data_recipe.chunk_bytes == 67108864
    assert data_recipe.chunk_size is None

    data_recipe = DataChunkRecipe(chunk_bytes=256)
    assert data_recipe.chunk_bytes == 256
    assert data_recipe.chunk_size is None

    data_recipe = DataChunkRecipe(chunk_size=2)
    assert data_recipe.chunk_bytes is None
    assert data_recipe.chunk_size == 2


def test_data_processor_start_method(monkeypatch):
    with pytest.raises(ValueError, match="cannot find context for 'blabla'"):
        DataProcessor(None, start_method="blabla")

    mp_mock = mock.MagicMock()

    monkeypatch.setattr(data_processor_module, "multiprocessing", mp_mock)

    DataProcessor(None)
    mp_mock.set_start_method.assert_called_with("spawn", force=True)

    monkeypatch.setattr(data_processor_module, "in_notebook", mock.MagicMock(return_value=True))

    DataProcessor(None)
    mp_mock.set_start_method.assert_called_with("fork", force=True)


@pytest.mark.parametrize("keep_data_ordered", [True, False])
def test_base_worker_collect_paths_no_downloader(keep_data_ordered):
    shared_queue = mp.Queue() if not keep_data_ordered else None
    msg_queue = mp.Queue()

    worker = BaseWorker(
        worker_index=0,
        num_workers=1,
        node_rank=0,
        msg_queue=msg_queue,
        data_recipe=DataRecipe(),
        input_dir=Dir(),
        output_dir=Dir(),
        items=list(range(10)),
        progress_queue=mp.Queue(),
        error_queue=mp.Queue(),
        stop_queue=mp.Queue(),
        num_downloaders=1,
        num_uploaders=1,
        remove=True,
        reader=None,
        keep_data_ordered=keep_data_ordered,
        shared_queue=shared_queue,
    )

    worker._collect_paths()

    expected_type = FakeQueue if keep_data_ordered else type(mp.Queue())

    assert isinstance(worker.ready_to_process_queue, expected_type)

    if keep_data_ordered:
        for index in range(10):
            assert worker.ready_to_process_queue.get() == (index, index, None)
    else:
        with pytest.raises(Empty):
            worker.ready_to_process_queue.get(timeout=0.05)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_download_data_target_with_data_connection_id(tmpdir, monkeypatch):
    """Test _download_data_target passes data_connection_id to fs_provider correctly."""
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    queue_in = mock.MagicMock()
    queue_out = mock.MagicMock()

    # Mock data with data_connection_id
    test_connection_id = "test-connection-123"
    input_dir_obj = Dir(path=input_dir, url="s3://test-bucket")
    input_dir_obj.data_connection_id = test_connection_id

    items = [10]
    paths = ["s3://test-bucket/a.txt", None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, items.pop(0), [value])

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty

    downloader = mock.MagicMock()
    get_downloader_mock = mock.MagicMock(return_value=downloader)
    monkeypatch.setattr(data_processor_module, "get_downloader", get_downloader_mock)
    monkeypatch.setattr(data_processor_module, "downloader_supports_adownload", lambda _d: False)
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())

    storage_options = {"key": "value"}

    _download_data_target(input_dir_obj, cache_dir, queue_in, queue_out, storage_options)

    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_downloader_mock.assert_called_with(input_dir_obj.url, cache_dir, [], expected_storage_options)
    downloader.download_file.assert_called_once()


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_download_data_target_remote_url_only(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    queue_in = mock.MagicMock()
    queue_out = mock.MagicMock()
    input_dir_obj = Dir(path=None, url="s3://test-bucket")
    paths = ["s3://test-bucket/a.txt", None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, "item", [value])

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty
    downloader = mock.MagicMock()
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=downloader))
    monkeypatch.setattr(data_processor_module, "downloader_supports_adownload", lambda _d: False)
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())

    _download_data_target(input_dir_obj, cache_dir, queue_in, queue_out, {}, emit_done=False)

    downloader.download_file.assert_called_once()
    src, dest = downloader.download_file.call_args[0]
    assert src == "s3://test-bucket/a.txt"
    assert dest == os.path.join(cache_dir, "a.txt")
    # emit_done=False: no None sentinel on the ready queue
    assert all(call.args[0] is not None for call in queue_out.put.call_args_list)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_download_data_target_uses_async_obstore_downloader(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    queue_in = mock.MagicMock()
    queue_out = mock.MagicMock()
    input_dir_obj = Dir(path=None, url="s3://test-bucket")
    paths = ["s3://test-bucket/a.txt", "s3://test-bucket/b.txt", None]
    items = iter([(0, "a", [paths[0]]), (1, "b", [paths[1]]), None])

    def fn(*_, **__):
        return next(items)

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty

    async def _adownload(remote, local):
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        with open(local, "w") as handle:
            handle.write(remote)

    downloader = mock.MagicMock()
    downloader.adownload_file = _adownload
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=downloader))
    monkeypatch.setattr(data_processor_module, "downloader_supports_adownload", lambda _d: True)
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())
    monkeypatch.setenv("LITDATA_OPTIMIZE_DOWNLOAD_BATCH", "1")

    _download_data_target(input_dir_obj, cache_dir, queue_in, queue_out, {}, emit_done=False)

    assert os.path.isfile(os.path.join(cache_dir, "a.txt"))
    assert os.path.isfile(os.path.join(cache_dir, "b.txt"))


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_download_data_target_skips_stat_on_fuse(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    fuse_dir = "/teamspace/lightning_storage/testing"
    fuse_file = f"{fuse_dir}/a.txt"
    input_dir_obj = Dir(path=fuse_dir, url="r2://bucket/testing")

    queue_in = mock.MagicMock()
    queue_out = mock.MagicMock()
    paths = [fuse_file, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, "item", [value])

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty

    def _boom(path):
        if "lightning_storage" in str(path):
            raise AssertionError(f"must not stat FUSE path {path}")
        return False

    monkeypatch.setattr(os.path, "isfile", _boom)
    downloader = mock.MagicMock()
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=downloader))
    monkeypatch.setattr(data_processor_module, "downloader_supports_adownload", lambda _d: False)
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())

    _download_data_target(input_dir_obj, cache_dir, queue_in, queue_out, {})
    downloader.download_file.assert_called_once()
    assert downloader.download_file.call_args[0][0] == "r2://bucket/testing/a.txt"


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_upload_fn_with_data_connection_id(tmpdir, monkeypatch):
    """Test _upload_fn passes data_connection_id to fs_provider correctly."""
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    # Create test file to upload
    test_file = os.path.join(cache_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")

    upload_queue = mock.MagicMock()
    remove_queue = mock.MagicMock()

    # Mock data with data_connection_id
    test_connection_id = "test-connection-456"
    output_dir = Dir(path=None, url="s3://output-bucket")
    output_dir.data_connection_id = test_connection_id

    paths = [test_file, None]

    def fn(*_, **__):
        return paths.pop(0)

    upload_queue.get = fn

    # Mock fs_provider
    fs_provider = mock.MagicMock()
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", get_fs_provider_mock)
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=mock.MagicMock()))
    monkeypatch.setattr(data_processor_module, "downloader_supports_aupload", lambda _d: False)

    storage_options = {"region": "us-west-2"}

    _upload_fn(upload_queue, remove_queue, cache_dir, output_dir, storage_options)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_with(output_dir.url, expected_storage_options)


def test_download_data_target_prefers_local_file_over_r2(tmpdir, monkeypatch):
    """Prefer local source files over R2 downloads.

    When a source file exists locally, _download_data_target should copy it
    directly instead of downloading from R2, even when input_dir.url is set.
    """
    input_dir = os.path.join(tmpdir, "input_dir")
    os.makedirs(input_dir, exist_ok=True)

    # Create a source file that is locally accessible (simulate a mounted drive)
    source_file = os.path.join(input_dir, "sample.txt")
    with open(source_file, "w") as f:
        f.write("hello")

    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    queue_in = mock.MagicMock()
    queue_out = mock.MagicMock()

    # input_dir has both path and url set, as happens with lightning_storage paths
    input_dir_obj = Dir(path=input_dir, url="r2://some-bucket")

    paths = [source_file, None]

    def fn(*_, **__):
        value = paths.pop(0)
        if value is None:
            return value
        return (0, 0, [value])

    queue_in.get = fn
    queue_in.get_nowait.side_effect = Empty

    downloader = mock.MagicMock()
    monkeypatch.setattr(data_processor_module, "get_downloader", mock.MagicMock(return_value=downloader))
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())

    _download_data_target(input_dir_obj, cache_dir, queue_in, queue_out)

    downloader.download_file.assert_not_called()
    downloader.adownload_file.assert_not_called()
    assert os.path.exists(os.path.join(cache_dir, "sample.txt"))
    with open(os.path.join(cache_dir, "sample.txt")) as f:
        assert f.read() == "hello"


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_chunk_recipe_upload_index_with_data_connection_id(tmpdir, monkeypatch):
    """Test DataChunkRecipe._upload_index passes data_connection_id correctly."""
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)

    # Create test index file
    index_file = os.path.join(cache_dir, "index.json")
    with open(index_file, "w") as f:
        f.write('{"test": "data"}')

    # Mock data with data_connection_id
    test_connection_id = "test-connection-789"
    output_dir = Dir(path=None, url="s3://output-bucket")
    output_dir.data_connection_id = test_connection_id

    put_mock = mock.MagicMock(return_value=(None, None))
    monkeypatch.setattr(data_processor_module, "_put_files_remote", put_mock)

    storage_options = {"timeout": 30}
    recipe = DataChunkRecipe(storage_options=storage_options)

    recipe._upload_index(output_dir, cache_dir, num_nodes=1, node_rank=None)

    put_mock.assert_called_once()
    assert put_mock.call_args[0][0] is output_dir
    assert put_mock.call_args[0][2] == storage_options


class _AppendChunkRecipe(DataChunkRecipe):
    def prepare_structure(self, input_dir: str | None) -> list:
        return []

    def prepare_item(self, item_metadata: Any) -> Any:
        return item_metadata


def _index_chunk(name: str) -> dict[str, Any]:
    return {"chunk_size": 1, "chunk_bytes": 4, "column_sizes": [4], "dim": None, "filename": name}


def _write_worker_index(directory: str, chunk_name: str) -> None:
    with open(os.path.join(directory, f"0.{_INDEX_FILENAME}"), "w") as f:
        json.dump({"chunks": [_index_chunk(chunk_name)], "config": None}, f)


def _chunk_names(index_path: str) -> list[str]:
    with open(index_path) as f:
        return [c["filename"] for c in json.load(f)["chunks"]]


def test_data_chunk_recipe_multinode_append_folds_existing_index_once(tmpdir, monkeypatch):
    """Regression for #865: ``existing_index`` must not be folded into every ``{node}-index.json``.

    ``DataChunkRecipe._done`` is the real two-stage merge (per-node, then last-node
    ``_upload_index``). Spying ``Cache._merge_no_wait`` asserts the existing chunks
    are passed only on the final merge.
    """
    output_path = str(tmpdir.mkdir("output"))
    merge_cache_dir = str(tmpdir.mkdir("merge_cache"))
    monkeypatch.setattr(data_processor_module, "_get_cache_dir", lambda name=None: merge_cache_dir)
    monkeypatch.setattr(data_processor_module, "_get_num_nodes", lambda: 2)

    existing_index = {"chunks": [_index_chunk("A.bin"), _index_chunk("B.bin")], "config": None}
    recipe = _AppendChunkRecipe()
    recipe.existing_index = existing_index
    output_dir = Dir(path=output_path, url=None)

    merge_calls: list[tuple[int | None, list[str] | None]] = []
    orig_merge = data_processor_module.Cache._merge_no_wait

    def _spy_merge(self, node_rank=None, existing_index=None):
        names = None if existing_index is None else [c["filename"] for c in existing_index["chunks"]]
        merge_calls.append((node_rank, names))
        return orig_merge(self, node_rank=node_rank, existing_index=existing_index)

    monkeypatch.setattr(data_processor_module.Cache, "_merge_no_wait", _spy_merge)

    monkeypatch.setattr(data_processor_module, "_get_node_rank", lambda: 0)
    _write_worker_index(output_path, "C.bin")
    recipe._done(size=None, delete_cached_files=False, output_dir=output_dir)
    assert _chunk_names(os.path.join(output_path, f"0-{_INDEX_FILENAME}")) == ["C.bin"]

    monkeypatch.setattr(data_processor_module, "_get_node_rank", lambda: 1)
    _write_worker_index(output_path, "D.bin")
    recipe._done(size=None, delete_cached_files=False, output_dir=output_dir)

    assert _chunk_names(os.path.join(output_path, _INDEX_FILENAME)) == ["A.bin", "B.bin", "C.bin", "D.bin"]
    assert merge_calls == [(0, None), (1, None), (None, ["A.bin", "B.bin"])]


def test_data_chunk_recipe_singlenode_append_folds_existing_index_on_node_merge(tmpdir, monkeypatch):
    output_path = str(tmpdir.mkdir("output"))
    monkeypatch.setattr(data_processor_module, "_get_num_nodes", lambda: 1)
    monkeypatch.setattr(data_processor_module, "_get_node_rank", lambda: 0)

    existing_index = {"chunks": [_index_chunk("A.bin"), _index_chunk("B.bin")], "config": None}
    recipe = _AppendChunkRecipe()
    recipe.existing_index = existing_index

    merge_calls: list[tuple[int | None, list[str] | None]] = []
    orig_merge = data_processor_module.Cache._merge_no_wait

    def _spy_merge(self, node_rank=None, existing_index=None):
        names = None if existing_index is None else [c["filename"] for c in existing_index["chunks"]]
        merge_calls.append((node_rank, names))
        return orig_merge(self, node_rank=node_rank, existing_index=existing_index)

    monkeypatch.setattr(data_processor_module.Cache, "_merge_no_wait", _spy_merge)

    _write_worker_index(output_path, "C.bin")
    recipe._done(size=None, delete_cached_files=False, output_dir=Dir(path=output_path, url=None))

    assert _chunk_names(os.path.join(output_path, _INDEX_FILENAME)) == ["A.bin", "B.bin", "C.bin"]
    assert merge_calls == [(None, ["A.bin", "B.bin"])]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processor_cleanup_checkpoints_with_data_connection_id(tmpdir, monkeypatch):
    """Test DataProcessor._cleanup_checkpoints passes data_connection_id correctly."""
    test_connection_id = "test-connection-cleanup"
    output_dir = Dir(path=None, url="s3://cleanup-bucket")
    output_dir.data_connection_id = test_connection_id

    # Mock fs_provider
    fs_provider = mock.MagicMock()
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", get_fs_provider_mock)

    storage_options = {"max_retries": 3}
    data_processor = DataProcessor(input_dir=str(tmpdir), output_dir=output_dir, storage_options=storage_options)

    data_processor._cleanup_checkpoints()

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_with(output_dir.url, expected_storage_options)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processor_save_current_config_with_data_connection_id(tmpdir, monkeypatch):
    """Test DataProcessor._save_current_config passes data_connection_id correctly."""
    test_connection_id = "test-connection-save"
    output_dir = Dir(path=None, url="s3://config-bucket")
    output_dir.data_connection_id = test_connection_id

    # Mock fs_provider
    fs_provider = mock.MagicMock()
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", get_fs_provider_mock)

    storage_options = {"connect_timeout": 10}
    data_processor = DataProcessor(
        input_dir=str(tmpdir), output_dir=output_dir, use_checkpoint=True, storage_options=storage_options
    )

    workers_user_items = [[1, 2], [3, 4]]
    data_processor._save_current_config(workers_user_items)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_with(output_dir.url, expected_storage_options)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processor_load_checkpoint_config_with_data_connection_id(tmpdir, monkeypatch):
    """Test DataProcessor._load_checkpoint_config passes data_connection_id correctly."""
    test_connection_id = "test-connection-load"
    output_dir = Dir(path=None, url="s3://load-bucket")
    output_dir.data_connection_id = test_connection_id

    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.download_directory = mock.MagicMock(return_value=str(tmpdir))
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", get_fs_provider_mock)

    # Create mock config file
    config_data = {"num_workers": 2, "workers_user_items": [[1, 2], [3, 4]]}
    config_file = os.path.join(tmpdir, "config.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    storage_options = {"read_timeout": 15}
    data_processor = DataProcessor(
        input_dir=str(tmpdir),
        output_dir=output_dir,
        use_checkpoint=True,
        num_workers=2,
        storage_options=storage_options,
    )

    workers_user_items = [[1, 2], [3, 4]]
    data_processor._load_checkpoint_config(workers_user_items)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_with(output_dir.url, expected_storage_options)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_connection_id_not_added_when_missing():
    """Test that data_connection_id is not added to storage_options when not present on Dir."""
    # Test with Dir object with default data_connection_id (None)
    output_dir = Dir(path=None, url="s3://test-bucket")

    # Verify that data_connection_id defaults to None
    assert output_dir.data_connection_id is None

    # Test with Dir object with data_connection_id explicitly set to None
    output_dir_none = Dir(path=None, url="s3://test-bucket", data_connection_id=None)

    assert output_dir_none.data_connection_id is None


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_storage_options_preserved_with_data_connection_id():
    """Test that original storage_options are preserved when adding data_connection_id."""
    original_storage_options = {
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
        "region_name": "us-east-1",
    }

    test_connection_id = "test-connection-preserve"

    # Simulate the merge operation that happens in the code
    merged_storage_options = original_storage_options.copy()
    merged_storage_options["data_connection_id"] = test_connection_id

    # Verify original is unchanged
    assert "data_connection_id" not in original_storage_options
    assert len(original_storage_options) == 3

    # Verify merged has all original keys plus data_connection_id
    assert len(merged_storage_options) == 4
    assert merged_storage_options["data_connection_id"] == test_connection_id
    assert merged_storage_options["aws_access_key_id"] == "test-key"
    assert merged_storage_options["aws_secret_access_key"] == "test-secret"
    assert merged_storage_options["region_name"] == "us-east-1"


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_connection_id_overrides_existing_value():
    """Test that data_connection_id from Dir overrides any existing value in storage_options."""
    original_storage_options = {"data_connection_id": "original-connection-id", "timeout": 30}

    dir_connection_id = "dir-connection-id"

    # Simulate the merge operation that happens in the code
    merged_storage_options = original_storage_options.copy()
    merged_storage_options["data_connection_id"] = dir_connection_id

    # Verify the Dir's data_connection_id overrides the original
    assert merged_storage_options["data_connection_id"] == dir_connection_id
    assert merged_storage_options["timeout"] == 30
    assert len(merged_storage_options) == 2


class CustomDataChunkRecipeWithConnectionId(DataChunkRecipe):
    """Custom recipe for testing data_connection_id integration."""

    is_generator = False

    def prepare_structure(self, input_dir: str) -> list[Any]:
        return ["test_item"]

    def prepare_item(self, item):
        return {"data": item}


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Not supported on windows")
def test_data_processor_end_to_end_with_data_connection_id(tmpdir, monkeypatch):
    """Test full data processing pipeline with data_connection_id."""
    test_connection_id = "test-connection-e2e"

    # Setup input and output dirs with data_connection_id
    input_dir = Dir(path=str(tmpdir), url="s3://input-bucket")
    input_dir.data_connection_id = test_connection_id

    output_dir = Dir(path=None, url="s3://output-bucket")
    output_dir.data_connection_id = test_connection_id

    # Mock fs_provider calls
    fs_provider = mock.MagicMock()
    fs_provider.exists = mock.MagicMock(return_value=True)
    fs_provider.download_file = mock.MagicMock()
    fs_provider.upload_file = mock.MagicMock()

    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(data_processor_module, "_get_fs_provider", get_fs_provider_mock)

    # Mock other dependencies
    monkeypatch.setattr(data_processor_module, "_wait_for_disk_usage_higher_than_threshold", mock.MagicMock())

    cache_dir = os.path.join(tmpdir, "cache")
    data_cache_dir = os.path.join(tmpdir, "data_cache")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", data_cache_dir)

    storage_options = {"custom_option": "test_value"}

    data_processor = DataProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=1,
        fast_dev_run=1,
        storage_options=storage_options,
        verbose=False,
    )

    # Run with custom recipe
    recipe = CustomDataChunkRecipeWithConnectionId(storage_options=storage_options)

    with suppress(Exception):
        # Expected to fail due to mocking, but we want to verify the calls were made
        data_processor.run(recipe)

    # Verify that fs_provider was called with data_connection_id
    # Check if any of the calls included the expected storage options
    calls_made = get_fs_provider_mock.call_args_list

    # Should have been called with merged storage options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id

    # Note: Due to the complexity of mocking the full pipeline, we mainly verify
    # that the fs_provider was called, indicating the data_connection_id code paths were executed
    assert len(calls_made) > 0, "fs_provider should have been called"


@pytest.mark.parametrize(
    ("output_dir", "broadcast_paths", "expect_broadcast"),
    [
        # Default off for ordinary paths
        ("/data/out", False, False),
        # Explicit True always broadcasts
        ("local/out", True, True),
        # `{%strftime}` time template auto-enables broadcast
        ("local/out_{%Y-%m-%d}", False, True),
        ("s3://bucket/run_{%Y-%m-%d_%H-%M-%S}", False, True),
    ],
)
def test_data_processor_broadcast_paths(tmpdir, monkeypatch, output_dir, broadcast_paths, expect_broadcast):
    """broadcast_paths defaults off; auto-on for `{%strftime}` paths; explicit True forces broadcast."""
    broadcast_mock = mock.MagicMock(side_effect=lambda key, obj, rank: obj)
    monkeypatch.setattr(data_processor_module, "broadcast_object", broadcast_mock)

    DataProcessor(
        input_dir=str(tmpdir),
        output_dir=output_dir,
        num_workers=1,
        verbose=False,
        broadcast_paths=broadcast_paths,
    )

    if expect_broadcast:
        assert broadcast_mock.call_count == 2
        assert broadcast_mock.call_args_list[0].args[0] == "input_dir"
        assert broadcast_mock.call_args_list[1].args[0] == "output_dir"
    else:
        broadcast_mock.assert_not_called()


def test_data_processor_broadcast_paths_default_false(tmpdir, monkeypatch):
    broadcast_mock = mock.MagicMock(side_effect=lambda key, obj, rank: obj)
    monkeypatch.setattr(data_processor_module, "broadcast_object", broadcast_mock)

    processor = DataProcessor(input_dir=str(tmpdir), output_dir=str(tmpdir / "out"), num_workers=1, verbose=False)

    assert processor.broadcast_paths is False
    broadcast_mock.assert_not_called()


def test_optimize_broadcast_paths_auto_on_for_time_template(tmpdir, monkeypatch):
    """Optimize detects `{%strftime}` on the unresolved path before `_resolve_dir`."""
    captured: dict[str, Any] = {}

    class CaptureDataProcessor(DataProcessor):
        def __init__(self, *args, **kwargs):
            captured["broadcast_paths"] = kwargs.get("broadcast_paths")
            super().__init__(*args, **kwargs)

        def run(self, data_recipe):
            return None

    monkeypatch.setattr(functions, "DataProcessor", CaptureDataProcessor)
    monkeypatch.setattr(functions, "_assert_dir_has_index_file", mock.MagicMock())

    optimize(
        fn=lambda x: x,
        inputs=[1, 2, 3],
        output_dir=str(tmpdir / "out_{%Y-%m-%d}"),
        chunk_size=2,
        num_workers=1,
        verbose=False,
    )

    assert captured["broadcast_paths"] is True


def test_optimize_broadcast_paths_default_off(tmpdir, monkeypatch):
    captured: dict[str, Any] = {}

    class CaptureDataProcessor(DataProcessor):
        def __init__(self, *args, **kwargs):
            captured["broadcast_paths"] = kwargs.get("broadcast_paths")
            super().__init__(*args, **kwargs)

        def run(self, data_recipe):
            return None

    monkeypatch.setattr(functions, "DataProcessor", CaptureDataProcessor)
    monkeypatch.setattr(functions, "_assert_dir_has_index_file", mock.MagicMock())

    optimize(
        fn=lambda x: x,
        inputs=[1, 2, 3],
        output_dir=str(tmpdir / "out"),
        chunk_size=2,
        num_workers=1,
        verbose=False,
    )

    assert captured["broadcast_paths"] is False


def test_optimize_broadcast_paths_explicit_true(tmpdir, monkeypatch):
    captured: dict[str, Any] = {}

    class CaptureDataProcessor(DataProcessor):
        def __init__(self, *args, **kwargs):
            captured["broadcast_paths"] = kwargs.get("broadcast_paths")
            super().__init__(*args, **kwargs)

        def run(self, data_recipe):
            return None

    monkeypatch.setattr(functions, "DataProcessor", CaptureDataProcessor)
    monkeypatch.setattr(functions, "_assert_dir_has_index_file", mock.MagicMock())

    optimize(
        fn=lambda x: x,
        inputs=[1, 2, 3],
        output_dir=str(tmpdir / "out"),
        chunk_size=2,
        num_workers=1,
        verbose=False,
        broadcast_paths=True,
    )

    assert captured["broadcast_paths"] is True


_LIGHTNING_STORAGE_TESTING = "/teamspace/lightning_storage/testing"


@pytest.mark.skipif(
    not os.path.isdir(_LIGHTNING_STORAGE_TESTING),
    reason="Requires Studio /teamspace/lightning_storage/testing",
)
def test_studio_lightning_storage_shared_node_queue(tmpdir):
    """Write files to lightning_storage and emulate two nodes via DATA_OPTIMIZER_* env."""
    import shutil
    import subprocess
    import uuid

    from litdata.streaming.resolver import _resolve_dir

    run_id = uuid.uuid4().hex[:8]
    root = os.path.join(_LIGHTNING_STORAGE_TESTING, "litdata_node_queue", run_id)
    input_dir = os.path.join(root, "input")
    output_dir = os.path.join(root, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    resolved = _resolve_dir(input_dir)
    assert resolved.path == input_dir
    assert resolved.url
    assert resolved.data_connection_id

    sizes = [64, 256, 1024, 4096, 16384, 65536, 128, 512]
    for i, size in enumerate(sizes):
        with open(os.path.join(input_dir, f"{i:02d}.bin"), "wb") as handle:
            handle.write(os.urandom(size))

    worker = os.path.join(os.path.dirname(__file__), "node_queue_multinode_worker.py")
    procs = []
    for rank in (0, 1):
        env = os.environ.copy()
        env["DATA_OPTIMIZER_NUM_NODES"] = "2"
        env["DATA_OPTIMIZER_NODE_RANK"] = str(rank)
        env["DATA_OPTIMIZER_CACHE_FOLDER"] = os.path.join(tmpdir, f"chunks-{rank}")
        env["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = os.path.join(tmpdir, f"data-{rank}")
        procs.append(
            subprocess.Popen(  # noqa: S603
                [sys.executable, worker, input_dir, output_dir],
                env=env,
                cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            )
        )

    codes = [proc.wait() for proc in procs]
    try:
        assert codes == [0, 0], f"node processes failed: {codes}"
        dataset = StreamingDataset(input_dir=output_dir)
        assert len(dataset) == len(sizes)
        names = {dataset[i][1] for i in range(len(dataset))}
        assert names == {f"{i:02d}.bin" for i in range(len(sizes))}
    finally:
        shutil.rmtree(root, ignore_errors=True)

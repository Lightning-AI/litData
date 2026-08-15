import os
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from litdata.constants import (
    _CRYPTOGRAPHY_AVAILABLE,
    _NUMPY_DTYPES_MAPPING,
    _POLARS_AVAILABLE,
    _PYARROW_AVAILABLE,
    _TORCH_DTYPES_MAPPING,
)
from litdata.streaming import Cache, item_loader
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.shuffle import _get_shared_chunks


def test_encode_data_size_header_is_little_endian_uint32():
    packed, dim = PyTreeLoader.encode_data([b"ab", b"cdef"], [2, 4], ["ab", "cdef"])
    assert dim is None
    assert packed[:8] == (2).to_bytes(4, "little") + (4).to_bytes(4, "little")
    assert packed[8:] == b"abcdef"


def _write_int_dataset(tmpdir, num_items: int = 40, chunk_size: int = 7) -> str:
    """Write a small integer StreamingDataset and return its directory."""
    cache = Cache(str(tmpdir), chunk_size=chunk_size)
    for i in range(num_items):
        cache[i] = i
    cache.done()
    cache.merge()
    return str(tmpdir)


def _read_all_with_mmap(dataset: StreamingDataset, allowed_chunks: set[int] | None) -> list:
    """Read every item, optionally forcing the mmap allow-set (empty = file path)."""
    # Force the Cache/reader to exist before mutating the loader.
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    loader.close(0)
    if allowed_chunks is None:
        loader.set_mmap_allowed_chunks(set())
    else:
        loader.set_mmap_allowed_chunks(allowed_chunks)
    return [dataset[i] for i in range(len(dataset))]


def test_serializer_setup():
    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    item_loader = PyTreeLoader()
    item_loader.setup(config_mock, [], {"fake": serializer_mock})
    assert len(item_loader._serializers) == 2
    assert item_loader._serializers["fake:12"]


def test_pytreeloader_with_no_header_tensor_serializer(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=10)
    assert isinstance(cache._reader._item_loader, PyTreeLoader)
    dtype_index_float = 1
    dtype_index_long = 18
    for i in range(10):
        cache[i] = {
            "float": i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_float]),
            "long": i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_long]),
        }

    data_format = [f"no_header_tensor:{dtype_index_float}", f"no_header_tensor:{dtype_index_long}"]
    assert cache._writer.get_config()["data_format"] == data_format
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))
    for i in range(len(dataset)):
        item = dataset[i]
        assert torch.allclose(i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_float]), item["float"])
        assert torch.allclose(i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_long]), item["long"])


def test_tokensloader_with_no_header_numpy_serializer(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=512, item_loader=TokensLoader())
    assert isinstance(cache._reader._item_loader, TokensLoader)

    dtype_index_int32 = 3
    dtype = _NUMPY_DTYPES_MAPPING[dtype_index_int32]

    for i in range(10):
        data = np.random.randint(0, 100, size=(256), dtype=dtype)
        cache._add_item(i, data)

    data_format = [f"no_header_numpy:{dtype_index_int32}"]
    assert cache._writer.get_config()["data_format"] == data_format
    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        input_dir=str(tmpdir),
        drop_last=True,
        item_loader=TokensLoader(block_size=256),
    )

    for data in dataset:
        assert data.shape == (256,)
        assert data.dtype == dtype


class TestPyTreeLoader(PyTreeLoader):
    def force_download(self, chunk_index):
        assert chunk_index == 0
        super().force_download(chunk_index)
        raise Exception("worked")


def test_force_download(monkeypatch, tmpdir):
    monkeypatch.setattr(item_loader, "_FORCE_DOWNLOAD_TIME", 1)
    monkeypatch.setattr(item_loader, "_FORCE_DOWNLOAD_TIME", 1)
    loader = TestPyTreeLoader()

    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    loader.setup(config_mock, [], {"fake": serializer_mock})

    with pytest.raises(Exception, match="worked"):
        loader.load_item_from_chunk(0, 0, "chunk_filepath", 0, 1)


def test_compiled_unflatten_matches_pytree(tmpdir):
    """The compiled treespec unflatten must match stock ``tree_unflatten``."""
    cache = Cache(str(tmpdir), chunk_size=5)
    for i in range(10):
        cache[i] = {"i": i, "coords": [float(i), float(i + 1)], "flag": i % 2 == 0}
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir))
    items = [dataset[i] for i in range(len(dataset))]
    assert items[0] == {"i": 0, "coords": [0.0, 1.0], "flag": True}
    assert items[9] == {"i": 9, "coords": [9.0, 10.0], "flag": False}

    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert loader._unflatten is not None


def test_compiled_unflatten_is_picklable_for_dataloader_workers(tmpdir):
    """Compiled unflatten must survive spawn pickling (used by DataLoader workers)."""
    cache = Cache(str(tmpdir), chunk_size=5)
    for i in range(5):
        cache[i] = {"i": i, "x": float(i)}
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir))
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert loader._unflatten is not None

    restored = pickle.loads(pickle.dumps(loader))  # noqa: S301
    assert restored._unflatten is not None
    leaves = [1, 2.0]
    assert restored._unflatten(leaves) == loader._unflatten(leaves) == {"i": 1, "x": 2.0}


def test_pre_load_chunk_does_not_mutate_mmap_from_prefetch_thread(tmpdir):
    """PrepareChunksThread may only WILLNEED; mmap state stays on the reader thread."""
    data_dir = _write_int_dataset(tmpdir, num_items=14, chunk_size=7)
    dataset = StreamingDataset(data_dir)
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    loader.close(0)
    chunk_path = dataset.cache._reader.config[ChunkedIndex(0, chunk_index=0)][0]
    loader.pre_load_chunk(0, chunk_path)
    assert loader._mapped == {}
    assert loader._mmap is None


def test_pytree_loader_mmap_matches_file_reads(tmpdir):
    """Mmap and unbuffered file reads must deserialize to identical items."""
    data_dir = _write_int_dataset(tmpdir, num_items=40, chunk_size=7)

    file_items = _read_all_with_mmap(StreamingDataset(data_dir), allowed_chunks=None)

    mmap_dataset = StreamingDataset(data_dir)
    _ = mmap_dataset[0]
    num_chunks = len(mmap_dataset.cache._reader.config._chunks)
    mmap_items = _read_all_with_mmap(mmap_dataset, allowed_chunks=set(range(num_chunks)))

    assert mmap_items == file_items == list(range(40))

    loader = mmap_dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    assert loader._mmap is not None
    assert loader._offsets is not None


def test_pytree_loader_mmap_close_releases_mapping(tmpdir):
    """Closing a mapped chunk must drop mmap state so the file can be deleted."""
    from litdata.streaming.sampler import ChunkedIndex

    data_dir = _write_int_dataset(tmpdir, num_items=14, chunk_size=7)
    dataset = StreamingDataset(data_dir)
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)

    chunk_filepath, _, _ = dataset.cache._reader.config[ChunkedIndex(0, chunk_index=0)]
    # Force a mapped open of chunk 0.
    loader.set_mmap_allowed_chunks({0})
    loader.close(0)
    _ = dataset[0]
    assert loader._mmap is not None

    loader.close(0)
    assert loader._mmap is None
    assert loader._offsets is None
    assert loader._open_handle is None
    assert loader._chunk_filepath is None

    # File should no longer be held open.
    os.remove(chunk_filepath)
    assert not os.path.exists(chunk_filepath)


def test_pytree_loader_mmap_pickle_roundtrip(tmpdir):
    """Mmap state is process-local and must not survive pickling."""
    data_dir = _write_int_dataset(tmpdir, num_items=14, chunk_size=7)
    dataset = StreamingDataset(data_dir)
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    loader.set_mmap_allowed_chunks({0})
    loader.close(0)
    _ = dataset[0]
    assert loader._mmap is not None

    restored = pickle.loads(pickle.dumps(loader))  # noqa: S301
    assert restored._mmap is None
    assert restored._offsets is None
    assert restored._open_handle is None
    assert restored._chunk_filepath is None
    assert restored._mmap_allowed_chunks == {0}

    # Lazily remaps and keeps reading correctly after unpickle.
    restored_dataset = StreamingDataset(data_dir)
    _ = restored_dataset[0]
    restored_loader = restored_dataset.cache._reader._item_loader
    restored_loader.set_mmap_allowed_chunks({0})
    restored_loader.close(0)
    assert restored_dataset[0] == 0
    assert restored_dataset[6] == 6


def test_tokens_loader_posix_warmup_is_picklable(tmpdir):
    """POSIX-fast must not pin token memmaps in the parent (that leaked fds); pickle still works."""
    cache = Cache(str(tmpdir), chunk_size=40, item_loader=TokensLoader(10))
    counter = 0
    for i in range(4):
        cache[i] = torch.arange(counter, counter + 20).to(torch.int)
        counter += 20
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), item_loader=TokensLoader(10), shuffle=False)
    assert len(dataset) == 8
    warmed = dataset.shuffler.cache._reader._item_loader
    assert isinstance(warmed, TokensLoader)
    assert warmed._posix_fast is True
    assert warmed._buffers == {}

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301
    restored_loader = restored.shuffler.cache._reader._item_loader
    assert restored_loader._buffers == {}
    assert restored_loader._mmaps == {}
    assert torch.equal(restored[0], torch.arange(0, 10).to(torch.int))


def test_shared_chunks_excluded_from_mmap_allow_set():
    """Only exclusive chunks are candidates for mmap; shared ones must be omitted."""
    workers_chunks = [[0, 1, 2], [2, 3, 4]]
    shared = _get_shared_chunks(workers_chunks)
    assert set(shared) == {2}

    my_chunks = workers_chunks[0]
    my_nonshared = {chunk_index for chunk_index in my_chunks if chunk_index not in shared}
    assert my_nonshared == {0, 1}

    loader = PyTreeLoader()
    loader.set_mmap_allowed_chunks(my_nonshared)
    assert 2 not in loader._mmap_allowed_chunks


@pytest.mark.skipif(not _CRYPTOGRAPHY_AVAILABLE, reason="Requires: ['cryptography']")
def test_encrypted_chunks_never_mmap(tmpdir):
    """Encrypted datasets must keep the file/decrypt path even when mmap is allowed."""
    from litdata.utilities.encryption import FernetEncryption

    fernet = FernetEncryption(password="password", level="chunk")
    cache = Cache(str(tmpdir), chunk_size=5, encryption=fernet)
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), encryption=fernet)
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)
    loader.set_mmap_allowed_chunks({0, 1})
    loader.close(0)

    assert dataset[0] == 0
    assert dataset[4] == 4
    assert loader._mmap is None


def test_pytree_loader_rejects_mismatched_chunk_header(tmpdir):
    """Mmap open must fail fast when the on-disk header disagrees with index.json."""
    from litdata.streaming.sampler import ChunkedIndex

    data_dir = _write_int_dataset(tmpdir, num_items=7, chunk_size=7)
    dataset = StreamingDataset(data_dir)
    _ = dataset[0]
    loader = dataset.cache._reader._item_loader
    assert isinstance(loader, PyTreeLoader)

    chunk_index = 0
    chunk_filepath, begin, filesize_bytes = dataset.cache._reader.config[ChunkedIndex(0, chunk_index=chunk_index)]
    # Corrupt only the in-memory index metadata used by the mmap open path.
    loader._chunks[chunk_index] = {**loader._chunks[chunk_index], "chunk_size": 3}
    loader.set_mmap_allowed_chunks({chunk_index})
    loader.close(chunk_index)

    with pytest.raises(RuntimeError, match="does not match index.json chunk_size"):
        loader.load_item_from_chunk(0, chunk_index, chunk_filepath, begin, filesize_bytes)


def _write_parquet_with_row_groups(path, row_group_values):
    """Write a parquet file where each element of row_group_values becomes its own row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("col", pa.int64())])
    with pq.ParquetWriter(path, schema) as writer:
        for values in row_group_values:
            writer.write_table(pa.table({"col": list(values)}, schema=schema))


@pytest.mark.parametrize(
    "row_group_sizes",
    [
        [10, 5, 5],  # regression: uneven groups, shrinking
        [3, 7, 2, 8],  # uneven groups, varying
        [20],  # single group
        [1, 1, 1, 1, 1],  # many size-1 groups
        [5, 5, 5],  # uniform control case
    ],
)
@pytest.mark.parametrize("low_memory", [True, False])
def test_parquet_loader_row_group_sizes(tmp_path, row_group_sizes, low_memory):
    """ParquetLoader must correctly read every row regardless of row-group layout."""
    parquet_dir = tmp_path / "pq"
    parquet_dir.mkdir()

    row_group_values = []
    expected = []

    for value, size in enumerate(row_group_sizes):
        row_group_values.append([value] * size)
        expected.extend([value] * size)
        value += 1
    _write_parquet_with_row_groups(parquet_dir / "data.parquet", row_group_values)

    index_parquet_dataset(str(parquet_dir))
    dataset = StreamingDataset(str(parquet_dir), item_loader=ParquetLoader(low_memory=low_memory))

    assert len(dataset) == sum(row_group_sizes)
    actual = [dataset[i]["col"] for i in range(len(dataset))]
    assert actual == expected


def test_parquet_loader_row_group_boundaries(tmp_path):
    """First and last row of each group (the modulo edges in the old implementation)."""
    parquet_dir = tmp_path / "pq"
    parquet_dir.mkdir()

    row_group_sizes = [10, 5, 5]
    _write_parquet_with_row_groups(
        parquet_dir / "data.parquet",
        [[v] * s for v, s in enumerate(row_group_sizes)],
    )

    index_parquet_dataset(str(parquet_dir))
    dataset = StreamingDataset(str(parquet_dir), item_loader=ParquetLoader(low_memory=True))

    boundaries = [0, 9, 10, 14, 15, 19]
    expected = [0, 0, 1, 1, 2, 2]
    for idx, exp in zip(boundaries, expected):
        assert dataset[idx]["col"] == exp


@pytest.mark.skipif(not _PYARROW_AVAILABLE or not _POLARS_AVAILABLE, reason="pyarrow and polars are required")
def test_parquet_loader_column_projection(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_dir = tmp_path / "pq"
    parquet_dir.mkdir()
    pq.write_table(pa.table({"keep": [1, 2, 3], "drop": [9, 8, 7]}), parquet_dir / "data.parquet")
    index_parquet_dataset(str(parquet_dir))

    dataset = StreamingDataset(str(parquet_dir), item_loader=ParquetLoader(low_memory=True, columns=["keep"]))
    row = dataset[0]
    assert row == {"keep": 1}


def test_parquet_loader_cache_eviction_with_uneven_groups(tmp_path):
    """After fully reading a row group, it must be evicted from the in-memory cache."""
    parquet_dir = tmp_path / "pq"
    parquet_dir.mkdir()

    row_group_sizes = [10, 5, 5]
    _write_parquet_with_row_groups(
        parquet_dir / "data.parquet",
        [[v] * s for v, s in enumerate(row_group_sizes)],
    )

    index_parquet_dataset(str(parquet_dir))
    loader = ParquetLoader(low_memory=True)
    dataset = StreamingDataset(str(parquet_dir), item_loader=loader)

    # Iterate through the whole dataset sequentially.
    for i in range(len(dataset)):
        dataset[i]

    # After a sequential pass every row group in the chunk should have been evicted.
    for chunk_index, groups in loader._chunk_row_groups.items():
        assert groups == {}, f"chunk {chunk_index} still has cached row groups: {groups}"


def test_wait_until_chunk_ready_raises_prefetch_crash_immediately(tmpdir):
    """A dead PrepareChunksThread must not surface as a 120s FileNotFoundError timeout."""
    from litdata.streaming.item_loader import BaseItemLoader

    class _Loader(BaseItemLoader):
        def generate_intervals(self):
            return []

        def pre_load_chunk(self, chunk_index, chunk_filepath):
            return None

        def load_item_from_chunk(self, index, chunk_index, chunk_filepath, begin, filesize_bytes):
            return None

        def delete(self, chunk_index, chunk_filepath):
            return None

        def encode_data(self, data, sizes, flattened):
            return b"", None

    loader = _Loader()
    path = os.path.join(tmpdir, "missing-chunk.bin")
    crash = TypeError("Session.__init__() got an unexpected keyword argument 'data_connection_id'")
    loader.set_prefetch_error_provider(lambda: crash)
    with pytest.raises(RuntimeError, match="prefetch thread crashed") as exc_info:
        loader._wait_until_chunk_ready(0, path, filesize_bytes=16)
    assert exc_info.value.__cause__ is crash


def test_wait_until_chunk_ready_times_out_as_chunk_wait_timeout(tmpdir, monkeypatch):
    from litdata.exceptions import ChunkWaitTimeoutError
    from litdata.streaming.item_loader import BaseItemLoader

    monkeypatch.setattr("litdata.streaming.item_loader._MAX_WAIT_TIME", 0.2)
    monkeypatch.setattr("litdata.streaming.item_loader._FORCE_DOWNLOAD_TIME", 10.0)

    class _Loader(BaseItemLoader):
        def generate_intervals(self):
            return []

        def pre_load_chunk(self, chunk_index, chunk_filepath):
            return None

        def load_item_from_chunk(self, index, chunk_index, chunk_filepath, begin, filesize_bytes):
            return None

        def delete(self, chunk_index, chunk_filepath):
            return None

        def encode_data(self, data, sizes, flattened):
            return b"", None

    loader = _Loader()
    path = os.path.join(tmpdir, "never-arrives.bin")
    with pytest.raises(ChunkWaitTimeoutError, match="Timed out") as exc_info:
        loader._wait_until_chunk_ready(0, path, filesize_bytes=16)
    assert isinstance(exc_info.value, FileNotFoundError)
    assert exc_info.value.path == path

import os
import pickle
import struct
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
from litdata.streaming.item_loader import (
    ParquetLoader,
    PyTreeLoader,
    TokensLoader,
    _auto_batch_rows,
    _batch_rows_for_format,
    _parse_batch_decode,
)
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.shuffle import _get_shared_chunks


def test_batch_rows_for_format(monkeypatch):
    monkeypatch.delenv("LITDATA_BATCH_DECODE", raising=False)
    monkeypatch.delenv("LITDATA_BATCH_ROWS", raising=False)
    assert _parse_batch_decode("auto") is None
    assert _parse_batch_decode("all") == -1
    assert _parse_batch_decode(32) == 32
    assert _batch_rows_for_format(["str", "int"]) == 256
    assert _batch_rows_for_format(["str", "str", "json", "json"]) == 256
    assert _batch_rows_for_format(["pickle"]) == 256
    assert _auto_batch_rows(["jpeg", "int"], [{"chunk_bytes": 8 << 20, "chunk_size": 4}]) == 1
    assert _auto_batch_rows(["jpeg"], [{"chunk_bytes": 512 << 10, "chunk_size": 8}]) == 16
    assert _batch_rows_for_format(["jpeg"], [{"chunk_bytes": 8 << 20, "chunk_size": 4}]) == 1
    assert _batch_rows_for_format(["str"], batch_decode=32) == 32
    assert _batch_rows_for_format(["jpeg"], [{"chunk_bytes": 8 << 20, "chunk_size": 4}], batch_decode=8) == 8
    monkeypatch.setenv("LITDATA_BATCH_DECODE", "0")
    assert _batch_rows_for_format(["json"]) == 0
    assert _batch_rows_for_format(["json"], batch_decode=64) == 64
    monkeypatch.setenv("LITDATA_BATCH_DECODE", "all")
    assert _batch_rows_for_format(["str"]) == -1
    monkeypatch.setenv("LITDATA_BATCH_DECODE", "1")
    assert _batch_rows_for_format(["json"]) == 1
    monkeypatch.delenv("LITDATA_BATCH_DECODE")
    monkeypatch.setenv("LITDATA_BATCH_ROWS", "256")
    assert _batch_rows_for_format(["json"]) == 256


def test_decode_window_is_aligned():
    loader = PyTreeLoader()
    assert loader._window_bounds(0, 1000, 256) == (0, 256)
    assert loader._window_bounds(255, 1000, 256) == (0, 256)
    assert loader._window_bounds(256, 1000, 256) == (256, 512)
    assert loader._window_bounds(900, 1000, 256) == (768, 1000)
    assert loader._window_bounds(3, 10, -1) == (0, 10)
    assert loader._window_bounds(3, 10, 1) == (3, 4)


def test_streaming_dataset_exposes_batch_decode(tmp_path):
    from litdata import optimize

    optimize(
        fn=_flat_arrow_sample,
        inputs=list(range(8)),
        output_dir=str(tmp_path / "ds"),
        chunk_size=8,
        num_workers=1,
    )
    ds = StreamingDataset(str(tmp_path / "ds"))
    assert ds.batch_decode == "auto"
    pinned = StreamingDataset(str(tmp_path / "ds"), batch_decode=4)
    assert pinned.batch_decode == 4
    loader = PyTreeLoader(batch_decode=4)
    assert _batch_rows_for_format(["str", "int"], [{"chunk_bytes": 100, "chunk_size": 8}], loader._batch_decode) == 4


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


def _nested_arrow_sample(i: int):
    from litdata.streaming.serializers import JsonLeaf

    return {
        "id": f"q{i}",
        "choices": JsonLeaf({"text": ["A", "B"], "label": ["1", "2"]}),
        "answers": JsonLeaf(["span"] * (i % 3)),
    }


def _flat_arrow_sample(i: int):
    return {"text": f"row {i}", "label": i % 2}


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
def test_nested_chunk_uses_arrow_footer(tmp_path):
    from litdata import optimize
    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC, load_arrow_row_footer

    out = tmp_path / "nested"
    optimize(fn=_nested_arrow_sample, inputs=list(range(32)), output_dir=str(out), chunk_size=32, num_workers=1)
    chunk = next(out.glob("*.bin"))
    raw = chunk.read_bytes()
    assert raw[-8:] == _ARROW_FOOTER_MAGIC
    rows = load_arrow_row_footer(raw)
    assert rows is not None
    assert len(rows) == 32
    assert rows[0]["choices"] == {"text": ["A", "B"], "label": ["1", "2"]}
    assert rows[0]["answers"] == []
    assert rows[2]["answers"] == ["span", "span"]

    n = struct.unpack_from("<I", raw, 0)[0]
    offsets = struct.unpack_from("<" + "I" * (n + 1), raw, 4)
    assert offsets[0] == offsets[-1], "nested chunks must not duplicate pytree item bytes"

    ds = StreamingDataset(str(out))
    assert ds[10]["id"] == "q10"
    assert ds[0]["choices"] == {"text": ["A", "B"], "label": ["1", "2"]}
    assert ds[2]["answers"] == ["span", "span"]
    assert ds[31]["id"] == "q31"


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
def test_nested_chunk_skips_file_zstd(tmp_path):
    """Nested chunks use Arrow IPC zstd, not LitData whole-file Python inflate."""
    from litdata import optimize
    from litdata.constants import _ZSTD_AVAILABLE
    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC, load_arrow_row_footer

    if not _ZSTD_AVAILABLE:
        pytest.skip("Requires zstd")

    out = tmp_path / "nested-zstd"
    optimize(
        fn=_nested_arrow_sample,
        inputs=list(range(32)),
        output_dir=str(out),
        chunk_size=32,
        num_workers=1,
        compression="zstd",
    )
    bins = list(out.glob("*.bin"))
    assert bins, "expected a chunk"
    assert not any(".zstd.bin" in p.name for p in bins)
    import json

    index = json.loads((out / "index.json").read_text())
    assert index["config"]["compression"] is None
    assert index["config"]["ipc_compression"] == "zstd"

    raw = bins[0].read_bytes()
    assert raw[-8:] == _ARROW_FOOTER_MAGIC
    rows = load_arrow_row_footer(raw)
    assert rows is not None
    assert len(rows) == 32

    ds = StreamingDataset(str(out))
    assert ds[10]["id"] == "q10"


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_nested_ipc_file_has_multiple_record_batches(tmp_path, compression):
    """Chunks with more than 256 nested rows write an IPC file with several batches."""
    import pyarrow as pa

    from litdata import optimize
    from litdata.constants import _ZSTD_AVAILABLE
    from litdata.streaming.item_loader import (
        _ARROW_FOOTER_MAGIC,
        _ARROW_IPC_FILE_MAGIC,
        _DEFAULT_BATCH_ROWS,
        _arrow_footer_span,
        load_arrow_row_footer,
    )

    if compression == "zstd" and not _ZSTD_AVAILABLE:
        pytest.skip("Requires zstd")

    n = _DEFAULT_BATCH_ROWS + 40
    out = tmp_path / f"nested-multi-{compression or 'none'}"
    optimize(
        fn=_nested_arrow_sample,
        inputs=list(range(n)),
        output_dir=str(out),
        chunk_size=n,
        num_workers=1,
        compression=compression,
    )
    bins = list(out.glob("*.bin"))
    assert bins, "expected a chunk"
    assert not any(".zstd.bin" in p.name for p in bins)
    chunk = bins[0]
    raw = chunk.read_bytes()
    assert raw[-8:] == _ARROW_FOOTER_MAGIC
    span = _arrow_footer_span(raw)
    assert span is not None
    start, ipc_len = span
    ipc = raw[start : start + ipc_len]
    assert ipc[:6] == _ARROW_IPC_FILE_MAGIC
    reader = pa.ipc.open_file(ipc)
    assert reader.num_record_batches > 1
    rows = load_arrow_row_footer(raw)
    assert rows is not None
    assert len(rows) == n

    ds = StreamingDataset(str(out))
    assert ds[0]["id"] == "q0"
    assert ds[_DEFAULT_BATCH_ROWS]["id"] == f"q{_DEFAULT_BATCH_ROWS}"
    assert ds[n - 1]["id"] == f"q{n - 1}"

    loader = PyTreeLoader()
    loader._batch_rows = _DEFAULT_BATCH_ROWS
    loader._config = {}
    assert loader._try_arrow_footer_rows(raw, 0, _DEFAULT_BATCH_ROWS)["id"] == f"q{_DEFAULT_BATCH_ROWS}"
    assert loader._arrow_reader_is_file is True
    assert loader._arrow_table is None
    assert loader._arrow_reader.num_record_batches > 1


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
def test_legacy_ipc_stream_footer_still_reads():
    """Old ``new_stream`` footers keep ``open_stream`` + ``read_all``."""
    import pyarrow as pa

    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC, load_arrow_row_footer

    table = pa.Table.from_pylist([{"id": f"q{i}", "answers": ["span"] * (i % 3)} for i in range(8)])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc = sink.getvalue().to_pybytes()
    assert ipc[:6] != b"ARROW1"
    n = 8
    header_len = 4 + 4 * (n + 1)
    blob = bytearray(header_len + len(ipc) + 12)
    struct.pack_into("<I", blob, 0, n)
    for i in range(n + 1):
        struct.pack_into("<I", blob, 4 + 4 * i, header_len)
    blob[header_len : header_len + len(ipc)] = ipc
    struct.pack_into("<I", blob, header_len + len(ipc), len(ipc))
    blob[-8:] = _ARROW_FOOTER_MAGIC
    rows = load_arrow_row_footer(bytes(blob))
    assert rows is not None
    assert len(rows) == 8
    assert rows[0]["id"] == "q0"
    assert rows[2]["answers"] == ["span", "span"]

    loader = PyTreeLoader()
    loader._batch_rows = 256
    loader._config = {}
    assert loader._try_arrow_footer_rows(bytes(blob), 0, 2)["id"] == "q2"
    assert loader._arrow_reader_is_file is False
    assert loader._arrow_table is not None


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
@pytest.mark.parametrize("compression", [None, "zstd"])
def test_nested_chunk_bytes_matches_on_disk(tmp_path, compression):
    """``chunk_bytes`` is the written file size, not discarded pytree JSON."""
    from litdata.constants import _ZSTD_AVAILABLE
    from litdata.streaming.cache import Cache

    if compression == "zstd" and not _ZSTD_AVAILABLE:
        pytest.skip("Requires zstd")

    target = 64 * 1024
    cache = Cache(str(tmp_path / "ds"), chunk_bytes=target, compression=compression)
    for i in range(2000):
        cache[i] = {
            "id": f"q{i}",
            "text": os.urandom(128).hex(),
            "choices": {"text": ["A", "B"], "label": ["1", "2"]},
            "answers": ["span"] * (i % 5),
        }
    cache.done()
    cache.merge()
    import json

    index = json.loads((tmp_path / "ds" / "index.json").read_text())
    chunks = index["chunks"]
    assert len(chunks) >= 2
    bins = sorted(p for p in (tmp_path / "ds").glob("*.bin") if ".zstd.bin" not in p.name)
    assert bins
    full = [int(c["chunk_bytes"]) for c in chunks[:-1]]
    for size in full:
        assert 0.45 * target <= size <= 1.6 * target, (size, target, full)


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
def test_flat_chunk_uses_arrow_footer(tmp_path):
    """Tabular HF rows (cnn_dailymail-style str dicts) use the IPC footer, not pytree-only."""
    from litdata import optimize
    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC, load_arrow_row_footer

    out = tmp_path / "flat"
    optimize(fn=_flat_arrow_sample, inputs=list(range(16)), output_dir=str(out), chunk_size=16, num_workers=1)
    chunk = next(out.glob("*.bin"))
    raw = chunk.read_bytes()
    assert raw[-8:] == _ARROW_FOOTER_MAGIC
    rows = load_arrow_row_footer(raw)
    assert rows is not None
    assert len(rows) == 16
    assert rows[0]["text"] == "row 0"
    assert rows[0]["label"] == 0
    ds = StreamingDataset(str(out))
    assert ds[0] == {"text": "row 0", "label": 0}
    assert ds[15] == {"text": "row 15", "label": 1}


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="Requires pyarrow")
def test_flat_chunk_skips_file_zstd(tmp_path):
    """Flat str dicts use Arrow IPC zstd, not LitData whole-file Python inflate."""
    from litdata import optimize
    from litdata.constants import _ZSTD_AVAILABLE
    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC, load_arrow_row_footer

    if not _ZSTD_AVAILABLE:
        pytest.skip("Requires zstd")

    out = tmp_path / "flat-zstd"
    optimize(
        fn=_flat_arrow_sample,
        inputs=list(range(32)),
        output_dir=str(out),
        chunk_size=32,
        num_workers=1,
        compression="zstd",
    )
    bins = list(out.glob("*.bin"))
    assert bins, "expected a chunk"
    assert not any(".zstd.bin" in p.name for p in bins)
    import json

    index = json.loads((out / "index.json").read_text())
    assert index["config"]["compression"] is None
    assert index["config"]["ipc_compression"] == "zstd"
    raw = bins[0].read_bytes()
    assert raw[-8:] == _ARROW_FOOTER_MAGIC
    rows = load_arrow_row_footer(raw)
    assert rows is not None
    assert len(rows) == 32
    ds = StreamingDataset(str(out))
    assert ds[10] == {"text": "row 10", "label": 0}

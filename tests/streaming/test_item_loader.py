from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from litdata.constants import _NUMPY_DTYPES_MAPPING, _TORCH_DTYPES_MAPPING
from litdata.streaming import Cache, item_loader
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader, PyTreeLoader, TokensLoader
from litdata.streaming.writer import index_parquet_dataset


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


@pytest.mark.parametrize("low_memory", [True, False])
def test_parquet_loader_random_access(tmp_path, low_memory):
    """Out-of-order access must return the right row for each index."""
    import random

    parquet_dir = tmp_path / "pq"
    parquet_dir.mkdir()

    row_group_sizes = [10, 5, 5]
    row_group_values = []
    expected = []

    for value, size in enumerate(row_group_sizes):
        row_group_values.append([value] * size)
        expected.extend([value] * size)

    _write_parquet_with_row_groups(parquet_dir / "data.parquet", row_group_values)

    index_parquet_dataset(str(parquet_dir))
    dataset = StreamingDataset(str(parquet_dir), item_loader=ParquetLoader(low_memory=low_memory))

    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    for i in indices:
        assert dataset[i]["col"] == expected[i]


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
        assert groups == {}, f"chunk {chunk_index} still holds row groups: {list(groups)}"

import os
import sys

import pytest

from litdata import map
from litdata.constants import _PYARROW_AVAILABLE
from litdata.processing.readers import BaseReader, ParquetReader


class DummyReader(BaseReader):
    def remap_items(self, items, num_workers: int):
        return [(worker_idx, idx, item) for idx, item in enumerate(items) for worker_idx in range(num_workers)]

    def read(self, item):
        return item


def fn(data: str, output_dir):
    worker_idx, idx, _ = data

    with open(os.path.join(output_dir, f"{worker_idx}_{idx}"), "w") as f:
        f.write("hello world")


def test_reader(tmpdir):
    map(fn, list(range(3)), output_dir=str(tmpdir), reader=DummyReader(), num_workers=2)
    assert sorted(os.listdir(tmpdir)) == ["0_0", "0_1", "0_2", "1_0", "1_1", "1_2"]


def map_parquet(df, output_dir):
    for row in df.iter_batches(batch_size=1):
        for row in row.to_pandas().values.tolist():
            filename = f"{row[0]}_{df.metadata.num_rows}"

            with open(os.path.join(output_dir, filename), "w") as f:
                f.write("hello world")

            return


@pytest.mark.skipif(not _PYARROW_AVAILABLE or sys.platform == "linux", reason="polars and pyarrow are required")
def test_parquet_reader(tmpdir):
    import pandas as pd

    inputs = []

    for i in range(3):
        parquet_path = os.path.join(tmpdir, f"{i}.parquet")
        df = pd.DataFrame(list(range(i * 10, (i + 1) * 10)), columns=["value"])
        df.to_parquet(parquet_path)
        inputs.append(parquet_path)

    cache_folder = os.path.join(tmpdir, "cache")

    map(
        map_parquet,
        inputs=inputs,
        output_dir=os.path.join(tmpdir, "output_dir"),
        reader=ParquetReader(cache_folder, num_rows=5, to_pandas=False),
        num_workers=2,
    )

    assert sorted(os.listdir(os.path.join(tmpdir, "output_dir"))) == ["0_5", "10_5", "15_5", "20_5", "25_5", "5_5"]


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="pyarrow is required")
def test_parquet_reader_row_group_shard_and_columns(tmpdir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = os.path.join(tmpdir, "wide.parquet")
    table = pa.table({"keep": list(range(12)), "drop": list(range(12, 24))})
    pq.write_table(table, path, row_group_size=4)

    reader = ParquetReader(os.path.join(tmpdir, "cache"), num_rows=5, to_pandas=False, columns=["keep"])
    shards = reader.remap_items([path], 1)
    assert len(shards) == 3

    first = reader.read(shards[0])
    assert first.column_names == ["keep"]
    assert first.num_rows == 5
    assert first.column("keep").to_pylist() == [0, 1, 2, 3, 4]


@pytest.mark.skipif(not _PYARROW_AVAILABLE, reason="pyarrow is required")
def test_parquet_reader_filters(tmpdir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = os.path.join(tmpdir, "vals.parquet")
    pq.write_table(pa.table({"value": list(range(10))}), path, row_group_size=10)

    reader = ParquetReader(
        os.path.join(tmpdir, "cache"),
        num_rows=10,
        to_pandas=True,
        filters=[("value", ">=", 7)],
    )
    shards = reader.remap_items([path], 1)
    frame = reader.read(shards[0])
    assert list(frame["value"]) == [7, 8, 9]

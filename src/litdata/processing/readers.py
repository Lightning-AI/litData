# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from abc import ABC, abstractmethod
from typing import Any

from litdata.constants import _PYARROW_AVAILABLE
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.utilities.format import _get_tqdm_iterator_if_available


class BaseReader(ABC):
    """The `BaseReader` interface defines how to read and preprocess data
    from various sources (files, directories, databases, etc.)
    for use with the `map` and `optimize` operations.

    Key concepts:
    - Reader.read(item): Processes a single item and returns data in a format expected by the mapping function (fn)
    - Reader.remap_items(items, num_workers): Optimizes data distribution by potentially resharding items
      (e.g., splitting large items or batching small ones) based on the processing requirements

    Implementation examples:
    - ParquetReader: Handles Parquet file reading with configurable chunk sizes
    - StreamingDataLoaderReader: Wraps a StreamingDataLoader for iterative data access

    Have a look at `tests/processing/test_readers.py::test_parquet_reader`.
    - fn `map_parquet` expects a dataframe object.
    - inputs are parquet files.
    - => `ParquetReader` will read the parquet files (passed as `inputs`) and return a dataframe object
      (which is passed to `map_parquet`).
    """

    def get_num_nodes(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))

    def get_node_rank(self) -> int:
        return int(os.getenv("DATA_OPTIMIZER_NODE_RANK", 0))

    @abstractmethod
    def remap_items(self, items: Any, num_workers: int) -> list[Any]:
        """Remap the items provided by the users into items more adapted to be distributed."""

    @abstractmethod
    def read(self, item: Any) -> Any:
        """Read the data associated to an item."""


class ParquetReader(BaseReader):
    def __init__(self, cache_folder: str, num_rows: int = 65536, to_pandas: bool = True) -> None:
        super().__init__()
        self.cache_folder = cache_folder
        self.num_rows = num_rows
        self.to_pandas = to_pandas

        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow`")

        self.parquet_file = None

    def _get_num_rows(self, path: str) -> int:
        import pyarrow.dataset as ds

        df = ds.dataset(path).scanner()
        return df.count_rows()

    def read(self, filepath: str) -> Any:
        """Read the parquet file and return a parquet file object."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Try to force dellocation to avoid memory leak
        with contextlib.suppress(Exception):
            pa.jemalloc_set_decay_ms(0)

        # close the previous parquet file to release the memory
        if self.parquet_file is not None:
            self.parquet_file.close()
            self.parquet_file = None

        self.parquet_file = pq.ParquetFile(filepath, memory_map=True)
        return self.parquet_file

    def remap_items(self, filepaths: list[str], _: int) -> list[str]:
        """Reshard the parquet files for optimized processing.

        If a parquet file contains more number of rows than a specified `num_rows`,
        it will be split into multiple files for faster processing.
        """
        import pyarrow.parquet as pq

        print("Starting resharding the parquet files for optimized processing.")

        new_items = []

        cache_folder = os.path.join(self.cache_folder, f"{self.num_rows}")
        os.makedirs(cache_folder, exist_ok=True)

        _tqdm = _get_tqdm_iterator_if_available()

        for filepath in filepaths:
            num_rows = self._get_num_rows(filepath)

            table = None
            parquet_filename = os.path.basename(filepath)

            for start in _tqdm(range(0, num_rows, self.num_rows)):
                end = min(start + self.num_rows, num_rows)
                chunk_filepath = os.path.join(cache_folder, f"{start}_{end}_{parquet_filename}")
                new_items.append(chunk_filepath)

                if os.path.exists(chunk_filepath):
                    continue

                if table is None:
                    table = pq.read_table(filepath, memory_map=True)

                pq.write_table(table[start:end], chunk_filepath)

        print("Finished resharding the parquet files for optimized processing.")

        return new_items


class StreamingDataLoaderReader(BaseReader):
    def __init__(self, dataloader: StreamingDataLoader) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.dataloader_iter: Any = None

    def read(self, _: int) -> Any:
        """Read the next item from the dataloader."""
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)

        try:
            # Data is distributed across workers through iterator, similar to DDP.
            # Although the iterator is created within this worker process,
            # distribution is already managed by the StreamingDataLoader and StreamingDataset.
            return next(self.dataloader_iter)
        except StopIteration:
            # This can happen when some workers finish their data slice before others.
            # In multiprocessing scenarios with StreamingDataLoader, this is expected behavior.
            # We return None to signal that this worker has no more data to process.
            return None

    def remap_items(self, items: Any, num_workers: int) -> list[Any]:
        """For StreamingDataLoader, we need to be smarter about item distribution.
        We create enough virtual items so that each worker can process until its
        portion of the dataloader is exhausted.
        """
        # The items parameter is the StreamingDataLoader in this case
        total_items = len(items)
        # Create more virtual items than actual items to ensure all workers
        # get enough chances to process their portion of the dataloader
        return list(range(total_items * 2))

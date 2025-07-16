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
import functools
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from copy import deepcopy
from io import BytesIO, FileIO
from multiprocessing import Queue
from time import sleep, time
from typing import Any, Optional, Union

import numpy as np
import torch

from litdata.constants import (
    _FORCE_DOWNLOAD_TIME,
    _MAX_WAIT_TIME,
    _NUMPY_DTYPES_MAPPING,
    _POLARS_AVAILABLE,
    _PYARROW_AVAILABLE,
    _TORCH_DTYPES_MAPPING,
)
from litdata.debugger import ChromeTraceColors, _get_log_msg
from litdata.streaming.serializers import Serializer
from litdata.utilities._pytree import PyTree, tree_unflatten
from litdata.utilities.encryption import Encryption, EncryptionLevel

Interval = namedtuple("Interval", ["chunk_start", "roi_start_idx", "roi_end_idx", "chunk_end"])

logger = logging.getLogger("litdata.streaming.item_loader")


class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
        force_download_queue: Optional[Queue] = None,
    ) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = {**serializers}
        self._data_format = self._config["data_format"]
        self._shift_idx = len(self._data_format) * 4  # each item takes 4 bytes
        self.region_of_interest = region_of_interest
        self._force_download_queue = force_download_queue

        # setup the serializers on restart
        for data_format in self._data_format:
            serializer = deepcopy(self._serializers[self._data_format_to_key(data_format)])
            serializer.setup(data_format)
            self._serializers[data_format] = serializer

    def force_download(self, chunk_index: int) -> None:
        if self._force_download_queue:
            self._force_download_queue.put(chunk_index)

    @functools.lru_cache(maxsize=128)
    def _data_format_to_key(self, data_format: str) -> str:
        if ":" in data_format:
            serialier, serializer_sub_type = data_format.split(":")
            if serializer_sub_type in self._serializers:
                return serializer_sub_type
            return serialier
        return data_format

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def generate_intervals(self) -> list[Interval]:
        """Returns a list of intervals.

        The structure is: [chunk_start, region_of_interest_start, region_of_interest_end, chunk_end]

        region_of_interest: indicates the indexes a chunk our StreamingDataset is allowed to read.
        """

    @abstractmethod
    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""

    @abstractmethod
    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        """Returns an item loaded from a chunk."""

    def load_item_from_bytes(
        self,
        raw_bytes: bytes,
        chunk_index: int,
    ) -> Any:
        """Returns an item loaded from bytes."""
        raise NotImplementedError("The `load_item_from_bytes` method is not implemented for this item loader.")

    @abstractmethod
    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""

    @abstractmethod
    def encode_data(self, data: list[bytes], sizes: list[int], flattened: list[Any]) -> Any:
        pass


class PyTreeLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(self) -> None:
        super().__init__()
        self._chunk_filepath: str | None = None
        self._decrypted_chunks: dict[int, bytes] = {}
        self._open_handle: FileIO | None = None

    def generate_intervals(self) -> list[Interval]:
        intervals = []
        begin = 0
        end = 0
        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        pass

    def load_item_from_bytes(
        self,
        raw_bytes: bytes,
        chunk_index: int,
    ) -> bytes:
        if self._config.get("encryption"):
            raise ValueError("The `load_item_from_bytes` method does not support encrypted data loading currently.")

        # check for mosaic mds format
        if "format" in self._config and self._config["format"] == "mds":
            item_data = self.mds_deserialize(raw_bytes, chunk_index)
        else:
            item_data = self.deserialize(raw_bytes)

        return item_data

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
        encryption: Optional[Encryption] = None,
    ) -> bytes:
        #
        # Let's say, a chunk contains items from [5,9] index.
        # And the index of the item we want to load is 7.
        # begin = 5
        # index = 7
        #
        # The chunk's binary format is structured as follows:
        #
        # +------------+---------------+-------------+
        # | num_items  | offset_array  | item_data   |
        # +------------+---------------+-------------+
        # | uint32     | uint32[N+1]   | bytes       |
        # | 4 bytes    | 4*(N+1) bytes | variable    |
        # +------------+---------------+-------------+
        #
        # To get to the offset index of the item we want to load, we need to jumpy by:
        #       => 1 + (index - begin) # 1 is added since first 4 bytes store `num_items` (1 uint32)
        #       => 1 + (7 - 5) = 3
        #       => 3 * 4 = 12          # each takes 4 bytes
        #       => offset = 12
        #
        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "B"})
        )
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath != self._chunk_filepath:
            start_time = time()
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes
            requested_force_download = False

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes

                if not requested_force_download and (time() - start_time) > _FORCE_DOWNLOAD_TIME:
                    self.force_download(chunk_index)
                    requested_force_download = True

                if (time() - start_time) > _MAX_WAIT_TIME:
                    raise FileNotFoundError(f"The {chunk_filepath} hasn't been found.")

            self._chunk_filepath = chunk_filepath

            if self._open_handle is not None:
                self._open_handle.close()

            self._open_handle = open(chunk_filepath, "rb", 0)  # noqa: SIM115

        if self._config.get("encryption"):
            data = self._load_encrypted_data(chunk_filepath, chunk_index, offset, encryption)
        else:
            assert self._open_handle
            # load the data from raw bytes using the offset for the item we want to load
            data = self._load_data(self._open_handle, offset)

        # check for mosaic mds format
        if "format" in self._config and self._config["format"] == "mds":
            item_data = self.mds_deserialize(data, chunk_index)
        else:
            item_data = self.deserialize(data)

        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "E"})
        )

        return item_data

    def _load_encrypted_data(
        self, chunk_filepath: str, chunk_index: int, offset: int, encryption: Optional[Encryption]
    ) -> bytes:
        """Load and decrypt data from chunk based on the encryption configuration."""
        # Validate the provided encryption object against the expected configuration.
        self._validate_encryption(encryption)

        # chunk-level decryption
        if self._config["encryption"]["level"] == EncryptionLevel.CHUNK:
            decrypted_data = self._decrypted_chunks.get(chunk_index, None)
            if decrypted_data is None:
                with open(chunk_filepath, "rb", 0) as fp:
                    encrypted_data = fp.read()
                    decrypted_data = encryption.decrypt(encrypted_data)  # type: ignore
                    # Store the decrypted chunk to avoid re-decryption,
                    # also allows to free the previous chunk from the memory
                    self._decrypted_chunks = {chunk_index: decrypted_data}
            data = self._load_data(BytesIO(decrypted_data), offset)

        # sample-level decryption
        elif self._config["encryption"]["level"] == EncryptionLevel.SAMPLE:
            with open(chunk_filepath, "rb", 0) as fp:
                data = self._load_data(fp, offset)
                data = encryption.decrypt(data)  # type: ignore

        else:
            raise ValueError("Invalid encryption level.")

        return data

    def _load_data(self, fp: Union[FileIO, BytesIO], offset: int) -> bytes:
        """Load the data from the file pointer."""
        fp.seek(offset)  # move the file pointer to the offset

        # Refer to `writer.py::_create_chunk` for more details on the chunk's binary format
        # We want to read the `offset_start` and `offset_end` for the item we want to load
        # 2 uint32 (4 bytes each) => 8 bytes; are read to get the offset_start and offset_end
        pair = fp.read(8)
        begin, end = np.frombuffer(pair, np.uint32)

        fp.seek(begin)  # move the file pointer to the offset_start where the item starts
        return fp.read(end - begin)  # read the item

    def mds_deserialize(self, raw_item_data: bytes, chunk_index: int) -> "PyTree":
        """Deserialize the mds raw bytes into their python equivalent."""
        idx = 0
        sizes = []
        column_sizes = self._chunks[chunk_index]["column_sizes"]
        # adapted from: MDSReader.deserialize : https://github.com/mosaicml/streaming/blob/main/streaming/base/format/mds/reader.py
        for size in column_sizes:
            if size:
                sizes.append(size)
            else:
                (size,) = np.frombuffer(raw_item_data[idx : idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        data = []
        for size, data_format in zip(sizes, self._data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def deserialize(self, raw_item_data: bytes) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = self._shift_idx
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def close(self, chunk_index: int) -> None:
        """Close the open file handle."""
        if self._open_handle is not None:
            self._open_handle.close()
            self._open_handle = None

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "B",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "E",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )

    def _validate_encryption(self, encryption: Optional[Encryption]) -> None:
        """Validate the encryption object."""
        if not encryption:
            raise ValueError("Data is encrypted but no encryption object was provided.")
        if encryption.algorithm != self._config["encryption"]["algorithm"]:
            raise ValueError("Encryption algorithm mismatch.")
        if encryption.level != self._config["encryption"]["level"]:
            raise ValueError("Encryption level mismatch.")

    @classmethod
    def encode_data(cls, data: list[bytes], sizes: list[int], flattened: list[Any]) -> tuple[bytes, Optional[int]]:
        """Encodes multiple serialized objects into a single binary format with size metadata.

        This method combines multiple serialized objects into a single byte array, prefixed with their sizes.
        The resulting format is: [size_header][concatenated_data], where size_header contains the byte sizes
        of each object encoded as uint32.

        Args:
            data: List of serialized objects as bytes
            sizes: List of integers representing the byte size of each object
            flattened: List of flattened pytree leaves

        Returns:
            Tuple containing:
                - bytes: Combined binary data with header
                - Optional[int]: dimension of the item (None for PyTreeLoader)

        Example:
            For a row containing [int, image, tensor]:
            - sizes might be [4, 100000, 1000] (number of bytes for each object)
            - data would be their respective serialized bytes
            The method combines these into:

                [size_bytes][int_bytes][image_bytes][tensor_bytes]
        """
        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_handle"] = None
        state["_chunk_filepath"] = None
        return state


class TokensLoader(BaseItemLoader):
    def __init__(self, block_size: Optional[int] = None):
        """The Tokens Loader is an optimizer item loader for NLP.

        Args:
            block_size: The context length to use during training.

        """
        super().__init__()
        self._block_size = block_size
        self._mmaps: dict[int, np.memmap] = {}
        self._buffers: dict[int, bytes] = {}
        # keeps track of number of readers for each chunk (can be more than 1 if multiple workers are reading)
        self._counter = defaultdict(int)
        self._dtype: Optional[torch.dtype] = None
        self._chunk_filepaths: dict[str, bool] = {}

    def state_dict(self) -> dict:
        assert self._block_size
        return {
            "block_size": self._block_size,
        }

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        super().setup(config, chunks, serializers, region_of_interest)

        serializer_name, dtype_index = self._data_format[0].split(":")
        if serializer_name not in ["no_header_numpy", "no_header_tensor"]:
            raise ValueError("The provided data format isn't supported.")

        self._serializer_name = serializer_name
        self._dtype = (
            _TORCH_DTYPES_MAPPING[int(dtype_index)]  # type: ignore
            if serializer_name == "no_header_tensor"
            else _NUMPY_DTYPES_MAPPING[int(dtype_index)]
        )
        if all(chunk["dim"] is None for chunk in self._chunks):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> list[Interval]:
        assert self._block_size
        intervals = []
        begin = 0
        end = 0
        for idx, chunk in enumerate(self._chunks):
            dim = chunk["dim"]  # number of tokens in the chunk
            num_blocks = dim // self._block_size
            end += num_blocks
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]
            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += num_blocks
        return intervals

    def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        self._counter[chunk_index] += 1
        if chunk_index in self._mmaps:
            return
        chunk = self._chunks[chunk_index]

        # Skip the header
        # [number of items] + [number of offsets (number of items in the chunk + 1)] {since offset starts at 0}
        # multiplied by the header encoding dtype (np.uint32)
        # for more details on the chunk's binary format, see `writer.py::_create_chunk`
        offset = (1 + chunk["chunk_size"] + 1) * 4
        mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
        self._mmaps[chunk_index] = mmap
        self._buffers[chunk_index] = memoryview(mmap)  # type: ignore

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        # This is called within the prepare chunks thread, so we overlap data loading with data reading.
        if chunk_filepath not in self._chunk_filepaths:
            self._chunk_filepaths[chunk_filepath] = True

        if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0:
            self._load_chunk(chunk_index, chunk_filepath)

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> torch.Tensor:
        assert self._block_size

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "B"})
        )

        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > filesize_bytes

            start_time = time()
            requested_force_download = False

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes

                if not requested_force_download and (time() - start_time) > _FORCE_DOWNLOAD_TIME:
                    self.force_download(chunk_index)
                    requested_force_download = True

                if (time() - start_time) > _MAX_WAIT_TIME:
                    raise FileNotFoundError(f"The {chunk_filepath} hasn't been found.")

            self._chunk_filepaths[chunk_filepath] = True

        self._load_chunk(chunk_index, chunk_filepath)
        assert self._dtype

        buffer: bytes = self._buffers[chunk_index]

        # offset: how many bytes to skip to get to the item we want to load
        #       -> if chunk begins at 5, and we want to load the item at index 7,
        #       -> we need to skip 2 items, and each item has `self._block_size` tokens
        #       -> and each token takes `self._dtype.itemsize` bytes
        #
        # Note: We have already accounted for offsets corresponding to starting bytes in `_load_chunk` function
        # while creating the memory map.
        offset = self._dtype.itemsize * (index - begin) * self._block_size

        if self._serializer_name == "no_header_tensor":
            # count: number of tokens to read from buffer => `self._block_size`
            data = torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        else:
            # count: number of tokens to read from buffer => `self._block_size`
            data = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)  # type: ignore

        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "E"})
        )
        return data

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "B",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )
        if os.path.exists(chunk_filepath):
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            if chunk_index in self._mmaps:
                # explicitly close before deleting. Won't raise error if already closed.
                self._mmaps[chunk_index]._mmap.close()
                del self._mmaps[chunk_index]
                del self._counter[chunk_index]
            os.remove(chunk_filepath)
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "E",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )

    def close(self, chunk_index: int) -> None:
        """Release the memory-mapped file for a specific chunk index."""
        self._counter[chunk_index] -= 1

        if self._counter[chunk_index] == 0:
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            if chunk_index in self._mmaps:
                self._mmaps[chunk_index]._mmap.close()
                del self._mmaps[chunk_index]

    @classmethod
    def encode_data(cls, data: list[bytes], _: list[int], flattened: list[Any]) -> tuple[bytes, Optional[int]]:
        r"""Encodes tokenized data into a raw byte format while preserving dimensional information.

        Parameters:
        - data (List[bytes]): A list containing a single element, which is the raw byte
          representation of tokenized data.
        - _ (List[int]): A list containing sizes of each PyTree leaf in the item.
          Since only one item (tokens) is present, this argument is ignored.
        - flattened (List[Any]): A list containing a single element, which is the list of tokens.

        Example:
            - Original data: "hello world"
            - Tokenized data: [1, 2] (word tokenizer)
            - Data (raw bytes): [b'\x01\x00\x00\x00\x02\x00\x00\x00']
              (raw bytes representing the tokenized data)
            - Flattened data: [[1, 2]] (returned by PyTree's `flatten` function)

        Returns:
        - Tuple[bytes, Optional[int]]:
            - bytes: The raw byte representation of tokenized data.
            - dimension: The number of tokens in the data (extracted from `flattened[0].shape[0]`).
        """
        return data[0], flattened[0].shape[0]


class ParquetLoader(BaseItemLoader):
    def __init__(self, pre_load_chunk: bool = False, low_memory: bool = True) -> None:
        if not _POLARS_AVAILABLE:
            raise ModuleNotFoundError(
                "You are using the Parquet item loader, which depends on `Polars > 1.0.0`.",
                "Please, run: `pip install polars>1.0.0`",
            )
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow`")

        self._chunk_filepaths: dict[str, bool] = {}
        self._pre_load_chunk = pre_load_chunk
        self._low_memory = low_memory

        if not self._low_memory:
            logger.warning(
                "You have set low_memory=False in ParquetLoader. "
                "This may result in high memory usage when processing large Parquet chunk files. "
                "Consider setting low_memory=True to reduce memory consumption."
            )

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = {**serializers}
        self._data_format = self._config["data_format"]
        self._shift_idx = len(self._data_format) * 4
        self.region_of_interest = region_of_interest
        self._df: dict[int, Any] = {}
        self._chunk_row_groups: dict[int, Any] = {}
        self._chunk_row_group_item_read_count: dict[int, Any] = {}

    def generate_intervals(self) -> list[Interval]:
        intervals = []
        begin = 0
        end = 0
        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Preload the chunk in the background to gain some time."""
        if not self._pre_load_chunk or self._low_memory:
            return

        import polars as pl

        if chunk_index not in self._df and os.path.exists(chunk_filepath):
            self._df[chunk_index] = pl.scan_parquet(chunk_filepath, low_memory=True).collect()

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        """Returns an item loaded from a parquet chunk."""
        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "B"})
        )
        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes

            self._chunk_filepaths[chunk_filepath] = True

        # relative index of the desired row within the chunk.
        relative_index = index - begin
        if self._low_memory:
            item_data = self._get_item_with_low_memory(chunk_index, chunk_filepath, relative_index)
        else:
            item_data = self._get_item(chunk_index, chunk_filepath, relative_index)

        logger.debug(
            _get_log_msg({"name": f"load_item_from_chunk_for_chunk_index_{chunk_index}_and_index_{index}", "ph": "E"})
        )
        return item_data

    def _get_item_with_low_memory(self, chunk_index: int, chunk_filepath: str, row_index: int) -> Any:
        """Retrieve a dataframe row from a parquet chunk in low memory mode.

        This method reads only the necessary row group from the parquet file using PyArrow and Polars,
        which helps in reducing memory usage.

        Args:
            chunk_index (int): The index of the chunk to be accessed.
            chunk_filepath (str): The file path of the parquet chunk.
            row_index (int): The relative row index within the loaded chunk.

        Returns:
            Any: The dataframe row corresponding to the specified index.
        """
        import polars as pl
        import pyarrow.parquet as pq

        # Load the Parquet file metadata if not already loaded
        if chunk_index not in self._df:
            self._df[chunk_index] = pq.ParquetFile(chunk_filepath)

        # Determine the row group and the row index within the row group
        parquet_file = self._df[chunk_index]
        num_rows_per_row_group = parquet_file.metadata.row_group(0).num_rows
        row_group_index = row_index // num_rows_per_row_group
        row_index_within_group = row_index % num_rows_per_row_group

        # Check if the row group is already loaded
        if chunk_index in self._chunk_row_groups and row_group_index in self._chunk_row_groups[chunk_index]:
            # Use the cached row group
            row_group_df = self._chunk_row_groups[chunk_index][row_group_index]
            # update read count
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] += 1
        else:
            # Load the row group and convert it to a Polars DataFrame
            row_group = self._df[chunk_index].read_row_group(row_group_index)
            row_group_df = pl.from_arrow(row_group)

            # Cache the loaded row group
            if chunk_index not in self._chunk_row_groups:
                self._chunk_row_groups[chunk_index] = {}
                self._chunk_row_group_item_read_count[chunk_index] = {}

            self._chunk_row_groups[chunk_index][row_group_index] = row_group_df
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] = 1

        # Check if the row group has been fully read and release memory if necessary
        read_count = self._chunk_row_group_item_read_count[chunk_index][row_group_index]
        if read_count >= num_rows_per_row_group:
            # Release memory for the fully read row group
            del self._chunk_row_groups[chunk_index][row_group_index]
            del self._chunk_row_group_item_read_count[chunk_index][row_group_index]

        # Return the specific row from the dataframe
        # Note: The `named=True` argument is used to return the row as a dictionary
        return row_group_df.row(row_index_within_group, named=True)  # type: ignore

    def _get_item(self, chunk_index: int, chunk_filepath: str, index: int) -> Any:
        """Retrieve a dataframe row from a parquet chunk by loading the entire chunk into memory.

        Note:
            This method reads the complete parquet file using Polars. Exercise caution with large files as it
            may significantly increase memory usage.

        Args:
            chunk_index (int): The index of the chunk to be accessed.
            chunk_filepath (str): The file path of the parquet chunk.
            index (int): The relative row index within the loaded chunk.

        Returns:
            Any: The dataframe row corresponding to the specified index.
        """
        import polars as pl

        if chunk_index not in self._df:
            self._df[chunk_index] = pl.scan_parquet(chunk_filepath, low_memory=True).collect()

        # Retrieve the specific row from the dataframe
        # Note: The `named=True` argument is used to return the row as a dictionary
        return self._df[chunk_index].row(index, named=True)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "B",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )
        if chunk_index in self._df:
            del self._df[chunk_index]
        if chunk_index in self._chunk_row_groups:
            del self._chunk_row_groups[chunk_index]

        if chunk_index in self._chunk_row_group_item_read_count:
            del self._chunk_row_group_item_read_count[chunk_index]
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)
        logger.debug(
            _get_log_msg(
                {
                    "name": f"delete_chunk_for_chunk_index_{chunk_index}",
                    "ph": "E",
                    "cname": ChromeTraceColors.BRIGHT_RED,
                }
            )
        )

    def close(self, chunk_index: int) -> None:
        """Release the memory-mapped file for a specific chunk index."""
        if chunk_index in self._df:
            del self._df[chunk_index]

        if chunk_index in self._chunk_row_groups:
            del self._chunk_row_groups[chunk_index]

        if chunk_index in self._chunk_row_group_item_read_count:
            del self._chunk_row_group_item_read_count[chunk_index]

    def encode_data(self, data: list[bytes], sizes: list[int], flattened: list[Any]) -> Any:
        pass

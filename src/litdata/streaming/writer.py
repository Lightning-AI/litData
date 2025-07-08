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

import json
import os
import uuid
import warnings
from dataclasses import dataclass
from multiprocessing import Queue
from time import sleep, time
from typing import Any, Optional, Union

import numpy as np

from litdata.constants import _INDEX_FILENAME, _POLARS_AVAILABLE, _TQDM_AVAILABLE
from litdata.processing.utilities import get_worker_rank
from litdata.streaming.compression import _COMPRESSORS, Compressor
from litdata.streaming.item_loader import BaseItemLoader, ParquetLoader, PyTreeLoader
from litdata.streaming.serializers import Serializer, _get_serializers
from litdata.utilities._pytree import PyTree, tree_flatten, treespec_dumps
from litdata.utilities.encryption import Encryption, EncryptionLevel
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.format import _convert_bytes_to_int, _human_readable_bytes
from litdata.utilities.parquet import get_parquet_indexer_cls
from litdata.utilities.torch_utils import is_local_rank_0, maybe_barrier


@dataclass
class Item:
    index: int
    data: bytes
    bytes: int
    dim: Optional[int] = None

    def __len__(self) -> int:
        return self.bytes


class BinaryWriter:
    def __init__(
        self,
        cache_dir: str,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        compression: Optional[str] = None,
        encryption: Optional[Encryption] = None,
        follow_tensor_dimension: bool = True,
        serializers: Optional[dict[str, Serializer]] = None,
        chunk_index: Optional[int] = None,
        item_loader: Optional[BaseItemLoader] = None,
        msg_queue: Optional[Queue] = None,
    ):
        """The BinaryWriter enables to chunk dataset into an efficient streaming format for cloud training.

        Args:
            cache_dir: The path to where the chunks will be saved.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.
            compression: The compression algorithm to use.
            encryption: The encryption algorithm to use.
            follow_tensor_dimension: Whether to follow the tensor dimension when serializing the data.
            serializers: Provide your own serializers.
            chunk_index: The index of the chunk to start from.
            item_loader: The object responsible to generate the chunk intervals and load an item from a chunk.
            msg_queue: Optional message queue to send messages to the main process.

        """
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)

        if (isinstance(self._cache_dir, str) and not os.path.exists(self._cache_dir)) or self._cache_dir is None:
            raise FileNotFoundError(f"The provided cache directory `{self._cache_dir}` doesn't exist.")

        if (chunk_size is None and chunk_bytes is None) or (chunk_size and chunk_bytes):
            raise ValueError("Either one of the `chunk_size` or the `chunk_bytes` need to be provided.")

        self._serializers: dict[str, Serializer] = _get_serializers(serializers)
        self._serializers_extra: dict[str, Serializer] = {}
        self._chunk_size = chunk_size
        self._chunk_bytes = _convert_bytes_to_int(chunk_bytes) if isinstance(chunk_bytes, str) else chunk_bytes
        self._compression = compression
        self._encryption = encryption
        self._item_loader = item_loader or PyTreeLoader()
        self.msg_queue = msg_queue

        self._data_format: Optional[list[str]] = None
        self._data_spec: Optional[PyTree] = None

        if self._compression:
            if len(_COMPRESSORS) == 0:
                raise ValueError("No compression algorithms are installed.")

            if self._compression not in _COMPRESSORS:
                raise ValueError(
                    f"The provided compression {self._compression} isn't available in {sorted(_COMPRESSORS)}"
                )
            self._compressor: Compressor = _COMPRESSORS[self._compression]

        self._serialized_items: dict[int, Item] = {}
        self._chunk_index = chunk_index or 0
        self._min_index: Optional[int] = None
        self._max_index: Optional[int] = None
        self._chunks_info: list[dict[str, Any]] = []
        self._worker_env: Optional[_WorkerEnv] = None
        self._rank: Optional[int] = None
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._follow_tensor_dimension = follow_tensor_dimension

        self._per_sample_num_bytes = 0
        self._per_sample_num_items = 0
        self.last_checkpoint_chunk_info: list[dict[str, Any]] = []

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._is_done:
            return True
        files = os.listdir(self._cache_dir)
        index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]
        worker_env = _WorkerEnv.detect()
        data_optimiser_num_workers = os.getenv("DATA_OPTIMIZER_NUM_WORKERS", None)
        if data_optimiser_num_workers is not None:
            self._is_done = len(index_files) == int(data_optimiser_num_workers)
        else:
            self._is_done = len(index_files) == self._distributed_env.world_size * worker_env.world_size
        return self._is_done

    @property
    def rank(self) -> int:
        """Returns the rank of the writer."""
        if self._rank is None:
            rank = os.getenv("DATA_OPTIMIZER_GLOBAL_RANK", None)
            if rank:
                self._rank = int(rank)
            else:
                self._worker_env = _WorkerEnv.detect()
                self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the writer."""
        return {
            "compression": self._compression,
            "chunk_size": self._chunk_size,
            "chunk_bytes": self._chunk_bytes,
            "data_format": self._data_format,
            "data_spec": treespec_dumps(self._data_spec) if self._data_spec else None,
            "encryption": self._encryption.state_dict() if self._encryption else None,
            "item_loader": self._item_loader.__class__.__name__,
        }

    def serialize(self, items: Any) -> tuple[bytes, Optional[int]]:
        """Serialize a dictionary into its binary format."""
        # Flatten the items provided by the users
        flattened, data_spec = tree_flatten(items)

        # Collect the sizes and associated bytes for each item
        sizes: list[int] = []
        data: list[bytes] = []

        if self._data_format is None:
            data_format: list[str] = []
            for item in flattened:
                data_format.append(self._serialize(item, sizes, data))

            worker_rank = get_worker_rank()
            if worker_rank is not None:
                msg = f"Rank {worker_rank} inferred the following `{data_format}` data format."
                if self.msg_queue is not None:
                    self.msg_queue.put_nowait(msg)
                else:
                    print(msg, flush=True)
            self._data_format = data_format
            self._data_spec = data_spec
        else:
            # tiny optimization to avoid looping over all the data format
            self._serialize_with_data_format(flattened, sizes, data, self._data_format)

        return self._item_loader.encode_data(data, sizes, flattened)

    def _serialize(self, item: Any, sizes: list[int], data: list[bytes]) -> str:
        """Serialize a given item and append its size and bytes to the sizes and data array."""
        for serializer_name, serializer in self._serializers.items():
            if serializer.can_serialize(item):
                serialized_item, name = serializer.serialize(item)
                data.append(serialized_item)
                sizes.append(serializer.size if hasattr(serializer, "size") else len(serialized_item))
                name = name or serializer_name
                if name and name not in self._serializers_extra:
                    self._serializers_extra[name] = serializer
                return name
        raise ValueError(f"The provided item isn't serializable. Found {item}")

    def _serialize_with_data_format(
        self, item: Any, sizes: list[int], data: list[bytes], data_format: list[str]
    ) -> None:
        """Serialize a given item and append its size and bytes to the sizes and data array."""
        assert data_format
        for element, item_format in zip(item, data_format):
            serializer = self._serializers_extra[item_format]
            serialized_item, _ = serializer.serialize(element)
            data.append(serialized_item)
            sizes.append(serializer.size if hasattr(serializer, "size") else len(serialized_item))

    def _create_chunk(self, filename: str, on_done: bool = False) -> bytes:
        """Creates a binary chunk file from serialized items."""
        # The chunk's binary format is structured as follows:

        # +------------+---------------+-------------+
        # | num_items  | offset_array  | item_data   |
        # +------------+---------------+-------------+
        # | uint32     | uint32[N+1]   | bytes       |
        # | 4 bytes    | 4*(N+1) bytes | variable    |
        # +------------+---------------+-------------+

        # Where:
        # - num_items: Number of items in the chunk (N)
        # - offset_array: Array of N+1 offsets indicating where each item begins/ends
        # - item_data: Concatenated binary data of all items

        # Example:
        #     For a chunk with 3 items of sizes [10, 20, 15] bytes:
        #     - num_items = 3 (4 bytes)
        #     - offset_array = [start, start+10, start+30, start+45]
        #       where start = 4 + (4 * 4) = 20 bytes (header size)
        #     - item_data = concatenated bytes of all items

        # This format allows direct access to any item by reading its offset
        #   from `offset_array[i]` to `offset_array[i+1]`.
        # Then, read bytes from `offset_start` to `offset_end` to get the item bytes.
        # Now, item_loader can use these raw bytes to deserialize the item.

        items: list[Item] = []

        if on_done:
            indices = sorted(self._serialized_items.keys())
            for i in range(len(indices) - 1):
                assert indices[i] == indices[i + 1] - 1, indices
            items = [self._serialized_items.pop(index) for index in indices]
        else:
            assert self._max_index is not None, (self._max_index, self._min_index)
            assert self._min_index is not None, (self._max_index, self._min_index)
            if self._max_index == self._min_index:
                # A single item is larger than the target chunk size; allow the chunk to be bigger than the target size
                items.append(self._serialized_items.pop(self._max_index))
            items.extend(self._serialized_items.pop(index) for index in range(self._min_index, self._max_index))

        if len(items) == 0:
            raise RuntimeError(
                "The items shouldn't have an empty length. Something went wrong."
                f" Found {self._pretty_serialized_items()} with boundaries: {self._min_index}, {self._max_index}."
            )

        num_items = np.uint32(len(items))  # total number of items in the chunk
        sizes = list(map(len, items))  # list of sizes (length of bytes) of each item
        offsets = np.array([0] + sizes).cumsum().astype(np.uint32)  # let's say: [0, 10, 30, 45]

        # add the number of bytes taken to store (num_items and offsets). Let's say 60: offsets -> [60, 70, 90, 105]
        offsets += len(num_items.tobytes()) + len(offsets.tobytes())
        sample_data = b"".join([item.data for item in items])

        # combine all bytes data which will be written to the chunk file
        data = num_items.tobytes() + offsets.tobytes() + sample_data

        # Whether to encrypt the data at the chunk level
        if self._encryption and self._encryption.level == EncryptionLevel.CHUNK:
            data = self._encryption.encrypt(data)

        current_chunk_bytes = len(data)

        if self._chunk_bytes and current_chunk_bytes > self._chunk_bytes:
            warnings.warn(
                f"An item was larger than the target chunk size ({_human_readable_bytes(self._chunk_bytes)})."
                f" The current chunk will be {_human_readable_bytes(current_chunk_bytes)} in size.",
                UserWarning,
            )

        if self._chunk_size:
            assert num_items.item() <= self._chunk_size

        dim: Optional[int] = None
        if items[0].dim:
            dim = sum([item.dim if item.dim is not None else 0 for item in items])

        chunk_info = {
            "chunk_bytes": current_chunk_bytes,
            "chunk_size": num_items.item(),
            "filename": filename,
            "dim": dim,
        }

        self._chunks_info.append(chunk_info)

        return data

    def get_chunk_filename(self) -> str:
        if self._compression:
            return f"chunk-{self.rank}-{self._chunk_index}.{self._compression}.bin"
        return f"chunk-{self.rank}-{self._chunk_index}.bin"

    def write_chunk(self, on_done: bool = False) -> str:
        """Write a chunk to the filesystem."""
        filename = self.get_chunk_filename()
        self.write_chunk_to_file(self._create_chunk(filename, on_done=on_done), filename)
        self._chunk_index += 1
        return os.path.join(self._cache_dir, filename)

    def __setitem__(self, index: int, items: Any) -> None:
        """Store an item to a chunk.

        The index needs to be provided in order.

        This is handled by the samplers automatically. This ensures we can map an index to a shard from an interval.

        """
        self.add_item(index, items)

    def add_item(self, index: int, items: Any) -> Optional[str]:
        """Given an index and items will serialize the items and store an Item object to the growing
        `_serialized_items`.
        """
        if index in self._serialized_items:
            raise ValueError(f"The provided index {index} already exists in the cache.")

        data, dim = self.serialize(items)

        # Whether to encrypt the data at the sample level
        if self._encryption and self._encryption.level == EncryptionLevel.SAMPLE:
            data = self._encryption.encrypt(data)

        self._serialized_items[index] = Item(
            index=index,
            data=data,
            bytes=len(data),
            dim=dim,
        )
        if self._min_index is None:
            # When processing the first item for the current chunk
            indexes = list(self._serialized_items.keys())
            self._max_index = self._min_index = indexes[0] if len(indexes) == 1 else min(*indexes)
            self._per_sample_num_items = self._per_sample_num_bytes = 0
            if not self._should_write():
                return None
        elif index < self._min_index:
            # reset the "temp" chunk
            self._max_index = self._min_index = index
            self._per_sample_num_items = self._per_sample_num_bytes = 0
            if not self._should_write():
                return None
        elif index == self._max_index:
            if not self._should_write():
                return None
        else:
            return None

        filepath = os.path.join(self._cache_dir, self.get_chunk_filename())

        self.write_chunk()

        # now to reset
        self._min_index = None
        self._max_index = None
        self._per_sample_num_bytes = 0
        self._per_sample_num_items = 0

        return filepath

    def _should_write(self) -> bool:
        # TODO: Misleading method name, it modifies `self._min_index` and `self._max_index`!
        if not self._serialized_items:
            return False

        if not isinstance(self._max_index, int):
            return False

        # We have already validated the indexes from the interval `min_index` to `max_index`` are in `_serialized_items`
        # Resetting the num_bytes and  num_items back the values.
        num_bytes = self._per_sample_num_bytes
        num_items = self._per_sample_num_items
        index = self._max_index
        while True:
            item = self._serialized_items.get(index, None)
            if item:
                num_bytes += item.bytes
                num_items += item.dim if item.dim else 1
                index += 1
                if (self._chunk_bytes and self._chunk_bytes < num_bytes) or (
                    self._chunk_size and num_items > self._chunk_size
                ):
                    self._max_index = index - 1
                    return True
            else:
                self._per_sample_num_bytes = num_bytes
                self._per_sample_num_items = num_items
                self._max_index = index
                return False

    def write_chunk_to_file(
        self,
        raw_data: bytes,
        filename: str,
    ) -> None:
        """Write chunk bytes to a file."""
        # Whether to compress the raw bytes
        if self._compression:
            raw_data = self._compressor.compress(raw_data)

        # Write the binary chunk file
        with open(os.path.join(self._cache_dir, filename), "wb") as out:
            out.write(raw_data)

    def write_chunks_index(self) -> str:
        """Write the chunks index to a JSON file."""
        if len(self._chunks_info) == 0:
            return ""
        filepath = os.path.join(self._cache_dir, f"{self.rank}.{_INDEX_FILENAME}")
        config = self.get_config()
        with open(filepath, "w") as out:
            json.dump({"chunks": self._chunks_info, "config": config}, out, sort_keys=True)
        return filepath

    def done(self) -> list[str]:
        """Called when StopIteration is triggered."""
        filepaths: list[str] = []
        if self.filled:
            return filepaths

        # Try writing down an chunks
        while self._should_write():
            filepaths.append(self.write_chunk())

        # If any elements is left, try writing one last chunk
        if self._serialized_items:
            filepaths.append(self.write_chunk(True))

        # Write down the index file
        self.write_chunks_index()

        self._is_done = True
        return filepaths

    def merge(self, num_workers: int = 1, node_rank: Optional[int] = None) -> None:
        """Once all the workers have written their own index, the merge function is responsible to read and merge them
        into a single index.
        """
        num_workers = num_workers or 1

        # Only for non rank 0
        if self.rank != 0:
            while not os.path.exists(os.path.join(self._cache_dir, _INDEX_FILENAME)):
                sleep(0.01)
            return

        # Wait for all indexes to be available
        is_done = False
        while not is_done:
            files = os.listdir(self._cache_dir)

            # Return if the index already exists
            if _INDEX_FILENAME in files:
                return

            index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]

            # When using the Data Optimizer, we don't use multi processes.
            is_done = len(index_files) == self._distributed_env.world_size * num_workers
            sleep(0.01)

        self._merge_no_wait(node_rank=node_rank)

    def _merge_no_wait(self, node_rank: Optional[int] = None, existing_index: Optional[dict[str, Any]] = None) -> None:
        """Once all the workers have written their own index, the merge function is responsible to read and merge them
        into a single index.

        Args:
            node_rank: The node rank of the index file
            existing_index: Existing index to be added to the newly created one.

        """
        files = os.listdir(self._cache_dir)
        index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]

        chunks_info = []
        config = None

        if existing_index is not None:
            chunks_info.extend(existing_index["chunks"])
            config = existing_index["config"]

        for index_filename in sorted(index_files):
            chunk_path = os.path.join(self._cache_dir, index_filename)
            with open(chunk_path) as f:
                data = json.load(f)

                if config is None:
                    config = data["config"]

                elif config != data["config"]:
                    raise Exception(
                        "The config isn't consistent between chunks. This shouldn't have happened."
                        f"Found {config}; {data['config']}."
                    )

                chunks_info.extend(data["chunks"])

            os.remove(chunk_path)

        if node_rank is None:
            with open(os.path.join(self._cache_dir, _INDEX_FILENAME), "w") as f:
                data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
                json.dump(data, f, sort_keys=True)
        else:
            with open(os.path.join(self._cache_dir, f"{node_rank}-{_INDEX_FILENAME}"), "w") as f:
                json.dump({"chunks": chunks_info, "config": config}, f, sort_keys=True)

    def _should_raise(self, data_format_1: list[str], data_format_2: list[str]) -> bool:
        if len(data_format_1) != len(data_format_2):
            return True

        def is_non_valid(f1: str, f2: str) -> bool:
            if f1 in ["pil", "jpeg"] and f2 in ["pil", "jpeg"]:
                return False
            return f1 != f2

        return any(is_non_valid(f1, f2) for f1, f2 in zip(data_format_1, data_format_2))

    def _pretty_serialized_items(self) -> dict[int, Item]:
        out = {}
        for key, value in self._serialized_items.items():
            # drop `data` as it would make logs unreadable.
            out[key] = Item(
                index=value.index,
                bytes=value.bytes,
                dim=value.dim,
                data=b"",
            )
        return out

    def save_checkpoint(self, checkpoint_dir: str = ".checkpoints") -> Optional[str]:
        """Save the current state of the writer to a checkpoint."""
        checkpoint_dir = os.path.join(self._cache_dir, checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        if self._chunks_info == self.last_checkpoint_chunk_info:
            # to avoid saving the same checkpoint twice
            return None

        unique_id = uuid.uuid4().hex
        done_till_index = sum(chnk_info["chunk_size"] for chnk_info in self._chunks_info)

        checkpoint_filepath = os.path.join(checkpoint_dir, f"checkpoint-{self.rank}-{unique_id}.json")

        checkPoint = {"chunks": self._chunks_info, "config": self.get_config(), "done_till_index": done_till_index}

        with open(checkpoint_filepath, "w") as f:
            json.dump(checkPoint, f)

        return checkpoint_filepath


def index_parquet_dataset(
    pq_dir_url: str,
    cache_dir: Optional[str] = None,
    storage_options: Optional[dict] = {},
    num_workers: int = 4,
) -> None:
    """Index a Parquet dataset from a specified URL.


    This function scans the metadata of all `.parquet` files in the specified directory URL, extracts metadata
    such as file size, chunk size, and data types, and indexes them. The index is saved as a JSON file in the
    specified cache directory.

    Args:
        pq_dir_url (str): URL of the directory containing the Parquet files.
        cache_dir (Optional[str]): Local cache directory for storing temporary files.
            For HF dataset, index.json file will be stored here.
        storage_options (Optional[Dict]): Additional storage options for accessing the Parquet files.
        num_workers (int): Number of workers to download metadata of Parquet files and index them.

    Raises:
        ModuleNotFoundError: If the required `polars` module is not installed.
    """
    if not _POLARS_AVAILABLE:
        raise ModuleNotFoundError("Please, run: `pip install polars`")

    if is_local_rank_0():
        # in multi-node setup, each node's first process should do the indexing,
        # but only first node's first process should upload the index (if uploading is needed)
        pq_chunks_info = []
        config: dict[str, Any] = {
            "compression": None,
            "chunk_size": None,
            "chunk_bytes": None,
            "data_format": [],
            "data_spec": None,
            "encryption": None,
            "item_loader": ParquetLoader.__name__,
        }

        pq_dir_class = get_parquet_indexer_cls(pq_dir_url, cache_dir, storage_options, num_workers)

        if _TQDM_AVAILABLE:
            from tqdm.auto import tqdm as _tqdm

            pbar = _tqdm(
                desc="Indexing progress",
                total=len(pq_dir_class.files),
                smoothing=0,
                mininterval=1,
                leave=True,
                dynamic_ncols=True,
                unit="step",
            )

        results = {}
        # iterate through the directory and index each file ending with ".parquet"
        for file_metadata, order in pq_dir_class:
            chunk_dtypes = file_metadata["data_types"]

            if len(config["data_format"]) != 0 and config["data_format"] != chunk_dtypes:
                raise Exception(
                    "The config isn't consistent between chunks. This shouldn't have happened."
                    f"Found {config}; {chunk_dtypes}."
                )
            config["data_format"] = chunk_dtypes
            chunk_info = {
                "chunk_bytes": file_metadata["file_size"],
                "chunk_size": file_metadata["num_rows"],
                "filename": file_metadata["file_name"],
                "dim": None,
            }
            results[order] = chunk_info
            if _TQDM_AVAILABLE:
                pbar.update(1)

        for i in sorted(results.keys()):
            pq_chunks_info.append(results[i])

        del results
        print(flush=True)  # to prevent truncated printing when using concurrent threads/processes
        pq_dir_class.write_index(pq_chunks_info, config)

    maybe_barrier()  # wait for all processes to reach this point

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

import logging
import os
from time import time
from typing import Any, Callable, Optional, Union

import numpy as np
from torch.utils.data import IterableDataset

from litdata import __version__
from litdata.constants import _INDEX_FILENAME
from litdata.debugger import _get_log_msg
from litdata.helpers import _check_version_and_prompt_upgrade
from litdata.streaming import Cache
from litdata.streaming.item_loader import BaseItemLoader, ParquetLoader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.streaming.shuffle import FullShuffle, NoShuffle, Shuffle
from litdata.utilities.dataset_utilities import _should_replace_path, _try_create_cache_dir, subsample_streaming_dataset
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _is_in_dataloader_worker, _WorkerEnv
from litdata.utilities.format import _convert_bytes_to_int
from litdata.utilities.hf_dataset import index_hf_dataset
from litdata.utilities.shuffle import (
    _find_chunks_per_workers_on_which_to_skip_deletion,
    _map_node_worker_rank_to_chunk_indexes_to_not_delete,
)

logger = logging.getLogger("litdata.streaming.dataset")


class StreamingDataset(IterableDataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        cache_dir: Optional[Union[str, "Dir"]] = None,
        item_loader: Optional[BaseItemLoader] = None,
        shuffle: bool = False,
        drop_last: Optional[bool] = None,
        seed: int = 42,
        serializers: Optional[dict[str, Serializer]] = None,
        max_cache_size: Union[int, str] = "100GB",
        subsample: float = 1.0,
        encryption: Optional[Encryption] = None,
        storage_options: Optional[dict] = {},
        session_options: Optional[dict] = {},
        max_pre_download: int = 2,
        index_path: Optional[str] = None,
        force_override_state_dict: bool = False,
        transform: Optional[Union[Callable, list[Callable]]] = None,
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Args:
            input_dir: Path to the folder where the input data is stored. Supports paths ending with `.parquet`
                with wildcards in the basename to stream specific Parquet files.
            cache_dir: Path to the folder where the cache data is stored. If not provided, the cache will be stored
                in the default cache directory.
            item_loader: The logic to load an item from a chunk.
            shuffle: Whether to shuffle the data.
            drop_last: If `True`, drops the last items to ensure that
                all processes/workers return the same amount of data.
                The argument `drop_last` is set to `True` in a distributed setting
                and `False` otherwise.
            seed: Random seed for shuffling.
            serializers: The serializers used to serialize and deserialize the chunks.
            max_cache_size: The maximum cache size used by the StreamingDataset.
            subsample: Float representing fraction of the dataset to be randomly sampled (e.g., 0.1 => 10% of dataset).
            encryption: The encryption object to use for decrypting the data.
            storage_options: Additional connection options for accessing storage services.
            session_options: Additional connection options for accessing S3 services.
            max_pre_download: Maximum number of chunks that can be pre-downloaded by the StreamingDataset.
            index_path: Path to `index.json` for the Parquet dataset.
                If `index_path` is a directory, the function will look for `index.json` within it.
                If `index_path` is a full file path, it will use that directly.
            force_override_state_dict: Boolean flag for allowing local arguments to override a loaded state dict.
            transform: Optional transformation function or list of functions to apply to each item in the dataset.
        """
        _check_version_and_prompt_upgrade(__version__)

        super().__init__()
        if not isinstance(shuffle, bool):
            raise ValueError(f"Shuffle should be a boolean. Found {shuffle}")

        if not isinstance(subsample, float) or subsample <= 0.0:
            raise ValueError("subsample must be a float with value greater than 0.")

        fnmatch_pattern = None
        if isinstance(input_dir, str) and input_dir.endswith(".parquet"):
            input_dir, fnmatch_pattern = os.path.split(input_dir)

        input_dir = _resolve_dir(input_dir)
        cache_dir = _resolve_dir(cache_dir)

        if input_dir.url is not None and input_dir.url.startswith("hf://"):
            if index_path is None:
                # No index_path was provided. Attempt to load it from cache or generate it dynamically on the fly.
                index_path = index_hf_dataset(input_dir.url, cache_dir.path)
            if item_loader is not None and not isinstance(item_loader, ParquetLoader):
                raise ValueError(
                    "Invalid item_loader for hf://datasets. "
                    "The item_loader must be an instance of ParquetLoader. "
                    "Please provide a valid ParquetLoader instance."
                )

            item_loader = item_loader or ParquetLoader()

        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.subsampled_files: list[str] = []
        self.region_of_interest: list[tuple[int, int]] = []
        self.subsampled_files, self.region_of_interest = subsample_streaming_dataset(
            self.input_dir,
            self.cache_dir,
            item_loader,
            subsample,
            shuffle,
            seed,
            storage_options,
            session_options,
            index_path,
            fnmatch_pattern,
        )

        self.item_loader = item_loader
        self.shuffle: bool = shuffle
        self.distributed_env = _DistributedEnv.detect()

        if self.distributed_env.world_size > 1:
            if drop_last is False:
                logger.warning(
                    "You're operating within a distributed environment and have disabled the `drop_last` option. "
                    "Please note that this configuration may lead to training interruptions if your system depends "
                    "on distributed collectives."
                )
            else:
                drop_last = True

        self.drop_last = drop_last or False

        self.seed = seed
        self.max_cache_size = max_cache_size

        max_cache_size_in_bytes = int(
            _convert_bytes_to_int(max_cache_size) if isinstance(max_cache_size, str) else max_cache_size,
        )
        min_cache_size_in_bytes = _convert_bytes_to_int("25GB")
        if max_cache_size_in_bytes < min_cache_size_in_bytes:
            logger.warning(
                "The provided `max_cache_size` is less than 25GB. "
                "This may lead to performance issues during the training process. "
                "Consider increasing the `max_cache_size` to at least 25GB to avoid potential performance degradation."
            )

        self.cache: Optional[Cache] = None
        self.worker_env: Optional[_WorkerEnv] = None
        self.worker_chunks: list[int] = []  # chunk indexes that the current worker will download, read & stream
        self.worker_intervals: list[list[int]] = []  # chunk index intervals for the current worker
        self.upcoming_indexes: list[int] = []  # contains list of upcoming indexes to be processed

        # which index of the array `self.worker_chunks` will we work on after this chunk is completely consumed
        self.worker_next_chunk_index = 0

        self.num_chunks: Optional[int] = None  # total number of chunks that the current worker will work on
        self.global_index = 0  # total number of samples processed by the current worker up until now

        # number of samples processed by the current worker in the current chunk
        self.consumed_sample_count_in_curr_chunk = 0
        self.has_triggered_download = False
        self.min_items_per_replica: Optional[int] = None
        self.current_epoch = 1
        self.random_state = None
        self.shuffler: Optional[Shuffle] = None
        self.serializers = serializers
        self._state_dict: Optional[dict[str, Any]] = None
        self._force_override_state_dict = force_override_state_dict
        # Has slightly different meaning in the context of the dataset
        # We consider `num_workers = 0` from `torch.utils.DataLoader` still as 1 worker (the main process)
        self.num_workers: int = 1
        self.batch_size: int = 1
        self._encryption = encryption
        self.storage_options = storage_options
        self.session_options = session_options
        self.max_pre_download = max_pre_download
        if transform is not None:
            transform = transform if isinstance(transform, list) else [transform]
            for t in transform:
                if not callable(t):
                    raise ValueError(f"Transform should be a callable. Found {t}")
            self.transform = transform
        self._on_demand_bytes = True  # true by default, when iterating, turn this off to store the chunks in the cache

    @property
    def on_demand_bytes(self) -> bool:
        return self._on_demand_bytes

    @on_demand_bytes.setter
    def on_demand_bytes(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"on_demand_bytes should be a boolean. Found {value}")
        self._on_demand_bytes = value
        assert self.cache is not None, "Cache must be initialized before setting on_demand_bytes."
        self.cache._reader.on_demand_bytes = value

    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle = shuffle

    def set_drop_last(self, drop_last: bool) -> None:
        self.drop_last = drop_last

    def set_epoch(self, current_epoch: int) -> None:
        """Set the current epoch to the dataset on epoch starts.

        When using the StreamingDataLoader, this is done automatically

        """
        # If the state dict has been reloaded, don't override the current epoch
        # The StreamingDataloader would clean this out
        if self._state_dict is None:
            self.current_epoch = current_epoch

    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        if _should_replace_path(self.input_dir.path):
            cache_path = _try_create_cache_dir(
                input_dir=self.input_dir.path if self.input_dir.path else self.input_dir.url,
                cache_dir=self.cache_dir.path,
            )
            if cache_path is not None:
                self.input_dir.path = cache_path

        cache = Cache(
            input_dir=self.input_dir,
            subsampled_files=self.subsampled_files,
            region_of_interest=self.region_of_interest,
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
            max_cache_size=self.max_cache_size,
            encryption=self._encryption,
            storage_options=self.storage_options,
            session_options=self.session_options,
            max_pre_download=self.max_pre_download,
            on_demand_bytes=self._on_demand_bytes,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                "\n HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache

    def _create_shuffler(self, cache: Cache) -> Shuffle:
        seed = self.seed
        drop_last = self.drop_last
        if self._state_dict is not None:
            state: dict[str, Any] = self._state_dict
            seed = state["seed"]
            drop_last = state["drop_last"]
        return FullShuffle(cache, seed, drop_last) if self.shuffle else NoShuffle(cache, seed, drop_last)

    def __len__(self) -> int:
        return self.get_len(self.num_workers, self.batch_size if self.batch_size else 1)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_num_workers(self, num_workers: int) -> None:
        self.num_workers = num_workers or 1

    def get_len(self, num_workers: int, batch_size: int) -> int:
        self.set_num_workers(num_workers)
        self.set_batch_size(batch_size)
        worker_env = _WorkerEnv.detect()
        if self.shuffler is None:
            cache = self._create_cache(worker_env=worker_env)
            self.shuffler = self._create_shuffler(cache)
        return self.shuffler.get_len(self.distributed_env, self.num_workers, self.batch_size, self.current_epoch)

    def __iter__(self) -> "StreamingDataset":
        # When the StreamingDataset is used within map or optimize, let's refetch the distributed env.
        logger.debug(_get_log_msg({"name": "iterating_dataset", "ph": "B"}))
        if os.getenv("DATA_OPTIMIZER_GLOBAL_RANK"):
            self.distributed_env = _DistributedEnv.detect()

        self.worker_env = _WorkerEnv.detect()
        self.cache = self._create_cache(worker_env=self.worker_env)
        self.shuffler = self._create_shuffler(self.cache)
        self.on_demand_bytes = False  # reset on_demand_bytes to False, and store chunks in the cache

        # Handle restart
        if self._state_dict:
            self._validate_state_dict()
            state: dict[str, Any] = self._state_dict
            self.current_epoch = state["current_epoch"]

        workers_chunks, workers_intervals = self.shuffler.get_chunks_and_intervals_per_workers(
            self.distributed_env, self.worker_env.world_size, self.batch_size, self.current_epoch
        )

        worker_rank = self.distributed_env.global_rank * self.worker_env.world_size + self.worker_env.rank
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        # The max number of samples to return from `__next__` (in worker)
        self.stop_length = sum(interval[2] - interval[1] for interval in self.worker_intervals)

        # Handle restart
        if self._state_dict:
            self._resume(workers_chunks, workers_intervals)
        else:
            # Find the chunks shared across all workers of the current node.
            # For each shared chunk, find the rank and worker to use the chunk last and prevent
            # premature deletion for the other workers.
            node_size = self.distributed_env.world_size // self.distributed_env.num_nodes
            first_rank_this_node = (self.distributed_env.global_rank // node_size) * node_size
            num_workers_per_node = node_size * self.num_workers
            worker_start = first_rank_this_node * num_workers_per_node
            worker_end = worker_start + num_workers_per_node
            local_rank = self.distributed_env.global_rank % node_size

            chunks_indexes_skip_deletion = _find_chunks_per_workers_on_which_to_skip_deletion(
                self.num_workers,
                self.batch_size,
                workers_chunks[worker_start:worker_end],
                workers_intervals[worker_start:worker_end],
            )
            worker_node_rank_to_chunk_indexes = _map_node_worker_rank_to_chunk_indexes_to_not_delete(
                chunks_indexes_skip_deletion
            )

            worker_rank_local_node = local_rank * self.num_workers + self.worker_env.rank
            if worker_rank_local_node in worker_node_rank_to_chunk_indexes:
                self.cache._reader.config.skip_chunk_indexes_deletion = worker_node_rank_to_chunk_indexes[
                    worker_rank_local_node
                ]

            self.num_chunks = len(self.worker_chunks)
            self.upcoming_indexes = []
            self.worker_next_chunk_index = 0
            self.global_index = 0
            self.consumed_sample_count_in_curr_chunk = 0

        self.has_triggered_download = False
        self.last_time = time()

        return self

    def _resume(self, workers_chunks: list[list[int]], workers_intervals: list[Any]) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.shuffler

        state: dict[str, Any] = self._state_dict

        num_workers = state["num_workers"]
        batch_size = state["batch_size"]

        # TODO: Implement elastic sampling where the number of workers, ranks can change.
        num_samples_yielded = self._state_dict["num_samples_yielded"]

        worker_start = self.distributed_env.global_rank * num_workers
        worker_end = worker_start + num_workers

        # replay sampling from each worker / chunks using the batch size
        indexes = _replay_sampling(num_samples_yielded, batch_size, num_workers)
        chunks_index, indexes = _replay_chunks_sampling(
            workers_intervals={i: workers_intervals[j] for i, j in enumerate(range(worker_start, worker_end))},
            indexes=indexes,
        )

        # select the chunks and intervals associated to this worker
        worker_rank = self.distributed_env.global_rank * self.worker_env.world_size + self.worker_env.rank
        worker_local_rank = self.worker_env.rank

        self.num_chunks = len(workers_intervals[worker_rank])
        self.worker_next_chunk_index = chunks_index[worker_local_rank]
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        if self.worker_next_chunk_index >= self.num_chunks:
            # This can happen when interrupting and resuming after some but not all workers are done.
            # Proceeding would result in an indexing error when attempting to access the next chunk.
            # To prevent this we exit early and let the worker raise a StopIteration in __next__.
            return

        # replay the indexes for the current chunks
        interval = self.worker_intervals[self.worker_next_chunk_index]
        current_indexes = np.arange(interval[1], interval[2])

        # re-shuffle the indexes
        current_indexes = self.shuffler(
            current_indexes, self.num_chunks, self.current_epoch, self.worker_next_chunk_index
        )

        # skip any indexes already consumed
        current_indexes = current_indexes[indexes[worker_local_rank] :]
        self.upcoming_indexes = current_indexes

        self.global_index = indexes[worker_local_rank]

        # bump the chunk_index
        self.worker_next_chunk_index += 1

    def __getitem__(self, index: Union[ChunkedIndex, int, slice]) -> Any:
        if self.cache is None:
            self.worker_env = _WorkerEnv.detect()
            self.cache = self._create_cache(worker_env=self.worker_env)
            self.shuffler = self._create_shuffler(self.cache)
        if isinstance(index, int):
            index = ChunkedIndex(*self.cache._get_chunk_index_from_index(index))
        elif isinstance(index, slice):
            self.on_demand_bytes = False  # for slices, we always want to store the chunks
            start, stop, step = index.indices(len(self))
            _my_indices = list(range(start, stop, step))
            _my_cache_indices = [ChunkedIndex(*self.cache._get_chunk_index_from_index(idx)) for idx in _my_indices]
            return [self.cache[chnk_idx] for chnk_idx in _my_cache_indices]
        logger.debug(
            _get_log_msg(
                {"name": f"getitem_dataset_for_chunk_index_{index.chunk_index}_and_index_{index.index}", "ph": "B"}
            )
        )
        item = self.cache[index]
        logger.debug(
            _get_log_msg(
                {"name": f"getitem_dataset_for_chunk_index_{index.chunk_index}_and_index_{index.index}", "ph": "E"}
            )
        )
        if hasattr(self, "transform"):
            if isinstance(self.transform, list):
                for transform_fn in self.transform:
                    item = transform_fn(item)
            else:
                item = self.transform(item)

        return item

    def __next__(self) -> Any:
        # check if we have reached the end of the dataset (i.e., all the chunks have been processed)
        if self.global_index >= self.stop_length:
            # global_index: total number of samples processed by the current worker across all chunks
            # stop_length: max number of samples that the current worker will process
            # if they are equal, means, worker has processed all the chunks
            self.current_epoch += 1
            self.reset_state_dict()
            logger.debug(_get_log_msg({"name": "iterating_dataset", "ph": "E"}))
            self.on_demand_bytes = True  # reset on_demand_bytes to True
            raise StopIteration

        # Lazily re-populate the interval to reduce memory usage.
        if len(self.upcoming_indexes) == 0:
            # check if it's not the end of the epoch
            if self.num_chunks is not None and self.worker_next_chunk_index >= self.num_chunks:
                self.current_epoch += 1
                self.reset_state_dict()
                self.on_demand_bytes = True  # reset on_demand_bytes to True
                raise StopIteration

            # if upcoming_indexes is empty, means either:
            #   - it's the start of a new epoch, or,
            #   - we have processed all the indexes in the current chunk
            #
            # we need to move to the next chunk (or first chunk if it's the start)
            # reset consumed_sample_count_in_curr_chunk as we are switching to a new chunk
            self.consumed_sample_count_in_curr_chunk = 0

            # `next_worker_chunks_index` is the index of the chunk that we will be working on now
            interval = self.worker_intervals[self.worker_next_chunk_index]
            current_indexes = np.arange(interval[1], interval[2])

            assert self.shuffler is not None
            assert self.num_chunks is not None
            self.upcoming_indexes = self.shuffler(
                current_indexes, self.num_chunks, self.current_epoch, self.worker_next_chunk_index
            )

            self.worker_next_chunk_index += 1  # bump the chunk_index

        # Get the first index
        index = self.upcoming_indexes.pop(0)

        # Call the `__getitem__` method.
        data = self.__getitem__(
            ChunkedIndex(
                index=index,
                chunk_index=self.worker_chunks[self.worker_next_chunk_index - 1],
                # We provide the chunks indexes only one the first
                chunk_indexes=None
                if self.has_triggered_download
                else self.worker_chunks[self.worker_next_chunk_index - 1 :],
                is_last_index=(self.worker_next_chunk_index) == self.num_chunks and len(self.upcoming_indexes) == 0,
            )
        )

        self.has_triggered_download = True
        self.global_index += 1  # total number of samples processed by the current worker
        self.consumed_sample_count_in_curr_chunk += 1  # number of samples processed in the current chunk

        return data

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> dict[str, Any]:
        if _is_in_dataloader_worker():
            raise RuntimeError("The method `state_dict` should only be called in the main process.")

        if self._state_dict is not None:
            self._state_dict["num_samples_yielded"] = num_samples_yielded
            self._state_dict["current_epoch"] = self.current_epoch
            return self._state_dict

        return {
            "num_samples_yielded": num_samples_yielded,
            "num_workers": num_workers or 1,
            "batch_size": batch_size,
            "current_epoch": self.current_epoch,
            "input_dir_path": self.input_dir.path,
            "input_dir_url": self.input_dir.url,
            "cache_dir_path": self.cache_dir.path,
            "item_loader": self.item_loader.state_dict() if self.item_loader else None,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "world_size": self.distributed_env.world_size,
            "shuffle": self.shuffle,
            "subsampled_files": self.subsampled_files,
            "region_of_interest": self.region_of_interest,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            # the state is restored within the workers
            self._state_dict = state_dict

    def reset_state_dict(self) -> None:
        self._state_dict = None

    def _validate_state_dict(self) -> None:
        if self._force_override_state_dict:
            logger.warning(
                "Using state dict override, may lead to unexpected behavior if you're not certain what you're doing."
            )

        assert self._state_dict
        assert self.worker_env
        assert self.cache

        state: dict[str, Any] = self._state_dict
        if state["shuffle"] != self.shuffle:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `shuffle` state doesn't match the current one. "
                    f"Found `{self.shuffle}` instead of `{state['shuffle']}`."
                )
            state["shuffle"] = self.shuffle
            logger.warning(
                f"Overriding state shuffle {state['shuffle']} to {self.shuffle}, "
                "this may lead to repeated or skipped datapoints within an episode."
            )

        if state["num_workers"] != self.worker_env.world_size:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `num_workers` state doesn't match the current one. "
                    f"Found `{self.worker_env.world_size}` instead of `{state['num_workers']}`."
                )
            state["num_workers"] = self.worker_env.world_size
            logger.warning(
                f"Overriding num workers {state['num_workers']} to {self.worker_env.world_size}. "
                "This may lead to repeated or skipped datapoints within an episode due to different shuffles."
            )

        # Note: We need to check whether the path has been resolved to its associated cache.
        # In this case, validate the cache folder is the same.
        if _should_replace_path(state["input_dir_path"]):
            cache_path = _try_create_cache_dir(
                input_dir=state["input_dir_path"] if state["input_dir_path"] else state["input_dir_url"],
                cache_dir=state.get("cache_dir_path"),
            )
            if cache_path != self.input_dir.path:
                if not self._force_override_state_dict:
                    raise ValueError(
                        "The provided `input_dir` path state doesn't match the current one. "
                        f"Found `{self.input_dir.path}` instead of `{cache_path}`."
                    )
                state["input_dir_path"] = self.input_dir.path
                logger.warning(
                    f"Overriding state input_dir_path {state['input_dir_path']} to {self.input_dir.path}, "
                    "this may lead to entirely different data loading."
                )

        elif state["input_dir_path"] != self.input_dir.path:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `input_dir` path state doesn't match the current one. "
                    f"Found `{self.input_dir.path}` instead of `{state['input_dir_path']}`."
                )
            state["input_dir_path"] = self.input_dir.path
            logger.warning(
                f"Overriding state input_dir_path {state['input_dir_path']} to {self.input_dir.path}, "
                "this may lead to entirely different data loading."
            )

        if state["input_dir_url"] != self.input_dir.url:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `input_dir` URL state doesn't match the current one. "
                    f"Found `{self.input_dir.url}` instead of `{state['input_dir_url']}`."
                )
            state["input_dir_url"] = self.input_dir.url
            logger.warning(
                f"Overriding state input_dir_url {state['input_dir_url']} to {self.input_dir.url}, "
                "this may lead to entirely different data loading."
            )

        if state["seed"] != self.seed:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `seed` state doesn't match the current one. "
                    f"Found `{self.seed}` instead of `{state['seed']}`."
                )
            state["seed"] = self.seed
            logger.warning(
                f"Overriding state seed {state['seed']} to {self.seed}, "
                "this may lead to repeated or skipped datapoints within an episode."
            )

        if self.item_loader and state["item_loader"] != self.item_loader.state_dict():
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `item_loader` state doesn't match the current one. "
                    f"Found `{self.item_loader.state_dict()}` instead of `{state['item_loader']}`."
                )
            logger.warning(f"Overriding state item_loader {state['item_loader']} to {self.item_loader.state_dict()}.")
            state["item_loader"] = self.item_loader.state_dict()

        if state["drop_last"] != self.drop_last:
            if not self._force_override_state_dict:
                raise ValueError(
                    "The provided `drop_last` state doesn't match the current one. "
                    f"Found `{self.drop_last}` instead of `{state['drop_last']}`."
                )
            state["drop_last"] = self.drop_last
            logger.warning(f"Overriding state drop_last {state['drop_last']} to {self.drop_last}.")

        if state["num_samples_yielded"] > len(self):
            raise ValueError(
                "The provided `num_samples_yielded` state is greater than the dataset length. "
                f"Found `{state['num_samples_yielded']}` instead of `{len(self)}`."
            )

    def reset(self) -> None:
        # undo all the properties associated with original dataset
        default_properties: dict[str, Any] = {
            "cache": None,
            "worker_env": None,
            "worker_chunks": [],
            "worker_intervals": [],
            "current_indexes": [],
            "chunk_index": 0,
            "num_chunks": None,
            "global_index": 0,
            "index": 0,
            "has_triggered_download": False,
            "min_items_per_replica": None,
            "current_epoch": 1,
            "random_state": None,
            "shuffler": None,
            "_state_dict": None,
            "num_workers": 1,
            "batch_size": 1,
        }

        for prop, value in default_properties.items():
            setattr(self, prop, value)


def is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _replay_sampling(num_samples_yielded: int, batch_size: int, num_workers: int) -> dict[int, int]:
    """This function replays the sampling from the dataloader."""
    divisible_num_batches_yielded = num_samples_yielded // (num_workers * batch_size)

    indexes = {}
    for worker_idx in range(num_workers):
        indexes[worker_idx] = divisible_num_batches_yielded * batch_size

    num_samples_yielded = num_samples_yielded - (num_workers * divisible_num_batches_yielded * batch_size)

    # take care of the reminder
    worker_idx = 0  # reset the worker_idx
    while True:
        if num_samples_yielded >= batch_size:
            indexes[worker_idx] += batch_size
            worker_idx = (worker_idx + 1) % num_workers
            num_samples_yielded -= batch_size
        else:
            indexes[worker_idx] += num_samples_yielded
            break
    return indexes


def _replay_chunks_sampling(
    workers_intervals: dict[int, list[Any]], indexes: dict[int, int]
) -> tuple[dict[int, int], dict[int, int]]:
    chunks_index = {}

    for worker_idx in range(len(workers_intervals)):
        chunks_index[worker_idx] = 0

    for worker_idx, intervals in workers_intervals.items():
        for interval in intervals:
            size = interval[2] - interval[1]
            if indexes[worker_idx] >= size:
                indexes[worker_idx] -= size
                chunks_index[worker_idx] += 1
            else:
                # We've reached the chunk where resuming needs to take place (for this worker)
                break

    return chunks_index, indexes

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

import asyncio
import inspect
import logging
import os
from copy import deepcopy
from importlib import reload
from itertools import cycle
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _DatasetKind,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)
from torch.utils.data.sampler import BatchSampler, Sampler

from litdata.constants import _DEFAULT_CHUNK_BYTES, _VIZ_TRACKER_AVAILABLE
from litdata.debugger import _get_log_msg
from litdata.streaming import Cache
from litdata.streaming.combined import CombinedStreamingDataset
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.parallel import ParallelStreamingDataset
from litdata.streaming.sampler import CacheBatchSampler
from litdata.utilities._pytree import tree_flatten
from litdata.utilities.base import (
    __NUM_CYCLES_KEY__,
    __NUM_SAMPLES_YIELDED_KEY__,
    __SAMPLES_KEY__,
    _BaseStreamingDatasetWrapper,
)
from litdata.utilities.env import _DistributedEnv

logger = logging.getLogger("litdata.streaming.dataloader")


def _equal_items(data_1: Any, data_2: Any) -> bool:
    data_1_flattened, _ = tree_flatten(data_1)
    data_2_flattened, _ = tree_flatten(data_2)

    if len(data_1_flattened) != len(data_2_flattened):
        return False

    return all(_equal_item(d1, d2) for d1, d2 in zip(data_1_flattened, data_2_flattened))


def _equal_item(d1: Any, d2: Any) -> bool:
    if not isinstance(d1, type(d2)):
        return False
    equality = d1 == d2
    if isinstance(equality, torch.Tensor):
        return bool(equality.all().item())
    return equality is True


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        cache_dir: str,
        chunk_bytes: Optional[int],
        chunk_size: Optional[int],
        compression: Optional[str],
    ):
        """The `CacheDataset` is a dataset wrapper to provide a beginner experience with the Cache.

        Args:
            dataset: The dataset of the user
            cache_dir: The folder where the chunks are written to.
            chunk_bytes: The maximal number of bytes to write within a chunk.
            chunk_size: The maximal number of items to write to a chunk.
            compression: The compression algorithm to use to reduce the size of the chunk.

        """
        self._dataset = dataset
        self._cache = Cache(cache_dir, chunk_bytes=chunk_bytes, chunk_size=chunk_size, compression=compression)
        self._is_deterministic = False

    def __len__(self) -> int:
        return len(self._cache) if self._cache.filled else len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        data_1 = self._cache[index] if self._cache.filled else self._dataset[index]
        if not self._cache.filled:
            if not self._is_deterministic:
                data2 = self._dataset[index]
                if not _equal_items(data_1, data2):
                    raise ValueError(
                        f"Your dataset items aren't deterministic. Found {data_1} and {data2} for index {index}."
                        "\n HINT: Use the `litdata.cache.Cache` directly within your dataset."
                    )
                self._is_deterministic = True
            self._cache[index] = data_1
        return data_1


class CacheCollateFn:
    """This CacheCollateFn is used to accelerate the processing of the data generated using the Cache.

    During the chunking phase, there is no need to return any data from the DataLoader reducing some time.

    Additionally, if the user makes their __getitem__ asynchronous, collate executes them in parallel.

    """

    def __init__(self, collate_fn: Optional[Callable] = None) -> None:
        self.collate_fn = collate_fn or default_collate

    def __call__(self, items: list[Any]) -> Any:
        if all(item is None for item in items):
            return None

        # If the __getitem__ method is asynchronous, collect all the items.
        if all(inspect.iscoroutine(item) for item in items):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            items = loop.run_until_complete(asyncio.gather(*items))

        return self.collate_fn([item for item in items if item is not None])


class _SingleProcessDataLoaderIterPatch(_SingleProcessDataLoaderIter):
    """This is ç to inform the cache is done chunking."""

    def _next_data(self) -> Any:
        try:
            data = None
            while data is None:
                data = super()._next_data()
            return data
        except StopIteration:
            for v in self._dataset_fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()
                    if not v.filled:
                        v.merge(1)
            raise StopIteration()


class WorkerLoop:
    """Wrap the PyTorch DataLoader WorkerLoop to perform caching and profiling."""

    def __init__(self, global_rank: int, profile: bool = False) -> None:
        self._global_rank = global_rank
        self._profile = profile

    def __call__(
        self,
        dataset_kind: Any,
        dataset: Any,
        index_queue: Any,
        data_queue: Any,
        done_event: Any,
        auto_collation: Any,
        collate_fn: Any,
        drop_last: Any,
        base_seed: Any,
        init_fn: Any,
        worker_id: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from torch.utils.data._utils import worker

        from litdata.streaming.cache import Cache

        enable_profiling = self._global_rank == 0 and worker_id == 0 and _VIZ_TRACKER_AVAILABLE and self._profile

        if enable_profiling:
            from viztracer import VizTracer

            tracer = VizTracer(output_file=os.path.join(os.getcwd(), "trace.json"))
            tracer.start()

        # Reload to remove the patching
        reloaded_worker = reload(worker)
        create_fetcher = _DatasetKind.create_fetcher
        fetcher = None

        def create_fetcher_fn(*args: Any, **kwargs: Any) -> "_BaseDatasetFetcher":
            nonlocal fetcher
            fetcher = create_fetcher(*args, **kwargs)
            return fetcher

        _DatasetKind.create_fetcher = create_fetcher_fn  # type: ignore

        reloaded_worker._worker_loop(
            dataset_kind,
            dataset,
            index_queue,
            data_queue,
            done_event,
            auto_collation,
            collate_fn,
            drop_last,
            base_seed,
            init_fn,
            worker_id,
            *args,
            **kwargs,
        )

        if dataset_kind == _DatasetKind.Map:
            assert fetcher
            for v in fetcher.dataset.__dict__.values():
                if isinstance(v, Cache):
                    v.done()

        if enable_profiling:
            tracer.stop()
            tracer.save()


class _MultiProcessingDataLoaderIterPatch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader) -> None:
        self._cache = loader._cache
        self._num_workers = loader.num_workers
        # Patch PyTorch worker loop to call the `cache.done()` method.
        from torch.utils.data._utils import worker

        worker._worker_loop = WorkerLoop(loader._global_rank, loader._profile)
        super().__init__(loader)

    def _shutdown_workers(self) -> None:
        super()._shutdown_workers()

        # If the data isn't filled, we trigger an indedm merge
        if not self._cache.filled:
            self._cache.merge(self._num_workers)

    def _next_data(self) -> Any:
        try:
            data = None
            while data is None:
                data = super()._next_data()
            return data
        except StopIteration as e:
            raise e


class CacheDataLoader(DataLoader):
    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset: Any,
        *args: Any,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        cache_dir: Optional[str] = None,
        chunk_bytes: Optional[int] = _DEFAULT_CHUNK_BYTES,
        compression: Optional[str] = None,
        profile: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        if sampler:
            raise ValueError(
                "The CacheDataLoader relies on its own internal sampler. Passing a sampler isn't supported."
            )

        if batch_sampler:
            raise ValueError(
                "The CacheDataLoader relies on its own internal sampler. Passing a batch_sampler isn't supported."
            )

        if isinstance(dataset, IterableDataset):
            raise ValueError("Only map-based dataset are supported by the CacheDataLoader for now.")

        if profile and not _VIZ_TRACKER_AVAILABLE:
            raise ModuleNotFoundError("To enable DataLoader profiling, run `pip install viztracer`.")

        cache_list = [v for v in dataset.__dict__.values() if isinstance(v, Cache)]

        if len(cache_list) > 1:
            raise ValueError(
                "We found several Cache used as attributes from your dataset. Only one is support for now."
            )

        if len(cache_list) == 0:
            if cache_dir is None:
                raise ValueError("You should provide a `cache_dir` filepath to the CacheDataLoader.")

            dataset = CacheDataset(dataset, cache_dir, chunk_bytes, batch_size, compression)
            cache = dataset._cache
        else:
            cache = cache_list[0]

        if not cache.filled and shuffle:
            logger.info("Shuffle is ignored during the caching phase phase.")

        self._cache = cache

        distributed_env = _DistributedEnv.detect()
        self._global_rank = distributed_env.global_rank

        batch_sampler = CacheBatchSampler(
            len(dataset),
            distributed_env.world_size,
            self._global_rank,
            num_workers,
            batch_size or 1,
            drop_last,
            shuffle,
            cache,
        )

        self._profile = profile

        super().__init__(
            dataset,
            *args,
            batch_sampler=batch_sampler,  # type: ignore
            collate_fn=CacheCollateFn(collate_fn),
            num_workers=num_workers,
            **kwargs,
        )

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        """Overridden to ensure the `Cache.done()` method is triggered on iteration done."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterPatch(self)
        self.check_worker_number_rationality()
        return _MultiProcessingDataLoaderIterPatch(self)


def _wrapper(fetcher: Any, func: Callable, tracer: Any, profile: int, profile_dir: str, skip_batches: int) -> Callable:
    counter = 0

    def wrap(*args: Any, **kwargs: Any) -> Any:
        nonlocal counter
        nonlocal tracer

        if skip_batches > 0 and skip_batches == counter and tracer is None:
            from viztracer import VizTracer

            output_file = os.path.join(profile_dir, "result.json")

            if os.path.exists(output_file):
                os.remove(output_file)

            tracer = VizTracer(output_file=output_file, verbose=0)
            tracer.start()

        result = func(*args, **kwargs)

        if tracer is not None and tracer.enable and counter == (profile + skip_batches):
            tracer.stop()
            tracer.save()
            print(
                f"Saved {os.path.join(profile_dir, 'result.json')} file after {profile} batches."
                "Use chrome://tracing/ to view it."
            )
            fetcher.fetch = func

        counter += 1
        return result

    return wrap


class _ProfileWorkerLoop:
    """Wrap the PyTorch DataLoader WorkerLoop to add profiling."""

    def __init__(self, profile: Union[int, bool], skip_batches: int, profile_dir: Optional[str] = None):
        self._profile = profile
        self._skip_batches = skip_batches
        self._profile_dir = profile_dir if profile_dir else os.getcwd()

    def __call__(
        self,
        dataset_kind: Any,
        dataset: Any,
        index_queue: Any,
        data_queue: Any,
        done_event: Any,
        auto_collation: Any,
        collate_fn: Any,
        drop_last: Any,
        base_seed: Any,
        init_fn: Any,
        worker_id: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from torch.utils.data._utils import worker
        from viztracer import VizTracer

        tracer = None

        if worker_id == 0 and self._skip_batches == 0:
            output_file = os.path.join(self._profile_dir, "result.json")

            if os.path.exists(output_file):
                os.remove(output_file)

            tracer = VizTracer(output_file=output_file, verbose=0)
            tracer.start()

        # Reload to remove the patching
        reloaded_worker = reload(worker)
        create_fetcher = _DatasetKind.create_fetcher
        fetcher = None

        def create_fetcher_fn(*args: Any, **kwargs: Any) -> "_BaseDatasetFetcher":
            nonlocal fetcher
            fetcher = create_fetcher(*args, **kwargs)

            if worker_id == 0 and isinstance(self._profile, int):
                fetcher.fetch = _wrapper(
                    fetcher, fetcher.fetch, tracer, self._profile, self._profile_dir, self._skip_batches
                )
            return fetcher

        _DatasetKind.create_fetcher = create_fetcher_fn  # type: ignore

        reloaded_worker._worker_loop(
            dataset_kind,
            dataset,
            index_queue,
            data_queue,
            done_event,
            auto_collation,
            collate_fn,
            drop_last,
            base_seed,
            init_fn,
            worker_id,
            *args,
            **kwargs,
        )

        if tracer is not None and worker_id == 0 and isinstance(self._profile, bool):
            tracer.stop()
            tracer.save()


class _StreamingMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._indexes = (
            list(range(self._loader._latest_worker_idx, self._loader.num_workers))
            if self._loader._latest_worker_idx > 0
            else []
        )
        self._num_workers = loader.num_workers

        distributed_env = _DistributedEnv.detect()

        if self._loader._profile_batches and distributed_env.global_rank == 0 and _VIZ_TRACKER_AVAILABLE:
            from torch.utils.data._utils import worker

            worker._worker_loop = _ProfileWorkerLoop(
                self._loader._profile_batches, self._loader._profile_skip_batches, self._loader._profile_dir
            )

        super().__init__(loader)

    def _try_put_index(self) -> None:
        # Used to restart on the right DataLoader worker
        if self._loader.restore and self._indexes:
            assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

            try:
                index = self._next_index()
            except StopIteration:
                return
            worker_queue_idx = self._indexes.pop(0)

            self._index_queues[worker_queue_idx].put((self._send_idx, index))
            self._task_info[self._send_idx] = (worker_queue_idx,)
            self._tasks_outstanding += 1
            self._send_idx += 1
        else:
            super()._try_put_index()


class StreamingDataLoaderCollateFn:
    def __init__(self, collate_fn: Optional[Callable] = None) -> None:
        self.collate_fn = collate_fn or default_collate

    def __call__(self, items: list[Any]) -> Any:
        if len(items) > 0 and isinstance(items[0], dict) and __NUM_SAMPLES_YIELDED_KEY__ in items[0]:
            batch = self.collate_fn([item[__SAMPLES_KEY__] for item in items])
            return {
                __SAMPLES_KEY__: batch,
                __NUM_SAMPLES_YIELDED_KEY__: list(torch.tensor(items[-1][__NUM_SAMPLES_YIELDED_KEY__]).reshape(-1, 1)),
                **(
                    {__NUM_CYCLES_KEY__: list(torch.tensor(items[-1][__NUM_CYCLES_KEY__]).reshape(-1, 1))}
                    if __NUM_CYCLES_KEY__ in items[0]
                    else {}
                ),
            }

        return self.collate_fn(items)


class StreamingDataLoader(DataLoader):
    r"""The StreamingDataLoader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~litdata.streaming.dataloader.StreamingDataLoader` supports either a
    StreamingDataset and CombinedStreamingDataset datasets with single- or multi-process loading,
    customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default `multiprocessing context`_ of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): the device to :attr:`pin_memory` to if ``pin_memory`` is
            ``True``.
        profile_skip_batches (int): How many batches to skip before recording
        profile_batches (int, bool, optional): Whether to record data loading profile and generate a result.json file.
        profile_dir (int, bool,  optional): Where to store the recorded trace when profile_batches is enabled.

    """

    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset: Union[StreamingDataset, _BaseStreamingDatasetWrapper],
        *args: Any,
        batch_size: int = 1,
        num_workers: int = 0,
        profile_batches: Union[bool, int] = False,
        profile_skip_batches: int = 0,
        profile_dir: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:  # pyright: ignore
        if not isinstance(dataset, (StreamingDataset, _BaseStreamingDatasetWrapper)):
            raise RuntimeError(
                "The provided dataset should be either an instance of StreamingDataset, CombinedStreamingDataset or "
                f"ParallelStreamingDataset. Found {dataset}."
            )

        if shuffle is not None:
            dataset.set_shuffle(shuffle)

        if drop_last is not None:
            dataset.set_drop_last(drop_last)

        dataset.set_batch_size(batch_size)
        dataset.set_num_workers(num_workers)

        shuffle = None

        if profile_batches and not _VIZ_TRACKER_AVAILABLE:
            raise ModuleNotFoundError("To use profile_batches, viztracer is required. Run `pip install viztracer`")

        if profile_batches and num_workers == 0:
            raise ValueError("Profiling is supported only with num_workers >= 1.")

        if collate_fn:
            collate_fn = StreamingDataLoaderCollateFn(collate_fn)

        self.current_epoch = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._profile_batches = profile_batches
        self._profile_skip_batches = profile_skip_batches
        self._profile_dir = profile_dir
        self._num_samples_yielded_streaming = 0
        self._num_samples_yielded_wrapper: dict[int, list[int]] = {}
        self._num_cycles: dict[int, list[int]] = {}
        self.rng_state: Optional[Any] = None
        self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
        self._worker_idx_iter: Optional[Any] = None
        self._latest_worker_idx = 0
        self.restore = False
        super().__init__(
            dataset,
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=(2 if num_workers > 0 else None) if prefetch_factor is None else prefetch_factor,
            collate_fn=collate_fn,
            **kwargs,
        )  # type: ignore

    def __iter__(self) -> Any:
        if not self.restore:
            if (
                isinstance(self.dataset, ParallelStreamingDataset)
                and self.dataset.is_cycling()
                and self.dataset.resume
                and self.current_epoch != 0
            ):
                # For ParallelStreamingDataset with _length != None we want to cycle the wrapped datasets i.e. we do not
                # want to restart at index 0 at every epoch. So we set them in restore state.
                self.load_state_dict(self.state_dict())
            else:
                self._latest_worker_idx = 0
                self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
                self._worker_idx_iter = iter(self._worker_idx)
                self._num_samples_yielded_wrapper = {}
                self._num_samples_yielded_streaming = 0
                self._num_cycles = {}
                self.dataset.reset_state_dict()
            self.current_epoch += 1

        self.dataset.set_epoch(self.current_epoch)
        logger.debug(_get_log_msg({"name": "iterating_dataloader", "ph": "B"}))

        if isinstance(self.dataset, StreamingDataset):
            assert self.batch_size
            for batch in super().__iter__():
                self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
                self._num_samples_yielded_streaming += self.batch_size
                yield batch
        else:
            self.dataset._set_use_streaming_dataloader(True)
            assert self.batch_size
            # TODO: Inject a custom collate function to avoid collating the __NUM_SAMPLES_YIELDED__ key
            for batch in super().__iter__():
                self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
                if isinstance(batch, dict) and __NUM_SAMPLES_YIELDED_KEY__ in batch:
                    self._num_samples_yielded_wrapper[self._latest_worker_idx] = [
                        sample[-1].item() if self.batch_size > 1 else sample.item()
                        for sample in batch[__NUM_SAMPLES_YIELDED_KEY__]
                    ]

                    if __NUM_CYCLES_KEY__ in batch:
                        self._num_cycles[self._latest_worker_idx] = [
                            cycle[-1].item() if self.batch_size > 1 else cycle.item()
                            for cycle in batch[__NUM_CYCLES_KEY__]
                        ]
                        self.dataset.update_epoch_counters(self._num_cycles[self._latest_worker_idx])

                    yield batch[__SAMPLES_KEY__]
                else:
                    yield batch

        logger.debug(_get_log_msg({"name": "iterating_dataloader", "ph": "E"}))
        self.restore = False

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = self.dataset.get_len(self.num_workers, self.batch_size)
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil

                return length // self.batch_size if self.drop_last else ceil(length / self.batch_size)
            return length
        return len(self._index_sampler)

    def state_dict(self) -> dict[str, Any]:
        if isinstance(self.dataset, StreamingDataset):
            assert self.batch_size
            return {
                "dataset": self.dataset.state_dict(
                    self._num_samples_yielded_streaming, self.num_workers, self.batch_size
                ),
                "current_epoch": self.current_epoch,
                "num_samples_yielded": self._num_samples_yielded_streaming,
                "latest_worker_idx": self._latest_worker_idx,
            }

        if isinstance(self.dataset, CombinedStreamingDataset):
            # initialize a list to track the number of samples yielded for each dataset
            num_samples_yielded = [0 for _ in range(len(self.dataset._datasets))]

            for worker_idx in self._num_samples_yielded_wrapper:
                for dataset_idx, samples_yielded in enumerate(self._num_samples_yielded_wrapper[worker_idx]):
                    num_samples_yielded[dataset_idx] += samples_yielded

        elif isinstance(self.dataset, ParallelStreamingDataset):
            num_samples_yielded, _ = self.dataset.get_num_samples_yielded(
                self._num_samples_yielded_wrapper, self._num_cycles
            )

        else:
            raise RuntimeError(
                "The provided dataset should be a `StreamingDataset`, a `CombinedStreamingDataset` or a "
                "`ParallelStreamingDataset`."
            )

        assert self.batch_size
        return {
            "dataset": self.dataset.state_dict(self.num_workers, self.batch_size, num_samples_yielded),
            "current_epoch": self.current_epoch,
            "latest_worker_idx": self._latest_worker_idx,
            "num_samples_yielded": deepcopy(self._num_samples_yielded_wrapper),
            **(
                {"num_cycles": deepcopy(self._num_cycles)} if isinstance(self.dataset, ParallelStreamingDataset) else {}
            ),
        }

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Load a dict containing training state (called from non-worker process).

        This is called on each copy of the dataset when resuming.

        Args:
            obj (Any): The state.

        """
        self.current_epoch = obj["current_epoch"]

        if isinstance(self.dataset, StreamingDataset):
            self._num_samples_yielded_streaming = obj["num_samples_yielded"]
        else:
            self._num_samples_yielded_wrapper = obj["num_samples_yielded"]

        if isinstance(self.dataset, ParallelStreamingDataset):
            self._num_cycles = obj["num_cycles"]

        # Used to restart on the next DataLoader worker from the previous run.
        self._latest_worker_idx = obj["latest_worker_idx"] + 1
        self._worker_idx_iter = iter(self._worker_idx)
        for _ in range(self._latest_worker_idx):
            next(self._worker_idx_iter)

        # Inform we are resuming and disable resetting the StreamingDataLoader state.
        # This is toggle back to False when the `__iter__` method of the StreamingDataLoader completes.
        # self.restore = True

        if isinstance(self.dataset, CombinedStreamingDataset):
            self.dataset._set_use_streaming_dataloader(True)
            self.dataset.load_state_dict(obj)

            total_samples_yielded = sum([sum(samples) for samples in self._num_samples_yielded_wrapper.values()])

            # Check if we need to restore for the case without weights.
            if (
                self.dataset._iterate_over_all
                and total_samples_yielded > 0
                and total_samples_yielded < len(self.dataset)  # type: ignore
            ):
                self.restore = True

            # Check if we need to restore for the case with weights.
            # Note: `len` is not available for CombinedStreamingDataset in case of provided weights.
            # TODO: handle the case with weights.
            if not self.dataset._iterate_over_all:
                self.restore = True

        elif isinstance(self.dataset, ParallelStreamingDataset):
            self.dataset._set_use_streaming_dataloader(True)
            self.dataset.load_state_dict(obj)
            num_samples_yielded, num_cycles = self.dataset.get_num_samples_yielded(
                self._num_samples_yielded_wrapper, self._num_cycles
            )
            if self.dataset.is_infinite():
                if any(samples > 0 for samples in num_samples_yielded):
                    self.restore = True
            else:
                dset_len = len(self.dataset)  # type: ignore
                if self.dataset.is_cycling():
                    dset_lens = self.dataset.get_all_lens()
                    for samples, cycles, length in zip(num_samples_yielded, num_cycles, dset_lens):
                        # infer where we are in the current epoch from the number of times we cycled, the dataset length
                        # i.e. cycle length, the number of samples yielded in the current cycle, and the epoch length
                        if 0 < (cycles * length + samples) % dset_len < dset_len:
                            self.restore = True
                            break
                elif any(0 < samples < dset_len for samples in num_samples_yielded):
                    self.restore = True

        elif isinstance(self.dataset, StreamingDataset):
            self.dataset.load_state_dict(obj["dataset"])

            # Inform that the dataloader is resuming.
            if self._num_samples_yielded_streaming > 0 and self._num_samples_yielded_streaming < len(self.dataset):
                self.restore = True
        else:
            raise RuntimeError(
                "The provided dataset should be a `StreamingDataset`, a `CombinedStreamingDataset` or a "
                "`ParallelStreamingDataset`."
            )

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        """Overridden to ensure the `Cache.done()` method is triggered on iteration done."""
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        self.check_worker_number_rationality()
        return _StreamingMultiProcessingDataLoaderIter(self)

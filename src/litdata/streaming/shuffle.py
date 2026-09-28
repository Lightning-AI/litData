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

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from litdata.streaming import Cache
from litdata.utilities.env import _DistributedEnv
from litdata.utilities.shuffle import (
    _associate_chunks_and_intervals_to_workers,
    _associate_whole_chunks_to_workers,
    _associate_within_nodes,
    _block_shuffle,
    _permute_node_chunk_indexes,
    _unique_chunk_indexes_per_node,
    _window_shuffle,
    _window_shuffle_chunks_and_intervals,
    node_shard_fits_in_cache,
)

_DEFAULT_POSIX_SHUFFLE_WINDOW = 16
# Match ``PyTreeLoader`` text ``batch_decode`` so shuffled training reuses a window.
_DEFAULT_ITEM_SHUFFLE_WINDOW = 256


def posix_shuffle_window() -> int:
    """Chunks each POSIX worker may mix among (sequential assignment, then local permute)."""
    raw = os.getenv("LITDATA_POSIX_SHUFFLE_WINDOW")
    if raw is None or not raw.strip():
        return _DEFAULT_POSIX_SHUFFLE_WINDOW
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_POSIX_SHUFFLE_WINDOW


def item_shuffle_window() -> int:
    """Env fallback for ``StreamingDataset(item_shuffle_window=...)``.

    ``LITDATA_ITEM_SHUFFLE_WINDOW=0`` / ``full`` restores a full in-chunk permutation.
    """
    raw = os.getenv("LITDATA_ITEM_SHUFFLE_WINDOW")
    if raw is None or not raw.strip():
        return _DEFAULT_ITEM_SHUFFLE_WINDOW
    key = raw.strip().lower()
    if key in {"auto", ""}:
        return _DEFAULT_ITEM_SHUFFLE_WINDOW
    return resolve_item_shuffle_window(key)


def resolve_item_shuffle_window(value: int | str | bool | None) -> int:
    """``None`` / ``"auto"`` → env or 256. ``0`` / ``"full"`` → full in-chunk permute."""
    if value is None:
        return item_shuffle_window()
    if isinstance(value, bool):
        return _DEFAULT_ITEM_SHUFFLE_WINDOW if value else 0
    if isinstance(value, int):
        return max(0, value)
    key = str(value).strip().lower()
    if key in {"auto", ""}:
        return item_shuffle_window()
    if key in {"0", "off", "full", "all"}:
        return 0
    try:
        return max(0, int(key))
    except ValueError:
        return _DEFAULT_ITEM_SHUFFLE_WINDOW


class Shuffle(ABC):
    """Shuffle describe how to distribute chunked datasets across processes and workers."""

    def __init__(self, cache: Cache, seed: int, drop_last: bool):
        self.cache = cache
        self.seed = seed
        self.drop_last = drop_last

    @lru_cache(maxsize=10)
    def get_len(self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int) -> int:
        _, workers_intervals = self.get_chunks_and_intervals_per_workers(
            distributed_env, num_workers, batch_size, current_epoch
        )
        worker_start = distributed_env.global_rank * num_workers
        worker_end = worker_start + num_workers
        return sum(
            (interval[2] - interval[1])
            for intervals in workers_intervals[worker_start:worker_end]
            for interval in intervals
        )

    @abstractmethod
    def get_chunks_and_intervals_per_workers(
        self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int
    ) -> Any:
        pass

    @abstractmethod
    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> list[int]:
        pass


class NoShuffle(Shuffle):
    """NoShuffle doesn't shuffle the items and ensure all the processes receive the same number of items if drop_last
    is True.
    """

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_workers(
        self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int
    ) -> Any:
        # 1. Get the intervals
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = range(len(chunk_intervals))

        # 2. Compute the items budget of each rank
        workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
            distributed_env, indexes, chunk_intervals, self.drop_last, num_workers, batch_size
        )
        return workers_chunks, workers_intervals

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> list[int]:
        return array.tolist()


class FullShuffle(Shuffle):
    """FullShuffle shuffles the chunks and associates them to the ranks.

    As the number of items in a chunk varies, it is possible for a rank to end up with more or less items.
    To ensure the same fixed dataset length for all ranks while dropping as few items as possible,
    we adopt the following strategy.

    We compute the maximum number of items per rank (M) and iterate through the chunks and ranks
    until we have associated at least M items per rank.

    As a result, we lose at most (number of ranks) items. However, as some chunks are shared across ranks. This leads to
    the same chunk to be downloaded multiple times.

    Multi-node: epoch 1 is a global permute. Later epochs stay inside each node's unique chunk
    set when that shard fits in ``max_cache_size``; otherwise chunks are re-scheduled globally.

    """

    def __init__(self, cache: Cache, seed: int, drop_last: bool, item_window: int | None = None):
        super().__init__(cache, seed, drop_last)
        self.node_shard_fits: bool = True
        self.item_window = resolve_item_shuffle_window(item_window)

    def _global_assign(
        self,
        distributed_env: _DistributedEnv,
        chunk_intervals: Any,
        num_workers: int,
        batch_size: int,
        seed_shift: int,
    ) -> tuple[Any, Any]:
        indexes = range(len(chunk_intervals))
        shuffled_indexes = np.random.RandomState([self.seed, seed_shift]).permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes].tolist()
        return _associate_chunks_and_intervals_to_workers(
            distributed_env, shuffled_indexes, shuffled_chunk_intervals, self.drop_last, num_workers, batch_size
        )

    def _chunk_byte_sizes(self) -> list[int]:
        config = getattr(self.cache._reader, "_config", None)
        if config is None:
            try_load = getattr(self.cache._reader, "_try_load_config", None)
            if callable(try_load):
                try_load()
            config = getattr(self.cache._reader, "_config", None)
        chunks = getattr(config, "_chunks", None) if config is not None else None
        if not chunks:
            return [0] * len(self.cache.get_chunk_intervals())
        return [int(chunk["chunk_bytes"]) for chunk in chunks]

    def _max_cache_size(self) -> int:
        return int(getattr(self.cache._reader, "_max_cache_size", 0) or 0)

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_workers(
        self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int
    ) -> Any:
        chunk_intervals = self.cache.get_chunk_intervals()
        sizes = self._chunk_byte_sizes()
        max_cache_size = self._max_cache_size()

        if distributed_env.num_nodes == 1:
            self.node_shard_fits = node_shard_fits_in_cache(sum(sizes), max_cache_size)
            return self._global_assign(distributed_env, chunk_intervals, num_workers, batch_size, current_epoch)

        workers_chunks, workers_intervals = self._global_assign(
            distributed_env, chunk_intervals, num_workers, batch_size, seed_shift=1
        )
        unique_per_node = _unique_chunk_indexes_per_node(workers_chunks, distributed_env, num_workers)
        node_bytes = [sum(sizes[int(i)] for i in ids) for ids in unique_per_node]
        self.node_shard_fits = node_shard_fits_in_cache(max(node_bytes, default=0), max_cache_size)

        if current_epoch == 1:
            return workers_chunks, workers_intervals

        if not self.node_shard_fits:
            return self._global_assign(distributed_env, chunk_intervals, num_workers, batch_size, current_epoch)

        permuted = _permute_node_chunk_indexes(unique_per_node, self.seed, current_epoch)
        return _associate_within_nodes(
            distributed_env, permuted, chunk_intervals, self.drop_last, num_workers, batch_size
        )

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> list[int]:
        rng = np.random.RandomState([self.seed, num_chunks, current_epoch, chunk_index])
        # Aligned blocks pair with ``batch_decode``; ``window <= 1`` is a full permute.
        return _block_shuffle(array.tolist(), self.item_window, rng)


class WindowShuffle(Shuffle):
    """POSIX / Vast shuffle: sequential chunk stripes, then a sliding-window permute per worker.

    ``FullShuffle`` globally permutes chunks before assignment. That is right for object storage
    (random GETs, cache copies). On a parallel filesystem it turns sequential 64MiB files into
    random IOPS and fights ``posix_fadvise`` / page cache.

    ``WindowShuffle`` assigns **whole chunks** (no split across workers) in sequential
    stripes, then window-shuffles each worker's list (default window 16). Item order
    *inside* a chunk uses the same window so the loader can view one contiguous mmap
    span and split samples from it.
    """

    def __init__(
        self,
        cache: Cache,
        seed: int,
        drop_last: bool,
        window: int | None = None,
        item_window: int | None = None,
    ):
        super().__init__(cache, seed, drop_last)
        self.window = posix_shuffle_window() if window is None else max(1, window)
        # None keeps the POSIX sliding item window. Dataset passes a resolved int.
        self.item_window = None if item_window is None else resolve_item_shuffle_window(item_window)

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_workers(
        self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int
    ) -> Any:
        chunk_intervals = self.cache.get_chunk_intervals()
        indexes = range(len(chunk_intervals))
        workers_chunks, workers_intervals = _associate_whole_chunks_to_workers(
            distributed_env, indexes, chunk_intervals, self.drop_last, num_workers, batch_size
        )
        return _window_shuffle_chunks_and_intervals(
            workers_chunks, workers_intervals, self.seed, current_epoch, self.window
        )

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> list[int]:
        rng = np.random.RandomState([self.seed, num_chunks, current_epoch, chunk_index])
        if self.item_window is not None:
            return _block_shuffle(array.tolist(), self.item_window, rng)
        return _window_shuffle(array.tolist(), self.window, rng)

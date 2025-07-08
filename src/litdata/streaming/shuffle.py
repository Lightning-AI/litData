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

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from litdata.streaming import Cache
from litdata.utilities.env import _DistributedEnv
from litdata.utilities.shuffle import (
    _associate_chunks_and_intervals_to_workers,
    _intra_node_chunk_shuffle,
)


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

    """

    @lru_cache(maxsize=10)
    def get_chunks_and_intervals_per_workers(
        self, distributed_env: _DistributedEnv, num_workers: int, batch_size: int, current_epoch: int
    ) -> Any:
        # 1. Get the intervals
        chunk_intervals = self.cache.get_chunk_intervals()

        # 2. Shuffle them
        indexes = range(len(chunk_intervals))

        # If we have multiple nodes, the seed_shift is constant here.
        # Here is why. When you are running epoch 1, we need to shuffle the chunks
        # and associate to each rank. This is done there.
        # When you are running epoch 2 or more, we need to keep the same shuffling
        # than in epoch 1 because shuffle a second time within the node.
        # This is done slighyly down this function.
        seed_shift = 1 if distributed_env.num_nodes > 1 else current_epoch
        shuffled_indexes = np.random.RandomState([self.seed, seed_shift]).permutation(indexes)
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes].tolist()

        # 3. Compute the items budget of each rank
        workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
            distributed_env, shuffled_indexes, shuffled_chunk_intervals, self.drop_last, num_workers, batch_size
        )

        # For the first epoch, no need of further shuffling
        if current_epoch == 1 or distributed_env.num_nodes == 1:
            return workers_chunks, workers_intervals

        # Perform shuffle within the nodes to avoid cache miss.
        # Note: It is possible for the overlapping chunks to change due to the changing order.
        shuffled_indexes = _intra_node_chunk_shuffle(
            distributed_env, num_workers, workers_chunks, self.seed, current_epoch
        )
        shuffled_chunk_intervals = np.asarray(chunk_intervals)[shuffled_indexes].tolist()

        workers_chunks, workers_intervals = _associate_chunks_and_intervals_to_workers(
            distributed_env, shuffled_indexes, shuffled_chunk_intervals, self.drop_last, num_workers, batch_size
        )

        return workers_chunks, workers_intervals

    def __call__(self, array: np.ndarray, num_chunks: int, current_epoch: int, chunk_index: int) -> list[int]:
        return np.random.RandomState([self.seed, num_chunks, current_epoch, chunk_index]).permutation(array).tolist()

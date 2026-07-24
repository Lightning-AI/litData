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

from typing import Any

import numpy as np

from litdata.streaming.item_loader import Interval
from litdata.utilities.env import _DistributedEnv


def _intra_node_chunk_shuffle(
    distributed_env: _DistributedEnv,
    num_workers: int,
    chunks_per_workers: list[list[int]],
    seed: int,
    current_epoch: int,
) -> list[int]:
    chunk_indexes_per_nodes = _group_chunks_by_nodes(
        chunks_per_workers=chunks_per_workers,
        world_size=distributed_env.world_size,
        num_nodes=distributed_env.num_nodes,
        num_workers_per_process=num_workers,
    )

    # shuffle the chunks associated to the node
    for i in range(len(chunk_indexes_per_nodes)):
        # permute the indexes within the node
        chunk_indexes_per_nodes[i] = list(
            np.random.RandomState([seed, current_epoch]).permutation(chunk_indexes_per_nodes[i])
        )

    return [index for chunks in chunk_indexes_per_nodes for index in chunks]


def _group_chunks_by_nodes(
    chunks_per_workers: list[list[int]],
    world_size: int,
    num_nodes: int,
    num_workers_per_process: int,
) -> list[list[int]]:
    """Takes a list representing chunks grouped by worker (global worker id across ranks and nodes) and returns a list
    in which the chunks are grouped by node.
    """
    chunk_indexes_per_nodes: Any = [[] for _ in range(num_nodes)]
    num_processes_per_node = world_size // num_nodes
    for worker_global_id, chunks in enumerate(chunks_per_workers):
        process_rank = worker_global_id // num_workers_per_process  # the process rank this worker belongs to
        node_rank = process_rank // num_processes_per_node  # the node rank this worker belongs to
        chunk_indexes_per_nodes[node_rank].extend(chunks)
    return chunk_indexes_per_nodes


def _associate_chunks_and_intervals_to_workers(
    distributed_env: _DistributedEnv,
    indexes: Any,
    chunk_intervals: list[Interval],
    drop_last: bool = False,
    num_workers: int = 1,
    batch_size: int = 1,
) -> tuple[list[list[int]], list[Any]]:
    num_items = sum([(interval[2] - interval[1]) for interval in chunk_intervals])
    max_batches = num_items // batch_size
    global_num_workers = distributed_env.world_size * num_workers

    num_items_per_workers: Any = []

    for rank in range(distributed_env.world_size):
        tmp_arr = [0 for _ in range(num_workers)]

        num_batches_per_rank = int(max_batches // distributed_env.world_size)
        base_batches = num_batches_per_rank // num_workers
        rem_batches = num_batches_per_rank % num_workers
        tmp_arr = [base_batches + (1 if i < rem_batches else 0) for i in range(num_workers)]

        if rank == distributed_env.world_size - 1:
            # Find how batches were associated
            num_assigned_items = batch_size * (sum(num_items_per_workers) + sum(tmp_arr))

            # Multiply with the batch_size to get the number of items
            if batch_size > 1:
                tmp_arr = [x * batch_size for x in tmp_arr]
                num_items_per_workers = [x * batch_size for x in num_items_per_workers]

            # If there are items left to assign, let's give it the last worker
            left_items = num_items - num_assigned_items
            if not drop_last and left_items > 0:
                tmp_arr[rem_batches % num_workers] += left_items

            num_items_per_workers.extend(tmp_arr)
        else:
            num_items_per_workers.extend(tmp_arr)

    chunks_per_workers: list[list[int]] = [[] for _ in range(global_num_workers)]
    intervals_per_workers: list[list[list[int]]] = [[] for _ in range(global_num_workers)]

    # 4. Assign the chunk & intervals to each rank
    for chunk_index, chunk_interval in zip(indexes, chunk_intervals):
        rank = 0

        while True:
            if rank == len(num_items_per_workers):
                break

            items_left_to_assign = num_items_per_workers[rank]

            if items_left_to_assign == 0:
                rank += 1
                continue

            items_in_chunk = chunk_interval[2] - chunk_interval[1]

            if items_in_chunk == 0:
                break

            if items_in_chunk > items_left_to_assign:
                chunks_per_workers[rank].append(chunk_index)

                chunk_start, chunk_roi_start, chunk_roi_end, chunk_end = chunk_interval

                intervals_per_workers[rank].append(
                    [chunk_start, chunk_roi_start, chunk_roi_start + items_left_to_assign, chunk_end]
                )
                chunk_interval = Interval(chunk_start, chunk_roi_start + items_left_to_assign, chunk_roi_end, chunk_end)
                num_items_per_workers[rank] = 0
                rank += 1
            else:
                chunks_per_workers[rank].append(chunk_index)
                intervals_per_workers[rank].append(list(chunk_interval))
                num_items_per_workers[rank] -= items_in_chunk
                break

    return chunks_per_workers, intervals_per_workers


def _get_shared_chunks(workers_chunks: list[list[int]]) -> dict[int, list[int]]:
    """Returns a dictionary mapping a chunk index to a list of workers that share that same chunk."""
    shared_chunks = {}
    for worker, chunks in enumerate(workers_chunks):
        for chunk in chunks:
            if chunk not in shared_chunks:
                shared_chunks[chunk] = [worker]
            else:
                shared_chunks[chunk].append(worker)
    # Remove chunk indexes that are only read by a single worker (and thus not shared)
    return {chunk: workers for chunk, workers in shared_chunks.items() if len(workers) > 1}

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

import copy
from typing import Any

import numpy as np

from litdata.streaming.item_loader import Interval
from litdata.utilities.env import _DistributedEnv


def _window_shuffle(items: list[Any], window: int, rng: np.random.RandomState) -> list[Any]:
    """Permute ``items`` so each swap stays inside a sliding window (FFCV-style locality).

    Windowed Fisher–Yates: at index ``i`` the partner is drawn from ``[i, min(i + window, n))``.
    ``window <= 1`` is the identity. A large window approaches a full permutation of this list.
    """
    n = len(items)
    if n < 2 or window <= 1:
        return list(items)
    out = list(items)
    for i in range(n - 1):
        high = min(i + window, n)
        j = int(rng.randint(i, high))
        out[i], out[j] = out[j], out[i]
    return out


def _block_shuffle(items: list[Any], window: int, rng: np.random.RandomState) -> list[Any]:
    """Shuffle aligned ``[kW, (k+1)W)`` blocks, then shuffle items inside each.

    Block order is random (not 0, 1, 2, …). Inside a block the items are
    shuffled too. Sequential consume of one block still hits one
    ``batch_decode`` window. ``window <= 1`` is a full permutation.
    """
    n = len(items)
    out = list(items)
    if n < 2:
        return out
    if window <= 1:
        return rng.permutation(out).tolist()
    blocks: list[list[Any]] = []
    for start in range(0, n, window):
        block = out[start : start + window]
        rng.shuffle(block)
        blocks.append(block)
    if len(blocks) > 1:
        rng.shuffle(blocks)
    flat: list[Any] = []
    for block in blocks:
        flat.extend(block)
    return flat


def _window_shuffle_chunks_and_intervals(
    workers_chunks: list[list[int]],
    workers_intervals: list[Any],
    seed: int,
    current_epoch: int,
    window: int,
) -> tuple[list[list[int]], list[Any]]:
    """Shuffle each worker's chunk list in place-local order; keep intervals aligned."""
    shuffled_chunks: list[list[int]] = []
    shuffled_intervals: list[Any] = []
    for worker_idx, (chunks, intervals) in enumerate(zip(workers_chunks, workers_intervals)):
        rng = np.random.RandomState([seed, current_epoch, worker_idx])
        paired = _window_shuffle(list(zip(chunks, intervals)), window, rng)
        if paired:
            new_chunks, new_intervals = zip(*paired)
            shuffled_chunks.append(list(new_chunks))
            shuffled_intervals.append(list(new_intervals))
        else:
            shuffled_chunks.append([])
            shuffled_intervals.append([])
    return shuffled_chunks, shuffled_intervals


def _trim_worker_to_item_count(
    chunks: list[int],
    intervals: list[Any],
    cap: int,
) -> tuple[list[int], list[Any]]:
    """Keep a prefix of ``intervals`` totaling ``cap`` items (shorten the last interval, never share)."""
    if cap <= 0:
        return [], []
    out_chunks: list[int] = []
    out_intervals: list[Any] = []
    remaining = cap
    for chunk_index, interval in zip(chunks, intervals):
        start, roi_start, roi_end, end = interval
        size = roi_end - roi_start
        if size <= 0:
            continue
        take = min(size, remaining)
        out_chunks.append(int(chunk_index))
        out_intervals.append([start, roi_start, roi_start + take, end])
        remaining -= take
        if remaining <= 0:
            break
    return out_chunks, out_intervals


def _associate_whole_chunks_to_workers(
    distributed_env: _DistributedEnv,
    indexes: Any,
    chunk_intervals: list[Interval],
    drop_last: bool = False,
    num_workers: int = 1,
    batch_size: int = 1,
) -> tuple[list[list[int]], list[Any]]:
    """Assign each chunk to exactly one worker as contiguous stripes (no shared mmap)."""
    indexes = [int(i) for i in indexes]
    global_n = distributed_env.world_size * num_workers
    chunks_per_workers: list[list[int]] = [[] for _ in range(global_n)]
    intervals_per_workers: list[list[Any]] = [[] for _ in range(global_n)]
    if not indexes or global_n == 0:
        return chunks_per_workers, intervals_per_workers

    sizes = [int(interval[2] - interval[1]) for interval in chunk_intervals]
    total = sum(sizes)
    worker = 0
    prefix = 0
    for chunk_index, interval, size in zip(indexes, chunk_intervals, sizes):
        while (
            worker < global_n - 1
            and chunks_per_workers[worker]
            and prefix + size / 2.0 > (worker + 1) * total / global_n
        ):
            worker += 1
        chunks_per_workers[worker].append(chunk_index)
        intervals_per_workers[worker].append(list(interval))
        prefix += size

    if drop_last:
        lengths = [sum(iv[2] - iv[1] for iv in ivs) for ivs in intervals_per_workers]
        positive = [length for length in lengths if length > 0]
        if positive:
            cap = min(positive)
            if batch_size > 1:
                cap = (cap // batch_size) * batch_size
            trimmed_chunks: list[list[int]] = []
            trimmed_intervals: list[list[Any]] = []
            for chunks, intervals in zip(chunks_per_workers, intervals_per_workers):
                new_chunks, new_intervals = _trim_worker_to_item_count(chunks, intervals, cap)
                trimmed_chunks.append(new_chunks)
                trimmed_intervals.append(new_intervals)
            return trimmed_chunks, trimmed_intervals

    return chunks_per_workers, intervals_per_workers


def node_shard_fits_in_cache(shard_bytes: int, max_cache_size: int | None) -> bool:
    """True when a node's unique chunk files can stay resident under ``max_cache_size``."""
    if not max_cache_size:
        return True
    return shard_bytes <= max_cache_size


def _unique_chunk_indexes_per_node(
    chunks_per_workers: list[list[int]],
    distributed_env: _DistributedEnv,
    num_workers: int,
) -> list[list[int]]:
    grouped = _group_chunks_by_nodes(
        chunks_per_workers=chunks_per_workers,
        world_size=distributed_env.world_size,
        num_nodes=distributed_env.num_nodes,
        num_workers_per_process=num_workers,
    )
    return [list(dict.fromkeys(chunk_indexes)) for chunk_indexes in grouped]


def _permute_node_chunk_indexes(
    chunk_indexes_per_nodes: list[list[int]],
    seed: int,
    current_epoch: int,
) -> list[list[int]]:
    permuted: list[list[int]] = []
    for node_idx, chunk_indexes in enumerate(chunk_indexes_per_nodes):
        if not chunk_indexes:
            permuted.append([])
            continue
        rng = np.random.RandomState([seed, current_epoch, node_idx])
        permuted.append([int(i) for i in rng.permutation(chunk_indexes)])
    return permuted


def _associate_within_nodes(
    distributed_env: _DistributedEnv,
    chunk_indexes_per_nodes: list[list[int]],
    chunk_intervals: list[Interval],
    drop_last: bool,
    num_workers: int,
    batch_size: int,
) -> tuple[list[list[int]], list[Any]]:
    """Associate each node's chunk ids only among that node's ranks (no cross-node move)."""
    num_nodes = max(1, distributed_env.num_nodes)
    ranks_per_node = distributed_env.world_size // num_nodes
    global_n = distributed_env.world_size * num_workers
    workers_chunks: list[list[int]] = [[] for _ in range(global_n)]
    workers_intervals: list[Any] = [[] for _ in range(global_n)]
    local_env = _DistributedEnv(ranks_per_node, 0, 1)
    for node_idx, node_chunk_ids in enumerate(chunk_indexes_per_nodes):
        node_intervals = [chunk_intervals[int(i)] for i in node_chunk_ids]
        local_chunks, local_intervals = _associate_chunks_and_intervals_to_workers(
            local_env, node_chunk_ids, node_intervals, drop_last, num_workers, batch_size
        )
        base = node_idx * ranks_per_node * num_workers
        for local_w, (chunks, intervals) in enumerate(zip(local_chunks, local_intervals)):
            workers_chunks[base + local_w] = list(chunks)
            workers_intervals[base + local_w] = list(intervals)
    return workers_chunks, workers_intervals


def _intra_node_chunk_shuffle(
    distributed_env: _DistributedEnv,
    num_workers: int,
    chunks_per_workers: list[list[int]],
    seed: int,
    current_epoch: int,
) -> list[int]:
    unique_per_node = _unique_chunk_indexes_per_node(chunks_per_workers, distributed_env, num_workers)
    permuted = _permute_node_chunk_indexes(unique_per_node, seed, current_epoch)
    shuffled = [index for chunks in permuted for index in chunks]
    return list(dict.fromkeys(shuffled))


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


def _find_chunks_per_workers_on_which_to_skip_deletion(
    num_workers: int,
    batch_size: int,
    workers_chunks: list[list[int]],
    workers_intervals: list[list[Interval]],
) -> dict[int, list[int]]:
    """Returns a dictionary mapping a chunk index to a list of workers that should not delete that chunk.

    If a worker is included in this list, it should not delete the chunk after fully reading it, because another worker
    will still have items left to read and therefore needs the chunk to be present. This mapping is used in the dataset
    to only let the worker delete a chunk when that worker is the last to read from it.

    """
    # Shared chunks across all workers and ranks
    shared_chunks = _get_shared_chunks(workers_chunks)

    # Shared chunks grouped together by rank
    shared_chunks_aggregated_by_rank = _aggregate_shared_chunks_per_rank(shared_chunks, num_workers)

    max_trackers = {}
    for chunk_index, map_local_rank_to_worker_ids in shared_chunks_aggregated_by_rank.items():
        for local_rank, workers_index_sharing_chunks_for_this_rank in map_local_rank_to_worker_ids.items():
            # Get all the worker chunks and intervals for this distributed rank
            workers_slice = slice(local_rank * num_workers, (local_rank + 1) * num_workers)
            workers_chunks_for_this_rank = copy.deepcopy(workers_chunks[workers_slice])
            workers_interval_sizes_for_this_rank = copy.deepcopy(
                [
                    [interval[2] - interval[1] for interval in worker_intervals]
                    for worker_intervals in workers_intervals[workers_slice]
                ]
            )

            num_shared_workers_for_this_rank = len(workers_index_sharing_chunks_for_this_rank)
            worker_tracker_idx = 0
            num_of_samples_to_carry_to_next_chunk = None
            counter = 0

            while True:
                # PART 1: Consume as many batches all at once for every worker and their respective current chunk
                if num_of_samples_to_carry_to_next_chunk is None:
                    sizes = [size for size in workers_interval_sizes_for_this_rank if len(size)]
                    min_interval_size = min(size[0] for size in sizes)
                    # -1 here because we need the logic in PART 2 to .pop() the list for the last batch
                    num_batches = (min_interval_size // batch_size) - 1
                    num_batches = max(num_batches, 0)
                    for i in range(len(workers_interval_sizes_for_this_rank)):
                        if workers_interval_sizes_for_this_rank[i]:
                            workers_interval_sizes_for_this_rank[i][0] -= num_batches * batch_size
                    worker_tracker_idx += num_batches * len(sizes)
                    counter += num_batches * batch_size * len(sizes)

                interval_size_of_current_worker = workers_interval_sizes_for_this_rank[worker_tracker_idx % num_workers]
                if len(interval_size_of_current_worker) == 0:
                    worker_tracker_idx += 1
                    continue

                # PART 2: We have leftover samples to consume
                # We consume them one by one because we're at the end of a chunk and may have to handle
                # a remainder from the previous iteration
                num_samples_left_for_this_worker_chunk = interval_size_of_current_worker[0]
                # To consume a batch, we want to subtract `batch_size` from the size we have left,
                # unless we had a remainder (< batch size) from the previous iteration/chunk
                remover = (
                    batch_size
                    if num_of_samples_to_carry_to_next_chunk is None
                    else num_of_samples_to_carry_to_next_chunk
                )

                if num_samples_left_for_this_worker_chunk > remover:
                    # There are samples left to consume, so we subtract the batch size (or a remainder)
                    workers_interval_sizes_for_this_rank[worker_tracker_idx % num_workers][0] -= remover
                    counter += remover
                    num_of_samples_to_carry_to_next_chunk = None
                else:
                    # There are fewer samples left in this chunk than we would like to consume for a full batch
                    # So we take what's left from the chunk and move to the next chunk to complete the batch
                    current_worker_chunk_index = workers_chunks_for_this_rank[worker_tracker_idx % num_workers].pop(0)
                    workers_interval_sizes_for_this_rank[worker_tracker_idx % num_workers].pop(0)
                    counter += remover

                    if current_worker_chunk_index == chunk_index:
                        num_shared_workers_for_this_rank -= 1

                    # TODO: Maybe, we can prevent loading over and over for each worker
                    if num_shared_workers_for_this_rank == 0 and current_worker_chunk_index == chunk_index:
                        # We consumed entirely the chunk of the worker we were tracking
                        # Keep track of how many samples this worker consumed for this chunk and which worker
                        # has consumed the most samples for this chunk
                        if chunk_index not in max_trackers:
                            max_trackers[chunk_index] = (
                                local_rank * num_workers + worker_tracker_idx % num_workers,
                                counter,
                            )
                        else:
                            if max_trackers[chunk_index][1] < counter:
                                max_trackers[chunk_index] = (
                                    local_rank * num_workers + worker_tracker_idx % num_workers,
                                    counter,
                                )
                        break

                    if num_samples_left_for_this_worker_chunk != batch_size:
                        # If a batch was not assembled completely because we're at the end of a chunk,
                        # we need to complete the assembly from samples in the next chunk and carry
                        # over that remainder to the next loop iteration
                        num_of_samples_to_carry_to_next_chunk = batch_size - num_samples_left_for_this_worker_chunk

                    if remover != batch_size:
                        # We've handled the remainder, reset it. Next iteration will start a fresh batch.
                        num_of_samples_to_carry_to_next_chunk = None

                if num_of_samples_to_carry_to_next_chunk is None:
                    # Only go to the next worker if we assembled a full batch. If we have a remainder,
                    # we need to go to the next chunk with the same worker and complete the batch.
                    worker_tracker_idx += 1

    to_disable = {}
    for chunk_index, worker_ids in shared_chunks.items():
        last_worker_idx = max_trackers[chunk_index][0]
        to_disable[chunk_index] = [worker_idx for worker_idx in worker_ids if worker_idx != last_worker_idx]
    return to_disable


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


def _aggregate_shared_chunks_per_rank(
    shared_chunks: dict[int, list[int]], num_workers: int
) -> dict[int, dict[int, list[int]]]:
    """Groups together shared chunks by rank.

    The output is a dictionary mapping a chunk index to a dictionary that maps a rank to a list of workers.

    """
    aggregated_shared_chunks_per_rank: dict[int, dict[int, list[int]]] = {}
    for chunk_index, workers_ids in shared_chunks.items():
        aggregated_shared_chunks_per_rank[chunk_index] = {}
        for worker_idx in workers_ids:
            if (worker_idx // num_workers) not in aggregated_shared_chunks_per_rank[chunk_index]:
                aggregated_shared_chunks_per_rank[chunk_index][worker_idx // num_workers] = []
            aggregated_shared_chunks_per_rank[chunk_index][worker_idx // num_workers].append(worker_idx)
    return aggregated_shared_chunks_per_rank


def _map_node_worker_rank_to_chunk_indexes_to_not_delete(to_disable: dict[int, list[int]]) -> dict[int, list[int]]:
    """Takes a dictionary mapping a chunk index to a list of workers and inverts the map such that it returns a
    dictionary mapping a worker to a list of chunk indexes (that should not be deleted by that worker).
    """
    map_node_worker_rank_to_chunk_indexes: dict[int, list[int]] = {}
    for chunk_index, worker_ids in to_disable.items():
        for worker_idx in worker_ids:
            if worker_idx not in map_node_worker_rank_to_chunk_indexes:
                map_node_worker_rank_to_chunk_indexes[worker_idx] = []
            map_node_worker_rank_to_chunk_indexes[worker_idx].append(chunk_index)
    return map_node_worker_rank_to_chunk_indexes

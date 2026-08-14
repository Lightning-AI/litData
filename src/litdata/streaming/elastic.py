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

"""Elastic resume: frozen canonical item order + global sample_in_epoch.

The epoch is a 1D stream of ``(chunk_index, item_index)`` that does not depend on
``world_size`` or ``num_workers``. Resume drops a prefix (``sample_in_epoch``) and
restripes the suffix onto the new ``(world_size, num_workers, batch_size)`` grid.

``granularity="item"`` restripes at batch boundaries. ``granularity="chunk"`` keeps
whole remaining chunks together (POSIX-fast / WindowShuffle).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from litdata.utilities.shuffle import _window_shuffle

Granularity = Literal["item", "chunk"]


def topology_changed(state: dict[str, Any], *, world_size: int, num_workers: int, batch_size: int) -> bool:
    """True when resume cannot use per-worker prefix replay."""

    def _workers(value: Any) -> int:
        n = int(value)
        return n if n > 0 else 1

    return (
        int(state.get("world_size", world_size)) != int(world_size)
        or _workers(state.get("num_workers", num_workers)) != _workers(num_workers)
        or int(state.get("batch_size", batch_size)) != int(batch_size)
    )


def sample_in_epoch_from_state(state: dict[str, Any]) -> int:
    """Global samples already taken in this epoch."""
    if "sample_in_epoch" in state:
        return max(0, int(state["sample_in_epoch"]))
    yielded = int(state.get("num_samples_yielded", 0))
    world_size = max(1, int(state.get("world_size", 1)))
    return yielded * world_size


def canonical_chunk_order(
    n_chunks: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    num_canonical_nodes: int,
    window: int | None = None,
) -> list[int]:
    """Chunk visit order independent of physical world size."""
    if n_chunks <= 0:
        return []
    ncn = max(1, int(num_canonical_nodes))
    indexes = list(range(n_chunks))
    if shuffle:
        indexes = np.random.RandomState([seed, epoch]).permutation(indexes).tolist()
    if ncn == 1 or n_chunks == 1:
        return indexes
    # Contiguous buckets so a chunk never straddles two canonical nodes.
    buckets: list[list[int]] = [[] for _ in range(ncn)]
    for i, chunk_index in enumerate(indexes):
        buckets[min(ncn - 1, i * ncn // n_chunks)].append(chunk_index)
    if window is not None and window > 1:
        out: list[int] = []
        for bucket_idx, bucket in enumerate(buckets):
            rng = np.random.RandomState([seed, epoch, bucket_idx])
            out.extend(_window_shuffle(bucket, window, rng))
        return out
    return [c for bucket in buckets for c in bucket]


def canonical_item_stream(
    chunk_intervals: list[Any],
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    num_canonical_nodes: int = 1,
    window: int | None = None,
) -> list[tuple[int, int]]:
    """1D ``(chunk_index, item_index)`` stream for the epoch."""
    n_chunks = len(chunk_intervals)
    order = canonical_chunk_order(
        n_chunks,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        num_canonical_nodes=num_canonical_nodes,
        window=window,
    )
    stream: list[tuple[int, int]] = []
    for chunk_index in order:
        interval = chunk_intervals[chunk_index]
        roi_start, roi_end = int(interval[1]), int(interval[2])
        items = list(range(roi_start, roi_end))
        if shuffle:
            if window is not None and window > 1:
                rng = np.random.RandomState([seed, n_chunks, epoch, chunk_index])
                items = _window_shuffle(items, window, rng)
            else:
                items = np.random.RandomState([seed, n_chunks, epoch, chunk_index]).permutation(items).tolist()
        stream.extend((chunk_index, item) for item in items)
    return stream


def _round_down_drop_first(drop_first: int, world_size: int, batch_size: int) -> int:
    """Align the prefix to a global batch so every rank stays in lockstep."""
    drop_first = max(0, int(drop_first))
    stride = max(1, int(world_size) * max(1, int(batch_size)))
    return drop_first - (drop_first % stride)


def _group_visits(pairs: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Collapse consecutive same-chunk items into visits."""
    visits: list[tuple[int, list[int]]] = []
    for chunk_index, item in pairs:
        if visits and visits[-1][0] == chunk_index:
            visits[-1][1].append(item)
        else:
            visits.append((chunk_index, [item]))
    return visits


def restripe_items(
    stream: list[tuple[int, int]],
    *,
    world_size: int,
    num_workers: int,
    batch_size: int,
    drop_first: int = 0,
    drop_last: bool = False,
    granularity: Granularity = "item",
) -> list[list[tuple[int, list[int]]]]:
    """Assign remaining canonical items to ``world_size * num_workers`` worker plans.

    Returns one plan per global worker: a list of ``(chunk_index, item_ids)`` visits.
    """
    world_size = max(1, int(world_size))
    num_workers = max(1, int(num_workers))
    batch_size = max(1, int(batch_size))
    global_workers = world_size * num_workers

    if granularity == "chunk":
        remaining = stream[max(0, drop_first) :]
        # Keep whole remaining chunks; skip a chunk that was already entered.
        if drop_first > 0 and remaining:
            first_chunk = remaining[0][0]
            if stream and drop_first < len(stream) and stream[drop_first - 1][0] == first_chunk:
                remaining = [p for p in remaining if p[0] != first_chunk]
        chunk_groups = _group_visits(remaining)
        if drop_last and chunk_groups:
            # Equal number of chunks per worker (trim).
            cap = len(chunk_groups) // global_workers
            chunk_groups = chunk_groups[: cap * global_workers]
        chunk_plans: list[list[tuple[int, list[int]]]] = [[] for _ in range(global_workers)]
        for i, visit in enumerate(chunk_groups):
            chunk_plans[i % global_workers].append(visit)
        return chunk_plans

    drop_first = _round_down_drop_first(drop_first, world_size, batch_size)
    remaining = stream[drop_first:]
    if drop_last:
        stride = global_workers * batch_size
        remaining = remaining[: (len(remaining) // stride) * stride]

    pair_plans: list[list[tuple[int, int]]] = [[] for _ in range(global_workers)]
    # Consecutive batches cycle ranks, then workers within a rank.
    for i, pair in enumerate(remaining):
        batch_idx = i // batch_size
        rank = batch_idx % world_size
        worker = (batch_idx // world_size) % num_workers
        global_worker = rank * num_workers + worker
        pair_plans[global_worker].append(pair)

    return [_group_visits(pairs) for pairs in pair_plans]


def lockstep_stream_from_worker_seqs(
    seqs: list[list[tuple[int, int]]],
    *,
    world_size: int,
    num_workers: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Rebuild the global 1D order implied by DDP lockstep + worker cycling.

    This is the inverse of ``restripe_items`` (item granularity): consecutive
    ``batch_size`` items go to rank ``batch_idx % world_size``, then workers
    cycle. Used so elastic ``drop_first`` skips the same IDs that were served
    under the original shuffler assignment.
    """
    world_size = max(1, int(world_size))
    num_workers = max(1, int(num_workers))
    batch_size = max(1, int(batch_size))
    heads = [0] * len(seqs)
    total = sum(len(s) for s in seqs)
    stream: list[tuple[int, int]] = []
    i = 0
    while len(stream) < total:
        batch_idx = i // batch_size
        rank = batch_idx % world_size
        worker = (batch_idx // world_size) % num_workers
        gw = rank * num_workers + worker
        if gw < len(seqs) and heads[gw] < len(seqs[gw]):
            stream.append(seqs[gw][heads[gw]])
            heads[gw] += 1
        i += 1
    return stream


def worker_plan_to_chunks(
    visits: list[tuple[int, list[int]]],
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Convert visits to ``worker_chunks``, dummy intervals, and per-visit item lists."""
    chunks: list[int] = []
    intervals: list[list[int]] = []
    item_lists: list[list[int]] = []
    for chunk_index, items in visits:
        if not items:
            continue
        chunks.append(int(chunk_index))
        n = len(items)
        # Dummy ROI length matches visit size so stop_length / chunk_size stay consistent.
        intervals.append([0, 0, n, n])
        item_lists.append(list(items))
    return chunks, intervals, item_lists

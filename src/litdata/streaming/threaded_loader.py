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

"""Experimental in-process threaded batch loader for StreamingDataset.

Opt-in via ``StreamingDataLoader(..., use_threading=True)`` on a free-threaded
(no-GIL) Python runtime only. Each thread owns an isolated dataset clone
(distinct ``_forced_worker_env`` rank), collates in-thread, and hands batches
through per-worker queues. The consumer round-robins workers so batch order is
deterministic.

Prefetch is bounded **per worker** (not globally). A global slot count plus
in-order round-robin can deadlock when later workers fill every slot while the
consumer waits on an earlier worker — the same class of bug as an undersized
in-order DataLoader queue.

This is experimental and not the default Lightning/PyTorch loading path.

Why not asyncio for this loader
-------------------------------
Asyncio is a poor fit as a replacement for ``use_threading``:

* Training loops are synchronous (``for batch in loader``); a fully async
  stack would need a second API or a sync façade that hides the event loop.
* The win here is avoiding **process IPC** while parallelizing CPU decode /
  collate. On free-threaded CPython, OS threads already do that; under the
  GIL, ``asyncio.to_thread`` collapses back to threads with extra scheduling.
* Most of the read path (item loaders, mmap, FileLock, serializers) is sync.

IO concurrency belongs in chunk prefetch (see ``async_prefetch`` /
``PrepareChunksThread``), not in a fully async DataLoader.
asyncio is **not** exposed as ``use_asyncio=`` on ``StreamingDataLoader``.

Learnings from FFCV (https://github.com/libffcv/ffcv) applied / skipped here
---------------------------------------------------------------------------
Adopted (lightweight):
  * Prefer threads over processes when the goal is to avoid result-queue pickle /
    SHM handoff — FFCV's Loader is thread-based for the same reason.
  * Bound in-flight work with a semaphore (similar in spirit to FFCV's slot /
    circular-buffer prefetch), not an unbounded queue.
  * Keep FileLock only for *cross-process* cache safety; same-process threads
    coordinate via in-process queues (FFCV uses background reader threads for
    page loads without per-sample process IPC).

Deliberately not ported (incompatible or out of scope for LitData):
  * Custom ``.beton`` page format + quasi-random page-local shuffle — LitData
    already has chunked ``.bin`` + deterministic FullShuffle / resume.
  * Numba JIT fused transform stages + pre-allocated batch memory arenas —
    LitData samples are arbitrary pytrees (PIL/video/pickle leaves); a
    Numba-only pipeline would break the public API. Compiled unflatten (#849)
    is the LitData-native analogue for structure rebuild.
  * Single shared OS-cached memmap of the whole dataset — LitData streams from
    remote object stores with bounded ``max_cache_size`` and shared-chunk
    deletion; whole-dataset RAM residency is not acceptable.
  * Transform-level ``prange`` parallelism as the main concurrency model —
    LitData's bottleneck is often remote download + deserialize; we keep
    per-thread readers (like process workers) rather than one reader + batch
    Numba fan-out.
"""

from __future__ import annotations

import pickle
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.timing import StreamingTimingStats
from litdata.utilities.env import _WorkerEnv

_SENTINEL = object()


def _clone_dataset_for_thread(dataset: StreamingDataset, rank: int, world_size: int) -> StreamingDataset:
    """Pickle-roundtrip clone so each thread has isolated cache/reader/state."""
    # Trusted in-process clone of our own dataset object (not untrusted input).
    clone: StreamingDataset = pickle.loads(pickle.dumps(dataset))  # noqa: S301
    clone._forced_worker_env = _WorkerEnv(world_size=world_size, rank=rank)
    clone.num_workers = world_size
    return clone


def _worker_loop(
    dataset: StreamingDataset,
    batch_size: int,
    collate_fn: Callable[[list[Any]], Any],
    out_queue: queue.SimpleQueue,
    stop_event: threading.Event,
    slot: threading.Semaphore,
) -> None:
    timing = StreamingTimingStats.instance()
    try:
        iterator = iter(dataset)
        while not stop_event.is_set():
            # Bound this worker's in-flight batches (mirrors DataLoader prefetch_factor).
            while not slot.acquire(timeout=0.05):
                if stop_event.is_set():
                    out_queue.put((_SENTINEL, None))
                    return

            samples: list[Any] = []
            try:
                for _ in range(batch_size):
                    samples.append(next(iterator))
            except StopIteration:
                if samples:
                    t0 = timing.start()
                    batch = collate_fn(samples)
                    timing.record("thread_collate_s", t0)
                    out_queue.put((True, batch))
                else:
                    slot.release()
                break

            t0 = timing.start()
            batch = collate_fn(samples)
            timing.record("thread_collate_s", t0)
            t1 = timing.start()
            out_queue.put((True, batch))
            timing.record("thread_queue_put_s", t1)
    except Exception as exc:
        out_queue.put((_SENTINEL, exc))
        return

    out_queue.put((_SENTINEL, None))


def iter_threaded_streaming_batches(
    dataset: StreamingDataset,
    *,
    num_workers: int,
    batch_size: int,
    collate_fn: Callable[[list[Any]], Any],
    prefetch_factor: int = 2,
) -> Iterator[Any]:
    """Yield collated batches from ``num_workers`` in-process loader threads.

    Batches are consumed in round-robin worker order so interleaving is
    deterministic for a given ``num_workers`` / shuffle seed.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1 for threaded loading")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1 for threaded loading")

    stop_event = threading.Event()
    out_queues: list[queue.SimpleQueue] = [queue.SimpleQueue() for _ in range(num_workers)]
    # Per-worker bound: avoids in-order deadlock from a single global slot pool.
    slots = [threading.Semaphore(max(prefetch_factor, 1)) for _ in range(num_workers)]
    threads: list[threading.Thread] = []
    finished = [False] * num_workers
    timing = StreamingTimingStats.instance()

    for rank in range(num_workers):
        worker_ds = _clone_dataset_for_thread(dataset, rank=rank, world_size=num_workers)
        t = threading.Thread(
            target=_worker_loop,
            name=f"litdata-thread-worker-{rank}",
            args=(worker_ds, batch_size, collate_fn, out_queues[rank], stop_event, slots[rank]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        next_worker = 0
        while not all(finished):
            if finished[next_worker]:
                next_worker = (next_worker + 1) % num_workers
                continue

            t0 = timing.start()
            tag, payload = out_queues[next_worker].get()
            timing.record("thread_queue_get_s", t0)

            if tag is _SENTINEL:
                finished[next_worker] = True
                if isinstance(payload, BaseException):
                    stop_event.set()
                    raise payload
                next_worker = (next_worker + 1) % num_workers
                continue

            try:
                yield payload
            finally:
                slots[next_worker].release()
            next_worker = (next_worker + 1) % num_workers
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

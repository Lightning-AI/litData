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

Opt-in via ``StreamingDataLoader(..., use_threading=True)``. Each thread owns an
isolated dataset clone (distinct ``_forced_worker_env`` rank), collates in-thread,
and hands batches through ``queue.SimpleQueue`` — no pickle / process IPC.

This is a spike for comparing against PyTorch process workers; it is not the
default Lightning/PyTorch loading path.

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
            # Bound in-flight batches across workers (mirrors DataLoader prefetch_factor).
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
    """Yield collated batches from ``num_workers`` in-process loader threads."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1 for threaded loading")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1 for threaded loading")

    stop_event = threading.Event()
    out_queue: queue.SimpleQueue = queue.SimpleQueue()
    # Cap total in-flight batches roughly like ``prefetch_factor * num_workers``.
    slot = threading.Semaphore(max(prefetch_factor, 1) * num_workers)
    threads: list[threading.Thread] = []
    finished = 0
    timing = StreamingTimingStats.instance()

    for rank in range(num_workers):
        worker_ds = _clone_dataset_for_thread(dataset, rank=rank, world_size=num_workers)
        t = threading.Thread(
            target=_worker_loop,
            name=f"litdata-thread-worker-{rank}",
            args=(worker_ds, batch_size, collate_fn, out_queue, stop_event, slot),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        while finished < num_workers:
            t0 = timing.start()
            tag, payload = out_queue.get()
            timing.record("thread_queue_get_s", t0)
            if tag is _SENTINEL:
                finished += 1
                if isinstance(payload, BaseException):
                    stop_event.set()
                    raise payload
                continue
            try:
                yield payload
            finally:
                slot.release()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

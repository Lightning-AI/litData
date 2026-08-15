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
import concurrent
import json
import logging
import multiprocessing
import multiprocessing.queues
import os
import pickle
import random
import shutil
import signal
import sys
import tempfile
import traceback
import warnings
from abc import abstractmethod
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Thread
from time import sleep, time
from typing import Any, TypeVar
from urllib import parse

import numpy as np
import torch

from litdata.constants import (
    _DEFAULT_FAST_DEV_RUN_ITEMS,
    _ENABLE_STATUS,
    _INDEX_FILENAME,
    _IS_IN_STUDIO,
    _SUPPORTED_PROVIDERS,
    _TQDM_AVAILABLE,
)
from litdata.processing.readers import BaseReader, StreamingDataLoaderReader
from litdata.processing.utilities import construct_storage_options, remove_uuid_from_filename
from litdata.streaming import Cache
from litdata.streaming.async_prefetch import downloader_supports_adownload, downloader_supports_aupload
from litdata.streaming.cache import Dir
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.downloader import get_downloader
from litdata.streaming.fs_provider import _get_fs_provider, not_supported_provider
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import _has_time_template, _resolve_dir
from litdata.utilities._pytree import tree_flatten, tree_unflatten, treespec_loads
from litdata.utilities.broadcast import broadcast_object
from litdata.utilities.dataset_utilities import load_index_file
from litdata.utilities.encryption import Encryption
from litdata.utilities.packing import _pack_greedily

logger = logging.getLogger(__name__)

ALL_DONE = "ALL_DONE"  # sentinel value for shared queue
_DEFAULT_PREFETCH_BYTES = 512 * 1024 * 1024
_PARENT_QUEUE_POLL_S = 1.0
_DEFAULT_DOWNLOAD_BATCH = 16
_DEFAULT_DOWNLOAD_CONCURRENCY = 16


def resolve_keep_data_ordered(
    keep_data_ordered: bool | None,
    *,
    use_checkpoint: bool = False,
    align_chunking: bool = False,
) -> bool:
    """Default is unordered (shared queue). Checkpoint and align_chunking require order."""
    if keep_data_ordered is None:
        return bool(use_checkpoint or align_chunking)
    if not keep_data_ordered and use_checkpoint:
        raise ValueError("Checkpoint feature is not supported for Queue based data processing, yet.")
    if not keep_data_ordered and align_chunking:
        raise ValueError("align_chunking requires keep_data_ordered=True.")
    return keep_data_ordered


def _prefetch_maxsize(num_workers: int, items: list[Any] | None = None) -> int:
    """Bound the ready queue by slot count and, when file sizes are cheap, a byte budget."""
    slot_cap = max(8, 2 * num_workers)
    raw = os.getenv("LITDATA_PREFETCH_BYTES", str(_DEFAULT_PREFETCH_BYTES))
    try:
        budget = int(raw)
    except ValueError:
        budget = _DEFAULT_PREFETCH_BYTES
    if budget <= 0 or not items:
        return slot_cap
    sizes: list[int] = []
    for item in items[:64]:
        if not isinstance(item, str) or _is_remote_path(item) or _is_studio_fuse_path(item):
            continue
        if not os.path.isfile(item):
            continue
        try:
            sizes.append(os.path.getsize(item))
        except OSError:
            continue
    if not sizes:
        return slot_cap
    avg = sum(sizes) / len(sizes)
    by_bytes = max(2, int(budget / max(avg, 1)))
    return max(2, min(slot_cap, by_bytes, len(items)))


def _optimize_download_batch() -> int:
    return max(1, int(os.getenv("LITDATA_OPTIMIZE_DOWNLOAD_BATCH", str(_DEFAULT_DOWNLOAD_BATCH))))


def _optimize_upload_batch() -> int:
    return max(1, int(os.getenv("LITDATA_OPTIMIZE_UPLOAD_BATCH", str(_DEFAULT_DOWNLOAD_BATCH))))


def _optimize_download_concurrency() -> int:
    return max(1, int(os.getenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY", str(_DEFAULT_DOWNLOAD_CONCURRENCY))))


def _download_via_streaming_downloader(
    downloader: Any,
    jobs: list[tuple[str, str]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Fetch ``(remote, local)`` pairs with the streaming Downloader (obstore async when usable)."""
    if not jobs:
        return
    if downloader_supports_adownload(downloader):

        async def _all() -> None:
            sem = asyncio.Semaphore(_adaptive_download_concurrency(len(jobs)))

            async def _one(remote: str, local: str) -> None:
                async with sem:
                    if os.path.exists(local):
                        return
                    await downloader.adownload_file(remote, local)

            await asyncio.gather(*[_one(remote, local) for remote, local in jobs])

        loop.run_until_complete(_all())
        return

    for remote, local in jobs:
        if not os.path.exists(local):
            downloader.download_file(remote, local)


def _upload_via_streaming_downloader(
    downloader: Any,
    jobs: list[tuple[str, str]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Put ``(local, remote)`` pairs with the streaming Downloader (obstore async when usable)."""
    if not jobs:
        return
    if downloader_supports_aupload(downloader):

        async def _all() -> None:
            sem = asyncio.Semaphore(_adaptive_download_concurrency(len(jobs)))

            async def _one(local: str, remote: str) -> None:
                async with sem:
                    await downloader.aupload_file(local, remote)

            await asyncio.gather(*[_one(local, remote) for local, remote in jobs])

        loop.run_until_complete(_all())
        return

    raise NotImplementedError(f"{type(downloader).__name__} does not support async upload")


def _same_local_file(src: str, dst: str) -> bool:
    """True when ``src`` and ``dst`` are the same inode (macOS ``/var`` vs ``/private/var``)."""
    try:
        return os.path.samefile(src, dst)
    except OSError:
        return os.path.normpath(os.path.abspath(src)) == os.path.normpath(os.path.abspath(dst))


def _upload_dest(output_dir: Dir, local_filepath: str, tmpdir: str | None) -> str:
    """Remote or local destination path for an optimized chunk or sidecar file."""
    url = output_dir.url
    remote = bool(url and parse.urlparse(url).scheme in _SUPPORTED_PROVIDERS)
    output_filepath = (url if remote else output_dir.path) or url
    assert output_filepath
    if ".checkpoints" in local_filepath:
        output_filepath = os.path.join(output_filepath, ".checkpoints")
    if tmpdir is None:
        output_filepath = os.path.join(output_filepath, os.path.basename(local_filepath))
    else:
        output_filepath = os.path.join(output_filepath, local_filepath.replace(tmpdir, "")[1:])
    return remove_uuid_from_filename(output_filepath)


def _put_files_remote(
    output_dir: Dir,
    jobs: list[tuple[str, str]],
    storage_options: dict[str, Any],
    cache_dir: str,
    downloader: Any | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[Any, asyncio.AbstractEventLoop | None]:
    """Upload local→remote pairs. Reuses ``downloader``/``loop`` when provided."""
    assert output_dir.url
    merged_storage_options = construct_storage_options(storage_options, output_dir)
    if downloader is None:
        downloader = get_downloader(output_dir.url, cache_dir, [], merged_storage_options)
    if downloader_supports_aupload(downloader):
        if loop is None:
            loop = asyncio.new_event_loop()
        _upload_via_streaming_downloader(downloader, jobs, loop)
        return downloader, loop
    fs_provider = _get_fs_provider(output_dir.url, merged_storage_options)
    for local_filepath, output_filepath in jobs:
        fs_provider.upload_file(local_filepath, output_filepath)
    return downloader, loop


def _io_thread_target(fn: Callable[..., None], error_queue: Queue, *args: Any) -> None:
    """Run an I/O loop in a thread and surface failures on ``error_queue``."""
    try:
        fn(*args)
    except Exception:
        error_queue.put(traceback.format_exc())


def _get_num_nodes() -> int:
    """Returns the number of nodes."""
    return int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 1))


def _get_node_rank() -> int:
    """Returns the current node rank of the instance."""
    return int(os.getenv("DATA_OPTIMIZER_NODE_RANK", 0))


def _get_fast_dev_run() -> int:
    """Returns whether fast dev mode is enabled."""
    return bool(int(os.getenv("DATA_OPTIMIZER_FAST_DEV_RUN", 1)))


def _get_default_cache() -> str:
    return "/cache" if _IS_IN_STUDIO else tempfile.gettempdir()


def _get_cache_dir(name: str | None = None) -> str:
    """Returns the cache directory used by the Cache to store the chunks."""
    cache_dir = os.getenv("DATA_OPTIMIZER_CACHE_FOLDER", f"{_get_default_cache()}/chunks")
    if name is None:
        return cache_dir
    return os.path.join(cache_dir, name.lstrip("/"))


def _is_local_write_through(output_dir: Dir | None) -> bool:
    """True when chunks can be written directly into a local output directory."""
    return bool(output_dir is not None and output_dir.url is None and output_dir.path)


def _chunks_dir(output_dir: Dir | None) -> str:
    """Directory that receives ``chunk-*.bin`` files (output path when local, else the cache)."""
    if _is_local_write_through(output_dir):
        assert output_dir is not None
        assert output_dir.path
        os.makedirs(output_dir.path, exist_ok=True)
        return output_dir.path
    return _get_cache_dir()


def _adaptive_download_concurrency(n_jobs: int, num_workers: int | None = None) -> int:
    """In-flight object-store GETs: env cap, else scale with workers/CPU and free disk."""
    env = os.getenv("LITDATA_OPTIMIZE_DOWNLOAD_CONCURRENCY")
    if env:
        cap = max(1, int(env))
    else:
        cpus = os.cpu_count() or 4
        workers = num_workers or int(os.getenv("DATA_OPTIMIZER_NUM_WORKERS", "0")) or cpus
        cap = min(32, max(8, int(workers) * 2, cpus))
    try:
        usage = shutil.disk_usage("/")
        if usage.total > 0:
            free_ratio = usage.free / usage.total
            if free_ratio < 0.10:
                cap = min(cap, 4)
            elif free_ratio < 0.25:
                cap = min(cap, 8)
    except OSError:
        pass
    return max(1, min(cap, max(1, n_jobs)))


def _get_cache_data_dir(name: str | None = None) -> str:
    """Returns the cache data directory used by the DataProcessor workers to download the files."""
    cache_dir = os.getenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", f"{_get_default_cache()}/data")
    if name is None:
        return cache_dir
    return os.path.join(cache_dir, name.lstrip("/"))


def _wait_for_file_to_exist(remote_filepath: str, sleep_time: int = 2, storage_options: dict[str, Any] = {}) -> Any:
    """Wait until the file exists."""
    file_exists = False
    fs_provider = _get_fs_provider(remote_filepath, storage_options)
    while not file_exists:
        file_exists = fs_provider.exists(remote_filepath)
        if not file_exists:
            sleep(sleep_time)


def _wait_for_disk_usage_higher_than_threshold(input_dir: str, threshold_in_gb: int = 25, sleep_time: int = 3) -> None:
    """Wait until the specified directory has more free disk space than the threshold."""
    usage = shutil.disk_usage(input_dir)

    while (usage.free / 1000 / 1000 / 1000) <= threshold_in_gb:
        sleep(sleep_time)
        usage = shutil.disk_usage(input_dir)

    return


#
# `_download_data_target` function accepts two queues:
# 1. `queue_in`: A queue that receives the (index, paths) from where the data is to be downloaded.
# 2. `queue_out`: A queue that sends the index after the files have been downloaded and ready to be used.
#
def _is_remote_path(element: str) -> bool:
    return parse.urlparse(element).scheme in _SUPPORTED_PROVIDERS


def _cache_local_path(path: str, input_dir: Dir, cache_dir: str) -> str:
    """Map a FUSE or remote object path onto the local download cache."""
    if input_dir.path and path.startswith(input_dir.path):
        return path.replace(input_dir.path, cache_dir)
    if input_dir.url and path.startswith(input_dir.url):
        rel = path[len(input_dir.url.rstrip("/")) :].lstrip("/")
        return os.path.join(cache_dir, rel)
    if _is_remote_path(path):
        return os.path.join(cache_dir, parse.urlparse(path).path.lstrip("/"))
    return os.path.join(cache_dir, os.path.basename(path))


def _source_path_for_download(path: str, input_dir: Dir) -> str:
    """Prefer the object-store URL over a Studio FUSE path when both exist."""
    if _is_remote_path(path):
        return path
    if input_dir.url and input_dir.path and path.startswith(input_dir.path):
        return path.replace(input_dir.path, input_dir.url)
    return path


def _dir_needs_download(input_dir: Dir | None, reader: BaseReader | None = None) -> bool:
    if reader is not None or input_dir is None:
        return False
    return input_dir.url is not None


def _download_data_target(
    input_dir: Dir,
    cache_dir: str,
    queue_in: Any,
    queue_out: Any,
    storage_options: dict[str, Any] = {},
    emit_done: bool = True,
    index_only: bool = False,
) -> None:
    """Download data from a remote directory to a cache directory to optimise reading."""
    downloader = None
    loop: asyncio.AbstractEventLoop | None = None

    def _emit(index: int, item: Any, paths: list[str]) -> None:
        queue_out.put(index if index_only else (index, item, paths))

    def _ensure_downloader(remote_url: str) -> Any:
        nonlocal downloader, loop
        if downloader is None:
            merged_storage_options = construct_storage_options(storage_options, input_dir)
            downloader = get_downloader(remote_url, cache_dir, [], merged_storage_options)
        if loop is None:
            loop = asyncio.new_event_loop()
        return downloader

    try:
        while True:
            first: tuple[int, Any, list[str]] | None = queue_in.get()
            if first is None:
                if emit_done:
                    queue_out.put(None)
                return

            batch: list[tuple[int, Any, list[str]]] = [first]
            saw_done = False
            limit = _optimize_download_batch()
            while len(batch) < limit:
                try:
                    nxt = queue_in.get_nowait()
                except Empty:
                    break
                if nxt is None:
                    saw_done = True
                    break
                batch.append(nxt)

            store_jobs: list[tuple[str, str, str]] = []
            pending_emit: list[tuple[int, Any, list[str]]] = []
            for index, item, paths in batch:
                if input_dir.path and all(os.path.exists(_cache_local_path(p, input_dir, cache_dir)) for p in paths):
                    # Keep batch order: a cache hit must not jump ahead of earlier
                    # items still waiting on copy/download (Windows local-file batches).
                    pending_emit.append((index, item, paths))
                    continue

                if input_dir.url is None and input_dir.path is None:
                    _emit(index, item, paths)
                    continue

                if input_dir.url:
                    _wait_for_disk_usage_higher_than_threshold("/", 25)

                for path in paths:
                    original_path = path
                    local_path = _cache_local_path(path, input_dir, cache_dir)
                    remote_path = _source_path_for_download(path, input_dir)
                    obj = parse.urlparse(remote_path)
                    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

                    fuse_or_local = original_path
                    if input_dir.path and _is_remote_path(original_path) and input_dir.url:
                        fuse_or_local = original_path.replace(input_dir.url, input_dir.path)

                    use_object_store = obj.scheme in _SUPPORTED_PROVIDERS and (
                        _is_studio_fuse_path(fuse_or_local) or not os.path.isfile(fuse_or_local)
                    )

                    if use_object_store:
                        store_jobs.append((remote_path, local_path, fuse_or_local))
                    elif os.path.isfile(fuse_or_local) and not fuse_or_local.startswith(
                        "/teamspace/studios/this_studio"
                    ):
                        shutil.copyfile(fuse_or_local, local_path)
                    elif (
                        os.path.isfile(remote_path)
                        and not _is_remote_path(remote_path)
                        and not _is_studio_fuse_path(remote_path)
                    ):
                        shutil.copyfile(remote_path, local_path)
                    else:
                        raise ValueError(f"The provided {input_dir.url or remote_path} isn't supported.")

                pending_emit.append((index, item, paths))

            if store_jobs:
                unique_jobs: list[tuple[str, str]] = []
                seen: set[tuple[str, str]] = set()
                for remote_path, local_path, _fuse in store_jobs:
                    key = (remote_path, local_path)
                    if key not in seen:
                        seen.add(key)
                        unique_jobs.append(key)
                remote_url = input_dir.url or store_jobs[0][0]
                try:
                    handle = _ensure_downloader(remote_url)
                    assert loop is not None
                    _download_via_streaming_downloader(handle, unique_jobs, loop)
                except Exception:
                    missing = [(remote, local, fuse) for remote, local, fuse in store_jobs if not os.path.exists(local)]
                    if not missing:
                        raise
                    for _remote, local, fuse_or_local in missing:
                        if os.path.isfile(fuse_or_local):
                            shutil.copyfile(fuse_or_local, local)
                        else:
                            raise

            for index, item, paths in pending_emit:
                _emit(index, item, paths)

            if saw_done:
                if emit_done:
                    queue_out.put(None)
                return
    finally:
        # Do not close ``loop``: node download and upload threads share this
        # process and obstore's tokio runtime. Closing the download loop drops
        # in-flight PUTs with "Event loop is closed".
        pass


#
# `_remove_target` function accepts a queue that receives the paths to delete files from the cache directory.
#
def _remove_target(input_dir: Dir, cache_dir: str, queue_in: Queue) -> None:
    """Delete files from the cache directory to minimise disk space."""
    while True:
        # 1. Collect paths
        paths = queue_in.get()

        # 2. Terminate the process if we received a termination signal
        if paths is None:
            return

        # 3. Iterate through the paths and delete them sequentially.
        for path in paths:
            if input_dir:
                if not path.startswith(cache_dir) and input_dir.path is not None:
                    path = path.replace(input_dir.path, cache_dir)

                if os.path.exists(path):
                    os.remove(path)

            elif keep_path(path) and os.path.exists(path):
                os.remove(path)


_STUDIO_FUSE_TOKENS = (
    "efs_connections",
    "efs_folders",
    "gcs_connections",
    "s3_connections",
    "s3_folders",
    "snowflake_connections",
    "lightning_storage",
)


def _is_studio_fuse_path(path: str | None) -> bool:
    """Studio object-store FUSE mounts — ``stat`` / ``listdir`` / ``exists`` are very slow."""
    if not path:
        return False
    return any(token in path for token in _STUDIO_FUSE_TOKENS)


def keep_path(path: str) -> bool:
    return not _is_studio_fuse_path(path)


#
# `_upload_fn` accepts two queues:
# 1. `upload_queue`: A queue that receives the local file paths ready to be uploaded.
# 2. `remove_queue`: After uploading, the file is sent to the remove queue,
#                    so it can be deleted from the cache directory.
#
def _upload_fn(
    upload_queue: Any,
    remove_queue: Any,
    cache_dir: str,
    output_dir: Dir,
    storage_options: dict[str, Any] = {},
) -> None:
    """Upload optimised chunks from a local to remote dataset directory."""
    obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
    remote = obj.scheme in _SUPPORTED_PROVIDERS
    downloader: Any | None = None
    loop: asyncio.AbstractEventLoop | None = None
    pending: list[tuple[str, str]] = []
    batch_limit = _optimize_upload_batch()

    def _mark_uploaded(local_filepath: str) -> None:
        if remove_queue and os.path.exists(local_filepath) and ".checkpoints" not in local_filepath:
            remove_queue.put([local_filepath])

    def _flush_remote() -> None:
        nonlocal downloader, loop
        if not pending:
            return
        try:
            downloader, loop = _put_files_remote(
                output_dir, pending, storage_options, cache_dir, downloader=downloader, loop=loop
            )
        except Exception:
            logger.exception("Failed to upload %s files to %s", len(pending), output_dir.url)
            raise
        for local_filepath, _remote in pending:
            _mark_uploaded(local_filepath)
        pending.clear()

    while True:
        data: str | tuple[str, str] | None = upload_queue.get()

        tmpdir = None

        if isinstance(data, str) or data is None:
            local_filepath = data
        else:
            tmpdir, local_filepath = data

        if local_filepath is None:
            if remote:
                _flush_remote()
            return

        if not local_filepath.startswith(cache_dir):
            local_filepath = os.path.join(cache_dir, local_filepath)

        if remote:
            if not os.path.exists(local_filepath) and ".checkpoints" in local_filepath:
                continue
            pending.append((local_filepath, _upload_dest(output_dir, local_filepath, tmpdir)))
            if len(pending) >= batch_limit:
                _flush_remote()
            continue

        if output_dir.path:
            output_filepath = _upload_dest(output_dir, local_filepath, tmpdir)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            if not os.path.exists(local_filepath):
                if ".checkpoints" in local_filepath:
                    continue
                raise FileNotFoundError(local_filepath)
            if not _same_local_file(local_filepath, output_filepath):
                shutil.copy(local_filepath, output_filepath)
            _mark_uploaded(local_filepath)
        else:
            raise ValueError(f"The provided {output_dir.path} isn't supported.")


def _map_items_to_workers_sequentially(
    num_workers: int, user_items: list[Any], chunk_size: int | None = None
) -> list[list[Any]]:
    """Map the items to the workers sequentially.

    Args:
        num_workers: The number of workers to assign items to.
        user_items: The list of items to be distributed among workers.
        chunk_size: Optional `chunk size` that enforces deterministic,
            single-worker-style chunk boundaries. When set, each worker is
            assigned only full chunks of this size, and the final worker
            receives any remaining items (which may form a partial chunk).


    >>> workers_user_items = _map_items_to_workers_sequentially(2, list(range(5)))
    >>> assert workers_user_items == [[0, 1], [2, 3, 4]]
    """
    assert isinstance(chunk_size, (int, type(None))), "chunk_size must be an integer or None"

    num_nodes = _get_num_nodes()
    node_rank = _get_node_rank()
    world_size = num_nodes * num_workers

    if chunk_size is not None:
        assert chunk_size > 0, "chunk_size must be a positive integer"

        # Compute how many full chunks each worker can take
        full_chunks = len(user_items) // chunk_size
        chunks_per_worker = full_chunks // world_size

        if chunks_per_worker == 0 and node_rank == 0:
            warnings.warn(
                f"chunk_size ({chunk_size}) is too large relative to dataset size ({len(user_items)}) "
                f"and world_size ({world_size}). This will result in idle workers. "
                f"Consider reducing chunk_size or using fewer workers."
            )

        # Assign full chunks to all workers except the last
        num_items_per_worker = [chunks_per_worker * chunk_size for _ in range(world_size - 1)]

        # Last worker receives all remaining items (full chunks + optional tail)
        remaining = len(user_items) - sum(num_items_per_worker)
        num_items_per_worker.append(remaining)

    else:
        items_per_worker_count = len(user_items) // world_size

        num_items_per_worker: list[int] = [items_per_worker_count for _ in range(world_size)]
        reminder = len(user_items) % world_size

        for worker_idx in range(len(num_items_per_worker) - 1, -1, -1):
            if reminder == 0:
                break
            num_items_per_worker[worker_idx] += 1
            reminder -= 1

    num_items_cumsum_per_worker = np.cumsum([0] + num_items_per_worker)

    out = []
    worker_idx_start = node_rank * num_workers
    worker_idx_end = (node_rank + 1) * num_workers

    for worker_idx in range(world_size):
        if worker_idx_start <= worker_idx and worker_idx < worker_idx_end:
            start = num_items_cumsum_per_worker[worker_idx]
            end = num_items_cumsum_per_worker[worker_idx + 1]
            out.append(user_items[start:end])

    if len(out) != num_workers:
        raise RuntimeError("The items didn't haven't been assigned properly. Please, open an issue on Github.")

    return out


def _map_items_to_workers_weighted(
    num_workers: int,
    user_items: list[Any],
    weights: list[int] | None = None,
    file_size: bool = True,
) -> list[list[Any]]:
    """Map the items to the workers based on the weights.

    >>> workers_user_items = _map_items_to_workers_weighted(2, list(range(5)), weights=[1, 2, 3, 4, 5])
    >>> assert workers_user_items == [[1, 4, 0], [3, 2]]
    """
    weights = [1] * len(user_items) if weights is None else weights
    num_nodes = _get_num_nodes()
    node_rank = _get_node_rank()
    world_size = num_nodes * num_workers

    worker_items, worker_weights = _pack_greedily(items=user_items, weights=weights, num_bins=world_size)
    worker_ids_this_node = range(node_rank * num_workers, (node_rank + 1) * num_workers)

    for worker_id, size in worker_weights.items():
        if worker_id not in worker_ids_this_node:
            continue

        label = "Node" if world_size == num_nodes else "Worker"
        if file_size:
            print(f"{label} {worker_id} gets {size / 1e6:.1f} MB ({len(worker_items[worker_id])} files)")
        else:
            print(f"{label} {worker_id} gets ({len(worker_items[worker_id])}) items for a total weight of {size}.")

    return [np.random.permutation(worker_items[worker_id]).tolist() for worker_id in worker_ids_this_node]


def _get_num_bytes(item: Any, base_path: str) -> int:
    """For the given item (PyTree), flatten it and return the total size in bytes of all file paths."""
    flattened_item, _ = tree_flatten(item)

    num_bytes = 0
    for element in flattened_item:
        if isinstance(element, str):
            if _is_remote_path(element) or _is_studio_fuse_path(element):
                continue
            element = Path(element).resolve()
            if not element.exists():
                continue
            file_bytes = os.path.getsize(element)
            if file_bytes == 0:
                raise RuntimeError(f"The file {element} has 0 bytes!")
            num_bytes += file_bytes
    return num_bytes


def _get_item_filesizes(items: list[Any], base_path: str = "") -> list[int]:
    """Computes the total size in bytes of all file paths for every datastructure in the given list."""
    item_sizes = []

    cpu_count = os.cpu_count() or 1

    # Parallelize to accelerate retrieving the number of file bytes to read for each item
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count * 2 if cpu_count > 4 else cpu_count) as executor:
        futures = [executor.submit(_get_num_bytes, item, base_path) for item in items]
        for future in futures:
            item_sizes.append(future.result())
    return item_sizes


def _to_path(element: str) -> str:
    if (
        _is_remote_path(element)
        or _is_studio_fuse_path(element)
        or (_IS_IN_STUDIO and element.startswith("/teamspace"))
    ):
        return element
    return str(Path(element).resolve())


def _is_path(input_dir: str | None, element: Any, input_dir_url: str | None = None) -> bool:
    if not isinstance(element, str):
        return False

    if _is_remote_path(element) or _is_studio_fuse_path(element):
        return True

    if input_dir_url is not None and element.startswith(input_dir_url):
        return True

    if input_dir is not None and _is_studio_fuse_path(input_dir) and element.startswith(input_dir):
        return True

    if _IS_IN_STUDIO and input_dir is not None:
        if element.startswith(input_dir):
            return True

        element = str(Path(element).absolute())
        if element.startswith(input_dir):
            # check whether the element has an extension.
            if os.path.splitext(element)[1]:
                return True
            return os.path.isfile(element)

    return os.path.isfile(element)


def _prepare_items_and_paths(
    items: list[Any], input_dir: Dir, cache_data_dir: str
) -> list[tuple[Any, list[str] | None]]:
    """Rewrite filepath leaves to the local cache and collect source paths for downloaders."""
    if input_dir.path is None and input_dir.url is None:
        return [(item, None) for item in items]

    prepared: list[tuple[Any, list[str] | None]] = []
    for item in items:
        flattened_item, spec = tree_flatten(item)
        indexed_paths = {
            index: _to_path(element)
            for index, element in enumerate(flattened_item)
            if _is_path(input_dir.path, element, input_dir.url)
        }
        if len(indexed_paths) == 0:
            raise ValueError(
                f"The provided item {item} didn't contain any filepaths. The input_dir is {input_dir.path}."
            )

        paths = []
        for index, path in indexed_paths.items():
            paths.append(path)
            if _is_remote_path(path) or (input_dir.url and path.startswith(input_dir.url)):
                flattened_item[index] = _cache_local_path(path, input_dir, cache_data_dir)
            elif (
                input_dir.path
                and isinstance(input_dir.path, str)
                and not input_dir.path.startswith("/teamspace/studios/this_studio")
            ):
                flattened_item[index] = path.replace(input_dir.path, cache_data_dir)
        prepared.append((tree_unflatten(flattened_item, spec), paths))
    return prepared


class FakeQueue:
    """This class enables us to replace multiprocessing Queue when not required and avoid serializing data."""

    def __init__(self) -> None:
        self._index: list[Any] = []
        self._items: list[Any] = []
        self._paths: list[Any] = []

    def add_items(self, index: list[Any], items: list[Any], paths: list[Any]) -> None:
        self._index.extend(index)
        self._items.extend(items)
        self._paths.extend(paths)

    def get(self, *args, **kwargs) -> None:
        try:
            return (self._index.pop(0), self._items.pop(0), self._paths.pop(0))
        except IndexError:
            raise Empty


class BaseWorker:
    """BaseWorker handles data processing using either map or optimize recipes.

    The worker follows this processing pipeline:
    1. Receives input data from ready_to_process_queue (structured via data_recipe.prepare_structure)
    2. If a reader is configured, reads and prepares data using data_recipe.prepare_item
    3. Processes data through either:
       - handle_data_chunk_recipe for optimization tasks (LambdaDataChunkRecipe)
       - handle_data_transform_recipe for mapping tasks (LambdaMapRecipe)
    4. Manages data lifecycle:
       - Uploads processed results
       - Optionally cleans up source data

    This class serves as the core processing unit in distributed data processing pipelines,
    supporting both data transformation and optimization workflows.
    """

    def __init__(
        self,
        worker_index: int,
        num_workers: int,
        node_rank: int,
        msg_queue: Queue,
        data_recipe: "DataRecipe",
        input_dir: Dir,
        output_dir: Dir,
        items: list[Any] | None,
        progress_queue: Queue,
        error_queue: Queue,
        stop_queue: Queue,
        num_downloaders: int,
        num_uploaders: int,
        remove: bool,
        reader: BaseReader | None = None,
        writer_starting_chunk_index: int = 0,
        use_checkpoint: bool = False,
        checkpoint_chunks_info: list[dict[str, Any]] | None = None,
        checkpoint_next_index: int | None = None,
        item_loader: BaseItemLoader | None = None,
        storage_options: dict[str, Any] = {},
        keep_data_ordered: bool = False,
        shared_queue: "Queue | FakeQueue | None" = None,
        using_queue_optimize: bool = False,  # using queues as inputs for optimize fn
        checkpoint_next_chunk_index: int | None = None,
        shared_upload_queue: "Queue | None" = None,
        shared_remove_queue: "Queue | None" = None,
        item_paths: list[list[str] | None] | None = None,
        items_lookup_path: str | None = None,
        shared_write_queue: "Queue | None" = None,
    ) -> None:
        """The BaseWorker is responsible to process the user data."""
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.node_rank = node_rank
        self.msg_queue = msg_queue
        self.data_recipe = data_recipe
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.items = items
        self.num_items = len(items) if items is not None else 0
        self._items_lookup_path = items_lookup_path
        self.shared_write_queue = shared_write_queue
        self.num_downloaders = num_downloaders
        self.num_uploaders = num_uploaders
        self.remove = remove
        self.reader = reader
        self.paths: list[list[str]] = []
        self.remover: Process | Thread | None = None
        self.downloaders: list[Process | Thread] = []
        self.uploaders: list[Process | Thread] = []
        self._node_io = (not keep_data_ordered) or shared_upload_queue is not None or shared_remove_queue is not None
        self.to_download_queues: list[Queue] = []
        self.to_upload_queues: list[Queue] = []
        self.stop_queue = stop_queue
        self.no_downloaders = (
            self.reader is not None
            or (items is None and not using_queue_optimize)
            or (self.input_dir.path is None and self.input_dir.url is None)
        )

        self.keep_data_ordered = keep_data_ordered
        if item_paths is not None:
            self.paths = item_paths

        if not keep_data_ordered or using_queue_optimize:
            assert shared_queue is not None
            self.ready_to_process_queue = shared_queue
        else:
            self.ready_to_process_queue: Queue | FakeQueue = FakeQueue() if self.no_downloaders else Queue()

        self.remove_queue: Queue = shared_remove_queue if shared_remove_queue is not None else Queue()
        if shared_upload_queue is not None:
            self.to_upload_queues = [shared_upload_queue]
        self.progress_queue: Queue = progress_queue
        self.error_queue: Queue = error_queue
        self.item_loader = item_loader
        self._counter = 0
        self._last_time = time()
        self._index_counter = 0
        self.writer_starting_chunk_index: int = writer_starting_chunk_index
        self.use_checkpoint: bool = use_checkpoint
        self.checkpoint_chunks_info: list[dict[str, Any]] | None = checkpoint_chunks_info
        self.checkpoint_next_index: int | None = checkpoint_next_index
        self.checkpoint_next_chunk_index: int | None = checkpoint_next_chunk_index
        self.storage_options = storage_options
        self.using_queue_optimize = using_queue_optimize
        # Explicit (writer_sample_index, key) pairs — index is the same value passed to Cache._add_item.
        self._key_pairs: list[tuple[int, Any]] = []

    def run(self) -> None:
        try:
            self._setup()
            self._loop()
            self._terminate()
        except Exception:
            traceback_format = traceback.format_exc()
            self.error_queue.put(traceback_format)
        self.msg_queue.put_nowait(f"Worker {str(_get_node_rank() * self.num_workers + self.worker_index)} is done.")

    def _setup(self) -> None:
        self._set_environ_variables()
        if self.items is None and self._items_lookup_path:
            with open(self._items_lookup_path, "rb") as handle:
                loaded_items, loaded_paths = pickle.load(handle)  # noqa: S301
            self.items = loaded_items
            self.num_items = len(loaded_items)
            if loaded_paths:
                self.paths = loaded_paths
        self._create_cache()
        self._collect_paths()
        self._start_downloaders()
        self._start_uploaders()
        self._start_remover()

    def _terminate(self) -> None:
        """Make sure all the uploaders, downloaders and removers are terminated."""
        for uploader in self.uploaders:
            if uploader.is_alive():
                uploader.join()

        for downloader in self.downloaders:
            if downloader.is_alive():
                downloader.join()

        if self.remover and self.remover.is_alive():
            self.remover.join()

    def _loop(self) -> None:
        """The main loop of the worker.

        It will get the item index from the `ready_to_process_queue`,
        and send it to the `handle_data_chunk_recipe` or `handle_data_transform_recipe` depending on the recipe type.
        finally, it will upload and remove the data depending on the recipe type.
        """
        num_downloader_finished = 0

        timeout = int(os.getenv("DATA_OPTIMIZER_TIMEOUT", 300))

        timed_out = False  # to avoid infinite waiting, and to know when shared_queue is completely empty
        combined_data = None

        while True:
            try:
                combined_data = self.ready_to_process_queue.get(timeout=timeout)

                if combined_data == ALL_DONE and (self.keep_data_ordered or self.using_queue_optimize):
                    # Ordered and queue-input: one sentinel is shared. Unordered list
                    # optimize already enqueues one ALL_DONE per worker — do not put it back.
                    self.ready_to_process_queue.put(ALL_DONE)
            except Empty:
                timed_out = True

            if combined_data in (None, ALL_DONE) or timed_out:
                num_downloader_finished += 1
                if (
                    timed_out
                    or combined_data == ALL_DONE
                    or (self.keep_data_ordered and num_downloader_finished == self.num_downloaders)
                ):
                    self.msg_queue.put_nowait(
                        f"Worker {str(_get_node_rank() * self.num_workers + self.worker_index)} is terminating."
                    )

                    if isinstance(self.data_recipe, DataChunkRecipe):
                        self._handle_data_chunk_recipe_end()

                    if not self._node_io:
                        if self.to_upload_queues and (self.output_dir.url or self.output_dir.path):
                            for queue in self.to_upload_queues:
                                queue.put(None)
                            for uploader in self.uploaders:
                                uploader.join()

                        if self.remove:
                            assert self.remover
                            self.remove_queue.put(None)
                            self.remover.join()

                    if self.progress_queue:
                        self.progress_queue.put((self.worker_index, self._counter))
                    return
                continue

            if self.using_queue_optimize:
                index = None
                item = combined_data
                paths = None
            elif isinstance(combined_data, int):
                index = combined_data
                assert self.items is not None
                item = self.items[index]
                paths = self.paths[index] if index < len(self.paths) else None
            else:
                assert isinstance(combined_data, tuple), f"Invalid data received from queue {combined_data=}."
                assert len(combined_data) == 3, f"Invalid data received from queue {combined_data=}."

                index, item, paths = combined_data
            if isinstance(self.data_recipe, DataChunkRecipe):
                self._handle_data_chunk_recipe(index, item)
            else:
                self._handle_data_transform_recipe(index, item)

            self._counter += 1

            # Send update after every 1 second.
            # When only few elements are left, send update every second.
            if self.progress_queue and (
                ((time() - self._last_time) > 1 and self._counter < (self.num_items - 20))
                or self._counter + 20 >= self.num_items
            ):
                self.progress_queue.put((self.worker_index, self._counter))
                self._last_time = time()

            if self.remove and self.input_dir.path is not None and self.reader is None and paths is not None:
                self.remove_queue.put(paths)

            try:
                self.stop_queue.get_nowait()
                return
            except Empty:
                pass

    def _set_environ_variables(self) -> None:
        # set the optimizer global rank and world_size
        os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(_get_node_rank() * self.num_workers + self.worker_index)
        os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_workers)

    def _create_cache(self) -> None:
        self.cache_data_dir = _get_cache_data_dir()
        self.cache_chunks_dir = _chunks_dir(self.output_dir)

        if self.shared_write_queue is not None and isinstance(self.data_recipe, DataChunkRecipe):
            return

        if isinstance(self.data_recipe, MapRecipe):
            return

        self.cache = Cache(
            self.cache_chunks_dir,
            chunk_bytes=self.data_recipe.chunk_bytes,
            chunk_size=self.data_recipe.chunk_size,
            compression=self.data_recipe.compression,
            encryption=self.data_recipe.encryption,
            writer_chunk_index=self.writer_starting_chunk_index,
            item_loader=self.item_loader,
            msg_queue=self.msg_queue,
        )
        rank = _get_node_rank() * self.num_workers + self.worker_index
        self.cache._reader._rank = rank
        self.cache._writer._rank = rank

        # return
        if self.use_checkpoint and all(
            [
                self.checkpoint_chunks_info is not None,
                self.checkpoint_next_index is not None,
            ]
        ):
            assert isinstance(self.checkpoint_next_index, int)
            assert isinstance(self.checkpoint_chunks_info, list)

            self.cache._writer._chunks_info = self.checkpoint_chunks_info
            self.cache._writer._chunk_index = _writer_chunk_index_from_checkpoint(
                self.writer_starting_chunk_index,
                self.checkpoint_chunks_info,
                self.checkpoint_next_chunk_index,
            )

    def _try_upload(self, data: str | tuple[str, str] | None) -> None:
        if not data or (self.output_dir.url if self.output_dir.url else self.output_dir.path) is None:
            return

        if isinstance(data, str):
            assert os.path.exists(data), data
        else:
            assert os.path.exists(data[-1]), data

        n_upload = len(self.to_upload_queues)
        if n_upload == 0:
            return
        self.to_upload_queues[self._counter % n_upload].put(data)

    def _collect_paths(self) -> None:
        if self.using_queue_optimize:
            # If using queues as inputs for optimize fn, we don't need to collect paths.
            # User should provide the paths in the queue.
            return

        if not self.keep_data_ordered:
            # Parent feeder owns the ready queue; items are only for index lookup.
            return

        if self.items is None:
            return

        if self.no_downloaders:
            # in queue, put (index, corresponding item, corresponding paths (None in this case))
            if isinstance(self.ready_to_process_queue, FakeQueue):
                self.ready_to_process_queue.add_items(
                    list(range(len(self.items))), self.items, [None for _ in self.items]
                )
            else:
                for index in range(len(self.items)):
                    self.ready_to_process_queue.put((index, self.items[index], None))
            return

        prepared = _prepare_items_and_paths(self.items, self.input_dir, self.cache_data_dir)
        self.items = [item for item, _ in prepared]
        self.paths = [paths or [] for _, paths in prepared]

    def _start_downloaders(self) -> None:
        if self.no_downloaders or self._node_io:
            return

        assert self.items is not None, "Items should be provided to the worker."

        for _ in range(self.num_downloaders):
            to_download_queue: Queue = Queue()
            p = Process(
                target=_download_data_target,
                args=(
                    self.input_dir,
                    self.cache_data_dir,
                    to_download_queue,
                    self.ready_to_process_queue,
                    self.storage_options,
                ),
            )
            p.start()
            self.downloaders.append(p)
            self.to_download_queues.append(to_download_queue)

        for index, paths in enumerate(self.paths):
            self.to_download_queues[index % self.num_downloaders].put((index, self.items[index], paths))

        for downloader_index in range(self.num_downloaders):
            self.to_download_queues[downloader_index].put(None)

    def _start_remover(self) -> None:
        if not self.remove or self._node_io:
            return

        self.remover = Process(
            target=_remove_target,
            args=(
                self.input_dir,
                self.cache_data_dir,
                self.remove_queue,
            ),
        )
        self.remover.start()

    def _start_uploaders(self) -> None:
        if self._node_io or (self.output_dir.path is None and self.output_dir.url is None):
            return
        if _is_local_write_through(self.output_dir) and not isinstance(self.data_recipe, MapRecipe):
            return

        for _ in range(self.num_uploaders):
            to_upload_queue: Queue = Queue()
            p = Process(
                target=_upload_fn,
                args=(
                    to_upload_queue,
                    self.remove_queue,
                    self.cache_chunks_dir,
                    self.output_dir,
                    self.storage_options,
                ),
            )
            p.start()
            self.uploaders.append(p)
            self.to_upload_queues.append(to_upload_queue)

    def _handle_data_chunk_recipe(self, index: int, item: Any) -> None:
        """Used by `optimize fn` to run the user provided fn on each item of the input data,
        and save (write) the output in the cache.
        """
        try:
            current_item = item if self.reader is None else self.reader.read(item)

            # Handle case where StreamingDataLoaderReader returns None (worker exhausted its data)
            if current_item is None:
                return

            item_data_or_generator = self.data_recipe.prepare_item(current_item)
            key_fn = getattr(self.data_recipe, "key_fn", None)
            if self.data_recipe.is_generator:
                for item_data in item_data_or_generator:
                    if item_data is not None:
                        sample_index = self._index_counter
                        key = None
                        if key_fn is not None:
                            from litdata.utilities.keys_index import normalize_key

                            key = normalize_key(key_fn(item_data))
                            self._key_pairs.append((sample_index, key))
                        if self.shared_write_queue is not None:
                            self.shared_write_queue.put((item_data, key))
                        else:
                            chunk_filepath = self.cache._add_item(sample_index, item_data)
                            self._try_upload(chunk_filepath)
                        self._index_counter += 1
            elif item_data_or_generator is not None:
                sample_index = self._index_counter
                key = None
                if key_fn is not None:
                    from litdata.utilities.keys_index import normalize_key

                    key = normalize_key(key_fn(item_data_or_generator))
                    self._key_pairs.append((sample_index, key))
                if self.shared_write_queue is not None:
                    self.shared_write_queue.put((item_data_or_generator, key))
                else:
                    chunk_filepath = self.cache._add_item(sample_index, item_data_or_generator)
                    self._try_upload(chunk_filepath)
                    if self.use_checkpoint:
                        checkpoint_filepath = self.cache.save_checkpoint(inputs_done=self._index_counter)
                        self._try_upload(checkpoint_filepath)
                self._index_counter += 1
        except Exception as e:
            raise RuntimeError(f"Failed processing {item=}; {index=}") from e

    def _handle_data_chunk_recipe_end(self) -> None:
        """Called when the `optimize fn` is done.

        It will save the cache to disk, and upload the chunks to the output directory.
        """
        if self.shared_write_queue is not None:
            return

        chunks_filepaths = self.cache.done()

        if chunks_filepaths and len(self.to_upload_queues):
            n_upload = len(self.to_upload_queues)
            for i, chunk_filepath in enumerate(chunks_filepaths):
                if isinstance(chunk_filepath, str) and os.path.exists(chunk_filepath):
                    self.to_upload_queues[i % n_upload].put(chunk_filepath)

        if getattr(self.data_recipe, "key_fn", None) is not None:
            from litdata.utilities.keys_index import save_rank_keys

            # Keep rank key files in the cache until `_merge_and_upload_keys` runs.
            # Do not `_try_upload` them — upload removes the local file and merge would miss them.
            rank = _get_node_rank() * self.num_workers + self.worker_index
            keys_filepath = os.path.join(self.cache_chunks_dir, f"{rank}.keys.parquet")
            save_rank_keys(keys_filepath, self._key_pairs)

        if self.use_checkpoint and not self.data_recipe.is_generator:
            checkpoint_filepath = self.cache.save_checkpoint(inputs_done=self._index_counter)
            self._try_upload(checkpoint_filepath)

    def _handle_data_transform_recipe(self, index: int, item: Any) -> None:
        """Used by map fn to run the user provided fn on each item of the input data.

        It should not return anything and write directly to the output directory.
        """
        # Don't use a context manager to avoid deleting files that are being uploaded.
        output_dir = tempfile.mkdtemp()
        item = item if self.reader is None else self.reader.read(item)
        is_last = (len(self.items) - 1 == index) if self.keep_data_ordered else False
        item_data = self.data_recipe.prepare_item(item, str(output_dir), is_last)
        if item_data is not None:
            raise ValueError(
                "When using a `MapRecipe`, the `prepare_item` shouldn't return anything."
                " Simply store your files under the output_dir."
            )
        filepaths = []
        for directory, _, filenames in os.walk(output_dir):
            for filename in filenames:
                filepaths.append(os.path.join(directory, filename))

        for filepath in filepaths:
            self._try_upload((output_dir, filepath))


class DataWorkerProcess(BaseWorker, Process):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """The DataWorkerProcess is responsible to process the user data inside processes."""
        BaseWorker.__init__(self, *args, **kwargs)
        Process.__init__(self)


class ChunkWriterProcess(Process):
    """Serialize and write chunks; transform workers only run ``fn``."""

    def __init__(
        self,
        writer_index: int,
        num_writers: int,
        node_rank: int,
        write_queue: Queue,
        error_queue: Queue,
        msg_queue: Queue,
        data_recipe: "DataChunkRecipe",
        output_dir: Dir,
        upload_queue: "Queue | None",
        item_loader: BaseItemLoader | None,
        storage_options: dict[str, Any],
    ) -> None:
        super().__init__()
        self.writer_index = writer_index
        self.num_writers = num_writers
        self.node_rank = node_rank
        self.write_queue = write_queue
        self.error_queue = error_queue
        self.msg_queue = msg_queue
        self.data_recipe = data_recipe
        self.output_dir = output_dir
        self.upload_queue = upload_queue
        self.item_loader = item_loader
        self.storage_options = storage_options

    def run(self) -> None:
        try:
            rank = self.node_rank * self.num_writers + self.writer_index
            os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = str(rank)
            os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_writers)
            chunks_dir = _chunks_dir(self.output_dir)
            cache = Cache(
                chunks_dir,
                chunk_bytes=self.data_recipe.chunk_bytes,
                chunk_size=self.data_recipe.chunk_size,
                compression=self.data_recipe.compression,
                encryption=self.data_recipe.encryption,
                writer_chunk_index=0,
                item_loader=self.item_loader,
                msg_queue=self.msg_queue,
            )
            cache._reader._rank = rank
            cache._writer._rank = rank
            counter = 0
            key_pairs: list[tuple[int, Any]] = []
            while True:
                payload = self.write_queue.get()
                if payload == ALL_DONE:
                    break
                item_data, key = payload
                if key is not None:
                    key_pairs.append((counter, key))
                chunk_filepath = cache._add_item(counter, item_data)
                if chunk_filepath and self.upload_queue is not None:
                    self.upload_queue.put(chunk_filepath)
                counter += 1
            for chunk_filepath in cache.done() or []:
                if self.upload_queue is not None and isinstance(chunk_filepath, str) and os.path.exists(chunk_filepath):
                    self.upload_queue.put(chunk_filepath)
            if getattr(self.data_recipe, "key_fn", None) is not None:
                from litdata.utilities.keys_index import save_rank_keys

                rank = self.node_rank * self.num_writers + self.writer_index
                save_rank_keys(os.path.join(chunks_dir, f"{rank}.keys.parquet"), key_pairs)
        except Exception:
            self.error_queue.put(traceback.format_exc())


@dataclass
class _Result:
    size: int | None = None
    num_bytes: str | None = None
    data_format: str | None = None
    compression: str | None = None
    encryption: Encryption | None = None
    num_chunks: int | None = None
    num_bytes_per_chunk: list[int] | None = None


T = TypeVar("T")


class DataRecipe:
    """Base class for all data recipes.

    It is responsible for preparing the `structure of the data (inputs)`
    and the `item (what is returned by the user fn)`.
    """

    @abstractmethod
    def prepare_structure(self, input_dir: str | None) -> list[T]:
        """Prepare the structure of the data.

        This is the structure of the data that will be used by the worker. (inputs)
        """
        pass

    @abstractmethod
    def prepare_item(self, *args: Any, **kwargs: Any) -> Any:
        """Prepare the item.

        This is the item that will be returned by the user `fn(input)`.
        For `optimize fn`, it will be saved in the cache.
        For `map fn`, it should return none, and should write directly to the output directory.
        """
        pass

    def __init__(self, storage_options: dict[str, Any] = {}) -> None:
        self._name: str | None = None
        self.storage_options = storage_options

    def _done(self, size: int | None, delete_cached_files: bool, output_dir: Dir) -> _Result:
        return _Result(size=size)


class DataChunkRecipe(DataRecipe):
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_bytes: int | str | None = None,
        compression: str | None = None,
        encryption: Encryption | None = None,
        storage_options: dict[str, Any] = {},
        key_fn: Callable[[Any], Any] | None = None,
    ):
        super().__init__(storage_options)
        if chunk_size is not None and chunk_bytes is not None:
            raise ValueError("Either one of the `chunk_size` or the `chunk_bytes` need to be provided.")

        self.chunk_size = chunk_size
        self.chunk_bytes = 1 << 26 if chunk_size is None and chunk_bytes is None else chunk_bytes  # 1<<26 = 64 MB
        self.compression = compression
        self.encryption = encryption
        self.key_fn = key_fn

    @abstractmethod
    def prepare_structure(self, input_dir: str | None) -> list[T]:
        """Return the structure of your data.

        Each element should contain at least a filepath.

        """

    @abstractmethod
    def prepare_item(self, item_metadata: T) -> Any:
        """Returns `prepare_item` method is persisted in chunked binary files."""

    def _done(self, size: int | None, delete_cached_files: bool, output_dir: Dir) -> _Result:
        num_nodes = _get_num_nodes()
        cache_dir = _chunks_dir(output_dir)

        chunks = [file for file in os.listdir(cache_dir) if file.endswith(".bin")] if os.path.isdir(cache_dir) else []
        if chunks and delete_cached_files and output_dir.path is not None and not _is_local_write_through(output_dir):
            raise RuntimeError(f"All the chunks should have been deleted. Found {chunks} in cache: {cache_dir}")

        merge_cache = Cache(cache_dir, chunk_bytes=1)
        node_rank = _get_node_rank()
        merge_cache._merge_no_wait(node_rank if num_nodes > 1 else None, getattr(self, "existing_index", None))

        self._merge_and_upload_keys(output_dir, cache_dir, num_nodes, node_rank)
        self._upload_index(output_dir, cache_dir, num_nodes, node_rank)

        if num_nodes == node_rank + 1:
            config = load_index_file(cache_dir)

            size = sum([c["dim"] if c["dim"] is not None else c["chunk_size"] for c in config["chunks"]])
            num_bytes = sum([c["chunk_bytes"] for c in config["chunks"]])
            if config["config"] is not None:
                data_format = tree_unflatten(
                    config["config"]["data_format"], treespec_loads(config["config"]["data_spec"])
                )
            else:
                data_format = None
            num_chunks = len(config["chunks"])

            # The platform can't store more than 1024 entries.
            # Note: This isn't really used right now, so it is fine to skip if too big.
            num_bytes_per_chunk = [c["chunk_size"] for c in config["chunks"]] if num_chunks < 1024 else []
            return _Result(
                size=size,
                num_bytes=num_bytes,
                data_format=data_format,
                compression=config["config"]["compression"] if config["config"] else None,
                num_chunks=len(config["chunks"]),
                num_bytes_per_chunk=num_bytes_per_chunk,
            )
        return _Result(
            size=size,
        )

    def _merge_and_upload_keys(self, output_dir: Dir, cache_dir: str, num_nodes: int, node_rank: int | None) -> None:
        """Merge per-rank key sidecars and publish ``keys/shard-*.parquet`` next to ``index.json``."""
        if getattr(self, "key_fn", None) is None:
            return

        from litdata.constants import _INDEX_FILENAME
        from litdata.utilities.keys_index import (
            concatenate_key_files,
            enrich_keys_with_chunks,
            has_keys_index,
            list_key_parquet_files,
            merge_rank_key_files,
        )

        def _enrich_if_index(dataset_dir: str) -> None:
            index_path = os.path.join(cache_dir, _INDEX_FILENAME)
            if os.path.isfile(index_path):
                # Close before enrich: enrich rewrites index.json, and Windows cannot
                # replace a path that still has an open handle.
                with open(index_path, encoding="utf-8") as f:
                    index_json = json.load(f)
                enrich_keys_with_chunks(dataset_dir, index_json)

        # Single-node (or per-node partial): merge rank files present in this cache.
        if num_nodes <= 1:
            merged = merge_rank_key_files(cache_dir)
            if merged is None:
                return
            existing = getattr(self, "existing_index", None)
            # Append mode: prepend keys from the previous dataset if present on output_dir.
            if existing is not None and output_dir.path and has_keys_index(output_dir.path):
                concatenate_key_files(
                    list_key_parquet_files(output_dir.path) + list_key_parquet_files(cache_dir),
                    cache_dir,
                )
            _enrich_if_index(cache_dir)
            self._upload_keys_store(output_dir, cache_dir)
            return

        # Multi-node: each node merges local rank keys into ``{node_rank}-keys.parquet``,
        # then the last node concatenates into the final ``keys/`` store.
        assert node_rank is not None
        node_keys_name = f"{node_rank}-keys.parquet"
        merged = merge_rank_key_files(cache_dir, output_filename=node_keys_name)
        if merged is None:
            return
        self._upload_file(output_dir, cache_dir, node_keys_name)

        if num_nodes != node_rank + 1:
            return

        obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
        local_paths: list[str] = []
        for nr in range(num_nodes):
            name = f"{nr}-keys.parquet"
            local_path = os.path.join(cache_dir, name)
            if nr != node_rank:
                remote_base = output_dir.url if output_dir.url else output_dir.path
                assert remote_base
                remote_filepath = os.path.join(remote_base, name)
                if obj.scheme in _SUPPORTED_PROVIDERS:
                    merged_storage_options = construct_storage_options(self.storage_options, output_dir)
                    _wait_for_file_to_exist(remote_filepath, storage_options=merged_storage_options)
                    fs_provider = _get_fs_provider(remote_filepath, merged_storage_options)
                    fs_provider.download_file(remote_filepath, local_path)
                elif output_dir.path and os.path.isdir(output_dir.path):
                    shutil.copyfile(remote_filepath, local_path)
            local_paths.append(local_path)

        concatenate_key_files(local_paths, cache_dir)
        _enrich_if_index(cache_dir)
        self._upload_keys_store(output_dir, cache_dir)

    def _upload_keys_store(self, output_dir: Dir, cache_dir: str) -> None:
        """Upload ``keys/shard-*.parquet`` next to the dataset (layout lives in ``index.json``)."""
        from litdata.constants import _KEYS_DIRNAME
        from litdata.utilities.keys_index import keys_dir, list_shard_files

        if output_dir.path is None and output_dir.url is None:
            return
        local_keys = keys_dir(cache_dir)
        if not os.path.isdir(local_keys):
            return

        rel_files = [os.path.basename(p) for p in list_shard_files(local_keys)]
        obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
        remote_jobs: list[tuple[str, str]] = []
        for name in rel_files:
            local_filepath = os.path.join(local_keys, name)
            if not os.path.isfile(local_filepath):
                continue
            remote_rel = os.path.join(_KEYS_DIRNAME, name)
            if obj.scheme in _SUPPORTED_PROVIDERS:
                assert output_dir.url
                remote_jobs.append((local_filepath, os.path.join(output_dir.url, remote_rel)))
            elif output_dir.path and os.path.isdir(output_dir.path):
                dest_dir = os.path.join(output_dir.path, _KEYS_DIRNAME)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, name)
                if not _same_local_file(local_filepath, dest):
                    shutil.copyfile(local_filepath, dest)
        if remote_jobs:
            _put_files_remote(output_dir, remote_jobs, self.storage_options, cache_dir)

    def _upload_file(self, output_dir: Dir, cache_dir: str, filename: str) -> None:
        if output_dir.path is None and output_dir.url is None:
            return
        local_filepath = os.path.join(cache_dir, filename)
        if not os.path.isfile(local_filepath):
            return
        obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
        if obj.scheme in _SUPPORTED_PROVIDERS:
            assert output_dir.url
            _put_files_remote(
                output_dir,
                [(local_filepath, os.path.join(output_dir.url, filename))],
                self.storage_options,
                cache_dir,
            )
        elif output_dir.path and os.path.isdir(output_dir.path):
            dest = os.path.join(output_dir.path, filename)
            if not _same_local_file(local_filepath, dest):
                shutil.copyfile(local_filepath, dest)

    def _upload_index(self, output_dir: Dir, cache_dir: str, num_nodes: int, node_rank: int | None) -> None:
        """Upload the index file to the remote cloud directory."""
        if output_dir.path is None and output_dir.url is None:
            return

        obj = parse.urlparse(output_dir.url if output_dir.url else output_dir.path)
        if num_nodes > 1:
            local_filepath = os.path.join(cache_dir, f"{node_rank}-{_INDEX_FILENAME}")
        else:
            local_filepath = os.path.join(cache_dir, _INDEX_FILENAME)

        if obj.scheme in _SUPPORTED_PROVIDERS:
            _put_files_remote(
                output_dir,
                [(local_filepath, os.path.join(output_dir.url, os.path.basename(local_filepath)))],
                self.storage_options,
                cache_dir,
            )
        elif output_dir.path and os.path.isdir(output_dir.path):
            dest = os.path.join(output_dir.path, os.path.basename(local_filepath))
            if not _same_local_file(local_filepath, dest):
                shutil.copyfile(local_filepath, dest)

        if num_nodes == 1 or node_rank is None:
            return

        # Merge the index files generated by each node.
        # Note: When using the Data Optimizer, they should be a single process on each node executing this section
        # So no risk to get race condition.
        if num_nodes == node_rank + 1:
            # Merge node shards in the cache, not the write-through output dir,
            # so ``{node}-index.json`` uploads stay in place for other ranks.
            merge_dir = _get_cache_dir() if _is_local_write_through(output_dir) else cache_dir
            os.makedirs(merge_dir, exist_ok=True)
            last_rank = node_rank
            for src_rank in range(num_nodes):
                name = f"{src_rank}-{_INDEX_FILENAME}"
                output_dir_path = output_dir.url if output_dir.url else output_dir.path
                assert output_dir_path
                remote_filepath = os.path.join(output_dir_path, name)
                node_index_filepath = os.path.join(merge_dir, name)
                if src_rank == last_rank and os.path.isfile(os.path.join(cache_dir, name)):
                    remote_filepath = os.path.join(cache_dir, name)
                if obj.scheme in _SUPPORTED_PROVIDERS:
                    if src_rank == last_rank and os.path.isfile(node_index_filepath):
                        continue
                    merged_storage_options = construct_storage_options(self.storage_options, output_dir)
                    _wait_for_file_to_exist(remote_filepath, storage_options=merged_storage_options)
                    fs_provider = _get_fs_provider(remote_filepath, merged_storage_options)
                    fs_provider.download_file(remote_filepath, node_index_filepath)
                elif os.path.isfile(remote_filepath) and not _same_local_file(remote_filepath, node_index_filepath):
                    shutil.copyfile(remote_filepath, node_index_filepath)

            merge_cache = Cache(merge_dir, chunk_bytes=1)
            merge_cache._merge_no_wait()
            self._upload_index(output_dir, merge_dir, 1, None)


class MapRecipe(DataRecipe):
    @abstractmethod
    def prepare_structure(self, input_dir: str | None) -> list[T]:
        """Return the structure of your data.

        Each element should contain at least a filepath.

        """

    @abstractmethod
    def prepare_item(self, item_metadata: T, output_dir: str, is_last: bool) -> None:
        """Use your item metadata to process your files and save the file outputs into `output_dir`."""


class DataProcessor:
    def __init__(
        self,
        input_dir: str | Dir,
        output_dir: str | Dir | None = None,
        num_workers: int | None = None,
        align_chunking: bool = False,
        num_downloaders: int | None = None,
        num_uploaders: int | None = None,
        delete_cached_files: bool = True,
        fast_dev_run: bool | int | None = None,
        random_seed: int | None = 42,
        reorder_files: bool = True,
        weights: list[int] | None = None,
        reader: BaseReader | None = None,
        state_dict: dict[int, int] | None = None,
        use_checkpoint: bool = False,
        item_loader: BaseItemLoader | None = None,
        start_method: str | None = None,
        storage_options: dict[str, Any] = {},
        keep_data_ordered: bool | None = None,
        verbose: bool = True,
        broadcast_paths: bool = False,
    ):
        """Provides an efficient way to process data across multiple machine into chunks to make training faster.

        Args:
            input_dir: The path to where the input data are stored.
            output_dir: The path to where the output data are stored.
            num_workers: The number of worker threads to use.
            align_chunking: Ensures chunk boundaries match the single-worker layout by packing full chunks first
                and placing all remaining items in the final worker.
            num_downloaders: The number of file downloaders to use.
            num_uploaders: The number of file uploaders to use.
            delete_cached_files: Whether to delete the cached files.
            fast_dev_run: Whether to run a quick dev run.
            random_seed: The random seed to be set before shuffling the data.
            reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
                Set this to ``False`` if the order in which samples are processed should be preserved.
            weights: Provide a list of weights associated to the inputs.
                This is used to evenly split the work among the workers.
            reader: Map the inputs to worker inputs and provides a read method to read a slice of the data.
            state_dict: The writer state dict. This is used to decide how to append data to an existing dataset.
            use_checkpoint: Whether to create checkpoints while processing the data, which can be used to resume the
                processing from the last checkpoint if the process is interrupted. (`Default: False`)
            item_loader: The item loader that will be used during loading in StreamingDataset. Determines
                    the format in which the data is stored and optimized for loading.
            start_method: The start method used by python multiprocessing package. Default to spawn unless running
                inside an interactive shell like Ipython.
            storage_options: Storage options for the cloud provider.
            keep_data_ordered: If False (default), shard per node and share one work queue.
                True keeps a static per-worker slice. ``None`` is False unless
                ``use_checkpoint`` or ``align_chunking`` is set.
            verbose: Whether to print the progress & logs of the workers. Defaults to True.
            broadcast_paths: When ``True``, broadcast resolved ``input_dir`` / ``output_dir`` across nodes via
                :func:`~litdata.utilities.broadcast.broadcast_object` (Studio multi-node, when
                ``LIGHTNING_APP_EXTERNAL_URL`` is set). Defaults to ``False``. Automatically enabled when
                ``input_dir`` or ``output_dir`` contains a ``{%strftime}`` time template (so every rank shares
                the same expanded path). When ``False`` and no time template is present, each rank keeps its
                locally resolved path — fine for stable ``s3://`` / connection paths, but unsafe if ranks
                would otherwise expand different timestamps.
        """
        # spawn doesn't work in IPython
        start_method = start_method or ("fork" if in_notebook() else "spawn")

        msg = f"Setting multiprocessing start_method to {start_method}. "
        if in_notebook() and start_method == "fork":
            msg += "Tip: Libraries relying on lock can hang with `fork`. To use `spawn` in notebooks, "
            msg += "move your code to files and import it within the notebook."

        if verbose:
            print(msg)

        multiprocessing.set_start_method(start_method, force=True)

        # Detect time templates on the unresolved path strings (before `_resolve_dir` expands them).
        self.broadcast_paths = broadcast_paths or _has_time_template(input_dir) or _has_time_template(output_dir)

        self.input_dir = _resolve_dir(input_dir)
        self.output_dir = _resolve_dir(output_dir)

        self.num_workers = num_workers or (1 if fast_dev_run else (os.cpu_count() or 1) * 4)
        self.align_chunking = align_chunking
        self.num_downloaders = num_downloaders or 2
        self.num_uploaders = num_uploaders or 1
        self.delete_cached_files = delete_cached_files
        self.fast_dev_run = _get_fast_dev_run() if fast_dev_run is None else fast_dev_run
        self.workers: Any = []
        self.workers_tracker: dict[int, int] = {}
        self.progress_queue: Queue | None = None
        self.error_queue: Queue = Queue()
        self.stop_queues: list[Queue] = []
        self.reorder_files = reorder_files
        self.weights = weights
        self.reader = reader
        self.use_checkpoint = use_checkpoint
        self.checkpoint_chunks_info: list[list[dict[str, Any]]] | None = None
        self.checkpoint_next_index: list[int] | None = None
        self.checkpoint_next_chunk_index: list[int | None] | None = None
        self.item_loader = item_loader
        self.storage_options = storage_options
        self.keep_data_ordered = resolve_keep_data_ordered(
            keep_data_ordered, use_checkpoint=use_checkpoint, align_chunking=align_chunking
        )
        self.shared_queue: Queue | FakeQueue | None = None
        self.shared_download_queue: ThreadQueue | None = None
        self.node_downloaders: list[Thread] = []
        self.node_uploaders: list[Thread] = []
        self.node_removers: list[Thread] = []
        self.shared_upload_queue: Queue | None = None
        self.shared_remove_queue: Queue | None = None
        self.node_user_items: list[Any] | None = None
        self.node_paths: list[list[str] | None] | None = None
        self._queue_input = False
        self._feeder_thread: Thread | None = None
        self._n_node_downloaders = 0
        self._n_node_uploaders = 0
        self.shared_write_queue: Queue | None = None
        self.chunk_writers: list[ChunkWriterProcess] = []
        self.data_recipe: DataRecipe | None = None

        # Queue for routing worker logs to the main process without breaking tqdm output.
        self.msg_queue: Queue = Queue()

        self.state_dict = state_dict or dict.fromkeys(range(self.num_workers), 0)

        if self.reader is not None and self.weights is not None:
            raise ValueError("Either the reader or the weights needs to be defined.")

        if self.broadcast_paths:
            # Align resolved dirs across nodes (needed when `{%strftime}` expands per-rank).
            self.input_dir = broadcast_object("input_dir", self.input_dir, rank=_get_node_rank())
            if self.output_dir:
                self.output_dir = broadcast_object("output_dir", self.output_dir, rank=_get_node_rank())

        if self.output_dir and verbose:
            print(f"Storing the files under {self.output_dir.path if self.output_dir.path else self.output_dir.url}")

        self.random_seed = random_seed
        self.verbose = verbose

    def run(self, data_recipe: DataRecipe) -> None:
        """Triggers the data recipe processing over your dataset."""
        if not isinstance(data_recipe, DataRecipe):
            raise ValueError("The provided value should be a data recipe.")
        if not self.use_checkpoint and isinstance(data_recipe, DataChunkRecipe):
            # clean up checkpoints if not using checkpoints
            self._cleanup_checkpoints()

        t0 = time()
        os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(self.num_workers)
        if self.verbose:
            print(f"Setup started with fast_dev_run={self.fast_dev_run}.")

        # Force random seed to be fixed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Call the setup method of the user
        user_items: list[Any] | StreamingDataLoader | Queue = data_recipe.prepare_structure(
            self.input_dir.path if self.input_dir else None
        )
        if not isinstance(user_items, (list, StreamingDataLoader, multiprocessing.queues.Queue)):
            raise ValueError("The `prepare_structure` should return a list of item metadata or a Queue.")

        if isinstance(user_items, StreamingDataLoader):
            self.reader = StreamingDataLoaderReader(user_items)
            # Each worker owns a dataloader iterator slice keyed by optimizer rank.
            # A shared node queue would only keep the first shard.
            self.keep_data_ordered = True

        if self.reader:
            user_items = self.reader.remap_items(user_items, self.num_workers)

        workers_user_items: list[list[int]] | None = None
        # Unordered: one bin per node; workers steal from a shared node queue.
        map_workers = 1 if not self.keep_data_ordered else self.num_workers

        if isinstance(user_items, list):
            assert isinstance(user_items, list)

            if self.weights is not None:
                if len(self.weights) != len(user_items):
                    raise ValueError("The provided weights length should match the inputs' length.")
                workers_user_items = _map_items_to_workers_weighted(
                    num_workers=map_workers, user_items=user_items, weights=self.weights, file_size=False
                )

            elif self.reorder_files and self.input_dir.path and not _is_studio_fuse_path(self.input_dir.path):
                # TODO: Only do this on node 0, and broadcast the item sizes to the other nodes.
                # Skip FUSE mounts: ``stat`` / ``getsize`` on those paths is extremely slow.
                item_sizes = _get_item_filesizes(user_items, base_path=self.input_dir.path)
                workers_user_items = _map_items_to_workers_weighted(
                    num_workers=map_workers, user_items=user_items, weights=item_sizes
                )
            else:
                if self.align_chunking and data_recipe.chunk_size is None:
                    raise ValueError(
                        "`align_chunking` is set to True, but the `chunk_size` is not defined in the data recipe."
                    )
                workers_user_items = _map_items_to_workers_sequentially(
                    num_workers=map_workers,
                    user_items=user_items,
                    chunk_size=data_recipe.chunk_size if self.align_chunking else None,
                )

            if not self.keep_data_ordered:
                self.node_user_items = workers_user_items[0]
                workers_user_items = None
        else:
            assert isinstance(user_items, multiprocessing.queues.Queue)
            self.shared_queue = user_items
            self._queue_input = True
            workers_user_items = None

        msg = (
            f"Found {len(user_items)} items to process."
            if isinstance(user_items, list)
            else "Using a Queue to process items on demand."
        )
        if self.verbose:
            print(f"Setup finished in {round(time() - t0, 3)} seconds. {msg}")

        if self.use_checkpoint:
            if isinstance(user_items, multiprocessing.queues.Queue) or not self.keep_data_ordered:
                raise ValueError("Checkpoint feature is not supported for Queue based data processing, yet.")

            assert isinstance(workers_user_items, list)

            if hasattr(data_recipe, "is_generator") and data_recipe.is_generator:
                #! TODO: Add checkpointing feature support for generators.
                # Checkpoint feature is not supported for generators for now.
                raise ValueError("Checkpoint feature is not supported for generators, yet.")
            # get the last checkpoint details
            if self.verbose:
                print("Resuming from last saved checkpoint...")
            self._load_checkpoint_config(workers_user_items)

            assert isinstance(self.checkpoint_next_index, list)

            if all(self.checkpoint_next_index[i] == 0 for i in range(self.num_workers)):
                # save the current configuration in the checkpoints.json file
                if self.verbose:
                    print("No checkpoints found. Saving current configuration...")
                self._save_current_config(workers_user_items)
            else:
                # load the last checkpoint details
                assert isinstance(self.checkpoint_next_index, list)
                workers_user_items = [w[self.checkpoint_next_index[i] :] for i, w in enumerate(workers_user_items)]
                if self.verbose:
                    print("Checkpoints loaded successfully.")

        if self.fast_dev_run and not isinstance(user_items, multiprocessing.queues.Queue):
            items_to_keep = self.fast_dev_run if isinstance(self.fast_dev_run, int) else _DEFAULT_FAST_DEV_RUN_ITEMS
            if workers_user_items is not None:
                workers_user_items = [w[:items_to_keep] for w in workers_user_items]
            elif self.node_user_items is not None:
                self.node_user_items = self.node_user_items[:items_to_keep]
            if self.verbose:
                print(f"Fast dev run is enabled. Limiting to {items_to_keep} items per process.")

        self._cleanup_cache()

        if workers_user_items is not None:
            num_items = sum(len(items) for items in workers_user_items)
        elif self.node_user_items is not None:
            num_items = len(self.node_user_items)
        else:
            num_items = -1

        if self.verbose:
            if num_items >= 0:
                print(
                    f"Starting {self.num_workers} workers with {num_items} items."
                    f" The progress bar is only updated when a worker finishes."
                )
            else:
                print(f"Starting {self.num_workers} workers with a Queue to process items on demand.")

        if self.input_dir is None and self.src_resolver is not None and self.input_dir:
            self.input_dir = self.src_resolver(self.input_dir)
            if self.verbose:
                print(f"The remote_dir is `{self.input_dir}`.")

        signal.signal(signal.SIGINT, self._signal_handler)

        self.data_recipe = data_recipe
        if not self.keep_data_ordered:
            self._start_node_io_pools()
            if self.node_user_items is not None:
                self._start_node_work_queue(self.node_user_items)
            if isinstance(data_recipe, DataChunkRecipe):
                self._start_chunk_writers(data_recipe)

        self._create_process_workers(data_recipe, workers_user_items)

        if self.verbose:
            print("Workers are ready ! Starting data processing...")

        current_total = 0
        show_pbar = bool(self.verbose and _TQDM_AVAILABLE)
        pbar = None
        if show_pbar:
            from tqdm.auto import tqdm as _tqdm

            pbar = _tqdm(
                desc="Progress",
                total=num_items if num_items >= 0 else None,
                smoothing=0,
                position=-1,
                mininterval=1,
                leave=True,
                dynamic_ncols=True,
            )
        num_nodes = _get_num_nodes()
        node_rank = _get_node_rank()
        total_num_items = len(user_items) if isinstance(user_items, list) else -1

        while True:
            if self.verbose:
                flush_msg_queue(self.msg_queue, pbar)

            # Exit early if all the workers are done.
            # This means either there were some kinda of errors, or optimize function was very small.
            if all(not w.is_alive() for w in self.workers):
                try:
                    error = self.error_queue.get_nowait()
                    self._exit_on_error(error)
                except Empty:
                    if self.verbose:
                        print("All workers are done. Exiting!")
                    break

            # Do not block on the error queue — that delayed progress by a full poll.
            try:
                error = self.error_queue.get_nowait()
                self._exit_on_error(error)
            except Empty:
                pass

            assert self.progress_queue
            try:
                index, counter = self.progress_queue.get(timeout=_PARENT_QUEUE_POLL_S)
            except Empty:
                continue
            self.workers_tracker[index] = counter
            new_total = sum(self.workers_tracker.values())

            if pbar is not None:
                pbar.update(new_total - current_total)

            current_total = new_total
            if current_total == num_items:
                for w in self.workers:
                    if w.is_alive():
                        w.join()
                break

            if _IS_IN_STUDIO and node_rank == 0 and _ENABLE_STATUS:
                with open("status.json", "w") as f:
                    json.dump({"progress": str(100 * current_total * num_nodes / total_num_items) + "%"}, f)

        if self.verbose:
            flush_msg_queue(self.msg_queue, pbar)

        if pbar is not None:
            pbar.clear()
            pbar.close()

        if self._feeder_thread is not None:
            self._feeder_thread.join(timeout=5)
        self._stop_chunk_writers()
        self._stop_node_io_pools()

        if self.verbose:
            print("Workers are finished.")
        size = self.num_workers if workers_user_items is not None or self.node_user_items is not None else None
        data_recipe._done(size, self.delete_cached_files, self.output_dir)

        if self.verbose:
            print("Finished data processing!")
        if self.use_checkpoint and isinstance(data_recipe, DataChunkRecipe):
            # clean up checkpoints
            self._cleanup_checkpoints()

    def _exit_on_error(self, error: str) -> None:
        for w in self.workers:
            # w.join(0)
            w.terminate()  # already error has occurred. So, no benefit of processing further.
        for writer in self.chunk_writers:
            if writer.is_alive():
                writer.terminate()
        if self._feeder_thread is not None:
            self._feeder_thread.join(timeout=1)
        raise RuntimeError(f"We found the following error {error}.")

    def _queue_prefetch(self) -> int:
        return _prefetch_maxsize(self.num_workers, self.node_user_items)

    def _n_download_threads(self) -> int:
        return max(self.num_downloaders, min(4, self.num_workers))

    def _n_upload_threads(self) -> int:
        if self.output_dir.path is None and self.output_dir.url is None:
            return 0
        # Chunk write-through already lands ``chunk-*.bin`` in the output dir.
        # ``map`` still writes into a temp dir and must copy those files out.
        if _is_local_write_through(self.output_dir) and not isinstance(getattr(self, "data_recipe", None), MapRecipe):
            return 0
        return max(self.num_uploaders, 2)

    def _n_chunk_writers(self) -> int:
        # Opt-in: shipping samples to a writer process is a win when serialize/compress
        # dominates ``fn``. The local hash bench regresses if this is always on.
        if os.getenv("LITDATA_OPTIMIZE_SPLIT_WRITERS", "0") not in {"1", "true", "True"}:
            return 0
        if self.keep_data_ordered or self.use_checkpoint or self._queue_input:
            return 0
        if self.state_dict and any(self.state_dict.values()):
            return 0
        if self.num_workers <= 1:
            return 0
        return 2 if self.num_workers >= 4 else 1

    def _start_chunk_writers(self, data_recipe: DataChunkRecipe) -> None:
        n_writers = self._n_chunk_writers()
        if n_writers <= 0:
            return
        self.shared_write_queue = Queue(maxsize=max(8, 2 * self.num_workers))
        for writer_idx in range(n_writers):
            writer = ChunkWriterProcess(
                writer_idx,
                n_writers,
                _get_node_rank(),
                self.shared_write_queue,
                self.error_queue,
                self.msg_queue,
                data_recipe,
                self.output_dir,
                self.shared_upload_queue,
                self.item_loader,
                self.storage_options,
            )
            writer.start()
            self.chunk_writers.append(writer)

    def _stop_chunk_writers(self) -> None:
        if self.shared_write_queue is None:
            return
        for _ in self.chunk_writers:
            self.shared_write_queue.put(ALL_DONE)
        for writer in self.chunk_writers:
            if writer.is_alive():
                writer.join(timeout=60)

    def _start_io_thread(self, fn: Callable[..., None], *args: Any) -> Thread:
        thread = Thread(target=_io_thread_target, args=(fn, self.error_queue, *args), daemon=True)
        thread.start()
        return thread

    def _start_node_io_pools(self) -> None:
        """Node-level I/O threads. Writers stay as processes; they do not spawn children."""
        prefetch = self._queue_prefetch()
        if self.shared_queue is None:
            self.shared_queue = Queue(maxsize=prefetch)
        cache_data_dir = _get_cache_data_dir()
        cache_chunks_dir = _chunks_dir(self.output_dir)

        if self.delete_cached_files and (self.input_dir.path or self.input_dir.url):
            self.shared_remove_queue = Queue()
            self.node_removers.append(
                self._start_io_thread(_remove_target, self.input_dir, cache_data_dir, self.shared_remove_queue)
            )

        self._n_node_uploaders = self._n_upload_threads()
        if self._n_node_uploaders:
            self.shared_upload_queue = Queue(maxsize=max(4, 2 * self._n_node_uploaders))
            for _ in range(self._n_node_uploaders):
                self.node_uploaders.append(
                    self._start_io_thread(
                        _upload_fn,
                        self.shared_upload_queue,
                        self.shared_remove_queue,
                        cache_chunks_dir,
                        self.output_dir,
                        self.storage_options,
                    )
                )

    def _stop_node_io_pools(self) -> None:
        if self.shared_download_queue is not None:
            for _ in range(self._n_node_downloaders):
                self.shared_download_queue.put(None)
        if self.shared_upload_queue is not None:
            for _ in range(self._n_node_uploaders):
                self.shared_upload_queue.put(None)
        for thread in (*self.node_downloaders, *self.node_uploaders):
            if thread.is_alive():
                thread.join(timeout=30)
        # Removers must see every path uploaders enqueued on the final flush.
        if self.shared_remove_queue is not None:
            self.shared_remove_queue.put(None)
        for thread in self.node_removers:
            if thread.is_alive():
                thread.join(timeout=30)

    def _start_node_work_queue(self, node_items: list[Any]) -> None:
        """Feed this node's items into shared download/ready queues for all local workers."""
        prefetch = self._queue_prefetch()
        if self.shared_queue is None:
            self.shared_queue = Queue(maxsize=prefetch)
        cache_data_dir = _get_cache_data_dir()
        needs_download = _dir_needs_download(self.input_dir, self.reader) or any(
            isinstance(el, str) and _is_remote_path(el) for item in node_items for el in tree_flatten(item)[0]
        )
        prepared = (
            _prepare_items_and_paths(node_items, self.input_dir, cache_data_dir)
            if needs_download
            else [(item, None) for item in node_items]
        )
        self.node_user_items = [item for item, _ in prepared]
        self.node_paths = [paths for _, paths in prepared]

        if needs_download:
            self.shared_download_queue = ThreadQueue(maxsize=prefetch)
            self._n_node_downloaders = self._n_download_threads()
            for _ in range(self._n_node_downloaders):
                self.node_downloaders.append(
                    self._start_io_thread(
                        _download_data_target,
                        self.input_dir,
                        cache_data_dir,
                        self.shared_download_queue,
                        self.shared_queue,
                        self.storage_options,
                        False,
                        True,
                    )
                )

            def _feed_downloads() -> None:
                assert self.shared_download_queue is not None
                assert self.shared_queue is not None
                for index, (item, paths) in enumerate(prepared):
                    self.shared_download_queue.put((index, item, paths or []))
                for _ in range(self._n_node_downloaders):
                    self.shared_download_queue.put(None)
                for thread in self.node_downloaders:
                    thread.join()
                for _ in range(self.num_workers):
                    self.shared_queue.put(ALL_DONE)

            self._feeder_thread = Thread(target=_feed_downloads, daemon=True)
            self._feeder_thread.start()
            return

        def _feed_ready() -> None:
            assert self.shared_queue is not None
            for index, _ in enumerate(prepared):
                self.shared_queue.put(index)
            for _ in range(self.num_workers):
                self.shared_queue.put(ALL_DONE)

        self._feeder_thread = Thread(target=_feed_ready, daemon=True)
        self._feeder_thread.start()

    def _create_process_workers(
        self, data_recipe: DataRecipe, workers_user_items: list[list[Any]] | None = None
    ) -> None:
        if not self.keep_data_ordered and workers_user_items is not None and self.shared_queue is None:
            self.shared_queue = Queue()

        self.progress_queue = Queue()
        workers: list[DataWorkerProcess] = []
        stop_queues: list[Queue] = []
        items_lookup_path: str | None = None
        if not self.keep_data_ordered and self.node_user_items is not None:
            items_lookup_path = os.path.join(_get_cache_dir(), f"node-{_get_node_rank()}-items.pkl")
            with open(items_lookup_path, "wb") as handle:
                pickle.dump((self.node_user_items, self.node_paths), handle, protocol=pickle.HIGHEST_PROTOCOL)
        for worker_idx in range(self.num_workers):
            worker_user_items = workers_user_items[worker_idx] if workers_user_items is not None else None
            stop_queues.append(Queue())
            worker = DataWorkerProcess(
                worker_idx,
                self.num_workers,
                _get_node_rank(),
                self.msg_queue,
                data_recipe,
                self.input_dir,
                self.output_dir,
                worker_user_items,
                self.progress_queue,
                self.error_queue,
                stop_queues[-1],
                self.num_downloaders,
                self.num_uploaders,
                self.delete_cached_files,
                self.reader,
                self.state_dict[worker_idx],
                self.use_checkpoint,
                self.checkpoint_chunks_info[worker_idx] if self.checkpoint_chunks_info else None,
                self.checkpoint_next_index[worker_idx] if self.checkpoint_next_index else None,
                self.item_loader,
                self.storage_options,
                self.keep_data_ordered,
                self.shared_queue,
                using_queue_optimize=self._queue_input,
                checkpoint_next_chunk_index=(
                    self.checkpoint_next_chunk_index[worker_idx] if self.checkpoint_next_chunk_index else None
                ),
                shared_upload_queue=self.shared_upload_queue,
                shared_remove_queue=self.shared_remove_queue,
                item_paths=None if items_lookup_path else self.node_paths,
                items_lookup_path=items_lookup_path,
                shared_write_queue=self.shared_write_queue,
            )
            worker.start()
            workers.append(worker)

        # Note: Don't store within the loop as weakref aren't serializable
        self.workers = workers
        self.stop_queues = stop_queues

    def _signal_handler(self, signal: Any, frame: Any) -> None:
        """On termination, we stop all the processes to avoid leaking RAM."""
        for stop_queue in self.stop_queues:
            stop_queue.put(None)
        for w in self.workers:
            w.join(0)
        os._exit(0)

    def _cleanup_cache(self) -> None:
        cache_dir = _get_cache_dir()

        # Cleanup the cache dir folder to avoid corrupted files from previous run to be there.
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        os.makedirs(cache_dir, exist_ok=True)

        cache_data_dir = _get_cache_data_dir()

        # Cleanup the cache data folder to avoid corrupted files from previous run to be there.
        if os.path.exists(cache_data_dir):
            shutil.rmtree(cache_data_dir, ignore_errors=True)

        os.makedirs(cache_data_dir, exist_ok=True)

    def _cleanup_checkpoints(self) -> None:
        if not isinstance(self.output_dir, Dir):
            raise ValueError("The provided output_dir isn't a Dir Object.")

        if self.output_dir.url is None:
            # this is a local directory
            if self.output_dir.path is None:
                return

            if os.path.exists(self.output_dir.path):
                # clear the checkpoints
                with suppress(FileNotFoundError):
                    shutil.rmtree(os.path.join(self.output_dir.path, ".checkpoints"))

            return

        obj = parse.urlparse(self.output_dir.url)
        if obj.scheme not in _SUPPORTED_PROVIDERS:
            not_supported_provider(self.output_dir.url)

        prefix = self.output_dir.url.rstrip("/") + "/"
        checkpoint_prefix = os.path.join(prefix, ".checkpoints")
        merged_storage_options = construct_storage_options(self.storage_options, self.output_dir)
        fs_provider = _get_fs_provider(self.output_dir.url, merged_storage_options)
        fs_provider.delete_file_or_directory(checkpoint_prefix)

    def _save_current_config(self, workers_user_items: list[list[Any]]) -> None:
        if not self.use_checkpoint:
            return

        # save the current configuration in the config.json file
        config = {
            "num_workers": self.num_workers,
            "workers_user_items": workers_user_items,
        }

        try:
            if self.output_dir.url is None:
                assert self.output_dir.path

                if not os.path.exists(os.path.join(self.output_dir.path, ".checkpoints")):
                    os.makedirs(os.path.join(self.output_dir.path, ".checkpoints"))

                with open(os.path.join(self.output_dir.path, ".checkpoints", "config.json"), "w") as f:
                    json.dump(config, f)

                return

            obj = parse.urlparse(self.output_dir.url)

            if obj.scheme not in _SUPPORTED_PROVIDERS:
                not_supported_provider(self.output_dir.url)
            merged_storage_options = construct_storage_options(self.storage_options, self.output_dir)
            fs_provider = _get_fs_provider(self.output_dir.url, merged_storage_options)

            prefix = self.output_dir.url.rstrip("/") + "/" + ".checkpoints/"

            # write config.json file to temp directory and upload it to the cloud provider
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_name = os.path.join(temp_dir, "config.json")
                with open(temp_file_name, "w") as f:
                    json.dump(config, f)
                fs_provider.upload_file(
                    temp_file_name,
                    os.path.join(prefix, "config.json"),
                )
        except Exception:
            logger.exception("Failed to persist optimize checkpoint config.json")
            raise

    def _load_checkpoint_config(self, workers_user_items: list[list[Any]]) -> None:
        if not self.use_checkpoint:
            return

        default_chunk_info: list[dict[str, Any]] = []

        self.checkpoint_chunks_info = [default_chunk_info for _ in range(self.num_workers)]
        self.checkpoint_next_index = [0 for _ in range(self.num_workers)]
        self.checkpoint_next_chunk_index = [None for _ in range(self.num_workers)]

        if self.output_dir.url is None:
            assert self.output_dir.path

            if not os.path.exists(os.path.join(self.output_dir.path, ".checkpoints")):
                return

            if not os.path.exists(os.path.join(self.output_dir.path, ".checkpoints", "config.json")):
                # if the config.json file doesn't exist, we don't have any checkpoint saved
                return

            with open(os.path.join(self.output_dir.path, ".checkpoints", "config.json")) as f:
                config = json.load(f)

            if config["num_workers"] != self.num_workers:
                raise ValueError(
                    "The number of workers in the checkpoints doesn't match the current number of workers."
                )

            if config["workers_user_items"] != workers_user_items:
                raise ValueError("Existing checkpoints are not compatible with the current configuration.")

            checkpoint_file_names = [f"checkpoint-{worker_idx}.json" for worker_idx in range(self.num_workers)]

            for i, checkpoint_file_name in enumerate(checkpoint_file_names):
                if not os.path.exists(os.path.join(self.output_dir.path, ".checkpoints", checkpoint_file_name)):
                    # if the checkpoint file doesn't exist, we don't have any checkpoint saved for this worker
                    continue

                with open(os.path.join(self.output_dir.path, ".checkpoints", checkpoint_file_name)) as f:
                    checkpoint = json.load(f)

                self.checkpoint_chunks_info[i], self.checkpoint_next_index[i], self.checkpoint_next_chunk_index[i] = (
                    _resume_fields_from_checkpoint(checkpoint)
                )
            return

        obj = parse.urlparse(self.output_dir.url)

        if obj.scheme not in _SUPPORTED_PROVIDERS:
            not_supported_provider(self.output_dir.url)

        prefix = self.output_dir.url.rstrip("/") + "/" + ".checkpoints/"

        # Delete all the files (including the index file in overwrite mode)

        # download all the checkpoint files in tempdir and read them
        with tempfile.TemporaryDirectory() as temp_dir:
            merged_storage_options = construct_storage_options(self.storage_options, self.output_dir)
            fs_provider = _get_fs_provider(self.output_dir.url, merged_storage_options)
            saved_file_dir = fs_provider.download_directory(prefix, temp_dir)

            if not os.path.exists(os.path.join(saved_file_dir, "config.json")):
                # if the config.json file doesn't exist, we don't have any checkpoint saved
                return

            # read the config.json file
            with open(os.path.join(saved_file_dir, "config.json")) as f:
                config = json.load(f)

            if config["num_workers"] != self.num_workers:
                raise ValueError(
                    "The number of workers in the checkpoints doesn't match the current number of workers."
                )

            if config["workers_user_items"] != workers_user_items:
                raise ValueError("Existing checkpoints are not compatible with the current configuration.")

            checkpoint_file_names = [f"checkpoint-{worker_idx}.json" for worker_idx in range(self.num_workers)]

            for i, checkpoint_file_name in enumerate(checkpoint_file_names):
                if not os.path.exists(os.path.join(saved_file_dir, checkpoint_file_name)):
                    # if the checkpoint file doesn't exist, we don't have any checkpoint saved for this worker
                    continue

                with open(os.path.join(saved_file_dir, checkpoint_file_name)) as f:
                    checkpoint = json.load(f)

                self.checkpoint_chunks_info[i], self.checkpoint_next_index[i], self.checkpoint_next_chunk_index[i] = (
                    _resume_fields_from_checkpoint(checkpoint)
                )
        return


def _resume_fields_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int | None]:
    chunks = checkpoint["chunks"]
    inputs_done = int(checkpoint.get("inputs_done", checkpoint["done_till_index"]))
    next_chunk_index = int(checkpoint["next_chunk_index"]) if "next_chunk_index" in checkpoint else None
    return chunks, inputs_done, next_chunk_index


def _writer_chunk_index_from_checkpoint(
    writer_starting_chunk_index: int,
    checkpoint_chunks: list[dict[str, Any]] | None,
    checkpoint_next_chunk_index: int | None,
) -> int:
    """Absolute next chunk file index for the writer after loading a checkpoint.

    ``next_chunk_index`` is already absolute (includes append offset). Older checkpoints
    only stored this-run ``chunks``; continue from ``writer_starting_chunk_index + len(chunks)``.
    Workers with no checkpoint file keep the append starting index.
    """
    if checkpoint_next_chunk_index is not None:
        return checkpoint_next_chunk_index
    n_chunks = len(checkpoint_chunks or [])
    return writer_starting_chunk_index + n_chunks


def in_notebook() -> bool:
    """Returns ``True`` if the module is running in IPython kernel, ``False`` if in IPython or other Python
    shell.
    """
    return "ipykernel" in sys.modules


def flush_msg_queue(msg_queue: Queue, pbar: Any | None = None):
    """Flush messages from a queue and print them without breaking the tqdm progress bar.

    This function drains all available messages from the given queue and prints them.
    If a tqdm progress bar is provided, it temporarily clears and restores the bar
    to avoid visual glitches during printing.

    Args:
        msg_queue (Queue): The queue containing log or status messages.
        pbar (Optional[tqdm]): The tqdm progress bar to preserve formatting. Optional.
    """
    # check if there're msgs in the msg queue
    msgs = []
    while True:
        try:
            msg = msg_queue.get_nowait()
            msgs.append(msg)
        except Empty:
            break
    if len(msgs) > 0:
        if pbar is not None:
            pbar.clear()  # clear the previous progress bar
        for msg in msgs:
            print(msg)
        if pbar is not None:
            pbar.display()  # display the progress bar again

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

import glob
import logging
import os
import sys
import traceback
import warnings
from collections import deque
from contextlib import suppress
from datetime import datetime
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import sleep, time
from typing import Any

import numpy as np
from filelock import FileLock, Timeout

from litdata.constants import _DEBUG
from litdata.debugger import CAT_CRASH, CAT_DOWNLOAD, CAT_READ, emit_trace, trace_span
from litdata.streaming.async_prefetch import (
    adaptive_pre_download,
    async_chunk_prefetch_enabled,
    async_download_concurrency,
    close_thread_event_loop,
    download_chunk_indexes_concurrently,
)
from litdata.streaming.config import ChunksConfig, Interval
from litdata.streaming.item_loader import BaseItemLoader, ParquetLoader, PyTreeLoader, TokensLoader
from litdata.streaming.posix_fast import advise_willneed, mean_chunk_bytes, posix_prefetch_fits_ram, posix_safe_keep
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer, _get_serializers
from litdata.streaming.timing import StreamingTimingStats
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.format import _resolve_max_cache_size

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")


logger = logging.getLogger("litdata.streaming.reader")


_END_TOKEN = "END"  # noqa: S105

# Note: The timeout here should not be too short. We need to prevent the caller from aggressively
# querying the queue and consuming too many CPU cycles.
_DEFAULT_TIMEOUT = 0.1
_LONG_DEFAULT_TIMEOUT = 5
# Reconcile the advisory cache-byte counter with a directory scan every N successful deletes.
_CACHE_SIZE_RECONCILE_EVERY = 8


class PrepareChunksThread(Thread):
    """Download chunks for a worker ahead of the reader.

    Default path downloads one chunk at a time (sync). For remote datasets,
    async chunk prefetch is **on by default** (override with
    ``LITDATA_ASYNC_CHUNK_PREFETCH=0``): pending indexes are drained up to the
    prefetch budget and downloaded concurrently via
    :mod:`litdata.streaming.async_prefetch`.

    Asyncio is scoped to **remote chunk IO overlap** only — not a fully async
    ``StreamingDataLoader`` / training loop.
    """

    def __init__(
        self,
        config: ChunksConfig,
        item_loader: BaseItemLoader,
        distributed_env: _DistributedEnv,
        max_cache_size: int | None = None,
        max_pre_download: int = 2,
        rank: int | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._item_loader = item_loader
        # Async gather needs enough in-flight slots to overlap RTT; raise the
        # floor when async prefetch is active (real-S3 benches: 2→4).
        self._max_pre_download = adaptive_pre_download(
            max_pre_download, remote_dir=config._remote_dir, chunks=config._chunks
        )
        self._pre_download_counter = 0
        self._distributed_env = distributed_env
        self._worker_env = _WorkerEnv.detect()

        self._chunks_index_to_be_deleted: deque[int] = deque()
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: Queue = Queue()
        self._to_delete_queue: Queue = Queue()
        self._force_stop_event = Event()
        self._end_requested = False

        # TODO: Find a real fix to this problem
        self._force_download_queue: Queue = Queue()

        # Per-chunk readiness signals for the in-process item-loader wait loop.
        self._chunk_ready: dict[int, Event] = {}
        self._chunk_ready_lock = Lock()

        self._rank = rank
        # Set when ``run()`` dies so the item-loader wait loop can fail fast with
        # the real exception instead of timing out as ``FileNotFoundError``.
        self._error: BaseException | None = None

        # Check whether a dataset slice fits on the node
        num_bytes_per_nodes = self._config.num_bytes // self._distributed_env.num_nodes
        self._delete_chunks_when_processed = num_bytes_per_nodes > max_cache_size if max_cache_size else False

        if _DEBUG and distributed_env.global_rank == 0 and self._worker_env.rank == 0:
            print(f"Delete chunks when used: {self._delete_chunks_when_processed}")

        self._has_exited = False
        # Advisory cache size for eviction decisions (avoid scanning the dir on every delete).
        self._approx_cache_bytes = 0
        self._cache_bytes_initialized = False
        self._deletes_since_reconcile = 0
        self._last_cache_reconcile_s = 0.0
        self._timing = StreamingTimingStats.instance()
        # Keep peak disk near ``max_cache_size`` under multi-worker prefetch.
        self._cap_pre_download_for_cache_budget()

    def _async_prefetch(self) -> bool:
        """True when this prepare thread should batch-download via asyncio."""
        if not async_chunk_prefetch_enabled(self._config._remote_dir):
            return False
        # One tiny chunk: asyncio/obstore startup is larger than the GET (sst2-sized).
        chunks = self._config._chunks or []
        return not (len(chunks) == 1 and int(chunks[0].get("chunk_bytes") or 0) < 8 * 1024 * 1024)

    def _async_gather_width(self) -> int:
        """How many queued chunk indexes to download together."""
        if not self._async_prefetch():
            return 1
        return max(1, min(self._max_pre_download, async_download_concurrency(self._max_pre_download)))

    def _free_prefetch_slots(self) -> int:
        return max(0, self._max_pre_download - self._pre_download_counter)

    def _should_start_download(self, *, over_budget: bool) -> bool:
        """True when we should pull work from the download queue.

        Use every free slot. Waiting for a gather-sized hole deadlocks when the
        reader is blocked on the next undownloaded chunk (deletes are only queued
        after that load succeeds). Overlap still happens when several slots are
        free: ``_run_loop`` drains the queue up to gather width.
        """
        del over_budget
        return self._free_prefetch_slots() > 0

    def get_ready_event(self, chunk_index: int) -> Event:
        """Return (creating if needed) the readiness event for ``chunk_index``."""
        with self._chunk_ready_lock:
            event = self._chunk_ready.get(chunk_index)
            if event is None:
                event = Event()
                self._chunk_ready[chunk_index] = event
            return event

    def mark_chunk_ready(self, chunk_index: int) -> None:
        """Signal that ``chunk_index`` is fully downloaded (and decompressed if needed)."""
        self.get_ready_event(chunk_index).set()

    def clear_chunk_ready(self, chunk_index: int) -> None:
        """Clear readiness after a chunk is deleted so waiters block for the next download."""
        with self._chunk_ready_lock:
            event = self._chunk_ready.get(chunk_index)
            if event is not None:
                event.clear()

    def download(self, chunk_indexes: list[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_download_queue.put(chunk_index)

    def delete(self, chunk_indexes: list[int]) -> None:
        """Receive the list of the chunk indices to delete for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_delete_queue.put(chunk_index)

    def _remaining_locks(self, chunkpath: str) -> int:
        countpath = chunkpath + ".cnt"
        if not os.path.exists(countpath):
            return 0
        with open(countpath) as count_f:
            try:
                return int(count_f.read().strip())
            except Exception:
                return 1

    def _decrement_local_lock(self, chunk_index: int) -> int:
        """Remove a count from the local lock, return the remaining count.

        Delegates to ``ChunksConfig`` so the reader can also release eagerly-acquired locks during
        teardown without the prefetch thread.
        """
        return self._config.decrement_local_lock(chunk_index)

    def _cleanup_download_locks(self, chunk_filepath: str, chunk_index: int) -> None:
        """Remove stale download lock files for a chunk.

        Download lock files (e.g. ``chunk-0-3.zstd.bin.lock``) are FileLock artifacts created
        during download. They are safe to remove once the chunk exists locally, regardless of
        the refcount held in ``.cnt`` files.  Reference-count lock files (``.cnt.lock``) are
        excluded because they may still be needed by concurrent refcount operations.

        """
        base_name = os.path.basename(chunk_filepath)
        base_prefix = os.path.splitext(base_name)[0]
        cache_dir = os.path.dirname(chunk_filepath)
        pattern = os.path.join(cache_dir, f"{base_prefix}*.lock")
        matched_locks = [p for p in glob.glob(pattern) if not p.endswith(".cnt.lock")]
        if matched_locks:
            logger.debug(f"_apply_delete({chunk_index}): glob matched {matched_locks}")
        for lock_path in matched_locks:
            try:
                os.remove(lock_path)
                logger.debug(f"_apply_delete({chunk_index}): removed {lock_path}")
            except (FileNotFoundError, PermissionError) as e:
                logger.warning(f"_apply_delete({chunk_index}): failed to remove {lock_path}: {e}")
            except Exception as e:
                logger.warning(f"_apply_delete({chunk_index}): unexpected error removing {lock_path}: {e}")

    def _chunk_byte_size(self, chunk_index: int) -> int:
        """Best-effort decompressed chunk size for advisory cache accounting."""
        chunk_filepath, _, filesize_bytes = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        filename = os.path.basename(chunk_filepath)
        return int(self._config.filename_to_size_map.get(filename, filesize_bytes))

    def _reconcile_cache_bytes(self) -> None:
        try:
            self._approx_cache_bytes = _get_folder_size(self._config._cache_dir, self._config)
            self._cache_bytes_initialized = True
            self._deletes_since_reconcile = 0
        except Exception as e:
            # Advisory only — never fail the prefetch loop because of a size scan.
            logger.debug(f"_reconcile_cache_bytes failed: {e}")

    def _ensure_cache_bytes(self) -> None:
        if not self._cache_bytes_initialized:
            self._reconcile_cache_bytes()

    def _note_chunk_added(self, chunk_index: int) -> None:
        try:
            self._ensure_cache_bytes()
            self._approx_cache_bytes += self._chunk_byte_size(chunk_index)
        except Exception as e:
            logger.debug(f"_note_chunk_added({chunk_index}) failed: {e}")

    def _note_chunk_removed(self, chunk_index: int) -> None:
        try:
            self._ensure_cache_bytes()
            self._approx_cache_bytes = max(0, self._approx_cache_bytes - self._chunk_byte_size(chunk_index))
            self._deletes_since_reconcile += 1
            if self._deletes_since_reconcile >= _CACHE_SIZE_RECONCILE_EVERY:
                self._reconcile_cache_bytes()
        except Exception as e:
            logger.debug(f"_note_chunk_removed({chunk_index}) failed: {e}")

    def _apply_delete(self, chunk_index: int, skip_lock: bool = False, *, release_slot: bool = True) -> None:
        """Inform the item loader of the chunk to delete.

        ``release_slot=False`` keeps the shared disk-slot reservation (used by
        force-redownload, which replaces the file in place).
        """
        logger.debug(f"_apply_delete({chunk_index}, skip_lock={skip_lock}) called")
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

        # A chunk is deleted only once its reference count reaches zero, i.e. every worker that
        # will read it has finished with it. Shared chunks are reference-counted *eagerly* (the
        # reader increments them before any reading begins, see BinaryReader._acquire_shared_locks),
        # so a zero count here reliably means "no worker still needs this chunk" — closing the race
        # where a fast worker deleted a shared chunk a slower co-worker had not yet incremented.
        # `skip_lock` is the force-redownload path (a worker re-fetching its own chunk).
        if not skip_lock:
            remaining_locks = self._remaining_locks(chunk_filepath)
            if remaining_locks > 0:  # Can't delete this, something has it
                logger.debug(f"_apply_delete({chunk_index}): skipping data deletion, remaining_locks={remaining_locks}")
                if _DEBUG:
                    print(f"Skip delete {chunk_filepath} by {self._rank or 0}, current lock count: {remaining_locks}")
                self._cleanup_download_locks(chunk_filepath, chunk_index)
                return

        if _DEBUG:
            with open(chunk_filepath + ".tmb", "w+") as tombstone_file:
                tombstone_file.write(f"Deleted {chunk_filepath} by {self._rank or 0}.")

        file_existed = os.path.exists(chunk_filepath)
        try:
            self._item_loader.delete(chunk_index, chunk_filepath)
            compressor = getattr(self._config, "_compressor_name", None)
            basename = os.path.basename(chunk_filepath)
            if compressor and chunk_filepath.endswith(".bin") and f".{compressor}." not in basename:
                compressed = chunk_filepath.replace(".bin", f".{compressor}.bin")
                with suppress(FileNotFoundError, PermissionError):
                    os.remove(compressed)
            self._note_chunk_removed(chunk_index)
            # Only free a slot when we removed a real file that was counted toward
            # the budget. Force-redownload keeps the reservation for the replacement.
            if release_slot and file_existed:
                self._release_cache_slot()
        except (FileNotFoundError, PermissionError) as e:
            logger.debug(f"_apply_delete({chunk_index}): could not remove data file: {e}")

        self.clear_chunk_ready(chunk_index)
        self._cleanup_download_locks(chunk_filepath, chunk_index)

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        self._end_requested = True
        self._to_download_queue.put(_END_TOKEN)

    def force_stop(self) -> None:
        self._force_stop_event.set()

    def _maybe_delete_chunks(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        # When the prefetch buffer is full we still use a short timeout so force-download
        # requests are not starved behind a multi-second delete-queue wait.
        chunk_index = _get_from_queue(self._to_delete_queue, timeout=timeout)

        if chunk_index is None:
            return

        # Store the current chunk index
        self._chunks_index_to_be_deleted.append(chunk_index)

        # Get the current cache size and decide whether we need to start cleanup. Otherwise, keep track of it
        while self._max_cache_size and self._chunks_index_to_be_deleted and self._can_delete_chunk():
            # Delete the oldest chunk
            self._apply_delete(self._chunks_index_to_be_deleted.popleft())
        # Decrement the pre-download counter
        self._pre_download_counter -= 1
        return

    def _drain_and_flush_deletes(self) -> None:
        """Drain the delete queue and remove pending chunks before thread exit.

        Normal ``_maybe_delete_chunks`` only pulls one queue item per call. At
        END_TOKEN several indexes can still be queued (or stuck behind the
        prefetch-window gate), which leaked the last chunk files / ``.lock``s.
        """
        while True:
            chunk_index = _get_from_queue(self._to_delete_queue, timeout=0.0)
            if chunk_index is None:
                break
            self._chunks_index_to_be_deleted.append(chunk_index)
            self._pre_download_counter -= 1

        if not self._max_cache_size:
            return

        # Shutdown: drop prefetch-window gating so delete-when-processed /
        # over-budget caches actually empty.
        while self._chunks_index_to_be_deleted:
            if self._delete_chunks_when_processed or self._cache_over_budget():
                self._apply_delete(self._chunks_index_to_be_deleted.popleft())
            else:
                break

    def _cap_pre_download_for_cache_budget(self) -> None:
        """Shrink per-worker prefetch so workers × chunks fit in ``max_cache_size``.

        Applies whenever delete-when-processed is on — including tiny unit-test
        budgets where the shared slot gate is disabled. Otherwise the async
        ``max_pre_download`` floor (4) can keep more chunks on disk than the
        budget allows (see ``test_reader_chunk_removal``).

        Never caps below 2: ``max_pre_download == 1`` deadlocks delete-when-processed
        gating (``_can_download_more`` needs a free slot while ``_can_delete_chunk``
        and refcount skips can leave ``pre_download_counter`` stuck at 1).
        """
        if not self._max_cache_size or not self._delete_chunks_when_processed:
            return
        max_cache_size = self._max_cache_size
        chunks = self._config._chunks or []
        mean_chunk = max(1, int(self._config.num_bytes // max(1, len(chunks))))
        n_workers = max(1, self._worker_env.world_size)
        budget_chunks = max(1, int(max_cache_size // mean_chunk))
        per_worker = max(1, budget_chunks // n_workers)
        # Floor at 2 so tiny budgets still allow one in-flight + one retained chunk.
        capped = max(2, per_worker)
        if capped < self._max_pre_download:
            logger.info(
                "max_cache_size=%s (~%d chunks) with %d workers: capping max_pre_download %d → %d to limit peak disk",
                self._max_cache_size,
                budget_chunks,
                n_workers,
                self._max_pre_download,
                capped,
            )
            self._max_pre_download = capped

    def _cache_over_budget(self, *, reconcile: bool = False, extra_bytes: int = 0) -> bool:
        """True when on-disk cache is at/over ``max_cache_size`` (node-wide folder)."""
        if not self._max_cache_size:
            return False
        if reconcile:
            # Rate-limit directory scans; multi-worker download decisions need a
            # fresh view of the shared cache but not a scandir on every loop tick.
            now = time()
            if (now - self._last_cache_reconcile_s) >= 0.25:
                self._reconcile_cache_bytes()
                self._last_cache_reconcile_s = now
            else:
                self._ensure_cache_bytes()
        else:
            self._ensure_cache_bytes()
        return (self._approx_cache_bytes + extra_bytes) >= self._max_cache_size

    def _budget_paths(self) -> tuple[str, str]:
        cache_dir = self._config._cache_dir
        return (
            os.path.join(cache_dir, ".litdata_cache_slots"),
            os.path.join(cache_dir, ".litdata_cache_slots.lock"),
        )

    def _max_chunk_slots(self) -> int:
        """How many chunk files may exist on disk for the configured budget."""
        assert self._max_cache_size
        chunks = self._config._chunks or []
        mean_chunk = max(1, int(self._config.num_bytes // max(1, len(chunks) or 1)))
        # +5% headroom matches the accepted overshoot for in-flight replace/tmp.
        return max(1, int((self._max_cache_size * 1.05) // mean_chunk))

    def _on_disk_chunk_count(self) -> int:
        """Count ``chunk-*.bin`` files currently under the cache dir."""
        cache_dir = self._config._cache_dir
        try:
            with os.scandir(cache_dir) as entries:
                return sum(
                    1
                    for e in entries
                    if e.is_file(follow_symlinks=False) and e.name.startswith("chunk-") and e.name.endswith(".bin")
                )
        except OSError:
            return 0

    def _read_used_slots_unlocked(self, slots_path: str) -> int:
        """Return reserved slots, never below the on-disk chunk count.

        The counter can drift low when force-redownload used to release-then-
        download without re-acquiring; clamping to disk keeps the gate honest.
        """
        on_disk = self._on_disk_chunk_count()
        try:
            with open(slots_path, encoding="utf-8") as f:
                used = int(f.read().strip() or "0")
        except (FileNotFoundError, ValueError):
            used = on_disk
        if used < on_disk:
            used = on_disk
            try:
                with open(slots_path, "w", encoding="utf-8") as f:
                    f.write(str(used))
            except OSError:
                pass
        return used

    def _slot_budget_enabled(self) -> bool:
        """Use shared chunk slots only for realistic on-disk budgets.

        Tiny ``max_cache_size`` values in unit tests (e.g. 512 bytes) only force
        delete-when-processed; engaging the multi-worker slot gate there deadlocks.
        Require the budget to fit at least two mean chunks and be ≥10MB.
        """
        if not self._max_cache_size or not self._delete_chunks_when_processed:
            return False
        chunks = self._config._chunks or []
        if not chunks:
            return False
        mean_chunk = max(1, int(self._config.num_bytes // max(1, len(chunks))))
        return self._max_cache_size >= max(mean_chunk * 2, 10 * 1024 * 1024)

    def _acquire_cache_slot(self, timeout_s: float = 30.0) -> bool:
        """Reserve one chunk slot in the shared on-disk budget (multi-worker safe).

        Returns False if no slot could be reserved — caller must **not** download
        and should re-queue the chunk index instead.
        """
        if not self._slot_budget_enabled():
            return True
        slots_path, lock_path = self._budget_paths()
        max_slots = self._max_chunk_slots()
        deadline = time() + timeout_s
        while time() < deadline:
            try:
                with FileLock(lock_path, timeout=0.2):
                    used = self._read_used_slots_unlocked(slots_path)
                    if used < max_slots:
                        with open(slots_path, "w", encoding="utf-8") as f:
                            f.write(str(used + 1))
                        return True
            except Timeout:
                pass
            # Free a slot by deleting a fully-consumed chunk, then retry.
            self._maybe_delete_chunks(timeout=0.05)
            sleep(0.05)
        return False

    def _release_cache_slot(self) -> None:
        """Release one chunk slot after a successful delete."""
        if not self._slot_budget_enabled():
            return
        slots_path, lock_path = self._budget_paths()
        try:
            with FileLock(lock_path, timeout=1):
                try:
                    with open(slots_path, encoding="utf-8") as f:
                        used = int(f.read().strip() or "0")
                except (FileNotFoundError, ValueError):
                    used = 0
                with open(slots_path, "w", encoding="utf-8") as f:
                    f.write(str(max(0, used - 1)))
        except Exception as e:
            logger.debug(f"_release_cache_slot failed: {e}")

    def _can_delete_chunk(self) -> bool:
        if self._max_cache_size is None:
            return False
        # Size always wins so multi-worker prefetch cannot ignore ``max_cache_size``.
        # Shared-chunk safety is enforced in `_apply_pending_deletes` (in-use /
        # refcount); ~5% overshoot from in-flight downloads is accepted.
        if self._cache_over_budget():
            return True
        if self._delete_chunks_when_processed:
            return self._pre_download_counter >= self._max_pre_download - 1
        return False

    def _pre_load_chunk(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def _force_download(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        if getattr(self._item_loader, "uses_direct_remote", False) is True:
            return
        chunk_index = _get_from_queue(self._force_download_queue, timeout=timeout)
        if chunk_index is None:
            return

        chunk_filepath, _, filesize_bytes = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        download_lock_path = self._config.download_filepath(chunk_index) + ".lock"
        try:
            with FileLock(download_lock_path, timeout=0):
                # The chunk may have been fully downloaded by the time this
                # request was processed, so double check that it still needs
                # downloading before we delete it.
                if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes:
                    self.mark_chunk_ready(chunk_index)
                    return

                had_file = os.path.exists(chunk_filepath)
                # Replace in place: keep the existing disk-slot reservation when
                # a partial/corrupt file occupied one. Missing files need a new
                # slot before we download.
                self._apply_delete(chunk_index, skip_lock=True, release_slot=False)
                if _DEBUG:
                    print(
                        f"[Reader] Requested force download for {chunk_filepath} "
                        f"by {self._rank} at {datetime.now().isoformat()}"
                    )

            if not had_file and not self._acquire_cache_slot():
                # No budget left — retry later instead of exceeding max_cache_size.
                self._force_download_queue.put(chunk_index)
                sleep(0.05)
                return

            self._config.download_chunk_from_index(chunk_index, skip_lock=True)
            self.mark_chunk_ready(chunk_index)
            self._note_chunk_added(chunk_index)
        except Timeout:
            # Another worker is actively downloading this chunk. Defer to them.
            return

        # Preload item if possible to gain some time but only
        # if this is one of the pre-downloaded chunk
        if self._pre_download_counter > 0:
            self._pre_load_chunk(chunk_index)

        # Avoid downloading too many chunks in advance at the risk of over using the disk space
        self._pre_download_counter += 1

    def _finalize_downloaded_chunk(self, chunk_index: int, *, existed: bool) -> None:
        """Mark ready / account cache / optionally pre-load after a chunk download."""
        chunk_filepath, _, filesize_bytes = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes:
            self.mark_chunk_ready(chunk_index)
        if not existed:
            self._note_chunk_added(chunk_index)
        if self._pre_download_counter > 0:
            self._pre_load_chunk(chunk_index)
        self._pre_download_counter += 1

    def _download_chunk_indexes(self, chunk_indexes: list[int]) -> None:
        """Download one or more chunk indexes (sync, or concurrent when env-enabled)."""
        if not chunk_indexes:
            return
        if getattr(self._item_loader, "uses_direct_remote", False) is True:
            for chunk_index in dict.fromkeys(int(idx) for idx in chunk_indexes):
                self._pre_load_chunk(chunk_index)
                self._pre_download_counter += 1
            return
        # Shuffle / queue can repeat an index; one GET per chunk per batch.
        chunk_indexes = list(dict.fromkeys(int(idx) for idx in chunk_indexes))
        # Respect the shared disk budget before bringing new bytes onto disk.
        pending: list[int] = []
        deferred: list[int] = []
        existed: dict[int, bool] = {}
        for idx in chunk_indexes:
            path = self._config[ChunkedIndex(index=-1, chunk_index=idx)][0]
            already = os.path.exists(path)
            existed[idx] = already
            if already:
                continue
            if self._acquire_cache_slot():
                # Another worker may have finished between the exists check and
                # the slot reservation — drop the spare slot and skip download.
                if os.path.exists(path):
                    self._release_cache_slot()
                    existed[idx] = True
                    continue
                pending.append(idx)
            else:
                # No disk slot — retry later instead of blowing past max_cache_size.
                deferred.append(idx)

        t0 = self._timing.start()
        with trace_span("prefetch", CAT_DOWNLOAD, chunks=len(pending)):
            if pending:
                if self._async_prefetch() and len(pending) > 1:
                    download_chunk_indexes_concurrently(self._config, pending)
                else:
                    for chunk_index in pending:
                        self._config.download_chunk_from_index(chunk_index)
        self._timing.record("chunk_download_s", t0)
        # Finalize only chunks we own / already had — not deferred ones.
        for chunk_index, was_there in existed.items():
            if chunk_index in deferred:
                continue
            self._finalize_downloaded_chunk(chunk_index, existed=was_there)
        if deferred:
            # Avoid a tight requeue spin when every worker is waiting on a slot.
            if not pending:
                sleep(0.05)
            for chunk_index in deferred:
                self._to_download_queue.put(chunk_index)

    def prefetch_error(self) -> BaseException | None:
        """Exception that killed this thread, if any."""
        return self._error

    def _report_crash(self, exc: BaseException) -> None:
        """Make a prefetch-thread death obvious in stderr and in the LitData tracer.

        DataLoader workers often swallow ``logger.exception`` from a daemon thread.
        Print a flushed traceback to stderr for the training log. Also emit a
        one-line Chrome-trace instant event (``ph: I``) so ``enable_tracer()`` /
        Litracer is not corrupted by a multi-line traceback.
        """
        rank = self._rank if self._rank is not None else self._distributed_env.global_rank
        worker = self._worker_env.rank
        header = (
            f"[litdata] PrepareChunksThread CRASHED (rank={rank}, worker={worker}): "
            f"{type(exc).__name__}: {exc}\n"
            "Chunk downloads have stopped. The reader will fail with this error "
            "instead of waiting until FileNotFoundError."
        )
        print(header, file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        emit_trace(
            "crash",
            "I",
            CAT_CRASH,
            exception=type(exc).__name__,
            error=f"{type(exc).__name__}: {exc}",
        )

    def run(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:
            # Do not re-raise: Thread.run would only print a traceback, and the
            # item loader would still wait until MAX_WAIT_TIME then raise
            # FileNotFoundError. Stash the cause so waiters fail immediately.
            self._error = exc
            self._report_crash(exc)
        finally:
            # Drop thread-local asyncio loop + default-executor workers
            # (``asyncio_N``) created by async chunk prefetch / to_thread.
            close_thread_event_loop()

    def _run_loop(self) -> None:
        while True:
            if self._force_stop_event.is_set():
                self._has_exited = True
                return

            over_budget = False
            if self._slot_budget_enabled():
                # Prefer eviction when over the shared disk budget so multi-worker
                # runs converge toward max_cache_size.
                over_budget = self._cache_over_budget(reconcile=True)
                if over_budget:
                    self._maybe_delete_chunks(timeout=0.0)

            can_download_more = self._should_start_download(over_budget=over_budget)

            # Non-blocking force/delete polls while download work can still proceed, so we do not
            # pay ~0.2s of empty-queue sleep per chunk. When the prefetch buffer is full, keep a
            # short timeout so force-download requests are not blocked behind a long delete wait.
            side_timeout = 0.0 if can_download_more else _DEFAULT_TIMEOUT

            # When the window is full, drain deletes before blocking on force-download
            # so slots free and we can refill instead of sitting on an empty force queue.
            if self._max_cache_size and not can_download_more:
                self._maybe_delete_chunks(timeout=0.0)

            self._force_download(timeout=side_timeout)

            if can_download_more:
                chunk_index = _get_from_queue(self._to_download_queue)
            elif self._end_requested:
                chunk_index = _END_TOKEN
            else:
                chunk_index = None

            if can_download_more or chunk_index == _END_TOKEN:
                if chunk_index == _END_TOKEN:
                    if self._max_cache_size:
                        self._drain_and_flush_deletes()
                    self._has_exited = True
                    return

                # Shuffle emits numpy.int64; do not use ``isinstance(..., int)``.
                if chunk_index is not None:
                    batch: list[int] = [int(chunk_index)]
                    # Drain more pending indexes so asyncio.gather can overlap remote
                    # downloads. When over disk budget, download one at a time.
                    if self._async_prefetch() and not over_budget:
                        target = min(self._free_prefetch_slots(), self._async_gather_width())
                        while len(batch) < target:
                            nxt = _get_from_queue(self._to_download_queue, timeout=0.0)
                            if nxt is None:
                                break
                            if nxt == _END_TOKEN:
                                self._to_download_queue.put(_END_TOKEN)
                                break
                            batch.append(int(nxt))
                    self._download_chunk_indexes(batch)

            if self._max_cache_size:
                self._maybe_delete_chunks(timeout=side_timeout)


# The BinaryReader operates as the inverse of the data optimization process:
# 1. Loads raw bytes from chunks based on specific indices
# 2. Uses deserializers to convert bytes back into Python objects
# 3. Reconstructs the original data structure with the data_spec from index.json and using `tree_unflatten function`
# 4. Supports features like compression, encryption, and distributed reading
class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        subsampled_files: list[str] | None = None,
        region_of_interest: list[tuple[int, int]] | None = None,
        max_cache_size: int | float | str | None = None,
        remote_input_dir: str | None = None,
        compression: str | None = None,
        encryption: Encryption | None = None,
        item_loader: BaseItemLoader | None = None,
        serializers: dict[str, Serializer] | None = None,
        storage_options: dict | None = {},
        session_options: dict | None = {},
        max_pre_download: int = 2,
        on_demand_bytes: bool = False,
    ) -> None:
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Args:
            cache_dir: The path to cache folder.
            subsampled_files: List of subsampled chunk files loaded from `input_dir/index.json` file.
            region_of_interest: List of tuples of {start,end} of region of interest for each chunk.
            remote_input_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            compression: The algorithm to decompress the chunks.
            encryption: The algorithm to decrypt the chunks or samples.
            item_loader: The chunk sampler to create sub arrays from a chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.
            serializers: Provide your own serializers.
            storage_options: Additional connection options for accessing storage services.
            session_options: Additional options for the S3 session.
            max_pre_download: Maximum number of chunks that can be pre-downloaded by the reader.
            on_demand_bytes: If True, fetch only the requested sample's bytes instead of downloading the entire chunk.

        """
        super().__init__()
        self._cache_dir = cache_dir
        self._remote_input_dir = remote_input_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._encryption = encryption
        self._intervals: list[str] | None = None
        self.subsampled_files = subsampled_files
        self.region_of_interest = region_of_interest
        self._serializers: dict[str, Serializer] = _get_serializers(serializers)
        self._distributed_env = _DistributedEnv.detect()
        self._rank: int | None = None
        self._config: ChunksConfig | None = None
        self._prepare_thread: PrepareChunksThread | None = None
        self._item_loader = item_loader or PyTreeLoader()
        self._last_chunk_index: int | None = None
        self._last_chunk_size: int | None = None
        self._chunks_queued_for_download = False
        # Shared chunks this worker reference-counts eagerly (before any reading), and the ones it
        # still holds. See `acquire_shared_locks` / `_release_shared_locks`.
        self._shared_chunk_indexes: set[int] = set()
        self._held_shared: set[int] = set()
        self._max_cache_size = _resolve_max_cache_size(max_cache_size, cache_dir)
        self._keep_node_shard: bool | None = None
        self._storage_options = storage_options
        self._session_options = session_options
        self._max_pre_download = max_pre_download
        self.on_demand_bytes = on_demand_bytes
        self._posix_fast = False
        self._posix_keep = 4
        self._posix_willneed = True
        self._timing = StreamingTimingStats.instance()
        self._pytree_loader = isinstance(self._item_loader, PyTreeLoader)

    def _get_chunk_index_from_index(self, index: int) -> tuple[int, int]:
        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self._config._get_chunk_index_from_index(index)  # type: ignore

    def _try_load_config(self) -> ChunksConfig | None:
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(
            self._cache_dir,
            self._serializers,
            self._remote_input_dir,
            self._item_loader,
            self.subsampled_files,
            self.region_of_interest,
            self._storage_options,
            self._session_options,
        )
        return self._config

    def set_keep_node_shard(self, keep: bool) -> None:
        """Pin this node's chunk files when the shard fits in ``max_cache_size``."""
        self._keep_node_shard = keep
        if self._prepare_thread is not None:
            self._prepare_thread._delete_chunks_when_processed = (not keep) and bool(self._max_cache_size)

    def acquire_shared_locks(self, shared_chunk_indexes: set[int]) -> None:
        """Eagerly reference-count the chunks this worker shares with other workers.

        Called once per epoch, before any item is read. Incrementing the shared chunks' reference
        counts up-front guarantees that every worker which will read a shared chunk has incremented
        its count before any worker can finish and delete it — closing the increment-lag race where
        a fast worker deletes a shared chunk a slower co-worker has not yet claimed. The matching
        decrements happen as each chunk is finished in `read`, and any still-held locks are released
        in `_release_shared_locks` on teardown (so early / partial iteration cannot leak counts).
        """
        if self._config is None and self._try_load_config() is None:
            return
        assert self._config is not None
        # Release anything left over from a previous epoch on this reader before re-acquiring.
        self._release_shared_locks()
        self._shared_chunk_indexes = set(shared_chunk_indexes)
        self._config._shared_chunk_indexes = self._shared_chunk_indexes
        for chunk_index in self._shared_chunk_indexes:
            self._config.increment_local_lock(chunk_index)
            self._held_shared.add(chunk_index)

    def enable_mmap_for_chunks(self, chunk_indexes: set[int]) -> None:
        """Tell the item loader which chunks are safe to memory-map (non-shared ones).

        Shared chunks are deliberately excluded: a co-worker could delete/replace a shared chunk
        while it is mapped, which crashes with SIGSEGV rather than a recoverable error.
        """
        self._item_loader.set_mmap_allowed_chunks(chunk_indexes)

    def _posix_node_readers(self) -> int:
        worker_env = getattr(self, "_worker_env", None) or _WorkerEnv.detect()
        self._worker_env = worker_env
        workers = max(1, worker_env.world_size)
        dist = self._distributed_env
        ranks_per_node = max(1, dist.world_size // max(1, dist.num_nodes))
        return workers * ranks_per_node

    def enable_posix_fast(self, chunk_indexes: list[int], keep: int = 4, *, prefetch: bool = True) -> None:
        """Read chunks from the dataset directory in place (Vast/NFS). Never delete sources."""
        self._posix_fast = True
        chunk_b = mean_chunk_bytes(self._config)
        readers = self._posix_node_readers()
        self._posix_keep = posix_safe_keep(keep=max(1, keep), chunk_bytes=chunk_b, num_readers=readers)
        self._posix_willneed = posix_prefetch_fits_ram(
            keep=self._posix_keep,
            chunk_bytes=chunk_b,
            num_readers=readers,
        )
        if not self._posix_willneed and getattr(self._worker_env, "rank", 0) == 0:
            logger.info(
                "POSIX-fast: skipping WILLNEED prefetch (%d readers × %d chunks × %d bytes would crowd RAM)",
                readers,
                self._posix_keep,
                chunk_b,
            )
        setter = getattr(self._item_loader, "set_posix_fast", None)
        if setter is not None:
            setter(True, keep=self._posix_keep, willneed=self._posix_willneed)
        self._item_loader.set_mmap_allowed_chunks(set(chunk_indexes))
        if prefetch:
            self.prefetch_posix_window(chunk_indexes[: self._posix_keep])

    def prefetch_posix_window(self, chunk_indexes: list[int]) -> None:
        """``posix_fadvise`` and mmap the next files in this worker's stripe (no download thread)."""
        if not self._posix_fast:
            return
        if self._config is None:
            self._try_load_config()
        if self._config is None:
            return
        warmer = getattr(self._item_loader, "warm_posix_chunk", None)
        for chunk_index in chunk_indexes:
            chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
            if not os.path.isfile(chunk_filepath):
                continue
            if self._posix_willneed:
                advise_willneed(chunk_filepath)
            if warmer is not None:
                warmer(chunk_index, chunk_filepath)
            else:
                self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def _release_shared_locks(self) -> None:
        """Release any eagerly-acquired shared-chunk locks this worker still holds."""
        if not self._held_shared:
            return
        if self._config is None:
            self._held_shared.clear()
            return
        for chunk_index in list(self._held_shared):
            self._config.decrement_local_lock(chunk_index)
            self._held_shared.discard(chunk_index)

    def setup_thread_and_download_chunk(self, index: ChunkedIndex) -> None:
        if self._config and (self._config._remote_dir or self._config._compressor):
            # Create and start the prepare chunks thread
            if self._prepare_thread is None and self._config:
                self._prepare_thread = PrepareChunksThread(
                    self._config,
                    self._item_loader,
                    self._distributed_env,
                    self._max_cache_size,
                    self._max_pre_download,
                    self._rank,
                )
                # Attach the force download queue and readiness signals used by the item loader wait loop.
                self._item_loader._force_download_queue = self._prepare_thread._force_download_queue  # type: ignore
                self._item_loader.set_chunk_ready_provider(self._prepare_thread.get_ready_event)
                self._item_loader.set_prefetch_error_provider(self._prepare_thread.prefetch_error)
                self._prepare_thread.start()
                if self._keep_node_shard is not None:
                    self._prepare_thread._delete_chunks_when_processed = (not self._keep_node_shard) and bool(
                        self._max_cache_size
                    )
                if index.chunk_indexes:
                    self._prepare_thread.download(index.chunk_indexes)
                    self._chunks_queued_for_download = True

            # Only request individual chunk download if:
            # 1. We haven't already queued all chunks for the download
            # 2. We're processing a new chunk (different from the last one)
            if not self._chunks_queued_for_download and index.chunk_index != self._last_chunk_index:
                assert self._prepare_thread
                self._prepare_thread.download([index.chunk_index])

    @property
    def config(self) -> ChunksConfig:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    @property
    def rank(self) -> int:
        """Returns the rank of the writer."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def read(self, index: ChunkedIndex) -> Any:
        """Read an item for the given from a chunk.

        If the chunk isn't available locally or in memory, it will be downloaded.

        Prefetching should reduce the wait time to be the batch available.

        """
        if index.__class__ is not ChunkedIndex:
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        # Fetch the element
        chunk_filepath, begin, filesize_bytes = self.config[index]
        timing = self._timing
        decode_t0 = timing.start() if timing.enabled else None

        if self._pytree_loader:
            if (
                self.on_demand_bytes
                and self._config
                and self._config._remote_dir
                and self._config._config
                and not self._config._config.get("encryption", None)
                and not self._config._config.get("compression", None)
            ):
                raw_bytes = self.read_item_bytes(index, begin)
                item = self._item_loader.load_item_from_bytes(raw_bytes, index.chunk_index)
            else:
                self.setup_thread_and_download_chunk(index)
                pytree_loader = self._item_loader
                assert isinstance(pytree_loader, PyTreeLoader)
                item = pytree_loader.load_item_from_chunk(
                    index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes, self._encryption
                )
        else:
            self.setup_thread_and_download_chunk(index)
            item = self._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes
            )
        timing.record("item_decode_s", decode_t0)

        # We need to request deletion after the latest element has been loaded.
        # Otherwise, this could trigger segmentation fault error depending on the item loader used.
        if (
            self._config
            and (self._config._remote_dir or self._config._compressor)
            and index.chunk_index != self._last_chunk_index
            and self._prepare_thread is not None
            and self._last_chunk_index is not None
        ):
            # inform the chunk has been completely consumed
            self._prepare_thread._decrement_local_lock(self._last_chunk_index)
            self._held_shared.discard(self._last_chunk_index)
            self._prepare_thread.delete([self._last_chunk_index])

        if index.chunk_index != self._last_chunk_index:
            if self._last_chunk_index is not None:
                emit_trace("read", "E", CAT_READ, chunk=self._last_chunk_index, size=self._last_chunk_size)

            emit_trace("read", "B", CAT_READ, chunk=index.chunk_index, size=index.chunk_size)

            # Close the memory-mapped file for the last chunk index.
            # PyTreeLoader is intentionally excluded: it keeps only one open chunk and already
            # unmaps the previous one inside `load_item_from_chunk` before this point. Calling
            # `close` here would unmap the newly opened chunk (its `close` ignores chunk_index).
            if isinstance(self._item_loader, (TokensLoader, ParquetLoader)) and self._last_chunk_index is not None:
                self._item_loader.close(self._last_chunk_index)

            # track the new chunk index as the latest one
            self._last_chunk_index = index.chunk_index
            self._last_chunk_size = index.chunk_size

        if index.is_last_index and self._prepare_thread:
            if self._last_chunk_index is not None:
                emit_trace("read", "E", CAT_READ, chunk=self._last_chunk_index, size=self._last_chunk_size)

            # Close the item loader's handle on the last chunk before requesting
            # deletion.  On Windows, os.remove fails if the file is still open.
            self._item_loader.close(self._last_chunk_index)

            # inform the thread it is time to stop
            self._prepare_thread._decrement_local_lock(index.chunk_index)
            self._held_shared.discard(index.chunk_index)
            self._prepare_thread.delete([index.chunk_index])
            # Release any shared-chunk locks still held (e.g. chunks assigned but never reached).
            self._release_shared_locks()
            self._prepare_thread.stop()
            if self._max_cache_size and self._prepare_thread.is_alive():
                try:
                    self._prepare_thread.join(timeout=_LONG_DEFAULT_TIMEOUT)
                except Timeout:
                    logger.warning(
                        "The prepare chunks thread didn't exit properly. "
                        "This can happen if the chunk files are too large."
                    )
            self._prepare_thread = None
            self._last_chunk_index = None
            self._last_chunk_size = None
            self._chunks_queued_for_download = False

        return item

    def read_item_bytes(self, index: ChunkedIndex, begin: int) -> bytes:
        """Reads the raw byte content for a specific item in a chunk without downloading the full chunk.

        Computes the byte offset for the item based on its index, retrieves the start and end positions
        from the chunk's index table, and downloads only the relevant byte range corresponding to the item.

        Args:
            index (ChunkedIndex): The index of the item within a chunk.
            begin (int): The starting index of the chunk (used to compute relative offset).

        Returns:
            bytes: The raw byte content for the specified item.
        """
        UINT32_BYTE_WIDTH = 4  # Number of bytes in a uint32
        offset_multiplier = 1 + (index.index - begin) if index.index >= begin else index.index + 1
        offset = offset_multiplier * UINT32_BYTE_WIDTH
        pair = self.config.download_chunk_bytes_from_index(index.chunk_index, offset, 8)
        begin, end = np.frombuffer(pair, np.uint32)
        actual_item_length = end - begin
        return self.config.download_chunk_bytes_from_index(index.chunk_index, begin, actual_item_length)

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return len(self.config)

    def get_chunk_intervals(self) -> list[Interval]:
        """Get the index interval of each chunk."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self.config.intervals

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_prepare_thread"] = None
        # StreamingTimingStats holds a threading.Lock and is process-local.
        state.pop("_timing", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._timing = StreamingTimingStats.instance()

    def __del__(self) -> None:
        # Release eagerly-acquired shared-chunk locks that were never released (e.g. the loop was
        # broken out of before the last item). Without this, an aborted epoch would leak reference
        # counts and prevent those chunks from ever being deleted.
        with suppress(Exception):
            self._release_shared_locks()
        closer = getattr(self._item_loader, "_close_open_chunk", None)
        if closer is not None:
            with suppress(Exception):
                closer()
        if self._prepare_thread and not self._prepare_thread._has_exited:
            self._prepare_thread.force_stop()
            self._prepare_thread = None


def _get_folder_size(path: str, config: ChunksConfig) -> int:
    """Calculate the total size of files in a directory based on specific rules.

    This method is robust to file deletion races.

    Args:
        path (str): Directory path to scan.
        config (ChunksConfig): Configuration object containing filename_to_size_map.

    Returns:
        int: Total size of valid files in bytes.

    """
    size = 0
    ignored_extensions = (".cnt", ".lock", ".json", ".zstd.bin")

    # os.scan_dir is more efficient than os.listdir
    with os.scandir(path) as dir_entries:
        for entry in dir_entries:
            # skip directories and symlinks
            if not entry.is_file(follow_symlinks=False):
                continue

            filename = entry.name

            # use size from config if available
            if filename in config.filename_to_size_map:
                size += config.filename_to_size_map[filename]

            # silently ignore specified extensions
            elif filename.endswith(ignored_extensions):
                continue

            # handle temporary files containing '.bin'
            elif ".bin" in filename:
                with suppress(FileNotFoundError):
                    size += entry.stat(follow_symlinks=False).st_size

            # warn about unrecognized files
            else:
                if _DEBUG:
                    logger.warning(
                        f"Ignoring '{filename}': This file doesn't appear to be a valid chunk file"
                        " and has been excluded from the cache size calculation."
                    )

    return size


def _get_from_queue(queue: Queue, timeout: float = _DEFAULT_TIMEOUT) -> Any | None:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        pass
    except OSError as err:
        # handle closed queue before the thread terminates
        if "handle is closed" in str(err) or "Bad file descriptor" in str(err):
            logger.debug(err)
        else:
            raise err
    except EOFError as err:
        logger.debug(err)
    return None

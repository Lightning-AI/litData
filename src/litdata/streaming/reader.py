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
import warnings
from contextlib import suppress
from datetime import datetime
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from filelock import FileLock, Timeout

from litdata.constants import _DEBUG
from litdata.debugger import _get_log_msg
from litdata.streaming.config import ChunksConfig, Interval
from litdata.streaming.item_loader import BaseItemLoader, ParquetLoader, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer, _get_serializers
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _WorkerEnv

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")


logger = logging.getLogger("litdata.streaming.reader")


_END_TOKEN = "END"  # noqa: S105

# Note: The timeout here should not be too short. We need to prevent the caller from aggressively
# querying the queue and consuming too many CPU cycles.
_DEFAULT_TIMEOUT = 0.1
_LONG_DEFAULT_TIMEOUT = 5


class PrepareChunksThread(Thread):
    """This thread is responsible to download the chunks associated to a given worker."""

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
        self._max_pre_download = max_pre_download
        self._pre_download_counter = 0
        self._distributed_env = distributed_env
        self._worker_env = _WorkerEnv.detect()

        self._chunks_index_to_be_deleted: list[int] = []
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: Queue = Queue()
        self._to_delete_queue: Queue = Queue()
        self._force_stop_event = Event()

        # TODO: Find a real fix to this problem
        self._force_download_queue: Queue = Queue()

        # Per-chunk readiness signals for the in-process item-loader wait loop.
        self._chunk_ready: dict[int, Event] = {}
        self._chunk_ready_lock = Lock()

        self._rank = rank

        # Check whether a dataset slice fits on the node
        num_bytes_per_nodes = self._config.num_bytes // self._distributed_env.num_nodes
        self._delete_chunks_when_processed = num_bytes_per_nodes > max_cache_size if max_cache_size else False

        if _DEBUG and distributed_env.global_rank == 0 and self._worker_env.rank == 0:
            print(f"Delete chunks when used: {self._delete_chunks_when_processed}")

        self._has_exited = False

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

    def _apply_delete(self, chunk_index: int, skip_lock: bool = False) -> None:
        """Inform the item loader of the chunk to delete."""
        logger.debug(f"_apply_delete({chunk_index}, skip_lock={skip_lock}) called")
        can_delete_chunk = self._config.can_delete(chunk_index)
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
                tombstone_file.write(f"Deleted {chunk_filepath} by {self._rank or 0}. Debug: {can_delete_chunk}")

        try:
            self._item_loader.delete(chunk_index, chunk_filepath)
        except (FileNotFoundError, PermissionError) as e:
            logger.debug(f"_apply_delete({chunk_index}): could not remove data file: {e}")

        self.clear_chunk_ready(chunk_index)
        self._cleanup_download_locks(chunk_filepath, chunk_index)

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
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
            self._apply_delete(self._chunks_index_to_be_deleted.pop(0))
        # Decrement the pre-download counter
        self._pre_download_counter -= 1
        return

    def _can_delete_chunk(self) -> bool:
        if self._delete_chunks_when_processed:
            return self._pre_download_counter >= self._max_pre_download - 1
        return (
            self._max_cache_size is not None
            and _get_folder_size(self._config._cache_dir, self._config) >= self._max_cache_size
        )

    def _pre_load_chunk(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def _force_download(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
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

                # force apply deletion before redownload
                self._apply_delete(chunk_index, skip_lock=True)
                if _DEBUG:
                    print(
                        f"[Reader] Requested force download for {chunk_filepath} "
                        f"by {self._rank} at {datetime.now().isoformat()}"
                    )

            self._config.download_chunk_from_index(chunk_index, skip_lock=True)
            self.mark_chunk_ready(chunk_index)
        except Timeout:
            # Another worker is actively downloading this chunk. Defer to them.
            return

        # Preload item if possible to gain some time but only
        # if this is one of the pre-downloaded chunk
        if self._pre_download_counter > 0:
            self._pre_load_chunk(chunk_index)

        # Avoid downloading too many chunks in advance at the risk of over using the disk space
        self._pre_download_counter += 1

    def run(self) -> None:
        while True:
            if self._force_stop_event.is_set():
                self._has_exited = True
                return

            can_download_more = self._pre_download_counter < self._max_pre_download
            # Non-blocking force/delete polls while download work can still proceed, so we do not
            # pay ~0.2s of empty-queue sleep per chunk. When the prefetch buffer is full, keep a
            # short timeout so force-download requests are not blocked behind a long delete wait.
            side_timeout = 0.0 if can_download_more else _DEFAULT_TIMEOUT

            self._force_download(timeout=side_timeout)

            if can_download_more:
                chunk_index = _get_from_queue(self._to_download_queue)
                if chunk_index == _END_TOKEN:
                    if self._max_cache_size:
                        self._maybe_delete_chunks(timeout=_DEFAULT_TIMEOUT)
                    self._has_exited = True
                    return

                if chunk_index is not None:
                    self._config.download_chunk_from_index(chunk_index)
                    chunk_filepath, _, filesize_bytes = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
                    if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes:
                        self.mark_chunk_ready(chunk_index)

                    # Preload item if possible to gain some time but only
                    # if this is one of the pre-downloaded chunk
                    if self._pre_download_counter > 0:
                        self._pre_load_chunk(chunk_index)

                    # Avoid downloading too many chunks in advance at the risk of over using the disk space
                    self._pre_download_counter += 1

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
        max_cache_size: int | str | None = None,
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
        warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

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
        self._max_cache_size = int(os.getenv("MAX_CACHE_SIZE", max_cache_size or 0))
        self._storage_options = storage_options
        self._session_options = session_options
        self._max_pre_download = max_pre_download
        self.on_demand_bytes = on_demand_bytes

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
                self._prepare_thread.start()
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
        if not isinstance(index, ChunkedIndex):
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        # Fetch the element
        chunk_filepath, begin, filesize_bytes = self.config[index]

        if isinstance(self._item_loader, PyTreeLoader):
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
                item = self._item_loader.load_item_from_chunk(
                    index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes, self._encryption
                )
        else:
            self.setup_thread_and_download_chunk(index)
            item = self._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes
            )

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
                # 2. Log the "End" event for the previous chunk.
                logger.debug(
                    _get_log_msg(
                        {"name": f"read_chunk_{self._last_chunk_index}_size_{self._last_chunk_size}", "ph": "E"}
                    )
                )

            # 2. Log the "Begin" event for the NEW chunk.
            logger.debug(_get_log_msg({"name": f"read_chunk_{index.chunk_index}_size_{index.chunk_size}", "ph": "B"}))

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
        return state

    def __del__(self) -> None:
        # Release eagerly-acquired shared-chunk locks that were never released (e.g. the loop was
        # broken out of before the last item). Without this, an aborted epoch would leak reference
        # counts and prevent those chunks from ever being deleted.
        with suppress(Exception):
            self._release_shared_locks()
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

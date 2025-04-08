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

import contextlib
import logging
import os
import warnings
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

from litdata.constants import _DEBUG
from litdata.streaming.config import ChunksConfig, Interval
from litdata.streaming.item_loader import BaseItemLoader, ParquetLoader, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer, _get_serializers
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _WorkerEnv

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")


logger = logging.getLogger(__name__)


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
        chunks_order: List[int],
        distributed_env: _DistributedEnv,
        max_cache_size: Optional[int] = None,
        max_pre_download: int = 2,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._config = config
        self._item_loader = item_loader
        self._chunks_order = chunks_order  # order in which chunks are to be downloaded
        self.current_downloading_chunk_index = -1
        self.current_reading_chunk_index = -1
        self._max_pre_download = max_pre_download
        self._pre_download_counter = 0
        self._distributed_env = distributed_env

        # self._chunks_index_to_be_deleted: List[int] = []
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: Queue = Queue()
        self._to_delete_queue: Queue = Queue()
        self._delete_queue_received_none: bool = False
        self._force_stop_event = Event()

        # TODO: Find a real fix to this problem
        self._force_download_queue: Queue = Queue()

        self._rank = rank

        # Check whether a dataset slice fits on the node
        num_bytes_per_nodes = self._config.num_bytes // self._distributed_env.num_nodes
        self._delete_chunks_when_processed = num_bytes_per_nodes > max_cache_size if max_cache_size else False
        # print(f"reader: {self._delete_chunks_when_processed=}")
        # if self._delete_chunks_when_processed:
        #     print(f"clearing cache dir {self._parent_cache_dir} because the dataset is too large to fit in memory")
        #     # means we can't keep all chunks in the cache directory, so we should clear it to minimize the size
        #     # clear the cache directory except the index.json file
        #     for root, _, files in os.walk(self._parent_cache_dir):
        #         for file in files:
        #             if file != _INDEX_FILENAME:
        #                 with contextlib.suppress(FileNotFoundError):
        #                     os.remove(os.path.join(root, file))
        self._has_exited = False

    # def download(self, chunk_indexes: List[int]) -> None:
    #     """Receive the list of the chunk indices to download for the current epoch."""
    #     print(f"thread: got indexes to download -> {chunk_indexes=};")
    #     for chunk_index in chunk_indexes:
    #         self._to_download_queue.put(chunk_index)

    def delete(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to delete for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_delete_queue.put(chunk_index)

    # def _remaining_locks(self, chunkpath: str) -> int:
    #     countpath = chunkpath + ".cnt"
    #     if not os.path.exists(countpath):
    #         return 0
    #     with open(countpath) as count_f:
    #         try:
    #             return int(count_f.read().strip())
    #         except Exception:
    #             return 1

    # def _decrement_local_lock(self, chunk_index: int) -> int:
    #     """Remove a count from the local lock, return the remaining count."""
    #     chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

    #     countpath = chunk_filepath + ".cnt"
    #     with suppress(Timeout), FileLock(countpath + ".lock", timeout=3):
    #         if not os.path.exists(countpath):
    #             return 0
    #         with open(countpath) as count_f:
    #             try:
    #                 curr_count = int(count_f.read().strip())
    #             except Exception:
    #                 curr_count = 1
    #         curr_count -= 1
    #         if curr_count <= 0:
    #             with contextlib.suppress(FileNotFoundError, PermissionError):
    #                 os.remove(countpath)

    #             with contextlib.suppress(FileNotFoundError, PermissionError):
    #                 os.remove(countpath + ".lock")
    #         else:
    #             with open(countpath, "w+") as count_f:
    #                 count_f.write(str(curr_count))
    #         return curr_count
    #     return 0

    def _apply_delete(self, chunk_index: int) -> None:
        """Inform the item loader of the chunk to delete."""
        # TODO: Fix the can_delete method
        # can_delete_chunk = self._config.can_delete(chunk_index)
        # print(f"apply delete called -> {chunk_index} {can_delete_chunk=}; by {self._rank or 0}")
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

        # remaining_locks = self._remaining_locks(chunk_filepath)
        # if remaining_locks > 0:  # Can't delete this, something has it
        #     if _DEBUG:
        #         print(f"Skip delete {chunk_filepath} by {self._rank or 0}, current lock count: {remaining_locks}")
        #     return

        # if _DEBUG:
        #     with open(chunk_filepath + ".tmb", "w+") as tombstone_file:
        #         tombstone_file.write(f"Deleted {chunk_filepath} by {self._rank or 0}. Debug: {can_delete_chunk}")

        self._item_loader.safe_delete(
            chunk_index, chunk_filepath, delete_original_file=self._delete_chunks_when_processed
        )

        # if _DEBUG:
        #     print(f"Deleted {chunk_filepath} by {self._rank or 0}. Debug: {can_delete_chunk}")

        # for lock_extension in [".lock", ".cnt.lock"]:
        #     try:
        #         locak_chunk_path = chunk_filepath + lock_extension
        #         if os.path.exists(locak_chunk_path):
        #             os.remove(locak_chunk_path)
        #     except FileNotFoundError:
        #         pass

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        # self._to_download_queue.put(_END_TOKEN)
        if self._delete_chunks_when_processed and not self._delete_queue_received_none:
            # for chnk_idx in self._chunks_index_to_be_deleted:
            #     self._apply_delete(chnk_idx)
            # read from delete queue until None is received and delete the chunks
            total_waiting_time = 0
            while not self._delete_queue_received_none:  # parallelly it can be set true by thread's run method
                try:
                    chunk_index = self._to_delete_queue.get(timeout=_DEFAULT_TIMEOUT)
                    if chunk_index is None:
                        self._delete_queue_received_none = True
                        break
                    self._apply_delete(chunk_index)
                    total_waiting_time = 0
                except Empty:
                    total_waiting_time += _DEFAULT_TIMEOUT
                    if total_waiting_time > _LONG_DEFAULT_TIMEOUT * 2:  # wait for 10 seconds
                        print("Timeout waiting for delete queue to be empty (None)")
                        break
        self.force_stop()

    def force_stop(self) -> None:
        self._force_stop_event.set()

    def _maybe_delete_chunks(self) -> None:
        # reached_pre_download = self._pre_download_counter == self._max_pre_download
        # reached_max_pre_download = (
        #     self.current_downloading_chunk_index < self.current_reading_chunk_index + self._max_pre_download
        # )

        # should_start_deleting_chunks = self._can_delete_chunk()
        # if not should_start_deleting_chunks:
        #     return

        # we have already pre-downloaded some chunks, we just need to wait for them to be processed.
        if self._delete_queue_received_none:
            return
        while True:
            try:
                chunk_index_to_be_deleted = self._to_delete_queue.get(timeout=_DEFAULT_TIMEOUT)

                if chunk_index_to_be_deleted is None:
                    self._delete_queue_received_none = True
                    return
                    # self._pre_download_counter -= 1

                    # Store the current chunk index
                    # self._chunks_index_to_be_deleted.append(chunk_index)

                # Get the current cache size and decide whether we need to start cleanup. Otherwise, keep track of it
                # Delete the oldest chunk
                self._apply_delete(chunk_index_to_be_deleted)
            except Empty:
                # Timeout waiting for delete queue to be empty
                break
            except Exception as e:
                raise RuntimeError(f"Error while deleting chunks: {e}") from e

        return

    def _can_delete_chunk(self) -> bool:
        if self._delete_chunks_when_processed:
            # return self._pre_download_counter >= self._max_pre_download - 1
            # if we have downloaded all chunks, we can delete the oldest one
            if self.current_downloading_chunk_index == len(self._chunks_order) - 1:
                return True
            return self.current_downloading_chunk_index >= (self.current_reading_chunk_index + self._max_pre_download)

        return False  # if complete dataset can be stored in the cache, we don't need to delete any chunk
        return (
            self._max_cache_size is not None
            and _get_folder_size(self._config._cache_dir, self._config) >= self._max_cache_size
        )

    def _can_download_chunk(self) -> bool:
        return not self._can_delete_chunk()

    def _pre_load_chunk(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def _force_download(self) -> None:
        chunk_index = _get_from_queue(self._force_download_queue)
        if chunk_index is not None:
            if _DEBUG:
                chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
                print(f"Requested force download for {chunk_filepath} by {self._rank}")

            # skip counter_file logic and directly download the chunk
            self._config._downloader.download_chunk_from_index(chunk_index)

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

            self._force_download()

            if self._can_download_chunk() and (self.current_downloading_chunk_index < len(self._chunks_order) - 1):
                self.current_downloading_chunk_index += 1
                # chunk_index = _get_from_queue(self._to_download_queue)
                chunk_index = self._chunks_order[self.current_downloading_chunk_index]
                # if chunk_index == _END_TOKEN:
                #     self._has_exited = True
                #     return

                if chunk_index is not None:
                    self._config.download_chunk_from_index(chunk_index)

                    # Preload item if possible to gain some time but only
                    # if this is one of the pre-downloaded chunk
                    # if self._pre_download_counter > 0:
                    #     self._pre_load_chunk(chunk_index)

                    # Avoid downloading too many chunks in advance at the risk of over using the disk space
                    # self._pre_download_counter += 1
            self._maybe_delete_chunks()


# The BinaryReader operates as the inverse of the data optimization process:
# 1. Loads raw bytes from chunks based on specific indices
# 2. Uses deserializers to convert bytes back into Python objects
# 3. Reconstructs the original data structure with the data_spec from index.json and using `tree_unflatten function`
# 4. Supports features like compression, encryption, and distributed reading
class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        subsampled_files: Optional[List[str]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
        max_cache_size: Optional[Union[int, str]] = None,
        remote_input_dir: Optional[str] = None,
        compression: Optional[str] = None,
        encryption: Optional[Encryption] = None,
        item_loader: Optional[BaseItemLoader] = None,
        serializers: Optional[Dict[str, Serializer]] = None,
        storage_options: Optional[dict] = {},
        max_pre_download: int = 2,
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
            max_pre_download: Maximum number of chunks that can be pre-downloaded by the reader.

        """
        super().__init__()
        warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

        self._cache_dir = cache_dir
        self._remote_input_dir = remote_input_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._encryption = encryption
        self._intervals: Optional[List[str]] = None
        self.subsampled_files = subsampled_files
        self.region_of_interest = region_of_interest
        self._serializers: Dict[str, Serializer] = _get_serializers(serializers)
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None
        self._config: Optional[ChunksConfig] = None
        self._prepare_thread: Optional[PrepareChunksThread] = None
        self._item_loader = item_loader or PyTreeLoader()
        self._last_chunk_index: Optional[int] = None
        self._chunks_queued_for_download = False
        self._max_cache_size = int(os.getenv("MAX_CACHE_SIZE", max_cache_size or 0))
        self._storage_options = storage_options
        self._max_pre_download = max_pre_download

    def _get_chunk_index_from_index(self, index: int) -> Tuple[int, int]:
        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self._config._get_chunk_index_from_index(index)  # type: ignore

    def _try_load_config(self) -> Optional[ChunksConfig]:
        """Try to load the chunks config if the index files are available."""
        self._config = ChunksConfig.load(
            self._cache_dir,
            self._serializers,
            self._remote_input_dir,
            self._item_loader,
            self.subsampled_files,
            self.region_of_interest,
            self._storage_options,
        )
        return self._config

    def prepare_downloader_thread(self, chunks_order: List[int]) -> None:
        """Prepare the downloader thread and start downloading the first few chunks."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        # Create and start the prepare chunks thread
        if self._prepare_thread is None:
            self._prepare_thread = PrepareChunksThread(
                config=self._config,
                item_loader=self._item_loader,
                chunks_order=chunks_order,
                distributed_env=self._distributed_env,
                max_cache_size=self._max_cache_size,
                max_pre_download=self._max_pre_download,
                rank=self.rank,
            )
            self._prepare_thread.start()

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

        if self._config is None or self._prepare_thread is None:
            raise Exception(
                "Reader's downloading thread is not started. Please call `reader.prepare_downloader_thread()` first."
            )

        # Load the config containing the index
        # if self._config is None and self._try_load_config() is None:
        #     raise Exception("The reader index isn't defined.")

        if self._config and (self._config._remote_dir or self._config._compressor):  # noqa: SIM102
            # Create and start the prepare chunks thread
            # if self._prepare_thread is None and self._config:
            #     self._prepare_thread = PrepareChunksThread(
            #         self._config,
            #         self._item_loader,
            #         self._distributed_env,
            #         self._max_cache_size,
            #         self._max_pre_download,
            #         self._rank,
            #     )
            #     # Attach the force download queue
            #     self._item_loader._force_download_queue = self._prepare_thread._force_download_queue  # type: ignore
            #     self._prepare_thread.start()
            #     if index.chunk_indexes:
            #         self._prepare_thread.download(index.chunk_indexes)
            #         self._chunks_queued_for_download = True

            # Only request individual chunk download if:
            # 1. We haven't already queued all chunks for the download
            # 2. We're processing a new chunk (different from the last one)
            # if not self._chunks_queued_for_download and index.chunk_index != self._last_chunk_index:
            #     assert self._prepare_thread
            #     self._prepare_thread.download([index.chunk_index])

            if self._last_chunk_index is None or index.chunk_index != self._last_chunk_index:
                self._prepare_thread.current_reading_chunk_index += 1

        # Fetch the element
        chunk_filepath, begin, filesize_bytes = self.config[index]

        if isinstance(self._item_loader, PyTreeLoader):
            item = self._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes, self._encryption
            )
        else:
            item = self._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes
            )

        # We need to request deletion after the latest element has been loaded.
        # Otherwise, this could trigger segmentation fault error depending on the item loader used.
        if self._last_chunk_index is None or index.chunk_index != self._last_chunk_index:
            # Close the memory-mapped file for the last chunk index
            if isinstance(self._item_loader, (TokensLoader, ParquetLoader)) and self._last_chunk_index is not None:
                self._item_loader.close(self._last_chunk_index)

            if self._config and (self._config._remote_dir or self._config._compressor):
                assert self._prepare_thread
                if self._last_chunk_index is not None:
                    # inform the chunk has been completely consumed
                    # self._prepare_thread._decrement_local_lock(self._last_chunk_index)
                    self._prepare_thread.delete([self._last_chunk_index])

            # track the new chunk index as the latest one
            self._last_chunk_index = index.chunk_index

        if index.is_last_index and self._prepare_thread:
            # inform the thread it is time to stop
            # self._prepare_thread._decrement_local_lock(index.chunk_index)
            self._item_loader.close(self._last_chunk_index)
            self._prepare_thread.delete([index.chunk_index, None])  # send this chunk for deletion
            self._prepare_thread.stop()
            self._prepare_thread.join()
            self._prepare_thread = None
            self._last_chunk_index = None
            self._chunks_queued_for_download = False

        return item

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return len(self.config)

    def get_chunk_intervals(self) -> List[Interval]:
        """Get the index interval of each chunk."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self.config.intervals

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_prepare_thread"] = None
        return state

    def __del__(self) -> None:
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
                with contextlib.suppress(FileNotFoundError):
                    size += entry.stat(follow_symlinks=False).st_size

            # warn about unrecognized files
            else:
                logger.warning(
                    f"Ignoring '{filename}': "
                    "This file doesn't appear to be a valid chunk file and has been excluded from the size calculation."
                )
    return size


def _get_from_queue(queue: Queue, timeout: float = _DEFAULT_TIMEOUT) -> Optional[Any]:
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

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
from collections import defaultdict
from contextlib import suppress
from time import sleep, time
from typing import Any, Optional

from filelock import FileLock, Timeout

from litdata.constants import _INDEX_FILENAME, _MAX_WAIT_TIME
from litdata.debugger import _get_log_msg
from litdata.streaming.compression import _COMPRESSORS, Compressor
from litdata.streaming.downloader import get_downloader
from litdata.streaming.item_loader import BaseItemLoader, Interval, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.utilities._pytree import tree_unflatten, treespec_loads
from litdata.utilities.dataset_utilities import load_index_file

logger = logging.getLogger("litdata.streaming.config")


class ChunksConfig:
    def __init__(
        self,
        cache_dir: str,
        serializers: dict[str, Serializer],
        remote_dir: str | None,
        item_loader: BaseItemLoader | None = None,
        subsampled_files: list[str] | None = None,
        region_of_interest: list[tuple[int, int]] | None = None,
        storage_options: dict | None = {},
        session_options: dict | None = {},
    ) -> None:
        """Reads the index files associated a chunked dataset and enables to map an index to its chunk.

        Arguments:
            cache_dir: The path to cache folder.
            serializers: The serializers used to serialize and deserialize the chunks.
            remote_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            item_loader: The item loader used to load the data from the chunks.
            subsampled_files: List of subsampled chunk files loaded from `input_dir/index.json` file.
            region_of_interest: List of tuples of {start,end} of region of interest for each chunk.
            storage_options: Additional connection options for accessing storage services.
            session_options: Additional options for S3 session.

        """
        self._cache_dir = cache_dir
        self._intervals: list[Interval] = []
        self._config = None
        self._chunks = None
        self._remote_dir = remote_dir
        self._item_loader = item_loader or PyTreeLoader()
        self._storage_options = storage_options
        self._session_options = session_options

        # load data from `index.json` file
        data = load_index_file(self._cache_dir)
        _original_chunks = data["chunks"]
        self._config = data["config"]
        self._validate_item_loader()

        assert _original_chunks is not None

        if subsampled_files is None:
            self._chunks = _original_chunks
        else:
            self._chunks = load_subsampled_chunks(subsampled_files, _original_chunks)

        if self._config["data_spec"] is not None:
            self._config["data_spec"] = treespec_loads(self._config["data_spec"])

        assert self._chunks is not None
        self._item_loader.setup(self._config, self._chunks, serializers, region_of_interest)
        self._intervals = self._item_loader.generate_intervals()
        self._length = self._intervals[-1][-1] if len(self._intervals) > 0 else 0
        self._downloader = None

        if remote_dir:
            self._downloader = get_downloader(
                remote_dir, cache_dir, self._chunks, self._storage_options, self._session_options
            )

        self._compressor_name = self._config["compression"]
        self._compressor: Compressor | None = None

        if self._compressor_name:
            if len(_COMPRESSORS) == 0:
                raise ValueError(
                    "No compression algorithms are installed. To use zstd compression,  run `pip install zstd`."
                )
            if self._compressor_name not in _COMPRESSORS:
                raise ValueError(
                    f"The provided compression {self._compressor_name} isn't available in {sorted(_COMPRESSORS)}",
                )
            self._compressor = _COMPRESSORS[self._compressor_name]

        self._skip_chunk_indexes_deletion: list[int] | None = None
        # Chunk indexes that are shared across workers on this node. Shared chunks are
        # reference-counted *eagerly* (incremented at iteration start by the reader), so their
        # lazy download-time increment is skipped to keep the count balanced. See reader.py.
        self._shared_chunk_indexes: set[int] = set()
        self.zero_based_roi: list[tuple[int, int]] | None = None
        # Memoizes ``__getitem__`` results per chunk_index (invariant once the config is loaded);
        # avoids rebuilding the chunk path on every item read.
        self._chunk_meta_cache: dict[int, tuple[str, int, int]] = {}
        self.filename_to_size_map: dict[str, int] = {}
        for cnk in _original_chunks:
            # since files downloaded while reading will be decompressed, we need to store the name without compression
            filename_without_compression = cnk["filename"].replace(f".{self._compressor_name}", "")
            self.filename_to_size_map[filename_without_compression] = cnk["chunk_bytes"]

    def can_delete(self, chunk_index: int) -> bool:
        if self._skip_chunk_indexes_deletion is None:
            return True
        return chunk_index not in self._skip_chunk_indexes_deletion

    def _chunk_lock_filepath(self, chunk_index: int) -> str:
        """The (decompressed) local chunk path whose ``.cnt``/``.lock`` files hold the refcount."""
        chunk_filepath, _, _ = self[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        return chunk_filepath

    def remaining_locks(self, chunk_index: int) -> int:
        """Return the current reference count held on a chunk (0 if none)."""
        countpath = self._chunk_lock_filepath(chunk_index) + ".cnt"
        if not os.path.exists(countpath):
            return 0
        with suppress(FileNotFoundError), open(countpath) as count_f:
            try:
                return int(count_f.read().strip())
            except Exception:
                return 1
        return 0

    def increment_local_lock(self, chunk_index: int) -> None:
        """Add one reference to a chunk's local lock (a co-reader intends to use it)."""
        if self._downloader is None:
            return
        self._downloader._increment_local_lock(self._chunk_lock_filepath(chunk_index), chunk_index)

    def decrement_local_lock(self, chunk_index: int) -> int:
        """Remove one reference from a chunk's local lock; return the remaining count.

        Moved here (from ``PrepareChunksThread``) so the reader can release eagerly-acquired locks
        during teardown without depending on the prefetch thread still being alive.
        """
        countpath = self._chunk_lock_filepath(chunk_index) + ".cnt"
        lock_path = countpath + ".lock"
        curr_count = 0
        remove_lock = False
        with suppress(Timeout, FileNotFoundError), FileLock(lock_path, timeout=3):
            if os.path.exists(countpath):
                with open(countpath) as count_f:
                    try:
                        curr_count = int(count_f.read().strip())
                    except Exception:
                        curr_count = 1
                curr_count -= 1
                if curr_count <= 0:
                    with suppress(FileNotFoundError, PermissionError):
                        os.remove(countpath)
                    remove_lock = True
                else:
                    with open(countpath, "w+") as count_f:
                        logger.debug(_get_log_msg({"name": f"decrement_lock_{chunk_index}_to_{curr_count}", "ph": "B"}))
                        count_f.write(str(curr_count))
                        logger.debug(_get_log_msg({"name": f"decrement_lock_{chunk_index}_to_{curr_count}", "ph": "E"}))
            else:
                remove_lock = True
        # FileLock doesn't delete its lock file on release — we clean it up manually.
        # This must happen after release (Windows can't delete open files) and after the
        # work is done (on Linux, deleting an in-use lock file lets other processes lock
        # on a new inode, bypassing mutual exclusion).
        if remove_lock:
            with suppress(FileNotFoundError, PermissionError):
                os.remove(lock_path)
        return curr_count

    @property
    def skip_chunk_indexes_deletion(self) -> list[int] | None:
        return self._skip_chunk_indexes_deletion

    @skip_chunk_indexes_deletion.setter
    def skip_chunk_indexes_deletion(self, skip_chunk_indexes_deletion: list[int]) -> None:
        self._skip_chunk_indexes_deletion = skip_chunk_indexes_deletion

    def download_chunk_from_index(self, chunk_index: int, skip_lock: bool = False) -> None:
        assert self._chunks is not None
        chunk_filename = self._chunks[chunk_index]["filename"]

        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)

        # Shared chunks are reference-counted eagerly by the reader (before any reading), so their
        # download-time increment is skipped here to avoid double-counting. Non-shared chunks keep
        # the original pay-as-you-download refcounting.
        lazily_ref_counted = chunk_index not in self._shared_chunk_indexes

        if os.path.exists(local_chunkpath):
            self.try_decompress(local_chunkpath)

            if self._downloader is not None and not skip_lock and lazily_ref_counted:
                # We don't want to redownload the base, but we should mark
                # it as having been requested by something
                self._downloader._increment_local_lock(
                    local_chunkpath.replace(f".{self._compressor_name}", ""), chunk_index
                )
            return

        if self._downloader is None:
            return

        if not skip_lock and lazily_ref_counted:
            self._downloader._increment_local_lock(
                local_chunkpath.replace(f".{self._compressor_name}", ""), chunk_index
            )

        self._downloader.download_chunk_from_index(chunk_index)

        self.try_decompress(local_chunkpath)

    def download_chunk_bytes_from_index(self, chunk_index: int, offset: int, length: int) -> bytes:
        assert self._chunks is not None
        chunk_filename = self._chunks[chunk_index]["filename"]

        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)

        if os.path.exists(local_chunkpath):
            with open(local_chunkpath, "rb") as f:
                f.seek(offset)
                return f.read(length)

        if self._compressor is not None:
            raise ValueError(
                "The `download_chunk_bytes_from_index` method is not supported for compressed chunks. "
                "Please, use `download_chunk_from_index` instead."
            )

        if self._downloader is None:
            raise RuntimeError("The downloader is not initialized. Please, initialize it before downloading chunks.")

        return self._downloader.download_chunk_bytes_from_index(chunk_index, offset, length)

    def try_decompress(self, local_chunkpath: str) -> None:
        if self._compressor is None:
            return

        target_local_chunkpath = local_chunkpath.replace(f".{self._compressor_name}", "")

        if os.path.exists(target_local_chunkpath):
            return

        # Wait until either the decompressed target appears (another worker finished) or the
        # compressed source exists. Cloud downloaders publish the compressed path atomically, so
        # existence of that path means the download is complete — do NOT use chunk_size (item
        # count) as a byte threshold.
        start_time = time()
        while not os.path.exists(local_chunkpath) and not os.path.exists(target_local_chunkpath):
            sleep(0.1)
            if (time() - start_time) > _MAX_WAIT_TIME:
                raise FileNotFoundError(f"The {local_chunkpath} hasn't been found.")

        if os.path.exists(target_local_chunkpath):
            return

        decompress_lock = target_local_chunkpath + ".decompress.lock"
        try:
            with FileLock(decompress_lock, timeout=_MAX_WAIT_TIME):
                if os.path.exists(target_local_chunkpath):
                    return

                with open(local_chunkpath, "rb") as f:
                    data = f.read()

                # delete the compressed file only if it was downloaded
                if self._downloader is not None:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(local_chunkpath)

                data = self._compressor.decompress(data)

                assert self._chunks is not None
                filename = os.path.basename(local_chunkpath)
                chunk_index = self._get_chunk_index_from_filename(filename)
                expected_bytes = int(self._chunks[chunk_index]["chunk_bytes"])

                tmp_path = f"{target_local_chunkpath}.tmp.{os.getpid()}"
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(data)
                    if os.stat(tmp_path).st_size < expected_bytes:
                        raise OSError(
                            f"Decompressed chunk {target_local_chunkpath} is smaller than expected "
                            f"({os.stat(tmp_path).st_size} < {expected_bytes})."
                        )
                    os.replace(tmp_path, target_local_chunkpath)
                except Exception:
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        os.remove(tmp_path)
                    raise
        finally:
            # FileLock leaves its lock file behind; remove after release.
            with contextlib.suppress(Exception):
                os.remove(decompress_lock)

    @property
    def intervals(self) -> list[Interval]:
        if self._intervals is None:
            raise RuntimeError("The intervals should be defined.")
        return self._intervals

    @property
    def num_bytes(self) -> int:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        assert self._chunks is not None
        return sum(c["chunk_bytes"] for c in self._chunks)

    @property
    def data_format(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["data_format"]

    @property
    def data_format_unflattened(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return tree_unflatten(self._config["data_format"], self._config["data_spec"])

    @property
    def compression(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["compression"]

    @property
    def chunk_bytes(self) -> int:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["chunk_bytes"]

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    def _get_chunk_index_from_index(self, index: int) -> tuple[int, int]:
        if self.zero_based_roi is None:
            # zero_based_roi is a list of tuples (start, end),
            # to efficiently find the chunk index.
            # Example:
            #  self._intervals = [(0, 5, 10, 10), (10, 10, 20, 20)]
            #  self.zero_based_roi = [(0, 5), (5, 15)]

            self.zero_based_roi = []
            start = 0
            for curr_interval in self._intervals:
                diff = curr_interval[2] - curr_interval[1]  # roi_start, roi_end
                self.zero_based_roi.append((start, start + diff))
                start += diff

        for chunk_index, internal in enumerate(self.zero_based_roi):
            if internal[0] <= index < internal[-1]:
                real_index_to_read_from = self._intervals[chunk_index][1] + (index - internal[0])
                return real_index_to_read_from, chunk_index
        raise ValueError(
            f"The provided index {index} didn't find a match within the chunk intervals {self._intervals}."
        )

    def __getitem__(self, index: ChunkedIndex) -> tuple[str, int, int]:
        """Find the associated chunk metadata.

        This is called once per item on the read hot path, but its result depends only on
        ``index.chunk_index`` (the local path, the chunk's begin offset and its byte size are all
        fixed once the config is loaded). The per-chunk tuple is therefore memoized to avoid
        rebuilding the path (``os.path.join`` + decompression-suffix stripping) on every item.
        """
        cached = self._chunk_meta_cache.get(index.chunk_index)
        if cached is not None:
            return cached

        assert self._chunks is not None
        chunk = self._chunks[index.chunk_index]

        local_chunkpath = os.path.join(self._cache_dir, chunk["filename"])

        if self._compressor is not None:
            local_chunkpath = local_chunkpath.replace(f".{self._compressor_name}", "")

        begin = self._intervals[index.chunk_index][0]

        filesize_bytes = chunk["chunk_bytes"]

        meta = (local_chunkpath, begin, filesize_bytes)
        self._chunk_meta_cache[index.chunk_index] = meta
        return meta

    def download_filepath(self, chunk_index: int) -> str:
        """The raw on-disk path that the chunk is downloaded to before any decompression."""
        assert self._chunks is not None
        return os.path.join(self._cache_dir, self._chunks[chunk_index]["filename"])

    def _get_chunk_index_from_filename(self, chunk_filename: str) -> int:
        """Retrieves the associated chunk_index for a given chunk filename."""
        assert self._chunks is not None
        for chunk_index, chunk in enumerate(self._chunks):
            if chunk["filename"] == chunk_filename:
                return chunk_index
        raise ValueError(f"The provided filename doesn't exist {chunk_filename}.")

    @classmethod
    def load(
        cls,
        cache_dir: str,
        serializers: dict[str, Serializer],
        remote_dir: str | None = None,
        item_loader: BaseItemLoader | None = None,
        subsampled_files: list[str] | None = None,
        region_of_interest: list[tuple[int, int]] | None = None,
        storage_options: dict | None = {},
        session_options: dict | None = {},
    ) -> Optional["ChunksConfig"]:
        cache_index_filepath = os.path.join(cache_dir, _INDEX_FILENAME)

        if isinstance(remote_dir, str):
            # for remote_dir, we try downloading `index.json` file.
            # If the files are stored on HF, they don't have an index file, so we can skip downloading it.
            if remote_dir.startswith("hf://"):
                if not os.path.exists(cache_index_filepath):
                    raise RuntimeError(
                        f"This should not have happened. No index.json file found in cache: {cache_index_filepath}"
                    )
            else:
                downloader = get_downloader(remote_dir, cache_dir, [], storage_options, session_options)
                downloader.download_file(os.path.join(remote_dir, _INDEX_FILENAME), cache_index_filepath)

        if not os.path.exists(cache_index_filepath):
            return None

        return ChunksConfig(
            cache_dir,
            serializers,
            remote_dir,
            item_loader,
            subsampled_files,
            region_of_interest,
            storage_options,
            session_options,
        )

    def __len__(self) -> int:
        return self._length

    def _validate_item_loader(self) -> None:
        assert self._config
        if "item_loader" in self._config:
            if self._item_loader.__class__.__name__ != self._config["item_loader"]:
                item_loader = self._config["item_loader"]
                raise ValueError(f"Please, use Cache(..., item_loader={item_loader}(...))")
        else:
            if (
                len(self._config["data_format"]) == 1
                and self._config["data_format"][0].startswith("no_header_tensor")
                and not isinstance(self._item_loader, TokensLoader)
            ):
                raise ValueError("Please, use Cache(..., item_loader=TokensLoader(block_size=...))")


def load_subsampled_chunks(subsampled_files: list[str], original_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Loads Chunks based on subsample provided."""
    _subsampled_chunks: list[dict[str, Any]] = [{} for _ in range(len(subsampled_files))]

    assert len(_subsampled_chunks) == len(subsampled_files)

    filename_dict = defaultdict(list)

    # Populate the dictionary with filenames and their indices
    for index, filename in enumerate(subsampled_files):
        filename_dict[filename].append(index)

    for curr_chunk in original_chunks:
        if curr_chunk["filename"] in filename_dict:
            for idx in filename_dict[curr_chunk["filename"]]:
                _subsampled_chunks[idx] = curr_chunk

    # if any idx of _subsampled_chunks is None, means,
    # some elements in subsampled_files were not actually part of chunks
    # raise error
    if any(not _subsampled_chunk for _subsampled_chunk in _subsampled_chunks):
        raise ValueError(
            "Mismatch in subsampled files and the chunks loaded",
            "Make sure subsampled chunks are actually part of the original chunk",
        )

    return _subsampled_chunks

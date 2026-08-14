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
import functools
import logging
import mmap
import os
import struct
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from io import BytesIO, FileIO
from multiprocessing import Queue
from threading import Event
from time import sleep, time
from typing import Any

import numpy as np
import torch

from litdata.constants import (
    _DEBUG,
    _FORCE_DOWNLOAD_TIME,
    _MAX_WAIT_TIME,
    _NUMPY_DTYPES_MAPPING,
    _POLARS_AVAILABLE,
    _PYARROW_AVAILABLE,
    _TORCH_DTYPES_MAPPING,
)
from litdata.debugger import CAT_DELETE, trace_span
from litdata.exceptions import ChunkWaitTimeoutError
from litdata.streaming.posix_fast import advise_willneed, madvise_mmap, posix_page_bytes
from litdata.streaming.serializers import Serializer
from litdata.utilities._pytree import SUPPORTED_NODES, PyTree, TreeSpec, tree_unflatten
from litdata.utilities.encryption import Encryption, EncryptionLevel

Interval = namedtuple("Interval", ["chunk_start", "roi_start_idx", "roi_end_idx", "chunk_end"])

logger = logging.getLogger("litdata.streaming.item_loader")


def _open_chunk_file(chunk_filepath: str) -> FileIO:
    """Open a chunk for reading, retrying Windows ``PermissionError`` races.

    On Windows, antivirus / ``os.replace`` from decompression can briefly deny
    ``open()`` even after the file exists at the expected size. Retry with a short
    backoff instead of failing the read.
    """
    last_err: PermissionError | None = None
    for attempt in range(20):
        try:
            return open(chunk_filepath, "rb", 0)
        except PermissionError as e:
            last_err = e
            sleep(0.05)
    assert last_err is not None
    raise last_err


# Module-level unflatten callables (not nested closures) so item loaders remain picklable for
# DataLoader workers under the ``spawn`` start method.


class _LeafUnflatten:
    __slots__ = ()

    def __call__(self, leaves: list[Any]) -> Any:
        return leaves[0]


class _DictOfLeavesUnflatten:
    __slots__ = ("keys",)

    def __init__(self, keys: tuple[Any, ...]) -> None:
        self.keys = keys

    def __call__(self, leaves: list[Any]) -> dict[Any, Any]:
        return dict(zip(self.keys, leaves))


class _ListOfLeavesUnflatten:
    __slots__ = ()

    def __call__(self, leaves: list[Any]) -> list[Any]:
        return list(leaves)


class _TupleOfLeavesUnflatten:
    __slots__ = ()

    def __call__(self, leaves: list[Any]) -> tuple[Any, ...]:
        return tuple(leaves)


class _NestedUnflatten:
    __slots__ = ("child_leaf_counts", "child_runners", "context", "unflatten_fn")

    def __init__(
        self,
        unflatten_fn: Any,
        child_runners: list[Any],
        child_leaf_counts: list[int],
        context: Any,
    ) -> None:
        self.unflatten_fn = unflatten_fn
        self.child_runners = child_runners
        self.child_leaf_counts = child_leaf_counts
        self.context = context

    def __call__(self, leaves: list[Any]) -> Any:
        values = []
        start = 0
        for runner, count in zip(self.child_runners, self.child_leaf_counts):
            end = start + count
            # Children that are themselves leaves can index directly; nested children get a slice.
            if count == 1 and isinstance(runner, _LeafUnflatten):
                values.append(leaves[start])
            else:
                values.append(runner(leaves[start:end]))
            start = end
        return self.unflatten_fn(values, self.context)


_LEAF_UNFLATTEN = _LeafUnflatten()
_LIST_OF_LEAVES_UNFLATTEN = _ListOfLeavesUnflatten()
_TUPLE_OF_LEAVES_UNFLATTEN = _TupleOfLeavesUnflatten()


def _compile_treespec_unflatten(spec: TreeSpec) -> Any:
    """Compile a ``TreeSpec`` into a fast, picklable ``leaves -> tree`` callable.

    The stock ``tree_unflatten`` is recursive and slices the leaves list at every node, which
    dominates the per-item cost for typical nested samples. Compiling once per dataset replaces
    that with a tight index walk over the flat leaf list.
    """
    if spec.is_leaf():
        return _LEAF_UNFLATTEN

    # Fast paths for the shapes that dominate StreamingDataset workloads.
    children = spec.children_specs
    if all(child.is_leaf() for child in children):
        if spec.type is dict:
            return _DictOfLeavesUnflatten(tuple(spec.context))
        if spec.type is list:
            return _LIST_OF_LEAVES_UNFLATTEN
        if spec.type is tuple:
            return _TUPLE_OF_LEAVES_UNFLATTEN

    return _NestedUnflatten(
        SUPPORTED_NODES[spec.type].unflatten_fn,
        [_compile_treespec_unflatten(child) for child in children],
        [child.num_leaves for child in children],
        spec.context,
    )


class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: list[tuple[int, int]] | None = None,
        force_download_queue: "Queue | None" = None,
    ) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = {**serializers}
        self._data_format = self._config["data_format"]
        self._shift_idx = len(self._data_format) * 4  # each item takes 4 bytes
        self.region_of_interest = region_of_interest
        self._force_download_queue = force_download_queue
        # Optional provider of per-chunk readiness Events from PrepareChunksThread.
        self._chunk_ready_provider: Callable[[int], Event] | None = getattr(self, "_chunk_ready_provider", None)
        # Optional provider of a crash from PrepareChunksThread (so waiters fail fast).
        self._prefetch_error_provider: Callable[[], BaseException | None] | None = getattr(
            self, "_prefetch_error_provider", None
        )

        # setup the serializers on restart
        for data_format in self._data_format:
            serializer = deepcopy(self._serializers[self._data_format_to_key(data_format)])
            serializer.setup(data_format)
            self._serializers[data_format] = serializer

        # Precompute the per-leaf serializer list and the data spec so the per-item `deserialize`
        # hot path avoids a dict lookup per leaf and a config lookup per item.
        self._serializers_list = [self._serializers[data_format] for data_format in self._data_format]
        self._data_spec = self._config["data_spec"]
        # Compile a specialized unflatten for this dataset's fixed treespec. Falls back to the
        # stock pytree path only when there is no data_spec (e.g. some parquet/MDS shapes).
        self._unflatten = (
            _compile_treespec_unflatten(self._data_spec) if isinstance(self._data_spec, TreeSpec) else None
        )
        # Fixed size-header layout: one little-endian uint32 per leaf.
        # Keep a format string (pickle-friendly) rather than a ``struct.Struct`` instance.
        self._sizes_fmt = "<" + "I" * len(self._data_format) if self._data_format else None

    def force_download(self, chunk_index: int) -> None:
        force_download_queue = getattr(self, "_force_download_queue", None)
        if force_download_queue:
            force_download_queue.put(chunk_index)

    def set_posix_fast(self, enabled: bool, keep: int = 4, *, willneed: bool = True) -> None:
        """Enable in-place parallel-FS reads (Vast/NFS). Default loaders ignore this."""
        del keep, willneed

    def warm_posix_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Advise and mmap ``chunk_filepath`` without making it the current item (POSIX-fast)."""
        self.pre_load_chunk(chunk_index, chunk_filepath)

    def set_mmap_allowed_chunks(self, chunk_indexes: set[int]) -> None:
        """Declare which chunks are safe to memory-map (i.e. not shared with another worker).

        Only ``PyTreeLoader`` acts on this; other loaders ignore it. Memory-mapping a chunk that a
        co-worker may delete/replace while it is mapped can crash with SIGSEGV (see issues #459,
        #756), so only non-shared chunks are mapped.
        """

    def set_chunk_ready_provider(self, provider: Callable[[int], Event] | None) -> None:
        """Install a provider of per-chunk readiness Events from the prefetch thread."""
        self._chunk_ready_provider = provider

    def set_prefetch_error_provider(self, provider: Callable[[], BaseException | None] | None) -> None:
        """Install a provider that returns a prefetch-thread crash, if any."""
        self._prefetch_error_provider = provider

    def _raise_if_prefetch_crashed(self, chunk_filepath: str) -> None:
        """Re-raise a PrepareChunksThread crash so waiters do not time out as FileNotFoundError."""
        prefetch_error_provider = getattr(self, "_prefetch_error_provider", None)
        if prefetch_error_provider is None:
            return
        err = prefetch_error_provider()
        if err is None:
            return
        raise RuntimeError(
            f"Chunk prefetch thread crashed while waiting for {chunk_filepath}. "
            f"Original error: {type(err).__name__}: {err}"
        ) from err

    def _wait_until_chunk_ready(self, chunk_index: int, chunk_filepath: str, filesize_bytes: int) -> None:
        """Block until ``chunk_filepath`` exists and is at least ``filesize_bytes``.

        Prefers the in-process readiness Event (set by ``PrepareChunksThread`` after download /
        decompress) and falls back to a short filesystem poll so co-worker downloads still work.

        If a readiness Event is already set but the file is missing (e.g. the chunk was deleted
        after a prior download), clear the Event and sleep so we do not busy-spin and starve the
        prefetch thread under the GIL.

        Without a prefetch thread / force-download queue (local uncompressed caches), the chunk
        should already be on disk — fail fast instead of polling for ``_MAX_WAIT_TIME`` (often the
        same as the test timeout), which otherwise looks like a DataLoader worker hang.

        Attributes are read via ``getattr`` because some loaders (e.g. ``ParquetLoader``) do not
        call ``BaseItemLoader.setup``, and unit tests may invoke this helper before setup.
        """
        start_time = time()
        requested_force_download = False
        chunk_ready_provider = getattr(self, "_chunk_ready_provider", None)
        force_download_queue = getattr(self, "_force_download_queue", None)
        # Remote/prefetch path keeps the long timeout; local-only missing files fail quickly.
        max_wait = (
            _MAX_WAIT_TIME
            if chunk_ready_provider is not None or force_download_queue is not None
            else min(2.0, float(_MAX_WAIT_TIME))
        )

        while True:
            self._raise_if_prefetch_crashed(chunk_filepath)
            if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes:
                return

            if chunk_ready_provider is not None:
                event = chunk_ready_provider(chunk_index)
                signaled = event.wait(timeout=0.1)
                # Stale signal: chunk was ready once, then deleted / not yet re-published.
                if signaled and not (
                    os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes
                ):
                    event.clear()
                    sleep(0.1)
            else:
                sleep(0.1)

            # Always attempt force-download after the grace period (no-op without a queue).
            # Tests override ``force_download`` to assert this path is reached.
            if not requested_force_download and (time() - start_time) > _FORCE_DOWNLOAD_TIME:
                if _DEBUG:
                    print(f"[ItemLoader] Requested force download for {chunk_filepath} at {datetime.now().isoformat()}")
                self.force_download(chunk_index)
                requested_force_download = True

            if (time() - start_time) > max_wait:
                raise ChunkWaitTimeoutError(chunk_filepath, time() - start_time)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Prefetch-thread handles are process-local and reattached after worker spawn.
        state["_chunk_ready_provider"] = None
        state["_force_download_queue"] = None
        state["_prefetch_error_provider"] = None
        return state

    @functools.lru_cache(maxsize=128)
    def _data_format_to_key(self, data_format: str) -> str:
        if ":" in data_format:
            serialier, serializer_sub_type = data_format.split(":")
            if serializer_sub_type in self._serializers:
                return serializer_sub_type
            return serialier
        return data_format

    def state_dict(self) -> dict:
        return {}

    @abstractmethod
    def generate_intervals(self) -> list[Interval]:
        """Returns a list of intervals.

        The structure is: [chunk_start, region_of_interest_start, region_of_interest_end, chunk_end]

        region_of_interest: indicates the indexes a chunk our StreamingDataset is allowed to read.
        """

    @abstractmethod
    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""

    @abstractmethod
    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        """Returns an item loaded from a chunk."""

    def load_item_from_bytes(
        self,
        raw_bytes: bytes,
        chunk_index: int,
    ) -> Any:
        """Returns an item loaded from bytes."""
        raise NotImplementedError("The `load_item_from_bytes` method is not implemented for this item loader.")

    @abstractmethod
    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""

    @abstractmethod
    def encode_data(self, data: list[bytes], sizes: list[int], flattened: list[Any]) -> Any:
        pass


class PyTreeLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(self) -> None:
        super().__init__()
        self._chunk_filepath: str | None = None
        self._decrypted_chunks: dict[int, bytes] = {}
        self._open_handle: FileIO | None = None
        # Memory-map + cached offset table for the current chunk, used only for chunks that are
        # safe to map (non-shared, unencrypted). Per-item reads then become one mmap slice instead
        # of two `seek`+`read` syscalls on the unbuffered handle.
        self._mmap: mmap.mmap | None = None
        # Owned copy of the chunk offset table as plain ints (not a view into the mmap).
        self._offsets: list[int] | None = None
        self._mmap_allowed_chunks: set[int] = set()
        self._posix_fast = False
        self._mmap_keep = 1
        self._mapped: OrderedDict[int, tuple[mmap.mmap, list[int], str]] = OrderedDict()
        self._mmap_handles: dict[int, FileIO] = {}
        self._page: memoryview | bytes | None = None
        self._page_chunk: int | None = None
        self._page_start = 0
        self._page_end = 0
        self._page_byte0 = 0
        self._page_bytes = 0
        self._mmap_view: memoryview | None = None
        self._posix_willneed = True

    def set_posix_fast(self, enabled: bool, keep: int = 4, *, willneed: bool = True) -> None:
        self._posix_fast = enabled
        self._posix_willneed = willneed
        self._mmap_keep = max(1, keep) if enabled else 1
        self._page_bytes = posix_page_bytes() if enabled else 0
        self._clear_item_page()

    def _clear_item_page(self) -> None:
        self._page = None
        self._page_chunk = None
        self._page_start = 0
        self._page_end = 0
        self._page_byte0 = 0

    def set_mmap_allowed_chunks(self, chunk_indexes: set[int]) -> None:
        self._mmap_allowed_chunks = chunk_indexes

    def generate_intervals(self) -> list[Interval]:
        intervals = []
        begin = 0
        end = 0
        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        if self._posix_fast and self._posix_willneed:
            advise_willneed(chunk_filepath)

    def warm_posix_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        self.pre_load_chunk(chunk_index, chunk_filepath)
        if not self._posix_fast:
            return
        if self._config.get("encryption") or chunk_index not in self._mmap_allowed_chunks:
            return
        self._ensure_chunk_mmap(chunk_filepath, chunk_index, make_current=False)

    def load_item_from_bytes(
        self,
        raw_bytes: bytes,
        chunk_index: int,
    ) -> bytes:
        if self._config.get("encryption"):
            raise ValueError("The `load_item_from_bytes` method does not support encrypted data loading currently.")

        # check for mosaic mds format
        if "format" in self._config and self._config["format"] == "mds":
            item_data = self.mds_deserialize(raw_bytes, chunk_index)
        else:
            item_data = self.deserialize(raw_bytes)

        return item_data

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
        encryption: Encryption | None = None,
    ) -> bytes:
        #
        # Let's say, a chunk contains items from [5,9] index.
        # And the index of the item we want to load is 7.
        # begin = 5
        # index = 7
        #
        # The chunk's binary format is structured as follows:
        #
        # +------------+---------------+-------------+
        # | num_items  | offset_array  | item_data   |
        # +------------+---------------+-------------+
        # | uint32     | uint32[N+1]   | bytes       |
        # | 4 bytes    | 4*(N+1) bytes | variable    |
        # +------------+---------------+-------------+
        #
        # To get to the offset index of the item we want to load, we need to jumpy by:
        #       => 1 + (index - begin) # 1 is added since first 4 bytes store `num_items` (1 uint32)
        #       => 1 + (7 - 5) = 3
        #       => 3 * 4 = 12          # each takes 4 bytes
        #       => offset = 12
        #
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath != self._chunk_filepath:
            start_time = time()
            self._wait_until_chunk_ready(chunk_index, chunk_filepath, filesize_bytes)

            if _DEBUG and time() - start_time > 5:
                print("WAIT TIME", time() - start_time)

            cached = self._mapped.get(chunk_index)
            if cached is not None and cached[2] == chunk_filepath:
                self._apply_mapped_chunk(chunk_index, cached)
            else:
                self._chunk_filepath = chunk_filepath
                if not self._posix_fast:
                    self._close_open_chunk()
                if self._config.get("encryption") or chunk_index not in self._mmap_allowed_chunks:
                    self._clear_item_page()
                    self._mmap_view = None
                    self._open_handle = _open_chunk_file(chunk_filepath)
                    self._mmap = None
                    self._offsets = None
                else:
                    self._ensure_chunk_mmap(chunk_filepath, chunk_index, make_current=True)

        if self._config.get("encryption"):
            data = self._load_encrypted_data(chunk_filepath, chunk_index, offset, encryption)
        elif self._mmap is not None:
            # `offset` points at the item's start entry in the offset table (byte (i+1)*4 holds
            # entry i), so this item's table index is `offset // 4 - 1`.
            # `mmap[start:end]` returns a fresh `bytes` object directly — no memoryview hop.
            assert self._offsets is not None
            table_idx = offset // 4 - 1
            data = self._slice_item_bytes(table_idx, chunk_index)
        else:
            assert self._open_handle
            # load the data from raw bytes using the offset for the item we want to load
            data = self._load_data(self._open_handle, offset)

        # check for mosaic mds format
        if "format" in self._config and self._config["format"] == "mds":
            item_data = self.mds_deserialize(data, chunk_index)
        else:
            item_data = self.deserialize(data)

        return item_data

    def _load_encrypted_data(
        self, chunk_filepath: str, chunk_index: int, offset: int, encryption: Encryption | None
    ) -> bytes:
        """Load and decrypt data from chunk based on the encryption configuration."""
        # Validate the provided encryption object against the expected configuration.
        self._validate_encryption(encryption)

        # chunk-level decryption
        if self._config["encryption"]["level"] == EncryptionLevel.CHUNK:
            decrypted_data = self._decrypted_chunks.get(chunk_index, None)
            if decrypted_data is None:
                with open(chunk_filepath, "rb", 0) as fp:
                    encrypted_data = fp.read()
                    decrypted_data = encryption.decrypt(encrypted_data)  # type: ignore
                    # Store the decrypted chunk to avoid re-decryption,
                    # also allows to free the previous chunk from the memory
                    self._decrypted_chunks = {chunk_index: decrypted_data}
            data = self._load_data(BytesIO(decrypted_data), offset)

        # sample-level decryption
        elif self._config["encryption"]["level"] == EncryptionLevel.SAMPLE:
            with open(chunk_filepath, "rb", 0) as fp:
                data = self._load_data(fp, offset)
                data = encryption.decrypt(data)  # type: ignore

        else:
            raise ValueError("Invalid encryption level.")

        return data

    def _load_data(self, fp: FileIO | BytesIO, offset: int) -> bytes:
        """Load the data from the file pointer."""
        fp.seek(offset)  # move the file pointer to the offset

        # Refer to `writer.py::_create_chunk` for more details on the chunk's binary format
        # We want to read the `offset_start` and `offset_end` for the item we want to load
        # 2 uint32 (4 bytes each) => 8 bytes; are read to get the offset_start and offset_end
        pair = fp.read(8)
        begin, end = np.frombuffer(pair, np.uint32)

        fp.seek(begin)  # move the file pointer to the offset_start where the item starts
        return fp.read(end - begin)  # read the item

    def _slice_item_bytes(self, table_idx: int, chunk_index: int) -> bytes | memoryview:
        """Return one item as a view into the mapped page when POSIX-fast."""
        assert self._mmap is not None
        assert self._offsets is not None
        start = self._offsets[table_idx]
        end = self._offsets[table_idx + 1]
        if self._posix_fast and self._page_bytes > 0:
            if (
                self._page is None
                or self._page_chunk != chunk_index
                or table_idx < self._page_start
                or table_idx >= self._page_end
            ):
                self._fill_item_page(table_idx, chunk_index)
            assert self._page is not None
            rel0 = start - self._page_byte0
            rel1 = end - self._page_byte0
            return self._page[rel0:rel1]
        if self._mmap_view is not None:
            return self._mmap_view[start:end]
        return self._mmap[start:end]

    def _fill_item_page(self, table_idx: int, chunk_index: int) -> None:
        """Remember a contiguous span of the mapping (no extra ``bytes`` copy)."""
        assert self._mmap is not None
        assert self._offsets is not None
        n_items = len(self._offsets) - 1
        byte0 = self._offsets[table_idx]
        limit = byte0 + self._page_bytes
        end_idx = table_idx + 1
        while end_idx < n_items and self._offsets[end_idx] <= limit:
            end_idx += 1
        byte1 = self._offsets[end_idx]
        view = self._mmap_view if self._mmap_view is not None else memoryview(self._mmap)
        self._page = view[byte0:byte1]
        self._page_chunk = chunk_index
        self._page_start = table_idx
        self._page_end = end_idx
        self._page_byte0 = byte0

    def mds_deserialize(self, raw_item_data: bytes | memoryview, chunk_index: int) -> "PyTree":
        """Deserialize the mds raw bytes into their python equivalent."""
        idx = 0
        sizes = []
        column_sizes = self._chunks[chunk_index]["column_sizes"]
        # adapted from: MDSReader.deserialize : https://github.com/mosaicml/streaming/blob/main/streaming/base/format/mds/reader.py
        for size in column_sizes:
            if size:
                sizes.append(size)
            else:
                (size,) = np.frombuffer(raw_item_data[idx : idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        data = []
        for size, data_format in zip(sizes, self._data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            if not isinstance(data_bytes, (bytes, bytearray)):
                data_bytes = bytes(data_bytes)
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def deserialize(self, raw_item_data: bytes | memoryview) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = self._shift_idx
        sizes = struct.unpack_from(self._sizes_fmt, raw_item_data, 0) if self._sizes_fmt is not None else ()
        data = []
        for size, serializer in zip(sizes, self._serializers_list):
            data_bytes = raw_item_data[idx : idx + size]
            if not isinstance(data_bytes, (bytes, bytearray)):
                data_bytes = bytes(data_bytes)
            data.append(serializer.deserialize(data_bytes))
            idx += size
        if self._unflatten is not None:
            return self._unflatten(data)
        return tree_unflatten(data, self._data_spec)

    def _ensure_chunk_mmap(self, chunk_filepath: str, chunk_index: int, *, make_current: bool) -> None:
        """Map ``chunk_filepath`` into the LRU; optionally make it the active item mapping."""
        cached = self._mapped.get(chunk_index)
        if cached is not None and cached[2] == chunk_filepath:
            if make_current:
                self._apply_mapped_chunk(chunk_index, cached)
            else:
                self._mapped.move_to_end(chunk_index)
            return
        handle = _open_chunk_file(chunk_filepath)
        chunk_mmap = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        if os.name == "nt":
            self._mmap_handles[chunk_index] = handle
        else:
            handle.close()
        header_num_items = int(np.frombuffer(chunk_mmap, dtype=np.uint32, count=1, offset=0)[0])
        index_num_items = int(self._chunks[chunk_index]["chunk_size"])
        if header_num_items != index_num_items:
            chunk_mmap.close()
            handle = self._mmap_handles.pop(chunk_index, None)
            if handle is not None:
                handle.close()
            raise RuntimeError(
                f"Chunk {chunk_index} header item count ({header_num_items}) does not match "
                f"index.json chunk_size ({index_num_items}) for {chunk_filepath}."
            )
        offsets = np.frombuffer(chunk_mmap, dtype=np.uint32, count=header_num_items + 1, offset=4).tolist()
        madvise_mmap(chunk_mmap, willneed=self._posix_willneed)
        self._mapped[chunk_index] = (chunk_mmap, offsets, chunk_filepath)
        self._mapped.move_to_end(chunk_index)
        self._evict_mapped_chunks(protect=None if make_current else chunk_index)
        if make_current:
            self._apply_mapped_chunk(chunk_index, self._mapped[chunk_index])

    def _apply_mapped_chunk(self, chunk_index: int, cached: tuple[mmap.mmap, list[int], str]) -> None:
        self._clear_item_page()
        chunk_mmap, offsets, chunk_filepath = cached
        self._mmap = chunk_mmap
        self._open_handle = None
        self._offsets = offsets
        self._chunk_filepath = chunk_filepath
        self._mmap_view = memoryview(chunk_mmap)
        self._mapped.move_to_end(chunk_index)

    def _close_mapping(self, chunk_index: int, chunk_mmap: mmap.mmap) -> None:
        """Drop views into ``chunk_mmap`` then close it so the fd is released."""
        if self._mmap is chunk_mmap:
            self._mmap = None
            self._mmap_view = None
            self._offsets = None
            self._open_handle = None
        if self._page_chunk == chunk_index:
            self._clear_item_page()
        handle = self._mmap_handles.pop(chunk_index, None)
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        with contextlib.suppress(BufferError, ValueError, OSError):
            chunk_mmap.close()

    def _evict_mapped_chunks(self, protect: int | None = None) -> None:
        for old_idx in list(self._mapped):
            if len(self._mapped) <= self._mmap_keep:
                break
            if old_idx == protect:
                continue
            old_mmap, _, _ = self._mapped[old_idx]
            if old_mmap is self._mmap:
                continue
            self._mapped.pop(old_idx)
            self._close_mapping(old_idx, old_mmap)

    def _close_open_chunk(self) -> None:
        """Release every memory-map / file handle this loader still holds."""
        self._clear_item_page()
        self._mmap_view = None
        self._offsets = None
        self._mmap = None
        self._open_handle = None
        for idx in list(self._mapped):
            mm, _, _ = self._mapped.pop(idx)
            self._close_mapping(idx, mm)

    def close(self, chunk_index: int) -> None:
        """Close the open file handle / memory-map for the current chunk."""
        del chunk_index
        self._close_open_chunk()
        self._chunk_filepath = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._close_open_chunk()

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if getattr(self, "_posix_fast", False):
            return
        with trace_span("delete", CAT_DELETE, chunk=chunk_index):
            if os.path.exists(chunk_filepath):
                if _DEBUG:
                    print(f"delete_chunk_{chunk_index}")
                os.remove(chunk_filepath)

    def _validate_encryption(self, encryption: Encryption | None) -> None:
        """Validate the encryption object."""
        if not encryption:
            raise ValueError("Data is encrypted but no encryption object was provided.")
        if encryption.algorithm != self._config["encryption"]["algorithm"]:
            raise ValueError("Encryption algorithm mismatch.")
        if encryption.level != self._config["encryption"]["level"]:
            raise ValueError("Encryption level mismatch.")

    @classmethod
    def encode_data(cls, data: list[bytes], sizes: list[int], flattened: list[Any]) -> tuple[bytes, int | None]:
        """Encodes multiple serialized objects into a single binary format with size metadata.

        This method combines multiple serialized objects into a single byte array, prefixed with their sizes.
        The resulting format is: [size_header][concatenated_data], where size_header contains the byte sizes
        of each object encoded as uint32.

        Args:
            data: List of serialized objects as bytes
            sizes: List of integers representing the byte size of each object
            flattened: List of flattened pytree leaves

        Returns:
            Tuple containing:
                - bytes: Combined binary data with header
                - Optional[int]: dimension of the item (None for PyTreeLoader)

        Example:
            For a row containing [int, image, tensor]:
            - sizes might be [4, 100000, 1000] (number of bytes for each object)
            - data would be their respective serialized bytes
            The method combines these into:

                [size_bytes][int_bytes][image_bytes][tensor_bytes]
        """
        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body, None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # File handle / memory-map are per-process and not picklable; lazily re-created on the
        # first read in the receiving process.
        state["_open_handle"] = None
        state["_chunk_filepath"] = None
        state["_mmap"] = None
        state["_offsets"] = None
        state["_mapped"] = OrderedDict()
        state["_mmap_handles"] = {}
        state["_page"] = None
        state["_page_chunk"] = None
        state["_mmap_view"] = None
        # Compiled unflatten closures aren't picklable; rebuild after unpickle.
        state["_unflatten"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_page"):
            self._page = None
            self._page_chunk = None
            self._page_start = 0
            self._page_end = 0
            self._page_byte0 = 0
            self._page_bytes = 0
        if not hasattr(self, "_mmap_view"):
            self._mmap_view = None
        if not hasattr(self, "_posix_willneed"):
            self._posix_willneed = True
        data_spec = getattr(self, "_data_spec", None)
        if isinstance(data_spec, TreeSpec):
            self._unflatten = _compile_treespec_unflatten(data_spec)


class TokensLoader(BaseItemLoader):
    def __init__(self, block_size: int | None = None):
        """The Tokens Loader is an optimizer item loader for NLP.

        Args:
            block_size: The context length to use during training.

        """
        super().__init__()
        self._block_size = block_size
        self._mmaps: dict[int, np.memmap] = {}
        self._buffers: dict[int, bytes] = {}
        # keeps track of number of readers for each chunk (can be more than 1 if multiple workers are reading)
        self._counter = defaultdict(int)
        self._posix_fast = False
        self._posix_willneed = True
        self._mmap_keep = 1
        self._dtype: torch.dtype | None = None
        self._chunk_filepaths: dict[str, bool] = {}

    def state_dict(self) -> dict:
        assert self._block_size
        return {
            "block_size": self._block_size,
        }

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: list[tuple[int, int]] | None = None,
    ) -> None:
        super().setup(config, chunks, serializers, region_of_interest)

        serializer_name, dtype_index = self._data_format[0].split(":")
        if serializer_name not in ["no_header_numpy", "no_header_tensor"]:
            raise ValueError("The provided data format isn't supported.")

        self._serializer_name = serializer_name
        self._dtype = (
            _TORCH_DTYPES_MAPPING[int(dtype_index)]  # type: ignore
            if serializer_name == "no_header_tensor"
            else _NUMPY_DTYPES_MAPPING[int(dtype_index)]
        )
        if all(chunk["dim"] is None for chunk in self._chunks):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> list[Interval]:
        assert self._block_size
        intervals = []
        begin = 0
        end = 0
        for idx, chunk in enumerate(self._chunks):
            dim = chunk["dim"]  # number of tokens in the chunk
            num_blocks = dim // self._block_size
            end += num_blocks
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]
            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += num_blocks
        return intervals

    def set_posix_fast(self, enabled: bool, keep: int = 4, *, willneed: bool = True) -> None:
        self._posix_fast = enabled
        self._posix_willneed = willneed
        self._mmap_keep = max(1, keep) if enabled else 1

    def warm_posix_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Page-cache hint only. Mapping every upcoming chunk leaked fds (CI EMFILE)."""
        del chunk_index
        if self._posix_willneed:
            advise_willneed(chunk_filepath)

    def _evict_token_mmaps(self, protect: int | None = None) -> None:
        if not self._posix_fast or self._mmap_keep <= 0:
            return
        for old_idx in list(self._mmaps):
            if len(self._mmaps) <= self._mmap_keep:
                break
            if old_idx == protect:
                continue
            buf = self._buffers.pop(old_idx, None)
            del buf
            mm = self._mmaps.pop(old_idx)
            with contextlib.suppress(BufferError, ValueError, OSError):
                mm._mmap.close()
            self._counter.pop(old_idx, None)

    def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        if chunk_index in self._mmaps:
            return
        chunk = self._chunks[chunk_index]

        # Skip the header
        # [number of items] + [number of offsets (number of items in the chunk + 1)] {since offset starts at 0}
        # multiplied by the header encoding dtype (np.uint32)
        # for more details on the chunk's binary format, see `writer.py::_create_chunk`
        offset = (1 + chunk["chunk_size"] + 1) * 4
        mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
        self._mmaps[chunk_index] = mmap
        self._buffers[chunk_index] = memoryview(mmap)  # type: ignore
        self._counter[chunk_index] += 1
        self._evict_token_mmaps(protect=chunk_index)

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        # This is called within the prepare chunks thread, so we overlap data loading with data reading.
        if chunk_filepath not in self._chunk_filepaths:
            self._chunk_filepaths[chunk_filepath] = True

        if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0:
            self._load_chunk(chunk_index, chunk_filepath)

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> torch.Tensor:
        assert self._block_size

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            self._wait_until_chunk_ready(chunk_index, chunk_filepath, filesize_bytes)
            self._chunk_filepaths[chunk_filepath] = True

        self._load_chunk(chunk_index, chunk_filepath)
        assert self._dtype

        buffer: bytes = self._buffers[chunk_index]

        # offset: how many bytes to skip to get to the item we want to load
        #       -> if chunk begins at 5, and we want to load the item at index 7,
        #       -> we need to skip 2 items, and each item has `self._block_size` tokens
        #       -> and each token takes `self._dtype.itemsize` bytes
        #
        # Note: We have already accounted for offsets corresponding to starting bytes in `_load_chunk` function
        # while creating the memory map.
        offset = self._dtype.itemsize * (index - begin) * self._block_size

        # Copy out of the memmap. ``close()`` unmaps the previous chunk on the next sample, and
        # DataLoader may still be pickling the last items — a view into a closed mmap is SIGSEGV.
        if self._serializer_name == "no_header_tensor":
            data = torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset).clone()
        else:
            data = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset).copy()  # type: ignore

        return data

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        with trace_span("delete", CAT_DELETE, chunk=chunk_index):
            if os.path.exists(chunk_filepath):
                if chunk_index in self._buffers:
                    del self._buffers[chunk_index]
                if chunk_index in self._mmaps:
                    # explicitly close before deleting. Won't raise error if already closed.
                    self._mmaps[chunk_index]._mmap.close()
                    del self._mmaps[chunk_index]
                    del self._counter[chunk_index]
                os.remove(chunk_filepath)

    def close(self, chunk_index: int) -> None:
        """Release the memory-mapped file for a specific chunk index."""
        self._counter[chunk_index] -= 1

        if self._posix_fast:
            # Keep mappings in the LRU; unmapping here races DataLoader IPC (SIGSEGV in CI).
            if self._counter[chunk_index] <= 0:
                self._counter.pop(chunk_index, None)
            self._evict_token_mmaps()
            return

        if self._counter[chunk_index] <= 0:
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            if chunk_index in self._mmaps:
                self._mmaps[chunk_index]._mmap.close()
                del self._mmaps[chunk_index]
            self._counter.pop(chunk_index, None)

    def _close_open_chunk(self) -> None:
        for idx in list(self._mmaps):
            self._buffers.pop(idx, None)
            mm = self._mmaps.pop(idx)
            with contextlib.suppress(BufferError, ValueError, OSError):
                mm._mmap.close()
        self._counter.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._close_open_chunk()

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # ``np.memmap`` / ``memoryview`` are not picklable. POSIX-fast warms these in the parent
        # before DataLoader workers spawn; drop them and remap after unpickle.
        state["_mmaps"] = {}
        state["_buffers"] = {}
        return state

    @classmethod
    def encode_data(cls, data: list[bytes], _: list[int], flattened: list[Any]) -> tuple[bytes, int | None]:
        r"""Encodes tokenized data into a raw byte format while preserving dimensional information.

        Parameters:
        - data (List[bytes]): A list containing a single element, which is the raw byte
          representation of tokenized data.
        - _ (List[int]): A list containing sizes of each PyTree leaf in the item.
          Since only one item (tokens) is present, this argument is ignored.
        - flattened (List[Any]): A list containing a single element, which is the list of tokens.

        Example:
            - Original data: "hello world"
            - Tokenized data: [1, 2] (word tokenizer)
            - Data (raw bytes): [b'\x01\x00\x00\x00\x02\x00\x00\x00']
              (raw bytes representing the tokenized data)
            - Flattened data: [[1, 2]] (returned by PyTree's `flatten` function)

        Returns:
        - Tuple[bytes, Optional[int]]:
            - bytes: The raw byte representation of tokenized data.
            - dimension: The number of tokens in the data (extracted from `flattened[0].shape[0]`).
        """
        return data[0], flattened[0].shape[0]


class ParquetLoader(BaseItemLoader):
    def __init__(self, pre_load_chunk: bool = False, low_memory: bool = True) -> None:
        if not _POLARS_AVAILABLE:
            raise ModuleNotFoundError(
                "You are using the Parquet item loader, which depends on `Polars > 1.0.0`.",
                "Please, run: `pip install polars>1.0.0`",
            )
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow`")

        self._chunk_filepaths: dict[str, bool] = {}
        self._pre_load_chunk = pre_load_chunk
        self._low_memory = low_memory

        if not self._low_memory:
            logger.warning(
                "You have set low_memory=False in ParquetLoader. "
                "This may result in high memory usage when processing large Parquet chunk files. "
                "Consider setting low_memory=True to reduce memory consumption."
            )

    def setup(
        self,
        config: dict,
        chunks: list,
        serializers: dict[str, Serializer],
        region_of_interest: list[tuple[int, int]] | None = None,
    ) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = {**serializers}
        self._data_format = self._config["data_format"]
        self._shift_idx = len(self._data_format) * 4
        self.region_of_interest = region_of_interest
        # ParquetLoader does not call ``BaseItemLoader.setup``; keep wait/force-download attrs defined.
        self._force_download_queue = None
        self._chunk_ready_provider = getattr(self, "_chunk_ready_provider", None)
        self._prefetch_error_provider = getattr(self, "_prefetch_error_provider", None)
        self._df: dict[int, Any] = {}
        self._chunk_row_groups: dict[int, Any] = {}
        self._chunk_row_group_item_read_count: dict[int, Any] = {}
        self._chunk_row_group_offsets: dict[int, list[int]] = {}

    def generate_intervals(self) -> list[Interval]:
        intervals = []
        begin = 0
        end = 0
        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Preload the chunk in the background to gain some time."""
        if not self._pre_load_chunk or self._low_memory:
            return

        import polars as pl

        if chunk_index not in self._df and os.path.exists(chunk_filepath):
            self._df[chunk_index] = pl.scan_parquet(chunk_filepath, low_memory=True).collect()

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        """Returns an item loaded from a parquet chunk."""
        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            self._wait_until_chunk_ready(chunk_index, chunk_filepath, filesize_bytes)
            self._chunk_filepaths[chunk_filepath] = True

        # relative index of the desired row within the chunk.
        relative_index = index - begin
        if self._low_memory:
            item_data = self._get_item_with_low_memory(chunk_index, chunk_filepath, relative_index)
        else:
            item_data = self._get_item(chunk_index, chunk_filepath, relative_index)

        return item_data

    def _get_item_with_low_memory(self, chunk_index: int, chunk_filepath: str, row_index: int) -> Any:
        """Retrieve a dataframe row from a parquet chunk in low memory mode.

        This method reads only the necessary row group from the parquet file using PyArrow and Polars,
        which helps in reducing memory usage.

        Args:
            chunk_index (int): The index of the chunk to be accessed.
            chunk_filepath (str): The file path of the parquet chunk.
            row_index (int): The relative row index within the loaded chunk.

        Returns:
            Any: The dataframe row corresponding to the specified index.
        """
        import bisect

        import polars as pl
        import pyarrow.parquet as pq

        # Load the Parquet file metadata if not already loaded
        if chunk_index not in self._df:
            parquet_file = pq.ParquetFile(chunk_filepath)
            self._df[chunk_index] = parquet_file
            # Precompute cumulative row offsets as a prefix-sum so lookup works for row groups of any size.
            offsets = [0]
            num_row_groups = parquet_file.metadata.num_row_groups
            for i in range(num_row_groups):
                num_rows = parquet_file.metadata.row_group(i).num_rows
                offsets.append(offsets[-1] + num_rows)
            self._chunk_row_group_offsets[chunk_index] = offsets

        # Locate the row group containing row_index and the offset inside it.
        offsets = self._chunk_row_group_offsets[chunk_index]
        row_group_index = bisect.bisect_right(offsets, row_index) - 1
        row_index_within_group = row_index - offsets[row_group_index]
        row_group_size = offsets[row_group_index + 1] - offsets[row_group_index]

        # Check if the row group is already loaded
        if chunk_index in self._chunk_row_groups and row_group_index in self._chunk_row_groups[chunk_index]:
            # Use the cached row group
            row_group_df = self._chunk_row_groups[chunk_index][row_group_index]
            # update read count
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] += 1
        else:
            # Load the row group and convert it to a Polars DataFrame
            row_group = self._df[chunk_index].read_row_group(row_group_index)
            row_group_df = pl.from_arrow(row_group)

            # Cache the loaded row group
            if chunk_index not in self._chunk_row_groups:
                self._chunk_row_groups[chunk_index] = {}
                self._chunk_row_group_item_read_count[chunk_index] = {}

            self._chunk_row_groups[chunk_index][row_group_index] = row_group_df
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] = 1

        # Check if the row group has been fully read and release memory if necessary
        read_count = self._chunk_row_group_item_read_count[chunk_index][row_group_index]
        if read_count >= row_group_size:
            # Release memory for the fully read row group
            del self._chunk_row_groups[chunk_index][row_group_index]
            del self._chunk_row_group_item_read_count[chunk_index][row_group_index]

        # Return the specific row from the dataframe
        # Note: The `named=True` argument is used to return the row as a dictionary
        return row_group_df.row(row_index_within_group, named=True)  # type: ignore

    def _get_item(self, chunk_index: int, chunk_filepath: str, index: int) -> Any:
        """Retrieve a dataframe row from a parquet chunk by loading the entire chunk into memory.

        Note:
            This method reads the complete parquet file using Polars. Exercise caution with large files as it
            may significantly increase memory usage.

        Args:
            chunk_index (int): The index of the chunk to be accessed.
            chunk_filepath (str): The file path of the parquet chunk.
            index (int): The relative row index within the loaded chunk.

        Returns:
            Any: The dataframe row corresponding to the specified index.
        """
        import polars as pl

        if chunk_index not in self._df:
            self._df[chunk_index] = pl.scan_parquet(chunk_filepath, low_memory=True).collect()

        # Retrieve the specific row from the dataframe
        # Note: The `named=True` argument is used to return the row as a dictionary
        return self._df[chunk_index].row(index, named=True)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        with trace_span("delete", CAT_DELETE, chunk=chunk_index):
            if chunk_index in self._df:
                del self._df[chunk_index]
            if chunk_index in self._chunk_row_groups:
                del self._chunk_row_groups[chunk_index]

            if chunk_index in self._chunk_row_group_item_read_count:
                del self._chunk_row_group_item_read_count[chunk_index]
            if chunk_index in self._chunk_row_group_offsets:
                del self._chunk_row_group_offsets[chunk_index]
            if os.path.exists(chunk_filepath):
                os.remove(chunk_filepath)

    def close(self, chunk_index: int) -> None:
        """Release the memory-mapped file for a specific chunk index."""
        if chunk_index in self._df:
            del self._df[chunk_index]

        if chunk_index in self._chunk_row_groups:
            del self._chunk_row_groups[chunk_index]

        if chunk_index in self._chunk_row_group_item_read_count:
            del self._chunk_row_group_item_read_count[chunk_index]

        if chunk_index in self._chunk_row_group_offsets:
            del self._chunk_row_group_offsets[chunk_index]

    def encode_data(self, data: list[bytes], sizes: list[int], flattened: list[Any]) -> Any:
        pass

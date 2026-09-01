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
from litdata.streaming.framed_zstd import (
    FramedHeader,
    frame_index_for_item,
    inflate_frame,
    is_framed_chunk,
    make_zstd_codec,
    parse_compression_level,
    parse_framed_header,
)
from litdata.streaming.posix_fast import advise_willneed, madvise_mmap, posix_page_bytes
from litdata.streaming.serializers import JsonLeaf, Serializer
from litdata.utilities._pytree import SUPPORTED_NODES, PyTree, TreeSpec, tree_unflatten
from litdata.utilities.encryption import Encryption, EncryptionLevel

Interval = namedtuple("Interval", ["chunk_start", "roi_start_idx", "roi_end_idx", "chunk_end"])

logger = logging.getLogger("litdata.streaming.item_loader")

_BATCH_SKIP = object()
# Cap for cheap leaves (text / nested JSON). Images/video scale down from avg bytes.
_DEFAULT_BATCH_ROWS = 256
_AUTO_WINDOW_BYTES = 16 << 20  # ~16MB of on-disk samples per decode window
# C++ / per-row deserialize is not amortizable. A window smaller than
# ``item_shuffle_window`` (default 256) re-decodes the same JPEG ~16×.
_PER_ITEM_LEAF = frozenset(
    {
        "jpeg",
        "pil",
        "image",
        "video",
        "audio",
        "file",
        "mesh",
        "pdf",
        "nifti",
        "tiff",
        "jpeg_array",
    }
)
_HEAVY_LEAF = frozenset(
    {
        *_PER_ITEM_LEAF,
        "tensor",
        "no_header_tensor",
        "graph",
    }
)
# Trailing Arrow IPC of the original rows. Nested Python decode is ~5–15× slower than
# Arrow ``to_pylist``; old readers ignore bytes past the last item offset.
_ARROW_FOOTER_MAGIC = b"LDARW01\0"
# Arrow IPC *file* magic (uncompressed directory + per-batch bodies). Stream
# footers from older writers start with a continuation / schema message instead.
_ARROW_IPC_FILE_MAGIC = b"ARROW1"


def _parse_batch_decode(value: Any) -> int | None:
    """``None`` means ``auto``. ``0`` = per item, ``-1`` = whole chunk, ``N`` = window."""
    if value is None:
        return None
    if isinstance(value, bool):
        return -1 if value else 0
    if isinstance(value, int):
        return -1 if value < 0 else value
    key = str(value).strip().lower()
    if key in {"auto", ""}:
        return None
    if key in {"0", "false", "no", "off", "item"}:
        return 0
    if key in {"all", "chunk", "true", "yes", "on"}:
        return -1
    try:
        parsed = int(key)
    except ValueError:
        return None
    return -1 if parsed < 0 else parsed


def _avg_sample_bytes(chunks: list | None) -> int:
    if not chunks:
        return 0
    total_b = 0
    total_n = 0
    for chunk in chunks[:16]:
        nbytes = int(chunk.get("chunk_bytes") or 0)
        n_items = int(chunk.get("chunk_size") or 0)
        if nbytes > 0 and n_items > 0:
            total_b += nbytes
            total_n += n_items
    return total_b // total_n if total_n else 0


def _leaf_key(name: str) -> str:
    return name.split(":", 1)[0].lower()


def _item_only_batch(batch_rows: int | None) -> bool:
    """``0`` / ``1`` decode the requested row; ``-1`` / ``N>1`` keep a window."""
    return batch_rows is not None and 0 <= batch_rows <= 1


def _auto_batch_rows(data_format: list[str] | None, chunks: list | None = None) -> int:
    """Pick a window from leaf types and mean on-disk sample size.

    Text/nested stay at 256 (measured winner). JPEG / image / audio always use
    1 (decode only the requested row) so shuffle cannot re-decode a dropped
    window. Large tensors still scale toward 1 from mean on-disk bytes.
    """
    avg = _avg_sample_bytes(chunks)
    keys = [_leaf_key(name) for name in (data_format or [])]
    if any(key in _PER_ITEM_LEAF for key in keys):
        return 1
    heavy = any(key in _HEAVY_LEAF for key in keys)
    if avg >= 1 << 20:
        return 1
    if heavy:
        if avg >= 256 << 10:
            return 8
        if avg >= 64 << 10:
            return 16
        return 32
    if avg >= 64 << 10:
        return max(1, min(_DEFAULT_BATCH_ROWS, _AUTO_WINDOW_BYTES // avg))
    return _DEFAULT_BATCH_ROWS


def _batch_rows_for_format(
    data_format: list[str] | None,
    chunks: list | None = None,
    batch_decode: Any = "auto",
) -> int:
    """How many items to deserialize together. ``0`` = per item, ``-1`` = whole chunk.

    Default is ``auto`` (format + sample size). ``StreamingDataset(batch_decode=...)``
    wins; ``LITDATA_BATCH_DECODE`` / ``LITDATA_BATCH_ROWS`` apply only when the
    dataset is still ``auto``.
    """
    parsed = _parse_batch_decode(batch_decode)
    if parsed is None:
        raw = os.getenv("LITDATA_BATCH_DECODE")
        if raw is not None and raw.strip():
            env_parsed = _parse_batch_decode(raw)
            if env_parsed is not None:
                return env_parsed
        raw_n = os.getenv("LITDATA_BATCH_ROWS")
        if raw_n is not None and raw_n.strip():
            with contextlib.suppress(ValueError):
                return max(0, int(raw_n))
        return _auto_batch_rows(data_format, chunks)
    return parsed


def _unwrap_arrow_sample(value: Any) -> Any:
    """Drop :class:`JsonLeaf` wrappers so Arrow sees plain Python values."""
    if isinstance(value, JsonLeaf):
        return _unwrap_arrow_sample(value.value)
    if isinstance(value, dict):
        return {key: _unwrap_arrow_sample(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap_arrow_sample(item) for item in value]
    return value


def _arrow_ipc_write_options(compression: str | None = "zstd") -> Any:
    """Arrow IPC compression (C++ inflate on ``get_batch`` / ``open_file``), or ``None``."""
    if not compression or not _PYARROW_AVAILABLE:
        return None
    import pyarrow as pa

    try:
        return pa.ipc.IpcWriteOptions(compression=compression)
    except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return None


def append_arrow_row_footer(data: bytes, samples: list[Any], ipc_compression: str | None = None) -> bytes:
    """Append a trailing Arrow IPC *file* when every sample is a dict (nested HF rows).

    Record batches are ``_DEFAULT_BATCH_ROWS`` (256) so the reader can
    ``open_file`` + ``get_batch(i)`` without inflating the whole chunk.
    ``ipc_compression`` (e.g. ``"zstd"``) is Arrow ``IpcWriteOptions`` — C++ inflate
    per batch. It is off unless the writer asked for compression. LitData whole-file
    ``compression="zstd"`` is a different wrap and is skipped for nested chunks.
    """
    if not samples or not _PYARROW_AVAILABLE:
        return data
    if any(sample is None or not isinstance(sample, dict) for sample in samples):
        return data
    import pyarrow as pa

    try:
        rows = [_unwrap_arrow_sample(sample) for sample in samples]
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                return data
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        unified = [{key: row.get(key) for key in keys} for row in rows]
        table = pa.Table.from_pylist(unified)
    except (TypeError, pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        return data
    sink = pa.BufferOutputStream()
    options = _arrow_ipc_write_options(ipc_compression)
    write_kw: dict[str, Any] = {"options": options} if options is not None else {}
    n_rows = table.num_rows
    batch_rows = _DEFAULT_BATCH_ROWS
    with pa.ipc.new_file(sink, table.schema, **write_kw) as writer:
        for start in range(0, n_rows, batch_rows):
            writer.write_table(table.slice(start, min(batch_rows, n_rows - start)))
    ipc = sink.getvalue().to_pybytes()
    return data + ipc + struct.pack("<I", len(ipc)) + _ARROW_FOOTER_MAGIC


def _arrow_footer_span(view: bytes | bytearray | memoryview) -> tuple[int, int] | None:
    """Return ``(start, ipc_len)`` for a trailing Arrow IPC blob, or ``None``."""
    raw = view if isinstance(view, (bytes, bytearray, memoryview)) else memoryview(view)
    if len(raw) < 12:
        return None
    if bytes(raw[-8:]) != _ARROW_FOOTER_MAGIC:
        return None
    ipc_len = struct.unpack_from("<I", raw, len(raw) - 12)[0]
    start = len(raw) - 12 - ipc_len
    if start < 0 or ipc_len <= 0:
        return None
    return start, ipc_len


def _ipc_is_file(ipc: bytes | bytearray | memoryview) -> bool:
    """True when the blob is Arrow IPC file format (``ARROW1``), not a stream."""
    if len(ipc) < 6:
        return False
    return bytes(ipc[:6]) == _ARROW_IPC_FILE_MAGIC


def open_arrow_footer_reader(view: bytes | bytearray | memoryview) -> tuple[Any, bool] | None:
    """Return ``(reader, is_file)`` for a trailing Arrow IPC blob, or ``None``.

    New chunks are IPC *files* (``open_file``). Older stream footers keep
    ``open_stream``.
    """
    if not _PYARROW_AVAILABLE:
        return None
    span = _arrow_footer_span(view)
    if span is None:
        return None
    start, ipc_len = span
    import pyarrow as pa

    ipc = memoryview(view)[start : start + ipc_len]
    try:
        is_file = _ipc_is_file(ipc)
        source = pa.py_buffer(ipc)
        reader = pa.ipc.open_file(source) if is_file else pa.ipc.open_stream(source)
        return reader, is_file
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError, TypeError):
        return None


def load_arrow_footer_table(view: bytes | bytearray | memoryview) -> Any | None:
    """Return the trailing Arrow ``Table``, or ``None`` if the chunk has no footer."""
    opened = open_arrow_footer_reader(view)
    if opened is None:
        return None
    reader, _is_file = opened
    import pyarrow as pa

    try:
        return reader.read_all()
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, ValueError):
        return None


def load_arrow_row_footer(view: bytes | bytearray | memoryview) -> list[Any] | None:
    """Return rows from a trailing Arrow IPC table, or ``None`` if the chunk has no footer."""
    table = load_arrow_footer_table(view)
    if table is None:
        return None
    return table.to_pylist()


def _as_chunk_index(chunk_index: int) -> int:
    """Normalize sampler / numpy indexes so mmap dict keys stay hash-stable."""
    return int(chunk_index)


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
        raw_level = self._config.get("compression_level") if isinstance(self._config, dict) else None
        self._compression_level = parse_compression_level(raw_level)
        self._sample_compression = self._compression_level == "sample"
        # Compile a specialized unflatten for this dataset's fixed treespec. Falls back to the
        # stock pytree path only when there is no data_spec (e.g. some parquet/MDS shapes).
        self._unflatten = (
            _compile_treespec_unflatten(self._data_spec) if isinstance(self._data_spec, TreeSpec) else None
        )
        # Fixed size-header layout: one little-endian uint32 per leaf.
        # Keep a format string (pickle-friendly) rather than a ``struct.Struct`` instance.
        self._sizes_fmt = "<" + "I" * len(self._data_format) if self._data_format else None
        self._sizes_struct = struct.Struct(self._sizes_fmt) if self._sizes_fmt else None
        self._batch_rows = _batch_rows_for_format(
            list(self._data_format) if self._data_format else None,
            chunks,
            getattr(self, "_batch_decode", "auto"),
        )

    def set_batch_decode(self, batch_decode: Any) -> None:
        """``auto`` / ``0`` / ``N`` / ``all``. Default loaders ignore this."""
        del batch_decode

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
                remaining = max_wait - (time() - start_time)
                if remaining <= 0:
                    raise ChunkWaitTimeoutError(chunk_filepath, time() - start_time)
                event = chunk_ready_provider(chunk_index)
                # Wait for prefetch to publish. Force-download is a last resort after
                # ``_FORCE_DOWNLOAD_TIME`` (default 30s) — not on the first wait.
                signaled = event.wait(timeout=min(1.0, remaining))
                if signaled and not (
                    os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= filesize_bytes
                ):
                    # Stale signal after delete (clear_chunk_ready may have been skipped).
                    # Chunk-ready is set only after decompress/finalize, so this is safe.
                    event.clear()
                    sleep(0.05)
            else:
                sleep(0.1)

            # Retry force-download after the grace period if the first request was deferred.
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
        # ``struct.Struct`` is not picklable; rebuild from ``_sizes_fmt`` after unpickle.
        state["_sizes_struct"] = None
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

    def __init__(self, batch_decode: Any = "auto") -> None:
        super().__init__()
        self._batch_decode: Any = batch_decode
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
        # Optional decode window (parquet row-group style). Sized from data_format.
        self._mmap_chunk_index: int | None = None
        self._chunk_rows: list[Any] | None = None
        self._chunk_rows_index: int | None = None
        self._batch_rows: int | None = None
        self._win_start = 0
        self._arrow_table: Any | None = None
        self._arrow_table_index: int | None = None
        self._arrow_reader: Any | None = None
        self._arrow_reader_index: int | None = None
        self._arrow_reader_is_file = False
        self._framed_meta: dict[int, FramedHeader] = {}
        self._framed_decompressor: Any | None = None
        self._framed_inflate_buf: bytes | memoryview | None = None
        self._framed_inflate_key: tuple[int, int] | None = None
        self._compression_level = "chunk"
        self._sample_compression = False

    def set_batch_decode(self, batch_decode: Any) -> None:
        self._batch_decode = batch_decode
        data_format = getattr(self, "_data_format", None)
        chunks = getattr(self, "_chunks", None)
        if data_format is not None:
            self._batch_rows = _batch_rows_for_format(list(data_format), chunks, batch_decode)

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
        # Called from PrepareChunksThread. Only advise the page cache — do not
        # mutate mmap state here (the reader thread owns ``_mapped``).
        del chunk_index
        if os.path.isfile(chunk_filepath) and (self._posix_willneed or not self._posix_fast):
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

        table_idx = offset // 4 - 1
        if self._config.get("encryption"):
            data = self._load_encrypted_data(chunk_filepath, chunk_index, offset, encryption)
        else:
            batched = self._load_batched_item(chunk_index, chunk_filepath, table_idx)
            if batched is not _BATCH_SKIP:
                return batched
            if self._mmap is not None:
                data = self._slice_item_bytes(table_idx, chunk_index)
            else:
                assert self._open_handle
                data = self._load_data(self._open_handle, offset)
            data = self._maybe_inflate_sample(data)

        if "format" in self._config and self._config["format"] == "mds":
            return self.mds_deserialize(data, chunk_index)
        return self.deserialize(data)

    def _framed_compressor(self) -> Any:
        """Reused Arrow C++ zstd codec (python-zstd if pyarrow is missing)."""
        if self._framed_decompressor is None:
            name = (self._config or {}).get("compression") or "zstd"
            self._framed_decompressor = make_zstd_codec(name)
        return self._framed_decompressor

    def _maybe_inflate_sample(self, raw: bytes | bytearray | memoryview) -> bytes | bytearray | memoryview:
        if not getattr(self, "_sample_compression", False):
            return raw
        payload = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        return self._framed_compressor().decompress(payload)

    def _missing_framed_magic_error(self, chunk_filepath: str) -> RuntimeError:
        return RuntimeError(
            f"Chunk {chunk_filepath} is indexed as compression_level='batch' but missing LDFZ01 magic. "
            "Refusing to unpack item bytes."
        )

    def _resolve_framed_header(self, chunk_index: int, view: bytes | bytearray | memoryview) -> FramedHeader | None:
        cached = self._framed_meta.get(chunk_index)
        if cached is not None:
            return cached
        if is_framed_chunk(view):
            header = parse_framed_header(view)
            self._framed_meta[chunk_index] = header
            return header
        if parse_compression_level((self._config or {}).get("compression_level")) == "batch":
            raise self._missing_framed_magic_error(self._chunk_filepath or "")
        return None

    def _fill_framed_window(
        self,
        view: bytes | bytearray | memoryview,
        chunk_index: int,
        table_idx: int,
        header: FramedHeader,
    ) -> Any:
        frame_i = frame_index_for_item(header, table_idx)
        frame = header.frames[frame_i]
        first = frame.first_item
        n_items = frame.n_items
        rows = self._chunk_rows
        if (
            rows is not None
            and self._chunk_rows_index == chunk_index
            and self._win_start == first
            and len(rows) == n_items
        ):
            return rows[table_idx - first]
        inflate_key = (chunk_index, frame_i)
        if self._framed_inflate_key == inflate_key and self._framed_inflate_buf is not None:
            raw = self._framed_inflate_buf
        else:
            raw = inflate_frame(view, header, frame_i, self._framed_compressor())
            self._framed_inflate_buf = raw
            self._framed_inflate_key = inflate_key
        base = header.offsets[first]
        local_offsets = [int(off) - base for off in header.offsets[first : first + n_items + 1]]
        if _item_only_batch(self._batch_rows):
            local_i = table_idx - first
            decoded = self._batch_deserialize_payload(raw, local_offsets, chunk_index, local_i, local_i + 1)
            return decoded[0]
        decoded = self._batch_deserialize_payload(raw, local_offsets, chunk_index, 0, n_items)
        return self._store_decode_window(chunk_index, first, decoded, table_idx)

    def _load_batched_item(self, chunk_index: int, chunk_filepath: str, table_idx: int) -> Any:
        """Return a cached/windowed row, or ``_BATCH_SKIP`` to decode this item alone."""
        batch_rows = self._batch_rows
        if batch_rows is None:
            batch_rows = _batch_rows_for_format(
                list(self._data_format) if self._data_format else None,
                getattr(self, "_chunks", None),
                getattr(self, "_batch_decode", "auto"),
            )
            self._batch_rows = batch_rows
        rows = self._chunk_rows
        if (
            rows is not None
            and self._chunk_rows_index == chunk_index
            and self._win_start <= table_idx < self._win_start + len(rows)
        ):
            return rows[table_idx - self._win_start]
        if self._mmap is not None:
            view = self._mmap_view
            if view is None:
                assert self._mmap is not None
                view = memoryview(self._mmap)
                self._mmap_view = view
            header = self._resolve_framed_header(chunk_index, view)
            if header is not None:
                return self._fill_framed_window(view, chunk_index, table_idx, header)
            arrow = self._try_arrow_footer_rows(view, chunk_index, table_idx)
            if arrow is not _BATCH_SKIP:
                return arrow
            if _item_only_batch(batch_rows):
                return _BATCH_SKIP
            return self._fill_decode_window_mmap(chunk_index, table_idx, batch_rows)
        if _item_only_batch(batch_rows):
            with open(chunk_filepath, "rb") as handle:
                blob = handle.read()
            header = self._resolve_framed_header(chunk_index, blob)
            if header is not None:
                return self._fill_framed_window(blob, chunk_index, table_idx, header)
            return _BATCH_SKIP
        return self._fill_decode_window_path(chunk_index, chunk_filepath, table_idx, batch_rows)

    def _clear_arrow_footer(self) -> None:
        self._arrow_table = None
        self._arrow_table_index = None
        self._arrow_reader = None
        self._arrow_reader_index = None
        self._arrow_reader_is_file = False

    def _try_arrow_footer_rows(self, view: bytes | memoryview, chunk_index: int, table_idx: int) -> Any:
        """Serve one IPC record batch (file) or a sliced stream table (legacy)."""
        reader = self._arrow_reader
        is_file = self._arrow_reader_is_file
        if reader is None or self._arrow_reader_index != chunk_index:
            opened = open_arrow_footer_reader(view)
            if opened is None:
                return _BATCH_SKIP
            reader, is_file = opened
            self._arrow_reader = reader
            self._arrow_reader_is_file = is_file
            self._arrow_reader_index = chunk_index
            self._arrow_table = None
            self._arrow_table_index = None
        if is_file:
            n_batches = int(reader.num_record_batches)
            if n_batches <= 0:
                return _BATCH_SKIP
            batch_i = table_idx // _DEFAULT_BATCH_ROWS
            if batch_i < 0 or batch_i >= n_batches:
                return _BATCH_SKIP
            rows = reader.get_batch(batch_i).to_pylist()
            start = batch_i * _DEFAULT_BATCH_ROWS
            if table_idx < start or table_idx >= start + len(rows):
                return _BATCH_SKIP
            return self._store_decode_window(chunk_index, start, rows, table_idx)
        if self._arrow_table is None or self._arrow_table_index != chunk_index:
            self._arrow_table = reader.read_all()
            self._arrow_table_index = chunk_index
        table = self._arrow_table
        n = table.num_rows
        if table_idx < 0 or table_idx >= n:
            return _BATCH_SKIP
        batch_rows = self._batch_rows
        if not batch_rows:
            start, end = 0, n
        else:
            start, end = self._window_bounds(table_idx, n, batch_rows)
        rows = table.slice(start, end - start).to_pylist()
        return self._store_decode_window(chunk_index, start, rows, table_idx)

    def _batch_deserialize_payload(
        self,
        view: bytes | memoryview,
        offsets: list[int],
        chunk_index: int,
        start: int,
        end: int,
    ) -> list[Any]:
        """Decode ``offsets[start:end]`` from a local chunk buffer."""
        n = end - start
        rows: list[Any] = [None] * n
        inflate = self._maybe_inflate_sample
        if self._config.get("format") == "mds":
            for i in range(n):
                idx = start + i
                rows[i] = self.mds_deserialize(inflate(view[offsets[idx] : offsets[idx + 1]]), chunk_index)
            return rows
        shift = self._shift_idx
        sizes_struct = self._sizes_struct
        serializers = self._serializers_list
        unflatten = self._unflatten
        n_leaves = len(serializers)
        if sizes_struct is not None and unflatten is not None:
            for i in range(n):
                idx = start + i
                raw = inflate(view[offsets[idx] : offsets[idx + 1]])
                sizes = sizes_struct.unpack_from(raw, 0)
                leaves: list[Any] = [None] * n_leaves
                cursor = shift
                for j, (size, serializer) in enumerate(zip(sizes, serializers)):
                    leaves[j] = serializer.deserialize(raw[cursor : cursor + size])
                    cursor += size
                rows[i] = unflatten(leaves)
            return rows
        deserialize = self.deserialize
        for i in range(n):
            idx = start + i
            rows[i] = deserialize(inflate(view[offsets[idx] : offsets[idx + 1]]))
        return rows

    def _store_decode_window(self, chunk_index: int, start: int, rows: list[Any], table_idx: int | None = None) -> Any:
        self._chunk_rows = rows
        self._chunk_rows_index = chunk_index
        self._win_start = start
        idx = 0 if table_idx is None else table_idx - start
        return rows[idx]

    def _window_bounds(self, table_idx: int, n_items: int, batch_rows: int) -> tuple[int, int]:
        """Aligned window so random hits in the same block share one decode.

        Sequential ``0..n`` already reused a forward window. A full in-chunk
        shuffle does not: starting at the requested index re-decoded overlapping
        ranges. Fixed blocks ``[kW, (k+1)W)`` decode each item once per visit.
        """
        if batch_rows < 0:
            return 0, n_items
        if batch_rows <= 1:
            end = min(n_items, table_idx + max(batch_rows, 1))
            return table_idx, end
        start = (table_idx // batch_rows) * batch_rows
        return start, min(n_items, start + batch_rows)

    def _fill_decode_window_mmap(self, chunk_index: int, table_idx: int, batch_rows: int) -> Any:
        offsets = self._offsets
        view = self._mmap_view
        assert offsets is not None
        if view is None:
            assert self._mmap is not None
            view = memoryview(self._mmap)
            self._mmap_view = view
        start, end = self._window_bounds(table_idx, len(offsets) - 1, batch_rows)
        rows = self._batch_deserialize_payload(view, offsets, chunk_index, start, end)
        return self._store_decode_window(chunk_index, start, rows, table_idx)

    def _fill_decode_window_path(self, chunk_index: int, chunk_filepath: str, table_idx: int, batch_rows: int) -> Any:
        with open(chunk_filepath, "rb") as handle:
            blob = handle.read()
        header = self._resolve_framed_header(chunk_index, blob)
        if header is not None:
            return self._fill_framed_window(blob, chunk_index, table_idx, header)
        n = struct.unpack_from("<I", blob, 0)[0]
        offsets = list(struct.unpack_from("<" + "I" * (n + 1), blob, 4))
        start, end = self._window_bounds(table_idx, n, batch_rows)
        arrow = self._try_arrow_footer_rows(blob, chunk_index, table_idx)
        if arrow is not _BATCH_SKIP:
            return arrow
        rows = self._batch_deserialize_payload(blob, offsets, chunk_index, start, end)
        return self._store_decode_window(chunk_index, start, rows, table_idx)

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
        begin, end = struct.unpack("<II", pair)

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
        sizes_struct = getattr(self, "_sizes_struct", None)
        if sizes_struct is not None:
            sizes = sizes_struct.unpack_from(raw_item_data, 0)
        elif self._sizes_fmt is not None:
            sizes = struct.unpack_from(self._sizes_fmt, raw_item_data, 0)
        else:
            sizes = ()
        data = [None] * len(sizes)
        for i, (size, serializer) in enumerate(zip(sizes, self._serializers_list)):
            data[i] = serializer.deserialize(raw_item_data[idx : idx + size])
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
        index_num_items = int(self._chunks[chunk_index]["chunk_size"])
        if is_framed_chunk(chunk_mmap):
            header = parse_framed_header(chunk_mmap)
            if header.num_items != index_num_items:
                chunk_mmap.close()
                handle = self._mmap_handles.pop(chunk_index, None)
                if handle is not None:
                    handle.close()
                raise RuntimeError(
                    f"Chunk {chunk_index} framed item count ({header.num_items}) does not match "
                    f"index.json chunk_size ({index_num_items}) for {chunk_filepath}."
                )
            self._framed_meta[chunk_index] = header
            offsets = header.offsets
        else:
            if parse_compression_level((self._config or {}).get("compression_level")) == "batch":
                chunk_mmap.close()
                handle = self._mmap_handles.pop(chunk_index, None)
                if handle is not None:
                    handle.close()
                raise self._missing_framed_magic_error(chunk_filepath)
            header_num_items = int(np.frombuffer(chunk_mmap, dtype=np.uint32, count=1, offset=0)[0])
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
        if self._mmap_chunk_index != chunk_index:
            self._chunk_rows = None
            self._chunk_rows_index = None
            self._win_start = 0
            self._clear_arrow_footer()
        chunk_mmap, offsets, chunk_filepath = cached
        self._mmap = chunk_mmap
        self._open_handle = None
        self._offsets = offsets
        self._chunk_filepath = chunk_filepath
        self._mmap_chunk_index = chunk_index
        self._mmap_view = memoryview(chunk_mmap)
        self._mapped.move_to_end(chunk_index)

    def _close_mapping(self, chunk_index: int, chunk_mmap: mmap.mmap) -> None:
        """Drop views into ``chunk_mmap`` then close it so the fd is released."""
        if self._mmap is chunk_mmap:
            self._mmap = None
            self._mmap_view = None
            self._offsets = None
            self._open_handle = None
            self._mmap_chunk_index = None
            if self._chunk_rows_index == chunk_index:
                self._chunk_rows = None
                self._chunk_rows_index = None
                self._win_start = 0
            if self._arrow_table_index == chunk_index or self._arrow_reader_index == chunk_index:
                self._clear_arrow_footer()
        if self._page_chunk == chunk_index:
            self._clear_item_page()
        handle = self._mmap_handles.pop(chunk_index, None)
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        self._framed_meta.pop(chunk_index, None)
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
        self._mmap_chunk_index = None
        self._chunk_rows = None
        self._chunk_rows_index = None
        self._win_start = 0
        self._clear_arrow_footer()
        self._framed_meta.clear()
        for idx in list(self._mapped):
            cached = self._mapped.pop(idx, None)
            if cached is not None:
                self._close_mapping(idx, cached[0])

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
        n = len(sizes)
        body_len = 0
        for size in sizes:
            body_len += size
        header_len = 4 * n
        out = bytearray(header_len + body_len)
        if n:
            out[0:header_len] = struct.pack("<" + "I" * n, *sizes)
        cursor = header_len
        for chunk, size in zip(data, sizes):
            out[cursor : cursor + size] = chunk
            cursor += size
        return bytes(out), None

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
        state["_mmap_chunk_index"] = None
        state["_chunk_rows"] = None
        state["_chunk_rows_index"] = None
        state["_win_start"] = 0
        state["_batch_rows"] = None
        state["_arrow_table"] = None
        state["_arrow_table_index"] = None
        state["_arrow_reader"] = None
        state["_arrow_reader_index"] = None
        state["_arrow_reader_is_file"] = False
        state["_framed_meta"] = {}
        state["_framed_decompressor"] = None
        state["_framed_inflate_buf"] = None
        state["_framed_inflate_key"] = None
        # Compiled unflatten closures aren't picklable; rebuild after unpickle.
        state["_unflatten"] = None
        state["_sizes_struct"] = None
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
        if not hasattr(self, "_chunk_rows"):
            self._chunk_rows = None
            self._chunk_rows_index = None
            self._mmap_chunk_index = None
            self._win_start = 0
            self._batch_rows = None
        if not hasattr(self, "_arrow_table"):
            self._arrow_table = None
            self._arrow_table_index = None
        if not hasattr(self, "_arrow_reader"):
            self._arrow_reader = None
            self._arrow_reader_index = None
            self._arrow_reader_is_file = False
        if not hasattr(self, "_framed_meta"):
            self._framed_meta = {}
            self._framed_decompressor = None
            self._framed_inflate_buf = None
        if not hasattr(self, "_framed_inflate_key"):
            self._framed_inflate_key = None
        data_spec = getattr(self, "_data_spec", None)
        if isinstance(data_spec, TreeSpec):
            self._unflatten = _compile_treespec_unflatten(data_spec)
        sizes_fmt = getattr(self, "_sizes_fmt", None)
        self._sizes_struct = struct.Struct(sizes_fmt) if sizes_fmt else None


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
        chunk_index = _as_chunk_index(chunk_index)
        if chunk_index in self._mmaps and chunk_index in self._buffers:
            return
        if chunk_index in self._mmaps and chunk_index not in self._buffers:
            mm = self._mmaps.pop(chunk_index)
            with contextlib.suppress(BufferError, ValueError, OSError):
                mm._mmap.close()
            self._counter.pop(chunk_index, None)
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
            self._load_chunk(_as_chunk_index(chunk_index), chunk_filepath)

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> torch.Tensor:
        assert self._block_size
        chunk_index = _as_chunk_index(chunk_index)

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            self._wait_until_chunk_ready(chunk_index, chunk_filepath, filesize_bytes)
            self._chunk_filepaths[chunk_filepath] = True

        self._load_chunk(chunk_index, chunk_filepath)
        assert self._dtype

        buffer = self._buffers.get(chunk_index)
        if buffer is None:
            self._mmaps.pop(chunk_index, None)
            self._load_chunk(chunk_index, chunk_filepath)
            buffer = self._buffers[chunk_index]

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
        chunk_index = _as_chunk_index(chunk_index)
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
        chunk_index = _as_chunk_index(chunk_index)
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
        leaf = flattened[0]
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) > 0:
            return data[0], int(shape[0])
        return data[0], len(leaf)


class ParquetLoader(BaseItemLoader):
    def __init__(
        self,
        pre_load_chunk: bool = False,
        low_memory: bool = True,
        columns: list[str] | None = None,
    ) -> None:
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
        self._columns = columns

        if not self._low_memory:
            logger.warning(
                "You have set low_memory=False in ParquetLoader. "
                "This may result in high memory usage when processing large Parquet chunk files. "
                "Consider setting low_memory=True to reduce memory consumption."
            )

        self._remote_dir: str | None = None
        self._storage_options: dict | None = {}

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

    def set_remote_source(self, remote_dir: str | None, storage_options: dict | None = None) -> None:
        self._remote_dir = remote_dir
        self._storage_options = storage_options or {}

    def _open_parquet_file(self, chunk_index: int, chunk_filepath: str) -> Any:
        del chunk_index
        import pyarrow.parquet as pq

        return pq.ParquetFile(chunk_filepath)

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
            scan = pl.scan_parquet(chunk_filepath, low_memory=True)
            if self._columns:
                scan = scan.select(self._columns)
            self._df[chunk_index] = scan.collect()

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

        This method reads only the necessary row group from the parquet file using PyArrow
        and materializes it once with ``to_pylist()`` (Hugging Face / Arrow batch conversion).

        Args:
            chunk_index (int): The index of the chunk to be accessed.
            chunk_filepath (str): The file path of the parquet chunk.
            row_index (int): The relative row index within the loaded chunk.

        Returns:
            Any: The dataframe row corresponding to the specified index.
        """
        import bisect

        # Load the Parquet file metadata if not already loaded
        if chunk_index not in self._df:
            parquet_file = self._open_parquet_file(chunk_index, chunk_filepath)
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

        # Cache the row group as a Python list of dicts. Hugging Face datasets / PyArrow
        # convert a whole RecordBatch at once (``to_pylist``); per-cell ``as_py()`` is ~2× slower.
        if chunk_index in self._chunk_row_groups and row_group_index in self._chunk_row_groups[chunk_index]:
            rows = self._chunk_row_groups[chunk_index][row_group_index]
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] += 1
        else:
            table = self._df[chunk_index].read_row_group(row_group_index, columns=self._columns)
            rows = table.to_pylist()
            if chunk_index not in self._chunk_row_groups:
                self._chunk_row_groups[chunk_index] = {}
                self._chunk_row_group_item_read_count[chunk_index] = {}

            self._chunk_row_groups[chunk_index][row_group_index] = rows
            self._chunk_row_group_item_read_count[chunk_index][row_group_index] = 1

        read_count = self._chunk_row_group_item_read_count[chunk_index][row_group_index]
        if read_count >= row_group_size:
            del self._chunk_row_groups[chunk_index][row_group_index]
            del self._chunk_row_group_item_read_count[chunk_index][row_group_index]

        return rows[row_index_within_group]

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
            scan = pl.scan_parquet(chunk_filepath, low_memory=True)
            if self._columns:
                scan = scan.select(self._columns)
            self._df[chunk_index] = scan.collect()

        # Retrieve the specific row from the dataframe
        # Note: The `named=True` argument is used to return the row as a dictionary
        return self._df[chunk_index].row(index, named=True)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        with trace_span("delete", CAT_DELETE, chunk=chunk_index):
            self.close(chunk_index)
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

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_df"] = {}
        state["_chunk_row_groups"] = {}
        state["_chunk_row_group_item_read_count"] = {}
        state["_chunk_row_group_offsets"] = {}
        return state

    def encode_data(self, data: list[bytes], sizes: list[int], flattened: list[Any]) -> Any:
        pass

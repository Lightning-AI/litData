"""Bounded, leased mappings for concurrent windows on immutable POSIX chunks."""

import mmap
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass


def _advise_window_ranges(mapping: mmap.mmap, ranges: list[tuple[int, int]]) -> None:
    """Prefetch selected page ranges, without pulling in unrelated track frames."""
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_WILLNEED"):
        return
    for offset, length in ranges:
        begin = offset // mmap.PAGESIZE * mmap.PAGESIZE
        end = min(len(mapping), (offset + length + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
        with suppress(OSError, ValueError):
            mapping.madvise(mmap.MADV_WILLNEED, begin, end - begin)


@dataclass
class _Mapping:
    buffer: mmap.mmap
    users: int = 0


class _WindowMMapCache:
    """Keep a small mapping LRU, protecting windows that are currently decoding.

    Mapping a chunk does not read or prefetch its entire payload. Unlike sequential
    record iteration, caller-directed windows must not advise whole chunks WILLNEED.
    Only requested views are touched. The OS controls page residency; this cache
    bounds mapping/file-descriptor resources, not resident dataset bytes.
    """

    def __init__(self, keep: int = 4) -> None:
        self.keep = max(1, keep)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._maps: OrderedDict[str, _Mapping] = OrderedDict()
        self._closed = False

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            # Never acquire an inherited lock: another parent thread may own it.
            self._lock = threading.Lock()
            for entry in self._maps.values():
                with suppress(BufferError, OSError):
                    entry.buffer.close()
            self._maps = OrderedDict()
            self._closed = False
            self._pid = os.getpid()

    def _evict(self) -> None:
        for path, entry in list(self._maps.items()):
            if not self._closed and len(self._maps) <= self.keep:
                break
            if entry.users:
                continue
            del self._maps[path]
            # A retained exception traceback can hold a temporary numpy view.
            # Its reference then owns the mapping until the traceback is freed.
            with suppress(BufferError, OSError):
                entry.buffer.close()

    @contextmanager
    def acquire(self, path: str, expected_size: int) -> Iterator[mmap.mmap]:
        self._ensure_process()
        with self._lock:
            if self._closed:
                raise RuntimeError("Window mapping cache is closed.")
            entry = self._maps.get(path)
            if entry is None:
                with open(path, "rb") as handle:
                    actual_size = os.fstat(handle.fileno()).st_size
                    if actual_size != expected_size:
                        raise OSError(f"Chunk size changed: expected {expected_size} bytes, received {actual_size}")
                    entry = _Mapping(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
                self._maps[path] = entry
            elif len(entry.buffer) != expected_size:
                raise OSError("Chunk metadata changed while mapped.")
            self._maps.move_to_end(path)
            entry.users += 1
            self._evict()
        try:
            yield entry.buffer
        finally:
            with self._lock:
                entry.users -= 1
                self._evict()

    def close(self) -> None:
        self._ensure_process()
        with self._lock:
            self._closed = True
            self._evict()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

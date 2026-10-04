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
r"""In-file zstd for pytree chunks (``compression="zstd"`` + ``compression_level``).

Omitted ``compression_level`` with ``compression="zstd"`` / ``"zstd:N"`` is
``"batch"`` (``compression_batch_size`` defaults to 256, matching Arrow IPC
and the aligned decode window). Explicit
``compression_level="chunk"`` is whole-file ``.zstd.bin``.
``compression_level="batch"`` writes ``.bin`` with this layout (magic first so
legacy readers fail closed — their uint32 ``num_items`` parse of ``LDFZ``
cannot match ``index.json`` ``chunk_size``, instead of ``unpack_from`` on
empty item slices)::

    LDFZ01\\0\\0                 8 bytes magic
    version u16                  1
    flags u16                    0
    num_items u32
    frame_rows u32               K (default 256)
    n_frames u32
    offsets u32[N+1]             relative to uncompressed concatenated item bytes
    frames[n_frames]:
      first_item u32
      n_items u32
      compressed_off u64         file offset of this zstd frame
      compressed_len u32
      uncompressed_len u32
    concatenated zstd frames of item bytes

``compression_level="sample"`` keeps the classic pytree header; each item
payload between offsets is its own zstd frame. The reader inflates one item
(or one batch frame) into the decode window.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from litdata.constants import _PYARROW_AVAILABLE, _ZSTD_AVAILABLE

_COMPRESSION_LEVELS = frozenset({"chunk", "batch", "sample"})
_FRAMED_MAGIC = b"LDFZ01\0\0"
_FRAMED_VERSION = 1
_PREFIX_STRUCT = struct.Struct("<8sHHIII")  # 24 bytes
_FRAME_STRUCT = struct.Struct("<IIQII")  # 24 bytes
_DEFAULT_ZSTD_LEVEL = 4
DEFAULT_ZSTD_WRAP = "batch"
DEFAULT_COMPRESSION_BATCH_SIZE = 256


def parse_compression_level(value: Any) -> str:
    """Normalize ``compression_level`` to ``chunk``, ``batch``, or ``sample``.

    Numeric zstd levels are ``compression="zstd:4"``, not this argument.
    ``None`` / omitted on **read** (``index.json``) means ``chunk`` so legacy
    whole-file ``.zstd.bin`` stays whole-file. Writers use
    :func:`resolve_write_compression_level` (omitted zstd → ``batch``).
    """
    if value is None or value == "":
        return "chunk"
    if isinstance(value, bool):
        raise ValueError("compression_level must be 'chunk', 'batch', or 'sample'.")
    if isinstance(value, int):
        raise ValueError(
            f"compression_level={value!r} is granularity, not a zstd numeric level. "
            "Use compression='zstd:4' for the algorithm level and "
            "compression_level='chunk'|'batch'|'sample' for how much to wrap."
        )
    key = str(value).strip().lower()
    if key not in _COMPRESSION_LEVELS:
        raise ValueError(f"compression_level must be 'chunk', 'batch', or 'sample', got {value!r}.")
    return key


def resolve_write_compression_level(compression: str | None, compression_level: str | None) -> str:
    """Write-time wrap. Omitted ``zstd`` / ``zstd:N`` is ``batch``; other codecs stay ``chunk``."""
    if compression_level is not None and compression_level != "":
        return parse_compression_level(compression_level)
    if compression and str(compression).startswith("zstd"):
        return DEFAULT_ZSTD_WRAP
    return "chunk"


def is_in_file_compression(level: Any) -> bool:
    """True when zstd frames live inside ``.bin`` (batch or sample), not ``.zstd.bin``."""
    return parse_compression_level(level) in {"batch", "sample"}


def is_framed_chunk(view: bytes | bytearray | memoryview) -> bool:
    return len(view) >= 8 and bytes(view[:8]) == _FRAMED_MAGIC


@dataclass(frozen=True, slots=True)
class FrameSpec:
    first_item: int
    n_items: int
    compressed_off: int
    compressed_len: int
    uncompressed_len: int


@dataclass(frozen=True, slots=True)
class FramedHeader:
    version: int
    flags: int
    num_items: int
    frame_rows: int
    offsets: list[int]
    frames: list[FrameSpec]


def pack_framed_zstd(classic_chunk: bytes, compressor: Any, frame_rows: int) -> bytes:
    """Wrap a classic pytree chunk (num_items + file offsets + items) as framed zstd."""
    if len(classic_chunk) < 8:
        raise RuntimeError("classic pytree chunk is truncated")
    n = int(struct.unpack_from("<I", classic_chunk, 0)[0])
    if n <= 0:
        raise RuntimeError("framed zstd chunk has no items")
    header_end = 4 + 4 * (n + 1)
    if len(classic_chunk) < header_end:
        raise RuntimeError("classic pytree chunk offset table is truncated")
    file_offsets = np.frombuffer(classic_chunk, dtype=np.uint32, count=n + 1, offset=4)
    k = max(1, int(frame_rows))
    logical = np.empty(n + 1, dtype=np.uint32)
    logical[0] = 0
    for i in range(n):
        logical[i + 1] = logical[i] + (int(file_offsets[i + 1]) - int(file_offsets[i]))

    n_frames = (n + k - 1) // k
    prefix_len = _PREFIX_STRUCT.size
    offsets_len = 4 * (n + 1)
    table_len = _FRAME_STRUCT.size * n_frames
    payload_cursor = prefix_len + offsets_len + table_len

    compressed_parts: list[bytes] = []
    frames: list[tuple[int, int, int, int, int]] = []
    for frame_i in range(n_frames):
        start = frame_i * k
        end = min(n, start + k)
        raw = classic_chunk[int(file_offsets[start]) : int(file_offsets[end])]
        compressed = compressor.compress(raw)
        frames.append((start, end - start, payload_cursor, len(compressed), len(raw)))
        compressed_parts.append(compressed)
        payload_cursor += len(compressed)

    out = bytearray(payload_cursor)
    _PREFIX_STRUCT.pack_into(out, 0, _FRAMED_MAGIC, _FRAMED_VERSION, 0, n, k, n_frames)
    out[prefix_len : prefix_len + offsets_len] = logical.tobytes()
    table_off = prefix_len + offsets_len
    for i, (first, n_items, coff, compressed_len, uncompressed_len) in enumerate(frames):
        _FRAME_STRUCT.pack_into(
            out, table_off + i * _FRAME_STRUCT.size, first, n_items, coff, compressed_len, uncompressed_len
        )
        out[coff : coff + compressed_len] = compressed_parts[i]
    return bytes(out)


def parse_framed_header(view: bytes | bytearray | memoryview) -> FramedHeader:
    """Parse the uncompressed directory. Raises if magic/tables are missing or truncated."""
    if not is_framed_chunk(view):
        raise RuntimeError("not a framed zstd chunk (missing LDFZ01 magic)")
    if len(view) < _PREFIX_STRUCT.size:
        raise RuntimeError("truncated framed zstd header")
    _magic, version, flags, n, k, n_frames = _PREFIX_STRUCT.unpack_from(view, 0)
    if version != _FRAMED_VERSION:
        raise RuntimeError(f"unsupported framed zstd version {version}")
    if n <= 0 or n_frames <= 0 or k <= 0:
        raise RuntimeError(f"invalid framed zstd header n={n} frames={n_frames} k={k}")
    offsets_off = _PREFIX_STRUCT.size
    offsets_end = offsets_off + 4 * (n + 1)
    table_end = offsets_end + _FRAME_STRUCT.size * n_frames
    if len(view) < table_end:
        raise RuntimeError("truncated framed zstd offset/frame table")
    offsets = np.frombuffer(view, dtype=np.uint32, count=n + 1, offset=offsets_off).tolist()
    frames: list[FrameSpec] = []
    pos = offsets_end
    view_len = len(view)
    for _ in range(n_frames):
        first, n_items, coff, compressed_len, uncompressed_len = _FRAME_STRUCT.unpack_from(view, pos)
        if (
            n_items <= 0
            or compressed_len <= 0
            or uncompressed_len <= 0
            or coff < table_end
            or coff + compressed_len > view_len
        ):
            raise RuntimeError("framed zstd frame table is invalid or truncated")
        frames.append(
            FrameSpec(
                first_item=int(first),
                n_items=int(n_items),
                compressed_off=int(coff),
                compressed_len=int(compressed_len),
                uncompressed_len=int(uncompressed_len),
            )
        )
        pos += _FRAME_STRUCT.size
    return FramedHeader(
        version=int(version),
        flags=int(flags),
        num_items=int(n),
        frame_rows=int(k),
        offsets=offsets,
        frames=frames,
    )


def zstd_level_from_name(name: str | None) -> int:
    """Numeric zstd level from ``compression='zstd'`` / ``'zstd:N'`` (default 4)."""
    if not name:
        return _DEFAULT_ZSTD_LEVEL
    text = str(name)
    if ":" not in text:
        return _DEFAULT_ZSTD_LEVEL
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return _DEFAULT_ZSTD_LEVEL


def _python_zstd_mod() -> Any:
    if not _ZSTD_AVAILABLE:
        raise ModuleNotFoundError(str(_ZSTD_AVAILABLE))
    import sys

    if sys.version_info >= (3, 14):
        from compression import zstd as mod
    else:
        import zstd as mod
    return mod


class _PythonZstdCodec:
    """``zstd`` package fallback (needs a ``bytes`` copy of mmap slices)."""

    name = "python-zstd"

    def __init__(self, level: int = _DEFAULT_ZSTD_LEVEL) -> None:
        self.level = level
        self._mod: Any | None = None
        self._buf: memoryview | None = None

    def _zstd(self) -> Any:
        if self._mod is None:
            self._mod = _python_zstd_mod()
        return self._mod

    def compress(self, data: bytes | bytearray | memoryview) -> bytes:
        payload = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        return self._zstd().compress(payload, self.level)

    def decompress(
        self,
        data: bytes | bytearray | memoryview,
        decompressed_size: int | None = None,
    ) -> bytes:
        blob = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        raw = self._zstd().decompress(blob)
        if decompressed_size is not None and len(raw) != decompressed_size:
            raise RuntimeError(f"zstd frame inflated to {len(raw)} bytes, expected {decompressed_size}")
        return raw


class _ArrowZstdCodec:
    """PyArrow C++ ``Codec('zstd')`` — GIL released, mmap ``memoryview``, reused per loader.

    ``decompress(..., decompressed_size=ulen)`` writes into a pool Buffer kept on
    ``_buf`` so the returned ``memoryview`` stays valid through the decode window.
    Sample-level inflate (no stored size) falls back to the ``zstd`` package.
    """

    name = "arrow-zstd"

    def __init__(self, level: int = _DEFAULT_ZSTD_LEVEL) -> None:
        import pyarrow as pa

        self.level = level
        self._pa = pa
        self._codec = pa.Codec("zstd", compression_level=level)
        self._pool = pa.default_memory_pool()
        self._buf: Any = None
        self._python: _PythonZstdCodec | None = None

    def compress(self, data: bytes | bytearray | memoryview) -> bytes:
        return self._codec.compress(data, asbytes=True)

    def decompress(
        self,
        data: bytes | bytearray | memoryview,
        decompressed_size: int | None = None,
    ) -> bytes | memoryview:
        if decompressed_size is None:
            if self._python is None:
                self._python = _PythonZstdCodec(self.level)
            return self._python.decompress(data)
        buf = self._codec.decompress(
            data,
            decompressed_size=decompressed_size,
            asbytes=False,
            memory_pool=self._pool,
        )
        if buf.size != decompressed_size:
            raise RuntimeError(f"zstd frame inflated to {buf.size} bytes, expected {decompressed_size}")
        self._buf = buf
        return memoryview(buf)


def make_zstd_codec(compression: str | None = None) -> _ArrowZstdCodec | _PythonZstdCodec:
    """Arrow C++ zstd when ``pyarrow`` has the codec; else the ``zstd`` package."""
    level = zstd_level_from_name(compression)
    if _PYARROW_AVAILABLE:
        try:
            import pyarrow as pa

            if pa.Codec.is_available("zstd"):
                return _ArrowZstdCodec(level)
        except (ImportError, ValueError, AttributeError, OSError):
            pass
    return _PythonZstdCodec(level)


def inflate_frame(
    view: bytes | bytearray | memoryview,
    header: FramedHeader,
    frame_index: int,
    compressor: Any,
) -> bytes | memoryview:
    """Decompress one zstd frame into the uncompressed item-byte payload for that window.

    Passes a ``memoryview`` slice (no ``bytes()`` copy of the compressed body) and
    ``decompressed_size=frame.uncompressed_len`` so Arrow's C++ codec can allocate
    once. Legacy ``Compressor.decompress(data)`` still works (copies to ``bytes``).
    """
    if frame_index < 0 or frame_index >= len(header.frames):
        raise RuntimeError(f"framed zstd frame index {frame_index} out of range")
    frame = header.frames[frame_index]
    src = memoryview(view)[frame.compressed_off : frame.compressed_off + frame.compressed_len]
    if len(src) != frame.compressed_len:
        raise RuntimeError("truncated zstd frame")
    ulen = frame.uncompressed_len
    decompress = compressor.decompress
    try:
        raw = decompress(src, decompressed_size=ulen)
    except TypeError:
        raw = decompress(bytes(src))
    if len(raw) != ulen:
        raise RuntimeError(f"zstd frame inflated to {len(raw)} bytes, expected {ulen}")
    return raw


def frame_index_for_item(header: FramedHeader, table_idx: int) -> int:
    if table_idx < 0 or table_idx >= header.num_items:
        raise RuntimeError(f"item {table_idx} out of range for framed chunk ({header.num_items} items)")
    k = header.frame_rows
    frame_i = table_idx // k
    if frame_i >= len(header.frames):
        raise RuntimeError(f"item {table_idx} maps to missing frame {frame_i}")
    frame = header.frames[frame_i]
    if not (frame.first_item <= table_idx < frame.first_item + frame.n_items):
        raise RuntimeError(
            f"item {table_idx} not in frame {frame_i} [{frame.first_item}, {frame.first_item + frame.n_items})"
        )
    return frame_i


def pack_sample_zstd(classic_chunk: bytes, compressor: Any) -> bytes:
    """Keep the classic pytree header; zstd-compress each item payload independently."""
    if len(classic_chunk) < 8:
        raise RuntimeError("classic pytree chunk is truncated")
    n = int(struct.unpack_from("<I", classic_chunk, 0)[0])
    if n <= 0:
        raise RuntimeError("sample zstd chunk has no items")
    header = 4 + 4 * (n + 1)
    if len(classic_chunk) < header:
        raise RuntimeError("classic pytree chunk offset table is truncated")
    file_offsets = np.frombuffer(classic_chunk, dtype=np.uint32, count=n + 1, offset=4)
    parts = [compressor.compress(classic_chunk[int(file_offsets[i]) : int(file_offsets[i + 1])]) for i in range(n)]
    offsets = np.empty(n + 1, dtype=np.uint32)
    offsets[0] = header
    for i, part in enumerate(parts):
        offsets[i + 1] = offsets[i] + len(part)
    out = bytearray(int(offsets[-1]))
    out[0:4] = np.uint32(n).tobytes()
    out[4:header] = offsets.tobytes()
    pos = header
    for part in parts:
        out[pos : pos + len(part)] = part
        pos += len(part)
    return bytes(out)

"""Read selected axis-0 frames without fetching complete records on S3/R2.

TemporalArrayLoader records use index metadata to plan one range per selected
group; see temporal.py for the byte layout and a complete two-track example.
Ordinary PyTree array records are also supported, but their per-field headers
must first be read and cached, and each selected field requires its own range.
Only metadata is retained here: repeated remote windows fetch their payload again.
"""

import asyncio
import math
import struct
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from litdata.constants import _NUMPY_DTYPES_MAPPING, _TORCH_DTYPES_MAPPING
from litdata.streaming.config import ChunksConfig
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.temporal import TemporalArrayLoader


class _WindowReader:
    """Plan bounded range reads; retain metadata only, never decoded payloads."""

    def __init__(self, config: ChunksConfig) -> None:
        self.config = config
        self.length = sum(interval[2] - interval[1] for interval in config.intervals)
        self.metadata: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        spec = config.config.get("data_spec")
        if (
            config.config.get("item_loader", "PyTreeLoader") not in ("PyTreeLoader", "TemporalArrayLoader")
            or config.config.get("compression")
            or config.config.get("encryption")
            or config.config.get("format") == "mds"
        ):
            raise ValueError("Window reads require uncompressed, unencrypted PyTree chunks from optimize().")
        self.temporal = config._item_loader if isinstance(config._item_loader, TemporalArrayLoader) else None
        if self.temporal is not None:
            assert self.temporal.schema is not None
            self.names = self.temporal.schema["field_order"]
            self.formats = []
            return
        if spec is None or spec.type is not dict or any(not child.is_leaf() for child in spec.children_specs):
            raise ValueError("Window reads require a flat dictionary of named array fields.")
        self.names = list(spec.context)
        if any(not isinstance(name, str) for name in self.names):
            raise ValueError("Window field names must be strings.")
        self.formats = config.config["data_format"]

    async def _read(self, index: ChunkedIndex, offset: int, length: int) -> bytes:
        path, _, size = self.config[index]
        if offset < 0 or length < 0 or offset + length > size:
            raise ValueError("Array metadata points outside its chunk.")
        if length == 0:
            return b""
        downloader = self.config._downloader
        if downloader is not None:
            # Always use immutable remote objects: a shared chunk cache can be evicted concurrently.
            filename = self.config._chunks[index.chunk_index]["filename"]  # type: ignore[index]
            remote = str(self.config._remote_dir).rstrip("/") + "/" + filename
            data = await downloader.adownload_bytes(remote, offset, length, path)
        else:

            def read_local() -> bytes:
                with open(path, "rb") as handle:
                    handle.seek(offset)
                    return handle.read(length)

            data = await asyncio.to_thread(read_local)
        if len(data) != length:
            raise OSError(f"Short window read: expected {length} bytes, received {len(data)}")
        return data

    async def _record(self, index: ChunkedIndex) -> dict[str, Any]:
        key = (index.chunk_index, index.index)
        if key in self.metadata:
            self.metadata.move_to_end(key)
            return self.metadata[key]
        _, begin, chunk_bytes = self.config[index]
        local_index = index.index - begin
        pair = await self._read(index, 4 * (1 + local_index), 8)
        start, end = np.frombuffer(pair, dtype=np.uint32).tolist()
        chunks = self.config._chunks
        assert chunks is not None
        chunk_header = 4 * (chunks[index.chunk_index]["chunk_size"] + 2)
        header_size = 4 * len(self.names)
        if start < chunk_header or end > chunk_bytes or end - start < header_size:
            raise ValueError("Invalid item offsets in chunk.")
        sizes = struct.unpack(f"<{len(self.names)}I", await self._read(index, start, header_size))
        if header_size + sum(sizes) != end - start:
            raise ValueError("Invalid field sizes in chunk.")
        offsets = []
        cursor = start + header_size
        for size in sizes:
            offsets.append(cursor)
            cursor += size
        result = {"offsets": offsets, "sizes": sizes, "fields": {}}
        self.metadata[key] = result
        # Bound memory independently of the number of records in the dataset.
        if len(self.metadata) > 256:
            self.metadata.popitem(last=False)
        return result

    async def _field(self, index: ChunkedIndex, record: dict[str, Any], position: int) -> tuple:
        if position in record["fields"]:
            return record["fields"][position]
        offset, size = record["offsets"][position], record["sizes"][position]
        format_name = self.formats[position]
        kind = format_name.split(":")[0]
        is_tensor = kind in ("tensor", "no_header_tensor")
        mapping: dict[int, Any] = _TORCH_DTYPES_MAPPING if is_tensor else _NUMPY_DTYPES_MAPPING
        header_size = 0
        if kind.startswith("no_header_"):
            dtype = mapping[int(format_name.split(":")[1])]
            width = torch.empty((), dtype=dtype).element_size() if is_tensor else np.dtype(dtype).itemsize
            shape = (size // width,) if width else ()
        else:
            if size < 8:
                raise ValueError("Truncated array header.")
            header = await self._read(index, offset, 8)
            dtype_id, rank = struct.unpack(">II" if is_tensor else "=II", header)
            if not 1 <= rank <= 64 or 8 + 4 * rank > size:
                raise ValueError("Invalid array rank.")
            dtype = mapping[dtype_id]
            shape_bytes = await self._read(index, offset + 8, 4 * rank)
            shape = struct.unpack((">" if is_tensor else "=") + f"{rank}I", shape_bytes)
            header_size = 8 + 4 * rank
            width = torch.empty((), dtype=dtype).element_size() if is_tensor else np.dtype(dtype).itemsize
        if not is_tensor and np.dtype(dtype).kind not in "biufc":
            raise ValueError("Window reads support fixed-width numeric/bool arrays only.")
        if not shape or any(d <= 0 for d in shape) or header_size + math.prod(shape) * width != size:
            raise ValueError("Invalid array shape or byte length.")
        result = (offset + header_size, shape, dtype, width, is_tensor)
        record["fields"][position] = result
        return result

    async def read(
        self, index: ChunkedIndex, start: int, frames: int, fields: Sequence[str] | None, max_concurrent_reads: int
    ) -> dict[str, Any]:
        names = self.names if fields is None else list(fields)
        if isinstance(fields, str) or any(not isinstance(name, str) for name in names):
            raise TypeError("fields must be a sequence of field names, not a string.")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate requested fields.")
        if self.temporal is not None:
            return await self._read_temporal(index, start, frames, names, max_concurrent_reads)
        positions = []
        for name in names:
            if name not in self.names:
                raise KeyError(name)
            position = self.names.index(name)
            if self.formats[position].split(":")[0] not in ("numpy", "tensor", "no_header_numpy", "no_header_tensor"):
                raise ValueError(f"Field {name!r} is not a fixed-width NumPy array or tensor.")
            positions.append(position)
        if not names:
            return {}
        record = await self._record(index)
        # Validate all selected fields before fetching any payload.
        plans = [await self._field(index, record, position) for position in positions]
        if len({plan[1][0] for plan in plans}) != 1:
            raise ValueError("Selected fields must share the same frame-axis length.")
        if start + frames > plans[0][1][0]:
            raise IndexError("Window is outside the record.")
        permits = asyncio.Semaphore(max_concurrent_reads)

        async def fetch(name: str, plan: tuple) -> tuple[str, Any]:
            offset, shape, dtype, width, is_tensor = plan
            stride = math.prod(shape[1:]) * width
            async with permits:
                data = await self._read(index, offset + start * stride, frames * stride)
            output_shape = (frames, *shape[1:])
            if is_tensor:
                value = torch.frombuffer(bytearray(data), dtype=dtype).reshape(output_shape)
            else:
                value = np.frombuffer(data, dtype=dtype).reshape(output_shape).copy()
            return name, value

        tasks = [asyncio.create_task(fetch(name, plan)) for name, plan in zip(names, plans)]
        try:
            return dict(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_temporal(
        self, index: ChunkedIndex, start: int, frames: int, names: list[str], max_concurrent_reads: int
    ) -> dict[str, Any]:
        loader = self.temporal
        assert loader is not None
        assert loader.schema is not None
        for name in names:
            if name not in loader.schema["fields"]:
                raise KeyError(name)
        _, begin, size = self.config[index]
        chunks = self.config._chunks
        assert chunks is not None
        chunk = chunks[index.chunk_index]
        position = index.index - begin
        offset = chunk["temporal_offsets"][position]
        count = chunk["temporal_frames"][position]
        if type(count) is not int or count < 1 or type(offset) is not int or offset < 4 * (chunk["chunk_size"] + 2):
            raise ValueError("Invalid temporal record metadata.")
        end = offset + 4 + count * sum(dtype.itemsize for dtype in loader._group_dtypes)
        expected_end = chunk["temporal_offsets"][position + 1] if position + 1 < chunk["chunk_size"] else size
        if end != expected_end or end > size:
            raise ValueError("Temporal record metadata exceeds its item boundary.")
        if start + frames > count:
            raise IndexError("Window is outside the record.")
        groups = loader.schema["groups"]
        # Skip the record's uint32 frame count. Each group contains ALL count
        # frames, so locating the next group advances by count, not window length.
        # For a group with row size R, read [base + start*R, base + (start+frames)*R).
        # In temporal.py's example, track 0 / start=3 / frames=4 yields:
        #   features:    base=20,  R=8 -> offset=44,  length=32 (S3 bytes 44..75)
        #   score/valid: base=100, R=8 -> offset=124, length=32 (S3 bytes 124..155)
        # Selecting features + valid therefore fetches 64 bytes in two requests,
        # including score/padding in the selected group, and returns only those
        # two fields. No other frames, tracks, or payload headers are fetched.
        cursor = offset + 4
        plans = []
        for group, dtype in enumerate(loader._group_dtypes):
            if any(name in names for name in groups[group]):
                plans.append((group, cursor + start * dtype.itemsize, frames * dtype.itemsize))
            cursor += count * dtype.itemsize
        permits = asyncio.Semaphore(max_concurrent_reads)

        async def fetch(group: int, offset: int, length: int) -> tuple[int, bytes]:
            async with permits:
                return group, await self._read(index, offset, length)

        # Parallelize selected groups within this window. The caller controls the
        # number of windows in flight and sampling/batching across ranks/workers.
        tasks = [asyncio.create_task(fetch(*plan)) for plan in plans]
        try:
            buffers = dict(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return loader.decode_groups(buffers, frames, names)

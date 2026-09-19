"""Temporal array item layout for optimize and StreamingDataset.

Store each complete track once; choose training windows later with read_window().
An input record is a flat dictionary of arrays with time on axis 0. For example,
features: float32[T, 2], score: float32[T], valid: bool[T]. Only T may vary between
records. The application chooses tracks/windows; LitData owns packing and I/O.

BinaryWriter supplies the usual chunk envelope (see writer.py):
    [record count: uint32][N + 1 absolute offsets: uint32][record 0][record 1]...
Each temporal record is:
    [frame count: little-endian uint32][group 0 bytes][group 1 bytes]...
Records stay whole within a chunk; chunk_bytes is a packing target, not a window
length. The existing chunk envelope uses native-endian uint32 (little-endian on
our usual platforms); array byte order is preserved in the schema's dtype strings.

Each group is a contiguous sequence of fixed-size frame rows, constructed with
np.dtype(..., align=True). With groups [["features"], ["score", "valid"]]:
    group 0: [x0, y0][x1, y1]...                       -> 8 bytes/frame
    group 1: [score0, valid0, padding][score1, ...]... -> 8 bytes/frame
The second row has a 4-byte float, a 1-byte bool and 3 zero padding bytes. This
alignment (including trailing padding) is part of the layout: range calculations
must use the structured dtype's itemsize, not the sum of useful field bytes.

Concrete example: one chunk containing tracks of 10 and 6 frames (280 bytes):
    bytes   0..3:   record count = 2
    bytes   4..15:  boundaries = [16, 180, 280]
    bytes  16..19:  track 0 frame count = 10
    bytes  20..99:  track 0 features
    bytes 100..179: track 0 score + valid
    bytes 180..183: track 1 frame count = 6
    bytes 184..231: track 1 features
    bytes 232..279: track 1 score + valid
All byte ranges in this example are inclusive. A record occupies
4 + T * sum(group.itemsize) bytes; no per-field headers are stored in its payload.

index.json stores temporal_schema once: field dtypes, trailing shapes, original
array/tensor types, field order and groups. Per chunk it also stores record starts
(temporal_offsets = [16, 180]) and lengths (temporal_frames = [10, 6]). Duplicating
these small headers in the index lets window.py compute ranges without fetching
chunk/record headers first. The same schema reconstructs group dtypes on read.

Grouping trades fewer requests for extra bytes: selecting valid also fetches
score and padding, but only valid is returned. Keep large independently selected
fields separate; group small fields commonly requested together. Unlisted fields
each form a separate group. Compression/encryption are unsupported because this
layout relies on directly addressable fixed-size rows.
"""

import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from litdata.streaming.item_loader import PyTreeLoader
from litdata.utilities.encryption import Encryption


class TemporalArrayLoader(PyTreeLoader):
    """Store complete array records with optional groups for efficient window reads.

    Pass the same loader type to ``optimize`` and ``StreamingDataset``. Field groups
    are configured only when writing; unlisted fields each get their own group.
    Grouping small fields reduces range requests but reads the entire group when
    any member is selected. Arrays must have a common leading frame dimension;
    names, dtypes, trailing shapes and array/tensor types must agree across records.
    Frame counts may vary. Compression, encryption and custom serializers are unsupported.
    """

    def __init__(self, field_groups: Sequence[Sequence[str]] | None = None) -> None:
        super().__init__(batch_decode=1)
        self.field_groups = [list(group) for group in field_groups] if field_groups is not None else []
        names = [name for group in self.field_groups for name in group]
        if (
            any(isinstance(group, str) or not group for group in field_groups or [])
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("field_groups must contain nonempty groups of distinct field names.")
        self.schema: dict[str, Any] | None = None
        self._group_dtypes: list[np.dtype] = []
        self._group_by_name: dict[str, int] = {}

    def set_batch_decode(self, batch_decode: Any) -> None:
        # This layout has its own decoder, not the default PyTree leaf batch decoder.
        super().set_batch_decode(1)

    def setup(self, config: dict, *args: Any, **kwargs: Any) -> None:
        super().setup(config, *args, **kwargs)
        self._set_schema(config["temporal_schema"])

    def _set_schema(self, schema: dict[str, Any]) -> None:
        if schema.get("version") != 1:
            raise ValueError("Unsupported temporal array schema version.")
        names = schema["field_order"]
        grouped = [name for group in schema["groups"] for name in group]
        if (
            not names
            or len(set(names)) != len(names)
            or set(names) != set(schema["fields"])
            or sorted(grouped) != sorted(names)
            or any(not group for group in schema["groups"])
        ):
            raise ValueError("Temporal groups must contain each field exactly once.")
        self.schema = schema
        self._group_by_name = {name: group for group, fields in enumerate(schema["groups"]) for name in fields}
        self._group_dtypes = []
        for group in schema["groups"]:
            descriptions = []
            for name in group:
                field = schema["fields"][name]
                dtype = np.dtype(field["dtype"])
                if dtype.kind not in "biufc" or any(type(n) is not int or n < 1 for n in field["shape"]):
                    raise ValueError("Invalid temporal array dtype or shape.")
                descriptions.append((name, dtype, tuple(field["shape"])))
            self._group_dtypes.append(np.dtype(descriptions, align=True))

    def serialize_record(self, record: Any) -> bytes:
        if not isinstance(record, Mapping) or not record:
            raise ValueError("TemporalArrayLoader expects a nonempty dictionary of arrays.")
        arrays = {}
        fields = {}
        frames = None
        for name, value in record.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Field names must be nonempty strings.")
            tensor = isinstance(value, torch.Tensor)
            if not tensor and not isinstance(value, np.ndarray):
                raise ValueError("Temporal fields must be NumPy arrays or tensors.")
            array = value.detach().cpu().numpy() if tensor else value
            if array.ndim < 1 or any(n < 1 for n in array.shape) or array.dtype.kind not in "biufc":
                raise ValueError("Temporal fields require nonempty fixed-width numeric/bool arrays.")
            if frames is not None and array.shape[0] != frames:
                raise ValueError("All temporal fields must share the frame axis.")
            frames = array.shape[0]
            arrays[name] = array
            fields[name] = {"dtype": array.dtype.str, "shape": list(array.shape[1:]), "tensor": tensor}
        if self.schema is None:
            grouped = {name for group in self.field_groups for name in group}
            if grouped - fields.keys():
                raise ValueError(f"Unknown grouped fields: {sorted(grouped - fields.keys())}")
            groups = self.field_groups + [[name] for name in fields if name not in grouped]
            self._set_schema({"version": 1, "fields": fields, "field_order": list(fields), "groups": groups})
        assert self.schema is not None
        assert frames is not None
        if fields != self.schema["fields"]:
            raise ValueError("Temporal record schema differs: field names, dtypes and trailing shapes must match.")
        data = [struct.pack("<I", frames)]
        for group, dtype in zip(self.schema["groups"], self._group_dtypes):
            # Interleave only within this group, frame by frame. Zero initialization
            # makes alignment padding deterministic instead of storing uninitialized bytes.
            packed = np.zeros(frames, dtype=dtype)
            for name in group:
                packed[name] = arrays[name]
            data.append(packed.tobytes())
        return b"".join(data)

    def decode_groups(
        self, buffers: Mapping[int, bytes | memoryview], frames: int, fields: Sequence[str]
    ) -> dict[str, Any]:
        assert self.schema is not None
        views = {
            group: np.frombuffer(data, dtype=self._group_dtypes[group], count=frames) for group, data in buffers.items()
        }
        result = {}
        for name in fields:
            # Structured views are strided and may reference immutable network bytes.
            # Copy only requested fields into independent, writable output arrays.
            array = views[self._group_by_name[name]][name].copy()
            result[name] = torch.from_numpy(array) if self.schema["fields"][name]["tensor"] else array
        return result

    def deserialize(self, raw_item_data: bytes | memoryview) -> Any:
        assert self.schema is not None
        frames = struct.unpack_from("<I", raw_item_data)[0]
        expected = 4 + frames * sum(dtype.itemsize for dtype in self._group_dtypes)
        if frames < 1 or len(raw_item_data) != expected:
            raise ValueError("Invalid temporal record length.")
        cursor = 4
        buffers = {}
        for group, dtype in enumerate(self._group_dtypes):
            length = frames * dtype.itemsize
            buffers[group] = raw_item_data[cursor : cursor + length]
            cursor += length
        return self.decode_groups(buffers, frames, self.schema["field_order"])

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
        encryption: Encryption | None = None,
    ) -> Any:
        # Full-record compatibility. Window reads use the direct range path instead.
        if encryption is not None:
            raise ValueError("TemporalArrayLoader does not support encryption.")
        self._wait_until_chunk_ready(chunk_index, chunk_filepath, filesize_bytes)
        with open(chunk_filepath, "rb", buffering=0) as handle:
            return self.deserialize(self._load_data(handle, 4 * (1 + index - begin)))

    @property
    def bytes_per_frame(self) -> int:
        """Useful bytes across all fields, excluding any group padding."""
        if self.schema is None:
            raise RuntimeError("The loader schema has not been initialized.")
        return sum(
            np.dtype(field["dtype"]).itemsize * math.prod(field["shape"]) for field in self.schema["fields"].values()
        )

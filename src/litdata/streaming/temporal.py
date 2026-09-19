"""Temporal array item layout for optimize and StreamingDataset."""

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

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

"""Typed media wrappers and a fused JSON schema for ``optimize``.

Sample wrappers (``Audio``, ``Image``, …) disambiguate a path from a caption.
The schema IR below (``JsonType``, ``infer_type``, ``fuse_type``) describes
columns. ``optimize`` infers a type per sample and **fuses** them: a field
seen on only some rows becomes ``optional``. The fused JSON lives on
``index.json`` as ``config.types``. PyArrow is used only to convert parquet /
the nested chunk footer — not as the source of truth for types.

    optimize(lambda p: Audio(path=p), inputs=wavs, output_dir=...)
    optimize(lambda x: Image(array=x, quality=95, format="jpeg"), ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def is_pyg_data(item: Any) -> bool:
    """True for PyG ``Data`` / ``HeteroData`` / ``Batch`` (has ``to_dict``)."""
    module = getattr(type(item), "__module__", "")
    return module.startswith("torch_geometric.data") and hasattr(item, "to_dict")


@dataclass
class _MediaRef:
    path: str | None = None
    bytes: bytes | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = str(self.path)
        if self.bytes is not None and not isinstance(self.bytes, (bytes, bytearray)):
            raise TypeError(f"{type(self).__name__}.bytes must be bytes, got {type(self.bytes)}")
        if self.bytes is not None:
            self.bytes = bytes(self.bytes)


@dataclass
class Audio(_MediaRef):
    """Audio sample: path, bytes, or ``array=`` + ``sampling_rate=``."""

    array: Any = None
    sampling_rate: int | None = None
    num_channels: int | None = None
    stream_index: int | None = None


@dataclass
class Video(_MediaRef):
    """Video sample: path, bytes, frame ``array=``, or a torchcodec decoder."""

    array: Any = None
    fps: float = 25.0
    stream_index: int | None = None
    dimension_order: str = "NCHW"
    num_ffmpeg_threads: int = 1
    seek_mode: str = "approximate"
    device: str = "cpu"


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _Unset()


def _reject_quality_and_max_quality(cls_name: str, quality: int | None, max_quality: int | None) -> None:
    if quality is not None and max_quality is not None:
        raise ValueError(
            f"{cls_name}.quality and {cls_name}.max_quality are mutually exclusive. "
            "Use quality= to re-encode at that JPEG quality, or max_quality= to cap "
            "(keep existing JPEG bytes when estimated quality is already at or below the cap)."
        )


@dataclass
class Image(_MediaRef):
    """Image sample: path, bytes, numpy/tensor ``array=``, or a PIL ``image=``.

    ``quality``: re-encode at this JPEG quality (from array/PIL, or force re-encode of
    path/bytes). ``max_quality``: cap. If the sample is already JPEG and estimated
    quality is at or below the cap, keep the original bytes (do not upgrade Hub
    tiny-imagenet q=75 to 95). Higher-quality JPEGs, PNG, or raw pixels are encoded
    at ``max_quality``. Only one of ``quality`` and ``max_quality`` may be set.
    Both default to ``None`` (path/bytes pass through). Prefer ``Jpeg`` when you
    want a default cap of 95.
    """

    array: Any = None
    image: Any = None
    mode: str | None = None
    format: str | None = None
    quality: int | None = None
    max_quality: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _reject_quality_and_max_quality("Image", self.quality, self.max_quality)


@dataclass
class Jpeg(_MediaRef):
    """JPEG sample.

    ``quality``: re-encode at this JPEG quality (from array/PIL, or force re-encode of
    path/bytes). ``max_quality``: cap (default 95). If the sample is already JPEG and
    estimated quality is at or below the cap, keep the original bytes (do not upgrade
    Hub tiny-imagenet q=75 to 95). Higher-quality JPEGs, PNG, or raw pixels are
    encoded at ``max_quality``. Only one of ``quality`` and ``max_quality`` may be
    set. ``Jpeg(quality=80)`` clears the default cap so the two are not both set.
    """

    array: Any = None
    image: Any = None
    mode: str | None = None
    quality: int | None = None
    max_quality: Any = _UNSET

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_quality is _UNSET:
            self.max_quality = None if self.quality is not None else 95
        _reject_quality_and_max_quality("Jpeg", self.quality, self.max_quality)


@dataclass
class JpegArray:
    """List of JPEGs (``JPEGArraySerializer``).

    Same ``quality`` / ``max_quality`` rules as ``Jpeg`` (default cap 95). Applied to
    child ``Jpeg`` samples that still use the default cap.
    """

    images: list[Any] = field(default_factory=list)
    quality: int | None = None
    max_quality: Any = _UNSET

    def __post_init__(self) -> None:
        if self.max_quality is _UNSET:
            self.max_quality = None if self.quality is not None else 95
        _reject_quality_and_max_quality("JpegArray", self.quality, self.max_quality)


@dataclass
class Pil(_MediaRef):
    """PIL image sample. ``array=`` is converted with ``Image.fromarray``."""

    array: Any = None
    image: Any = None
    mode: str | None = None


@dataclass
class Tiff(_MediaRef):
    """TIFF sample: path, bytes, or ``array=`` (written with tifffile)."""

    array: Any = None
    image: Any = None


@dataclass
class Text(_MediaRef):
    """Text sample: path, UTF-8 bytes, or ``text=`` string. Streams as ``str``."""

    text: str | None = None


@dataclass
class File(_MediaRef):
    """Generic file bytes (``FileSerializer``)."""


@dataclass
class Mesh(_MediaRef):
    """3D mesh: path, bytes, or a trimesh ``mesh=``."""

    mesh: Any = None
    file_type: str = "glb"


@dataclass
class Pdf(_MediaRef):
    """PDF sample: path, bytes, or a pdfplumber ``pdf=``."""

    pdf: Any = None


@dataclass
class Nifti(_MediaRef):
    """NIfTI volume: path, bytes, a nibabel ``image=``, or ``array=`` + ``affine=``."""

    image: Any = None
    array: Any = None
    affine: Any = None


GRAPH_FIELDS = ("x", "edge_index", "edge_attr", "y", "pos", "batch")


@dataclass
class Graph:
    """Graph sample stored as packed tensors from PyG ``Data.to_dict()``.

    Prefer passing a PyG ``Data`` / ``HeteroData`` (or ``Graph(...)``). On read,
    LitData calls ``Data.from_dict`` / ``HeteroData.from_dict``. Do not
    ``torch.save`` the graph — that is still pickle/zip. ``data=`` is only a
    pickle fallback for NetworkX or other opaque objects.

        Graph(x=node_feat, edge_index=edge_index, edge_attr=edge_w, y=label)
        Graph(data=pyg_data)  # uses pyg_data.to_dict(); field kwargs override
    """

    x: Any = None
    edge_index: Any = None
    edge_attr: Any = None
    y: Any = None
    pos: Any = None
    batch: Any = None
    data: Any = None

    def to_mapping(self) -> dict[str, Any]:
        """Flat mapping for pack / ``Data.from_dict``. Field kwargs override ``data=``."""
        mapping: dict[str, Any] = {}
        if self.data is not None and is_pyg_data(self.data):
            mapping.update(self.data.to_dict())
        elif isinstance(self.data, dict):
            mapping.update(self.data)
        elif self.data is not None:
            if any(getattr(self, name) is not None for name in GRAPH_FIELDS):
                raise TypeError(
                    "Graph.data= must be a PyG Data, a dict, or omitted. "
                    "Do not mix tensor fields with an opaque data= payload."
                )
            return mapping
        for name in GRAPH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                mapping[name] = value
        return mapping

    def to_pyg(self) -> Any:
        """``Data.from_dict`` / ``HeteroData.from_dict`` (requires torch-geometric)."""
        from torch_geometric.data import Data, HeteroData

        mapping = self.to_mapping()
        if any(
            isinstance(value, dict) and any(hasattr(child, "shape") for child in value.values())
            for value in mapping.values()
        ):
            return HeteroData.from_dict(mapping)
        return Data.from_dict(mapping)


@dataclass
class Tensor(_MediaRef):
    """Tensor sample for ``TensorSerializer`` / ``NoHeaderTensorSerializer`` / ``TokensLoader``.

    ``array=`` is a ``torch.Tensor`` or NumPy array. 1-D arrays use the no-header
    token layout (``TokensLoader``). ``shape`` is exposed so ``TokensLoader.encode_data``
    can read ``flattened[0].shape[0]``.
    """

    array: Any = None
    dtype: Any = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self.array is None:
            raise AttributeError("Tensor.shape requires array=")
        return tuple(self.array.shape)


# --- Fused JSON schema (index.json ``config.types``) -------------------------

_MEDIA_KINDS = {
    "Audio": "audio",
    "Video": "video",
    "Image": "image",
    "Jpeg": "jpeg",
    "JpegArray": "jpeg_array",
    "Pil": "pil",
    "Tiff": "tiff",
    "Text": "text",
    "File": "file",
    "Mesh": "mesh",
    "Pdf": "pdf",
    "Nifti": "nifti",
    "Graph": "graph",
    "Tensor": "tensor",
}
_SCALAR_KINDS = frozenset({"str", "int", "float", "bool", "bytes", "null", "json", *_MEDIA_KINDS.values()})


@dataclass
class JsonType:
    """One JSON type. ``optional`` means the field was missing or null on some rows."""

    kind: str
    value: JsonType | None = None
    fields: dict[str, JsonType] | None = None
    optional: bool = False


def _as_optional(schema: JsonType) -> JsonType:
    if schema.optional:
        return schema
    return JsonType(schema.kind, value=schema.value, fields=schema.fields, optional=True)


def infer_type(value: Any) -> JsonType:
    """Infer a JSON type from one Python value."""
    if value is None:
        return JsonType("null", optional=True)
    cls_name = type(value).__name__
    if cls_name == "JsonLeaf" and hasattr(value, "value"):
        return infer_type(value.value)
    media = _MEDIA_KINDS.get(cls_name)
    if media is not None:
        return JsonType(media)
    # PIL plugin classes (JpegImageFile, PngImageFile, …) are not the wrapper names above.
    module = getattr(type(value), "__module__", "") or ""
    if module.startswith("PIL.") and (cls_name.endswith("ImageFile") or cls_name == "Image"):
        return JsonType("pil")
    if isinstance(value, bool):
        return JsonType("bool")
    if isinstance(value, int):
        return JsonType("int")
    if isinstance(value, float):
        return JsonType("float")
    if isinstance(value, str):
        return JsonType("str")
    if isinstance(value, (bytes, bytearray)):
        return JsonType("bytes")
    if isinstance(value, list):
        if not value:
            return JsonType("list", value=None)
        fused = infer_type(value[0])
        for item in value[1:]:
            fused = fuse_type(fused, infer_type(item))
        return JsonType("list", value=fused)
    if isinstance(value, dict):
        return JsonType("struct", fields={key: infer_type(item) for key, item in value.items()})
    return JsonType("json")


def fuse_type(left: JsonType | None, right: JsonType | None) -> JsonType:
    """Merge two observations. Keys only on one side become ``optional``."""
    if left is None:
        return _as_optional(right) if right is not None else JsonType("json", optional=True)
    if right is None:
        return _as_optional(left)
    optional = left.optional or right.optional
    if left.kind == "null":
        return _as_optional(right)
    if right.kind == "null":
        return _as_optional(left)
    if left.kind == right.kind:
        if left.kind == "list":
            if left.value is None:
                value = right.value
            elif right.value is None:
                value = left.value
            else:
                value = fuse_type(left.value, right.value)
            return JsonType("list", value=value, optional=optional)
        if left.kind == "map":
            if left.value is None:
                value = right.value
            elif right.value is None:
                value = left.value
            else:
                value = fuse_type(left.value, right.value)
            return JsonType("map", value=value, optional=optional)
        if left.kind == "struct":
            keys = list(left.fields or {})
            for key in right.fields or {}:
                if key not in (left.fields or {}):
                    keys.append(key)
            fields: dict[str, JsonType] = {}
            left_fields = left.fields or {}
            right_fields = right.fields or {}
            for key in keys:
                if key in left_fields and key in right_fields:
                    fields[key] = fuse_type(left_fields[key], right_fields[key])
                elif key in left_fields:
                    fields[key] = _as_optional(left_fields[key])
                else:
                    fields[key] = _as_optional(right_fields[key])
            return JsonType("struct", fields=fields, optional=optional)
        return JsonType(left.kind, optional=optional)
    if left.kind == "map" and right.kind == "struct":
        return fuse_type(left, JsonType("map", value=_struct_value_type(right), optional=right.optional))
    if left.kind == "struct" and right.kind == "map":
        return fuse_type(JsonType("map", value=_struct_value_type(left), optional=left.optional), right)
    return JsonType("json", optional=optional)


def _struct_value_type(schema: JsonType) -> JsonType:
    fused: JsonType | None = None
    for item in (schema.fields or {}).values():
        fused = item if fused is None else fuse_type(fused, item)
    return fused or JsonType("json")


def type_to_json(schema: JsonType) -> Any:
    """Compact JSON: scalars are strings; lists/structs are objects."""
    if schema.kind in _SCALAR_KINDS and schema.kind not in {"json"} and not schema.optional:
        return schema.kind
    payload: dict[str, Any] = {"type": schema.kind}
    if schema.optional:
        payload["optional"] = True
    if schema.kind == "list" or schema.kind == "map":
        payload["value"] = type_to_json(schema.value) if schema.value else "json"
    elif schema.kind == "struct":
        payload["fields"] = {key: type_to_json(item) for key, item in (schema.fields or {}).items()}
    elif schema.kind == "json" and not schema.optional:
        return "json"
    return payload


def type_from_json(obj: Any) -> JsonType:
    if obj is None:
        return JsonType("json")
    if isinstance(obj, str):
        return JsonType(obj)
    if not isinstance(obj, dict):
        return JsonType("json")
    kind = obj.get("type") or obj.get("kind")
    if kind is None:
        return JsonType("struct", fields={key: type_from_json(item) for key, item in obj.items()})
    optional = bool(obj.get("optional", False))
    if kind == "list":
        return JsonType("list", value=type_from_json(obj.get("value", "json")), optional=optional)
    if kind == "map":
        return JsonType("map", value=type_from_json(obj.get("value", "json")), optional=optional)
    if kind == "struct":
        raw_fields = obj.get("fields") or {}
        return JsonType(
            "struct", fields={key: type_from_json(item) for key, item in raw_fields.items()}, optional=optional
        )
    return JsonType(str(kind), optional=optional)


def schema_to_json(schema: JsonType | None) -> Any:
    """``config.types``: a struct becomes a field map (plan / index shape)."""
    if schema is None:
        return None
    if schema.kind == "struct":
        return {key: type_to_json(item) for key, item in (schema.fields or {}).items()}
    return type_to_json(schema)


def schema_from_json(obj: Any) -> JsonType | None:
    if obj is None:
        return None
    if isinstance(obj, dict) and "type" not in obj and "kind" not in obj:
        return JsonType("struct", fields={key: type_from_json(item) for key, item in obj.items()})
    return type_from_json(obj)


def fuse_schema_json(left: Any, right: Any) -> Any:
    """Fuse two ``config.types`` payloads (worker index merge)."""
    if left is None:
        return right
    if right is None:
        return left
    return schema_to_json(fuse_type(schema_from_json(left), schema_from_json(right)))


def types_from_arrow(schema: Any, columns: list[str] | None = None) -> JsonType:
    """Build a struct schema from a PyArrow schema (HF / parquet). Nullable → optional."""
    names = columns if columns is not None else [field.name for field in schema]
    fields: dict[str, JsonType] = {}
    for name in names:
        try:
            pa_field = schema.field(name)
        except KeyError:
            fields[name] = JsonType("json", optional=True)
            continue
        fields[name] = _from_pa_type(pa_field.type, optional=bool(pa_field.nullable))
    return JsonType("struct", fields=fields)


def _from_pa_type(pa_type: Any, *, optional: bool = False) -> JsonType:
    import pyarrow as pa

    if pa.types.is_dictionary(pa_type):
        return _from_pa_type(pa_type.value_type, optional=optional)
    is_list_type = pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type)
    is_fixed = getattr(pa.types, "is_fixed_size_list", lambda _: False)(pa_type)
    if is_list_type or is_fixed:
        value_type = getattr(pa_type, "value_type", None) or getattr(pa_type, "type", None)
        return JsonType(
            "list", value=_from_pa_type(value_type) if value_type is not None else JsonType("json"), optional=optional
        )
    if pa.types.is_map(pa_type):
        item_type = getattr(pa_type, "item_type", None)
        return JsonType(
            "map", value=_from_pa_type(item_type) if item_type is not None else JsonType("json"), optional=optional
        )
    if pa.types.is_struct(pa_type):
        fields = {
            pa_type.field(i).name: _from_pa_type(pa_type.field(i).type, optional=bool(pa_type.field(i).nullable))
            for i in range(pa_type.num_fields)
        }
        return JsonType("struct", fields=fields, optional=optional)
    if pa.types.is_integer(pa_type):
        return JsonType("int", optional=optional)
    if pa.types.is_floating(pa_type):
        return JsonType("float", optional=optional)
    if pa.types.is_boolean(pa_type):
        return JsonType("bool", optional=optional)
    if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
        return JsonType("bytes", optional=optional)
    return JsonType("str", optional=optional)


def default_for(schema: JsonType) -> Any:
    if schema.kind == "list":
        return []
    if schema.kind == "map":
        return {}
    if schema.kind == "struct":
        return {key: default_for(item) for key, item in (schema.fields or {}).items()}
    if schema.kind == "int":
        return 0
    if schema.kind == "float":
        return 0.0
    if schema.kind == "bool":
        return False
    if schema.kind == "bytes":
        return b""
    if schema.kind == "str":
        return ""
    return None


def is_json_row(value: Any) -> bool:
    """True for dict/list/scalar JSON or raw bytes — safe to keep for the Arrow footer."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return True
    if type(value).__name__ == "JsonLeaf" and hasattr(value, "value"):
        return is_json_row(value.value)
    if isinstance(value, list):
        return all(is_json_row(item) for item in value)
    if isinstance(value, dict):
        return all(is_json_row(item) for item in value.values())
    return False


_ARROW_JSON_KINDS = frozenset({"str", "int", "float", "bool", "bytes", "null", "list", "map", "struct"})


def is_nested_type(schema: JsonType | None) -> bool:
    """True if any list / map / json / nested struct should use the Arrow footer."""
    if schema is None:
        return False
    if schema.kind in {"list", "map", "json"}:
        return True
    if schema.kind == "struct":
        return any(is_nested_type(item) for item in (schema.fields or {}).values())
    return False


def _is_arrow_json_schema(schema: JsonType) -> bool:
    """True when a field is JSON (not media / tensor) and can live in an Arrow table."""
    if schema.kind not in _ARROW_JSON_KINDS:
        return False
    if schema.kind == "struct":
        return all(_is_arrow_json_schema(item) for item in (schema.fields or {}).values())
    if schema.kind in {"list", "map"} and schema.value is not None:
        return _is_arrow_json_schema(schema.value)
    return True


def is_arrow_footer_type(schema: JsonType | None) -> bool:
    """True when rows should be stored as an Arrow IPC footer.

    Nested lists/maps already need it (pytree JSON is slow). Flat structs of
    strings (cnn_dailymail, IMDB, sst2) need it too: Arrow ``to_pylist`` of a
    record batch beats per-item pytree str deserialize, and IPC zstd skips
    whole-file Python inflate.
    """
    if schema is None:
        return False
    if schema.kind == "struct":
        fields = schema.fields or {}
        return bool(fields) and all(_is_arrow_json_schema(item) for item in fields.values())
    return is_nested_type(schema)


def wrap_for_pytree(
    sample: Any,
    schema: JsonType,
    *,
    keys: list[str] | None = None,
    wrap_leaf: Any = None,
) -> Any:
    """Fill missing optional fields and wrap lists/maps as one leaf (stable pytree).

    ``keys`` locks the first-sample field set so later extra fields do not change
    ``data_format``. Those extras still land in fused ``config.types`` and the
    Arrow footer (original sample).
    """
    if wrap_leaf is None:
        wrap_leaf = lambda value: value
    if schema.kind == "struct" and isinstance(sample, dict):
        fields = schema.fields or {}
        order = keys if keys is not None else list(fields)
        out: dict[str, Any] = {}
        for key in order:
            field_t = fields.get(key)
            if field_t is None:
                continue
            value = sample.get(key)
            if value is not None and type(value).__name__ == "JsonLeaf":
                value = getattr(value, "value", value)
            if value is None:
                value = default_for(field_t)
            out[key] = _wrap_value(value, field_t, wrap_leaf)
        return out
    return _wrap_value(sample, schema, wrap_leaf)


def _wrap_value(value: Any, schema: JsonType, wrap_leaf: Any) -> Any:
    if schema.kind in {"list", "map", "json"}:
        # Lists of PIL/tensors stay one pytree leaf (jpeg_array, …). Do not wrap
        # them as JsonLeaf — orjson cannot dump JpegImageFile.
        if not is_json_row(value):
            return value
        return wrap_leaf(value)
    if schema.kind == "struct":
        if not isinstance(value, dict):
            value = default_for(schema)
        return wrap_for_pytree(value, schema, wrap_leaf=wrap_leaf)
    return value

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

import io
import os
import pickle
import struct
import sys
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import suppress
from copy import copy
from dataclasses import asdict
from itertools import chain
from typing import Any

import numpy as np
import torch

from litdata.constants import (
    _AV_AVAILABLE,
    _NIBABEL_AVAILABLE,
    _NUMPY_DTYPES_MAPPING,
    _PDFPLUMBER_AVAILABLE,
    _PIL_AVAILABLE,
    _TORCH_DTYPES_MAPPING,
    _TORCHCODEC_AVAILABLE,
    _TRIMESH_AVAILABLE,
)
from litdata.types import (
    GRAPH_FIELDS,
    Audio,
    File,
    Graph,
    Image,
    Jpeg,
    JpegArray,
    Mesh,
    Nifti,
    Pdf,
    Pil,
    Tensor,
    Text,
    Tiff,
    Video,
    _MediaRef,
    is_pyg_data,
)

_torchcodec_ok: bool | None = None


def _torchcodec_usable() -> bool:
    """``RequirementCache`` only checks install, not whether the native lib loads."""
    global _torchcodec_ok
    if not _TORCHCODEC_AVAILABLE:
        return False
    if _torchcodec_ok is not None:
        return _torchcodec_ok
    try:
        import torch
        from torchcodec.encoders import AudioEncoder

        # Importing decoders is not enough: Windows often has the wheel but no FFmpeg DLLs.
        AudioEncoder(torch.zeros(1, 8), sample_rate=8000).to_file_like(io.BytesIO(), format="wav")
        _torchcodec_ok = True
    except Exception:
        _torchcodec_ok = False
    return _torchcodec_ok


def _torchvision_read_video_available() -> bool:
    """``torchvision.io.read_video`` was removed in recent torchvision builds."""
    if not _AV_AVAILABLE:
        return False
    try:
        import torchvision.io

        return hasattr(torchvision.io, "read_video")
    except Exception:
        return False


def _as_bytes(data: bytes | bytearray | memoryview) -> bytes:
    """Convert mmap slices to ``bytes`` so torchcodec can construct a decoder."""
    return bytes(data)


class Serializer(ABC):
    """The base interface for any serializers.

    A Serializer serialize and deserialize to and from bytes.

    """

    @abstractmethod
    def serialize(self, data: Any) -> tuple[bytes, str | None]:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        pass

    @abstractmethod
    def can_serialize(self, data: Any) -> bool:
        pass

    def setup(self, metadata: Any) -> None:
        pass


class PILSerializer(Serializer):
    """The PILSerializer serialize and deserialize PIL Image to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if isinstance(item, Pil):
            target_mode = item.mode
            if item.image is not None:
                item = item.image
            elif item.array is not None:
                item = _pil_from_array(item.array)
            else:
                if not _PIL_AVAILABLE:
                    raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")
                from PIL import Image as PILImage

                data, _ = _read_media_bytes(item)
                item = PILImage.open(io.BytesIO(data))
            if target_mode and item.mode != target_mode:
                item = item.convert(target_mode)
        mode = item.mode.encode("utf-8")
        width, height = item.size
        raw = item.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32)
        return ints.tobytes() + mode + raw, None

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        if not _PIL_AVAILABLE:
            raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")
        from PIL import Image

        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = bytes(data[idx:idx2]).decode("utf-8")
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore

    def can_serialize(self, item: Any) -> bool:
        if isinstance(item, Pil):
            return True
        if not _PIL_AVAILABLE:
            return False

        from PIL import Image
        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, Image.Image) and not isinstance(item, JpegImageFile)


class JPEGSerializer(Serializer):
    """The JPEGSerializer serialize and deserialize JPEG image to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if not _PIL_AVAILABLE:
            raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")

        from PIL import Image
        from PIL.GifImagePlugin import GifImageFile
        from PIL.JpegImagePlugin import JpegImageFile
        from PIL.PngImagePlugin import PngImageFile
        from PIL.WebPImagePlugin import WebPImageFile

        if isinstance(item, Jpeg):
            if item.array is not None or item.image is not None or item.mode:
                data, _ = _encode_image_ref(item, default_format="JPEG", default_quality=item.quality)
                return data, None
            data, ext = _read_media_bytes(item)
            if item.quality != 95:
                data, _ = _encode_image_ref(item, default_format="JPEG", default_quality=item.quality)
            return data, None

        if isinstance(item, JpegImageFile):
            if not hasattr(item, "filename"):
                raise ValueError(
                    "The JPEG Image's filename isn't defined."
                    "\n HINT: Open the image in your Dataset `__getitem__` method."
                )
            if item.filename and os.path.isfile(item.filename):
                # read the content of the file directly
                with open(item.filename, "rb") as f:
                    return f.read(), None
            else:
                item_bytes = io.BytesIO()
                item.save(item_bytes, format="JPEG")
                item_bytes = item_bytes.getvalue()
                return item_bytes, None

        if isinstance(item, (PngImageFile, WebPImageFile, GifImageFile, Image.Image)):
            buff = io.BytesIO()
            item.convert("RGB").save(buff, quality=100, format="JPEG")
            buff.seek(0)
            return buff.read(), None

        raise TypeError(f"The provided item should be of type `JpegImageFile`. Found {item}.")

    def deserialize(self, data: bytes) -> torch.Tensor:
        return _decode_image_tensor(data)

    def can_serialize(self, item: Any) -> bool:
        if isinstance(item, Jpeg):
            return True
        if not _PIL_AVAILABLE:
            return False
        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, JpegImageFile)


class ImageSerializer(Serializer):
    """Store image bytes from ``Image(path=...)`` / ``Image(bytes=...)``."""

    _EXTENSIONS = ("jpg", "jpeg", "png", "webp", "bmp", "gif")

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if isinstance(item, Image) and (
            item.array is not None or item.image is not None or item.quality is not None or item.mode or item.format
        ):
            data, ext = _encode_image_ref(item, default_format=item.format or "PNG", default_quality=item.quality)
            return data, f"image:{ext}"
        data, ext = _read_media_bytes(item, self._EXTENSIONS)
        return data, f"image:{ext or 'jpg'}"

    def deserialize(self, data: bytes) -> torch.Tensor:
        return _decode_image_tensor(data)

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, Image) or _has_media_extension(item, self._EXTENSIONS)


class JPEGArraySerializer(Serializer):
    """The JPEGArraySerializer serializes and deserializes lists of JPEG images to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        images = item.images if isinstance(item, JpegArray) else item
        if not images:
            raise ValueError("JpegArray.images must be a non-empty list.")
        # Store number of images as first 4 bytes
        n_images_bytes = np.uint32(len(images)).tobytes()

        # create a instance of JPEGSerializer
        if not hasattr(self, "_jpeg_serializer"):
            self._jpeg_serializer = JPEGSerializer()
        # convert each image to bytes and store in a list
        image_bytes = []
        for image in images:
            if isinstance(image, Jpeg):
                if isinstance(item, JpegArray) and image.quality == 95 and item.quality != 95:
                    image = Jpeg(
                        path=image.path,
                        bytes=image.bytes,
                        array=image.array,
                        image=image.image,
                        mode=image.mode,
                        quality=item.quality,
                    )
                image_bytes.append(self._jpeg_serializer.serialize(image)[0])
            elif isinstance(image, Image):
                image_bytes.append(ImageSerializer().serialize(image)[0])
            elif isinstance(image, (bytes, bytearray, str, dict, _MediaRef)):
                data, _ = _read_media_bytes(image)
                image_bytes.append(data)
            else:
                image_bytes.append(self._jpeg_serializer.serialize(image)[0])

        # Store all image sizes as uint32 array and convert to bytes
        image_sizes_bytes = np.array([len(elem) for elem in image_bytes], dtype=np.uint32).tobytes()

        # Concatenate all data: n_images + sizes + image bytes
        return b"".join(chain([n_images_bytes, image_sizes_bytes], image_bytes)), None

    def deserialize(self, data: bytes) -> list[torch.Tensor]:
        if len(data) < 4:
            raise ValueError("Input data is too short to contain valid list of images")

        # Extract number of images from the first 4 bytes
        n_images = np.frombuffer(data[:4], dtype=np.uint32)[0]

        # Ensure the number of images is positive
        if n_images <= 0:
            raise ValueError("Number of images must be positive")

        # Calculate the offset where image bytes start
        image_bytes_offset = 4 + 4 * n_images

        if len(data) < image_bytes_offset:
            raise ValueError("Data is too short for the number of images specified")

        # Extract the sizes of each image
        image_sizes = np.frombuffer(data[4:image_bytes_offset], dtype=np.uint32)

        # Calculate offsets for each image's data
        offsets = np.cumsum(np.concatenate(([image_bytes_offset], image_sizes)))

        if len(offsets) != n_images + 1:
            raise ValueError("Mismatch between number of images and offsets")

        if not hasattr(self, "_jpeg_serializer"):
            self._jpeg_serializer = JPEGSerializer()

        # Extract and decode each image data
        images = []
        for i in range(n_images):
            # Extract the image data using the offsets
            image_data = data[offsets[i] : offsets[i + 1]]
            # Convert the image data to a tensor
            images.append(self._jpeg_serializer.deserialize(image_data))
        return images

    def can_serialize(self, item: Any) -> bool:
        """Check if the item is a list of JPEG images."""
        if isinstance(item, JpegArray):
            return True
        if not _PIL_AVAILABLE:
            return False
        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, (list, tuple)) and all(isinstance(elem, JpegImageFile) for elem in item)


class BytesSerializer(Serializer):
    """The BytesSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: bytes) -> tuple[bytes, str | None]:
        return item, None

    def deserialize(self, item: bytes) -> bytes:
        return item

    def can_serialize(self, item: bytes) -> bool:
        return isinstance(item, bytes)


def _coerce_torch_tensor(item: Any) -> torch.Tensor:
    if isinstance(item, Tensor):
        if item.array is not None:
            tensor = item.array if isinstance(item.array, torch.Tensor) else torch.as_tensor(item.array)
            if item.dtype is not None:
                tensor = tensor.to(item.dtype)
            return tensor
        data, _ = _read_media_bytes(item)
        if item.dtype is None:
            raise TypeError("Tensor(path=...) / Tensor(bytes=...) requires dtype=.")
        return torch.frombuffer(data, dtype=item.dtype)
    if isinstance(item, torch.Tensor):
        return item
    return torch.as_tensor(item)


class TensorSerializer(Serializer):
    """An optimized TensorSerializer that is compatible with deepcopy/pickle."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}
        self._header_struct_format = ">II"
        self._header_struct = struct.Struct(self._header_struct_format)

    def serialize(self, item: torch.Tensor) -> tuple[bytes, str | None]:
        item = _coerce_torch_tensor(item)
        if item.device.type != "cpu":
            item = item.cpu()

        dtype_indice = self._dtype_to_indices[item.dtype]

        numpy_item = item.numpy(force=True)
        rank = len(numpy_item.shape)
        shape_format = f">{rank}I"
        header_bytes = self._header_struct.pack(dtype_indice, rank)
        shape_bytes = struct.pack(shape_format, *numpy_item.shape)
        data_bytes = numpy_item.tobytes()
        return b"".join([header_bytes, shape_bytes, data_bytes]), None

    def deserialize(self, data: bytes) -> torch.Tensor:
        buffer_view = memoryview(data)
        dtype_indice, rank = self._header_struct.unpack_from(buffer_view, 0)
        dtype = _TORCH_DTYPES_MAPPING[dtype_indice]
        header_size = self._header_struct.size
        shape = struct.unpack_from(f">{rank}I", buffer_view, header_size)
        data_start_offset = header_size + (rank * 4)
        if data_start_offset < len(buffer_view):
            tensor_1d = torch.frombuffer(buffer_view[data_start_offset:], dtype=dtype)
            return tensor_1d.reshape(shape)
        return torch.empty(shape, dtype=dtype)

    def can_serialize(self, item: Any) -> bool:
        if isinstance(item, Tensor):
            return item.array is not None and len(item.array.shape) != 1
        return isinstance(item, torch.Tensor) and len(item.shape) != 1

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_header_struct"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._header_struct = struct.Struct(self._header_struct_format)


class NoHeaderTensorSerializer(Serializer):
    """The TensorSerializer serialize and deserialize tensor to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}
        self._dtype: torch.dtype | None = None

    def setup(self, data_format: str) -> None:
        self._dtype = _TORCH_DTYPES_MAPPING[int(data_format.split(":")[1])]

    def serialize(self, item: torch.Tensor) -> tuple[bytes, str | None]:
        item = _coerce_torch_tensor(item)
        dtype_indice = self._dtype_to_indices[item.dtype]
        return item.numpy().tobytes(order="C"), f"no_header_tensor:{dtype_indice}"

    def deserialize(self, data: bytes) -> torch.Tensor:
        assert self._dtype
        return torch.frombuffer(data, dtype=self._dtype) if len(data) > 0 else torch.empty((0,), dtype=self._dtype)

    def can_serialize(self, item: torch.Tensor) -> bool:
        if isinstance(item, Tensor):
            if item.array is not None:
                return len(item.array.shape) == 1
            return (item.path is not None or item.bytes is not None) and item.dtype is not None
        return isinstance(item, torch.Tensor) and len(item.shape) == 1


class NumpySerializer(Serializer):
    """The NumpySerializer serialize and deserialize numpy to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _NUMPY_DTYPES_MAPPING.items()}

    def serialize(self, item: np.ndarray) -> tuple[bytes, str | None]:
        dtype_indice = self._dtype_to_indices[item.dtype]
        data = [np.uint32(dtype_indice).tobytes()]
        data.append(np.uint32(len(item.shape)).tobytes())
        for dim in item.shape:
            data.append(np.uint32(dim).tobytes())
        data.append(item.tobytes(order="C"))
        return b"".join(data), None

    def deserialize(self, data: bytes) -> np.ndarray:
        dtype_indice = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _NUMPY_DTYPES_MAPPING[dtype_indice]
        shape_size = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        # deserialize the shape header
        # Note: The start position of the shape value: 8 (dtype + shape length) + 4 * shape_idx
        for shape_idx in range(shape_size):
            shape.append(np.frombuffer(data[8 + 4 * shape_idx : 8 + 4 * (shape_idx + 1)], np.uint32).item())

        # deserialize the numpy array bytes
        tensor = np.frombuffer(data[8 + 4 * shape_size : len(data)], dtype=dtype).copy()
        if tensor.shape == shape:
            return tensor
        return np.reshape(tensor, shape)

    def can_serialize(self, item: np.ndarray) -> bool:
        return isinstance(item, np.ndarray) and len(item.shape) > 1


class NoHeaderNumpySerializer(Serializer):
    """The NoHeaderNumpySerializer serialize and deserialize numpy to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _NUMPY_DTYPES_MAPPING.items()}
        self._dtype: np.dtype | None = None

    def setup(self, data_format: str) -> None:
        self._dtype = _NUMPY_DTYPES_MAPPING[int(data_format.split(":")[1])]

    def serialize(self, item: np.ndarray) -> tuple[bytes, str | None]:
        dtype_indice: int = self._dtype_to_indices[item.dtype]
        return item.tobytes(order="C"), f"no_header_numpy:{dtype_indice}"

    def deserialize(self, data: bytes) -> np.ndarray:
        assert self._dtype
        return np.frombuffer(data, dtype=self._dtype).copy()

    def can_serialize(self, item: np.ndarray) -> bool:
        return isinstance(item, np.ndarray) and len(item.shape) == 1


_GRAPH_MAGIC = b"LDGR"
_GRAPH_KIND_TENSOR = 0
_GRAPH_KIND_PICKLE = 1
_GRAPH_META_KEY = "_meta"
_GRAPH_TENSOR_MARK = "__ldgr_t__"
_GRAPH_VERSION = 3


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, (Tensor, torch.Tensor, np.ndarray))


def _pyg_cls(item: Any) -> str:
    return "hetero" if type(item).__name__ == "HeteroData" else "data"


def _looks_hetero(mapping: dict[str, Any]) -> bool:
    return any(
        isinstance(value, dict) and any(_is_tensor_like(child) for child in value.values())
        for value in mapping.values()
    )


def _mapping_from_item(item: Any) -> tuple[dict[str, Any], str] | None:
    """PyG ``to_dict()`` (flat ``Data`` or nested ``HeteroData``). Opaque objects return ``None``."""
    if is_pyg_data(item):
        return dict(item.to_dict()), _pyg_cls(item)
    if isinstance(item, Graph):
        mapping = item.to_mapping()
        if not mapping:
            return None
        cls = _pyg_cls(item.data) if is_pyg_data(item.data) else "hetero" if _looks_hetero(mapping) else "data"
        return mapping, cls
    raise TypeError(f"Cannot pack graph from {type(item)}")


def _flatten_graph(obj: Any, prefix: tuple[Any, ...] = ()) -> tuple[dict[tuple[Any, ...], Any], Any] | None:
    if _is_tensor_like(obj):
        return {prefix: obj}, _GRAPH_TENSOR_MARK
    if isinstance(obj, dict):
        tensors: dict[tuple[Any, ...], Any] = {}
        tree: dict[Any, Any] = {}
        for key, value in obj.items():
            nested = _flatten_graph(value, prefix + (key,))
            if nested is None:
                return None
            child_tensors, child_tree = nested
            tensors.update(child_tensors)
            tree[key] = child_tree
        return tensors, tree
    if isinstance(obj, (list, tuple, set)):
        return None
    return {}, obj


def _path_name(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in path:
        if isinstance(key, tuple):
            parts.append("__".join(str(part) for part in key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def _fill_graph_tree(tree: Any, tensors: dict[tuple[Any, ...], Any], prefix: tuple[Any, ...] = ()) -> Any:
    if tree == _GRAPH_TENSOR_MARK:
        return tensors[prefix]
    if isinstance(tree, dict):
        return {key: _fill_graph_tree(value, tensors, prefix + (key,)) for key, value in tree.items()}
    return tree


def _reconstruct_graph(values: dict[str, Any]) -> Any:
    """Re-read with ``Data.from_dict`` / ``HeteroData.from_dict`` when PyG is installed."""
    tree = values.pop("_tree", None)
    paths = values.pop("_paths", None)
    fields = values.pop("_fields", None)
    cls = values.pop("_cls", "data")
    if tree is not None and paths is not None and fields is not None:
        by_path = {path: values[name] for path, name in zip(paths, fields)}
        values = _fill_graph_tree(tree, by_path)

    try:
        if cls == "hetero":
            from torch_geometric.data import HeteroData

            return HeteroData.from_dict(values)
        from torch_geometric.data import Data

        return Data.from_dict(values)
    except ImportError:
        if cls == "hetero":
            return Graph(data=values)
        known = {name: values.pop(name) for name in list(values) if name in GRAPH_FIELDS}
        leftover = values or None
        return Graph(**known, data=leftover)


class GraphSerializer(Serializer):
    """Store a PyG ``Data.to_dict()`` as packed tensors (not ``torch.save``).

    Reconstructs with ``Data.from_dict`` / ``HeteroData.from_dict`` when
    torch-geometric is installed. ``graph:pickle`` is only for NetworkX / opaque ``data=``.
    """

    def __init__(self) -> None:
        self._tensor = TensorSerializer()
        self._pickle_fallback = False

    def setup(self, metadata: Any) -> None:
        self._pickle_fallback = isinstance(metadata, str) and metadata.endswith("pickle")

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        packed = _mapping_from_item(item)
        if packed is None:
            payload = item.data if isinstance(item, Graph) and item.data is not None else item
            return pickle.dumps(payload), "graph:pickle"

        mapping, cls = packed
        flattened = _flatten_graph(mapping)
        if flattened is None or not flattened[0]:
            payload = item.data if isinstance(item, Graph) and item.data is not None else item
            return pickle.dumps(payload), "graph:pickle"

        tensor_map, tree = flattened
        paths = list(tensor_map.keys())
        names = [_path_name(path) for path in paths]
        if len(set(names)) != len(names):
            names = [f"t{index}" for index in range(len(paths))]
        meta = {"_cls": cls, "_tree": tree, "_paths": paths, "_fields": names}

        parts = [_GRAPH_MAGIC, struct.pack(">B", _GRAPH_VERSION)]
        fields: list[tuple[bytes, int, bytes]] = []
        for name, path in zip(names, paths):
            tensor = _coerce_torch_tensor(tensor_map[path]).cpu().contiguous()
            payload, _ = self._tensor.serialize(tensor)
            fields.append((name.encode("utf-8"), _GRAPH_KIND_TENSOR, payload))
        fields.append((_GRAPH_META_KEY.encode("utf-8"), _GRAPH_KIND_PICKLE, pickle.dumps(meta)))
        if len(fields) > 65535:
            raise ValueError("graph pack exceeds 65535 fields")
        parts.append(struct.pack(">H", len(fields)))
        for name_b, kind, payload in fields:
            if len(name_b) > 65535:
                raise ValueError("graph field name exceeds 65535 bytes")
            parts.append(struct.pack(">H", len(name_b)))
            parts.append(name_b)
            parts.append(struct.pack(">B", kind))
            parts.append(struct.pack(">I", len(payload)))
            parts.append(payload)
        return b"".join(parts), "graph"

    def deserialize(self, data: bytes) -> Any:
        view = data if isinstance(data, memoryview) else memoryview(data)
        if self._pickle_fallback or bytes(view[: len(_GRAPH_MAGIC)]) != _GRAPH_MAGIC:
            loaded = pickle.loads(bytes(view))  # noqa: S301
            return loaded if isinstance(loaded, Graph) else Graph(data=loaded)
        offset = len(_GRAPH_MAGIC)
        version = view[offset]
        offset += 1
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported graph pack version: {version}")
        if version >= 3:
            n_fields = struct.unpack_from(">H", view, offset)[0]
            offset += 2
        else:
            n_fields = view[offset]
            offset += 1
        values: dict[str, Any] = {}
        for _ in range(n_fields):
            if version >= 3:
                name_len = struct.unpack_from(">H", view, offset)[0]
                offset += 2
            else:
                name_len = view[offset]
                offset += 1
            name = bytes(view[offset : offset + name_len]).decode("utf-8")
            offset += name_len
            if version >= 2:
                kind = view[offset]
                offset += 1
            else:
                kind = _GRAPH_KIND_TENSOR
            payload_len = struct.unpack_from(">I", view, offset)[0]
            offset += 4
            payload = bytes(view[offset : offset + payload_len])
            offset += payload_len
            if kind == _GRAPH_KIND_PICKLE or name == _GRAPH_META_KEY:
                values.update(pickle.loads(payload))  # noqa: S301
            else:
                values[name] = self._tensor.deserialize(payload)
        return _reconstruct_graph(values)

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, Graph) or is_pyg_data(item)


class PickleSerializer(Serializer):
    """The PickleSerializer serialize and deserialize python objects to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        return pickle.dumps(item), None

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)  # noqa: S301

    def can_serialize(self, _: Any) -> bool:
        return True


class FileSerializer(Serializer):
    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if isinstance(item, (File, _MediaRef)):
            data, ext = _read_media_bytes(item)
            return data, f"file:{ext}" if ext else "file"
        filepath = item
        _, file_extension = os.path.splitext(filepath)
        with open(filepath, "rb") as f:
            file_extension = file_extension.replace(".", "").lower()
            return f.read(), f"file:{file_extension}"

    def deserialize(self, data: bytes) -> Any:
        return bytes(data)

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, File)


def _extension(path: str | None, default: str = "") -> str:
    if not path:
        return default
    return os.path.splitext(str(path).split("?", 1)[0])[1].replace(".", "").lower()


def _media_path_bytes(item: Any) -> tuple[str | None, bytes | None]:
    if isinstance(item, _MediaRef):
        return item.path, item.bytes
    if isinstance(item, dict):
        payload = item.get("bytes")
        return item.get("path"), bytes(payload) if payload is not None else None
    return None, None


def _read_media_bytes(item: Any, _extensions: tuple[str, ...] = ()) -> tuple[bytes, str]:
    """Normalize a typed ref, path, bytes, or ``{path, bytes}`` into ``(bytes, extension)``."""
    path, payload = _media_path_bytes(item)
    if payload is not None:
        return payload, _extension(path)
    if isinstance(item, _MediaRef) and item.path is None and item.bytes is None:
        extras = [name for name in ("array", "image", "mesh", "pdf", "text") if getattr(item, name, None) is not None]
        if extras:
            raise TypeError(
                f"{type(item).__name__} has {extras} but no path/bytes; the serializer should encode those first."
            )
        raise TypeError(f"{type(item).__name__} needs path=, bytes=, or a native payload (array=/image=/...).")
    if path:
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(), _extension(path)
        raise FileNotFoundError(f"Media path does not exist: {path}")
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), ""
    if isinstance(item, os.PathLike):
        item = os.fspath(item)
    if isinstance(item, str) and os.path.isfile(item):
        with open(item, "rb") as handle:
            return handle.read(), _extension(item)
    raise TypeError(f"Unsupported media input: {type(item)}")


_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"
_VALID_IMAGE_ARRAY_DTYPES = (
    np.dtype("|b1"),
    np.dtype("|u1"),
    np.dtype("<u2"),
    np.dtype(">u2"),
    np.dtype("<i2"),
    np.dtype(">i2"),
    np.dtype("<u4"),
    np.dtype(">u4"),
    np.dtype("<i4"),
    np.dtype(">i4"),
    np.dtype("<f4"),
    np.dtype(">f4"),
    np.dtype("<f8"),
    np.dtype(">f8"),
)


def _as_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _image_array_for_pil(array: Any) -> np.ndarray:
    """Downcast arrays so Pillow can save them (integers to uint8, float ``[0, 1]`` × 255)."""
    array = _as_numpy(array)
    dtype = array.dtype
    byteorder = dtype.byteorder if dtype.byteorder != "=" else _NATIVE_BYTEORDER
    if array.ndim >= 3:
        dest = np.dtype("|u1")
        if np.issubdtype(dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        elif dtype != dest:
            warnings.warn(f"Downcasting array dtype {dtype} to {dest} for image encoding.", stacklevel=2)
        return array.astype(dest)
    if dtype in _VALID_IMAGE_ARRAY_DTYPES:
        return array
    itemsize = dtype.itemsize
    while itemsize >= 1:
        dest = np.dtype(f"{byteorder}{dtype.kind}{itemsize}")
        if dest in _VALID_IMAGE_ARRAY_DTYPES:
            warnings.warn(f"Downcasting array dtype {dtype} to {dest} for image encoding.", stacklevel=2)
            return array.astype(dest)
        itemsize //= 2
    raise TypeError(f"Cannot downcast dtype {dtype} to a Pillow-compatible image dtype.")


def _pil_from_array(array: Any) -> Any:
    if not _PIL_AVAILABLE:
        raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")
    from PIL import Image as PILImage

    return PILImage.fromarray(_image_array_for_pil(array))


def _native_pil_format(image: Any) -> str:
    fmt = getattr(image, "format", None)
    if fmt:
        return str(fmt)
    return "PNG" if image.mode in {"1", "L", "LA", "RGB", "RGBA"} else "TIFF"


def _save_pil(image: Any, fmt: str, quality: int | None = None, mode: str | None = None) -> tuple[bytes, str]:
    if mode and image.mode != mode:
        image = image.convert(mode)
    fmt = "JPEG" if fmt.upper() in {"JPG", "JPEG"} else fmt.upper()
    if fmt == "JPEG" and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"format": fmt}
    if quality is not None and fmt in {"JPEG", "WEBP"}:
        kwargs["quality"] = int(quality)
    image.save(buffer, **kwargs)
    ext = "jpg" if fmt == "JPEG" else fmt.lower()
    return buffer.getvalue(), ext


def _jpeg_has_exif_app1(data: bytes) -> bool:
    """Cheap JPEG scan: EXIF lives in an APP1 marker. Avoid PIL on the ImageNet hot path."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return False
    idx = 2
    while idx + 4 <= len(data):
        if data[idx] != 0xFF:
            return False
        marker = data[idx + 1]
        if marker == 0xDA:  # SOS — image data
            return False
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            idx += 2
            continue
        if idx + 4 > len(data):
            return False
        length = int.from_bytes(data[idx + 2 : idx + 4], "big")
        if marker == 0xE1 and data[idx + 4 : idx + 8] == b"Exif":
            return True
        idx += 2 + length
    return False


def _decode_image_tensor(data: bytes) -> torch.Tensor:
    """Decode image bytes to an RGB CHW tensor. Apply EXIF orientation when a JPEG APP1 Exif marker is present."""
    if _PIL_AVAILABLE and _jpeg_has_exif_app1(data):
        from PIL import Image as PILImage
        from PIL import ImageOps

        image = PILImage.open(io.BytesIO(data))
        image.load()
        image = ImageOps.exif_transpose(image)
        return torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).contiguous()

    from torchvision.io import ImageReadMode, decode_image, decode_jpeg

    array = torch.frombuffer(data, dtype=torch.uint8)
    with suppress(RuntimeError):
        return decode_jpeg(array, mode=ImageReadMode.RGB)
    return decode_image(array, mode=ImageReadMode.RGB)


class _LitAudioDecoder:
    """Wrap torchcodec ``AudioDecoder`` so ``audio["array"]`` / ``audio["sampling_rate"]`` work.

    Assigning ``__getitem__`` on the native decoder is ignored (C extension).
    """

    def __init__(self, decoder: Any) -> None:
        self._decoder = decoder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decoder, name)

    def __getitem__(self, key: str) -> Any:
        if key == "array":
            samples = self._decoder.get_all_samples().data.cpu().numpy()
            if samples.ndim == 2 and samples.shape[0] == 1:
                return samples[0]
            return samples
        if key == "sampling_rate":
            return self._decoder.get_samples_played_in_range(0, 0).sample_rate
        raise KeyError(key)


def _unwrap_audio_decoder(item: Any) -> Any:
    return item._decoder if isinstance(item, _LitAudioDecoder) else item


def _attach_audio_hf_api(decoder: Any) -> Any:
    if isinstance(decoder, _LitAudioDecoder):
        return decoder
    return _LitAudioDecoder(decoder)


def _nifti_encoded_bytes(image: Any) -> tuple[bytes, str]:
    file_map = getattr(image, "file_map", None)
    if file_map is not None and "image" in file_map:
        filename = getattr(file_map["image"], "filename", None)
        if filename and os.path.isfile(filename):
            with open(filename, "rb") as handle:
                ext = "nii.gz" if str(filename).lower().endswith(".nii.gz") else "nii"
                return handle.read(), f"nifti:{ext}"
    return image.to_bytes(), "nifti:nii"


def _encode_image_ref(item: Any, default_format: str = "PNG", default_quality: int | None = None) -> tuple[bytes, str]:
    """Encode ``Image`` / ``Jpeg`` / ``Pil`` from path, bytes, array, or PIL image."""
    array = getattr(item, "array", None)
    image = getattr(item, "image", None)
    mode = getattr(item, "mode", None)
    quality = getattr(item, "quality", default_quality)
    explicit_format = getattr(item, "format", None)
    if image is not None or array is not None:
        pil = image if image is not None else _pil_from_array(array)
        fmt = explicit_format or (_native_pil_format(pil) if image is not None else default_format)
        return _save_pil(pil, fmt, quality=quality, mode=mode)
    data, ext = _read_media_bytes(item)
    if quality is None and mode is None and getattr(item, "format", None) is None:
        return data, ext or default_format.lower()
    if not _PIL_AVAILABLE:
        raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")
    from PIL import Image as PILImage

    pil = PILImage.open(io.BytesIO(data))
    return _save_pil(pil, explicit_format or ext or default_format, quality=quality, mode=mode)


def _encode_video_array(array: Any, fps: float) -> bytes:
    frames = torch.as_tensor(_as_numpy(array))
    if frames.ndim != 4:
        raise ValueError("Video(array=...) must be (N, H, W, C) or (N, C, H, W).")
    if frames.shape[1] in {1, 3} and frames.shape[-1] not in {1, 3}:
        frames = frames.permute(0, 2, 3, 1)
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    with tempfile.TemporaryDirectory() as dirname:
        path = os.path.join(dirname, "clip.mp4")
        write_video = None
        with suppress(Exception):
            from torchvision.io import write_video as _write_video

            write_video = _write_video
        if write_video is not None:
            write_video(path, frames, fps=float(fps))
        elif _torchcodec_usable():
            try:
                from torchcodec.encoders import VideoEncoder

                VideoEncoder(frames.permute(0, 3, 1, 2), frame_rate=float(fps)).to_file(path)
            except Exception as exc:
                raise TypeError(
                    "Encoding Video(array=...) requires torchvision.io.write_video or torchcodec. "
                    "Pass Video(path=...) or Video(bytes=...)."
                ) from exc
        else:
            raise TypeError(
                "Encoding Video(array=...) requires torchvision.io.write_video or torchcodec. "
                "Pass Video(path=...) or Video(bytes=...)."
            )
        with open(path, "rb") as handle:
            return handle.read()


def _has_media_extension(item: Any, extensions: tuple[str, ...]) -> bool:
    if isinstance(item, _MediaRef):
        return bool(item.path) and _extension(item.path) in extensions
    if isinstance(item, dict):
        path = item.get("path")
        return bool(path) and _extension(path) in extensions
    if isinstance(item, os.PathLike):
        item = os.fspath(item)
    return isinstance(item, str) and os.path.isfile(item) and _extension(item) in extensions


def _safe_decode_device(device: str) -> str:
    """CUDA decode is not safe in DataLoader or optimize workers. Force CPU there."""
    requested = "cpu" if device is None else str(device)
    if requested == "cpu" or requested.startswith("cpu:"):
        return "cpu"
    try:
        from torch.utils.data import get_worker_info

        in_dataloader_worker = get_worker_info() is not None
    except Exception:
        in_dataloader_worker = False
    in_optimize_worker = os.getenv("DATA_OPTIMIZER_GLOBAL_RANK") is not None
    if in_dataloader_worker or in_optimize_worker:
        return "cpu"
    return requested


class VideoSerializer(Serializer):
    """Store video bytes. Decode with torchcodec (lazy ``VideoDecoder`` by default).

    ``decode="decoder"`` (default) returns a torchcodec ``VideoDecoder`` so callers
    can ``get_frames_in_range`` / ``get_frames_at`` without materializing the clip.
    ``decode="all"`` still returns ``(frames, audio, metadata)`` for older code.

    ``device="cuda"`` is ignored inside DataLoader / optimize workers (CPU decode).
    Move frames to GPU after the batch is collated on the main process.
    """

    _EXTENSIONS = ("mp4", "ogv", "mjpeg", "avi", "mov", "h264", "mpg", "mpeg", "webm", "wmv", "mkv")

    def __init__(
        self,
        decode: str = "decoder",
        seek_mode: str = "approximate",
        device: str = "cpu",
        num_ffmpeg_threads: int = 1,
        dimension_order: str = "NCHW",
        stream_index: int | None = None,
    ) -> None:
        if decode not in {"all", "decoder", "bytes"}:
            raise ValueError("VideoSerializer decode must be 'all', 'decoder', or 'bytes'.")
        if seek_mode not in {"exact", "approximate"}:
            raise ValueError("seek_mode must be 'exact' or 'approximate'.")
        self.decode = decode
        self.seek_mode = seek_mode
        self.device = device
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.dimension_order = dimension_order
        self.stream_index = stream_index

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if _torchcodec_usable():
            from torchcodec.decoders import VideoDecoder

            if isinstance(item, VideoDecoder):
                encoded = getattr(item, "_hf_encoded", None) or getattr(item, "_litdata_encoded", None)
                if encoded and encoded.get("bytes") is not None:
                    return bytes(encoded["bytes"]), f"video:{_extension(encoded.get('path')) or 'mp4'}"
                if encoded and encoded.get("path") and os.path.isfile(encoded["path"]):
                    data, ext = _read_media_bytes(encoded["path"])
                    return data, f"video:{ext or 'mp4'}"
                raise TypeError(
                    "Encoding a VideoDecoder that was not produced by LitData/HF decode is not supported. "
                    "Pass Video(path=...) or Video(bytes=...) instead."
                )
        if isinstance(item, Video) and item.array is not None:
            return _encode_video_array(item.array, item.fps), "video:mp4"
        data, ext = _read_media_bytes(item, self._EXTENSIONS)
        return data, f"video:{ext or 'mp4'}"

    def _video_decoder(self, data: bytes, dimension_order: str | None = None) -> Any:
        if not _torchcodec_usable():
            raise ModuleNotFoundError("torchcodec is required. Run `pip install torchcodec`")
        from torchcodec.decoders import VideoDecoder

        kwargs: dict[str, Any] = {
            "dimension_order": dimension_order or self.dimension_order,
            "num_ffmpeg_threads": self.num_ffmpeg_threads,
            "device": _safe_decode_device(self.device),
            "seek_mode": self.seek_mode,
        }
        if self.stream_index is not None:
            kwargs["stream_index"] = self.stream_index
        data = _as_bytes(data)
        decoder = VideoDecoder(data, **kwargs)
        decoder._litdata_encoded = {"path": None, "bytes": data}
        if getattr(decoder, "metadata", None) is not None:
            decoder.metadata.path = None
        return decoder

    def deserialize(self, data: bytes) -> Any:
        if self.decode == "bytes":
            return data
        if self.decode == "decoder":
            return self._video_decoder(data)
        if _torchcodec_usable():
            return self._deserialize_with_torchcodec(data)
        if _torchvision_read_video_available():
            return self._deserialize_with_torchvision_io(data)
        raise ModuleNotFoundError("torchcodec is required. Run `pip install torchcodec`")

    def _deserialize_with_torchvision_io(self, data: bytes) -> Any:
        if not _AV_AVAILABLE:
            raise ModuleNotFoundError("av is required. Run `pip install av`")

        import torchvision.io

        read_video = getattr(torchvision.io, "read_video", None)
        if read_video is None:
            raise ModuleNotFoundError("torchcodec is required. Run `pip install torchcodec`")

        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, "file.mp4")
            with open(fname, "wb") as stream:
                stream.write(_as_bytes(data))
            return read_video(fname, pts_unit="sec")

    def _deserialize_with_torchcodec(self, data: bytes) -> Any:
        import torch
        from torchcodec.decoders import AudioDecoder

        # NHWC matches the historical torchvision.io.read_video layout.
        dec = self._video_decoder(data, dimension_order="NHWC")
        metadata = asdict(dec.metadata) if dec.metadata is not None else {}
        video = dec.get_all_frames().data
        try:
            audio = AudioDecoder(_as_bytes(data)).get_all_samples().data
        except ValueError:
            audio = torch.zeros(1, 0)
        return video, audio, metadata

    def can_serialize(self, data: Any) -> bool:
        if isinstance(data, Video):
            return True
        if _torchcodec_usable():
            from torchcodec.decoders import VideoDecoder

            if isinstance(data, VideoDecoder):
                return True
        return _has_media_extension(data, self._EXTENSIONS)


class AudioSerializer(Serializer):
    """Store audio bytes. Decode with torchcodec ``AudioDecoder`` by default."""

    _EXTENSIONS = ("wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "wma", "pcm")

    def __init__(
        self,
        decode: str = "decoder",
        sampling_rate: int | None = None,
        num_channels: int | None = None,
        stream_index: int | None = None,
    ) -> None:
        if decode not in {"decoder", "bytes", "samples"}:
            raise ValueError("AudioSerializer decode must be 'decoder', 'bytes', or 'samples'.")
        self.decode = decode
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.stream_index = stream_index

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if _torchcodec_usable():
            from torchcodec.decoders import AudioDecoder

            item = _unwrap_audio_decoder(item)
            if isinstance(item, AudioDecoder):
                return self._serialize_decoder(item), "audio:wav"

        array, rate = None, None
        num_channels = self.num_channels
        if isinstance(item, Audio):
            array, rate = item.array, item.sampling_rate
            if item.num_channels is not None:
                num_channels = item.num_channels
            if item.sampling_rate is not None:
                rate = item.sampling_rate
        elif isinstance(item, dict) and item.get("array") is not None:
            array, rate = item["array"], item.get("sampling_rate")

        if array is not None:
            if rate is None:
                raise KeyError("Audio waveform encode requires sampling_rate.")
            return self._serialize_array(array, int(rate), num_channels=num_channels), "audio:wav"

        path, payload = _media_path_bytes(item)
        if path and str(path).lower().endswith(".pcm"):
            pcm_rate = rate or self.sampling_rate
            return self._serialize_pcm(path, payload, pcm_rate, num_channels=num_channels), "audio:wav"

        data, ext = _read_media_bytes(item, self._EXTENSIONS)
        return data, f"audio:{ext or 'wav'}"

    def _serialize_decoder(self, audio: Any) -> bytes:
        encoded = getattr(audio, "_hf_encoded", None) or getattr(audio, "_litdata_encoded", None)
        if encoded and encoded.get("bytes") is not None:
            return bytes(encoded["bytes"])
        if encoded and encoded.get("path") and os.path.isfile(encoded["path"]):
            data, _ = _read_media_bytes(encoded["path"])
            return data
        samples = audio.get_all_samples()
        channels = self.num_channels
        if channels is None and getattr(samples.data, "ndim", 0) > 0:
            channels = int(samples.data.shape[0])
        return self._serialize_array(samples.data.cpu().numpy(), int(samples.sample_rate), num_channels=channels)

    def _serialize_pcm(
        self, path: str, payload: bytes | None, sampling_rate: int | None, num_channels: int | None = None
    ) -> bytes:
        if sampling_rate is None:
            raise KeyError("To encode PCM, set Audio(sampling_rate=...) or AudioSerializer(sampling_rate=...).")
        if payload is not None:
            waveform = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            waveform = np.memmap(path, dtype="h", mode="r").astype(np.float32) / 32767.0
        return self._serialize_array(waveform, int(sampling_rate), num_channels=num_channels)

    def _serialize_array(self, array: Any, rate: int, num_channels: int | None = None) -> bytes:
        array = np.asarray(array)
        channels = self.num_channels if num_channels is None else num_channels
        if _torchcodec_usable():
            import torch
            from torchcodec.encoders import AudioEncoder

            tensor = array if isinstance(array, torch.Tensor) else torch.from_numpy(array.astype(np.float32))
            buffer = io.BytesIO()
            AudioEncoder(tensor, sample_rate=rate).to_file_like(buffer, format="wav", num_channels=channels)
            return buffer.getvalue()
        import wave

        pcm = np.clip(np.asarray(array, dtype=np.float32), -1.0, 1.0)
        if pcm.ndim == 1:
            pcm = pcm[None, :]
        interleaved = (pcm.T.reshape(-1) * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(int(pcm.shape[0]))
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(interleaved.tobytes())
        return buffer.getvalue()

    def _audio_decoder(self, data: bytes) -> Any:
        if not _torchcodec_usable():
            raise ModuleNotFoundError("torchcodec is required. Run `pip install torchcodec`")
        from torchcodec.decoders import AudioDecoder

        kwargs: dict[str, Any] = {}
        if self.stream_index is not None:
            kwargs["stream_index"] = self.stream_index
        if self.sampling_rate is not None:
            kwargs["sample_rate"] = self.sampling_rate
        if self.num_channels is not None:
            kwargs["num_channels"] = self.num_channels
        data = _as_bytes(data)
        decoder = AudioDecoder(data, **kwargs)
        decoder._litdata_encoded = {"path": None, "bytes": data}
        if getattr(decoder, "metadata", None) is not None:
            decoder.metadata.path = None
        return _attach_audio_hf_api(decoder)

    def deserialize(self, data: bytes) -> Any:
        if self.decode == "bytes":
            return data
        decoder = self._audio_decoder(data)
        if self.decode == "samples":
            samples = decoder.get_all_samples()
            return samples.data, int(samples.sample_rate)
        return decoder

    def can_serialize(self, data: Any) -> bool:
        if isinstance(data, Audio):
            return True
        if _torchcodec_usable():
            from torchcodec.decoders import AudioDecoder

            if isinstance(_unwrap_audio_decoder(data), AudioDecoder):
                return True
        if isinstance(data, dict) and data.get("array") is not None and data.get("sampling_rate") is not None:
            return True
        return _has_media_extension(data, self._EXTENSIONS)


class NiftiSerializer(Serializer):
    """Store NIfTI bytes (``.nii`` / ``.nii.gz``). Decode with nibabel when installed."""

    def __init__(self, decode: bool = True) -> None:
        self.decode = decode

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if _NIBABEL_AVAILABLE:
            import nibabel as nib

            if isinstance(item, nib.spatialimages.SpatialImage):
                return _nifti_encoded_bytes(item)

        if isinstance(item, Nifti):
            if item.image is not None:
                return _nifti_encoded_bytes(item.image)
            if item.array is not None:
                if not _NIBABEL_AVAILABLE:
                    raise ModuleNotFoundError("nibabel is required. Run `pip install nibabel`")
                import nibabel as nib

                affine = item.affine if item.affine is not None else np.eye(4)
                return nib.Nifti1Image(_as_numpy(item.array), affine).to_bytes(), "nifti:nii"
            data, ext = _read_media_bytes(item)
            return data, f"nifti:{ext or 'nii'}"

        path = item.get("path") if isinstance(item, dict) else item
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        ext = "nii.gz" if isinstance(path, str) and path.lower().endswith(".nii.gz") else "nii"
        data, _ = _read_media_bytes(item)
        return data, f"nifti:{ext}"

    def deserialize(self, data: bytes) -> Any:
        if not self.decode:
            return data
        if not _NIBABEL_AVAILABLE:
            raise ModuleNotFoundError("nibabel is required. Run `pip install nibabel`")
        import gzip

        import nibabel as nib

        payload = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        return nib.Nifti1Image.from_bytes(payload)

    def can_serialize(self, data: Any) -> bool:
        if _NIBABEL_AVAILABLE:
            import nibabel as nib

            if isinstance(data, nib.spatialimages.SpatialImage):
                return True
        if isinstance(data, Nifti):
            return True
        path = data.get("path") if isinstance(data, dict) else data
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        if not isinstance(path, str):
            return False
        lower = path.lower().split("?", 1)[0]
        return (lower.endswith(".nii") or lower.endswith(".nii.gz")) and (
            isinstance(data, dict) or os.path.isfile(path)
        )


class MeshSerializer(Serializer):
    """Store mesh bytes (``.glb``, ``.ply``, ``.stl``). Decode with trimesh when installed."""

    _EXTENSIONS = ("glb", "ply", "stl")

    def __init__(self, decode: bool = True) -> None:
        self.decode = decode

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        mesh_obj = item.mesh if isinstance(item, Mesh) else item
        file_type = item.file_type if isinstance(item, Mesh) else "glb"
        if _TRIMESH_AVAILABLE:
            import trimesh

            if isinstance(mesh_obj, (trimesh.Trimesh, trimesh.Scene)):
                item = mesh_obj
            if isinstance(item, (trimesh.Trimesh, trimesh.Scene)):
                metadata = getattr(item, "metadata", None) or {}
                path = metadata.get("file_path") or metadata.get("file_name") if isinstance(metadata, dict) else None
                if path and os.path.isfile(path):
                    data, ext = _read_media_bytes(path, self._EXTENSIONS)
                    return data, f"mesh:{ext or 'glb'}"
                exported = item.export(file_type=file_type)
                payload = exported if isinstance(exported, (bytes, bytearray)) else bytes(exported)
                return bytes(payload), f"mesh:{file_type}"

        data, ext = _read_media_bytes(item, self._EXTENSIONS)
        if ext not in self._EXTENSIONS:
            raise ValueError("Mesh path must end with .glb, .ply, or .stl.")
        return data, f"mesh:{ext}"

    def deserialize(self, data: bytes) -> Any:
        if not self.decode:
            return data
        if not _TRIMESH_AVAILABLE:
            raise ModuleNotFoundError("trimesh is required. Run `pip install trimesh`")
        import trimesh

        ext = getattr(self, "_file_type", "glb")
        return trimesh.load(io.BytesIO(data), file_type=ext)

    def setup(self, metadata: Any) -> None:
        if isinstance(metadata, str) and metadata.startswith("mesh:"):
            self._file_type = metadata.split(":", 1)[1] or "glb"

    def can_serialize(self, data: Any) -> bool:
        if _TRIMESH_AVAILABLE:
            import trimesh

            if isinstance(data, (trimesh.Trimesh, trimesh.Scene)):
                return True
        if isinstance(data, Mesh):
            return True
        return _has_media_extension(data, self._EXTENSIONS)


class PDFSerializer(Serializer):
    """Store PDF bytes. Decode with pdfplumber when installed."""

    _EXTENSIONS = ("pdf",)

    def __init__(self, decode: bool = True) -> None:
        self.decode = decode

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if _PDFPLUMBER_AVAILABLE:
            import pdfplumber

            if isinstance(item, pdfplumber.pdf.PDF):
                stream = getattr(item, "stream", None)
                name = getattr(stream, "name", None) if stream is not None else None
                if name and os.path.isfile(name):
                    with open(name, "rb") as handle:
                        return handle.read(), "pdf"
                if stream is None:
                    raise TypeError("PDF object has no readable stream.")
                stream.seek(0)
                return stream.read(), "pdf"

        if isinstance(item, Pdf) and item.pdf is not None:
            item = item.pdf
            if _PDFPLUMBER_AVAILABLE:
                import pdfplumber

                if isinstance(item, pdfplumber.pdf.PDF):
                    stream = getattr(item, "stream", None)
                    name = getattr(stream, "name", None) if stream is not None else None
                    if name and os.path.isfile(name):
                        with open(name, "rb") as handle:
                            return handle.read(), "pdf"
                    if stream is None:
                        raise TypeError("PDF object has no readable stream.")
                    stream.seek(0)
                    return stream.read(), "pdf"
        if isinstance(item, (bytes, bytearray)) and bytes(item[:4]) == b"%PDF":
            return bytes(item), "pdf"
        data, _ext = _read_media_bytes(item, self._EXTENSIONS)
        return data, "pdf"

    def deserialize(self, data: bytes) -> Any:
        if not self.decode:
            return data
        if not _PDFPLUMBER_AVAILABLE:
            raise ModuleNotFoundError("pdfplumber is required. Run `pip install pdfplumber`")
        import pdfplumber

        return pdfplumber.open(io.BytesIO(data))

    def can_serialize(self, data: Any) -> bool:
        if _PDFPLUMBER_AVAILABLE:
            import pdfplumber

            if isinstance(data, pdfplumber.pdf.PDF):
                return True
        if isinstance(data, Pdf):
            return True
        if isinstance(data, (bytes, bytearray)) and bytes(data[:4]) == b"%PDF":
            return True
        return _has_media_extension(data, self._EXTENSIONS)


class StringSerializer(Serializer):
    def serialize(self, obj: str) -> tuple[bytes, str | None]:
        return obj.encode("utf-8"), None

    def deserialize(self, data: bytes) -> str:
        return bytes(data).decode("utf-8")

    def can_serialize(self, data: str) -> bool:
        return isinstance(data, str) and not os.path.isfile(data)


class TextSerializer(Serializer):
    """Store UTF-8 text. ``path=`` / ``bytes=`` are stored as-is; ``text=`` encodes."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if isinstance(item, Text):
            if item.text is not None:
                return item.text.encode("utf-8"), "text"
            if item.path is None and item.bytes is None:
                raise TypeError("Text needs path=, bytes=, or text=.")
        data, _ext = _read_media_bytes(item)
        return data, "text"

    def deserialize(self, data: bytes) -> str:
        return bytes(data).decode("utf-8")

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, Text)


class NumericSerializer:
    """Store scalar."""

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype
        self.size = self.dtype().nbytes
        # Prefer ``struct`` on the hot deserialize path — it avoids a numpy array allocation
        # for every scalar leaf. Store the format string (not a ``struct.Struct``) so the
        # serializer stays deepcopy/pickle friendly for DataLoader workers.
        self._struct_fmt: str | None
        if dtype is np.int64:
            self._struct_fmt = "<q"
        elif dtype is np.float64:
            self._struct_fmt = "<d"
        else:
            self._struct_fmt = None

    def serialize(self, obj: Any) -> tuple[bytes, str | None]:
        if self._struct_fmt is not None:
            return struct.pack(self._struct_fmt, obj), None
        return self.dtype(obj).tobytes(), None

    def deserialize(self, data: bytes) -> Any:
        if self._struct_fmt is not None:
            return struct.unpack(self._struct_fmt, data)[0]
        return np.frombuffer(data, self.dtype)[0]


class IntegerSerializer(NumericSerializer, Serializer):
    def __init__(self) -> None:
        super().__init__(np.int64)

    def can_serialize(self, data: int) -> bool:
        return isinstance(data, int)


class FloatSerializer(NumericSerializer, Serializer):
    def __init__(self) -> None:
        super().__init__(np.float64)

    def can_serialize(self, data: float) -> bool:
        return isinstance(data, float)


class BooleanSerializer(Serializer):
    """The BooleanSerializer serializes and deserializes boolean values to and from bytes."""

    size = 1

    def serialize(self, item: bool) -> tuple[bytes, str | None]:
        """Serialize a boolean value to bytes.

        Args:
            item: Boolean value to serialize

        Returns:
            Tuple containing the serialized bytes and None for the format string
        """
        return (b"\x01" if item else b"\x00"), None

    def deserialize(self, data: bytes) -> bool:
        """Deserialize bytes back into a boolean value.

        Args:
            data: Bytes to deserialize

        Returns:
            The deserialized boolean value
        """
        return data[0] != 0

    def can_serialize(self, item: Any) -> bool:
        """Check if the item can be serialized by this serializer.

        Args:
            item: Item to check

        Returns:
            True if the item is a boolean, False otherwise
        """
        return isinstance(item, bool)


class TIFFSerializer(Serializer):
    """Serializer for TIFF files using tifffile."""

    def serialize(self, item: Any) -> tuple[bytes, str | None]:
        if isinstance(item, Tiff) and (item.array is not None or item.image is not None):
            array = item.array if item.array is not None else np.asarray(item.image)
            buffer = io.BytesIO()
            import tifffile

            tifffile.imwrite(buffer, _as_numpy(array))
            return buffer.getvalue(), "tiff"
        if isinstance(item, (Tiff, dict)):
            data, _ = _read_media_bytes(item)
            return data, "tiff"
        if not isinstance(item, str) or not os.path.isfile(item):
            raise ValueError(f"The item to serialize must be a valid file path. Received: {item}")

        with open(item, "rb") as f:
            data = f.read()

        return data, None

    def deserialize(self, data: bytes) -> Any:
        import tifffile

        return tifffile.imread(io.BytesIO(data))  # This is a NumPy array

    def can_serialize(self, item: Any) -> bool:
        if isinstance(item, Tiff):
            return True
        return isinstance(item, str) and os.path.isfile(item) and item.lower().endswith((".tif", ".tiff"))


_SERIALIZERS = OrderedDict(
    **{
        "str": StringSerializer(),
        "text": TextSerializer(),
        "bool": BooleanSerializer(),
        "int": IntegerSerializer(),
        "float": FloatSerializer(),
        "video": VideoSerializer(),
        "audio": AudioSerializer(),
        "image": ImageSerializer(),
        "nifti": NiftiSerializer(),
        "mesh": MeshSerializer(),
        "pdf": PDFSerializer(),
        "graph": GraphSerializer(),
        "tifffile": TIFFSerializer(),
        "file": FileSerializer(),
        "pil": PILSerializer(),
        "jpeg": JPEGSerializer(),
        "jpeg_array": JPEGArraySerializer(),
        "bytes": BytesSerializer(),
        "no_header_numpy": NoHeaderNumpySerializer(),
        "numpy": NumpySerializer(),
        "no_header_tensor": NoHeaderTensorSerializer(),
        "tensor": TensorSerializer(),
        "pickle": PickleSerializer(),
    }
)


def _get_serializers(serializers: dict[str, Serializer] | None) -> dict[str, Serializer]:
    if serializers is None:
        serializers = {}
    serializers = OrderedDict(serializers)

    for key, value in _SERIALIZERS.items():
        if key not in serializers:
            # Shallow copy is enough: serializers are flat objects. deepcopy was
            # paid on every BinaryWriter / BinaryReader construct.
            serializers[key] = copy(value)

    return serializers

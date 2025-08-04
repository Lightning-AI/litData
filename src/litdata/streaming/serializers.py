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
import tempfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from itertools import chain
from typing import Any, Optional

import numpy as np
import tifffile
import torch

from litdata.constants import (
    _AV_AVAILABLE,
    _NUMPY_DTYPES_MAPPING,
    _PIL_AVAILABLE,
    _TORCH_DTYPES_MAPPING,
)


class Serializer(ABC):
    """The base interface for any serializers.

    A Serializer serialize and deserialize to and from bytes.

    """

    @abstractmethod
    def serialize(self, data: Any) -> tuple[bytes, Optional[str]]:
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

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
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
        mode = data[idx:idx2].decode("utf-8")
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore

    def can_serialize(self, item: Any) -> bool:
        if not _PIL_AVAILABLE:
            return False

        from PIL import Image
        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, Image.Image) and not isinstance(item, JpegImageFile)


class JPEGSerializer(Serializer):
    """The JPEGSerializer serialize and deserialize JPEG image to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        if not _PIL_AVAILABLE:
            raise ModuleNotFoundError("PIL is required. Run `pip install pillow`")

        from PIL import Image
        from PIL.GifImagePlugin import GifImageFile
        from PIL.JpegImagePlugin import JpegImageFile
        from PIL.PngImagePlugin import PngImageFile
        from PIL.WebPImagePlugin import WebPImageFile

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
        from torchvision.io import decode_image, decode_jpeg

        array = torch.frombuffer(data, dtype=torch.uint8)
        # Try decoding as JPEG. Some datasets (e.g., ImageNet) may have PNG images with a JPEG extension,
        # which will cause decode_jpeg to fail. In that case, fall back to a generic image decoder.
        with suppress(RuntimeError):
            return decode_jpeg(array)

        # Fallback: decode as a generic image (handles PNG, etc.)
        return decode_image(array)

    def can_serialize(self, item: Any) -> bool:
        if not _PIL_AVAILABLE:
            return False

        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, JpegImageFile)


class JPEGArraySerializer(Serializer):
    """The JPEGArraySerializer serializes and deserializes lists of JPEG images to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        # Store number of images as first 4 bytes
        n_images_bytes = np.uint32(len(item)).tobytes()

        # create a instance of JPEGSerializer
        if not hasattr(self, "_jpeg_serializer"):
            self._jpeg_serializer = JPEGSerializer()

        # convert each image to bytes and store in a list
        image_bytes = []
        for image in item:
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
        if not _PIL_AVAILABLE:
            return False
        from PIL.JpegImagePlugin import JpegImageFile

        return isinstance(item, (list, tuple)) and all(isinstance(elem, JpegImageFile) for elem in item)


class BytesSerializer(Serializer):
    """The BytesSerializer serialize and deserialize integer to and from bytes."""

    def serialize(self, item: bytes) -> tuple[bytes, Optional[str]]:
        return item, None

    def deserialize(self, item: bytes) -> bytes:
        return item

    def can_serialize(self, item: bytes) -> bool:
        return isinstance(item, bytes)


class TensorSerializer(Serializer):
    """The TensorSerializer serialize and deserialize tensor to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}

    def serialize(self, item: torch.Tensor) -> tuple[bytes, Optional[str]]:
        dtype_indice = self._dtype_to_indices[item.dtype]
        data = [np.uint32(dtype_indice).tobytes()]
        data.append(np.uint32(len(item.shape)).tobytes())
        for dim in item.shape:
            data.append(np.uint32(dim).tobytes())
        data.append(item.numpy().tobytes(order="C"))
        return b"".join(data), None

    def deserialize(self, data: bytes) -> torch.Tensor:
        dtype_indice = np.frombuffer(data[0:4], np.uint32).item()
        dtype = _TORCH_DTYPES_MAPPING[dtype_indice]
        shape_size = np.frombuffer(data[4:8], np.uint32).item()
        shape = []
        for shape_idx in range(shape_size):
            shape.append(np.frombuffer(data[8 + 4 * shape_idx : 8 + 4 * (shape_idx + 1)], np.uint32).item())
        idx_start = 8 + 4 * shape_size
        idx_end = len(data)
        if idx_end > idx_start:
            tensor = torch.frombuffer(data[idx_start:idx_end], dtype=dtype)
        else:
            assert idx_start == idx_end, "The starting index should never be greater than end ending index."
            tensor = torch.empty(shape, dtype=dtype)
        shape = torch.Size(shape)
        if tensor.shape == shape:
            return tensor
        return torch.reshape(tensor, shape)

    def can_serialize(self, item: torch.Tensor) -> bool:
        return isinstance(item, torch.Tensor) and len(item.shape) != 1


class NoHeaderTensorSerializer(Serializer):
    """The TensorSerializer serialize and deserialize tensor to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _TORCH_DTYPES_MAPPING.items()}
        self._dtype: Optional[torch.dtype] = None

    def setup(self, data_format: str) -> None:
        self._dtype = _TORCH_DTYPES_MAPPING[int(data_format.split(":")[1])]

    def serialize(self, item: torch.Tensor) -> tuple[bytes, Optional[str]]:
        dtype_indice = self._dtype_to_indices[item.dtype]
        return item.numpy().tobytes(order="C"), f"no_header_tensor:{dtype_indice}"

    def deserialize(self, data: bytes) -> torch.Tensor:
        assert self._dtype
        return torch.frombuffer(data, dtype=self._dtype) if len(data) > 0 else torch.empty((0,), dtype=self._dtype)

    def can_serialize(self, item: torch.Tensor) -> bool:
        return isinstance(item, torch.Tensor) and len(item.shape) == 1


class NumpySerializer(Serializer):
    """The NumpySerializer serialize and deserialize numpy to and from bytes."""

    def __init__(self) -> None:
        super().__init__()
        self._dtype_to_indices = {v: k for k, v in _NUMPY_DTYPES_MAPPING.items()}

    def serialize(self, item: np.ndarray) -> tuple[bytes, Optional[str]]:
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
        tensor = np.frombuffer(data[8 + 4 * shape_size : len(data)], dtype=dtype)
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
        self._dtype: Optional[np.dtype] = None

    def setup(self, data_format: str) -> None:
        self._dtype = _NUMPY_DTYPES_MAPPING[int(data_format.split(":")[1])]

    def serialize(self, item: np.ndarray) -> tuple[bytes, Optional[str]]:
        dtype_indice: int = self._dtype_to_indices[item.dtype]
        return item.tobytes(order="C"), f"no_header_numpy:{dtype_indice}"

    def deserialize(self, data: bytes) -> np.ndarray:
        assert self._dtype
        return np.frombuffer(data, dtype=self._dtype)

    def can_serialize(self, item: np.ndarray) -> bool:
        return isinstance(item, np.ndarray) and len(item.shape) == 1


class PickleSerializer(Serializer):
    """The PickleSerializer serialize and deserialize python objects to and from bytes."""

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        return pickle.dumps(item), None

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)  # noqa: S301

    def can_serialize(self, _: Any) -> bool:
        return True


class FileSerializer(Serializer):
    def serialize(self, filepath: str) -> tuple[bytes, Optional[str]]:
        print("FileSerializer will be removed in the future.")
        _, file_extension = os.path.splitext(filepath)
        with open(filepath, "rb") as f:
            file_extension = file_extension.replace(".", "").lower()
            return f.read(), f"file:{file_extension}"

    def deserialize(self, data: bytes) -> Any:
        return data

    def can_serialize(self, data: Any) -> bool:
        # return isinstance(data, str) and os.path.isfile(data)
        # FileSerializer will be removed in the future.
        return False


class VideoSerializer(Serializer):
    _EXTENSIONS = ("mp4", "ogv", "mjpeg", "avi", "mov", "h264", "mpg", "webm", "wmv")

    def serialize(self, filepath: str) -> tuple[bytes, Optional[str]]:
        _, file_extension = os.path.splitext(filepath)
        with open(filepath, "rb") as f:
            file_extension = file_extension.replace(".", "").lower()
            return f.read(), f"video:{file_extension}"

    def deserialize(self, data: bytes) -> Any:
        if not _AV_AVAILABLE:
            raise ModuleNotFoundError("av is required. Run `pip install av`")

        # Add support for a better deserialization mechanism for videos
        # TODO: Investigate https://pytorch.org/audio/main/generated/torchaudio.io.StreamReader.html
        import torchvision.io

        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, "file.mp4")
            with open(fname, "wb") as stream:
                stream.write(data)
            return torchvision.io.read_video(fname, pts_unit="sec")

    def can_serialize(self, data: Any) -> bool:
        return isinstance(data, str) and os.path.isfile(data) and any(data.endswith(ext) for ext in self._EXTENSIONS)


class StringSerializer(Serializer):
    def serialize(self, obj: str) -> tuple[bytes, Optional[str]]:
        return obj.encode("utf-8"), None

    def deserialize(self, data: bytes) -> str:
        return data.decode("utf-8")

    def can_serialize(self, data: str) -> bool:
        return isinstance(data, str) and not os.path.isfile(data)


class NumericSerializer:
    """Store scalar."""

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype
        self.size = self.dtype().nbytes

    def serialize(self, obj: Any) -> tuple[bytes, Optional[str]]:
        return self.dtype(obj).tobytes(), None

    def deserialize(self, data: bytes) -> Any:
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

    def serialize(self, item: bool) -> tuple[bytes, Optional[str]]:
        """Serialize a boolean value to bytes.

        Args:
            item: Boolean value to serialize

        Returns:
            Tuple containing the serialized bytes and None for the format string
        """
        return np.bool_(item).tobytes(), None

    def deserialize(self, data: bytes) -> bool:
        """Deserialize bytes back into a boolean value.

        Args:
            data: Bytes to deserialize

        Returns:
            The deserialized boolean value
        """
        return bool(np.frombuffer(data, dtype=np.bool_)[0])

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

    def serialize(self, item: Any) -> tuple[bytes, Optional[str]]:
        if not isinstance(item, str) or not os.path.isfile(item):
            raise ValueError(f"The item to serialize must be a valid file path. Received: {item}")

        # Read the TIFF file as bytes
        with open(item, "rb") as f:
            data = f.read()

        return data, None

    def deserialize(self, data: bytes) -> Any:
        return tifffile.imread(io.BytesIO(data))  # This is a NumPy array

    def can_serialize(self, item: Any) -> bool:
        return isinstance(item, str) and os.path.isfile(item) and item.lower().endswith((".tif", ".tiff"))


_SERIALIZERS = OrderedDict(
    **{
        "str": StringSerializer(),
        "bool": BooleanSerializer(),
        "int": IntegerSerializer(),
        "float": FloatSerializer(),
        "video": VideoSerializer(),
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


def _get_serializers(serializers: Optional[dict[str, Serializer]]) -> dict[str, Serializer]:
    if serializers is None:
        serializers = {}
    serializers = OrderedDict(serializers)

    for key, value in _SERIALIZERS.items():
        if key not in serializers:
            serializers[key] = deepcopy(value)

    return serializers

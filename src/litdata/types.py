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

"""Typed media wrappers for ``optimize`` / ``map``.

These are sample values, not Arrow schemas. Wrapping a path makes it unambiguous
to the serializer (a caption string is not an audio file). Native objects
(``array=``, ``image=``, ``mesh=``) are encoded with the matching serializer.

    optimize(lambda p: Audio(path=p), inputs=wavs, output_dir=...)
    optimize(lambda x: Image(array=x, quality=95, format="jpeg"), ...)
    optimize(lambda g: g, inputs=pyg_data_list, output_dir=...)  # Data.to_dict packed
    optimize(lambda g: Graph(x=g.x, edge_index=g.edge_index), ...)
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


@dataclass
class Image(_MediaRef):
    """Image sample: path, bytes, numpy/tensor ``array=``, or a PIL ``image=``."""

    array: Any = None
    image: Any = None
    mode: str | None = None
    format: str | None = None
    quality: int | None = None


@dataclass
class Jpeg(_MediaRef):
    """JPEG sample. ``quality`` defaults to 95 when encoding from ``array=`` / ``image=``."""

    array: Any = None
    image: Any = None
    mode: str | None = None
    quality: int = 95


@dataclass
class JpegArray:
    """List of JPEGs (``JPEGArraySerializer``)."""

    images: list[Any] = field(default_factory=list)
    quality: int = 95


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

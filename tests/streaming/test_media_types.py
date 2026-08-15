# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
import sys
import types

import numpy as np
import pytest
import torch

from litdata.streaming.serializers import (
    _SERIALIZERS,
    AudioSerializer,
    GraphSerializer,
    ImageSerializer,
    JPEGArraySerializer,
    NoHeaderTensorSerializer,
    TensorSerializer,
    VideoSerializer,
    _get_serializers,
    _image_array_for_pil,
    _jpeg_has_exif_app1,
    _read_media_bytes,
)
from litdata.types import Audio, Graph, Image, Jpeg, JpegArray, Tensor
from litdata.utilities._pytree import tree_flatten

try:
    import torch_geometric  # noqa: F401

    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


def test_image_serializer_claims_bare_jpeg_path(tmpdir):
    from PIL import Image as PILImage

    path = os.path.join(tmpdir, "img.jpeg")
    PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, format="JPEG")
    assert ImageSerializer().can_serialize(path)
    data, name = ImageSerializer().serialize(path)
    assert name.startswith("image:")
    tensor = ImageSerializer().deserialize(data)
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.uint8


def test_writer_picks_audio_not_string_for_wrapper():
    serializers = _get_serializers(None)
    caption = "dog barking.wav"
    audio = Audio(bytes=b"RIFF....", path="dog.wav")
    picked = next(name for name, serializer in serializers.items() if serializer.can_serialize(audio))
    assert picked == "audio"
    picked_caption = next(name for name, serializer in serializers.items() if serializer.can_serialize(caption))
    assert picked_caption == "str"


def test_empty_image_raises():
    with pytest.raises(TypeError, match="needs path="):
        _read_media_bytes(Image())


def test_float_unit_interval_scales_to_uint8():
    array = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = _image_array_for_pil(array)
    assert out.dtype == np.dtype("|u1")
    assert int(out.max()) == 127 or int(out.max()) == 128


def test_jpeg_exif_detector():
    assert not _jpeg_has_exif_app1(b"\xff\xd8\xff\xda")
    # APP1 + Exif
    payload = b"Exif\x00\x00xxxx"
    marker = b"\xff\xe1" + (2 + len(payload)).to_bytes(2, "big") + payload
    assert _jpeg_has_exif_app1(b"\xff\xd8" + marker + b"\xff\xda")


def test_image_roundtrip_array_quality():
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[0, 0] = [255, 0, 0]
    data, name = ImageSerializer().serialize(Image(array=array, quality=95, format="jpeg"))
    assert name == "image:jpg"
    tensor = ImageSerializer().deserialize(data)
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.uint8


def _tiny_jpeg_bytes() -> bytes:
    from PIL import Image as PILImage

    buf = __import__("io").BytesIO()
    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_jpeg_array_does_not_mutate_quality():
    jpeg = Jpeg(bytes=_tiny_jpeg_bytes(), quality=95)
    JPEGArraySerializer().serialize(JpegArray(images=[jpeg], quality=80))
    assert jpeg.quality == 95


def test_video_does_not_claim_audio_wrapper():
    assert not VideoSerializer().can_serialize(Audio(bytes=b"RIFF"))
    assert AudioSerializer().can_serialize(Audio(bytes=b"RIFF"))


def test_writer_image_type_roundtrip(tmpdir):
    from litdata.streaming.reader import BinaryReader
    from litdata.streaming.sampler import ChunkedIndex
    from litdata.streaming.writer import BinaryWriter

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    writer = BinaryWriter(cache_dir, chunk_size=2)
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :] = [10, 20, 30]
    writer[0] = {"id": 0, "caption": "a red square", "image": Image(array=array, quality=95, format="jpeg")}
    writer[1] = {"id": 1, "caption": "a red square", "image": Image(array=array, quality=95, format="jpeg")}
    writer.done()
    writer.merge()

    reader = BinaryReader(cache_dir)
    sample = reader.read(ChunkedIndex(0, chunk_index=0))
    assert sample["id"] == 0
    assert sample["caption"] == "a red square"
    assert sample["image"].shape == (3, 8, 8)


def test_writer_audio_type_vs_caption(tmpdir):
    import json

    from litdata.streaming.reader import BinaryReader
    from litdata.streaming.sampler import ChunkedIndex
    from litdata.streaming.writer import BinaryWriter

    wav, _ = AudioSerializer(decode="bytes").serialize(
        {"array": np.zeros(800, dtype=np.float32), "sampling_rate": 8000}
    )
    path = os.path.join(tmpdir, "tone.wav")
    with open(path, "wb") as handle:
        handle.write(wav)

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    writer = BinaryWriter(cache_dir, chunk_size=1, serializers={"audio": AudioSerializer(decode="bytes")})
    writer[0] = {"audio": Audio(path=path), "caption": "tone.wav"}
    writer.done()
    writer.merge()
    with open(os.path.join(cache_dir, "index.json")) as handle:
        formats = json.load(handle)["config"]["data_format"]
    assert any(fmt.startswith("audio") for fmt in formats)
    assert "str" in formats
    sample = BinaryReader(cache_dir, serializers={"audio": AudioSerializer(decode="bytes")}).read(
        ChunkedIndex(0, chunk_index=0)
    )
    assert sample["caption"] == "tone.wav"
    assert sample["audio"][:4] == b"RIFF"


def test_pytree_wrapper_is_single_leaf():
    sample = {"a": Audio(path="x.wav"), "b": {"c": Image(bytes=b"\xff\xd8")}}
    leaves, _ = tree_flatten(sample)
    assert len(leaves) == 2
    assert isinstance(leaves[0], Audio)
    assert isinstance(leaves[1], Image)


def test_serializers_registry_has_media_keys():
    for key in ("video", "audio", "image", "nifti", "mesh", "pdf"):
        assert key in _SERIALIZERS


def test_tensor_wrapper_routes_1d_and_nd():
    tokens = Tensor(array=torch.arange(8, dtype=torch.int64))
    image = Tensor(array=torch.zeros(3, 4, 4))
    assert NoHeaderTensorSerializer().can_serialize(tokens)
    assert not TensorSerializer().can_serialize(tokens)
    assert TensorSerializer().can_serialize(image)
    assert not NoHeaderTensorSerializer().can_serialize(image)
    data, name = NoHeaderTensorSerializer().serialize(tokens)
    assert name.startswith("no_header_tensor:")
    serializer = NoHeaderTensorSerializer()
    serializer.setup(name)
    assert torch.equal(serializer.deserialize(data), tokens.array)
    packed, _ = TensorSerializer().serialize(image)
    assert TensorSerializer().deserialize(packed).shape == (3, 4, 4)


def test_tokens_loader_reads_tensor_wrapper_dim():
    from litdata.streaming.item_loader import TokensLoader

    tokens = Tensor(array=torch.arange(32, dtype=torch.int64))
    data, _ = NoHeaderTensorSerializer().serialize(tokens)
    encoded, dim = TokensLoader.encode_data([data], [len(data)], [tokens])
    assert dim == 32
    assert encoded == data


def test_graph_serializer_packs_coo_tensors():
    graph = Graph(
        x=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.ones(3, 1),
        y=torch.tensor([7]),
    )
    picked = next(name for name, serializer in _get_serializers(None).items() if serializer.can_serialize(graph))
    assert picked == "graph"
    data, name = GraphSerializer().serialize(graph)
    assert name == "graph"
    assert data.startswith(b"LDGR")
    out = GraphSerializer().deserialize(data)
    assert out.x.shape == (4, 3)
    assert out.edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert out.y.tolist() == [7]


def test_graph_serializer_packs_extra_keys_and_meta():
    graph = Graph(
        x=torch.ones(2, 1),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        data={"train_mask": torch.tensor([True, False]), "num_nodes": 2},
    )
    data, name = GraphSerializer().serialize(graph)
    assert name == "graph"
    out = GraphSerializer().deserialize(data)
    assert out.x.shape == (2, 1)
    if hasattr(out, "train_mask"):
        mask, num_nodes = out.train_mask, out.num_nodes
    else:
        mask, num_nodes = out.data["train_mask"], out.data["num_nodes"]
    assert mask.tolist() == [True, False]
    assert int(num_nodes) == 2


def test_graph_reconstruct_uses_pyg_from_dict(monkeypatch):
    class FakeData:
        def __init__(self, mapping):
            self.mapping = mapping

        @classmethod
        def from_dict(cls, mapping):
            return cls(mapping)

    pyg = types.ModuleType("torch_geometric")
    pyg_data = types.ModuleType("torch_geometric.data")
    pyg_data.Data = FakeData
    pyg.data = pyg_data
    monkeypatch.setitem(sys.modules, "torch_geometric", pyg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", pyg_data)

    graph = Graph(x=torch.ones(2, 1), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    packed, _ = GraphSerializer().serialize(graph)
    out = GraphSerializer().deserialize(packed)
    assert isinstance(out, FakeData)
    assert out.mapping["x"].shape == (2, 1)


@pytest.mark.skipif(not _PYG_AVAILABLE, reason="Requires torch-geometric")
def test_pyg_data_roundtrip_and_loader(tmpdir):
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool

    from litdata import StreamingDataLoader, StreamingDataset
    from litdata.streaming.cache import Cache

    cache = Cache(str(tmpdir), chunk_size=8)
    for i in range(16):
        n = 6 + i % 3
        cache[i] = Data(
            x=torch.randn(n, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            y=torch.tensor(i % 2, dtype=torch.long),
            num_nodes=n,
        )
    cache.done()
    cache.merge()

    ds = StreamingDataset(str(tmpdir))
    assert type(ds[0]).__name__ == "Data"
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, Batch)
    assert batch.y.numel() == 4

    conv = GCNConv(4, 2)
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(list(conv.parameters()) + list(lin.parameters()), lr=1e-2)
    logits = lin(global_mean_pool(conv(batch.x, batch.edge_index), batch.batch))
    loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
    loss.backward()
    opt.step()
    assert loss.ndim == 0


def test_graph_pickle_fallback_for_opaque_data():
    graph = Graph(data={"nodes": [1, 2], "edges": [(0, 1)]})
    data, name = GraphSerializer().serialize(graph)
    assert name == "graph:pickle"
    serializer = GraphSerializer()
    serializer.setup(name)
    out = serializer.deserialize(data)
    assert out.data == {"nodes": [1, 2], "edges": [(0, 1)]}

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

import json
import os
import random
import sys

import numpy as np
import pytest
import torch
from lightning_utilities.core.imports import RequirementCache

from litdata.streaming.compression import _ZSTD_AVAILABLE
from litdata.streaming.reader import BinaryReader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.writer import BinaryWriter
from litdata.utilities.format import _FORMAT_TO_RATIO


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


_PIL_AVAILABLE = RequirementCache("PIL")


def test_binary_writer_with_ints_and_chunk_bytes(tmpdir):
    match = (
        "The provided compression something_else isn't available"
        if _ZSTD_AVAILABLE
        else "No compression algorithms are installed."
    )

    with pytest.raises(ValueError, match=match):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, chunk_bytes=90)

    for i in range(100):
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    # JSON int rows use an Arrow IPC footer; schema overhead is larger than 90 bytes,
    # so chunks are 1-row each instead of the old 2-row pytree packing.
    assert len(os.listdir(tmpdir)) >= 1
    binary_writer.done()
    binary_writer.merge()

    with open(os.path.join(tmpdir, "index.json")) as f:
        index = json.load(f)

    assert sum(chunk["chunk_size"] for chunk in index["chunks"]) == 100
    chunk_sizes = np.cumsum([chunk["chunk_size"] for chunk in index["chunks"]])

    reader = BinaryReader(tmpdir, max_cache_size=10 ^ 9)
    for i in range(100):
        for chunk_index, chunk_start in enumerate(chunk_sizes):
            if i >= chunk_start:
                continue
            break
        data = reader.read(ChunkedIndex(i, chunk_index=chunk_index))
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}


def test_binary_writer_with_ints_and_chunk_size(tmpdir):
    seed_everything(42)

    match = (
        "The provided compression something_else isn't available"
        if _ZSTD_AVAILABLE
        else "No compression algorithms are installed."
    )

    with pytest.raises(ValueError, match=match):
        BinaryWriter(tmpdir, {"i": "int"}, compression="something_else")

    binary_writer = BinaryWriter(tmpdir, chunk_size=25)

    indices = list(range(100))
    indices = indices[:5] + np.random.permutation(indices[5:]).tolist()

    for i in indices:
        binary_writer[i] = {"i": i, "i+1": i + 1, "i+2": i + 2}

    assert len(os.listdir(tmpdir)) >= 2
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(tmpdir)) == 5

    with open(os.path.join(tmpdir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 25
    assert data["chunks"][1]["chunk_size"] == 25
    assert data["chunks"][-1]["chunk_size"] == 25

    reader = BinaryReader(tmpdir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 25))
        assert data == {"i": i, "i+1": i + 1, "i+2": i + 2}


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "darwin", reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_and_int(tmpdir):
    """Validate the writer and reader can serialize / deserialize a pair of image and label."""
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_bytes=2 << 12)

    imgs = []

    for i in range(100):
        path = os.path.join(tmpdir, f"img{i}.jpeg")
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(np_data).convert("L")
        img.save(path, format="jpeg", quality=100)
        img = Image.open(path)
        imgs.append(img)
        binary_writer[i] = {"x": img, "y": i}

    assert len(os.listdir(cache_dir)) == 24
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(cache_dir)) == 26

    with open(os.path.join(cache_dir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 4
    assert data["chunks"][1]["chunk_size"] == 4
    assert data["chunks"][-1]["chunk_size"] == 4

    reader = BinaryReader(cache_dir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 4))
        # JPEG deserialize uses ImageReadMode.RGB (CHW), including grayscale sources.
        got = np.asarray(data["x"])
        assert got.shape == (3, 28, 28)
        expected = np.asarray(imgs[i].convert("RGB")).transpose(2, 0, 1)
        np.testing.assert_array_equal(got, expected)
        assert data["y"] == i


@pytest.mark.skipif(condition=not _PIL_AVAILABLE or sys.platform == "darwin", reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_filepath_and_int(tmpdir):
    """Validate the writer and reader can serialize / deserialize a pair of image and label."""
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_size=7)  # each chunk will have 7 items

    imgs = []

    for i in range(100):
        path = os.path.join(tmpdir, f"img{i}.jpeg")
        np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(np_data).convert("L")
        img.save(path, format="jpeg", quality=100)
        img = Image.open(path)
        imgs.append(img)
        binary_writer[i] = {"x": path, "y": i}

    assert len(os.listdir(cache_dir)) == 14  # 100 items / 7 items per chunk = 14 chunks
    binary_writer.done()
    binary_writer.merge()
    assert len(os.listdir(cache_dir)) == 16  # 2 items in last chunk and index.json file

    with open(os.path.join(cache_dir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 7
    assert data["chunks"][1]["chunk_size"] == 7
    assert data["chunks"][-1]["chunk_size"] == 2
    assert sum([chunk["chunk_size"] for chunk in data["chunks"]]) == 100

    reader = BinaryReader(cache_dir, max_cache_size=10 ^ 9)
    for i in range(100):
        data = reader.read(ChunkedIndex(i, chunk_index=i // 7))
        # Filepath → image bytes → RGB CHW tensor (torchvision). No PIL on read.
        got = np.asarray(data["x"])
        assert got.shape == (3, 28, 28)
        expected = np.asarray(imgs[i].convert("RGB")).transpose(2, 0, 1)
        np.testing.assert_array_equal(got, expected)
        assert data["y"] == i


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_binary_writer_with_jpeg_and_png(tmpdir):
    from PIL import Image

    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    binary_writer = BinaryWriter(cache_dir, chunk_bytes=2 << 12)

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint8)
    img = Image.fromarray(np_data).convert("L")
    path = os.path.join(tmpdir, "img.jpeg")
    img.save(path, format="jpeg", quality=100)
    img_jpeg = Image.open(path)

    binary_writer[0] = {"x": img_jpeg, "y": 0}
    binary_writer[1] = {"x": img, "y": 1}

    with pytest.raises(TypeError, match="The provided item should be of type"):
        binary_writer[2] = {"x": 2, "y": 1}


def test_writer_human_format(tmpdir):
    for k, v in _FORMAT_TO_RATIO.items():
        binary_writer = BinaryWriter(tmpdir, chunk_bytes=f"{1}{k}")
        assert binary_writer._chunk_bytes == v

    binary_writer = BinaryWriter(tmpdir, chunk_bytes="64MB")
    assert binary_writer._chunk_bytes == 64000000


def test_writer_unordered_indexes(tmpdir):
    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)

    binary_writer = BinaryWriter(cache_dir, chunk_size=5)

    arr = [2, 3, 1, 4, 6, 5, 7, 8, 11, 9, 10, 12]

    for i in arr:
        binary_writer[i] = i - 1

    binary_writer.done()
    binary_writer.merge()

    reader = BinaryReader(cache_dir)
    for i in range(12):
        assert i == reader.read(ChunkedIndex(i, chunk_index=i // 5))

    with open(os.path.join(cache_dir, "index.json")) as f:
        data = json.load(f)

    assert data["chunks"][0]["chunk_size"] == 5
    assert data["chunks"][1]["chunk_size"] == 5
    assert data["chunks"][2]["chunk_size"] == 2


def test_chunk_bytes_consistency(tmpdir):
    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)

    binary_writer = BinaryWriter(cache_dir, chunk_size=5)

    for i in range(100):
        binary_writer[i] = i

    binary_writer.done()
    binary_writer.merge()

    with open(os.path.join(cache_dir, "index.json")) as f:
        config_data = json.load(f)

    for chunk in config_data["chunks"]:
        chunk_file = os.path.join(cache_dir, chunk["filename"])
        chunk_size = os.path.getsize(chunk_file)
        assert chunk_size == chunk["chunk_bytes"]


def test_writer_save_checkpoint(tmpdir):
    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)

    binary_writer = BinaryWriter(cache_dir, chunk_size=5)

    arr = [2, 3, 1, 4, 6, 5, 7, 8, 11, 9, 10, 12]

    for i in arr:
        binary_writer[i] = i - 1

    binary_writer.done()
    binary_writer.merge()
    binary_writer.save_checkpoint()

    checkpoint_dir = os.path.join(cache_dir, ".checkpoints")
    files = os.listdir(checkpoint_dir)
    assert files == ["checkpoint-0.json"]
    with open(os.path.join(checkpoint_dir, "checkpoint-0.json")) as f:
        payload = json.load(f)
    assert payload["inputs_done"] == payload["samples_written"] == 12
    assert payload["next_chunk_index"] == binary_writer._chunk_index
    assert "chunks" in payload

    binary_writer.save_checkpoint()  # no-op when unchanged
    assert os.listdir(checkpoint_dir) == ["checkpoint-0.json"]


def test_merge_natural_sort_order_with_many_workers(tmpdir):
    # With 11+ workers, index files are named 0.index.json ... 10.index.json.
    # sorted() puts "10.index.json" before "2.index.json" alphabetically, making
    # chunk-10-0.bin appear before chunk-2-0.bin in the merged index. Fixes #826.
    config = {
        "chunk_bytes": None,
        "chunk_size": 1,
        "compression": None,
        "data_format": ["scalar"],
        "data_spec": None,
        "encryption": None,
        "item_loader": "PyTreeLoader",
    }
    from litdata.constants import _INDEX_FILENAME

    n_workers = 11
    for rank in range(n_workers):
        chunk = {"chunk_size": 1, "column_sizes": [4], "dim": None, "filename": f"chunk-{rank}-0.bin"}
        with open(os.path.join(str(tmpdir), f"{rank}.{_INDEX_FILENAME}"), "w") as f:
            json.dump({"chunks": [chunk], "config": config}, f, sort_keys=True)

    writer = BinaryWriter(str(tmpdir), chunk_size=1)
    writer._is_done = True
    writer._rank = 0
    writer._merge_no_wait()

    with open(os.path.join(str(tmpdir), _INDEX_FILENAME)) as f:
        data = json.load(f)

    filenames = [c["filename"] for c in data["chunks"]]
    assert filenames == [f"chunk-{i}-0.bin" for i in range(n_workers)]


@pytest.mark.skipif(not _ZSTD_AVAILABLE, reason="Requires zstd")
def test_zstd_decompress_file_roundtrip(tmpdir):
    from litdata.streaming.compression import ZSTDCompressor

    compressor = ZSTDCompressor(4)
    payload = os.urandom(80_000)
    src = os.path.join(tmpdir, "chunk.bin.zstd")
    dst = os.path.join(tmpdir, "chunk.bin")
    with open(src, "wb") as f:
        f.write(compressor.compress(payload))
    compressor.decompress_file(src, dst)
    with open(dst, "rb") as f:
        assert f.read() == payload


def test_sample_account_bytes_and_binary_heavy():
    from litdata.streaming.writer import _sample_account_bytes, _sample_is_binary_heavy

    jpeg = {"image": {"bytes": b"\xff\xd8" + b"x" * 4000, "path": "a.jpg"}, "label": 1}
    text = {"text": "hello world " * 20, "id": "n"}
    assert _sample_is_binary_heavy(jpeg)
    assert not _sample_is_binary_heavy(text)
    assert _sample_account_bytes(jpeg) >= 4000
    assert _sample_account_bytes(text) == len("hello world " * 20) + 1


def test_writer_filled_false_during_optimize_append(tmpdir, monkeypatch):
    from litdata.constants import _INDEX_FILENAME

    with open(os.path.join(tmpdir, _INDEX_FILENAME), "w") as f:
        json.dump({"chunks": [], "config": {}}, f)
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", "0")
    writer = BinaryWriter(str(tmpdir), chunk_bytes=90)
    assert writer.filled is False
    writer[0] = 1
    assert writer.done()
    assert os.path.isfile(os.path.join(tmpdir, "0.index.json"))

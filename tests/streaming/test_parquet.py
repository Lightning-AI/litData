import json
import os
import sys
import tempfile
from contextlib import nullcontext
from fnmatch import fnmatch
from unittest.mock import Mock, patch

import pytest

from litdata.constants import _DEFAULT_CACHE_DIR, _DEFAULT_LIGHTNING_CACHE_DIR, _INDEX_FILENAME
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader, PyTreeLoader
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.hf_dataset import (
    _hf_cache_root,
    _persist_hf_parquet_files,
    _stabilize_hf_row,
    hf_parquet_cache_path,
    index_hf_dataset,
    optimize_hf,
    resolve_hf_dataset_url,
)
from litdata.utilities.parquet import (
    CloudParquetDir,
    HFParquetDir,
    LocalParquetDir,
    _hf_relative_name,
    get_parquet_indexer_cls,
)


#! TODO: Fix test failing on windows
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows and test gets cancelled")
@pytest.mark.usefixtures("clean_pq_index_cache")
@pytest.mark.parametrize(
    ("pq_dir_url"),
    [
        "s3://some_bucket/some_path",
        "gs://some_bucket/some_path",
        "hf://datasets/some_org/some_repo/some_path",
    ],
)
@pytest.mark.parametrize(("num_worker"), [None, 1, 2, 4])
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
@patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
@patch("litdata.utilities.parquet._FSSPEC_AVAILABLE", True)
def test_parquet_index_write(
    monkeypatch, tmp_path, pq_data, huggingface_hub_fs_mock, fsspec_pq_mock, pq_dir_url, num_worker
):
    if pq_dir_url is None:
        pq_dir_url = os.path.join(tmp_path, "pq-dataset")

    cache_dir = os.path.join(tmp_path, "pq-cache")
    os.makedirs(cache_dir, exist_ok=True)

    index_file_path = os.path.join(tmp_path, "pq-dataset", _INDEX_FILENAME)
    if pq_dir_url.startswith("hf://"):
        index_file_path = os.path.join(cache_dir, _INDEX_FILENAME)

    assert not os.path.exists(index_file_path)

    # call the write_parquet_index fn
    index_parquet_dataset(pq_dir_url=pq_dir_url, cache_dir=cache_dir, num_workers=num_worker)
    assert os.path.exists(index_file_path)

    if pq_dir_url.startswith("hf://"):
        assert len(os.listdir(cache_dir)) == 1
    elif pq_dir_url.startswith(("gs://", "s3://")):
        assert len(os.listdir(cache_dir)) == 0

    # Read JSON file into a dictionary
    with open(index_file_path) as f:
        data = json.load(f)
        assert len(data["chunks"]) == 5
        for cnk in data["chunks"]:
            assert cnk["chunk_size"] == 5
        assert data["config"]["item_loader"] == "ParquetLoader"

    # no test for streaming on s3 and gs
    if pq_dir_url is None or pq_dir_url.startswith("hf://"):
        ds = StreamingDataset(pq_dir_url)

        assert len(ds) == 25  # 5 datasets for 5 loops

        for i, _ds in enumerate(ds):
            idx = i % 5
            assert isinstance(_ds, dict)
            assert _ds["name"] == pq_data["name"][idx]
            assert _ds["weight"] == pq_data["weight"][idx]
            assert _ds["height"] == pq_data["height"][idx]


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows and test gets cancelled")
@pytest.mark.usefixtures("clean_pq_index_cache")
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
def test_index_hf_dataset(monkeypatch, tmp_path, huggingface_hub_fs_mock):
    with pytest.raises(ValueError, match="Invalid Hugging Face dataset URL"):
        index_hf_dataset("invalid_url")

    hf_url = "hf://datasets/some_org/some_repo/some_path"
    cache_dir = index_hf_dataset(hf_url)
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, _INDEX_FILENAME))
    # Stable path: a second call must reuse the same directory (no re-index).
    assert index_hf_dataset(hf_url) == cache_dir


def test_resolve_hf_dataset_url(monkeypatch):
    def fake_list(url, storage_options=None):
        if url.startswith("hf://") and ("plain_text/train" in url or url.endswith("default/train")):
            return [f"{url}/0000.parquet"]
        return []

    monkeypatch.setattr("litdata.utilities.hf_dataset._list_hf_parquet_urls", fake_list)
    assert resolve_hf_dataset_url("hf://datasets/org/name/data") == "hf://datasets/org/name/data"
    assert resolve_hf_dataset_url("org/name", split="train") == (
        "hf://datasets/org/name@refs/convert/parquet/default/train"
    )
    assert resolve_hf_dataset_url("org/name", revision="v1", split="train", config="plain_text") == (
        "hf://datasets/org/name@v1/plain_text/train"
    )


@pytest.mark.usefixtures("clean_pq_index_cache")
def test_optimize_hf(tmp_path, write_pq_data, monkeypatch):
    pq_dir = tmp_path / "pq-dataset"
    files = [str(pq_dir / name) for name in sorted(os.listdir(pq_dir)) if name.endswith(".parquet")]
    monkeypatch.setattr("litdata.utilities.hf_dataset.resolve_hf_dataset_url", lambda *a, **k: "hf://datasets/org/name")
    monkeypatch.setattr("litdata.utilities.hf_dataset._prepare_optimize_inputs", lambda *a, **k: files)
    out = tmp_path / "imdb-opt"
    returned = optimize_hf("org/name", output_dir=str(out), chunk_size=10, num_workers=1)
    assert returned == str(out)
    ds = StreamingDataset(str(out))
    assert len(ds) == 25
    assert isinstance(ds[0], dict)
    assert optimize_hf("org/name", output_dir=str(out), chunk_size=10) == str(out)


@pytest.mark.parametrize("remote_dir", ["r2://bucket/imdb-opt", "s3://bucket/imdb-opt"])
def test_optimize_hf_reuses_remote_index(monkeypatch, remote_dir):
    dummy_index = {"chunks": [{"filename": "chunk-0-0.bin"}], "config": {}}

    def fake_download(remote_path, local_path):
        assert remote_path.rstrip("/").endswith(_INDEX_FILENAME)
        with open(local_path, "w") as f:
            json.dump(dummy_index, f)

    fs_provider = Mock()
    fs_provider.download_file = fake_download
    monkeypatch.setattr("litdata.processing.utilities._get_fs_provider", Mock(return_value=fs_provider))

    optimize_calls = []

    def fake_optimize(*_args, **_kwargs):
        optimize_calls.append(True)

    monkeypatch.setattr("litdata.processing.functions.optimize", fake_optimize)
    monkeypatch.setattr(
        "litdata.utilities.hf_dataset.resolve_hf_dataset_url", lambda *_a, **_k: "hf://datasets/org/name"
    )
    monkeypatch.setattr("litdata.utilities.hf_dataset._prepare_optimize_inputs", lambda *_a, **_k: ["x.parquet"])

    assert optimize_hf("org/name", output_dir=remote_dir, overwrite=False) == remote_dir
    assert optimize_calls == []

    assert optimize_hf("org/name", output_dir=remote_dir, overwrite=True, chunk_size=10, num_workers=1) == remote_dir
    assert optimize_calls == [True]


def test_optimize_hf_reuses_lightning_storage_index(monkeypatch):
    from litdata.streaming.resolver import Dir

    dummy_index = {"chunks": [{"filename": "chunk-0-0.bin"}], "config": {}}
    dest = Dir(
        path="/teamspace/lightning_storage/ds/imdb-opt",
        url="r2://bucket/imdb-opt",
        data_connection_id="dc-1",
    )
    monkeypatch.setattr("litdata.streaming.resolver._resolve_dir", lambda _p: dest)

    seen_storage_options = []

    def fake_download(remote_path, local_path):
        with open(local_path, "w") as f:
            json.dump(dummy_index, f)

    def fake_get_fs_provider(url, storage_options=None):
        seen_storage_options.append(dict(storage_options or {}))
        provider = Mock()
        provider.download_file = fake_download
        return provider

    monkeypatch.setattr("litdata.processing.utilities._get_fs_provider", fake_get_fs_provider)

    optimize_calls = []
    monkeypatch.setattr("litdata.processing.functions.optimize", lambda *_a, **_k: optimize_calls.append(True))
    monkeypatch.setattr(
        "litdata.utilities.hf_dataset.resolve_hf_dataset_url", lambda *_a, **_k: "hf://datasets/org/name"
    )
    monkeypatch.setattr("litdata.utilities.hf_dataset._prepare_optimize_inputs", lambda *_a, **_k: ["x.parquet"])

    out = "/teamspace/lightning_storage/ds/imdb-opt"
    assert optimize_hf("org/name", output_dir=out, overwrite=False) == out
    assert optimize_calls == []
    assert seen_storage_options
    assert seen_storage_options[0].get("data_connection_id") == "dc-1"


def test_hf_parquet_cache_is_outside_chunk_dir(monkeypatch):
    monkeypatch.delenv("LITDATA_HF_CACHE_DIR", raising=False)
    chunk_dir = os.path.join(os.sep, "cache", "chunks")
    monkeypatch.setattr("litdata.utilities.hf_dataset.get_default_cache_dir", lambda: chunk_dir)
    root = _hf_cache_root()
    assert os.path.normpath(root) == os.path.normpath(os.path.dirname(os.path.normpath(chunk_dir)))
    dest = hf_parquet_cache_path("hf://datasets/org/name")
    parquet_root = os.path.normpath(os.path.join(root, "hf-parquet"))
    assert os.path.normpath(dest).startswith(parquet_root)
    assert f"{os.sep}chunks{os.sep}" not in dest + os.sep


def test_stabilize_hf_row_wraps_variable_lists():
    from litdata.streaming.serializers import JsonLeaf

    row = {
        "id": "a",
        "answers": ["x", "y"],
        "choices": {"text": ["A", "B"], "label": ["1", "2"]},
        "n": None,
    }
    out = _stabilize_hf_row(row, {"id": "", "answers": [], "choices": {}, "n": 0})
    assert out["id"] == "a"
    assert out["n"] == 0
    assert isinstance(out["answers"], JsonLeaf)
    assert out["answers"].value == ["x", "y"]
    assert isinstance(out["choices"], JsonLeaf)
    assert out["choices"].value == {"text": ["A", "B"], "label": ["1", "2"]}

    flat = _stabilize_hf_row({"id": "a", "n": 1}, {"id": "", "n": 0})
    assert flat == {"id": "a", "n": 1}


def test_wrap_hf_media_features_keeps_bytes_for_arrow_footer():
    from litdata.types import Audio, Image, infer_type, is_arrow_footer_type
    from litdata.utilities.hf_dataset import _wrap_hf_media_features

    text = _wrap_hf_media_features({"text": "hi", "n": 3})
    assert text == {"text": "hi", "n": 3}
    assert is_arrow_footer_type(infer_type(text))

    img = _wrap_hf_media_features({"img": {"bytes": b"\xff\xd8xx", "path": "a.jpg"}, "label": 1})
    assert img["img"] == {"bytes": b"\xff\xd8xx", "path": "a.jpg"}
    assert not isinstance(img["img"], Image)
    img_t = infer_type(img)
    assert is_arrow_footer_type(img_t)
    assert img_t.fields["img"].kind == "struct"
    assert img_t.fields["img"].fields["bytes"].kind == "bytes"
    assert img_t.fields["img"].fields["path"].kind == "str"

    aud = _wrap_hf_media_features({"audio": {"bytes": b"RIFF....", "path": "a.wav"}})
    assert aud["audio"] == {"bytes": b"RIFF....", "path": "a.wav"}
    assert not isinstance(aud["audio"], Audio)
    assert is_arrow_footer_type(infer_type(aud))

    vid = _wrap_hf_media_features({"video": {"bytes": b"ftyp", "path": "a.mp4"}})
    assert vid["video"] == {"bytes": b"ftyp", "path": "a.mp4"}
    assert is_arrow_footer_type(infer_type(vid))


def test_optimize_hf_media_bytes_stay_arrow_binary(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from litdata.streaming.item_loader import _ARROW_FOOTER_MAGIC

    jpeg = b"\xff\xd8\xff\xdbxx"
    wav = b"RIFF....WAVE"
    table = pa.table(
        {
            "img": [{"bytes": jpeg, "path": "a.jpg"}, {"bytes": jpeg, "path": "b.jpg"}],
            "audio": [{"bytes": wav, "path": "a.wav"}, {"bytes": wav, "path": "b.wav"}],
            "label": [1, 2],
        }
    )
    pq_path = tmp_path / "media.parquet"
    pq.write_table(table, pq_path)
    monkeypatch.setattr(
        "litdata.utilities.hf_dataset.resolve_hf_dataset_url", lambda *a, **k: "hf://datasets/org/media"
    )
    monkeypatch.setattr("litdata.utilities.hf_dataset._prepare_optimize_inputs", lambda *a, **k: [str(pq_path)])
    out = tmp_path / "media-opt"
    optimize_hf("org/media", output_dir=str(out), chunk_size=10, num_workers=1, compression="zstd")
    ds = StreamingDataset(str(out))
    assert len(ds) == 2
    row = ds[0]
    assert row["img"] == {"bytes": jpeg, "path": "a.jpg"}
    assert row["audio"] == {"bytes": wav, "path": "a.wav"}
    assert row["label"] == 1
    assert not hasattr(row["img"], "array")
    types = json.loads((out / "index.json").read_text())["config"]["types"]
    assert types["img"]["type"] == "struct"
    assert types["img"]["fields"]["bytes"] == "bytes"
    assert types["audio"]["fields"]["bytes"] == "bytes"
    fmt = json.loads((out / "index.json").read_text())["config"]["data_format"]
    names = fmt if isinstance(fmt, list) else [fmt]
    assert not any("image" in str(name).lower() or "audio" in str(name).lower() for name in names)
    chunk = next(p for p in out.iterdir() if p.name.endswith(".bin"))
    assert chunk.read_bytes()[-8:] == _ARROW_FOOTER_MAGIC


def test_optimize_hf_variable_length_lists(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": ["a", "b", "c"],
            "answers": [["one"], [], ["two", "three"]],
            "choices": [
                {"text": ["A", "B"], "label": ["1", "2"]},
                {"text": ["C", "D"], "label": ["3", "4"]},
                {"text": ["E", "F"], "label": ["5", "6"]},
            ],
            "n": [1, None, 3],
        }
    )
    pq_path = tmp_path / "nested.parquet"
    pq.write_table(table, pq_path)
    monkeypatch.setattr("litdata.utilities.hf_dataset.resolve_hf_dataset_url", lambda *a, **k: "hf://datasets/org/qa")
    monkeypatch.setattr("litdata.utilities.hf_dataset._prepare_optimize_inputs", lambda *a, **k: [str(pq_path)])
    out = tmp_path / "qa-opt"
    optimize_hf("org/qa", output_dir=str(out), chunk_size=10, num_workers=1)
    ds = StreamingDataset(str(out))
    assert len(ds) == 3
    assert ds[0]["answers"] == ["one"]
    assert ds[1]["answers"] == []
    assert ds[2]["answers"] == ["two", "three"]
    assert ds[0]["choices"] == {"text": ["A", "B"], "label": ["1", "2"]}
    assert ds[1]["n"] == 0
    types = json.loads((out / "index.json").read_text())["config"]["types"]
    assert types["answers"]["type"] == "list"
    assert types["choices"]["type"] == "struct"
    assert types["n"] == "int"  # nulls are filled before write; optionality is for missing keys


def test_materialize_uses_persist_then_falls_back(tmp_path, monkeypatch):
    from litdata.utilities.hf_dataset import _materialize_hf_parquet

    persist = tmp_path / "hf-parquet"
    persist.mkdir()
    local = persist / "0000.parquet"
    local.write_bytes(b"parquet")
    path = _materialize_hf_parquet(
        "hf://datasets/org/name/0000.parquet",
        persist_dir=str(persist),
        index_url="hf://datasets/org/name",
    )
    assert path == str(local)

    def boom(*args, **kwargs):
        raise AssertionError("should not download when persist exists")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    again = _materialize_hf_parquet(
        "hf://datasets/org/name/0000.parquet",
        persist_dir=str(persist),
        index_url="hf://datasets/org/name",
    )
    assert again == str(local)


def test_persist_hf_parquet_skips_complete_file(tmp_path, monkeypatch):
    url = "hf://datasets/org/name"
    dest = hf_parquet_cache_path(url, str(tmp_path))
    os.makedirs(dest, exist_ok=True)
    local = os.path.join(dest, "a.parquet")
    with open(local, "wb") as f:
        f.write(b"0123456789")

    def boom(*args, **kwargs):
        raise AssertionError("should not download a complete file")

    monkeypatch.setattr("litdata.streaming.downloader.get_downloader", boom)
    paths = _persist_hf_parquet_files(url, [{"filename": "a.parquet", "chunk_bytes": 10}], str(tmp_path), {})
    assert paths == [local]


#! TODO: Fix test failing on windows
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows bcoz of urllib.parse")
@pytest.mark.parametrize(
    ("pq_url", "cls", "expectation"),
    [
        ("s3://some_bucket/somepath", CloudParquetDir, nullcontext()),
        ("gs://some_bucket/somepath", CloudParquetDir, nullcontext()),
        ("hf://some_bucket/somepath", HFParquetDir, nullcontext()),
        ("local://some_bucket/somepath", LocalParquetDir, nullcontext()),
        ("/home/some_user/some_bucket/somepath", LocalParquetDir, nullcontext()),
        ("meow://some_bucket/somepath", None, pytest.raises(ValueError, match="The provided")),
    ],
)
def test_get_parquet_indexer_cls(pq_url, tmp_path, cls, expectation, monkeypatch, fsspec_mock, huggingface_hub_fs_mock):
    os = Mock()
    os.listdir = Mock(return_value=[])

    fsspec_fs_mock = Mock()
    fsspec_fs_mock.ls = Mock(return_value=[])
    fsspec_mock.filesystem = Mock(return_value=fsspec_fs_mock)

    hf_fs_mock = Mock()
    hf_fs_mock.ls = Mock(return_value=[])
    huggingface_hub_fs_mock.HfFileSystem = Mock(return_value=hf_fs_mock)

    monkeypatch.setattr("litdata.utilities.parquet.os", os)
    monkeypatch.setattr("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)

    with expectation:
        indexer_obj = get_parquet_indexer_cls(pq_url, tmp_path)
        assert isinstance(indexer_obj, cls)


@pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows bcoz of urllib.parse")
def test_cloud_parquet_dir_forwards_storage_options(tmp_path, monkeypatch, fsspec_mock):
    fsspec_fs_mock = Mock()
    fsspec_fs_mock.ls = Mock(return_value=[])
    fsspec_mock.filesystem = Mock(return_value=fsspec_fs_mock)

    monkeypatch.setattr("litdata.utilities.parquet._FSSPEC_AVAILABLE", True)

    storage_options = {"key": "ACCESS_KEY", "secret": "SECRET_KEY", "endpoint_url": "https://s3.example.com"}
    CloudParquetDir("s3://some_bucket/some_path", tmp_path, storage_options=storage_options)

    fsspec_mock.filesystem.assert_called_once_with("s3", **storage_options)


@pytest.mark.usefixtures("clean_pq_index_cache")
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
@patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
@pytest.mark.parametrize(("pre_load_chunk"), [False, True])
@pytest.mark.parametrize(("low_memory"), [False, True])
def test_stream_hf_parquet_dataset(monkeypatch, huggingface_hub_fs_mock, pq_data, pre_load_chunk, low_memory):
    hf_url = "hf://datasets/some_org/some_repo/some_path"

    # Test case 1: Invalid item_loader
    with pytest.raises(ValueError, match="Invalid item_loader for hf://datasets."):
        StreamingDataset(hf_url, item_loader=PyTreeLoader)

    # Test case 2: Streaming without passing item_loader
    ds = StreamingDataset(hf_url)
    assert len(ds) == 25  # 5 datasets for 5 loops
    for i, _ds in enumerate(ds):
        idx = i % 5
        assert isinstance(_ds, dict)
        assert _ds["name"] == pq_data["name"][idx]
        assert _ds["weight"] == pq_data["weight"][idx]
        assert _ds["height"] == pq_data["height"][idx]

    # Test case 3: Streaming with passing item_loader
    ds = StreamingDataset(hf_url, item_loader=ParquetLoader(pre_load_chunk, low_memory))
    assert len(ds) == 25
    for i, _ds in enumerate(ds):
        idx = i % 5
        assert isinstance(_ds, dict)
        assert _ds["name"] == pq_data["name"][idx]
        assert _ds["weight"] == pq_data["weight"][idx]
        assert _ds["height"] == pq_data["height"][idx]


@pytest.mark.usefixtures("clean_pq_index_cache")
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
@patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
@pytest.mark.parametrize(
    ("hf_url", "length", "context"),
    [
        ("hf://datasets/some_org/some_repo/some_path/*.parquet", 25, nullcontext()),
        ("hf://datasets/some_org/some_repo/some_path/tmp-?.parquet", 25, nullcontext()),
        ("hf://datasets/some_org/some_repo/some_path/tmp-[012].parquet", 15, nullcontext()),
        ("hf://datasets/some_org/some_repo/some_path/tmp-0.parquet", 5, nullcontext()),
        ("hf://datasets/some_org/some_repo/some_path/foo.parquet", 0, pytest.raises(AssertionError, match="No chunks")),
    ],
)
def test_input_dir_wildcard(monkeypatch, huggingface_hub_fs_mock, hf_url, length, context):
    with context:
        ds = StreamingDataset(hf_url)
        pattern = os.path.basename(hf_url)
        assert all(fnmatch(fn, pattern) for fn in ds.subsampled_files)
        assert len(ds) == length  # 5 datasets for 5 loops


@pytest.mark.usefixtures("clean_pq_index_cache")
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
@patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
@pytest.mark.parametrize("default", [False, True])
def test_cache_dir_option(monkeypatch, huggingface_hub_fs_mock, default):
    hf_url = "hf://datasets/some_org/some_repo/some_path"
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = StreamingDataset(hf_url, cache_dir=None if default else tmpdir)
        assert ds.cache_dir.path == (None if default else os.path.realpath(tmpdir))
        assert ds.input_dir.path.startswith(
            (_DEFAULT_CACHE_DIR, _DEFAULT_LIGHTNING_CACHE_DIR) if default else os.path.realpath(tmpdir)
        )
        # check index file is sole file in chunk cache dir
        assert len(os.listdir(ds.input_dir.path)) == 1
        assert os.path.exists(os.path.join(ds.input_dir.path, _INDEX_FILENAME))
        # iterate over dataset — default path downloads parquet into the cache
        for _ in ds:
            pass
        assert len([f for f in os.listdir(ds.input_dir.path) if f.endswith(".parquet")]) == 5


@pytest.mark.parametrize(
    ("pq_url"),
    [
        "s3://some_bucket/some_path",
        "gs://some_bucket/some_path",
        "hf://datasets/some_org/some_repo/some_path",
    ],
)
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
@patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
@patch("litdata.utilities.parquet._FSSPEC_AVAILABLE", True)
def test_no_parquet_files(pq_url, tmpdir, huggingface_hub_fs_mock, fsspec_pq_mock):
    ls_mock = Mock()
    ls_mock.ls = Mock(side_effect=lambda *args, **kwargs: [])
    huggingface_hub_fs_mock.HfFileSystem = Mock(return_value=ls_mock)
    fsspec_pq_mock.filesystem = Mock(return_value=ls_mock)

    with pytest.raises(RuntimeError, match="No Parquet files were found"):
        index_parquet_dataset(pq_url, cache_dir=tmpdir)


def test_hf_relative_name():
    url = "hf://datasets/org/name"
    assert _hf_relative_name(url, "datasets/org/name/data/train-00000.parquet") == "data/train-00000.parquet"
    assert _hf_relative_name(url, "hf://datasets/org/name/train.parquet") == "train.parquet"
    assert _hf_relative_name(url, "train.parquet") == "train.parquet"


def test_parse_hf_url():
    from litdata.utilities.hf_fs import parse_hf_url

    assert parse_hf_url("hf://datasets/org/name/data/train.parquet") == ("org/name", None, "data/train.parquet")
    assert parse_hf_url("hf://datasets/yahma/alpaca-cleaned@refs/convert/parquet/default/train/0000.parquet") == (
        "yahma/alpaca-cleaned",
        "refs/convert/parquet",
        "default/train/0000.parquet",
    )
    assert parse_hf_url("hf://datasets/org/name@main/split/file.parquet") == (
        "org/name",
        "main",
        "split/file.parquet",
    )

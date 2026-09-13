import importlib
import json
import os
from unittest import mock

import litdata.constants
import litdata.utilities
import litdata.utilities.dataset_utilities
from litdata.constants import _DEFAULT_CACHE_DIR, _DEFAULT_LIGHTNING_CACHE_DIR, _INDEX_FILENAME
from litdata.streaming.resolver import Dir
from litdata.utilities.dataset_utilities import (
    _read_updated_at,
    _should_replace_path,
    _try_create_cache_dir,
    adapt_mds_shards_to_chunks,
    generate_roi,
    get_default_cache_dir,
    load_index_file,
)


def test_should_replace_path():
    assert _should_replace_path(None)
    assert _should_replace_path("")
    assert not _should_replace_path(".../datasets/...")
    assert not _should_replace_path(".../s3__connections/...")
    assert _should_replace_path("/teamspace/datasets/...")
    assert _should_replace_path("/teamspace/s3_connections/...")
    assert _should_replace_path("/teamspace/s3_folders/...")
    assert _should_replace_path("/teamspace/gcs_folders/...")
    assert _should_replace_path("/teamspace/gcs_connections/...")
    assert _should_replace_path("/teamspace/lightning_storage/...")
    assert not _should_replace_path("something_else")


def test_try_create_cache_dir():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert os.path.join(
            "chunks", "d41d8cd98f00b204e9800998ecf8427e", "100b8cad7cf2a56f6df78f171f97a1ec"
        ) in _try_create_cache_dir("any")

    # the cache dir creating at /cache requires root privileges, so we need to mock `os.makedirs()`
    with (
        mock.patch.dict("os.environ", {"LIGHTNING_CLUSTER_ID": "abc", "LIGHTNING_CLOUD_PROJECT_ID": "123"}),
        mock.patch("litdata.streaming.dataset.os.makedirs") as makedirs_mock,
    ):
        cache_dir_1 = _try_create_cache_dir("")
        cache_dir_2 = _try_create_cache_dir("ssdf")
        assert cache_dir_1 != cache_dir_2
        assert cache_dir_1 == os.path.join(
            "/cache", "chunks", "d41d8cd98f00b204e9800998ecf8427e", "d41d8cd98f00b204e9800998ecf8427e"
        )
        assert len(makedirs_mock.mock_calls) == 2


def test_try_create_cache_dir_with_custom_cache_dir(tmpdir):
    cache_dir = str(tmpdir.join("cache"))
    with mock.patch.dict(os.environ, {}, clear=True):
        assert os.path.join(
            cache_dir, "d41d8cd98f00b204e9800998ecf8427e", "100b8cad7cf2a56f6df78f171f97a1ec"
        ) in _try_create_cache_dir("any", cache_dir)

    with (
        mock.patch.dict("os.environ", {"LIGHTNING_CLUSTER_ID": "abc", "LIGHTNING_CLOUD_PROJECT_ID": "123"}),
        mock.patch("litdata.streaming.dataset.os.makedirs") as makedirs_mock,
    ):
        cache_dir_1 = _try_create_cache_dir("", cache_dir)
        cache_dir_2 = _try_create_cache_dir("ssdf", cache_dir)
        assert cache_dir_1 != cache_dir_2
        assert cache_dir_1 == os.path.join(
            cache_dir, "d41d8cd98f00b204e9800998ecf8427e", "d41d8cd98f00b204e9800998ecf8427e"
        )
        assert len(makedirs_mock.mock_calls) == 2


def test_generate_roi():
    my_chunks = [
        {"chunk_size": 30},
        {"chunk_size": 50},
        {"chunk_size": 20},
        {"chunk_size": 10},
    ]
    my_roi = generate_roi(my_chunks)

    assert my_roi == [(0, 30), (0, 50), (0, 20), (0, 10)]


def test_load_index_file(tmpdir, mosaic_mds_index_data):
    with open(os.path.join(tmpdir, _INDEX_FILENAME), "w") as f:
        f.write(json.dumps(mosaic_mds_index_data))
    index_data = load_index_file(tmpdir)
    assert "chunks" in index_data
    assert "config" in index_data
    assert len(mosaic_mds_index_data["shards"]) == len(index_data["chunks"])


def test_adapt_mds_shards_to_chunks(mosaic_mds_index_data):
    adapted_data = adapt_mds_shards_to_chunks(mosaic_mds_index_data)
    assert "chunks" in adapted_data
    assert "config" in adapted_data
    assert len(mosaic_mds_index_data["shards"]) == len(adapted_data["chunks"])


def test_get_default_cache_dir():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert get_default_cache_dir() == _DEFAULT_CACHE_DIR

    with mock.patch.dict(os.environ, {"LIGHTNING_CLUSTER_ID": "abc", "LIGHTNING_CLOUD_PROJECT_ID": "123"}):
        assert get_default_cache_dir() == _DEFAULT_LIGHTNING_CACHE_DIR

    with mock.patch.dict(os.environ, {"LITDATA_CACHE_DIR": "/custom/cache/dir"}):
        importlib.reload(litdata.constants)
        importlib.reload(litdata.utilities.dataset_utilities)
        assert litdata.utilities.dataset_utilities.get_default_cache_dir() == "/custom/cache/dir"


def test_read_updated_at_falls_back_on_truncated_local_index(tmp_path, monkeypatch):
    """FUSE can serve a leftover truncated index.json; use the object-store copy."""
    local = tmp_path / "fuse"
    local.mkdir()
    (local / _INDEX_FILENAME).write_text('{"chunks":[{"filename":"chunk-0-0.bin"')

    class FakeDownloader:
        def download_file(self, remote, dest):
            with open(dest, "w") as fh:
                json.dump({"chunks": [], "config": {}, "updated_at": "42"}, fh)

    monkeypatch.setattr(
        "litdata.utilities.dataset_utilities.get_downloader",
        lambda *args, **kwargs: FakeDownloader(),
    )
    d = Dir(path=str(local), url="r2://bucket/ds", data_connection_id="cid")
    assert _read_updated_at(d, {"data_connection_id": "cid"}) == "42"


def test_subsample_streaming_dataset_uses_index_path_for_local_dir(tmp_path):
    """Custom index_path must work when input_dir is a plain local directory (issue #800)."""
    from litdata.utilities.dataset_utilities import subsample_streaming_dataset

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "chunk-0.bin").write_bytes(b"x")

    index_dir = tmp_path / "custom_cache"
    index_dir.mkdir()
    index_payload = {
        "chunks": [
            {
                "filename": "chunk-0.bin",
                "chunk_size": 2,
                "chunk_bytes": 1,
                "dim": None,
            }
        ],
        "config": {},
    }
    index_file = index_dir / _INDEX_FILENAME
    index_file.write_text(json.dumps(index_payload))

    files, roi = subsample_streaming_dataset(
        Dir(path=str(data_dir), url=None),
        index_path=str(index_file),
    )
    assert files == ["chunk-0.bin"]
    assert roi == [(0, 2)]
    # Index was copied into the local input_dir so subsequent loads find it there.
    assert (data_dir / _INDEX_FILENAME).is_file()


def test_subsample_streaming_dataset_uses_index_path_dir_for_local_dir(tmp_path):
    """index_path may be a directory containing index.json (issue #800)."""
    from litdata.utilities.dataset_utilities import subsample_streaming_dataset

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    index_dir = tmp_path / "custom_cache"
    index_dir.mkdir()
    index_payload = {
        "chunks": [
            {
                "filename": "chunk-0.bin",
                "chunk_size": 3,
                "chunk_bytes": 1,
                "dim": None,
            }
        ],
        "config": {},
    }
    (index_dir / _INDEX_FILENAME).write_text(json.dumps(index_payload))

    files, roi = subsample_streaming_dataset(
        Dir(path=str(data_dir), url=None),
        index_path=str(index_dir),
    )
    assert files == ["chunk-0.bin"]
    assert roi == [(0, 3)]

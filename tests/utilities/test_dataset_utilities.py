import importlib
import json
import os
from unittest import mock

import litdata.constants
import litdata.utilities
import litdata.utilities.dataset_utilities
from litdata.constants import _DEFAULT_CACHE_DIR, _DEFAULT_LIGHTNING_CACHE_DIR, _INDEX_FILENAME
from litdata.utilities.dataset_utilities import (
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

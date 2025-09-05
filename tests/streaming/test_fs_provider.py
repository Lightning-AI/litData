from unittest.mock import Mock

import pytest

from litdata.streaming import fs_provider as fs_provider_module
from litdata.streaming.fs_provider import (
    GCPFsProvider,
    R2FsProvider,
    S3FsProvider,
    _get_fs_provider,
    get_bucket_and_path,
    not_supported_provider,
)


def test_get_bucket_and_path():
    bucket, path = get_bucket_and_path("s3://bucket/path/to/file.txt")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"

    bucket, path = get_bucket_and_path("s3://bucket/path/to/file.txt", "s3")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"

    bucket, path = get_bucket_and_path("gs://bucket/path/to/file.txt", "gs")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"

    bucket, path = get_bucket_and_path("r2://bucket/path/to/file.txt", "r2")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"


def test_get_fs_provider(monkeypatch, google_mock):
    google_mock.cloud.storage.Client = Mock()
    monkeypatch.setattr(fs_provider_module, "_GOOGLE_STORAGE_AVAILABLE", True)
    monkeypatch.setattr(fs_provider_module, "S3Client", Mock())

    fs_provider = _get_fs_provider("s3://bucket/path/to/file.txt")
    assert isinstance(fs_provider, S3FsProvider)

    fs_provider = _get_fs_provider("gs://bucket/path/to/file.txt")
    assert isinstance(fs_provider, GCPFsProvider)

    fs_provider = _get_fs_provider("r2://bucket/path/to/file.txt")
    assert isinstance(fs_provider, R2FsProvider)

    with pytest.raises(ValueError, match="Unsupported scheme"):
        _get_fs_provider("http://bucket/path/to/file.txt")


def test_not_supported_provider():
    with pytest.raises(ValueError, match="URL should start with one of"):
        not_supported_provider("http://bucket/path/to/file.txt")

import contextlib
import io
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

from litdata.streaming.downloader import (
    _DOWNLOADERS,
    AzureDownloader,
    Downloader,
    GCPDownloader,
    HFDownloader,
    LocalDownloaderWithCache,
    R2Downloader,
    S3Downloader,
    get_downloader,
    register_downloader,
    shutil,
    unregister_downloader,
)


class DummyDownloader(Downloader):
    def download_file(self, remote_path: str, local_path: str) -> None:
        pass


def test_register_downloader():
    assert "dummy://" not in _DOWNLOADERS
    register_downloader("dummy://", DummyDownloader)
    assert "dummy://" in _DOWNLOADERS
    unregister_downloader("dummy://")
    assert "dummy://" not in _DOWNLOADERS


def test_register_downloader_overwrite():
    register_downloader("dummy://", DummyDownloader)
    with pytest.raises(ValueError, match="Downloader with prefix dummy:// already registered."):
        register_downloader("dummy://", DummyDownloader)

    register_downloader("dummy://", DummyDownloader, overwrite=True)
    assert "dummy://" in _DOWNLOADERS
    unregister_downloader("dummy://")


def test_get_downloader(tmpdir):
    register_downloader("dummy://", DummyDownloader)
    assert isinstance(get_downloader("dummy://dummy", tmpdir, []), DummyDownloader)
    unregister_downloader("dummy://")


def _write_download_target(*args, **kwargs):
    """Side effect for mocked cloud downloads that write to a local path argument."""
    # boto: (bucket, key, filename, ...); gcp blob: (filename,)
    path = args[2] if len(args) >= 3 else args[0]
    with open(path, "wb") as f:
        f.write(b"ok")


@mock.patch("litdata.streaming.downloader.R2Client")
def test_r2_downloader_fast(r2_client_mock, tmpdir):
    # Mock the R2Client
    r2_client_instance = MagicMock()
    r2_client_mock.return_value = r2_client_instance

    # Mock the download_file method to avoid credential errors
    r2_client_instance.client.download_file = MagicMock(side_effect=_write_download_target)

    downloader = R2Downloader("r2://random_bucket", str(tmpdir), [])
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("r2://random_bucket/a.txt", local_filepath)

    # Verify R2Client download_file was called and the final path was published atomically
    r2_client_instance.client.download_file.assert_called_once()
    assert os.path.exists(local_filepath)
    assert r2_client_instance.client.download_file.call_args.args[2].startswith(local_filepath + ".tmp.")


@mock.patch("litdata.streaming.downloader.R2Client")
def test_r2_downloader_with_storage_options(r2_client_mock, tmpdir):
    storage_options = {"data_connection_id": "test_connection_id"}

    # Mock the R2Client
    r2_client_instance = MagicMock()
    r2_client_mock.return_value = r2_client_instance

    # Mock the download_file method to avoid credential errors
    r2_client_instance.client.download_file = MagicMock(side_effect=_write_download_target)

    # Initialize the R2Downloader with storage options
    downloader = R2Downloader("r2://random_bucket", str(tmpdir), [], storage_options)

    # Action: Call the download_file method
    remote_filepath = "r2://random_bucket/sample_file.txt"
    local_filepath = os.path.join(tmpdir, "sample_file.txt")
    downloader.download_file(remote_filepath, local_filepath)

    # Assertion: Verify R2Client was initialized with storage options
    r2_client_mock.assert_called_once_with(storage_options=storage_options, session_options={})

    # Assertion: Verify R2Client download_file was called
    r2_client_instance.client.download_file.assert_called_once()


@mock.patch("litdata.streaming.downloader.R2Client")
def test_r2_downloader_error_handling(r2_client_mock, tmpdir):
    # Mock the R2Client to raise an exception
    r2_client_instance = MagicMock()
    r2_client_mock.return_value = r2_client_instance

    # Mock the download_file method to raise an exception
    r2_client_instance.client.download_file.side_effect = Exception("Simulated R2 error")

    # Initialize the R2Downloader
    downloader = R2Downloader("r2://random_bucket", str(tmpdir), [])

    # Action: Call the download_file method and expect an exception
    remote_filepath = "r2://random_bucket/sample_file.txt"
    local_filepath = os.path.join(tmpdir, "sample_file.txt")

    with pytest.raises(Exception, match="Simulated R2 error"):
        downloader.download_file(remote_filepath, local_filepath)

    # Assertion: Verify R2Client download_file was called
    r2_client_instance.client.download_file.assert_called_once()


@mock.patch("litdata.streaming.downloader.R2Client")
def test_r2_downloader_download_bytes_reuses_client(r2_client_mock, tmpdir):
    r2_client_instance = MagicMock()
    r2_client_mock.return_value = r2_client_instance

    body = MagicMock()
    body.read.return_value = b"hello"
    r2_client_instance.client.get_object.return_value = {"Body": body}

    downloader = R2Downloader("r2://random_bucket", str(tmpdir), [])

    assert downloader.download_bytes("r2://random_bucket/a.txt", 0, 5, os.path.join(tmpdir, "a.txt")) == b"hello"
    assert downloader.download_bytes("r2://random_bucket/a.txt", 5, 5, os.path.join(tmpdir, "a.txt")) == b"hello"

    r2_client_mock.assert_called_once_with(storage_options={}, session_options={})
    assert r2_client_instance.client.get_object.call_args_list == [
        mock.call(Bucket="random_bucket", Key="a.txt", Range="bytes=0-4"),
        mock.call(Bucket="random_bucket", Key="a.txt", Range="bytes=5-9"),
    ]


@mock.patch("litdata.streaming.downloader.S3Client")
def test_s3_downloader_download_bytes_reuses_client(s3_client_mock, tmpdir):
    s3_client_instance = MagicMock()
    s3_client_mock.return_value = s3_client_instance

    body = MagicMock()
    body.read.return_value = b"hello"
    s3_client_instance.client.get_object.return_value = {"Body": body}

    downloader = S3Downloader("s3://random_bucket", str(tmpdir), [])
    # __init__ already creates _client; download_bytes must not recreate it.
    assert hasattr(downloader, "_client")
    client_id = id(downloader._client)

    assert downloader.download_bytes("s3://random_bucket/a.txt", 0, 5, os.path.join(tmpdir, "a.txt")) == b"hello"
    assert downloader.download_bytes("s3://random_bucket/a.txt", 5, 5, os.path.join(tmpdir, "a.txt")) == b"hello"

    assert id(downloader._client) == client_id
    s3_client_mock.assert_called_once_with(storage_options={}, session_options={})
    assert s3_client_instance.client.get_object.call_args_list == [
        mock.call(Bucket="random_bucket", Key="a.txt", Range="bytes=0-4"),
        mock.call(Bucket="random_bucket", Key="a.txt", Range="bytes=5-9"),
    ]


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader_download_bytes_reuses_client(tmpdir, google_mock):
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = b"hello"

    google_mock.cloud.storage.Client = MagicMock(return_value=mock_client)
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    downloader = GCPDownloader("gs://random_bucket", str(tmpdir), [], {"project": "p"})
    assert downloader.download_bytes("gs://random_bucket/a.txt", 0, 5, os.path.join(tmpdir, "a.txt")) == b"hello"
    assert downloader.download_bytes("gs://random_bucket/a.txt", 5, 5, os.path.join(tmpdir, "a.txt")) == b"hello"

    google_mock.cloud.storage.Client.assert_called_once_with(project="p")
    assert mock_blob.download_as_bytes.call_args_list == [
        mock.call(start=0, end=4),
        mock.call(start=5, end=9),
    ]


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader(tmpdir, monkeypatch, google_mock):
    # Create mock objects
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    def _write_ok(path: str) -> None:
        with open(path, "wb") as f:
            f.write(b"ok")

    mock_blob.download_to_filename = MagicMock(side_effect=_write_ok)

    # Patch the storage client to return the mock client
    google_mock.cloud.storage.Client = MagicMock(return_value=mock_client)

    # Configure the mock client to return the mock bucket and blob
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = GCPDownloader("gs://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("gs://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    google_mock.cloud.storage.Client.assert_called_with(**storage_options)
    mock_client.bucket.assert_called_with("random_bucket")
    mock_bucket.blob.assert_called_with("a.txt")
    assert mock_blob.download_to_filename.call_args.args[0].startswith(local_filepath + ".tmp.")
    assert os.path.exists(local_filepath)


@mock.patch("litdata.streaming.downloader._AZURE_STORAGE_AVAILABLE", True)
def test_azure_downloader(tmpdir, monkeypatch, azure_mock):
    mock_blob = MagicMock()
    mock_blob_data = MagicMock()
    mock_blob.download_blob.return_value = mock_blob_data
    service_mock = MagicMock()
    service_mock.get_blob_client.return_value = mock_blob

    azure_mock.storage.blob.BlobServiceClient = MagicMock(return_value=service_mock)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = AzureDownloader("azure://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("azure://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    azure_mock.storage.blob.BlobServiceClient.assert_called_with(**storage_options)
    service_mock.get_blob_client.assert_called_with(container="random_bucket", blob="a.txt")
    mock_blob.download_blob.assert_called()
    mock_blob_data.readinto.assert_called()


def test_download_with_cache(tmpdir, monkeypatch):
    # Create a file to download/cache
    with open("a.txt", "w") as f:
        f.write("hello")

    try:
        local_downloader = LocalDownloaderWithCache(tmpdir, tmpdir, [])
        shutil_mock = MagicMock()
        os_mock = MagicMock()
        monkeypatch.setattr(shutil, "copy", shutil_mock)
        monkeypatch.setattr(os, "rename", os_mock)

        local_downloader.download_file("local:a.txt", os.path.join(tmpdir, "a.txt"))
        shutil_mock.assert_called()
        os_mock.assert_called()
    finally:
        os.remove("a.txt")


@mock.patch("litdata.streaming.downloader._HF_HUB_AVAILABLE", True)
def test_hf_downloader(tmpdir, huggingface_hub_mock):
    # Create a mock for hf_hub_download
    mock_hf_hub_download = MagicMock()
    huggingface_hub_mock.hf_hub_download = mock_hf_hub_download

    # Initialize the downloader
    storage_options = {}
    downloader = HFDownloader("hf://datasets/sample_org/sample_repo", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")

    # Configure the mock to return the local_filepath
    mock_hf_hub_download.return_value = local_filepath

    # Test case 1: File doesn’t exist, should download
    with contextlib.suppress(FileNotFoundError):
        downloader.download_file("hf://datasets/sample_org/sample_repo/a.txt", local_filepath)

    # Verify that hf_hub_download was called with the correct arguments
    huggingface_hub_mock.hf_hub_download.assert_called_once()

    # Reset the mock for the next test case
    mock_hf_hub_download.reset_mock()

    # Test case 2: File exists, should skip download
    with open(local_filepath, "w") as f:
        f.write("dummy content")

    with contextlib.suppress(FileNotFoundError):
        downloader.download_file("hf://datasets/sample_org/sample_repo/a.txt", local_filepath)

    # Verify that hf_hub_download was not called
    mock_hf_hub_download.assert_not_called()


# Test cases for download_fileobj method
def test_s3_downloader_download_fileobj():
    with mock.patch("os.system", return_value=1), mock.patch("litdata.streaming.downloader.S3Client") as S3ClientMock:
        mock_client = MagicMock()
        S3ClientMock.return_value.client = mock_client

        downloader = S3Downloader("s3://bucket", "", [])
        fileobj = io.BytesIO()

        downloader.download_fileobj("s3://bucket/file.txt", fileobj)
        mock_client.download_fileobj.assert_called_once_with("bucket", "file.txt", fileobj)


def test_r2_downloader_download_fileobj():
    with mock.patch("os.system", return_value=1), mock.patch("litdata.streaming.downloader.R2Client") as R2ClientMock:
        mock_client = MagicMock()
        R2ClientMock.return_value.client = mock_client

        downloader = R2Downloader("r2://bucket", "", [])
        fileobj = io.BytesIO()

        downloader.download_fileobj("r2://bucket/file.txt", fileobj)
        mock_client.download_fileobj.assert_called_once_with("bucket", "file.txt", fileobj)


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader_download_fileobj(google_mock):
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    google_mock.cloud.storage.Client = MagicMock(return_value=mock_client)
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    downloader = GCPDownloader("gs://bucket", "", [])
    fileobj = io.BytesIO()

    downloader.download_fileobj("gs://bucket/file.txt", fileobj)
    mock_blob.download_to_file.assert_called_with(fileobj)


@mock.patch("litdata.streaming.downloader._AZURE_STORAGE_AVAILABLE", True)
def test_azure_downloader_download_fileobj(azure_mock):
    mock_blob = MagicMock()
    mock_blob_data = MagicMock()
    mock_blob.download_blob.return_value = mock_blob_data
    service_mock = MagicMock()
    service_mock.get_blob_client.return_value = mock_blob

    azure_mock.storage.blob.BlobServiceClient = MagicMock(return_value=service_mock)

    downloader = AzureDownloader("azure://container", "", [])
    fileobj = io.BytesIO()

    downloader.download_fileobj("azure://container/file.txt", fileobj)
    mock_blob_data.readinto.assert_called_with(fileobj)


@pytest.mark.asyncio
@mock.patch("litdata.streaming.downloader._OBSTORE_AVAILABLE", True)
async def test_s3_downloader_adownload_fileobj(obstore_mock):
    with mock.patch("litdata.streaming.downloader.S3Downloader._get_store") as get_store_mock:
        store_mock = MagicMock()
        get_store_mock.return_value = store_mock
        resp_mock = MagicMock()
        obstore_mock.get_async = mock.AsyncMock(return_value=resp_mock)
        stream_mock = [b"chunk1", b"chunk2"]
        resp_mock.bytes_async = mock.AsyncMock(return_value=b"".join(stream_mock))
        downloader = S3Downloader("s3://bucket", "", [])
        result = await downloader.adownload_fileobj("s3://bucket/file.txt")
        assert isinstance(result, bytes)
        for chunk in stream_mock:
            assert chunk in result


@pytest.mark.asyncio
@mock.patch("litdata.streaming.downloader._OBSTORE_AVAILABLE", True)
async def test_r2_downloader_adownload_fileobj(obstore_mock):
    with mock.patch("litdata.streaming.downloader.R2Downloader._get_store") as get_store_mock:
        store_mock = MagicMock()
        get_store_mock.return_value = store_mock
        resp_mock = MagicMock()
        obstore_mock.get_async = mock.AsyncMock(return_value=resp_mock)
        stream_mock = [b"chunk1", b"chunk2"]
        resp_mock.bytes_async = mock.AsyncMock(return_value=b"".join(stream_mock))
        downloader = R2Downloader("r2://bucket", "", [])
        result = await downloader.adownload_fileobj("r2://bucket/file.txt")
        assert isinstance(result, bytes)
        for chunk in stream_mock:
            assert chunk in result


@pytest.mark.asyncio
@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
async def test_gcp_downloader_adownload_fileobj(obstore_mock):
    with mock.patch("litdata.streaming.downloader.GCPDownloader._get_store") as get_store_mock:
        store_mock = MagicMock()
        get_store_mock.return_value = store_mock
        resp_mock = MagicMock()
        obstore_mock.get_async = mock.AsyncMock(return_value=resp_mock)
        stream_mock = [b"chunk1", b"chunk2"]
        resp_mock.bytes_async = mock.AsyncMock(return_value=b"".join(stream_mock))
        downloader = GCPDownloader("gs://bucket", "", [])
        result = await downloader.adownload_fileobj("gs://bucket/file.txt")
        assert isinstance(result, bytes)
        for chunk in stream_mock:
            assert chunk in result


@pytest.mark.asyncio
@mock.patch("litdata.streaming.downloader._AZURE_STORAGE_AVAILABLE", True)
async def test_azure_downloader_adownload_fileobj(obstore_mock):
    with mock.patch("litdata.streaming.downloader.AzureDownloader._get_store") as get_store_mock:
        store_mock = MagicMock()
        get_store_mock.return_value = store_mock
        resp_mock = MagicMock()
        obstore_mock.get_async = mock.AsyncMock(return_value=resp_mock)
        stream_mock = [b"chunk1", b"chunk2"]
        resp_mock.bytes_async = mock.AsyncMock(return_value=b"".join(stream_mock))
        downloader = AzureDownloader("azure://container", "", [])
        result = await downloader.adownload_fileobj("azure://container/file.txt")
        assert isinstance(result, bytes)
        for chunk in stream_mock:
            assert chunk in result

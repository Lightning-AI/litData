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

import contextlib
import logging
import os
import shutil
import tempfile
import threading
from abc import ABC
from contextlib import suppress
from time import time
from typing import TYPE_CHECKING, Any, cast
from urllib import parse

from filelock import FileLock, Timeout

from litdata.constants import (
    _AZURE_STORAGE_AVAILABLE,
    _DEBUG,
    _GOOGLE_STORAGE_AVAILABLE,
    _HF_HUB_AVAILABLE,
    _INDEX_FILENAME,
    _OBSTORE_AVAILABLE,
)
from litdata.debugger import _get_log_msg
from litdata.streaming.client import R2Client, S3Client

if TYPE_CHECKING:
    from obstore.store import ClientConfig

logger = logging.getLogger("litdata.streaming.downloader")


# Obstore stream yield size. Default matches boto3 multipart chunksize (8MB).
# Override with LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB (integer MiB) for benches.
def _obstore_stream_min_chunk_size() -> int:
    raw = os.getenv("LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB")
    if raw:
        return max(1, int(raw)) * 1024 * 1024
    return 8 * 1024 * 1024


# Obstore default request timeout is 30s; large chunk GETs under worker
# contention can exceed that. Speed-neutral, avoids spurious retries.
_OBSTORE_CLIENT_OPTIONS = cast("ClientConfig", {"timeout": "200s"})


class Downloader(ABC):
    """Cloud/local chunk downloader.

    Implementers should:
    - Publish cache files atomically (temp path + ``os.replace``; see ``_temp_download_path``).
    - Be safe for concurrent calls from multiple threads (or document otherwise).
    - Prefer real HTTP Range in ``download_bytes`` when the backend supports it.
    - Clean up ``.tmp.*`` paths on failure.
    """

    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._storage_options = storage_options or {}

    def _increment_local_lock(self, chunkpath: str, chunk_index: int) -> None:
        countpath = chunkpath + ".cnt"
        with suppress(Timeout, FileNotFoundError), FileLock(countpath + ".lock", timeout=1):
            try:
                with open(countpath) as count_f:
                    curr_count = int(count_f.read().strip())
            except Exception:
                curr_count = 0
            curr_count += 1
            with open(countpath, "w+") as count_f:
                logger.debug(_get_log_msg({"name": f"increment_lock_chunk_{chunk_index}_to_{curr_count}", "ph": "B"}))
                count_f.write(str(curr_count))
                logger.debug(_get_log_msg({"name": f"increment_lock_chunk_{chunk_index}_to_{curr_count}", "ph": "E"}))

    def download_chunk_from_index(self, chunk_index: int) -> None:
        logger.debug(_get_log_msg({"name": f"download_chunk_{chunk_index}", "ph": "B"}))

        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)

        self.download_file(remote_chunkpath, local_chunkpath)

        logger.debug(_get_log_msg({"name": f"download_chunk_{chunk_index}", "ph": "E"}))

    def download_chunk_bytes_from_index(self, chunk_index: int, offset: int, length: int) -> bytes:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)

        return self.download_bytes(remote_chunkpath, offset, length, local_chunkpath)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str) -> None:
        pass

    @staticmethod
    def _temp_download_path(local_filepath: str) -> str:
        """Return a process-unique temp path used for atomic downloads."""
        return f"{local_filepath}.tmp.{os.getpid()}"

    @staticmethod
    def _atomic_replace(tmp_path: str, local_filepath: str) -> None:
        """Publish a completed download by atomically replacing the destination path."""
        os.replace(tmp_path, local_filepath)

    def download_bytes(self, remote_chunkpath: str, offset: int, length: int, local_chunkpath: str) -> bytes:
        """Download a specific range of bytes from the remote file.

        If this method is not overridden in a subclass, it defaults to downloading the full file
        by calling `download_file` and then reading the desired byte range from the local copy.
        """
        self.download_file(remote_chunkpath, local_chunkpath)
        # read the specified byte range from the local file
        with open(local_chunkpath, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def download_fileobj(self, remote_filepath: str, fileobj: Any) -> None:
        """Download a file from remote storage directly to a file-like object."""
        pass

    async def adownload_fileobj(self, remote_filepath: str) -> Any:
        """Download a file from remote storage directly to a file-like object asynchronously."""
        pass

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Async download of ``remote_filepath`` straight to ``local_filepath``.

        Subclasses that can stream should override this to avoid buffering the
        entire object in memory (important for large chunks under
        ``LITDATA_ASYNC_CHUNK_PREFETCH``). The base implementation falls back to
        :meth:`adownload_fileobj` + an atomic write.
        """
        if os.path.exists(local_filepath):
            return
        data = await self.adownload_fileobj(remote_filepath)
        if data is None:
            raise NotImplementedError(
                f"{type(self).__name__}.adownload_fileobj returned None; "
                "override adownload_file or adownload_fileobj for async prefetch."
            )
        tmp_path = self._temp_download_path(local_filepath)
        try:
            os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(data)
            self._atomic_replace(tmp_path, local_filepath)
        except Exception:
            with contextlib.suppress(FileNotFoundError, PermissionError):
                os.remove(tmp_path)
            raise


class S3Downloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        # check if kwargs contains session_options
        self.session_options = kwargs.get("session_options", {})
        self._client = S3Client(storage_options=self._storage_options, session_options=self.session_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0),
        ):
            if os.path.exists(local_filepath):
                return
            # Prefer obstore for sync downloads too (Studio: often faster than
            # boto3 serial on ~64MB chunks). Fall back to boto3 if unavailable.
            tmp_path = self._temp_download_path(local_filepath)
            try:
                os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
                if _OBSTORE_AVAILABLE:
                    import obstore as obs

                    store = self._get_store(obj.netloc)
                    resp = obs.get(store, obj.path.lstrip("/"))
                    with open(tmp_path, "wb", buffering=1024 * 1024) as f:
                        for chunk in resp.stream(min_chunk_size=_obstore_stream_min_chunk_size()):
                            f.write(chunk)
                else:
                    from boto3.s3.transfer import TransferConfig

                    self._client.client.download_file(
                        obj.netloc,
                        obj.path.lstrip("/"),
                        tmp_path,
                        Config=TransferConfig(use_threads=False),
                    )
                self._atomic_replace(tmp_path, local_filepath)
            except Exception:
                with suppress(FileNotFoundError, PermissionError):
                    os.remove(tmp_path)
                raise

    def download_bytes(self, remote_filepath: str, offset: int, length: int, local_chunkpath: str) -> bytes:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        # self._client is created in __init__; S3Client.client serializes create/refresh.
        bucket = obj.netloc
        key = obj.path.lstrip("/")

        byte_range = f"bytes={offset}-{offset + length - 1}"

        response = self._client.client.get_object(Bucket=bucket, Key=key, Range=byte_range)

        return response["Body"].read()

    def download_fileobj(self, remote_filepath: str, fileobj: Any) -> None:
        """Download a file from S3 directly to a file-like object."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        bucket = obj.netloc
        key = obj.path.lstrip("/")

        self._client.client.download_fileobj(
            bucket,
            key,
            fileobj,
        )

    def _get_store(self, bucket: str) -> Any:
        """Return an obstore S3Store instance for the given bucket, initializing if needed."""
        if not hasattr(self, "_store"):
            if not _OBSTORE_AVAILABLE:
                raise ModuleNotFoundError(str(_OBSTORE_AVAILABLE))
            import boto3
            from obstore.auth.boto3 import Boto3CredentialProvider
            from obstore.store import S3Store

            session = boto3.Session(**self._storage_options, **self.session_options)
            credential_provider = Boto3CredentialProvider(session)
            self._store = S3Store(
                bucket,
                credential_provider=credential_provider,
                client_options=_OBSTORE_CLIENT_OPTIONS,
            )
        return self._store

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from S3 into memory asynchronously (prefer :meth:`adownload_file`)."""
        import obstore as obs

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        bucket = obj.netloc
        key = obj.path.lstrip("/")

        store = self._get_store(bucket)
        resp = await obs.get_async(store, key)
        bytes_object = await resp.bytes_async()
        return bytes(bytes_object)  # Convert obstore.Bytes to bytes

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Stream an S3 object to ``local_filepath`` without buffering the full body."""
        import obstore as obs

        if os.path.exists(local_filepath):
            return

        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        bucket = obj.netloc
        key = obj.path.lstrip("/")
        store = self._get_store(bucket)
        tmp_path = self._temp_download_path(local_filepath)
        try:
            os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
            resp = await obs.get_async(store, key)
            with open(tmp_path, "wb", buffering=1024 * 1024) as f:
                async for chunk in resp.stream(min_chunk_size=_obstore_stream_min_chunk_size()):
                    f.write(chunk)
            self._atomic_replace(tmp_path, local_filepath)
        except Exception:
            with suppress(FileNotFoundError, PermissionError):
                os.remove(tmp_path)
            raise


class R2Downloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        # check if kwargs contains session_options
        self.session_options = kwargs.get("session_options", {})
        self._client = R2Client(storage_options=self._storage_options, session_options=self.session_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0),
        ):
            from boto3.s3.transfer import TransferConfig

            extra_args: dict[str, Any] = {}

            if not os.path.exists(local_filepath):
                # Issue: https://github.com/boto/boto3/issues/3113
                t0 = time()
                tmp_path = self._temp_download_path(local_filepath)
                try:
                    self._client.client.download_file(
                        obj.netloc,
                        obj.path.lstrip("/"),
                        tmp_path,
                        ExtraArgs=extra_args,
                        Config=TransferConfig(use_threads=False),
                    )
                    self._atomic_replace(tmp_path, local_filepath)
                except Exception:
                    with suppress(FileNotFoundError, PermissionError):
                        os.remove(tmp_path)
                    raise
                if _DEBUG:
                    print("DOWNLOAD TIME", time() - t0)

    def download_bytes(self, remote_filepath: str, offset: int, length: int, local_chunkpath: str) -> bytes:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        # self._client is created in __init__; R2Client.client serializes create/refresh.
        bucket = obj.netloc
        key = obj.path.lstrip("/")

        byte_range = f"bytes={offset}-{offset + length - 1}"

        response = self._client.client.get_object(Bucket=bucket, Key=key, Range=byte_range)

        return response["Body"].read()

    def download_fileobj(self, remote_filepath: str, fileobj: Any) -> None:
        """Download a file from R2 directly to a file-like object."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        bucket = obj.netloc
        key = obj.path.lstrip("/")

        self._client.client.download_fileobj(
            bucket,
            key,
            fileobj,
        )

    def _get_store(self, bucket: str) -> Any:
        """Return an obstore S3Store instance for the given bucket, initializing if needed."""
        if not hasattr(self, "_store"):
            if not _OBSTORE_AVAILABLE:
                raise ModuleNotFoundError(str(_OBSTORE_AVAILABLE))
            import boto3
            from obstore.auth.boto3 import Boto3CredentialProvider
            from obstore.store import S3Store

            session = boto3.Session(**self._storage_options, **self.session_options)
            credential_provider = Boto3CredentialProvider(session)
            self._store = S3Store(bucket, credential_provider=credential_provider)
        return self._store

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from R2 directly to a file-like object asynchronously."""
        import obstore as obs

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        bucket = obj.netloc
        key = obj.path.lstrip("/")

        store = self._get_store(bucket)
        resp = await obs.get_async(store, key)
        bytes_object = await resp.bytes_async()
        return bytes(bytes_object)  # Convert obstore.Bytes to bytes


class GCPDownloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        self._client: Any | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        """Return a cached ``google.cloud.storage.Client`` (thread-safe lazy init)."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                from google.cloud import storage

                self._client = storage.Client(**self._storage_options)
            return self._client

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected obj.scheme to be `gs`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0),
        ):
            if os.path.exists(local_filepath):
                return

            bucket_name = obj.netloc
            key = obj.path
            # Remove the leading "/":
            if key[0] == "/":
                key = key[1:]

            client = self._get_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(key)
            tmp_path = self._temp_download_path(local_filepath)
            try:
                blob.download_to_filename(tmp_path)
                self._atomic_replace(tmp_path, local_filepath)
            except Exception:
                with suppress(FileNotFoundError, PermissionError):
                    os.remove(tmp_path)
                raise

    def download_bytes(self, remote_filepath: str, offset: int, length: int, local_chunkpath: str) -> bytes:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")

        bucket_name = obj.netloc
        key = obj.path.lstrip("/")

        client = self._get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)

        # GCS uses end as *inclusive*, so end = offset + length - 1
        end = offset + length - 1

        return blob.download_as_bytes(start=offset, end=end)

    def download_fileobj(self, remote_filepath: str, fileobj: Any) -> None:
        """Download a file from GCS directly to a file-like object."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")

        bucket_name = obj.netloc
        key = obj.path.lstrip("/")

        client = self._get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)

        blob.download_to_file(fileobj)

    def _get_store(self, bucket: str) -> Any:
        """Return an obstore GCSStore instance for the given bucket, initializing if needed."""
        if not hasattr(self, "_store"):
            if not _OBSTORE_AVAILABLE:
                raise ModuleNotFoundError(str(_OBSTORE_AVAILABLE))
            from obstore.auth.google import GoogleCredentialProvider
            from obstore.store import GCSStore

            client = self._get_client()
            credential_provider = GoogleCredentialProvider(credentials=client._credentials)
            self._store = GCSStore(bucket, credential_provider=credential_provider)
        return self._store

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from GCS directly to a file-like object asynchronously."""
        import obstore as obs

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")

        bucket_name = obj.netloc
        key = obj.path.lstrip("/")

        store = self._get_store(bucket_name)
        resp = await obs.get_async(store, key)
        bytes_object = await resp.bytes_async()
        return bytes(bytes_object)  # Convert obstore.Bytes to bytes


class AzureDownloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        if not _AZURE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_AZURE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        from azure.storage.blob import BlobServiceClient

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        if os.path.exists(local_filepath):
            return

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0),
        ):
            if os.path.exists(local_filepath):
                return

            service = BlobServiceClient(**self._storage_options)
            blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip("/"))
            tmp_path = self._temp_download_path(local_filepath)
            try:
                with open(tmp_path, "wb") as download_file:
                    blob_data = blob_client.download_blob()
                    blob_data.readinto(download_file)
                self._atomic_replace(tmp_path, local_filepath)
            except Exception:
                with suppress(FileNotFoundError, PermissionError):
                    os.remove(tmp_path)
                raise

    def download_fileobj(self, remote_filepath: str, fileobj: Any) -> None:
        """Download a file from Azure Blob Storage directly to a file-like object."""
        from azure.storage.blob import BlobServiceClient

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        service = BlobServiceClient(**self._storage_options)
        blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip("/"))

        blob_data = blob_client.download_blob()
        blob_data.readinto(fileobj)

    def _get_store(self, bucket: str) -> Any:
        """Return an obstore GCSStore instance for the given bucket, initializing if needed."""
        if not hasattr(self, "_store"):
            if not _OBSTORE_AVAILABLE:
                raise ModuleNotFoundError(str(_OBSTORE_AVAILABLE))
            from obstore.auth.azure import AzureCredentialProvider
            from obstore.store import AzureStore

            # TODO: Check how to pass storage options to AzureCredentialProvider
            credential_provider = AzureCredentialProvider()
            self._store = AzureStore(bucket, credential_provider=credential_provider)
        return self._store

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from Azure Blob Storage directly to a file-like object asynchronously."""
        import obstore as obs

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        bucket_name = obj.netloc
        key = obj.path.lstrip("/")

        store = self._get_store(bucket_name)
        resp = await obs.get_async(store, key)
        bytes_object = await resp.bytes_async()
        return bytes(bytes_object)  # Convert obstore.Bytes to bytes


class LocalDownloader(Downloader):
    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Read a local file (sync I/O; avoids leaking default-executor threads in tests)."""
        from pathlib import Path

        return Path(remote_filepath).read_bytes()

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Copy a local file into the cache path."""
        if os.path.exists(local_filepath):
            return
        self.download_file(remote_filepath, local_filepath)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        lock_path = local_filepath + ".lock"
        lock_acquired = False
        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(lock_path, timeout=1 if remote_filepath.endswith(_INDEX_FILENAME) else 0),
        ):
            lock_acquired = True
            if not (remote_filepath == local_filepath or os.path.exists(local_filepath)):
                # make an atomic operation to be safe
                temp_file_path = local_filepath + ".tmp"
                shutil.copy(remote_filepath, temp_file_path)
                os.rename(temp_file_path, local_filepath)
        # FileLock leaves the lock path behind; remove it after release when we held it.
        # Delete only after the critical section so other waiters do not race a new inode
        # while we still expected exclusive access.
        if lock_acquired:
            with contextlib.suppress(Exception):
                os.remove(lock_path)


class HFDownloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: list[dict[str, Any]],
        storage_options: dict | None = {},
        **kwargs: Any,
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Downloading HF dataset depends on `huggingface_hub`.",
                "Please, run: `pip install huggingface_hub",
            )

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Download a file from the Hugging Face Hub.
        The remote_filepath should be in the format `hf://<repo_type>/<repo_org>/<repo_name>/path`. For more
        information, see
        https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system#integrations.
        """
        from huggingface_hub import hf_hub_download

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "hf":
            raise ValueError(f"Expected obj.scheme to be `hf`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=0),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            _, _, _, repo_org, repo_name, path = remote_filepath.split("/", 5)
            repo_id = f"{repo_org}/{repo_name}"
            downloaded_path = hf_hub_download(
                repo_id,
                path,
                cache_dir=tmpdir,
                repo_type="dataset",
                **self._storage_options,
            )
            if downloaded_path != local_filepath and os.path.exists(downloaded_path):
                temp_file_path = local_filepath + ".tmp"
                shutil.copyfile(downloaded_path, temp_file_path)
                os.rename(temp_file_path, local_filepath)


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


# TODO(follow-up): parametrized Downloader conformance suite over _DOWNLOADERS
# (atomic publish, tmp cleanup, download_bytes correctness + concurrent safety).
_DOWNLOADERS: dict[str, type[Downloader]] = {
    "s3://": S3Downloader,
    "gs://": GCPDownloader,
    "azure://": AzureDownloader,
    "hf://": HFDownloader,
    "local:": LocalDownloaderWithCache,
    "r2://": R2Downloader,
}


def register_downloader(prefix: str, downloader_cls: type[Downloader], overwrite: bool = False) -> None:
    """Register a new downloader class with a specific prefix.

    Args:
        prefix (str): The prefix associated with the downloader.
        downloader_cls (type[Downloader]): The downloader class to register.
        overwrite (bool, optional): Whether to overwrite an existing downloader with the same prefix. Defaults to False.

    Raises:
        ValueError: If a downloader with the given prefix is already registered and overwrite is False.
    """
    if prefix in _DOWNLOADERS and not overwrite:
        raise ValueError(f"Downloader with prefix {prefix} already registered.")

    _DOWNLOADERS[prefix] = downloader_cls


def unregister_downloader(prefix: str) -> None:
    """Unregister a downloader class associated with a specific prefix.

    Args:
        prefix (str): The prefix associated with the downloader to unregister.
    """
    del _DOWNLOADERS[prefix]


def get_downloader(
    remote_dir: str,
    cache_dir: str,
    chunks: list[dict[str, Any]],
    storage_options: dict | None = {},
    session_options: dict | None = {},
) -> Downloader:
    """Get the appropriate downloader instance based on the remote directory prefix.

    Args:
        remote_dir (str): The remote directory URL.
        cache_dir (str): The local cache directory.
        chunks (List[Dict[str, Any]]): List of chunks to managed by the downloader.
        storage_options (Optional[Dict], optional): Additional storage options. Defaults to {}.
        session_options (Optional[Dict], optional): Additional S3 session options. Defaults to {}.

    Returns:
        Downloader: An instance of the appropriate downloader class.
    """
    for k, cls in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return cls(remote_dir, cache_dir, chunks, storage_options, session_options=session_options)
    else:
        # Default to LocalDownloader if no prefix is matched
        return LocalDownloader(remote_dir, cache_dir, chunks, storage_options)

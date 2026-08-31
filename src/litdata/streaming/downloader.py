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
from collections.abc import Callable
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
from litdata.debugger import CAT_DOWNLOAD, CAT_LOCK, emit_trace
from litdata.streaming.client import R2Client, S3Client

if TYPE_CHECKING:
    from obstore.store import ClientConfig, S3Config

logger = logging.getLogger("litdata.streaming.downloader")


# Obstore stream yield size. Default matches boto3 multipart chunksize (8MB).
# Override with LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB (integer MiB) for benches.
def _obstore_stream_min_chunk_size() -> int:
    raw = os.getenv("LITDATA_OBSTORE_STREAM_MIN_CHUNK_MIB")
    if raw:
        return max(1, int(raw)) * 1024 * 1024
    return 8 * 1024 * 1024


def _write_obstore_chunk(fileobj: Any, chunk: Any) -> None:
    """Write a stream yield without an extra ``bytes()`` copy when possible."""
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        fileobj.write(chunk)
        return
    try:
        fileobj.write(memoryview(chunk))
    except TypeError:
        fileobj.write(bytes(chunk))


def _obstore_stream_resp_to_tmp(tmp_path: str, resp: Any) -> None:
    with open(tmp_path, "wb") as f:
        for chunk in resp.stream(min_chunk_size=_obstore_stream_min_chunk_size()):
            _write_obstore_chunk(f, chunk)


async def _obstore_astream_resp_to_tmp(tmp_path: str, resp: Any) -> None:
    with open(tmp_path, "wb") as f:
        async for chunk in resp.stream(min_chunk_size=_obstore_stream_min_chunk_size()):
            _write_obstore_chunk(f, chunk)


async def _obstore_adownload_file(downloader: "Downloader", store: Any, key: str, local_filepath: str) -> None:
    """Stream an object to ``local_filepath`` (prefer this over :meth:`Downloader.adownload_fileobj`)."""
    import obstore as obs

    if os.path.exists(local_filepath):
        return
    tmp_path = downloader._temp_download_path(local_filepath)
    try:
        os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
        resp = await obs.get_async(store, key)
        await _obstore_astream_resp_to_tmp(tmp_path, resp)
        downloader._publish_file(tmp_path, local_filepath)
    except Exception:
        with suppress(FileNotFoundError, PermissionError):
            os.remove(tmp_path)
        raise


async def _obstore_adownload_bytes(store: Any, key: str) -> bytes:
    """In-memory GET. Callers that write to disk should use :func:`_obstore_adownload_file`."""
    import obstore as obs

    resp = await obs.get_async(store, key)
    return bytes(await resp.bytes_async())


async def _obstore_aiter_file_chunks(local_filepath: str) -> Any:
    """Yield ``local_filepath`` in the same chunk size used for download streams."""
    chunk_size = _obstore_stream_min_chunk_size()
    with open(local_filepath, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return
            yield chunk


async def _obstore_aupload_file(store: Any, key: str, local_filepath: str) -> None:
    """Stream ``local_filepath`` to ``key`` (same chunking as download-to-tmp)."""
    import obstore as obs

    await obs.put_async(store, key, _obstore_aiter_file_chunks(local_filepath))


# Obstore default request timeout is 30s; large chunk GETs under worker
# contention can exceed that. Speed-neutral, avoids spurious retries.
_OBSTORE_CLIENT_OPTIONS = cast("ClientConfig", {"timeout": "200s"})

# PID that first built an obstore store in this process lineage. Obstore's
# Rust/tokio runtime is process-global and not fork-safe: a new S3Store in a
# forked child still hangs if the parent already started tokio. So we:
#   * never start obstore for ``index.json`` (parent DataLoader setup)
#   * lazy-create a store in the process that first downloads a chunk
#   * fall back to boto3 only if this process forked *after* that init
_OBSTORE_INIT_PID: int | None = None


def obstore_usable() -> bool:
    """Return whether obstore may be used in *this* process.

    True when obstore is installed and this process did not inherit a tokio
    runtime started by its parent (fork). Spawn children start clean.
    Re-instantiating ``S3Store`` after fork is not enough — the runtime is
    process-global, not the Python object.
    """
    if not _OBSTORE_AVAILABLE:
        return False
    return _OBSTORE_INIT_PID is None or os.getpid() == _OBSTORE_INIT_PID


def _use_obstore_for_s3_key(object_path: str) -> bool:
    """Whether this S3 GET should go through obstore.

    ``index.json`` is fetched in the DataLoader parent before fork. Using
    obstore there starts tokio in the parent and poisons every worker.
    Chunk downloads then lazy-init obstore in the worker (or in the parent
    when ``num_workers=0``).
    """
    return obstore_usable() and not object_path.endswith(_INDEX_FILENAME)


def _note_obstore_init() -> None:
    global _OBSTORE_INIT_PID
    if _OBSTORE_INIT_PID is None:
        _OBSTORE_INIT_PID = os.getpid()


def _discard_forked_obstore_store(downloader: Any) -> None:
    """Drop an S3Store inherited across fork so it cannot be reused."""
    if getattr(downloader, "_store_pid", None) == os.getpid():
        return
    if hasattr(downloader, "_store"):
        del downloader._store
    downloader._store_pid = os.getpid()


def _obstore_credential_provider(s3_client: S3Client) -> Any:
    """Return an obstore credential callback that follows ``S3Client``/``R2Client`` refresh.

    LitData's boto3 wrappers resolve Studio temp credentials, ``data_connection_id``,
    custom endpoints, and IMDS. The async/obstore path must reuse that client instead
    of dumping ``storage_options`` into ``boto3.Session`` (which rejects LitData
    metadata keys such as ``data_connection_id`` and client-only keys such as
    ``endpoint_url``).
    """

    def _provider() -> dict[str, Any]:
        from datetime import datetime, timedelta, timezone

        boto_client = s3_client.client
        frozen = boto_client._get_credentials().get_frozen_credentials()
        if frozen.access_key is None or frozen.secret_key is None:
            raise ValueError("boto3 client returned incomplete credentials")
        return {
            "access_key_id": frozen.access_key,
            "secret_access_key": frozen.secret_key,
            "token": frozen.token,
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=30),
        }

    return _provider


def _build_obstore_s3_store(bucket: str, s3_client: S3Client) -> Any:
    """Build an obstore ``S3Store`` that matches an already-configured boto3 S3 client.

    Sync downloads go through ``S3Client``/``R2Client``. This helper is the async
    equivalent: same credentials, endpoint, and region, so R2, S3-compatible
    endpoints, and Lightning data connections work on the prefetch path.
    """
    from obstore.store import S3Store

    boto_client = s3_client.client
    endpoint_url = boto_client.meta.endpoint_url
    region = boto_client.meta.region_name
    config: dict[str, Any] = {}
    if region:
        config["region"] = region
    if endpoint_url:
        config["endpoint"] = endpoint_url
        # Path-style addressing is required for R2 and most S3-compatible endpoints.
        if "amazonaws.com" not in endpoint_url:
            config["virtual_hosted_style_request"] = False

    return S3Store(
        bucket,
        config=cast("S3Config", config) if config else None,
        credential_provider=_obstore_credential_provider(s3_client),
        client_options=_OBSTORE_CLIENT_OPTIONS,
    )


def _cached_obstore_store(downloader: Any, factory: Any) -> Any:
    """Return a process-local obstore store, or raise if this process forked after init."""
    _discard_forked_obstore_store(downloader)
    if not obstore_usable():
        raise RuntimeError(
            "obstore is not fork-safe after the parent process initialized it; fall back to the SDK client"
        )
    if not hasattr(downloader, "_store"):
        if not _OBSTORE_AVAILABLE:
            raise ModuleNotFoundError(str(_OBSTORE_AVAILABLE))
        downloader._store = factory()
        downloader._store_pid = os.getpid()
        _note_obstore_init()
    return downloader._store


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
        # Set by ChunksConfig: called after an atomic publish so waiters can Event.wait
        # instead of polling the filesystem.
        self._on_file_published: Callable[[str], None] | None = None

    def __getstate__(self) -> dict[str, Any]:
        # Spawn workers must not inherit a live obstore S3Store / tokio runtime.
        state = self.__dict__.copy()
        state.pop("_store", None)
        state.pop("_store_pid", None)
        # Bound method / lock-holding callback is process-local; rebound in ChunksConfig.__setstate__.
        state.pop("_on_file_published", None)
        return state

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
                emit_trace("lock", "B", CAT_LOCK, op="increment", chunk=chunk_index, count=curr_count)
                count_f.write(str(curr_count))
                emit_trace("lock", "E", CAT_LOCK, op="increment", chunk=chunk_index, count=curr_count)

    def download_chunk_from_index(self, chunk_index: int) -> None:
        emit_trace("download", "B", CAT_DOWNLOAD, chunk=chunk_index)

        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)

        self.download_file(remote_chunkpath, local_chunkpath)
        self._notify_published(local_chunkpath)

        emit_trace("download", "E", CAT_DOWNLOAD, chunk=chunk_index)

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
        try:
            os.replace(tmp_path, local_filepath)
        except FileNotFoundError:
            # Same-pid gather of a duplicate index, or another worker already published.
            if os.path.exists(local_filepath):
                return
            raise

    def _notify_published(self, local_filepath: str) -> None:
        """Signal that ``local_filepath`` is visible (atomic rename finished, or already present)."""
        callback = getattr(self, "_on_file_published", None)
        if callback is not None:
            callback(local_filepath)

    def _publish_file(self, tmp_path: str, local_filepath: str) -> None:
        """Atomically publish ``tmp_path`` and notify in-process waiters."""
        self._atomic_replace(tmp_path, local_filepath)
        self._notify_published(local_filepath)

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

    async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
        """Async upload of ``local_filepath`` to ``remote_filepath``.

        Cloud subclasses that have an obstore store should override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support async upload")

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Async download of ``remote_filepath`` straight to ``local_filepath``.

        Subclasses that can stream should override this to avoid buffering the
        entire object in memory (important for large chunks under
        ``LITDATA_ASYNC_CHUNK_PREFETCH``). The base implementation falls back to
        :meth:`adownload_fileobj` + an atomic write.
        """
        if os.path.exists(local_filepath):
            self._notify_published(local_filepath)
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
            self._publish_file(tmp_path, local_filepath)
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
            # Prefer obstore for chunk GETs (Studio: often faster than boto3
            # serial on ~64MB objects). Index fetches stay on boto3 so the
            # DataLoader parent does not start tokio before fork.
            tmp_path = self._temp_download_path(local_filepath)
            try:
                os.makedirs(os.path.dirname(local_filepath) or ".", exist_ok=True)
                if _use_obstore_for_s3_key(obj.path):
                    import obstore as obs

                    store = self._get_store(obj.netloc)
                    resp = obs.get(store, obj.path.lstrip("/"))
                    _obstore_stream_resp_to_tmp(tmp_path, resp)
                else:
                    from boto3.s3.transfer import TransferConfig

                    self._client.client.download_file(
                        obj.netloc,
                        obj.path.lstrip("/"),
                        tmp_path,
                        Config=TransferConfig(use_threads=False),
                    )
                self._publish_file(tmp_path, local_filepath)
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
        return _cached_obstore_store(self, lambda: _build_obstore_s3_store(bucket, self._client))

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from S3 into memory asynchronously (prefer :meth:`adownload_file`)."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        return await _obstore_adownload_bytes(self._get_store(obj.netloc), obj.path.lstrip("/"))

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Stream an S3 object to ``local_filepath`` without buffering the full body."""
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        await _obstore_adownload_file(self, self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)

    async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")
        await _obstore_aupload_file(self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)


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
                    self._publish_file(tmp_path, local_filepath)
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
        return _cached_obstore_store(self, lambda: _build_obstore_s3_store(bucket, self._client))

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from R2 into memory asynchronously (prefer :meth:`adownload_file`)."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        return await _obstore_adownload_bytes(self._get_store(obj.netloc), obj.path.lstrip("/"))

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Stream an R2 object to ``local_filepath`` without buffering the full body."""
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")

        await _obstore_adownload_file(self, self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)

    async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "r2":
            raise ValueError(f"Expected obj.scheme to be `r2`, instead, got {obj.scheme} for remote={remote_filepath}")
        await _obstore_aupload_file(self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)


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
                self._publish_file(tmp_path, local_filepath)
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

        def _factory() -> Any:
            from obstore.auth.google import GoogleCredentialProvider
            from obstore.store import GCSStore

            client = self._get_client()
            credential_provider = GoogleCredentialProvider(credentials=client._credentials)
            return GCSStore(bucket, credential_provider=credential_provider)

        return _cached_obstore_store(self, _factory)

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from GCS into memory asynchronously (prefer :meth:`adownload_file`)."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")

        return await _obstore_adownload_bytes(self._get_store(obj.netloc), obj.path.lstrip("/"))

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Stream a GCS object to ``local_filepath`` without buffering the full body."""
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")

        await _obstore_adownload_file(self, self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)

    async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "gs":
            raise ValueError(f"Expected scheme 'gs', got '{obj.scheme}' for remote={remote_filepath}")
        await _obstore_aupload_file(self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)


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
                self._publish_file(tmp_path, local_filepath)
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
        """Return an obstore AzureStore instance for the given bucket, initializing if needed."""

        def _factory() -> Any:
            from obstore.auth.azure import AzureCredentialProvider
            from obstore.store import AzureStore

            # TODO: Check how to pass storage options to AzureCredentialProvider
            credential_provider = AzureCredentialProvider()
            return AzureStore(bucket, credential_provider=credential_provider)

        return _cached_obstore_store(self, _factory)

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Download a file from Azure into memory asynchronously (prefer :meth:`adownload_file`)."""
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        return await _obstore_adownload_bytes(self._get_store(obj.netloc), obj.path.lstrip("/"))

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Stream an Azure object to ``local_filepath`` without buffering the full body."""
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        await _obstore_adownload_file(self, self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)

    async def aupload_file(self, local_filepath: str, remote_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )
        await _obstore_aupload_file(self._get_store(obj.netloc), obj.path.lstrip("/"), local_filepath)


class LocalDownloader(Downloader):
    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Read a local file (sync I/O; avoids leaking default-executor threads in tests)."""
        from pathlib import Path

        return Path(remote_filepath).read_bytes()

    async def adownload_file(self, remote_filepath: str, local_filepath: str) -> None:
        """Copy a local file into the cache path."""
        if os.path.exists(local_filepath):
            self._notify_published(local_filepath)
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
                self._publish_file(temp_file_path, local_filepath)
        # FileLock leaves the lock path behind; remove it after release when we held it.
        # Delete only after the critical section so other waiters do not race a new inode
        # while we still expected exclusive access.
        if lock_acquired:
            with contextlib.suppress(Exception):
                os.remove(lock_path)
        if os.path.exists(local_filepath):
            self._notify_published(local_filepath)


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
                self._publish_file(temp_file_path, local_filepath)


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

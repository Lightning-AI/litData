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
import os
import shutil
import subprocess
from abc import ABC
from contextlib import suppress
from typing import Any, Dict, List, Optional, Type
from urllib import parse

from filelock import FileLock, Timeout

from litdata.constants import (
    _AZURE_STORAGE_AVAILABLE,
    _DISABLE_S5CMD,
    _DOWNLOAD_RETRIES,
    _GOOGLE_STORAGE_AVAILABLE,
    _HF_HUB_AVAILABLE,
    _INDEX_FILENAME,
)
from litdata.streaming.client import S3Client


class Downloader(ABC):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._storage_options = storage_options or {}

    def _increment_local_lock(self, chunkpath: str) -> None:
        countpath = chunkpath + ".cnt"
        with suppress(Timeout), FileLock(countpath + ".lock", timeout=1):
            try:
                with open(countpath) as count_f:
                    curr_count = int(count_f.read().strip())
            except Exception:
                curr_count = 0
            curr_count += 1
            with open(countpath, "w+") as count_f:
                count_f.write(str(curr_count))

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)

        self.download_file(remote_chunkpath, local_chunkpath, chunk_filename)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str, remote_chunk_filename: str = "") -> None:
        pass


class S3Downloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        self._s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

        if not self._s5cmd_available or _DISABLE_S5CMD:
            self._client = S3Client(storage_options=self._storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            for _attempt in range(_DOWNLOAD_RETRIES):
                if os.path.exists(local_filepath):
                    return

                if self._s5cmd_available and not _DISABLE_S5CMD:
                    env = None
                    if self._storage_options:
                        env = os.environ.copy()
                        env.update(self._storage_options)

                    aws_no_sign_request = self._storage_options.get("AWS_NO_SIGN_REQUEST", "no").lower() == "yes"
                    # prepare the s5cmd command
                    no_signed_option = "--no-sign-request" if aws_no_sign_request else None
                    cmd_parts = ["s5cmd", no_signed_option, "--log", "info", "cp", remote_filepath, local_filepath]
                    cmd = " ".join(part for part in cmd_parts if part)

                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    return_code = proc.wait()

                    if return_code != 0:
                        stderr_output = proc.stderr.read().decode().strip() if proc.stderr else ""
                        error_message = (
                            f"Failed to execute command `{cmd}` (exit code: {return_code}). "
                            "This might be due to an incorrect file path, insufficient permissions, or network issues. "
                            "To resolve this issue, you can either:\n"
                            "- Pass `storage_options` with the necessary credentials and endpoint. \n"
                            "- Example:\n"
                            "  storage_options = {\n"
                            '      "AWS_ACCESS_KEY_ID": "your-key",\n'
                            '      "AWS_SECRET_ACCESS_KEY": "your-secret",\n'
                            '      "S3_ENDPOINT_URL": "https://s3.example.com" (Optional if using AWS)\n'
                            "  }\n"
                            "- or disable `s5cmd` by setting `DISABLE_S5CMD=1` if `storage_options` do not work.\n"
                        )
                        if stderr_output:
                            error_message += (
                                f"For further debugging, please check the command output below:\n{stderr_output}"
                            )
                        raise RuntimeError(error_message)
                else:
                    from boto3.s3.transfer import TransferConfig

                    extra_args: Dict[str, Any] = {}

                    if not os.path.exists(local_filepath):
                        # Issue: https://github.com/boto/boto3/issues/3113
                        self._client.client.download_file(
                            obj.netloc,
                            obj.path.lstrip("/"),
                            local_filepath,
                            ExtraArgs=extra_args,
                            Config=TransferConfig(use_threads=False),
                        )


class GCPDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        from google.cloud import storage

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected obj.scheme to be `gs`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            for _attempt in range(_DOWNLOAD_RETRIES):
                if os.path.exists(local_filepath):
                    return
                
                bucket_name = obj.netloc
                key = obj.path
                # Remove the leading "/":
                if key[0] == "/":
                    key = key[1:]

                client = storage.Client(**self._storage_options)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(key)
                blob.download_to_filename(local_filepath)


class AzureDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _AZURE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_AZURE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        from azure.storage.blob import BlobServiceClient

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=1 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            for _attempt in range(_DOWNLOAD_RETRIES):
                if os.path.exists(local_filepath):
                    return

                service = BlobServiceClient(**self._storage_options)
                blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip("/"))
                with open(local_filepath, "wb") as download_file:
                    blob_data = blob_client.download_blob()
                    blob_data.readinto(download_file)


class LocalDownloader(Downloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=1 if remote_filepath.endswith(_INDEX_FILENAME) else 0
        ):
            for _attempt in range(_DOWNLOAD_RETRIES):
                if remote_filepath == local_filepath or os.path.exists(local_filepath):
                    return
                # make an atomic operation to be safe
                temp_file_path = local_filepath + ".tmp"
                shutil.copy(remote_filepath, temp_file_path)
                os.rename(temp_file_path, local_filepath)
                with contextlib.suppress(Exception):
                    os.remove(local_filepath + ".lock")


class HFDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Downloading HF dataset depends on `huggingface_hub`.",
                "Please, run: `pip install huggingface_hub",
            )

        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        # for HF dataset downloading, we don't need remote_filepath, but remote_chunk_filename
        with suppress(Timeout), FileLock(local_filepath + ".lock", timeout=0):
            for _attempt in range(_DOWNLOAD_RETRIES):
                if os.path.exists(local_filepath):
                    return

                temp_path = local_filepath + ".tmp"  # Avoid partial writes
                try:
                    with self.fs.open(remote_chunk_filename, "rb") as cloud_file, open(temp_path, "wb") as local_file:
                        for chunk in iter(lambda: cloud_file.read(4096), b""):  # Stream in 4KB chunks local_file.
                            local_file.write(chunk)

                    os.rename(temp_path, local_filepath)  # Atomic move after successful write

                except Exception as e:
                    print(f"Error processing {remote_chunk_filename}: {e}")

                finally:
                    # Ensure cleanup of temp file if an error occurs
                    if os.path.exists(temp_path):
                        os.remove(temp_path)


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


_DOWNLOADERS: Dict[str, Type[Downloader]] = {
    "s3://": S3Downloader,
    "gs://": GCPDownloader,
    "azure://": AzureDownloader,
    "hf://": HFDownloader,
    "local:": LocalDownloaderWithCache,
}


def register_downloader(prefix: str, downloader_cls: Type[Downloader], overwrite: bool = False) -> None:
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
    remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
) -> Downloader:
    """Get the appropriate downloader instance based on the remote directory prefix.

    Args:
        remote_dir (str): The remote directory URL.
        cache_dir (str): The local cache directory.
        chunks (List[Dict[str, Any]]): List of chunks to managed by the downloader.
        storage_options (Optional[Dict], optional): Additional storage options. Defaults to {}.

    Returns:
        Downloader: An instance of the appropriate downloader class.
    """
    for k, cls in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return cls(remote_dir, cache_dir, chunks, storage_options)
    else:
        # Default to LocalDownloader if no prefix is matched
        return LocalDownloader(remote_dir, cache_dir, chunks, storage_options)

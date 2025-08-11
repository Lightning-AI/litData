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
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from litdata.constants import _FSSPEC_AVAILABLE, _TQDM_AVAILABLE, _ZSTD_AVAILABLE

logger = logging.getLogger(__name__)
_SUPPORTED_PROVIDERS = ("s3", "gs", "azure")
_INDEX_FILENAME = "index.json.zstd"


@dataclass
class FileMetadata:
    """Metadata for a single file in the dataset."""

    path: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        return cls(path=data["path"], size=data["size"])


class BaseIndexer(ABC):
    """Abstract base class for file indexing strategies."""

    @abstractmethod
    def discover_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Discover dataset files and return their metadata."""

    def build_or_load_index(
        self,
        input_dir: str,
        cache_dir: str,
        storage_options: Optional[dict[str, Any]],
        recompute_index: bool = False,
    ) -> list[FileMetadata]:
        """Loads or builds a ZSTD-compressed index of dataset file metadata.
        This method attempts to load an existing index from local or remote cache, or builds a new one if needed.
        Use `recompute_index=True` to force rebuilding the index from the input directory.

        Args:
            input_dir: Path to the dataset root directory.
            cache_dir: Directory for storing the index cache.
            storage_options: Optional storage backend options.
            recompute_index: If True, always rebuild the index.

        Returns:
            List of FileMetadata objects for discovered files.

        Raises:
            ModuleNotFoundError: If required dependencies are missing.
            ValueError: If no files are found in the input directory.
        """
        if not _ZSTD_AVAILABLE:
            raise ModuleNotFoundError(str(_ZSTD_AVAILABLE))

        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))

        parsed_url = urlparse(input_dir)
        if parsed_url.scheme and parsed_url.scheme not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported input directory scheme: `{parsed_url.scheme}`. "
                f"Supported schemes are: {_SUPPORTED_PROVIDERS}"
            )

        if not recompute_index:
            files = self._load_index_from_cache(input_dir, cache_dir, storage_options)
            if files:
                return files

        return self._build_and_cache_index(input_dir, cache_dir, storage_options)

    def _load_index_from_cache(
        self, input_dir: str, cache_dir: str, storage_options: Optional[dict[str, Any]]
    ) -> Optional[list[FileMetadata]]:
        """Tries to load the index from local or remote cache."""
        # 1. Try to load index from local cache.
        local_index_path = Path(cache_dir) / _INDEX_FILENAME
        if local_index_path.exists():
            logger.info(f"Loading index from local cache: {local_index_path}")
            files = self._load_index_file(str(local_index_path))
            if files:
                logger.info(f"Loaded index from local cache: {local_index_path}")
                return files

        # 2. If not found, try remote cache.
        remote_index_path = os.path.join(input_dir, _INDEX_FILENAME)
        try:
            self._download_from_cloud(remote_index_path, str(local_index_path), storage_options)
            files = self._load_index_file(str(local_index_path))
            if files:
                logger.info(f"Loaded index from remote cache: {remote_index_path}")
                return files
        except FileNotFoundError:
            logger.warning(f"Remote index not found at {remote_index_path}")
        except Exception as e:
            logger.error(f"Failed to download or load remote index: {e}")

        return None

    def _build_and_cache_index(
        self, input_dir: str, cache_dir: str, storage_options: Optional[dict[str, Any]]
    ) -> list[FileMetadata]:
        """Builds a new index and caches it locally and remotely."""
        local_index_path = Path(cache_dir) / _INDEX_FILENAME
        logger.info(f"Building index for {input_dir} at {local_index_path}")
        files = self.discover_files(input_dir, storage_options)
        if not files:
            raise ValueError(f"No files found in {input_dir}")

        self._save_index_file(str(local_index_path), files, input_dir)

        # Upload to remote cache
        remote_index_path = os.path.join(input_dir, _INDEX_FILENAME)
        try:
            self._upload_to_cloud(str(local_index_path), remote_index_path, storage_options)
            logger.info(f"Uploaded index to remote cache: {remote_index_path}")
        except Exception as e:
            logger.warning(f"Failed to upload index to remote cache: {e}")

        logger.info(f"Built index with {len(files)} files from {input_dir} at {local_index_path}")
        return files

    def _load_index_file(self, index_path: str) -> Optional[list[FileMetadata]]:
        """Loads and decodes an index file."""
        import zstd

        try:
            with open(index_path, "rb") as f:
                compressed_data = f.read()
            metadata = json.loads(zstd.decompress(compressed_data).decode("utf-8"))
            return [FileMetadata.from_dict(file_data) for file_data in metadata["files"]]
        except (FileNotFoundError, json.JSONDecodeError, zstd.ZstdError, KeyError) as e:
            logger.warning(f"Failed to load index from local cache at `{index_path}`: {e}. ")
            return None

    def _save_index_file(self, index_path: str, files: list[FileMetadata], source: str) -> None:
        """Encodes and saves an index file."""
        import zstd

        try:
            metadata = {
                "source": source,
                "files": [file.to_dict() for file in files],
                "created_at": time.time(),
            }
            with open(index_path, "wb") as f:
                f.write(zstd.compress(json.dumps(metadata).encode("utf-8")))
        except (OSError, zstd.ZstdError) as e:
            logger.warning(f"Error caching index to {index_path}: {e}")

    def _download_from_cloud(
        self,
        remote_path: str,
        local_path: str,
        storage_options: Optional[dict[str, Any]],
    ) -> None:
        """Downloads a file from cloud storage."""
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))
        import fsspec

        parsed_url = urlparse(remote_path)
        fs = fsspec.filesystem(parsed_url.scheme, **(storage_options or {}))
        fs.get(remote_path, local_path)

    def _upload_to_cloud(
        self,
        local_path: str,
        remote_path: str,
        storage_options: Optional[dict[str, Any]],
    ) -> None:
        """Uploads a file to cloud storage."""
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))
        import fsspec

        parsed_url = urlparse(remote_path)
        fs = fsspec.filesystem(parsed_url.scheme, **(storage_options or {}))
        fs.put(local_path, remote_path)


class FileIndexer(BaseIndexer):
    """Indexes files recursively from cloud or local storage with optional extension filtering."""

    def __init__(
        self,
        max_depth: int = 5,
        extensions: Optional[list[str]] = None,
    ):
        self.max_depth = max_depth
        self.extensions = [ext.lower() for ext in (extensions or [])]

    def discover_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Discover dataset files and return their metadata."""
        parsed_url = urlparse(input_dir)
        if parsed_url.scheme and parsed_url.scheme not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported input directory scheme: `{parsed_url.scheme}`. "
                f"Supported schemes are: {_SUPPORTED_PROVIDERS}"
            )

        if parsed_url.scheme in _SUPPORTED_PROVIDERS:  # Cloud storage
            return self._discover_cloud_files(input_dir, storage_options)

        # Local filesystem
        return self._discover_local_files(input_dir)

    def _discover_cloud_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Recursively list files in a cloud storage bucket."""
        import fsspec

        obj = urlparse(input_dir)

        # TODO: Research on switching to 'obstore' for file listing to potentially improve performance.
        # Currently using 'fsspec' due to some issues with 'obstore' when handling multiple instances.
        fs = fsspec.filesystem(obj.scheme, **(storage_options or {}))
        files = fs.find(input_dir, maxdepth=self.max_depth, detail=True, withdirs=False)

        if _TQDM_AVAILABLE:
            from tqdm.auto import tqdm

            pbar = tqdm(desc="Discovering files", total=len(files))

        metadatas = []
        for _, file_info in files.items():
            if file_info.get("type") != "file":
                continue

            file_path = file_info["name"]
            if self._should_include_file(file_path):
                metadata = FileMetadata(
                    path=f"{obj.scheme}://{file_path}",
                    size=file_info.get("size", 0),
                )
                metadatas.append(metadata)
            if _TQDM_AVAILABLE:
                pbar.update(1)
        if _TQDM_AVAILABLE:
            pbar.close()
        return metadatas

    def _discover_local_files(self, input_dir: str) -> list[FileMetadata]:
        """Recursively list files in the local filesystem."""
        path = Path(input_dir)
        metadatas = []

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            if self._should_include_file(str(file_path)):
                metadata = FileMetadata(
                    path=str(file_path),
                    size=file_path.stat().st_size,
                )
                metadatas.append(metadata)

        return metadatas

    def _should_include_file(self, file_path: str) -> bool:
        """Return True if file matches allowed extensions and is not an index file."""
        path = Path(file_path)
        if path.name == _INDEX_FILENAME:
            return False
        file_ext = path.suffix.lower()
        return not self.extensions or file_ext in self.extensions

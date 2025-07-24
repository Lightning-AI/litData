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

import asyncio
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse

import fsspec
import zstd
from torch.utils.data import Dataset
from tqdm import tqdm

from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming.downloader import Downloader, get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("s3", "gs", "azure")


@dataclass(slots=True)
class FileMetadata:
    """Metadata for a single file in the dataset."""

    path: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            size=data["size"],
        )


class BaseIndexer(ABC):
    """Abstract base class for file indexing strategies."""

    @abstractmethod
    def discover_files(self, input_dir: str, storage_options: dict[str, Any]) -> list[FileMetadata]:
        """Discover files and return metadata."""

    @abstractmethod
    def get_cache_key(self) -> str:
        """Get a unique cache key for this indexer configuration."""

    def build_or_load_index(
        self, input_dir: str, cache_dir: str, storage_options: dict[str, Any]
    ) -> list[FileMetadata]:
        """Build or load cached file index using ZSTD compression."""
        index_path = os.path.join(cache_dir, "index.json.zstd")
        if not _ZSTD_AVAILABLE:
            raise ModuleNotFoundError(
                "ZSTD compression is required to use dataset indexing. "
                "Please install the zstd package by running: pip install zstd"
            )

        # Check if cached index exists and is fresh
        if os.path.exists(index_path):
            try:
                with open(index_path, "rb") as f:
                    compressed_data = f.read()
                metadata = json.loads(zstd.decompress(compressed_data).decode("utf-8"))

                if metadata.get("cache_key") == self.get_cache_key():
                    logger.info(f"Loaded cached index with {len(metadata['files'])} files from {index_path}")
                    return [FileMetadata.from_dict(file_data) for file_data in metadata["files"]]
            except Exception as e:
                logger.warning(f"Error loading cached index: {e}")

        # Build fresh index
        logger.info(f"Building index for {input_dir} at {index_path}")
        files = self.discover_files(input_dir, storage_options)
        if not files:
            raise ValueError(f"No files found in {input_dir}")

        # Cache the index with ZSTD compression
        try:
            metadata = {
                "cache_key": self.get_cache_key(),
                "source": input_dir,
                "files": [file.to_dict() for file in files],
                "created_at": time.time(),
            }
            with open(os.path.join(index_path), "wb") as f:
                f.write(zstd.compress(json.dumps(metadata).encode("utf-8")))
        except Exception as e:
            logger.warning(f"Error caching index: {e}")

        logger.info(f"Built index with {len(files)} files from {input_dir} at {index_path}")
        return files


class FileIndexer(BaseIndexer):
    """File indexer that discovers files recursively by extension and depth."""

    def __init__(
        self,
        max_depth: int = 5,
        extensions: Optional[list[str]] = None,
    ):
        self.max_depth = max_depth
        self.extensions = [ext.lower() for ext in (extensions or [])]

    def discover_files(self, input_dir: str, storage_options: dict[str, Any] = {}) -> list[FileMetadata]:
        """Discover files using recursive search."""
        parsed_url = urlparse(input_dir)

        if parsed_url.scheme in SUPPORTED_PROVIDERS:
            return self._discover_cloud_files(input_dir, storage_options)
        return self._discover_local_files(input_dir)

    def _discover_cloud_files(self, input_dir: str, storage_options: dict[str, Any]) -> list[FileMetadata]:
        """Discover files in cloud storage."""
        parsed_url = urlparse(input_dir)
        fs = fsspec.filesystem(parsed_url.scheme, **storage_options)
        files = fs.find(
            input_dir, maxdepth=self.max_depth, detail=True, withdirs=False
        )  # returns dict with file details

        all_metadata = []
        for _, file_info in tqdm(files.items(), desc="Discovering files"):
            if file_info.get("type") != "file":
                continue

            file_path = file_info["name"]

            if self._should_include_file(file_path):
                metadata = FileMetadata(
                    path=f"{parsed_url.scheme}://{file_path}",
                    size=file_info.get("size", 0),
                )
                all_metadata.append(metadata)

        return all_metadata

    def _discover_local_files(self, input_dir: str) -> list[FileMetadata]:
        """Discover files in local filesystem."""
        path = Path(input_dir)
        all_metadata = []

        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue

            # Filter by depth
            if self.max_depth is not None:
                rel_depth = len(file_path.relative_to(path).parts)
                if rel_depth > self.max_depth + 1:  # +1 to account for file name in parts
                    continue

            if self._should_include_file(str(file_path)):
                metadata = FileMetadata(
                    path=str(file_path),
                    size=file_path.stat().st_size,
                )
                all_metadata.append(metadata)

        return all_metadata

    def _should_include_file(self, file_path: str) -> bool:
        """Check if file should be included based on extension filters."""
        file_ext = Path(file_path).suffix.lower()
        return not self.extensions or file_ext in self.extensions

    def get_cache_key(self) -> str:
        """Get a unique cache key for this indexer configuration."""
        config_str = f"{self.max_depth}_{'_'.join(self.extensions) if self.extensions else 'all'}"
        return generate_md5_hash(config_str)


class CacheManager:
    """Manages local file caching with directory structure preservation."""

    def __init__(
        self,
        input_dir: Union[Dir, str],
        cache_dir: Optional[Union[Dir, str]] = None,
        storage_options: Optional[dict] = None,
        cache_files: bool = False,
    ):
        self.input_dir = _resolve_dir(input_dir)
        self._input_dir_path = self.input_dir.path or self.input_dir.url
        self.cache_files = cache_files

        self.cache_dir = self._create_cache_dir(self._input_dir_path, cache_dir)

        self.storage_options = storage_options or {}
        self._downloader: Optional[Downloader] = None
        self._loop = None
        self._closed = False

    @property
    def downloader(self) -> Downloader:
        """Lazily initialize the downloader."""
        if self._downloader is None:
            self._downloader = get_downloader(
                remote_dir=self._input_dir_path,
                cache_dir=self.cache_dir,
                chunks=[],
                storage_options=self.storage_options,
            )
        return self._downloader

    def _create_cache_dir(self, input_dir: str, cache_dir: Optional[str] = None) -> str:
        """Create cache directory if it doesn't exist."""
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        cache_path = os.path.join(cache_dir, generate_md5_hash(input_dir))
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def get_local_path(self, file_path: str) -> str:
        """Map a remote file path to a local cache path."""
        prefix = self._input_dir_path.rstrip("/") + "/"
        if not file_path.startswith(prefix):
            raise ValueError(f"File path {file_path} does not start with input dir {prefix}")

        relative_path = file_path[len(prefix) :].lstrip("/")
        local_path = os.path.join(self.cache_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return local_path

    async def download_file_async(self, file_path: str) -> bytes:
        """Download file asynchronously and return content."""
        if self.cache_files:
            local_path = self.get_local_path(file_path)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    return f.read()

        # Download to BytesIO
        file_obj = io.BytesIO()
        try:
            await self.downloader.adownload_fileobj(file_path, file_obj)
            return file_obj.getvalue()
        except Exception as e:
            raise RuntimeError(f"Error downloading file {file_path}: {e}") from e

    def __del__(self) -> None:
        """Close the event loop when the object is destroyed."""
        if not self._closed:
            self.close()

    def close(self) -> None:
        """Close the downloader and the event loop."""
        if self._closed:
            return
        if self._downloader:
            if self._loop and self._loop.is_running():
                self._loop.create_task(self._downloader.close())
            else:
                asyncio.run(self._downloader.close())
        if self._loop:
            self._loop.stop()
            self._loop.close()
        self._closed = True


class StreamingRawDataset(Dataset):
    """Stream raw files from cloud storage with fast indexing and caching.

    Supports any folder structure, automatically indexing individual files.

    Features:
    - Simple synchronous __getitem__ for single items
    - Efficient async __getitems__ for batch operations
    - Clean resource management
    - Minimal memory footprint
    """

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        cache_dir: Optional[Union[str, "Dir"]] = None,
        indexer: Optional[BaseIndexer] = None,
        storage_options: Optional[dict] = None,
        cache_files: bool = False,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g. s3://bucket/dataset/)
            cache_dir: Directory for caching files (optional)
            indexer: Custom file indexer (default: FileIndexer)
            storage_options: Cloud storage options
            cache_files: Whether to cache files locally (default: False)
            transform: A function to apply to each downloaded item.
        """
        # Resolve directories
        self.input_dir = _resolve_dir(input_dir)
        self.cache_manager = CacheManager(self.input_dir, cache_dir, storage_options, cache_files)

        # Configuration
        self.indexer = indexer or FileIndexer()
        self.storage_options = storage_options or {}
        self.transform = transform

        # Discover files and build index
        self.files = self.indexer.build_or_load_index(
            self.input_dir.path or self.input_dir.url,
            self.cache_manager.cache_dir,
            self.storage_options,
        )

        logger.info(f"Initialized StreamingRawDataset with {len(self.files)} files")

    @lru_cache(maxsize=1)
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        """Get single item by index - simple synchronous download."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")

        file_path = self.files[index].path
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(self.cache_manager.download_file_async(file_path))
        if self.transform:
            return self.transform(data)
        return data

    def __getitems__(self, indices: list[int]) -> list[Any]:
        """Get multiple items efficiently using async batch download."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._download_batch(indices))

    async def _download_batch(self, indices: list[int]) -> list[Any]:
        """Asynchronously download and transform items."""
        file_paths = [self.files[index].path for index in indices]
        coros = [self._process_item(path) for path in file_paths]
        return await asyncio.gather(*coros)

    async def _process_item(self, file_path: str) -> Any:
        """Download a single file and apply the transform."""
        data = await self.cache_manager.download_file_async(file_path)
        if self.transform:
            return await asyncio.to_thread(self.transform, data)
        return data

    def __del__(self) -> None:
        """Close the cache manager when the object is destroyed."""
        self.cache_manager.close()

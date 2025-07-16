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
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import fsspec
import zstd
from torch.utils.data import Dataset
from tqdm import tqdm

from litdata.streaming.downloader import get_downloader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import generate_md5_hash, get_default_cache_dir

logger = logging.getLogger(__name__)

INDEX_METADATA_FILE = "index_metadata.json.zstd"


@dataclass(slots=True)
class FileMetadata:
    """Metadata for a single file in the dataset."""

    path: str
    size: int
    modified_time: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "size": self.size,
            "modified_time": self.modified_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            size=data["size"],
            modified_time=data.get("modified_time"),
            metadata=(data.get("metadata") or {}).copy(),  # Ensure metadata is a copy to avoid shared references
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
        index_path = os.path.join(cache_dir, INDEX_METADATA_FILE)

        # Check if cached index exists and is fresh
        if os.path.exists(index_path):
            try:
                logger.info(f"Loading cached index from {index_path}")
                # Decompress and load
                with open(index_path, "rb") as f:
                    compressed_data = f.read()
                json_data = zstd.decompress(compressed_data)
                cached_data = json.loads(json_data.decode("utf-8"))

                # Verify cache key matches current configuration
                if cached_data.get("cache_key") == self.get_cache_key():
                    files = [FileMetadata.from_dict(file_data) for file_data in cached_data["files"]]
                    logger.info(f"Loaded cached index with {len(files)} files from {index_path}")
                    return files
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
    """File indexer that discovers files recursively by extension and depth.

    - Supports both local and cloud storage (S3, GCS, etc.)
    - Filters files by extension if provided.
    - Uses max_depth for recursion in cloud storage.
    """

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

        if parsed_url.scheme in ("s3", "gs", "gcs"):
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
            modified_time = file_info.get("LastModified")
            modified_time = modified_time.timestamp() if modified_time else None

            if self._should_include_file(file_path):
                metadata = FileMetadata(
                    path=f"{parsed_url.scheme}://{file_path}",
                    size=file_info.get("size", 0),
                    modified_time=modified_time,
                    metadata={"etag": file_info.get("ETag")},
                )
                all_metadata.append(metadata)

        return all_metadata

    def _discover_local_files(self, input_dir: str) -> list[FileMetadata]:
        """Discover files in local filesystem."""
        path = Path(input_dir)
        all_metadata = []
        # Use rglob for recursive search, respects max_depth by filtering
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            # Filter by depth
            if self.max_depth is not None:
                rel_depth = len(file_path.relative_to(path).parts)
                if rel_depth > self.max_depth:
                    continue
            if self._should_include_file(str(file_path)):
                metadata = FileMetadata(
                    path=str(file_path),
                    size=file_path.stat().st_size,
                    modified_time=file_path.stat().st_mtime,
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
    ):
        self.input_dir = _resolve_dir(input_dir)
        self.cache_dir = self._try_create_cache_dir(self.input_dir.path or self.input_dir.url, cache_dir)
        assert self.cache_dir is not None, "Cache directory must be specified or created"
        self.storage_options = storage_options or {}
        self.downloader = None

    def _try_create_cache_dir(self, input_dir: str, cache_dir: Optional[str] = None) -> str:
        """Create cache directory if it doesn't exist."""
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        cache_path = os.path.join(cache_dir, generate_md5_hash(input_dir))
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def get_local_path(self, file_path: str, class_name: str) -> str:
        """Get local cache path maintaining class directory structure."""
        filename = os.path.basename(file_path)
        class_cache_dir = os.path.join(self.cache_dir, class_name)
        os.makedirs(class_cache_dir, exist_ok=True)
        return os.path.join(class_cache_dir, filename)

    def download_file(self, file_path: str, class_name: str) -> str:
        """Download file to cache and return local path."""
        local_path = self.get_local_path(file_path, class_name)

        # Return if already cached
        if os.path.exists(local_path):
            return local_path

        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Create a temporary downloader for this specific file
            filename = os.path.basename(file_path)

            # Create chunk info for this specific file
            chunks = [{"filename": filename}]

            # Get downloader instance
            if self.downloader is None:
                # Initialize downloader only once
                self.downloader = get_downloader(
                    remote_dir=self.remote_dir,
                    cache_dir=self.cache_dir,
                    chunks=chunks,
                    storage_options=self.storage_options,
                )

            # Download the file
            self.downloader.download_file(file_path, local_path)

            # Verify download was successful
            if os.path.exists(local_path):
                return local_path

            raise FileNotFoundError(f"Downloaded file not found at {local_path}")

        except Exception as e:
            logger.warning(f"Failed to download {file_path}: {e}")
            return file_path  # Return remote path as fallback


class StreamingRawDataset(Dataset):
    """Stream raw files from cloud storage with fast indexing and caching.

    Supports any folder structure, automatically indexing individual files.

    Features:
    - Fast multithreaded indexing
    - Automatic index caching
    - Preloading for performance
    - PyTorch DataLoader compatible
    """

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        cache_dir: Optional[Union[str, "Dir"]] = None,
        indexer: Optional[BaseIndexer] = None,
        max_preload_size: int = 10,
        storage_options: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g. s3://bucket/dataset/)
            cache_dir: Directory for caching files (optional)
            indexer: Custom file indexer (default: FileIndexer)
            max_preload_size: Maximum number of files to preload (default: 10)
            storage_options: Cloud storage options
            **kwargs: Additional arguments
        """
        # Resolve directories
        self.input_dir = _resolve_dir(input_dir)
        self.cache_manager = CacheManager(self.input_dir, cache_dir, storage_options)

        # Configuration
        self.indexer = indexer or FileIndexer()
        self.max_preload_size = max_preload_size
        self.storage_options = storage_options or {}

        # Preloading with adaptive performance tracking
        self._preload_executor = None
        self._preload_futures = {}
        self._cache_hit_count = 0
        self._total_requests = 0

        # Discover files and build index
        self.files = self.indexer.build_or_load_index(
            self.input_dir.path or self.input_dir.url, self.cache_manager.cache_dir, self.storage_options
        )

        logger.info(f"Initialized StreamingRawDataset with {len(self.files)} samples")

    def _init_preloading(self) -> None:
        """Initialize preloading executor with adaptive worker count."""
        if self._preload_executor is None and self.max_preload_size > 0:
            # Adaptive worker count based on dataset size
            worker_count = min(self.download_workers, len(self.files) // 100 + 1, 8)
            self._preload_executor = ThreadPoolExecutor(max_workers=worker_count)

            # Start preloading first files with adaptive batch size
            initial_batch = min(self.max_preload_size, len(self.files), 50)
            for i in range(initial_batch):
                self._submit_preload(i)

    def _submit_preload(self, index: int) -> None:
        """Submit a file for preloading."""
        if index < len(self.files) and index not in self._preload_futures:
            file_path, class_name = self.files[index]
            future = self._preload_executor.submit(self.cache_manager.download_file, file_path, class_name)
            self._preload_futures[index] = future

    def _get_file(self, index: int) -> str:
        """Get local file path with adaptive preloading strategy."""
        file_path, class_name = self.files[index]
        self._total_requests += 1

        # Check if preloaded
        if index in self._preload_futures:
            future = self._preload_futures.pop(index)
            try:
                local_path = future.result(timeout=30)
                self._cache_hit_count += 1

                # Adaptive preloading: adjust based on cache hit rate
                cache_hit_rate = self._cache_hit_count / self._total_requests
                preload_window = min(self.max_preload_size * 2, 100) if cache_hit_rate > 0.8 else self.max_preload_size

                # Queue next files for preloading with smart scheduling
                for offset in range(1, min(4, preload_window // 4 + 1)):
                    next_idx = index + offset * preload_window // 4
                    if next_idx < len(self.files):
                        self._submit_preload(next_idx)

                return local_path
            except Exception as e:
                logger.warning(f"Preload failed for {index}: {e}")

        # Download directly
        return self.cache_manager.download_file(file_path, class_name)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics for monitoring."""
        hit_rate = self._cache_hit_count / max(self._total_requests, 1)
        return {
            "cache_hit_rate": hit_rate,
            "total_requests": self._total_requests,
            "preload_queue_size": len(self._preload_futures),
            "cache_dir": self.cache_manager.cache_dir,
            "adaptive_preload_window": min(self.max_preload_size * 2, 100) if hit_rate > 0.8 else self.max_preload_size,
        }

    def load_sample(self, local_path: str, file_path: str, class_name: str, index: int) -> Any:
        """Load sample data. Override this method to customize loading.

        Args:
            local_path: Path to local cached file
            file_path: Original remote file path
            class_name: Class name
            index: Sample index

        Returns:
            Loaded sample data
        """
        # Default: return file content as bytes
        try:
            from PIL import Image

            image = Image.open(local_path)
            return {"index": index, "path": local_path, "image": image, "label": class_name}
        except Exception:
            # Fallback for remote files
            return {"path": file_path, "class_name": class_name, "index": index}

    def __iter__(self):
        """Iterate over dataset."""
        # Initialize preloading on first iteration
        if self._preload_executor is None:
            self._init_preloading()

        for i in range(len(self.files)):
            yield self[i]

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        if index >= len(self.files):
            raise IndexError(f"Index {index} out of range")

        # Initialize preloading only when actually needed
        if self._preload_executor is None:
            self._init_preloading()

        file_path, class_name = self.files[index]
        local_path = self._get_file(index)
        return self.load_sample(local_path, file_path, class_name, index)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.files)

    def get_class_counts(self) -> dict[str, int]:
        """Get samples per class."""
        counts = {}
        for _, class_name in self.files:
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


if __name__ == "__main__":
    # Example usage on litserve teamspace
    import time

    logging.basicConfig(level=logging.INFO)

    start = time.perf_counter()
    dataset = StreamingRawDataset(
        input_dir="s3://imagenet-1m-template/raw/train",
        # input_dir="s3://grid-cloud-litng-ai-03/projects/01jpacd4y2yza88t23wf049m0t/datasets/caltech101/101_ObjectCategories",
        cache_dir="cache",
        max_preload_size=20,
    )
    print(f"Discovered {len(dataset.files)} files", dataset.files[:5])
    end = time.perf_counter()
    print(f"Dataset loaded in {end - start:.2f} seconds")
    # print("sample files :", dataset.files[:3])

    # print(f"Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")

    # # Test iteration
    # for i, sample in enumerate(dataset):
    #     if i >= 3:
    #         break
    #     print(f"Sample {i}: {sample} ({len(sample) if isinstance(sample, bytes) else 'metadata'})")

    # print("✅ Test completed")

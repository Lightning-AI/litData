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
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
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
        self._input_dir_path = self.input_dir.path or self.input_dir.url
        self.cache_dir = self._try_create_cache_dir(self._input_dir_path, cache_dir)
        assert self.cache_dir is not None, "Cache directory must be specified or created"
        self.storage_options = storage_options or {}
        self._downloader = None

    def _try_create_cache_dir(self, input_dir: str, cache_dir: Optional[str] = None) -> str:
        """Create cache directory if it doesn't exist."""
        if cache_dir is None:
            cache_dir = get_default_cache_dir()
        cache_path = os.path.join(cache_dir, generate_md5_hash(input_dir))
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def get_local_path(self, file_path: str) -> str:
        """Get local cache path maintaining class directory structure."""
        try:
            # Remove the input directory prefix from file_path to preserve relative directory structure in cache
            file_path = file_path.lstrip(self._input_dir_path)
            # Ensure cache directory exists
            os.makedirs(os.path.join(self.cache_dir, os.path.dirname(file_path)), exist_ok=True)
            return os.path.join(self.cache_dir, file_path)
        except Exception as e:
            logger.error(f"Error getting local path for {file_path}: {e}")
            raise

    def download_file(self, file_path: str) -> str:
        """Download file to cache and return local path."""
        local_path = self.get_local_path(file_path)

        # Return if already cached
        if os.path.exists(local_path):
            return local_path
        try:
            # Get downloader instance
            if self._downloader is None:
                # Initialize downloader only once
                self._downloader = get_downloader(
                    remote_dir=self._input_dir_path,
                    cache_dir=self.cache_dir,
                    chunks=[],
                    storage_options=self.storage_options,
                )

            # Download the file
            # logger.info(f"Downloading {file_path} to {local_path}")
            self._downloader.download_file(file_path, local_path)
            # logger.info(f"Downloaded {file_path} to {local_path}")
            return local_path

        except Exception as e:
            logger.exception(f"Error downloading file {file_path}: {e}")
            return None


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
        download_timeout: int = 30,
        **kwargs,
    ):
        """Initialize StreamingRawDataset.

        Args:
            input_dir: Path to dataset root (e.g. s3://bucket/dataset/)
            cache_dir: Directory for caching files (optional)
            indexer: Custom file indexer (default: FileIndexer)
            max_preload_size: Maximum number of files to preload (default: 10)
            storage_options: Cloud storage options
            download_timeout: Timeout for downloading files (default: 30 seconds)
            **kwargs: Additional arguments
        """
        # Resolve directories
        self.input_dir = _resolve_dir(input_dir)
        self.cache_manager = CacheManager(self.input_dir, cache_dir, storage_options)

        # Configuration
        self.indexer = indexer or FileIndexer()
        self.max_preload_size = max_preload_size
        self.storage_options = storage_options or {}
        self.download_timeout = download_timeout

        # Preloading with adaptive performance tracking
        self._preload_executor = None
        self._preload_futures = {}
        self._cache_hit_count = 0
        self._total_requests = 0

        # Discover files and build index
        self.files = self.indexer.build_or_load_index(
            self.input_dir.path or self.input_dir.url, self.cache_manager.cache_dir, self.storage_options
        )

        # Queue
        self._request_queue = Queue()
        self._loop_thread = None
        self._lock = threading.Lock()

        logger.info(f"Initialized StreamingRawDataset with {len(self.files)} samples")

    @lru_cache(maxsize=1)
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")

        # start the background thread + event loop once
        with self._lock:
            if self._loop_thread is None:
                # Start the preloading executor in a separate thread
                self._loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
                self._loop_thread.start()

        # create a pre-request event and send it to the queue wit the index
        event = threading.Event()
        self._request_queue.put((index, event))

        # Wait for the event to be set by the preloading thread
        if not event.wait(timeout=self.download_timeout):
            raise TimeoutError(f"Timeout waiting for file at index {index} to be preloaded")

        return event.data

        # TODO: cleanup later
        # file_path = self.files[index].path
        # local_path = self.cache_manager.get_local_path(file_path)
        # self.cache_manager.download_file(file_path)
        # if not os.path.exists(local_path):
        #     raise FileNotFoundError(f"File not found in cache: {local_path}")
        # with open(local_path, "rb") as f:
        #     data = f.read()
        # return data

    def _start_event_loop(self) -> None:
        """Event loop for handling preloading requests."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # run the event loop until stopped
        loop.run_until_complete(self._handle_requests())

    async def _handle_requests(self):
        """Handle requests from the queue."""
        event_loop = asyncio.get_event_loop()
        pending_requests = set()
        while True:
            # print("Waiting for requests...")
            try:
                index, event = await event_loop.run_in_executor(
                    None, self._request_queue.get, 0.1
                )  # Non-blocking get with timeout
            except Empty:
                continue

            if index is None:
                print("Received stop signal, exiting request handler")
                # Stop signal received
                break

            if index < 0 or index >= len(self):
                logger.error(f"Invalid index {index} requested")
                continue

            # Get the remote and local paths
            remote_path = self.files[index].path
            # process incoming request asynchronously to enable concurrent downloads
            task = asyncio.create_task(self._download_file(remote_path, event), name=f"download_task_{index}")

            # save reference to the task result to prevent it from being garbage collected
            pending_requests.add(task)
            task.add_done_callback(pending_requests.discard)

    async def _download_file(self, remote_path: str, event: threading.Event) -> str:
        """Download file asynchronously and set event data."""
        try:
            event_loop = asyncio.get_event_loop()
            local_path = await event_loop.run_in_executor(None, self.cache_manager.download_file, remote_path)
            with open(local_path, "rb") as f:
                event.data = f.read()
            event.set()
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file {remote_path}: {e}")
            raise

    def _cleanup(self):
        """Internal cleanup method."""
        try:
            if self._loop_thread and self._loop_thread.is_alive():
                # Signal to stop the loop
                self._request_queue.put((None, None))
                # Wait for the thread to finish
                self._loop_thread.join(timeout=2)
                if self._loop_thread.is_alive():
                    logger.warning("Thread did not terminate gracefully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        finally:
            self._loop_thread = None
            self._request_queue = None
            logger.info("StreamingRawDataset cleaned up successfully.")

    def close(self):
        """Explicitly close the dataset and clean up resources."""
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Clean up resources."""
        self._cleanup()


if __name__ == "__main__":
    # Example usage on litserve teamspace
    import time

    logging.basicConfig(level=logging.INFO)

    start = time.perf_counter()
    with StreamingRawDataset(
        # input_dir="s3://imagenet-1m-template/raw/train",
        input_dir="s3://grid-cloud-litng-ai-03/projects/01jpacd4y2yza88t23wf049m0t/datasets/caltech101/101_ObjectCategories",
        cache_dir="cache",
        max_preload_size=20,
    ) as dataset:
        print("Dataset initialized")
        print(f"Dataset size: {len(dataset)}")
        print(f"Discovered {len(dataset.files)} files", dataset.files[:1])
        end = time.perf_counter()
        print(f"Dataset loaded in {end - start:.2f} seconds")
        # print("sample files :", dataset.files[:3])
        sample = dataset[0]
        print("sample", type(sample))  # Access the first item to trigger preloading
        print("Dataset cleaned up")
        print("✅ Test completed")

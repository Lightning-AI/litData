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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from litdata.constants import _FSSPEC_AVAILABLE, _TQDM_AVAILABLE, _ZSTD_AVAILABLE

logger = logging.getLogger(__name__)
_SUPPORTED_PROVIDERS = ("s3", "gs", "azure")


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
        self, input_dir: str, cache_dir: str, storage_options: Optional[dict[str, Any]]
    ) -> list[FileMetadata]:
        """Build or load a ZSTD-compressed index of file metadata."""
        if not _ZSTD_AVAILABLE:
            raise ModuleNotFoundError(str(_ZSTD_AVAILABLE))

        import zstd

        index_path = Path(cache_dir) / "index.json.zstd"

        # Try loading cached index if it exists
        if index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    compressed_data = f.read()
                metadata = json.loads(zstd.decompress(compressed_data).decode("utf-8"))

                return [FileMetadata.from_dict(file_data) for file_data in metadata["files"]]
            except (FileNotFoundError, json.JSONDecodeError, zstd.ZstdError, KeyError) as e:
                logger.warning(f"Failed to load cached index from {index_path}: {e}")

        # Build fresh index
        logger.info(f"Building index for {input_dir} at {index_path}")
        files = self.discover_files(input_dir, storage_options)
        if not files:
            raise ValueError(f"No files found in {input_dir}")

        # Cache the index with ZSTD compression
        # TODO: upload the index to cloud storage
        try:
            metadata = {
                "source": input_dir,
                "files": [file.to_dict() for file in files],
                "created_at": time.time(),
            }
            with open(index_path, "wb") as f:
                f.write(zstd.compress(json.dumps(metadata).encode("utf-8")))
        except (OSError, zstd.ZstdError) as e:
            logger.warning(f"Error caching index to {index_path}: {e}")

        logger.info(f"Built index with {len(files)} files from {input_dir} at {index_path}")
        return files


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

        if parsed_url.scheme in _SUPPORTED_PROVIDERS:  # Cloud storage
            return self._discover_cloud_files(input_dir, storage_options)

        if not parsed_url.scheme:  # Local filesystem
            return self._discover_local_files(input_dir)

        raise ValueError(
            f"Unsupported input directory scheme: {parsed_url.scheme}. Supported schemes are: {_SUPPORTED_PROVIDERS}"
        )

    def _discover_cloud_files(self, input_dir: str, storage_options: Optional[dict[str, Any]]) -> list[FileMetadata]:
        """Recursively list files in a cloud storage bucket."""
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))
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
        """Return True if file matches allowed extensions."""
        file_ext = Path(file_path).suffix.lower()
        return not self.extensions or file_ext in self.extensions

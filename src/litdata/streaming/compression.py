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

import logging
from abc import ABC, abstractmethod
from typing import TypeVar

from litdata.constants import _ZSTD_AVAILABLE
from litdata.debugger import ChromeTraceColors, _get_log_msg

TCompressor = TypeVar("TCompressor", bound="Compressor")

logger = logging.getLogger("litdata.streaming.compression")


class Compressor(ABC):
    """Base class for compression algorithm."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

    @classmethod
    @abstractmethod
    def register(cls, compressors: dict[str, "Compressor"]) -> None:
        pass


class ZSTDCompressor(Compressor):
    """Compressor for the zstd package."""

    def __init__(self, level: int) -> None:
        super().__init__()
        if not _ZSTD_AVAILABLE:
            raise ModuleNotFoundError(str(_ZSTD_AVAILABLE))
        self.level = level
        self.extension = "zstd"

    @property
    def name(self) -> str:
        return f"{self.extension}:{self.level}"

    def compress(self, data: bytes) -> bytes:
        import zstd

        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        import zstd

        logger.debug(_get_log_msg({"name": "Decompressing data", "ph": "B", "cname": ChromeTraceColors.MUSTARD_YELLOW}))
        decompressed_data = zstd.decompress(data)
        logger.debug(_get_log_msg({"name": "Decompressed data", "ph": "E", "cname": ChromeTraceColors.MUSTARD_YELLOW}))
        return decompressed_data

    @classmethod
    def register(cls, compressors: dict[str, "Compressor"]) -> None:
        if not _ZSTD_AVAILABLE:
            return

        # default
        compressors["zstd"] = ZSTDCompressor(4)

        for level in list(range(1, 23)):
            compressors[f"zstd:{level}"] = ZSTDCompressor(level)


_COMPRESSORS: dict[str, Compressor] = {}

ZSTDCompressor.register(_COMPRESSORS)

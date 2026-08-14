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

import shutil
from abc import ABC, abstractmethod
from typing import TypeVar

from litdata.constants import _PYTHON_GREATER_EQUAL_3_14, _ZSTD_AVAILABLE
from litdata.debugger import CAT_DECOMPRESS, trace_span

TCompressor = TypeVar("TCompressor", bound="Compressor")


class Compressor(ABC):
    """Base class for compression algorithm."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

    def decompress_file(self, src: str, dst: str) -> None:
        """Decompress ``src`` onto ``dst``. Default reads the whole file."""
        with open(src, "rb") as inf:
            data = self.decompress(inf.read())
        with open(dst, "wb") as outf:
            outf.write(data)

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
        if _PYTHON_GREATER_EQUAL_3_14:
            from compression import zstd
        else:
            import zstd

        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        if _PYTHON_GREATER_EQUAL_3_14:
            from compression import zstd
        else:
            import zstd

        with trace_span("decompress", CAT_DECOMPRESS):
            return zstd.decompress(data)

    def decompress_file(self, src: str, dst: str) -> None:
        if _PYTHON_GREATER_EQUAL_3_14:
            from compression.zstd import ZstdFile

            with open(src, "rb") as inf, open(dst, "wb") as outf, ZstdFile(inf, "r") as zf:
                shutil.copyfileobj(zf, outf, length=1024 * 1024)
            return
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            with open(src, "rb") as inf, open(dst, "wb") as outf:
                dctx.copy_stream(inf, outf)
            return
        except ImportError:
            super().decompress_file(src, dst)

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

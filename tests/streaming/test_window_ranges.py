"""Local range reads must not fetch extra bytes or reject valid partial reads."""

import asyncio
import io

import pytest

from litdata.streaming.window import _WindowReader


@pytest.mark.parametrize("max_read", [None, 3])
def test_local_range_reads_fetch_only_requested_bytes(monkeypatch, max_read):
    payload = bytes(range(256)) * 32
    fetched = []

    class RawFile(io.RawIOBase):
        def __init__(self):
            self.position = 0

        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return self.position

        def seek(self, offset, whence=0):
            assert whence == 0
            self.position = offset
            return offset

        def readinto(self, buffer):
            count = min(len(buffer), len(payload) - self.position)
            if max_read is not None:
                count = min(count, max_read)
            buffer[:count] = payload[self.position : self.position + count]
            self.position += count
            fetched.append(count)
            return count

    def open_local(*args, buffering=-1, **kwargs):
        raw = RawFile()
        return raw if buffering == 0 else io.BufferedReader(raw, buffer_size=1024)

    monkeypatch.setattr("litdata.streaming.window.open", open_local, raising=False)
    reader = _WindowReader.__new__(_WindowReader)

    class Config:
        _downloader = None

        def __getitem__(self, index):
            return "example.bin", 0, len(payload)

    reader.config = Config()
    result = asyncio.run(reader._read(None, 7, 19))
    assert result == payload[7:26]
    assert sum(fetched) == 19


def test_local_range_read_rejects_truncated_file(tmp_path):
    path = tmp_path / "truncated.bin"
    path.write_bytes(b"data")

    class Config:
        _downloader = None

        def __getitem__(self, index):
            return str(path), 0, 100

    reader = _WindowReader.__new__(_WindowReader)
    reader.config = Config()
    with pytest.raises(OSError, match="Short window read"):
        asyncio.run(reader._read(None, 0, 16))

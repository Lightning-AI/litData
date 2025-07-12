import os

import numpy as np
import pytest

from litdata.streaming.cache import Cache
from litdata.streaming.config import load_subsampled_chunks


def test_load_subsampled_chunks():
    my_subsampled_files = ["1.txt", "2.txt", "5.txt", "3.txt", "9.txt"]

    original_chunks = [
        {"foo": "a", "filename": "1.txt"},
        {"foo": "b", "filename": "2.txt"},
        {"foo": "c", "filename": "3.txt"},
        {"foo": "d", "filename": "4.txt"},
        {"foo": "e", "filename": "5.txt"},
        {"foo": "f", "filename": "6.txt"},
        {"foo": "g", "filename": "7.txt"},
        {"foo": "h", "filename": "8.txt"},
        {"foo": "i", "filename": "9.txt"},
    ]

    assert load_subsampled_chunks(my_subsampled_files, original_chunks) == [
        {"foo": "a", "filename": "1.txt"},
        {"foo": "b", "filename": "2.txt"},
        {"foo": "e", "filename": "5.txt"},
        {"foo": "c", "filename": "3.txt"},
        {"foo": "i", "filename": "9.txt"},
    ]

    my_subsampled_files = ["1.txt", "21.txt", "5.txt", "3.txt", "9.txt"]

    with pytest.raises(ValueError, match="Mismatch in subsampled files"):
        load_subsampled_chunks(my_subsampled_files, original_chunks)


def test_config_download_chunk_bytes(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    cache._reader._try_load_config()

    chunk_idx = 0
    offset = 4
    length = 8

    data = cache._reader._config.download_chunk_bytes_from_index(chunk_index=chunk_idx, offset=offset, length=length)
    assert isinstance(data, bytes)
    assert len(data) == length
    original_data = np.frombuffer(data, dtype=np.int32)  # original data is expected to be []
    assert isinstance(original_data, np.ndarray)

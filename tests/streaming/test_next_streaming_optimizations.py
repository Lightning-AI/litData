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

"""Tests for next-streaming optimizations (deque, timing, cache bytes)."""

from __future__ import annotations

import os
from collections import deque
from unittest.mock import MagicMock

from litdata.streaming import Cache
from litdata.streaming.config import ChunksConfig
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, PyTreeLoader
from litdata.streaming.reader import PrepareChunksThread
from litdata.streaming.serializers import _get_serializers
from litdata.streaming.timing import StreamingTimingStats
from litdata.utilities.env import _DistributedEnv


def _seed_cache(tmpdir, n_items: int = 64, chunk_size: int = 8) -> str:
    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_items):
        cache[i] = i
    cache.done()
    cache.merge(1)
    return cache_dir


def test_upcoming_indexes_uses_deque_popleft(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=40, chunk_size=10)
    ds = StreamingDataset(input_dir=cache_dir, shuffle=True, seed=0)
    values = list(ds)
    assert sorted(values) == list(range(40))
    assert isinstance(ds.upcoming_indexes, deque)


def test_wait_until_chunk_ready_exact_size(tmpdir):
    """Shared wait helper accepts exact ``filesize_bytes`` (>=)."""

    class _Loader(BaseItemLoader):
        def generate_intervals(self):
            return []

        def pre_load_chunk(self, chunk_index, chunk_filepath):
            return None

        def load_item_from_chunk(self, index, chunk_index, chunk_filepath, begin, filesize_bytes):
            return None

        def delete(self, chunk_index, chunk_filepath):
            return None

        def encode_data(self, data, sizes=None, dimensions=None):
            return b"", None

    path = os.path.join(tmpdir, "chunk.bin")
    payload = b"0123456789"
    with open(path, "wb") as f:
        f.write(payload)

    loader = _Loader()
    loader.setup({"data_format": [], "data_spec": None}, [], {})
    loader._wait_until_chunk_ready(0, path, len(payload))


def test_advisory_cache_bytes_updated_on_remove(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    thread = PrepareChunksThread(cfg, MagicMock(), _DistributedEnv(1, 0, 1), max_cache_size=10**9)
    thread._reconcile_cache_bytes()
    baseline = thread._approx_cache_bytes
    assert baseline > 0
    thread._note_chunk_removed(0)
    assert thread._approx_cache_bytes < baseline


def test_can_delete_chunk_respects_size_budget_when_delete_when_processed(tmpdir):
    """Even with delete-when-processed, over-budget must allow eviction.

    Otherwise multi-worker prefetch ignores ``max_cache_size`` until each
    worker's local prefetch window fills.
    """
    cache_dir = _seed_cache(tmpdir, n_items=32, chunk_size=4)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    # Tiny budget -> delete_chunks_when_processed=True
    thread = PrepareChunksThread(cfg, MagicMock(), _DistributedEnv(1, 0, 1), max_cache_size=1)
    assert thread._delete_chunks_when_processed
    thread._reconcile_cache_bytes()
    assert thread._approx_cache_bytes > 1
    thread._pre_download_counter = 0  # prefetch window NOT full
    assert thread._can_delete_chunk() is True
    assert thread._cache_over_budget() is True


def test_timing_stats_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LITDATA_TIMING", raising=False)
    stats = StreamingTimingStats.reset_instance()
    assert stats.enabled is False
    t0 = stats.start()
    stats.record("x", t0)
    assert stats.snapshot() == {}


def test_timing_stats_records_when_enabled(monkeypatch):
    monkeypatch.setenv("LITDATA_TIMING", "1")
    stats = StreamingTimingStats.reset_instance()
    assert stats.enabled is True
    t0 = stats.start()
    assert t0 is not None
    stats.record("item_decode_s", t0)
    snap = stats.snapshot()
    assert snap["item_decode_s"]["count"] == 1
    assert snap["item_decode_s"]["total_s"] >= 0.0

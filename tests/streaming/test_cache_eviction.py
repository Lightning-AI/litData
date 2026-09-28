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

"""Unit tests for ``max_cache_size`` eviction / shared chunk-slot budgeting."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from litdata.streaming import Cache
from litdata.streaming.config import ChunksConfig
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.reader import PrepareChunksThread
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import _get_serializers
from litdata.utilities.env import _DistributedEnv


def _seed_cache(tmpdir, n_items: int = 64, chunk_size: int = 8) -> str:
    cache_dir = os.path.join(str(tmpdir), "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_items):
        cache[i] = i
    cache.done()
    cache.merge(1)
    return cache_dir


def _make_thread(cache_dir: str, max_cache_size: int, max_pre_download: int = 2) -> PrepareChunksThread:
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    return PrepareChunksThread(
        cfg,
        MagicMock(),
        _DistributedEnv(1, 0, 1),
        max_cache_size=max_cache_size,
        max_pre_download=max_pre_download,
    )


def _enable_slot_budget(thread: PrepareChunksThread, max_slots: int = 2) -> None:
    """Force the multi-worker slot gate on for tiny unit-test datasets."""
    thread._delete_chunks_when_processed = True
    # Realistic budgets only; keep >=10MB so `_slot_budget_enabled` is True.
    thread._max_cache_size = max(thread._max_cache_size or 0, 20 * 1024 * 1024)
    assert thread._slot_budget_enabled()
    thread._max_chunk_slots = lambda: max_slots  # type: ignore[method-assign]
    # Seeded fixture chunks would otherwise clamp used-slots to on-disk count
    # and immediately exhaust a tiny test budget.
    thread._on_disk_chunk_count = lambda: 0  # type: ignore[method-assign]
    slots_path, _ = thread._budget_paths()
    with open(slots_path, "w", encoding="utf-8") as f:
        f.write("0")


def test_over_budget_allows_eviction_before_prefetch_window_full(tmpdir):
    """Size wins over delete-when-processed prefetch gating."""
    cache_dir = _seed_cache(tmpdir, n_items=32, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=1)
    assert thread._delete_chunks_when_processed
    thread._reconcile_cache_bytes()
    assert thread._approx_cache_bytes > 1
    thread._pre_download_counter = 0
    assert thread._cache_over_budget() is True
    assert thread._can_delete_chunk() is True


def test_under_budget_waits_for_prefetch_window_when_delete_when_processed(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=8, chunk_size=4)
    # Budget larger than dataset → delete_when_processed=False; force the flag.
    thread = _make_thread(cache_dir, max_cache_size=10**9, max_pre_download=4)
    thread._delete_chunks_when_processed = True
    thread._approx_cache_bytes = 0
    thread._cache_bytes_initialized = True
    thread._pre_download_counter = 0
    assert thread._cache_over_budget() is False
    assert thread._can_delete_chunk() is False
    thread._pre_download_counter = thread._max_pre_download - 1
    assert thread._can_delete_chunk() is True


def test_slot_budget_disabled_for_tiny_max_cache_size(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=64, chunk_size=4)
    # Sub-10MB budgets keep delete-when-processed but skip the slot gate
    # (avoids deadlocks in tiny unit-test caches).
    thread = _make_thread(cache_dir, max_cache_size=1)
    assert thread._delete_chunks_when_processed
    assert thread._slot_budget_enabled() is False


def test_tiny_budget_still_caps_async_pre_download_floor(tmpdir, monkeypatch):
    """Async floor must not defeat tiny max_cache_size (e.g. reader eviction tests)."""
    monkeypatch.setenv("LITDATA_ASYNC_CHUNK_PREFETCH", "1")
    cache_dir = _seed_cache(tmpdir, n_items=64, chunk_size=4)
    cfg = ChunksConfig.load(cache_dir, _get_serializers(None), None, PyTreeLoader())
    assert cfg is not None
    # Pretend remote so the async max_pre_download floor (default 4) applies.
    cfg._remote_dir = "s3://bucket/prefix"
    thread = PrepareChunksThread(
        cfg,
        MagicMock(),
        _DistributedEnv(1, 0, 1),
        max_cache_size=90,  # ~2 mean chunks for this fixture
        max_pre_download=2,
    )
    assert thread._delete_chunks_when_processed
    assert thread._slot_budget_enabled() is False
    # Floor would raise to 4; budget cap must pull it back toward ~2.
    assert thread._max_pre_download <= 2


def test_slot_acquire_and_release_roundtrip(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024)
    _enable_slot_budget(thread, max_slots=2)

    assert thread._acquire_cache_slot(timeout_s=1.0) is True
    assert thread._acquire_cache_slot(timeout_s=1.0) is True
    # Budget full — short timeout must fail rather than download past the limit.
    assert thread._acquire_cache_slot(timeout_s=0.2) is False

    thread._release_cache_slot()
    assert thread._acquire_cache_slot(timeout_s=1.0) is True


def test_download_defers_when_no_slot_available(tmpdir, monkeypatch):
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024)
    _enable_slot_budget(thread, max_slots=1)

    # Hold the only slot.
    assert thread._acquire_cache_slot(timeout_s=1.0) is True

    downloaded: list[int] = []
    monkeypatch.setattr(
        thread._config,
        "download_chunk_from_index",
        lambda idx: downloaded.append(idx),
    )
    chunk_path = thread._config[ChunkedIndex(index=-1, chunk_index=0)][0]
    if os.path.exists(chunk_path):
        os.remove(chunk_path)

    monkeypatch.setattr(thread, "_acquire_cache_slot", lambda timeout_s=30.0: False)
    thread._download_chunk_indexes([0])
    assert downloaded == []
    assert thread._to_download_queue.get_nowait() == 0


def test_apply_delete_releases_slot_only_when_file_existed(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024)
    _enable_slot_budget(thread, max_slots=2)
    assert thread._acquire_cache_slot(timeout_s=1.0) is True
    assert thread._acquire_cache_slot(timeout_s=1.0) is True

    path, _, _ = thread._config[ChunkedIndex(index=-1, chunk_index=0)]
    assert os.path.exists(path)
    thread._item_loader = MagicMock()
    thread._apply_delete(0)
    # One slot freed → acquire should succeed again.
    assert thread._acquire_cache_slot(timeout_s=1.0) is True


def test_apply_delete_skip_release_keeps_reservation(tmpdir):
    """Force-redownload path: delete file but keep the slot for the replacement."""
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024)
    _enable_slot_budget(thread, max_slots=1)
    assert thread._acquire_cache_slot(timeout_s=1.0) is True

    thread._item_loader = MagicMock()
    thread._apply_delete(0, skip_lock=True, release_slot=False)
    # Still full — no free slot for a second acquire.
    assert thread._acquire_cache_slot(timeout_s=0.2) is False


def test_cap_pre_download_under_shared_budget(tmpdir, monkeypatch):
    cache_dir = _seed_cache(tmpdir, n_items=32, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024, max_pre_download=8)
    thread._delete_chunks_when_processed = True
    monkeypatch.setattr(thread._worker_env, "world_size", 16)

    # Force a large mean chunk so budget_chunks // workers < 8.
    class _Cfg:
        _chunks = [{"filename": f"c{i}.bin"} for i in range(4)]
        num_bytes = 16 * 1024 * 1024

    thread._config = _Cfg()  # type: ignore[assignment]
    assert thread._slot_budget_enabled()
    thread._max_pre_download = 8
    thread._cap_pre_download_for_cache_budget()
    # 20MB / 4MB mean = 5 chunks / 16 workers → per_worker 1, floored to 2
    # (max_pre=1 deadlocks delete-when-processed gating).
    assert thread._max_pre_download == 2


def test_max_chunk_slots_includes_five_percent_headroom(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=16, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=20 * 1024 * 1024)
    thread._delete_chunks_when_processed = True
    mean_chunk = max(1, int(thread._config.num_bytes // max(1, len(thread._config._chunks or [1]))))
    expected = max(1, int((thread._max_cache_size * 1.05) // mean_chunk))
    assert thread._max_chunk_slots() == expected


def test_on_disk_chunk_count_matches_bin_files(tmpdir):
    cache_dir = _seed_cache(tmpdir, n_items=24, chunk_size=4)
    thread = _make_thread(cache_dir, max_cache_size=10**9)
    n_bins = sum(1 for name in os.listdir(cache_dir) if name.startswith("chunk-") and name.endswith(".bin"))
    assert thread._on_disk_chunk_count() == n_bins
    assert n_bins > 0

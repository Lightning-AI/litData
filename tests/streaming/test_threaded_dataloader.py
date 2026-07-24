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

"""Deep coverage for experimental ``StreamingDataLoader(use_threading=True)``.

This path is opt-in and requires a free-threaded (no-GIL) runtime. Production
callers must not enable it on GIL builds; these tests patch the GIL gate so the
loader logic can be exercised on standard CI interpreters.
"""

from __future__ import annotations

import os
import pickle
import threading
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from litdata.streaming import Cache
from litdata.streaming.combined import CombinedStreamingDataset
from litdata.streaming.dataloader import StreamingDataLoader, _batch_num_samples
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.parallel import ParallelStreamingDataset
from litdata.streaming.threaded_loader import (
    _clone_dataset_for_thread,
    iter_threaded_streaming_batches,
)


def _seed_cache(tmpdir, n_items: int = 64, chunk_size: int = 8, *, as_dict: bool = False) -> str:
    cache_dir = os.path.join(tmpdir, "chunks")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=cache_dir, chunk_size=chunk_size)
    for i in range(n_items):
        cache[i] = {"x": i, "y": i * 10} if as_dict else i
    cache.done()
    cache.merge(1)
    return cache_dir


def _enable_threading_gate(monkeypatch) -> None:
    monkeypatch.setattr("litdata.streaming.dataloader._is_gil_disabled", lambda: True)


def _flatten_int_batches(batches: list) -> list[int]:
    return [int(x) for batch in batches for x in batch]


def _add_1000(x: int) -> int:
    return x + 1000


def _boom_at_5(x: int) -> int:
    if x == 5:
        raise ValueError("boom-at-5")
    return x


# ---------------------------------------------------------------------------
# Gates / API surface
# ---------------------------------------------------------------------------


def test_is_gil_disabled_false_on_standard_cpython():
    """On GIL builds (typical CI), the probe reports GIL enabled / unavailable."""
    import litdata.streaming.dataloader as dl

    # Either ``sys._is_gil_enabled`` is missing (pre-3.13) or returns True.
    assert dl._is_gil_disabled() is False


def test_is_gil_disabled_respects_patched_probe(monkeypatch):
    import litdata.streaming.dataloader as dl

    monkeypatch.setattr(dl, "_is_gil_disabled", lambda: True)
    assert dl._is_gil_disabled() is True
    monkeypatch.setattr(dl, "_is_gil_disabled", lambda: False)
    assert dl._is_gil_disabled() is False


def test_use_threading_requires_nogil(tmpdir, monkeypatch):
    monkeypatch.setattr("litdata.streaming.dataloader._is_gil_disabled", lambda: False)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=8), shuffle=False)
    with pytest.raises(RuntimeError, match="no-GIL"):
        StreamingDataLoader(ds, batch_size=2, num_workers=2, use_threading=True)


def test_use_threading_rejects_plain_mock(monkeypatch):
    _enable_threading_gate(monkeypatch)
    with pytest.raises(RuntimeError, match="use_threading"):
        StreamingDataLoader(MagicMock(), use_threading=True)  # type: ignore[arg-type]


def test_use_threading_rejects_combined_dataset(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    a = StreamingDataset(input_dir=_seed_cache(os.path.join(tmpdir, "a"), n_items=8), shuffle=False)
    b = StreamingDataset(input_dir=_seed_cache(os.path.join(tmpdir, "b"), n_items=8), shuffle=False)
    combined = CombinedStreamingDataset([a, b], seed=0, weights=(0.5, 0.5), iterate_over_all=False)
    with pytest.raises(RuntimeError, match="StreamingDataset"):
        StreamingDataLoader(combined, batch_size=2, num_workers=2, use_threading=True)


def test_use_threading_rejects_parallel_dataset(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    a = StreamingDataset(input_dir=_seed_cache(os.path.join(tmpdir, "a"), n_items=8), shuffle=False)
    b = StreamingDataset(input_dir=_seed_cache(os.path.join(tmpdir, "b"), n_items=8), shuffle=False)
    parallel = ParallelStreamingDataset([a, b])
    with pytest.raises(RuntimeError, match="StreamingDataset"):
        StreamingDataLoader(parallel, batch_size=2, num_workers=2, use_threading=True)


def test_use_threading_keeps_process_path_default(tmpdir, monkeypatch):
    """Default loader must not enter the threaded path even if GIL is disabled."""
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=16), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=0)
    assert loader.use_threading is False
    assert _flatten_int_batches(list(loader)) == list(range(16))


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_threaded_covers_all_items(tmpdir, monkeypatch, num_workers, batch_size):
    _enable_threading_gate(monkeypatch)
    n_items = 48
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=n_items, chunk_size=8), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=batch_size, num_workers=num_workers, use_threading=True)
    batches = list(loader)
    flat = _flatten_int_batches(batches)
    assert sorted(flat) == list(range(n_items))
    assert all(isinstance(b, torch.Tensor) for b in batches)
    # Last batch may be short when n_items % batch_size != 0.
    assert all(1 <= int(b.shape[0]) <= batch_size for b in batches)


def test_threaded_matches_single_process_item_set(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    cache_dir = _seed_cache(tmpdir, n_items=60, chunk_size=10)
    process = StreamingDataLoader(
        StreamingDataset(input_dir=cache_dir, shuffle=False),
        batch_size=5,
        num_workers=0,
    )
    threaded = StreamingDataLoader(
        StreamingDataset(input_dir=cache_dir, shuffle=False),
        batch_size=5,
        num_workers=3,
        use_threading=True,
    )
    assert sorted(_flatten_int_batches(list(process))) == sorted(_flatten_int_batches(list(threaded)))


def test_threaded_shuffle_covers_all_items(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    n_items = 40
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=n_items), shuffle=True, seed=123)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=2, use_threading=True, shuffle=True)
    assert sorted(_flatten_int_batches(list(loader))) == list(range(n_items))


def test_threaded_order_is_deterministic_across_runs(tmpdir, monkeypatch):
    """Round-robin worker queues must produce stable batch sequences for a fixed seed."""
    _enable_threading_gate(monkeypatch)
    cache_dir = _seed_cache(tmpdir, n_items=36, chunk_size=6)

    def _run() -> list[list[int]]:
        ds = StreamingDataset(input_dir=cache_dir, shuffle=True, seed=7)
        loader = StreamingDataLoader(ds, batch_size=3, num_workers=3, use_threading=True)
        return [[int(x) for x in batch] for batch in loader]

    assert _run() == _run()


def test_threaded_dict_samples_collate(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=24, as_dict=True), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=2, use_threading=True)
    batches = list(loader)
    assert len(batches) > 0
    for batch in batches:
        assert set(batch.keys()) == {"x", "y"}
        assert batch["x"].shape[0] == batch["y"].shape[0]
    xs = [int(v) for batch in batches for v in batch["x"]]
    assert sorted(xs) == list(range(24))


def test_threaded_custom_collate_fn(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=16), shuffle=False)

    def collate(samples: list[int]) -> dict[str, torch.Tensor]:
        t = default_collate(samples)
        return {"values": t, "doubled": t * 2}

    loader = StreamingDataLoader(ds, batch_size=4, num_workers=2, use_threading=True, collate_fn=collate)
    batches = list(loader)
    assert all("values" in b and "doubled" in b for b in batches)
    assert torch.equal(batches[0]["doubled"], batches[0]["values"] * 2)


def test_threaded_transform_applied(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=20), shuffle=False, transform=_add_1000)
    loader = StreamingDataLoader(ds, batch_size=5, num_workers=2, use_threading=True)
    flat = _flatten_int_batches(list(loader))
    assert sorted(flat) == list(range(1000, 1020))


# ---------------------------------------------------------------------------
# Epochs / resume accounting
# ---------------------------------------------------------------------------


def test_threaded_two_epochs_each_cover_dataset(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    n_items = 30
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=n_items), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=2, use_threading=True)
    epoch0 = sorted(_flatten_int_batches(list(loader)))
    epoch1 = sorted(_flatten_int_batches(list(loader)))
    assert epoch0 == list(range(n_items))
    assert epoch1 == list(range(n_items))
    assert loader.current_epoch == 2


def test_threaded_counts_partial_last_batch_in_state(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    # 10 items, batch_size 4 → batches of 4, 4, 2 = 10 samples accounted.
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=10), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=1, use_threading=True)
    list(loader)
    assert loader._num_samples_yielded_streaming == 10
    state = loader.state_dict()
    assert state["num_samples_yielded"] == 10


def test_threaded_state_dict_after_partial_epoch(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=40, chunk_size=8), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=2, use_threading=True)
    it = iter(loader)
    seen = 0
    for _ in range(3):
        batch = next(it)
        seen += int(batch.shape[0])
    state = loader.state_dict()
    assert state["num_samples_yielded"] == seen
    assert state["current_epoch"] == 1


def test_batch_num_samples_helpers():
    assert _batch_num_samples(torch.arange(5), 8) == 5
    assert _batch_num_samples({"a": torch.arange(3), "b": torch.arange(3)}, 8) == 3
    assert _batch_num_samples([1, 2, 3], 8) == 3
    assert _batch_num_samples(object(), 8) == 8


# ---------------------------------------------------------------------------
# Failure / lifecycle
# ---------------------------------------------------------------------------


def test_threaded_propagates_worker_exception(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=20), shuffle=False, transform=_boom_at_5)
    loader = StreamingDataLoader(ds, batch_size=2, num_workers=2, use_threading=True)
    with pytest.raises(ValueError, match="boom-at-5"):
        list(loader)


def test_threaded_rejects_unpicklable_transform(tmpdir, monkeypatch):
    """Cloning workers requires a picklable dataset (including transforms)."""
    _enable_threading_gate(monkeypatch)

    def local_transform(x: int) -> int:
        return x

    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=8), shuffle=False, transform=local_transform)
    loader = StreamingDataLoader(ds, batch_size=2, num_workers=2, use_threading=True)
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        list(loader)


def test_threaded_early_break_does_not_hang(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=200, chunk_size=10), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=8, num_workers=4, use_threading=True)
    count = 0
    for _ in loader:
        count += 1
        if count == 2:
            break
    assert count == 2
    # A second epoch must still start cleanly after GeneratorExit cleanup.
    assert sorted(_flatten_int_batches(list(loader))) == list(range(200))


def test_threaded_loader_rejects_invalid_args(tmpdir):
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=8), shuffle=False)
    with pytest.raises(ValueError, match="num_workers"):
        list(iter_threaded_streaming_batches(ds, num_workers=0, batch_size=2, collate_fn=default_collate))
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_threaded_streaming_batches(ds, num_workers=1, batch_size=0, collate_fn=default_collate))


def test_clone_dataset_assigns_distinct_worker_ranks(tmpdir):
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=16), shuffle=False)
    clones = [_clone_dataset_for_thread(ds, rank=r, world_size=3) for r in range(3)]
    ranks = [c._forced_worker_env.rank for c in clones]
    assert ranks == [0, 1, 2]
    assert all(c._forced_worker_env.world_size == 3 for c in clones)
    # Clones must not share identity with the parent or each other.
    assert len({id(c) for c in clones} | {id(ds)}) == 4


def test_threaded_sets_logical_num_workers_on_dataset(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=16), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=3, use_threading=True)
    assert loader.num_workers == 3
    # Torch DataLoader underneath stays single-process; logical workers live on the wrapper.
    assert loader.use_threading is True
    list(loader)


def test_threaded_worker_threads_use_expected_names(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=24), shuffle=False)
    seen: set[str] = set()
    original_thread = threading.Thread

    class _NamedThread(original_thread):
        def start(self):
            seen.add(self.name)
            return super().start()

    monkeypatch.setattr("litdata.streaming.threaded_loader.threading.Thread", _NamedThread)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=3, use_threading=True)
    list(loader)
    assert seen == {f"litdata-thread-worker-{i}" for i in range(3)}


def test_drop_last_length_with_threading(tmpdir, monkeypatch):
    _enable_threading_gate(monkeypatch)
    ds = StreamingDataset(input_dir=_seed_cache(tmpdir, n_items=10), shuffle=False)
    loader = StreamingDataLoader(ds, batch_size=4, num_workers=1, use_threading=True, drop_last=True)
    assert len(loader) == 10 // 4

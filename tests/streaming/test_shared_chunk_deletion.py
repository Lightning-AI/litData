"""Regression tests for the shared-chunk deletion race.

A chunk can be assigned to several workers on the same node (the shuffle splits a chunk's
interval across worker boundaries). If a worker deletes such a chunk after finishing its own
slice while another worker still needs it, the other worker raises
``FileNotFoundError: The <...>.bin hasn't been found`` (item_loader.py).

The fix is *eager reference counting*: each worker increments the ``.cnt`` of every shared chunk
it will read BEFORE reading anything (``BinaryReader.acquire_shared_locks``), decrements as it
finishes each chunk, and drains any still-held locks on teardown. A shared chunk is deleted only
once its reference count reaches zero — i.e. every worker that will read it has incremented (so the
increment can never lag behind another worker's delete) and every worker has finished. These tests
pin that behaviour, including that counts stay balanced (no leak) across full, partial, and
multi-epoch iteration.
"""

import os
import shutil
import sys
import time

import pytest

from litdata import StreamingDataLoader, StreamingDataset
from litdata.constants import _INDEX_FILENAME
from litdata.streaming.cache import Cache
from litdata.streaming.config import ChunkedIndex, ChunksConfig
from litdata.streaming.downloader import LocalDownloaderWithCache, register_downloader
from litdata.streaming.item_loader import PyTreeLoader
from litdata.streaming.reader import PrepareChunksThread
from litdata.streaming.resolver import Dir
from litdata.streaming.serializers import _get_serializers
from litdata.utilities.env import _DistributedEnv

# Env flag that turns on the flaky-remote behaviour. Off by default so registering this downloader
# for the ``local:`` scheme is a no-op for every other test (it then behaves exactly like the stock
# LocalDownloaderWithCache). The flag is set only inside the flaky test and, being an env var, is
# inherited by spawned DataLoader workers.
_FLAKY_ENV = "LITDATA_TEST_FLAKY_REMOTE"


class _FlakyRemoteDownloader(LocalDownloaderWithCache):
    """Emulates a slow/flaky remote (e.g. S3) that will NOT re-serve an already-evicted chunk.

    The first fetch of a chunk copies it and drops a ``.served`` marker. If that chunk is later
    evicted from the cache and requested again, the "remote" refuses to serve it. A *premature*
    deletion of a shared chunk (one another worker still needs) therefore becomes an unrecoverable
    ``FileNotFoundError`` — the exact production failure — while a legitimate final deletion (nobody
    needs the chunk again) stays harmless. This lets an in-process test distinguish the fix, which a
    plain local remote cannot (it just re-downloads instantly).
    """

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        if os.getenv(_FLAKY_ENV) != "1" or local_filepath.endswith(_INDEX_FILENAME):
            super().download_file(remote_filepath, local_filepath)
            return
        served_marker = local_filepath + ".served"
        if os.path.exists(served_marker) and not os.path.exists(local_filepath):
            # Chunk was fetched before, then evicted, and is now requested again: emulate a remote
            # that cannot recover it in time. Do nothing -> the reader eventually times out.
            return
        super().download_file(remote_filepath, local_filepath)
        with open(served_marker, "w") as marker:
            marker.write("1")


# Route the ``local:`` scheme (the resolver only treats a fixed set of schemes as remote) through
# the flaky downloader. Harmless when the env flag is off.
register_downloader("local:", _FlakyRemoteDownloader, overwrite=True)


def _build_remote_dataset(tmpdir, num_items=10, chunk_size=2):
    """Optimize a tiny dataset into ``remote_dir`` and return an empty local ``cache_dir``."""
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)

    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=chunk_size, compression="zstd")
    for i in range(num_items):
        cache[i] = i
    cache.done()
    cache.merge()

    shutil.copytree(cache_dir, remote_dir)
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, remote_dir


def test_lock_primitives_balance(tmpdir):
    """`increment_local_lock` / `decrement_local_lock` / `remaining_locks` are balanced."""
    cache_dir, remote_dir = _build_remote_dataset(tmpdir)
    config = ChunksConfig.load(cache_dir, _get_serializers(None), remote_dir, PyTreeLoader())
    assert config is not None

    assert config.remaining_locks(0) == 0
    config.increment_local_lock(0)
    assert config.remaining_locks(0) == 1
    config.increment_local_lock(0)
    assert config.remaining_locks(0) == 2
    assert config.decrement_local_lock(0) == 1
    assert config.remaining_locks(0) == 1
    assert config.decrement_local_lock(0) == 0
    assert config.remaining_locks(0) == 0  # count file removed at zero


def test_shared_chunk_deleted_only_at_refcount_zero(tmpdir):
    """A shared chunk (eagerly counted by both readers) is deleted only when the last one finishes.

    Reproduces the customer's ``FileNotFoundError`` at the mechanism level: two workers share chunk
    0. Because both increment its count eagerly (before reading), the first worker to finish sees a
    non-zero count and does NOT delete it — so the second worker still finds the file.
    """
    cache_dir, remote_dir = _build_remote_dataset(tmpdir)
    item_loader = PyTreeLoader()
    config = ChunksConfig.load(cache_dir, _get_serializers(None), remote_dir, item_loader)
    assert config is not None

    # Chunk 0 is shared -> download must not add its own (lazy) reference.
    config._shared_chunk_indexes = {0}
    config.increment_local_lock(0)  # worker A claims it eagerly
    config.increment_local_lock(0)  # worker B claims it eagerly  -> count 2
    config.download_chunk_from_index(0)  # materialise the file (shared -> no extra increment)

    chunk0_path, _, _ = config[ChunkedIndex(index=-1, chunk_index=0)]
    assert os.path.exists(chunk0_path)
    assert config.remaining_locks(0) == 2

    thread = PrepareChunksThread(config, item_loader, _DistributedEnv.detect(), max_cache_size=1, rank=0)

    config.decrement_local_lock(0)  # worker A finishes
    thread._apply_delete(0)
    assert os.path.exists(chunk0_path), "shared chunk must survive while a co-worker still holds it"

    config.decrement_local_lock(0)  # worker B finishes -> count 0
    thread._apply_delete(0)
    assert not os.path.exists(chunk0_path), "shared chunk should be deleted once the last reader is done"


def test_reader_acquire_and_release_shared_locks_balanced(tmpdir):
    """`acquire_shared_locks` increments eagerly and every count is released on teardown."""
    from litdata.streaming.reader import BinaryReader

    cache_dir, remote_dir = _build_remote_dataset(tmpdir)
    reader = BinaryReader(cache_dir, remote_input_dir=remote_dir, item_loader=PyTreeLoader())
    assert reader._try_load_config() is not None

    reader.acquire_shared_locks({0, 1})
    assert reader._held_shared == {0, 1}
    assert reader.config.remaining_locks(0) == 1
    assert reader.config.remaining_locks(1) == 1

    # Re-acquiring must first release the previous holds (no double counting across epochs).
    reader.acquire_shared_locks({0, 1})
    assert reader.config.remaining_locks(0) == 1
    assert reader.config.remaining_locks(1) == 1

    reader._release_shared_locks()
    assert reader._held_shared == set()
    assert reader.config.remaining_locks(0) == 0
    assert reader.config.remaining_locks(1) == 0


def test_apply_delete_still_deletes_when_not_protected(tmpdir):
    """Sanity check: with no skip list, a chunk at refcount zero is deleted as before."""
    cache_dir, remote_dir = _build_remote_dataset(tmpdir)

    item_loader = PyTreeLoader()
    config = ChunksConfig.load(cache_dir, _get_serializers(None), remote_dir, item_loader)
    assert config is not None

    config.download_chunk_from_index(0)
    thread = PrepareChunksThread(config, item_loader, _DistributedEnv.detect(), max_cache_size=1, rank=0)

    chunk0_path, _, _ = config[ChunkedIndex(index=-1, chunk_index=0)]
    assert os.path.exists(chunk0_path)

    thread._decrement_local_lock(0)
    assert thread._remaining_locks(chunk0_path) == 0
    assert config.skip_chunk_indexes_deletion is None

    thread._apply_delete(0)
    assert not os.path.exists(chunk0_path)


def test_apply_delete_skips_when_refcount_positive(tmpdir):
    """A positive ``.cnt`` refcount blocks deletion regardless of the skip list."""
    cache_dir, remote_dir = _build_remote_dataset(tmpdir)

    item_loader = PyTreeLoader()
    config = ChunksConfig.load(cache_dir, _get_serializers(None), remote_dir, item_loader)
    assert config is not None

    # Two holders incremented the lock; only one has finished (one decrement).
    config.download_chunk_from_index(0)
    config._downloader._increment_local_lock(chunk0 := config[ChunkedIndex(index=-1, chunk_index=0)][0], 0)

    thread = PrepareChunksThread(config, item_loader, _DistributedEnv.detect(), max_cache_size=1, rank=0)
    thread._decrement_local_lock(0)
    assert thread._remaining_locks(chunk0) == 1

    thread._apply_delete(0)
    assert os.path.exists(chunk0), "A chunk still referenced by another holder must not be deleted."


def _skewed_transform(x, *args, **kwargs):
    """Slow down worker 0 so it reads its (shared) chunks much later than the other workers.

    Worker 0 owns the first slice of the dataset, so the chunk on the worker-0/worker-1 boundary
    is worker 0's *last* chunk but worker 1's *first*. Slowing worker 0 guarantees worker 1 reaches
    and finishes that shared chunk (and, with a tiny cache, tries to evict it) long before worker 0
    reads it -- deterministically exercising the shared-chunk deletion race.
    """
    import torch.utils.data

    info = torch.utils.data.get_worker_info()
    if info is not None and info.id == 0:
        time.sleep(0.03)
    return x


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_multiworker_dataloader_shared_chunk_not_deleted_early(tmpdir):
    """End-to-end: streaming a full dataset across workers must not lose shared chunks.

    Uses a ``local:`` remote (so the download / lock / delete path is active) and a tiny
    ``max_cache_size`` (so eviction runs aggressively). Chunks straddle worker boundaries, and
    worker 0 is deliberately slowed so a faster co-worker would, pre-fix, evict a shared chunk that
    worker 0 still needs -> ``FileNotFoundError``. After the fix every item must be returned exactly
    once with no error.
    """
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    num_items = 200
    # chunk_size=7 makes chunk boundaries misalign with the per-worker item split, so several
    # chunks are shared between adjacent workers.
    cache = Cache(str(data_dir), chunk_size=7)
    for i in range(num_items):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        f"local:{data_dir}",
        cache_dir=str(cache_dir),
        shuffle=False,
        drop_last=False,
        # Tiny cache -> num_bytes_per_node > max_cache_size -> aggressive "delete when processed".
        max_cache_size=512,
        transform=_skewed_transform,
    )
    assert len(dataset) == num_items

    dataloader = StreamingDataLoader(dataset, batch_size=1, num_workers=3)

    seen = []
    for batch in dataloader:
        seen.extend(int(v) for v in batch)

    assert sorted(seen) == list(range(num_items)), (
        f"Expected all {num_items} items exactly once; got {len(seen)} items "
        f"(missing: {sorted(set(range(num_items)) - set(seen))}, "
        f"duplicated: {sorted({v for v in seen if seen.count(v) > 1})})."
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_multiworker_flaky_remote_shared_chunk_race(tmpdir, monkeypatch):
    """The distinguishing end-to-end test: fails pre-fix, passes post-fix.

    Same setup as above but over a ``flaky:`` remote that will not re-serve an evicted chunk, and
    with short timeouts so a lost chunk fails fast. Pre-fix, a non-designated worker deletes a
    shared chunk that the (slowed) worker 0 still needs and the remote won't give it back ->
    ``FileNotFoundError``. Post-fix, the shared chunk is protected until its last reader is done.
    """
    # Short timeouts so a genuinely lost chunk raises quickly. Patch both the env (spawn workers
    # re-import and read it) and the already-imported module globals (fork workers inherit them).
    monkeypatch.setenv(_FLAKY_ENV, "1")
    monkeypatch.setenv("MAX_WAIT_TIME", "4")
    monkeypatch.setenv("FORCE_DOWNLOAD_TIME", "1")
    import litdata.streaming.config as config_mod
    import litdata.streaming.item_loader as item_loader_mod

    monkeypatch.setattr(item_loader_mod, "_MAX_WAIT_TIME", 4, raising=False)
    monkeypatch.setattr(item_loader_mod, "_FORCE_DOWNLOAD_TIME", 1, raising=False)
    monkeypatch.setattr(config_mod, "_MAX_WAIT_TIME", 4, raising=False)

    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    num_items = 200
    cache = Cache(str(data_dir), chunk_size=7)
    for i in range(num_items):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        f"local:{data_dir}",
        cache_dir=str(cache_dir),
        shuffle=False,
        drop_last=False,
        max_cache_size=512,
        transform=_skewed_transform,
    )
    assert len(dataset) == num_items

    dataloader = StreamingDataLoader(dataset, batch_size=1, num_workers=3)

    seen = []
    for batch in dataloader:
        seen.extend(int(v) for v in batch)

    assert sorted(seen) == list(range(num_items)), (
        f"Lost or duplicated items streaming a shared-chunk dataset over a flaky remote. "
        f"Missing: {sorted(set(range(num_items)) - set(seen))}."
    )


def _count_files(root, suffix):
    return sum(1 for dirpath, _, files in os.walk(root) for f in files if f.endswith(suffix))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_no_refcount_leak_after_full_and_multi_epoch_iteration(tmpdir):
    """Full iteration (repeated) must leave no ``.cnt`` reference-count files behind.

    Guards the eager-refcount balance: an over-increment would leave ``.cnt`` files (and prevent
    chunk deletion) that accumulate across epochs.
    """
    cache_dir = os.path.join(tmpdir, "cache_dir")
    data_dir = os.path.join(tmpdir, "data_dir")
    os.makedirs(cache_dir)
    os.makedirs(data_dir)

    num_items = 120
    cache = Cache(str(data_dir), chunk_size=7)
    for i in range(num_items):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        f"local:{data_dir}", cache_dir=str(cache_dir), shuffle=False, drop_last=False, max_cache_size=512
    )
    dataloader = StreamingDataLoader(dataset, batch_size=1, num_workers=3)

    for _ in range(3):  # multiple epochs
        seen = [int(v) for batch in dataloader for v in batch]
        assert sorted(seen) == list(range(num_items))

    leftover_cnt = _count_files(cache_dir, ".cnt")
    assert leftover_cnt == 0, f"Reference-count (.cnt) files leaked after full iteration: {leftover_cnt}"


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_reader_teardown_releases_held_shared_locks(tmpdir):
    """Abandoning a reader mid-epoch (early break) must release its eagerly-acquired locks.

    Without the teardown release, aborted iteration would leak reference counts and permanently
    prevent those shared chunks from being deleted.
    """
    import gc

    from litdata.streaming.reader import BinaryReader

    cache_dir, remote_dir = _build_remote_dataset(tmpdir, num_items=40, chunk_size=4)
    reader = BinaryReader(cache_dir, remote_input_dir=remote_dir, item_loader=PyTreeLoader())
    assert reader._try_load_config() is not None

    reader.acquire_shared_locks({0, 1, 2})  # pretend this worker shares chunks 0,1,2
    assert _count_files(cache_dir, ".cnt") == 3

    # Simulate: worker consumed chunk 0 only, then the loop was broken.
    reader.config.decrement_local_lock(0)
    reader._held_shared.discard(0)

    del reader
    gc.collect()  # triggers BinaryReader.__del__ -> _release_shared_locks()

    leftover = _count_files(cache_dir, ".cnt")
    assert leftover == 0, f"Held shared-chunk locks leaked after reader teardown: {leftover}"

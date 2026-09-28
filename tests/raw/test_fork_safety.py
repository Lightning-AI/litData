"""Regression: fork/loop lifecycle, atomic cache publish, prefetch failures."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from torch.utils.data import DataLoader

from litdata.raw.dataset import (
    _HEDGE_ASSUMED_BANDWIDTH_BPS,
    _LOCK_SUFFIX,
    CacheManager,
    StreamingRawDataset,
    _create_event_loop,
    _effective_hedge_delay,
    _get_loop_runner,
    _loop_backend_name,
    _run_async,
    _sweep_orphan_tmp_files,
)


@pytest.mark.skipif(sys.platform == "win32", reason="fork semantics differ on Windows")
def test_spawn_pickle_strips_runtime_state(tmp_path: Path) -> None:
    """After warm-up, pickle must succeed and omit loop/task/downloader state."""
    import pickle

    for i in range(8):
        (tmp_path / f"f{i}.bin").write_bytes(f"data-{i}".encode())

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=4,
        hedge_delay=0,
    )
    _ = ds.__getitems__([0, 1, 2])
    # Accidental instance attrs must not enter the allowlisted pickle payload.
    ds._bad_lock = threading.Lock()  # type: ignore[attr-defined]
    ds.cache_manager._bad_lock = threading.Lock()  # type: ignore[attr-defined]

    blob = pickle.dumps(ds, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(blob)  # noqa: S301
    assert not hasattr(restored, "_bad_lock")
    assert not hasattr(restored.cache_manager, "_bad_lock")
    assert restored.cache_manager._downloader is None
    assert restored.cache_manager._semaphore is None
    assert restored.cache_manager._range_executor is None
    assert restored._inflight == {}
    assert restored._inflight_loop is None
    assert restored._owner_pid == os.getpid()
    assert restored[0].startswith(b"data-")


@pytest.mark.skipif(sys.platform == "win32", reason="fork semantics differ on Windows")
@pytest.mark.parametrize("mp_context", ["spawn", "fork"])
@pytest.mark.parametrize("cache_files", [False, True])
def test_parent_worker_parent_lifecycle(tmp_path: Path, mp_context: str, cache_files: bool) -> None:
    """Touch dataset in main, iterate with workers, touch again in main."""
    for i in range(8):
        (tmp_path / f"f{i}.bin").write_bytes(f"data-{i}".encode())

    cache = tmp_path / "cache"
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(cache),
        cache_files=cache_files,
        max_prefetch=0,
        max_concurrent_downloads=8,
        hedge_delay=0,
    )

    first = ds[0]
    assert first.startswith(b"data-")

    loader = DataLoader(ds, batch_size=2, num_workers=2, multiprocessing_context=mp_context)
    batches = list(loader)
    assert len(batches) == 4
    assert len(batches[0]) == 2

    again = ds[0]
    assert again == first


@pytest.mark.skipif(sys.platform == "win32", reason="fork semantics differ on Windows")
def test_os_fork_clears_runner_and_lock(tmp_path: Path) -> None:
    """Child after os.fork gets a fresh runner; module locks/futures are reinitialized."""
    (tmp_path / "a.bin").write_bytes(b"fork-me")
    # cache_files=True so write-behind futures may be in flight around fork.
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=True,
        hedge_delay=0,
        max_prefetch=0,
    )
    parent_runner = _get_loop_runner()
    assert parent_runner.is_alive()
    _ = ds[0]

    import litdata.raw.dataset as raw_dataset

    parent_lock = raw_dataset._RUNNER_LOCK
    parent_wb_lock = raw_dataset._WRITE_BEHIND_LOCK
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child — register_at_fork already cleared runner + reinit locks.
        os.close(rfd)
        try:
            assert raw_dataset._RUNNER is None
            assert raw_dataset._RUNNER_LOCK is not parent_lock
            assert raw_dataset._WRITE_BEHIND_LOCK is not parent_wb_lock
            assert set() == raw_dataset._WRITE_BEHIND_FUTURES
            child_runner = _get_loop_runner()
            assert child_runner.pid == os.getpid()
            assert child_runner is not parent_runner
            val = ds[0]
            os.write(wfd, b"ok:" + val)
            code = 0
        except Exception as exc:
            os.write(wfd, f"err:{exc!r}".encode())
            code = 1
        finally:
            os.close(wfd)
            os._exit(code)

    os.close(wfd)
    with os.fdopen(rfd, "rb") as rf:
        msg = rf.read(4096)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status), msg
    assert os.WEXITSTATUS(status) == 0, msg
    assert msg.startswith(b"ok:")
    assert msg[3:] == b"fork-me"
    # Parent runner still usable.
    assert ds[0] == b"fork-me"


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_loop_runner_deadlock_guard() -> None:
    runner = _get_loop_runner()

    async def boom() -> None:
        # Calling run() on the loop thread itself must fail fast.
        runner.run(asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="event-loop thread"):
        runner.run(boom())


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_create_event_loop_matches_backend_name() -> None:
    loop = _create_event_loop()
    try:
        name = _loop_backend_name()
        if name == "uvloop":
            assert type(loop).__module__.startswith("uvloop")
        else:
            assert not type(loop).__module__.startswith("uvloop")
    finally:
        loop.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_cache_files_dedupes_concurrent_downloads(tmp_path: Path) -> None:
    """Concurrent ensure_file_async for the same key shares one download task."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"hello-cache")
    cache = tmp_path / "cache"

    cm = CacheManager(str(src), cache_dir=str(cache), cache_files=True, max_concurrent_downloads=4)
    remote = str(src / "a.bin")
    calls = {"n": 0}

    async def run() -> None:
        downloader = cm.downloader
        orig = downloader.adownload_file

        async def counting_adownload(remote_filepath: str, local_filepath: str) -> None:
            calls["n"] += 1
            await asyncio.sleep(0.05)
            await orig(remote_filepath, local_filepath)

        downloader.adownload_file = counting_adownload  # type: ignore[method-assign]
        paths = await asyncio.gather(
            cm.ensure_file_async(remote),
            cm.ensure_file_async(remote),
            cm.ensure_file_async(remote),
        )
        assert len(set(paths)) == 1
        assert Path(paths[0]).read_bytes() == b"hello-cache"

    asyncio.run(run())
    assert calls["n"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_cache_publish_is_atomic_against_direct_writes(tmp_path: Path) -> None:
    """Even if the downloader writes non-atomically to its target, final path is atomic."""
    src = tmp_path / "src"
    src.mkdir()
    payload = b"x" * 64_000
    (src / "big.bin").write_bytes(payload)
    cache = tmp_path / "cache"
    cm = CacheManager(str(src), cache_dir=str(cache), cache_files=True)
    remote = str(src / "big.bin")
    local = cm.get_local_path(remote)
    seen_partial = {"value": False}

    async def slow_direct_write(remote_filepath: str, local_filepath: str) -> None:
        # Intentionally non-atomic write to whatever path CacheManager asks for (the tmp path).
        with open(local_filepath, "wb") as f:
            f.write(payload[:1000])
            f.flush()
            os.fsync(f.fileno())
            # Final publish target must still be absent while the temp is partial.
            if not os.path.exists(local):
                seen_partial["value"] = True
            await asyncio.sleep(0.05)
            f.write(payload[1000:])

    async def run() -> None:
        cm.downloader.adownload_file = slow_direct_write  # type: ignore[method-assign]
        path = await cm.ensure_file_async(remote)
        assert path == local
        assert Path(local).read_bytes() == payload
        assert not list(Path(local).parent.glob("*.tmp.*"))

    asyncio.run(run())
    assert seen_partial["value"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_prefetch_failure_retries_on_demand(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Failed prefetch must not poison the item; on-demand fetch succeeds without GC warnings."""
    for i in range(6):
        (tmp_path / f"f{i}.bin").write_bytes(f"ok-{i}".encode())

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=4,
        prefetch_cache_size=8,
        hedge_delay=0,
    )
    # Force known order.
    ds.items = sorted(ds.items, key=lambda m: m.path)

    fail_once = {"n": 0}
    real = ds.cache_manager.download_file_async

    async def flaky(file_path: str, size: int | None = None) -> bytes:
        # Fail the first download of index-1's file during prefetch window.
        if file_path.endswith("f1.bin") and fail_once["n"] == 0:
            fail_once["n"] += 1
            raise RuntimeError("boom-prefetch")
        return await real(file_path)

    ds.cache_manager.download_file_async = flaky  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger="litdata.raw.dataset"):
        # Batch [0] schedules prefetch of later indices (worker stride=1).
        batch0 = ds.__getitems__([0])
        assert batch0[0] == b"ok-0"
        # Allow prefetch tasks to settle (including the failed one + done callback).
        time.sleep(0.1)
        # On-demand fetch of the previously failed index must succeed.
        batch1 = ds.__getitems__([1])
        assert batch1[0] == b"ok-1"

    # No asyncio "never retrieved" noise expected in our logger; callback retrieved it.
    joined = "\n".join(r.message for r in caplog.records)
    assert "exception was never retrieved" not in joined


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_getitem_from_running_event_loop(tmp_path: Path) -> None:
    """``__getitem__`` works when the caller already has a running loop (notebook path)."""
    (tmp_path / "a.bin").write_bytes(b"notebook")
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        max_prefetch=0,
    )

    async def from_running_loop() -> bytes:
        # Nested: running loop + dataset sync API.
        return await asyncio.to_thread(lambda: ds[0])

    assert asyncio.run(from_running_loop()) == b"notebook"

    async def via_run_async_nested() -> bytes:
        return _run_async(ds._download_batch([0]))[0]

    assert asyncio.run(via_run_async_nested()) == b"notebook"


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_cache_manager_resets_runtime_state(tmp_path: Path) -> None:
    cm = CacheManager(str(tmp_path), cache_dir=str(tmp_path / "c"), cache_files=False)
    cm._downloader = object()  # type: ignore[assignment]
    cm._downloader_pid = 1
    cm._semaphore = object()  # type: ignore[assignment]
    cm.reset_runtime_state()
    assert cm._downloader is None
    assert cm._downloader_pid is None
    assert cm._semaphore is None
    assert cm._path_inflight == {}


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_atomic_publish_never_exposes_short_file(tmp_path: Path) -> None:
    """Readers polling the final path must never see a partial publish."""
    src = tmp_path / "src"
    src.mkdir()
    payload = b"ABCDEFGH" * 8_000  # 64 KiB
    (src / "big.bin").write_bytes(payload)
    cache = tmp_path / "cache"
    cm = CacheManager(str(src), cache_dir=str(cache), cache_files=True)
    remote = str(src / "big.bin")
    local = cm.get_local_path(remote)
    observed_lengths: list[int] = []
    stop = threading.Event()

    def poller() -> None:
        while not stop.wait(0.001):
            if os.path.exists(local):
                with contextlib.suppress(OSError):
                    observed_lengths.append(os.path.getsize(local))

    async def chunky_write(remote_filepath: str, local_filepath: str) -> None:
        with open(local_filepath, "wb") as f:
            for i in range(0, len(payload), 4096):
                f.write(payload[i : i + 4096])
                f.flush()
                await asyncio.sleep(0.002)

    async def run() -> None:
        cm.downloader.adownload_file = chunky_write  # type: ignore[method-assign]
        path = await cm.ensure_file_async(remote)
        assert Path(path).read_bytes() == payload

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    asyncio.run(run())
    # Allow the poller to observe the atomically published final path.
    deadline = time.monotonic() + 2.0
    while not observed_lengths and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    t.join(timeout=2)
    assert observed_lengths, "poller should observe the published file"
    assert all(n == len(payload) for n in observed_lengths)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_two_threads_same_file_cache_publish(tmp_path: Path) -> None:
    """Two caller threads sharing the process LoopRunner can cache-publish the same file.

    Note: both threads dispatch onto the single process-local ``_LoopRunner`` loop
    (not distinct event loops). Cross-process coordination is covered by the lock tests.
    """
    src = tmp_path / "src"
    src.mkdir()
    payload = b"thread-race-" + os.urandom(32_768)
    (src / "shared.bin").write_bytes(payload)
    cache = tmp_path / "cache"
    cm = CacheManager(str(src), cache_dir=str(cache), cache_files=True)
    remote = str(src / "shared.bin")
    results: list[bytes] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            path = _run_async(cm.ensure_file_async(remote))
            results.append(Path(path).read_bytes())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert len(results) == 2
    assert all(r == payload for r in results)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_cancelled_inflight_retries_on_demand(tmp_path: Path) -> None:
    """Cancelled prefetch task is dropped; on-demand resolve still returns data."""
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(f"val-{i}".encode())

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=0,
        hedge_delay=0,
    )
    ds.items = sorted(ds.items, key=lambda m: m.path)

    async def run() -> bytes:
        # Simulate a cancelled prefetch entry for index 1.
        async def forever() -> bytes:
            await asyncio.sleep(3600)
            return b"never"

        task = asyncio.create_task(forever())
        ds._inflight[1] = task
        ds._inflight_loop = asyncio.get_running_loop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # _resolve_index should treat cancelled task as miss and re-fetch.
        return await ds._resolve_index(1)

    assert asyncio.run(run()) == b"val-1"


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_nested_loop_still_serves_items(tmp_path: Path) -> None:
    """Caller with a running loop can still fetch via the process LoopRunner thread."""
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(f"n-{i}".encode())

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=2,
        hedge_delay=0,
    )
    ds.items = sorted(ds.items, key=lambda m: m.path)

    async def nested() -> list[Any]:
        return await asyncio.to_thread(lambda: ds.__getitems__([0]))

    batch = asyncio.run(nested())
    assert batch[0] == b"n-0"


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_sweep_orphan_tmp_files_removes_dead_pid(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    dead_pid = os.getpid() + 100000
    dead = cache / f"file.bin.tmp.{dead_pid}.1"
    # Likely-dead pid; if somehow alive, skip assertion rather than flaking.
    try:
        os.kill(dead_pid, 0)
        pytest.skip("unexpected live pid collision")
    except ProcessLookupError:
        pass
    except PermissionError:
        pytest.skip("pid exists")
    dead.write_bytes(b"orphan")
    live_pid_tmp = cache / f"other.bin.tmp.{os.getpid()}.99"
    live_pid_tmp.write_bytes(b"keep")
    stale_lock = cache / f"stale.bin{_LOCK_SUFFIX}"
    stale_lock.write_text(f"{dead_pid}\n")
    dead_scratch = cache / f".range-scratch.{dead_pid}.1.0.abc123"
    dead_scratch.write_bytes(b"scratch-orphan")
    live_scratch = cache / f".range-scratch.{os.getpid()}.1.0.def456"
    live_scratch.write_bytes(b"keep-scratch")
    _sweep_orphan_tmp_files(str(cache))
    assert not dead.exists()
    assert live_pid_tmp.exists()
    assert not stale_lock.exists()
    assert not dead_scratch.exists()
    assert live_scratch.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_write_through_cache_returns_bytes_before_publish(tmp_path: Path) -> None:
    """cache_files + bytes path returns network bytes and eventually publishes the file."""
    src = tmp_path / "src"
    src.mkdir()
    payload = b"write-through-" + os.urandom(4096)
    (src / "a.bin").write_bytes(payload)
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    remote = str(src / "a.bin")
    local = cm.get_local_path(remote)

    data = _run_async(cm.download_file_async(remote))
    assert data == payload
    # Write-behind is async; wait briefly for publish.
    deadline = time.monotonic() + 5.0
    while not os.path.exists(local) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert Path(local).read_bytes() == payload


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_write_through_dedupes_concurrent_fetches(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"dedupe-bytes")
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    remote = str(src / "a.bin")
    calls = {"n": 0}

    async def run() -> None:
        real = cm.downloader.adownload_fileobj

        async def counting(path: str) -> bytes:
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return await real(path)

        cm.downloader.adownload_fileobj = counting  # type: ignore[method-assign]
        results = await asyncio.gather(
            cm.download_file_async(remote),
            cm.download_file_async(remote),
            cm.download_file_async(remote),
        )
        assert results == [b"dedupe-bytes"] * 3

    asyncio.run(run())
    assert calls["n"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_loop_runner_survives_across_calls(tmp_path: Path) -> None:
    """Prefetch can progress between __getitems__ because the loop thread persists."""
    for i in range(16):
        (tmp_path / f"f{i:02d}.bin").write_bytes(f"x-{i}".encode())
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=8,
        hedge_delay=0,
    )
    ds.items = sorted(ds.items, key=lambda m: m.path)
    runner = _get_loop_runner()
    assert runner.is_alive()
    _ = ds.__getitems__([0, 1])
    time.sleep(0.2)
    # Same runner instance should still be alive (not recreated per call).
    assert _get_loop_runner() is runner
    batch = ds.__getitems__([2, 3])
    assert batch[0].startswith(b"x-")


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_cross_process_lock_claim(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"locked")
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    local = cm.get_local_path(str(src / "a.bin"))
    assert cm._try_claim_lock(local) is True
    assert Path(cm._lock_path(local)).name.endswith(_LOCK_SUFFIX.lstrip(".")) or cm._lock_path(local).endswith(
        _LOCK_SUFFIX
    )
    assert cm._try_claim_lock(local) is False
    cm._release_lock(local)
    assert cm._try_claim_lock(local) is True
    cm._release_lock(local)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_lock_peer_fails_then_claim(tmp_path: Path) -> None:
    """If a peer held the lock then failed/released, the waiter can claim and finish."""
    src = tmp_path / "src"
    src.mkdir()
    payload = b"peer-fail-then-ok"
    (src / "a.bin").write_bytes(payload)
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    remote = str(src / "a.bin")
    local = cm.get_local_path(remote)
    # Simulate a peer that claimed the lock then died without publishing.
    assert cm._try_claim_lock(local) is True
    lock_path = cm._lock_path(local)
    dead_pid = os.getpid() + 100000
    try:
        os.kill(dead_pid, 0)
        pytest.skip("unexpected live pid collision")
    except ProcessLookupError:
        pass
    except PermissionError:
        pytest.skip("pid exists")
    Path(lock_path).write_text(f"{dead_pid}\n")
    assert cm._lock_owner_alive(lock_path) is False
    path = _run_async(cm.ensure_file_async(remote))
    assert Path(path).read_bytes() == payload


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_lock_dead_pid_takeover(tmp_path: Path) -> None:
    """``_try_claim_lock`` takes over when the recorded owner pid is dead."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"takeover")
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    local = cm.get_local_path(str(src / "a.bin"))
    lock_path = cm._lock_path(local)
    dead_pid = os.getpid() + 100000
    try:
        os.kill(dead_pid, 0)
        pytest.skip("unexpected live pid collision")
    except ProcessLookupError:
        pass
    except PermissionError:
        pytest.skip("pid exists")
    Path(lock_path).write_text(f"{dead_pid}\n")
    assert cm._lock_owner_alive(lock_path) is False
    assert cm._try_claim_lock(local) is True
    assert Path(lock_path).read_text().strip() == str(os.getpid())
    cm._release_lock(local)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_hedge_slow_first_wins(tmp_path: Path) -> None:
    """Slow first request is beaten by a hedged second request."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"hedge-ok")
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0.05,
        max_concurrent_downloads=8,
    )
    remote = "s3://bucket/data/a.bin"  # remote so hedging is eligible
    cm._input_dir_path = "s3://bucket/data"
    calls = {"n": 0}

    async def run() -> bytes:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        async def flaky(path: str) -> bytes:
            n = calls["n"]
            calls["n"] = n + 1
            if n == 0:
                await asyncio.sleep(0.5)
                return b"slow"
            return b"hedge-ok"

        cm._downloader = SimpleNamespace(adownload_fileobj=flaky)  # type: ignore[assignment]
        return await cm._fetch_bytes(remote, size=8)

    t0 = time.monotonic()
    assert asyncio.run(run()) == b"hedge-ok"
    # Hedge delay is 0.05s; allow scheduling jitter on loaded CI.
    assert time.monotonic() - t0 < 2.0
    assert calls["n"] >= 2
    assert cm._hedge_fired >= 1


def test_hedge_delay_default_is_zero() -> None:
    """Hedging is opt-in (default 0), matching range_parallel_threshold."""
    import inspect

    ds_default = inspect.signature(StreamingRawDataset.__init__).parameters["hedge_delay"].default
    cm_default = inspect.signature(CacheManager.__init__).parameters["hedge_delay"].default
    assert float(ds_default) == 0.0
    assert float(cm_default) == 0.0


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
@pytest.mark.parametrize("download_timeout", [0.0, 120.0])
def test_fetch_bytes_fast_path_when_hedge_off(tmp_path: Path, download_timeout: float) -> None:
    """Hedging off takes the bare path even when download_timeout defaults to 120."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"fast")
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        download_timeout=download_timeout,
    )
    remote = "s3://bucket/data/a.bin"
    cm._input_dir_path = "s3://bucket/data"
    calls = {"n": 0}
    wait_for_calls = {"n": 0}
    real_wait_for = asyncio.wait_for

    async def counting_wait_for(awaitable, timeout=None, **kwargs):  # type: ignore[no-untyped-def]
        wait_for_calls["n"] += 1
        return await real_wait_for(awaitable, timeout=timeout, **kwargs)

    async def run() -> bytes:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        async def once(path: str) -> bytes:
            calls["n"] += 1
            return b"fast"

        cm._downloader = SimpleNamespace(adownload_fileobj=once)  # type: ignore[assignment]
        return await cm._fetch_bytes(remote, size=4)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "wait_for", counting_wait_for)
        assert asyncio.run(run()) == b"fast"
    assert calls["n"] == 1
    assert cm._hedge_fired == 0
    assert wait_for_calls["n"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_download_batch_applies_timeout_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default download_timeout wraps the batch gather once, not each item."""
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(f"data-{i}".encode())
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=0,
        hedge_delay=0,
        download_timeout=120.0,
    )
    assert ds._batch_download_budget([0, 1, 2]) == 120.0
    wait_for_calls = {"n": 0}
    real_wait_for = asyncio.wait_for

    async def counting_wait_for(awaitable, timeout=None, **kwargs):  # type: ignore[no-untyped-def]
        wait_for_calls["n"] += 1
        return await real_wait_for(awaitable, timeout=timeout, **kwargs)

    monkeypatch.setattr(asyncio, "wait_for", counting_wait_for)
    items = ds.__getitems__([0, 1, 2])
    assert len(items) == 3
    assert wait_for_calls["n"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_batch_timeout_cancels_hung_inflight_for_recovery(tmp_path: Path) -> None:
    """Batch timeout must cancel poisoned ``_inflight`` so a retry can succeed promptly."""
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(f"val-{i}".encode())

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        max_prefetch=0,
        hedge_delay=0,
        download_timeout=0.25,
    )
    ds.items = sorted(ds.items, key=lambda m: m.path)

    hang = {"enabled": True}
    real = ds.cache_manager.download_file_async

    async def stub(file_path: str, size: int | None = None) -> bytes:
        if hang["enabled"] and file_path.endswith("f1.bin"):
            await asyncio.sleep(3600)
            return b"never"
        return await real(file_path, size=size)

    ds.cache_manager.download_file_async = stub  # type: ignore[method-assign]

    async def run() -> None:
        # Seed a hung prefetch entry whose download sleeps forever.
        task = asyncio.create_task(ds._prefetch_index(1))
        ds._inflight[1] = task
        ds._inflight_loop = asyncio.get_running_loop()
        await asyncio.sleep(0.05)

        with pytest.raises(TimeoutError, match="Batch download timed out"):
            await ds._download_batch([1])

        # Without cancelling _inflight, retry would await the same hung task and
        # pay the full budget again. Healthy stub + prompt success is the recovery.
        hang["enabled"] = False
        t0 = time.perf_counter()
        result = await ds._download_batch([1])
        elapsed = time.perf_counter() - t0
        assert result[0] == b"val-1"
        assert elapsed < 1.0
        assert 1 not in ds._inflight

    asyncio.run(run())


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_download_budget_scales_with_size(tmp_path: Path) -> None:
    """Sized objects use download_timeout as a floor, not a hard cap."""
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        download_timeout=10.0,
    )
    large = 2 * 1024 * 1024 * 1024  # 2 GiB
    size_floor = large / _HEDGE_ASSUMED_BANDWIDTH_BPS * 3.0
    budget = cm._download_budget(large)
    assert budget is not None
    assert budget >= size_floor
    assert budget >= 10.0
    assert cm._download_budget(None) == 10.0
    assert cm._download_budget(1024) == 10.0  # tiny → floor is download_timeout
    assert cm._download_budget(large, timeout=1.5) == 1.5  # explicit override


def test_path_is_cached_discards_stale_present_mark(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True)
    local = str(tmp_path / "cache" / "gone.bin")
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    Path(local).write_bytes(b"x")
    assert cm._path_is_cached(local)
    assert local in cm._present_paths
    Path(local).unlink()
    assert not cm._path_is_cached(local)
    assert local not in cm._present_paths


def test_hedge_skipped_for_large_file(tmp_path: Path) -> None:
    """Files >= 8 MB must not issue a duplicate whole-object GET."""
    assert _effective_hedge_delay(1.0, 8 * 1024 * 1024) is None
    assert _effective_hedge_delay(1.0, 16 * 1024 * 1024) is None
    assert _effective_hedge_delay(1.0, None) is None
    assert _effective_hedge_delay(1.0, 0) is None
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0.01,
        range_parallel_threshold=0,  # force whole-object path
    )
    remote = "s3://bucket/data/big.bin"
    calls = {"n": 0}
    payload = b"x" * 100

    async def run(size: int | None) -> bytes:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        async def once(path: str) -> bytes:
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return payload

        cm._downloader = SimpleNamespace(adownload_fileobj=once)  # type: ignore[assignment]
        return await cm._fetch_bytes(remote, size=size)

    assert asyncio.run(run(8 * 1024 * 1024)) == payload
    assert calls["n"] == 1
    calls["n"] = 0
    assert asyncio.run(run(None)) == payload
    assert calls["n"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_ranged_download_validates_short_part(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        range_chunk_size=4,
        range_parallel_threshold=1,
    )
    remote = "s3://bucket/data/obj.bin"
    size = 12

    async def run() -> None:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        def short_bytes(path: str, offset: int, length: int, scratch: str) -> bytes:
            return b"x" * (length - 1)  # always short

        cm._downloader = type("D", (), {"download_bytes": staticmethod(short_bytes)})()  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="short read"):
            await cm._ranged_download_bytes(remote, size)

    asyncio.run(run())


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_ranged_download_reassembles(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    payload = b"abcdefghijklmnopqrstuvwx"  # 24 bytes
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        range_chunk_size=8,
        range_parallel_threshold=1,
    )
    remote = "s3://bucket/data/obj.bin"
    scratches: list[str] = []
    call_order: list[int] = []

    async def run() -> bytes:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        def ranged(path: str, offset: int, length: int, scratch: str) -> bytes:
            scratches.append(scratch)
            call_order.append(offset)
            return payload[offset : offset + length]

        cm._downloader = type("D", (), {"download_bytes": staticmethod(ranged)})()  # type: ignore[assignment]
        return await cm._ranged_download_bytes(remote, len(payload))

    assert asyncio.run(run()) == payload
    assert len(scratches) == 3
    assert len(set(scratches)) == 3  # per-chunk scratch paths
    # Reassembly must be by ascending offset regardless of completion order.
    assert sorted(call_order) == [0, 8, 16]


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_ranged_download_size_mismatch(tmp_path: Path) -> None:
    """Joined payload length must match the declared object size."""
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        range_chunk_size=8,
        range_parallel_threshold=1,
    )
    remote = "s3://bucket/data/obj.bin"

    async def run() -> None:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        def ok_chunk(path: str, offset: int, length: int, scratch: str) -> bytes:
            return b"x" * length

        cm._downloader = type("D", (), {"download_bytes": staticmethod(ok_chunk)})()  # type: ignore[assignment]
        real_gather = asyncio.gather

        async def gather_truncate(*aws: Any, **kwargs: Any) -> Any:
            parts = await real_gather(*aws, **kwargs)
            parts = list(parts)
            off, data = parts[-1]
            parts[-1] = (off, data[:-1])
            return parts

        asyncio.gather = gather_truncate  # type: ignore[method-assign, assignment]
        try:
            with pytest.raises(RuntimeError, match="size mismatch"):
                await cm._ranged_download_bytes(remote, 24)
        finally:
            asyncio.gather = real_gather  # type: ignore[method-assign, assignment]

    asyncio.run(run())


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_ranged_chunk_hedge_uses_distinct_scratch(tmp_path: Path) -> None:
    """Concurrent first/hedge attempts for the same chunk must not share scratch paths."""
    src = tmp_path / "src"
    src.mkdir()
    cm = CacheManager(
        str(src),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0.05,
        range_chunk_size=8,
        range_parallel_threshold=1,
        max_concurrent_downloads=8,
    )
    remote = "s3://bucket/data/obj.bin"
    scratches: list[str] = []
    barrier = threading.Barrier(2, timeout=5)

    async def run() -> bytes:
        cm._downloader_pid = os.getpid()
        cm._downloader_loop = asyncio.get_running_loop()

        def ranged(path: str, offset: int, length: int, scratch: str) -> bytes:
            scratches.append(scratch)
            # Block both concurrent attempts (first + hedge) until both have started.
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait()
            return b"y" * length

        cm._downloader = type("D", (), {"download_bytes": staticmethod(ranged)})()  # type: ignore[assignment]
        return await cm._ranged_download_bytes(remote, 8)

    assert asyncio.run(run()) == b"yyyyyyyy"
    assert len(scratches) >= 2
    assert len(set(scratches)) == len(scratches)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_largest_first_preserves_result_order(tmp_path: Path) -> None:
    """Downloads may start largest-first, but batch results keep caller index order."""
    sizes = [10, 1000, 50]
    for i, n in enumerate(sizes):
        (tmp_path / f"f{i}.bin").write_bytes(bytes([i]) * n)

    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        cache_files=False,
        hedge_delay=0,
        max_prefetch=0,
    )
    # Stable item order f0, f1, f2 by path.
    ds.items = sorted(ds.items, key=lambda m: m.path)
    order: list[int] = []

    real = ds.cache_manager.download_file_async

    async def tracking(file_path: str, size: int | None = None) -> bytes:
        idx = int(Path(file_path).stem[1:])
        order.append(idx)
        await asyncio.sleep(0.01 * (1 if idx != 1 else 0))  # let scheduling show LPT start
        return await real(file_path, size=size)

    ds.cache_manager.download_file_async = tracking  # type: ignore[method-assign]
    batch = ds.__getitems__([0, 1, 2])
    assert [b[0] for b in batch] == [0, 1, 2]
    # Largest (index 1, 1000 bytes) should be started first among the three.
    assert order[0] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_duplicate_batch_indices_fanout(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"same")
    (tmp_path / "b.bin").write_bytes(b"other")
    ds = StreamingRawDataset(
        str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        hedge_delay=0,
        max_prefetch=0,
    )
    ds.items = sorted(ds.items, key=lambda m: m.path)
    calls = {"n": 0}
    real = ds.cache_manager.download_file_async

    async def counting(file_path: str, size: int | None = None) -> bytes:
        calls["n"] += 1
        return await real(file_path, size=size)

    ds.cache_manager.download_file_async = counting  # type: ignore[method-assign]
    batch = ds.__getitems__([0, 0, 1, 0])
    assert batch[0] == batch[1] == batch[3]
    assert batch[2] == b"other"
    # Index 0 materialized once despite three positions.
    assert calls["n"] == 2


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_download_to_cache_logs_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(b"fallback-ok")
    cm = CacheManager(str(src), cache_dir=str(tmp_path / "cache"), cache_files=True, hedge_delay=0)
    remote = str(src / "a.bin")

    async def run() -> str:
        async def boom(remote_filepath: str, local_filepath: str) -> None:
            raise OSError("stream failed")

        cm.downloader.adownload_file = boom  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="litdata.raw.dataset"):
            return await cm.ensure_file_async(remote)

    path = asyncio.run(run())
    assert Path(path).read_bytes() == b"fallback-ok"
    assert any("falling back to bytes path" in r.message for r in caplog.records)

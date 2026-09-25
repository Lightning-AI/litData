#!/usr/bin/env python3
"""Behavioral Adaptive Concurrency Stress Benchmark.

Demonstrates cause-and-effect chain:
  simulated network condition -> real async delay -> downloader boundary
  -> BandwidthTracker observation -> class-gated sample count
  -> stateful backoff/recovery -> dynamic semaphore budget -> runtime concurrency.

Calculates per-worker permit allocations across worker count configurations w in [4, 8, 16, 24].
"""

import asyncio
import os
import sys
import tempfile
import time
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath("src"))

from litdata.raw.dataset import (
    StreamingRawDataset,
    _aggregate_concurrency_budget,
    _DynamicSemaphore,
    _effective_concurrency,
)
from litdata.streaming.downloader import Downloader


class FakeDownloader(Downloader):
    """Deterministic network simulator at the Downloader ABC boundary."""

    def __init__(
        self,
        latency_s: float = 0.010,
        bandwidth_bps: float = 100 * 1024 * 1024,
        error_rate: float = 0.0,
        default_size: int = 30 * 1024,
        **kwargs: Any,
    ):
        """Initialize FakeDownloader with latency, bandwidth, and error rates."""
        super().__init__("", "", [], **kwargs)
        self.latency_s = latency_s
        self.bandwidth_bps = bandwidth_bps
        self.error_rate = error_rate
        self.default_size = default_size
        self._call_count = 0

    def _extract_requested_size(self, remote_filepath: str) -> int:
        """Parse size from remote_filepath query parameter or return default_size."""
        if "?size=" in remote_filepath:
            try:
                return int(remote_filepath.split("?size=")[1])
            except ValueError:
                pass
        return self.default_size

    async def adownload_fileobj(self, remote_filepath: str) -> bytes:
        """Simulate async object download with dual-component latency and payload transfer delay."""
        self._call_count += 1
        # Deterministic 429 rate-limiting simulation when error_rate > 0
        if self.error_rate > 0 and (self._call_count % 2 == 0):
            raise RuntimeError("HTTP 429 Too Many Requests")
        req_size = self._extract_requested_size(remote_filepath)
        transfer_s = req_size / max(1.0, self.bandwidth_bps)
        await asyncio.sleep(self.latency_s + transfer_s)
        return b"x" * req_size

    def download_bytes(self, remote_filepath: str, offset: int, length: int, local_chunkpath: str) -> bytes:
        """Simulate sync ranged download with dual-component timing delay."""
        self._call_count += 1
        if self.error_rate > 0 and (self._call_count % 2 == 0):
            raise RuntimeError("HTTP 429 Too Many Requests")
        transfer_s = length / max(1.0, self.bandwidth_bps)
        time.sleep(self.latency_s + transfer_s)
        data = b"x" * length
        with open(local_chunkpath, "wb") as f:
            f.write(data)
        return data


def run_benchmark() -> None:
    """Execute behavioral adaptive concurrency stress benchmark across worker matrix."""
    header_title = "BEHAVIORAL ADAPTIVE CONCURRENCY STRESS BENCHMARK"
    print("=" * 115)
    print(f"{header_title:^115}")
    print("=" * 115)
    print("Note: BPS (MB/s) = 0.00 in latency-only rows (<64 KiB GETs) is expected class-gated sample isolation.")
    print(
        f"{'Phase':<12} | {'Workers':<8} | {'Budget':<8} | {'Permits/W':<10} | "
        f"{'BPS (MB/s)':<12} | {'Lat (ms)':<10} | {'Tput (MB/s)':<12} | {'Errors':<8}"
    )
    print("-" * 115)

    worker_counts = [4, 8, 16, 24]
    latency_test_size = 30 * 1024  # <64 KiB (latency population only)
    medium_test_size = 128 * 1024  # 64-256 KiB (both latency and bandwidth populations)
    bandwidth_test_size = 500 * 1024  # >=256 KiB (bandwidth population only)

    for w in worker_counts:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = os.path.join(tmp_dir, "sample.bin")
            with open(sample_file, "wb") as f:
                f.write(b"x" * 1000)

            ds = StreamingRawDataset(
                input_dir=tmp_dir,
                cache_dir=tmp_dir,
            )

            # Phase 1: Healthy Baseline (10 ms latency, 100 MB/s bandwidth)
            fake_dl_healthy = FakeDownloader(
                latency_s=0.010,
                bandwidth_bps=100 * 1024 * 1024,
                default_size=latency_test_size,
            )

            # Correction 2: Direct timing fidelity check on FakeDownloader using tolerance
            t0_direct = time.monotonic()
            asyncio.run(fake_dl_healthy.adownload_fileobj(f"s3://mock-bucket/file.bin?size={latency_test_size}"))
            direct_dur = time.monotonic() - t0_direct
            expected_dur = fake_dl_healthy.latency_s + (latency_test_size / fake_dl_healthy.bandwidth_bps)
            tolerance = max(0.015, expected_dur * 0.5)
            assert abs(direct_dur - expected_dur) <= tolerance, (
                f"Direct timing ({direct_dur:.4f}s) must be within tolerance of expected ({expected_dur:.4f}s)"
            )

            async def _run_phase1():
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_healthy
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={latency_test_size}", size=latency_test_size
                    )
                    for _ in range(6)
                ]
                await asyncio.gather(*tasks)

            t0 = time.monotonic()
            asyncio.run(_run_phase1())
            elapsed = time.monotonic() - t0

            tracker = ds.cache_manager._bandwidth_tracker
            budget_healthy = _aggregate_concurrency_budget(latency_test_size, tracker=tracker)
            permits_w_h = _effective_concurrency(
                None, num_workers=w, median_file_bytes=latency_test_size, tracker=tracker
            )
            bps_h, lat_h, _, lat_cnt_h = tracker.get_metrics()
            bps_mbs_h = (bps_h / (1024 * 1024)) if bps_h else 0.0
            lat_ms_h = (lat_h * 1000) if lat_h else 0.0
            tput_h = (6 * latency_test_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

            print(
                f"{'Healthy':<12} | {w:<8} | {budget_healthy:<8} | {permits_w_h:<10} | "
                f"{bps_mbs_h:<12.2f} | {lat_ms_h:<10.2f} | {tput_h:<12.2f} | {0:<8}"
            )

            assert lat_cnt_h >= 5, "Small GETs must update latency sample count"

            # Phase 2: Congested / Slow Link (200 ms latency = 5x target)
            fake_dl_congested = FakeDownloader(
                latency_s=0.200,
                bandwidth_bps=10 * 1024 * 1024,
                default_size=latency_test_size,
            )

            async def _run_phase2():
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_congested
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={latency_test_size}", size=latency_test_size
                    )
                    for _ in range(6)
                ]
                await asyncio.gather(*tasks)

            t0 = time.monotonic()
            asyncio.run(_run_phase2())
            elapsed = time.monotonic() - t0

            budget_congested = _aggregate_concurrency_budget(latency_test_size, tracker=tracker)
            permits_w_c = _effective_concurrency(
                None, num_workers=w, median_file_bytes=latency_test_size, tracker=tracker
            )
            bps_c, lat_c, _, _ = tracker.get_metrics()
            bps_mbs_c = (bps_c / (1024 * 1024)) if bps_c else 0.0
            lat_ms_c = (lat_c * 1000) if lat_c else 0.0
            tput_c = (6 * latency_test_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

            print(
                f"{'Congested':<12} | {w:<8} | {budget_congested:<8} | {permits_w_c:<10} | "
                f"{bps_mbs_c:<12.2f} | {lat_ms_c:<10.2f} | {tput_c:<12.2f} | {0:<8}"
            )

            assert budget_congested < budget_healthy, (
                f"Congestion budget ({budget_congested}) must be < healthy ({budget_healthy})"
            )

            # Phase 3: Operational Rate-Limiting (Deterministic HTTP 429 Simulation)
            fake_dl_429 = FakeDownloader(
                latency_s=0.010,
                bandwidth_bps=100 * 1024 * 1024,
                error_rate=0.5,
                default_size=latency_test_size,
            )

            errors_429 = 0

            async def _run_phase3():
                nonlocal errors_429
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_429
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={latency_test_size}", size=latency_test_size
                    )
                    for _ in range(10)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        errors_429 += 1

            t0 = time.monotonic()
            asyncio.run(_run_phase3())
            elapsed = time.monotonic() - t0

            budget_429 = _aggregate_concurrency_budget(latency_test_size, tracker=tracker)
            permits_w_429 = _effective_concurrency(
                None, num_workers=w, median_file_bytes=latency_test_size, tracker=tracker
            )
            bps_429, lat_429_v, _, _ = tracker.get_metrics()
            bps_mbs_429 = (bps_429 / (1024 * 1024)) if bps_429 else 0.0
            lat_ms_429 = (lat_429_v * 1000) if lat_429_v else 0.0
            tput_429 = (10 * latency_test_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

            print(
                f"{'Rate-Limit':<12} | {w:<8} | {budget_429:<8} | {permits_w_429:<10} | "
                f"{bps_mbs_429:<12.2f} | {lat_ms_429:<10.2f} | {tput_429:<12.2f} | {errors_429:<8}"
            )

            # Phase 4: Multi-Step Gradual Recovery (Healthy latency restored)
            fake_dl_recovery = FakeDownloader(
                latency_s=0.010,
                bandwidth_bps=100 * 1024 * 1024,
                default_size=latency_test_size,
            )

            recovery_budgets = [budget_congested]

            async def _run_phase4_step():
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_recovery
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={latency_test_size}", size=latency_test_size
                    )
                    for _ in range(2)
                ]
                await asyncio.gather(*tasks)

            for step in range(3):
                asyncio.run(_run_phase4_step())
                b_step = _aggregate_concurrency_budget(latency_test_size, tracker=tracker)
                recovery_budgets.append(b_step)

            budget_recovered = recovery_budgets[-1]
            permits_w_rec = _effective_concurrency(
                None, num_workers=w, median_file_bytes=latency_test_size, tracker=tracker
            )
            bps_rec, lat_rec, _, _ = tracker.get_metrics()
            bps_mbs_rec = (bps_rec / (1024 * 1024)) if bps_rec else 0.0
            lat_ms_rec = (lat_rec * 1000) if lat_rec else 0.0

            print(
                f"{'Recovered':<12} | {w:<8} | {budget_recovered:<8} | {permits_w_rec:<10} | "
                f"{bps_mbs_rec:<12.2f} | {lat_ms_rec:<10.2f} | {'N/A':<12} | {0:<8}"
            )

            for prev_b, curr_b in zip(recovery_budgets, recovery_budgets[1:]):
                assert curr_b >= prev_b, f"Recovery sequence must be non-decreasing: {recovery_budgets}"

            assert budget_recovered > budget_congested, (
                f"Recovered budget ({budget_recovered}) must exceed congested budget ({budget_congested})"
            )

            assert budget_recovered <= budget_healthy, (
                f"Recovered budget ({budget_recovered}) must not exceed baseline ({budget_healthy})"
            )

            assert any(b < budget_healthy for b in recovery_budgets[1:-1]), (
                f"Recovery sequence ({recovery_budgets}) must be gradual and not jump immediately "
                f"to healthy baseline ({budget_healthy})"
            )

            # Explicit Medium-Object Test (64 KiB <= size < 256 KiB) -> Increments BOTH counters
            fake_dl_medium = FakeDownloader(
                latency_s=0.010,
                bandwidth_bps=50 * 1024 * 1024,
                default_size=medium_test_size,
            )

            async def _run_medium_test():
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_medium
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={medium_test_size}", size=medium_test_size
                    )
                    for _ in range(1)
                ]
                await asyncio.gather(*tasks)

            _, _, bps_before, lat_before = tracker.get_metrics()
            asyncio.run(_run_medium_test())
            _, _, bps_after, lat_after = tracker.get_metrics()

            assert bps_after == bps_before + 1, "Medium GET (128 KiB) must increment bps_sample_count"
            assert lat_after == lat_before + 1, "Medium GET (128 KiB) must increment lat_sample_count"

            # Large Object Test (>=256 KiB): Bandwidth Population Isolation
            fake_dl_large = FakeDownloader(
                latency_s=0.010,
                bandwidth_bps=50 * 1024 * 1024,
                default_size=bandwidth_test_size,
            )

            async def _run_large_test():
                loop = asyncio.get_running_loop()
                ds.cache_manager._downloader = fake_dl_large
                ds.cache_manager._downloader_pid = os.getpid()
                ds.cache_manager._downloader_loop = loop
                tasks = [
                    ds.cache_manager._fetch_bytes(
                        f"s3://mock-bucket/file.bin?size={bandwidth_test_size}", size=bandwidth_test_size
                    )
                    for _ in range(5)
                ]
                await asyncio.gather(*tasks)

            asyncio.run(_run_large_test())
            _, _, bps_cnt_large, _ = tracker.get_metrics()

            assert bps_cnt_large >= 5, "Large GETs must update bandwidth sample count"

            ds.cache_manager.reset_runtime_state()
            print("-" * 115)

    # Correction 1: Focused in-flight semaphore lifecycle check
    # Proves 32 initial holders remain valid on downscale to 16, release 16 -> 17th acquire blocks,
    # release remaining -> new acquisitions succeed up to 16

    async def _run_semaphore_invariant_check():
        sem = _DynamicSemaphore(32)
        # Step 1: Acquire all 32 initial permits (32 holders in-flight)
        for _ in range(32):
            await sem.acquire()

        # Step 2: Target reduced to 16 mid-flight
        sem.update_target(16)
        assert sem.target_permits == 16

        # Step 3: Existing 32 holders remain valid, release 16 of them
        for _ in range(16):
            sem.release()

        # Step 4: 16 holders still remain active — new acquisition attempt must block/timeout
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)

        # Step 5: Release remaining 16 holders
        for _ in range(16):
            sem.release()

        # Step 6: Verify 16 new acquisitions succeed cleanly
        for _ in range(16):
            await sem.acquire()
        for _ in range(16):
            sem.release()

    asyncio.run(_run_semaphore_invariant_check())

    print("=" * 115)
    print("Behavioral adaptive concurrency benchmark completed successfully; all assertions passed.")


if __name__ == "__main__":
    run_benchmark()

"""Fixtures for raw streaming tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _shutdown_raw_loop_runner():
    """Ensure the process-local LoopRunner does not leak threads across tests."""
    yield
    from litdata.raw import dataset as raw_dataset

    runner = raw_dataset._RUNNER
    raw_dataset._shutdown_runner_before_fork()
    if runner is not None:
        # Production shutdown is best-effort. Tests must drain the executor before
        # the session's thread-leak assertion, including already-completed local I/O.
        runner._executor.shutdown(wait=True, cancel_futures=True)

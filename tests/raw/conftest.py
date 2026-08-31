"""Fixtures for raw streaming tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _shutdown_raw_loop_runner():
    """Ensure the process-local LoopRunner does not leak threads across tests."""
    yield
    from litdata.raw import dataset as raw_dataset

    raw_dataset._shutdown_runner_before_fork()

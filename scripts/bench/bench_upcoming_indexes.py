#!/usr/bin/env python3
"""Microbench: list.pop(0) vs deque.popleft for upcoming_indexes consumption.

Example:
  .venv/bin/python scripts/bench/bench_upcoming_indexes.py
"""

from __future__ import annotations

import time
from collections import deque


def _bench_list(n: int, rounds: int = 5) -> float:
    best = float("inf")
    for _ in range(rounds):
        indexes = list(range(n))
        t0 = time.perf_counter()
        while indexes:
            indexes.pop(0)
        best = min(best, time.perf_counter() - t0)
    return best


def _bench_deque(n: int, rounds: int = 5) -> float:
    best = float("inf")
    for _ in range(rounds):
        indexes = deque(range(n))
        t0 = time.perf_counter()
        while indexes:
            indexes.popleft()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Print pop(0) vs popleft wall times for increasing index-list sizes."""
    for n in (1_000, 10_000, 100_000):
        list_s = _bench_list(n)
        deque_s = _bench_deque(n)
        print(
            f"n={n:>7}  list.pop(0)={list_s * 1e3:.3f}ms  "
            f"deque.popleft={deque_s * 1e3:.3f}ms  "
            f"speedup={list_s / deque_s:.1f}x"
        )


if __name__ == "__main__":
    main()

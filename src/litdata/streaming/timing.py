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

"""Lightweight streaming timing counters for prefetch / decode / yield analysis.

Enable with ``LITDATA_TIMING=1``. Counters are process-local and cheap (``time.perf_counter``
deltas aggregated under a lock). With ``num_workers > 0``, each DataLoader worker has its
own singleton — the parent process snapshot will not include worker download/decode time.
Use :meth:`StreamingTimingStats.snapshot` from benches (prefer ``num_workers=0`` for a
single-process view) or debug sessions; chrome-trace spans remain in :mod:`litdata.debugger`.
"""

from __future__ import annotations

import os
import threading
import time


def _timing_enabled() -> bool:
    return os.getenv("LITDATA_TIMING", "0").lower() in {"1", "true", "yes"}


class StreamingTimingStats:
    """Aggregated wall-clock seconds for streaming hot-path stages."""

    _singleton: StreamingTimingStats | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = _timing_enabled()
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.totals_s: dict[str, float] = {}

    @classmethod
    def instance(cls) -> StreamingTimingStats:
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = StreamingTimingStats()
        return cls._singleton

    @classmethod
    def reset_instance(cls) -> StreamingTimingStats:
        """Replace the process singleton (useful in tests / benches)."""
        with cls._singleton_lock:
            cls._singleton = StreamingTimingStats()
            return cls._singleton

    def start(self) -> float | None:
        if not self.enabled:
            return None
        return time.perf_counter()

    def record(self, name: str, start: float | None) -> None:
        if start is None or not self.enabled:
            return
        elapsed = time.perf_counter() - start
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            self.totals_s[name] = self.totals_s.get(name, 0.0) + elapsed

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            out: dict[str, dict[str, float | int]] = {}
            for name, total in self.totals_s.items():
                count = self.counts.get(name, 0)
                out[name] = {
                    "count": count,
                    "total_s": total,
                    "mean_s": (total / count) if count else 0.0,
                }
            return out

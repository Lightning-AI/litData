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

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from litdata.utilities.env import _DistributedEnv, _WorkerEnv, _is_in_dataloader_worker

# ---------------------------------------------------------------------------
# Trace categories / levels
# ---------------------------------------------------------------------------

CAT_EPOCH = "epoch"
CAT_BATCH = "batch"
CAT_DOWNLOAD = "download"
CAT_READ = "read"
CAT_DELETE = "delete"
CAT_DECOMPRESS = "decompress"
CAT_SAMPLE = "sample"
CAT_LOCK = "lock"
CAT_CRASH = "crash"

ALL_CATEGORIES = (
    CAT_EPOCH,
    CAT_BATCH,
    CAT_DOWNLOAD,
    CAT_READ,
    CAT_DELETE,
    CAT_DECOMPRESS,
    CAT_SAMPLE,
    CAT_LOCK,
    CAT_CRASH,
)

LEVEL_CATEGORIES: dict[str, frozenset[str]] = {
    "off": frozenset(),
    "batch": frozenset({CAT_EPOCH, CAT_BATCH, CAT_CRASH}),
    "chunk": frozenset(
        {CAT_EPOCH, CAT_BATCH, CAT_DOWNLOAD, CAT_READ, CAT_DELETE, CAT_DECOMPRESS, CAT_CRASH}
    ),
    "sample": frozenset(
        {
            CAT_EPOCH,
            CAT_BATCH,
            CAT_DOWNLOAD,
            CAT_READ,
            CAT_DELETE,
            CAT_DECOMPRESS,
            CAT_SAMPLE,
            CAT_CRASH,
        }
    ),
    "debug": frozenset(ALL_CATEGORIES),
}

_CAT_CNAME = {
    CAT_EPOCH: "memory_dump",
    CAT_BATCH: "thread_state_runnable",
    CAT_DOWNLOAD: "rail_load",
    CAT_READ: "good",
    CAT_DELETE: "rail_animate",
    CAT_DECOMPRESS: "cq_build_attempt_running",
    CAT_SAMPLE: "generic_work",
    CAT_LOCK: "thread_state_sleeping",
    CAT_CRASH: "terrible",
}

_ACTIVE_CATEGORIES: frozenset[str] = frozenset()
_tracer_logger = logging.getLogger("litdata")


def is_tracing(category: str) -> bool:
    """True when ``enable_tracer()`` turned on this category. Cheap enough for hot paths."""
    return category in _ACTIVE_CATEGORIES


def active_categories() -> frozenset[str]:
    return _ACTIVE_CATEGORIES


def _parse_categories(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    out = set()
    for part in raw.split(","):
        name = part.strip().lower()
        if name:
            out.add(name)
    return frozenset(out)


def _categories_for_level(level: str) -> frozenset[str]:
    key = level.strip().lower()
    if key not in LEVEL_CATEGORIES:
        valid = ", ".join(LEVEL_CATEGORIES)
        raise ValueError(f"Unknown tracer level {level!r}. Expected one of: {valid}")
    return LEVEL_CATEGORIES[key]


def _set_active_categories(cats: frozenset[str]) -> None:
    global _ACTIVE_CATEGORIES
    _ACTIVE_CATEGORIES = cats
    os.environ["LITDATA_TRACE_CATEGORIES"] = ",".join(sorted(cats))


def _load_active_categories_from_env() -> frozenset[str]:
    cats = _parse_categories(os.getenv("LITDATA_TRACE_CATEGORIES"))
    if cats:
        return cats
    level = os.getenv("LITDATA_TRACE_LEVEL", "")
    if level:
        return _categories_for_level(level)
    return frozenset()


_ACTIVE_CATEGORIES = _load_active_categories_from_env()


class TimedFlushFileHandler(logging.FileHandler):
    """FileHandler that flushes every N seconds in a background thread."""

    def __init__(self, filename, mode="a", flush_interval=2):
        super().__init__(filename, mode)
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()
        t = threading.Thread(target=self._flusher, daemon=True, name="TimedFlushFileHandler._flusher")
        t.start()

    def _flusher(self):
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self.flush()

    def close(self):
        self._stop_event.set()
        self.flush()
        super().close()


class EnvConfigFilter(logging.Filter):
    """Drop tracer records whose ``cat:`` is not in the active set."""

    def filter(self, record):
        if not _ACTIVE_CATEGORIES:
            return True
        msg = record.getMessage()
        marker = "cat:"
        idx = msg.find(marker)
        if idx < 0:
            return True
        rest = msg[idx + len(marker) :]
        cat = rest.split(";", 1)[0].strip()
        return cat in _ACTIVE_CATEGORIES


def get_logger_level(level: str) -> int:
    level = level.upper()
    if level in logging._nameToLevel:
        return logging._nameToLevel[level]
    raise ValueError(f"Invalid log level: {level}")


class LitDataLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name="litdata", flush_interval=2):
        if hasattr(self, "logger"):
            return  # Already initialized

        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.log_file, self.log_level = self.get_log_file_and_level()
        self.flush_interval = flush_interval
        self._setup_logger()

    @staticmethod
    def get_log_file_and_level():
        log_file = os.getenv("LITDATA_LOG_FILE", "litdata_debug.log")
        log_lvl = os.getenv("LITDATA_LOG_LEVEL", "DEBUG")
        return log_file, get_logger_level(log_lvl)

    def _setup_logger(self):
        if self.logger.handlers:
            return
        self.logger.setLevel(self.log_level)
        formatter = _OneLineTraceFormatter("ts:%(asctime)s;PID:%(process)d; TID:%(thread)d; %(message)s")
        handler = TimedFlushFileHandler(self.log_file, flush_interval=self.flush_interval)
        handler.setFormatter(formatter)
        handler.setLevel(self.log_level)
        self.logger.addHandler(handler)

        self.logger.filters = [f for f in self.logger.filters if not isinstance(f, EnvConfigFilter)]
        self.logger.addFilter(EnvConfigFilter())

    def get_logger(self):
        return self.logger


def enable_tracer(
    level: str = "chunk",
    *,
    categories: Sequence[str] | None = None,
    flush_interval: int = 5,
    log_file: str = "litdata_debug.log",
    item_loader: bool | None = None,
    iterating_dataset: bool | None = None,
    getitem_dataset_for_chunk_index: bool | None = None,
) -> logging.Logger:
    """Enable LitData pipeline tracing for Litracer / Perfetto.

    Call once per process before creating the DataLoader.

    Levels (each includes the ones above it, plus crashes):

    - ``batch``: dataloader epoch + per-batch spans
    - ``chunk`` (default): download, read, delete, decompress, prefetch
    - ``sample``: per-item getitem spans (high volume)
    - ``debug``: lock refcount spans as well
    - ``off``: disable

    Or pass an explicit ``categories`` list such as
    ``["download", "read", "delete"]``.

    ``log_file`` is the Litracer input (one semicolon-delimited event per line).
    """
    os.environ["LITDATA_LOG_FILE"] = log_file
    os.environ["LITDATA_TRACE_LEVEL"] = level
    os.environ["LITDATA_LOG_LEVEL"] = os.getenv("LITDATA_LOG_LEVEL", "DEBUG")

    cats = frozenset(c.strip().lower() for c in categories) if categories is not None else _categories_for_level(level)

    if iterating_dataset is False:
        cats -= {CAT_EPOCH}
    if item_loader is False or getitem_dataset_for_chunk_index is False:
        cats -= {CAT_SAMPLE}

    os.environ["LITDATA_LOG_ITEM_LOADER"] = str(item_loader if item_loader is not None else CAT_SAMPLE in cats)
    os.environ["LITDATA_LOG_ITERATING_DATASET"] = str(
        iterating_dataset if iterating_dataset is not None else CAT_EPOCH in cats
    )
    os.environ["LITDATA_LOG_GETITEM"] = str(
        getitem_dataset_for_chunk_index if getitem_dataset_for_chunk_index is not None else CAT_SAMPLE in cats
    )

    _set_active_categories(cats)
    return LitDataLogger(flush_interval=flush_interval).get_logger()


def emit_trace(name: str, ph: str, cat: str, **fields: Any) -> None:
    """Write one Chrome-trace event if ``cat`` is enabled. No-op when the tracer is off."""
    if cat not in _ACTIVE_CATEGORIES:
        return
    payload: dict[str, Any] = {"name": name, "ph": ph, "cat": cat}
    cname = fields.pop("cname", None) or _CAT_CNAME.get(cat)
    if cname:
        payload["cname"] = cname
    payload.update(fields)
    _tracer_logger.debug(_get_log_msg(payload))


@contextmanager
def trace_span(name: str, cat: str, **fields: Any) -> Iterator[None]:
    """Begin/end a duration event. No-op when ``cat`` is disabled."""
    if cat not in _ACTIVE_CATEGORIES:
        yield
        return
    emit_trace(name, "B", cat, **fields)
    try:
        yield
    finally:
        emit_trace(name, "E", cat, **fields)


def _sanitize_log_value(value: object) -> str:
    """Keep a tracer field on one semicolon-delimited kv pair.

    Litracer parses each line as ``key: value;`` pairs. Newlines split the file
    into extra records (usually dropped for missing ``ph``). Semicolons inside a
    value would invent extra keys.
    """
    return str(value).replace(";", ",").replace("\r", " ").replace("\n", " ")


class _OneLineTraceFormatter(logging.Formatter):
    """Write exactly one Litracer line per log record.

    Chrome / Perfetto timestamps are microseconds. ``record.created`` is Unix
    seconds, so ``formatTime`` emits ``created * 1e6``. ``logger.exception``
    would otherwise append a traceback, which Litracer cannot parse.
    """

    def formatTime(self, record, datefmt=None):  # noqa: N802
        return f"{record.created * 1_000_000:.3f}"

    def formatException(self, ei):  # noqa: N802
        return ""

    def formatStack(self, stack_info):  # noqa: N802
        return ""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace("\r", " ").replace("\n", " ")


def _get_log_msg(data: dict) -> str:
    if "name" not in data or "ph" not in data:
        raise ValueError(f"Missing required keys in data dictionary. Required keys: 'name', 'ph'. Received: {data}")
    parts: list[str] = []
    for key, value in data.items():
        parts.append(f"{key}: {_sanitize_log_value(value)};")
    info = env_info()
    for key, value in info.items():
        if key not in data:
            parts.append(f"{key}: {_sanitize_log_value(value)};")
    return "".join(parts)


def env_info() -> dict:
    if _is_in_dataloader_worker():
        return _cached_env_info()

    dist_env = _DistributedEnv.detect()
    worker_env = _WorkerEnv.detect()
    return {
        "dist_world_size": dist_env.world_size,
        "dist_global_rank": dist_env.global_rank,
        "dist_num_nodes": dist_env.num_nodes,
        "worker_world_size": worker_env.world_size,
        "worker_rank": worker_env.rank,
    }


@lru_cache(maxsize=1)
def _cached_env_info() -> dict:
    dist_env = _DistributedEnv.detect()
    worker_env = _WorkerEnv.detect()
    return {
        "dist_world_size": dist_env.world_size,
        "dist_global_rank": dist_env.global_rank,
        "dist_num_nodes": dist_env.num_nodes,
        "worker_world_size": worker_env.world_size,
        "worker_rank": worker_env.rank,
    }


# Chrome trace colors
class ChromeTraceColors:
    PINK = "thread_state_iowait"
    GREEN = "thread_state_running"
    LIGHT_BLUE = "thread_state_runnable"
    LIGHT_GRAY = "thread_state_sleeping"
    BROWN = "thread_state_unknown"
    BLUE = "memory_dump"
    GRAY = "generic_work"
    DARK_GREEN = "good"
    ORANGE = "bad"
    RED = "terrible"
    BLACK = "black"
    BRIGHT_BLUE = "rail_response"
    BRIGHT_RED = "rail_animate"
    ORANGE_YELLOW = "rail_idle"
    TEAL = "rail_load"
    DARK_BLUE = "used_memory_column"
    LIGHT_SKY_BLUE = "older_used_memory_column"
    MEDIUM_GRAY = "tracing_memory_column"
    PALE_YELLOW = "cq_build_running"
    LIGHT_GREEN = "cq_build_passed"
    LIGHT_RED = "cq_build_failed"
    MUSTARD_YELLOW = "cq_build_attempt_running"
    NEON_GREEN = "cq_build_attempt_passed"
    DARK_RED = "cq_build_attempt_failed"

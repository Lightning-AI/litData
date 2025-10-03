# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import threading
import time
from functools import lru_cache

from litdata.utilities.env import _DistributedEnv, _is_in_dataloader_worker, _WorkerEnv


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
    """A logging filter that reads its configuration from environment variables."""

    def __init__(self):
        super().__init__()
        self.name_re = re.compile(r"name:\s*([^;]+);")

    def _get_name_from_msg(self, msg):
        match = self.name_re.search(msg)
        return match.group(1).strip() if match else None

    def filter(self, record):
        """Determine if a log record should be processed by checking env vars."""
        is_iterating_dataset_enabled = os.getenv("LITDATA_LOG_ITERATING_DATASET", "True").lower() == "true"
        is_getitem_enabled = os.getenv("LITDATA_LOG_GETITEM", "True").lower() == "true"
        is_item_loader_enabled = os.getenv("LITDATA_LOG_ITEM_LOADER", "True").lower() == "true"

        log_name = self._get_name_from_msg(record.getMessage())

        if log_name:
            if not is_iterating_dataset_enabled and log_name.startswith("iterating_dataset"):
                return False
            if not is_getitem_enabled and log_name.startswith("getitem_dataset_for_chunk_index"):
                return False
            if not is_item_loader_enabled and log_name.startswith("item_loader"):
                return False

        return True


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
        formatter = logging.Formatter("ts:%(created)s;PID:%(process)d; TID:%(thread)d; %(message)s")
        handler = TimedFlushFileHandler(self.log_file, flush_interval=self.flush_interval)
        handler.setFormatter(formatter)
        handler.setLevel(self.log_level)
        self.logger.addHandler(handler)

        self.logger.filters = [f for f in self.logger.filters if not isinstance(f, EnvConfigFilter)]
        self.logger.addFilter(EnvConfigFilter())

    def get_logger(self):
        return self.logger


def enable_tracer(
    flush_interval: int = 5, item_loader=True, iterating_dataset=True, getitem_dataset_for_chunk_index=True
) -> logging.Logger:
    """Convenience function to enable and configure litdata logging.
    This function SETS the environment variables that control the logging behavior.
    """
    os.environ["LITDATA_LOG_FILE"] = "litdata_debug.log"
    os.environ["LITDATA_LOG_ITEM_LOADER"] = str(item_loader)
    os.environ["LITDATA_LOG_ITERATING_DATASET"] = str(iterating_dataset)
    os.environ["LITDATA_LOG_GETITEM"] = str(getitem_dataset_for_chunk_index)

    master_logger = LitDataLogger(flush_interval=flush_interval).get_logger()
    return master_logger


def _get_log_msg(data: dict) -> str:
    log_msg = ""
    if "name" not in data or "ph" not in data:
        raise ValueError(f"Missing required keys in data dictionary. Required keys: 'name', 'ph'. Received: {data}")
    env_info_data = env_info()
    data.update(env_info_data)
    for key, value in data.items():
        log_msg += f"{key}: {value};"
    return log_msg


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

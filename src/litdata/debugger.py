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

import logging
import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Tuple

from litdata.constants import _PRINT_DEBUG_LOGS
from litdata.utilities.env import _DistributedEnv, _WorkerEnv

# Create the root logger for the library
root_logger = logging.getLogger("litdata")


def get_logger_level(level: str) -> int:
    """Get the log level from the level string."""
    level = level.upper()
    if level in logging._nameToLevel:
        return logging._nameToLevel[level]
    raise ValueError(f"Invalid log level: {level}. Valid levels: {list(logging._nameToLevel.keys())}.")


class TimeWindowFilter(logging.Filter):
    """Filter log records based on a time window."""

    def __init__(self, start_time: int = None, end_time: int = None):
        """Initialize the filter with start and end time in seconds."""
        super().__init__()
        self.start = datetime.now()
        self.start_time = timedelta(seconds=start_time) if start_time else timedelta(seconds=0)
        self.end_time = timedelta(seconds=end_time) if end_time else None
        if self.end_time and self.start_time > self.end_time:
            raise ValueError("Start time must be less than or equal to end time.")

    def filter(self, record: logging.LogRecord) -> bool:
        now = datetime.now()
        elapsed = now - self.start
        if self.end_time:
            return self.start_time <= elapsed <= self.end_time
        return elapsed >= self.start_time


class LitDataLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.log_file, self.log_level = self.get_log_file_and_level()
        self.setup_logger()

    @staticmethod
    def get_log_file_and_level() -> Tuple[str, int]:
        log_file = os.getenv("LITDATA_LOG_FILE", "litdata_debug.log")
        log_lvl = os.getenv("LITDATA_LOG_LEVEL", "DEBUG")

        log_lvl = get_logger_level(log_lvl)

        return log_file, log_lvl

    def setup_logger(self) -> None:
        """Configures logging by adding handlers and formatting."""
        if len(self.logger.handlers) > 0:  # Avoid duplicate handlers
            return

        self.logger.setLevel(self.log_level)

        # Log format
        formatter = logging.Formatter(
            "ts:%(created)s; logger_name:%(name)s; level:%(levelname)s; PID:%(process)d; TID:%(thread)d; %(message)s"
        )
        # ENV - f"{WORLD_SIZE, GLOBAL_RANK, NNODES, LOCAL_RANK, NODE_RANK}"

        # Time window filter
        start_time = int(os.getenv("LITDATA_LOG_START_TIME", 0))
        end_time = os.getenv("LITDATA_LOG_END_TIME")
        end_time = int(end_time) if end_time else None
        time_window_filter = TimeWindowFilter(start_time, end_time)

        # Console handler
        if _PRINT_DEBUG_LOGS:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(time_window_filter)
            self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(time_window_filter)
        self.logger.addHandler(file_handler)


def enable_tracer() -> None:
    os.environ["LITDATA_LOG_FILE"] = "litdata_debug.log"
    LitDataLogger("litdata")


def _get_log_msg(data: dict) -> str:
    log_msg = ""

    if "name" not in data or "ph" not in data:
        raise ValueError(f"Missing required keys in data dictionary. Required keys: 'name', 'ph'. Received: {data}")

    env_info_data = env_info()
    data.update(env_info_data)

    for key, value in data.items():
        log_msg += f"{key}: {value};"
    return log_msg


@lru_cache(maxsize=1)
def env_info() -> dict:
    dist_env = _DistributedEnv.detect()
    worker_env = _WorkerEnv.detect()  # will all threads read the same value if decorate this function with `@cache`

    return {
        "dist_world_size": dist_env.world_size,
        "dist_global_rank": dist_env.global_rank,
        "dist_num_nodes": dist_env.num_nodes,
        "worker_world_size": worker_env.world_size,
        "worker_rank": worker_env.rank,
    }


# -> Chrome tracing colors
#     url: https://chromium.googlesource.com/external/trace-viewer/+/bf55211014397cf0ebcd9e7090de1c4f84fc3ac0/tracing/tracing/ui/base/color_scheme.html

# # ------


# thread_state_iowait: {r: 182, g: 125, b: 143},
# thread_state_running: {r: 126, g: 200, b: 148},
# thread_state_runnable: {r: 133, g: 160, b: 210},
# ....
class ChromeTraceColors:
    """Predefined Chrome tracing colors."""

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

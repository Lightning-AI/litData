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
import shutil
from typing import Any

from litdata.constants import _TQDM_AVAILABLE

_FORMAT_TO_RATIO = {
    "tb": 1000**4,
    "gb": 1000**3,
    "mb": 1000**2,
    "kb": 1000,
    "t": 1000**4,
    "g": 1000**3,
    "m": 1000**2,
    "k": 1000,
    "b": 1,
}


def _convert_bytes_to_int(bytes_str: str) -> int:
    """Convert human-readable byte format to an integer (``100G`` / ``100GB``)."""
    raw = bytes_str.lower().strip()
    for suffix in _FORMAT_TO_RATIO:
        if raw.endswith(suffix):
            try:
                return int(float(raw[: -len(suffix)]) * _FORMAT_TO_RATIO[suffix])
            except ValueError:
                raise ValueError(
                    f"Unsupported value/suffix {bytes_str}. Supported suffix are {list(_FORMAT_TO_RATIO.keys())}."
                ) from None
    raise ValueError(f"The supported units are {list(_FORMAT_TO_RATIO.keys())}.")


def _human_readable_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1000.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1000.0
    return f"{num_bytes:.1f} PB"


def _get_tqdm_iterator_if_available() -> Any:
    if _TQDM_AVAILABLE:
        from tqdm.auto import tqdm as _tqdm

        return _tqdm

    def _pass_through(iterator: Any, *args: Any, **kwargs: Any) -> Any:
        yield from iterator

    return _pass_through


_CACHE_FREE_FRACTION = 0.75
_CACHE_RESERVE_BYTES = _FORMAT_TO_RATIO["gb"] * 50  # leave room for checkpoints

logger = logging.getLogger("litdata.utilities.format")


def _probe_existing_dir(path: str | None) -> str:
    probe = os.path.abspath(path or os.getcwd())
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return os.getcwd()
        probe = parent
    return probe or os.getcwd()


def _adaptive_max_cache_size(
    path: str | None = None,
    *,
    fraction: float = _CACHE_FREE_FRACTION,
    leave_reserve: bool = True,
) -> int:
    """Cache budget from currently free disk.

    ``leave_reserve`` keeps ≥50GB free when that much is available (default ``None``
    path). An explicit fraction such as ``0.90`` uses that share of free space as-is.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"`max_cache_size` fraction must be in (0, 1], got {fraction}.")
    probe = _probe_existing_dir(path)
    try:
        free = int(shutil.disk_usage(probe).free)
    except OSError:
        free = _CACHE_RESERVE_BYTES
    budget = int(free * fraction)
    if leave_reserve and free > _CACHE_RESERVE_BYTES:
        return min(budget, free - _CACHE_RESERVE_BYTES)
    return budget


def _parse_cache_fraction(value: str | float) -> float | None:
    """Return a (0, 1] free-disk fraction, or ``None`` if ``value`` is a byte size."""
    if isinstance(value, float):
        if not 0 < value <= 1:
            raise ValueError(f"`max_cache_size` fraction must be in (0, 1], got {value}.")
        return value
    stripped = value.strip()
    if stripped.isdigit() or stripped.lower().endswith(tuple(_FORMAT_TO_RATIO)):
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if 0 < parsed <= 1:
        return parsed
    return None


def _parse_max_cache_size(value: int | float | str) -> int:
    """Parse an absolute cache size (bytes). Fractions must go through ``_resolve_max_cache_size``."""
    if isinstance(value, float):
        raise TypeError("fractional `max_cache_size` needs a cache path; use `_resolve_max_cache_size`.")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return _convert_bytes_to_int(stripped)
    return int(value)


def _is_absolute_cache_size(value: int | float | str) -> bool:
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return False
    return _parse_cache_fraction(value) is None


def _resolve_max_cache_size(max_cache_size: int | float | str | None, cache_path: str | None = None) -> int:
    """``MAX_CACHE_SIZE`` wins; else size (``100G``), fraction (``0.90``), or default 75% + 50GB reserve."""
    env = os.getenv("MAX_CACHE_SIZE")
    if env is not None and str(env).strip():
        max_cache_size = str(env).strip()
    if max_cache_size is None:
        resolved = _adaptive_max_cache_size(cache_path, fraction=_CACHE_FREE_FRACTION, leave_reserve=True)
        logger.info(
            "max_cache_size is unset; using %s (75%% of free disk, leaving ≥50GB when possible) on %s.",
            _human_readable_bytes(resolved),
            _probe_existing_dir(cache_path),
        )
        return resolved
    fraction = _parse_cache_fraction(max_cache_size) if not isinstance(max_cache_size, int) else None
    if fraction is not None:
        resolved = _adaptive_max_cache_size(cache_path, fraction=fraction, leave_reserve=False)
        logger.info(
            "max_cache_size=%s of free disk; using %s on %s.",
            fraction,
            _human_readable_bytes(resolved),
            _probe_existing_dir(cache_path),
        )
        return resolved
    return _parse_max_cache_size(max_cache_size)

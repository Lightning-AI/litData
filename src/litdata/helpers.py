import functools
import warnings
from typing import Any, Optional

import requests
from packaging import version as packaging_version


class WarningCache(set):
    """Cache for warnings."""

    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Trigger warning message."""
        if message not in self:
            self.add(message)
            warnings.warn(message, stacklevel=stacklevel, **kwargs)


warning_cache = WarningCache()

__package_name__ = "litdata"


@functools.lru_cache(maxsize=1)
def _get_newer_version(curr_version: str) -> Optional[str]:
    """Check PyPI for newer versions of ``litdata``.

    Returning the newest version if different from the current or ``None`` otherwise.

    """
    if packaging_version.parse(curr_version).is_prerelease:
        return None
    try:
        response = requests.get(f"https://pypi.org/pypi/{__package_name__}/json", timeout=30)
        response_json = response.json()
        releases = response_json["releases"]
        if curr_version not in releases:
            # Always return None if not installed from PyPI (e.g. dev versions)
            return None
        latest_version = response_json["info"]["version"]
        parsed_version = packaging_version.parse(latest_version)
        is_invalid = response_json["info"]["yanked"] or parsed_version.is_devrelease or parsed_version.is_prerelease
        return None if curr_version == latest_version or is_invalid else latest_version
    except requests.exceptions.RequestException:
        return None


def _check_version_and_prompt_upgrade(curr_version: str) -> None:
    """Checks that the current version of ``litdata`` is the latest on PyPI.

    If not, warn the user to upgrade ``litdata``.

    """
    new_version = _get_newer_version(curr_version)
    if new_version:
        warning_cache.warn(
            f"A newer version of {__package_name__} is available ({new_version}). "
            f"Please consider upgrading with `pip install -U {__package_name__}`. "
            "Not all functionalities of the platform can be guaranteed to work with the current version.",
        )
    return

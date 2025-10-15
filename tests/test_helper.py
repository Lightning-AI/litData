import os
import warnings
from unittest.mock import patch

import pytest

from litdata.helpers import _check_version_and_prompt_upgrade, _get_newer_version


@pytest.fixture
def set_env_var(monkeypatch):
    """Fixture to set and reset the env var."""
    original_value = os.environ.get("LITDATA_DISABLE_VERSION_CHECK")
    yield
    if original_value is None:
        monkeypatch.delenv("LITDATA_DISABLE_VERSION_CHECK", raising=False)
    else:
        monkeypatch.setenv("LITDATA_DISABLE_VERSION_CHECK", original_value)


@patch("litdata.helpers._LITDATA_DISABLE_VERSION_CHECK", new=1)
@patch("litdata.helpers.requests.get")
def test_get_newer_version_disabled_no_request(mock_get, set_env_var):
    """Ensure _get_newer_version returns None without making a request when disabled."""
    _get_newer_version.cache_clear()
    result = _get_newer_version("0.2.50")
    assert result is None
    mock_get.assert_not_called()  # Verify no network call


@patch("litdata.helpers._LITDATA_DISABLE_VERSION_CHECK", new=0)
@patch("litdata.helpers.requests.get")
def test_get_newer_version_default_behavior_enabled(mock_get, set_env_var):
    """Test default behavior (env var unset): makes request and handles response."""
    # Clear the LRU cache to ensure fresh execution
    _get_newer_version.cache_clear()
    os.environ.pop("LITDATA_DISABLE_VERSION_CHECK", None)

    # Simulate a successful response with a newer version
    current_version = "0.2.50"
    newer_version = "0.2.58"  # Newer version from PyPI
    mock_response = {
        "releases": {current_version: [], newer_version: []},
        "info": {"version": newer_version, "yanked": False},
    }
    mock_get.return_value.json.return_value = mock_response
    result = _get_newer_version(current_version)
    assert result == newer_version
    mock_get.assert_called_once_with("https://pypi.org/pypi/litdata/json", timeout=30)


@patch("litdata.helpers._LITDATA_DISABLE_VERSION_CHECK", new=0)
@patch("litdata.helpers._get_newer_version")
def test_check_version_default_behavior_warning(mock_get_newer, set_env_var):
    """Test default behavior: calls _get_newer_version and warns if newer version exists."""
    _get_newer_version.cache_clear()
    os.environ.pop("LITDATA_DISABLE_VERSION_CHECK", None)

    mock_get_newer.return_value = "0.2.58"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_version_and_prompt_upgrade("0.2.50")
        assert len(w) == 1
        assert f"A newer version of litdata is available ({mock_get_newer.return_value})" in str(w[0].message)
    mock_get_newer.assert_called_once_with("0.2.50")

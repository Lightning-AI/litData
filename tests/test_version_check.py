import os
import warnings
from unittest.mock import patch

import pytest

from litdata.helpers import _check_version_and_prompt_upgrade, _get_newer_version, _is_version_check_disabled


@pytest.fixture
def set_env_var():
    """Fixture to set and reset the env var."""
    original_value = os.environ.get("LITDATA_DISABLE_VERSION_CHECK")
    yield
    if original_value is None:
        os.environ.pop("LITDATA_DISABLE_VERSION_CHECK", None)
    else:
        os.environ["LITDATA_DISABLE_VERSION_CHECK"] = original_value


def test_is_version_check_disabled_true(set_env_var):
    """Test _is_version_check_disabled returns True when env var is '1'."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"
    assert _is_version_check_disabled() is True


def test_is_version_check_disabled_false_empty(set_env_var):
    """Test _is_version_check_disabled returns False when env var is empty."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = ""
    assert _is_version_check_disabled() is False


def test_is_version_check_disabled_false_zero(set_env_var):
    """Test _is_version_check_disabled returns False when env var is '0'."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "0"
    assert _is_version_check_disabled() is False


def test_is_version_check_disabled_false_unset(set_env_var):
    """Test _is_version_check_disabled returns False when env var is unset."""
    assert "LITDATA_DISABLE_VERSION_CHECK" not in os.environ
    assert _is_version_check_disabled() is False


def test_is_version_check_disabled_case_sensitivity_upper(set_env_var):
    """Test _is_version_check_disabled is case-sensitive (uppercase '1' should not disable)."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"  # Exact match only
    assert _is_version_check_disabled() is True
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "ONE"  # Not "1"
    assert _is_version_check_disabled() is False


@patch("litdata.helpers.requests.get")
def test_get_newer_version_disabled_no_request(mock_get, set_env_var):
    """Ensure _get_newer_version returns None without making a request when disabled."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"
    result = _get_newer_version("0.2.50")
    assert result is None
    mock_get.assert_not_called()  # Verify no network call


def test_get_newer_version_prerelease_disabled(set_env_var):
    """Ensure existing logic for pre-releases returns None even when disabled (though disabled takes precedence)."""
    # When disabled, it returns None early, so pre-release check is skipped
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"
    result = _get_newer_version("0.2.50")  # Pre-release
    assert result is None


@patch("litdata.helpers.requests.get")
def test_get_newer_version_prerelease_enabled(mock_get, set_env_var):
    """Test existing logic for pre-releases when enabled."""
    result = _get_newer_version("0.2.50")  # Pre-release
    assert result is None
    mock_get.assert_not_called()  # Pre-release check happens before request


@patch("litdata.helpers.requests.get")
def test_get_newer_version_default_behavior_enabled(mock_get, set_env_var):
    """Test default behavior (env var unset): makes request and handles response."""
    # Clear the LRU cache to ensure fresh execution
    _get_newer_version.cache_clear()

    # Ensure environment variable is not set
    if "LITDATA_DISABLE_VERSION_CHECK" in os.environ:
        del os.environ["LITDATA_DISABLE_VERSION_CHECK"]

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


@patch("litdata.helpers._get_newer_version")
def test_check_version_disabled_no_warning(mock_get_newer, set_env_var):
    """Ensure no warning or call to _get_newer_version when disabled."""
    os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_version_and_prompt_upgrade("0.2.50")
        assert len(w) == 0  # No warnings
    mock_get_newer.assert_not_called()


@patch("litdata.helpers._get_newer_version")
def test_check_version_default_behavior_warning(mock_get_newer, set_env_var):
    """Test default behavior: calls _get_newer_version and warns if newer version exists."""
    mock_get_newer.return_value = "0.2.58"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_version_and_prompt_upgrade("0.2.50")
        assert len(w) == 1
        assert f"A newer version of litdata is available ({mock_get_newer.return_value})" in str(w[0].message)
    mock_get_newer.assert_called_once_with("0.2.50")


@patch("litdata.helpers._get_newer_version")
def test_check_version_no_newer_version(mock_get_newer, set_env_var):
    """Test no warning when no newer version."""
    mock_get_newer.return_value = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_version_and_prompt_upgrade("0.2.50")
        assert len(w) == 0
    mock_get_newer.assert_called_once_with("0.2.50")

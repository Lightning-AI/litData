import importlib
import sys
import warnings
from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("disable_version_check", ["1", "0", None])
def test_get_newer_version_respects_env_flag(monkeypatch, disable_version_check):
    """Verify that _get_newer_version respects LITDATA_DISABLE_VERSION_CHECK and skips requests when disabled."""
    monkeypatch.delenv("LITDATA_DISABLE_VERSION_CHECK", raising=False)

    if disable_version_check is not None:
        monkeypatch.setenv("LITDATA_DISABLE_VERSION_CHECK", disable_version_check)

    # Reload both modules so constants re-evaluate environment variables
    sys.modules.pop("litdata.constants", None)
    sys.modules.pop("litdata.helpers", None)
    importlib.import_module("litdata.helpers")
    from litdata import helpers

    # Mock requests.get
    mock_get = Mock()
    mock_get.return_value.json.return_value = {
        "releases": {"0.2.50": [], "2.51.0": []},
        "info": {"version": "2.51.0", "yanked": False},
    }

    monkeypatch.setattr("litdata.helpers.requests.get", mock_get)

    # Clear cached function results
    helpers._get_newer_version.cache_clear()

    result = helpers._get_newer_version("0.2.50")

    if disable_version_check == "1":
        assert result is None
        mock_get.assert_not_called()
    else:
        assert result == "2.51.0"
        mock_get.assert_called_once_with("https://pypi.org/pypi/litdata/json", timeout=30)


@patch("litdata.helpers._get_newer_version")
def test_check_version_default_behavior_warning(mock_get_newer, monkeypatch):
    """Test default behavior: calls _get_newer_version and warns if newer version exists."""
    mock_get_newer.return_value = "0.2.58"

    from litdata.helpers import _check_version_and_prompt_upgrade

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_version_and_prompt_upgrade("0.2.50")
        assert len(w) == 1
        assert f"A newer version of litdata is available ({mock_get_newer.return_value})" in str(w[0].message)
    mock_get_newer.assert_called_once_with("0.2.50")

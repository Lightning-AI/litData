import operator
import warnings
from unittest.mock import MagicMock, patch

import pytest

from litdata.imports import (
    LazyModule,
    ModuleAvailableCache,
    RequirementCache,
    compare_version,
    lazy_import,
    requires,
)


# ---------------------------------------------------------------------------
# compare_version
# ---------------------------------------------------------------------------

class TestCompareVersion:
    def test_existing_package_passing(self):
        assert compare_version("pytest", operator.ge, "0.1") is True

    def test_existing_package_failing(self):
        assert compare_version("pytest", operator.ge, "9999.0") is False

    def test_nonexistent_package(self):
        assert compare_version("totally_nonexistent_pkg_xyz", operator.ge, "0.0") is False

    def test_use_base_version(self):
        assert compare_version("pytest", operator.ge, "0.1", use_base_version=True) is True


# ---------------------------------------------------------------------------
# RequirementCache
# ---------------------------------------------------------------------------

class TestRequirementCache:
    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            RequirementCache()

    def test_known_package_met(self):
        rc = RequirementCache("pytest>=0.1")
        assert bool(rc) is True

    def test_known_package_not_met(self):
        rc = RequirementCache("pytest>=9999.0")
        assert bool(rc) is False

    def test_unknown_package(self):
        rc = RequirementCache("totally_nonexistent_pkg_xyz>=0.1")
        assert bool(rc) is False

    def test_module_only(self):
        rc = RequirementCache(module="os.path")
        assert bool(rc) is True

    def test_module_only_missing(self):
        rc = RequirementCache(module="totally_nonexistent_pkg_xyz.sub")
        assert bool(rc) is False

    def test_check_module_path(self):
        rc = RequirementCache(requirement="pytest", module="pytest")
        assert bool(rc) is True

    def test_str_met(self):
        rc = RequirementCache("pytest>=0.1")
        s = str(rc)
        assert "pytest" in s

    def test_repr(self):
        rc = RequirementCache("pytest>=0.1")
        assert repr(rc) == str(rc)


# ---------------------------------------------------------------------------
# ModuleAvailableCache
# ---------------------------------------------------------------------------

class TestModuleAvailableCache:
    def test_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ModuleAvailableCache("os")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_delegates_to_requirement_cache(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mac = ModuleAvailableCache("os")
        assert bool(mac) is True

    def test_unknown_module(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mac = ModuleAvailableCache("totally_nonexistent_pkg_xyz")
        assert bool(mac) is False


# ---------------------------------------------------------------------------
# LazyModule / lazy_import
# ---------------------------------------------------------------------------

class TestLazyModule:
    def test_does_not_import_immediately(self):
        lm = LazyModule("json")
        assert lm._module is None

    def test_getattr_triggers_import(self):
        lm = lazy_import("json")
        _ = lm.dumps
        assert lm._module is not None

    def test_dir_works_after_import(self):
        lm = lazy_import("json")
        entries = dir(lm)
        assert "dumps" in entries
        assert "loads" in entries

    def test_callback_called(self):
        cb = MagicMock()
        lm = lazy_import("json", callback=cb)
        _ = lm.dumps
        cb.assert_called_once()


# ---------------------------------------------------------------------------
# requires decorator
# ---------------------------------------------------------------------------

class TestRequires:
    def test_available_requirement_runs(self):
        @requires("pytest")
        def fn():
            return 42

        assert fn() == 42

    def test_missing_requirement_raises(self):
        @requires("totally_nonexistent_pkg_xyz", raise_exception=True)
        def fn():
            return 42

        with pytest.raises(ModuleNotFoundError, match="Required dependencies not available"):
            fn()

    def test_missing_requirement_warns(self):
        @requires("totally_nonexistent_pkg_xyz", raise_exception=False)
        def fn():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fn()
            assert result == 42
            assert any("Required dependencies not available" in str(x.message) for x in w)

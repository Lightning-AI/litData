import pytest

from litdata.requirements import (
    _RequirementWithComment,
    _parse_requirements,
    _yield_lines,
    load_requirements,
)


# ---------------------------------------------------------------------------
# _yield_lines
# ---------------------------------------------------------------------------

class TestYieldLines:
    def test_string_input(self):
        result = list(_yield_lines("foo\nbar\nbaz"))
        assert result == ["foo", "bar", "baz"]

    def test_list_input(self):
        result = list(_yield_lines(["foo", "bar"]))
        assert result == ["foo", "bar"]

    def test_skips_empty_and_comments(self):
        result = list(_yield_lines(["# comment", "", "  ", "pkg", "# another"]))
        assert result == ["pkg"]


# ---------------------------------------------------------------------------
# _RequirementWithComment
# ---------------------------------------------------------------------------

class TestRequirementWithComment:
    def test_init_with_comment_and_pip_argument(self):
        r = _RequirementWithComment("arrow>=1.0", comment="# note", pip_argument="--extra-index-url x")
        assert r.comment == "# note"
        assert r.pip_argument == "--extra-index-url x"

    def test_empty_pip_argument_raises(self):
        with pytest.raises(RuntimeError, match="wrong pip argument"):
            _RequirementWithComment("arrow>=1.0", pip_argument="")

    def test_strict_detection(self):
        r = _RequirementWithComment("arrow>=1.0", comment="# strict")
        assert r.strict is True

    def test_not_strict(self):
        r = _RequirementWithComment("arrow>=1.0", comment="# anything")
        assert r.strict is False

    def test_adjust_none(self):
        r = _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# anything")
        assert r.adjust("none") == "arrow<=1.2.2,>=1.2.0"

    def test_adjust_none_strict(self):
        result = _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# strict").adjust("none")
        assert result == "arrow<=1.2.2,>=1.2.0  # strict"

    def test_adjust_all(self):
        result = _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# my name").adjust("all")
        assert result == "arrow>=1.2.0"

    def test_adjust_all_strict_overrides(self):
        result = _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("all")
        assert result == "arrow<=1.2.2,>=1.2.0  # strict"

    def test_adjust_all_no_specifier(self):
        assert _RequirementWithComment("arrow").adjust("all") == "arrow"

    def test_adjust_major(self):
        result = _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# cool").adjust("major")
        assert result == "arrow<2.0,>=1.2.0"

    def test_adjust_major_strict_overrides(self):
        result = _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("major")
        assert result == "arrow<=1.2.2,>=1.2.0  # strict"

    def test_adjust_major_no_upper_bound(self):
        assert _RequirementWithComment("arrow>=1.2.0").adjust("major") == "arrow>=1.2.0"

    def test_adjust_major_no_specifier(self):
        assert _RequirementWithComment("arrow").adjust("major") == "arrow"

    def test_adjust_invalid_raises(self):
        with pytest.raises(ValueError, match="Unexpected unfreeze"):
            _RequirementWithComment("arrow>=1.0").adjust("invalid")


# ---------------------------------------------------------------------------
# _parse_requirements
# ---------------------------------------------------------------------------

class TestParseRequirements:
    def test_parses_list(self):
        lines = ["# ignored", "", "this # is an", "--piparg", "example", "foo # strict", "thing"]
        results = [r.adjust("none") for r in _parse_requirements(lines)]
        assert results == ["this", "example", "foo  # strict", "thing"]

    def test_parses_multiline_string(self):
        txt = "# ignored\n\nthis # is an\n--piparg\nexample\nfoo # strict\nthing"
        results = [r.adjust("none") for r in _parse_requirements(txt)]
        assert results == ["this", "example", "foo  # strict", "thing"]

    def test_preserves_comments_and_pip_arguments(self):
        lines = ["--extra-index-url http://x", "pkg>=1.0 # strict"]
        reqs = list(_parse_requirements(lines))
        assert len(reqs) == 1
        assert reqs[0].pip_argument == "--extra-index-url http://x"
        assert reqs[0].comment == " # strict"
        assert reqs[0].strict is True

    def test_skips_dash_r_lines(self):
        lines = ["-r other/file.txt", "arrow"]
        results = list(_parse_requirements(lines))
        assert len(results) == 1
        assert str(results[0]) == "arrow"

    def test_skips_url_lines(self):
        lines = ["pesq @ git+https://github.com/ludlows/python-pesq", "arrow"]
        results = list(_parse_requirements(lines))
        assert len(results) == 1
        assert str(results[0]) == "arrow"

    def test_line_continuation(self):
        lines = ["foo\\", ">=1.0"]
        results = list(_parse_requirements(lines))
        assert len(results) == 1
        assert "foo" in str(results[0])


# ---------------------------------------------------------------------------
# load_requirements
# ---------------------------------------------------------------------------

class TestLoadRequirements:
    def test_loads_from_temp_file(self, tmp_path):
        req_file = tmp_path / "base.txt"
        req_file.write_text("arrow>=1.2.0,<=1.2.2\npytest>=0.1\n")
        result = load_requirements(str(tmp_path), "base.txt", unfreeze="all")
        assert "arrow>=1.2.0" in result
        assert "pytest>=0.1" in result

    def test_invalid_unfreeze_raises(self, tmp_path):
        req_file = tmp_path / "base.txt"
        req_file.write_text("arrow>=1.0\n")
        with pytest.raises(ValueError, match="unsupported option"):
            load_requirements(str(tmp_path), "base.txt", unfreeze="invalid")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing file"):
            load_requirements(str(tmp_path), "nonexistent.txt")

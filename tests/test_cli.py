from typer.testing import CliRunner

from litdata.cli import app

runner = CliRunner()


def test_litdata_help_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LitData CLI" in result.output
    assert "cache" in result.output


def test_cache_path_command():
    result = runner.invoke(app, ["cache", "path"])
    assert result.exit_code == 0
    assert "Default cache directory" in result.output


def test_cache_clear_command(tmp_path, monkeypatch):
    result = runner.invoke(app, ["cache", "clear"])
    assert result.exit_code == 0
    assert "cleared" in result.output

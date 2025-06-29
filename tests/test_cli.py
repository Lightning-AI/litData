import io
import sys
from contextlib import redirect_stdout, suppress
from unittest.mock import patch

from litdata.__main__ import app  # Your main entrypoint (like in your typer example)


def run_cli(args_list):
    f = io.StringIO()

    # argparse calls sys.exit(), which raises SystemExit
    with patch.object(sys, "argv", ["litdata"] + args_list), redirect_stdout(f), suppress(SystemExit):
        app()
    return f.getvalue()


def test_litdata_help_command():
    output = run_cli(["--help"])
    assert "LitData CLI" in output
    assert "cache" in output


def test_cache_path_command():
    output = run_cli(["cache", "path"])
    assert "Default cache directory" in output


def test_cache_clear_command(tmp_path, monkeypatch):
    # if your CLI uses default cache paths like ~/.cache/litdata, monkeypatch it here
    monkeypatch.setenv("LITDATA_CACHE_DIR", str(tmp_path))  # if applicable
    output = run_cli(["cache", "clear"])
    assert "cleared" in output.lower()

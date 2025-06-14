import subprocess
import sys
from pathlib import Path

import pytest

CLI_SCRIPT = Path("src/litdata/cli.py")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "LitData CLI"),
        (["cache", "--help"], "cache-related operations"),
        (["cache", "path"], "Default cache directory:"),
        (["cache", "clear"], "cleared."),
    ],
)
def test_litdata_cli_commands(args, expected):
    result = subprocess.run(  # noqa: S603
        [sys.executable, CLI_SCRIPT] + args, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert expected in result.stdout

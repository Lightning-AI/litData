import io
import logging
from unittest import mock

import pytest

from litdata.debugger import configure_logging


@pytest.fixture
def log_stream():
    return io.StringIO()


def test_get_logger_level():
    from litdata.debugger import get_logger_level

    assert get_logger_level("DEBUG") == logging.DEBUG
    assert get_logger_level("INFO") == logging.INFO
    assert get_logger_level("WARNING") == logging.WARNING
    assert get_logger_level("ERROR") == logging.ERROR
    assert get_logger_level("CRITICAL") == logging.CRITICAL
    with pytest.raises(ValueError, match="Invalid log level"):
        get_logger_level("INVALID")


def test_configure_logging(log_stream):
    # Configure logging with test stream
    configure_logging(level=logging.DEBUG, stream=log_stream)

    # Get logger and log a test message
    logger = logging.getLogger("litdata")
    test_message = "Test debug message"
    logger.debug(test_message)

    # Verify log output
    log_contents = log_stream.getvalue()
    assert test_message in log_contents
    assert "DEBUG" in log_contents
    assert logger.propagate is False


def test_configure_logging_2():
    configure_logging(use_rich=False)
    assert logging.getLogger("litdata").handlers[0].__class__.__name__ == "StreamHandler"


def test_configure_logging_rich_not_installed():
    # patch builtins.__import__ to raise ImportError
    with mock.patch("builtins.__import__", side_effect=ImportError):
        configure_logging(use_rich=True)
        assert logging.getLogger("litdata").handlers[0].__class__.__name__ == "StreamHandler"

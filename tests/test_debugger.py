import logging
import sys

import pytest


def test_get_logger_level():
    from litdata.debugger import get_logger_level

    assert get_logger_level("DEBUG") == logging.DEBUG
    assert get_logger_level("INFO") == logging.INFO
    assert get_logger_level("WARNING") == logging.WARNING
    assert get_logger_level("ERROR") == logging.ERROR
    assert get_logger_level("CRITICAL") == logging.CRITICAL
    with pytest.raises(ValueError, match="Invalid log level"):
        get_logger_level("INVALID")


def test_get_log_msg_sanitizes_semicolons_and_newlines():
    from litdata.debugger import _get_log_msg

    msg = _get_log_msg({"name": "crash", "ph": "I", "error": "a;b\nc"})
    assert "\n" not in msg
    assert "\r" not in msg
    assert "name: crash;" in msg
    assert "ph: I;" in msg
    assert "error: a,b c;" in msg


def test_one_line_trace_formatter_drops_traceback():
    from litdata.debugger import _get_log_msg, _OneLineTraceFormatter

    formatter = _OneLineTraceFormatter("ts:%(asctime)s;PID:%(process)d; TID:%(thread)d; %(message)s")
    try:
        raise TypeError("Session.__init__() got an unexpected keyword argument 'data_connection_id'")
    except TypeError:
        record = logging.LogRecord(
            name="litdata",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=_get_log_msg({"name": "prepare_chunks_thread_crashed_TypeError", "ph": "I"}),
            args=(),
            exc_info=sys.exc_info(),
        )

    line = formatter.format(record)
    assert "\n" not in line
    assert "Traceback" not in line
    assert "name: prepare_chunks_thread_crashed_TypeError;" in line
    assert "ph: I;" in line
    assert line.startswith("ts:")
    ts_val = float(line.split(";", 1)[0].split(":", 1)[1])
    assert ts_val > 1e12, f"expected Chrome microseconds, got {ts_val}"


def test_trace_levels_and_categories():
    from litdata.debugger import (
        CAT_BATCH,
        CAT_DOWNLOAD,
        CAT_SAMPLE,
        _categories_for_level,
        _set_active_categories,
        active_categories,
        is_tracing,
    )

    prev = active_categories()
    try:
        _set_active_categories(_categories_for_level("batch"))
        assert is_tracing(CAT_BATCH)
        assert not is_tracing(CAT_DOWNLOAD)
        assert not is_tracing(CAT_SAMPLE)

        _set_active_categories(_categories_for_level("chunk"))
        assert is_tracing(CAT_DOWNLOAD)
        assert not is_tracing(CAT_SAMPLE)

        _set_active_categories(_categories_for_level("sample"))
        assert is_tracing(CAT_SAMPLE)

        _set_active_categories(frozenset({"download", "read", "delete"}))
        assert is_tracing(CAT_DOWNLOAD)
        assert not is_tracing(CAT_BATCH)

        _set_active_categories(_categories_for_level("off"))
        assert not is_tracing(CAT_DOWNLOAD)
        with pytest.raises(ValueError, match="Unknown tracer level"):
            _categories_for_level("nope")
    finally:
        _set_active_categories(prev)


def test_emit_trace_noop_when_disabled(caplog):
    from litdata.debugger import CAT_DOWNLOAD, _set_active_categories, active_categories, emit_trace

    prev = active_categories()
    _set_active_categories(frozenset())
    try:
        with caplog.at_level("DEBUG", logger="litdata"):
            emit_trace("download", "B", CAT_DOWNLOAD, chunk=1)
        assert not any("name: download;" in r.getMessage() for r in caplog.records)
    finally:
        _set_active_categories(prev)

from collections import namedtuple

from litdata.utilities.format import (
    _adaptive_max_cache_size,
    _convert_bytes_to_int,
    _human_readable_bytes,
    _resolve_max_cache_size,
)


def test_human_readable_bytes():
    assert _human_readable_bytes(0) == "0.0 B"
    assert _human_readable_bytes(1) == "1.0 B"
    assert _human_readable_bytes(999) == "999.0 B"
    assert _human_readable_bytes(int(1e3)) == "1.0 KB"
    assert _human_readable_bytes(int(1e3 + 1e2)) == "1.1 KB"
    assert _human_readable_bytes(int(1e6)) == "1.0 MB"
    assert _human_readable_bytes(int(1e6 + 2e5)) == "1.2 MB"
    assert _human_readable_bytes(int(1e9)) == "1.0 GB"
    assert _human_readable_bytes(int(1e9 + 3e8)) == "1.3 GB"
    assert _human_readable_bytes(int(1e12)) == "1.0 TB"
    assert _human_readable_bytes(int(1e12 + 4e11)) == "1.4 TB"
    assert _human_readable_bytes(int(1e15)) == "1.0 PB"
    assert _human_readable_bytes(int(1e15 + 5e14)) == "1.5 PB"
    assert _human_readable_bytes(int(1e18)) == "1000.0 PB"


def test_adaptive_max_cache_size_leaves_checkpoint_headroom(monkeypatch, tmpdir):
    usage = namedtuple("Usage", "total used free")
    gb = 1000**3
    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(10 * 1000 * gb, 0, 200 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 40 * gb  # 20% of 200GB, still ≥50GB free

    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(80 * gb, 0, 80 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 16 * gb  # 20% of 80GB

    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(40 * gb, 0, 40 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 4 * gb  # 10% when free ≤50GB


def test_resolve_max_cache_size_user_and_env_win(monkeypatch, tmpdir):
    monkeypatch.delenv("MAX_CACHE_SIZE", raising=False)
    assert _resolve_max_cache_size("10GB", str(tmpdir)) == _convert_bytes_to_int("10GB")
    monkeypatch.setenv("MAX_CACHE_SIZE", "2GB")
    assert _resolve_max_cache_size("10GB", str(tmpdir)) == _convert_bytes_to_int("2GB")

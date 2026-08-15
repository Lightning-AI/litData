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


def test_convert_bytes_short_suffix():
    assert _convert_bytes_to_int("100G") == _convert_bytes_to_int("100GB")
    assert _convert_bytes_to_int("100g") == _convert_bytes_to_int("100GB")


def test_adaptive_max_cache_size_leaves_checkpoint_headroom(monkeypatch, tmpdir):
    usage = namedtuple("Usage", "total used free")
    gb = 1000**3
    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(10 * 1000 * gb, 0, 200 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 150 * gb  # 75% of 200GB, still ≥50GB free

    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(80 * gb, 0, 80 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 30 * gb  # capped so ≥50GB remains

    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(40 * gb, 0, 40 * gb))
    assert _adaptive_max_cache_size(str(tmpdir)) == 30 * gb  # 75% when free ≤50GB


def test_resolve_max_cache_size_fraction_of_free(monkeypatch, tmpdir):
    usage = namedtuple("Usage", "total used free")
    gb = 1000**3
    monkeypatch.setattr("litdata.utilities.format.shutil.disk_usage", lambda _p: usage(10 * 1000 * gb, 0, 200 * gb))
    monkeypatch.delenv("MAX_CACHE_SIZE", raising=False)
    assert _resolve_max_cache_size(0.90, str(tmpdir)) == 180 * gb
    assert _resolve_max_cache_size("0.90", str(tmpdir)) == 180 * gb
    monkeypatch.setenv("MAX_CACHE_SIZE", "0.5")
    assert _resolve_max_cache_size("10G", str(tmpdir)) == 100 * gb  # env fraction wins


def test_resolve_max_cache_size_user_and_env_win(monkeypatch, tmpdir):
    monkeypatch.delenv("MAX_CACHE_SIZE", raising=False)
    assert _resolve_max_cache_size("10GB", str(tmpdir)) == _convert_bytes_to_int("10GB")
    assert _resolve_max_cache_size("100G", str(tmpdir)) == _convert_bytes_to_int("100GB")
    monkeypatch.setenv("MAX_CACHE_SIZE", "2GB")
    assert _resolve_max_cache_size("10GB", str(tmpdir)) == _convert_bytes_to_int("2GB")

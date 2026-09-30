import errno
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from litdata.utilities import direct_io

LINUX = sys.platform == "linux" and platform.machine() in ("x86_64", "aarch64")


def test_unsupported_platform(monkeypatch):
    monkeypatch.setattr(direct_io.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        direct_io.open_nfs_direct("unused")
    with pytest.raises(RuntimeError, match="Linux"), direct_io.nfs_direct_io_environment("unused", suffixes=[".bin"]):
        pass


@pytest.mark.skipif(not LINUX, reason="Linux descriptor flags")
def test_opener_rejects_local_files_without_leaking_fd(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"payload")
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError, match="NFS filesystem"):
        direct_io.open_nfs_direct(path)
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.skipif(not LINUX, reason="Linux descriptor flags")
def test_opener_flags_and_owned_handle(tmp_path, monkeypatch):
    import fcntl

    path = tmp_path / "data.bin"
    path.write_bytes(b"0123456789")
    monkeypatch.setattr(direct_io, "_check_nfs", lambda fd: None)
    original = fcntl.fcntl
    changes = []

    def set_flags(fd, op, *args):
        if op == fcntl.F_SETFL:
            changes.append(args[0])
            return 0
        return original(fd, op, *args)

    monkeypatch.setattr(fcntl, "fcntl", set_flags)
    with direct_io.open_nfs_direct(path) as handle:
        fd = handle.fileno()
        handle.seek(3)
        assert handle.read(5) == b"34567"
        assert not os.get_inheritable(fd)
    assert changes[0] & os.O_DIRECT
    assert not changes[0] & os.O_NONBLOCK
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(fd)


@pytest.mark.skipif(not LINUX, reason="Linux descriptor flags")
def test_opener_failure_closes_descriptor(tmp_path, monkeypatch):
    import fcntl

    path = tmp_path / "data.bin"
    path.write_bytes(b"payload")
    monkeypatch.setattr(direct_io, "_check_nfs", lambda fd: None)
    seen = []

    def fail(fd, op, *args):
        seen.append(fd)
        raise OSError(errno.EINVAL, "flag failure")

    monkeypatch.setattr(fcntl, "fcntl", fail)
    with pytest.raises(OSError, match="flag failure"):
        direct_io.open_nfs_direct(path)
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(seen[0])


@pytest.mark.skipif(not LINUX, reason="Linux descriptor flags")
def test_opener_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "pipe.bin"
    os.mkfifo(path)
    with pytest.raises(OSError, match="regular file"):
        direct_io.open_nfs_direct(path)


@pytest.mark.parametrize("suffixes", [[], ".bin", ["bin"], [".a:b"], [".a/b"], [".a\0b"]])
def test_launcher_rejects_invalid_suffixes(tmp_path, monkeypatch, suffixes):
    monkeypatch.setattr(direct_io, "_check_platform", lambda: None)
    with pytest.raises(ValueError, match="suffix"), direct_io.nfs_direct_io_environment(tmp_path, suffixes=suffixes):
        pass


@pytest.mark.skipif(not LINUX, reason="Linux launcher preflight")
def test_launcher_cleans_build_and_preserves_parent_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_io, "_check_platform", lambda: None)
    monkeypatch.setattr(direct_io, "_check_nfs", lambda fd: None)
    monkeypatch.setattr(direct_io.shutil, "which", lambda compiler: "/usr/bin/cc")
    monkeypatch.setattr(direct_io.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.delenv("LITDATA_DIRECT_IO_ROOT", raising=False)
    monkeypatch.setenv("LD_PRELOAD", "/existing.so")
    # macOS lacks O_DIRECTORY's Linux semantics, but opening a directory is supported.
    with direct_io.nfs_direct_io_environment(tmp_path, suffixes=[".bin", ".lance"]) as env:
        assert env["LITDATA_DIRECT_IO_ROOT"] == str(tmp_path.resolve())
        assert env["LITDATA_DIRECT_IO_SUFFIXES"] == ".bin:.lance"
        build = Path(env["LD_PRELOAD"].split()[0]).parent
        assert build.exists()
        assert env["LD_PRELOAD"].endswith(" /existing.so")
        assert "LITDATA_DIRECT_IO_ROOT" not in os.environ
        assert os.environ["LD_PRELOAD"] == "/existing.so"
    assert not build.exists()


@pytest.fixture
def native_env(tmp_path, monkeypatch):
    if not LINUX or not shutil.which("cc"):
        pytest.skip("Linux and a C compiler are required")
    # Only the Python launch preflight is bypassed. The compiled helper still checks the real filesystem.
    monkeypatch.setattr(direct_io, "_check_nfs", lambda fd: None)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("LITDATA_DIRECT_IO_ROOT", raising=False)
    root = tmp_path / "selected"
    root.mkdir()
    (root / "data.bin").write_bytes(b"data")
    (root / "index.json").write_text("{}")
    outside = tmp_path / "selected-other"
    outside.mkdir()
    (outside / "data.bin").write_bytes(b"outside")
    (root / "escape.bin").symlink_to(outside / "data.bin")
    (outside / "enter.bin").symlink_to(root / "data.bin")
    with direct_io.nfs_direct_io_environment(root, suffixes=[".bin"]) as env:
        yield root, outside, env


def test_native_scope_relative_paths_symlinks_and_writes(native_env):
    root, outside, env = native_env
    script = r"""
import errno, json, os, sys
root, outside = sys.argv[1:]
def check(path, flags=os.O_RDONLY, dir_fd=None):
    try:
        fd=os.open(path, flags, 0o600, dir_fd=dir_fd)
    except OSError as ex:
        return ex.errno
    os.close(fd)
    return 0
directory=os.open(root, os.O_RDONLY|os.O_DIRECTORY)
result=[check(root+'/data.bin'), check('data.bin',dir_fd=directory),
 check(root+'/index.json'), check(outside+'/data.bin'), check(root+'/escape.bin'),
 check(outside+'/enter.bin'), check(root+'/data.bin',os.O_WRONLY),
 check(root+'/new.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL), check(root+'/missing.bin'),
 check(root+'/../selected-other/data.bin')]
os.close(directory)
print(json.dumps(result))
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(root), str(outside)], env=env, check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == [
        errno.EOPNOTSUPP,
        errno.EOPNOTSUPP,
        0,
        0,
        0,
        errno.EOPNOTSUPP,
        0,
        0,
        errno.ENOENT,
        0,
    ]


def test_nfs_unaligned_reads_and_native_launcher():
    mount = os.environ.get("LITDATA_TEST_NFS_DIR")
    if not mount:
        pytest.skip("Set LITDATA_TEST_NFS_DIR for a real NFS integration test")
    import fcntl
    import tempfile

    with tempfile.TemporaryDirectory(prefix="litdata-test-", dir=mount) as directory:
        path = Path(directory) / "data.bin"
        payload = bytes(range(256)) * 64
        path.write_bytes(payload)
        with direct_io.open_nfs_direct(path) as handle:
            assert fcntl.fcntl(handle.fileno(), fcntl.F_GETFL) & os.O_DIRECT
            handle.seek(7)
            assert handle.read(123) == payload[7:130]
            handle.seek(len(payload) - 3)
            assert handle.read(100) == payload[-3:]
        script = """
import fcntl, os, sys
with open(sys.argv[1], 'rb', buffering=0) as f:
    assert fcntl.fcntl(f.fileno(), fcntl.F_GETFL) & os.O_DIRECT
    f.seek(7)
    assert f.read(123) == (bytes(range(256))*64)[7:130]
"""
        with direct_io.nfs_direct_io_environment(directory, suffixes=[".bin"]) as env:
            subprocess.run([sys.executable, "-c", script, str(path)], env=env, check=True)  # noqa: S603


@pytest.mark.skipif(not LINUX, reason="POSIX process groups")
def test_command_propagates_exit_code():
    assert direct_io._run_command([sys.executable, "-c", "raise SystemExit(7)"], os.environ.copy()) == 7


@pytest.mark.skipif(not LINUX, reason="POSIX process groups")
def test_command_forwards_termination(tmp_path):
    module = str(Path(direct_io.__file__))
    ready = str(tmp_path / "handlers-ready")
    script = r"""
import importlib.util, os, pathlib, signal, sys
spec=importlib.util.spec_from_file_location('direct_io',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ready=sys.argv[2];original=signal.signal
def install(sig, handler):
    old=original(sig,handler)
    if sig==signal.SIGTERM and callable(handler):pathlib.Path(ready).touch()
    return old
signal.signal=install
child=("import os,pathlib,signal,sys,time; p=pathlib.Path(sys.argv[1]);\n"
       "while not p.exists(): time.sleep(.005)\n"
       "os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)")
raise SystemExit(m._run_command([sys.executable,'-c',child,ready],os.environ.copy()))
"""
    result = subprocess.run([sys.executable, "-c", script, module, ready], timeout=10)  # noqa: S603
    assert result.returncode == 128 + 15

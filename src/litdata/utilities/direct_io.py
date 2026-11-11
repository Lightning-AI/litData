"""Opt-in direct reads on Linux NFS, including a launcher for native readers."""

import argparse
import ctypes
import errno
import io
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

_NFS_SUPER_MAGIC = 0x6969


class _StatFS(ctypes.Structure):
    # Linux x86_64/aarch64 use the same 64-bit statfs layout.
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong),
        ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong),
        ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    ]


def _check_platform() -> None:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64") or ctypes.sizeof(ctypes.c_long) != 8:
        raise RuntimeError("NFS direct I/O currently supports Linux x86_64 and aarch64 only.")


def _check_nfs(fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    fstatfs = libc.fstatfs
    fstatfs.argtypes = [ctypes.c_int, ctypes.POINTER(_StatFS)]
    fstatfs.restype = ctypes.c_int
    info = _StatFS()
    if fstatfs(fd, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    if info.f_type != _NFS_SUPER_MAGIC:
        raise OSError(errno.EOPNOTSUPP, "Direct reads require an NFS filesystem; buffered fallback is disabled.")


def open_nfs_direct(path: str | os.PathLike) -> io.FileIO:
    """Open a regular NFS file for unbuffered, read-only direct I/O.

    The returned handle supports arbitrary byte offsets on Linux NFS. It bypasses
    the client page cache; the server may still cache data. Open handles inside
    each worker, close them normally, and do not mmap the same payload. Other
    filesystems/platforms raise an error. No compiler or native extension is required.
    """
    _check_platform()
    import fcntl

    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Direct reads require a regular file.")
        _check_nfs(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, (flags & ~os.O_NONBLOCK) | os.O_DIRECT)
        return io.FileIO(fd, mode="rb", closefd=True)
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def nfs_direct_io_environment(
    root: str | os.PathLike, *, suffixes: Sequence[str], compiler: str = "cc"
) -> Iterator[dict[str, str]]:
    """Build an experimental preload helper and yield an environment for a new process.

    Only read-only regular files with a selected suffix underneath the resolved
    root get O_DIRECT. The helper checks the opened descriptor's filesystem and
    actual path, including relative openat calls and symlinks. It does not change
    the current process. Wait for the child and all its workers before exiting
    this context, which removes the compiled helper. A C compiler is required.

    Native readers must use dynamically linked libc open/openat calls. mmap,
    static binaries, raw syscalls and io_uring open operations are not supported.
    """
    _check_platform()
    directory = Path(root).resolve(strict=True)
    if not directory.is_dir() or directory == Path("/"):
        raise ValueError("root must be a dataset directory, not the filesystem root.")
    if isinstance(suffixes, str) or not suffixes:
        raise ValueError("Provide a nonempty sequence of payload suffixes, for example ['.lance'].")
    if any(not isinstance(s, str) or not s.startswith(".") or any(c in s for c in "/:\0") for s in suffixes):
        raise ValueError("Each suffix must start with '.' and contain no slash, colon or NUL.")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        _check_nfs(fd)
    finally:
        os.close(fd)
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError(f"C compiler {compiler!r} not found. Install a compiler to use the native-reader launcher.")
    source = Path(__file__).with_name("_nfs_direct_io.c")
    # A private build directory avoids executing a shared or stale cached library.
    with tempfile.TemporaryDirectory(prefix="litdata-direct-io-") as build:
        library = Path(build) / "nfs_direct_io.so"
        if any(c.isspace() or c == ":" for c in str(library)):
            raise ValueError("The temporary directory path must contain no whitespace or colon for LD_PRELOAD.")
        subprocess.run(  # noqa: S603
            [
                executable,
                "-shared",
                "-fPIC",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(library),
                "-ldl",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        env = os.environ.copy()
        if env.get("LITDATA_DIRECT_IO_ROOT"):
            raise RuntimeError("Nested direct-I/O launchers are not supported.")
        env["LITDATA_DIRECT_IO_ROOT"] = str(directory)
        env["LITDATA_DIRECT_IO_SUFFIXES"] = ":".join(suffixes)
        env["LD_PRELOAD"] = str(library) + (" " + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        yield env


def _run_command(command: Sequence[str], env: dict[str, str]) -> int:
    # A separate group lets the launcher forward termination to DataLoader descendants too.
    with subprocess.Popen(command, env=env, start_new_session=True) as child:  # noqa: S603

        def forward(signum: int, frame: object) -> None:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signum)

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return code if code >= 0 else 128 - code


def main() -> int:
    """Run a command with explicitly scoped NFS direct reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="NFS dataset directory")
    parser.add_argument("--suffix", action="append", required=True, help="Payload suffix; repeat for more types")
    parser.add_argument("--compiler", default="cc", help="C compiler executable")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- python train.py [arguments]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --.")
    try:
        with nfs_direct_io_environment(args.root, suffixes=args.suffix, compiler=args.compiler) as env:
            return _run_command(command, env)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as ex:
        detail = f"\n{ex.stderr}" if isinstance(ex, subprocess.CalledProcessError) and ex.stderr else ""
        parser.exit(1, f"NFS direct I/O: {ex}{detail}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

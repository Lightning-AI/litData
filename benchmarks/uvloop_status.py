"""Shared uvloop detection and LoopRunner backend logging for raw benchmarks."""

from __future__ import annotations

from collections.abc import Callable


def uvloop_package_status() -> str:
    """Return a short string describing uvloop install and preferred loop backend."""
    from litdata.raw.dataset import _loop_backend_name

    try:
        import uvloop
    except ImportError:
        return "not installed (stdlib asyncio fallback)"
    version = getattr(uvloop, "__version__", "?")
    backend = _loop_backend_name()
    return f"available (uvloop {version}; create→{backend})"


def log_loop_runner_backend(log_fn: Callable[[str], None], *, prefix: str = "") -> bool:
    """Log whether the process-local LoopRunner loop is uvloop-backed."""
    from litdata.raw.dataset import _get_loop_runner, _loop_backend_name

    runner = _get_loop_runner()
    loop = runner.loop
    loop_type = f"{type(loop).__module__}.{type(loop).__name__}"
    active = type(loop).__module__.startswith("uvloop")
    tag = f"{prefix} " if prefix else ""
    log_fn(
        f"{tag}LoopRunner event loop: {loop_type} "
        f"(preferred={_loop_backend_name()}, "
        f"{'uvloop active' if active else 'stdlib asyncio'}) pid={runner.pid}"
    )
    return active

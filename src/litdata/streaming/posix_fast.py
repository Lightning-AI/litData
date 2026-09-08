# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-place POSIX reads for local / parallel filesystems (Vast, NFS, Lustre, GPFS).

StreamingDataset already packs samples into chunks. On object storage those chunks are
downloaded into a local cache. On a POSIX path the copy is wasted: FFCV-style reads mmap
the chunk in place and ``posix_fadvise`` the next files so the page cache fills ahead of
the reader. Shared-chunk mmap is safe because source objects are never deleted.

This is automatic for any local ``input_dir`` (no ``s3://`` URL). Users do not pass a flag.

When ``shuffle=True``, chunk **and** in-chunk item order use a sliding-window permute
so the loader can copy a contiguous byte span (a page) and split samples from it.
See ``WindowShuffle``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("litdata.streaming.posix_fast")

_OBJECT_PREFIXES = ("s3://", "gs://", "r2://", "hf://", "azure://", "local:")
_PARALLEL_FS = frozenset({"nfs", "nfs4", "nfs3", "lustre", "gpfs", "panfs", "beegfs", "fuse.vast"})
_VAST_MARKERS = ("vast", "vastdata")


def _is_object_url(value: str | None) -> bool:
    return value is not None and value.startswith(_OBJECT_PREFIXES)


_WINDOW_SHUFFLE_KINDS = frozenset({"vast", "nfs", "lustre", "gpfs", "panfs", "beegfs", "forced"})


@dataclass(frozen=True)
class PosixFastProfile:
    """How StreamingDataset should read a local/POSIX dataset."""

    kind: str  # posix | vast | nfs | lustre | gpfs | forced
    in_place: bool = True
    mmap_shared: bool = True
    skip_cache_copy: bool = True
    skip_chunk_delete: bool = True

    @property
    def window_shuffle(self) -> bool:
        """Sliding-window shuffle on parallel FS; local CI disks keep FullShuffle."""
        return self.kind in _WINDOW_SHUFFLE_KINDS


def _env_override() -> bool | None:
    raw = os.getenv("LITDATA_POSIX_FAST")
    if raw is None:
        return None
    return raw.strip() not in {"0", "false", "False", ""}


def _path_looks_vast(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in _VAST_MARKERS)


def parse_proc_mounts(text: str) -> list[tuple[str, str, str]]:
    """Return ``(mountpoint, fstype, source)`` rows from ``/proc/mounts`` text."""
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        source, mountpoint, fstype = parts[0], parts[1], parts[2]
        mountpoint = mountpoint.replace("\\040", " ")
        rows.append((mountpoint, fstype.lower(), source.lower()))
    return rows


def _path_match_forms(path: str) -> list[str]:
    """Unix and Windows spellings so injected ``/proc/mounts`` text matches in CI."""
    unix = path.replace("\\", "/")
    forms = [unix]
    abs_p = os.path.abspath(path).replace("\\", "/")
    if abs_p not in forms:
        forms.append(abs_p)
    for form in list(forms):
        if len(form) >= 2 and form[1] == ":":
            rest = form[2:] if form[2:].startswith("/") else "/" + form[2:]
            if rest not in forms:
                forms.append(rest)
    return forms


def _mount_for_path(path: str, mounts: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    forms = _path_match_forms(path)
    best: tuple[str, str, str] | None = None
    for mountpoint, fstype, source in mounts:
        mp = mountpoint.replace("\\", "/").rstrip("/") or "/"
        if any(form == mp or form.startswith(mp + "/") for form in forms) and (best is None or len(mp) > len(best[0])):
            best = (mountpoint, fstype, source)
    return best


def _profile_from_mount(fstype: str, source: str) -> PosixFastProfile | None:
    blob = f"{fstype} {source}"
    if any(marker in blob for marker in _VAST_MARKERS) or fstype in {"fuse.vast"}:
        return PosixFastProfile(kind="vast")
    if fstype in _PARALLEL_FS:
        kind = "nfs" if fstype.startswith("nfs") else fstype
        return PosixFastProfile(kind=kind)
    return None


def detect_posix_fast(
    path: str | None,
    storage_options: dict[str, Any] | None = None,
    *,
    remote_url: str | None = None,
    mounts_text: str | None = None,
) -> PosixFastProfile | None:
    """Return a POSIX-fast profile when chunks should be mmapped in place.

    Automatic for every local directory. Object URLs stay on the GET path.
    ``LITDATA_POSIX_FAST=0`` disables; ``=1`` forces on a local path.
    """
    del storage_options  # detection is path/URL based; not a user-facing switch
    forced = _env_override()
    if forced is False:
        return None

    if _is_object_url(path) or _is_object_url(remote_url):
        return None

    if not path:
        return None

    if forced is True:
        return PosixFastProfile(kind="forced")

    kind = "posix"
    if _path_looks_vast(path):
        kind = "vast"
    else:
        if mounts_text is None:
            try:
                with open("/proc/mounts", encoding="utf-8") as fh:
                    mounts_text = fh.read()
            except OSError:
                mounts_text = ""
        if mounts_text:
            mount = _mount_for_path(path, parse_proc_mounts(mounts_text))
            if mount is not None:
                from_fs = _profile_from_mount(mount[1], mount[2])
                if from_fs is not None:
                    kind = from_fs.kind

    return PosixFastProfile(kind=kind)


_DEFAULT_PAGE_BYTES = 256 * 1024
_DEFAULT_RAM_FRACTION = 0.5
_DEFAULT_RAM_CEILING = 0.95  # used fraction of MemTotal; leave ~5% for SSH/OS
_DEFAULT_RAM_SOFT_MARGIN = 0.20  # start slowing at 75% used RAM
_DEFAULT_RAM_THROTTLE_MAX_S = 0.50  # sleep per batch at the ceiling
_DEFAULT_RAM_WAIT_TIMEOUT = 120.0  # seconds; fail closed rather than wait forever
_DEFAULT_WORKER_RSS = 256 * 1024 * 1024  # process + one collated JPEG batch, not four WILLNEED chunks
_logged_willneed_skip = False


def _read_memory_counter(path: str, *, allow_zero: bool = False) -> int | None:
    """Read a cgroup memory counter, treating ``max`` and huge sentinels as unlimited."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        value = int(raw)
    except (OSError, ValueError):
        return None
    # cgroup v1 commonly represents "no limit" with a near-int64 maximum.
    return value if (value >= 0 if allow_zero else value > 0) and value < (1 << 60) else None


def cgroup_memory_limit_bytes(cgroup_root: str | None = None) -> int | None:
    """Return the finite cgroup memory limit for this process, if present.

    Container runtimes normally mount the process's cgroup at ``/sys/fs/cgroup``.
    The optional root exists for deterministic tests.
    """
    root = cgroup_root or "/sys/fs/cgroup"
    return _read_memory_counter(os.path.join(root, "memory.max")) or _read_memory_counter(
        os.path.join(root, "memory", "memory.limit_in_bytes")
    )


def cgroup_memory_current_bytes(cgroup_root: str | None = None) -> int | None:
    """Return current cgroup memory usage for this process, if available."""
    root = cgroup_root or "/sys/fs/cgroup"
    current = _read_memory_counter(os.path.join(root, "memory.current"), allow_zero=True)
    return (
        current
        if current is not None
        else _read_memory_counter(os.path.join(root, "memory", "memory.usage_in_bytes"), allow_zero=True)
    )


def _read_meminfo(meminfo_text: str | None = None) -> dict[str, int]:
    """Parse ``/proc/meminfo`` keys (bytes). Empty dict if unknown."""
    text = meminfo_text
    if text is None:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        with contextlib.suppress(ValueError):
            out[key] = int(parts[1]) * 1024
    return out


def available_ram_bytes(meminfo_text: str | None = None, *, cgroup_root: str | None = None) -> int | None:
    """Available bytes, bounded by the process's cgroup memory headroom when finite."""
    info = _read_meminfo(meminfo_text)
    if "MemAvailable" in info:
        host_available: int | None = info["MemAvailable"]
    else:
        host_available = info.get("MemFree", 0) + info.get("Cached", 0) or None
    limit = cgroup_memory_limit_bytes(cgroup_root)
    current = cgroup_memory_current_bytes(cgroup_root)
    cgroup_available = max(0, limit - current) if limit is not None and current is not None else None
    if host_available is not None and cgroup_available is not None:
        return min(host_available, cgroup_available)
    return host_available if host_available is not None else cgroup_available


def mem_total_bytes(meminfo_text: str | None = None, *, cgroup_root: str | None = None) -> int | None:
    """Total addressable bytes, bounded by the process's finite cgroup limit."""
    info = _read_meminfo(meminfo_text)
    host_total = info.get("MemTotal") or None
    cgroup_limit = cgroup_memory_limit_bytes(cgroup_root)
    if host_total is not None and cgroup_limit is not None:
        return min(host_total, cgroup_limit)
    return host_total or cgroup_limit


def posix_ram_fraction() -> float:
    raw = os.getenv("LITDATA_POSIX_RAM_FRACTION")
    if raw is None or not raw.strip():
        return _DEFAULT_RAM_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RAM_FRACTION
    return min(1.0, max(0.0, value))


def ram_ceiling_fraction() -> float:
    """Max used fraction of ``MemTotal`` LitData should leave the host at.

    Override with ``LITDATA_RAM_CEILING`` (default 0.95). Clamped to ``[0.50, 0.99]``.
    """
    raw = os.getenv("LITDATA_RAM_CEILING")
    if raw is None or not raw.strip():
        return _DEFAULT_RAM_CEILING
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RAM_CEILING
    return min(0.99, max(0.50, value))


def ram_budget_bytes(
    *,
    ram_bytes: int | None = None,
    mem_total: int | None = None,
) -> int | None:
    """Bytes LitData may still fill while staying under ``LITDATA_RAM_CEILING``.

    Keeps ``(1 - ceiling) * MemTotal`` as ``MemAvailable`` headroom for sshd / the
    OS. When ``MemTotal`` is unknown (tests passing only ``ram_bytes``), falls
    back to ``LITDATA_POSIX_RAM_FRACTION`` of available.
    """
    available = ram_bytes if ram_bytes is not None else available_ram_bytes()
    if available is None:
        return None
    total = mem_total if mem_total is not None else mem_total_bytes()
    if total is None:
        return max(1, int(available * posix_ram_fraction()))
    reserve = max(0, total - int(total * ram_ceiling_fraction()))
    return max(0, available - reserve)


def ram_in_flight_budget_bytes(*, mem_total: int | None = None) -> int | None:
    """Max decoded-batch RAM that can sit in the DataLoader queue without crossing the ceiling.

    The SSH reserve: ``workers × prefetch × batch`` can be outstanding when the
    scheduler starts draining, so it must fit in the headroom below the ceiling.
    """
    total = mem_total if mem_total is not None else mem_total_bytes()
    if total is None:
        return None
    reserve = max(0, total - int(total * ram_ceiling_fraction()))
    return max(1, reserve)


def ram_soft_margin() -> float:
    """How far below ``LITDATA_RAM_CEILING`` we start throttling (default 0.20)."""
    raw = os.getenv("LITDATA_RAM_SOFT_MARGIN")
    if raw is None or not raw.strip():
        return _DEFAULT_RAM_SOFT_MARGIN
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RAM_SOFT_MARGIN
    return min(0.30, max(0.02, value))


def ram_used_fraction(
    *,
    ram_bytes: int | None = None,
    mem_total: int | None = None,
) -> float | None:
    """Used RAM as a fraction of ``MemTotal`` (``1 - MemAvailable/MemTotal``)."""
    available = ram_bytes if ram_bytes is not None else available_ram_bytes()
    total = mem_total if mem_total is not None else mem_total_bytes()
    if available is None or total is None or total <= 0:
        return None
    return min(1.0, max(0.0, 1.0 - (available / total)))


def ram_pressure_fraction(
    *,
    ram_bytes: int | None = None,
    mem_total: int | None = None,
) -> float:
    """0 below the soft limit, 1 at ``LITDATA_RAM_CEILING``, linear in between.

    With the defaults (ceiling 0.95, margin 0.20) throttling starts at 75% used.
    """
    used = ram_used_fraction(ram_bytes=ram_bytes, mem_total=mem_total)
    if used is None:
        return 0.0
    ceiling = ram_ceiling_fraction()
    soft = max(0.50, ceiling - ram_soft_margin())
    if used <= soft:
        return 0.0
    if used >= ceiling:
        return 1.0
    return (used - soft) / (ceiling - soft)


def ram_throttle_sleep_s(pressure: float) -> float:
    """Backoff sleep for one batch. Quadratic so we ease in, then bite near the ceiling."""
    p = min(1.0, max(0.0, pressure))
    return _DEFAULT_RAM_THROTTLE_MAX_S * p * p


def posix_prefetch_fits_ram(
    *,
    keep: int,
    chunk_bytes: int,
    num_readers: int,
    ram_bytes: int | None = None,
    ram_fraction: float | None = None,
) -> bool:
    """Whether ``WILLNEED`` of ``keep`` chunks per reader fits in a fraction of RAM.

    H100 boxes often set ``num_workers`` to all CPUs. Prefaulting
    ``workers × ranks × keep × 64MiB`` into the page cache thrashes when that
    window is close to ``MemAvailable``.
    """
    force = os.getenv("LITDATA_POSIX_WILLNEED")
    if force is not None:
        return force.strip() not in {"0", "false", "False", ""}
    ram = ram_bytes if ram_bytes is not None else available_ram_bytes()
    projected = max(1, keep) * max(1, chunk_bytes) * max(1, num_readers)
    if ram is None:
        return projected < 8 * 1024 * 1024 * 1024
    frac = posix_ram_fraction() if ram_fraction is None else ram_fraction
    return projected <= int(ram * frac)


def posix_safe_keep(
    *,
    keep: int,
    chunk_bytes: int,
    num_readers: int,
    ram_bytes: int | None = None,
) -> int:
    """Shrink mapped-chunk LRU so ``readers × keep × chunk`` fits the RAM budget."""
    keep = max(1, keep)
    if posix_prefetch_fits_ram(keep=keep, chunk_bytes=chunk_bytes, num_readers=num_readers, ram_bytes=ram_bytes):
        return keep
    ram = ram_bytes if ram_bytes is not None else available_ram_bytes()
    if ram is None:
        return 1
    budget = max(1, int(ram * posix_ram_fraction()))
    per = max(1, num_readers) * max(1, chunk_bytes)
    return max(1, min(keep, budget // per))


def ram_prefetch_keep(
    *,
    keep: int,
    chunk_bytes: int,
    num_readers: int,
    ram_bytes: int | None = None,
    min_keep: int = 2,
) -> int:
    """Cap per-worker chunk prefetch so ``readers × keep × chunk`` fits RAM.

    Unlike :func:`posix_safe_keep`, this ignores ``LITDATA_POSIX_WILLNEED`` (that
    flag only controls page-cache hints) and never returns below ``min_keep``
    (``max_pre_download == 1`` deadlocks delete-when-processed).
    """
    keep = max(min_keep, keep)
    budget = ram_budget_bytes(ram_bytes=ram_bytes)
    if budget is None:
        return keep
    per = max(1, num_readers) * max(1, chunk_bytes)
    if budget <= 0:
        return min_keep
    keep = max(min_keep, min(keep, budget // per))
    if ram_bytes is None:
        pressure = ram_pressure_fraction()
        if pressure > 0:
            keep = max(min_keep, int(round(keep * (1.0 - pressure) + min_keep * pressure)))
    return keep


def mean_sample_bytes(config: Any) -> int:
    """Mean payload bytes per sample from ``index.json`` chunks, else mean chunk size."""
    chunks = getattr(config, "_chunks", None) or []
    total_b = 0
    total_n = 0
    for chunk in chunks:
        size = chunk.get("chunk_bytes")
        n_items = chunk.get("chunk_size")
        if size and n_items:
            total_b += int(size)
            total_n += int(n_items)
    if total_n:
        return max(1, total_b // total_n)
    return mean_chunk_bytes(config)


def ram_prefetch_factor(
    requested: int,
    *,
    num_workers: int,
    batch_bytes: int,
    ram_bytes: int | None = None,
    min_prefetch: int = 1,
) -> int:
    """Cap DataLoader ``prefetch_factor`` so ``workers × prefetch × batch`` fits the RAM budget.

    ``LITDATA_POSIX_MAX_WORKERS=0`` disables this cap (same hatch as worker capping).
    Torch forbids ``prefetch_factor`` below 1 when ``num_workers > 0``.
    """
    requested = max(min_prefetch, requested)
    if num_workers <= 0:
        return requested
    raw = os.getenv("LITDATA_POSIX_MAX_WORKERS")
    if raw is not None and raw.strip() == "0":
        return requested
    budget = ram_budget_bytes(ram_bytes=ram_bytes)
    if ram_bytes is None:
        in_flight = ram_in_flight_budget_bytes()
        if in_flight is not None and budget is not None:
            budget = min(budget, in_flight)
        elif in_flight is not None:
            budget = in_flight
    if budget is None:
        return requested
    if budget <= 0:
        return min_prefetch
    per = max(1, num_workers) * max(1, batch_bytes)
    return max(min_prefetch, min(requested, budget // per))


_logged_ram_wait = False


def ram_wait_timeout_s() -> float:
    """Max seconds to block once used RAM is at the ceiling. ``0`` disables.

    Default is 120 seconds. When it expires, raise instead of allocating more
    memory beyond the ceiling. Override with ``LITDATA_RAM_WAIT_TIMEOUT``.
    Below the ceiling we only sleep a few milliseconds (see
    :func:`ram_throttle_sleep_s`), not this timeout.
    """
    raw = os.getenv("LITDATA_RAM_WAIT_TIMEOUT")
    if raw is None or not raw.strip():
        return _DEFAULT_RAM_WAIT_TIMEOUT
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_RAM_WAIT_TIMEOUT


def wait_for_ram_budget(
    *,
    need_bytes: int = 1,
    timeout_s: float | None = None,
    poll_s: float = 0.05,
) -> int | None:
    """Slow down as used RAM approaches ``LITDATA_RAM_CEILING``, then hold the band.

    * Pressure 0 (below ceiling − margin): no wait.
    * Pressure in (0, 1): short quadratic sleep so we do not slam into the limit.
    * Pressure 1 (at/over ceiling): wait until pressure is back to 0.5, then
      resume. If this does not happen within the timeout, fail closed rather
      than allocating beyond the target.
    """
    global _logged_ram_wait
    need = max(1, need_bytes)
    limit = ram_wait_timeout_s() if timeout_s is None else timeout_s
    budget = ram_budget_bytes()
    if budget is None or limit <= 0:
        return budget
    pressure = ram_pressure_fraction()
    if pressure <= 0 and budget >= need:
        return budget
    if pressure < 1.0 and budget >= need:
        time.sleep(ram_throttle_sleep_s(pressure))
        return ram_budget_bytes()
    resume_pressure = 0.5
    if not _logged_ram_wait:
        logger.warning(
            "RAM ceiling: holding new DataLoader work until used RAM drops "
            "(LITDATA_RAM_CEILING=%s). Reduce batch_size or num_workers if this persists.",
            ram_ceiling_fraction(),
        )
        _logged_ram_wait = True
    deadline = time.monotonic() + limit
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _logged_ram_wait = False
            raise RuntimeError(
                "LitData RAM ceiling was reached and MemAvailable did not recover within "
                f"{limit:.0f}s. Reduce batch_size, prefetch_factor, or retained batches; "
                "set LITDATA_RAM_WAIT_TIMEOUT=0 to disable this protection."
            )
        time.sleep(min(poll_s, remaining))
        pressure = ram_pressure_fraction()
        budget = ram_budget_bytes()
        if budget is None or (pressure <= resume_pressure and budget >= need):
            _logged_ram_wait = False
            return budget
    return budget


def posix_max_data_workers(
    *,
    requested: int,
    ram_bytes: int | None = None,
    rss_bytes: int | None = None,
) -> int:
    """Cap DataLoader workers so process RSS fits ``MemAvailable``.

    ``num_workers=os.cpu_count()`` on a loaded H100/Vast node (hundreds of cores,
    tens of GiB free) OOMs / EMFILE even when WILLNEED is skipped.
    ``LITDATA_POSIX_MAX_WORKERS=0`` disables the cap.
    """
    requested = max(0, requested)
    if requested == 0:
        return 0
    raw = os.getenv("LITDATA_POSIX_MAX_WORKERS")
    if raw is not None and raw.strip():
        try:
            forced = int(raw)
        except ValueError:
            forced = -1
        if forced == 0:
            return requested
        if forced > 0:
            return min(requested, forced)
    ram = ram_bytes if ram_bytes is not None else available_ram_bytes()
    if ram is None:
        return requested
    rss = rss_bytes if rss_bytes is not None else _DEFAULT_WORKER_RSS
    raw_rss = os.getenv("LITDATA_POSIX_WORKER_RSS")
    if raw_rss and rss_bytes is None:
        with contextlib.suppress(ValueError):
            rss = max(1, int(raw_rss))
    budget = ram_budget_bytes(ram_bytes=ram)
    if budget is None:
        return requested
    capped = max(1, max(0, budget) // max(1, rss))
    return min(requested, capped)


def raise_nofile_limit(target: int = 1_048_576) -> int | None:
    """Raise the soft ``RLIMIT_NOFILE`` toward ``target`` (best-effort)."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except OSError:
        return None
    inf = getattr(resource, "RLIM_INFINITY", -1)
    ceiling = target if hard in (inf, -1) else min(target, hard)
    if ceiling <= soft:
        return soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))
        return ceiling
    except (ValueError, OSError):
        return soft


def mean_chunk_bytes(config: Any) -> int:
    chunks = getattr(config, "_chunks", None) or []
    if not chunks:
        return 64 * 1024 * 1024
    total = 0
    n = 0
    for chunk in chunks:
        size = chunk.get("chunk_bytes")
        if size:
            total += int(size)
            n += 1
    if n:
        return max(1, total // n)
    try:
        num_bytes = int(config.num_bytes)
    except (AttributeError, TypeError, ValueError):
        return 64 * 1024 * 1024
    if num_bytes <= 0:
        return 64 * 1024 * 1024
    return max(1, num_bytes // max(1, len(chunks)))


def posix_page_bytes() -> int:
    """How many sequential payload bytes to keep as a mapped view (then split into items)."""
    raw = os.getenv("LITDATA_POSIX_PAGE_BYTES")
    if raw is None or not raw.strip():
        return _DEFAULT_PAGE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_PAGE_BYTES


def advise_willneed(path: str) -> None:
    """Ask the kernel to pull ``path`` into the page cache (FFCV-style page warm)."""
    if os.name != "posix" or not os.path.isfile(path):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        if hasattr(os, "posix_fadvise"):
            size = 0
            try:
                size = os.fstat(fd).st_size
            except OSError:
                size = 0
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_SEQUENTIAL)
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
    except OSError:
        logger.debug("posix_fadvise failed for %s", path)
    finally:
        os.close(fd)


def advise_dontneed(path: str) -> None:
    """Drop ``path`` from the page cache so consumed chunks do not pin RAM."""
    if os.name != "posix" or not os.path.isfile(path):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        if hasattr(os, "posix_fadvise"):
            size = 0
            try:
                size = os.fstat(fd).st_size
            except OSError:
                size = 0
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    except OSError:
        logger.debug("posix_fadvise DONTNEED failed for %s", path)
    finally:
        os.close(fd)


def madvise_mmap(mapping: Any, *, willneed: bool = True) -> None:
    """Hint sequential access; ``WILLNEED`` only when the prefetch window fits in RAM."""
    madvise = getattr(mapping, "madvise", None)
    if madvise is None:
        return
    mmap_mod = __import__("mmap")
    names = ("MADV_SEQUENTIAL",) + (("MADV_WILLNEED",) if willneed else ())
    for name in names:
        flag = getattr(mmap_mod, name, None)
        if flag is None:
            continue
        try:
            madvise(flag)
        except (OSError, OverflowError, ValueError):
            continue


def madvise_mmap_dontneed(mapping: Any) -> None:
    """Release pages of an mmap that is leaving the local LRU."""
    madvise = getattr(mapping, "madvise", None)
    if madvise is None:
        return
    flag = getattr(__import__("mmap"), "MADV_DONTNEED", None)
    if flag is None:
        return
    try:
        madvise(flag)
    except (OSError, OverflowError, ValueError):
        pass


def posix_fast_supports_config(config: Any) -> bool:
    """Compressed and Mosaic MDS chunks are not LitData mmap payloads."""
    if config is None:
        return False
    if getattr(config, "_compressor", None) is not None:
        return False
    cfg = getattr(config, "_config", None) or {}
    return cfg.get("format") != "mds"

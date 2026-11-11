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
_DEFAULT_WORKER_RSS = 256 * 1024 * 1024  # process + one collated JPEG batch, not four WILLNEED chunks
_logged_willneed_skip = False


def available_ram_bytes(meminfo_text: str | None = None) -> int | None:
    """``MemAvailable`` in bytes from ``/proc/meminfo``, or ``None`` if unknown."""
    text = meminfo_text
    if text is None:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return None
    available = None
    fallback = 0
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            available = int(parts[1]) * 1024
            break
        if line.startswith("MemFree:") or line.startswith("Cached:"):
            parts = line.split()
            fallback += int(parts[1]) * 1024
    if available is not None:
        return available
    return fallback or None


def posix_ram_fraction() -> float:
    raw = os.getenv("LITDATA_POSIX_RAM_FRACTION")
    if raw is None or not raw.strip():
        return _DEFAULT_RAM_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RAM_FRACTION
    return min(1.0, max(0.0, value))


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
    budget = max(1, int(ram * posix_ram_fraction()))
    capped = max(1, budget // max(1, rss))
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
    return max(1, num_bytes // len(chunks))


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


def posix_fast_supports_config(config: Any) -> bool:
    """Compressed and Mosaic MDS chunks are not LitData mmap payloads."""
    if config is None:
        return False
    if getattr(config, "_compressor", None) is not None:
        return False
    cfg = getattr(config, "_config", None) or {}
    return cfg.get("format") != "mds"

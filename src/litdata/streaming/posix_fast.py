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

When ``shuffle=True``, chunk order is a per-worker sliding-window permute (not a global
chunk permutation) so sequential POSIX reads stay in the page cache. See ``WindowShuffle``.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class PosixFastProfile:
    """How StreamingDataset should read a local/POSIX dataset."""

    kind: str  # posix | vast | nfs | lustre | gpfs | forced
    in_place: bool = True
    mmap_shared: bool = True
    skip_cache_copy: bool = True
    skip_chunk_delete: bool = True


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


def _mount_for_path(path: str, mounts: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    abs_path = os.path.abspath(path)
    best: tuple[str, str, str] | None = None
    for mountpoint, fstype, source in mounts:
        if (abs_path == mountpoint or abs_path.startswith(mountpoint.rstrip("/") + "/")) and (
            best is None or len(mountpoint) > len(best[0])
        ):
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

    if forced is True:
        return PosixFastProfile(kind="forced")

    if not path:
        return None

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


def madvise_mmap(mapping: Any) -> None:
    """Hint sequential / will-need on an ``mmap.mmap`` (Linux)."""
    madvise = getattr(mapping, "madvise", None)
    if madvise is None:
        return
    mmap_mod = __import__("mmap")
    for name in ("MADV_SEQUENTIAL", "MADV_WILLNEED"):
        flag = getattr(mmap_mod, name, None)
        if flag is None:
            continue
        try:
            madvise(flag)
        except (OSError, OverflowError, ValueError):
            continue

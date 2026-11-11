"""Hugging Face Hub helpers for listing, parsing, and range-reading parquet.

Listing follows Datatrove ``DataFolder.list_files``: recursive ``find`` with
``expand_info=False`` so a tree like FineWeb / OpenThoughts is cheap to scan.
Reads open the object with a 32 MiB fsspec block (HF ``datasets`` range cache
and Datatrove ``block_size``) so PyArrow footer / row-group GETs coalesce.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Hugging Face special refs that contain slashes (see huggingface_hub.utils._hf_uris).
_HF_SPECIAL_REVISION = re.compile(r"^(refs/(?:convert/[\w.-]+|pr/\d+))(?:/(.*))?$")

# Hugging Face datasets CacheOptions.range_size_limit; Datatrove token I/O uses ~20 MiB.
HF_PARQUET_BLOCK_SIZE = 32 * 1024 * 1024


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """Split an ``hf://`` URI into ``(repo_id, revision, path_in_repo)``.

    Understands ``hf://datasets/org/name@refs/convert/parquet/default/train/0000.parquet``
    (revision ``refs/convert/parquet``). Uses ``huggingface_hub.utils.parse_hf_uri`` when
    available; otherwise a local split that keeps slash-containing special refs intact.
    """
    try:
        from huggingface_hub.utils import parse_hf_uri

        parsed = parse_hf_uri(url)
        return parsed.id, parsed.revision, parsed.path_in_repo or ""
    except Exception:
        return _parse_hf_url_fallback(url)


def _parse_hf_url_fallback(url: str) -> tuple[str, str | None, str]:
    rest = url.removeprefix("hf://")
    for prefix in ("datasets/", "models/", "spaces/"):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    if "@" in rest:
        repo_id, after = rest.split("@", 1)
        match = _HF_SPECIAL_REVISION.match(after)
        if match:
            return repo_id, match.group(1), match.group(2) or ""
        revision, _, path = after.partition("/")
        return repo_id, revision, path
    parts = rest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid Hugging Face URI (expected namespace/name): {url}")
    return f"{parts[0]}/{parts[1]}", None, parts[2] if len(parts) > 2 else ""


def hf_token(storage_options: dict | None = None) -> str | None:
    opts = storage_options or {}
    return opts.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def hf_relative_name(listed_url: str, file_name: str) -> str:
    """Path of a parquet file relative to the listed ``hf://`` prefix (not just the basename)."""
    prefix = listed_url.removeprefix("hf://").rstrip("/")
    name = str(file_name).removeprefix("hf://").lstrip("/")
    if prefix and name.startswith(prefix + "/"):
        return name[len(prefix) + 1 :]
    return os.path.basename(name)


def get_hf_filesystem(storage_options: dict | None = None) -> Any:
    from huggingface_hub import HfFileSystem

    return HfFileSystem(token=hf_token(storage_options), block_size=HF_PARQUET_BLOCK_SIZE)


def open_hf_parquet(fs: Any, url: str) -> Any:
    """Open a Hub parquet for PyArrow range reads (no full-file download)."""
    try:
        return fs.open(url, "rb", block_size=HF_PARQUET_BLOCK_SIZE)
    except TypeError:
        return fs.open(url, "rb")


def list_hf_parquet_files(fs: Any, url: str) -> list[dict[str, Any]]:
    """List ``.parquet`` files under ``url``, recursively when the prefix is a tree."""
    files = _find_parquet(fs, url)
    if files:
        return files

    files = _ls_parquet(fs, url)
    if files:
        return files

    return _glob_parquet(fs, url)


def _entry(name: str, info: Any) -> dict[str, Any] | None:
    if not isinstance(name, str) or not name.endswith(".parquet"):
        return None
    rec: dict[str, Any] = {"name": name, "type": "file"}
    if isinstance(info, dict):
        if info.get("type") == "directory":
            return None
        if "size" in info:
            rec["size"] = info["size"]
        if info.get("name"):
            rec["name"] = info["name"]
    return rec


def _find_parquet(fs: Any, url: str) -> list[dict[str, Any]]:
    find_fn = getattr(fs, "find", None)
    if not callable(find_fn):
        return []
    try:
        listing = find_fn(url, detail=True, expand_info=False)
    except TypeError:
        try:
            listing = find_fn(url, detail=True)
        except Exception:
            return []
    except Exception:
        return []
    if not isinstance(listing, dict):
        return []
    files: list[dict[str, Any]] = []
    for name, info in listing.items():
        rec = _entry(str(name), info)
        if rec is not None:
            files.append(rec)
    files.sort(key=lambda item: item["name"])
    return files


def _ls_parquet(fs: Any, url: str) -> list[dict[str, Any]]:
    try:
        entries = fs.ls(url, detail=True) or []
    except Exception:
        return []
    files: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            rec = _entry(str(entry.get("name", "")), entry)
            if rec is not None:
                files.append(rec)
    return files


def _glob_parquet(fs: Any, url: str) -> list[dict[str, Any]]:
    glob_fn = getattr(fs, "glob", None)
    if not callable(glob_fn):
        return []
    try:
        raw = glob_fn(f"{url.rstrip('/')}/**/*.parquet")
    except Exception:
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    files: list[dict[str, Any]] = []
    for name in raw:
        if not isinstance(name, str) or not name.endswith(".parquet"):
            continue
        rec: dict[str, Any] = {"name": name, "type": "file"}
        try:
            info = fs.info(name)
        except Exception:
            info = None
        if isinstance(info, dict) and "size" in info:
            rec["size"] = info["size"]
        files.append(rec)
    return files

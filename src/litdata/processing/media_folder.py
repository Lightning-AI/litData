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

"""Folder and WebDataset-style ingest helpers."""

from __future__ import annotations

import os
import tarfile
from collections.abc import Iterator
from typing import Any

TEXT_EXTENSIONS = ("txt", "md")
IMAGE_EXTENSIONS = (
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif",
    "webp",
    "tif",
    "tiff",
)
VIDEO_EXTENSIONS = ("mp4", "ogv", "mjpeg", "avi", "mov", "h264", "mpg", "mpeg", "webm", "wmv", "mkv")
AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "wma", "pcm")
MESH_EXTENSIONS = ("glb", "ply", "stl")
PDF_EXTENSIONS = ("pdf",)
NIFTI_EXTENSIONS = ("nii", "nii.gz")

_KIND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "text": TEXT_EXTENSIONS,
    "image": IMAGE_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
    "mesh": MESH_EXTENSIONS,
    "pdf": PDF_EXTENSIONS,
    "nifti": NIFTI_EXTENSIONS,
}


def _file_matches(name: str, extensions: tuple[str, ...]) -> bool:
    lower = name.lower().split("?", 1)[0]
    if lower.endswith(".nii.gz"):
        return "nii.gz" in extensions
    ext = os.path.splitext(lower)[1].lstrip(".")
    return ext in extensions


def list_media_folder(
    root: str,
    kind: str = "image",
    drop_labels: bool = False,
) -> list[dict[str, Any]]:
    """List a class-folder tree (``root/class_name/file.ext``).

    Each item is ``{"path": abs_path, "label": parent_dir_name_or_None}``.
    Files directly under ``root`` have ``label=None``. Pass the list to
    ``optimize`` / ``map``. ``kind`` selects extensions (``text``, ``image``,
    ``video``, ``audio``, ``mesh``, ``pdf``, ``nifti``).
    """
    if kind not in _KIND_EXTENSIONS:
        raise ValueError(f"Unknown media kind {kind!r}. Expected one of {sorted(_KIND_EXTENSIONS)}.")
    extensions = _KIND_EXTENSIONS[kind]
    root = os.path.abspath(root)
    items: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not _file_matches(filename, extensions):
                continue
            path = os.path.join(dirpath, filename)
            label = None
            if not drop_labels:
                parent = os.path.basename(dirpath)
                if os.path.abspath(dirpath) != root:
                    label = parent
            items.append({"path": path, "label": label})
    items.sort(key=lambda item: item["path"])
    return items


def iter_webdataset_tar(tar_path: str) -> Iterator[dict[str, Any]]:
    """Yield one dict per WebDataset key from a ``.tar`` (``__key__``, plus each suffix)."""
    current: dict[str, Any] = {}
    current_key: str | None = None
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            base = os.path.basename(name)
            if "." not in base:
                continue
            key, field = base.split(".", 1)
            if current_key is not None and key != current_key:
                current["__key__"] = current_key
                current["__url__"] = tar_path
                yield current
                current = {}
            current_key = key
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            current[field] = extracted.read()
    if current_key is not None:
        current["__key__"] = current_key
        current["__url__"] = tar_path
        yield current

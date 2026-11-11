#!/usr/bin/env python3
"""Resolve a StreamingDataset-style path via litdata.streaming.resolver._resolve_dir.

Prints JSON: path, url, data_connection_id, index_json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

try:
    from litdata.streaming.resolver import _resolve_dir
except ImportError as e:
    print(
        json.dumps({"error": (f"litdata resolver unavailable; install litdata or set PYTHONPATH to src. {e}")}),
        file=sys.stderr,
    )
    sys.exit(2)


def _index_json(path: str | None, url: str | None, user: str) -> str:
    u = user.rstrip("/")
    if u.endswith("index.json"):
        return u
    if path:
        p = path.rstrip("/")
        if p.endswith("index.json"):
            return p
        return os.path.join(p, "index.json")
    if url:
        base = url.rstrip("/")
        if base.endswith("index.json"):
            return base
        return base + "/index.json"
    return u + "/index.json"


def main() -> None:
    """Print JSON for the resolved dataset directory and index.json path."""
    if len(sys.argv) != 2:
        print("usage: resolve_path.py <dataset-path-or-index.json>", file=sys.stderr)
        sys.exit(1)
    user = sys.argv[1]
    d = _resolve_dir(user)
    out = {
        "path": d.path,
        "url": d.url,
        "data_connection_id": d.data_connection_id,
        "index_json": _index_json(d.path, d.url, user),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()

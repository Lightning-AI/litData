"""Adaptable indexed-array example, not a public dataset or training sampler.

Store records once, then read arbitrary named fields and contiguous frame windows.
All fields in one record must share the same frame axis. See README.md before
adapting this to compressed media, multiple tables, or distributed resume.
"""

import argparse
import asyncio
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from litdata.streaming.downloader import Downloader, get_downloader
from litdata.streaming.resolver import _resolve_dir


def write_records(
    records: Iterable[Mapping[str, np.ndarray]], output_dir: str, shard_bytes: int = 128 * 1024**2
) -> None:
    """Write fixed-width array fields into raw binary shards and a JSON index.

    This reference index stays in memory; partition it for very large datasets.
    Each field is contiguous within a shard, so one window needs one range/field.
    These are indexed raw files, not the optimized LitData chunk format.
    """
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    shard = 0
    position = 0
    handle = None
    try:
        for record in records:
            if not record:
                raise ValueError("Records must contain fields")
            fields = {}
            frames = None
            for name, value in record.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Field names must be nonempty strings")
                array = np.asarray(value)
                if array.ndim < 1 or array.dtype.kind not in "biufc" or any(n == 0 for n in array.shape):
                    raise ValueError("Fields must be nonempty, fixed-width numeric/bool arrays with a frame axis")
                if frames is not None and frames != array.shape[0]:
                    raise ValueError("Fields within a record must share the frame axis")
                frames = array.shape[0]
                if handle is None or (position and position + array.nbytes > shard_bytes):
                    if handle is not None:
                        handle.close()
                        shard += 1
                    handle = (root / f"part-{shard:06d}.bin").open("wb")
                    position = 0
                fields[name] = {
                    "file": f"part-{shard:06d}.bin",
                    "offset": position,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
                handle.write(array.tobytes(order="C"))
                position += array.nbytes
            entries.append({"frames": frames, "fields": fields})
    finally:
        if handle is not None:
            handle.close()
    # Publication marker last. A failed write intentionally has no readable index.
    (root / "index.json").write_text(json.dumps({"version": 1, "records": entries}) + "\n")


class ArrayWindowReader:
    """Explicit indexed reads; sampling, batching and checkpoint state belong to the caller."""

    def __init__(
        self, uri: str, index: dict[str, Any], downloader: Downloader, cache_dir: str, max_concurrent_reads: int = 8
    ) -> None:
        if index.get("version") != 1 or max_concurrent_reads < 1:
            raise ValueError("Unsupported index version or invalid concurrency")
        self.uri = uri.rstrip("/")
        self.records = index["records"]
        self.downloader = downloader
        self.cache_dir = Path(cache_dir)
        self.permits = asyncio.Semaphore(max_concurrent_reads)
        for record in self.records:
            if not isinstance(record["frames"], int) or record["frames"] < 1:
                raise ValueError("Invalid frame count")
            for field in record["fields"].values():
                path = PurePosixPath(field["file"])
                shape = field["shape"]
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or ":" in field["file"]
                    or "\\" in field["file"]
                    or len(path.parts) != 1
                ):
                    raise ValueError("Index shard paths must be plain relative filenames")
                if (
                    not shape
                    or any(not isinstance(n, int) or n < 1 for n in shape)
                    or shape[0] != record["frames"]
                    or np.dtype(field["dtype"]).kind not in "biufc"
                    or not isinstance(field["offset"], int)
                    or field["offset"] < 0
                ):
                    raise ValueError("Invalid field metadata")

    @classmethod
    async def open(
        cls,
        input_dir: str,
        cache_dir: str,
        storage_options: dict[str, Any] | None = None,
        max_concurrent_reads: int = 8,
    ) -> "ArrayWindowReader":
        """Resolve Studio mounts to direct cloud URLs and download metadata before reading ranges."""
        resolved = _resolve_dir(input_dir)
        uri = resolved.url or resolved.path
        if uri is None:
            raise ValueError("An input directory is required")
        options = dict(storage_options or {})
        if resolved.data_connection_id:
            options["data_connection_id"] = resolved.data_connection_id
        if uri.startswith("r2://"):
            options.setdefault("region_name", "auto")
        cache = Path(cache_dir) / hashlib.sha256(uri.encode()).hexdigest()[:16]
        cache.mkdir(parents=True, exist_ok=True)
        downloader = get_downloader(uri, str(cache), [], storage_options=options)
        await downloader.adownload_file(uri.rstrip("/") + "/index.json", str(cache / "index.json"))
        index = json.loads((cache / "index.json").read_text())
        return cls(uri, index, downloader, str(cache), max_concurrent_reads)

    async def read(self, record: int, start: int, frames: int, fields: Sequence[str]) -> dict[str, np.ndarray]:
        """Return owned arrays for [start, start + frames), preserving request field order."""
        if not 0 <= record < len(self.records):
            raise IndexError("Unknown record")
        entry = self.records[record]
        if start < 0 or frames < 1 or start + frames > entry["frames"]:
            raise IndexError("Window is outside the record")
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate requested fields")
        plans = []
        for name in fields:
            field = entry["fields"][name]  # Unknown fields fail before any payload I/O.
            dtype = np.dtype(field["dtype"])
            stride = dtype.itemsize * math.prod(field["shape"][1:])
            plans.append((name, field, dtype, field["offset"] + start * stride, frames * stride))

        async def fetch(plan: tuple) -> tuple[str, np.ndarray]:
            name, field, dtype, offset, length = plan
            async with self.permits:
                data = await self.downloader.adownload_bytes(
                    self.uri + "/" + field["file"], offset, length, str(self.cache_dir / field["file"])
                )
            shape = (frames, *field["shape"][1:])
            return name, np.frombuffer(data, dtype=dtype).reshape(shape).copy()

        # Drain every task on failure so later reads do not inherit pending requests.
        tasks = [asyncio.create_task(fetch(plan)) for plan in plans]
        try:
            return dict(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", help="Create a small synthetic dataset in a NEW local directory")
    parser.add_argument("--input", help="Read a dataset directory, cloud URI or Studio connection path")
    parser.add_argument("--cache", default=".cache/litdata-array-example")
    args = parser.parse_args()
    if args.write:
        rng = np.random.default_rng(42)
        write_records(
            (
                {"features": rng.normal(size=(n, 8)).astype(np.float32), "valid": np.ones(n, dtype=bool)}
                for n in (128, 192, 256)
            ),
            args.write,
        )
    if args.input:

        async def run() -> None:
            reader = await ArrayWindowReader.open(args.input, args.cache)
            sample = await reader.read(record=1, start=7, frames=64, fields=["features", "valid"])
            print({name: {"shape": value.shape, "dtype": str(value.dtype)} for name, value in sample.items()})

        asyncio.run(run())


if __name__ == "__main__":
    main()

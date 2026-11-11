#!/usr/bin/env python3
"""Before/after microbench for BinaryWriter, BinaryReader, and serializers.

PYTHONPATH=src python scripts/bench/bench_writer_reader.py --label after
PYTHONPATH=/path/to/old/src python scripts/bench/bench_writer_reader.py --label before
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = os.environ.get("LITDATA_BENCH_SRC", str(REPO_ROOT / "src"))
sys.path.insert(0, _SRC)

from litdata.streaming.reader import BinaryReader  # noqa: E402
from litdata.streaming.sampler import ChunkedIndex  # noqa: E402
from litdata.streaming.serializers import (  # noqa: E402
    BooleanSerializer,
    IntegerSerializer,
    JPEGSerializer,
    _get_serializers,
)
from litdata.streaming.writer import BinaryWriter  # noqa: E402


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _repeat(fn, n: int = 5) -> tuple[float, float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), _median(times)


def _jpeg_file(tmpdir: str, size: int = 224) -> tuple[str, object]:
    path = os.path.join(tmpdir, f"im_{size}.jpg")
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path, format="JPEG", quality=95)
    return path, PILImage.open(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    import litdata

    src = Path(litdata.__file__).resolve().parent
    print(f"label={args.label}  litdata={src}")

    rows: list[tuple[str, float, float]] = []

    def add(name: str, fn) -> None:
        best, med = _repeat(fn, args.repeats)
        rows.append((name, best, med))
        print(f"  {name:42s}  best={best * 1e3:8.2f} ms  median={med * 1e3:8.2f} ms")

    add("_get_serializers x200", lambda: [_get_serializers(None) for _ in range(200)])

    ints = IntegerSerializer()
    add("IntegerSerializer.serialize x20000", lambda: [ints.serialize(i) for i in range(20000)])
    packed, _ = ints.serialize(42)
    add("IntegerSerializer.deserialize x20000", lambda: [ints.deserialize(packed) for _ in range(20000)])

    booleans = BooleanSerializer()
    add("BooleanSerializer.serialize x20000", lambda: [booleans.serialize(True) for _ in range(20000)])

    with tempfile.TemporaryDirectory() as tmp:
        path, jpeg = _jpeg_file(tmp, 224)
        jser = JPEGSerializer()
        add("JPEGSerializer.serialize JpegImageFile x50", lambda: [jser.serialize(jpeg) for _ in range(50)])
        data, _ = jser.serialize(jpeg)
        add("JPEGSerializer.deserialize → tensor x50", lambda: [jser.deserialize(data) for _ in range(50)])

        def write_ints() -> None:
            d = os.path.join(tmp, "ints")
            os.makedirs(d, exist_ok=True)
            for name in os.listdir(d):
                os.remove(os.path.join(d, name))
            writer = BinaryWriter(d, chunk_size=256)
            for i in range(4000):
                writer[i] = {"id": i, "flag": i % 2 == 0, "x": float(i)}
            writer.done()
            writer.merge()

        add("BinaryWriter 4000 int/bool/float", write_ints)

        write_ints()
        ints_dir = os.path.join(tmp, "ints")

        def read_ints() -> None:
            reader = BinaryReader(ints_dir)
            for i in range(4000):
                sample = reader.read(ChunkedIndex(i, chunk_index=i // 256))
                assert sample["id"] == i

        add("BinaryReader 4000 int/bool/float", read_ints)

        def write_jpegs() -> None:
            d = os.path.join(tmp, "jpegs")
            os.makedirs(d, exist_ok=True)
            for name in os.listdir(d):
                os.remove(os.path.join(d, name))
            writer = BinaryWriter(d, chunk_size=64)
            for i in range(256):
                writer[i] = {"image": jpeg, "id": i}
            writer.done()
            writer.merge()

        add("BinaryWriter 256 JpegImageFile 224", write_jpegs)
        write_jpegs()
        jpeg_dir = os.path.join(tmp, "jpegs")

        def read_jpegs() -> None:
            reader = BinaryReader(jpeg_dir)
            for i in range(256):
                sample = reader.read(ChunkedIndex(i, chunk_index=i // 64))
                assert sample["id"] == i

        add("BinaryReader 256 JPEG → tensor", read_jpegs)

        def write_paths() -> None:
            d = os.path.join(tmp, "paths")
            os.makedirs(d, exist_ok=True)
            for name in os.listdir(d):
                os.remove(os.path.join(d, name))
            writer = BinaryWriter(d, chunk_size=64)
            for i in range(128):
                writer[i] = {"image": path, "id": i}
            writer.done()
            writer.merge()

        add("BinaryWriter 128 jpeg filepath", write_paths)
        write_paths()
        path_dir = os.path.join(tmp, "paths")

        def read_paths() -> None:
            reader = BinaryReader(path_dir)
            for i in range(128):
                sample = reader.read(ChunkedIndex(i, chunk_index=i // 64))
                assert sample["id"] == i

        add("BinaryReader 128 jpeg filepath", read_paths)

    print()
    print(f"{'bench':42s}  {'best_ms':>10s}  {'median_ms':>10s}  [{args.label}]")
    for name, best, med in rows:
        print(f"{name:42s}  {best * 1e3:10.2f}  {med * 1e3:10.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Before/after bench for VideoSerializer / AudioSerializer.

HEAD video always materializes (frames, audio, metadata). After defaults to a
lazy torchcodec decoder. AudioSerializer did not exist on HEAD.

  ffmpeg -y -f lavfi -i testsrc=size=64x64:rate=8:duration=1 -pix_fmt yuv420p /tmp/litdata_av.mp4
  PYTHONPATH=src python scripts/bench/bench_video_audio.py --label after --video /tmp/litdata_av.mp4
  LITDATA_BENCH_SRC=/path/to/old/src python scripts/bench/bench_video_audio.py --label before --video CLIP.mp4
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = os.environ.get("LITDATA_BENCH_SRC")
if _SRC:
    sys.path.insert(0, _SRC)
else:
    sys.path.insert(0, str(REPO_ROOT / "src"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="current")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--video", default="/tmp/litdata_av.mp4")
    args = parser.parse_args()

    import litdata
    from litdata.streaming.serializers import VideoSerializer

    src = Path(litdata.__file__).resolve().parent
    print(f"label={args.label}  litdata={src}")
    if not os.path.isfile(args.video):
        raise SystemExit(f"missing video {args.video}; generate with ffmpeg (see docstring)")

    video_path = args.video
    supports_decode = "decode" in inspect.signature(VideoSerializer.__init__).parameters
    print(f"  VideoSerializer.decode_kw={supports_decode}")

    def add(name: str, fn) -> None:
        best, med = _repeat(fn, args.repeats)
        print(f"  {name:48s}  best={best * 1e3:8.2f} ms  median={med * 1e3:8.2f} ms")

    default_ser = VideoSerializer()
    packed, _ = default_ser.serialize(video_path)
    print(f"  serialized_video_bytes={len(packed)}")

    add("video.serialize(path) x20", lambda: [VideoSerializer().serialize(video_path) for _ in range(20)])
    try:
        add("video.deserialize default x20", lambda: [default_ser.deserialize(packed) for _ in range(20)])
    except Exception as exc:
        print(f"  video.deserialize default skipped: {type(exc).__name__}: {exc}")

    if supports_decode:
        bytes_ser = VideoSerializer(decode="bytes")
        add("video.deserialize decode=bytes x200", lambda: [bytes_ser.deserialize(packed) for _ in range(200)])
        try:
            all_ser = VideoSerializer(decode="all")
            add("video.deserialize decode=all x5", lambda: [all_ser.deserialize(packed) for _ in range(5)])
        except Exception as exc:
            print(f"  video.deserialize decode=all skipped: {type(exc).__name__}: {exc}")

        from litdata.streaming.serializers import AudioSerializer
        from litdata.types import Audio

        wav = np.zeros(16000, dtype=np.float32)
        audio = Audio(array=wav, sampling_rate=16000)
        audio_serializer = AudioSerializer(decode="bytes")
        audio_bytes, _ = audio_serializer.serialize(audio)
        print(f"  serialized_audio_bytes={len(audio_bytes)}")
        add(
            "audio.serialize(array) x50",
            lambda: [AudioSerializer(decode="bytes").serialize(audio) for _ in range(50)],
        )
        add(
            "audio.deserialize decode=bytes x200",
            lambda: [audio_serializer.deserialize(audio_bytes) for _ in range(200)],
        )
        try:
            dec_audio = AudioSerializer(decode="decoder")
            add("audio.deserialize default x50", lambda: [dec_audio.deserialize(audio_bytes) for _ in range(50)])
            samples_serializer = AudioSerializer(decode="samples")
            add(
                "audio.deserialize decode=samples x20",
                lambda: [samples_serializer.deserialize(audio_bytes) for _ in range(20)],
            )
        except Exception as exc:
            print(f"  audio decoder skipped: {type(exc).__name__}: {exc}")
    else:
        try:
            add(
                "video.deserialize (HEAD always materializes) x5",
                lambda: [default_ser.deserialize(packed) for _ in range(5)],
            )
        except Exception as exc:
            print(f"  video.deserialize HEAD skipped: {type(exc).__name__}: {exc}")
        print("  (HEAD: no decode= / no AudioSerializer)")


if __name__ == "__main__":
    main()

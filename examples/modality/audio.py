"""Audio: wrap WAV files on disk, stream array + sampling rate.

Needs torchcodec to decode (included in litdata[extra] on Linux/macOS).
Decoders do not stack — collate to a list, or decode in collate_fn.
"""

import wave
from pathlib import Path

import numpy as np

from litdata import Audio, StreamingDataLoader, StreamingDataset, list_media_folder, optimize


def make_sample(item: dict) -> dict:
    return {
        "audio": Audio(path=item["path"]),
        "caption": item["label"],
    }


def seed_folder(root: Path) -> None:
    t = np.arange(16000, dtype=np.float32) / 16000.0
    for label, freq in (("tone_a", 440.0), ("tone_b", 554.0)):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        pcm = (0.2 * np.sin(2 * np.pi * freq * t) * 32767.0).astype(np.int16)
        for index in range(4):
            path = folder / f"{index}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(pcm.tobytes())


def collate_fn(samples: list) -> dict:
    return {
        "array": [sample["audio"]["array"] for sample in samples],
        "sampling_rate": [sample["audio"]["sampling_rate"] for sample in samples],
        "caption": [sample["caption"] for sample in samples],
    }


if __name__ == "__main__":
    media_dir = Path("example_optimize_dataset/source/audio")
    seed_folder(media_dir)

    optimize(
        fn=make_sample,
        inputs=list_media_folder(str(media_dir), kind="audio"),
        output_dir="example_optimize_dataset/audio",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/audio")
    sample = dataset[0]
    array = sample["audio"]["array"]
    rate = sample["audio"]["sampling_rate"]
    print(sample["caption"], array.shape, rate)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)))
    print(len(batch["array"]), batch["array"][0].shape, batch["caption"])

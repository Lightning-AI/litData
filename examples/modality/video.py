"""Video: wrap MP4 files on disk, stream a torchcodec VideoDecoder.

Needs torchcodec to decode. Keep decode on CPU in DataLoader workers.
Decoders do not stack — collate to a list.
"""

from pathlib import Path

import numpy as np

from litdata import StreamingDataLoader, StreamingDataset, Video, list_media_folder, optimize
from litdata.streaming.serializers import _encode_video_array


def make_sample(item: dict) -> dict:
    return {
        "video": Video(path=item["path"]),
        "caption": item["label"],
    }


def seed_folder(root: Path) -> None:
    for label in ("clip_a", "clip_b"):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            frames = np.random.randint(0, 256, (8, 64, 64, 3), np.uint8)
            (folder / f"{index}.mp4").write_bytes(_encode_video_array(frames, 25))


def collate_fn(samples: list) -> dict:
    return {
        "video": [sample["video"] for sample in samples],
        "caption": [sample["caption"] for sample in samples],
    }


if __name__ == "__main__":
    media_dir = Path("example_optimize_dataset/source/video")
    seed_folder(media_dir)

    optimize(
        fn=make_sample,
        inputs=list_media_folder(str(media_dir), kind="video"),
        output_dir="example_optimize_dataset/video",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/video")
    sample = dataset[0]
    video = sample["video"]
    frame = video.get_frames_at(0)
    clip = video.get_frames_in_range(0, 8)
    print(sample["caption"], frame, clip)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)))
    print(len(batch["video"]), batch["caption"])
    print(batch["video"][0].get_frames_in_range(0, 8))

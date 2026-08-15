"""Image: wrap files on disk, stream CHW tensors.

list_media_folder walks a class-folder tree (root/label/file.jpg).
The caption stays a string — only the image is wrapped.
"""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from litdata import Image, StreamingDataLoader, StreamingDataset, list_media_folder, optimize


def make_sample(item: dict) -> dict:
    return {
        "image": Image(path=item["path"]),
        "caption": item["label"],
    }


def seed_folder(root: Path) -> None:
    rng = np.random.default_rng(0)
    for label in ("cat", "dog"):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            array = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            PILImage.fromarray(array).save(folder / f"{index}.jpg", quality=95)


if __name__ == "__main__":
    media_dir = Path("example_optimize_dataset/source/image")
    seed_folder(media_dir)

    optimize(
        fn=make_sample,
        inputs=list_media_folder(str(media_dir), kind="image"),
        output_dir="example_optimize_dataset/image",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/image")
    sample = dataset[0]
    image = sample["image"]  # Tensor CHW
    print(sample["caption"], image.shape, image.dtype)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["image"].shape, batch["caption"])

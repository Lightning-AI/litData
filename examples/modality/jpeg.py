"""Jpeg: always JPEG (quality defaults to 95), stream a CHW tensor."""

import numpy as np

from litdata import Jpeg, StreamingDataLoader, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    hwc = np.random.randint(0, 256, (64, 64, 3), np.uint8)
    return {
        "index": index,
        # Jpeg(path="a.jpg")
        "image": Jpeg(array=hwc, quality=95),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/jpeg",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/jpeg")
    sample = dataset[0]
    image = sample["image"]  # Tensor CHW
    print(image.shape, image.dtype)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["image"].shape)

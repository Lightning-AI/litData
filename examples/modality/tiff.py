"""Tiff: scientific / multi-page images, stream a NumPy array.

Needs tifffile.
"""

import numpy as np

from litdata import StreamingDataset, Tiff, optimize


def make_sample(index: int) -> dict:
    hw = np.random.randint(0, 256, (64, 64), np.uint16)
    return {
        "index": index,
        # Tiff(path="a.tif")
        "image": Tiff(array=hw),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/tiff",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/tiff")[0]
    array = sample["image"]  # NumPy
    print(array.shape, array.dtype)

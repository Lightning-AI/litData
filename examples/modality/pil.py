"""Pil: uncompressed pixels, stream a PIL.Image.

Use Image / Jpeg for training photos. Use Pil when you need lossless pixels.
"""

import numpy as np
from PIL import Image as PILImage

from litdata import Pil, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    hwc = np.random.randint(0, 256, (64, 64, 3), np.uint8)
    return {
        "index": index,
        # Pil(path="a.png")
        "image": Pil(image=PILImage.fromarray(hwc), mode="RGB"),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/pil",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/pil")[0]
    pil_img = sample["image"]  # PIL.Image
    print(pil_img.size, pil_img.mode)

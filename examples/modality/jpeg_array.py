"""JpegArray: a list of JPEGs, stream a list of CHW tensors."""

import numpy as np

from litdata import Jpeg, JpegArray, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    frames = [np.random.randint(0, 256, (32, 32, 3), np.uint8) for _ in range(4)]
    return {
        "index": index,
        "frames": JpegArray(images=[Jpeg(array=frame, quality=95) for frame in frames]),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/jpeg_array",
        num_workers=2,
        chunk_bytes="64MB",
    )

    sample = StreamingDataset("example_optimize_dataset/jpeg_array")[0]
    images = sample["frames"]  # list of Tensor CHW
    print(len(images), images[0].shape)

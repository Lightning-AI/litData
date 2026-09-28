"""Write a mixed-modality chunk dataset.

Wrap each media field so a caption string is never treated as a file path.
"""

import numpy as np
import torch

from litdata import Audio, File, Graph, Image, Tensor, optimize


def make_sample(index: int) -> dict:
    image = np.random.randint(0, 256, (64, 64, 3), np.uint8)
    # 0.1 s of a 440 Hz tone at 16 kHz
    t = np.arange(1600, dtype=np.float32) / 16000.0
    wave = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    return {
        "index": index,
        "caption": "a synthetic sample",  # stays a string — do not pass this as Image(path=...)
        "image": Image(array=image, quality=95, format="jpeg"),
        "audio": Audio(array=wave, sampling_rate=16000),
        "tokens": Tensor(array=torch.randint(0, 1000, (32,))),
        "graph": Graph(x=x, edge_index=edge_index, y=torch.tensor(index % 3)),
        "sidecar": File(bytes=b'{"split":"train"}'),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(64)),
        output_dir="example_optimize_dataset",
        num_workers=2,
        chunk_bytes="64MB",
    )

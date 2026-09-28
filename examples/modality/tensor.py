"""Tensor: store a torch / NumPy array (2-D+).

1-D token ids for LLM training belong with Text + TokensLoader (see text.py).
"""

import torch

from litdata import StreamingDataLoader, StreamingDataset, Tensor, optimize


def make_sample(index: int) -> dict:
    return {
        "index": index,
        "feat": Tensor(array=torch.randn(3, 4, 4)),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/tensor",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/tensor")
    sample = dataset[0]
    feat = sample["feat"]  # Tensor
    print(feat.shape, feat.dtype)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["feat"].shape)

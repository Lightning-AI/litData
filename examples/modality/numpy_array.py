"""NumPy: load .npy files, stream ndarrays.

A 2-D+ array keeps its shape. A 1-D array uses the no-header layout
(same idea as token ids). Use Tensor(...) for torch tensors.
"""

from pathlib import Path

import numpy as np

from litdata import StreamingDataLoader, StreamingDataset, optimize


def make_sample(path: str) -> dict:
    return {
        "array": np.array(np.load(path), copy=True),
        "caption": Path(path).stem,
    }


def seed_folder(root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for index in range(8):
        path = root / f"feat_{index}.npy"
        np.save(path, rng.standard_normal((3, 4, 4)).astype(np.float32))
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    paths = seed_folder(Path("example_optimize_dataset/source/numpy"))

    optimize(
        fn=make_sample,
        inputs=paths,
        output_dir="example_optimize_dataset/numpy",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/numpy")
    sample = dataset[0]
    array = sample["array"]  # NumPy
    print(sample["caption"], array.shape, array.dtype)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["array"].shape)

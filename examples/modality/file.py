"""File: wrap files on disk, stream raw bytes.

Use this when you want the original file, not a decoded tensor.
"""

from pathlib import Path

from litdata import File, StreamingDataLoader, StreamingDataset, optimize


def make_sample(path: str) -> dict:
    return {
        "sidecar": File(path=path),
        "caption": Path(path).stem,
    }


def seed_folder(root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(8):
        path = root / f"item_{index}.json"
        path.write_bytes(f'{{"split":"train","index":{index}}}'.encode())
        paths.append(str(path))
    return paths


def collate_fn(samples: list) -> dict:
    return {
        "sidecar": [bytes(sample["sidecar"]) for sample in samples],
        "caption": [sample["caption"] for sample in samples],
    }


if __name__ == "__main__":
    paths = seed_folder(Path("example_optimize_dataset/source/file"))

    optimize(
        fn=make_sample,
        inputs=paths,
        output_dir="example_optimize_dataset/file",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/file")
    sample = dataset[0]
    sidecar = bytes(sample["sidecar"])  # raw bytes
    print(sample["caption"], sidecar)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)))
    print(len(batch["sidecar"]), batch["caption"])

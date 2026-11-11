"""Write full records with optimize; select windows using built-in LitData APIs."""

import argparse

import numpy as np

from litdata import StreamingDataset, TemporalArrayLoader, optimize


def make_record(frames: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(frames)
    return {
        "features": rng.normal(size=(frames, 8)).astype(np.float32),
        "clock": np.arange(frames, dtype=np.int64),
        "valid": np.ones(frames, dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", help="Create a dataset in a NEW directory or cloud prefix")
    parser.add_argument("--input", help="Read a dataset directory, cloud URI or Studio connection path")
    parser.add_argument("--cache", default=".cache/litdata-array-example")
    args = parser.parse_args()
    if args.write:
        optimize(
            make_record,
            [128, 192, 256],
            output_dir=args.write,
            chunk_bytes="128MB",
            num_workers=1,
            item_loader=TemporalArrayLoader(field_groups=[["clock", "valid"]]),
        )
    if args.input:
        dataset = StreamingDataset(args.input, cache_dir=args.cache, item_loader=TemporalArrayLoader())
        sample = dataset.read_window(1, start=7, frames=64, fields=["features", "valid"])
        print({name: {"shape": value.shape, "dtype": str(value.dtype)} for name, value in sample.items()})


if __name__ == "__main__":
    main()

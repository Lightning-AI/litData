"""Parquet: stream existing .parquet files (no optimize).

StreamingDataset indexes a parquet folder automatically. Pass ParquetLoader
so each sample is a dict of columns. On Linux, use spawn if num_workers > 0.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from litdata import StreamingDataLoader, StreamingDataset
from litdata.streaming.item_loader import ParquetLoader


def seed_folder(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for part in range(2):
        table = pa.table(
            {
                "id": list(range(part * 4, part * 4 + 4)),
                "x": rng.standard_normal(4).astype(np.float32),
                "split": ["train"] * 4,
            }
        )
        pq.write_table(table, root / f"part-{part}.parquet")


if __name__ == "__main__":
    media_dir = Path("example_optimize_dataset/source/parquet")
    seed_folder(media_dir)

    dataset = StreamingDataset(str(media_dir), item_loader=ParquetLoader())
    row = dataset[0]  # dict of columns
    print(row["id"], row["x"], row["split"])

    # Linux + num_workers>0: pass multiprocessing_context="spawn" (Polars + fork).
    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["id"], batch["x"])

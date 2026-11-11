# Copyright The Lightning AI team.
"""Child process for emulating a data-prep node via DATA_OPTIMIZER_* env vars."""

from __future__ import annotations

import os
import sys

from litdata import optimize


def _fn(path: str) -> tuple[int, str]:
    with open(path, "rb") as handle:
        payload = handle.read()
    return len(payload), os.path.basename(path)


def main() -> None:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    manifest = sys.argv[4] if len(sys.argv) > 4 else None
    if manifest:
        with open(manifest) as handle:
            inputs = [line.strip() for line in handle if line.strip()]
    else:
        inputs = sorted(os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith(".bin"))
    optimize(
        fn=_fn,
        inputs=inputs,
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_bytes="64MB",
        num_workers=num_workers,
        num_downloaders=2,
        keep_data_ordered=False,
        reorder_files=False,
        mode="overwrite",
        verbose=True,
    )


if __name__ == "__main__":
    main()

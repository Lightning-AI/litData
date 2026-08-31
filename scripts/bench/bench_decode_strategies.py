#!/usr/bin/env python3
"""Per-item vs windowed vs whole-chunk binary decode, vs parquet, by schema.

PYTHONPATH=src python scripts/bench/bench_decode_strategies.py
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from litdata import StreamingDataset, optimize
from litdata.streaming.item_loader import ParquetLoader
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.hf_dataset import _optimize_hf_file

N = 4000
ROWS = 3000
WARMUP = 200
REPEATS = 5
ROOT = Path("/tmp/litdata_decode_strategies")


def consume(it, limit: int) -> tuple[int, float]:
    n = 0
    t0 = time.perf_counter()
    for _ in it:
        n += 1
        if n >= limit:
            break
    return n, time.perf_counter() - t0


def bench_path(path: Path, *, loader=None, env: dict[str, str] | None = None) -> float:
    old = {key: os.environ.get(key) for key in ("LITDATA_BATCH_DECODE", "LITDATA_BATCH_ROWS")}
    try:
        os.environ.pop("LITDATA_BATCH_DECODE", None)
        os.environ.pop("LITDATA_BATCH_ROWS", None)
        if env:
            os.environ.update(env)
        kwargs: dict = {"shuffle": False}
        if loader is not None:
            kwargs["item_loader"] = loader
        consume(StreamingDataset(str(path), **kwargs), WARMUP)
        n, elapsed = consume(StreamingDataset(str(path), **kwargs), ROWS)
        return n / elapsed if elapsed else 0.0
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def median_rps(path: Path, **kwargs) -> float:
    return statistics.median(bench_path(path, **kwargs) for _ in range(REPEATS))


def write_schemas(root: Path) -> dict[str, tuple[Path, Path]]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    schemas = {
        "flat_short": pa.table(
            {
                "text": [f"review {i} " * 8 for i in range(N)],
                "label": [i % 2 for i in range(N)],
            }
        ),
        "flat_long": pa.table(
            {
                "article": [("word " * 80) + str(i) for i in range(N)],
                "summary": [("sum " * 12) + str(i) for i in range(N)],
            }
        ),
        "nested_qa": pa.table(
            {
                "id": [f"q{i}" for i in range(N)],
                "question": [f"What is the answer to question {i}?" for i in range(N)],
                "choices": [{"text": [f"opt{j}" for j in range(4)], "label": list("ABCD")} for _ in range(N)],
                "answers": [["span"] * (i % 4) for i in range(N)],
                "answer_start": [[10] * (i % 4) for i in range(N)],
            }
        ),
        "chat_recs": pa.table(
            {
                "prompt": [f"user prompt {i}" for i in range(N)],
                "messages": [
                    [
                        {"role": "user", "content": f"hello {i}"},
                        {"role": "assistant", "content": f"hi {i} " * 4},
                    ]
                    for i in range(N)
                ],
            }
        ),
    }
    out: dict[str, tuple[Path, Path]] = {}
    for name, table in schemas.items():
        pq_path = root / f"{name}.parquet"
        opt = root / f"{name}-opt"
        pq_dir = root / f"{name}-pqdir"
        pq.write_table(table, pq_path)
        optimize(
            fn=partial(_optimize_hf_file),
            inputs=[str(pq_path)],
            output_dir=str(opt),
            chunk_bytes="64MB",
            num_workers=1,
            mode="overwrite",
        )
        pq_dir.mkdir()
        os.symlink(pq_path, pq_dir / pq_path.name)
        index_parquet_dataset(str(pq_dir), str(pq_dir), num_workers=1)
        fmt = json.loads((opt / "index.json").read_text())["config"]["data_format"]
        print(f"prepared {name} format={fmt}")
        out[name] = (opt, pq_dir)
    return out


def main() -> None:
    prepared = write_schemas(ROOT)
    modes = [
        ("per_item", {"LITDATA_BATCH_DECODE": "0"}),
        ("win_256", {"LITDATA_BATCH_ROWS": "256"}),
        ("win_1024", {"LITDATA_BATCH_ROWS": "1024"}),
        ("chunk", {"LITDATA_BATCH_DECODE": "all"}),
    ]
    print(
        f"\n{'schema':<12} {'per_item':>10} {'win256':>10} {'win1024':>10} {'chunk':>10} {'parquet':>10} {'best':>10}"
    )
    results = []
    for name, (opt, pq_dir) in prepared.items():
        scores = {}
        for label, env in modes:
            clean = {k: v for k, v in env.items() if v}
            scores[label] = median_rps(opt, env=clean)
        scores["parquet"] = median_rps(pq_dir, loader=ParquetLoader())
        best_bin = max(
            label
            for label in ("per_item", "win_256", "win_1024", "chunk")
            if scores[label] == max(scores[k] for k in ("per_item", "win_256", "win_1024", "chunk"))
        )
        print(
            f"{name:<12} {scores['per_item']:10.0f} {scores['win_256']:10.0f} "
            f"{scores['win_1024']:10.0f} {scores['chunk']:10.0f} {scores['parquet']:10.0f} {best_bin:>10}"
        )
        results.append({"schema": name, **scores, "best_binary": best_bin})
    Path("/tmp/bench_decode_strategies.json").write_text(json.dumps(results, indent=2))
    print("\nwrote /tmp/bench_decode_strategies.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Why nested QA lost to parquet, and whether struct-as-pytree closes it.

PYTHONPATH=src python scripts/bench/bench_nested_qa_gap.py
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
from litdata.streaming.serializers import JsonLeaf, JsonSerializer, _nested_dumps, _nested_loads
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.hf_dataset import _optimize_hf_file, _stabilize_hf_row

N = 4000
ROWS = 3000
WARMUP = 200
REPEATS = 5
ROOT = Path("/tmp/litdata_nested_qa_gap")


def qa_table(n: int = N) -> pa.Table:
    return pa.table(
        {
            "id": [f"q{i}" for i in range(n)],
            "question": [f"What is the answer to question {i}?" for i in range(n)],
            "choices": [{"text": [f"opt{j}" for j in range(4)], "label": list("ABCD")} for _ in range(n)],
            "answers": [["span"] * (i % 4) for i in range(n)],
            "answer_start": [[10] * (i % 4) for i in range(n)],
        }
    )


def consume(it, limit: int) -> tuple[int, float]:
    n = 0
    t0 = time.perf_counter()
    for _ in it:
        n += 1
        if n >= limit:
            break
    return n, time.perf_counter() - t0


def median_rps(path: Path, *, loader=None) -> float:
    rates = []
    for _ in range(REPEATS):
        kwargs: dict = {"shuffle": False}
        if loader is not None:
            kwargs["item_loader"] = loader
        consume(StreamingDataset(str(path), **kwargs), WARMUP)
        n, elapsed = consume(StreamingDataset(str(path), **kwargs), ROWS)
        rates.append(n / elapsed if elapsed else 0.0)
    return statistics.median(rates)


def _old_optimize_file(url: str) -> None:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(url)
    for batch in parquet_file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            yield _stabilize_hf_row(
                row,
                {key: [] if isinstance(row[key], list) else {} if isinstance(row[key], dict) else "" for key in row},
            )


def micro_decode(table: pa.Table) -> dict[str, float]:
    rows = table.to_pylist()
    choices = [row["choices"] for row in rows]
    answers = [row["answers"] for row in rows]
    starts = [row["answer_start"] for row in rows]
    json_ser = JsonSerializer()
    choice_blobs = [json_ser.serialize(JsonLeaf(c))[0] for c in choices]
    answer_blobs = [json_ser.serialize(JsonLeaf(a))[0] for a in answers]
    start_blobs = [json_ser.serialize(JsonLeaf(s))[0] for s in starts]
    text_blobs = [[t.encode() for t in c["text"]] for c in choices]
    label_blobs = [[t.encode() for t in c["label"]] for c in choices]
    list_text = [_nested_dumps(c["text"]) for c in choices]
    list_label = [_nested_dumps(c["label"]) for c in choices]

    def timed(fn, loops: int = 8) -> float:
        fn()
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        return (time.perf_counter() - t0) / loops / N * 1e6

    pq_path = ROOT / "micro.parquet"
    pq.write_table(table, pq_path)

    def pq_pylist() -> None:
        pq.read_table(pq_path).to_pylist()

    return {
        "parquet_to_pylist_us": timed(pq_pylist, loops=5),
        "opaque_choices_dict_us": timed(lambda: [_nested_loads(b) for b in choice_blobs]),
        "two_list_leaves_us": timed(
            lambda: [(_nested_loads(a), _nested_loads(b)) for a, b in zip(list_text, list_label)]
        ),
        "answers_list_us": timed(lambda: [_nested_loads(b) for b in answer_blobs]),
        "starts_list_us": timed(lambda: [_nested_loads(b) for b in start_blobs]),
        "utf8_decode_8_strs_us": timed(
            lambda: [
                [x.decode() for x in texts] + [x.decode() for x in labels]
                for texts, labels in zip(text_blobs, label_blobs)
            ]
        ),
    }


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir()
    table = qa_table()
    pq_path = ROOT / "nested_qa.parquet"
    pq.write_table(table, pq_path)
    pq_dir = ROOT / "pqdir"
    pq_dir.mkdir()
    os.symlink(pq_path, pq_dir / pq_path.name)
    index_parquet_dataset(str(pq_dir), str(pq_dir), num_workers=1)

    new_opt = ROOT / "struct-pytree"
    old_opt = ROOT / "opaque-json"
    optimize(
        fn=partial(_optimize_hf_file),
        inputs=[str(pq_path)],
        output_dir=str(new_opt),
        chunk_bytes="64MB",
        num_workers=1,
        mode="overwrite",
    )
    optimize(
        fn=_old_optimize_file,
        inputs=[str(pq_path)],
        output_dir=str(old_opt),
        chunk_bytes="64MB",
        num_workers=1,
        mode="overwrite",
    )
    new_fmt = json.loads((new_opt / "index.json").read_text())["config"]["data_format"]
    old_fmt = json.loads((old_opt / "index.json").read_text())["config"]["data_format"]
    print(f"struct-pytree format={new_fmt}")
    print(f"opaque-json   format={old_fmt}")

    scores = {
        "opaque_json": median_rps(old_opt),
        "struct_pytree": median_rps(new_opt),
        "parquet": median_rps(pq_dir, loader=ParquetLoader()),
    }
    print(f"\n{'mode':<16} {'rows/s':>10} {'vs pq':>8}")
    for name, rate in scores.items():
        print(f"{name:<16} {rate:10.0f} {rate / scores['parquet']:8.2f}x")

    micros = micro_decode(table)
    print("\nper-row decode (µs, median-ish mean of loops)")
    for key, value in micros.items():
        print(f"  {key:<28} {value:6.2f}")

    out = {"e2e_rps": scores, "new_format": new_fmt, "old_format": old_fmt, "micro_us": micros}
    Path("/tmp/bench_nested_qa_gap.json").write_text(json.dumps(out, indent=2))
    print("\nwrote /tmp/bench_nested_qa_gap.json")


if __name__ == "__main__":
    main()

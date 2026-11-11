#!/usr/bin/env python3
"""LitData binary chunks (optimize_hf) vs hf:// parquet streaming.

Uses the Hub ids that already resolved in the parquet-vs-HF suite when present.
Resume: writes --out after every dataset. optimize_hf reuses output_dir/index.json.

  PYTHONPATH=src python scripts/bench/bench_hf_opt_vs_parquet.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# (repo, split, config) — skip gated / too-many-files / huge-shard sets
DATASETS: list[tuple[str, str, str | None]] = [
    ("stanfordnlp/imdb", "train", None),
    ("yahma/alpaca-cleaned", "train", None),
    ("openai/gsm8k", "train", "main"),
    ("tatsu-lab/alpaca", "train", None),
    ("databricks/databricks-dolly-15k", "train", None),
    ("garage-bAInd/Open-Platypus", "train", None),
    ("HuggingFaceH4/no_robots", "train", None),
    ("SetFit/sst2", "train", None),
    ("SetFit/ag_news", "train", None),
    ("SetFit/emotion", "train", None),
    ("dair-ai/emotion", "train", None),
    ("cardiffnlp/tweet_eval", "train", "sentiment"),
    ("google/boolq", "train", None),
    ("allenai/winogrande", "train", "winogrande_xl"),
    ("allenai/sciq", "train", None),
    ("allenai/openbookqa", "train", "main"),
    ("allenai/qasc", "train", None),
    ("tau/commonsense_qa", "train", None),
    ("rajpurkar/squad", "train", None),
    ("rajpurkar/squad_v2", "train", None),
    ("knkarthick/samsum", "train", None),
    ("Salesforce/wikitext", "train", "wikitext-2-raw-v1"),
    ("nyu-mll/glue", "train", "sst2"),
    ("nyu-mll/glue", "train", "qnli"),
    ("nyu-mll/glue", "train", "rte"),
    ("nyu-mll/glue", "validation", "stsb"),
    ("google-research-datasets/paws", "train", "labeled_final"),
    ("google-research-datasets/nq_open", "train", None),
    ("stanfordnlp/snli", "train", None),
    ("facebook/anli", "train_r3", None),
    ("Rowan/hellaswag", "train", None),
    ("Intel/orca_dpo_pairs", "train", None),
    ("HuggingFaceH4/ultrafeedback_binarized", "train_sft", None),
    ("OpenAssistant/oasst1", "train", None),
    ("EdinburghNLP/xsum", "train", None),
    ("abisee/cnn_dailymail", "train", "3.0.0"),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", None),
    ("Anthropic/hh-rlhf", "train", None),
]


def consume(it, limit: int) -> tuple[int, float]:
    n = 0
    t0 = time.perf_counter()
    for _ in it:
        n += 1
        if n >= limit:
            break
    return n, time.perf_counter() - t0


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/bench_hf_opt_vs_parquet.json")
    parser.add_argument(
        "--opt-root",
        default="/teamspace/lightning_storage/testing/litdata_hf_opt",
        help="Write optimized 64MB chunks here (lightning_storage → R2, not FUSE reads).",
    )
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--limit-datasets", type=int, default=40)
    args = parser.parse_args()

    from litdata import StreamingDataset, optimize_hf
    from litdata.utilities.hf_dataset import resolve_hf_dataset_url

    out_path = Path(args.out)
    results: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = list(prev.get("datasets") or [])
        except Exception:
            results = []
    done = {
        (r.get("repo"), r.get("split"), r.get("config"))
        for r in results
        if r.get("error") is None and "binary_rows_per_s" in r
    }

    picked = DATASETS[: args.limit_datasets]
    print(f"{'dataset':<48} {'opt_s':>7} {'bin/s':>9} {'pq/s':>9} {'bin×':>7}")
    for name, split, config in picked:
        key = (name, split, config)
        if key in done:
            print(f"skip existing {name} {split} {config}")
            continue
        label = f"{name}/{split}" + (f"/{config}" if config else "")
        opt_dir = Path(args.opt_root) / name.replace("/", "--") / f"{split}-{config or 'default'}"
        rec: dict = {"repo": name, "split": split, "config": config, "error": None}
        try:
            uri = resolve_hf_dataset_url(name, split=split, config=config)
            rec["url"] = uri
            t0 = time.perf_counter()
            optimize_hf(
                name,
                output_dir=str(opt_dir),
                split=split,
                config=config,
                chunk_bytes="64MB",
            )
            rec["optimize_s"] = time.perf_counter() - t0
            rec["binary_bytes"] = dir_size(opt_dir)

            ds_bin = StreamingDataset(str(opt_dir), shuffle=False)
            rec["binary_length"] = len(ds_bin)
            limit = min(args.rows, rec["binary_length"])
            consume(StreamingDataset(str(opt_dir), shuffle=False), min(args.warmup, limit))
            n_bin, elapsed_bin = consume(StreamingDataset(str(opt_dir), shuffle=False), limit)

            consume(StreamingDataset(uri, shuffle=False), min(args.warmup, limit))
            n_pq, elapsed_pq = consume(StreamingDataset(uri, shuffle=False), limit)

            rec["rows"] = n_bin
            rec["binary_rows_per_s"] = n_bin / elapsed_bin if elapsed_bin else 0.0
            rec["parquet_rows"] = n_pq
            rec["parquet_rows_per_s"] = n_pq / elapsed_pq if elapsed_pq else 0.0
            rec["ratio"] = rec["binary_rows_per_s"] / rec["parquet_rows_per_s"] if rec["parquet_rows_per_s"] else None
            rec["binary_faster"] = bool(rec["ratio"] and rec["ratio"] > 1)
            print(
                f"{label:<48} {rec['optimize_s']:7.1f} {rec['binary_rows_per_s']:9.0f} "
                f"{rec['parquet_rows_per_s']:9.0f} {rec['ratio']:7.2f}"
            )
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{label:<48} FAIL {rec['error'][:120]}")

        results = [r for r in results if (r.get("repo"), r.get("split"), r.get("config")) != key]
        results.append(rec)
        wins = sum(1 for r in results if r.get("binary_faster") is True)
        losses = sum(1 for r in results if r.get("binary_faster") is False)
        out_path.write_text(
            json.dumps(
                {
                    "rows": args.rows,
                    "warmup": args.warmup,
                    "wins": wins,
                    "losses": losses,
                    "n": len(results),
                    "datasets": results,
                },
                indent=2,
            )
        )

    wins = sum(1 for r in results if r.get("binary_faster") is True)
    losses = sum(1 for r in results if r.get("binary_faster") is False)
    print(f"\ndone binary_faster={wins} parquet_faster={losses} errors={sum(1 for r in results if r.get('error'))}")


if __name__ == "__main__":
    main()

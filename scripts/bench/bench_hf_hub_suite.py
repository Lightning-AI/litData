#!/usr/bin/env python3
"""LitData binary vs LitData parquet vs HF streaming on Hub parquet sets.

~15 significant, mixed-modality splits (text/JSON, images, audio). Tiny NLP
toys are omitted. Each timed pass is **one epoch**, capped at ``len`` — never
wrap or cycle a short split to invent runtime.

Methodology: 200-row warmup, wipe payload cache, then a **cold** first pass
capped at ``--rows`` (default 200000) or ``len``. From first-pass ``n`` and
``elapsed``, large sets may raise the row cap toward ``--target-seconds``
(default 120):

    target_rows = min(len, max(n, ceil(rate * target_seconds)))

If ``target_rows > n``, a second cold pass uses that cap (still one epoch).
If the split is shorter than the heuristic, the first pass at ``len`` is the
result. ``--target-seconds 0`` keeps v1 first-pass-only behavior.

Each timed pass starts from a **cold** payload cache: warmup downloads are
deleted so LitData parquet is not reading Hub files already on disk, and
binary is not reading R2 chunks already in ``cache_dir``. ``index.json`` is
kept (metadata only).

* **Parquet** — ``StreamingDataset("hf://...")``. Files are pulled from the Hub
  (``hf_hub_download``), not from a local folder or FUSE mount. No range_read.
* **HF** — ``load_dataset(..., streaming=True)`` (library defaults).
* **Binary** — ``optimize_hf(..., compression="zstd")`` (default ``chunk_bytes="256MB"``) into
  ``--opt-root`` on lightning_storage (R2 upload), then
  ``StreamingDataset(opt_dir, max_pre_download=8)``. Nested JSON uses the Arrow
  IPC footer (including Hub ``{bytes, path}`` media as Arrow binary). ``chunk_bytes`` is
  on-disk size. Do not pass R2 ``storage_options`` into ``optimize_hf``.

Token: ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``. Never printed.
Resume: writes ``--out`` after every dataset. Previous runs with a different
``rows`` or ``target_seconds`` value are not treated as done.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
from contextlib import suppress
from pathlib import Path

# (hf_id, split, optional load_dataset config, optional explicit hf:// directory)
# Significant splits only: ≥50k rows and/or ≥50–100MB parquet. No wrap-padding.
DATASETS: list[tuple[str, str, str | None, str | None]] = [
    # nested / long text
    (
        "HuggingFaceH4/ultrachat_200k",
        "train_sft",
        None,
        "hf://datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-*.parquet",
    ),
    (
        "open-thoughts/OpenThoughts-114k",
        "train",
        None,
        "hf://datasets/open-thoughts/OpenThoughts-114k/data/train-*.parquet",
    ),
    ("abisee/cnn_dailymail", "train", "3.0.0", None),
    ("teknium/OpenHermes-2.5", "train", None, None),
    ("roneneldan/TinyStories", "train", None, None),
    ("fancyzhx/amazon_polarity", "train", None, None),
    ("Yelp/yelp_review_full", "train", None, None),
    ("EdinburghNLP/xsum", "train", None, None),
    ("Anthropic/hh-rlhf", "train", None, None),
    ("JeanKaddour/minipile", "train", None, None),
    # images (Hub {bytes, path} → Arrow binary, same as parquet to_pylist)
    ("ethz/food101", "train", None, None),
    ("zh-plus/tiny-imagenet", "train", None, None),
    ("uoft-cs/cifar10", "train", None, None),
    ("uoft-cs/cifar100", "train", None, None),
    # audio (Hub {bytes, path} → Arrow binary, same as parquet)
    ("s3prl/superb", "train", "ks", None),
]

# repo, split, config → modality label for JSON / logs
MODALITY: dict[tuple[str, str, str | None], str] = {
    ("HuggingFaceH4/ultrachat_200k", "train_sft", None): "nested_text",
    ("open-thoughts/OpenThoughts-114k", "train", None): "nested_text",
    ("abisee/cnn_dailymail", "train", "3.0.0"): "long_text",
    ("teknium/OpenHermes-2.5", "train", None): "instruct_text",
    ("roneneldan/TinyStories", "train", None): "text",
    ("fancyzhx/amazon_polarity", "train", None): "text",
    ("Yelp/yelp_review_full", "train", None): "text",
    ("EdinburghNLP/xsum", "train", None): "text",
    ("Anthropic/hh-rlhf", "train", None): "dialogue_text",
    ("JeanKaddour/minipile", "train", None): "text",
    ("ethz/food101", "train", None): "image",
    ("zh-plus/tiny-imagenet", "train", None): "image",
    ("uoft-cs/cifar10", "train", None): "image",
    ("uoft-cs/cifar100", "train", None): "image",
    ("s3prl/superb", "train", "ks"): "audio",
}


def consume(it, limit: int) -> tuple[int, float, float]:
    ttfb = None
    n = 0
    t0 = time.perf_counter()
    for _ in it:
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        n += 1
        if n >= limit:
            break
    return n, ttfb or 0.0, time.perf_counter() - t0


def target_rows_for_seconds(n: int, elapsed: float, length: int, target_seconds: float) -> int:
    """Raise a first-pass row cap toward ``target_seconds``, never past ``length``.

    No wrap / second epoch. ``target_seconds <= 0``, ``n == 0``, or a first pass
    that already lasted long enough returns ``n``.
    """
    if target_seconds <= 0 or n <= 0 or elapsed <= 0 or length <= 0:
        return n
    if elapsed >= target_seconds:
        return n
    return min(length, max(n, math.ceil((n / elapsed) * target_seconds)))


def _arm_metrics(
    n: int, ttfb: float, elapsed: float, first_n: int, first_ttfb: float, first_elapsed: float, target: int
) -> dict:
    rps = n / elapsed if elapsed else 0.0
    first_rps = first_n / first_elapsed if first_elapsed else 0.0
    return {
        "n": n,
        "ttfb": ttfb,
        "elapsed": elapsed,
        "rps": rps,
        "first_n": first_n,
        "first_ttfb": first_ttfb,
        "first_elapsed": first_elapsed,
        "first_rps": first_rps,
        "target": target,
    }


def resolve_hf_url(repo: str, split: str, config: str | None, explicit: str | None, max_files: int) -> str | None:
    from litdata.utilities.hf_fs import get_hf_filesystem

    fs = get_hf_filesystem()
    configs = [c for c in (config, "default", "plain_text") if c]
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for cfg in configs:
        candidates.append(f"hf://datasets/{repo}@refs/convert/parquet/{cfg}/{split}")
    candidates.extend(
        [
            f"hf://datasets/{repo}/data/{split}-*.parquet",
            f"hf://datasets/{repo}/data/{split}*.parquet",
            f"hf://datasets/{repo}/{split}-*.parquet",
        ]
    )
    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        try:
            files = _list_for_url(fs, url)
        except Exception:
            continue
        if files and len(files) <= max_files:
            return url
    return None


def _list_for_url(fs, url: str) -> list:
    from fnmatch import fnmatch

    from litdata.utilities.hf_fs import list_hf_parquet_files

    base = os.path.basename(url)
    if base.endswith(".parquet") and any(ch in base for ch in "*?["):
        parent = url.rsplit("/", 1)[0]
        files = list_hf_parquet_files(fs, parent)
        return [f for f in files if fnmatch(os.path.basename(f.get("name", "")), base)]
    return list_hf_parquet_files(fs, url)


def _storage_options_for_opt(opt_dir: str) -> tuple[object, dict]:
    """Resolve lightning_storage to R2 and require ``data_connection_id`` before optimize."""
    from litdata.processing.utilities import construct_storage_options
    from litdata.streaming.resolver import _resolve_dir

    dest = _resolve_dir(opt_dir)
    storage_options: dict = {}
    if dest.url:
        if not dest.data_connection_id:
            raise RuntimeError(
                f"Resolved {opt_dir} to {dest.url} without data_connection_id; cannot mint R2 credentials."
            )
        storage_options = construct_storage_options({"data_connection_id": dest.data_connection_id}, dest)
    return dest, storage_options


def _existing_opt_is_packed(existing: dict | None, need_rows: int) -> bool:
    """Reuse chunks if they already have large IPC objects or tiny whole-file zstd."""
    if not existing:
        return False
    chunks = existing.get("chunks") or []
    rows = sum(int(c.get("chunk_size") or 0) for c in chunks)
    cfg = existing.get("config") or {}
    sizes = [int(c.get("chunk_bytes") or 0) for c in chunks]
    if not sizes:
        return False
    tiny = sum(sizes) < 50 * 1024 * 1024
    large = max(sizes) >= 50 * 1024 * 1024
    # Text keeps IPC zstd. JPEG/WAV skip it; large chunks are still packed.
    if cfg.get("ipc_compression") != "zstd":
        if tiny and cfg.get("compression") == "zstd" and rows > 0:
            return True
        return bool(large and rows >= need_rows)
    if not (tiny or large):
        return False
    return tiny or rows >= need_rows


def optimize_to_storage(
    url: str,
    opt_dir: str,
    cache_root: str,
    need_rows: int,
    overwrite: bool,
    slug: str,
) -> tuple[float, int]:
    """``optimize_hf`` into lightning_storage; workers upload via R2."""
    from litdata import optimize_hf
    from litdata.processing.utilities import read_index_file_content

    dest, storage_options = _storage_options_for_opt(opt_dir)
    if not overwrite:
        existing = read_index_file_content(dest, storage_options)
        if _existing_opt_is_packed(existing, need_rows):
            return 0.0, len((existing or {}).get("chunks") or [])

    # Isolated chunk cache per dataset. Hub auth is HF_TOKEN; do not pass R2
    # storage_options into optimize_hf (HFDownloader forwards them to hf_hub_download).
    opt_cache = f"/tmp/litdata-hf-opt-cache/{slug}"
    os.makedirs(opt_cache, exist_ok=True)
    prev_cache = os.environ.get("DATA_OPTIMIZER_CACHE_FOLDER")
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = opt_cache
    t0 = time.perf_counter()
    try:
        optimize_hf(
            url,
            output_dir=opt_dir,
            compression="zstd",
            overwrite=True,
            cache_dir=cache_root,
            num_workers=min(os.cpu_count() or 4, 4),
        )
    finally:
        if prev_cache is None:
            os.environ.pop("DATA_OPTIMIZER_CACHE_FOLDER", None)
        else:
            os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = prev_cache
    idx = read_index_file_content(dest, storage_options)
    return time.perf_counter() - t0, len((idx or {}).get("chunks") or [])


def _wipe_cached_payloads(cache_dir: str | None) -> None:
    """Drop downloaded objects; keep ``index.json`` so we do not re-scan footers."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    for root, _, files in os.walk(cache_dir):
        for name in files:
            if name == "index.json":
                continue
            with suppress(FileNotFoundError, PermissionError, IsADirectoryError):
                os.remove(os.path.join(root, name))


def _stop_streaming_dataset(ds: object) -> None:
    """Stop prefetch before deleting cache files (else ``os.replace`` races the wipe)."""
    cache = getattr(ds, "cache", None)
    reader = getattr(cache, "_reader", None)
    thread = getattr(reader, "_prepare_thread", None)
    if thread is None:
        return
    with suppress(Exception):
        thread.force_stop()
    with suppress(Exception):
        thread.join(timeout=120)
    if reader is not None:
        reader._prepare_thread = None


def _stream_litdata(
    input_dir: str,
    cache_dir: str | None,
    warmup: int,
    rows: int,
    max_pre_download: int | None = None,
) -> tuple[int, int, float, float]:
    from litdata import StreamingDataset

    kwargs: dict = {"shuffle": False}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if max_pre_download is not None:
        kwargs["max_pre_download"] = max_pre_download
    ds = StreamingDataset(input_dir, **kwargs)
    length = len(ds)
    _stop_streaming_dataset(ds)
    limit = min(rows, length)
    warm = StreamingDataset(input_dir, **kwargs)
    try:
        consume(warm, min(warmup, limit))
    finally:
        _stop_streaming_dataset(warm)
    # Warmup would otherwise leave Hub/R2 objects on disk for the timed pass.
    _wipe_cached_payloads(cache_dir)
    timed = StreamingDataset(input_dir, **kwargs)
    try:
        n, ttfb, elapsed = consume(timed, limit)
    finally:
        _stop_streaming_dataset(timed)
    return length, n, ttfb, elapsed


def _stream_hf(repo: str, split: str, config: str | None, warmup: int, rows: int) -> tuple[int, float, float]:
    from datasets import load_dataset

    kwargs: dict = {"split": split, "streaming": True}
    load_args = (repo, config) if config else (repo,)
    consume(iter(load_dataset(*load_args, **kwargs)), min(warmup, rows))
    return consume(iter(load_dataset(*load_args, **kwargs)), rows)


def _extend_litdata(
    input_dir: str,
    cache_dir: str | None,
    warmup: int,
    first_n: int,
    first_ttfb: float,
    first_elapsed: float,
    length: int,
    target_seconds: float,
    max_pre_download: int | None = None,
) -> dict:
    target = target_rows_for_seconds(first_n, first_elapsed, length, target_seconds)
    if target <= first_n:
        return _arm_metrics(first_n, first_ttfb, first_elapsed, first_n, first_ttfb, first_elapsed, target)
    _len, n, ttfb, elapsed = _stream_litdata(input_dir, cache_dir, warmup, target, max_pre_download)
    return _arm_metrics(n, ttfb, elapsed, first_n, first_ttfb, first_elapsed, target)


def _extend_hf(
    repo: str,
    split: str,
    config: str | None,
    warmup: int,
    first_n: int,
    first_ttfb: float,
    first_elapsed: float,
    length: int,
    target_seconds: float,
) -> dict:
    target = target_rows_for_seconds(first_n, first_elapsed, length, target_seconds)
    if target <= first_n:
        return _arm_metrics(first_n, first_ttfb, first_elapsed, first_n, first_ttfb, first_elapsed, target)
    n, ttfb, elapsed = _stream_hf(repo, split, config, warmup, target)
    return _arm_metrics(n, ttfb, elapsed, first_n, first_ttfb, first_elapsed, target)


def _store_arm(rec: dict, prefix: str, arm: dict) -> None:
    rec[f"{prefix}_rows"] = arm["n"]
    rec[f"{prefix}_rows_per_s"] = arm["rps"]
    rec[f"{prefix}_ttfb_s"] = arm["ttfb"]
    rec[f"{prefix}_first_pass_rows"] = arm["first_n"]
    rec[f"{prefix}_first_pass_rows_per_s"] = arm["first_rps"]
    rec[f"{prefix}_first_pass_ttfb_s"] = arm["first_ttfb"]
    rec[f"{prefix}_target_rows"] = arm["target"]


def bench_one(
    repo: str,
    split: str,
    config: str | None,
    url: str,
    cache_root: str,
    opt_dir: str,
    bin_cache: str,
    slug: str,
    rows: int,
    warmup: int,
    overwrite_opt: bool,
    target_seconds: float,
) -> dict:
    rec: dict = {
        "repo": repo,
        "split": split,
        "config": config,
        "url": url,
        "opt_dir": opt_dir,
        "modality": MODALITY.get((repo, split, config)),
        "error": None,
    }

    # 1) Parquet from Hub (hf:// download), before optimize fills persist.
    pq_cache = str(Path("/tmp/litdata-hf-pq-cache") / slug)
    if os.path.isdir(pq_cache):
        shutil.rmtree(pq_cache)
    os.makedirs(pq_cache, exist_ok=True)
    t0 = time.perf_counter()
    pq_len, n_pq, ttfb_pq, elapsed_pq = _stream_litdata(url, pq_cache, warmup, rows)
    index_s = time.perf_counter() - t0 - elapsed_pq
    pq = _extend_litdata(url, pq_cache, warmup, n_pq, ttfb_pq, elapsed_pq, pq_len, target_seconds)
    rec.update({"length": pq_len, "index_s": index_s})
    _store_arm(rec, "parquet", pq)
    pq_rps = pq["rps"]

    # 2) HF datasets streaming (same Hub split). Same row cap / length as parquet.
    n_hf, ttfb_hf, elapsed_hf = _stream_hf(repo, split, config, warmup, min(rows, n_pq or rows))
    hf = _extend_hf(repo, split, config, warmup, n_hf, ttfb_hf, elapsed_hf, pq_len, target_seconds)
    _store_arm(rec, "hf", hf)
    rec["pq_vs_hf"] = (pq_rps / hf["rps"]) if hf["rps"] else None

    # 3) Optimize onto lightning_storage, then stream binary from R2 (fresh cache).
    try:
        need_rows = max(rows, pq["target"], n_pq or 0) + warmup
        opt_s, opt_files = optimize_to_storage(url, opt_dir, cache_root, need_rows, overwrite=overwrite_opt, slug=slug)
        if os.path.isdir(bin_cache):
            shutil.rmtree(bin_cache)
        os.makedirs(bin_cache, exist_ok=True)
        bin_len, n_bin, ttfb_bin, elapsed_bin = _stream_litdata(
            opt_dir, bin_cache, warmup, min(rows, n_pq or rows), max_pre_download=8
        )
        binary = _extend_litdata(
            opt_dir, bin_cache, warmup, n_bin, ttfb_bin, elapsed_bin, bin_len, target_seconds, max_pre_download=8
        )
        _store_arm(rec, "binary", binary)
        rec.update(
            {
                "optimize_s": opt_s,
                "optimize_files": opt_files,
                "binary_length": bin_len,
                "bin_vs_pq": (binary["rps"] / pq_rps) if pq_rps else None,
                "bin_vs_hf": (binary["rps"] / hf["rps"]) if hf["rps"] else None,
                "binary_fastest": bool(binary["rps"] >= pq_rps >= hf["rps"]) if hf["rps"] else None,
            }
        )
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        print(rec["traceback"], flush=True)

    return rec


def _done_key(rec: dict) -> bool:
    return (
        rec.get("error") is None
        and "binary_rows_per_s" in rec
        and "parquet_rows_per_s" in rec
        and "hf_rows_per_s" in rec
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="/tmp/litdata-hf-triad-cache")
    parser.add_argument(
        "--opt-root",
        default="/teamspace/lightning_storage/testing/litdata_hf_opt_v2",
        help="Optimized chunks (lightning_storage → R2, not FUSE reads).",
    )
    parser.add_argument("--out", default="/tmp/bench_hf_triad.json")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=120.0,
        help="Raise --rows toward this many seconds on splits that have enough "
        "rows (clamped to len; never wraps). 0 = first pass only.",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument(
        "--limit-datasets",
        type=int,
        default=None,
        help=f"Cap how many DATASETS entries to run (default: all {len(DATASETS)}).",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=None,
        help="Only these Hub ids (repeatable). Default: the full list.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Only these splits (repeatable). Default: all splits for the selected repos.",
    )
    parser.add_argument(
        "--overwrite-opt",
        action="store_true",
        help="Rebuild binary chunks even if opt-dir already has packed IPC-zstd chunks.",
    )
    args = parser.parse_args()

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise SystemExit("Set HF_TOKEN")

    out_path = Path(args.out)
    results: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("rows") == args.rows and prev.get("target_seconds", 0) == args.target_seconds:
                results = list(prev.get("datasets") or [])
        except Exception:
            results = []
    done = {(r.get("repo"), r.get("split"), r.get("config")) for r in results if _done_key(r)}

    os.makedirs(args.cache_root, exist_ok=True)
    os.makedirs(args.opt_root, exist_ok=True)
    picked: list[tuple[str, str, str | None, str | None]] = []
    seen_keys: set[tuple] = set()
    allow = set(args.repo) if args.repo else None
    splits = set(args.split) if args.split else None
    for item in DATASETS:
        if allow is not None and item[0] not in allow:
            continue
        if splits is not None and item[1] not in splits:
            continue
        key = (item[0], item[1], item[2])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picked.append(item)
        if args.limit_datasets is not None and len(picked) >= args.limit_datasets:
            break
    if (allow or splits) and not picked:
        raise SystemExit(f"No datasets matched --repo {sorted(allow or [])} --split {sorted(splits or [])}")

    print(f"{'dataset':<48} {'bin/s':>8} {'pq/s':>8} {'hf/s':>8} {'bin/pq':>7} {'pq/hf':>7}")
    for repo, split, config, explicit in picked:
        key = (repo, split, config)
        if key in done:
            print(f"skip existing {repo} {split} {config}")
            continue
        label = f"{repo}/{split}" + (f"/{config}" if config else "")
        print(f"\n## {label}")
        rec: dict = {"repo": repo, "split": split, "config": config, "error": None}
        try:
            url = resolve_hf_url(repo, split, config, explicit, args.max_files)
            if url is None:
                rec["error"] = "no parquet (or too many files)"
                print(f"  skip {rec['error']}")
            else:
                print(f"  url={url}")
                opt_dir = str(Path(args.opt_root) / repo.replace("/", "--") / f"{split}-{config or 'default'}")
                slug = f"{repo.replace('/', '--')}-{split}-{config or 'default'}"
                rec = bench_one(
                    repo,
                    split,
                    config,
                    url,
                    args.cache_root,
                    opt_dir,
                    str(Path("/tmp/litdata-hf-bin-cache") / slug),
                    slug,
                    args.rows,
                    args.warmup,
                    args.overwrite_opt,
                    args.target_seconds,
                )
                if rec.get("error"):
                    print(f"  FAIL {rec['error']}")
                else:
                    print(
                        f"{label:<48} {rec['binary_rows_per_s']:8.0f} {rec['parquet_rows_per_s']:8.0f} "
                        f"{rec['hf_rows_per_s']:8.0f} {rec['bin_vs_pq']:7.2f} {rec['pq_vs_hf']:7.2f}"
                    )
        except Exception as exc:
            rec = {
                "repo": repo,
                "split": split,
                "config": config,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(rec["traceback"], flush=True)
            print(f"  FAIL {rec['error']}")

        results = [r for r in results if (r.get("repo"), r.get("split"), r.get("config")) != key]
        results.append(rec)
        ranked = [r for r in results if r.get("binary_fastest") is not None]
        payload = {
            "rows": args.rows,
            "target_seconds": args.target_seconds,
            "warmup": args.warmup,
            "cache_root": args.cache_root,
            "opt_root": args.opt_root,
            "n": len(results),
            "binary_fastest": sum(1 for r in ranked if r.get("binary_fastest")),
            "not_ordered": sum(1 for r in ranked if r.get("binary_fastest") is False),
            "errors": sum(1 for r in results if r.get("error")),
            "datasets": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  saved {out_path}")

    print(
        f"\ndone n={len(results)} binary>pq>hf={sum(1 for r in results if r.get('binary_fastest'))} "
        f"errors={sum(1 for r in results if r.get('error'))}"
    )


if __name__ == "__main__":
    main()

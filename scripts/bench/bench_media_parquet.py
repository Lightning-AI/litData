#!/usr/bin/env python3
"""Bench parquet + video/mesh/pdf on Studio lightning_storage.

Uses public Hugging Face datasets. Set HF_TOKEN in the environment for gated
repos (never pass the token on the command line or commit it).

  HF_TOKEN=... python scripts/bench/bench_media_parquet.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os.environ.get("LITDATA_BENCH_SRC", REPO_ROOT / "src")))

from litdata import optimize  # noqa: E402
from litdata.processing.readers import ParquetReader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.item_loader import ParquetLoader  # noqa: E402
from litdata.streaming.serializers import MeshSerializer, PDFSerializer, VideoSerializer  # noqa: E402
from litdata.streaming.writer import index_parquet_dataset  # noqa: E402

_STUDIO_ROOT = Path("/teamspace/lightning_storage/testing/litdata_media_parquet_bench")

# Public parquet (IMDB train shard).
_IMDB_REPO = "stanfordnlp/imdb"
_IMDB_FILE = "plain_text/train-00000-of-00001.parquet"

# Short public MP4 used by torchaudio tutorials.
_VIDEO_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/mptestsrc.mp4"

# Tiny public PDF from HF datasets-server fixtures if available; fallback written locally.
_PDF_REPO = "hf-internal-testing/document-question-answering"
_PDF_FILE = None  # resolved at runtime if present


def _hf_download(repo_id: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or True,
    )
    if dest.exists():
        dest.unlink()
    shutil.copy2(cached, dest)
    return dest


def _download_url(url: str, dest: Path) -> Path:
    from urllib.request import urlretrieve

    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)  # noqa: S310
    return dest


def _legacy_remap(src: str, cache_dir: Path, num_rows: int) -> list[str]:
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src, memory_map=True)
    items = []
    name = os.path.basename(src)
    for start in range(0, table.num_rows, num_rows):
        end = min(start + num_rows, table.num_rows)
        out = cache_dir / f"{start}_{end}_{name}"
        pq.write_table(table[start:end], out)
        items.append(str(out))
    return items


def _time(fn, *args, **kwargs) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - start, result


def _bench_parquet(pq_path: Path, work: Path, num_rows: int) -> dict[str, float]:
    import pyarrow.parquet as pq

    results: dict[str, float] = {}
    meta = pq.ParquetFile(str(pq_path)).metadata
    print(
        f"parquet: {pq_path.name} rows={meta.num_rows} row_groups={meta.num_row_groups} bytes={pq_path.stat().st_size}"
    )

    legacy_dir = work / "legacy_shards"
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)
    results["remap_legacy_s"], _ = _time(_legacy_remap, str(pq_path), legacy_dir, num_rows)

    new_dir = work / "new_shards"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    reader = ParquetReader(str(new_dir), num_rows=num_rows, to_pandas=False, columns=["text"])
    results["remap_rowgroup_cols_s"], shards = _time(reader.remap_items, [str(pq_path)], 1)
    print(f"  shards={len(shards)} first_cols={reader.read(shards[0]).column_names}")

    indexed = work / "indexed"
    if indexed.exists():
        shutil.rmtree(indexed)
    indexed.mkdir(parents=True)
    shutil.copy2(pq_path, indexed / pq_path.name)
    index_parquet_dataset(str(indexed))

    def _scan(columns: list[str] | None, n: int) -> int:
        ds = StreamingDataset(str(indexed), item_loader=ParquetLoader(low_memory=True, columns=columns))
        seen = 0
        for i, row in enumerate(ds):
            seen += 1
            if i + 1 >= n:
                break
            _ = row.get("text") if columns else row
        return seen

    # Stream from a local index copy. lightning_storage FUSE writes are not
    # always visible on the R2 URL yet, so ParquetLoader would 404 index.json.
    local_pq = Path(tempfile.mkdtemp(prefix="litdata-pq-index-"))
    shutil.copy2(pq_path, local_pq / pq_path.name)
    shutil.copy2(indexed / "index.json", local_pq / "index.json")

    def _scan(columns: list[str] | None, n: int) -> int:
        ds = StreamingDataset(
            str(local_pq),
            item_loader=ParquetLoader(low_memory=True, columns=columns),
            index_path=str(local_pq / "index.json"),
        )
        seen = 0
        for i, row in enumerate(ds):
            seen += 1
            if i + 1 >= n:
                break
            _ = row.get("text") if columns else row
        return seen

    n = min(2000, meta.num_rows)
    results["stream_all_cols_s"], _ = _time(_scan, None, n)
    results["stream_text_only_s"], _ = _time(_scan, ["text"], n)
    results["stream_rows"] = float(n)

    opt_dir = work / "opt_from_reader"
    if opt_dir.exists():
        shutil.rmtree(opt_dir)
    reader = ParquetReader(str(work / "opt_reader_cache"), num_rows=num_rows, to_pandas=False, columns=["text"])
    results["optimize_parquet_s"], _ = _time(
        lambda: optimize(
            fn=_parquet_shard_num_rows,
            inputs=[str(local_pq / pq_path.name)],
            output_dir=str(opt_dir),
            reader=reader,
            num_workers=2,
            chunk_size=256,
            keep_data_ordered=False,
            reorder_files=False,
        )
    )
    return results


def _identity(path: str) -> str:
    return path


def _parquet_shard_num_rows(table) -> int:
    return int(table.num_rows)


def _bench_media(kind: str, paths: list[str], output_dir: Path, workers: int) -> dict[str, float]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    t, _ = _time(
        lambda: optimize(
            fn=_identity,
            inputs=paths,
            output_dir=str(output_dir),
            num_workers=workers,
            chunk_bytes=64 << 20,
            keep_data_ordered=False,
            reorder_files=False,
        )
    )
    try:
        ds = StreamingDataset(str(output_dir))
        read_t, n = _time(lambda: sum(1 for _ in ds))
        sample = ds[0]
        sample_type = type(sample).__name__
    except Exception as exc:
        print(f"  {kind}: optimize={t:.2f}s stream failed: {exc}")
        return {"optimize_s": t, "stream_s": -1.0, "n": 0.0}
    print(f"  {kind}: optimize={t:.2f}s stream={read_t:.2f}s n={n} sample_type={sample_type}")
    return {"optimize_s": t, "stream_s": read_t, "n": float(n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studio-root", default=str(_STUDIO_ROOT))
    parser.add_argument("--num-rows", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    root = Path(args.studio_root)
    raw = Path(tempfile.mkdtemp(prefix="litdata-media-raw-"))
    staged = root / "raw"
    staged.mkdir(parents=True, exist_ok=True)

    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    print(f"studio root: {root}")

    pq_path = raw / "imdb-train.parquet"
    if not pq_path.exists():
        print(f"downloading {_IMDB_REPO}/{_IMDB_FILE}")
        _hf_download(_IMDB_REPO, _IMDB_FILE, pq_path)
    pq_stats = _bench_parquet(pq_path, root / "parquet", args.num_rows)

    media_stats: dict[str, dict[str, float]] = {}
    if not args.skip_media:
        video_path = raw / "mptestsrc.mp4"
        if not video_path.exists():
            print(f"downloading video {_VIDEO_URL}")
            _download_url(_VIDEO_URL, video_path)
        videos = [str(video_path)] * 8
        media_stats["video"] = _bench_media("video", videos, root / "opt_video", args.workers)

        stl = raw / "triangle.stl"
        stl.write_bytes(
            b"solid simple\n  facet normal 0 0 1\n    outer loop\n"
            b"      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n"
            b"    endloop\n  endfacet\nendsolid simple\n"
        )
        meshes = [str(stl)] * 16
        media_stats["mesh"] = _bench_media("mesh", meshes, root / "opt_mesh", args.workers)

        pdf = raw / "dummy.pdf"
        if not pdf.exists():
            _download_url("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", pdf)
        pdfs = [str(pdf)] * 16
        media_stats["pdf"] = _bench_media("pdf", pdfs, root / "opt_pdf", args.workers)

        # Local encode/decode microbench (no FUSE).
        vser = VideoSerializer(decode="bytes")
        start = time.perf_counter()
        blob, _ = vser.serialize(str(video_path))
        _ = vser.deserialize(blob)
        media_stats["video"]["serialize_bytes_s"] = time.perf_counter() - start

        mser = MeshSerializer(decode=True)
        start = time.perf_counter()
        blob, name = mser.serialize(str(stl))
        mser.setup(name)
        _ = mser.deserialize(blob)
        media_stats["mesh"]["serialize_decode_s"] = time.perf_counter() - start

        pser = PDFSerializer(decode=True)
        start = time.perf_counter()
        blob, _ = pser.serialize(str(pdf))
        _ = pser.deserialize(blob)
        media_stats["pdf"]["serialize_decode_s"] = time.perf_counter() - start

    print("\n=== parquet (lightning_storage) ===")
    for key, value in pq_stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    if media_stats:
        print("=== media (lightning_storage optimize) ===")
        for kind, stats in media_stats.items():
            print(f"  {kind}: " + " ".join(f"{k}={v:.3f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()

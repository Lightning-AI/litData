#!/usr/bin/env python3
"""Bench every format/ingest helper added after the HF datasets pass.

Inputs stay on local disk (FUSE writes are not always on R2). optimize outputs
go to lightning_storage.

  python scripts/bench/bench_new_formats.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os.environ.get("LITDATA_BENCH_SRC", REPO_ROOT / "src")))

from litdata import iter_webdataset_tar, list_media_folder, optimize  # noqa: E402
from litdata.processing.readers import ParquetReader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402
from litdata.streaming.item_loader import ParquetLoader  # noqa: E402
from litdata.streaming.serializers import (  # noqa: E402
    AudioSerializer,
    MeshSerializer,
    NiftiSerializer,
    PDFSerializer,
    VideoSerializer,
    _safe_decode_device,
)
from litdata.streaming.writer import index_parquet_dataset  # noqa: E402

_STUDIO = Path("/teamspace/lightning_storage/testing/litdata_new_formats_bench")
_VIDEO_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/mptestsrc.mp4"
_PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
_IMDB_REPO = "stanfordnlp/imdb"
_IMDB_FILE = "plain_text/train-00000-of-00001.parquet"
_STL = (
    b"solid simple\n  facet normal 0 0 1\n    outer loop\n"
    b"      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n"
    b"    endloop\n  endfacet\nendsolid simple\n"
)


def _time(fn, *args, **kwargs) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - start, result


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    from urllib.request import urlretrieve

    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)  # noqa: S310
    return dest


def _hf_parquet(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=_IMDB_REPO,
        filename=_IMDB_FILE,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or True,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def _identity(item):
    if isinstance(item, dict) and "path" in item:
        return item["path"], item.get("label")
    return item


def _opt(paths, out: Path, workers: int = 2) -> float:
    if out.exists():
        shutil.rmtree(out)
    t, _ = _time(
        lambda: optimize(
            fn=_identity,
            inputs=paths,
            output_dir=str(out),
            num_workers=workers,
            chunk_bytes=32 << 20,
            keep_data_ordered=False,
            reorder_files=False,
        )
    )
    return t


def _stream_n(out: Path, n: int = 8) -> tuple[float, str]:
    try:
        ds = StreamingDataset(str(out))
        t, _ = _time(lambda: [ds[i] for i in range(min(n, len(ds)))])
        sample = ds[0]
        return t, type(sample).__name__ if not isinstance(sample, tuple) else type(sample[0]).__name__
    except Exception as exc:
        return -1.0, f"fail:{exc.__class__.__name__}"


def main() -> None:
    raw = Path(tempfile.mkdtemp(prefix="litdata-new-formats-"))
    _STUDIO.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, float, str]] = []

    # --- parquet ---
    pq_path = _hf_parquet(raw / "imdb.parquet")
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(str(pq_path)).metadata
    print(f"parquet rows={meta.num_rows} groups={meta.num_row_groups} bytes={pq_path.stat().st_size}")

    def _legacy():
        table = pq.read_table(str(pq_path), memory_map=True)
        return table.slice(0, 4096)

    t, _ = _time(_legacy)
    results.append(("parquet_legacy_slice_4k", t, "table"))

    reader = ParquetReader(str(raw / "pq_cache"), num_rows=4096, to_pandas=False, columns=["text"])
    t, shards = _time(reader.remap_items, [str(pq_path)], 1)
    results.append(("parquet_rowgroup_cols_remap", t, f"shards={len(shards)}"))

    local_idx = raw / "pq_index"
    local_idx.mkdir()
    shutil.copy2(pq_path, local_idx / pq_path.name)
    index_parquet_dataset(str(local_idx))

    def _scan(columns):
        ds = StreamingDataset(str(local_idx), item_loader=ParquetLoader(low_memory=True, columns=columns))
        for i, row in enumerate(ds):
            if i >= 1999:
                break
            _ = row.get("text") if columns else row

    t, _ = _time(_scan, None)
    results.append(("parquet_stream_all_cols_2k", t, ""))
    t, _ = _time(_scan, ["text"])
    results.append(("parquet_stream_text_2k", t, ""))

    # --- serializers (local microbench) ---
    video = _download(_VIDEO_URL, raw / "clip.mp4")
    pdf = _download(_PDF_URL, raw / "dummy.pdf")
    stl = raw / "mesh.stl"
    stl.write_bytes(_STL)
    wav_serializer = AudioSerializer(decode="bytes")
    wav_bytes, _ = wav_serializer.serialize({"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000})
    wav = raw / "tone.wav"
    wav.write_bytes(wav_bytes)
    nii = raw / "vol.nii"
    nii.write_bytes(b"niftifake" * 64)

    for name, serializer, path, loops in (
        ("video_bytes", VideoSerializer(decode="bytes"), video, 20),
        ("audio_bytes", AudioSerializer(decode="bytes"), wav, 200),
        ("mesh_bytes", MeshSerializer(decode=False), stl, 200),
        ("pdf_bytes", PDFSerializer(decode=False), pdf, 200),
        ("nifti_bytes", NiftiSerializer(decode=False), nii, 200),
    ):

        def _roundtrip(serializer=serializer, src=str(path), n=loops):
            for _ in range(n):
                blob, _fmt = serializer.serialize(src)
                serializer.deserialize(blob)

        t, _ = _time(_roundtrip)
        results.append((f"{name}_x{loops}", t, f"{path.stat().st_size}B"))

    t, _ = _time(
        lambda: AudioSerializer(decode="bytes").serialize(
            {"array": np.zeros(16000, np.float32), "sampling_rate": 16000}
        )
    )
    results.append(("audio_from_array", t, "1s@16k"))

    mesh_dec = MeshSerializer(decode=True)
    blob, fmt = mesh_dec.serialize(str(stl))
    mesh_dec.setup(fmt)
    t, _ = _time(lambda: mesh_dec.deserialize(blob))
    results.append(("mesh_trimesh_decode", t, fmt or ""))

    pdf_dec = PDFSerializer(decode=True)
    t, _ = _time(lambda: pdf_dec.deserialize(pdf_dec.serialize(str(pdf))[0]))
    results.append(("pdf_pdfplumber_decode", t, ""))

    # --- folder + webdataset ---
    folder = raw / "folder"
    for label in ("cats", "dogs"):
        (folder / label).mkdir(parents=True)
        for i in range(50):
            (folder / label / f"{i}.jpg").write_bytes(b"\xff\xd8\xff" + os.urandom(64))
    t, items = _time(list_media_folder, str(folder), "image")
    results.append(("list_media_folder_100", t, f"n={len(items)}"))

    tar_path = raw / "wds.tar"
    with tarfile.open(tar_path, "w") as archive:
        for i in range(200):
            jpg = raw / f"{i:03d}.jpg"
            txt = raw / f"{i:03d}.txt"
            jpg.write_bytes(b"img" + str(i).encode())
            txt.write_bytes(b"cap" + str(i).encode())
            archive.add(jpg, arcname=jpg.name)
            archive.add(txt, arcname=txt.name)
    t, samples = _time(lambda: list(iter_webdataset_tar(str(tar_path))))
    results.append(("iter_webdataset_tar_200", t, f"n={len(samples)}"))

    # --- optimize to lightning_storage ---
    t = _opt([str(video)] * 4, _STUDIO / "opt_video", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_video", 4)
    results.append(("optimize_video_x4", t, f"stream={st:.3f}s {typ}"))

    t = _opt([str(wav)] * 16, _STUDIO / "opt_audio", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_audio", 8)
    results.append(("optimize_audio_x16", t, f"stream={st:.3f}s {typ}"))

    t = _opt([str(stl)] * 16, _STUDIO / "opt_mesh", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_mesh", 8)
    results.append(("optimize_mesh_x16", t, f"stream={st:.3f}s {typ}"))

    t = _opt([str(pdf)] * 16, _STUDIO / "opt_pdf", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_pdf", 8)
    results.append(("optimize_pdf_x16", t, f"stream={st:.3f}s {typ}"))

    t = _opt([str(nii)] * 16, _STUDIO / "opt_nifti", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_nifti", 8)
    results.append(("optimize_nifti_x16", t, f"stream={st:.3f}s {typ}"))

    t = _opt(items[:40], _STUDIO / "opt_folder", workers=2)
    st, typ = _stream_n(_STUDIO / "opt_folder", 8)
    results.append(("optimize_list_media_folder_40", t, f"stream={st:.3f}s {typ}"))

    os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = "0"
    forced = _safe_decode_device("cuda")
    del os.environ["DATA_OPTIMIZER_GLOBAL_RANK"]
    results.append(("cuda_in_worker_forced_cpu", 0.0, forced))

    print("\n=== bench_new_formats ===")
    width = max(len(name) for name, _, _ in results)
    for name, elapsed, note in results:
        print(f"  {name:<{width}}  {elapsed:7.3f}s  {note}")


if __name__ == "__main__":
    main()

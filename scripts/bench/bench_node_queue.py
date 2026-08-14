#!/usr/bin/env python3
"""Large-scale ``keep_data_ordered`` True vs False optimize bench.

Writes a heavy-tailed file set (first slice is all large files so a static
per-worker assignment overloads worker 0). Then runs ``optimize`` both ways.

Examples:
  python scripts/bench/bench_node_queue.py --files 4000 --workers 8
  python scripts/bench/bench_node_queue.py --files 800 --workers 4 --studio --nodes 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = Path(os.environ.get("LITDATA_BENCH_SRC", REPO_ROOT / "src"))
sys.path.insert(0, str(_SRC))

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
from litdata import optimize  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402

HEAVY_BYTES = 1 << 20  # 1 MiB
LIGHT_BYTES = 8 << 10  # 8 KiB
_STUDIO_ROOT = "/teamspace/lightning_storage/testing"


def _process_file(path: str) -> tuple[int, str, str]:
    with open(path, "rb") as handle:
        payload = handle.read()
    # Cost scales with file size (read + hash + reduce) so a static heavy
    # slice on worker 0 is visible in wall time.
    digest = hashlib.sha256(payload).hexdigest()
    mean = float(np.frombuffer(payload, dtype=np.uint8).mean())
    return len(payload), os.path.basename(path), f"{digest[:8]}:{mean:.3f}"


def _write_file(path: str, blob: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(blob)


def _make_inputs(input_dir: Path, n_files: int, n_workers: int) -> list[str]:
    input_dir.mkdir(parents=True, exist_ok=True)
    n_heavy = max(n_workers, n_files // n_workers)
    heavy = os.urandom(HEAVY_BYTES)
    light = os.urandom(LIGHT_BYTES)
    jobs: list[tuple[str, bytes]] = []
    for i in range(n_files):
        blob = heavy if i < n_heavy else light
        jobs.append((str(input_dir / f"{i:06d}.bin"), blob))
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as pool:
        list(pool.map(lambda job: _write_file(*job), jobs))
    return [path for path, _ in jobs]


def _verify_dataset(output_dir: str, n_files: int) -> int:
    dataset = StreamingDataset(output_dir)
    names = [dataset[i][1] for i in range(len(dataset))]
    if len(dataset) != n_files or len(set(names)) != n_files:
        raise SystemExit(
            f"{output_dir}: expected {n_files} unique samples, got len={len(dataset)} unique={len(set(names))}"
        )
    return len(dataset)


def _optimize(inputs: list[str], input_dir: str, output_dir: str, workers: int, ordered: bool) -> float:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    t0 = time.perf_counter()
    optimize(
        fn=_process_file,
        inputs=inputs,
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_bytes="64MB",
        num_workers=workers,
        num_downloaders=2,
        keep_data_ordered=ordered,
        reorder_files=False,
        mode="overwrite",
        verbose=os.environ.get("LITDATA_BENCH_VERBOSE", "0") == "1",
    )
    return time.perf_counter() - t0


def _run_nodes(inputs: list[str], input_dir: str, output_dir: str, workers: int, nodes: int, cache_root: Path) -> float:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest = cache_root / "paths.txt"
    manifest.write_text("\n".join(inputs) + "\n")
    worker_py = REPO_ROOT / "tests" / "processing" / "node_queue_multinode_worker.py"
    t0 = time.perf_counter()
    procs = []
    for rank in range(nodes):
        env = os.environ.copy()
        env["DATA_OPTIMIZER_NUM_NODES"] = str(nodes)
        env["DATA_OPTIMIZER_NODE_RANK"] = str(rank)
        env["DATA_OPTIMIZER_CACHE_FOLDER"] = str(cache_root / f"chunks-{rank}")
        env["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = str(cache_root / f"data-{rank}")
        procs.append(
            subprocess.Popen(
                [sys.executable, str(worker_py), input_dir, output_dir, str(workers), str(manifest)],
                env=env,
                cwd=str(REPO_ROOT),
            )
        )
    codes = [proc.wait() for proc in procs]
    elapsed = time.perf_counter() - t0
    if codes != [0] * nodes:
        raise RuntimeError(f"node processes failed: {codes}")
    return elapsed


def _run_child(args: argparse.Namespace) -> None:
    os.environ.pop("DATA_OPTIMIZER_NUM_NODES", None)
    os.environ.pop("DATA_OPTIMIZER_NODE_RANK", None)
    input_dir = Path(args.input_dir)
    inputs = sorted(str(path) for path in input_dir.glob("*.bin"))
    cache_root = Path(tempfile.mkdtemp(prefix="litdata-node-queue-cache-"))
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = str(cache_root / "chunks")
    os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = str(cache_root / "data")
    output_dir = args.output_dir
    try:
        elapsed = _optimize(inputs, str(input_dir), output_dir, args.workers, ordered=args.only == "ordered")
        n = _verify_dataset(output_dir, len(inputs))
        print(f"BENCH_RESULT {elapsed:.4f} {n}", flush=True)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def _spawn_child(src: Path, input_dir: Path, output_dir: Path, workers: int, only: str) -> tuple[float, int]:
    env = os.environ.copy()
    env["LITDATA_BENCH_SRC"] = str(src)
    env.pop("DATA_OPTIMIZER_NUM_NODES", None)
    env.pop("DATA_OPTIMIZER_NODE_RANK", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(workers),
            "--only",
            only,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"child failed ({proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    sys.stderr.write(proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT "):
            _, elapsed, n = line.split()
            return float(elapsed), int(n)
        print(line)
    raise RuntimeError(f"no BENCH_RESULT in child stdout:\n{proc.stdout}\n{proc.stderr}")


def _run_before_after(args: argparse.Namespace) -> None:
    worktree = Path("/tmp/litdata-before-node-queue")
    if worktree.exists():
        subprocess.check_call(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO_ROOT)
    subprocess.check_call(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=REPO_ROOT)
    before_src = worktree / "src"
    after_src = REPO_ROOT / "src"

    root = Path(tempfile.mkdtemp(prefix="litdata-node-queue-ba-"))
    input_dir = root / "input"
    n_heavy = max(args.workers, args.files // args.workers)
    print(
        f"before=git HEAD ({subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT, text=True).strip()})"
    )
    print(f"after =working tree  files={args.files} workers={args.workers}  first {n_heavy} files are {HEAVY_BYTES // 1024}KiB")
    t_gen = time.perf_counter()
    _make_inputs(input_dir, args.files, args.workers)
    print(f"generated in {time.perf_counter() - t_gen:.1f}s")

    rows: list[tuple[str, str, float, int]] = []
    try:
        for tree_name, src in (("before", before_src), ("after", after_src)):
            for only, label in (("ordered", "keep_data_ordered=True"), ("shared", "keep_data_ordered=False")):
                out = root / f"{tree_name}-{only}"
                print(f"\n=== {tree_name} / {label} ===", flush=True)
                elapsed, n = _spawn_child(src, input_dir, out, args.workers, only)
                rows.append((tree_name, label, elapsed, n))
                print(f"{tree_name:6s} {label:28s} {elapsed:7.2f}s  {args.files / elapsed:7.1f} files/s  n={n}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        subprocess.check_call(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO_ROOT)

    print("\n## Before vs after (same files, same machine)\n")
    print("| Tree | Mode | Time | Throughput | Samples |")
    print("|---|---|---:|---:|---:|")
    for tree_name, label, elapsed, n in rows:
        print(f"| {tree_name} | `{label}` | {elapsed:.2f}s | {args.files / elapsed:.1f} files/s | {n} |")
    by_key = {(t, m): e for t, m, e, _ in rows}
    if ("before", "keep_data_ordered=False") in by_key and ("after", "keep_data_ordered=False") in by_key:
        b, a = by_key[("before", "keep_data_ordered=False")], by_key[("after", "keep_data_ordered=False")]
        print(f"\nShared-queue speedup (before / after): **{b / a:.2f}x**")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Write inputs/outputs under /teamspace/lightning_storage/testing",
    )
    parser.add_argument("--skip-ordered", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--only", choices=("ordered", "shared"), default=None)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the same workload on git HEAD (before) vs this tree (after).",
    )
    args = parser.parse_args()

    if args.child:
        _run_child(args)
        return
    if args.compare:
        _run_before_after(args)
        return

    if args.studio and not os.path.isdir(_STUDIO_ROOT):
        raise SystemExit(f"missing {_STUDIO_ROOT}")
    os.environ.pop("DATA_OPTIMIZER_NUM_NODES", None)
    os.environ.pop("DATA_OPTIMIZER_NODE_RANK", None)

    run_id = uuid.uuid4().hex[:8]
    if args.studio:
        root = Path(_STUDIO_ROOT) / "litdata_node_queue_bench" / run_id
    else:
        root = Path(tempfile.mkdtemp(prefix=f"litdata-node-queue-{run_id}-"))

    input_dir = root / "input"
    cache_root = Path(tempfile.mkdtemp(prefix="litdata-node-queue-cache-"))
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = str(cache_root / "chunks")
    os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = str(cache_root / "data")

    print(f"generating {args.files} files under {input_dir} (first {max(args.workers, args.files // args.workers)} are {HEAVY_BYTES // 1024}KiB)")
    t_gen = time.perf_counter()
    inputs = _make_inputs(input_dir, args.files, args.workers)
    print(f"generated in {time.perf_counter() - t_gen:.1f}s")

    results: list[tuple[str, float, int]] = []
    try:
        if args.nodes > 1:
            out = str(root / "shared-nodes")
            elapsed = _run_nodes(inputs, str(input_dir), out, args.workers, args.nodes, cache_root)
            n = _verify_dataset(out, args.files)
            results.append((f"shared queue, {args.nodes} nodes x {args.workers} workers", elapsed, n))
        else:
            if not args.skip_ordered:
                out_o = str(root / "ordered")
                t_o = _optimize(inputs, str(input_dir), out_o, args.workers, ordered=True)
                results.append(("keep_data_ordered=True", t_o, _verify_dataset(out_o, args.files)))
            out_s = str(root / "shared")
            t_s = _optimize(inputs, str(input_dir), out_s, args.workers, ordered=False)
            results.append(("keep_data_ordered=False", t_s, _verify_dataset(out_s, args.files)))
    finally:
        if args.studio:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cache_root, ignore_errors=True)

    print()
    print(f"files={args.files} workers={args.workers} nodes={args.nodes} studio={args.studio}")
    for label, elapsed, n in results:
        if n != args.files:
            raise SystemExit(f"{label}: expected {args.files} samples, got {n}")
        print(f"{label:42s}  {elapsed:7.2f}s  {args.files / elapsed:7.1f} files/s  n={n}")
    if len(results) == 2:
        print(f"speedup  {results[0][1] / results[1][1]:.2f}x  (ordered / shared)")


if __name__ == "__main__":
    main()

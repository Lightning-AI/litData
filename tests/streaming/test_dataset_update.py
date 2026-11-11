# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0.

import json
import os

import polars as pl
import pytest

from litdata import StreamingDataset, build_keys_index, dataset_update, optimize
from litdata.constants import _INDEX_FILENAME, _KEYS_DIRNAME
from litdata.utilities.keys_index import (
    KeyIndex,
    enrich_keys_with_chunks,
    iter_key_indexes,
    keys_dir,
    save_keys,
    save_rank_keys,
    shard_path,
    write_keys_store,
)


@pytest.fixture(autouse=True)
def _isolate_optimizer_cache(tmpdir, monkeypatch):
    """Keep optimize() scratch dirs unique per test (xdist-safe).

    Without this, workers share ``/tmp/chunks`` and leftover ``*.bin`` /
    ``*-index.json`` files from other tests fail ``_done`` cleanup checks and
    can merge inconsistent configs into the dataset under test.
    """
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", os.path.join(str(tmpdir), "opt_chunks"))
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", os.path.join(str(tmpdir), "opt_data"))


def _fn(i: int) -> dict:
    return {"id": f"item-{i}", "value": i}


def _key_fn(sample: dict) -> str:
    return sample["id"]


def _int_fn(i: int) -> dict:
    return {"id": i, "value": i * 10}


def _int_key_fn(sample: dict) -> int:
    return sample["id"]


def test_keys_parquet_int64_roundtrip(tmpdir):
    path = os.path.join(tmpdir, "keys.parquet")
    keys = list(range(1000))
    save_keys(path, keys)
    with KeyIndex(path) as idx:
        assert len(idx) == 1000
        assert idx[0] == 0
        assert idx[999] == 999
        assert 42 in idx
        assert idx.key_at(42) == 42
        assert idx.get(-1) is None
        gidx, chunk_i, chunk_off = idx.resolve(42)
        assert gidx == 42
        assert chunk_i == -1  # not enriched yet


def test_keys_parquet_utf8_roundtrip(tmpdir):
    path = os.path.join(tmpdir, "keys.parquet")
    keys = [f"id-{i}" for i in range(500)]
    save_keys(path, keys)
    df = pl.read_parquet(path)
    assert df.columns == ["key", "index"]
    with KeyIndex(path) as idx:
        assert len(idx) == 500
        assert idx["id-7"] == 7
        assert idx.key_at(7) == "id-7"


def test_write_keys_store_default_shard_layout(tmpdir):
    # Minimal index.json so keys config is recorded there (not a second manifest).
    with open(os.path.join(tmpdir, _INDEX_FILENAME), "w", encoding="utf-8") as f:
        json.dump({"chunks": [], "config": {}}, f)

    out = write_keys_store(str(tmpdir), ["a", "b", "c"], indices=[0, 1, 2])
    assert out == keys_dir(str(tmpdir))
    assert os.path.isfile(shard_path(str(tmpdir), 0))
    assert not os.path.isfile(os.path.join(out, "manifest.json"))
    with open(os.path.join(tmpdir, _INDEX_FILENAME), encoding="utf-8") as f:
        index = json.load(f)
    assert index["keys"]["num_shards"] == 1
    assert index["keys"]["sharding"] == "none"
    with KeyIndex(str(tmpdir)) as idx:
        assert idx["b"] == 1


def test_write_keys_store_multi_shard(tmpdir):
    keys = [f"k-{i}" for i in range(100)]
    write_keys_store(str(tmpdir), keys, num_shards=4)
    for shard in range(4):
        assert os.path.isfile(shard_path(str(tmpdir), shard))
    with KeyIndex(str(tmpdir)) as idx:
        assert len(idx) == 100
        assert idx["k-7"] == 7
        assert idx.resolve_many(["k-1", "k-50", "k-99"])["k-99"][0] == 99


def test_save_rank_keys_pairs_and_merge_remaps_indexes(tmpdir):
    from litdata.utilities.keys_index import merge_rank_key_files

    # Rank 0: local indexes 0..2
    save_rank_keys(
        os.path.join(tmpdir, "0.keys.parquet"),
        [(0, "a"), (1, "b"), (2, "c")],
    )
    # Rank 1: local indexes 0..1
    save_rank_keys(
        os.path.join(tmpdir, "1.keys.parquet"),
        [(0, "d"), (1, "e")],
    )
    merged = merge_rank_key_files(str(tmpdir))
    assert merged is not None
    assert merged == keys_dir(str(tmpdir))
    with KeyIndex(merged) as idx:
        assert len(idx) == 5
        assert idx["a"] == 0
        assert idx["c"] == 2
        assert idx["d"] == 3  # remapped after rank 0
        assert idx["e"] == 4


def test_merge_rank_key_files_skips_empty_utf8_when_other_rank_is_int(tmpdir):
    from litdata.utilities.keys_index import merge_rank_key_files

    save_rank_keys(os.path.join(tmpdir, "0.keys.parquet"), [])
    save_rank_keys(os.path.join(tmpdir, "1.keys.parquet"), [(0, 3), (1, 7)])
    merged = merge_rank_key_files(str(tmpdir))
    assert merged is not None
    with KeyIndex(merged) as idx:
        assert len(idx) == 2
        assert idx[3] == 0
        assert idx[7] == 1


def test_merge_rank_key_files_multi_node_then_concatenate(tmpdir):
    """Simulate last-node merge: each node writes ``{rank}-keys.parquet``, then concatenate."""
    from litdata.utilities.keys_index import concatenate_key_files, merge_rank_key_files

    node0 = os.path.join(tmpdir, "n0")
    node1 = os.path.join(tmpdir, "n1")
    os.makedirs(node0)
    os.makedirs(node1)
    save_rank_keys(os.path.join(node0, "0.keys.parquet"), [(0, "a"), (1, "b")])
    save_rank_keys(os.path.join(node0, "1.keys.parquet"), [(0, "c")])
    save_rank_keys(os.path.join(node1, "0.keys.parquet"), [(0, "d")])
    save_rank_keys(os.path.join(node1, "1.keys.parquet"), [(0, "e"), (1, "f")])

    p0 = merge_rank_key_files(node0, output_filename="0-keys.parquet")
    p1 = merge_rank_key_files(node1, output_filename="1-keys.parquet")
    assert p0
    assert p1
    out = os.path.join(tmpdir, "merged")
    os.makedirs(out)
    concatenate_key_files([p0, p1], out)
    with KeyIndex(out) as idx:
        assert len(idx) == 6
        assert idx["a"] == 0
        assert idx["c"] == 2
        assert idx["d"] == 3
        assert idx["f"] == 5


def test_enrich_keys_with_chunks(tmpdir):
    write_keys_store(str(tmpdir), ["x", "y", "z"], indices=[0, 1, 2])
    index_json = {
        "chunks": [
            {"filename": "chunk-0-0.bin", "chunk_bytes": 10, "chunk_size": 2},
            {"filename": "chunk-0-1.bin", "chunk_bytes": 10, "chunk_size": 1},
        ]
    }
    enrich_keys_with_chunks(str(tmpdir), index_json)
    df = pl.read_parquet(shard_path(str(tmpdir), 0))
    assert "chunk_index" in df.columns
    assert "chunk_offset" in df.columns
    with KeyIndex(str(tmpdir)) as idx:
        g0, c0, o0 = idx.resolve("x")
        g2, c2, o2 = idx.resolve("z")
        assert (g0, c0) == (0, 0)
        assert (g2, c2) == (2, 1)


def test_optimize_writes_keys_store(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(10)),
        output_dir=out,
        chunk_size=4,
        num_workers=2,
        reorder_files=False,
        key_fn=_key_fn,
    )

    keys_file = shard_path(out, 0)
    assert os.path.isfile(keys_file)
    assert os.path.isdir(os.path.join(out, _KEYS_DIRNAME))
    with open(os.path.join(out, _INDEX_FILENAME), encoding="utf-8") as f:
        index = json.load(f)
    assert index["keys"]["num_shards"] == 1
    df = pl.read_parquet(keys_file)
    assert "key" in df.columns
    assert "index" in df.columns
    assert "chunk_index" in df.columns
    assert df.height == 10

    ds = StreamingDataset(out, shuffle=False)
    assert len(ds) == 10
    sample = ds["item-3"]
    assert sample["id"] == "item-3"
    assert sample["value"] == 3


def test_streaming_dataset_key_access_matches_int_index(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(8)),
        output_dir=out,
        chunk_size=3,
        num_workers=2,
        reorder_files=False,
        key_fn=_key_fn,
    )
    ds = StreamingDataset(out, shuffle=False)
    with KeyIndex(out) as index:
        for key in ("item-0", "item-4", "item-7"):
            assert ds[key] == ds[index[key]]


def test_dataset_update_replaces_sample_by_key(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(12)),
        output_dir=out,
        chunk_size=5,
        num_workers=2,
        reorder_files=False,
        key_fn=_key_fn,
    )

    with dataset_update(out) as update:
        update["item-3"] = {"id": "item-3", "value": 999}
        update["item-11"] = {"id": "item-11", "value": -1}
        update.commit()
        with pytest.raises(RuntimeError, match="after commit"):
            update["item-0"] = {"id": "item-0", "value": 0}

    ds = StreamingDataset(out, shuffle=False)
    assert ds["item-3"]["value"] == 999
    assert ds["item-11"]["value"] == -1
    assert ds["item-0"]["value"] == 0


def test_dataset_update_without_commit_discards_changes(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(4)),
        output_dir=out,
        chunk_size=2,
        num_workers=1,
        key_fn=_key_fn,
    )
    with dataset_update(out) as update:
        update["item-1"] = {"id": "item-1", "value": 999}
        # no commit()
    assert StreamingDataset(out, shuffle=False)["item-1"]["value"] == 1


def test_dataset_update_int_keys(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_int_fn,
        inputs=list(range(8)),
        output_dir=out,
        chunk_size=3,
        num_workers=2,
        reorder_files=False,
        key_fn=_int_key_fn,
    )
    with dataset_update(out) as update:
        update[3] = {"id": 3, "value": 123}
        update.commit()
    ds = StreamingDataset(out, shuffle=False)
    # Int entity keys use get_by_key (ds[3] remains a global sample index).
    assert ds.get_by_key(3)["value"] == 123


def test_dataset_update_unknown_key_raises(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(4)),
        output_dir=out,
        chunk_size=2,
        num_workers=1,
        key_fn=_key_fn,
    )
    with dataset_update(out) as update:
        update["missing"] = {"id": "missing", "value": 0}
        with pytest.raises(KeyError, match="Unknown dataset key"):
            update.commit()


def test_dataset_update_requires_keys_file(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(4)),
        output_dir=out,
        chunk_size=2,
        num_workers=1,
    )
    with pytest.raises(FileNotFoundError, match="keys/"):
        dataset_update(out)


def test_build_keys_index_backfills_sidecar(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(9)),
        output_dir=out,
        chunk_size=4,
        num_workers=2,
        reorder_files=False,
        keep_data_ordered=True,
    )
    assert not os.path.isfile(shard_path(out, 0))

    pairs = list(iter_key_indexes(out, _key_fn, verbose=False))
    assert pairs[0] == ("item-0", 0)
    assert pairs[-1] == ("item-8", 8)

    path = build_keys_index(out, _key_fn, verbose=False)
    assert path == keys_dir(out)
    df = pl.read_parquet(shard_path(out, 0))
    assert df.height == 9
    assert "chunk_index" in df.columns

    ds = StreamingDataset(out, shuffle=False)
    assert ds["item-5"]["value"] == 5
    # Release chunk mmaps before in-place rewrite (required on Windows).
    item_loader = getattr(getattr(ds, "cache", None), "_reader", None)
    item_loader = getattr(item_loader, "_item_loader", None) if item_loader is not None else None
    close_open = getattr(item_loader, "_close_open_chunk", None)
    if callable(close_open):
        close_open()
    del ds

    with dataset_update(out) as update:
        update["item-5"] = {"id": "item-5", "value": 50}
        update.commit()
    assert StreamingDataset(out, shuffle=False)["item-5"]["value"] == 50

    with pytest.raises(FileExistsError, match="already exists"):
        build_keys_index(out, _key_fn, verbose=False)

    build_keys_index(out, _key_fn, overwrite=True, verbose=False)


def test_duplicate_keys_rejected(tmpdir):
    path = os.path.join(tmpdir, "keys.parquet")
    with pytest.raises(ValueError, match="Duplicate key"):
        save_keys(path, ["a", "b", "a"])


def test_legacy_keys_parquet_still_readable(tmpdir):
    out = os.path.join(tmpdir, "data")
    os.makedirs(out)
    save_keys(os.path.join(out, "keys.parquet"), ["legacy-a", "legacy-b"], indices=[0, 1])
    with KeyIndex(out) as idx:
        assert idx["legacy-a"] == 0
        assert idx["legacy-b"] == 1


def test_remote_key_index_scans_without_full_download(monkeypatch, tmpdir):
    """Remote KeyIndex should scan cloud URIs with storage_options, not cache shards locally."""
    import polars as pl

    import litdata.streaming.fs_provider as fs_mod

    dataset_url = "s3://bucket/dataset"
    with open(os.path.join(tmpdir, _INDEX_FILENAME), "w", encoding="utf-8") as f:
        json.dump({"chunks": [], "config": {}, "keys": {"version": 1, "num_shards": 1, "sharding": "none"}}, f)
    write_keys_store(str(tmpdir), ["a", "b"], indices=[0, 1], num_shards=1)
    local_shard = shard_path(str(tmpdir), 0)
    local_index = os.path.join(tmpdir, _INDEX_FILENAME)

    class _FakeFS:
        def exists(self, path: str) -> bool:
            return path.endswith("index.json") or path.endswith("shard-00000.parquet")

        def download_file(self, remote_path: str, local_path: str) -> None:
            assert remote_path.endswith("index.json")
            with open(local_index, encoding="utf-8") as src, open(local_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())

    scanned: dict[str, object] = {}
    real_scan = pl.scan_parquet

    def _tracking_scan(source, *args, **kwargs):
        scanned["source"] = source
        scanned["storage_options"] = kwargs.get("storage_options")
        if isinstance(source, str) and source.startswith("s3://"):
            source = local_shard
        elif isinstance(source, list):
            source = [local_shard if str(s).startswith("s3://") else s for s in source]
        return real_scan(source, *args, **{k: v for k, v in kwargs.items() if k != "storage_options"})

    monkeypatch.setattr(fs_mod, "_get_fs_provider", lambda url, opts=None: _FakeFS())
    monkeypatch.setattr(pl, "scan_parquet", _tracking_scan)

    storage_options = {"aws_region": "us-east-1"}
    with KeyIndex(dataset_url, storage_options=storage_options) as idx:
        assert idx["a"] == 0

    source = scanned["source"]
    assert (isinstance(source, str) and source.startswith("s3://")) or (
        isinstance(source, list) and str(source[0]).startswith("s3://")
    )
    assert scanned["storage_options"] == storage_options


def test_streaming_dataset_missing_keys_raises_on_str_key(tmpdir):
    out = os.path.join(tmpdir, "data")
    optimize(
        fn=_fn,
        inputs=list(range(4)),
        output_dir=out,
        chunk_size=2,
        num_workers=1,
    )
    ds = StreamingDataset(out, shuffle=False)
    with pytest.raises(KeyError, match="keys/"):
        _ = ds["item-0"]

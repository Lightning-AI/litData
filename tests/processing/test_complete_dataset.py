import json
import os

import torch

from litdata import Graph, StreamingDataset, complete_dataset, is_complete_dataset
from litdata.constants import _INDEX_FILENAME
from litdata.streaming.cache import Cache


def test_complete_dataset_merges_worker_shards(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(20):
        cache[i] = i
    cache.done()
    cache.merge()

    index_path = os.path.join(str(tmpdir), _INDEX_FILENAME)
    with open(index_path) as f:
        payload = json.load(f)
    os.remove(index_path)
    shard = os.path.join(str(tmpdir), f"0.{_INDEX_FILENAME}")
    with open(shard, "w") as f:
        json.dump(payload, f)

    assert not is_complete_dataset(str(tmpdir))
    complete_dataset(str(tmpdir))
    assert is_complete_dataset(str(tmpdir))
    assert not os.path.exists(shard)

    ds = StreamingDataset(str(tmpdir))
    assert len(ds) == 20
    assert ds[0] == 0


def test_streaming_dataset_completes_shards_on_open(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=5)
    for i in range(10):
        cache[i] = {"x": i}
    cache.done()
    cache.merge()

    index_path = os.path.join(str(tmpdir), _INDEX_FILENAME)
    with open(index_path) as f:
        payload = json.load(f)
    os.remove(index_path)
    with open(os.path.join(str(tmpdir), f"1.{_INDEX_FILENAME}"), "w") as f:
        json.dump(payload, f)

    ds = StreamingDataset(str(tmpdir))
    assert len(ds) == 10


def test_graph_and_list_samples(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=4)
    for i in range(8):
        cache[i] = {
            "vals": [float(i), float(i) + 0.5],
            "graph": Graph(
                x=torch.ones(4, 3) * i,
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                y=torch.tensor([i]),
            ),
        }
    cache.done()
    cache.merge()

    ds = StreamingDataset(str(tmpdir))
    sample = ds[3]
    assert sample["vals"] == [3.0, 3.5]
    assert sample["graph"].x.shape == (4, 3)
    assert int(sample["graph"].y[0]) == 3

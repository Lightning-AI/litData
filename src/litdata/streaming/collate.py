# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Default collate for ``StreamingDataLoader``.

Uses PyTorch ``default_collate`` unless a sample (or a dict value) is a graph,
in which case it batches with ``Batch.from_data_list``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data._utils.collate import default_collate

from litdata.types import Graph, is_pyg_data


def _is_graph_sample(item: Any) -> bool:
    return isinstance(item, Graph) or is_pyg_data(item)


def _collate_graphs(items: list[Any]) -> Any:
    try:
        from torch_geometric.data import Batch
    except ImportError:
        return items
    graphs = [item.to_pyg() if isinstance(item, Graph) else item for item in items]
    return Batch.from_data_list(graphs)


def litdata_collate(items: list[Any]) -> Any:
    """Collate a batch: graphs via PyG ``Batch``, everything else via ``default_collate``.

    Dict samples recurse only when a value is a graph, so ``{"graph": data, "id": i}``
    still batches. Without torch-geometric, graph batches stay a list of ``Graph``.
    """
    if not items:
        return items
    elem = items[0]
    if _is_graph_sample(elem):
        return _collate_graphs(items)
    if isinstance(elem, Mapping) and any(_is_graph_sample(elem[key]) for key in elem):
        return {key: litdata_collate([item[key] for item in items]) for key in elem}
    return default_collate(items)

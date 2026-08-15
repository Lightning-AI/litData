# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collate that batches PyG ``Data`` the way ``torch_geometric.loader.DataLoader`` does."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data._utils.collate import default_collate

from litdata.types import Graph


def _is_pyg_data(item: Any) -> bool:
    module = getattr(type(item), "__module__", "")
    return module.startswith("torch_geometric.data") and hasattr(item, "to_dict")


def pyg_collate(items: list[Any]) -> Any:
    """``Batch.from_data_list`` for PyG graphs; otherwise ``default_collate``.

    Recurses into dict samples so ``{"graph": data, "id": i}`` still batches.
    """
    if not items:
        return items
    elem = items[0]
    if _is_pyg_data(elem) or isinstance(elem, Graph):
        from torch_geometric.data import Batch

        graphs = [item.to_pyg() if isinstance(item, Graph) else item for item in items]
        return Batch.from_data_list(graphs)
    if isinstance(elem, Mapping):
        return {key: pyg_collate([item[key] for item in items]) for key in elem}
    return default_collate(items)

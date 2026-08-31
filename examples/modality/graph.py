"""Graph: pack tensors (not pickle), stream PyG Data or Graph.

Pass a PyG Data / HeteroData or Graph(x=..., edge_index=..., y=...).
Do not torch.save the graph. StreamingDataLoader uses litdata_collate:
graphs become a PyG Batch when torch-geometric is installed.
"""

import torch

from litdata import Graph, StreamingDataLoader, StreamingDataset, optimize


def make_sample(index: int) -> dict:
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor(index % 3)
    return {
        "index": index,
        # Data(x=x, edge_index=edge_index, y=y)
        # Graph(data=pyg_data)
        "graph": Graph(x=x, edge_index=edge_index, y=y),
    }


if __name__ == "__main__":
    optimize(
        fn=make_sample,
        inputs=list(range(8)),
        output_dir="example_optimize_dataset/graph",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    dataset = StreamingDataset("example_optimize_dataset/graph")
    sample = dataset[0]
    graph = sample["graph"]  # PyG Data or Graph
    print(graph.x.shape, graph.edge_index.shape, graph.y)

    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(batch["graph"])

"""Stream the dataset written by optimize.py."""

from litdata import StreamingDataLoader, StreamingDataset

if __name__ == "__main__":
    dataset = StreamingDataset("example_optimize_dataset", shuffle=True)
    sample = dataset[0]

    caption = sample["caption"]
    image = sample["image"]  # Tensor CHW
    audio = sample["audio"]
    wave, rate = audio["array"], audio["sampling_rate"]
    tokens = sample["tokens"]  # Tensor (1-D is TokensLoader)
    graph = sample["graph"]  # PyG Data or Graph
    sidecar = sample["sidecar"]  # raw bytes

    print(caption, image.shape, wave.shape, rate, tokens.shape)
    print(graph.x.shape, graph.edge_index.shape, graph.y, sidecar)

    loader = StreamingDataLoader(dataset, batch_size=8, num_workers=0)
    batch = next(iter(loader))
    print(batch["image"].shape, batch["graph"])

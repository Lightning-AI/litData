"""Text as strings, or as tokens with TokensLoader.

String samples: wrap Text(path=...) so a filepath is not a caption.
Token samples: yield a 1-D Tensor of ids and pass item_loader=TokensLoader()
so chunks are one contiguous token stream (LLM pre-training).
Swap encode() for your HF / SentencePiece tokenizer.
"""

from pathlib import Path

import torch

from litdata import (
    StreamingDataLoader,
    StreamingDataset,
    Tensor,
    Text,
    TokensLoader,
    list_media_folder,
    optimize,
)


def make_text(item: dict) -> dict:
    return {
        "text": Text(path=item["path"]),
        "caption": item["label"],
    }


def encode(text: str) -> torch.Tensor:
    # Toy tokenizer: UTF-8 bytes as token ids. Replace with tokenizer.encode(text).
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.int64)


def make_tokens(item: dict) -> Tensor:
    # The sample must be a single 1-D tensor (not a dict) for TokensLoader.
    return Tensor(array=encode(Path(item["path"]).read_text(encoding="utf-8")))


def seed_folder(root: Path) -> None:
    for label, body in (
        ("news", "The market opened higher after the announcement."),
        ("sports", "The home team won in overtime."),
    ):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            (folder / f"{index}.txt").write_text(f"{body} ({index})\n", encoding="utf-8")


if __name__ == "__main__":
    media_dir = Path("example_optimize_dataset/source/text")
    seed_folder(media_dir)
    inputs = list_media_folder(str(media_dir), kind="text")

    # --- strings ---
    optimize(
        fn=make_text,
        inputs=inputs,
        output_dir="example_optimize_dataset/text",
        num_workers=2,
        chunk_bytes="64MB",
        mode="overwrite",
    )
    dataset = StreamingDataset("example_optimize_dataset/text")
    sample = dataset[0]
    text = sample["text"]  # str
    print(sample["caption"], text)
    batch = next(iter(StreamingDataLoader(dataset, batch_size=4, num_workers=0)))
    print(len(batch["text"]), batch["caption"])

    # --- tokens (TokensLoader concatenates 1-D ids; stream windows of block_size) ---
    optimize(
        fn=make_tokens,
        inputs=inputs,
        output_dir="example_optimize_dataset/text_tokens",
        num_workers=2,
        chunk_size=256,
        item_loader=TokensLoader(),
        mode="overwrite",
    )
    token_dataset = StreamingDataset(
        "example_optimize_dataset/text_tokens",
        item_loader=TokensLoader(block_size=16),
    )
    tokens = token_dataset[0]  # Tensor, length block_size
    print(tokens.shape, tokens.dtype)
    token_batch = next(iter(StreamingDataLoader(token_dataset, batch_size=4, num_workers=0)))
    print(token_batch.shape)

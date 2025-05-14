# LitData Benchmarks: Optimize & Stream Datasets

This directory provides practical scripts and guidance for benchmarking data optimization and streaming with [LitData](https://github.com/Lightning-AI/litdata). Use these tools to accelerate your AI workflows, whether you want to optimize raw datasets for fast training or stream already-optimized datasets from local or cloud storage.

---

## 🚀 Quick Start: Streaming an Optimized ImageNet Dataset

If you already have an optimized dataset (e.g., from Lightning Platform), you can stream it directly for training or benchmarking. For example, to stream the 1M ImageNet dataset:

```bash
python stream_imagenet.py \
    --input_dir /teamspace/datasets/imagenet-1m-optimized-0.2.41-v2/train/ \
    --batch_size 256 \
    --epochs 2 \
    --num_workers 32 \
    --clear_cache
```

- `--input_dir`: Path to the optimized dataset (can be local or cloud/S3 path)
- `--batch_size`: Batch size for training (default: 256)
- `--epochs`: Number of epochs to train (default: 2)
- `--num_workers`: Number of parallel workers (default: 32)
- `--clear_cache`: Clear the cache before/after benchmarking (optional)

---

## 🛠️ Optimize a Raw Dataset for Fast Streaming

To optimize a raw dataset (e.g., from S3 or local), use the `optimize_imagenet.py` script. This will convert your raw images into a format that supports fast streaming and training.

Example: Optimize the raw ImageNet-1M dataset from Lightning Platform S3 to a local output directory:

```bash
python optimize_imagenet.py \
    --input_dir /teamspace/s3_connections/imagenet-1m-template/raw/train/ \
    --output_dir /teamspace/datasets/imagenet-1m-optimized \
    --resize --resize_size 256 \
    --write_mode jpeg \
    --quality 90 \
    --num_workers 32
```

- `--input_dir`: Path to the raw dataset (local or S3)
- `--output_dir`: Where to write the optimized dataset
- `--resize`: Enable resizing (recommended for training)
- `--resize_size`: Resize the largest dimension to this value (preserving aspect ratio)
- `--write_mode`: `jpeg` (recommended) or `pil` (raw PIL format)
- `--quality`: JPEG quality (default: 90)
- `--num_workers`: Number of parallel workers (adjust for your machine)

---

## 💡 Tips & Scenarios

- **Streaming from Cloud**: Both scripts support S3 or local paths. Use S3 URIs (e.g., `s3://my-bucket/imagenet-optimized`) for cloud streaming.
- **Format Choice**: Use `jpeg` for best speed and compatibility. Use `pil` only if you need raw PIL images (slower, larger files).
- **Cache Management**: Use `--clear_cache` in `stream_imagenet.py` to clear the cache before/after benchmarking.
- **Custom Transforms**: You can edit the scripts to add your own torchvision transforms for data augmentation or preprocessing.

# LitData Benchmarks

This folder contains scripts to optimize and stream datasets using LitData.

## Optimize ImageNet

Optimize a raw ImageNet dataset for fast training and streaming:

```bash
python optimize_imagenet.py \
    --input_dir /path/to/raw/imagenet/train \
    --output_dir /path/to/optimized/imagenet \
    --resize --resize_size 256 \
    --write_mode jpeg \
    --quality 90 \
    --num_workers 32
```

- `--resize_size 256` resizes the largest image dimension to 256 (preserving aspect ratio).
- `--write_mode jpeg` stores images as JPEGs for best speed.

## Stream Optimized ImageNet

Stream an already optimized dataset for benchmarking or training:

```bash
python stream_imagenet.py \
    --input_dir /path/to/optimized/imagenet \
    --batch_size 256 \
    --epochs 2
```

- Use `--use_pil` if you optimized with raw PIL images.

---

These scripts are easy to use and work with both local and cloud datasets. For more details, see the script docstrings or run with `--help`.

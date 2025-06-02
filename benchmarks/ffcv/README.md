# FFCV Benchmarks  

This folder contains scripts to convert, write, and stream datasets using FFCV for benchmarking.

## 1. Prepare the Dataset

First, copy the raw ImageNet dataset to your machine (if not already present):

```sh
s5cmd cp "s3://imagenet-1m-template/raw/train/*" data/imagenet-1m-raw/train
```

Convert the raw ImageNet synset folders to PyTorch ImageFolder format (class index folders):

```sh
python convert_imagenet.py --data_dir data/imagenet-1m-raw/train
```

## 2. Install FFCV

Install the FFCV library (if not already installed):

```sh
sh install_ffcv.sh
```

## 3. Write FFCV Dataset

Prepare imagenet dataset to FFCV format. Example for different settings:

- (max 256px, 0% JPEG, quality 100)

```sh
python write_imagenet.py \
    --cfg.dataset=imagenet \
    --cfg.split=train \
    --cfg.data_dir=/path/to/imagenet/train \
    --cfg.write_path=/your/output/path/train_256_0.0_100.ffcv \
    --cfg.max_resolution=256 \
    --cfg.write_mode=proportion \
    --cfg.compress_probability=0.0 \
    --cfg.jpeg_quality=100
```

- (max 256px, 100% JPEG, quality 90)

```sh
python write_imagenet.py \
    --cfg.dataset=imagenet \
    --cfg.split=train \
    --cfg.data_dir=/path/to/imagenet/train \
    --cfg.write_path=/your/output/path/train_256_100.0_90.ffcv \
    --cfg.max_resolution=256 \
    --cfg.write_mode=proportion \
    --cfg.compress_probability=100.0 \
    --cfg.jpeg_quality=90
```

## 4. Stream FFCV Dataset

Stream an FFCV .ffcv dataset for benchmarking or training:

```sh
python stream_imagenet.py \
    --cfg.data_path=/path/to/train_256_0.0_100.ffcv \
    --cfg.batch_size=256 \
    --cfg.num_workers=32 \
    --cfg.epochs=2
```

---

These scripts are easy to use and work with both local and cloud datasets. For more details, see the script docstrings or run with `--help`.

## 5. Benchmark LitData vs FFCV

You can use already prepared datasets to quickly run your benchmarks. Simply copy the optimized datasets from S3 to your teamspace, then run the provided streaming or benchmarking scripts.

Example S3 structure:

```
s3://xxxxxxx/datasets/imagenet-1m-ffcv/
    train_256_0.0_100.ffcv
    train_256_100.0_90.ffcv
s3://xxxxxxx/datasets/imagenet-1m-litdata/
    train_256_jpg_90/
    train_256_raw_pil/
```

To extract the real S3 path for a dataset in your teamspace, use:

```sh
python3 -c "from litdata.streaming.resolver import _resolve_dir; path=_resolve_dir('/teamspace/datasets/imagenet-1m-litdata/'); print(path.url)"
```
You can also prepare the datasets yourself using the earlier steps if you prefer.

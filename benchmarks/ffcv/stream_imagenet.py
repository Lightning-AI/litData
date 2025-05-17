"""Stream an FFCV ImageNet dataset for benchmarking.

Adapted from: https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py

    This script streams an FFCV ImageNet dataset and benchmarks the streaming speed.
    It uses the FFCV library to load and process the dataset efficiently.

Example usage:
    python stream_imagenet.py --cfg.data_path=/path/to/train_256_0.0_100.ffcv
"""

import os
import time

import lightning as L
import numpy as np
import torch
import torchvision.transforms.v2 as T
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param, section
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import NormalizeImage, RandomHorizontalFlip, Squeeze, ToTensor, ToTorchImage
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

Section("cfg", "arguments for streaming FFCV dataset").params(
    data_path=Param(str, "Path to the FFCV .ffcv file", required=True),
    batch_size=Param(int, "Batch size for streaming", default=256),
    num_workers=Param(int, "Number of workers for loader", default=os.cpu_count()),
    drop_last=Param(
        bool,
        "Drop the last incomplete batch (default: True)",
        default=False,
    ),
    epochs=Param(int, "Number of epochs to run benchmark", default=2),
    order=Param(str, "Order: SEQUENTIAL or RANDOM or QUASI_RANDOM", default="SEQUENTIAL"),
    os_cache=Param(bool, "Use OS cache if the dataset can fit in memory", default=False),
    normalize=Param(
        bool,
        "If True, applies normalization using ImageNet mean and std; if False, uses scaling via T.ToDtype.",
        default=False,
    ),
)


@section("cfg")
@param("data_path")
@param("batch_size")
@param("num_workers")
@param("drop_last")
@param("epochs")
@param("order")
@param("os_cache")
@param("normalize")
def main(data_path, batch_size, num_workers, drop_last, epochs, order, os_cache, normalize):
    """Stream and benchmark an FFCV ImageNet dataset."""
    L.seed_everything(42)

    # Set up FFCV pipelines
    image_pipeline = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32) if normalize else T.ToDtype(torch.float32, scale=True),
    ]

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]
    pipelines = {"image": image_pipeline, "label": label_pipeline}

    order_option = getattr(OrderOption, order.upper())
    loader = Loader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order_option,
        pipelines=pipelines,
        os_cache=os_cache,
        drop_last=drop_last,
    )

    print("[INFO] Starting streaming benchmark...")
    for epoch in range(epochs):
        num_samples = 0
        t0 = time.perf_counter()
        for data in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", smoothing=0, mininterval=1):
            num_samples += data[0].shape[0]
        elapsed = time.perf_counter() - t0
        print(
            f"[RESULT] Epoch {epoch + 1}: Streamed {num_samples} samples in"
            f" {elapsed:.2f}s ({num_samples / elapsed:.2f} images/sec)"
        )
    print("[INFO] Finished streaming benchmark.")


if __name__ == "__main__":
    config = get_current_config()
    import argparse

    parser = argparse.ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    main()

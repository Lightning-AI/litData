# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Optional, Union

import yaml

from litdata.constants import _PIL_AVAILABLE, _ULTRALYTICS_AVAILABLE
from litdata.processing.functions import optimize
from litdata.streaming.resolver import Dir, _resolve_dir


def _ultralytics_optimize_fn(img_path: str):
    """Internal function that will be passed to the `optimize` function."""
    from PIL import Image

    img = Image.open(img_path)
    if not img_path.endswith((".jpg", ".jpeg", ".png")):
        raise ValueError(f"Unsupported image format: {img_path}. Supported formats are .jpg, .jpeg, and .png.")

    img_ext = os.path.splitext(img_path)[-1].lower()  # get the file extension

    label = ""
    label_path = img_path.replace("images", "labels").replace(img_ext, ".txt")

    # read label file if it exists, else raise an error
    if os.path.isfile(label_path):
        with open(label_path) as f:
            # don't convert to lists, as labels might've different lengths and hence config won't be same for all images
            label = f.read().strip()  # read the entire file content as a single string
    else:
        raise FileNotFoundError(f"Label file not found: {label_path}")

    return {
        "image": img,
        "label": label,
    }


def optimize_ultralytics_dataset(
    yaml_path: str,
    output_dir: str,
    chunk_size: Optional[int] = None,
    chunk_bytes: Optional[Union[int, str]] = None,
    num_workers: int = 1,
    verbose: bool = False,
) -> None:
    """Optimize an Ultralytics dataset by converting it into chunks and resizing images.

    Args:
        yaml_path: Path to the dataset YAML file.
        output_dir: Directory where the optimized dataset will be saved.
        chunk_size: Number of samples per chunk. If None, no chunking is applied.
        chunk_bytes: Maximum size of each chunk in bytes. If None, no size limit is applied.
        num_workers: Number of worker processes to use for optimization. Defaults to 1.
        verbose: Whether to print progress messages. Defaults to False.
    """
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Ultralytics is not installed. Please install it with `pip install ultralytics` to use this function."
        )
    if not _PIL_AVAILABLE:
        raise ImportError("PIL is not installed. Please install it with `pip install pillow` to use this function.")

    # check if the YAML file exists and is a file
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    if chunk_bytes is None and chunk_size is None:
        raise ValueError("Either chunk_bytes or chunk_size must be specified.")

    if chunk_bytes is not None and chunk_size is not None:
        raise ValueError("Only one of chunk_bytes or chunk_size should be specified, not both.")

    from ultralytics.data.utils import check_det_dataset

    # parse the YAML file & make sure data exists, else download it
    dataset_config = check_det_dataset(yaml_path)
    print(f"checked dataset structure: {dataset_config=}")

    output_dir = _resolve_dir(output_dir)

    mode_to_dir = {}

    for mode in ("train", "val", "test"):
        if dataset_config[mode] is None:
            continue
        if not os.path.exists(dataset_config[mode]):
            raise FileNotFoundError(f"Dataset directory not found for {mode}: {dataset_config[mode]}")
        mode_output_dir = get_output_dir(output_dir, mode)
        inputs = list_all_files(dataset_config[mode])

        optimize(
            fn=_ultralytics_optimize_fn,
            inputs=inputs,
            output_dir=mode_output_dir,
            chunk_bytes=chunk_bytes,
            chunk_size=chunk_size,
            num_workers=num_workers,
            mode="overwrite",
            verbose=verbose,
        )

        mode_to_dir[mode] = mode_output_dir
        print(f"Optimized {mode} dataset and saved to {mode_output_dir} ✅")

    # update the YAML file with the new paths
    for mode, dir in mode_to_dir.items():
        if mode in dataset_config:
            dataset_config[mode] = dir.url if dir.url else dir.path
        else:
            raise ValueError(f"Mode '{mode}' not found in dataset configuration.")

    # convert path to string if it's a Path object
    for key, value in dataset_config.items():
        if isinstance(value, Path):
            dataset_config[key] = str(value)

    # save the updated YAML file
    with open("litdata_" + yaml_path, "w") as f:
        yaml.dump(dataset_config, f)


def get_output_dir(output_dir: Dir, mode: str) -> Dir:
    if not isinstance(output_dir, Dir):
        raise TypeError(f"Expected output_dir to be of type Dir, got {type(output_dir)} instead.")
    print(f"Using output_dir: {output_dir}")
    url, path = output_dir.url, output_dir.path
    if url is not None:
        url = url.rstrip("/") + f"/{mode}"
    if path is not None:
        path = os.path.join(path, f"{mode}")

    updated_output_dir = Dir(url=url, path=path)
    print(f"Updated output_dir for mode '{mode}': {updated_output_dir}")
    return updated_output_dir


def list_all_files(path: str) -> list[str]:
    return [str(p) for p in Path(path).rglob("*") if p.is_file()]

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
from typing import Dict, Optional, Union

import yaml

from litdata.constants import _PIL_AVAILABLE, _ULTRALYTICS_AVAILABLE
from litdata.processing.functions import optimize
from litdata.streaming.resolver import Dir, _resolve_dir


def _ultralytics_optimize_fn(img_path: str) -> Optional[Dict]:
    """Optimized function for Ultralytics that reads image + label and optionally re-encodes to reduce size."""
    if not img_path.endswith((".jpg", ".jpeg", ".png")):
        raise ValueError(f"Unsupported image format: {img_path}. Supported formats are .jpg, .jpeg, and .png.")

    import cv2

    img_ext = os.path.splitext(img_path)[-1].lower()

    # Read image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # JPEG re-encode if image is jpeg or png
    if img_ext in [".jpg", ".jpeg", ".png"]:
        # Reduce quality to 90%
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, encoded = cv2.imencode(".jpg", img, encode_param)
        if not success:
            raise ValueError(f"JPEG encoding failed for: {img_path}")

        # Decode it back to a numpy array (OpenCV default format)
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Load the label
    label = ""
    label_path = img_path.replace("images", "labels").replace(img_ext, ".txt")
    if os.path.isfile(label_path):
        with open(label_path) as f:
            label = f.read().strip()
    else:
        return None  # skip this sample

    return {
        "img": img,
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
            output_dir=mode_output_dir.url or mode_output_dir.path or "optimized_data",
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
    output_yaml = Path(yaml_path).with_name("litdata_" + Path(yaml_path).name)
    with open(output_yaml, "w") as f:
        yaml.dump(dataset_config, f)


def get_output_dir(output_dir: Dir, mode: str) -> Dir:
    if not isinstance(output_dir, Dir):
        raise TypeError(f"Expected output_dir to be of type Dir, got {type(output_dir)} instead.")
    url, path = output_dir.url, output_dir.path
    if url is not None:
        url = url.rstrip("/") + f"/{mode}"
    if path is not None:
        path = os.path.join(path, f"{mode}")

    return Dir(url=url, path=path)


def list_all_files(_path: str) -> list[str]:
    path = Path(_path)

    if path.is_dir():
        # Recursively list all files under the directory
        return [str(p) for p in path.rglob("*") if p.is_file()]

    if path.is_file() and path.suffix == ".txt":
        # Read lines and return cleaned-up paths
        base_dir = path.parent  # use the parent of the txt file to resolve relative paths
        with open(path) as f:
            return [str((base_dir / line.strip()).resolve()) for line in f if line.strip()]

    else:
        raise ValueError(f"Unsupported path: {path}")

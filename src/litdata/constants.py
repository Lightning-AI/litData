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

import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache

_INDEX_FILENAME = "index.json"
_DEFAULT_CHUNK_BYTES = 1 << 26  # 64M B
_DEFAULT_FAST_DEV_RUN_ITEMS = 10
_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".lightning", "chunks")
_DEFAULT_LIGHTNING_CACHE_DIR = os.path.join("/cache", "chunks")
_LITDATA_CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", None)
_SUPPORTED_PROVIDERS = (
    "s3",
    "gs",
    "r2",
)  # cloud providers supported by litdata for uploading (optimize, map, merge, etc)

# This is required for full pytree serialization / deserialization support
_TORCH_GREATER_EQUAL_2_1_0 = RequirementCache("torch>=2.1.0")
_VIZ_TRACKER_AVAILABLE = RequirementCache("viztracer")
_BOTO3_AVAILABLE = RequirementCache("boto3")
_FSSPEC_AVAILABLE = RequirementCache("fsspec")
_TORCH_AUDIO_AVAILABLE = RequirementCache("torchaudio")
_ZSTD_AVAILABLE = RequirementCache("zstd")
_CRYPTOGRAPHY_AVAILABLE = RequirementCache("cryptography")
_GOOGLE_STORAGE_AVAILABLE = RequirementCache("google.cloud.storage")
_AZURE_STORAGE_AVAILABLE = RequirementCache("azure.storage.blob")
_TQDM_AVAILABLE = RequirementCache("tqdm")
_LIGHTNING_SDK_AVAILABLE = RequirementCache("lightning_sdk")
_HF_HUB_AVAILABLE = RequirementCache("huggingface_hub")
_PYARROW_AVAILABLE = RequirementCache("pyarrow")
_POLARS_AVAILABLE = RequirementCache("polars>1.0.0")
_PIL_AVAILABLE = RequirementCache("PIL")
_TORCH_VISION_AVAILABLE = RequirementCache("torchvision")
_AV_AVAILABLE = RequirementCache("av")
_OBSTORE_AVAILABLE = RequirementCache("obstore")

_DEBUG = bool(int(os.getenv("DEBUG_LITDATA", "0")))
_PRINT_DEBUG_LOGS = bool(int(os.getenv("PRINT_DEBUG_LOGS", "0")))

_MAX_WAIT_TIME = int(os.getenv("MAX_WAIT_TIME", "120"))
_FORCE_DOWNLOAD_TIME = int(os.getenv("FORCE_DOWNLOAD_TIME", "30"))
_DISABLE_S5CMD = bool(int(os.getenv("DISABLE_S5CMD", "0")))

# DON'T CHANGE ORDER
_TORCH_DTYPES_MAPPING = {
    0: torch.float32,
    1: torch.float,
    2: torch.float64,
    3: torch.double,
    4: torch.complex64,
    5: torch.cfloat,
    6: torch.complex128,
    7: torch.cdouble,
    8: torch.float16,
    9: torch.half,
    10: torch.bfloat16,  # Not supported https://github.com/pytorch/pytorch/issues/110285
    11: torch.uint8,
    12: torch.int8,
    13: torch.int16,
    14: torch.short,
    15: torch.int32,
    16: torch.int,
    17: torch.int64,
    18: torch.long,
    19: torch.bool,
    20: torch.uint16,
}

_NUMPY_SCTYPES = [  # All NumPy scalar types from np.core.sctypes.values()
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    bool,
    object,
    bytes,
    str,
    np.void,
]
_NUMPY_DTYPES_MAPPING: dict[int, np.dtype] = {i: np.dtype(v) for i, v in enumerate(_NUMPY_SCTYPES)}

_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S.%fZ"
_IS_IN_STUDIO = bool(os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)) and bool(os.getenv("LIGHTNING_CLUSTER_ID", None))
_ENABLE_STATUS = bool(int(os.getenv("ENABLE_STATUS_REPORT", "0")))

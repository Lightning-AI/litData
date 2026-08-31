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
import warnings

from litdata.__about__ import *  # noqa: F403
from litdata.constants import _LIGHTNING_SDK_AVAILABLE
from litdata.exceptions import ChunkWaitTimeoutError
from litdata.processing.complete import complete_dataset, is_complete_dataset
from litdata.processing.functions import map, merge_datasets, optimize, walk
from litdata.processing.media_folder import iter_webdataset_tar, list_media_folder
from litdata.raw.dataset import StreamingRawDataset
from litdata.streaming.collate import litdata_collate
from litdata.streaming.combined import CombinedStreamingDataset
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.dataset_update import dataset_update
from litdata.streaming.item_loader import TokensLoader
from litdata.streaming.parallel import ParallelStreamingDataset
from litdata.streaming.writer import index_parquet_dataset
from litdata.types import Audio, File, Graph, Image, Jpeg, JpegArray, Mesh, Nifti, Pdf, Pil, Tensor, Text, Tiff, Video
from litdata.utilities.breakpoint import breakpoint
from litdata.utilities.hf_dataset import index_hf_dataset
from litdata.utilities.keys_index import build_keys_index
from litdata.utilities.train_test_split import train_test_split

warnings.filterwarnings(
    "ignore",
    message=r"A newer version of lightning-sdk.*",
    category=UserWarning,
)

__all__ = [
    "StreamingDataset",
    "StreamingRawDataset",
    "CombinedStreamingDataset",
    "StreamingDataLoader",
    "litdata_collate",
    "TokensLoader",
    "ParallelStreamingDataset",
    "map",
    "optimize",
    "dataset_update",
    "build_keys_index",
    "walk",
    "list_media_folder",
    "iter_webdataset_tar",
    "train_test_split",
    "merge_datasets",
    "complete_dataset",
    "is_complete_dataset",
    "index_parquet_dataset",
    "index_hf_dataset",
    "breakpoint",
    "ChunkWaitTimeoutError",
    "Audio",
    "Video",
    "Image",
    "Jpeg",
    "JpegArray",
    "Pil",
    "Tiff",
    "File",
    "Mesh",
    "Pdf",
    "Nifti",
    "Tensor",
    "Text",
    "Graph",
]

if _LIGHTNING_SDK_AVAILABLE:
    from lightning_sdk import Machine  # noqa: F401

    __all__.append("Machine")

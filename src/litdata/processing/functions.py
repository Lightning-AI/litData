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

import concurrent.futures
import inspect
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union
from urllib import parse

import torch

from litdata import __version__
from litdata.constants import _INDEX_FILENAME, _IS_IN_STUDIO, _SUPPORTED_PROVIDERS
from litdata.helpers import _check_version_and_prompt_upgrade
from litdata.processing.data_processor import DataChunkRecipe, DataProcessor, MapRecipe
from litdata.processing.readers import BaseReader
from litdata.processing.utilities import (
    _get_work_dir,
    extract_rank_and_index_from_filename,
    optimize_dns_context,
    read_index_file_content,
)
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.fs_provider import _get_fs_provider
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import (
    Dir,
    _assert_dir_has_index_file,
    _assert_dir_is_empty,
    _execute,
    _resolve_dir,
)
from litdata.utilities._pytree import tree_flatten
from litdata.utilities.encryption import Encryption
from litdata.utilities.format import _get_tqdm_iterator_if_available

if TYPE_CHECKING:
    from lightning_sdk import Machine


def _is_remote_file(path: str) -> bool:
    obj = parse.urlparse(path)
    return obj.scheme in _SUPPORTED_PROVIDERS


def _get_indexed_paths(data: Any) -> dict[int, str]:
    flattened_item, _ = tree_flatten(data)

    return {
        index: element
        for index, element in enumerate(flattened_item)
        if isinstance(element, str) and (os.path.exists(element) or _is_remote_file(element))
    }


def _get_input_dir(inputs: Sequence[Any]) -> Optional[str]:
    indexed_paths = _get_indexed_paths(inputs[0])

    if len(indexed_paths) == 0:
        if len(inputs) < 2:
            return None
        # Check whether the second element has any input_path
        indexed_paths = _get_indexed_paths(inputs[1])
        if len(indexed_paths) == 0:
            return None

        # Every element should have filepaths if any contains one.
        raise ValueError(f"The provided item {inputs[0]} didn't contain any filepaths.")

    path = list(indexed_paths.values())[0]
    if _is_remote_file(path):
        return os.path.dirname(path)
    absolute_path = str(Path(path).resolve())

    if _IS_IN_STUDIO or absolute_path.startswith("/teamspace"):
        if "/.project" in absolute_path:
            return "/" + os.path.join(*str(list(indexed_paths.values())[0]).split("/")[:4])
        return "/" + os.path.join(*str(absolute_path).split("/")[:4])
    return None


def _get_default_num_workers() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return os.cpu_count() or 1


class LambdaMapRecipe(MapRecipe):
    """Recipe for `map`."""

    def __init__(
        self,
        fn: Callable[[str, Any], None],
        inputs: Union[Sequence[Any], StreamingDataLoader],
        storage_options: dict[str, Any] = {},
    ):
        super().__init__(storage_options)
        self._fn = fn
        self._inputs = inputs
        self._device: Optional[str] = None

        _fn = self._fn if isinstance(self._fn, FunctionType) else self._fn.__call__  # type: ignore
        params = inspect.signature(_fn).parameters
        self._contains_device = "device" in params
        self._contains_is_last = "is_last" in params

    def prepare_structure(self, _: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any, output_dir: str, is_last: bool) -> None:
        if self._contains_device and self._device is None:
            self._find_device()

        kwargs: dict[str, Any] = {}

        if self._contains_device:
            kwargs["device"] = self._device

        if self._contains_is_last:
            kwargs["is_last"] = is_last

        if isinstance(self._fn, (FunctionType, partial)):
            self._fn(item_metadata, output_dir, **kwargs)

        elif callable(self._fn):
            self._fn.__call__(item_metadata, output_dir, **kwargs)  # type: ignore
        else:
            raise ValueError(f"The provided {self._fn} isn't supported.")

    def _find_device(self) -> None:
        global_rank = os.getenv("DATA_OPTIMIZER_GLOBAL_RANK", None)
        if torch.cuda.is_available() and global_rank:
            num_gpus = torch.cuda.device_count()
            device = int(global_rank) % num_gpus
            self._device = f"cuda:{device}"


class LambdaDataChunkRecipe(DataChunkRecipe):
    """Recipe for `optimize`."""

    def __init__(
        self,
        fn: Callable[[Any], None],
        inputs: Union[Sequence[Any], StreamingDataLoader],
        chunk_size: Optional[int],
        chunk_bytes: Optional[Union[int, str]],
        compression: Optional[str],
        encryption: Optional[Encryption] = None,
        existing_index: Optional[dict[str, Any]] = None,
        storage_options: dict[str, Any] = {},
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            compression=compression,
            encryption=encryption,
            storage_options=storage_options,
        )
        self._fn = fn
        self._inputs = inputs
        self.is_generator = False
        self.existing_index = existing_index

        self.check_fn()

        self.prepare_item = self._prepare_item_generator if self.is_generator else self._prepare_item  # type: ignore

    def check_fn(self) -> None:
        if (
            isinstance(self._fn, (partial, FunctionType))
            and inspect.isgeneratorfunction(self._fn)
            or (callable(self._fn) and inspect.isgeneratorfunction(self._fn.__call__))  # type: ignore
        ):
            self.is_generator = True

    def _prepare_item(self, item_metadata: Any) -> Any:
        return self._fn(item_metadata)

    def _prepare_item_generator(self, item_metadata: Any) -> Any:
        yield from self._fn(item_metadata)  # type: ignore

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any) -> Any:
        """Being overridden dynamically."""


class QueueDataChunkRecipe(DataChunkRecipe):
    """A recipe to optimize data using a shared `multiprocessing.Queue`."""

    def __init__(
        self,
        fn: Callable[[Any], None],
        queue: mp.Queue,
        chunk_size: Optional[int],
        chunk_bytes: Optional[Union[int, str]],
        compression: Optional[str],
        encryption: Optional[Encryption] = None,
        existing_index: Optional[dict[str, Any]] = None,
        storage_options: dict[str, Any] = {},
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            compression=compression,
            encryption=encryption,
            storage_options=storage_options,
        )
        self._fn = fn
        self._queue = queue
        self.existing_index = existing_index
        self.is_generator = False

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._queue

    def prepare_item(self, item_metadata: Any) -> Any:
        return self._fn(item_metadata)


def map(
    fn: Callable[[str, Any], None],
    inputs: Union[Sequence[Any], StreamingDataLoader],
    output_dir: Union[str, Path, Dir],
    input_dir: Optional[Union[str, Path]] = None,
    weights: Optional[list[int]] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: Union[bool, int] = False,
    num_nodes: Optional[int] = None,
    machine: Optional[Union["Machine", str]] = None,
    num_downloaders: Optional[int] = None,
    num_uploaders: Optional[int] = None,
    reorder_files: bool = True,
    error_when_not_empty: bool = False,
    reader: Optional[BaseReader] = None,
    batch_size: Optional[int] = None,
    start_method: Optional[str] = None,
    optimize_dns: Optional[bool] = None,
    storage_options: dict[str, Any] = {},
    keep_data_ordered: bool = True,
) -> None:
    """Maps a callable over a collection of inputs, possibly in a distributed way.

    Args:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function, or a streaming dataloader.
        output_dir: The folder where the processed data should be stored.
        input_dir: Provide the path where your files are stored. If the files are on a remote storage,
            they will be downloaded in the background while processed.
        weights: Provide an associated weight to each input. This is used to balance work among workers.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use. Only supported on https://lightning.ai/.
        machine: When doing remote execution, the machine to use. Only supported on https://lightning.ai/.
        num_downloaders: The number of downloaders per worker.
        num_uploaders: The number of uploaders per workers.
        reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
            Set this to ``False`` if the order in which samples are processed should be preserved.
        error_when_not_empty: Whether we should error if the output folder isn't empty.
        reader: The reader to use when reading the data. By default, it uses the `BaseReader`.
        batch_size: Group the inputs into batches of batch_size length.
        start_method: The start method used by python multiprocessing package. Default to spawn unless running
            inside an interactive shell like Ipython.
        optimize_dns: Whether the optimized dns should be used.
        storage_options: Storage options for the cloud provider.
        keep_data_ordered (bool): Whether to use a shared queue for item distribution among workers.
            If False, all workers will fetch items dynamically from a shared queue, which helps balance
            workload and reduce idle time when some workers finish early. This may lead to unordered
            processing of items. If True, each worker processes a statically assigned subset of items
            in order.
    """
    _check_version_and_prompt_upgrade(__version__)

    if isinstance(inputs, StreamingDataLoader) and batch_size is not None:
        raise ValueError("When providing a streaming dataloader, pass the batch_size to the dataloader directly.")

    if isinstance(inputs, StreamingDataLoader) and weights is not None:
        raise ValueError("When providing a streaming dataloader, weights isn't supported.")

    if not isinstance(inputs, (Sequence, StreamingDataLoader)):
        raise ValueError(
            f"The provided inputs should be a non-empty sequence or a streaming dataloader. Found {inputs}."
        )

    if len(inputs) == 0:
        raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if not _IS_IN_STUDIO and (machine is not None or num_nodes is not None):
        raise ValueError(
            "Only https://lightning.ai/ supports multiple nodes or selecting a machine."
            " Create an account to try it out."
        )

    if not _IS_IN_STUDIO:
        print(
            "Create an account on https://lightning.ai/ to transform your data faster using "
            "multiple nodes and large machines."
        )

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        _output_dir: Dir = _resolve_dir(output_dir)

        if _output_dir.url and "cloudspaces" in _output_dir.url:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {_output_dir.path if _output_dir else None}."
                "\n HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        if error_when_not_empty:
            _assert_dir_is_empty(_output_dir, storage_options=storage_options)

        if not isinstance(inputs, StreamingDataLoader):
            input_dir = input_dir or _get_input_dir(inputs)
            resolved_dir = _resolve_dir(input_dir)

            if isinstance(batch_size, int) and batch_size > 1:
                inputs = [inputs[pos : pos + batch_size] for pos in range(0, len(inputs), batch_size)]
        else:
            resolved_dir = Dir()

        if num_workers == 0:
            num_workers = 1

        data_processor = DataProcessor(
            input_dir=resolved_dir,
            output_dir=_output_dir,
            num_workers=num_workers or _get_default_num_workers(),
            fast_dev_run=fast_dev_run,
            num_downloaders=num_downloaders,
            num_uploaders=num_uploaders,
            reorder_files=reorder_files,
            weights=weights,
            reader=reader,
            start_method=start_method,
            storage_options=storage_options,
            keep_data_ordered=keep_data_ordered,
        )

        with optimize_dns_context(optimize_dns if optimize_dns is not None else False):
            return data_processor.run(LambdaMapRecipe(fn, inputs, storage_options=storage_options))
    return _execute(
        f"litdata-map-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )


#
# `Optimize` Pipeline:
#
# 1. optimize() function uses LambdaDataChunkRecipe to process inputs
# 2. LambdaDataChunkRecipe is passed to DataProcess
# 3. DataProcess spawns DataWorkerProcess (BaseWorker) instances
# 4. Each worker:
#    a. Processes a single input through the optimize fn
#    b. Flattens the output using pytree:
#       - Converts nested structures (dict, list, tuple) into flat list
#       - Generates data_spec to preserve structure for later reconstruction
#    c. Uses Writer to serialize (bytes) the flattened data
#    d. Writes chunk files when either limit is reached:
#       - Total bytes > chunk_bytes
#       - Total items > chunk_size
#    e. Creates chunk_{idx}.bin files in cache
#    f. Saves data_spec in index.json for data structure preservation
#
def optimize(
    fn: Callable[[Any], Any],
    inputs: Optional[Union[Sequence[Any], StreamingDataLoader]] = None,
    output_dir: str = "optimized_data",
    queue: Optional[mp.Queue] = None,
    input_dir: Optional[str] = None,
    weights: Optional[list[int]] = None,
    chunk_size: Optional[int] = None,
    chunk_bytes: Optional[Union[int, str]] = None,
    compression: Optional[str] = None,
    encryption: Optional[Encryption] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: bool = False,
    num_nodes: Optional[int] = None,
    machine: Optional[Union["Machine", str]] = None,
    num_downloaders: Optional[int] = None,
    num_uploaders: Optional[int] = None,
    reorder_files: bool = True,
    reader: Optional[BaseReader] = None,
    batch_size: Optional[int] = None,
    mode: Optional[Literal["append", "overwrite"]] = None,
    use_checkpoint: bool = False,
    item_loader: Optional[BaseItemLoader] = None,
    start_method: Optional[str] = None,
    optimize_dns: Optional[bool] = None,
    storage_options: dict[str, Any] = {},
    keep_data_ordered: bool = True,
    verbose: bool = True,
) -> None:
    """This function converts a dataset into chunks, possibly in a distributed way.

    Args:
        fn: A function to be executed over each input element. The function should return the data sample that
            corresponds to the input. Every invocation of the function should return a similar hierarchy of objects,
            where the object types and list sizes don't change.
        inputs: A sequence of input to be processed by the `fn` function, or a streaming dataloader.
        queue: A multiprocessing queue to be used for processing the inputs. If provided, the `inputs` argument
            will be ignored, and the function will process the items from the queue instead.
        output_dir: The folder where the processed data should be stored.
        input_dir: Provide the path where your files are stored. If the files are on a remote storage,
            they will be downloaded in the background while processed.
        weights: Provide an associated weight to each input. This is used to balance work among workers.
        chunk_size: The maximum number of elements to hold within a chunk.
        chunk_bytes: The maximum number of bytes to hold within a chunk.
        compression: The compression algorithm to use over the chunks.
        encryption: The encryption algorithm to use over the chunks.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use. Only supported on https://lightning.ai/.
        machine: When doing remote execution, the machine to use. Only supported on https://lightning.ai/.
        num_downloaders: The number of downloaders per worker.
        num_uploaders: The numbers of uploaders per worker.
        reader: The reader to use when reading the data. By default, it uses the `BaseReader`.
        reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
            Set this to ``False`` if the order in which samples are processed should be preserved.
        batch_size: Group the inputs into batches of batch_size length.
        mode: The mode to use when writing the data. Accepts either ``append`` or ``overwrite`` or None.
            Defaults to None.
        use_checkpoint: Whether to create checkpoints while processing the data, which can be used to resume the
            processing from the last checkpoint if the process is interrupted. (`Default: False`)
        item_loader: The item loader that will be used during loading in StreamingDataset. Determines
                the format in which the data is stored and optimized for loading.
        start_method: The start method used by python multiprocessing package. Default to spawn unless running
            inside an interactive shell like Ipython.
        optimize_dns: Whether the optimized dns should be used.
        storage_options: Storage options for the cloud provider.
        keep_data_ordered (bool): Whether to use a shared queue for item distribution among workers.
            If False, all workers will fetch items dynamically from a shared queue, which helps balance
            workload and reduce idle time when some workers finish early. This may lead to unordered
            processing of items. If True, each worker processes a statically assigned subset of items
            in order.
        verbose: Whether to print the progress of the optimization. Defaults to True.
    """
    _check_version_and_prompt_upgrade(__version__)

    if queue is not None and inputs is not None:
        raise ValueError("You can either provide `inputs` or `queue`, not both.")

    if queue is None and inputs is None:
        raise ValueError("At least one of `inputs` or `queue` must be provided.")

    if queue is not None and not isinstance(queue, mp.queues.Queue):
        raise ValueError(f"The provided `queue` should be a multiprocessing queue. Found {type(queue)}.")

    if mode is not None and mode not in ["append", "overwrite"]:
        raise ValueError(f"The provided `mode` should be either `append` or `overwrite`. Found {mode}.")

    if isinstance(inputs, StreamingDataLoader) and batch_size is not None:
        raise ValueError("When providing a streaming dataloader, pass the batch_size to the dataloader directly.")

    if isinstance(inputs, StreamingDataLoader) and weights is not None:
        raise ValueError("When providing a streaming dataloader, weights isn't supported.")

    if queue is None:
        if not isinstance(inputs, (Sequence, StreamingDataLoader)):
            raise ValueError(
                f"The provided inputs should be a non-empty sequence or a streaming dataloader. Found {inputs}."
            )

        if len(inputs) == 0:
            raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if chunk_size is None and chunk_bytes is None:
        raise ValueError("Either `chunk_size` or `chunk_bytes` needs to be defined.")

    if not _IS_IN_STUDIO and (machine is not None or num_nodes is not None):
        raise ValueError(
            "Only https://lightning.ai/ supports multiple nodes or selecting a machine.Create an account to try it out."
        )

    if not _IS_IN_STUDIO and verbose:
        print(
            "Create an account on https://lightning.ai/ to optimize your data faster "
            "using multiple nodes and large machines."
        )

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        DATA_OPTIMIZER_NUM_NODES = int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0))
        _output_dir: Dir = _resolve_dir(output_dir)

        if (
            _output_dir.url is None
            and _output_dir.path
            and _output_dir.path.startswith("/teamspace/studios/this_studio")
            and DATA_OPTIMIZER_NUM_NODES > 0
        ):
            assert _output_dir.path
            output_dir = _output_dir.path.replace("/teamspace/studios/this_studio", "")
            output_dir = _get_work_dir().lstrip("/").rstrip("/") + "/" + output_dir.lstrip("/").rstrip("/")
            _output_dir = _resolve_dir(output_dir)

        if _output_dir.url is not None and "cloudspaces" in _output_dir.url:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {_output_dir.path}."
                "\n HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        _assert_dir_has_index_file(
            _output_dir, mode=mode, use_checkpoint=use_checkpoint, storage_options=storage_options
        )

        if not isinstance(inputs, StreamingDataLoader) and queue is None:
            assert inputs is not None
            resolved_dir = _resolve_dir(input_dir or _get_input_dir(inputs))

            if isinstance(batch_size, int) and batch_size > 1:
                inputs = [inputs[pos : pos + batch_size] for pos in range(0, len(inputs), batch_size)]
        else:
            resolved_dir = Dir()

        if num_workers == 0:
            num_workers = 1

        num_workers = num_workers or _get_default_num_workers()
        state_dict = dict.fromkeys(range(num_workers), 0)

        existing_index_file_content = (
            read_index_file_content(_output_dir, storage_options) if mode == "append" else None
        )

        if existing_index_file_content is not None:
            for chunk in existing_index_file_content["chunks"]:
                rank, index = extract_rank_and_index_from_filename(chunk["filename"])

                if rank < num_workers and state_dict[rank] <= index:
                    state_dict[rank] = index + 1  # +1 because we want to start from the next index

        data_processor = DataProcessor(
            input_dir=resolved_dir,
            output_dir=_output_dir,
            num_workers=num_workers or _get_default_num_workers(),
            fast_dev_run=fast_dev_run,
            num_downloaders=num_downloaders,
            num_uploaders=num_uploaders,
            reorder_files=reorder_files,
            reader=reader,
            state_dict=state_dict,
            use_checkpoint=use_checkpoint,
            item_loader=item_loader,
            start_method=start_method,
            storage_options=storage_options,
            keep_data_ordered=keep_data_ordered,
            verbose=verbose,
        )

        with optimize_dns_context(optimize_dns if optimize_dns is not None else False):
            recipe: Optional[Union[LambdaDataChunkRecipe, QueueDataChunkRecipe]] = None
            if queue is None:
                assert isinstance(inputs, (Sequence, StreamingDataLoader))
                recipe = LambdaDataChunkRecipe(
                    fn,
                    inputs,
                    chunk_size=chunk_size,
                    chunk_bytes=chunk_bytes,
                    compression=compression,
                    encryption=encryption,
                    existing_index=existing_index_file_content,
                    storage_options=storage_options,
                )
            else:
                assert queue is not None
                recipe = QueueDataChunkRecipe(
                    fn,
                    queue,
                    chunk_size=chunk_size,
                    chunk_bytes=chunk_bytes,
                    compression=compression,
                    encryption=encryption,
                    existing_index=existing_index_file_content,
                    storage_options=storage_options,
                )
            assert recipe is not None, "Recipe should be defined at this point."
            data_processor.run(recipe)
        return None
    return _execute(
        f"litdata-optimize-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )


def _listdir(folder: str) -> tuple[str, list[str]]:
    return folder, os.listdir(folder)


class walk:
    """This class is an optimized version of os.walk for listing files and folders from cloud filesystem.

    .. note:: The order of files and folders yielded aren't depth-first anymore due to the asynchronous listing call.

    """

    def __init__(self, folder: str, max_workers: Optional[int] = os.cpu_count()) -> None:
        self.folders = [folder]
        self.max_workers = max_workers or 1
        self.futures: list[concurrent.futures.Future] = []

        if not _IS_IN_STUDIO:
            print("This method is optimized to run on https://lightning.ai/. Don't use it otherwise.")

    def __iter__(self) -> Any:
        """This function queues the folders to perform listdir across multiple workers."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while len(self.folders):
                folder = self.folders.pop(0)
                future = executor.submit(_listdir, folder)
                self.futures.append(future)

            while self.futures:
                for future in concurrent.futures.as_completed(self.futures):
                    filenames = []
                    folders = []

                    folder, files_or_folders = future.result()
                    self.futures = [f for f in self.futures if f != future]

                    for file_or_folder in files_or_folders:
                        if os.path.isfile(os.path.join(folder, file_or_folder)):
                            filenames.append(file_or_folder)
                        else:
                            folders.append(file_or_folder)
                            self.folders.append(os.path.join(folder, file_or_folder))

                    yield folder, folders, filenames

                    while len(self.folders) and len(self.futures) <= self.max_workers * 2:
                        folder = self.folders.pop(0)
                        future = executor.submit(_listdir, folder)
                        self.futures.append(future)
        return


@dataclass
class CopyInfo:
    input_dir: Dir
    old_filename: str
    new_filename: str


def merge_datasets(
    input_dirs: list[str],
    output_dir: str,
    max_workers: Optional[int] = os.cpu_count(),
    storage_options: dict[str, Any] = {},
) -> None:
    """Enables to merge multiple existing optimized datasets into a single optimized dataset.

    Args:
        input_dirs: A list of directories pointing to the existing optimized datasets.
        output_dir: The directory where the merged dataset would be stored.
        max_workers: Number of workers for multithreading
        storage_options: Storage options for the cloud provider.
    """
    if len(input_dirs) == 0:
        raise ValueError("The input directories needs to be defined.")

    if len(input_dirs) == 1:
        raise ValueError("There should be more than 1 input directory")

    resolved_input_dirs = [_resolve_dir(input_dir) for input_dir in input_dirs]
    resolved_output_dir = _resolve_dir(output_dir)
    max_workers = max_workers or 1

    if any(input_dir == resolved_output_dir for input_dir in resolved_input_dirs):
        raise ValueError("The provided output_dir was found within the input_dirs. This isn't supported.")

    input_dirs_file_content = [read_index_file_content(input_dir, storage_options) for input_dir in resolved_input_dirs]

    if any(file_content is None for file_content in input_dirs_file_content):
        raise ValueError("One of the provided input_dir doesn't have an index file.")

    output_dir_file_content = read_index_file_content(resolved_output_dir, storage_options)

    if output_dir_file_content is not None:
        raise ValueError("The output_dir already contains an optimized dataset")

    assert input_dirs_file_content

    for input_dir_file_content in input_dirs_file_content[1:]:
        if input_dirs_file_content[0]["config"]["data_format"] != input_dir_file_content["config"]["data_format"]:  # type: ignore
            raise ValueError("Your are trying to merge datasets with different data formats")

        if input_dirs_file_content[0]["config"]["compression"] != input_dir_file_content["config"]["compression"]:  # type: ignore
            raise ValueError("Your are trying to merge datasets with different compression configuration.")

    chunks = []
    copy_infos: list[CopyInfo] = []
    counter = 0
    for input_dir, input_dir_file_content in zip(resolved_input_dirs, input_dirs_file_content):
        compression = input_dir_file_content["config"]["compression"]  # type: ignore
        for chunk in input_dir_file_content["chunks"]:  # type: ignore
            assert isinstance(chunk, dict)
            old_filename = chunk["filename"]
            new_filename = (
                f"chunk-0-{counter}.{compression}.bin" if compression is not None else f"chunk-0-{counter}.bin"
            )
            copy_infos.append(CopyInfo(input_dir=input_dir, old_filename=old_filename, new_filename=new_filename))
            chunk["filename"] = new_filename
            chunks.append(chunk)
            counter += 1

    index_json = {"config": input_dirs_file_content[0]["config"], "chunks": chunks}  # type: ignore

    _tqdm = _get_tqdm_iterator_if_available()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[concurrent.futures.Future] = []
        for copy_info in copy_infos:
            future = executor.submit(_apply_copy, copy_info, resolved_output_dir, storage_options)
            futures.append(future)

        for future in _tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    _save_index(index_json, resolved_output_dir, storage_options)


def _apply_copy(copy_info: CopyInfo, output_dir: Dir, storage_options: dict[str, Any] = {}) -> None:
    if output_dir.url is None and copy_info.input_dir.url is None:
        assert copy_info.input_dir.path
        assert output_dir.path
        input_filepath = os.path.join(copy_info.input_dir.path, copy_info.old_filename)
        output_filepath = os.path.join(output_dir.path, copy_info.new_filename)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        shutil.copyfile(input_filepath, output_filepath)

    elif output_dir.url and copy_info.input_dir.url:
        input_filepath = os.path.join(copy_info.input_dir.url, copy_info.old_filename)
        output_filepath = os.path.join(output_dir.url, copy_info.new_filename)

        fs_provider = _get_fs_provider(output_dir.url, storage_options)
        fs_provider.copy(input_filepath, output_filepath)
    else:
        raise NotImplementedError


def _save_index(index_json: dict, output_dir: Dir, storage_options: dict[str, Any] = {}) -> None:
    if output_dir.url is None:
        assert output_dir.path
        with open(os.path.join(output_dir.path, _INDEX_FILENAME), "w") as f:
            json.dump(index_json, f)
    else:
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(index_json, f)

            f.flush()

            remote_path = os.path.join(output_dir.url, _INDEX_FILENAME)

            fs_provider = _get_fs_provider(output_dir.url, storage_options)
            fs_provider.upload_file(f.name, remote_path)

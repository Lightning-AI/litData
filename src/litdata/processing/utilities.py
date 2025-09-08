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

import io
import json
import os
import tempfile
import urllib
from contextlib import contextmanager
from subprocess import DEVNULL, Popen
from typing import Any, Callable, Optional, Union
from urllib import parse

from litdata.constants import _INDEX_FILENAME, _IS_IN_STUDIO, _SUPPORTED_PROVIDERS
from litdata.streaming.cache import Dir
from litdata.streaming.fs_provider import _get_fs_provider, not_supported_provider


#! TODO: Not sure what this function is used for.
def _create_dataset(
    input_dir: Optional[str],
    storage_dir: str,
    dataset_type: Any,
    empty: Optional[bool] = None,
    size: Optional[int] = None,
    num_bytes: Optional[str] = None,
    data_format: Optional[Union[str, tuple[str]]] = None,
    compression: Optional[str] = None,
    num_chunks: Optional[int] = None,
    num_bytes_per_chunk: Optional[list[int]] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> None:
    """Create a dataset with metadata information about its source and destination using the Lightning SDK.

    This function will be called only when:
        - you're on last node (num_nodes == node_rank + 1)
        - `output_dir.url` and `output_dir.path` are defined
        - You're using Lightning Studio
    """
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    user_id = os.getenv("LIGHTNING_USER_ID", None)
    studio_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)
    lightning_app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)

    if project_id is None:
        return

    if not storage_dir:
        raise ValueError("The storage_dir should be defined.")

    from lightning_sdk.lightning_cloud.openapi import ProjectIdDatasetsBody
    from lightning_sdk.lightning_cloud.openapi.rest import ApiException
    from lightning_sdk.lightning_cloud.rest_client import LightningClient

    client = LightningClient(retry=False)

    try:
        client.dataset_service_create_dataset(
            body=ProjectIdDatasetsBody(
                cloud_space_id=studio_id if lightning_app_id is None else None,
                cluster_id=cluster_id,
                creator_id=user_id,
                empty=empty,
                input_dir=input_dir,
                lightning_app_id=lightning_app_id,
                name=name,
                size=size,
                num_bytes=num_bytes,
                data_format=str(data_format) if data_format else data_format,
                compression=compression,
                num_chunks=num_chunks,
                num_bytes_per_chunk=num_bytes_per_chunk,
                storage_dir=storage_dir,
                type=dataset_type,
                version=version,
            ),
            project_id=project_id,
        )
    except ApiException as ex:
        if "already exists" in str(ex.body):
            pass
        else:
            raise ex


def get_worker_rank() -> Optional[str]:
    return os.getenv("DATA_OPTIMIZER_GLOBAL_RANK")


#! TODO: Do we still need this? It is not used anywhere.
def catch(func: Callable) -> Callable:
    def _wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Optional[Exception]]:
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e

    return _wrapper


# Credit to the https://github.com/rom1504/img2dataset Github repo
# The code was taken from there. It has a MIT License.


def make_request(
    url: str,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",  # noqa: S107
) -> io.BytesIO:
    """Download an image with urllib."""
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/Lightning-AI/pytorch-lightning)"

    with urllib.request.urlopen(
        urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string}), timeout=timeout
    ) as r:
        return io.BytesIO(r.read())


@contextmanager
def optimize_dns_context(enable: bool) -> Any:
    """Optimize the DNS resolution for the Lightning Studio machine.

    This speeds up the DNS resolution for the current machine as it reduces the number of DNS requests.
    """
    optimize_dns(enable)
    try:
        yield
        optimize_dns(False)  # always disable the optimize DNS
    except Exception as e:
        optimize_dns(False)  # always disable the optimize DNS
        raise e


def optimize_dns(enable: bool) -> None:
    if not _IS_IN_STUDIO:
        return

    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (enable and any("127.0.0.53" in line for line in lines)) or (
        not enable and any("127.0.0.1" in line for line in lines)
    ):
        cmd = (
            f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python"
            f" -c 'from litdata.processing.utilities import _optimize_dns; _optimize_dns({enable})'"
        )
        Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()  # E501


def _optimize_dns(enable: bool) -> None:
    """Optimize the DNS resolution for the Lightning Studio machine.

    When enable=True: Switches to using localhost (127.0.0.1) as the DNS server, which typically means using a
        `local DNS cache`.
    When enable=False: It switches back to using systemd-resolved (127.0.0.53), which is the default DNS resolver
        in many modern Linux distro.
    """
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    write_lines = []
    for line in lines:
        if "nameserver 127" in line:
            if enable:
                write_lines.append("nameserver 127.0.0.1\n")
            else:
                write_lines.append("nameserver 127.0.0.53\n")
        else:
            write_lines.append(line)

    with open("/etc/resolv.conf", "w") as f:
        for line in write_lines:
            f.write(line)


def _get_work_dir() -> str:
    # Provides the storage path associated to the current Lightning Work.
    bucket_name = os.getenv("LIGHTNING_BUCKET_NAME")
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID")
    work_id = os.getenv("LIGHTNING_CLOUD_WORK_ID")
    assert bucket_name is not None
    assert project_id is not None
    assert work_id is not None
    return f"s3://{bucket_name}/projects/{project_id}/lightningapps/{app_id}/artifacts/{work_id}/content/"


def read_index_file_content(output_dir: Dir, storage_options: dict[str, Any] = {}) -> Optional[dict[str, Any]]:
    """Read the index file content."""
    if not isinstance(output_dir, Dir):
        raise ValueError("The provided output_dir should be a Dir object.")

    if output_dir.url is None:
        if output_dir.path is None:
            return None
        index_file_path = os.path.join(output_dir.path, _INDEX_FILENAME)
        if not os.path.exists(index_file_path):
            return None
        with open(index_file_path) as f:
            return json.load(f)

    else:
        # download the index file from the cloud provider, and read it
        obj = parse.urlparse(output_dir.url)

        if obj.scheme not in _SUPPORTED_PROVIDERS:
            not_supported_provider(output_dir.url)

        fs_provider = _get_fs_provider(output_dir.url, storage_options)

        prefix = output_dir.url.rstrip("/") + "/"

        # Check the index file exists
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_file_name = temp_file.name
                fs_provider.download_file(os.path.join(prefix, _INDEX_FILENAME), temp_file_name)
            # Read data from the temporary file
            with open(temp_file_name) as temp_file:
                data = json.load(temp_file)
            # Delete the temporary file
            os.remove(temp_file_name)
            return data
        except Exception:
            return None


def extract_rank_and_index_from_filename(chunk_filename: str) -> tuple[int, int]:
    """Extract the rank and index from the filename.

    It is assumed that the filename is in the format `chunk-<rank>-<index>.bin` or
    `chunk-<rank>-<index>.compressionAlgorithm.bin`.

    """
    # remove chunk and bin
    chunk_filename = chunk_filename[6:-4].split("-")  # (0, 0) or (0, 0.compressionAlgorithm)
    assert len(chunk_filename) == 2

    # get the rank and index
    rank = int(chunk_filename[0])
    index = int(chunk_filename[1].split(".")[0])

    return rank, index


def remove_uuid_from_filename(filepath: str) -> str:
    """Remove the unique id from the filepath. Expects the filepath to be in the format
    `checkpoint-<rank>-<uuid>.json`.

    e.g.: `checkpoint-0-9fe2c4e93f654fdbb24c02b15259716c.json`
        -> `checkpoint-0.json`

    """
    if not filepath.__contains__(".checkpoints"):
        return filepath

    # uuid is of 32 characters, '.json' is 5 characters and '-' is 1 character
    return filepath[:-38] + ".json"


def construct_storage_options(storage_options: dict[str, Any], input_dir: Dir) -> dict[str, Any]:
    merged_storage_options = storage_options.copy()
    if hasattr(input_dir, "data_connection_id") and input_dir.data_connection_id:
        merged_storage_options["data_connection_id"] = input_dir.data_connection_id
    return merged_storage_options

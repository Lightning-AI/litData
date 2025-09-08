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

import datetime
import os
import re
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Literal, Optional, Union
from urllib import parse

from litdata.constants import _LIGHTNING_SDK_AVAILABLE, _SUPPORTED_PROVIDERS
from litdata.streaming.fs_provider import _get_fs_provider, not_supported_provider

if TYPE_CHECKING:
    from lightning_sdk import Machine
    from lightning_sdk.lightning_cloud.openapi import V1DataConnection


@dataclass
class Dir:
    """Holds a directory path and possibly its associated remote URL."""

    path: Optional[str] = None
    url: Optional[str] = None
    data_connection_id: Optional[str] = None


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"


def _resolve_dir(dir_path: Optional[Union[str, Path, Dir]]) -> Dir:
    if isinstance(dir_path, Dir):
        return Dir(
            path=str(dir_path.path) if dir_path.path else None,
            url=str(dir_path.url) if dir_path.url else None,
            data_connection_id=dir_path.data_connection_id if dir_path.data_connection_id else None,
        )

    if dir_path is None:
        return Dir()

    if not isinstance(dir_path, (str, Path)):
        raise ValueError(f"`dir_path` must be either a string, Path, or Dir, got: {type(dir_path)}")

    if isinstance(dir_path, str):
        cloud_prefixes = ("s3://", "gs://", "azure://", "hf://")
        if dir_path.startswith(cloud_prefixes):
            return Dir(path=None, url=dir_path)

        if dir_path.startswith("local:"):
            return Dir(path=None, url=dir_path)

        dir_path = _resolve_time_template(dir_path)

    dir_path_absolute = str(Path(dir_path).absolute().resolve())
    dir_path = str(dir_path)  # Convert to string if it was a Path object

    # Handle special teamspace paths
    if dir_path_absolute.startswith("/teamspace/studios/this_studio"):
        return Dir(path=dir_path_absolute, url=None)

    if dir_path_absolute.startswith("/.project"):
        dir_path_absolute = dir_path

    if dir_path_absolute.startswith("/teamspace/studios") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_studio(dir_path_absolute, dir_path_absolute.split("/")[3], None)

    if dir_path_absolute.startswith("/teamspace/s3_connections") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_s3_connections(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/gcs_connections") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_gcs_connections(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/s3_folders") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_s3_folders(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/gcs_folders") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_gcs_folders(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/lightning_storage") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_lightning_storage(dir_path_absolute)

    if dir_path_absolute.startswith("/teamspace/datasets") and len(dir_path_absolute.split("/")) > 3:
        return _resolve_datasets(dir_path_absolute)

    return Dir(path=dir_path_absolute, url=None)


def _match_studio(target_id: Optional[str], target_name: Optional[str], cloudspace: Any) -> bool:
    if cloudspace.name is not None and target_name is not None and cloudspace.name.lower() == target_name.lower():
        return True

    if target_id is not None and cloudspace.id == target_id:
        return True

    return bool(
        cloudspace.display_name is not None
        and target_name is not None
        and cloudspace.display_name.lower() == target_name.lower()
    )


def _resolve_studio(dir_path: str, target_name: Optional[str], target_id: Optional[str]) -> Dir:
    from lightning_sdk.lightning_cloud.rest_client import LightningClient

    client = LightningClient(max_tries=2)

    # Get the ids from env variables
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    provider = os.getenv("LIGHTNING_CLOUD_PROVIDER", CloudProvider.AWS)

    if cluster_id is None:
        raise RuntimeError("The `LIGHTNING_CLUSTER_ID` couldn't be found from the environment variables.")

    if project_id is None:
        raise RuntimeError("The `LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables.")

    clusters = client.cluster_service_list_project_clusters(project_id).clusters

    cloudspaces = client.cloud_space_service_list_cloud_spaces(project_id=project_id, cluster_id=cluster_id).cloudspaces
    target_cloud_space = [cloudspace for cloudspace in cloudspaces if _match_studio(target_id, target_name, cloudspace)]

    if not target_cloud_space:
        raise ValueError(f"We didn't find any matching Studio for the provided name `{target_name}`.")

    target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

    if not target_cluster:
        raise ValueError(
            f"We didn't find a matching cluster associated with the id {target_cloud_space[0].cluster_id}."
        )

    if provider == CloudProvider.AWS:
        bucket_name = target_cluster[0].spec.aws_v1.bucket_name
        scheme = "s3"
    elif provider == CloudProvider.GCP:
        bucket_name = target_cluster[0].spec.google_cloud_v1.bucket_name
        scheme = "gs"
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}. Supported providers are AWS and GCP.")

    return Dir(
        path=dir_path,
        url=os.path.join(
            f"{scheme}://{bucket_name}/projects/{project_id}/cloudspaces/{target_cloud_space[0].id}/code/content",
            *dir_path.split("/")[4:],
        ),
    )


@lru_cache(maxsize=5)
def _resolve_data_connection(dir_path: str) -> "V1DataConnection":
    """Resolve data connection from teamspace path with caching.

    Extracts connection name from paths like '/teamspace/s3_connections/my_dataset'
    and fetches the corresponding data connection from Lightning Cloud.

    Args:
        dir_path: Teamspace path containing the connection name

    Returns:
        V1DataConnection: The resolved data connection object

    Raises:
        ValueError: If path format is invalid or connection not found
        RuntimeError: If required environment variables are missing
    """
    # Validate dir_path format - must start with /teamspace/ and have expected structure
    if not dir_path.startswith("/teamspace/"):
        raise ValueError(f"Invalid dir_path format: {dir_path}. Expected path starting with '/teamspace/'.")

    path_parts = dir_path.split("/")
    if len(path_parts) < 4:
        raise ValueError(f"Invalid teamspace path: {dir_path}. Expected at least 4 path components.")

    # Get the ids from env variables
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    if project_id is None:
        raise RuntimeError("The `LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables.")

    target_name = path_parts[3]

    from lightning_sdk.lightning_cloud.rest_client import LightningClient

    client = LightningClient(max_tries=2)

    data_connections = client.data_connection_service_list_data_connections(project_id).data_connections

    data_connection = [dc for dc in data_connections if dc.name == target_name]

    if not data_connection:
        raise ValueError(f"We didn't find any matching data connection with the provided name `{target_name}`.")

    return data_connection[0]


def _resolve_s3_connections(dir_path: str) -> Dir:
    data_connection = _resolve_data_connection(dir_path)

    return Dir(path=dir_path, url=os.path.join(data_connection.aws.source, *dir_path.split("/")[4:]))


def _resolve_gcs_connections(dir_path: str) -> Dir:
    data_connection = _resolve_data_connection(dir_path)

    return Dir(path=dir_path, url=os.path.join(data_connection.gcp.source, *dir_path.split("/")[4:]))


def _resolve_s3_folders(dir_path: str) -> Dir:
    data_connection = _resolve_data_connection(dir_path)

    return Dir(path=dir_path, url=os.path.join(data_connection.s3_folder.source, *dir_path.split("/")[4:]))


def _resolve_gcs_folders(dir_path: str) -> Dir:
    data_connection = _resolve_data_connection(dir_path)

    return Dir(path=dir_path, url=os.path.join(data_connection.gcs_folder.source, *dir_path.split("/")[4:]))


def _resolve_lightning_storage(dir_path: str) -> Dir:
    data_connection = _resolve_data_connection(dir_path)

    return Dir(
        path=dir_path,
        url=os.path.join(data_connection.r2.source, *dir_path.split("/")[4:]),
        data_connection_id=data_connection.id,
    )


def _resolve_datasets(dir_path: str) -> Dir:
    from lightning_sdk.lightning_cloud.rest_client import LightningClient

    client = LightningClient(max_tries=2)

    # Get the ids from env variables
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    studio_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)

    if cluster_id is None:
        raise RuntimeError("The `LIGHTNING_CLUSTER_ID` couldn't be found from the environment variables.")

    if project_id is None:
        raise RuntimeError("The `LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables.")

    if studio_id is None:
        raise RuntimeError("The `LIGHTNING_CLOUD_SPACE_ID` couldn't be found from the environment variables.")

    clusters = client.cluster_service_list_project_clusters(project_id).clusters

    target_cloud_space = [
        cloudspace
        for cloudspace in client.cloud_space_service_list_cloud_spaces(
            project_id=project_id, cluster_id=cluster_id
        ).cloudspaces
        if cloudspace.id == studio_id
    ]

    if not target_cloud_space:
        raise ValueError(f"We didn't find any matching Studio for the provided id `{studio_id}`.")

    target_cluster = [cluster for cluster in clusters if cluster.id == target_cloud_space[0].cluster_id]

    if not target_cluster:
        raise ValueError(
            f"We didn't find a matching cluster associated with the id `{target_cloud_space[0].cluster_id}`."
        )

    return Dir(
        path=dir_path,
        url=os.path.join(
            f"s3://{target_cluster[0].spec.aws_v1.bucket_name}/projects/{project_id}/datasets/",
            *dir_path.split("/")[3:],
        ),
    )


def _assert_dir_is_empty(
    output_dir: Dir, append: bool = False, overwrite: bool = False, storage_options: dict[str, Any] = {}
) -> None:
    if not isinstance(output_dir, Dir):
        raise ValueError("The provided output_dir isn't a `Dir` Object.")

    if output_dir.url is None:
        return

    obj = parse.urlparse(output_dir.url)

    if obj.scheme not in _SUPPORTED_PROVIDERS:
        not_supported_provider(output_dir.url)

    # Add data connection ID to storage_options for R2 connections
    merged_storage_options = storage_options.copy()
    if output_dir.data_connection_id:
        merged_storage_options["data_connection_id"] = output_dir.data_connection_id

    fs_provider = _get_fs_provider(output_dir.url, merged_storage_options)

    is_empty = fs_provider.is_empty(output_dir.url)

    # We aren't allowing to add more data
    # TODO: Add support for `append` and `overwrite`.
    if not is_empty:
        raise RuntimeError(
            f"The provided output_dir `{output_dir.path}` already contains data and datasets are meant to be immutable."
            "\n HINT: Did you consider changing the `output_dir` with your own versioning as a suffix?"
        )


def _assert_dir_has_index_file(
    output_dir: Dir,
    mode: Optional[Literal["append", "overwrite"]] = None,
    use_checkpoint: bool = False,
    storage_options: dict[str, Any] = {},
) -> None:
    if mode is not None and mode not in ["append", "overwrite"]:
        raise ValueError(f"The provided `mode` should be either `append` or `overwrite`. Found {mode}.")

    if mode == "append":
        # in append mode, we neither need to delete the index file nor the chunks
        return

    if not isinstance(output_dir, Dir):
        raise ValueError("The provided output_dir isn't a Dir Object.")

    if output_dir.url is None:
        # this is a local directory
        assert output_dir.path

        if os.path.exists(output_dir.path):
            # we need to delete the index file
            index_file = os.path.join(output_dir.path, "index.json")
            if os.path.exists(index_file) and mode is None:
                raise RuntimeError(
                    f"The provided output_dir `{output_dir.path}` already contains an optimized immutable datasets."
                    "\n HINT: Did you consider changing the `output_dir` with your own versioning as a suffix?"
                    "\n HINT: If you want to append/overwrite to the existing dataset, use `mode='append | overwrite'`."
                )

            # delete index.json file and chunks
            if os.path.exists(os.path.join(output_dir.path, "index.json")):
                # only possible if mode = "overwrite"
                os.remove(os.path.join(output_dir.path, "index.json"))

            if mode == "overwrite" or (mode is None and not use_checkpoint):
                for file in os.listdir(output_dir.path):
                    if file.endswith(".bin"):
                        os.remove(os.path.join(output_dir.path, file))

                # delete checkpoints
                with suppress(FileNotFoundError):
                    shutil.rmtree(os.path.join(output_dir.path, ".checkpoints"))

        return

    obj = parse.urlparse(output_dir.url)

    if obj.scheme not in _SUPPORTED_PROVIDERS:
        not_supported_provider(output_dir.url)

    # Add data connection ID to storage_options for R2 connections
    merged_storage_options = storage_options.copy()
    if output_dir.data_connection_id:
        merged_storage_options["data_connection_id"] = output_dir.data_connection_id

    fs_provider = _get_fs_provider(output_dir.url, merged_storage_options)

    prefix = output_dir.url.rstrip("/") + "/"

    is_empty = fs_provider.is_empty(output_dir.url)

    # No files are found in this folder
    if is_empty:
        return

    # Check the index file exists
    try:
        fs_provider.exists(os.path.join(prefix, "index.json"))
        has_index_file = True
    except Exception:
        has_index_file = False

    if has_index_file and mode is None:
        raise RuntimeError(
            f"The provided output_dir `{output_dir.path}` already contains an optimized immutable datasets."
            "\n HINT: Did you consider changing the `output_dir` with your own versioning as a suffix?"
            "\n HINT: If you want to append/overwrite to the existing dataset, use `mode='append | overwrite'`."
        )

    # delete all the files (including the index file in overwrite mode)
    if mode == "overwrite" or (mode is None and not use_checkpoint):
        fs_provider.delete_file_or_directory(output_dir.url)


def _get_lightning_cloud_url() -> str:
    # detect local development
    if os.getenv("VSCODE_PROXY_URI", "").startswith("http://localhost:9800"):
        return "http://localhost:9800"
    # DO NOT CHANGE!
    return os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")


def _resolve_time_template(path: str) -> str:
    """Resolves a datetime pattern in the given path string.

    If the path contains a placeholder in the form `{%Y-%m-%d}`, it replaces it
    with the current date/time formatted using the specified `strftime` pattern.

    Example:
        Input: "/logs/log_{%Y-%m-%d}.txt"
        Output (on May 5, 2025): "/logs/log_2025-05-05.txt"

    Args:
        path (str): The file path containing an optional datetime placeholder.

    Returns:
        str: The path with the datetime placeholder replaced by the current timestamp.
    """
    match = re.search("^.*{%.*}.*$", path)
    if match is None:
        return path

    pattern = path.split("{")[1].split("}")[0]

    return path.replace("{" + pattern + "}", datetime.datetime.now().strftime(pattern))


def _execute(
    name: str,
    num_nodes: int,
    machine: Optional[Union["Machine", str]] = None,
    command: Optional[str] = None,
    interruptible: bool = False,
) -> None:
    """Remotely execute the current operator."""
    if not _LIGHTNING_SDK_AVAILABLE:
        raise ModuleNotFoundError("The `lightning_sdk` is required.")

    from lightning_sdk import Studio

    lightning_skip_install = os.getenv("LIGHTNING_SKIP_INSTALL", "")
    if lightning_skip_install:
        lightning_skip_install = f" LIGHTNING_SKIP_INSTALL={lightning_skip_install} "

    lightning_branch = os.getenv("LIGHTNING_BRANCH", "")
    if lightning_branch:
        lightning_branch = f" LIGHTNING_BRANCH={lightning_branch} "

    studio = Studio()
    job = studio._studio_api.create_data_prep_machine_job(
        command or f"cd {os.getcwd()} &&{lightning_skip_install}{lightning_branch} python {' '.join(sys.argv)}",
        name=name,
        num_instances=num_nodes,
        studio_id=studio._studio.id,
        teamspace_id=studio._teamspace.id,
        cloud_account=studio._studio.cluster_id,
        machine=machine or studio._studio_api.get_machine(studio._studio.id, studio._teamspace.id),
        interruptible=interruptible,
    )

    has_printed = False

    while True:
        curr_job = studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance(
            project_id=studio._teamspace.id, id=job.id
        )
        if not has_printed:
            cloud_url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai").replace(":443", "")
            job_url = f"{cloud_url}/{studio.owner.name}/{studio._teamspace.name}"
            job_url += f"/studios/{studio.name}/app?app_id=litdata&app_tab=Runs&job_name={curr_job.name}"
            print(f"Find your job at {job_url}")
            has_printed = True

        if curr_job.status.phase == "LIGHTNINGAPP_INSTANCE_STATE_FAILED":
            raise RuntimeError(f"job {curr_job.name} failed!")

        if curr_job.status.phase in ["LIGHTNINGAPP_INSTANCE_STATE_STOPPED", "LIGHTNINGAPP_INSTANCE_STATE_COMPLETED"]:
            break

        sleep(1)

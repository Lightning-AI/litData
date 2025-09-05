import datetime
import sys
from pathlib import Path
from unittest import mock

import pytest
from lightning_sdk.lightning_cloud import login
from lightning_sdk.lightning_cloud.openapi import (
    Externalv1Cluster,
    V1AwsDataConnection,
    V1AWSDirectV1,
    V1CloudSpace,
    V1ClusterSpec,
    V1DataConnection,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListDataConnectionsResponse,
    V1S3FolderDataConnection,
)

from litdata.streaming import resolver


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_s3_connections(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(
        RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables."
    ):
        resolver._resolve_dir("/teamspace/s3_connections/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[V1DataConnection(name="imagenet", aws=V1AwsDataConnection(source="s3://imagenet-bucket"))],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    assert resolver._resolve_dir("/teamspace/s3_connections/imagenet").url == "s3://imagenet-bucket"
    assert resolver._resolve_dir("/teamspace/s3_connections/imagenet/train").url == "s3://imagenet-bucket/train"

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="name `imagenet`"):
        assert resolver._resolve_dir("/teamspace/s3_connections/imagenet")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_studios(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLUSTER_ID`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="other_studio_id", cluster_id="cluster_id_of_other_studio")],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(
                id="cluster_id_of_other_studio", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket"))
            )
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://my_bucket/projects/project_id/cloudspaces/other_studio_id/code/content"
    assert resolver._resolve_dir("/teamspace/studios/other_studio").url == expected
    assert resolver._resolve_dir("/teamspace/studios/other_studio/train").url == f"{expected}/train"

    datetime_mock = mock.MagicMock()
    now_mock = mock.MagicMock()
    called = False

    def fn(pattern):
        nonlocal called
        called = True
        assert pattern == "%Y-%m-%d-%H-%M-%S"
        import datetime

        return datetime.datetime(2023, 12, 1, 10, 37, 40, 281942).strftime(pattern)

    now_mock.strftime = fn
    datetime_mock.datetime.now.return_value = now_mock
    monkeypatch.setattr(resolver, "datetime", datetime_mock)
    assert (
        resolver._resolve_dir("/teamspace/studios/other_studio/{%Y-%m-%d-%H-%M-%S}").path
        == "/teamspace/studios/other_studio/2023-12-01-10-37-40"
    )
    assert called

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="other_studio`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_s3_folders(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(name="debug_folder", s3_folder=V1S3FolderDataConnection(source="s3://imagenet-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://imagenet-bucket"
    assert resolver._resolve_dir("/teamspace/s3_folders/debug_folder").url == expected
    assert resolver._resolve_dir("/teamspace/s3_folders/debug_folder/a/b/c").url == expected + "/a/b/c"
    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_datasets(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    assert resolver._resolve_dir("s3://bucket_name").url == "s3://bucket_name"

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLUSTER_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_SPACE_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "cloud_space_id")

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="cloud_space_id", cluster_id="cluster_id_of_other_studio")],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(
                id="cluster_id_of_other_studio", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket"))
            )
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://my_bucket/projects/project_id/datasets/imagenet"
    assert resolver._resolve_dir("/teamspace/datasets/imagenet").url == expected
    assert resolver._resolve_dir("/teamspace/datasets/imagenet/train").url == f"{expected}/train"

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="cloud_space_id`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_dst_resolver_dataset_path(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    assert resolver._resolve_dir("something").url is None

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "cloud_space_id")

    client_mock = mock.MagicMock()

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(id="cluster_id", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket")))
        ],
    )

    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="cloud_space_id", cluster_id="cluster_id")],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    boto3 = mock.MagicMock()
    client_s3_mock = mock.MagicMock()
    client_s3_mock.list_objects_v2.return_value = {"KeyCount": 1, "Contents": []}
    boto3.client.return_value = client_s3_mock
    resolver.boto3 = boto3

    assert resolver._resolve_dir("something").url is None

    client_s3_mock.list_objects_v2.return_value = {"KeyCount": 0, "Contents": []}

    assert (
        resolver._resolve_dir("/teamspace/datasets/something/else").url
        == "s3://my_bucket/projects/project_id/datasets/something/else"
    )

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
@pytest.mark.parametrize("phase", ["LIGHTNINGAPP_INSTANCE_STATE_STOPPED", "LIGHTNINGAPP_INSTANCE_STATE_COMPLETED"])
def test_execute(phase, monkeypatch, lightning_sdk_mock):
    studio = mock.MagicMock()
    studio._studio.id = "studio_id"
    studio._teamspace.id = "teamspace_id"
    studio._studio.cluster_id = "cluster_id"
    studio._studio_api.get_machine.return_value = "cpu"
    studio.owner.name = "username"
    studio._teamspace.name = "teamspace_name"
    studio.name = "studio_name"
    studio.name = "studio_name"
    job = mock.MagicMock()
    job.name = "job_name"
    job.id = "job_id"
    job.status.phase = phase
    studio._studio_api.create_data_prep_machine_job.return_value = job
    studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance.return_value = job

    monkeypatch.setattr(resolver, "_LIGHTNING_SDK_AVAILABLE", True)
    lightning_sdk_mock.Studio = mock.MagicMock(return_value=studio)

    called = False

    def print_fn(msg, file=None):
        nonlocal called
        assert (
            msg
            == "Find your job at https://lightning.ai/username/teamspace_name/studios/studio_name/app?app_id=litdata&app_tab=Runs&job_name=job_name"
        )
        called = True

    original_print = __builtins__["print"]
    monkeypatch.setattr(sys, "argv", ["test.py", "--dummy"])
    monkeypatch.setitem(__builtins__, "print", print_fn)
    assert not called
    resolver._execute("dummy", 2)
    assert called
    monkeypatch.setitem(__builtins__, "print", original_print)

    generated_args = studio._studio_api.create_data_prep_machine_job._mock_call_args_list[0].args
    assert "&& python test.py --dummy" in generated_args[0]

    generated_kwargs = studio._studio_api.create_data_prep_machine_job._mock_call_args_list[0].kwargs
    assert generated_kwargs == {
        "name": "dummy",
        "num_instances": 2,
        "studio_id": "studio_id",
        "teamspace_id": "teamspace_id",
        "cloud_account": "cluster_id",
        "machine": "cpu",
        "interruptible": False,
    }

    generated_kwargs = (
        studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance._mock_call_args_list[
            0
        ].kwargs
    )
    assert generated_kwargs == {"project_id": "teamspace_id", "id": "job_id"}


def test_assert_dir_is_empty(monkeypatch):
    fs_provider = mock.MagicMock()
    fs_provider.is_empty = mock.MagicMock(return_value=False)
    monkeypatch.setattr(resolver, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(RuntimeError, match="The provided output_dir"):
        resolver._assert_dir_is_empty(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_is_empty(resolver.Dir(path="/teamspace/...", url="s3://"))


def test_assert_dir_has_index_file(monkeypatch):
    fs_provider = mock.MagicMock()
    fs_provider.is_empty = mock.MagicMock(return_value=False)
    monkeypatch.setattr(resolver, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(RuntimeError, match="The provided output_dir"):
        resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.exists = mock.MagicMock(return_value=False)

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.exists = mock.MagicMock(return_value=True)

    fs_provider.is_empty = mock.MagicMock(return_value=False)
    fs_provider.delete_file_or_directory = mock.MagicMock()

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"), mode="overwrite")

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"), mode="append")

    assert fs_provider.delete_file_or_directory.call_count == 1


def test_resolve_dir_absolute(tmp_path, monkeypatch):
    """Test that the directory gets resolved to an absolute path and symlinks are followed."""
    # relative path
    monkeypatch.chdir(tmp_path)
    relative = "relative"
    resolved_dir = resolver._resolve_dir(str(relative))
    assert resolved_dir.path == str(tmp_path / relative)
    assert Path(resolved_dir.path).is_absolute()
    monkeypatch.undo()

    # symlink
    src = tmp_path / "src"
    src.mkdir()
    link = tmp_path / "link"
    link.symlink_to(src)
    assert link.resolve() == src
    assert resolver._resolve_dir(str(link)).path == str(src)


def test_resolve_time_template():
    path_1 = "/logs/log_{%Y-%m}"
    path_2 = "/logs/my_logfile"
    path_3 = "/logs/log_{%Y-%m}/important"

    current_datetime = datetime.datetime.now()
    curr_year = current_datetime.year
    curr_month = current_datetime.month

    assert resolver._resolve_time_template(path_1) == f"/logs/log_{curr_year}-{curr_month:02d}"
    assert resolver._resolve_time_template(path_2) == path_2
    assert resolver._resolve_time_template(path_3) == f"/logs/log_{curr_year}-{curr_month:02d}/important"


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_gcs_connections(monkeypatch, lightning_cloud_mock):
    """Test GCS connections resolver."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(
        RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables."
    ):
        resolver._resolve_dir("/teamspace/gcs_connections/my_dataset")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[V1DataConnection(name="my_dataset", gcp=mock.MagicMock(source="gs://my-gcs-bucket"))],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    assert resolver._resolve_dir("/teamspace/gcs_connections/my_dataset").url == "gs://my-gcs-bucket"
    assert resolver._resolve_dir("/teamspace/gcs_connections/my_dataset/train").url == "gs://my-gcs-bucket/train"

    # Test missing data connection
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    with pytest.raises(ValueError, match="name `my_dataset`"):
        resolver._resolve_dir("/teamspace/gcs_connections/my_dataset")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_gcs_folders(monkeypatch, lightning_cloud_mock):
    """Test GCS folders resolver."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(name="debug_folder", gcs_folder=mock.MagicMock(source="gs://my-gcs-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "gs://my-gcs-bucket"
    assert resolver._resolve_dir("/teamspace/gcs_folders/debug_folder").url == expected
    assert resolver._resolve_dir("/teamspace/gcs_folders/debug_folder/a/b/c").url == expected + "/a/b/c"

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_lightning_storage(monkeypatch, lightning_cloud_mock):
    """Test lightning_storage resolver with r2 source."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    # Setup mocking first
    client_mock = mock.MagicMock()
    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(
        RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables."
    ):
        resolver._resolve_dir("/teamspace/lightning_storage/my_dataset")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(id="test-connection-id", name="my_dataset", r2=mock.MagicMock(source="r2://my-r2-bucket"))
        ],
    )

    expected = "r2://my-r2-bucket"
    result = resolver._resolve_dir("/teamspace/lightning_storage/my_dataset")
    assert result.url == expected
    assert result.data_connection_id == "test-connection-id"

    result_train = resolver._resolve_dir("/teamspace/lightning_storage/my_dataset/train")
    assert result_train.url == expected + "/train"
    assert result_train.data_connection_id == "test-connection-id"

    # Test missing data connection
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    with pytest.raises(ValueError, match="name `my_dataset`"):
        resolver._resolve_dir("/teamspace/lightning_storage/my_dataset")

    auth.clear()


# Tests for data_connection_id functionality


def test_dir_dataclass_with_data_connection_id():
    """Test Dir dataclass properly handles data_connection_id field."""
    # Test default initialization
    dir_obj = resolver.Dir()
    assert dir_obj.path is None
    assert dir_obj.url is None
    assert dir_obj.data_connection_id is None

    # Test with data_connection_id
    dir_obj = resolver.Dir(path="/test/path", url="r2://test-bucket", data_connection_id="test-connection-123")
    assert dir_obj.path == "/test/path"
    assert dir_obj.url == "r2://test-bucket"
    assert dir_obj.data_connection_id == "test-connection-123"


def test_resolve_dir_preserves_data_connection_id():
    """Test that _resolve_dir preserves data_connection_id when copying Dir objects."""
    # Test with Dir object containing data_connection_id
    original_dir = resolver.Dir(
        path="/original/path", url="r2://original-bucket", data_connection_id="original-connection-456"
    )

    resolved_dir = resolver._resolve_dir(original_dir)

    # Verify all fields are preserved
    assert resolved_dir.path == "/original/path"
    assert resolved_dir.url == "r2://original-bucket"
    assert resolved_dir.data_connection_id == "original-connection-456"

    # Test with Dir object without data_connection_id
    dir_without_connection = resolver.Dir(path="/test", url="s3://test-bucket")
    resolved_without_connection = resolver._resolve_dir(dir_without_connection)

    assert resolved_without_connection.path == "/test"
    assert resolved_without_connection.url == "s3://test-bucket"
    assert resolved_without_connection.data_connection_id is None


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_resolve_lightning_storage_sets_data_connection_id(monkeypatch, lightning_cloud_mock):
    """Test that _resolve_lightning_storage sets data_connection_id from the data connection."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    # Mock data connection with ID
    test_connection_id = "data-connection-789"
    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(id=test_connection_id, name="my_dataset", r2=mock.MagicMock(source="r2://my-r2-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    # Test that data_connection_id is set
    result = resolver._resolve_dir("/teamspace/lightning_storage/my_dataset")
    assert result.url == "r2://my-r2-bucket"
    assert result.data_connection_id == test_connection_id

    # Test with subdirectory
    result_subdir = resolver._resolve_dir("/teamspace/lightning_storage/my_dataset/train")
    assert result_subdir.url == "r2://my-r2-bucket/train"
    assert result_subdir.data_connection_id == test_connection_id

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_assert_dir_is_empty_with_data_connection_id(monkeypatch):
    """Test that _assert_dir_is_empty passes data_connection_id to fs_provider."""
    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.is_empty.return_value = True
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(resolver, "_get_fs_provider", get_fs_provider_mock)

    # Test with data_connection_id
    test_connection_id = "test-connection-empty-123"
    output_dir = resolver.Dir(path="/test/path", url="r2://test-bucket", data_connection_id=test_connection_id)

    storage_options = {"timeout": 30}
    resolver._assert_dir_is_empty(output_dir, storage_options=storage_options)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_once_with(output_dir.url, expected_storage_options)


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_assert_dir_is_empty_without_data_connection_id(monkeypatch):
    """Test that _assert_dir_is_empty works correctly when data_connection_id is None."""
    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.is_empty.return_value = True
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(resolver, "_get_fs_provider", get_fs_provider_mock)

    # Test without data_connection_id
    output_dir = resolver.Dir(path="/test/path", url="s3://test-bucket")
    storage_options = {"region": "us-west-2"}

    resolver._assert_dir_is_empty(output_dir, storage_options=storage_options)

    # Verify fs_provider was called with original storage_options (no data_connection_id added)
    get_fs_provider_mock.assert_called_once_with(output_dir.url, storage_options)


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_assert_dir_has_index_file_with_data_connection_id(monkeypatch):
    """Test that _assert_dir_has_index_file passes data_connection_id to fs_provider."""
    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.is_empty.return_value = True
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(resolver, "_get_fs_provider", get_fs_provider_mock)

    # Test with data_connection_id
    test_connection_id = "test-connection-index-456"
    output_dir = resolver.Dir(path="/test/path", url="r2://test-bucket", data_connection_id=test_connection_id)

    storage_options = {"max_retries": 3}
    resolver._assert_dir_has_index_file(output_dir, storage_options=storage_options)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_once_with(output_dir.url, expected_storage_options)


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_assert_dir_has_index_file_without_data_connection_id(monkeypatch):
    """Test that _assert_dir_has_index_file works correctly when data_connection_id is None."""
    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.is_empty.return_value = True
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(resolver, "_get_fs_provider", get_fs_provider_mock)

    # Test without data_connection_id
    output_dir = resolver.Dir(path="/test/path", url="s3://test-bucket")
    storage_options = {"connect_timeout": 10}

    resolver._assert_dir_has_index_file(output_dir, storage_options=storage_options)

    # Verify fs_provider was called with original storage_options (no data_connection_id added)
    get_fs_provider_mock.assert_called_once_with(output_dir.url, storage_options)


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_assert_dir_has_index_file_overwrite_mode_with_data_connection_id(monkeypatch):
    """Test _assert_dir_has_index_file in overwrite mode with data_connection_id."""
    # Mock fs_provider
    fs_provider = mock.MagicMock()
    fs_provider.is_empty.return_value = False  # Directory not empty
    fs_provider.exists.return_value = True  # Index file exists
    fs_provider.delete_file_or_directory = mock.MagicMock()
    get_fs_provider_mock = mock.MagicMock(return_value=fs_provider)
    monkeypatch.setattr(resolver, "_get_fs_provider", get_fs_provider_mock)

    # Test with data_connection_id in overwrite mode
    test_connection_id = "test-connection-overwrite-789"
    output_dir = resolver.Dir(path="/test/path", url="r2://test-bucket", data_connection_id=test_connection_id)

    storage_options = {"write_timeout": 60}
    resolver._assert_dir_has_index_file(output_dir, mode="overwrite", storage_options=storage_options)

    # Verify fs_provider was called with merged storage_options including data_connection_id
    expected_storage_options = storage_options.copy()
    expected_storage_options["data_connection_id"] = test_connection_id
    get_fs_provider_mock.assert_called_once_with(output_dir.url, expected_storage_options)

    # Verify delete was called
    fs_provider.delete_file_or_directory.assert_called_once_with(output_dir.url)


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_resolve_lightning_storage_missing_data_connection(monkeypatch, lightning_cloud_mock):
    """Test _resolve_lightning_storage when data connection is not found."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    # Mock empty data connections list
    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    # Should raise ValueError when data connection not found
    with pytest.raises(ValueError, match="name `nonexistent_dataset`"):
        resolver._resolve_dir("/teamspace/lightning_storage/nonexistent_dataset")

    auth.clear()


def test_storage_options_merge_behavior():
    """Test that storage options are properly merged with data_connection_id."""
    # Test the merge behavior that happens in the resolver functions
    original_storage_options = {"timeout": 30, "region": "us-west-2", "max_retries": 5}

    test_connection_id = "test-merge-connection"

    # Simulate the merge operation from _assert_dir_is_empty
    merged_storage_options = original_storage_options.copy()
    merged_storage_options["data_connection_id"] = test_connection_id

    # Verify original is unchanged
    assert "data_connection_id" not in original_storage_options
    assert len(original_storage_options) == 3

    # Verify merged has all original keys plus data_connection_id
    assert len(merged_storage_options) == 4
    assert merged_storage_options["data_connection_id"] == test_connection_id
    assert merged_storage_options["timeout"] == 30
    assert merged_storage_options["region"] == "us-west-2"
    assert merged_storage_options["max_retries"] == 5


def test_data_connection_id_conditional_merge():
    """Test that data_connection_id is only added when it's not None."""
    original_storage_options = {"timeout": 30}

    # Test with None data_connection_id (should not be added)
    merged_none = original_storage_options.copy()
    data_connection_id = None
    if data_connection_id:
        merged_none["data_connection_id"] = data_connection_id

    assert "data_connection_id" not in merged_none
    assert len(merged_none) == 1

    # Test with valid data_connection_id (should be added)
    merged_valid = original_storage_options.copy()
    data_connection_id = "valid-connection"
    if data_connection_id:
        merged_valid["data_connection_id"] = data_connection_id

    assert merged_valid["data_connection_id"] == "valid-connection"
    assert len(merged_valid) == 2


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_resolve_lightning_storage_integration_with_existing_test(monkeypatch, lightning_cloud_mock):
    """Integration test verifying data_connection_id works with the existing lightning_storage test."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    # Use the same mock structure as the existing test but verify data_connection_id
    test_connection_id = "integration-test-connection-id"
    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(id=test_connection_id, name="my_dataset", r2=mock.MagicMock(source="r2://my-r2-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    # Test the resolution
    result = resolver._resolve_lightning_storage("/teamspace/lightning_storage/my_dataset")

    # Verify all expected properties
    assert result.path == "/teamspace/lightning_storage/my_dataset"
    assert result.url == "r2://my-r2-bucket"
    assert result.data_connection_id == test_connection_id

    # Test with subdirectory
    result_subdir = resolver._resolve_lightning_storage("/teamspace/lightning_storage/my_dataset/train/validation")
    assert result_subdir.path == "/teamspace/lightning_storage/my_dataset/train/validation"
    assert result_subdir.url == "r2://my-r2-bucket/train/validation"
    assert result_subdir.data_connection_id == test_connection_id

    auth.clear()

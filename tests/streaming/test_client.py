import logging
import sys
import threading
from time import sleep, time
from unittest import mock

import pytest
import requests

from litdata.streaming import client


@pytest.fixture(autouse=True)
def _clear_temp_bucket_credentials_cache():
    """Isolate HTTP mocks: a warm process cache must not satisfy a later test."""
    client.clear_temp_bucket_credentials_cache()
    yield
    client.clear_temp_bucket_credentials_cache()


def test_s3_client_with_storage_options(monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Create S3Client with storage options
    storage_options = {
        "region_name": "us-west-2",
        "endpoint_url": "https://custom.endpoint",
        "config": botocore.config.Config(retries={"max_attempts": 100}),
    }
    s3_client = client.S3Client(storage_options=storage_options)

    assert s3_client.client

    boto3_session().client.assert_called_with(
        "s3",
        region_name="us-west-2",
        endpoint_url="https://custom.endpoint",
        config=botocore.config.Config(retries={"max_attempts": 100}),
    )

    # Create S3Client without storage options (force non-Studio path so IMDS is not used).
    monkeypatch.setattr(client, "_IS_IN_STUDIO", False)
    s3_client = client.S3Client()
    assert s3_client.client

    # Verify that boto3.Session().client was called with the default parameters
    boto3_session().client.assert_called_with(
        "s3",
        config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
    )


def test_s3_client_without_cloud_space_id(monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client

    boto3_session().client.assert_called_once()


def test_s3_client_pickle_drops_boto_client():
    import pickle

    s3 = client.S3Client()
    s3._client = mock.sentinel.live
    restored = pickle.loads(pickle.dumps(s3))  # noqa: S301
    assert restored._client is None
    assert restored._client_lock is not None


def test_s3_client_recreates_after_pid_change(monkeypatch):
    import os

    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)
    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)
    monkeypatch.setattr(client, "_IS_IN_STUDIO", False)

    s3 = client.S3Client()
    _ = s3.client
    s3._owner_pid = os.getpid() + 1
    s3._client = mock.sentinel.stale
    second = s3.client
    assert second is not mock.sentinel.stale
    assert boto3_session().client.call_count == 2


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows")
@pytest.mark.parametrize("use_shared_credentials", [False, True, None])
def test_s3_client_with_cloud_space_id(use_shared_credentials, monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    if isinstance(use_shared_credentials, bool):
        monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "dummy")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/.credentials/.aws_credentials")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/.credentials/.aws_credentials")

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    boto3_session().client.assert_called_once()
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3_session().client._mock_mock_calls) == 6
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3_session().client._mock_mock_calls) == 9

    assert instance_metadata_provider._mock_call_count == 0 if use_shared_credentials else 3


# Tests for R2Client functionality


def test_r2_client_initialization():
    """Test R2Client initialization with different parameters."""
    # Test with default parameters
    r2_client = client.R2Client()
    assert r2_client._refetch_interval == 2700
    assert r2_client._last_time is None
    assert r2_client._client is None
    assert r2_client._base_storage_options == {}
    assert r2_client._session_options == {}

    # Test with custom parameters
    storage_options = {"data_connection_id": "test-connection-123"}
    session_options = {"region_name": "us-west-2"}
    r2_client = client.R2Client(refetch_interval=1800, storage_options=storage_options, session_options=session_options)
    assert r2_client._refetch_interval == 1800
    assert r2_client._base_storage_options == storage_options
    assert r2_client._base_storage_options is not storage_options
    assert r2_client._session_options == session_options


def test_r2_client_missing_data_connection_id(monkeypatch):
    """Test R2Client raises error when data_connection_id is missing."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Create R2Client without data_connection_id
    r2_client = client.R2Client(storage_options={})

    # Accessing client should raise error
    with pytest.raises(RuntimeError, match="data_connection_id is required"):
        _ = r2_client.client


def test_r2_client_get_r2_bucket_credentials_success(monkeypatch):
    """Test successful R2 credential fetching."""
    # Mock environment variables
    monkeypatch.setenv("LIGHTNING_CLOUD_URL", "https://test.lightning.ai")
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-api-key")
    monkeypatch.setenv("LIGHTNING_USERNAME", "test-user")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "test-project-123")

    # Mock requests
    requests_mock = mock.MagicMock()
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))

    # Mock login response
    login_response = mock.MagicMock()
    login_response.status_code = 200
    login_response.json.return_value = {"token": "test-token-456"}

    # Mock credentials response
    credentials_response = mock.MagicMock()
    credentials_response.status_code = 200
    credentials_response.json.return_value = {
        "accessKeyId": "test-access-key",
        "secretAccessKey": "test-secret-key",
        "sessionToken": "test-session-token",
        "accountId": "test-account-id",
    }

    # Configure mock to return different responses for different calls
    def mock_request(*args, **kwargs):
        if "auth/login" in args[0]:
            return login_response
        return credentials_response

    requests_mock.post = mock_request
    requests_mock.get = mock_request

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: credentials_response)

    r2_client = client.R2Client()
    credentials = r2_client.get_r2_bucket_credentials("test-connection-789")

    expected_credentials = {
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
        "endpoint_url": "https://test-account-id.r2.cloudflarestorage.com",
    }

    assert credentials == expected_credentials


def test_r2_client_get_r2_bucket_credentials_missing_env_vars(monkeypatch):
    """Test R2 credential fetching fails with missing environment variables."""
    # Don't set required environment variables
    monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
    monkeypatch.delenv("LIGHTNING_USERNAME", raising=False)
    monkeypatch.delenv("LIGHTNING_CLOUD_PROJECT_ID", raising=False)

    r2_client = client.R2Client()

    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        r2_client.get_r2_bucket_credentials("test-connection")


def _mock_login_env(monkeypatch):
    monkeypatch.setenv("LIGHTNING_CLOUD_URL", "https://test.lightning.ai")
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-api-key")
    monkeypatch.setenv("LIGHTNING_USERNAME", "test-user")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "test-project-123")


def test_r2_client_get_r2_bucket_credentials_login_rejected(monkeypatch):
    """A non-200 from the login endpoint reports the status, not a missing-token error."""
    _mock_login_env(monkeypatch)

    login_response = mock.MagicMock()
    login_response.status_code = 401

    requests_mock = mock.MagicMock()
    requests_mock.post = mock.MagicMock(return_value=login_response)
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))

    r2_client = client.R2Client()

    with pytest.raises(RuntimeError, match="Failed to log in to the Lightning Cloud API: 401"):
        r2_client.get_r2_bucket_credentials("test-connection")


@pytest.mark.parametrize("body", [{"error": "Invalid credentials"}, ValueError("not json")])
def test_r2_client_get_r2_bucket_credentials_login_without_token(body, monkeypatch):
    """A 200 login that carries no usable token is reported as a missing token."""
    _mock_login_env(monkeypatch)

    login_response = mock.MagicMock()
    login_response.status_code = 200
    if isinstance(body, Exception):
        login_response.json.side_effect = body
    else:
        login_response.json.return_value = body

    requests_mock = mock.MagicMock()
    requests_mock.post = mock.MagicMock(return_value=login_response)
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))

    r2_client = client.R2Client()

    with pytest.raises(RuntimeError, match="Failed to get authentication token"):
        r2_client.get_r2_bucket_credentials("test-connection")


def test_r2_client_get_r2_bucket_credentials_api_failure(monkeypatch):
    """Test R2 credential fetching fails when credentials API fails."""
    # Mock environment variables
    monkeypatch.setenv("LIGHTNING_CLOUD_URL", "https://test.lightning.ai")
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-api-key")
    monkeypatch.setenv("LIGHTNING_USERNAME", "test-user")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "test-project-123")

    # Mock successful login response
    login_response = mock.MagicMock()
    login_response.status_code = 200
    login_response.json.return_value = {"token": "test-token-456"}

    # Mock failed credentials response
    credentials_response = mock.MagicMock()
    credentials_response.status_code = 403

    # Mock requests
    requests_mock = mock.MagicMock()
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))
    requests_mock.post = mock.MagicMock(return_value=login_response)
    requests_mock.get = mock.MagicMock(return_value=credentials_response)

    r2_client = client.R2Client()

    with pytest.raises(RuntimeError, match="Failed to get credentials: 403"):
        r2_client.get_r2_bucket_credentials("test-connection")


def test_r2_client_create_client_success(monkeypatch):
    """Test successful R2 client creation."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Mock the credential fetching method
    mock_credentials = {
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
        "endpoint_url": "https://test-account.r2.cloudflarestorage.com",
    }

    r2_client = client.R2Client(storage_options={"data_connection_id": "test-connection"})
    r2_client.get_r2_bucket_credentials = mock.MagicMock(return_value=mock_credentials)

    # Call _create_client
    r2_client._create_client()

    # Verify boto3 session was created and client was configured correctly
    boto3_session.assert_called_once()
    boto3_session().client.assert_called_once_with(
        "s3",
        config=client._r2_botocore_config(),
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
        endpoint_url="https://test-account.r2.cloudflarestorage.com",
    )


def test_s3_client_uses_temp_credentials_with_data_connection_id(monkeypatch):
    """S3Client should mint temporary project-role creds when a data_connection_id is provided.

    This is the path for S3 connections marked available on non-AWS providers.
    """
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # The S3 temp-credentials response has no accountId (real AWS S3, no custom endpoint).
    temp_credentials = {
        "accessKeyId": "test-access-key",
        "secretAccessKey": "test-secret-key",
        "sessionToken": "test-session-token",
    }
    monkeypatch.setattr(client, "_login_and_get_temp_bucket_credentials", mock.MagicMock(return_value=temp_credentials))

    s3_client = client.S3Client(storage_options={"data_connection_id": "test-connection", "region_name": "us-west-2"})
    assert s3_client.client

    # data_connection_id is dropped before boto3; temp creds + remaining options are forwarded.
    boto3_session().client.assert_called_with(
        "s3",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
        config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
        region_name="us-west-2",
    )


def test_r2_client_filters_metadata_from_storage_options(monkeypatch):
    """Test that R2Client filters out metadata keys from storage options."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Mock the credential fetching method
    mock_credentials = {
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
        "endpoint_url": "https://test-account.r2.cloudflarestorage.com",
    }

    storage_options = {"data_connection_id": "test-connection", "timeout": 30, "region_name": "auto"}

    r2_client = client.R2Client(storage_options=storage_options)
    r2_client.get_r2_bucket_credentials = mock.MagicMock(return_value=mock_credentials)

    # Call _create_client
    r2_client._create_client()

    # Verify that data_connection_id was filtered out but other options were preserved
    expected_call_kwargs = {
        "config": client._r2_botocore_config(),
        "timeout": 30,
        "region_name": "auto",
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
        "endpoint_url": "https://test-account.r2.cloudflarestorage.com",
    }

    boto3_session().client.assert_called_once_with("s3", **expected_call_kwargs)


def test_r2_client_keeps_data_connection_id_when_caller_pops_shared_dict():
    """R2Client must copy storage_options so a later pop cannot starve _create_client."""
    storage_options = {"data_connection_id": "conn-shared", "timeout": 30}
    r2_client = client.R2Client(storage_options=storage_options)
    storage_options.pop("data_connection_id")
    assert r2_client._base_storage_options["data_connection_id"] == "conn-shared"
    assert "data_connection_id" not in storage_options


def test_r2_client_property_creates_client_on_first_access(monkeypatch):
    """Test that accessing client property creates client on first access."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    r2_client = client.R2Client(storage_options={"data_connection_id": "test-connection"})
    r2_client.get_r2_bucket_credentials = mock.MagicMock(
        return_value={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_session_token": "test-token",
            "endpoint_url": "https://test.r2.cloudflarestorage.com",
        }
    )

    # Initially no client
    assert r2_client._client is None
    assert r2_client._last_time is None

    # Access client property
    client_instance = r2_client.client

    # Verify client was created
    assert r2_client._client is not None
    assert r2_client._last_time is not None
    assert client_instance == r2_client._client


def test_r2_client_property_refreshes_expired_credentials(monkeypatch):
    """Test that client property refreshes credentials when they expire."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Set short refresh interval for testing
    r2_client = client.R2Client(
        refetch_interval=1,  # 1 second
        storage_options={"data_connection_id": "test-connection"},
    )
    r2_client.get_r2_bucket_credentials = mock.MagicMock(
        return_value={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_session_token": "test-token",
            "endpoint_url": "https://test.r2.cloudflarestorage.com",
        }
    )

    # First access
    r2_client.client
    first_call_count = boto3_session().client.call_count

    # Wait for credentials to expire
    sleep(1.1)

    # Second access should refresh credentials
    r2_client.client
    second_call_count = boto3_session().client.call_count

    # Verify client was created twice (initial + refresh)
    assert second_call_count == first_call_count + 1


def test_s3_client_refresh_is_serialized_under_threads(monkeypatch):
    """Concurrent .client access at a refresh boundary must not race-create clients."""
    in_create = {"n": 0, "max": 0}
    counter_lock = threading.Lock()
    barrier = threading.Barrier(8)

    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    s3 = client.S3Client(refetch_interval=0, storage_options={"region_name": "us-east-1"})
    original_create = s3._create_client

    def slow_create() -> None:
        with counter_lock:
            in_create["n"] += 1
            in_create["max"] = max(in_create["max"], in_create["n"])
        try:
            sleep(0.01)
            original_create()
        finally:
            with counter_lock:
                in_create["n"] -= 1

    s3._create_client = slow_create  # type: ignore[method-assign]

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(3):
                assert s3.client is not None
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert in_create["max"] == 1


def test_r2_client_with_session_options(monkeypatch):
    """Test R2Client with custom session options."""
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    session_options = {"profile_name": "test-profile"}
    r2_client = client.R2Client(
        storage_options={"data_connection_id": "test-connection"}, session_options=session_options
    )
    r2_client.get_r2_bucket_credentials = mock.MagicMock(
        return_value={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_session_token": "test-token",
            "endpoint_url": "https://test.r2.cloudflarestorage.com",
        }
    )

    # Access client to trigger creation
    r2_client.client

    # Verify session was created with custom options
    boto3.Session.assert_called_once_with(profile_name="test-profile")


def test_r2_client_api_call_format(monkeypatch):
    """Test that R2Client makes correct API calls for credential fetching."""
    # Mock environment variables
    monkeypatch.setenv("LIGHTNING_CLOUD_URL", "https://api.lightning.ai")
    monkeypatch.setenv("LIGHTNING_API_KEY", "sk-test123")
    monkeypatch.setenv("LIGHTNING_USERNAME", "testuser")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "proj-456")

    # Mock requests
    mock_post = mock.MagicMock()
    mock_get = mock.MagicMock()

    # Mock login response
    login_response = mock.MagicMock()
    login_response.status_code = 200
    login_response.json.return_value = {"token": "bearer-token-789"}
    mock_post.return_value = login_response

    # Mock credentials response
    credentials_response = mock.MagicMock()
    credentials_response.status_code = 200
    credentials_response.json.return_value = {
        "accessKeyId": "AKIATEST123",
        "secretAccessKey": "secrettest456",
        "sessionToken": "sessiontest789",
        "accountId": "account123",
    }
    mock_get.return_value = credentials_response

    requests_mock = mock.MagicMock()
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))
    requests_mock.post = mock_post
    requests_mock.get = mock_get

    r2_client = client.R2Client()
    r2_client.get_r2_bucket_credentials("conn-abc123")

    # Verify login API call
    mock_post.assert_called_once_with(
        "https://api.lightning.ai/v1/auth/login", data='{"apiKey": "sk-test123", "username": "testuser"}'
    )

    # Verify credentials API call
    mock_get.assert_called_once_with(
        "https://api.lightning.ai/v1/projects/proj-456/data-connections/conn-abc123/temp-bucket-credentials",
        headers={"Authorization": "Bearer bearer-token-789", "Content-Type": "application/json"},
        timeout=10,
    )


def _successful_login_session(monkeypatch):
    """Wire requests.Session so a full credential fetch succeeds, and hand back the mock."""
    login_response = mock.MagicMock()
    login_response.status_code = 200
    login_response.json.return_value = {"token": "test-token"}

    credentials_response = mock.MagicMock()
    credentials_response.status_code = 200
    credentials_response.json.return_value = {
        "accessKeyId": "test-access-key",
        "secretAccessKey": "test-secret-key",
        "sessionToken": "test-session-token",
        "accountId": "test-account-id",
    }

    requests_mock = mock.MagicMock()
    requests_mock.post = mock.MagicMock(return_value=login_response)
    requests_mock.get = mock.MagicMock(return_value=credentials_response)
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))
    return requests_mock


def test_login_post_is_retried(monkeypatch):
    """urllib3 leaves POST out of its default allowed_methods, so the login must opt in."""
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)

    client._login_and_get_temp_bucket_credentials("test-connection")

    mounted_adapters = [mount_call.args[1] for mount_call in requests_mock.mount.call_args_list]
    assert mounted_adapters
    for adapter in mounted_adapters:
        assert "POST" in adapter.max_retries.allowed_methods
        assert 429 in adapter.max_retries.status_forcelist


def _client_with_failing_refresh(monkeypatch, refetch_interval=0):
    """An S3Client holding a live client whose next refresh will fail."""
    boto3_session = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", mock.MagicMock(Session=boto3_session))
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    s3 = client.S3Client(refetch_interval=refetch_interval, storage_options={"region_name": "us-east-1"})
    live_client = s3.client
    # Windows resolves time.time() to ~15ms, so `elapsed > deadline` can still be False on the
    # next access. Age the stamp rather than depending on the clock having ticked.
    s3._last_time -= 1

    attempts = {"n": 0}

    def failing_create():
        attempts["n"] += 1
        raise client._CredentialsUnavailableError("control plane unavailable")

    s3._create_client = failing_create
    return s3, live_client, attempts


def test_failed_refresh_keeps_serving_the_current_client(monkeypatch, caplog):
    """Credentials are refreshed early, so a failed refresh must not fail the read."""
    s3, live_client, attempts = _client_with_failing_refresh(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="litdata.streaming.client"):
        assert s3.client is live_client

    assert attempts["n"] == 1
    assert "reusing the current ones" in caplog.text


def test_failed_refresh_is_not_retried_on_every_access(monkeypatch):
    """One outage must not become a request storm from every worker."""
    s3, live_client, attempts = _client_with_failing_refresh(monkeypatch)

    for _ in range(10):
        assert s3.client is live_client

    assert attempts["n"] == 1


def test_failed_refresh_raises_once_past_the_grace_period(monkeypatch):
    """Past the grace period the credentials are assumed dead, so stop pretending."""
    s3, _, _ = _client_with_failing_refresh(monkeypatch)
    s3._last_time = time() - (client._REFRESH_GRACE_PERIOD + 60)

    with pytest.raises(RuntimeError, match="assumed expired"):
        _ = s3.client


def test_refetch_deadline_is_jittered_below_the_interval(monkeypatch):
    """Forked workers all reach the interval together, so each refreshes slightly early."""
    monkeypatch.setattr(client, "boto3", mock.MagicMock())
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    deadlines = {client.S3Client(refetch_interval=3600)._refetch_deadline for _ in range(20)}

    assert len(deadlines) > 1
    assert all(3600 * (1 - client._REFETCH_JITTER_RATIO) <= deadline <= 3600 for deadline in deadlines)


def test_unpickled_client_rerolls_its_refresh_jitter():
    """A DataLoader worker inherits the parent's schedule unless the jitter is re-rolled."""
    import pickle

    s3 = client.S3Client(refetch_interval=3600)
    restored = [pickle.loads(pickle.dumps(s3)) for _ in range(20)]  # noqa: S301

    assert len({r._refetch_deadline for r in restored} | {s3._refetch_deadline}) > 1


def _s3_client_failing_n_times(monkeypatch, failures):
    """An S3Client whose first `failures` creation attempts fail, then succeed."""
    boto3_session = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", mock.MagicMock(Session=boto3_session))
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    s3 = client.S3Client(storage_options={"region_name": "us-east-1"})
    real_create = s3._create_client
    attempts = {"n": 0}

    def flaky_create():
        attempts["n"] += 1
        if attempts["n"] <= failures:
            raise client._CredentialsUnavailableError("control plane unavailable")
        real_create()

    s3._create_client = flaky_create
    return s3, attempts


def test_initial_creation_retries_until_the_control_plane_returns(monkeypatch, caplog):
    """The first client has nothing to fall back on, so it waits the outage out."""
    monkeypatch.setattr(client, "_REFRESH_RETRY_INTERVAL", 0)
    s3, attempts = _s3_client_failing_n_times(monkeypatch, failures=3)

    with caplog.at_level(logging.WARNING, logger="litdata.streaming.client"):
        assert s3.client is not None

    assert attempts["n"] == 4
    assert caplog.text.count("data loading is blocked") == 3


def test_initial_creation_gives_up_after_the_grace_period(monkeypatch):
    """Waiting is bounded: a control plane that never comes back fails with a clear reason."""
    monkeypatch.setattr(client, "_REFRESH_RETRY_INTERVAL", 0)
    monkeypatch.setattr(client, "_INITIAL_RETRY_BUDGET", 0)
    s3, attempts = _s3_client_failing_n_times(monkeypatch, failures=99)

    with pytest.raises(RuntimeError, match="Could not get credentials after"):
        _ = s3.client

    assert attempts["n"] == 1


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (client._CredentialsConfigurationError("data_connection_id is required"), "data_connection_id is required"),
        (client._credentials_error(403, "Failed to get credentials: 403"), "Failed to get credentials: 403"),
    ],
)
def test_initial_creation_does_not_retry_a_permanent_failure(failure, match, monkeypatch):
    """Missing config or rejected auth must fail now, not after minutes of pointless retrying."""
    monkeypatch.setattr(client, "boto3", mock.MagicMock())
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    s3 = client.S3Client(storage_options={"region_name": "us-east-1"})
    attempts = {"n": 0}

    def failing_create():
        attempts["n"] += 1
        raise failure

    s3._create_client = failing_create

    with pytest.raises(RuntimeError, match=match):
        _ = s3.client

    assert attempts["n"] == 1


def test_refresh_rides_out_a_rejected_response(monkeypatch):
    """A 403 mid-refresh may be a proxy misbehaving, and the current credentials still work."""
    s3, live_client, _ = _client_with_failing_refresh(monkeypatch)

    def rejected_create():
        raise client._credentials_error(403, "Failed to get credentials: 403")

    s3._create_client = rejected_create

    assert s3.client is live_client

    # ...but the deadline still catches a real revocation.
    s3._last_time = time() - (client._REFRESH_GRACE_PERIOD + 60)
    s3._refresh_retry_time = None
    with pytest.raises(RuntimeError, match="assumed expired"):
        _ = s3.client


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_statuses_that_may_clear_are_retryable(status):
    """408 and 429 are the 4xx that do fix themselves; 5xx always might."""
    assert isinstance(client._credentials_error(status, "x"), client._CredentialsUnavailableError)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_statuses_that_will_not_clear_are_permanent(status):
    assert isinstance(client._credentials_error(status, "x"), client._CredentialsConfigurationError)


def test_local_failures_are_not_retried(monkeypatch):
    """A bad storage_options key is a caller mistake, not an outage: fail on the first attempt."""
    monkeypatch.setattr(client, "_REFRESH_RETRY_INTERVAL", 0)
    monkeypatch.setattr(client, "boto3", mock.MagicMock())
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    s3 = client.S3Client(storage_options={"region_name": "us-east-1"})
    attempts = {"n": 0}

    def bad_kwarg_create():
        attempts["n"] += 1
        raise TypeError("client() got an unexpected keyword argument 'bogus_option'")

    s3._create_client = bad_kwarg_create

    with pytest.raises(TypeError, match="bogus_option"):
        _ = s3.client

    assert attempts["n"] == 1


def test_adapter_applies_its_default_timeout(monkeypatch):
    """Requests passes timeout=None explicitly, so the adapter has to fill it in itself."""
    captured = {}

    class _Recorder(client._CustomRetryAdapter):
        def send(self, request, *args, **kwargs):
            super().send(request, *args, **kwargs)

    def fake_send(self, request, **kwargs):
        captured.update(kwargs)
        raise requests.exceptions.ConnectionError("stop")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)

    session = requests.Session()
    session.mount("http://", _Recorder(timeout=client._DEFAULT_REQUEST_TIMEOUT))
    with pytest.raises(requests.exceptions.ConnectionError):
        session.post("http://127.0.0.1:1/x", data="{}")

    assert captured["timeout"] == client._DEFAULT_REQUEST_TIMEOUT


def test_temp_bucket_credentials_are_cached_per_connection(monkeypatch):
    """A second login for the same data_connection_id must not hit the control plane."""
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)

    first = client._login_and_get_temp_bucket_credentials("conn-a")
    second = client._login_and_get_temp_bucket_credentials("conn-a")
    other = client._login_and_get_temp_bucket_credentials("conn-b")

    assert first == second
    assert first["accessKeyId"] == "test-access-key"
    assert other["accessKeyId"] == "test-access-key"
    assert requests_mock.post.call_count == 2
    assert requests_mock.get.call_count == 2


def test_temp_bucket_credentials_cache_clear_forces_refetch(monkeypatch):
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)

    client._login_and_get_temp_bucket_credentials("conn-a")
    client.clear_temp_bucket_credentials_cache()
    client._login_and_get_temp_bucket_credentials("conn-a")

    assert requests_mock.post.call_count == 2


def test_temp_bucket_credentials_force_refresh_bypasses_cache(monkeypatch):
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)

    client._login_and_get_temp_bucket_credentials("conn-a")
    client._login_and_get_temp_bucket_credentials("conn-a", force_refresh=True)

    assert requests_mock.post.call_count == 2


def test_temp_bucket_credentials_ttl_expiry_refetches(monkeypatch):
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)
    monkeypatch.setattr(client, "_DEFAULT_REFETCH_INTERVAL", 10)

    now = {"t": 1000.0}
    monkeypatch.setattr(client, "time", lambda: now["t"])

    client._login_and_get_temp_bucket_credentials("conn-a")
    now["t"] = 1009.0
    client._login_and_get_temp_bucket_credentials("conn-a")
    assert requests_mock.post.call_count == 1

    now["t"] = 1010.0
    client._login_and_get_temp_bucket_credentials("conn-a")
    assert requests_mock.post.call_count == 2


def test_temp_bucket_credentials_failed_fetch_is_not_cached(monkeypatch):
    _mock_login_env(monkeypatch)
    login_response = mock.MagicMock()
    login_response.status_code = 503
    requests_mock = mock.MagicMock()
    requests_mock.post = mock.MagicMock(return_value=login_response)
    monkeypatch.setattr("requests.Session", mock.MagicMock(return_value=requests_mock))

    with pytest.raises(client._CredentialsUnavailableError, match="Failed to log in"):
        client._login_and_get_temp_bucket_credentials("conn-a")
    assert client._temp_creds_cache == {}


def test_two_r2_clients_share_cached_credentials(monkeypatch):
    """A new R2Client in the same process must not login again (bench warmup vs timed pass)."""
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)
    boto3_session = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", mock.MagicMock(Session=boto3_session))
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    first = client.R2Client(storage_options={"data_connection_id": "conn-shared"})
    second = client.R2Client(storage_options={"data_connection_id": "conn-shared"})
    assert first.client is not None
    assert second.client is not None
    assert requests_mock.post.call_count == 1
    assert boto3_session().client.call_count == 1
    assert second.client is first.client


def test_r2_client_refresh_mints_new_credentials(monkeypatch):
    """Scheduled refresh must bypass the process cache so creds are not held past TTL."""
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)
    boto3_session = mock.MagicMock()
    monkeypatch.setattr(client, "boto3", mock.MagicMock(Session=boto3_session))
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    r2 = client.R2Client(refetch_interval=1, storage_options={"data_connection_id": "conn-refresh"})
    _ = r2.client
    assert requests_mock.post.call_count == 1
    r2._last_time -= 2
    _ = r2.client
    assert requests_mock.post.call_count == 2


def test_cached_credentials_reuse_original_fetched_at(monkeypatch):
    """A new client that hits the cache must age credentials from mint time, not now."""
    _mock_login_env(monkeypatch)
    _successful_login_session(monkeypatch)
    monkeypatch.setattr(client, "boto3", mock.MagicMock())
    monkeypatch.setattr(client, "botocore", mock.MagicMock())

    first = client.R2Client(storage_options={"data_connection_id": "conn-age"})
    _ = first.client
    minted = first._creds_fetched_at
    assert minted is not None

    second = client.R2Client(storage_options={"data_connection_id": "conn-age"})
    _ = second.client
    assert second._creds_fetched_at == minted
    assert second._last_time == minted


def test_temp_bucket_credentials_concurrent_first_access_fetches_once(monkeypatch):
    _mock_login_env(monkeypatch)
    requests_mock = _successful_login_session(monkeypatch)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            client._login_and_get_temp_bucket_credentials("conn-race")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert requests_mock.post.call_count == 1

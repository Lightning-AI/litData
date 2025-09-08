import sys
from time import sleep, time
from unittest import mock

import pytest

from litdata.streaming import client


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

    # Create S3Client without storage options
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
    assert r2_client._refetch_interval == 3600  # 1 hour default
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


def test_r2_client_get_r2_bucket_credentials_login_failure(monkeypatch):
    """Test R2 credential fetching fails when login fails."""
    # Mock environment variables
    monkeypatch.setenv("LIGHTNING_CLOUD_URL", "https://test.lightning.ai")
    monkeypatch.setenv("LIGHTNING_API_KEY", "test-api-key")
    monkeypatch.setenv("LIGHTNING_USERNAME", "test-user")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "test-project-123")

    # Mock failed login response
    login_response = mock.MagicMock()
    login_response.json.return_value = {"error": "Invalid credentials"}

    requests_mock = mock.MagicMock(return_value=login_response)
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
        config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
        endpoint_url="https://test-account.r2.cloudflarestorage.com",
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
        "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
        "timeout": 30,
        "region_name": "auto",
        "aws_access_key_id": "test-access-key",
        "aws_secret_access_key": "test-secret-key",
        "aws_session_token": "test-session-token",
        "endpoint_url": "https://test-account.r2.cloudflarestorage.com",
    }

    boto3_session().client.assert_called_once_with("s3", **expected_call_kwargs)


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

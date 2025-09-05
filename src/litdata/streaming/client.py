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
from time import time
from typing import Any, Optional

import boto3
import botocore
from botocore.credentials import InstanceMetadataProvider
from botocore.utils import InstanceMetadataFetcher

from litdata.constants import _IS_IN_STUDIO


class S3Client:
    # TODO: Generalize to support more cloud providers.

    def __init__(
        self,
        refetch_interval: int = 3300,
        storage_options: Optional[dict] = {},
        session_options: Optional[dict] = {},
    ) -> None:
        self._refetch_interval = refetch_interval
        self._last_time: Optional[float] = None
        self._client: Optional[Any] = None
        self._storage_options: dict = storage_options or {}
        self._session_options: dict = session_options or {}

    def _create_client(self) -> None:
        has_shared_credentials_file = (
            os.getenv("AWS_SHARED_CREDENTIALS_FILE") == os.getenv("AWS_CONFIG_FILE") == "/.credentials/.aws_credentials"
        )

        if has_shared_credentials_file or not _IS_IN_STUDIO or self._storage_options or self._session_options:
            session = boto3.Session(**self._session_options)  # If additional options are provided
            self._client = session.client(
                "s3",
                **{
                    "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
                    **self._storage_options,  # If additional options are provided
                },
            )
        else:
            provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=3600, num_attempts=5))
            credentials = provider.load()
            session = boto3.Session()
            self._client = session.client(
                "s3",
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._create_client()
            self._last_time = time()

        # Re-generate credentials for EC2
        if self._last_time is None or (time() - self._last_time) > self._refetch_interval:
            self._create_client()
            self._last_time = time()

        return self._client


class R2Client:
    """R2 client with refreshable credentials for Cloudflare R2 storage."""

    def __init__(
        self,
        refetch_interval: int = 3600,  # 1 hour - this is the default refresh interval for R2 credentials
        storage_options: Optional[dict] = {},
        session_options: Optional[dict] = {},
    ) -> None:
        self._refetch_interval = refetch_interval
        self._last_time: Optional[float] = None
        self._client: Optional[Any] = None
        self._base_storage_options: dict = storage_options or {}
        self._session_options: dict = session_options or {}

    def get_r2_bucket_credentials(self, data_connection_id: str) -> dict[str, str]:
        """Fetch temporary R2 credentials for the current lightning storage connection."""
        import json

        import requests

        try:
            # Get Lightning Cloud API token
            cloud_url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")
            api_key = os.getenv("LIGHTNING_API_KEY")
            username = os.getenv("LIGHTNING_USERNAME")
            project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")

            if not all([api_key, username, project_id]):
                raise RuntimeError("Missing required environment variables")

            # Login to get token
            payload = {"apiKey": api_key, "username": username}
            login_url = f"{cloud_url}/v1/auth/login"
            response = requests.post(login_url, data=json.dumps(payload))

            if "token" not in response.json():
                raise RuntimeError("Failed to get authentication token")

            token = response.json()["token"]

            # Get temporary bucket credentials
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            credentials_url = (
                f"{cloud_url}/v1/projects/{project_id}/data-connections/{data_connection_id}/temp-bucket-credentials"
            )

            credentials_response = requests.get(credentials_url, headers=headers)

            if credentials_response.status_code != 200:
                raise RuntimeError(f"Failed to get credentials: {credentials_response.status_code}")

            temp_credentials = credentials_response.json()

            endpoint_url = f"https://{temp_credentials['accountId']}.r2.cloudflarestorage.com"

            # Format credentials for S3Client
            return {
                "aws_access_key_id": temp_credentials["accessKeyId"],
                "aws_secret_access_key": temp_credentials["secretAccessKey"],
                "aws_session_token": temp_credentials["sessionToken"],
                "endpoint_url": endpoint_url,
            }

        except Exception as e:
            # Fallback to hardcoded credentials if API call fails
            print(f"Failed to get R2 credentials from API: {e}. Using fallback credentials.")
            raise RuntimeError(f"Failed to get R2 credentials and no fallback available: {e}")

    def _create_client(self) -> None:
        """Create a new R2 client with fresh credentials."""
        # Get data connection ID from storage options
        data_connection_id = self._base_storage_options.get("lightning_data_connection_id")
        if not data_connection_id:
            raise RuntimeError("lightning_data_connection_id is required in storage_options for R2 client")

        # Get fresh R2 credentials
        r2_credentials = self.get_r2_bucket_credentials(data_connection_id)

        # Filter out metadata keys that shouldn't be passed to boto3
        filtered_storage_options = {
            k: v for k, v in self._base_storage_options.items() if k not in ["lightning_data_connection_id"]
        }

        # Combine filtered storage options with fresh credentials
        storage_options = {**filtered_storage_options, **r2_credentials}

        # Create session and client
        session = boto3.Session(**self._session_options)
        self._client = session.client(
            "s3",
            **{
                "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
                **storage_options,
            },
        )

    @property
    def client(self) -> Any:
        """Get the R2 client, refreshing credentials if necessary."""
        if self._client is None:
            self._create_client()
            self._last_time = time()

        # Re-generate credentials when they expire
        if self._last_time is None or (time() - self._last_time) > self._refetch_interval:
            self._create_client()
            self._last_time = time()

        return self._client

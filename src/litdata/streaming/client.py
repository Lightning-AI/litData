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
from typing import Any, Dict, Optional

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
        storage_options: Optional[Dict] = {},
        session_options: Optional[Dict] = {},
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

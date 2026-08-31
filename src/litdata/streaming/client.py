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

import json
import logging
import os
import random
import threading
from time import sleep, time
from typing import Any

import boto3
import botocore
import requests
from botocore.credentials import InstanceMetadataProvider
from botocore.utils import InstanceMetadataFetcher
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from litdata.constants import _IS_IN_STUDIO

logger = logging.getLogger("litdata.streaming.client")

# Constants for the retry adapter. Docs: https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html
# Retries per request. Deliberately small: a refresh runs inline on whichever DataLoader worker
# thread crossed the interval and holds the client lock for its whole duration, so it has to
# return in seconds, not hours. `S3Client._refresh_client` is what rides out a longer outage.
_CONNECTION_RETRY_TOTAL = 4
# Backoff factor for connection retries (wait time increases by this factor after each failure)
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5
# Default timeout for each HTTP request in seconds
_DEFAULT_REQUEST_TIMEOUT = 30  # seconds

# The control plane mints credentials with a 1 hour TTL for S3 project-role data connections, and
# the response carries no expiry for us to read, so refresh well inside it. The remaining time is
# the window in which a failed refresh can be retried while the credentials in hand still work.
_DEFAULT_REFETCH_INTERVAL = 2700  # seconds
# How long past the refetch interval we keep serving existing credentials while refreshes fail.
# Sized against the TTL, not comfort: 2700 + 600 leaves ~5 minutes before the 1 hour S3 expiry,
# so we stop before reads start failing as unexplained S3 403s.
_REFRESH_GRACE_PERIOD = 600  # seconds
# How long to wait for the control plane when there are no credentials yet. No TTL constrains
# this one — nothing is being served — so it can be more patient than the refresh grace.
_INITIAL_RETRY_BUDGET = 900  # seconds
# Spacing between refresh attempts once one has failed. Without it every subsequent read
# re-requests credentials, turning one control-plane blip into a request storm per worker.
_REFRESH_RETRY_INTERVAL = 60  # seconds
# Fraction of the interval by which each process refreshes early. DataLoader workers are forked
# together, so without jitter they all reach the interval — and stampede — in the same instant.
_REFETCH_JITTER_RATIO = 0.1


class _CredentialsError(RuntimeError):
    """Raised when credentials could not be obtained from the control plane or IMDS.

    Only these are retried. Anything else out of ``_create_client`` — a bad ``storage_options``
    key, a malformed endpoint — is a local mistake that no amount of waiting fixes, and must
    reach the caller immediately rather than stalling the job for the retry budget.
    """


class _CredentialsUnavailableError(_CredentialsError):
    """A credential failure that may clear on its own: unreachable, timed out, 5xx, 429."""


class _CredentialsConfigurationError(_CredentialsError):
    """A credential failure that retrying cannot fix: missing configuration, or rejected auth."""


def _credentials_error(status_code: int, message: str) -> _CredentialsError:
    """Classify an HTTP failure. A 4xx other than 408/429 will not fix itself on a retry."""
    if 400 <= status_code < 500 and status_code not in (408, 429):
        return _CredentialsConfigurationError(message)
    return _CredentialsUnavailableError(message)


class _CustomRetryAdapter(HTTPAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT)
        super().__init__(*args, **kwargs)

    def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        # requests always passes `timeout` explicitly, as None when the caller gave none, so a
        # `kwargs.get("timeout", self.timeout)` default never applied and every request without
        # an explicit timeout could hang forever.
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def _login_and_get_temp_bucket_credentials(data_connection_id: str) -> dict[str, Any]:
    """Mint temporary bucket credentials for a data connection via the Lightning Cloud API.

    Shared by R2 (lightning storage) connections and by S3 connections marked
    ``available_in_non_aws_providers``: both bypass the FUSE mount and need short-lived creds
    from the same control-plane ``temp-bucket-credentials`` endpoint. Returns the raw response
    (accessKeyId/secretAccessKey/sessionToken, plus accountId for R2).
    """
    retry_strategy = Retry(
        total=_CONNECTION_RETRY_TOTAL,
        backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
        # urllib3 leaves POST out of its default allowed_methods, which left the login below
        # unretried: a single 429 or 502 there failed the whole refresh even though the
        # credentials GET would have retried it. Minting a token is safe to repeat.
        allowed_methods=Retry.DEFAULT_ALLOWED_METHODS | {"POST"},
        status_forcelist=[
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        ],
    )
    adapter = _CustomRetryAdapter(max_retries=retry_strategy, timeout=_DEFAULT_REQUEST_TIMEOUT)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    cloud_url = os.getenv("LIGHTNING_CLOUD_URL", "https://lightning.ai")
    api_key = os.getenv("LIGHTNING_API_KEY")
    username = os.getenv("LIGHTNING_USERNAME")
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")

    if not all([api_key, username, project_id]):
        raise _CredentialsConfigurationError("Missing required environment variables")

    # Login to get token
    payload = {"apiKey": api_key, "username": username}
    login_url = f"{cloud_url}/v1/auth/login"
    try:
        response = session.post(login_url, data=json.dumps(payload))
    except requests.exceptions.RequestException as e:
        raise _CredentialsUnavailableError(f"Could not reach the Lightning Cloud API to log in: {e}") from e

    # Check the status before the body: a proxy in front of the API answers with HTML, which
    # would otherwise surface as a JSONDecodeError with nothing pointing back at the login call.
    if response.status_code != 200:
        raise _credentials_error(
            response.status_code, f"Failed to log in to the Lightning Cloud API: {response.status_code}"
        )

    try:
        token = response.json()["token"]
    except (ValueError, KeyError) as e:
        raise RuntimeError("Failed to get authentication token") from e

    # Get temporary bucket credentials
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    credentials_url = (
        f"{cloud_url}/v1/projects/{project_id}/data-connections/{data_connection_id}/temp-bucket-credentials"
    )

    try:
        credentials_response = session.get(credentials_url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        raise _CredentialsUnavailableError(f"Could not reach the Lightning Cloud API for credentials: {e}") from e

    if credentials_response.status_code != 200:
        raise _credentials_error(
            credentials_response.status_code, f"Failed to get credentials: {credentials_response.status_code}"
        )

    return credentials_response.json()


class S3Client:
    # TODO: Generalize to support more cloud providers.

    def __init__(
        self,
        refetch_interval: int = _DEFAULT_REFETCH_INTERVAL,
        storage_options: dict | None = {},
        session_options: dict | None = {},
    ) -> None:
        self._refetch_interval = refetch_interval
        self._last_time: float | None = None
        self._storage_options: dict = storage_options or {}
        self._session_options: dict = session_options or {}
        self._reset_process_state()

    def _reset_process_state(self) -> None:
        """Reset the state that must not be shared across a fork.

        Re-rolls the refresh jitter too: DataLoader workers inherit the parent's timings, so
        without a fresh roll per process they would all refresh in the same instant.
        """
        self._client: Any | None = None
        # Guards lazy create + credential refresh (range GETs hit .client from many threads).
        self._client_lock = threading.Lock()
        self._owner_pid = os.getpid()
        self._refetch_deadline = self._jittered_refetch_interval()
        self._refresh_retry_time: float | None = None

    def _jittered_refetch_interval(self) -> float:
        # Only ever early, never late: callers set the interval as an upper bound on how long
        # a set of credentials is held, and the grace period below is measured against it.
        return self._refetch_interval * (1.0 - random.uniform(0.0, _REFETCH_JITTER_RATIO))  # noqa: S311

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_client_lock", None)
        # Recreate the boto3 client in the child (fork/spawn). Connection pools
        # and threading.Locks inside botocore are not safe to inherit.
        state.pop("_client", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._reset_process_state()

    def _create_client(self) -> None:
        # S3 data connections marked available on non-AWS providers can't reach the bucket via the
        # FUSE mount or an instance profile off AWS, so mint temporary project-role credentials from
        # the control plane instead (mirrors R2Client). Gated on data_connection_id being threaded
        # through storage_options, so plain S3 access is unaffected.
        data_connection_id = self._storage_options.get("data_connection_id")
        if data_connection_id:
            self._create_client_from_temp_credentials(data_connection_id)
            return

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
            try:
                credentials = provider.load()
            except Exception as e:
                raise _CredentialsUnavailableError(f"Could not load instance metadata credentials: {e}") from e
            session = boto3.Session()
            self._client = session.client(
                "s3",
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
            )

    def _create_client_from_temp_credentials(self, data_connection_id: str) -> None:
        """Create an S3 client backed by temporary project-role credentials for a data connection.

        Used for S3 connections available on non-AWS providers. Unlike R2 there is no custom
        endpoint — this is real AWS S3, so botocore resolves the bucket region on first use.
        """
        temp_credentials = _login_and_get_temp_bucket_credentials(data_connection_id)

        # data_connection_id is our own metadata; drop it before handing options to boto3.
        storage_options = {k: v for k, v in self._storage_options.items() if k != "data_connection_id"}

        session = boto3.Session(**self._session_options)
        self._client = session.client(
            "s3",
            **{
                "aws_access_key_id": temp_credentials["accessKeyId"],
                "aws_secret_access_key": temp_credentials["secretAccessKey"],
                "aws_session_token": temp_credentials["sessionToken"],
                "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
                **storage_options,
            },
        )

    def _mark_refreshed(self) -> None:
        self._last_time = time()
        self._refetch_deadline = self._jittered_refetch_interval()
        self._refresh_retry_time = None

    def _create_initial_client(self) -> None:
        """Create the first client, waiting out a control plane that is briefly unavailable.

        Unlike a refresh there are no credentials to fall back on, so this has to keep trying
        rather than hand the caller an error. Bounded by ``_INITIAL_RETRY_BUDGET``: a job that
        cannot reach the control plane for that long should fail with a clear reason.
        """
        started = time()
        attempt = 0
        while True:
            attempt += 1
            try:
                self._create_client()
            # Only a control plane that might come back is worth waiting for. A configuration
            # error, or any local failure building the client, propagates on the first attempt.
            except _CredentialsUnavailableError as e:
                waited = time() - started
                if waited + _REFRESH_RETRY_INTERVAL >= _INITIAL_RETRY_BUDGET:
                    raise RuntimeError(f"Could not get credentials after {waited:.0f}s of retrying: {e}") from e
                logger.warning(
                    "Could not get credentials (attempt %d, %.0fs elapsed), so data loading is blocked; "
                    "retrying in %ds: %s",
                    attempt,
                    waited,
                    _REFRESH_RETRY_INTERVAL,
                    e,
                )
                sleep(_REFRESH_RETRY_INTERVAL)
            else:
                self._mark_refreshed()
                return

    def _refresh_client(self) -> None:
        """Re-mint credentials, tolerating a control plane that is briefly unavailable.

        Credentials are refreshed before they expire, so a failed refresh is not immediately
        fatal — the ones already in hand still work. Keep serving those and retry on a timer
        until the grace period runs out, rather than failing the read on the first error, which
        kills the DataLoader worker and with it the run.
        """
        now = time()
        if self._refresh_retry_time is not None and now < self._refresh_retry_time:
            return

        try:
            self._create_client()
        # Both kinds get the grace period here, unlike initial creation. The credentials in hand
        # still work, so there is nothing to gain by failing fast on a 403 that might be a proxy
        # misbehaving mid-deploy — and if it is a real revocation, the deadline still catches it.
        except _CredentialsError as e:
            held_for = 0.0 if self._last_time is None else now - self._last_time
            if held_for > self._refetch_deadline + _REFRESH_GRACE_PERIOD:
                raise RuntimeError(
                    f"Failed to refresh credentials for {held_for:.0f}s, so they are assumed expired: {e}"
                ) from e
            self._refresh_retry_time = now + _REFRESH_RETRY_INTERVAL
            logger.warning(
                "Could not refresh credentials (%.0fs since the last successful refresh); reusing the current "
                "ones and retrying in %ds: %s",
                held_for,
                _REFRESH_RETRY_INTERVAL,
                e,
            )
            return

        self._mark_refreshed()

    @property
    def client(self) -> Any:
        # boto3 clients are thread-safe for requests; construction/refresh is not.
        if getattr(self, "_owner_pid", None) != os.getpid():
            # DataLoader fork: drop the inherited client, lock and refresh schedule.
            self._reset_process_state()

        with self._client_lock:
            if self._client is None:
                self._create_initial_client()
            # Re-generate credentials for EC2 / temporary Studio creds
            elif self._last_time is None or (time() - self._last_time) > self._refetch_deadline:
                self._refresh_client()

            return self._client


class R2Client(S3Client):
    """R2 client with refreshable credentials for Cloudflare R2 storage."""

    def __init__(
        self,
        refetch_interval: int = _DEFAULT_REFETCH_INTERVAL,
        storage_options: dict | None = {},
        session_options: dict | None = {},
    ) -> None:
        # Store R2-specific options before calling super()
        self._base_storage_options: dict = storage_options or {}

        # Call parent constructor with R2-specific refetch interval
        super().__init__(
            refetch_interval=refetch_interval,
            storage_options={},  # storage options handled in _create_client
            session_options=session_options,
        )

    def get_r2_bucket_credentials(self, data_connection_id: str) -> dict[str, str]:
        """Fetch temporary R2 credentials for the current lightning storage connection."""
        try:
            temp_credentials = _login_and_get_temp_bucket_credentials(data_connection_id)

            endpoint_url = f"https://{temp_credentials['accountId']}.r2.cloudflarestorage.com"

            # Format credentials for S3Client
            return {
                "aws_access_key_id": temp_credentials["accessKeyId"],
                "aws_secret_access_key": temp_credentials["secretAccessKey"],
                "aws_session_token": temp_credentials["sessionToken"],
                "endpoint_url": endpoint_url,
            }

        except _CredentialsError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get R2 credentials: {e}") from e

    def _create_client(self) -> None:
        """Create a new R2 client with fresh credentials."""
        # Get data connection ID from storage options
        data_connection_id = self._base_storage_options.get("data_connection_id")
        if not data_connection_id:
            raise _CredentialsConfigurationError("data_connection_id is required in storage_options for R2 client")

        # Get fresh R2 credentials
        r2_credentials = self.get_r2_bucket_credentials(data_connection_id)

        # Filter out metadata keys that shouldn't be passed to boto3
        filtered_storage_options = {
            k: v for k, v in self._base_storage_options.items() if k not in ["data_connection_id"]
        }

        # Combine filtered storage options with fresh credentials
        combined_storage_options = {**filtered_storage_options, **r2_credentials}

        # Update the inherited storage options with R2 credentials
        self._storage_options = combined_storage_options

        # Create session and client
        session = boto3.Session(**self._session_options)
        self._client = session.client(
            "s3",
            **{
                "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
                **combined_storage_options,
            },
        )

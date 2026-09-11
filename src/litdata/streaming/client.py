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
import re
import threading
from datetime import datetime, timezone
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

# Fallback for a control plane that reports no expiry: assume the 1 hour TTL S3 project-role
# connections have always had, and refresh well inside it. Also an upper bound on how long any
# credentials are held, so a longer reported TTL does not stretch the window between refreshes.
_DEFAULT_REFETCH_INTERVAL = 2700  # seconds
# Fraction of a reported lifetime to hold credentials for. The remainder is the window in which
# a failed refresh can be retried while the credentials in hand still work. 0.75 of the 1 hour
# TTL is the 2700s above, so a control plane that reports its expiry changes nothing.
_REFETCH_FRACTION = 0.75
# How long past the refetch interval we keep serving existing credentials while refreshes fail.
# Sized against the TTL, not comfort: 2700 + 600 leaves ~5 minutes before a 1 hour expiry, so we
# stop before reads start failing as unexplained S3 403s. A reported expiry bounds this directly.
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


def _parse_reported_expiry(value: Any) -> float | None:
    """Parse the control plane's RFC 3339 ``expiresAt`` into a unix timestamp.

    Anything unreadable is reported as absent rather than raised: a deadline we cannot
    parse should fall back to the assumed TTL, not fail the read.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # The proto3 JSON mapping emits Z-normalized RFC 3339 with 0, 3, 6 or 9 fractional digits.
    # `fromisoformat` rejects the Z before 3.11, and the 9-digit form on every version we support.
    if text[-1] in "Zz":
        text = text[:-1] + "+00:00"
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.warning("Ignoring unparsable credential expiry %r from the control plane", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _credentials_expiry(creds: dict[str, Any] | None) -> float | None:
    """Unix timestamp at which ``creds`` stop working, or ``None`` when none was reported."""
    if not creds:
        return None
    return _parse_reported_expiry(creds.get("expiresAt"))


def _refetch_interval_for(expires_at: float | None, fetched_at: float, upper_bound: float) -> float:
    """How long after ``fetched_at`` credentials expiring at ``expires_at`` should be replaced.

    ``upper_bound`` still applies once an expiry is known, because it is what the caller
    asked for: a 12 hour R2 lifetime should not stretch the gap between refreshes to match.
    """
    if expires_at is None:
        return upper_bound
    return max(0.0, min(upper_bound, (expires_at - fetched_at) * _REFETCH_FRACTION))


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


# Process-level cache of control-plane temp-bucket credentials. Each StreamingDataset
# builds a new R2Client, and the first GET used to login again (~1s). Warmup then a
# timed pass in the same process should reuse creds. Keyed by data_connection_id;
# TTL matches the client refetch interval so we do not serve credentials past the
# window a live client would already have refreshed.
_temp_creds_lock = threading.Lock()
_temp_creds_pid = os.getpid()
_temp_creds_cache: dict[str, tuple[float, dict[str, Any]]] = {}
# boto3 clients are thread-safe for requests; building one is ~50ms and is not
# fork-safe. Cache them with the credentials, drop on fork / explicit clear.
_temp_creds_boto_clients: dict[str, tuple[float, Any]] = {}


def _temp_creds_lock_for_pid() -> threading.Lock:
    """Recreate the cache lock after fork; an inherited held lock is unsafe."""
    global _temp_creds_lock, _temp_creds_pid
    pid = os.getpid()
    if _temp_creds_pid != pid:
        _temp_creds_lock = threading.Lock()
        _temp_creds_pid = pid
        _temp_creds_boto_clients.clear()
    return _temp_creds_lock


def _r2_botocore_config() -> Any:
    """R2 client config: adaptive retries, no extra checksum round-trips on tiny GETs."""
    # Adaptive retry mode adds ~30ms to tiny R2 GETs vs standard.
    retries = {"max_attempts": 1000, "mode": "standard"}
    try:
        return botocore.config.Config(
            retries=retries,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
    except TypeError:
        return botocore.config.Config(retries=retries)


def clear_temp_bucket_credentials_cache() -> None:
    """Drop cached Lightning Cloud temp-bucket credentials.

    Tests must call this (or use the autouse fixture in ``test_client.py``) so a
    previous case cannot satisfy a later HTTP mock via a warm cache.
    """
    with _temp_creds_lock_for_pid():
        _temp_creds_cache.clear()
        _temp_creds_boto_clients.clear()


def _cached_temp_bucket_credentials(
    data_connection_id: str, *, force_refresh: bool = False
) -> tuple[float, dict[str, Any]]:
    """Return ``(fetched_at, creds)``, reusing a process-level cache until refetch.

    ``force_refresh`` bypasses the cache so a client that has reached its deadline
    mints new credentials rather than inheriting ones that are about to expire.
    """
    lock = _temp_creds_lock_for_pid()
    with lock:
        if not force_refresh:
            cached = _temp_creds_cache.get(data_connection_id)
            if cached is not None:
                fetched_at, creds = cached
                reuse_for = _refetch_interval_for(_credentials_expiry(creds), fetched_at, _DEFAULT_REFETCH_INTERVAL)
                if time() - fetched_at < reuse_for:
                    return fetched_at, dict(creds)
        creds = _fetch_temp_bucket_credentials(data_connection_id)
        fetched_at = time()
        _temp_creds_cache[data_connection_id] = (fetched_at, creds)
        return fetched_at, dict(creds)


def _login_and_get_temp_bucket_credentials(data_connection_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Mint temporary bucket credentials for a data connection via the Lightning Cloud API.

    Shared by R2 (lightning storage) connections and by S3 connections marked
    ``available_in_non_aws_providers``: both bypass the FUSE mount and need short-lived creds
    from the same control-plane ``temp-bucket-credentials`` endpoint. Returns the raw response
    (accessKeyId/secretAccessKey/sessionToken, plus accountId for R2, and expiresAt/region/
    endpoint from control planes new enough to report them).

    Results are cached per process (see ``_cached_temp_bucket_credentials``).
    """
    _, creds = _cached_temp_bucket_credentials(data_connection_id, force_refresh=force_refresh)
    return creds


def _fetch_temp_bucket_credentials(data_connection_id: str) -> dict[str, Any]:
    """Hit the Lightning Cloud API for temp-bucket credentials (no cache)."""
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
        # Set before the deadline below, which is derived from them once credentials exist.
        self._creds_fetched_at: float | None = None
        self._creds_expires_at: float | None = None
        self._refetch_deadline = self._jittered_refetch_interval()
        self._refresh_retry_time: float | None = None
        self._force_refresh_credentials = False

    def _jittered_refetch_interval(self) -> float:
        # Only ever early, never late: callers set the interval as an upper bound on how long
        # a set of credentials is held, and the grace period below is measured against it.
        interval = _refetch_interval_for(
            self._creds_expires_at,
            self._creds_fetched_at if self._creds_fetched_at is not None else time(),
            self._refetch_interval,
        )
        return interval * (1.0 - random.uniform(0.0, _REFETCH_JITTER_RATIO))  # noqa: S311

    def _record_credentials_window(self, data_connection_id: str, creds: dict[str, Any]) -> None:
        """Note when these credentials were minted and when the control plane says they die."""
        with _temp_creds_lock_for_pid():
            cached = _temp_creds_cache.get(data_connection_id)
        self._creds_fetched_at = cached[0] if cached is not None else None
        self._creds_expires_at = _credentials_expiry(creds)

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

        Used for S3 connections available on non-AWS providers.
        """
        temp_credentials = _login_and_get_temp_bucket_credentials(
            data_connection_id, force_refresh=self._force_refresh_credentials
        )
        self._record_credentials_window(data_connection_id, temp_credentials)

        # data_connection_id is our own metadata; drop it before handing options to boto3.
        storage_options = {k: v for k, v in self._storage_options.items() if k != "data_connection_id"}

        # Inside a Studio, AWS_CONFIG_FILE points the default profile at Lightning Storage, and
        # botocore applies that endpoint and region to any client built here — sending these AWS
        # keys to Cloudflare, which rejects them as InvalidAccessKeyId. The control plane reports
        # where the bucket actually lives, so set both rather than letting the SDK resolve them.
        # Older control planes omit these fields; nothing to correct with, so leave them unset.
        resolved: dict[str, Any] = {}
        if temp_credentials.get("endpoint"):
            resolved["endpoint_url"] = temp_credentials["endpoint"]
        if temp_credentials.get("region"):
            resolved["region_name"] = temp_credentials["region"]

        session = boto3.Session(**self._session_options)
        self._client = session.client(
            "s3",
            **{
                "aws_access_key_id": temp_credentials["accessKeyId"],
                "aws_secret_access_key": temp_credentials["secretAccessKey"],
                "aws_session_token": temp_credentials["sessionToken"],
                "config": botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
                **resolved,
                **storage_options,
            },
        )

    def _mark_refreshed(self) -> None:
        # Prefer the mint time from the process cache so a new client that reused credentials
        # still refreshes before they expire, rather than a full interval from now.
        fetched_at = getattr(self, "_creds_fetched_at", None)
        self._last_time = fetched_at if fetched_at is not None else time()
        self._refetch_deadline = self._jittered_refetch_interval()
        self._refresh_retry_time = None

    def next_refresh_time(self) -> datetime:
        """When :attr:`client` will next mint credentials, as an aware UTC datetime.

        Callers that cache credentials of their own need this rather than the expiry, so they
        come back while the credentials they hold are still the ones this client is serving.
        Bounded by the reported expiry so it can never name a time the credentials are dead by.
        """
        last = self._last_time if self._last_time is not None else time()
        deadline = last + self._refetch_deadline
        if self._creds_expires_at is not None:
            deadline = min(deadline, self._creds_expires_at)
        return datetime.fromtimestamp(deadline, tz=timezone.utc)

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
            self._force_refresh_credentials = True
            try:
                self._create_client()
            finally:
                self._force_refresh_credentials = False
        # Both kinds get the grace period here, unlike initial creation. The credentials in hand
        # still work, so there is nothing to gain by failing fast on a 403 that might be a proxy
        # misbehaving mid-deploy — and if it is a real revocation, the deadline still catches it.
        except _CredentialsError as e:
            held_for = 0.0 if self._last_time is None else now - self._last_time
            # Once the control plane has told us when these die there is nothing left to serve
            # past that point, and continuing only turns the failure into an opaque S3 403.
            expired = self._creds_expires_at is not None and now >= self._creds_expires_at
            if expired or held_for > self._refetch_deadline + _REFRESH_GRACE_PERIOD:
                reason = "they have expired" if expired else "they are assumed expired"
                raise RuntimeError(f"Failed to refresh credentials for {held_for:.0f}s, so {reason}: {e}") from e
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
        storage_options: dict | None = None,
        session_options: dict | None = None,
    ) -> None:
        # Copy so a later pop of data_connection_id on a shared dict cannot
        # starve _create_client (R2Downloader / R2FsProvider pass their dict).
        self._base_storage_options: dict = dict(storage_options or {})

        # Call parent constructor with R2-specific refetch interval
        super().__init__(
            refetch_interval=refetch_interval,
            storage_options=None,  # storage options handled in _create_client
            session_options=session_options,
        )

    def get_r2_bucket_credentials(self, data_connection_id: str, *, force_refresh: bool = False) -> dict[str, str]:
        """Fetch temporary R2 credentials for the current lightning storage connection."""
        try:
            temp_credentials = _login_and_get_temp_bucket_credentials(data_connection_id, force_refresh=force_refresh)
            self._record_credentials_window(data_connection_id, temp_credentials)

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

        if not self._force_refresh_credentials:
            with _temp_creds_lock_for_pid():
                cached_client = _temp_creds_boto_clients.get(data_connection_id)
            if cached_client is not None:
                fetched_at, boto_client = cached_client
                # The credentials behind this client are cached separately; their reported
                # deadline is what says whether it is still safe to hand back.
                with _temp_creds_lock_for_pid():
                    cached_creds = _temp_creds_cache.get(data_connection_id)
                expires_at = _credentials_expiry(cached_creds[1] if cached_creds is not None else None)
                if time() - fetched_at < _refetch_interval_for(expires_at, fetched_at, _DEFAULT_REFETCH_INTERVAL):
                    self._client = boto_client
                    self._creds_fetched_at = fetched_at
                    self._creds_expires_at = expires_at
                    return

        # Get R2 credentials (process cache on first use; mint on scheduled refresh).
        r2_credentials = self.get_r2_bucket_credentials(
            data_connection_id, force_refresh=self._force_refresh_credentials
        )

        # Filter out metadata keys that shouldn't be passed to boto3
        filtered_storage_options = {
            k: v for k, v in self._base_storage_options.items() if k not in ["data_connection_id"]
        }

        # Combine filtered storage options with fresh credentials
        combined_storage_options = {**filtered_storage_options, **r2_credentials}

        # Update the inherited storage options with R2 credentials
        self._storage_options = combined_storage_options

        # Create session and client. Default SDK checksums add ~100ms per tiny R2 GET.
        session = boto3.Session(**self._session_options)
        self._client = session.client(
            "s3",
            **{
                "config": _r2_botocore_config(),
                **combined_storage_options,
            },
        )
        with _temp_creds_lock_for_pid():
            _temp_creds_boto_clients[data_connection_id] = (self._creds_fetched_at or time(), self._client)

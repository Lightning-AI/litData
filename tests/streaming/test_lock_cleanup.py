import os
import shutil
from contextlib import suppress

import pytest
from filelock import FileLock, Timeout

from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming.cache import Cache
from litdata.streaming.config import ChunkedIndex
from litdata.streaming.downloader import LocalDownloader, register_downloader, unregister_downloader
from litdata.streaming.reader import BinaryReader
from litdata.streaming.resolver import Dir


class LocalDownloaderNoLockCleanup(LocalDownloader):
    """A Local downloader variant that does NOT remove the `.lock` file after download.

    This simulates behavior of non-local downloaders where the lockfile persists on disk
    until Reader cleanup runs. Used to verify our centralized lock cleanup.
    """

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:  # type: ignore[override]
        # Strip the custom scheme used for testing to map to local FS
        if remote_filepath.startswith("s3+local://"):
            remote_filepath = remote_filepath.replace("s3+local://", "")
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        with (
            suppress(Timeout, FileNotFoundError),
            FileLock(local_filepath + ".lock", timeout=0),
        ):
            if remote_filepath == local_filepath or os.path.exists(local_filepath):
                return
            temp_file_path = local_filepath + ".tmp"
            shutil.copy(remote_filepath, temp_file_path)
            os.rename(temp_file_path, local_filepath)
            # Intentionally do NOT remove `local_filepath + ".lock"` here


@pytest.mark.skipif(not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")
def test_reader_lock_cleanup_with_nonlocal_like_downloader(tmpdir):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)

    # Build a small compressed dataset
    cache = Cache(input_dir=Dir(path=cache_dir, url=None), chunk_size=3, compression="zstd")
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    # Copy to a "remote" directory
    shutil.copytree(cache_dir, remote_dir)

    # Use a custom scheme that we register to our test downloader
    prefix = "s3+local://"
    remote_url = prefix + remote_dir

    # Register the downloader and ensure we unregister afterwards
    register_downloader(prefix, LocalDownloaderNoLockCleanup, overwrite=True)
    try:
        # Fresh cache dir for reading
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        reader = BinaryReader(cache_dir=cache_dir, remote_input_dir=remote_url, compression="zstd", max_cache_size=1)

        # Iterate across enough samples to trigger multiple chunk downloads and deletions
        for i in range(10):
            idx = reader._get_chunk_index_from_index(i)
            chunk_idx = ChunkedIndex(index=idx[0], chunk_index=idx[1], is_last_index=(i == 9))
            reader.read(chunk_idx)

        # At the end, no chunk-related lock files should remain
        leftover_locks = [f for f in os.listdir(cache_dir) if f.endswith(".lock") and f.startswith("chunk-")]
        assert leftover_locks == []
    finally:
        unregister_downloader(prefix)

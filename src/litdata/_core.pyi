def hello_from_bin() -> str: ...

# StreamingDataProvider
#     -> on start, download x upcoming items in advance
#     -> get_next_k_item() => get next k upcomig items
#
# ------ how it works ------
# 1. ChunksConfig has a property `self.streaming_data_provider` which is an instance of StreamingDataProvider
# 2. When dataset.py __iter__() is called, it gets the chunk order and __next__() will get the sample item order.
# 3. The chunk order and sample item order is stored in `set_chunk` and `set_sample_index`.
# 4. But, we will not only get chunk and sample order for current epoch, but also for next epoch to be better prepared.
# 5. For dataset's epoch 1, we will call on_start() to download offset array for all chunk indexes in parallel.
# 6. Downloaded items returned by on_start() and in future by get_next_k_item()
#       are deserialized and then stored in `config.index_to_sample_data`.
# 7. when an item read is requested, get_next_k_item() will be called to get the next k items.
# 8. For every subsequent epoch (2, 3, ...), we will get the chunk and sample order for the next epoch
#       and then call `set_chunk` and `set_sample_index` to update the chunk and sample order for next epoch.
class StreamingDataProvider:
    def __init__(
        self,
        epoch: int,
        remote_dir: str,
        chunks: list[dict[str, str]],
        on_start_pre_item_download_count: int,
        get_next_k_item_count: int,
    ) -> None: ...
    def on_start(self) -> list[tuple[int, int, int, bytes]]: ...
    def get_next_k_item(self) -> list[tuple[int, int, int, bytes]]: ...
    def set_epoch(self, epoch: int) -> None: ...
    def set_chunk_and_sample_index(self, epoch: int, chunk_index: list[int], sample_index: list[list[int]]) -> None: ...
    def set_chunk(
        self, epoch: int, chunk_index: list[int], chunk_index_begin: list[tuple[int, int, int, int]]
    ) -> None: ...
    def set_sample_index(self, epoch: int, sample_index: list[list[int]]) -> None: ...

# S3Storage
class S3Storage:
    def __init__(self, remote_dir: str) -> None: ...
    def byte_range_download(self, remote_path: str, local_path: str, num_workers: int) -> None: ...

class LitDataLoaderCore:
    index: int
    worker_chunks: list[int]
    worker_intervals: list[tuple[int, int]]
    batch_size: int
    pre_download: int
    prefetch_workers: int
    prefetch_factor: int

    def __init__(
        self,
        worker_chunks: list[int],
        worker_intervals: list[tuple[int, int]],
        batch_size: int,
        pre_download: int,
        prefetch_workers: int,
        prefetch_factor: int,
    ) -> None: ...
    def __iter__(self) -> any: ...

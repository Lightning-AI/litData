def hello_from_bin() -> str: ...

class S3Storage:
    def list(self, path: str) -> list[str]: ...
    def does_file_exist(self, path: str) -> bool: ...
    def upload(self, local_path: str, remote_path: str) -> None: ...
    def delete(self, path: str) -> None: ...
    def download(self, remote_path: str, local_path: str) -> None: ...
    def byte_range_download(self, remote_path: str, local_path: str, num_threads: int) -> None: ...

class StreamingDataProvider:
    def __init__(
        self,
        epoch: int,
        remote_dir: str,
        chunks: list[dict[str, str]],
        chunk_index_odd_epoch: list[int],
        chunk_index_even_epoch: list[int],
        sample_index_odd_epoch: list[list[int]],
        sample_index_even_epoch: list[list[int]],
        on_start_pre_item_download_count: int,
        get_next_k_item_count: int,
    ) -> None: ...
    def on_start(self) -> None: ...
    def get_next_k_item(self) -> None: ...
    def set_epoch(self, epoch: int) -> None: ...
    def set_chunk_and_sample_index(self, epoch: int, chunk_index: list[int], sample_index: list[list[int]]) -> None: ...
    def set_chunk(self, epoch: int, chunk_index: list[int]) -> None: ...
    def set_sample_index(self, epoch: int, sample_index: list[list[int]]) -> None: ...

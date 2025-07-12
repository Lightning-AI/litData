import glob
import io
import os
import random
import shutil
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from unittest import mock

import cryptography
import numpy as np
import pytest
import requests
import torch
from PIL import Image

from litdata import StreamingDataLoader, StreamingDataset, index_parquet_dataset, map, merge_datasets, optimize, walk
from litdata.processing.data_processor import ALL_DONE
from litdata.processing.functions import _get_input_dir, _resolve_dir
from litdata.streaming.cache import Cache
from litdata.streaming.item_loader import ParquetLoader
from litdata.utilities.encryption import FernetEncryption, RSAEncryption


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
def test_get_input_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock.MagicMock(return_value=True))
    assert _get_input_dir(["/teamspace/studios/here/a", "/teamspace/studios/here/b"]) == "/teamspace/studios/here"

    exists_res = [False, True]

    def fn(*_, **__):
        return exists_res.pop(0)

    monkeypatch.setattr(os.path, "exists", fn)

    with pytest.raises(ValueError, match="The provided item  didn't contain any filepaths."):
        assert _get_input_dir(["", "/teamspace/studios/asd/b"])


def test_walk(tmpdir):
    for i in range(5):
        folder_path = os.path.join(tmpdir, str(i))
        os.makedirs(folder_path, exist_ok=True)
        for j in range(5):
            filepath = os.path.join(folder_path, f"{j}.txt")
            with open(filepath, "w") as f:
                f.write("hello world !")

    walks_os = sorted(os.walk(tmpdir))
    walks_function = sorted(walk(tmpdir))
    assert walks_os == walks_function


def test_get_input_dir_with_s3_path():
    input_dir = _get_input_dir(["s3://my_bucket/my_folder/a.txt"])
    assert input_dir == "s3://my_bucket/my_folder"
    input_dir = _resolve_dir(input_dir)
    assert not input_dir.path
    assert input_dir.url == "s3://my_bucket/my_folder"


def update_msg(file_path: Path, output_dir: Path):
    with open(os.path.join(output_dir, file_path.name), "w") as f:
        f.write("Bonjour!")


def test_map_with_path(tmpdir):
    input_dir = Path(tmpdir) / "input_dir"
    output_dir = Path(tmpdir) / "output_dir"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(5):
        filepath = os.path.join(input_dir, f"{i}.txt")
        with open(filepath, "w") as f:
            f.write("hello world!")

    # read all files in the input directory, and assert it contains hello world!
    for file in input_dir.iterdir():
        with open(file) as f:
            content = f.read()
            assert content == "hello world!"

    inputs = list(input_dir.iterdir())  # List all files in the directory

    map(
        fn=update_msg,
        inputs=inputs,
        output_dir=output_dir,
    )

    # read all files in the output directory, and assert it contains Bonjour!
    for file in output_dir.iterdir():
        with open(file) as f:
            content = f.read()
            assert content == "Bonjour!"


def compress(index):
    return index, index**2


def different_compress(index):
    return index, index**2, index**3


def fn(i: int):
    if i in [1, 2, 4]:
        raise ValueError("An error occurred")
    return i, i**2


def another_fn(i: int):
    return i, i**2


def random_image(index):
    fake_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return {"image": fake_img, "class": index}


@pytest.mark.skipif(sys.platform == "win32", reason="too slow")
def test_optimize_append_overwrite(tmpdir):
    output_dir = str(tmpdir / "output_dir")

    optimize(
        fn=compress,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5)]

    with pytest.raises(RuntimeError, match="HINT: If you want to append/overwrite to the existing dataset"):
        optimize(
            fn=compress,
            inputs=list(range(5, 10)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
        )

    with pytest.raises(ValueError, match="The provided `mode` should be either `append` or `overwrite`"):
        optimize(
            fn=compress,
            inputs=list(range(5, 10)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode="some-random-mode",
        )

    optimize(
        fn=compress,
        inputs=list(range(5, 10)),
        num_workers=2,
        output_dir=output_dir,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5, 10)]

    optimize(
        fn=compress,
        inputs=list(range(10, 15)),
        num_workers=os.cpu_count(),
        output_dir=output_dir,
        chunk_bytes="64MB",
        mode="append",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 10
    assert ds[:] == [(i, i**2) for i in range(5, 15)]

    optimize(
        fn=compress,
        inputs=list(range(15, 20)),
        num_workers=2,
        output_dir=output_dir,
        chunk_bytes="64MB",
        mode="append",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 15
    assert ds[:] == [(i, i**2) for i in range(5, 20)]

    with pytest.raises(Exception, match="The config isn't consistent between chunks"):
        optimize(
            fn=different_compress,
            inputs=list(range(100, 200)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode="append",
        )

    optimize(
        fn=different_compress,
        inputs=list(range(0, 5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        mode="overwrite",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 5
    assert ds[:] == [(i, i**2, i**3) for i in range(0, 5)]


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="Multiprocessing issues on Python 3.12+")
@pytest.mark.skipif(sys.platform == "win32", reason="too slow")
def test_optimize_checkpoint_in_none_and_append_mode(tmpdir):
    output_dir = str(tmpdir / "output_dir")

    with pytest.raises(RuntimeError, match="We found the following error"):
        optimize(
            fn=fn,
            inputs=list(range(4)),
            output_dir=output_dir,
            chunk_size=1,
            num_workers=2,
            use_checkpoint=True,
            start_method="fork",
        )

    # check that the checkpoints are created
    assert os.path.exists(os.path.join(output_dir, ".checkpoints"))
    assert os.path.exists(os.path.join(output_dir, ".checkpoints", "config.json"))

    optimize(
        fn=another_fn,
        inputs=list(range(4)),
        output_dir=output_dir,
        chunk_size=1,
        num_workers=2,
        use_checkpoint=True,
        start_method="fork",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 4
    assert ds[:] == [(i, i**2) for i in range(4)]
    # checkpoints should be deleted
    assert not os.path.exists(os.path.join(output_dir, ".checkpoints"))

    # --------- now test for append mode ---------

    with pytest.raises(RuntimeError, match="We found the following error"):
        optimize(
            fn=fn,
            inputs=list(range(4, 8)),
            output_dir=output_dir,
            chunk_size=1,
            num_workers=2,
            use_checkpoint=True,
            mode="append",
            start_method="fork",
        )

    # check that the checkpoints are created
    assert os.path.exists(os.path.join(output_dir, ".checkpoints"))
    assert os.path.exists(os.path.join(output_dir, ".checkpoints", "config.json"))
    print("-" * 80)
    # print all the files in the checkpoints folder
    for f in os.listdir(os.path.join(output_dir, ".checkpoints")):
        print(f)
    print("-" * 80)

    optimize(
        fn=another_fn,
        inputs=list(range(4, 8)),
        output_dir=output_dir,
        chunk_size=1,
        num_workers=2,
        use_checkpoint=True,
        mode="append",
        start_method="fork",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 8
    assert ds[:] == [(i, i**2) for i in range(8)]
    # checkpoints should be deleted
    assert not os.path.exists(os.path.join(output_dir, ".checkpoints"))


def test_merge_datasets(tmpdir):
    folder_1 = os.path.join(tmpdir, "folder_1")
    folder_2 = os.path.join(tmpdir, "folder_2")
    folder_3 = os.path.join(tmpdir, "folder_3")

    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)

    cache_1 = Cache(input_dir=folder_1, chunk_bytes="64MB")
    for i in range(10):
        cache_1[i] = i

    cache_1.done()
    cache_1.merge()

    cache_2 = Cache(input_dir=folder_2, chunk_bytes="64MB")
    for i in range(10, 20):
        cache_2[i] = i

    cache_2.done()
    cache_2.merge()

    merge_datasets(
        input_dirs=[folder_1, folder_2],
        output_dir=folder_3,
    )

    ds = StreamingDataset(input_dir=folder_3)

    assert len(ds) == 20
    assert ds[:] == list(range(20))


@pytest.mark.timeout(10)
def test_merge_compressed_datasets(tmpdir):
    folder_1 = os.path.join(tmpdir, "folder_1")
    folder_2 = os.path.join(tmpdir, "folder_2")
    folder_3 = os.path.join(tmpdir, "folder_3")

    os.makedirs(folder_1, exist_ok=True)
    os.makedirs(folder_2, exist_ok=True)

    cache_1 = Cache(input_dir=folder_1, chunk_bytes="64MB", compression="zstd")
    for i in range(10):
        cache_1[i] = i

    cache_1.done()
    cache_1.merge()

    cache_2 = Cache(input_dir=folder_2, chunk_bytes="64MB", compression="zstd")
    for i in range(10, 20):
        cache_2[i] = i

    cache_2.done()
    cache_2.merge()

    merge_datasets(
        input_dirs=[folder_1, folder_2],
        output_dir=folder_3,
    )

    ds = StreamingDataset(input_dir=folder_3)

    assert len(ds) == 20
    assert ds[:] == list(range(20))


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_optimize_with_fernet_encryption(tmpdir):
    output_dir = str(tmpdir / "output_dir")

    # ----------------- sample level -----------------
    fernet = FernetEncryption(password="password", level="sample")
    optimize(
        fn=compress,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=fernet,
    )

    ds = StreamingDataset(output_dir, encryption=fernet)
    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5)]

    # ----------------- chunk level -----------------
    fernet = FernetEncryption(password="password", level="chunk")
    optimize(
        fn=compress,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=fernet,
        mode="overwrite",
    )

    ds = StreamingDataset(output_dir, encryption=fernet)
    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5)]

    # ----------------- test with appending more -----------------
    optimize(
        fn=compress,
        inputs=list(range(5, 10)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=fernet,
        mode="append",
    )
    ds = StreamingDataset(output_dir, encryption=fernet)
    assert len(ds) == 10
    assert ds[:] == [(i, i**2) for i in range(10)]

    # ----------------- decrypt with different conf  -----------------
    ds = StreamingDataset(output_dir)
    with pytest.raises(ValueError, match="Data is encrypted but no encryption object was provided."):
        ds[0]

    fernet.level = "sample"
    ds = StreamingDataset(output_dir, encryption=fernet)
    with pytest.raises(ValueError, match="Encryption level mismatch."):
        ds[0]

    fernet = FernetEncryption(password="password", level="chunk")
    ds = StreamingDataset(output_dir, encryption=fernet)
    with pytest.raises(cryptography.fernet.InvalidToken, match=""):
        ds[0]

    # ----------------- test with other alg -----------------
    rsa = RSAEncryption(password="password", level="sample")
    ds = StreamingDataset(output_dir, encryption=rsa)
    with pytest.raises(ValueError, match="Encryption algorithm mismatch."):
        ds[0]

    # ----------------- test with random images -----------------

    fernet = FernetEncryption(password="password", level="chunk")
    optimize(
        fn=random_image,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=fernet,
        mode="overwrite",
    )

    ds = StreamingDataset(output_dir, encryption=fernet)

    assert len(ds) == 5
    assert ds[0]["class"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_optimize_with_rsa_encryption(tmpdir):
    output_dir = str(tmpdir / "output_dir")

    # ----------------- sample level -----------------
    rsa = RSAEncryption(password="password", level="sample")
    optimize(
        fn=compress,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=rsa,
    )

    ds = StreamingDataset(output_dir, encryption=rsa)
    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5)]

    # ----------------- chunk level -----------------
    rsa = RSAEncryption(password="password", level="chunk")
    optimize(
        fn=compress,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=rsa,
        mode="overwrite",
    )

    ds = StreamingDataset(output_dir, encryption=rsa)
    assert len(ds) == 5
    assert ds[:] == [(i, i**2) for i in range(5)]

    # ----------------- test with appending more -----------------
    optimize(
        fn=compress,
        inputs=list(range(5, 10)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
        encryption=rsa,
        mode="append",
    )
    ds = StreamingDataset(output_dir, encryption=rsa)
    assert len(ds) == 10
    assert ds[:] == [(i, i**2) for i in range(10)]

    # ----------------- decrypt with different conf  -----------------
    ds = StreamingDataset(output_dir)
    with pytest.raises(ValueError, match="Data is encrypted but no encryption object was provided."):
        ds[0]

    # ----------------- test with random images  -----------------
    # RSA Encryption throws an error: ValueError: Encryption failed, when trying to encrypt large data
    # optimize(
    #     fn=random_image,
    #     inputs=list(range(5)),
    #     num_workers=1,
    #     output_dir=output_dir,
    #     chunk_bytes="64MB",
    #     encryption=rsa,
    #     mode="overwrite",
    # )


def tokenize(filename: str):
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    text = text.strip().split(" ")
    word_to_int = {word: random.randint(1, 1000) for word in set(text)}  # noqa: S311
    tokenized = [word_to_int[word] for word in text]

    yield tokenized


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_optimize_race_condition(tmpdir):
    # issue: https://github.com/Lightning-AI/litdata/issues/367
    # run_commands = [
    #     "mkdir -p tempdir/custom_texts",
    #     "curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output tempdir/custom_texts/book1.txt",
    #     "curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output tempdir/custom_texts/book2.txt",
    # ]
    # The files were moved to S3
    shutil.rmtree(f"{tmpdir}/custom_texts", ignore_errors=True)
    os.makedirs(f"{tmpdir}/custom_texts", exist_ok=True)

    urls = [
        "https://pl-flash-data.s3.us-east-1.amazonaws.com/pg24440.txt",
        "https://pl-flash-data.s3.us-east-1.amazonaws.com/pg26393.txt",
    ]

    for i, url in enumerate(urls):
        print(f"downloading {i + 1} file")
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()  # Raise an exception for bad status codes

            with open(f"{tmpdir}/custom_texts/book{i + 1}.txt", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    train_files = sorted(glob.glob(str(Path(f"{tmpdir}/custom_texts") / "*.txt")))
    optimize(
        fn=tokenize,
        inputs=train_files,
        output_dir=f"{tmpdir}/temp",
        num_workers=1,
        chunk_bytes="50MB",
    )


def create_test_images(num_images=3, width=64, height=64):
    """Create a list JPEG images."""
    image_list = []

    for i in range(num_images):
        img = Image.new("RGB", (width, height))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        image_list.append(Image.open(io.BytesIO(img_bytes.getvalue())))
    return image_list


def process_with_jpeg_array(index):
    """Return a dict with an index and list of image bytearrays."""
    width = height = 32 + (index * 8)
    images = create_test_images(3, width, height)
    return {"index": index, "images": images}


def test_optimize_with_jpeg_array(tmpdir):
    """Test optimizing data containing lists of bytearrays which should use the jpeg_array serializer."""
    output_dir = str(tmpdir)
    # Run the optimization
    optimize(
        fn=process_with_jpeg_array,
        inputs=list(range(5)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="10MB",
    )

    # Load as streaming dataset
    ds = StreamingDataset(output_dir)

    # Verify data length
    assert len(ds) == 5

    # Check images are properly deserialized
    for i in range(5):
        item = ds[i]
        assert item["index"] == i

        # Check images
        images = item["images"]
        assert len(images) == 3
        assert all(isinstance(img, (Image.Image, torch.Tensor)) for img in images)

    data_format = ds.cache._reader._item_loader._data_format
    assert data_format == [
        "int",
        "jpeg_array",
    ], f"Expected data format to be ['int', 'jpeg_array'], but got {data_format}"


def text_optimize_fn(filename: str, *args, **kwargs):
    """A simple text processing function."""
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    return text.strip()


@pytest.mark.parametrize("keep_data_ordered", [False, True])
def test_optimize_with_text_files(tmpdir, keep_data_ordered):
    """Test optimizing data containing text manipulation."""
    input_dir = Path(tmpdir) / "files"
    os.makedirs(input_dir, exist_ok=True)
    for i in range(1, 6):
        with open(input_dir / f"file-{i}.txt", "w") as f:
            f.write("hello world")

    # Run the optimization
    optimize(
        fn=text_optimize_fn,
        inputs=[str(input_dir / f"file-{i}.txt") for i in range(1, 6)],
        input_dir=str(input_dir),
        output_dir=str(tmpdir / "output"),
        num_workers=1,
        chunk_bytes="10MB",
        num_downloaders=1,
        num_uploaders=1,
        mode="overwrite",
        keep_data_ordered=keep_data_ordered,
    )

    # Load as streaming dataset
    ds = StreamingDataset(str(tmpdir / "output"))

    # Verify data length
    assert len(ds) == 5

    # Check tokenized data
    for i in range(5):
        item = ds[i]
        assert isinstance(item, str)
        assert item == "hello world"


def text_map_fn(filename: str, output_dir: str):
    """A simple text processing function."""
    with open(filename, encoding="utf-8") as file:
        text = file.read().strip()
        filename_only = os.path.basename(filename)

        with open(os.path.join(output_dir, filename_only), "a", encoding="utf-8") as file:
            file.write(f"{text}\tBonjour!")


@pytest.mark.parametrize("keep_data_ordered", [False, True])
def test_map_with_text_files(tmpdir, keep_data_ordered):
    """Test optimizing data containing text manipulation."""
    input_dir = Path(tmpdir) / "files"
    os.makedirs(input_dir, exist_ok=True)
    for i in range(1, 6):
        with open(input_dir / f"file-{i}.txt", "w") as f:
            f.write("hello world")

    # Run the optimization
    map(
        fn=text_map_fn,
        inputs=[str(input_dir / f"file-{i}.txt") for i in range(1, 6)],
        input_dir=str(input_dir),
        output_dir=str(tmpdir / "output"),
        num_workers=1,
        keep_data_ordered=keep_data_ordered,
    )

    # Verify in files are updated
    for i in range(1, 6):
        with open(tmpdir / "output" / f"file-{i}.txt", encoding="utf-8") as file:
            content = file.read()
            assert content == "hello world\tBonjour!"


def yield_numbers():
    for i in range(100):
        time.sleep(0.01)
        yield i


def data_producer(q: Queue):
    for item in yield_numbers():
        q.put(item)

    q.put(ALL_DONE)  # Sentinel value to indicate end


def simple_optimize_fn(index):
    return index, index**2


@pytest.mark.parametrize("num_workers", [1, 2])
def test_optimize_with_queues_as_input(tmpdir, num_workers):
    output_dir = str(tmpdir / "output_dir")
    q = Queue(maxsize=100)

    producer = Process(target=data_producer, args=(q,))
    producer.start()

    optimize(
        fn=simple_optimize_fn,
        queue=q,
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_size=100,
    )

    producer.join()

    ds = StreamingDataset(output_dir)
    assert len(ds) == 100
    complete_data = sorted(ds[:])  # Sort to ensure order
    for idx, data in enumerate(complete_data):
        assert data == (idx, idx**2)


def optimize_fn(data):
    # Extract single elements from list-based record
    index = data["index"][0]
    question = data["question"][0]
    answer = data["answer"][0]
    return {"index": index, "question": question, "answer": answer}


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
@pytest.mark.parametrize("num_workers", [5, 6, 8])
def test_optimize_with_streaming_dataloader_on_parquet_data(tmpdir, num_workers):
    """Test optimization with StreamingDataLoader on parquet data with multiple workers.

    This test ensures that when using StreamingDataLoader as input to optimize(),
    all items are processed correctly without loss due to StopIteration issues
    that can occur with multiple workers.

    Reproduces issue: https://github.com/Lightning-AI/litdata/issues/599
    """
    # Prepare parquet dataset
    parquet_dir = os.path.join(tmpdir, "parquet")
    os.makedirs(parquet_dir, exist_ok=True)
    import polars as pl

    num_items = 500
    indexes = list(range(num_items))
    questions = [f"What is the capital of country {i}?" for i in range(num_items)]
    answers = [f"The capital of country {i} is city {i}." for i in range(num_items)]

    df = pl.DataFrame({"index": indexes, "question": questions, "answer": answers})
    parquet_file = os.path.join(parquet_dir, "sample.parquet")
    df.write_parquet(parquet_file)

    # Index the parquet dataset and create a streaming dataset and dataloader
    index_parquet_dataset(parquet_dir)
    dataset = StreamingDataset(parquet_dir, item_loader=ParquetLoader())
    dataloader = StreamingDataLoader(dataset)

    # Verify the dataloader has the expected length
    assert len(dataloader) == num_items, f"Expected dataloader length {num_items}, got {len(dataloader)}"

    # Optimize the dataset using the streaming dataloader as input
    output_dir = os.path.join(tmpdir, "out")
    os.makedirs(output_dir, exist_ok=True)

    optimize(
        fn=optimize_fn,
        inputs=dataloader,
        num_workers=num_workers,
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    # Verify optimized dataset length - this is the critical test
    ds = StreamingDataset(output_dir)
    actual_length = len(ds)
    assert actual_length == num_items, (
        f"Expected {num_items} items, got {actual_length}. "
        f"Missing {num_items - actual_length} items with {num_workers} workers."
    )

    # Verify a sample record structure
    sample_record = ds[0]
    assert "index" in sample_record, "Missing 'index' field in sample record"
    assert "question" in sample_record, "Missing 'question' field in sample record"
    assert "answer" in sample_record, "Missing 'answer' field in sample record"

    # Verify the first record has expected values
    assert sample_record["index"] == 0, f"Expected index 0, got {sample_record['index']}"
    assert sample_record["question"] == "What is the capital of country 0?", (
        f"Unexpected question: {sample_record['question']}"
    )
    assert sample_record["answer"] == "The capital of country 0 is city 0.", (
        f"Unexpected answer: {sample_record['answer']}"
    )

    # check all the indexes are correct
    indexes = [sample_record["index"].item() for sample_record in ds]
    assert indexes == list(range(num_items)), f"Expected indexes to be {list(range(num_items))}, but got {indexes}"

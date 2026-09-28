import os
import tarfile

from litdata.processing.media_folder import iter_webdataset_tar, list_media_folder
from litdata.streaming.serializers import VideoSerializer, _safe_decode_device


def test_list_media_folder_labels(tmpdir):
    cats = os.path.join(tmpdir, "cats")
    dogs = os.path.join(tmpdir, "dogs")
    os.makedirs(cats)
    os.makedirs(dogs)
    open(os.path.join(cats, "a.jpg"), "wb").close()
    open(os.path.join(dogs, "b.png"), "wb").close()
    open(os.path.join(dogs, "skip.txt"), "wb").close()

    items = list_media_folder(str(tmpdir), kind="image")
    assert [(os.path.basename(item["path"]), item["label"]) for item in items] == [
        ("a.jpg", "cats"),
        ("b.png", "dogs"),
    ]

    texts = list_media_folder(str(tmpdir), kind="text")
    assert [os.path.basename(item["path"]) for item in texts] == ["skip.txt"]


def test_iter_webdataset_tar(tmpdir):
    tar_path = os.path.join(tmpdir, "shard.tar")
    with tarfile.open(tar_path, "w") as archive:
        for name, payload in (("000.jpg", b"img0"), ("000.txt", b"cap0"), ("001.jpg", b"img1")):
            path = os.path.join(tmpdir, name)
            with open(path, "wb") as handle:
                handle.write(payload)
            archive.add(path, arcname=name)

    samples = list(iter_webdataset_tar(tar_path))
    assert [sample["__key__"] for sample in samples] == ["000", "001"]
    assert samples[0]["jpg"] == b"img0"
    assert samples[0]["txt"] == b"cap0"
    assert samples[1]["jpg"] == b"img1"


def test_video_cuda_forced_cpu_in_optimize_worker(monkeypatch):
    assert _safe_decode_device("cpu") == "cpu"
    monkeypatch.delenv("DATA_OPTIMIZER_GLOBAL_RANK", raising=False)
    assert _safe_decode_device("cuda") == "cuda"
    monkeypatch.setenv("DATA_OPTIMIZER_GLOBAL_RANK", "0")
    assert _safe_decode_device("cuda") == "cpu"
    serializer = VideoSerializer(device="cuda")
    assert serializer.device == "cuda"

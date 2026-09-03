import json
from unittest.mock import MagicMock

import pytest

from litdata.processing import utilities as utilities_module
from litdata.processing.utilities import (
    extract_rank_and_index_from_filename,
    optimize_dns_context,
    read_index_file_content,
    remove_uuid_from_filename,
)
from litdata.streaming.resolver import _resolve_dir


def test_optimize_dns_context(monkeypatch):
    popen_mock = MagicMock()

    monkeypatch.setattr(utilities_module, "_IS_IN_STUDIO", True)
    monkeypatch.setattr(utilities_module, "Popen", popen_mock)

    class FakeFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            return self

        def readlines(self):
            return ["127.0.0.53"]

    monkeypatch.setitem(__builtins__, "open", MagicMock(return_value=FakeFile()))

    with optimize_dns_context(True):
        pass

    cmd = popen_mock._mock_call_args_list[0].args[0]
    expected_cmd = (
        "sudo /home/zeus/miniconda3/envs/cloudspace/bin/python"
        " -c 'from litdata.processing.utilities import _optimize_dns; _optimize_dns(True)'"
    )
    assert cmd == expected_cmd


def test_extract_rank_and_index_from_filename():
    file_names = [
        "chunk-0-0.bin",
        "chunk-0-0.compressionAlgorithm.bin",
        "chunk-1-4.bin",
        "chunk-1-9.compressionAlgorithm.bin",
        "chunk-22-10.bin",
        "chunk-2-3.compressionAlgorithm.bin",
        "chunk-31-3.bin",
        "chunk-3-110.compressionAlgorithm.bin",
    ]

    rank_and_index = [
        (0, 0),
        (0, 0),
        (1, 4),
        (1, 9),
        (22, 10),
        (2, 3),
        (31, 3),
        (3, 110),
    ]

    for idx, file_name in enumerate(file_names):
        rank, index = extract_rank_and_index_from_filename(file_name)
        assert rank == rank_and_index[idx][0]
        assert index == rank_and_index[idx][1]


def test_read_index_file_content(tmpdir, monkeypatch):
    output_dir = tmpdir / "output_dir"

    assert read_index_file_content(_resolve_dir(str(output_dir))) is None

    output_dir.mkdir()
    assert read_index_file_content(_resolve_dir(str(output_dir))) is None

    with open(output_dir / "index.json", "w") as f:
        dummy_dict = {"chunks": ["abc.bin", "def.bin"], "config": {"data_format": "a", "data_spec": "b"}}
        json.dump(dummy_dict, f)

    assert read_index_file_content(_resolve_dir(str(output_dir))) == dummy_dict

    def _fn(remote_path, local_path):
        with open(local_path, "w") as f:
            json.dump(dummy_dict, f)

    fs_provider = MagicMock()
    fs_provider.download_file = _fn

    monkeypatch.setattr(utilities_module, "_get_fs_provider", MagicMock(return_value=fs_provider))
    assert read_index_file_content(_resolve_dir("s3://bucket/path")) == dummy_dict


def test_remove_uuid_from_filename():
    checkpoint_dir = "output/data/train/.checkpoints"
    uuid = "9fe2c4e93f654fdbb24c02b15259716c"

    for rank in (0, 1, 2, 12, 101, 267):
        filepath = f"{checkpoint_dir}/checkpoint-{rank}-{uuid}.json"
        assert remove_uuid_from_filename(filepath) == f"{checkpoint_dir}/checkpoint-{rank}.json"


@pytest.mark.parametrize(
    "filepath",
    [
        "output/data/train/.checkpoints/checkpoint-0.json",
        "input/data/val/.checkpoints/config.json",
        "output/data/train/.checkpoints/checkpoint-0-not-a-uuid.json",
        "output/data/train/checkpoint-0-9fe2c4e93f654fdbb24c02b15259716c.json",
    ],
)
def test_remove_uuid_from_filename_leaves_other_paths_unchanged(filepath):
    assert remove_uuid_from_filename(filepath) == filepath

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
from unittest import mock

import pytest

from litdata.integrations.ultralytics.optimize import get_output_dir, list_all_files, optimize_ultralytics_dataset
from litdata.streaming.resolver import Dir


@mock.patch("litdata.integrations.ultralytics.optimize._ULTRALYTICS_AVAILABLE", True)
@mock.patch("litdata.integrations.ultralytics.optimize.optimize")
def test_optimize_ultralytics_dataset_mocked(optimize_mock, tmp_path, mock_ultralytics):
    os.makedirs(tmp_path / "images/train", exist_ok=True)
    os.makedirs(tmp_path / "images/val", exist_ok=True)
    for split in ["train", "val"]:
        (tmp_path / f"images/{split}/img1.jpg").touch()
        (tmp_path / f"labels/{split}/img1.txt").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"labels/{split}/img1.txt").write_text("0 0.5 0.5 0.2 0.2")

    yaml_file = tmp_path / "coco8.yaml"
    yaml_file.write_text(
        "path: {}\ntrain: {}\nval: {}\n".format(tmp_path, tmp_path / "images" / "train", tmp_path / "images" / "val")
    )

    optimize_ultralytics_dataset(str(yaml_file), str(tmp_path / "out"), chunk_size=1, num_workers=1)

    assert optimize_mock.called
    assert (tmp_path / "litdata_coco8.yaml").exists()


def test_get_output_dir():
    # Case 1: Both url and path provided
    d1 = Dir(path="/data/output", url="s3://bucket/output/")
    r1 = get_output_dir(d1, "train")
    assert r1.path == os.path.join("/data/output", "train")
    assert r1.url == "s3://bucket/output/train"

    # Case 2: Only url provided
    d2 = Dir(url="s3://bucket/output/")
    r2 = get_output_dir(d2, "val")
    assert r2.path is None
    assert r2.url == "s3://bucket/output/val"

    # Case 3: Only path provided
    d3 = Dir(path="/data/output")
    r3 = get_output_dir(d3, "test")
    assert r3.url is None
    assert r3.path == os.path.join("/data/output", "test")

    # Case 4: Neither url nor path provided
    d4 = Dir()
    r4 = get_output_dir(d4, "debug")
    assert r4.url is None
    assert r4.path is None

    # Case 5: Invalid type
    with pytest.raises(TypeError):
        get_output_dir("not_a_dir_obj", "fail")


def test_list_all_files_combined(tmp_path):
    # --- Case 1: Directory with nested files ---
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "b.txt").write_text("world")

    result = list_all_files(str(tmp_path))
    expected = {str(tmp_path / "a.txt"), str(tmp_path / "subdir" / "b.txt")}
    assert set(result) == expected, "Should list all files recursively from directory"

    # --- Case 2: .txt file listing files ---
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()
    txt_file = tmp_path / "train.txt"
    txt_file.write_text("img1.jpg\nimg2.jpg\n")

    result = list_all_files(str(txt_file))
    expected = {
        str((tmp_path / "img1.jpg").resolve()),
        str((tmp_path / "img2.jpg").resolve()),
    }
    assert set(result) == expected, ".txt path list should resolve correctly"

    # --- Case 3: Unsupported file path ---
    bad_file = tmp_path / "unsupported.md"
    bad_file.write_text("invalid")

    with pytest.raises(ValueError, match="Unsupported path"):
        list_all_files(str(bad_file))

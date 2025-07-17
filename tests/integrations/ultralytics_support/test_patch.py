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
from unittest.mock import patch

import numpy as np
import pytest

from litdata.integrations.ultralytics.patch import parse_labels, ultralytics_detection_transform


def make_lit_args(**kwargs):
    return {
        "names": ["class0", "class1"],
        "kpt_shape": (5, 3),
        **kwargs,
    }


def test_parse_labels():
    # --- Basic test ---
    labels = "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.7 0.8"
    lit_args = make_lit_args()
    lb, segments, keypoints = parse_labels(labels, lit_args=lit_args)
    assert lb.shape == (2, 5)
    assert segments == []
    assert keypoints is None
    assert np.all(lb[:, 0] < len(lit_args["names"]))

    # --- Single class ---
    labels = "1 0.1 0.2 0.3 0.4"
    lb, _, _ = parse_labels(labels, lit_args=lit_args, single_cls=True)
    assert np.all(lb[:, 0] == 0)

    # --- With segments ---
    segment_label = "0 " + " ".join([str(round(0.01 * i, 3)) for i in range(14)])  # 7 xy pairs
    lb, segments, _ = parse_labels(segment_label, lit_args=lit_args)
    assert lb.shape == (1, 5)
    assert len(segments) == 1

    # --- With keypoints ---
    keypoint_str = "0 " + " ".join(["0.1"] * 19)
    lb, _, keypoints = parse_labels(keypoint_str, lit_args=lit_args, keypoint=True)
    assert lb.shape == (1, 5)
    assert keypoints.shape == (1, 5, 3)

    # --- Duplicate removal ---
    dup_labels = "0 0.1 0.2 0.3 0.4\n0 0.1 0.2 0.3 0.4"
    lb, _, _ = parse_labels(dup_labels, lit_args=lit_args)
    assert lb.shape == (1, 5)

    # --- Out of bounds class ---
    bad_class_label = "99 0.1 0.2 0.3 0.4"
    with pytest.raises(AssertionError, match="Label class 99 exceeds"):
        parse_labels(bad_class_label, lit_args=lit_args)

    # --- Coordinates out of bounds ---
    bad_coords_label = "0 1.2 1.2 1.2 1.2"
    with pytest.raises(AssertionError, match="non-normalized or out of bounds"):
        parse_labels(bad_coords_label, lit_args=lit_args)

    # --- Negative class ---
    negative_label = "-1 0.1 0.2 0.3 0.4"
    with pytest.raises(AssertionError, match="negative class labels"):
        parse_labels(negative_label, lit_args=lit_args)

    # --- Wrong shape ---
    wrong_shape_label = "0 0.1 0.2 0.3"  # Only 4 elements
    with pytest.raises(AssertionError, match="labels require 5 columns"):
        parse_labels(wrong_shape_label, lit_args=lit_args)


def test_ultralytics_detection_transform():
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    valid_label = "1 0.1 0.2 0.3 0.4\n0 0.5 0.6 0.7 0.8"
    invalid_label = 123  # not a string

    dummy_img_resized = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    ori_shape = dummy_img.shape[:2]
    resized_shape = dummy_img_resized.shape[:2]

    lit_args = {"names": ["class0", "class1"], "kpt_shape": (0, 0)}

    with (
        patch(
            "litdata.integrations.ultralytics.patch.image_resize",
            return_value=(dummy_img_resized, ori_shape, resized_shape),
        ),
        patch(
            "litdata.integrations.ultralytics.patch.parse_labels",
            return_value=(
                np.array([[1, 0.1, 0.2, 0.3, 0.4], [0, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
                [],
                None,
            ),
        ),
    ):
        # ✅ Valid transformation
        data = {"img": dummy_img, "label": valid_label}
        out = ultralytics_detection_transform(data, index=42, channels=3, img_size=640, lit_args=lit_args)

        assert isinstance(out, dict)
        assert out["img"].shape == (640, 640, 3)
        assert out["batch_idx"].item() == 42
        assert out["cls"].shape == (2, 1)
        assert out["bboxes"].shape == (2, 4)
        assert out["segments"] == []
        assert out["keypoints"] is None
        assert out["normalized"] is True
        assert out["bbox_format"] == "xywh"
        assert out["ori_shape"] == ori_shape
        assert out["resized_shape"] == resized_shape
        assert isinstance(out["ratio_pad"], tuple)
        assert out["channels"] == 3

        # Missing index
        with pytest.raises(ValueError, match="Index must be provided"):
            ultralytics_detection_transform(
                {"img": dummy_img, "label": valid_label}, index=None, channels=3, lit_args=lit_args
            )

        # Invalid label type
        with pytest.raises(ValueError, match="Label must be a string"):
            ultralytics_detection_transform(
                {"img": dummy_img, "label": invalid_label}, index=0, channels=3, lit_args=lit_args
            )

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

from collections import OrderedDict, defaultdict, deque, namedtuple
from typing import NamedTuple

import pytest

from litdata.utilities._pytree import (
    tree_flatten,
    tree_iter,
    tree_leaves,
    tree_map,
    tree_unflatten,
    treespec_dumps,
    treespec_loads,
)


def _roundtrip(tree):
    leaves, spec = tree_flatten(tree)
    assert list(tree_iter(tree)) == leaves
    assert tree_leaves(tree) == leaves
    restored = tree_unflatten(leaves, spec)
    assert restored == tree
    assert treespec_loads(treespec_dumps(spec)).num_leaves == spec.num_leaves
    assert tree_unflatten(leaves, treespec_loads(treespec_dumps(spec))) == tree
    return leaves, spec


def test_flatten_primitives_are_single_leaves():
    for value in (0, 1, -3, 1.5, True, False, "hi", b"raw", None):
        leaves, spec = tree_flatten(value)
        assert leaves == [value]
        assert spec.is_leaf()
        assert tree_unflatten(leaves, spec) == value


def test_flatten_nested_dict_list_tuple():
    tree = {"i": 1, "coords": [1.0, 2.0], "flag": True, "meta": ("a", {"b": 3})}
    leaves, spec = _roundtrip(tree)
    assert leaves == [1, 1.0, 2.0, True, "a", 3]
    assert spec.type is dict
    assert spec.context == ["i", "coords", "flag", "meta"]
    assert spec.num_leaves == 6


def test_empty_containers():
    for tree in ([], {}, (), OrderedDict(), defaultdict(int), deque()):
        leaves, spec = tree_flatten(tree)
        assert leaves == []
        assert spec.num_leaves == 0
        assert tree_unflatten(leaves, spec) == tree


def test_ordered_dict_and_defaultdict_and_deque():
    od = OrderedDict([("z", 1), ("a", 2)])
    leaves, spec = _roundtrip(od)
    assert leaves == [1, 2]
    assert spec.type is OrderedDict

    dd = defaultdict(int, {"x": 4})
    leaves, spec = _roundtrip(dd)
    assert leaves == [4]
    assert spec.type is defaultdict
    assert tree_unflatten(leaves, spec).default_factory is int

    dq = deque([1, 2, 3], maxlen=5)
    leaves, spec = _roundtrip(dq)
    assert leaves == [1, 2, 3]
    assert spec.type is deque
    assert tree_unflatten(leaves, spec).maxlen == 5


def test_namedtuple_is_a_node_not_a_tuple_of_leaves_only():
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    leaves, spec = tree_flatten(point)
    assert leaves == [1, 2]
    restored = tree_unflatten(leaves, spec)
    assert restored == point
    assert type(restored) is Point

    class TypedPoint(NamedTuple):
        x: int
        y: int

    typed = TypedPoint(3, 4)
    leaves, spec = tree_flatten(typed)
    assert leaves == [3, 4]
    assert type(tree_unflatten(leaves, spec)) is TypedPoint


def test_dict_and_list_subclasses_are_leaves():
    class MyDict(dict):
        pass

    class MyList(list):
        pass

    wrapped = {"d": MyDict(a=1), "l": MyList([2, 3])}
    leaves, spec = tree_flatten(wrapped)
    assert len(leaves) == 2
    assert leaves[0] is wrapped["d"]
    assert leaves[1] is wrapped["l"]
    assert tree_unflatten(leaves, spec) == wrapped


def test_custom_is_leaf_stops_descent():
    tree = {"keep": [1, 2], "split": [3, 4]}
    leaves, spec = tree_flatten(tree, is_leaf=lambda x: isinstance(x, list) and x == [1, 2])
    assert leaves == [[1, 2], 3, 4]
    assert tree_unflatten(leaves, spec) == tree


def test_tree_map_and_key_order():
    tree = {"b": 2, "a": 1}
    mapped = tree_map(lambda x: x * 10, tree)
    assert mapped == {"b": 20, "a": 10}
    assert list(mapped) == ["b", "a"]


def test_treespec_dumps_stable_for_common_sample():
    tree = {"i": 0, "coords": [0.0, 1.0], "flag": True}
    _, spec = tree_flatten(tree)
    dumped = treespec_dumps(spec)
    assert treespec_dumps(treespec_loads(dumped)) == dumped
    assert "builtins.dict" in dumped
    assert "builtins.list" in dumped


def test_jpeg_array_is_one_leaf():
    pytest.importorskip("PIL")
    import io

    from PIL import Image as PILImage

    frames = []
    for _ in range(2):
        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4)).save(buf, format="JPEG")
        frames.append(PILImage.open(io.BytesIO(buf.getvalue())))
    empty: list = []
    mixed = [frames[0], "not-jpeg"]
    tree = {"images": frames, "empty": empty, "mixed": mixed, "n": 1}
    leaves, spec = tree_flatten(tree)
    # Empty list is a node with zero leaves; JPEG list is a single leaf.
    assert leaves == [frames, frames[0], "not-jpeg", 1]
    restored = tree_unflatten(leaves, spec)
    assert restored["images"] is frames
    assert restored["empty"] == []


def test_writer_fixed_size_layout_roundtrip(tmpdir):
    from litdata.streaming.reader import BinaryReader
    from litdata.streaming.sampler import ChunkedIndex
    from litdata.streaming.writer import BinaryWriter

    writer = BinaryWriter(str(tmpdir), chunk_size=8)
    samples = [{"i": i, "flag": i % 2 == 0, "x": float(i)} for i in range(8)]
    for i, sample in enumerate(samples):
        writer[i] = sample
    writer.done()
    writer.merge()
    assert writer._fixed_header is not None
    assert writer._fixed_body_len == 8 + 1 + 8
    reader = BinaryReader(str(tmpdir))
    for i, sample in enumerate(samples):
        assert reader.read(ChunkedIndex(i, chunk_index=0)) == sample


def test_writer_leaves_match_flatten_after_first_sample(tmpdir):
    from litdata.streaming.writer import BinaryWriter

    samples = [{"i": i, "coords": [float(i), float(i + 1)], "flag": i % 2 == 0} for i in range(8)]
    writer = BinaryWriter(str(tmpdir), chunk_size=8)
    for i, sample in enumerate(samples):
        writer[i] = sample
    writer.done()
    writer.merge()

    from litdata.streaming.reader import BinaryReader
    from litdata.streaming.sampler import ChunkedIndex

    reader = BinaryReader(str(tmpdir))
    for i, sample in enumerate(samples):
        assert reader.read(ChunkedIndex(i, chunk_index=0)) == sample

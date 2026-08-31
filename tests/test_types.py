# Copyright The Lightning AI team.
import json

import pytest

from litdata.streaming.serializers import JsonLeaf
from litdata.types import (
    Image,
    JsonType,
    default_for,
    fuse_schema_json,
    fuse_type,
    infer_type,
    is_arrow_footer_type,
    is_json_row,
    is_nested_type,
    schema_from_json,
    schema_to_json,
    type_from_json,
    type_to_json,
    wrap_for_pytree,
)


def test_infer_and_fuse_optional_fields():
    first = infer_type({"id": "a", "n": 1, "answers": ["x"]})
    second = infer_type({"id": "b", "n": 2, "choices": {"text": ["A"], "label": ["1"]}})
    fused = fuse_type(first, second)
    assert fused.kind == "struct"
    assert fused.fields is not None
    assert fused.fields["id"].kind == "str"
    assert fused.fields["answers"].optional
    assert fused.fields["choices"].optional
    assert fused.fields["choices"].kind == "struct"
    assert fused.fields["choices"].fields["text"].kind == "list"
    assert not fused.fields["n"].optional


def test_schema_json_roundtrip_matches_index_shape():
    schema = infer_type(
        {
            "id": "q1",
            "question": "why?",
            "choices": {"text": ["A", "B"], "label": ["1", "2"]},
            "answers": ["span"],
        }
    )
    payload = schema_to_json(schema)
    assert payload["id"] == "str"
    assert payload["answers"]["type"] == "list"
    assert payload["answers"]["value"] == "str"
    assert payload["choices"]["type"] == "struct"
    assert payload["choices"]["fields"]["text"]["type"] == "list"
    restored = schema_from_json(payload)
    assert restored is not None
    assert schema_to_json(restored) == payload
    assert json.loads(json.dumps(payload)) == payload


def test_null_and_missing_are_optional():
    fused = fuse_type(infer_type({"n": 1}), infer_type({"n": None}))
    assert fused.fields["n"].kind == "int"
    assert fused.fields["n"].optional
    assert fuse_schema_json({"a": "str"}, {"a": "str", "b": {"type": "int", "optional": True}})["b"]["optional"]


def test_wrap_fills_missing_and_wraps_lists():
    schema = infer_type({"id": "a", "answers": ["x"], "choices": {"text": ["A"], "label": ["1"]}})
    out = wrap_for_pytree({"id": "b"}, schema, wrap_leaf=JsonLeaf)
    assert out["id"] == "b"
    assert isinstance(out["answers"], JsonLeaf)
    assert out["answers"].value == []
    assert isinstance(out["choices"]["text"], JsonLeaf)


def test_wrap_does_not_jsonleaf_jpeg_lists():
    """PIL JPEG lists stay unwrapped so jpeg_array can serialize them."""
    pytest.importorskip("PIL")
    import io

    from PIL import Image as PILImage

    frames = []
    for _ in range(2):
        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4)).save(buf, format="JPEG")
        frames.append(PILImage.open(io.BytesIO(buf.getvalue())))
    schema = infer_type({"index": 0, "images": frames})
    out = wrap_for_pytree({"index": 1, "images": frames}, schema, wrap_leaf=JsonLeaf)
    assert out["index"] == 1
    assert out["images"] is frames
    assert not isinstance(out["images"], JsonLeaf)
    assert not is_arrow_footer_type(schema)


def test_wrap_locks_first_sample_keys():
    schema = fuse_type(infer_type({"id": "a"}), infer_type({"id": "b", "extra": ["z"]}))
    locked = wrap_for_pytree({"id": "c", "extra": ["z"]}, schema, keys=["id"], wrap_leaf=JsonLeaf)
    assert list(locked) == ["id"]
    assert "extra" not in locked


def test_is_nested_and_defaults():
    assert is_nested_type(infer_type({"answers": ["x"]}))
    assert not is_nested_type(infer_type({"id": "a", "n": 1}))
    assert is_arrow_footer_type(infer_type({"article": "x", "highlights": "y", "id": "z"}))
    assert is_arrow_footer_type(infer_type({"id": "a", "n": 1}))
    assert is_arrow_footer_type(infer_type({"answers": ["x"]}))
    assert is_arrow_footer_type(infer_type({"img": {"bytes": b"xx", "path": "a.jpg"}, "label": 1}))
    assert is_json_row({"img": {"bytes": b"xx", "path": "a.jpg"}, "label": 1})
    assert not is_arrow_footer_type(infer_type(1))
    assert not is_arrow_footer_type(infer_type({"image": Image(bytes=b"xx", path="a.jpg"), "label": 1}))
    assert default_for(JsonType("list", value=JsonType("str"))) == []
    assert type_to_json(JsonType("str")) == "str"
    assert type_from_json("int").kind == "int"


def _fused_types_sample(i: int):
    row = {"id": f"q{i}", "n": i}
    if i % 2:
        row["answers"] = ["span"] * (i % 3)
    if i > 4:
        row["note"] = "late"
    return row


def test_optimize_writes_fused_types(tmp_path):
    from litdata import optimize
    from litdata.streaming.dataset import StreamingDataset

    out = tmp_path / "opt"
    optimize(fn=_fused_types_sample, inputs=list(range(8)), output_dir=str(out), chunk_size=8, num_workers=1)
    types = json.loads((out / "index.json").read_text())["config"]["types"]
    assert types["id"] == "str"
    assert types["n"] == "int"
    assert types["answers"]["type"] == "list"
    assert types["answers"]["optional"] is True
    assert types["note"]["optional"] is True
    ds = StreamingDataset(str(out))
    assert ds[0]["id"] == "q0"
    assert ds[1]["answers"] == ["span"]
    assert ds[5]["note"] == "late"

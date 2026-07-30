"""TaskSpec: round-trip, derived fields, schema export."""

import json

import pytest
from pydantic import ValidationError

from jobpipe import TaskSpec


def base(**over):
    d = dict(task="embeddings", dataset="org/my-set", model="BAAI/bge-m3",
             engine="tei", output="hf://buckets/org/b/out")
    d.update(over)
    return TaskSpec(**d)


def test_round_trip():
    s = base(limit=1000, num_examples=5000)
    assert TaskSpec(**json.loads(s.model_dump_json())) == s


def test_slug():
    assert base().slug == "embed_my-set"
    assert base(dataset="org/My Set.v2").slug == "embed_my-set-v2"
    assert len(base(dataset="org/" + "x" * 100).slug) == 40


def test_expected_rows_denominator():
    # ceil(min(limit, num_examples) / batch) — the console's batchCap
    assert base(limit=12800, num_examples=10**9).expected_rows == 200
    assert base(limit=None, num_examples=100).expected_rows == 2
    assert base(limit=65, num_examples=None).expected_rows == 2
    assert base(limit=None, num_examples=None).expected_rows is None


def test_validation():
    with pytest.raises(ValidationError):
        base(engine="tgi")  # not a serving arm
    with pytest.raises(ValidationError):
        base(world=0)
    with pytest.raises(ValidationError):
        base(flavor="h100-mega")  # not in the proven flavor set


def test_json_schema_exports():
    schema = TaskSpec.model_json_schema()
    assert set(schema["properties"]) >= {
        "task", "dataset", "config", "split", "column", "model", "engine",
        "flavor", "world", "output", "publish", "batch", "limit", "num_examples",
    }

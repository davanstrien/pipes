"""Shared stubs: no test touches the network."""

import pytest

import jobpipe.introspect as intro
import jobpipe.resolve as res

PINNED = "https://huggingface.co/datasets/uv-scripts/embeddings/raw/abc123/generate-embeddings-saturate.py"


@pytest.fixture
def offline(monkeypatch):
    """A Hub dataset with config default/train, a text and an id column, 10,000 rows."""
    monkeypatch.setattr(intro, "num_examples", lambda *a, **k: 10_000)
    monkeypatch.setattr(intro, "dataset_layout",
                        lambda *a, **k: [{"config": "default", "split": "train"}])
    monkeypatch.setattr(intro, "string_columns",
                        lambda *a, **k: [{"name": "text", "avg_len": 500.0},
                                         {"name": "id", "avg_len": 30.0}])
    monkeypatch.setattr(intro, "dataset_exists", lambda *a, **k: True)
    monkeypatch.setattr(intro, "small_columns", lambda *a, **k: [])
    monkeypatch.setattr(res, "_whoami", lambda token: "tester")
    monkeypatch.setattr(res, "script_path", lambda recipe: PINNED)
    return monkeypatch

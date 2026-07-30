"""resolve(): zero-config filling + provenance receipts."""

import jobpipe.introspect as intro
import jobpipe.resolve as res
from jobpipe.resolve import resolve


def _wire(monkeypatch, *, layout, columns, meta=None):
    monkeypatch.setattr(intro, "dataset_layout", lambda ds, token=None: layout)
    monkeypatch.setattr(
        intro, "string_columns", lambda ds, c, s, token=None: columns
    )
    monkeypatch.setattr(
        intro, "model_metadata",
        lambda m, token=None: meta or {"params": 568_000_000, "model_type": "xlm-roberta"},
    )
    monkeypatch.setattr(res, "_whoami", lambda token: "tester")


def test_zero_config_fills_everything(monkeypatch):
    _wire(
        monkeypatch,
        layout=[{"config": "corpus", "split": "validation"}],
        columns=[{"name": "content", "avg_len": 512.3}, {"name": "title", "avg_len": 40.0}],
    )
    r = resolve("embeddings", "org/ds")
    s = r.spec
    assert (s.config, s.split) == ("corpus", "validation")
    assert r.provenance["config"] == "detected"
    assert s.column == "content" and r.provenance["column"] == "detected"
    assert "longest varying string column" in r.notes["column"]
    assert s.model == "BAAI/bge-m3" and r.provenance["model"] == "curated"
    assert s.engine == "tei" and r.provenance["engine"] == "curated"  # catalogue entry
    assert s.flavor == "a10g-small"
    assert s.output == "hf://datasets/tester/ds-embeddings/data"
    assert r.provenance["output"] == "default"


def test_text_column_preferred_and_default_config_kept(monkeypatch):
    _wire(
        monkeypatch,
        layout=[{"config": "default", "split": "train"},
                {"config": "default", "split": "test"}],
        columns=[{"name": "body", "avg_len": 900.0}, {"name": "text", "avg_len": 300.0}],
    )
    r = resolve("embeddings", "org/ds")
    assert (r.spec.config, r.spec.split) == ("default", "train")
    assert r.spec.column == "text"  # named 'text' beats longer strings


def test_user_values_win_and_skip_model_reads(monkeypatch):
    _wire(monkeypatch, layout=[], columns=[])
    monkeypatch.setattr(
        intro, "model_metadata",
        lambda m, token=None: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    r = resolve(
        "embeddings", "org/ds",
        column="prompt", model="Qwen/Qwen3-Embedding-0.6B", engine="vllm",
        flavor="l4x1", output="hf://buckets/me/out", world=4, limit=1000,
    )
    for f in ("column", "model", "engine", "flavor", "output", "world", "limit"):
        assert r.provenance[f] == "user"
    assert r.spec.world == 4 and r.spec.limit == 1000


def test_no_viewer_falls_back_to_defaults(monkeypatch):
    _wire(monkeypatch, layout=[], columns=[])
    r = resolve("embeddings", "org/private-ds", engine="tei", flavor="a10g-small")
    assert (r.spec.config, r.spec.split, r.spec.column) == ("default", "train", "text")
    assert r.provenance["column"] == "default"
    assert "no preview" in r.notes["column"]


def test_describe_renders_provenance(monkeypatch):
    _wire(
        monkeypatch,
        layout=[{"config": "default", "split": "train"}],
        columns=[{"name": "text", "avg_len": 100.0}],
    )
    out = resolve("embeddings", "org/ds").describe()
    assert "[default]" in out and "[detected]" in out and "'BAAI/bge-m3'" in out


def test_catalogue_model_gets_curated_engine(monkeypatch):
    _wire(monkeypatch, layout=[], columns=[])
    r = resolve("embeddings", "org/ds", model="Qwen/Qwen3-Embedding-0.6B",
                flavor="a10g-small")
    assert r.provenance["model"] == "user"
    assert "in catalogue" in r.notes["model"]
    assert r.spec.engine == "vllm" and r.provenance["engine"] == "curated"


def test_uncatalogued_model_warns_and_detects(monkeypatch):
    _wire(monkeypatch, layout=[], columns=[],
          meta={"params": 4_000_000_000, "model_type": "qwen3"})
    r = resolve("embeddings", "org/ds", model="some/new-embedder")
    assert "not in catalogue" in r.notes["model"]
    assert r.spec.engine == "vllm" and r.provenance["engine"] == "detected"
    assert r.spec.flavor == "a10g-large"  # 4B params -> wizard rule

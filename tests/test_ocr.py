"""OCR task: spec validation, catalogue-driven template render, resolve branch."""

import json

import pytest

import jobpipe.resolve as res
from jobpipe import TaskSpec, compile
from jobpipe.resolve import resolve

GLOB = "hf://buckets/me/scans/**/*.jpg"


def spec(**kw):
    base = dict(task="ocr", dataset=GLOB, model="lightonai/LightOnOCR-2-1B",
                output="hf://buckets/me/scans-ocr", num_examples=None)
    return TaskSpec(**(base | kw))


def test_pdf_glob_refused_with_pointer():
    with pytest.raises(ValueError, match="rasterize stage"):
        spec(dataset="hf://buckets/me/docs/**/*.pdf")


def test_dataset_id_refused():
    with pytest.raises(ValueError, match="bucket glob"):
        spec(dataset="biglam/londons-pulse-moh")


def test_driver_renders_catalogue_serving():
    c = compile(spec(), run_id="ocrgold1")
    # serving values come from the catalogue entry, not template constants
    assert "--max-model-len" in c.driver and "8192" in c.driver
    assert "--mm-processor-cache-gb" in c.driver
    assert "TARGET_SIZE = 1540" in c.driver
    assert "PROMPT = None" in c.driver  # verified image-only trained format
    assert "bucket_rows(GLOB)" in c.driver
    assert "# /// script" in c.driver and "[tool.hf-jobs]" in c.driver
    assert c.slug.startswith("ocr_")


def test_uncatalogued_model_gets_generic_serving():
    c = compile(spec(model="some/new-vlm"), run_id="ocrgold2")
    assert "TARGET_SIZE = 0" in c.driver  # unknown training resolution: no resize
    assert "Convert this page to markdown" in c.driver


def test_resolve_bucket_branch(monkeypatch):
    monkeypatch.setattr(res, "_whoami", lambda token: "tester")
    r = resolve("ocr", GLOB)
    assert r.spec.model == "lightonai/LightOnOCR-2-1B"
    assert r.provenance["model"] == "curated"
    assert r.spec.output == "hf://datasets/tester/scans-ocr/data"
    assert "config" not in r.provenance  # no dataset introspection ran


def test_compile_skips_dataset_introspection():
    import jobpipe.introspect as intro
    called = []
    orig = intro.num_examples
    intro.num_examples = lambda *a, **k: called.append(a)
    try:
        compile(spec(), run_id="ocrgold3")
    finally:
        intro.num_examples = orig
    assert called == []


def test_driver_parses_and_compiles():
    c = compile(spec(), run_id="ocrgold4")
    import ast
    ast.parse(c.driver)  # emitted driver is valid python
    rj = c.run_json
    assert rj["expected_rows"] is None  # honest-None denominator for buckets
    assert json.dumps(rj)  # serializable

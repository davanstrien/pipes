"""Task verbs: flat args -> TaskSpec -> compile; Run wraps launch output."""

import jobpipe.introspect as intro
from jobpipe import Run, embed, generate


def _no_network(monkeypatch):
    import jobpipe.resolve as res

    monkeypatch.setattr(intro, "num_examples", lambda *a, **k: 10_000)
    monkeypatch.setattr(intro, "dataset_layout",
                        lambda *a, **k: [{"config": "default", "split": "train"}])
    monkeypatch.setattr(intro, "string_columns",
                        lambda *a, **k: [{"name": "text", "avg_len": 500.0}])
    monkeypatch.setattr(intro, "model_metadata",
                        lambda *a, **k: {"params": 568_000_000, "model_type": "xlm-roberta"})
    monkeypatch.setattr(res, "_whoami", lambda token: "tester")


def test_embed_compiles_without_launch(monkeypatch):
    _no_network(monkeypatch)
    c = embed("HuggingFaceFW/fineweb-edu", model="BAAI/bge-m3", engine="tei",
              output="hf://buckets/me/out", limit=12800, launch=False)
    assert c.spec.task == "embeddings" and c.spec.batch == 64
    assert c.run_json["items_per_row"] == 64
    # router lifecycle moved into the driver (Engine(cmd=...)), 2026-07-30
    assert "text-embeddings-router" in c.driver
    assert 'saturate[hf]' in c.commands[-1]  # PyPI install, no wheel-curl
    assert c.driver_name.startswith("embed_fineweb-edu-")


def test_generate_compiles_without_launch(monkeypatch):
    _no_network(monkeypatch)
    c = generate("fka/prompts.chat", column="prompt", model="Qwen/Qwen2.5-0.5B-Instruct",
                 output="hf://buckets/me/gen-out", limit=100, launch=False)
    assert c.spec.task == "generation" and c.spec.batch == 1
    assert "from saturate import" in c.driver
    assert c.run_json["expected_rows"] == 100


def test_run_handle_surfaces(monkeypatch):
    _no_network(monkeypatch)
    c = embed("a/b", model="BAAI/bge-m3", engine="tei",
              output="hf://buckets/me/out", launch=False)
    run = Run(run_json={**c.run_json, "jobs": [{"rank": 0, "id": "j1"}]}, spec=c.spec)
    assert run.run_id == c.run_json["run_id"]
    assert run.output == "hf://buckets/me/out"
    assert run.jobs[0]["id"] == "j1"


def test_publish_shell_shape(monkeypatch):
    _no_network(monkeypatch)
    c = embed("a/b", model="BAAI/bge-m3", engine="tei",
              output="hf://buckets/me/out", launch=False)
    run = Run(run_json=c.run_json, spec=c.spec)
    captured = {}

    class FakeJob:
        id = "job1"
        url = "https://hf.co/jobs/x/job1"

    class FakeApi:
        def __init__(self, token=None): ...
        def run_job(self, **kw):
            captured.update(kw)
            return FakeJob()
        def whoami(self):
            return {"name": "me"}

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr("jobpipe.api.PUBLISH_TIMEOUT_S", 60)
    out = run.publish("me/embeds", token="t")
    assert out["repo_id"] == "me/embeds" and out["job_id"] == "job1"
    shell = captured["command"][-1]
    assert 'pip install -q "saturate[hf]"' in shell
    assert "read_output" in shell and "push_to_hub" in shell
    assert captured["flavor"] == "cpu-upgrade"

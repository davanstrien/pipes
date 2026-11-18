"""launch(): run.json before and after each Job; the resume guard."""

import json

import pytest

import jobpipe.launcher as launcher
from jobpipe.resolve import resolve_embed

FLAGS = "".join(f'"{flag}"' for flag in (
    "--model", "--column", "--max-seq-len", "--skip-empty", "--private", "--config", "--split",
    "--id-column", "--prompt", "--keep", "--max-samples", "--no-text", "--shard"))
HEADER = FLAGS + '''
# /// script
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "2h"
# ///
'''


class FakeJob:
    def __init__(self, n):
        self.id = f"job{n}"
        self.url = f"https://hf.co/jobs/tester/job{n}"


class FakeApi:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def whoami(self):
        return {"name": "tester"}

    def create_bucket(self, *a, **k): ...
    def create_repo(self, *a, **k): ...
    def sync_bucket(self, *a, **k):
        self.calls.append(("sync", a, k))

    def run_uv_job(self, script, **kw):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("quota")
        self.calls.append(("job", script, kw))
        return FakeJob(len(self.calls))


@pytest.fixture
def wired(offline, tmp_path, monkeypatch):
    store = {}

    def write(output, run_json, token=None):
        store[output] = json.loads(json.dumps(run_json))
        store.setdefault("history", []).append(len(run_json["jobs"]))

    monkeypatch.setattr(launcher, "write_run_json", write)
    monkeypatch.setattr(launcher, "read_run_json", lambda output, token=None: store.get(output))
    monkeypatch.setattr(launcher, "fetch_text", lambda url, token=None: HEADER)
    monkeypatch.setattr(launcher, "get_token", lambda: "hf_test")
    return store


def test_launch_writes_run_json_around_each_job(wired, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(launcher, "HfApi", lambda token=None: api)
    plan = resolve_embed("org/ds", world=3, keep=["lang"])
    doc = launcher.launch(plan)
    jobs = [c for c in api.calls if c[0] == "job"]
    assert len(jobs) == 3
    _, script, kw = jobs[2]
    assert script == plan.recipe_url
    assert kw["script_args"][-2:] == ["--shard", "2/3"]
    assert kw["image"] == "vllm/vllm-openai:v0.30.0" and kw["flavor"] == "a10g-small"
    assert kw["timeout"] == "2h" and kw["secrets"] == {"HF_TOKEN": "hf_test"}
    assert kw["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert kw["labels"]["run-id"] == doc["run_id"] and kw["labels"]["rank"] == "2"
    # written before the first Job, after each one, and once more at the end
    assert wired["history"] == [0, 1, 2, 3, 3]
    assert doc["state"] == "launched" and doc["expected_rows"] == 10_000


def test_partial_failure_leaves_a_record(wired, monkeypatch):
    api = FakeApi(fail_at=1)
    monkeypatch.setattr(launcher, "HfApi", lambda token=None: api)
    plan = resolve_embed("org/ds", world=3)
    with pytest.raises(RuntimeError):
        launcher.launch(plan)
    record = wired[plan.output]
    assert record["state"] == "launching" and [j["rank"] for j in record["jobs"]] == [0]


def test_resume_guard(wired, monkeypatch):
    monkeypatch.setattr(launcher, "HfApi", lambda token=None: FakeApi())
    first = launcher.launch(resolve_embed("org/ds", output="hf://buckets/me/b/out"))
    again = launcher.launch(resolve_embed("org/ds", output="hf://buckets/me/b/out", world=2))
    assert again["resumes"] == first["run_id"]  # same setup: resume is allowed
    with pytest.raises(launcher.OutputMismatch, match="model"):
        launcher.launch(resolve_embed("org/ds", output="hf://buckets/me/b/out",
                                      model="other/model"))


def test_local_directory_is_uploaded_before_jobs(wired, monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(launcher, "HfApi", lambda token=None: api)
    (tmp_path / "a.txt").write_text("hello")
    launcher.launch(resolve_embed(str(tmp_path)))
    assert api.calls[0][0] == "sync" and api.calls[0][2]["include"] == ["*.txt"]
    assert api.calls[1][0] == "job"


def test_old_recipe_is_refused_before_any_job(wired, monkeypatch):
    from jobpipe.recipes import RecipeTooOld

    api = FakeApi()
    monkeypatch.setattr(launcher, "HfApi", lambda token=None: api)
    monkeypatch.setattr(launcher, "fetch_text", lambda url, token=None: '"--model" "--column"')
    with pytest.raises(RecipeTooOld, match="--shard"):
        launcher.launch(resolve_embed("org/ds"))
    assert [c for c in api.calls if c[0] == "job"] == []

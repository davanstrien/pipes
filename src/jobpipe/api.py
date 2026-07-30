"""Task verbs — the python layer.

One call per common task: build the spec (emitted, not authored), compile the
driver, launch the jobs, hand back a Run whose status() reads the progress
document from storage. `launch=False` returns the CompiledRun for inspection
without spending anything.

    from jobpipe import embed

    run = embed("HuggingFaceFW/fineweb-edu", column="text", model="BAAI/bge-m3",
                output="hf://buckets/me/fw-embeds", world=4, limit=200_000)
    run.spec       # the TaskSpec the verb built
    run.status()   # 0/N with a real denominator before boot; exact counts at end
    run.publish("me/fineweb-edu-embeddings")   # healing reader -> dataset repo (CPU job)

Simple-case posture: verbs ship sensible hand defaults (engine/flavor are
plain arguments); resolving them from model metadata or the IE catalogue is a
later, optional layer — deliberately not a dependency of the simple path.
"""

from __future__ import annotations

from dataclasses import dataclass

from jobpipe.compiler import CompiledRun, compile
from jobpipe.spec import TaskSpec

PUBLISH_TIMEOUT_S = 60 * 60


@dataclass
class Run:
    """A launched run: the intent record plus storage-backed observation."""

    run_json: dict
    spec: TaskSpec

    @property
    def run_id(self) -> str:
        return self.run_json["run_id"]

    @property
    def jobs(self) -> list[dict]:
        return self.run_json.get("jobs", [])

    @property
    def output(self) -> str:
        return self.spec.output

    def status(self, include_jobs: bool = True, token: str | None = None) -> dict:
        from jobpipe.status import status

        return status(self.spec.output, include_jobs=include_jobs, token=token)

    def publish(self, repo_id: str, private: bool = True,
                token: str | None = None, namespace: str | None = None) -> dict:
        """Compile-to-dataset: launch a CPU job that reads the finished output
        through saturate's healing reader and pushes a clean dataset repo.
        Returns {job_id, url, repo_id}. Run it after status() shows every
        shard's marker."""
        from huggingface_hub import HfApi, get_token

        code = (
            "from saturate import read_output\n"
            "from datasets import Dataset\n"
            "def gen():\n"
            f"    for _id, row in read_output({self.spec.output!r}):\n"
            "        yield row\n"
            f"Dataset.from_generator(gen).push_to_hub({repo_id!r}, private={private})\n"
        )
        shell = (
            'pip install -q "saturate[hf]" && '
            f"python3 -c {_sh_quote(code)}"
        )
        api = HfApi(token=token)
        job = api.run_job(
            image="python:3.12-slim",
            command=["bash", "-c", shell],
            flavor="cpu-upgrade",
            secrets={"HF_TOKEN": token or get_token()},
            timeout=PUBLISH_TIMEOUT_S,
            namespace=namespace or api.whoami()["name"],
        )
        return {"job_id": job.id, "url": job.url, "repo_id": repo_id}


def _sh_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _run(spec: TaskSpec, *, launch: bool, token: str | None,
         namespace: str | None, resolution=None) -> Run | CompiledRun:
    compiled = compile(spec, token=token)
    if resolution is not None:
        # zero-config receipts travel with the run (console renders them)
        compiled.run_json["resolved"] = {
            "provenance": resolution.provenance,
            "notes": resolution.notes,
        }
    if not launch:
        return compiled
    from jobpipe.launcher import launch as _launch

    run_json = _launch(compiled.spec, token=token, namespace=namespace, compiled=compiled)
    return Run(run_json=run_json, spec=compiled.spec)


def _resolved(task: str, dataset: str, *, token: str | None, verbose: bool, **user):
    import sys

    from jobpipe.resolve import resolve

    r = resolve(task, dataset, token=token, **user)
    if verbose:
        print(f"resolved {task} spec:\n{r.describe()}", file=sys.stderr)
    return r


def embed(dataset: str, *, model: str | None = None, output: str | None = None,
          column: str | None = None, engine: str | None = None,
          flavor: str | None = None, world: int | None = None,
          batch: int | None = None, limit: int | None = None,
          config: str | None = None, split: str | None = None,
          launch: bool = True, token: str | None = None,
          namespace: str | None = None, verbose: bool = True) -> Run | CompiledRun:
    """Embed a dataset. Everything except the dataset id is optional:
    resolve() fills column/model/engine/flavor/output from the dataset and
    model metadata (hand defaults where noted) and prints the filled spec
    with per-field provenance before anything launches."""
    r = _resolved("embeddings", dataset, token=token, verbose=verbose,
                  model=model, output=output, column=column, engine=engine,
                  flavor=flavor, world=world, batch=batch, limit=limit,
                  config=config, split=split)
    return _run(r.spec, launch=launch, token=token, namespace=namespace, resolution=r)


def ocr(glob: str, *, model: str | None = None, output: str | None = None,
        flavor: str | None = None, world: int | None = None,
        limit: int | None = None, max_tokens: int | None = None,
        launch: bool = True, token: str | None = None,
        namespace: str | None = None, verbose: bool = True) -> Run | CompiledRun:
    """OCR a bucket glob of page images to markdown. Zero-config: the curated
    OCR model (with its receipted serving arrangement + prompt contract)
    resolves from the catalogue; any other model runs as an unverified path.
    PDFs are refused — rasterization is a separate stage (issue #4)."""
    r = _resolved("ocr", glob, token=token, verbose=verbose,
                  model=model, output=output, flavor=flavor, world=world,
                  limit=limit, max_tokens=max_tokens)
    return _run(r.spec, launch=launch, token=token, namespace=namespace, resolution=r)


def generate(dataset: str, *, model: str, output: str | None = None,
             column: str | None = None, max_tokens: int | None = None,
             temperature: float | None = None, flavor: str | None = None,
             world: int | None = None, limit: int | None = None,
             config: str | None = None, split: str | None = None,
             launch: bool = True, token: str | None = None,
             namespace: str | None = None, verbose: bool = True) -> Run | CompiledRun:
    """Prompt-in-column text generation (one request per row, content-hash
    ids). model stays required — there is no honest hand default for open-set
    generation; the rest resolves like embed()."""
    r = _resolved("generation", dataset, token=token, verbose=verbose,
                  model=model, output=output, column=column, flavor=flavor,
                  world=world, limit=limit, max_tokens=max_tokens,
                  temperature=temperature, config=config, split=split)
    return _run(r.spec, launch=launch, token=token, namespace=namespace, resolution=r)

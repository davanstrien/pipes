"""launch(spec): stage the driver, fan out one job per rank, write run.json.

The API path and the printed `hf jobs run` commands are the same payload —
one delivery mode (staged driver), so the copy-paste escape hatch and the
programmatic path can't drift apart.

Run identity is carried on the jobs themselves as labels (`jobpipe`,
`run-id`, `rank`) — no database: a UI recovers the
whole run from a label-filtered job list. run.json carries what labels can't.
"""

from __future__ import annotations

import datetime as _dt
import json
import re

import fsspec
from huggingface_hub import HfApi, get_token

from jobpipe.compiler import CompiledRun, compile
from jobpipe.spec import TaskSpec
from jobpipe.templates import get_template

LABEL = "jobpipe"  # bare marker label
JOB_TIMEOUT_S = 3600


def _ensure_output_container(api: HfApi, output: str) -> None:
    # run.json is written at launch, before any job runs — the container must
    # exist now (also spares status() the existing_ids-on-missing-repo trap).
    stripped = re.sub(r"^hf://(datasets|buckets)/", "", output)
    repo = "/".join(stripped.split("/")[:2])
    if output.startswith("hf://buckets/"):
        api.create_bucket(repo, private=True, exist_ok=True)
    elif output.startswith("hf://datasets/"):
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)


def write_run_json(output: str, run_json: dict, token: str | None = None) -> str:
    # token must be passed explicitly: in hosted contexts (Spaces) there is no
    # ambient cached token, and fsspec alone can't see the private repo the
    # same launch just created (found live: Space launch, 2026-07-30)
    uri = f"{output.rstrip('/')}/run.json"
    with fsspec.open(uri, "w", token=token) as f:
        f.write(json.dumps(run_json, indent=2))
    return uri


def launch(
    spec: TaskSpec,
    token: str | None = None,
    namespace: str | None = None,
    compiled: CompiledRun | None = None,
) -> dict:
    """Returns the launched run.json (also written to {output}/run.json)."""
    api = HfApi(token=token)
    compiled = compiled or compile(spec, token=token)
    spec = compiled.spec
    template = get_template(spec.task)
    namespace = namespace or api.whoami()["name"]

    staging = template.staging_repo(spec)
    api.create_repo(staging, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=compiled.driver.encode(),
        path_in_repo=compiled.driver_name,
        repo_id=staging,
        repo_type="dataset",
    )
    _ensure_output_container(api, spec.output)

    jobs = []
    for rank in range(spec.world):
        job = api.run_job(
            image=template.image(spec),
            command=["bash", "-c", template.job_shell(spec, compiled.driver_name, rank)],
            flavor=spec.flavor,
            secrets={"HF_TOKEN": token or get_token()},
            timeout=JOB_TIMEOUT_S,
            namespace=namespace,
            labels={
                LABEL: "",
                "run-id": compiled.run_id,
                "rank": str(rank),
                # labels must match ^[a-zA-Z0-9._-]*$ (Jobs API validation)
                "name": f"{compiled.slug}-{compiled.run_id}-r{rank}",
            },
        )
        jobs.append(
            {
                "rank": rank,
                "id": job.id,
                "owner": namespace,
                "url": getattr(job, "url", None),
            }
        )

    run_json = dict(compiled.run_json)
    run_json["jobs"] = jobs
    run_json["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    write_run_json(spec.output, run_json, token=token)
    return run_json

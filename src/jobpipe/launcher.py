"""launch(plan): upload inputs, check the output, launch one Job per rank, record run.json.

Run identity is carried on the Jobs as labels (`jobpipe`, `run-id`, `rank`), so a
UI can find a run from a label-filtered Job list with no database. `run.json`,
next to the outputs, records what labels cannot: the script's sha256, the exact
arguments, the expected row count and the fingerprint that guards resume.

`run.json` is written before the first Job is submitted and again after each
one, so a failure part-way through leaves a record of the Jobs already running.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import uuid

import fsspec
from huggingface_hub import HfApi, get_token

from jobpipe.recipes import HF_JOBS_DEFAULT_TIMEOUT, check_flags, parse_jobs_header, read_script
from jobpipe.resolve import Plan

LABEL = "jobpipe"


class OutputMismatch(RuntimeError):
    """The output already holds a run with a different setup."""


def _container(output: str) -> tuple[str, str]:
    """('bucket' | 'dataset', 'ns/name') for an output URI."""
    if output.startswith("hf://buckets/"):
        return "bucket", "/".join(output.removeprefix("hf://buckets/").split("/")[:2])
    return "dataset", "/".join(output.removeprefix("hf://datasets/").split("/")[:2])


def _ensure_output_container(api: HfApi, output: str) -> None:
    kind, repo = _container(output)
    if kind == "bucket":
        api.create_bucket(repo, private=True, exist_ok=True)
    else:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)


def read_run_json(output: str, token: str | None = None) -> dict | None:
    """The output's run.json; None only when there is none. A file that cannot be
    read or parsed raises: the resume guard must not pass by default."""
    try:
        with fsspec.open(f"{output.rstrip('/')}/run.json", "r", token=token) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_run_json(output: str, run_json: dict, token: str | None = None) -> str:
    # pass the token explicitly: a Space has no cached token, and fsspec alone
    # cannot see a private repo the same launch just created
    uri = f"{output.rstrip('/')}/run.json"
    with fsspec.open(uri, "w", token=token) as f:
        f.write(json.dumps(run_json, indent=2))
    return uri


def upload_inputs(api: HfApi, uploads) -> None:
    """Sync local directories to their bucket prefixes (only the chosen file type)."""
    for directory, prefix, pattern in uploads:
        api.create_bucket(_container(prefix)[1], private=True, exist_ok=True)
        print(f"uploading {pattern} from {directory} to {prefix}", file=sys.stderr)
        # the prefix belongs to this directory alone, so mirror it exactly
        api.sync_bucket(directory, prefix, include=[pattern], delete=True, quiet=True)


def check_resume(plan: Plan, token: str | None = None) -> dict | None:
    """Refuse an output that holds a run with another fingerprint (model, text handling...)."""
    earlier = read_run_json(plan.output, token=token)
    if earlier and earlier.get("fingerprint", {}).get("sha256") != plan.fingerprint["sha256"]:
        differs = sorted(
            key for key in plan.fingerprint
            if key != "sha256" and earlier.get("fingerprint", {}).get(key) != plan.fingerprint[key]
        )
        raise OutputMismatch(
            f"{plan.output} holds run {earlier.get('run_id')} with a different setup "
            f"({', '.join(differs)}). Its rows would be skipped as done. "
            "Choose another --output."
        )
    return earlier


def launch(plan: Plan, token: str | None = None, namespace: str | None = None) -> dict:
    """Launch the plan. Returns the run.json document (also written next to the outputs)."""
    api = HfApi(token=token)
    token = token or get_token()
    namespace = namespace or api.whoami()["name"]
    recipe_text = read_script(plan.recipe_url, token=token)
    check_flags(recipe_text, plan.rank_args(0), plan.recipe_url)
    header = parse_jobs_header(recipe_text)

    _ensure_output_container(api, plan.output)
    earlier = check_resume(plan, token=token)
    recipe_sha = hashlib.sha256(recipe_text.encode()).hexdigest()
    earlier_sha = ((earlier or {}).get("recipe") or {}).get("sha256")
    if earlier_sha and earlier_sha != recipe_sha:
        print("warning: the script differs from the one that wrote this output; "
              "rows already written are kept as they are", file=sys.stderr)
    upload_inputs(api, plan.uploads)

    run_id = uuid.uuid4().hex[:8]
    run_json = {
        "version": 2,
        "task": plan.task,
        "run_id": run_id,
        "resumes": earlier.get("run_id") if earlier else None,
        "recipe": {"location": plan.recipe_url, "sha256": recipe_sha},
        "args": plan.args,
        "world": plan.world,
        "flavor": plan.flavor or header.flavor,
        "image": header.image,
        "timeout": plan.timeout or header.timeout or HF_JOBS_DEFAULT_TIMEOUT,
        "fingerprint": plan.fingerprint,
        "expected_items": plan.expected_items,
        # status.py reads rows; one text per request, so rows == items
        "expected_rows": plan.expected_items,
        "items_per_row": 1,
        "resolved": {"provenance": plan.provenance, "notes": plan.notes},
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "jobs": [],
        "state": "launching",
    }
    write_run_json(plan.output, run_json, token=token)

    # The recipes need HF_TOKEN only; other secrets a header names are not ours to fill.
    secrets = {"HF_TOKEN": token}
    env = {**header.env, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        _submit_ranks(api, plan, run_json, header, secrets, env, namespace, token)
    except Exception:
        submitted = ", ".join(f"rank {job['rank']}: {job['id']}" for job in run_json["jobs"])
        print(f"error during launch of run {run_id}. Jobs submitted: {submitted or 'none'}. "
              f"Find them with the label run-id={run_id}.", file=sys.stderr)
        raise
    run_json["state"] = "launched"
    write_run_json(plan.output, run_json, token=token)
    return run_json


def _submit_ranks(api, plan, run_json, header, secrets, env, namespace, token) -> None:
    for rank in range(plan.world):
        job = api.run_uv_job(
            plan.recipe_url,
            script_args=plan.rank_args(rank),
            image=header.image,
            flavor=run_json["flavor"],
            timeout=run_json["timeout"],
            secrets=secrets,
            env=env,
            namespace=namespace,
            labels={LABEL: "", "run-id": run_json["run_id"], "rank": str(rank)},
        )
        run_json["jobs"].append(
            {"rank": rank, "id": job.id, "owner": namespace, "url": getattr(job, "url", None)}
        )
        write_run_json(plan.output, run_json, token=token)

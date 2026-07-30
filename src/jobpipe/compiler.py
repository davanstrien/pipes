"""compile(spec) -> CompiledRun: driver .py + launch commands + run.json.

The driver is generated source, not a call into a generic runner:
- self-contained inside the job (only saturate[hf] at runtime — this
  repo is never a job dependency);
- inspectable before spending (`ip compile` shows the literal code and
  commands before anything is spent);
- every run gets a code artifact: the driver is staged per-run as
  {slug}-{run_id}.py and run.json records its URI + sha256, so
  run <-> exact-code-that-ran is a durable link.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from jobpipe import introspect
from jobpipe.spec import TaskSpec
from jobpipe.templates import get_template
from jobpipe.templates.embeddings import SATURATE_PIN


@dataclass
class CompiledRun:
    spec: TaskSpec
    run_id: str
    slug: str
    driver: str  # the .py text
    driver_name: str  # "{slug}-{run_id}.py" — per-run code artifact
    commands: list[str] = field(default_factory=list)  # stage + launches (+ publish)
    run_json: dict = field(default_factory=dict)  # jobs[]/created_at filled by launch()


def _saturate_version() -> str:
    try:
        from importlib.metadata import version

        return version("saturate")
    except Exception:
        return "unknown"


def _script_header(spec: TaskSpec, template) -> str:
    """PEP 723 metadata + [tool.hf-jobs] table, prepended to every emitted
    driver. Makes the staged artifact a self-describing uv script: deps pin
    saturate (already-emitted drivers never rot against saturate main), and
    the hf-jobs table carries the job config so `hf jobs uv run <driver>`
    needs zero flags once huggingface_hub#4598 lands (plain uv ignores
    unknown tool tables — verified during that PR's review). Until then the
    launch commands below remain the working path; the header is the eject
    contract."""
    secrets = 'secrets = ["HF_TOKEN"]'
    return (
        "# /// script\n"
        '# requires-python = ">=3.11"\n'
        f'# dependencies = ["{SATURATE_PIN}"]\n'
        "#\n"
        "# [tool.hf-jobs]\n"
        f'# flavor = "{spec.flavor}"\n'
        f'# image = "{template.image(spec)}"\n'
        f"# {secrets}\n"
        '# timeout = "4h"\n'
        "# ///\n"
    )


def compile(
    spec: TaskSpec,
    run_id: str | None = None,
    token: str | None = None,
) -> CompiledRun:
    if spec.num_examples is None and spec.task != "ocr":
        # ocr's `dataset` is a bucket glob — the viewer has nothing to say;
        # the denominator stays honest-None (page count unknown pre-listing)
        n = introspect.num_examples(spec.dataset, spec.config, spec.split, token=token)
        spec = spec.model_copy(update={"num_examples": n})

    run_id = run_id or uuid.uuid4().hex[:8]
    template = get_template(spec.task)
    driver = _script_header(spec, template) + template.driver(spec)
    driver_name = f"{spec.slug}-{run_id}.py"

    commands = [template.stage_command(spec, driver_name)]
    commands += template.launch_commands(spec, driver_name)
    if spec.publish:
        # publish target owner = the output's namespace (first path segment)
        owner = spec.output.removeprefix("hf://buckets/").removeprefix(
            "hf://datasets/"
        ).split("/")[0]
        commands.append(template.publish_command(spec, owner))

    # run.json: launch INTENT, written next to the outputs at launch time.
    # saturate records what happened; this records what was meant (G1/G4:
    # expected rows, items_per_row, world are known here and nowhere else).
    run_json = {
        "version": 1,
        "task": spec.task,
        "run_id": run_id,
        "slug": spec.slug,
        "spec": spec.model_dump(),
        "expected_rows": spec.expected_rows,  # contract rows (batches) — G1 denominator
        "items_per_row": spec.batch,  # G4
        "expected_items": spec.expected_items,
        "world": spec.world,  # launch-time cover for G2's dead-on-arrival case
        "num_examples": spec.num_examples,
        "driver": {
            "staged": f"hf://datasets/{template.staging_repo(spec)}/{driver_name}",
            "sha256": hashlib.sha256(driver.encode()).hexdigest(),
        },
        "jobs": [],  # filled by launch(): [{rank, id, owner, url}]
        "created_at": None,  # stamped by launch()
        "saturate_version": _saturate_version(),
    }
    return CompiledRun(
        spec=spec,
        run_id=run_id,
        slug=spec.slug,
        driver=driver,
        driver_name=driver_name,
        commands=commands,
        run_json=run_json,
    )

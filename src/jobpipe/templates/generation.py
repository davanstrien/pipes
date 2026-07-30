"""Generation template: TaskSpec(task="generation") -> driver .py + launch commands.

Prompt-in-column, one request per contract row (batch=1 — rollouts/agent loops
are a different tool), vLLM only, content-hash ids (identical prompts dedupe;
receipt: 38 real dolly duplicates caught by the same id path, saturate
spikes/RESULTS.md Tier-1 C).

Unlike the embeddings template (pinned byte-for-byte to the console POC's
generated driver, pre-rename `pumpjack` wheel), this template targets the
renamed engine: `saturate` imports and the saturate wheel. The input side is
`dataset_rows` from saturate's sources module — streaming by default, `limit=`
handled inside the source, no hand-rolled buffering loop.
"""

from __future__ import annotations

import re

from jobpipe.spec import TaskSpec
from jobpipe.templates.embeddings import (
    SATURATE_PIN,
    VLLM_IMAGE,
    _out_repo_line,
    staging_repo,
)


def _render(template: str, subs: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: subs[m.group(1)], template)


_DRIVER = '''"""{{slug}} — generation pipeline compiled by jobpipe.

Source: {{dataset}} · {{split}}/{{column}} (prompt-in-column)
Model:  {{model}} (vLLM) · {{world}} job(s) on {{flavor}}
Output: {{output}}
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, dataset_rows, pump
from saturate.source import shard_select

MODEL = "{{model}}"
OUTPUT = "{{output}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default={{world}})
    args = ap.parse_args()
{{repo_line}}
    def to_request(row: dict) -> dict:
        return {"model": MODEL,
                "messages": [{"role": "user", "content": row["{{column}}"]}],
                "max_tokens": {{max_tokens}}, "temperature": {{temperature}}}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"{{column}}": row["{{column}}"],
                "response": body["choices"][0]["message"]["content"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    rows = dataset_rows("{{dataset}}", config="{{config}}", split="{{split}}",
                        columns=["{{column}}"], ids="content"{{limit_arg}})
    rows = shard_select(rows, rank=args.rank, world=args.world)
    with Engine(MODEL, engine="vllm",
                extra_args=["--gpu-memory-utilization", "0.90"]) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     window=Auto(initial=8),
                     shard=(args.rank, args.world), flush_every=100)
    print(f"GEN rank={args.rank} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
'''


def driver(spec: TaskSpec) -> str:
    limit_arg = f", limit={spec.expected_items}" if spec.expected_items else ""
    repo_line = _out_repo_line(spec)
    return _render(
        _DRIVER,
        {
            "slug": spec.slug,
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
            "column": spec.column,
            "model": spec.model,
            "flavor": spec.flavor,
            "output": spec.output,
            "world": str(spec.world),
            "max_tokens": str(spec.max_tokens),
            "temperature": str(spec.temperature),
            "limit_arg": limit_arg,
            "repo_line": repo_line,
        },
    )


def image(spec: TaskSpec) -> str:
    return VLLM_IMAGE


def job_shell(spec: TaskSpec, driver_filename: str, rank: int) -> str:
    base = f"https://huggingface.co/datasets/{staging_repo(spec)}/resolve/main"
    return (
        f'pip install -q "{SATURATE_PIN}" && '
        f'curl -sL -H "Authorization: Bearer $HF_TOKEN" {base}/{driver_filename}'
        " -o /tmp/run.py && "
        f"python3 /tmp/run.py --rank {rank} --world {spec.world}"
    )


def stage_command(spec: TaskSpec, driver_filename: str) -> str:
    return (f"hf upload {staging_repo(spec)} {driver_filename} {driver_filename}"
            " --repo-type dataset")


def launch_commands(spec: TaskSpec, driver_filename: str) -> list[str]:
    return [
        f"hf jobs run --flavor {spec.flavor} --secrets HF_TOKEN \\\n"
        f"  {image(spec)} \\\n  bash -c '{job_shell(spec, driver_filename, rank)}'"
        for rank in range(spec.world)
    ]


def publish_command(spec: TaskSpec, owner: str) -> str:
    target = f"{owner}/{spec.slug.replace('_', '-')}"
    return (
        f"# after all {spec.world} shard marker(s) appear in completions/\n"
        "hf jobs run --flavor cpu-upgrade --secrets HF_TOKEN \\\n"
        "  ghcr.io/astral-sh/uv:python3.12-bookworm \\\n"
        f"  bash -c 'uv pip install --system -q \"{SATURATE_PIN}\" && python3 -c \"\n"
        "from saturate import read_output          "
        "# healing reader: error-IS-NULL record wins\n"
        "from datasets import Dataset\n"
        'def gen():\n'
        f'    for _id, row in read_output(\\"{spec.output}\\"): yield row\n'
        f'Dataset.from_generator(gen).push_to_hub(\\"{target}\\", private=True)\n'
        "\"'"
    )

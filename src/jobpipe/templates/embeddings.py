"""Embeddings template: TaskSpec -> driver .py + launch commands.

Modernized 2026-07-30 (console-session steer): saturate imports (the library
renamed from pumpjack on 2026-07-29), `dataset_rows` streaming source instead
of a hand-rolled `load_dataset` loop, engine lifecycle owned by the driver
(TEI included, via Engine(cmd=...)), and PyPI install — saturate 0.1.1 is
live, the wheel-curl scaffolding is gone. Presets keep their receipts:

Serving-arm rule (which engine to default to lives in introspect.py):
- decoder-style embedding model -> vLLM and TEI are a tie at the GPU compute
  ceiling: Qwen3-Embedding-0.6B on a10g-small, identical 400x64-text workload,
  vLLM pooling 37.8k tok/s vs TEI 36.9k (2.3% apart; jobs 6a68b9fc / 6a68bbfc).
  Default vLLM (one image serves every model family).
- classic encoder (BERT-family) -> TEI wins clearly: BAAI/bge-m3, identical
  25,600-text workload, TEI 63.0k tok/s vs vLLM 53.4k (+18%; jobs
  6a68c171a9f4 / 6a68c18115e8).
- NO reranker template yet: TEI's native /rerank returns a bare JSON array,
  which saturate's transport (JSON-object assumption) turns into 100% durable
  error rows (job 6a68c17115e8). Known gap; vLLM /score works.

Scale receipt: the same driver at world=1 vs world=4 (200k fineweb-edu texts,
a10g-small) went 163 -> 633 texts/s, 3.81x, ~$1.70/M texts, with a 22-line
driver diff — fan-out is strided shard_select, no coordinator.

Micro-batching stays caller-side deliberately (rows-only boundary): the
inline `batched()` in the driver is the missing `saturate.source.batched`
combinator (upstream ask, saturate#18). Batch ids are `b-{first-member-id}`
— resume-stable, no counter.
"""

from __future__ import annotations

import re

from jobpipe.spec import TaskSpec

# [hf] extra = huggingface_hub (hf:// output) + datasets (dataset_rows)
SATURATE_PIN = "saturate[hf]>=0.1.1"


def staging_repo(spec: TaskSpec) -> str:
    """Per-run drivers are staged into a repo in the OUTPUT's namespace —
    derivable from the spec with no network, and writable by whoever can
    write the output. Portable by construction: no author-owned repo."""
    stripped = re.sub(r"^hf://(datasets|buckets)/", "", spec.output)
    return f"{stripped.split('/')[0]}/jobpipe-staging"

# TEI image tag follows the GPU architecture; a10g's 86 tag is the proven one
# (86-latest carried both the decoder tie and the +18% encoder win above).
TEI_IMAGES = {
    "t4-small": "turing-latest",
    "a10g-small": "86-latest",
    "a10g-large": "86-latest",
    "l4x1": "89-latest",
    "l40sx1": "89-latest",
}

VLLM_IMAGE = "vllm/vllm-openai:latest"

# vLLM pooling flags: the 32k-text spike (500/500 rows, 7.85M tokens, 33.5k
# tok/s, spikes/RESULTS.md). `--task embed` is gone upstream; current form is
# `--runner pooling --convert embed` — flag churn found by that spike.
VLLM_ENGINE_ARGS = [
    "--runner", "pooling", "--convert", "embed",
    "--max-num-seqs", "256", "--gpu-memory-utilization", "0.90",
]

# Units warning (gap G4): with batched texts a contract "row" is one request of
# `batch` items. The 32k-text spike settled at final_limit=2 REQUESTS — i.e.
# ~128 texts in flight. Neither window values nor advisor hints are
# items-normalized; run.json's items_per_row is how readers recover the factor.

# Window preset: Auto(target_waiting=8, initial=4, max_limit=64) + flush_every=50
# carried every embed run above (32k spike, 4-job fan-out, TEI A/Bs) unchanged.


def tei_image(flavor: str) -> str:
    return f"ghcr.io/huggingface/text-embeddings-inference:{TEI_IMAGES.get(flavor, 'latest')}"


def _render(template: str, subs: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: subs[m.group(1)], template)


# ---------------------------------------------------------------------------
# drivers — saturate-native; the parity test pins the TEI arm against
# tests/golden (regenerated 2026-07-30, provenance in REGEN.md).
# ---------------------------------------------------------------------------

_DRIVER_TEI = '''"""{{slug}} — embeddings pipeline compiled by jobpipe (TEI arm).

Source: {{dataset}} · {{split}}/{{column}}
Model:  {{model}} (TEI) · {{world}} job(s) on {{flavor}}
Output: {{output}}
The driver owns the engine: text-embeddings-router boots on :8080
(health-gated by Engine) and is torn down with the run.
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, dataset_rows, pump
from saturate.source import shard_select

MODEL = "{{model}}"
OUTPUT = "{{output}}"
BATCH = {{batch}}
TRUNCATE = 1200  # chars; guards context blowups client-side (--auto-truncate guards tokens)

TEI_CMD = ["text-embeddings-router", "--model-id", MODEL, "--port", "8080",
           "--max-batch-tokens", "32768", "--max-client-batch-size", "128",
           "--auto-truncate"]


def batched(rows):
    # caller-side micro-batching (rows-only boundary); empty texts skipped,
    # tail batch flushes — a short final batch is data, not a remainder.
    buf = []
    for rid, row in rows:
        text = (row.get("{{column}}") or "").strip()
        if not text:
            continue
        buf.append((rid, text[:TRUNCATE]))
        if len(buf) == BATCH:
            yield f"b-{buf[0][0]}", {"texts": [t for _, t in buf]}
            buf = []
    if buf:
        yield f"b-{buf[0][0]}", {"texts": [t for _, t in buf]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default={{world}})
    args = ap.parse_args()
{{repo_line}}
    def to_request(row: dict) -> dict:
        return {"model": "tei", "input": row["texts"]}

    def parse(row: dict, body: dict) -> dict:
        data = body["data"]
        if len(data) != len(row["texts"]):  # schema-invalid: never mark as success
            raise ValueError(f"expected {len(row['texts'])} embeddings, got {len(data)}")
        return {"texts": row["texts"],
                "embeddings": [d["embedding"] for d in data],
                "n_texts": len(data),
                "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}

    rows = dataset_rows("{{dataset}}", config="{{config}}", split="{{split}}",
                        columns=["{{column}}"]{{limit_arg}})
    rows = shard_select(batched(rows), rank=args.rank, world=args.world)
    with Engine(MODEL, cmd=TEI_CMD, port=8080,
                ready_route="/embeddings",
                ready_payload={"model": "tei", "input": ["ready?"]},
                ready_accept=lambda r: r.status_code == 200) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     route="/embeddings",
                     window=Auto(target_waiting=8, initial=4, max_limit=64),
                     shard=(args.rank, args.world), flush_every=50)
    print(f"EMBED rank={args.rank} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
'''

_DRIVER_VLLM = '''"""{{slug}} — embeddings pipeline compiled by jobpipe (vLLM pooling).

Source: {{dataset}} · {{split}}/{{column}}
Model:  {{model}} (vLLM pooling) · {{world}} job(s) on {{flavor}}
Output: {{output}}
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, dataset_rows, pump
from saturate.source import shard_select

MODEL = "{{model}}"
OUTPUT = "{{output}}"
BATCH = {{batch}}
TRUNCATE = 1200  # chars; guards model max-len client-side

VLLM_ARGS = ["--runner", "pooling", "--convert", "embed",
             "--max-num-seqs", "256", "--gpu-memory-utilization", "0.90"]


def batched(rows):
    # caller-side micro-batching (rows-only boundary); empty texts skipped,
    # tail batch flushes — a short final batch is data, not a remainder.
    buf = []
    for rid, row in rows:
        text = (row.get("{{column}}") or "").strip()
        if not text:
            continue
        buf.append((rid, text[:TRUNCATE]))
        if len(buf) == BATCH:
            yield f"b-{buf[0][0]}", {"texts": [t for _, t in buf]}
            buf = []
    if buf:
        yield f"b-{buf[0][0]}", {"texts": [t for _, t in buf]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default={{world}})
    args = ap.parse_args()
{{repo_line}}
    def to_request(row: dict) -> dict:
        return {"model": MODEL, "input": row["texts"]}

    def parse(row: dict, body: dict) -> dict:
        data = body["data"]
        if len(data) != len(row["texts"]):  # schema-invalid: never mark as success
            raise ValueError(f"expected {len(row['texts'])} embeddings, got {len(data)}")
        return {"texts": row["texts"],
                "embeddings": [d["embedding"] for d in data],
                "n_texts": len(data),
                "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}

    rows = dataset_rows("{{dataset}}", config="{{config}}", split="{{split}}",
                        columns=["{{column}}"]{{limit_arg}})
    rows = shard_select(batched(rows), rank=args.rank, world=args.world)
    with Engine(MODEL, engine="vllm", extra_args=VLLM_ARGS) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     route="/embeddings",
                     window=Auto(target_waiting=8, initial=4, max_limit=64),
                     shard=(args.rank, args.world), flush_every=50)
    print(f"EMBED rank={args.rank} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
'''


def _out_repo_line(spec: TaskSpec) -> str:
    # Pre-create the output container: a first run into a fresh repo/bucket
    # otherwise dies in existing_ids() — glob on a missing repo raises instead
    # of reading as an empty done-set (reported upstream; found by a real
    # launch). Dataset outputs also get a viewer card: without explicit configs
    # the viewer auto-globs ALL parquet and lands on the _manifest sidecars.
    if spec.output.startswith("hf://datasets/"):
        out_type = "dataset"
    elif spec.output.startswith("hf://buckets/"):
        out_type = "bucket"
    else:
        return ""
    stripped = re.sub(r"^hf://(datasets|buckets)/", "", spec.output)
    out_repo = "/".join(stripped.split("/")[:2])
    if out_type == "bucket":
        return (
            "\n    from huggingface_hub import HfApi\n\n"
            f'    HfApi().create_bucket("{out_repo}", private=True, exist_ok=True)\n'
        )
    out_sub = "/".join(stripped.split("/")[2:]) or "."
    return (
        "\n    from huggingface_hub import HfApi\n\n"
        "    api = HfApi()\n"
        f'    api.create_repo("{out_repo}", repo_type="dataset", private=True, exist_ok=True)\n'
        f'    if not api.file_exists("{out_repo}", "README.md", repo_type="dataset"):\n'
        '        card = ("---\\nconfigs:\\n- config_name: data\\n  data_files:\\n"\n'
        f'                "  - split: train\\n    path: {out_sub}/part-*.parquet\\n'
        '  default: true\\n"\n'
        '                "- config_name: manifest\\n  data_files:\\n"\n'
        f'                "  - split: train\\n    path: {out_sub}/_manifest/ids-*.parquet\\n'
        '---\\n"\n'
        '                "# saturate output\\n")\n'
        "        api.upload_file(path_or_fileobj=card.encode(), path_in_repo=\"README.md\",\n"
        f'                        repo_id="{out_repo}", repo_type="dataset")\n'
    )


def driver(spec: TaskSpec) -> str:
    template = _DRIVER_TEI if spec.engine == "tei" else _DRIVER_VLLM
    # limit lives in the source as a texts cap (steer: kills the derived
    # N_BATCHES guard); expected_items folds spec.limit and the viewer count.
    limit_arg = f", limit={spec.expected_items}" if spec.expected_items else ""
    return _render(
        template,
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
            "batch": str(spec.batch),
            "limit_arg": limit_arg,
            "repo_line": _out_repo_line(spec),
        },
    )


# ---------------------------------------------------------------------------
# launch commands — the inspectable/portable artifact. `ip run` executes the
# same payloads via the API; these strings are the copy-paste escape hatch.
# ---------------------------------------------------------------------------


def job_shell(spec: TaskSpec, driver_filename: str, rank: int) -> str:
    """The bash payload for one rank's job — identical text in the printed
    command and in the API launch (one delivery mode: staged driver)."""
    base = f"https://huggingface.co/datasets/{staging_repo(spec)}/resolve/main"
    fetch_run = (
        f'curl -sL -H "Authorization: Bearer $HF_TOKEN" {base}/{driver_filename}'
        " -o /tmp/run.py"
    )
    if spec.engine == "tei":
        # TEI image has no usable python/pip — uv brings a managed 3.12
        # (learned from failed jobs on 2026-07-28; the router now boots inside
        # the driver via Engine(cmd=...), so the shell only installs and runs).
        return (
            "set -e\n"
            "curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            "uv venv /tmp/venv --python 3.12 > /dev/null\n"
            f'uv pip install -q --python /tmp/venv/bin/python "{SATURATE_PIN}"\n'
            f"{fetch_run}\n"
            f"/tmp/venv/bin/python /tmp/run.py --rank {rank} --world {spec.world}"
        )
    return (
        f'pip install -q "{SATURATE_PIN}" && '
        f"{fetch_run} && "
        f"python3 /tmp/run.py --rank {rank} --world {spec.world}"
    )


def image(spec: TaskSpec) -> str:
    return tei_image(spec.flavor) if spec.engine == "tei" else VLLM_IMAGE


def stage_command(spec: TaskSpec, driver_filename: str) -> str:
    return (f"hf upload {staging_repo(spec)} {driver_filename} {driver_filename}"
            " --repo-type dataset")


def launch_commands(spec: TaskSpec, driver_filename: str) -> list[str]:
    cmds = []
    for rank in range(spec.world):
        shell = job_shell(spec, driver_filename, rank)
        if spec.engine == "tei":
            # printed form: `set -e` folds into the quoted bash -c body
            body = shell.removeprefix("set -e\n")
            cmds.append(
                f"hf jobs run --flavor {spec.flavor} --secrets HF_TOKEN \\\n"
                f"  {image(spec)} \\\n  bash -c 'set -e\n{body}'"
            )
        else:
            cmds.append(
                f"hf jobs run --flavor {spec.flavor} --secrets HF_TOKEN \\\n"
                f"  {image(spec)} \\\n  bash -c '{shell}'"
            )
    return cmds


def publish_command(spec: TaskSpec, owner: str) -> str:
    # Compile-to-dataset step: reads finished parts from the output, applies
    # the healing reader rule (error-IS-NULL record wins), pushes a clean
    # dataset. Runs on CPU after all shard markers appear.
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

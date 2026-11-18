"""Embeddings template: TaskSpec -> driver .py + launch commands.

Refreshed 2026-11-18. The driver serves the model with vLLM (`--runner pooling`) and
sends one text per request through saturate; vLLM batches the requests on the GPU.
This is the arrangement of the uv-scripts recipe `generate-embeddings-saturate.py`
(2026-11-10, A10G, 20k IMDb texts: bge-m3 276 rows/s, Qwen3-Embedding-0.6B 187 rows/s).

What changed from the July template:
- One text per request, not batches of 64. Each output row is one input row, with
  its own id (a source column, or saturate's position id) and any kept columns.
- vLLM only. TEI was 18% faster on bge-m3 in July, but saturate 0.3.1 counts TEI's
  capacity in the wrong unit for batch rows, and the TEI image has no uv.
- Texts are cut in tokens by the server (`truncate_prompt_tokens`), not in characters.
- float32 embeddings (the July output was float64, twice the size).
- Input can be a Hub dataset or a bucket glob of parquet/json/csv/text files.
- The driver logs its resident memory every minute (the August run lost ranks to
  memory growth of about 2.4 KB per source row), and exits with os._exit(0)
  after success (saturate README: avoids the exit hang after a Parquet stream).

Fan-out: `world` Jobs read the same input; each keeps the rows whose id hashes to
its rank (saturate shard_of). Every rank still streams the whole input.
"""

from __future__ import annotations

import pprint
import re

from jobpipe.spec import TaskSpec

# [hf] extra = huggingface_hub (hf:// output) + datasets (dataset_rows).
# Exact pin: saturate 0.x minor releases can change the output layout.
SATURATE_PIN = "saturate[hf]==0.3.1"

# Exact tag: `latest` moves with each vLLM release. Generation and OCR use this one.
VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
# Embeddings: the tag the uv-scripts recipe and the 2026-11-18 smoke run used.
EMBED_IMAGE = "vllm/vllm-openai:v0.30.0"


def staging_repo(spec: TaskSpec) -> str:
    """Per-run drivers are staged into a repo in the OUTPUT's namespace —
    derivable from the spec with no network, and writable by whoever can
    write the output. Portable by construction: no author-owned repo."""
    stripped = re.sub(r"^hf://(datasets|buckets)/", "", spec.output)
    return f"{stripped.split('/')[0]}/jobpipe-staging"


def _render(template: str, subs: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: subs[m.group(1)], template)


_DRIVER = '''"""{{slug}} — embeddings compiled by jobpipe (vLLM pooling, one text per request).

Source: {{dataset}}
Model:  {{model}} · {{world}} job(s) on {{flavor}}
Output: {{output}}

Run it again with the same arguments to resume: rows already written are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

SPEC = {{spec_literal}}


def report_memory(every_s: int = 60) -> None:
    """Print this process's resident memory now and then, to stderr."""

    def loop() -> None:
        while True:
            try:
                with open("/proc/self/status") as status:
                    for line in status:
                        if line.startswith("VmRSS"):
                            megabytes = int(line.split()[1]) // 1024
                            print(f"[rss] {megabytes} MB", file=sys.stderr, flush=True)
            except OSError:
                return
            time.sleep(every_s)

    threading.Thread(target=loop, daemon=True).start()


def row_ids():
    """The `ids` argument for dataset_rows: a column name, or the row's position."""
    column = SPEC["id_column"]
    if column is None:
        return "index"
    if column in ("index", "content"):
        # saturate reads these two names as id strategies, so a column with one of
        # them needs a callable.
        return lambda example: example[column]
    return column


def source_rows(shard):
    from saturate import dataset_rows

    columns = [SPEC["column"], *SPEC["keep"]]
    dataset = SPEC["dataset"]
    if dataset.startswith("hf://buckets/"):
        from datasets import load_dataset

        extra = {"sample_by": "document"} if SPEC["source_format"] == "text" else {}
        loaded = load_dataset(SPEC["source_format"], data_files=dataset, split="train",
                              streaming=True, **extra)
        return dataset_rows(loaded, split="train", columns=columns, ids=row_ids(),
                            limit=SPEC["limit"], shard=shard)
    return dataset_rows(dataset, config=SPEC["config"], split=SPEC["split"], columns=columns,
                        ids=row_ids(), limit=SPEC["limit"], shard=shard)


def served_max_tokens(endpoint: str) -> int:
    """The most tokens the server accepts for one text of this model."""
    with urllib.request.urlopen(f"{endpoint}/models", timeout=60) as response:
        for entry in json.load(response)["data"]:
            if entry["id"] == SPEC["model"]:
                return entry["max_model_len"]
    sys.exit(f"the server does not list {SPEC['model']}")


def keep_value(value):
    """Kept columns are written as strings; lists and dicts as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default={{world}})
    args = parser.parse_args()
    shard = (args.rank, args.world)
{{repo_line}}
    import pyarrow as pa
    from saturate import Engine, pump

    report_memory()
    text_column = SPEC["column"]
    rows = source_rows(shard)
    rows = ((row_id, row) for row_id, row in rows if (row[text_column] or "").strip())
    cut = {"tokens": None}  # set once the server reports the model's maximum

    def to_request(row: dict) -> dict:
        return {"model": SPEC["model"], "input": SPEC["prefix"] + row[text_column],
                "truncate_prompt_tokens": cut["tokens"]}

    def parse(row: dict, body: dict) -> dict:
        result = {"embedding": body["data"][0]["embedding"], "model": SPEC["model"],
                  "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens")}
        for column in SPEC["keep"]:
            result[column] = keep_value(row.get(column))
        return result

    # Declared, so a part holding only error rows has the same columns as the rest.
    schema = pa.schema(
        [("id", pa.string()), ("error", pa.string()), ("embedding", pa.list_(pa.float32())),
         ("model", pa.string()), ("prompt_tokens", pa.int64())]
        + [(column, pa.string()) for column in SPEC["keep"]]
    )
    server_args = ["--runner", "pooling", *SPEC["serve_args"]]
    if SPEC["max_seq_len"]:
        server_args += ["--max-model-len", str(SPEC["max_seq_len"])]
    engine = Engine(SPEC["model"], engine="vllm", extra_args=server_args,
                    ready_route="/embeddings",
                    ready_payload={"model": SPEC["model"], "input": "ready"})
    with engine as endpoint:
        cut["tokens"] = served_max_tokens(endpoint)
        print(f"texts are cut at {cut['tokens']} tokens", file=sys.stderr, flush=True)
        started = time.monotonic()
        stats = pump(rows, to_request, parse, endpoint, SPEC["output"], route="/embeddings",
                     shard=shard, schema=schema)
        elapsed = time.monotonic() - started
    print(f"rank {args.rank}/{args.world}: {stats.rows_processed} ok, {stats.rows_failed} failed, "
          f"{stats.rows_processed / max(elapsed, 1e-6):.0f} rows/s", file=sys.stderr)
    print(f"EMBED rank={args.rank} " + stats.to_json(), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # saturate README: avoids the exit hang after a stopped Parquet stream


if __name__ == "__main__":
    main()
'''


def _out_repo_line(spec: TaskSpec) -> str:
    # Pre-create the output container: a first run into a fresh repo/bucket
    # otherwise dies in existing_ids() — glob on a missing repo raises instead
    # of reading as an empty done-set (reported upstream; found by a real
    # launch). Dataset outputs also get a viewer card that names the data parts.
    # saturate 0.3 writes its sidecars under __saturate/, which the viewer and
    # loaders skip, so the card declares one config only.
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
        '  default: true\\n---\\n"\n'
        '                "# saturate output\\n")\n'
        "        api.upload_file(path_or_fileobj=card.encode(), path_in_repo=\"README.md\",\n"
        f'                        repo_id="{out_repo}", repo_type="dataset")\n'
    )


def _spec_literal(spec: TaskSpec) -> str:
    """The values the driver needs, as a Python literal (repr-quoted, so any text is safe)."""
    values = {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "source_format": spec.source_format,
        "column": spec.column,
        "id_column": spec.id_column,
        "keep": list(spec.keep),
        "limit": spec.expected_items,
        "model": spec.model,
        "prefix": spec.prefix,
        "max_seq_len": spec.max_seq_len,
        "serve_args": list(spec.serve_args),
        "output": spec.output,
    }
    return pprint.pformat(values, sort_dicts=False, width=96)


def driver(spec: TaskSpec) -> str:
    return _render(
        _DRIVER,
        {
            "slug": spec.slug,
            "dataset": spec.dataset,
            "model": spec.model,
            "flavor": spec.flavor,
            "output": spec.output,
            "world": str(spec.world),
            "spec_literal": _spec_literal(spec),
            "repo_line": _out_repo_line(spec),
        },
    )


# ---------------------------------------------------------------------------
# launch commands — the inspectable/portable artifact. `ip run` executes the
# same payloads via the API; these strings are the copy-paste escape hatch.
# ---------------------------------------------------------------------------


def job_shell(spec: TaskSpec, driver_filename: str, rank: int) -> str:
    """The bash payload for one rank's job — identical text in the printed
    command and in the API launch (one delivery mode: staged driver).
    `uv run` installs the driver's PEP 723 dependencies; `vllm` comes from the image."""
    base = f"https://huggingface.co/datasets/{staging_repo(spec)}/resolve/main"
    return (
        f'curl -sfL -H "Authorization: Bearer $HF_TOKEN" {base}/{driver_filename}'
        " -o /tmp/run.py && "
        f"uv run /tmp/run.py --rank {rank} --world {spec.world}"
    )


def image(spec: TaskSpec) -> str:
    return EMBED_IMAGE


def stage_command(spec: TaskSpec, driver_filename: str) -> str:
    return (f"hf upload {staging_repo(spec)} {driver_filename} {driver_filename}"
            " --repo-type dataset")


def launch_commands(spec: TaskSpec, driver_filename: str) -> list[str]:
    return [
        f"hf jobs run --flavor {spec.flavor} --timeout {spec.timeout} --secrets HF_TOKEN \\\n"
        f"  -e HF_HUB_DISABLE_PROGRESS_BARS=1 {image(spec)} \\\n"
        f"  bash -c '{job_shell(spec, driver_filename, rank)}'"
        for rank in range(spec.world)
    ]


def publish_command(spec: TaskSpec, owner: str) -> str:
    # Compile-to-dataset step: reads finished parts from the output, applies
    # the healing reader rule (error-IS-NULL record wins), pushes a clean
    # dataset. Runs on CPU after all shard markers appear.
    target = f"{owner}/{spec.slug.replace('_', '-')}"
    return (
        f"# after all {spec.world} shard marker(s) appear in __saturate/completions/\n"
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

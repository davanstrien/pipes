"""OCR template: TaskSpec(task="ocr") -> driver .py + launch commands.

Input is a bucket glob of page images (`hf://buckets/you/scans/**/*.jpg`) via
saturate's `bucket_rows` — ids are file paths, so resume is per-page and a
re-run pays listing, not transfer. One request per page (batch=1), vLLM only.

Mechanism only: everything model-specific (serve args, sampling, resize
target, prompt) interpolates from the catalogue entry's `serving` data at
compile time. Models without a catalogue entry get honest generic defaults
and are labeled unverified by resolve().

PDFs are refused at the spec layer — rasterization is a separate stage
(hf-pipe issue #4), not a driver concern.
"""

from __future__ import annotations

import re

from jobpipe import catalogue
from jobpipe.spec import TaskSpec
from jobpipe.templates.embeddings import (
    SATURATE_PIN,
    VLLM_IMAGE,
    _out_repo_line,
    staging_repo,
)

# generic fallback for uncatalogued models: conservative context math, a plain
# instruction, no resize (we don't know the model's training resolution)
_GENERIC_SERVING = {
    "image": VLLM_IMAGE,
    "max_model_len": 16384,
    "serve_args": ["--limit-mm-per-prompt", '{"image": 1}',
                   "--mm-processor-cache-gb", "0"],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "target_size": 0,
    "prompt": "Convert this page to markdown. Output only the markdown.",
}


def serving_for(model: str) -> dict:
    entry = catalogue.lookup("ocr", model)
    if entry and "serving" in entry:
        s = dict(_GENERIC_SERVING) | dict(entry["serving"])
        # prompts: None on the entry = verified image-only trained format
        if entry.get("prompts", "absent") is None:
            s["prompt"] = None
        return s
    return dict(_GENERIC_SERVING)


def _render(template: str, subs: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: subs[m.group(1)], template)


_DRIVER = '''"""{{slug}} — OCR pipeline compiled by jobpipe.

Source: {{glob}}  (bucket glob; ids = file paths, resume per page)
Model:  {{model}} (vLLM) · {{world}} job(s) on {{flavor}}
Output: {{output}}
"""

from __future__ import annotations

import argparse
import base64
import io

from saturate import Auto, Engine, bucket_rows, pump
from saturate.source import shard_select

MODEL = "{{model}}"
GLOB = "{{glob}}"
OUTPUT = "{{output}}"
TARGET_SIZE = {{target_size}}  # 0 = no resize; else longest side, px
PROMPT = {{prompt}}

VLLM_ARGS = ["--max-model-len", "{{max_model_len}}"{{serve_args}}]


def encode_image(raw: bytes) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if TARGET_SIZE:
        w, h = img.size
        if max(w, h) != TARGET_SIZE:
            scale = TARGET_SIZE / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default={{world}})
    args = ap.parse_args()
{{repo_line}}
    def to_request(row: dict) -> dict:
        b64 = encode_image(row["bytes"])
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        if PROMPT:
            content.insert(0, {"type": "text", "text": PROMPT})
        return {"model": MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": {{max_tokens}},
                "temperature": {{temperature}}, "top_p": {{top_p}}}

    def parse(row: dict, body: dict) -> dict:
        usage = body.get("usage") or {}
        return {"markdown": body["choices"][0]["message"]["content"],
                "model": MODEL,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    rows = bucket_rows(GLOB{{limit_arg}})
    rows = shard_select(rows, rank=args.rank, world=args.world)
    with Engine(MODEL, engine="vllm", extra_args=VLLM_ARGS) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, OUTPUT,
                     window=Auto(initial=8),
                     shard=(args.rank, args.world), flush_every=100)
    print(f"OCR rank={args.rank} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
'''


def driver(spec: TaskSpec) -> str:
    s = serving_for(spec.model)
    serve_args = "".join(f',\n             "{a}"' if not a.startswith("{")
                         else f",\n             '{a}'" for a in s["serve_args"])
    limit_arg = f", limit={spec.limit}" if spec.limit else ""
    return _render(
        _DRIVER,
        {
            "slug": spec.slug,
            "glob": spec.dataset,
            "model": spec.model,
            "flavor": spec.flavor,
            "output": spec.output,
            "world": str(spec.world),
            "target_size": str(s["target_size"]),
            "prompt": repr(s["prompt"]),
            "max_model_len": str(s["max_model_len"]),
            "serve_args": serve_args,
            "max_tokens": str(spec.max_tokens if spec.max_tokens != 512 else s["max_tokens"]),
            "temperature": str(s["temperature"]),
            "top_p": str(s["top_p"]),
            "limit_arg": limit_arg,
            "repo_line": _out_repo_line(spec),
        },
    )


def image(spec: TaskSpec) -> str:
    return serving_for(spec.model)["image"]


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
        "from saturate import read_output\n"
        "from datasets import Dataset\n"
        'def gen():\n'
        f'    for _id, row in read_output(\\"{spec.output}\\"): yield row\n'
        f'Dataset.from_generator(gen).push_to_hub(\\"{target}\\", private=True)\n'
        "\"'"
    )

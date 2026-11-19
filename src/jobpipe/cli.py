"""jobpipe / `hf pipe`: run a tested uv-scripts recipe over a dataset on HF Jobs.

Agent contract: with --json, stdout carries exactly one JSON document and
nothing else; everything for humans goes to stderr. No interactive prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from jobpipe.status import prog


def _hub_url(output: str) -> str | None:
    if output.startswith("hf://datasets/"):
        repo = "/".join(output.removeprefix("hf://datasets/").split("/")[:2])
        return f"https://huggingface.co/datasets/{repo}"
    if output.startswith("hf://buckets/"):
        repo = "/".join(output.removeprefix("hf://buckets/").split("/")[:2])
        return f"https://huggingface.co/buckets/{repo}"
    return None


def cmd_embed(args) -> int:
    from jobpipe.api import embed
    from jobpipe.launcher import OutputMismatch
    from jobpipe.recipes import RecipeTooOld

    try:
        result = embed(
            args.source, model=args.model, column=args.column, id_column=args.id_column,
            keep=[] if args.keep == ["none"] else args.keep,
            max_seq_len=args.max_seq_len, prefix=args.prefix,
            config=args.config, split=args.split, output=args.output, world=args.world,
            limit=args.limit, flavor=args.flavor, timeout=args.timeout,
            recipe_url=args.recipe_url, keep_text=args.keep_text, launch=not args.dry_run,
            token=args.token,
            namespace=args.namespace,
        )
    except (ValueError, OutputMismatch, RecipeTooOld) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return _report(result, args)


def cmd_ocr(args) -> int:
    from jobpipe.api import ocr
    from jobpipe.launcher import OutputMismatch
    from jobpipe.recipes import RecipeTooOld

    try:
        result = ocr(
            args.source, model=args.model, mode=args.mode, image_column=args.image_column,
            id_column=args.id_column, config=args.config, split=args.split, output=args.output,
            world=args.world, limit=args.limit, flavor=args.flavor, timeout=args.timeout,
            recipe_url=args.recipe_url, launch=not args.dry_run, token=args.token,
            namespace=args.namespace,
        )
    except (ValueError, OutputMismatch, RecipeTooOld) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return _report(result, args)


def cmd_extract(args) -> int:
    from jobpipe.api import extract
    from jobpipe.launcher import OutputMismatch
    from jobpipe.recipes import RecipeTooOld

    try:
        result = extract(
            args.source, prompt=args.prompt, schema=args.schema, system=args.system,
            model=args.model, id_column=args.id_column,
            keep=[] if args.keep == ["none"] else args.keep,
            max_tokens=args.max_tokens, max_chars=args.max_chars, config=args.config,
            split=args.split, output=args.output, world=args.world, limit=args.limit,
            flavor=args.flavor, timeout=args.timeout, recipe_url=args.recipe_url,
            launch=not args.dry_run, token=args.token, namespace=args.namespace,
        )
    except (ValueError, OutputMismatch, RecipeTooOld) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return _report(result, args)


def cmd_new(args) -> int:
    """Print a new recipe made from the template (write it to NAME.py and edit its TASK part)."""
    from jobpipe.new import render

    try:
        sys.stdout.write(render(args.name))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"(edit the part between the TASK markers; run it with hf jobs uv run {args.name}.py "
          "INPUT OUTPUT --max-samples 20)", file=sys.stderr)
    return 0


def _report(result, args) -> int:
    """Print a dry-run plan (commands) or a launched run, the same way for every verb."""
    from jobpipe.api import Run

    if not isinstance(result, Run):  # --dry-run: a Plan, nothing launched
        if args.json:
            print(json.dumps({"plan": result.to_json(), "script_command": result.script_command(),
                              "commands": result.commands()}))
        else:
            for directory, prefix, pattern in result.uploads:
                print(f"hf buckets sync {directory} {prefix} --include '{pattern}' --delete")
            if result.script_command():
                print(result.script_command())
            print("\n" + "\n\n".join(result.commands()))
        return 0
    if args.json:
        print(json.dumps(result.run_json))
        return 0
    print(f"run {result.run_id}: {len(result.jobs)} job(s) launched", file=sys.stderr)
    url = _hub_url(result.output)
    print(f"output: {result.output}" + (f"  ({url})" if url else ""), file=sys.stderr)
    print(f"watch: {prog()} status '{result.output}' --watch", file=sys.stderr)
    return 0


ACTIVE_STAGES = ("SCHEDULING", "RUNNING")


def watch_done(doc: dict) -> bool:
    """`status --watch` ends when every Job in run.json has stopped, or, without
    Job stages, when every shard has its completion marker."""
    if doc["jobs"]:
        return not any(job.get("stage") in ACTIVE_STAGES for job in doc["jobs"])
    return bool(doc["shards"]) and all(shard["done"] for shard in doc["shards"])


def cmd_status(args) -> int:
    from jobpipe.status import render_human, status

    while True:
        doc = status(args.output, include_jobs=not args.no_jobs, token=args.token)
        if args.json:
            print(json.dumps(doc), flush=True)
        else:
            print(render_human(doc), flush=True)
        if not args.watch or watch_done(doc):
            return 0
        time.sleep(args.interval)


def cmd_script(args) -> int:
    """Print jobpipe's copy of a recipe, to read, edit or run with `hf jobs uv run`."""
    import pathlib

    from jobpipe.recipes import RECIPES, script_path

    if args.name not in RECIPES:
        print(f"error: no script {args.name!r}; known: {', '.join(sorted(RECIPES))}",
              file=sys.stderr)
        return 2
    sys.stdout.write(pathlib.Path(script_path(RECIPES[args.name])).read_text())
    return 0


def cmd_logs(args) -> int:
    """Job logs for a run, addressed by its output (reads run.json)."""
    import subprocess

    from jobpipe.launcher import read_run_json

    run_json = read_run_json(args.output, token=args.token)
    jobs = (run_json or {}).get("jobs") or []
    if not jobs:
        print(f"no run.json with jobs at {args.output}", file=sys.stderr)
        return 1
    chosen = [job for job in jobs if args.rank is None or job["rank"] == args.rank]
    if not chosen:
        print(f"no job with rank {args.rank}; ranks: {[job['rank'] for job in jobs]}",
              file=sys.stderr)
        return 1
    if args.follow and len(chosen) > 1:
        chosen = chosen[:1]
        print("(-f follows rank 0; use --rank N for another shard)", file=sys.stderr)
    code = 0
    for job in chosen:
        print(f"--- rank {job['rank']}: {job['owner']}/{job['id']} ---", file=sys.stderr)
        command = ["hf", "jobs", "logs", *(["-f"] if args.follow else []), job["id"]]
        code = subprocess.run(command).returncode or code
    return code


def _shared_options(p) -> None:
    """Options every verb takes: input layout, output, scale, hardware, plan or launch."""
    p.add_argument("--config", help="dataset input: default detected (prefers 'default')")
    p.add_argument("--split", help="dataset input: default detected (prefers 'train')")
    p.add_argument("--output", help="hf://buckets/me/b/prefix or me/dataset "
                   "(default: a private bucket under you, named after source and model)")
    p.add_argument("--world", type=int, help="Jobs to fan out over (default 1)")
    p.add_argument("--limit", type=int, help="process only the first N rows or files")
    p.add_argument("--flavor", help="default: the script header's flavor")
    p.add_argument("--timeout", help="per Job, e.g. 30m or 4h (default: the script header's)")
    p.add_argument("--recipe-url", help="run another copy of the script: a path or URL "
                   "(default: the copy shipped with jobpipe)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the hf jobs commands; upload and launch nothing")
    p.add_argument("--json", action="store_true")
    p.add_argument("--token")
    p.add_argument("--namespace", help="Jobs namespace (default: you)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog(), description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "embed",
        help="embed a text column: Hub dataset, bucket glob or local directory",
        description="SOURCE is a Hub dataset id, a bucket glob such as "
        "hf://buckets/me/b/data/*.parquet (.jsonl, .csv, .txt, .md too), or a local "
        "directory (its files are uploaded to a bucket at launch). Runs jobpipe's copy of "
        "the uv-scripts recipe embeddings/generate-embeddings-saturate.py.",
    )
    p.add_argument("source")
    p.add_argument("--model", help="default: BAAI/bge-m3")
    p.add_argument("--column", help="text column (default: detected, else 'text')")
    p.add_argument("--id-column",
                   help="row id column (default: 'id' if the dataset has one, else the position)")
    p.add_argument("--keep", nargs="+", metavar="COLUMN",
                   help="source columns to copy (default: every scalar column and list of "
                   "scalars, except the text; 'none' for no columns)")
    p.add_argument("--max-seq-len", type=int, help="tokens per text (default 512)")
    p.add_argument("--keep-text", action="store_true",
                   help="store the text in the output too (default: only when ids are positions)")
    p.add_argument("--prefix", help="text put before every input (default: the model's "
                   "registered document prompt, chosen by the recipe)")
    _shared_options(p)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser(
        "ocr",
        help="OCR page images to markdown: bucket glob, local folder or dataset image column",
        description="SOURCE is a bucket glob of page images (hf://buckets/me/scans/**/*.jpg; "
        "also .png .tif .webp .jp2), a local folder of them (uploaded to a bucket at launch), "
        "or a Hub dataset with an image column. Runs jobpipe's copy of the uv-scripts recipe "
        "ocr/lighton-ocr3-saturate.py. PDFs are not supported yet.",
    )
    p.add_argument("source")
    p.add_argument("--model", help="default: lightonai/LightOnOCR-3-0.8B (or the 4B)")
    p.add_argument("--mode", choices=["plain", "grounding"], help="default: plain markdown")
    p.add_argument("--image-column", help="dataset input (default: detected)")
    p.add_argument("--id-column", help="dataset input (default: 'id' if present)")
    _shared_options(p)
    p.set_defaults(fn=cmd_ocr)

    p = sub.add_parser(
        "extract",
        help="ask an instruct model about every row, with an optional JSON schema for the reply",
        description="SOURCE is a Hub dataset id, a bucket glob (.parquet .jsonl .csv, or "
        ".txt/.md files: one row per file) or a local directory. --prompt has {column} "
        "placeholders filled from each row. With --schema the server constrains each reply "
        "to that JSON schema, and its top-level properties become output columns. Runs "
        "jobpipe's extract.py, made from the recipe template.",
    )
    p.add_argument("source")
    p.add_argument("--prompt", required=True,
                   help="template text or a file, e.g. 'Classify this review: {text}'")
    p.add_argument("--schema", help="JSON schema text or a .json file (an object schema)")
    p.add_argument("--system", help="system message")
    p.add_argument("--model", help="default: Qwen/Qwen3.5-4B (thinking off)")
    p.add_argument("--id-column", help="dataset input (default: 'id' if present)")
    p.add_argument("--keep", nargs="+", metavar="COLUMN",
                   help="source columns to copy (default: every scalar column and list of "
                   "scalars, except the prompt's; 'none' for no columns)")
    p.add_argument("--max-tokens", type=int, help="longest reply (default 1024)")
    p.add_argument("--max-chars", type=int,
                   help="cut each filled-in value at this many characters (default 20000)")
    _shared_options(p)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("new", help="print a new saturate recipe made from the template")
    p.add_argument("name", help="recipe name, e.g. caption (lower case)")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("status", help="progress of a run, read from its output")
    p.add_argument("output", help="the run's output URI")
    p.add_argument("--json", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--no-jobs", action="store_true", help="storage only, no Jobs API")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("script", help="print the script a task runs (to read, edit or run "
                       "yourself with hf jobs uv run)")
    p.add_argument("name", help="task name, e.g. embed")
    p.set_defaults(fn=cmd_script)

    p = sub.add_parser("logs", help="Job logs for a run, by output URI (reads run.json)")
    p.add_argument("output")
    p.add_argument("--rank", type=int, help="one shard only (default: all; -f implies rank 0)")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_logs)

    args = parser.parse_args(argv)
    # run_uv_job warns on every call that it is experimental; we pin to it knowingly.
    os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
    # Hub upload bars (local-directory sync) drown the provenance block.
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

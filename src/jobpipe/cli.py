"""jobpipe / `hf pipe`: run a pinned uv-scripts recipe over a dataset on HF Jobs.

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
    from jobpipe.api import Run, embed
    from jobpipe.launcher import OutputMismatch
    from jobpipe.recipes import RecipeTooOld

    try:
        result = embed(
            args.source, model=args.model, column=args.column, id_column=args.id_column,
            keep=args.keep, max_seq_len=args.max_seq_len, prefix=args.prefix,
            config=args.config, split=args.split, output=args.output, world=args.world,
            limit=args.limit, flavor=args.flavor, timeout=args.timeout,
            recipe_url=args.recipe_url, keep_text=args.keep_text, launch=not args.dry_run,
            token=args.token,
            namespace=args.namespace,
        )
    except (ValueError, OutputMismatch, RecipeTooOld) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not isinstance(result, Run):  # --dry-run: a Plan, nothing launched
        if args.json:
            print(json.dumps({"plan": result.to_json(), "commands": result.commands()}))
        else:
            if result.uploads:
                for directory, prefix, pattern in result.uploads:
                    print(f"# first: hf buckets sync {directory} {prefix} --include '{pattern}'")
            print("\n\n".join(result.commands()))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog(), description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "embed",
        help="embed a text column: Hub dataset, bucket glob or local directory",
        description="SOURCE is a Hub dataset id, a bucket glob such as "
        "hf://buckets/me/b/data/*.parquet (.jsonl, .csv, .txt, .md too), or a local "
        "directory (its files are uploaded to a bucket at launch). Runs the uv-scripts "
        "recipe embeddings/generate-embeddings-saturate.py, pinned to a commit.",
    )
    p.add_argument("source")
    p.add_argument("--model", help="default: BAAI/bge-m3")
    p.add_argument("--column", help="text column (default: detected, else 'text')")
    p.add_argument("--id-column",
                   help="row id column (default: 'id' if the dataset has one, else the position)")
    p.add_argument("--keep", nargs="+", metavar="COLUMN", help="source columns to copy")
    p.add_argument("--max-seq-len", type=int, help="tokens per text (default 512)")
    p.add_argument("--keep-text", action="store_true",
                   help="store the text in the output too (default: only when ids are positions)")
    p.add_argument("--prefix", help="text put before every input (default: the model's "
                   "registered document prompt, chosen by the recipe)")
    p.add_argument("--config", help="default: detected (prefers 'default')")
    p.add_argument("--split", help="default: detected (prefers 'train')")
    p.add_argument("--output", help="hf://buckets/me/b/prefix or me/dataset "
                   "(default: a private bucket under you, named after source and model)")
    p.add_argument("--world", type=int, help="Jobs to fan out over (default 1)")
    p.add_argument("--limit", type=int, help="embed only the first N rows")
    p.add_argument("--flavor", help="default: the recipe header's flavor")
    p.add_argument("--timeout", help="per Job, e.g. 30m or 4h (default: the recipe header's)")
    p.add_argument("--recipe-url", help="run another copy of the recipe (testing a branch)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the hf jobs commands; upload and launch nothing")
    p.add_argument("--json", action="store_true")
    p.add_argument("--token")
    p.add_argument("--namespace", help="Jobs namespace (default: you)")
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("status", help="progress of a run, read from its output")
    p.add_argument("output", help="the run's output URI")
    p.add_argument("--json", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--no-jobs", action="store_true", help="storage only, no Jobs API")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_status)

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

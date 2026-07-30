"""jobpipe — compile / run / status for dataset-scale inference on HF Jobs.

Agent contract (mirrors the saturate CONTRACT): with --json, stdout carries
exactly the document and nothing else; anything human goes to stderr.
Installed both as `jobpipe` (direct) and `hf-pipe` (the `hf pipe` extension);
spec in, JSON out, no interactive state.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from jobpipe.spec import TaskSpec
from jobpipe.status import prog


def _load_spec(path: str) -> TaskSpec:
    return TaskSpec(**json.loads(pathlib.Path(path).read_text()))


def cmd_compile(args) -> int:
    from jobpipe.compiler import compile

    c = compile(_load_spec(args.spec), token=args.token)
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / c.driver_name).write_text(c.driver)
        (out / "commands.sh").write_text("\n\n".join(c.commands) + "\n")
        (out / "run.json").write_text(json.dumps(c.run_json, indent=2))
        print(f"wrote {c.driver_name}, commands.sh, run.json to {out}", file=sys.stderr)
    if args.json:
        print(json.dumps({"driver_name": c.driver_name, "driver": c.driver,
                          "commands": c.commands, "run_json": c.run_json}))
    elif not args.out:
        print(f"# --- driver: {c.driver_name} ---\n")
        print(c.driver)
        print("# --- commands ---\n")
        print("\n\n".join(c.commands))
    return 0


def cmd_run(args) -> int:
    from jobpipe.launcher import launch

    run_json = launch(_load_spec(args.spec), token=args.token, namespace=args.namespace)
    if args.json:
        print(json.dumps(run_json))
    else:
        print(f"run {run_json['run_id']}: {len(run_json['jobs'])} job(s) launched")
        for j in run_json["jobs"]:
            print(f"  rank {j['rank']}: {j['id']}")
        print(f"run.json -> {run_json['spec']['output'].rstrip('/')}/run.json")
        print(f"watch: {prog()} status '{run_json['spec']['output']}'")
    return 0


def cmd_status(args) -> int:
    from jobpipe.status import render_human, status

    while True:
        doc = status(args.output, include_jobs=not args.no_jobs, token=args.token)
        if args.json:
            print(json.dumps(doc), flush=True)
        else:
            print(render_human(doc), flush=True)
        done = doc["shards"] and all(s["done"] for s in doc["shards"])
        if not args.watch or done:
            return 0
        time.sleep(args.interval)


def _hub_url(output: str) -> str | None:
    if output.startswith("hf://datasets/"):
        repo = "/".join(output.removeprefix("hf://datasets/").split("/")[:2])
        return f"https://huggingface.co/datasets/{repo}"
    if output.startswith("hf://buckets/"):
        repo = "/".join(output.removeprefix("hf://buckets/").split("/")[:2])
        return f"https://huggingface.co/buckets/{repo}"
    return None


def _dump(result, path: str) -> None:
    """Eject: write the compiled driver — a self-describing uv script
    (PEP 723 deps + [tool.hf-jobs] flavor/image) — for a human or agent to
    edit and run. Thin by design: it imports from the pinned saturate."""
    p = pathlib.Path(path)
    p.write_text(result.driver)
    spec = result.run_json["spec"]
    img = next((line.split('"')[1] for line in result.driver.splitlines()
                if line.startswith('# image = ')), None)
    print(f"wrote {p} — edit freely, then run it on Jobs:", file=sys.stderr)
    print(f"  hf jobs uv run --flavor {spec['flavor']}"
          + (f" --image {img}" if img else "")
          + f" -s HF_TOKEN {p}", file=sys.stderr)
    print("(once huggingface_hub#4598 ships, the flags come from the script's"
          " own [tool.hf-jobs] header)", file=sys.stderr)


def _verb(args, fn, **extra) -> int:
    kwargs = dict(model=args.model, output=args.output, column=args.column,
                  flavor=args.flavor, world=args.world, limit=args.limit,
                  config=args.config, split=args.split,
                  launch=not (args.compile_only or args.dump),
                  token=args.token, namespace=args.namespace, **extra)
    result = fn(args.dataset, **kwargs)
    from jobpipe.api import Run

    if isinstance(result, Run):
        doc = result.run_json
        if args.json:
            print(json.dumps(doc))
        else:
            print(f"run {result.run_id}: {len(result.jobs)} job(s) launched")
            url = _hub_url(result.output)
            print(f"output: {result.output}" + (f"  ({url})" if url else ""))
            print(f"watch: {prog()} status '{result.output}'")
    else:  # CompiledRun (--compile-only / --dump: nothing launched)
        if args.dump:
            _dump(result, args.dump)
        if args.json:
            print(json.dumps({"driver_name": result.driver_name, "driver": result.driver,
                              "commands": result.commands, "run_json": result.run_json}))
        elif not args.dump:
            print(result.driver)
    return 0


def cmd_embed(args) -> int:
    from jobpipe.api import embed

    return _verb(args, embed, engine=args.engine, batch=args.batch)


def cmd_generate(args) -> int:
    from jobpipe.api import generate

    return _verb(args, generate, max_tokens=args.max_tokens, temperature=args.temperature)


def cmd_ocr(args) -> int:
    from jobpipe.api import ocr

    kwargs = dict(model=args.model, output=args.output, flavor=args.flavor,
                  world=args.world, limit=args.limit, max_tokens=args.max_tokens,
                  launch=not (args.compile_only or args.dump),
                  token=args.token, namespace=args.namespace)
    result = ocr(args.glob, **kwargs)
    from jobpipe.api import Run

    if isinstance(result, Run):
        print(f"run {result.run_id}: {len(result.jobs)} job(s) launched")
        url = _hub_url(result.output)
        print(f"output: {result.output}" + (f"  ({url})" if url else ""))
        print(f"watch: {prog()} status '{result.output}'")
    else:
        if args.dump:
            _dump(result, args.dump)
        elif args.json:
            print(json.dumps({"driver_name": result.driver_name, "driver": result.driver,
                              "commands": result.commands, "run_json": result.run_json}))
        else:
            print(result.driver)
    return 0


def cmd_schema(args) -> int:
    print(json.dumps(TaskSpec.model_json_schema(), indent=2))
    return 0


def cmd_logs(args) -> int:
    """Job logs for a run, addressed by its OUTPUT (the one handle users have).
    Reads run.json's job list and delegates to `hf jobs logs`."""
    import subprocess

    import fsspec

    with fsspec.open(f"{args.output.rstrip('/')}/run.json", "r") as f:
        rj = json.load(f)
    jobs = rj.get("jobs") or []
    if not jobs:
        print("run.json has no jobs (launched outside the API path?)", file=sys.stderr)
        return 1
    sel = [j for j in jobs if args.rank is None or j["rank"] == args.rank]
    if not sel:
        print(f"no job with rank {args.rank}; ranks: {[j['rank'] for j in jobs]}",
              file=sys.stderr)
        return 1
    if args.follow and len(sel) > 1:
        sel = sel[:1]
        print("(-f follows rank 0; use --rank N for another shard)", file=sys.stderr)
    rc = 0
    for j in sel:
        ref = f"{j['owner']}/{j['id']}"
        print(f"--- rank {j['rank']}: {ref} ---", file=sys.stderr)
        cmd = ["hf", "jobs", "logs"] + (["-f"] if args.follow else []) + [ref]
        rc = subprocess.run(cmd).returncode or rc
    return rc


def cmd_catalogue(args) -> int:
    from jobpipe import catalogue

    if args.json:
        print(json.dumps(catalogue.CATALOGUE if not args.task
                         else {args.task: catalogue.entries(args.task)}))
        return 0
    tasks = [args.task] if args.task else sorted(catalogue.CATALOGUE)
    for t in tasks:
        print(f"{t}:")
        for i, e in enumerate(catalogue.entries(t)):
            star = " (default)" if i == 0 else ""
            print(f"  {e['model']}  [{e.get('engine', '?')}]{star}")
            print(f"    {e['why']}")
            for r in e.get("receipts", []):
                print(f"    receipt {r['date']}  job {r['job']}: {r['note']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog(), description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compile", help="spec.json -> driver + commands + run.json (dry run)")
    p.add_argument("spec")
    p.add_argument("-o", "--out", help="write artifacts to this directory")
    p.add_argument("--json", action="store_true")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("run", help="compile, stage, launch the fan-out, write run.json")
    p.add_argument("spec")
    p.add_argument("--namespace", help="jobs namespace (default: whoami)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="read the run progress document")
    p.add_argument("output", help="output URI (the spec's `output`)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--no-jobs", action="store_true", help="storage channel only")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_status)

    def _verb_args(p, model_required: bool):
        # Everything except the dataset is optional: resolve() fills the rest
        # and prints per-field provenance (user/detected/default) to stderr.
        p.add_argument("dataset")
        p.add_argument("--model", required=model_required,
                       help="default: resolved per task (embeddings: BAAI/bge-m3)")
        p.add_argument("--output", help="default: private dataset repo under you")
        p.add_argument("--column", help="default: detected from the dataset preview")
        p.add_argument("--flavor", help="default: suggested from model size")
        p.add_argument("--world", type=int, help="jobs to fan out (default 1)")
        p.add_argument("--limit", type=int)
        p.add_argument("--config", help="default: detected (prefers 'default')")
        p.add_argument("--split", help="default: detected (prefers 'train')")
        p.add_argument("--compile-only", action="store_true",
                       help="print the driver, launch nothing")
        p.add_argument("--dump", metavar="FILE",
                       help="eject: write the driver as an editable uv script, launch nothing")
        p.add_argument("--json", action="store_true")
        p.add_argument("--token")
        p.add_argument("--namespace")

    p = sub.add_parser("embed", help="embed a dataset (zero-config: `ip embed user/ds`)")
    _verb_args(p, model_required=False)
    p.add_argument("--engine", choices=["vllm", "tei"],
                   help="default: detected from model_type (encoder -> tei)")
    p.add_argument("--batch", type=int)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("generate", help="prompt-in-column generation (verb over run)")
    _verb_args(p, model_required=True)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("schema", help="print the TaskSpec JSON schema (the UI contract)")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("ocr", help="OCR a bucket glob of page images to markdown")
    p.add_argument("glob", help="hf://buckets/owner/name/**/*.jpg (images; PDFs need issue #4)")
    p.add_argument("--model", help="default: curated OCR model from the catalogue")
    p.add_argument("--output", help="default: private dataset repo under you")
    p.add_argument("--flavor")
    p.add_argument("--world", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--dump", metavar="FILE")
    p.add_argument("--json", action="store_true")
    p.add_argument("--token")
    p.add_argument("--namespace")
    p.set_defaults(fn=cmd_ocr)

    p = sub.add_parser("logs", help="job logs for a run, by output URI (reads run.json)")
    p.add_argument("output", help="the run's output URI (same handle as status)")
    p.add_argument("--rank", type=int, help="one shard only (default: all; -f implies rank 0)")
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("catalogue", help="curated model paths per task, with run receipts")
    p.add_argument("task", nargs="?", help="task to show (default: all)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_catalogue)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

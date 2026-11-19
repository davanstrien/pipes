# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.2",
#     "huggingface-hub>=1.12",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///
"""
extract: one request per row to a vLLM server in the Job, driven by saturate.

    hf jobs uv run extract.py INPUT OUTPUT --max-samples 20

Made from the jobpipe recipe template (`hf pipe new extract`). The parts between the
TASK markers are this recipe's own; the rest is the template's plumbing, the same in every
recipe made from it:

INPUT: a Hub dataset id, or a bucket glob. A glob of .parquet, .jsonl, .json or .csv files
is read as a table. A glob of other files (see FILE_SUFFIXES) gives one row per file, with
its path (relative to the glob's fixed prefix) as the id.

OUTPUT: a dataset id (the parts go in its data/), or a bucket path hf://buckets/you/b/run1.
One record per input row: `id`, `error`, the task's columns, and the --keep columns.
Run the same command again to resume; --retry-errors tries the failed rows again.

FAN-OUT: --shard 2/4 processes shard 2 of 4 (a hash of the row id). Start one Job per rank
with the same output. Use a bucket output for fan-out: each flush to a dataset is a commit.
"""

import argparse
import json
import os
import shlex
import sys
import time

TASK_NAME = "extract"
TABLE_FORMATS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv"}


# ==== TASK: settings ===========================================================
# extract: fill a prompt template from each row, ask an instruct model, and (with --schema)
# get JSON back that the server constrains to the schema. Each top-level property of the
# schema becomes an output column.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
FILE_SUFFIXES = {".txt", ".md"}  # one row per file for a bucket glob of these
ROUTE = "/chat/completions"
ENGINE_ARGS = ["--max-model-len", "16384"]
FIXED_COLUMNS = ("raw", "model", "finish_reason", "prompt_tokens", "completion_tokens")


def add_task_arguments(parser):
    parser.add_argument("--prompt", required=True,
                        help="Prompt template; {column} is replaced by the row's value, e.g. "
                             "'Classify this review: {text}'")
    parser.add_argument("--schema", default=None,
                        help="JSON schema (an object) the reply must match. Each top-level "
                             "property becomes a column. Without it the reply is free text in `raw`.")
    parser.add_argument("--system", default=None, help="System message")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Longest reply, in tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Cut each filled-in value at this many characters")
    parser.add_argument("--thinking", action="store_true",
                        help="Leave the model's thinking on (off by default: it uses the token budget)")


def prompt_fields(prompt):
    import string

    return [name for _, name, _, _ in string.Formatter().parse(prompt) if name]


def check_task_arguments(args, parser):
    fields = prompt_fields(args.prompt)
    if not fields:
        parser.error("--prompt has no {column} to fill in, e.g. 'Summarise: {text}'")
    args.schema_obj = None
    if args.schema is not None:
        try:
            args.schema_obj = json.loads(args.schema)
        except json.JSONDecodeError as error:
            parser.error(f"--schema is not JSON: {error}")
        if args.schema_obj.get("type") != "object" or not args.schema_obj.get("properties"):
            parser.error("--schema must be an object schema with properties")
        clashes = [name for name in args.schema_obj["properties"]
                   if name in ("id", "error", *FIXED_COLUMNS)]
        if clashes:
            parser.error(f"--schema properties {clashes} clash with output columns {FIXED_COLUMNS}")


def input_columns(args):
    return prompt_fields(args.prompt)


def file_row(args, path, data):
    """A file fills every {field} of the prompt with its text; {path} with its path."""
    text = data.decode("utf-8", errors="replace")
    return {name: (path if name == "path" else text) for name in prompt_fields(args.prompt)}


def skip_row(args, row):
    """Rows whose fields are all empty get no request."""
    return not any(str(row.get(name) or "").strip() for name in prompt_fields(args.prompt))


def arrow_type(prop):
    import pyarrow as pa

    scalar = {"string": pa.string(), "integer": pa.int64(), "number": pa.float64(),
              "boolean": pa.bool_()}
    kind = prop.get("type", "string" if "enum" in prop else None)
    if isinstance(kind, list):  # ["string", "null"]
        kind = next((k for k in kind if k != "null"), None)
    if kind in scalar:
        return scalar[kind]
    if kind == "array":
        item = (prop.get("items") or {}).get("type")
        if item in scalar:
            return pa.list_(scalar[item])
    return None  # anything else is written as a JSON string


def output_fields(args):
    import pyarrow as pa

    fields = []
    if args.schema_obj:
        for name, prop in args.schema_obj["properties"].items():
            fields.append((name, arrow_type(prop) or pa.string()))
    return fields + [("raw", pa.string()), ("model", pa.string()), ("finish_reason", pa.string()),
                     ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64())]


def ready_payload(args):
    return {"model": args.model, "messages": [{"role": "user", "content": "ready"}],
            "max_tokens": 1}


def to_request(args, row):
    values = {}
    for name in prompt_fields(args.prompt):
        value = "" if row.get(name) is None else str(row[name])
        values[name] = value[:args.max_chars] if args.max_chars else value
    messages = [{"role": "user", "content": args.prompt.format_map(values)}]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})
    body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": args.temperature}
    if not args.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.schema_obj:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "extract", "schema": args.schema_obj}}
    return body


def parse(args, row, body):
    choice = body["choices"][0]
    raw = choice["message"]["content"] or ""
    usage = body.get("usage") or {}
    record = {"raw": raw, "model": args.model, "finish_reason": choice.get("finish_reason"),
              "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens")}
    if args.schema_obj:
        if choice.get("finish_reason") == "length":
            raise ValueError(f"reply cut at --max-tokens {args.max_tokens}: JSON incomplete")
        reply = json.loads(raw)
        for name, prop in args.schema_obj["properties"].items():
            value = reply.get(name)
            record[name] = value if arrow_type(prop) is not None or value is None \
                else json.dumps(value, ensure_ascii=False)
    return record
# ==== END TASK =================================================================


def parse_shard(text):
    """'2/4' -> (2, 4). The rank counts from 0."""
    try:
        rank_text, world_text = text.split("/")
        rank, world = int(rank_text), int(world_text)
    except ValueError:
        sys.exit(f"--shard takes RANK/WORLD, for example 0/4; got {text!r}")
    if world < 1 or not 0 <= rank < world:
        sys.exit(f"--shard {text}: the rank must be from 0 to {world - 1}")
    return rank, world


def bucket_id(uri):
    """'hf://buckets/ns/name/any/prefix' -> 'ns/name'."""
    return "/".join(uri.removeprefix("hf://buckets/").split("/")[:2])


def glob_root(pattern):
    """The fixed directory a glob starts from: every path segment before the first one
    with a wildcard (*, ? or [). For a pattern with no wildcard, its parent directory."""
    scheme, separator, rest = pattern.partition("://")
    if not separator:
        scheme, rest = "", pattern
    fixed = []
    for segment in rest.split("/")[:-1]:
        if any(character in segment for character in "*?["):
            break
        fixed.append(segment)
    return f"{scheme}{separator}" + "/".join(fixed)


def row_ids(args):
    """saturate's id rule: the row's position, or a column's value."""
    if args.id_column is None:
        return "index"
    if args.id_column in ("index", "content"):
        # saturate reads these two names as id strategies, so such a column needs a callable.
        def ids(example):
            return example[args.id_column]

        return ids
    return args.id_column


def source_features(args):
    """The Hub dataset's column types, or None when the repo does not declare them."""
    try:
        from datasets import load_dataset_builder

        return load_dataset_builder(args.input, args.config).info.features
    except Exception as error:  # noqa: BLE001  any failure here: kept columns become strings
        print(f"Column types unknown ({type(error).__name__}); kept columns are written as strings",
              file=sys.stderr)
        return None


def kept_column_types(features, keep):
    """Arrow type for each kept column from the source features; None means 'write a string'."""
    from datasets import Features

    types = {}
    for column in keep:
        feature = (features or {}).get(column)
        if feature is None:
            types[column] = None
            continue
        if type(feature).__name__ in ("Image", "Audio", "Video", "Pdf"):
            sys.exit(f"--keep {column}: media columns cannot be copied; keep a path or id column instead")
        types[column] = Features({column: feature}).arrow_schema.field(column).type
    return types


def keep_value(value):
    """Kept columns of unknown type are written as strings; lists and dicts as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def input_rows(args, shard):
    """(rows, features) for a Hub dataset, a bucket glob of tables, or a bucket glob of files."""
    from saturate import bucket_rows, dataset_rows

    columns = list(dict.fromkeys([*input_columns(args), *args.keep]))
    if not args.input.startswith("hf://buckets/"):
        rows = dataset_rows(args.input, config=args.config, split=args.split, columns=columns,
                            ids=row_ids(args), limit=args.max_samples, shard=shard)
        return rows, source_features(args)
    suffix = os.path.splitext(args.input)[1].lower()
    if suffix in TABLE_FORMATS:
        from datasets import load_dataset

        loaded = load_dataset(TABLE_FORMATS[suffix], data_files=args.input, split="train",
                              streaming=True)
        rows = dataset_rows(loaded, split="train", columns=columns, ids=row_ids(args),
                            limit=args.max_samples, shard=shard)
        return rows, loaded.features
    if suffix not in FILE_SUFFIXES:
        sys.exit(f"A bucket input is a glob ending in one of "
                 f"{sorted([*TABLE_FORMATS, *FILE_SUFFIXES])}; got {args.input!r}")
    if args.keep:
        sys.exit("--keep needs a table input; files have no columns to keep")
    if args.id_column:
        print("--id-column is ignored for files: the id is the file path", file=sys.stderr)
    found = bucket_rows(args.input, root=glob_root(args.input), limit=args.max_samples,
                        shard=shard)
    return ((path, file_row(args, path, row["bytes"])) for path, row in found), None


def open_output(args):
    """Create the output bucket or dataset if needed; return saturate's output URI."""
    from huggingface_hub import HfApi

    if args.output.startswith("hf://buckets/"):
        HfApi().create_bucket(bucket_id(args.output), private=args.private, exist_ok=True)
        return args.output
    HfApi().create_repo(args.output, repo_type="dataset", private=args.private, exist_ok=True)
    return f"hf://datasets/{args.output}/data"


def report_memory(every_s=60):
    """Print this process's resident memory every minute, to stderr (Linux only)."""
    import threading

    def loop():
        while True:
            try:
                with open("/proc/self/status") as status:
                    for line in status:
                        if line.startswith("VmRSS"):
                            print(f"[rss] {int(line.split()[1]) // 1024} MB", file=sys.stderr, flush=True)
            except OSError:
                return
            time.sleep(every_s)

    threading.Thread(target=loop, daemon=True).start()


def write_card(args, stats):
    """A short dataset card with the command that made the output (dataset output only)."""
    from huggingface_hub import DatasetCard

    command = " ".join(shlex.quote(part) for part in sys.argv[1:])
    text = (f"---\ntags:\n- uv-script\n- {TASK_NAME}\nconfigs:\n- config_name: default\n"
            f"  data_files: data/*.parquet\n---\n\n# {args.output}\n\n"
            f"Made with the `{TASK_NAME}` recipe from `{args.input}` and `{args.model}`.\n\n"
            f"```\n{TASK_NAME}.py {command}\n```\n\n"
            f"Rows written by shard 0: {stats.rows_processed} ok, {stats.rows_failed} errors.\n")
    DatasetCard(text).push_to_hub(args.output, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", help="Hub dataset id, or a bucket glob hf://buckets/you/b/data/*.parquet")
    parser.add_argument("output", help="Dataset id (created if missing), or a bucket path hf://buckets/you/b/run1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="A model the vLLM server serves")
    parser.add_argument("--config", default=None, help="Dataset config name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--id-column", default=None,
                        help="Column to use as the row id (default: the row's position)")
    parser.add_argument("--max-samples", type=int, default=None, help="Process only the first N rows")
    parser.add_argument("--keep", nargs="+", default=[], metavar="COLUMN",
                        help="Source columns to copy into the output, with their types")
    parser.add_argument("--private", action="store_true", help="Make a new output private")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Try again the rows whose only record is an error")
    parser.add_argument("--shard", default="0/1", metavar="RANK/WORLD",
                        help="Process only shard RANK of WORLD (fan-out over several Jobs), e.g. 0/4")
    add_task_arguments(parser)
    args = parser.parse_args()
    check_task_arguments(args, parser)
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    shard = parse_shard(args.shard)
    args.keep = list(dict.fromkeys(args.keep))  # a column named twice is kept once

    task_fields = output_fields(args)
    taken = {"id", "error", *(name for name, _ in task_fields)}
    clashes = [column for column in args.keep if column in taken]
    if clashes:
        sys.exit(f"--keep {' '.join(clashes)}: already an output column "
                 f"(the output has {', '.join(sorted(taken))})")

    import pyarrow as pa
    from saturate import Engine, pump

    output_uri = open_output(args)
    rows, features = input_rows(args, shard)
    keep_types = kept_column_types(features, args.keep)
    rows = ((row_id, row) for row_id, row in rows if not skip_row(args, row))
    report_memory()

    # Declared, so a part holding only error rows has the same columns as the rest.
    schema = pa.schema([("id", pa.string()), ("error", pa.string()), *task_fields]
                       + [(column, keep_types[column] or pa.string()) for column in args.keep])

    def request(row):
        return to_request(args, row)

    def result(row, body):
        record = parse(args, row, body)
        for column in args.keep:
            value = row.get(column)
            record[column] = value if keep_types[column] is not None else keep_value(value)
        return record

    engine = Engine(args.model, engine="vllm", extra_args=ENGINE_ARGS,
                    ready_route=ROUTE, ready_payload=ready_payload(args))
    with engine as endpoint:
        started = time.monotonic()
        stats = pump(rows, request, result, endpoint, output_uri, route=ROUTE,
                     retry_errors=args.retry_errors, schema=schema, shard=shard)
        elapsed = time.monotonic() - started

    if not args.output.startswith("hf://buckets/") and shard[0] == 0:
        write_card(args, stats)
    print(f"{args.output} shard {shard[0]}/{shard[1]} ({stats.rows_processed} ok, "
          f"{stats.rows_failed} error rows, {stats.rows_processed / max(elapsed, 1e-6):.1f} rows/s)",
          file=sys.stderr)
    print(f"{TASK_NAME.upper()}_SATURATE " + stats.to_json(), flush=True)
    sys.stderr.flush()
    # saturate README: a process can hang or abort at interpreter exit after a Hub Parquet
    # stream stopped early (--max-samples); everything is written, so leave now.
    os._exit(0)


if __name__ == "__main__":
    main()

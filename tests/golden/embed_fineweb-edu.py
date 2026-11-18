# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]==0.3.1"]
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# image = "vllm/vllm-openai:v0.30.0"
# secrets = ["HF_TOKEN"]
# timeout = "2h"
# ///
"""embed_fineweb-edu — embeddings compiled by jobpipe (vLLM pooling, one text per request).

Source: HuggingFaceFW/fineweb-edu
Model:  BAAI/bge-m3 · 1 job(s) on a10g-small
Output: hf://buckets/davanstrien/pumpjack/fineweb-edu-bge-m3

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

SPEC = {'dataset': 'HuggingFaceFW/fineweb-edu',
 'config': 'default',
 'split': 'train',
 'source_format': None,
 'column': 'text',
 'id_column': 'id',
 'keep': ['url'],
 'limit': 12800,
 'model': 'BAAI/bge-m3',
 'prefix': '',
 'max_seq_len': 512,
 'serve_args': [],
 'output': 'hf://buckets/davanstrien/pumpjack/fineweb-edu-bge-m3'}


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
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    shard = (args.rank, args.world)

    from huggingface_hub import HfApi

    HfApi().create_bucket("davanstrien/pumpjack", private=True, exist_ok=True)

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

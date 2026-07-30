# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1"]
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# image = "ghcr.io/huggingface/text-embeddings-inference:86-latest"
# secrets = ["HF_TOKEN"]
# timeout = "4h"
# ///
"""embed_fineweb-edu — embeddings pipeline compiled by jobpipe (TEI arm).

Source: HuggingFaceFW/fineweb-edu · train/text
Model:  BAAI/bge-m3 (TEI) · 1 job(s) on a10g-small
Output: hf://buckets/davanstrien/pumpjack/fineweb-edu-bge-m3
The driver owns the engine: text-embeddings-router boots on :8080
(health-gated by Engine) and is torn down with the run.
"""

from __future__ import annotations

import argparse

from saturate import Auto, Engine, dataset_rows, pump
from saturate.source import shard_select

MODEL = "BAAI/bge-m3"
OUTPUT = "hf://buckets/davanstrien/pumpjack/fineweb-edu-bge-m3"
BATCH = 64
TRUNCATE = 1200  # chars; guards context blowups client-side (--auto-truncate guards tokens)

TEI_CMD = ["text-embeddings-router", "--model-id", MODEL, "--port", "8080",
           "--max-batch-tokens", "32768", "--max-client-batch-size", "128",
           "--auto-truncate"]


def batched(rows):
    # caller-side micro-batching (rows-only boundary); empty texts skipped,
    # tail batch flushes — a short final batch is data, not a remainder.
    buf = []
    for rid, row in rows:
        text = (row.get("text") or "").strip()
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
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    HfApi().create_bucket("davanstrien/pumpjack", private=True, exist_ok=True)

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

    rows = dataset_rows("HuggingFaceFW/fineweb-edu", config="default", split="train",
                        columns=["text"], limit=12800)
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

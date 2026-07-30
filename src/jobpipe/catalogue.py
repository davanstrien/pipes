"""Curated model catalogue — tested paths per task, as DATA with receipts.

This is the layer's third resolution source (user > curated > detected >
default) and a working sample of the row format a Scripts-repo-type registry
would serve: model + serving arrangement + receipts, machine-readable.
Long-term this table is FETCHED (pinned Scripts repos / IE catalog) rather
than shipped; the builtin dict keeps the demo honest without front-running
that product conversation. Rules:

- every entry carries at least one run receipt (date + job + result) —
  an entry without receipts is not curated, it's a guess;
- `prompts` is explicit: None means "verified prompt-free", absent means
  unknown (models needing doc/query prompts get entries only once the
  driver applies them — doc/query prompt support is a known gap);
- entries are consumed by resolve(), so they cannot drift silently from
  what runs (the uv-scripts SERVING-dict rule).
"""

from __future__ import annotations

CATALOGUE: dict[str, list[dict]] = {
    "ocr": [
        {
            "model": "lightonai/LightOnOCR-2-1B",
            "engine": "vllm",
            "why": "1B Apache-2.0 page-to-markdown; deepest receipts in the collection",
            "prompts": None,  # trained format is the image ONLY (model card)
            # per-value provenance: serve_args + sampling + 1540px = card
            # verbatim; max_model_len 8192 = house choice (halves KV on 24GB,
            # fits a 1540px page + 4096 output) — same split as the uv-scripts
            # SERVING dict this mirrors
            "serving": {
                "image": "vllm/vllm-openai:latest",
                "max_model_len": 8192,
                "serve_args": ["--limit-mm-per-prompt", '{"image": 1}',
                               "--mm-processor-cache-gb", "0",
                               "--no-enable-prefix-caching"],
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.9,
                "target_size": 1540,
            },
            "receipts": [
                {"date": "2026-07-30", "job": "6a6a7ead",
                 "note": "uv-scripts companion validation: 20/20 pages, 1,010 tok/s, 0 error rows"},
                {"date": "2026-07-29", "job": "1k-A/B",
                 "note": "1,000-page A/B: 0.955 img/s incl. streaming = offline recipe parity"},
            ],
        },
    ],
    "embeddings": [
        {
            "model": "BAAI/bge-m3",
            "engine": "tei",
            "why": "multilingual all-rounder; encoder -> TEI (+18% vs vLLM receipt)",
            "prompts": None,  # card explicitly: no instruction needed, either side
            "receipts": [
                {"date": "2026-07-30", "job": "6a6b1603b36a6516e96a243c",
                 "note": "zero-config fka/prompts.chat: 2,112 texts, 37.6k tok/s, 0 failed"},
                {"date": "2026-07-28", "job": "6a68c171a9f4",
                 "note": "25.6k-text TEI-vs-vLLM A/B: TEI 63.0k tok/s (+18%)"},
            ],
        },
        {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "engine": "vllm",
            "why": "decoder-style, strong retrieval; vLLM/TEI tie at GPU ceiling",
            "receipts": [
                {"date": "2026-07-30", "job": "6a6b161bb36a6516e96a2440",
                 "note": "10,048 imdb texts, 34.4k tok/s, 0 failed"},
                {"date": "2026-07-28", "job": "6a68b9fc",
                 "note": "400x64-text A/B: vLLM pooling 37.8k vs TEI 36.9k tok/s"},
            ],
        },
    ],
}


def entries(task: str) -> list[dict]:
    return CATALOGUE.get(task, [])


def lookup(task: str, model: str) -> dict | None:
    return next((e for e in entries(task) if e["model"] == model), None)


def default_entry(task: str) -> dict | None:
    """First entry is the task's hand-picked default."""
    es = entries(task)
    return es[0] if es else None

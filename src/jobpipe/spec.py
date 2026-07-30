"""TaskSpec — the wizard <-> compiler contract.

TaskSpec is the single typed contract between every front door (python API,
CLI, a UI posting spec.json) and the compiler. Field names follow saturate's
own vocabulary (``engine``, ``world`` as in ``pump(shard=(rank, world))``).

``TaskSpec.model_json_schema()`` is the cross-language contract for UIs.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ServingEngine = Literal["vllm", "tei"]

Flavor = Literal[
    "t4-small", "l4x1", "a10g-small", "a10g-large", "l40sx1", "a100-large"
]


class TaskSpec(BaseModel):
    task: Literal["embeddings", "generation", "translation", "ocr"] = "embeddings"
    dataset: str  # "HuggingFaceFW/fineweb-edu"
    config: str = "default"
    split: str = "train"
    column: str = "text"
    model: str  # "BAAI/bge-m3"
    engine: ServingEngine = "vllm"
    flavor: Flavor = "a10g-small"
    world: int = Field(default=1, ge=1)
    output: str  # fsspec URI: hf://buckets/… | hf://datasets/… | local path
    publish: bool = False  # compile-to-dataset post-run step
    # items per contract row (run.json items_per_row); default 64 embeddings, 1 generation
    batch: int | None = Field(default=None, ge=1)
    limit: int | None = None  # cap on items (texts/prompts), not batches
    num_examples: int | None = None  # dataset-viewer count; resolved at compile if None
    # generation/translation sampling knobs (ignored by the embeddings template)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    # translation-only (TranslateGemma-style models)
    source_lang: str = "en"
    target_lang: str | None = None

    @model_validator(mode="after")
    def _task_rules(self) -> TaskSpec:
        if self.task == "ocr":
            if not self.dataset.startswith("hf://buckets/"):
                raise ValueError(
                    "ocr reads a bucket glob of page images "
                    "(hf://buckets/owner/name/**/*.jpg) — got a dataset id; "
                    "dataset-image-column input is a planned follow-up"
                )
            if ".pdf" in self.dataset.lower():
                raise ValueError(
                    "PDFs need a rasterize stage first (hf-pipe issue #4) — "
                    "ocr v1 reads page IMAGES (*.jpg / *.png)"
                )
            if self.engine != "vllm":
                raise ValueError("ocr currently supports engine='vllm' only")
            if self.batch not in (None, 1):
                raise ValueError("ocr is one page per request (batch=1)")
            self.batch = 1
            return self
        if self.task in ("generation", "translation"):
            if self.batch not in (None, 1):
                raise ValueError(
                    f"{self.task} is one request per row (batch=1); "
                    "micro-batching is an embeddings-shaped concept"
                )
            if self.engine != "vllm":
                raise ValueError(f"{self.task} currently supports engine='vllm' only")
            self.batch = 1
        elif self.batch is None:
            self.batch = 64
        if self.task == "translation" and not self.target_lang:
            raise ValueError("translation requires target_lang (ISO 639-1, e.g. 'de')")
        return self

    @property
    def slug(self) -> str:
        if self.task == "ocr":  # dataset carries a bucket glob, not a repo id
            parts = self.dataset.removeprefix("hf://buckets/").split("/")
            name = parts[1] if len(parts) > 1 else (parts[0] or "bucket")
        else:
            name = (self.dataset.split("/")[1] if "/" in self.dataset else "ds") or "ds"
        name = re.sub(r"[^a-z0-9-]+", "-", name.lower())
        prefix = {"generation": "gen", "translation": "translate",
                  "ocr": "ocr"}.get(self.task, "embed")
        return f"{prefix}_{name}"[:40]

    @property
    def expected_items(self) -> int | None:
        """Items (texts) this run will admit: min(limit, split size), if either is known."""
        items = min(self.limit or math.inf, self.num_examples or math.inf)
        return int(items) if math.isfinite(items) else None

    @property
    def expected_rows(self) -> int | None:
        """Contract rows (= batches) — the progress denominator (gap G1)."""
        items = self.expected_items
        return math.ceil(items / self.batch) if items is not None else None

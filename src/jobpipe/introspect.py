"""Dataset/model introspection helpers.

The G1 denominator problem in miniature: streaming `datasets` never exposes a
row count, so the layer that compiles a run has to find it elsewhere. Chain:

1. dataset-viewer `/info` — num_examples for anything the viewer has processed
   (the read the console wizard proved out).
2. Parquet metadata footers — for viewer-less or private repos: list the
   repo's parquet files, read only their footers over hf://, sum num_rows.
   Counts without downloading data.

If this proves broadly useful it belongs upstream in `datasets`/`huggingface_hub`
— this module is the working demonstration.
"""

from __future__ import annotations

import httpx

DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Encoder model_types serve best on TEI (+18% on bge-m3, jobs 6a68c171a9f4 /
# 6a68c18115e8); decoder-style models are a vLLM/TEI tie — default vLLM.
# Set copied from the console wizard, which auto-suggested the arm from
# config.model_type.
ENCODER_TYPES = {
    "bert", "roberta", "xlm-roberta", "camembert", "distilbert", "mpnet",
    "modernbert", "electra", "deberta-v2", "nomic_bert", "new", "gte",
}


def num_examples(
    dataset: str, config: str = "default", split: str = "train", token: str | None = None
) -> int | None:
    """Row count for a Hub dataset split without materializing it."""
    n = _viewer_num_examples(dataset, config, split, token)
    if n is None:
        n = _parquet_footer_count(dataset, token)
    return n


def _viewer_num_examples(
    dataset: str, config: str, split: str, token: str | None
) -> int | None:
    headers = {"authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(
            f"{DATASETS_SERVER}/info",
            params={"dataset": dataset, "config": config},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["dataset_info"]["splits"][split]["num_examples"]
    except Exception:
        return None


def _parquet_footer_count(dataset: str, token: str | None) -> int | None:
    """Sum row counts from parquet footers — metadata reads only."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem

        api = HfApi(token=token)
        files = [
            f for f in api.list_repo_files(dataset, repo_type="dataset")
            if f.endswith(".parquet")
        ]
        if not files:
            return None
        fs = HfFileSystem(token=token)
        total = 0
        for f in files:
            with fs.open(f"datasets/{dataset}/{f}", "rb") as fh:
                total += pq.ParquetFile(fh).metadata.num_rows
        return total
    except Exception:
        return None


def dataset_layout(dataset: str, token: str | None = None) -> list[dict]:
    """All (config, split) pairs the viewer knows for a dataset.

    The zero-config entry point can't assume config="default"/split="train"
    exist — /splits is the read that makes `embed <any dataset>` honest.
    Returns [] when the viewer has nothing (private w/o token, errored, ...).
    """
    headers = {"authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(
            f"{DATASETS_SERVER}/splits",
            params={"dataset": dataset},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        return [
            {"config": s["config"], "split": s["split"]}
            for s in r.json().get("splits", [])
        ]
    except Exception:
        return []


def string_columns(
    dataset: str, config: str, split: str, token: str | None = None
) -> list[dict]:
    """String-typed columns with a mean-length sample, for text-column detection.

    Reads /first-rows (the console wizard's preview read). Returns
    [{name, avg_len}] sorted longest-first; [] when the viewer can't serve rows.
    """
    headers = {"authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(
            f"{DATASETS_SERVER}/first-rows",
            params={"dataset": dataset, "config": config, "split": split},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        doc = r.json()
        names = [
            f["name"]
            for f in doc.get("features", [])
            if f.get("type", {}).get("dtype") == "string"
            and f.get("type", {}).get("_type") == "Value"
        ]
        rows = [row["row"] for row in doc.get("rows", [])]
        out = []
        for name in names:
            vals = [row.get(name) or "" for row in rows]
            avg = sum(len(v) for v in vals) / len(vals) if vals else 0.0
            uniq = len(set(vals)) / len(vals) if vals else 0.0
            out.append({"name": name, "avg_len": round(avg, 1), "uniq": round(uniq, 2)})
        # near-constant columns (system prompts, boilerplate) lose to shorter
        # but actually-varying text; length only breaks ties within a band.
        return sorted(out, key=lambda c: (-(c["uniq"] >= 0.3), -c["avg_len"]))
    except Exception:
        return []


def model_metadata(model: str, token: str | None = None) -> dict:
    """The two reads the wizard bases suggestions on: param count + model_type."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(model)
    params = (info.safetensors.total if info.safetensors else None) or None
    model_type = (getattr(info, "config", None) or {}).get("model_type")
    return {"params": params, "model_type": model_type}


def suggest_engine(model_type: str | None) -> str:
    return "tei" if model_type in ENCODER_TYPES else "vllm"


def suggest_flavor(params: int | None) -> str:
    """Flavor from param count — the console wizard's rule of thumb."""
    if params is None:
        return "a10g-small"
    b = params / 1e9
    if b <= 1.5:
        return "a10g-small"
    if b <= 4:
        return "a10g-large"
    return "a100-large"

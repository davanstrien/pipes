"""resolve() — fill a partial intent into a launchable TaskSpec, with receipts.

The zero-config seam: verbs and CLI pass what the user actually said; this
module fills the rest and records WHERE each value came from —

    "user"      the caller passed it explicitly
    "curated"   from the catalogue: a tested, receipted model path
    "detected"  read from the dataset/model (datasets-server, model config)
    "default"   the layer's hand default (receipted in the templates)

The provenance map is printed before launch (and belongs in run.json), so a
zero-config run is never magic: every filled blank is attributable. Same
receipted-vs-derived distinction as the uv-scripts SERVING dicts.

Rule: model-specific knowledge is DATA (catalogue/config), never template code. resolve()
may look values up (model_type -> engine, params -> flavor) but templates
never hard-code per-model values.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

from jobpipe import catalogue, introspect
from jobpipe.spec import TaskSpec


@dataclass
class Resolution:
    spec: TaskSpec
    provenance: dict[str, str] = field(default_factory=dict)  # field -> user|detected|default
    notes: dict[str, str] = field(default_factory=dict)  # field -> one-line why
    # (local directory, bucket prefix, file pattern) to upload before launch
    uploads: list[tuple[str, str, str]] = field(default_factory=list)

    def describe(self) -> str:
        """Human block printed before launch (stderr)."""
        lines = []
        for f in ("dataset", "config", "split", "column", "id_column", "keep", "model",
                  "engine", "flavor", "max_seq_len", "world", "output", "limit"):
            if f not in self.provenance:
                continue
            val = getattr(self.spec, f)
            note = self.notes.get(f)
            lines.append(f"  {f:<11} {val!r:<44} [{self.provenance[f]}]"
                         + (f"  {note}" if note else ""))
        return "\n".join(lines)


def _repo_slug(dataset: str, task: str) -> str:
    if dataset.startswith("hf://buckets/"):
        parts = dataset.removeprefix("hf://buckets/").split("/")
        name = parts[1] if len(parts) > 1 else (parts[0] or "bucket")
        if name == "jobpipe-inputs" and len(parts) > 2:  # an uploaded local directory
            name = parts[2]
    else:
        name = dataset.split("/")[-1] or "ds"
    name = re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-") or "ds"
    suffix = {"embeddings": "embeddings", "generation": "generated", "ocr": "ocr"}.get(task, task)
    return f"{name}-{suffix}"[:96]


def _whoami(token: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi(token=token).whoami()["name"]


def resolve(
    task: str,
    dataset: str,
    *,
    token: str | None = None,
    **user: object,
) -> Resolution:
    """Build a launchable TaskSpec from (task, dataset) + whatever the user
    passed in `user` (only NON-None values count as user-passed)."""
    user = {k: v for k, v in user.items() if v is not None}
    prov: dict[str, str] = {"dataset": "user"}
    notes: dict[str, str] = {}
    vals: dict[str, object] = {"task": task, "dataset": dataset}

    if task == "ocr":
        return _resolve_bucket_task(task, dataset, user, prov, notes, vals, token)
    uploads: list[tuple[str, str, str]] = []
    if task == "embeddings" and pathlib.Path(dataset).expanduser().is_dir():
        dataset, upload = _local_dir_source(dataset, token)
        uploads.append(upload)
        vals["dataset"] = dataset
        notes["dataset"] = f"local directory, uploaded to {upload[1]} before launch"
    if task == "embeddings" and dataset.startswith("hf://buckets/"):
        return _resolve_embed_bucket(dataset, user, prov, notes, vals, token, uploads)

    # --- config / split: never assume default/train exist -------------------
    layout = None
    for f, prefer in (("config", "default"), ("split", "train")):
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"
            continue
        if layout is None:
            layout = introspect.dataset_layout(dataset, token=token)
        if layout:
            if f == "config":
                configs = list(dict.fromkeys(s["config"] for s in layout))
                vals[f] = prefer if prefer in configs else configs[0]
            else:
                splits = [s["split"] for s in layout if s["config"] == vals["config"]]
                vals[f] = prefer if prefer in splits else (splits[0] if splits else prefer)
            prov[f] = "detected"
        else:
            vals[f] = prefer
            prov[f] = "default"
            notes[f] = "viewer had no layout; assuming"

    # --- column: prefer 'text', else longest string column in the preview ---
    if "column" in user:
        vals["column"] = user["column"]
        prov["column"] = "user"
    else:
        cols = introspect.string_columns(
            dataset, str(vals["config"]), str(vals["split"]), token=token
        )
        names = [c["name"] for c in cols]
        if "text" in names:
            vals["column"] = "text"
            prov["column"] = "detected"
        elif cols:
            vals["column"] = cols[0]["name"]
            prov["column"] = "detected"
            others = ", ".join(names[1:4])
            notes["column"] = (
                f"longest varying string column in preview "
                f"(avg {cols[0]['avg_len']} chars, {int(cols[0].get('uniq', 1) * 100)}% distinct"
                + (f"; others: {others}" if others else "") + ")"
            )
        else:
            vals["column"] = "text"
            prov["column"] = "default"
            notes["column"] = "no preview available; assuming"
    if task == "embeddings" and "id_column" not in user:
        cols = introspect.string_columns(
            dataset, str(vals["config"]), str(vals["split"]), token=token
        )
        if "id" in [c["name"] for c in cols]:
            vals["id_column"] = "id"
            prov["id_column"] = "detected"
            notes["id_column"] = "string column named 'id'; output rows join back on it"

    # --- model: catalogue first (curated, receipted), else caller's ---------
    entry = None
    if "model" in user:
        vals["model"] = user["model"]
        prov["model"] = "user"
        entry = catalogue.lookup(task, str(user["model"]))
        if entry:
            notes["model"] = f"in catalogue: {entry['why']}"
        elif catalogue.entries(task):
            notes["model"] = "not in catalogue — unverified path, resolved from metadata"
    else:
        entry = catalogue.default_entry(task)
        if entry is None:
            raise ValueError(f"no curated model for task {task!r} — pass model=")
        vals["model"] = entry["model"]
        prov["model"] = "curated"
        notes["model"] = entry["why"]

    # --- engine / flavor: catalogue entry, else read the model --------------
    need_meta = (
        "flavor" not in user and (entry is None or "flavor" not in entry.get("serving", {}))
    ) and task == "embeddings"
    meta = introspect.model_metadata(str(vals["model"]), token=token) if need_meta else {}
    if "engine" in user:
        vals["engine"] = user["engine"]
        prov["engine"] = "user"
    elif entry is not None and "engine" in entry:
        vals["engine"] = entry["engine"]
        prov["engine"] = "curated"
        notes["engine"] = "from catalogue entry (receipted arrangement)"
    elif task == "embeddings":
        vals["engine"] = "vllm"
        prov["engine"] = "default"
        notes["engine"] = "vLLM pooling; unverified model path"
    if "flavor" in user:
        vals["flavor"] = user["flavor"]
        prov["flavor"] = "user"
    elif task == "embeddings" and entry is not None and "flavor" in entry.get("serving", {}):
        vals["flavor"] = entry["serving"]["flavor"]
        prov["flavor"] = "curated"
    elif task == "embeddings":
        vals["flavor"] = introspect.suggest_flavor(meta.get("params"))
        prov["flavor"] = "detected" if meta.get("params") else "default"
    if task == "embeddings":
        _apply_serving(entry, user, vals, prov)

    # --- output: auto-named private dataset repo under the caller -----------
    _resolve_output(task, dataset, user, prov, notes, vals, token)

    # --- pass-through knobs --------------------------------------------------
    for f in PASS_THROUGH:
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"

    return Resolution(spec=TaskSpec(**vals), provenance=prov, notes=notes, uploads=uploads)


PASS_THROUGH = ("world", "limit", "batch", "max_tokens", "temperature", "publish",
                "keep", "timeout")

# File extension -> `datasets` builder, for bucket and local-directory inputs.
FORMATS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv",
           ".txt": "text", ".md": "text"}


def _resolve_output(task, dataset, user, prov, notes, vals, token) -> None:
    if "output" in user:
        vals["output"] = user["output"]
        prov["output"] = "user"
        return
    owner = _whoami(token)
    if task == "embeddings":
        # buckets take concurrent writes from several ranks without commit contention
        vals["output"] = f"hf://buckets/{owner}/jobpipe-outputs/{_repo_slug(dataset, task)}"
        prov["output"] = "default"
        notes["output"] = "private bucket, created on first write"
        return
    vals["output"] = f"hf://datasets/{owner}/{_repo_slug(dataset, task)}/data"
    prov["output"] = "default"
    notes["output"] = "private dataset repo, created on first write"


def _apply_serving(entry, user, vals, prov) -> None:
    """Text handling for embeddings: user > catalogue serving entry > default."""
    serving = (entry or {}).get("serving", {})
    for f in ("id_column", "max_seq_len", "prefix", "serve_args"):
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"
        elif f in serving:
            vals[f] = serving[f]
            prov[f] = "curated"


def _local_dir_source(path: str, token: str | None) -> tuple[str, tuple[str, str, str]]:
    """A local directory becomes a bucket glob under the caller's jobpipe-inputs bucket.

    The most common supported file type in the directory decides the format; only
    files of that type are uploaded (at launch, not here)."""
    directory = pathlib.Path(path).expanduser().resolve()
    counts: dict[str, int] = {}
    for file in directory.rglob("*"):
        if file.is_file() and file.suffix.lower() in FORMATS:
            counts[file.suffix.lower()] = counts.get(file.suffix.lower(), 0) + 1
    if not counts:
        raise ValueError(f"{directory} has no {', '.join(sorted(FORMATS))} files")
    suffix = max(counts, key=counts.get)
    name = re.sub(r"[^A-Za-z0-9-]+", "-", directory.name).strip("-") or "input"
    prefix = f"hf://buckets/{_whoami(token)}/jobpipe-inputs/{name}"
    return f"{prefix}/**/*{suffix}", (str(directory), prefix, f"*{suffix}")


def _resolve_embed_bucket(dataset, user, prov, notes, vals, token, uploads) -> Resolution:
    """Embeddings from a bucket glob: the format comes from the glob's extension,
    and there is no viewer to read columns from."""
    suffix = pathlib.PurePosixPath(dataset).suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(
            f"bucket input needs a glob ending in one of {', '.join(sorted(FORMATS))}, "
            f"for example hf://buckets/me/b/data/*.parquet — got {dataset!r}"
        )
    vals["source_format"] = FORMATS[suffix]
    prov.setdefault("dataset", "user")
    notes.setdefault("dataset", f"bucket glob read as {FORMATS[suffix]}")
    if FORMATS[suffix] == "text":
        vals["column"] = "text"  # the datasets text builder names its column "text"
        prov["column"] = "detected"
        notes["column"] = "one row per file"
    elif "column" in user:
        vals["column"] = user["column"]
        prov["column"] = "user"
    else:
        vals["column"] = "text"
        prov["column"] = "default"
        notes["column"] = "no viewer for bucket files; pass --column if not 'text'"
    if "id_column" in user:
        vals["id_column"] = user["id_column"]
        prov["id_column"] = "user"

    entry = None
    if "model" in user:
        vals["model"] = user["model"]
        prov["model"] = "user"
        entry = catalogue.lookup("embeddings", str(user["model"]))
        if entry is None:
            notes["model"] = "not in catalogue — unverified path"
    else:
        entry = catalogue.default_entry("embeddings")
        vals["model"] = entry["model"]
        prov["model"] = "curated"
        notes["model"] = entry["why"]
    vals["engine"] = "vllm"
    prov["engine"] = "curated" if entry is not None else "default"
    serving = (entry or {}).get("serving", {})
    if "flavor" in user:
        vals["flavor"] = user["flavor"]
        prov["flavor"] = "user"
    else:
        vals["flavor"] = serving.get("flavor", "a10g-small")
        prov["flavor"] = "curated" if "flavor" in serving else "default"
    _apply_serving(entry, {k: v for k, v in user.items() if k != "id_column"}, vals, prov)
    _resolve_output("embeddings", dataset, user, prov, notes, vals, token)
    for f in PASS_THROUGH:
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"
    return Resolution(spec=TaskSpec(**vals), provenance=prov, notes=notes, uploads=uploads)


def _resolve_bucket_task(task, dataset, user, prov, notes, vals, token) -> Resolution:
    """Bucket-glob tasks: no config/split/column reads — the introspection
    surface is the glob itself (validated by the spec) + the catalogue."""
    notes["dataset"] = "bucket glob; ids = file paths, resume per page"
    entry = None
    if "model" in user:
        vals["model"] = user["model"]
        prov["model"] = "user"
        entry = catalogue.lookup(task, str(user["model"]))
        if entry:
            notes["model"] = f"in catalogue: {entry['why']}"
        elif catalogue.entries(task):
            notes["model"] = ("not in catalogue — unverified path: generic prompt, "
                             "conservative serving defaults")
    else:
        entry = catalogue.default_entry(task)
        if entry is None:
            raise ValueError(f"no curated model for task {task!r} — pass model=")
        vals["model"] = entry["model"]
        prov["model"] = "curated"
        notes["model"] = entry["why"]

    vals["engine"] = user.get("engine", "vllm")
    prov["engine"] = "user" if "engine" in user else "curated"
    if "flavor" in user:
        vals["flavor"] = user["flavor"]
        prov["flavor"] = "user"
    else:
        vals["flavor"] = "a10g-small"
        prov["flavor"] = "default"
        notes["flavor"] = "the port-matrix workhorse; override for bigger pages/models"

    if "output" in user:
        vals["output"] = user["output"]
        prov["output"] = "user"
    else:
        owner = _whoami(token)
        vals["output"] = f"hf://datasets/{owner}/{_repo_slug(dataset, task)}/data"
        prov["output"] = "default"
        notes["output"] = "private dataset repo, created on first write"

    for f in ("world", "limit", "max_tokens", "publish"):
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"
    return Resolution(spec=TaskSpec(**vals), provenance=prov, notes=notes)

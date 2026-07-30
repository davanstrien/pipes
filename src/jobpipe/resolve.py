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

import re
from dataclasses import dataclass, field

from jobpipe import catalogue, introspect
from jobpipe.spec import TaskSpec


@dataclass
class Resolution:
    spec: TaskSpec
    provenance: dict[str, str] = field(default_factory=dict)  # field -> user|detected|default
    notes: dict[str, str] = field(default_factory=dict)  # field -> one-line why

    def describe(self) -> str:
        """Human block printed before launch (stderr)."""
        lines = []
        for f in ("dataset", "config", "split", "column", "model", "engine",
                  "flavor", "world", "output", "limit"):
            if f not in self.provenance:
                continue
            val = getattr(self.spec, f)
            note = self.notes.get(f)
            lines.append(f"  {f:<8} {val!r:<44} [{self.provenance[f]}]"
                         + (f"  {note}" if note else ""))
        return "\n".join(lines)


def _repo_slug(dataset: str, task: str) -> str:
    if dataset.startswith("hf://buckets/"):
        parts = dataset.removeprefix("hf://buckets/").split("/")
        name = parts[1] if len(parts) > 1 else (parts[0] or "bucket")
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
        ("engine" not in user and entry is None) or "flavor" not in user
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
        vals["engine"] = introspect.suggest_engine(meta.get("model_type"))
        prov["engine"] = "detected"
        notes["engine"] = (
            f"model_type={meta.get('model_type')!r} "
            + ("(encoder -> TEI, +18% receipt)" if vals["engine"] == "tei"
               else "(decoder -> vLLM)")
        )
    if "flavor" in user:
        vals["flavor"] = user["flavor"]
        prov["flavor"] = "user"
    elif task == "embeddings":
        vals["flavor"] = introspect.suggest_flavor(meta.get("params"))
        prov["flavor"] = "detected" if meta.get("params") else "default"

    # --- output: auto-named private dataset repo under the caller -----------
    if "output" in user:
        vals["output"] = user["output"]
        prov["output"] = "user"
    else:
        owner = _whoami(token)
        vals["output"] = f"hf://datasets/{owner}/{_repo_slug(dataset, task)}/data"
        prov["output"] = "default"
        notes["output"] = "private dataset repo, created on first write"

    # --- pass-through knobs --------------------------------------------------
    for f in ("world", "limit", "batch", "max_tokens", "temperature", "publish"):
        if f in user:
            vals[f] = user[f]
            prov[f] = "user"

    return Resolution(spec=TaskSpec(**vals), provenance=prov, notes=notes)


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

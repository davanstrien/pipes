"""resolve: fill a partial request into a launch Plan, with receipts.

The zero-config seam. Verbs pass what the user said; this module fills the
rest and records where each value came from:

    "user"      the caller passed it
    "detected"  read from the dataset (datasets-server) or the input files
    "default"   this layer's hand default

The result is a Plan: the script to run (jobpipe's own copy of a uv-scripts
recipe) plus the exact arguments for each Job. Nothing here launches or
uploads; `launcher.launch(plan)` does that.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shlex
from dataclasses import asdict, dataclass, field

from jobpipe import introspect
from jobpipe.recipes import RECIPES, script_path

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_MAX_SEQ_LEN = 512

# OCR: the models jobpipe's OCR script serves. Another model family needs its own
# script (serving flags, prompt and page size differ per family).
OCR_MODELS = ["lightonai/LightOnOCR-3-0.8B", "lightonai/LightOnOCR-3-4B"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".jp2", ".bmp"}

# File extension -> `datasets` builder, for bucket and local-directory inputs.
# Must match BUCKET_FORMATS in the embeddings recipe.
FORMATS = {".parquet": "parquet", ".jsonl": "json", ".json": "json", ".csv": "csv",
           ".txt": "text", ".md": "text"}


@dataclass
class Plan:
    """What a launch will do. `args` is the recipe's argv without `--shard`."""

    task: str
    recipe_url: str  # a local path (jobpipe's copy) or a URL
    # set when recipe_url is jobpipe's own copy: `hf pipe script <name>` writes it out
    args: list[str]
    world: int
    output: str  # storage URI holding the parts, __saturate/ and run.json
    script_name: str | None = None
    flavor: str | None = None  # None: the recipe header's flavor
    timeout: str | None = None  # None: the recipe header's timeout
    expected_items: int | None = None
    fingerprint: dict = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    values: dict = field(default_factory=dict)  # the resolved fields, for describe()
    # (local directory, bucket prefix, file pattern) to upload before launch
    uploads: list[tuple[str, str, str]] = field(default_factory=list)

    def rank_args(self, rank: int) -> list[str]:
        return [*self.args, "--shard", f"{rank}/{self.world}"]

    def commands(self) -> list[str]:
        """The equivalent `hf jobs uv run` commands, one per rank. With jobpipe's own
        copy of the script they name `<name>.py`, written out by `script_command()`."""
        script = f"{self.script_name}.py" if self.script_name else self.recipe_url
        options = []
        if self.flavor:
            options.append(f"--flavor {self.flavor}")
        if self.timeout:
            options.append(f"--timeout {self.timeout}")
        options.append("-s HF_TOKEN -e HF_HUB_DISABLE_PROGRESS_BARS=1")
        return [
            f"hf jobs uv run {' '.join(options)} \\\n  {script} \\\n  "
            + " ".join(shlex.quote(arg) for arg in self.rank_args(rank))
            for rank in range(self.world)
        ]

    def script_command(self) -> str | None:
        """The command that writes jobpipe's copy of the script next to the commands."""
        if not self.script_name:
            return None
        return f"hf pipe script {self.script_name} > {self.script_name}.py"

    def describe(self) -> str:
        """The provenance block printed before launch (stderr)."""
        lines = []
        for name, value in self.values.items():
            if name not in self.provenance:
                continue
            note = self.notes.get(name)
            lines.append(f"  {name:<11} {value!r:<46} [{self.provenance[name]}]"
                         + (f"  {note}" if note else ""))
        return "\n".join(lines)

    def to_json(self) -> dict:
        return asdict(self)


def _whoami(token: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi(token=token).whoami()["name"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", text).strip("-") or "x"


def _source_name(source: str) -> str:
    if source.startswith("hf://buckets/"):
        parts = source.removeprefix("hf://buckets/").split("/")
        if len(parts) > 2 and parts[1] == "jobpipe-inputs":  # an uploaded local directory
            return parts[2]
        return parts[1] if len(parts) > 1 else parts[0]
    return source.split("/")[-1]


def _local_dir_source(path: str, token: str | None,
                      suffixes=FORMATS) -> tuple[str, tuple[str, str, str]]:
    """A local directory becomes a bucket glob under the caller's jobpipe-inputs bucket.

    The most common supported file type decides the format; only files of that
    type are uploaded (by the launcher, not here)."""
    directory = pathlib.Path(path).expanduser().resolve()
    counts: dict[str, int] = {}
    for file in directory.rglob("*"):
        suffix = file.suffix.lower()
        if file.is_file() and suffix in suffixes:
            counts[suffix] = counts.get(suffix, 0) + 1
    if not counts:
        raise ValueError(f"{directory} has no {', '.join(sorted(suffixes))} files")
    suffix = max(counts, key=counts.get)
    # One prefix per absolute path: two folders called "docs" never share a snapshot.
    place = hashlib.sha256(str(directory).encode()).hexdigest()[:8]
    prefix = f"hf://buckets/{_whoami(token)}/jobpipe-inputs/{_slug(directory.name)}-{place}"
    return f"{prefix}/**/*{suffix}", (str(directory), prefix, f"*{suffix}")


def local_digest(directory: str, pattern: str) -> str:
    """Names, sizes and modification times of the files a run uploads.

    Path ids mark a file done; if a file changes, its id does not. Putting this
    digest in the fingerprint makes a changed folder a different run."""
    root = pathlib.Path(directory)
    entries = sorted(
        (str(file.relative_to(root)), file.stat().st_size, int(file.stat().st_mtime))
        for file in root.rglob(pattern) if file.is_file()
    )
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()[:16]


def _normalise_output(output: str) -> tuple[str, str]:
    """User output -> (recipe positional argument, storage URI of the parts).

    Accepts hf://buckets/ns/b/prefix, hf://datasets/ns/name or ns/name."""
    if output.startswith("hf://buckets/"):
        if len(output.removeprefix("hf://buckets/").split("/")) < 2:
            raise ValueError(f"a bucket output needs ns/bucket at least: {output!r}")
        return output, output.rstrip("/")
    repo = output.removeprefix("hf://datasets/").strip("/")
    if len(repo.split("/")) != 2:
        raise ValueError(f"a dataset output is ns/name (the parts go in its data/): {output!r}")
    return repo, f"hf://datasets/{repo}/data"


EMBED_KEYS = ("source", "config", "split", "column", "id_column", "model", "max_seq_len",
              "prefix", "keep", "text_out", "local_digest")
OCR_KEYS = ("source", "config", "split", "image_column", "id_column", "model", "mode",
            "local_digest")


def fingerprint_of(values: dict, task: str = "embed", keys: tuple = EMBED_KEYS) -> dict:
    """What must match for a re-run to resume into an existing output.

    saturate's row ids do not include the model or the text handling, so a
    re-run with another model would skip the rows as done. The output's columns
    (kept columns, text in or out) are part of it too: parts with two shapes in one
    output do not load as one table. --limit and --world are not: a larger limit
    adds rows, and a smaller one leaves the rows already written in place."""
    data = {key: values.get(key) for key in keys}
    data["recipe"] = RECIPES[task].script
    data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return data


class _Fill:
    """The resolved values, where each came from, and a one-line why."""

    def __init__(self):
        self.values: dict = {}
        self.prov: dict[str, str] = {}
        self.notes: dict[str, str] = {}
        self.uploads: list[tuple[str, str, str]] = []

    def put(self, name, value, how, note=None):
        self.values[name] = value
        self.prov[name] = how
        if note:
            self.notes[name] = note


def _check_numbers(**numbers) -> None:
    for name, number in numbers.items():
        if number is not None and number < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1, got {number}")


def _take_source(fill: _Fill, source: str, token, suffixes) -> str:
    """A local directory becomes a bucket glob (uploaded at launch); anything else as given."""
    if pathlib.Path(source).expanduser().is_dir():
        source, upload = _local_dir_source(source, token, suffixes)
        fill.uploads.append(upload)
        fill.put("source", source, "user", f"local directory, uploaded to {upload[1]} at launch")
        fill.values["local_digest"] = local_digest(upload[0], upload[2])
    else:
        fill.put("source", source, "user")
    return source


def _require_dataset(source: str, token) -> None:
    if not introspect.dataset_exists(source, token=token):
        raise ValueError(
            f"dataset {source!r} not found on the Hub"
            + ("" if token else " (or it is private: sign in or pass a token)")
        )


def _output_and_scale(fill: _Fill, source, output, world, limit, flavor, timeout, owner,
                      token) -> tuple[str, str]:
    """Output (default: a private bucket named after source and model), Jobs, limits."""
    if output:
        fill.put("output", output, "user")
    else:
        name = f"{_slug(_source_name(source))}-{_slug(fill.values['model'].split('/')[-1])}"
        fill.put("output", f"hf://buckets/{owner or _whoami(token)}/jobpipe-outputs/{name}",
                 "default", "private bucket; the name includes the model")
    output_arg, output_uri = _normalise_output(fill.values["output"])
    fill.put("world", world or 1, "user" if world else "default")
    if limit is not None:
        fill.put("limit", limit, "user")
    if flavor:
        fill.put("flavor", flavor, "user")
    if timeout:
        fill.put("timeout", timeout, "user")
    return output_arg, output_uri


def _script(fill: _Fill, task: str, recipe_url: str | None, upstream: str) -> str:
    if recipe_url:
        fill.put("recipe", recipe_url, "user")
        return recipe_url
    recipe = RECIPES[task]
    fill.put("recipe", f"scripts/{recipe.script}", "default",
             f"jobpipe's copy of uv-scripts {upstream}")
    return script_path(recipe)


def _expected(source, from_bucket, fill: _Fill, limit, token) -> int | None:
    total = None
    if not from_bucket:
        total = introspect.num_examples(source, fill.values["config"], fill.values["split"],
                                        token=token)
    known = [count for count in (total, limit) if count is not None]
    return min(known) if known else None


def _hub_split(fill: _Fill, source, config, split, token) -> None:
    """config and split of a Hub dataset: user, else the viewer (prefer default/train)."""
    layout = None
    for name, given, prefer in (("config", config, "default"), ("split", split, "train")):
        if given:
            fill.put(name, given, "user")
            continue
        if layout is None:
            layout = introspect.dataset_layout(source, token=token)
        if layout:
            if name == "config":
                options = list(dict.fromkeys(s["config"] for s in layout))
            else:
                options = [s["split"] for s in layout if s["config"] == fill.values["config"]]
            fill.put(name, prefer if prefer in options else (options[0] if options else prefer),
                     "detected")
        else:
            fill.put(name, prefer, "default", "viewer had no layout; assuming")


def resolve_embed(
    source: str,
    *,
    model: str | None = None,
    column: str | None = None,
    id_column: str | None = None,
    keep: list[str] | None = None,
    max_seq_len: int | None = None,
    prefix: str | None = None,
    config: str | None = None,
    split: str | None = None,
    output: str | None = None,
    world: int | None = None,
    limit: int | None = None,
    flavor: str | None = None,
    timeout: str | None = None,
    recipe_url: str | None = None,
    keep_text: bool = False,
    owner: str | None = None,
    token: str | None = None,
) -> Plan:
    """Plan an embeddings run over a Hub dataset, a bucket glob or a local directory.

    `owner` names the default output's namespace (default: the token's user)."""
    _check_numbers(limit=limit, world=world, max_seq_len=max_seq_len)
    fill = _Fill()
    put, values = fill.put, fill.values

    source = _take_source(fill, source, token, FORMATS)
    from_bucket = source.startswith("hf://buckets/")

    if from_bucket:
        suffix = pathlib.PurePosixPath(source).suffix.lower()
        if suffix not in FORMATS:
            raise ValueError(
                f"a bucket input is a glob ending in one of {', '.join(sorted(FORMATS))}, "
                f"for example hf://buckets/me/b/data/*.parquet; got {source!r}"
            )
        if FORMATS[suffix] == "text":
            put("column", "text", "detected", "one row per file")
            put("id_column", "(file path)", "detected", "relative to the glob's fixed prefix")
        elif column:
            put("column", column, "user")
        else:
            put("column", "text", "default",
                "no viewer for bucket files; pass --column if not 'text'")
        if id_column:
            put("id_column", id_column, "user")
    else:
        _require_dataset(source, token)
        _hub_split(fill, source, config, split, token)
        _text_columns(fill, source, column, id_column, token)

    # --- model and text handling ---------------------------------------------------
    if model:
        put("model", model, "user")
    else:
        put("model", DEFAULT_MODEL, "default", "multilingual (100+ languages), 1024-d")
    if max_seq_len:
        put("max_seq_len", max_seq_len, "user")
    else:
        put("max_seq_len", DEFAULT_MAX_SEQ_LEN, "default",
            "predictable cost; the recipe alone would use the model maximum")
    if prefix is not None:
        put("prefix", prefix, "user")
    if keep is not None:
        put("keep", list(keep), "user")
    elif not from_bucket:
        skip = {values["column"], values.get("id_column")}
        small = [name for name in introspect.small_columns(
            source, values["config"], values["split"], token=token) if name not in skip]
        if small:
            put("keep", small, "detected",
                "every scalar column and list of scalars; --keep none to skip, or name them")

    output_arg, output_uri = _output_and_scale(fill, source, output, world, limit, flavor,
                                               timeout, owner, token)
    location = _script(fill, "embed", recipe_url, "generate-embeddings-saturate.py")
    expected = _expected(source, from_bucket, fill, limit, token)

    # --- the recipe's arguments ------------------------------------------------------
    args = [source, output_arg, "--model", values["model"], "--column", values["column"],
            "--max-seq-len", str(values["max_seq_len"]), "--skip-empty", "--private"]
    if not from_bucket:
        args += ["--config", values["config"], "--split", values["split"]]
    text_files = from_bucket and FORMATS[pathlib.PurePosixPath(source).suffix.lower()] == "text"
    if values.get("id_column") and not text_files:
        args += ["--id-column", values["id_column"]]
    # The text is only worth storing when the id alone cannot find the row again.
    stable_ids = bool(values.get("id_column"))
    if keep_text:
        put("text_out", True, "user")
    elif stable_ids:
        args.append("--no-text")
        put("text_out", False, "default", "rows join back on the id; --keep-text to store it")
    else:
        put("text_out", True, "default", "position ids: the text is kept so rows stay identifiable")
    if "prefix" in values:
        args += ["--prompt", values["prefix"]]
    if values.get("keep"):
        args += ["--keep", *values["keep"]]
    if limit is not None:
        args += ["--max-samples", str(limit)]

    return Plan(
        task="embed",
        recipe_url=location,
        script_name=None if recipe_url else "embed",
        args=args,
        world=values["world"],
        output=output_uri,
        flavor=flavor,
        timeout=timeout,
        expected_items=expected,
        fingerprint=fingerprint_of(values, "embed", EMBED_KEYS),
        provenance=fill.prov,
        notes=fill.notes,
        values=values,
        uploads=fill.uploads,
    )


def _text_columns(fill: _Fill, source, column, id_column, token) -> None:
    """Text column and id column of a Hub dataset, from the viewer's preview."""
    put = fill.put
    columns = introspect.string_columns(source, fill.values["config"], fill.values["split"],
                                        token=token)
    names = [c["name"] for c in columns]
    if column:
        put("column", column, "user")
    elif "text" in names:
        put("column", "text", "detected")
    elif columns:
        others = ", ".join(names[1:4])
        put("column", columns[0]["name"], "detected",
            f"longest varying string column in preview (avg {columns[0]['avg_len']} chars"
            + (f"; others: {others}" if others else "") + ")")
    else:
        put("column", "text", "default", "no preview available; assuming")

    if id_column:
        put("id_column", id_column, "user")
    elif "id" in names:
        put("id_column", "id", "detected", "string column named 'id'; output rows join back on it")


def resolve_ocr(
    source: str,
    *,
    model: str | None = None,
    mode: str | None = None,
    image_column: str | None = None,
    id_column: str | None = None,
    config: str | None = None,
    split: str | None = None,
    output: str | None = None,
    world: int | None = None,
    limit: int | None = None,
    flavor: str | None = None,
    timeout: str | None = None,
    recipe_url: str | None = None,
    owner: str | None = None,
    token: str | None = None,
) -> Plan:
    """Plan an OCR run over a bucket glob of page images, a local folder of them, or a Hub
    dataset with an image column. Output rows: id, markdown, raw, finish_reason..."""
    _check_numbers(limit=limit, world=world)
    fill = _Fill()
    put, values = fill.put, fill.values

    source = _take_source(fill, source, token, IMAGE_SUFFIXES)
    from_bucket = source.startswith("hf://buckets/")
    if from_bucket:
        suffix = pathlib.PurePosixPath(source).suffix.lower()
        if suffix == ".pdf":
            raise ValueError("PDFs are not supported yet: OCR reads page images")
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(
                f"a bucket input is a glob of page images ending in one of "
                f"{', '.join(sorted(IMAGE_SUFFIXES))}, for example "
                f"hf://buckets/me/scans/**/*.jpg; got {source!r}"
            )
        put("id_column", "(file path)", "detected", "relative to the glob's fixed prefix")
        if id_column:
            raise ValueError("--id-column applies to dataset input; image files use their path")
    else:
        _require_dataset(source, token)
        _hub_split(fill, source, config, split, token)
        found = introspect.image_columns(source, values["config"], values["split"], token=token)
        if image_column:
            put("image_column", image_column, "user")
        elif found:
            put("image_column", found[0], "detected",
                "first image column in the preview" + (f"; others: {', '.join(found[1:3])}"
                                                       if len(found) > 1 else ""))
        else:
            raise ValueError(f"no image column found in {source!r}; pass --image-column")
        names = [c["name"] for c in introspect.string_columns(
            source, values["config"], values["split"], token=token)]
        if id_column:
            put("id_column", id_column, "user")
        elif "id" in names:
            put("id_column", "id", "detected", "string column named 'id'")

    if model:
        if model not in OCR_MODELS:
            raise ValueError(
                f"no OCR script for {model!r} yet; jobpipe's script serves "
                f"{', '.join(OCR_MODELS)}"
            )
        put("model", model, "user")
    else:
        put("model", OCR_MODELS[0], "default", "LightOnOCR-3 0.8B; 14,483 IDL pages, 10 Nov")
    put("mode", mode or "plain", "user" if mode else "default",
        None if mode else "markdown text; 'grounding' adds layout blocks")

    output_arg, output_uri = _output_and_scale(fill, source, output, world, limit, flavor,
                                               timeout, owner, token)
    location = _script(fill, "ocr", recipe_url, "ocr/lighton-ocr3-saturate.py")
    expected = _expected(source, from_bucket, fill, limit, token)

    args = [source, output_arg, "--model", values["model"], "--mode", values["mode"]]
    if not from_bucket:
        args += ["--image-column", values["image_column"],
                 "--config", values["config"], "--split", values["split"]]
        if values.get("id_column"):
            args += ["--id-column", values["id_column"]]
    if limit is not None:
        args += ["--max-samples", str(limit)]

    return Plan(
        task="ocr",
        recipe_url=location,
        script_name=None if recipe_url else "ocr",
        args=args,
        world=values["world"],
        output=output_uri,
        flavor=flavor,
        timeout=timeout,
        expected_items=expected,
        fingerprint=fingerprint_of(values, "ocr", OCR_KEYS),
        provenance=fill.prov,
        notes=fill.notes,
        values=values,
        uploads=fill.uploads,
    )

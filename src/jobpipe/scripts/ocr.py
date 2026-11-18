# Copy of the uv-scripts recipe ocr/lighton-ocr3-saturate.py
# (github.com/davanstrien/uv-scripts, main @ f2792c8).
# jobpipe launches this copy. Refresh it on purpose: see src/jobpipe/scripts/README.md.
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.1",
#     "pillow>=10",
#     # LightOn's official client, pinned to a commit (not on PyPI). Used for its page
#     # resize, grounding parser and dollar-delimiter repair so outputs match upstream.
#     "lightonocr @ https://github.com/lightonai/LightOnOCR/archive/36755d461be079737860a5f03ae0c803501269e9.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
Convert document images to markdown (or layout blocks) using LightOnOCR-3 via saturate.

Same model, prompt modes, sampling and page preparation as `lighton-ocr3.py`, but the
script starts `vllm serve` in the Job and the `saturate` library sends the pages to it:
many pages in flight, resumable parquet output and durable error rows. Use this recipe
for large runs.

Run on HF Jobs. The script starts the server itself; its [tool.hf-jobs] header declares
the image that provides `vllm`, the hardware and the HF_TOKEN secret (`hf` CLI 1.32+).
Flags on the command line override the header:

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-saturate.py \\
      <input-dataset> <output-dataset>

Smoke test first with --max-samples 10. Layout blocks with labels and boxes:

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-saturate.py \\
      <input-dataset> <output-dataset> --mode grounding

Output: the output repo holds `data/part-*.parquet` with rows
`{id, markdown, raw, finish_reason, layout_blocks, model, prompt_tokens, completion_tokens,
error}` keyed by the input row id (`--id-column`, or `<repo>:<config>:<split>:<000000000>`
by default). Read it with `saturate.read_output`, which keeps one row per id (a success
over an earlier error). `datasets.load_dataset(<output>, data_dir="data")` returns every
record, so after a --retry-errors run it also holds the old error rows: drop rows with an
`error` before you join back to the input on id. Re-running the same command skips
finished rows; --retry-errors re-admits failed ones.

`finish_reason` = "length" marks a page that hit --max-tokens (long tables, or the model
repeating a line). --repair-max-tokens N sends those pages once more, at the end of the
run, with a cap of N tokens and temperature 0.1, and writes the results to `repair/`
(`load_dataset(<output>, data_dir="repair")`); join on id and prefer the repair row where
its finish_reason is "stop". A restarted run also repairs the pages that an earlier,
interrupted run saved as truncated.

Models (Apache-2.0):
  lightonai/LightOnOCR-3-0.8B  (default)  85.5 olmOCR-Bench, 74.6 ParseBench
  lightonai/LightOnOCR-3-4B               86.3 olmOCR-Bench, 75.1 ParseBench
LightOnOCR-3-1B is not supported: it keeps the LightOnOCR-2 (Pixtral) architecture, and
LightOn reports that transformers 5.17 (the version in the vLLM 0.30.0 image) breaks it.

Upstream settings kept:
- Serve flags: the LightOnOCR README's flags for the 0.8B and 4B (`--max-model-len 18000`,
  thinking off, `--mm-processor-cache-gb 0`, `--no-enable-prefix-caching`,
  `--limit-mm-per-prompt '{"image": 1}'`). vLLM 0.30.0, the version LightOn tests.
- Prompt modes: plain sends the image alone; grounding adds the text "grounding". The card
  says other instructions are out of distribution.
- Sampling: the model card's vLLM example (temperature 0.2, top_p 0.9, max_tokens 4096).
- Pages: downscaled to 2048 px on the longest edge with LightOn's own resize (never
  upscaled), sent as PNG, as the official client does. The PNG uses a fast compression
  level; PNG is lossless, so the server decodes the same pixels.
- Escaped dollar delimiters are repaired (`\\$x\\$` -> `$x$`) as the client does; `raw`
  keeps the model's text.

Speed settings: pages are prepared in `--prepare-workers` threads (default: the CPU_CORES
that HF Jobs sets), and saturate's adaptive window sets how many pages are in flight, up to
a cap that fits the Job's memory (window_cap below). `--window N` pins it instead. The
server also gets `--max-num-batched-tokens 16384`, and 4 API server processes on flavors
with 12 or more CPU cores (see SERVING). Pass --serve-args "--api-server-count N" to choose.

Measured speed, LightOnOCR-3-0.8B (2026-11-05, vLLM 0.30.0). 600 pages: 350 book pages from
finebooks/bhl-impact-gt and 250 pages of typed industry documents, all fitted to 2048 px.
Steady state, excluding the server start (about 4 min, also paid by lighton-ocr3.py):

  flavor          pages/s   $ per 1,000 pages   lighton-ocr3.py (offline batches)
  a10g-small        2.2         0.12                0.90 pages/s, $0.31
  l4x1              1.6         0.14
  a10g-large        2.7         0.15
  a100-large        7.2         0.10                1.27 pages/s, $0.55
  rtx-pro-6000     10.0         0.08

The 4B on a100-large (one API server): 2.7 pages/s, $0.26 per 1,000 pages; on these book
pages its character error rate was higher than the 0.8B's (0.10 mean, 0.023 median, one
run), so check a sample before paying for it. The header's a10g-small is a good default
for small and medium runs; for large runs add --flavor rtx-pro-6000 (or a100-large). In
every run the outputs differed from lighton-ocr3.py's no more than two runs of this recipe
differ from each other (sampling at temperature 0.2). The character error rate on the 282
book pages with at least 20 characters of ground truth was 0.077 to 0.083 mean and 0.020
median in every 0.8B run.

Truncation and repair: 9 to 12 of the 600 pages stopped at max_tokens 4096 in each run,
most of them typed tables. With --repair-max-tokens 12000, 8 of 12 such pages finished:
3 in fewer than 4,096 tokens on the second try, and 5 long pages that needed 4,900 to
6,100 tokens. The other 4 hit the cap again. The pass added 22% to the run's output
tokens. With --repair-max-tokens 8192 (suggested), 5 of 9 such pages finished, the same 4
hit the cap again, and the pass added 16%. Pages that hit the cap on both tries keep
their first output in data/.

BUCKETS: the input can be a glob of page images in a Storage Bucket instead of a dataset id,
for example hf://buckets/you/scans/books/**/*.jpg (also .jpeg, .png, .tif, .tiff, .webp, .jp2,
.bmp; PDFs are not accepted). Each file is one page, and its id is its path below the glob's
fixed prefix (books/ in the example). --id-column, --config, --split and --image-column apply
to dataset input only. The output can be a bucket path, for example
hf://buckets/you/ocr/run1: the parts land there (repairs in run1/repair) and no card is written.

FAN-OUT: --shard 2/4 makes this Job process the pages whose id hashes to shard 2 of 4. Start
4 Jobs with --shard 0/4 ... 3/4 and the same output; together they cover the input once.
Each Job lists (or streams) the whole input and fetches or decodes only its own pages. Use a
bucket output for fan-out: each flush to a dataset repo is a commit.

The SERVING dict below is the per-model tuning prior. Agents can `ast.literal_eval` it
without running the script; the script itself consumes it.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import sys
import tempfile

# Serving starting values for LightOnOCR-3 (0.8B and 4B). Per-value provenance:
# - image: vLLM 0.30.0, the version LightOn's client pins and tests (0.27.1 makes the
#   models loop, per the LightOnOCR README).
# - context 18000 and serve_args: the LightOnOCR README "Models" table for the 0.8B and 4B,
#   plus its `--limit-mm-per-prompt '{"image": 1}'`.
# - max_tokens/temperature/top_p: the model card's vLLM example, verbatim.
# - longest_edge 2048: the card's rendering tip and the client's page size for these models.
# - patch 32: 16 px patches with a 2x2 merge, so a 2048x2048 page is at most 64x64 = 4,096
#   image tokens (used for the context check).
# - --max-num-batched-tokens 16384: speed only. vLLM 0.30 serves with 2048 on A10G, L4 and
#   A100 (8192 on other GPUs of 70 GB or more), less than one page's image tokens. Measured on a100-large:
#   6.20 and 6.30 pages/s against 5.83 without it, outputs within the run-to-run noise.
# - pages_per_gib: see window_cap().
# - api_servers 4 when CPU_CORES >= 12: speed only. With one vLLM API server, a100-large and
#   rtx-pro-6000 had 128-256 pages in flight but only 36-60 running in the engine and none
#   waiting: the rest queued for image preprocessing in the API process. Measured, 4 API
#   servers against one: a100-large 7.29 and 7.18 against 6.20 and 6.30 pages/s,
#   rtx-pro-6000 10.0 against 9.15, a10g-large 2.74 against 2.49; outputs within the
#   run-to-run noise. Small flavors (a10g-small 3 cores, l4x1 8) keep one, to leave CPU
#   for page preparation.
SERVING = {
    "models": ["lightonai/LightOnOCR-3-0.8B", "lightonai/LightOnOCR-3-4B"],
    "image": "vllm/vllm-openai:v0.30.0",
    "context": 18000,
    "serve_args": [
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens", "16384",
    ],
    "pages_per_gib": 6,
    "api_servers": 4,
    "api_servers_min_cores": 12,
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "repair_temperature": 0.1,
    "longest_edge": 2048,
    "patch": 32,
}


def window_cap() -> int:
    """Most pages in flight that this machine's memory allows.

    Every page in flight holds about 0.1 GiB of image data on the host (the client's copy and
    the server's preprocessed image), on top of 7 to 9 GiB for the engine itself. On
    a10g-small (14.9 GiB limit), 64 pages in flight reached the limit and the Job stopped
    making progress; 32 peaked at 12.0 GiB, and up to 39 peaked at 13.0 and 14.8 GiB on two
    runs. So the window may grow by SERVING "pages_per_gib" pages per GiB above 10 GiB: 29 on
    a10g-small (which is GPU-bound from about 20 pages in flight), about 100 on l4x1, the full
    512 on a100-large. The limit is read from the container (cgroup v2), then from the MEMORY
    variable that HF Jobs sets.
    """
    limit_gib = None
    try:
        with open("/sys/fs/cgroup/memory.max") as handle:
            value = handle.read().strip()
        if value != "max":
            limit_gib = int(value) / 2**30
    except (OSError, ValueError):
        pass
    if limit_gib is None and os.environ.get("MEMORY"):
        try:  # e.g. "16.0G" (decimal gigabytes)
            limit_gib = float(os.environ["MEMORY"].upper().rstrip("GIB ")) * 1e9 / 2**30
        except ValueError:
            pass
    if limit_gib is None:
        return 512
    return max(8, min(512, int((limit_gib - 10) * SERVING["pages_per_gib"])))


def flag_value(tokens: list[str], name: str):
    """Last value given for `name` in a `vllm serve` argument list (`--x v` or `--x=v`), else None."""
    value = None
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def saved_truncations(data_uri: str, repair_uri: str) -> set:
    """Ids whose saved result in data_uri stopped at max_tokens and that have no saved repair.

    A restarted run skips rows already in data/, so without this the pages truncated
    before the restart would never reach the repair pass. Reads only three columns.
    """
    import fsspec
    import pyarrow.parquet as pq

    def records(uri, columns):
        fs, root = fsspec.core.url_to_fs(uri)
        if not fs.exists(root):
            return
        for path in fs.glob(f"{root}/part-*.parquet"):
            with fs.open(path, "rb") as handle:
                yield from pq.read_table(handle, columns=columns).to_pylist()

    finish = {}
    for record in records(data_uri, ["id", "error", "finish_reason"]):
        if record["error"] is None:
            finish[record["id"]] = record["finish_reason"]
    repaired = {r["id"] for r in records(repair_uri, ["id", "error"]) if r["error"] is None}
    return {row_id for row_id, reason in finish.items() if reason == "length" and row_id not in repaired}


def max_image_tokens(longest_edge: int) -> int:
    """Upper bound on the image tokens of a page fitted to longest_edge (square page)."""
    side = -(-longest_edge // SERVING["patch"])
    return side * side


# Page image extensions accepted in a bucket glob (compared in lower case).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".jp2", ".bmp")


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


def check_image_glob(pattern):
    """Exit with a message unless `pattern` ends in a page image extension (no network call)."""
    suffix = os.path.splitext(pattern)[1].lower()
    if suffix == ".pdf":
        sys.exit(f"PDF input is not supported: {pattern!r}. Give a glob of page images "
                 f"({', '.join(IMAGE_EXTENSIONS)}), or a dataset with an image column.")
    if suffix not in IMAGE_EXTENSIONS:
        sys.exit(f"A bucket input is a glob of page images ending in one of "
                 f"{', '.join(IMAGE_EXTENSIONS)}, for example "
                 f"hf://buckets/you/scans/books/**/*.jpg; got {pattern!r}")


def image_file_rows(pattern, image_column, limit, shard):
    """One row per page image in a bucket glob: (id, {image_column: the file's bytes}).

    The id is the file's path below the glob's fixed prefix, so widening the glob later
    keeps every id. The shard filter runs on the path before any file is read."""
    from saturate import bucket_rows

    root = glob_root(pattern)
    for row_id, row in bucket_rows(pattern, root=root, limit=limit, shard=shard):
        yield row_id, {image_column: row["bytes"]}


def to_pil(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise ValueError(f"unsupported image value: {type(value)}")


def page_png(value, longest_edge: int) -> bytes:
    """RGB page fitted to longest_edge with LightOn's resize, as PNG bytes.

    compress_level=1 instead of Pillow's default 6: PNG is lossless at every level, so the
    pixels are identical, and level 1 encodes several times faster (CPU is the limit on
    small flavors).
    """
    from lightonocr import fit

    image = fit(to_pil(value).convert("RGB"), longest_edge)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def chat_request(model: str, png: bytes, mode: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """One page as the client sends it: the image, plus the word "grounding" in grounding mode."""
    from lightonocr import GROUNDING_PROMPT

    url = "data:image/png;base64," + base64.b64encode(png).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}]
    if mode == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def postprocess(raw: str, mode: str, fix_dollars: bool):
    """Return (markdown, layout_blocks JSON or None) for one model output, as lighton-ocr3.py."""
    from lightonocr import fix_escaped_dollars, parse_blocks, to_markdown

    text = fix_escaped_dollars(raw) if fix_dollars else raw
    if mode != "grounding":
        return text, None
    blocks = parse_blocks(text)
    if not blocks:
        return text, json.dumps([])  # no markers parsed: keep the text so nothing is lost
    return to_markdown(blocks), json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="LightOnOCR-3 batch OCR via saturate")
    ap.add_argument("input_dataset",
                    help="Input dataset repo id (rows with an image column), or a bucket glob of "
                         "page images such as hf://buckets/you/scans/books/**/*.jpg")
    ap.add_argument("output_dataset",
                    help="Output dataset repo id (created private if missing), or a bucket path "
                         "such as hf://buckets/you/ocr/run1")
    ap.add_argument("--model", default=SERVING["models"][0], choices=SERVING["models"])
    ap.add_argument("--mode", default="plain", choices=["plain", "grounding"],
                    help="plain = markdown; grounding = labelled layout blocks with 0-1000 boxes")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"],
                    help="Output cap per page (default 4096, the model card's)")
    ap.add_argument("--temperature", type=float, default=SERVING["temperature"])
    ap.add_argument("--top-p", type=float, default=SERVING["top_p"],
                    help="Default 0.9, the model card's; LightOn's client uses 1.0")
    ap.add_argument("--longest-edge", type=int, default=SERVING["longest_edge"],
                    help="Downscale pages to at most this many pixels on the longest edge")
    ap.add_argument("--no-fix-dollars", action="store_true",
                    help="Keep the model's text in `markdown` without repairing escaped dollars")
    ap.add_argument("--repair-max-tokens", type=int, default=0,
                    help="Re-send pages that hit --max-tokens once, with this cap and "
                         "temperature 0.1, into repair/ (default 0: off; 8192 suggested)")
    ap.add_argument("--window", default="auto",
                    help="Pages in flight: 'auto' (saturate adapts it) or a fixed number")
    ap.add_argument("--prepare-workers", type=int, default=None,
                    help="Threads that decode, resize and encode pages (default: CPU_CORES)")
    ap.add_argument("--serve-args", default="",
                    help="Extra `vllm serve` flags, one quoted string, e.g. "
                         "\"--max-num-batched-tokens 16384\"")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    ap.add_argument("--shard", default="0/1", metavar="RANK/WORLD",
                    help="Process only shard RANK of WORLD (fan-out over several Jobs), e.g. 0/4")
    args = ap.parse_args()
    shard = parse_shard(args.shard)
    bucket_input = args.input_dataset.startswith("hf://buckets/")
    bucket_output = args.output_dataset.startswith("hf://buckets/")
    if bucket_input:
        check_image_glob(args.input_dataset)
        if args.id_column is not None:
            print("--id-column is ignored for a bucket input: the id is the file path",
                  file=sys.stderr)

    user_serve_args = shlex.split(args.serve_args)
    if args.longest_edge < 1:
        ap.error("--longest-edge must be at least 1")
    try:  # --serve-args may set its own --max-model-len; vLLM uses the last one given
        context = int(flag_value(user_serve_args, "--max-model-len") or SERVING["context"])
    except ValueError:
        ap.error("--max-model-len in --serve-args must be a whole number of tokens")
    room = context - max_image_tokens(args.longest_edge) - 64
    for flag, value in (("--max-tokens", args.max_tokens), ("--repair-max-tokens", args.repair_max_tokens)):
        if value > room:
            ap.error(f"{flag} {value} can overflow the {context}-token context with a "
                     f"{args.longest_edge} px page; use at most {room}")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be at least 1")
    if args.repair_max_tokens and args.repair_max_tokens <= args.max_tokens:
        ap.error("--repair-max-tokens must be larger than --max-tokens")
    if args.window != "auto" and not (args.window.isdigit() and int(args.window) >= 1):
        ap.error("--window must be 'auto' or a number of at least 1")

    import pyarrow as pa
    from saturate import Auto, Engine, Fixed, dataset_rows, pump

    if args.id_column is None:
        ids = "index"
    elif args.id_column in ("index", "content"):
        # saturate reads these two names as id strategies, so a column with one of them
        # needs a callable. It returns the raw value, so saturate still refuses a null id.
        def ids(example):
            return example[args.id_column]
    else:
        ids = args.id_column

    if bucket_input:
        rows = image_file_rows(args.input_dataset, args.image_column, args.limit, shard)
    else:
        rows = dataset_rows(
            args.input_dataset, config=args.config, split=args.split,
            columns=[args.image_column], ids=ids, limit=args.limit, shard=shard,
        )

    # Results go to data/ of a dataset repo, or straight under a bucket path; repairs go to
    # repair/ beside them. saturate lists only part-*.parquet directly in an output
    # directory, so the repair/ parts never mix with a bucket output's own parts.
    if bucket_output:
        from huggingface_hub import HfApi

        HfApi().create_bucket(bucket_id(args.output_dataset), private=True, exist_ok=True)
        data_uri = args.output_dataset.rstrip("/")
        repair_uri = f"{data_uri}/repair"
    else:
        data_uri = f"hf://datasets/{args.output_dataset}/data"
        repair_uri = f"hf://datasets/{args.output_dataset}/repair"

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()), ("raw", pa.string()),
        ("finish_reason", pa.string()), ("layout_blocks", pa.string()), ("model", pa.string()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    # Pages that hit --max-tokens, kept on local disk for the repair pass: those truncated
    # in this run (stashed in parse) and those saved as truncated by an earlier, interrupted
    # run (stashed as the stream passes them; the main pass skips them).
    repair_dir = tempfile.mkdtemp(prefix="lighton-ocr3-repair-") if args.repair_max_tokens else None
    earlier = saved_truncations(data_uri, repair_uri) if repair_dir else set()
    truncated = []

    def stash(row_id, image):
        path = os.path.join(repair_dir, hashlib.sha1(row_id.encode()).hexdigest() + ".png")
        with open(path, "wb") as handle:
            handle.write(page_png(image, args.longest_edge))
        truncated.append((row_id, path))

    def to_request(row):
        png = page_png(row[args.image_column], args.longest_edge)
        return chat_request(args.model, png, args.mode, args.max_tokens, args.temperature, args.top_p)

    def make_parse(stash_truncated: bool):
        def parse(row, body):
            choice = body["choices"][0]
            raw = (choice["message"].get("content") or "").strip()
            markdown, blocks = postprocess(raw, args.mode, fix_dollars=not args.no_fix_dollars)
            usage = body.get("usage") or {}
            if stash_truncated and choice.get("finish_reason") == "length":
                stash(row["__id"], row[args.image_column])
            return {
                "markdown": markdown,
                "raw": raw,
                "finish_reason": choice.get("finish_reason"),
                "layout_blocks": blocks,
                "model": args.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        return parse

    def with_ids(source):
        # Keep each row's id on the row, so parse can name the page it stashes for repair.
        for row_id, row in source:
            if row_id in earlier:
                stash(row_id, row[args.image_column])
            yield row_id, {**row, "__id": row_id}

    cap = window_cap()
    if args.window == "auto":
        window = Auto(initial=min(64, cap), max_limit=cap)
        print(f"window: auto, at most {cap} pages in flight (host memory)", file=sys.stderr)
    else:
        window = Fixed(int(args.window))
        if int(args.window) > cap:
            print(f"warning: --window {args.window} is above the {cap} pages this machine's "
                  "memory allows; the Job may run out of memory", file=sys.stderr)
    cores = max(1, int(float(os.environ.get("CPU_CORES") or "4")))
    workers = args.prepare_workers if args.prepare_workers is not None else cores
    extra = [
        "--max-model-len", str(SERVING["context"]),
        *SERVING["serve_args"],
        *user_serve_args,
    ]
    # Several API server processes share the image decoding and preprocessing, which one
    # process cannot keep up with on a fast GPU. Only where there are cores to spare.
    if cores >= SERVING["api_servers_min_cores"] and flag_value(extra, "--api-server-count") is None:
        extra += ["--api-server-count", str(SERVING["api_servers"])]
        print(f"server: --api-server-count {SERVING['api_servers']} ({cores} CPU cores)", file=sys.stderr)

    with Engine(args.model, engine="vllm", extra_args=extra) as endpoint:
        stats = pump(with_ids(rows) if repair_dir else rows, to_request, make_parse(bool(repair_dir)),
                     endpoint, data_uri, window=window, prepare_workers=workers,
                     retry_errors=args.retry_errors, schema=schema, shard=shard)
        print("LIGHTON_OCR3_SATURATE " + stats.to_json(), flush=True)

        if repair_dir and truncated:
            print(f"repair: re-sending {len(truncated)} pages that hit --max-tokens "
                  f"with a cap of {args.repair_max_tokens}", file=sys.stderr, flush=True)

            def repair_rows():
                for row_id, path in truncated:
                    with open(path, "rb") as handle:
                        yield row_id, {"png": handle.read()}

            def repair_request(row):
                return chat_request(args.model, row["png"], args.mode, args.repair_max_tokens,
                                    SERVING["repair_temperature"], args.top_p)

            repair_stats = pump(repair_rows(), repair_request, make_parse(False), endpoint,
                                repair_uri, window=window,
                                retry_errors=args.retry_errors, schema=schema, shard=shard)
            print("LIGHTON_OCR3_SATURATE_REPAIR " + repair_stats.to_json(), flush=True)

    where = (f"https://huggingface.co/buckets/{bucket_id(args.output_dataset)}" if bucket_output
             else f"https://huggingface.co/datasets/{args.output_dataset}")
    print(f"{where} shard {shard[0]}/{shard[1]} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows, "
          f"{len(truncated) if repair_dir else 'n/a'} truncated pages re-sent to repair/)", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    # saturate README: a process can hang or abort at interpreter exit after a Hub Parquet
    # stream stopped early (--max-samples); everything is written, so leave now.
    os._exit(0)


if __name__ == "__main__":
    main()

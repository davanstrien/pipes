# hf-pipe · `jobpipe`

Run a model over a whole dataset on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) with one command.

```
hf pipe embed fka/prompts.chat
```

That's the entire invocation: no GPU on your machine, no config file, no concurrency
tuning. It resolves everything it needs, launches a Job, streams results to a
dataset repo as they arrive, and tells you exactly what it decided:

```
resolved embeddings spec:
  dataset  'fka/prompts.chat'   [user]
  config   'default'            [detected]
  split    'train'              [detected]
  column   'prompt'             [detected]  longest varying string column in preview
  model    'BAAI/bge-m3'        [curated]   multilingual all-rounder; encoder -> TEI
  engine   'tei'                [curated]   from catalogue entry (receipted arrangement)
  flavor   'a10g-small'         [default]
  output   'hf://datasets/you/prompts-chat-embeddings/data'  [default]
run 9fee4f3d: 1 job(s) launched
output: hf://datasets/you/prompts-chat-embeddings/data
watch: hf pipe status 'hf://datasets/you/prompts-chat-embeddings/data'
```

Zero-config is never magic: every filled blank is labeled with where it came from
(`user` / `curated` / `detected` / `default`), and every line is overridable
(`--column`, `--model`, `--flavor`, ...). `--world 4` fans the same run out over
4 Jobs. Kill anything anytime; re-running the same command resumes exactly
(results are keyed by row id — done work is never repeated).

**v0 does one task: `embed`.** The design is task-general (the same spec/compile
path is built to carry generation, OCR, transcription); more tasks land as they
earn receipts.

## Install

As an `hf` CLI extension:

```
hf extensions install davanstrien/hf-pipe
hf pipe embed user/dataset
```

Or standalone:

```
pip install git+https://github.com/davanstrien/hf-pipe   # CLI: jobpipe embed ...
```

Or from Python:

```python
from jobpipe import embed

run = embed("user/dataset")          # same resolution path, same receipts
run.status()                          # progress read from storage, exact counts
run.publish("you/my-embeddings")      # optional: compile output to a clean dataset
```

You need a Hugging Face token with Jobs access (`hf auth login`). Jobs bill by
the minute; the runs above cost a few cents each.

## See before you spend

```
hf pipe embed user/dataset --compile-only    # print the exact code + commands, launch nothing
hf pipe embed user/dataset --dump my.py      # eject an editable uv script, launch nothing
hf jobs uv run --flavor a10g-small --image vllm/vllm-openai:latest -s HF_TOKEN my.py
```

The generated driver is a self-contained [PEP 723](https://peps.python.org/pep-0723/)
uv script (with a `[tool.hf-jobs]` header carrying its Job config) built on
[saturate](https://github.com/davanstrien/saturate) for adaptive concurrency,
crash-safe resumable output, and durable error rows. Edit the ejected script
freely — it has no dependency on this repo.

## Tested model paths

```
hf pipe catalogue
```

Curated models per task ship with run receipts (date, job, throughput, failure
count). Any other model works too — it's resolved from Hub metadata and labeled
`unverified path` so you know which regime you're in.

## Anatomy of a run

Every run writes to its output location: parquet result parts (streamed while
hot), a `run.json` intent record (spec, expected counts, the staged driver's
sha256, resolution provenance), exact-count stats, and completion markers.
`hf pipe status <output>` — or anything else that can read the repo — renders
progress from storage alone; no Jobs API required.

## Status

Early, working, moving. One task, receipts for everything it claims. Issues and
model-path contributions (with receipts) welcome.

Extending (new tasks, new curated models): see [EXTENDING.md](EXTENDING.md).

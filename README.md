# hf-pipe · `jobpipe`

Run a model over a whole dataset on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) with one command.

```
hf pipe embed biglam/europeana_newspapers --keep language date --world 4
```

`hf pipe` does not generate code. It runs a tested single-file recipe from
[uv-scripts](https://huggingface.co/uv-scripts), pinned to a commit, and adds
the parts a recipe alone does not have:

- **Fills the blanks, with receipts.** Config, split, text column and id column
  are read from the dataset viewer. Every value is labelled with where it came
  from (`user`, `detected`, `default`) before anything launches.
- **Takes three kinds of input.** A Hub dataset id, a bucket glob
  (`hf://buckets/me/b/data/*.parquet`; also `.jsonl`, `.csv`, `.txt`, `.md`),
  or a local directory, which is synced to a bucket first.
- **Fans out.** `--world 4` starts 4 Jobs with `--shard 0/4` ... `3/4` and one
  output. Each row's id decides its shard, so the Jobs need no coordinator.
- **Records the run.** `run.json` next to the output holds the pinned recipe,
  the exact arguments, the Job ids and a fingerprint of the setup. A re-run
  with the same setup resumes. A re-run with another model or text handling is
  refused, because its rows would be skipped as done.
- **Reads progress from storage.** `hf pipe status <output>` counts rows from
  saturate's files; no database.

## See before you spend

```
hf pipe embed biglam/europeana_newspapers --limit 200000 --world 4 --dry-run
```

prints the resolved values and the equivalent `hf jobs uv run` command for each
Job. Copy one and edit it, or run without `--dry-run`. `--json` gives one JSON
document for agents.

```
resolved embed plan:
  source      'biglam/europeana_newspapers'   [user]
  config      'default'                       [detected]
  column      'text'                          [detected]
  id_column   'id'                            [detected]  string column named 'id'; output rows join back on it
  model       'BAAI/bge-m3'                   [default]   multilingual (100+ languages), 1024-d
  max_seq_len 512                             [default]   predictable cost; the recipe alone would use the model maximum
  output      'hf://buckets/you/jobpipe-outputs/europeana_newspapers-bge-m3' [default]
```

## Install

```
hf extensions install davanstrien/hf-pipe
hf pipe embed user/dataset
```

or `pip install git+https://github.com/davanstrien/hf-pipe` for the `jobpipe`
command and the Python API:

```python
from jobpipe import embed

plan = embed("user/dataset", launch=False)   # resolve only
print(*plan.commands(), sep="\n")
run = embed("user/dataset", world=2)
run.status()
```

You need a token with Jobs access (`hf auth login`). Jobs bill by the minute.

## The pieces

- [saturate](https://github.com/davanstrien/saturate): the batch client. It
  drives the vLLM server, sets how many requests are in flight, and writes
  resumable parquet with durable error rows.
- [uv-scripts](https://huggingface.co/uv-scripts) recipes: the code that runs.
  `embed` uses `embeddings/generate-embeddings-saturate.py` (vLLM pooling, one
  text per request, float32 output, row ids and kept columns).
- `hf pipe`: resolve, launch, record, status.

## Known limits

- Each Job reads the whole input and keeps its share of the rows. For a very
  large dataset with many Jobs, that repeats the download per Job.
- Kept columns are written as strings (lists and dicts as JSON).
- One task so far (`embed`). Another task is another recipe plus a resolver.

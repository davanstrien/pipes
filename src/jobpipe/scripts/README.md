# Scripts jobpipe launches

Each file here is a copy of a recipe from [uv-scripts](https://github.com/davanstrien/uv-scripts).
jobpipe launches its own copy, so a change in uv-scripts does not change what `hf pipe` runs
until someone copies it here on purpose, and uv-scripts does not have to follow jobpipe's needs.

| File | Copied from | Commit |
|---|---|---|
| `embed.py` | `embeddings/generate-embeddings-saturate.py` | `b2cb30f` (branch `embeddings/shard-keep`) |

To refresh a copy: copy the file, update the first comment lines and this table, then run
`uv run --group dev pytest`. `tests/test_scripts.py` checks that every flag jobpipe passes is
still accepted by the copy.

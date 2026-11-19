# Scripts jobpipe launches

Each file here is a copy of a recipe from [uv-scripts](https://github.com/davanstrien/uv-scripts).
jobpipe launches its own copy, so a change in uv-scripts does not change what `hf pipe` runs
until someone copies it here on purpose, and uv-scripts does not have to follow jobpipe's needs.

| File | Copied from | Commit |
|---|---|---|
| `embed.py` | `embeddings/generate-embeddings-saturate.py` | `b2cb30f` (branch `embeddings/shard-keep`) |
| `ocr.py` | `ocr/lighton-ocr3-saturate.py` | `f2792c8` (`main`) |
| `extract.py` | made here with `hf pipe new extract` (no uv-scripts original) | |

To refresh a copy: copy the file, update the first comment lines and this table, then run
`uv run --group dev pytest`. `tests/test_scripts.py` checks that every flag jobpipe passes is
still accepted by the copy.

## Recipes made from the template

`../templates/recipe.py` is a whole saturate recipe with the task left out. `hf pipe new NAME`
prints it with the name filled in. The part between the `# ==== TASK` markers is the recipe's
own (its flags, request, reply and output columns); the rest is plumbing that is the same in
every recipe made from the template: Hub dataset, bucket table and bucket file inputs,
dataset or bucket output, `--shard`, `--keep` with source types, resume, the stats line.
`tests/test_extract.py` checks that `extract.py`'s plumbing still matches the template. A
change to the plumbing is made in the template and then copied into each recipe made from it.

# Extending jobpipe

Two ways to extend: add a **model** (data, minutes) or add a **task** (a
template module, an afternoon). The rule that keeps both sane: **templates
carry mechanism only; anything model-specific is data.** Serve flags, prompts,
sampling — never hard-coded in a template, always interpolated from the spec
or the catalogue.

## The plumbing in one diagram

```
 embed("org/ds") / CLI / spec.json          resolve()                data sources
        │                            fills blanks, stamps      ┌ dataset introspection [detected]
        └──────────────────────────▶ provenance:          ────▶├ model metadata        [detected]
                                     user > curated >          ├ catalogue.py          [curated]
                                     detected > default        └ hand defaults         [default]
                                            │
                                            ▼
                                    TaskSpec (typed; `jobpipe schema` = the UI contract)
                                            ▼
                                     compile()  ── templates/<task>.py renders a
                                            │      self-contained uv-script driver
                                            │      (PEP 723 + [tool.hf-jobs] header)
                                            ▼
                                     launch()   ── stage driver (sha256 in run.json)
                                            │      → one Job per rank
                                            ▼
                                    output storage: parquet parts + run.json +
                                    exact-count stats  ◀── `jobpipe status` reads
                                                            storage only
```

## Add a model (data change)

Append an entry to `CATALOGUE` in `src/jobpipe/catalogue.py`:

```python
{
    "model": "org/model-id",
    "engine": "tei",            # the receipted serving arrangement
    "why": "one line on when to pick it",
    "prompts": None,            # None = verified prompt-free; omit = unknown
    "receipts": [
        {"date": "2026-07-30", "job": "<job id>",
         "note": "N texts, X tok/s, 0 failed"},
    ],
}
```

The contract: **an entry without a run receipt is not curated, it's a guess** —
run the model through `jobpipe embed <ds> --model <id> --limit 1000` first and
cite the job. Models outside the catalogue still work; they resolve from Hub
metadata and are labeled `unverified path` so users know which regime they're in.

## Add a task (template change)

Four steps, ~150 lines total (generation.py is the model to copy):

1. **Spec**: add the task name to the `Literal` in `TaskSpec.task`
   (`src/jobpipe/spec.py`) plus any task-specific fields; add its rules to the
   validator (e.g. generation forces `batch=1`).
2. **Template**: write `src/jobpipe/templates/<task>.py` exposing
   `staging_repo(spec)`, `driver(spec)`, `image(spec)`, `job_shell(...)`,
   `stage_command(...)`, `launch_commands(...)`, `publish_command(...)`.
   In practice: copy `generation.py` and change the driver body — the input
   shape, the `to_request`, the `parse`. Keep `{{placeholders}}` for anything
   that varies; no model names in the file.
3. **Register**: one line in `templates/__init__.py`'s `_TEMPLATES` dict.
4. **Verb**: a thin `def <task>(dataset, *, model, ...)` in `api.py` (build
   spec via `resolve()`, compile, launch — ~15 lines) and a subparser in
   `cli.py`.

Then: a golden test (compile a fixture spec, commit the emitted driver —
`tests/golden/REGEN.md` has the regen snippet) and at least one live run on a
small dataset as the receipt before it's documented.

There is no plugin framework and no base class on purpose — at this size, the
convention is the interface.

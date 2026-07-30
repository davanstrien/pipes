# Golden files

The golden driver + launch command are the canonical emitted artifacts for the
spec in `spec_fineweb_tei.json` (TEI embeddings arm; `numExamples` pinned so
the test never touches the network). Purpose: rot-guard — any change to
emitted-driver text or launch commands must appear as a conscious golden diff
in review, never as silent drift.

Regenerate after an intentional template/compiler change:

    uv run python - << 'PY'
    import json, pathlib
    from jobpipe import TaskSpec, compile
    G = pathlib.Path('tests/golden')
    spec = TaskSpec(**json.loads((G/'spec_fineweb_tei.json').read_text()))
    c = compile(spec, run_id='goldrun1')
    (G/'embed_fineweb-edu.py').write_text(c.driver)
    (G/'commands.txt').write_text(c.commands[1].replace(c.driver_name, 'embed_fineweb-edu.py') + '\n')
    PY

One deliberate delta asserted separately: real runs stage `{slug}-{run_id}.py`
(immutable per-run code artifact, sha256 in run.json); the golden normalizes
the filename before comparison.

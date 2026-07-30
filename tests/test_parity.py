"""Golden-driver test: compile() reproduces the committed canonical driver.

Direction flipped 2026-07-30 (rename + modernization): the golden is now the
SEED's canonical saturate-native form, regenerated from this compiler — the
console POC mirrors it, not the other way round. Provenance + history in
tests/golden/REGEN.md. Purpose is rot-guard: any change to emitted driver
text must show up as a conscious golden diff in review, never silent drift.
"""

import json
import pathlib

from jobpipe import TaskSpec, compile

GOLDEN = pathlib.Path(__file__).parent / "golden"
RUN_ID = "goldrun1"


def golden_spec() -> TaskSpec:
    return TaskSpec(**json.loads((GOLDEN / "spec_fineweb_tei.json").read_text()))


def test_driver_byte_exact():
    c = compile(golden_spec(), run_id=RUN_ID)
    assert c.driver == (GOLDEN / "embed_fineweb-edu.py").read_text()


def test_launch_command_matches_console():
    c = compile(golden_spec(), run_id=RUN_ID)
    stage, launch = c.commands[0], c.commands[1]
    assert len(c.commands) == 2  # publish=false, world=1

    # the one deliberate delta: per-run code artifact vs overwrite-in-place
    assert c.driver_name == f"embed_fineweb-edu-{RUN_ID}.py"
    assert c.driver_name in stage and c.driver_name in launch

    normalized = launch.replace(c.driver_name, "embed_fineweb-edu.py")
    assert normalized == (GOLDEN / "commands.txt").read_text().rstrip("\n")


def test_run_json_intent_fields():
    c = compile(golden_spec(), run_id=RUN_ID)
    rj = c.run_json
    # G1: denominator = ceil(min(limit, num_examples) / batch) = 12800/64
    assert rj["expected_rows"] == 200
    assert rj["items_per_row"] == 64  # G4
    assert rj["expected_items"] == 12800
    assert rj["world"] == 1  # G2's launch-time cover
    assert rj["jobs"] == [] and rj["created_at"] is None  # launch() fills these
    # run <-> code artifact link
    assert rj["driver"]["staged"].endswith(c.driver_name)
    assert len(rj["driver"]["sha256"]) == 64


def test_no_network_when_num_examples_pinned():
    # compile() must not touch the network when the spec carries the count
    import jobpipe.introspect as intro

    called = []
    orig = intro.num_examples
    intro.num_examples = lambda *a, **k: called.append(a) or orig(*a, **k)
    try:
        compile(golden_spec(), run_id=RUN_ID)
    finally:
        intro.num_examples = orig
    assert called == []


def test_vllm_arm_renders():
    spec = golden_spec().model_copy(update={"engine": "vllm", "world": 4})
    c = compile(spec, run_id=RUN_ID)
    assert "Engine(MODEL, engine=\"vllm\"" in c.driver
    assert "--runner\", \"pooling\", \"--convert\", \"embed\"" in c.driver
    # limit lives in the source now (dataset_rows texts cap; N_BATCHES guard gone)
    assert ", limit=12800)" in c.driver
    assert 'ap.add_argument("--world", type=int, default=4)' in c.driver
    assert len(c.commands) == 1 + 4  # stage + one launch per rank
    assert "--rank 3 --world 4" in c.commands[-1]


def test_dataset_output_gets_viewer_card():
    spec = golden_spec().model_copy(
        update={"output": "hf://datasets/davanstrien/embed-out/run1"}
    )
    c = compile(spec, run_id=RUN_ID)
    assert 'create_repo("davanstrien/embed-out"' in c.driver
    assert "config_name: manifest" in c.driver  # explicit configs beat viewer glob
    assert "run1/part-*.parquet" in c.driver

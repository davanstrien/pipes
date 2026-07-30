"""Generation template: prompt-in-column, batch=1, vLLM, content-hash ids.

No console-POC golden exists for this task (the byte-parity contract is the
embeddings template's alone) — these tests pin the compile SHAPE and that the
generated driver is valid Python targeting the renamed engine (saturate).
"""

import py_compile
import tempfile

import pytest

from jobpipe import TaskSpec, compile

RUN_ID = "goldrun1"


def gen_spec(**over) -> TaskSpec:
    base = dict(
        task="generation",
        dataset="databricks/databricks-dolly-15k",
        column="instruction",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        output="hf://buckets/davanstrien/gen-out/run1",
        limit=1000,
        num_examples=15011,
    )
    base.update(over)
    return TaskSpec(**base)


def test_driver_shape_and_syntax():
    c = compile(gen_spec(), run_id=RUN_ID)
    d = c.driver
    # renamed engine + sources input, not a hand-rolled loop
    assert "from saturate import Auto, Engine, dataset_rows, pump" in d
    assert 'ids="content"' in d and "limit=1000" in d
    assert '"max_tokens": 512, "temperature": 0.7' in d
    assert "pumpjack" not in d  # the codename must not leak into new drivers
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(d)
    py_compile.compile(f.name, doraise=True)  # generated source is valid Python


def test_batch_is_one_request_per_row():
    spec = gen_spec()
    assert spec.batch == 1  # coerced, not defaulted to the embeddings 64
    c = compile(spec, run_id=RUN_ID)
    assert c.run_json["items_per_row"] == 1
    assert c.run_json["expected_rows"] == 1000  # rows == items at batch=1
    with pytest.raises(ValueError, match="one request per row"):
        gen_spec(batch=8)


def test_generation_is_vllm_only():
    with pytest.raises(ValueError, match="vllm"):
        gen_spec(engine="tei")


def test_slug_and_wheel_target():
    c = compile(gen_spec(), run_id=RUN_ID)
    assert c.slug.startswith("gen_")
    stage, launch = c.commands[0], c.commands[1]
    assert c.driver_name in stage and c.driver_name in launch
    assert 'saturate[hf]' in launch  # PyPI install (0.1.1 live), wheel-curl gone
    assert "vllm/vllm-openai" in launch


def test_publish_uses_saturate_reader():
    c = compile(gen_spec(publish=True), run_id=RUN_ID)
    assert "from saturate import read_output" in c.commands[-1]


def test_embeddings_defaults_unchanged():
    spec = TaskSpec(dataset="a/b", model="m", output="hf://buckets/o/n/p")
    assert spec.task == "embeddings" and spec.batch == 64
    assert spec.slug.startswith("embed_")

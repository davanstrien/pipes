"""extract: the first recipe made from the template, and its resolver."""

import importlib.util
import json
import pathlib
import sys

import pytest

import jobpipe.introspect as intro
from jobpipe.new import TEMPLATE, plumbing_of, render
from jobpipe.recipes import RECIPES, check_flags, script_path
from jobpipe.resolve import resolve_extract

SCHEMA = {"type": "object", "properties": {
    "sentiment": {"enum": ["positive", "negative"]},
    "score": {"type": "integer"},
    "topics": {"type": "array", "items": {"type": "string"}},
    "detail": {"type": "object"},
}}


def arg(plan, flag):
    return plan.args[plan.args.index(flag) + 1]


def load_recipe():
    path = script_path(RECIPES["extract"])
    spec = importlib.util.spec_from_file_location("extract_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe_args(module, *argv):
    """Parse argv the way the recipe's main() does, without starting a server."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default=module.DEFAULT_MODEL)
    module.add_task_arguments(parser)
    args = parser.parse_args(list(argv))
    module.check_task_arguments(args, parser)
    return args


def test_recipe_plumbing_matches_the_template():
    text = pathlib.Path(script_path(RECIPES["extract"])).read_text()
    assert plumbing_of(text, "extract") == plumbing_of(TEMPLATE.read_text(), "__TASK__")


def test_new_renders_a_named_recipe():
    text = render("caption")
    assert 'TASK_NAME = "caption"' in text and "__TASK__" not in text
    compile(text, "caption.py", "exec")
    with pytest.raises(ValueError, match="lower case"):
        render("My Recipe")


def test_request_and_reply_with_a_schema():
    module = load_recipe()
    args = recipe_args(module, "in", "out", "--prompt", "Title: {title}\n{text}",
                       "--schema", json.dumps(SCHEMA), "--max-chars", "5")
    assert module.input_columns(args) == ["title", "text"]
    body = module.to_request(args, {"title": "A", "text": "abcdefgh"})
    assert body["messages"][-1]["content"] == "Title: A\nabcde"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA

    import pyarrow as pa

    fields = dict(module.output_fields(args))
    assert fields["sentiment"] == pa.string() and fields["score"] == pa.int64()
    assert fields["topics"] == pa.list_(pa.string()) and fields["detail"] == pa.string()

    reply = {"sentiment": "positive", "score": 4, "topics": ["film"], "detail": {"a": 1}}
    record = module.parse(args, {}, {"choices": [{"message": {"content": json.dumps(reply)},
                                                  "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 9, "completion_tokens": 3}})
    assert record["sentiment"] == "positive" and record["detail"] == '{"a": 1}'
    assert record["prompt_tokens"] == 9
    with pytest.raises(ValueError, match="max-tokens"):
        module.parse(args, {}, {"choices": [{"message": {"content": "{"},
                                             "finish_reason": "length"}]})


def test_recipe_refuses_bad_task_flags():
    module = load_recipe()
    with pytest.raises(SystemExit):
        recipe_args(module, "in", "out", "--prompt", "no fields here")
    with pytest.raises(SystemExit):
        recipe_args(module, "in", "out", "--prompt", "{text}", "--schema",
                    '{"type": "object", "properties": {"raw": {}}}')


def test_plan_over_a_dataset(offline):
    offline.setattr(intro, "column_names", lambda *a, **k: ["id", "text", "label"])
    offline.setattr(intro, "small_columns", lambda *a, **k: ["id", "label"])
    plan = resolve_extract("org/reviews", prompt="Review: {text}", schema=json.dumps(SCHEMA),
                           limit=20)
    assert plan.task == "extract" and plan.script_name == "extract"
    assert arg(plan, "--prompt") == "Review: {text}" and arg(plan, "--model") == "Qwen/Qwen3.5-4B"
    assert json.loads(arg(plan, "--schema")) == SCHEMA
    assert arg(plan, "--id-column") == "id" and plan.values["keep"] == ["label"]
    assert plan.args[1].startswith("hf://buckets/tester/jobpipe-outputs/reviews-Qwen3-5-4B-")


def test_another_prompt_gets_another_default_output(offline):
    offline.setattr(intro, "column_names", lambda *a, **k: ["text"])
    first = resolve_extract("org/reviews", prompt="Summarise: {text}")
    second = resolve_extract("org/reviews", prompt="Translate: {text}")
    assert first.output != second.output
    assert first.fingerprint["sha256"] != second.fingerprint["sha256"]


def test_refusals(offline):
    offline.setattr(intro, "column_names", lambda *a, **k: ["id", "text"])
    with pytest.raises(ValueError, match="no {column}"):
        resolve_extract("org/reviews", prompt="Summarise this")
    with pytest.raises(ValueError, match="not columns"):
        resolve_extract("org/reviews", prompt="{body}")
    with pytest.raises(ValueError, match="not JSON"):
        resolve_extract("org/reviews", prompt="{text}", schema="{nope")
    with pytest.raises(ValueError, match="clash"):
        resolve_extract("org/reviews", prompt="{text}",
                        schema='{"type": "object", "properties": {"model": {}}}')
    with pytest.raises(ValueError, match="valid template"):
        resolve_extract("org/reviews", prompt="{text")


def test_prompt_and_schema_from_files(offline, tmp_path):
    offline.setattr(intro, "column_names", lambda *a, **k: ["text"])
    (tmp_path / "prompt.txt").write_text("Label: {text}")
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    plan = resolve_extract("org/reviews", prompt=str(tmp_path / "prompt.txt"),
                           schema=str(tmp_path / "schema.json"))
    assert arg(plan, "--prompt") == "Label: {text}"
    assert "from" in plan.notes["prompt"] and "from" in plan.notes["schema"]


def test_text_files_and_flags_the_copy_accepts(offline, monkeypatch, tmp_path):
    import jobpipe.resolve as res

    monkeypatch.setattr(res, "script_path", script_path)  # the real copy
    text = pathlib.Path(script_path(RECIPES["extract"])).read_text()
    offline.setattr(intro, "column_names", lambda *a, **k: ["id", "text"])
    (tmp_path / "a.txt").write_text("x")
    plans = [
        resolve_extract("org/reviews", prompt="{text}", schema=json.dumps(SCHEMA),
                        system="Be brief.", keep=["id"], world=2, limit=5),
        resolve_extract("hf://buckets/me/notes/**/*.md", prompt="{text}"),
        resolve_extract(str(tmp_path), prompt="{text}"),
    ]
    assert "--id-column" not in plans[1].args and "--config" not in plans[1].args
    for plan in plans:
        check_flags(text, plan.rank_args(0), plan.recipe_url)
    with pytest.raises(ValueError, match="no columns"):
        resolve_extract("hf://buckets/me/notes/*.md", prompt="{text}", keep=["a"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

"""resolve_embed(): zero-config filling, provenance, and the recipe's arguments."""

import pytest
from conftest import PINNED

import jobpipe.introspect as intro
from jobpipe.resolve import resolve_embed


def arg(plan, flag):
    return plan.args[plan.args.index(flag) + 1]


def test_zero_config_hub_dataset(offline):
    plan = resolve_embed("org/ds")
    assert plan.recipe_url == PINNED and plan.provenance["recipe"] == "default"
    assert plan.args[:2] == ["org/ds", "hf://buckets/tester/jobpipe-outputs/ds-bge-m3"]
    assert arg(plan, "--model") == "BAAI/bge-m3" and plan.provenance["model"] == "default"
    assert arg(plan, "--column") == "text" and plan.provenance["column"] == "detected"
    assert arg(plan, "--id-column") == "id" and plan.provenance["id_column"] == "detected"
    assert arg(plan, "--max-seq-len") == "512"
    assert (arg(plan, "--config"), arg(plan, "--split")) == ("default", "train")
    assert "--skip-empty" in plan.args and "--private" in plan.args
    assert "--prompt" not in plan.args  # the recipe picks the model's registered prompt
    assert plan.output == "hf://buckets/tester/jobpipe-outputs/ds-bge-m3"
    assert plan.expected_items == 10_000 and plan.world == 1


def test_user_values_win(offline):
    plan = resolve_embed(
        "org/ds", model="Qwen/Qwen3-Embedding-0.6B", column="body", id_column="doc",
        keep=["lang", "date"], max_seq_len=256, prefix="passage: ", output="me/out",
        world=4, limit=1000, flavor="l4x1", timeout="30m",
    )
    for name in ("model", "column", "id_column", "keep", "max_seq_len", "prefix",
                 "output", "world", "limit", "flavor", "timeout"):
        assert plan.provenance[name] == "user", name
    assert plan.args[1] == "me/out" and plan.output == "hf://datasets/me/out/data"
    assert plan.args[plan.args.index("--keep") + 1:][:2] == ["lang", "date"]
    assert arg(plan, "--prompt") == "passage: " and arg(plan, "--max-samples") == "1000"
    assert plan.expected_items == 1000
    assert plan.rank_args(3)[-2:] == ["--shard", "3/4"]
    commands = plan.commands()
    assert len(commands) == 4 and "--flavor l4x1 --timeout 30m" in commands[0]
    assert "--shard 3/4" in commands[3] and "  embed.py " in commands[0]
    assert plan.script_command() == "hf pipe script embed > embed.py"


def test_longest_column_when_no_text(offline):
    offline.setattr(intro, "string_columns",
                    lambda *a, **k: [{"name": "content", "avg_len": 900.0},
                                     {"name": "title", "avg_len": 40.0}])
    plan = resolve_embed("org/ds")
    assert arg(plan, "--column") == "content"
    assert "longest varying string column" in plan.notes["column"]
    assert "--id-column" not in plan.args  # no id column: the recipe uses the position


def test_no_viewer_falls_back(offline):
    offline.setattr(intro, "dataset_layout", lambda *a, **k: [])
    offline.setattr(intro, "string_columns", lambda *a, **k: [])
    offline.setattr(intro, "num_examples", lambda *a, **k: None)
    plan = resolve_embed("org/private-ds")
    assert (arg(plan, "--config"), arg(plan, "--split"), arg(plan, "--column")) == (
        "default", "train", "text")
    assert plan.provenance["column"] == "default" and plan.expected_items is None


def test_bucket_glob(offline):
    def refuse(*a, **k):
        raise AssertionError("bucket inputs must not call the viewer")

    for name in ("dataset_layout", "string_columns", "num_examples"):
        offline.setattr(intro, name, refuse)
    plan = resolve_embed("hf://buckets/me/corpus/data/*.parquet", column="body", limit=50)
    assert plan.args[0] == "hf://buckets/me/corpus/data/*.parquet"
    assert "--config" not in plan.args and arg(plan, "--column") == "body"
    assert plan.output == "hf://buckets/tester/jobpipe-outputs/corpus-bge-m3"
    assert plan.expected_items == 50 and plan.uploads == []


def test_bucket_glob_needs_a_known_extension(offline):
    with pytest.raises(ValueError, match="glob ending in"):
        resolve_embed("hf://buckets/me/corpus/data/")


def test_local_directory_is_planned_as_an_upload(offline, tmp_path):
    notes = tmp_path / "My Notes"
    (notes / "sub").mkdir(parents=True)
    for name in ("a.txt", "b.txt", "sub/c.txt"):
        (notes / name).write_text("some text")
    (notes / "readme.md").write_text("# fewer of these")
    plan = resolve_embed(str(notes))
    assert plan.args[0].startswith("hf://buckets/tester/jobpipe-inputs/My-Notes-")
    assert plan.args[0].endswith("/**/*.txt")
    assert arg(plan, "--column") == "text" and plan.provenance["column"] == "detected"
    directory, prefix, pattern = plan.uploads[0]
    assert directory == str(notes.resolve()) and pattern == "*.txt"
    assert plan.args[0] == f"{prefix}/**/*.txt"
    name = prefix.rsplit("/", 1)[-1]  # My-Notes-<hash of the folder's path>
    assert plan.output == f"hf://buckets/tester/jobpipe-outputs/{name}-bge-m3"


def test_fingerprint_tracks_model_and_text_handling(offline):
    base = resolve_embed("org/ds").fingerprint
    assert resolve_embed("org/ds").fingerprint == base
    assert resolve_embed("org/ds", model="other/model").fingerprint["sha256"] != base["sha256"]
    assert resolve_embed("org/ds", max_seq_len=128).fingerprint["sha256"] != base["sha256"]
    # scale and limits do not change what a row's result is
    assert resolve_embed("org/ds", world=8, limit=5).fingerprint == base


def test_bad_output(offline):
    with pytest.raises(ValueError, match="ns/name"):
        resolve_embed("org/ds", output="just-a-name")


def test_describe(offline):
    text = resolve_embed("org/ds").describe()
    assert "[detected]" in text and "[default]" in text and "'BAAI/bge-m3'" in text


def test_text_left_out_only_when_ids_are_stable(offline):
    assert "--no-text" in resolve_embed("org/ds").args  # 'id' column detected
    assert "--no-text" not in resolve_embed("org/ds", keep_text=True).args
    offline.setattr(intro, "string_columns", lambda *a, **k: [{"name": "text", "avg_len": 9.0}])
    plan = resolve_embed("org/ds")  # position ids: keep the text
    assert "--no-text" not in plan.args and plan.values["text_out"] is True


def test_text_files_use_path_ids(offline):
    plan = resolve_embed("hf://buckets/me/notes/**/*.md")
    assert "--id-column" not in plan.args and "--no-text" in plan.args
    assert plan.values["id_column"] == "(file path)"


def test_two_folders_with_one_name_get_two_snapshots(offline, tmp_path):
    for parent in ("a", "b"):
        (tmp_path / parent / "docs").mkdir(parents=True)
        (tmp_path / parent / "docs" / "x.txt").write_text(parent)
    first = resolve_embed(str(tmp_path / "a" / "docs")).uploads[0][1]
    second = resolve_embed(str(tmp_path / "b" / "docs")).uploads[0][1]
    assert first != second


def test_numbers_must_be_positive(offline):
    for bad in ({"limit": 0}, {"world": 0}, {"max_seq_len": 0}):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_embed("org/ds", **bad)


def test_owner_names_the_default_output_without_a_whoami(offline):
    import jobpipe.resolve as res

    def refuse(token):
        raise AssertionError("no whoami when owner is given")

    offline.setattr(res, "_whoami", refuse)
    plan = resolve_embed("org/ds", owner="someone")
    assert plan.output == "hf://buckets/someone/jobpipe-outputs/ds-bge-m3"

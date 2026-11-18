"""CLI: --dry-run prints the plan and commands; --json is one document."""

import json

from jobpipe.cli import main, watch_done


def test_dry_run_prints_commands(offline, capsys):
    assert main(["embed", "org/ds", "--world", "2", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    assert out.count("hf jobs uv run") == 2 and "--shard 1/2" in out
    assert "resolved embed plan" in err and "[detected]" in err


def test_dry_run_json(offline, capsys):
    assert main(["embed", "org/ds", "--dry-run", "--json"]) == 0
    out, _ = capsys.readouterr()
    doc = json.loads(out)
    assert doc["plan"]["recipe_url"].startswith("https://") and len(doc["commands"]) == 1


def test_bad_source_is_a_clean_error(offline, capsys):
    assert main(["embed", "hf://buckets/me/b/data/", "--dry-run"]) == 2
    assert "error:" in capsys.readouterr().err


def test_watch_done():
    shards_done = [{"rank": 0, "done": True}, {"rank": 1, "done": True}]
    assert watch_done({"shards": shards_done, "jobs": []})
    assert watch_done({"shards": [], "jobs": [{"rank": 0, "stage": "ERROR"}]})
    assert not watch_done({"shards": shards_done, "jobs": [{"rank": 0, "stage": "RUNNING"}]})
    assert not watch_done({"shards": [{"rank": 0, "done": False}], "jobs": []})
    assert not watch_done({"shards": [], "jobs": []})


def test_dry_run_commands_use_the_written_out_script(offline, capsys):
    import jobpipe.resolve as res
    from jobpipe.recipes import script_path

    offline.setattr(res, "script_path", script_path)
    assert main(["embed", "org/ds", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("hf pipe script embed > embed.py")
    assert "\n  embed.py \\\n" in out and "site-packages" not in out


def test_script_prints_the_copy(capsys):
    assert main(["script", "embed"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Copy of the uv-scripts recipe") and "[tool.hf-jobs]" in out
    assert main(["script", "nope"]) == 2

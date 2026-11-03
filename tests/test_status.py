"""status() against a CONTRACT-shaped local fixture (fsspec local path).

Fixture mirrors pumpjack CONTRACT §1/§5/§6: parts + manifest sidecars (one
orphan), completions markers + stats, telemetry jsonl, plus this layer's
run.json. Asserts the exact/approximate/inferred semantics the document
promises, for both sidecar layouts: saturate 0.3 (`__saturate/...`, telemetry
names with a hex suffix) and the root layout written before 0.3.
"""

import json

import pytest

from jobpipe.status import render_human, status

STATS_0 = {
    "rows_total": 100, "rows_done_prior": 0, "rows_processed": 100,
    "rows_failed": 2, "rows_deduped": 0, "prompt_tokens": 640000,
    "completion_tokens": 0, "elapsed_s": 60.0, "final_limit": 4,
    "input_bound": False, "breaker_opens": 0, "hints": [],
    "tokens_per_sec": 10666.0, "rank": 0, "world": 2,
}


LAYOUTS = {
    # saturate 0.3: every sidecar under __saturate/, telemetry names end in -{hex6}
    "0.3": {"manifest": "__saturate/manifest", "completions": "__saturate/completions",
            "telemetry": "__saturate/telemetry", "suffix": "-a1b2c3"},
    # before 0.3: sidecars at the output root, no telemetry suffix
    "0.2": {"manifest": "_manifest", "completions": "completions",
            "telemetry": ".", "suffix": ""},
}


def make_run(tmp_path, with_stats=True, with_run_json=True, layout="0.3"):
    lay = LAYOUTS[layout]
    out = tmp_path / "run"
    for key in ("manifest", "completions", "telemetry"):
        (out / lay[key]).mkdir(parents=True, exist_ok=True)
    # two parts with sidecars + one orphan (crash window)
    for name in ["part-1000-aaaaaaaa.parquet", "part-2000-bbbbbbbb.parquet"]:
        (out / name).write_bytes(b"pq")
        (out / lay["manifest"] / f"ids-{name}").write_bytes(b"pq")
    (out / "part-3000-cccccccc.parquet").write_bytes(b"pq")

    (out / lay["completions"] / "shard-0.done").write_bytes(b"done")
    if with_stats:
        (out / lay["completions"] / "stats-0.json").write_text(json.dumps(STATS_0))

    telem = out / lay["telemetry"]
    sfx = lay["suffix"]
    # shard 0: two telemetry files (a resume) — ok sums across BOTH,
    # tick window comes from the newest only
    (telem / f"telemetry-shard0-100{sfx}.jsonl").write_text(
        "\n".join(json.dumps({"t": i, "limit": 4, "inflight": 2, "waiting": 1,
                              "running": 2, "bp": 0, "ok": 10, "input_bound": False,
                              "tok_s": 9000.0}) for i in range(3))
    )
    (telem / f"telemetry-shard0-200{sfx}.jsonl").write_text(
        "\n".join(json.dumps({"t": i, "limit": 4, "inflight": 2, "waiting": 1,
                              "running": 2, "bp": 0, "ok": 20, "input_bound": False,
                              "tok_s": 11000.0}) for i in range(3))
    )
    # shard 1: telemetry only — no marker, no stats (still running / crashed)
    (telem / f"telemetry-shard1-150{sfx}.jsonl").write_text(
        json.dumps({"t": 0, "limit": 4, "inflight": 2, "waiting": 0,
                    "running": 2, "bp": 0, "ok": 15, "input_bound": False}) + "\n"
    )

    if with_run_json:
        (out / "run.json").write_text(json.dumps({
            "version": 1, "task": "embeddings", "run_id": "testrun1",
            "slug": "embed_test", "spec": {"output": str(out)},
            "expected_rows": 200, "items_per_row": 64, "expected_items": 12800,
            "world": 2, "jobs": [], "created_at": "2026-07-28T00:00:00+00:00",
        }))
    return out


@pytest.mark.parametrize("layout", ["0.3", "0.2"])
def test_exact_channel(tmp_path, layout):
    doc = status(str(make_run(tmp_path, layout=layout)), include_jobs=False)
    p = doc["progress"]
    assert p["exact"] is True
    assert p["rows_ok"] == 100  # from stats, not telemetry's 105
    assert p["rows_failed"] == 2
    assert p["expected_rows"] == 200 and p["pct"] == 0.5  # G1: denominator from run.json
    assert p["items_ok"] == 6400 and p["items_per_row"] == 64  # G4
    assert doc["storage"] == {"parts": 3, "orphans": 1, "world": 2, "world_inferred": False}
    assert doc["run_id"] == "testrun1"

    shards = {s["rank"]: s for s in doc["shards"]}
    assert shards[0]["done"] and shards[0]["stats"]["rows_processed"] == 100
    assert not shards[1]["done"] and shards[1]["stats"] is None

    # tick window from the NEWEST telemetry file per shard
    assert [t["ok"] for t in doc["telemetry"]["0"]] == [20, 20, 20]
    assert render_human(doc)  # smoke: renders without crashing


@pytest.mark.parametrize("layout", ["0.3", "0.2"])
def test_approximate_fallback_without_stats(tmp_path, layout):
    doc = status(str(make_run(tmp_path, with_stats=False, layout=layout)), include_jobs=False)
    p = doc["progress"]
    assert p["exact"] is False
    # Sum(ok) across ALL telemetry files: 3*10 + 3*20 + 15 — the documented
    # undercount-the-tail approximation
    assert p["rows_ok"] == 105
    assert p["rows_failed"] is None
    assert doc["storage"]["world"] == 2  # run.json still supplies world (G2 cover)


@pytest.mark.parametrize("layout", ["0.3", "0.2"])
def test_world_inferred_without_run_json(tmp_path, layout):
    doc = status(str(make_run(tmp_path, with_stats=False, with_run_json=False, layout=layout)),
                 include_jobs=False)
    assert doc["storage"]["world"] == 2  # max shard index seen + 1
    assert doc["storage"]["world_inferred"] is True  # G2: flagged, never silent
    assert doc["progress"]["expected_rows"] is None  # no denominator without run.json
    assert doc["progress"]["pct"] is None


def test_empty_output(tmp_path):
    doc = status(str(tmp_path / "nothing-here"), include_jobs=False)
    assert doc["storage"]["parts"] == 0
    assert doc["progress"]["rows_ok"] == 0


@pytest.mark.parametrize("version_key", ["version"])
def test_document_is_versioned(tmp_path, version_key):
    doc = status(str(make_run(tmp_path)), include_jobs=False)
    assert doc[version_key] == 1


def test_resumed_02_output_prefers_03_sidecars(tmp_path):
    """A 0.2 output resumed under 0.3 keeps its old sidecars at the root and
    gains new ones under __saturate/. The 0.3 stats file is the newer count."""
    out = make_run(tmp_path, layout="0.2")
    new = out / "__saturate" / "completions"
    new.mkdir(parents=True)
    (new / "shard-0.done").write_bytes(b"done")
    # 2 of the 100 prior rows have only an error record: the resume skipped them
    resumed = {
        **STATS_0,
        "rows_done_prior": 100,
        "rows_errored_prior": 2,
        "rows_processed": 50,
        "rows_failed": 1,
    }
    (new / "stats-0.json").write_text(json.dumps(resumed))

    doc = status(str(out), include_jobs=False)
    assert doc["progress"]["rows_ok"] == 148  # 0.3 file: 100 - 2 errored + 50
    assert doc["progress"]["rows_failed"] == 3  # 1 this run + 2 skipped error rows
    assert [s["rank"] for s in doc["shards"] if s["done"]] == [0]  # one marker, not two


def test_stale_root_marker_does_not_mark_a_resumed_shard_done(tmp_path):
    """A 0.2 output resumed under 0.3: shard 0 has a root marker from the
    earlier run and 0.3 telemetry from the running one, but no 0.3 marker."""
    out = make_run(tmp_path, layout="0.2")
    telem = out / "__saturate" / "telemetry"
    telem.mkdir(parents=True)
    (telem / "telemetry-shard0-300-a1b2c3.jsonl").write_text(json.dumps({"t": 0, "ok": 5}) + "\n")

    doc = status(str(out), include_jobs=False)
    assert not {s["rank"]: s for s in doc["shards"]}[0]["done"]


def test_telemetry_epoch_tie_prefers_sidecar_dir(tmp_path):
    out = tmp_path / "run"
    (out / "__saturate" / "telemetry").mkdir(parents=True)
    (out / "telemetry-shard0-100.jsonl").write_text(json.dumps({"t": 0, "ok": 1}) + "\n")
    (out / "__saturate" / "telemetry" / "telemetry-shard0-100-a1b2c3.jsonl").write_text(
        json.dumps({"t": 0, "ok": 2}) + "\n"
    )

    doc = status(str(out), include_jobs=False)
    assert [t["ok"] for t in doc["telemetry"]["0"]] == [2]  # the __saturate/ file
    assert doc["progress"]["rows_ok"] == 3  # Sum(ok) still counts both files

"""status(output_uri) -> the run progress document.

One canonical, versioned JSON document is the protocol for every consumer:
`ip status --json` prints it, `--watch` re-emits it, a UI renders the same
shape. Ported from the console POC's dashboard reads (analyze/renderRun),
which the pumpjack CONTRACT doubles as the storage side of.

Two channels merge here:
- exact/storage: part files + completions/stats-{n}.json (exact counts) with
  telemetry Sum(ok) as the approximate fallback — approximate because ticks
  sample a counter and the tail goes uncounted (measured: a 3,125-row run
  reporting 3,120 from telemetry alone);
- live: job stages from the Jobs API via run.json's job ids — determinate
  "k of n running" before any telemetry flushes (the G3 workaround; a
  structured per-tick stderr line from pumpjack would upgrade this for free).

The denominator (expected_rows) and the rows->items factor (items_per_row)
come from run.json — launch intent, gaps G1/G4 closed at this layer.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict

import fsspec

# part names carry no semantics (CONTRACT); match any dash-joined tail so the
# 2026-07-29 saturate filename change (part-{epoch}-{seq}-{uuid8}) and any
# future segment additions don't zero the count (bug found by live run 9fee4f3d)
PART_RE = re.compile(r"^part-[0-9a-f-]+\.parquet$")
MANIFEST_RE = re.compile(r"^_manifest/ids-(part-[0-9a-f-]+\.parquet)$")
MARKER_RE = re.compile(r"^completions/shard-(\d+)\.done$")
STATS_RE = re.compile(r"^completions/stats-(\d+)\.json$")
TELEMETRY_RE = re.compile(r"^telemetry-shard(\d+)-(\d+)\.jsonl$")

TICK_WINDOW = 120  # ticks kept per shard in the document (~4 min at 2s/tick)


def _read_json(fs, path: str):
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def status(
    output_uri: str,
    include_jobs: bool = True,
    token: str | None = None,
) -> dict:
    # token threads into the filesystem too — hosted contexts (Spaces) have no
    # ambient token, and the storage channel must see private repos (same bug
    # family as the launcher's run.json write; found live twice on 2026-07-30)
    fs, root = fsspec.core.url_to_fs(output_uri, **({"token": token} if token else {}))
    root = root.rstrip("/")
    try:
        found = fs.find(root)
    except FileNotFoundError:
        found = []
    rel = [f[len(root) + 1 :] for f in found]

    parts = sorted(p for p in rel if PART_RE.match(p))
    manifests = {m.group(1) for p in rel if (m := MANIFEST_RE.match(p))}
    orphans = [p for p in parts if p not in manifests]  # crash window <= flush_every rows
    markers = sorted(int(m.group(1)) for p in rel if (m := MARKER_RE.match(p)))
    stats_paths = {int(m.group(1)): p for p in rel if (m := STATS_RE.match(p))}
    stats = {
        rank: s
        for rank, p in sorted(stats_paths.items())
        if (s := _read_json(fs, f"{root}/{p}")) is not None
    }

    # telemetry: all files count toward Sum(ok) (resumes append new files);
    # the newest file per shard feeds the tick window
    telem_files = defaultdict(list)  # rank -> [(ts, relpath)]
    for p in rel:
        if m := TELEMETRY_RE.match(p):
            telem_files[int(m.group(1))].append((int(m.group(2)), p))
    ok_sum = 0
    ticks_by_rank: dict[str, list[dict]] = {}
    for rank, files in sorted(telem_files.items()):
        files.sort()
        for i, (_ts, p) in enumerate(files):
            ticks = []
            try:
                with fs.open(f"{root}/{p}", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ticks.append(json.loads(line))
            except Exception:
                continue
            ok_sum += sum(t.get("ok", 0) for t in ticks)
            if i == len(files) - 1:  # newest file: the chart feed
                ticks_by_rank[str(rank)] = ticks[-TICK_WINDOW:]

    run_json = _read_json(fs, f"{root}/run.json")

    # world: exact from any stats file, else inferred from traces (gap G2 —
    # a dead-on-arrival shard leaves no trace, hence the flag)
    world = None
    world_inferred = False
    if stats:
        world = next(iter(stats.values())).get("world")
    if world is None:
        seen = set(markers) | set(telem_files) | set(stats)
        if run_json:
            world = run_json.get("world")
        elif seen:
            world, world_inferred = max(seen) + 1, True

    exact = bool(stats)
    rows_ok = (
        sum(s.get("rows_done_prior", 0) + s.get("rows_processed", 0) for s in stats.values())
        if exact
        else ok_sum
    )
    rows_failed = sum(s.get("rows_failed", 0) for s in stats.values()) if exact else None

    expected_rows = run_json.get("expected_rows") if run_json else None
    items_per_row = run_json.get("items_per_row") if run_json else None
    progress = {
        "rows_ok": rows_ok,
        "rows_failed": rows_failed,
        "expected_rows": expected_rows,
        "pct": min(1.0, rows_ok / expected_rows) if expected_rows else None,
        "exact": exact,  # False -> telemetry Sum(ok), undercounts the tail
        "items_per_row": items_per_row,
        "items_ok": rows_ok * items_per_row if items_per_row else None,
        "expected_items": run_json.get("expected_items") if run_json else None,
        "has_run_json": run_json is not None,
    }

    jobs = []
    if include_jobs and run_json and run_json.get("jobs"):
        jobs = _job_stages(run_json["jobs"], token=token)

    shard_ranks = sorted(
        set(markers) | set(stats) | set(telem_files) | set(range(world or 0))
    )
    shards = [
        {"rank": r, "done": r in markers, "stats": stats.get(r)} for r in shard_ranks
    ]

    return {
        "version": 1,
        "run_id": run_json.get("run_id") if run_json else None,
        "output": output_uri,
        "spec": run_json.get("spec") if run_json else None,
        "progress": progress,
        "jobs": jobs,
        "shards": shards,
        "storage": {
            "parts": len(parts),
            "orphans": len(orphans),
            "world": world,
            "world_inferred": world_inferred,
        },
        "telemetry": ticks_by_rank,
    }


def _job_stages(jobs: list[dict], token: str | None) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    out = []
    for j in jobs:
        stage = None
        try:
            info = api.inspect_job(job_id=j["id"], namespace=j.get("owner"))
            stage = getattr(getattr(info, "status", None), "stage", None)
        except Exception:
            pass
        out.append({"rank": j.get("rank"), "id": j["id"], "stage": stage})
    return out


def prog() -> str:
    """The command name the user actually typed: `hf pipe` when running as the
    extension (argv[0] is the hf-pipe console script), else `jobpipe`."""
    import sys

    base = pathlib.Path(sys.argv[0]).name
    return "hf pipe" if base.startswith("hf-pipe") else "jobpipe"


def render_human(doc: dict) -> str:
    """Compact terminal view of the progress document."""
    lines = []
    p = doc["progress"]
    marker = "exact" if p["exact"] else "~approx"
    if p["expected_rows"]:
        pct = f"{100 * (p['pct'] or 0):.1f}%"
        lines.append(
            f"progress {pct}  {p['rows_ok']}/{p['expected_rows']} rows ({marker})"
            + (f"  ≈{p['items_ok']:,} items" if p["items_ok"] else "")
        )
    else:
        why = ("dataset size unknown at launch" if p.get("has_run_json")
               else "no run.json")
        lines.append(f"progress {p['rows_ok']} rows ({marker}) — no denominator ({why})")
    if p["rows_failed"]:
        lines.append(f"failed rows: {p['rows_failed']}")
    stage_by_rank = {j["rank"]: j["stage"] for j in doc["jobs"]}
    for s in doc["shards"]:
        light = "✓" if s["done"] else "●" if stage_by_rank.get(s["rank"]) == "RUNNING" else "○"
        extra = ""
        if s["stats"]:
            rows = s["stats"].get("rows_processed", 0)
            tok_s = s["stats"].get("tokens_per_sec", 0)
            extra = f"  {rows} rows, {tok_s:,.0f} tok/s"
        elif stage_by_rank.get(s["rank"]):
            extra = f"  {stage_by_rank[s['rank']]}"
        lines.append(f"  shard {s['rank']} {light}{extra}")
    st = doc["storage"]
    world = f"{st['world']}{' (inferred)' if st['world_inferred'] else ''}"
    live = [j for j in doc["jobs"] if j.get("stage") not in ("COMPLETED", None)]
    if live:
        j = live[0]
        lines.append(f"logs: {prog()} logs '{doc['output']}'"
                     + (f"  (rank {j['rank']}: job {j['id']})" if j.get("id") else ""))
    lines.append(f"storage: {st['parts']} parts, {st['orphans']} orphans, world={world}")
    return "\n".join(lines)

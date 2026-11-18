"""Task verbs: the Python layer.

    from jobpipe import embed

    plan = embed("biglam/europeana_newspapers", keep=["language"], limit=200_000,
                 world=4, launch=False)          # resolve only: inspect before spending
    print(plan.describe()); print(*plan.commands(), sep="\\n")
    run = embed("biglam/europeana_newspapers", keep=["language"], limit=200_000, world=4)
    run.status()

A verb resolves the request into a Plan (a pinned recipe plus its arguments)
and, unless launch=False, launches it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from jobpipe.resolve import Plan, resolve_embed


@dataclass
class Run:
    """A launched run: the run.json document plus storage-backed status."""

    run_json: dict
    plan: Plan

    @property
    def run_id(self) -> str:
        return self.run_json["run_id"]

    @property
    def jobs(self) -> list[dict]:
        return self.run_json.get("jobs", [])

    @property
    def output(self) -> str:
        return self.plan.output

    def status(self, include_jobs: bool = True, token: str | None = None) -> dict:
        from jobpipe.status import status

        return status(self.plan.output, include_jobs=include_jobs, token=token)


def embed(source: str, *, model: str | None = None, column: str | None = None,
          id_column: str | None = None, keep: list[str] | None = None,
          max_seq_len: int | None = None, prefix: str | None = None,
          config: str | None = None, split: str | None = None,
          output: str | None = None, world: int | None = None,
          limit: int | None = None, flavor: str | None = None,
          timeout: str | None = None, recipe_url: str | None = None,
          launch: bool = True, token: str | None = None,
          namespace: str | None = None, verbose: bool = True) -> Run | Plan:
    """Embed a text column of a Hub dataset, a bucket glob (hf://buckets/me/b/*.parquet;
    also .jsonl, .csv, .txt, .md) or a local directory (uploaded to a bucket at launch).
    Everything but the source is optional; the filled values and where each came from
    are printed before anything launches."""
    plan = resolve_embed(
        source, model=model, column=column, id_column=id_column, keep=keep,
        max_seq_len=max_seq_len, prefix=prefix, config=config, split=split,
        output=output, world=world, limit=limit, flavor=flavor, timeout=timeout,
        recipe_url=recipe_url, token=token,
    )
    if verbose:
        print(f"resolved embed plan:\n{plan.describe()}", file=sys.stderr)
    if not launch:
        return plan
    from jobpipe.launcher import launch as _launch

    return Run(run_json=_launch(plan, token=token, namespace=namespace), plan=plan)

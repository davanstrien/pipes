"""Recipes: the pinned uv scripts this layer launches. It generates no code.

A recipe is a single-file uv script in the uv-scripts collection. Its PEP 723
header carries its dependencies, and its `[tool.hf-jobs]` table carries the
Job's image, flavor, timeout and secrets. A launch pins the recipe to a commit,
so `run.json` names the exact code that ran.
"""

from __future__ import annotations

import tomllib
import urllib.request
from dataclasses import dataclass, field

HF_JOBS_DEFAULT_TIMEOUT = "2h"


@dataclass(frozen=True)
class Recipe:
    task: str
    repo_id: str  # Hub dataset repo the uv-scripts sync publishes to
    path: str  # file path inside that repo
    source: str  # where it is maintained (GitHub is the source of truth)


RECIPES = {
    "embed": Recipe(
        task="embed",
        repo_id="uv-scripts/embeddings",
        path="generate-embeddings-saturate.py",
        source="https://github.com/davanstrien/uv-scripts/blob/main/embeddings/"
        "generate-embeddings-saturate.py",
    ),
}


def pinned_url(recipe: Recipe, token: str | None = None) -> str:
    """The recipe's raw URL at the current commit of its Hub repo."""
    from huggingface_hub import HfApi

    sha = HfApi(token=token).dataset_info(recipe.repo_id).sha
    return f"https://huggingface.co/datasets/{recipe.repo_id}/raw/{sha}/{recipe.path}"


@dataclass
class JobsHeader:
    """The `[tool.hf-jobs]` values this layer uses. `HfApi.run_uv_job` does not read
    the header itself (huggingface_hub 2.4: only the `hf jobs uv run` CLI does)."""

    image: str | None = None
    flavor: str | None = None
    timeout: str | None = None
    secrets: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def parse_jobs_header(text: str) -> JobsHeader:
    """Read `[tool.hf-jobs]` from a script's `# /// script` block (PEP 723)."""
    lines = text.splitlines()
    try:
        start = lines.index("# /// script")
        end = lines.index("# ///", start + 1)
    except ValueError:
        return JobsHeader()
    block = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                      for line in lines[start + 1:end])
    table = tomllib.loads(block).get("tool", {}).get("hf-jobs", {})
    return JobsHeader(
        image=table.get("image"),
        flavor=table.get("flavor"),
        timeout=table.get("timeout"),
        secrets=list(table.get("secrets", [])),
        env={key: str(value) for key, value in table.get("env", {}).items()},
    )


def fetch_text(url: str, token: str | None = None) -> str:
    request = urllib.request.Request(url)
    if token and "huggingface.co" in url:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


class RecipeTooOld(RuntimeError):
    """The pinned recipe does not accept a flag this layer passes."""


def check_flags(recipe_text: str, args: list[str], url: str) -> None:
    """Fail before launching if the recipe has no add_argument for a flag we pass.

    The recipe and this layer live in different repos; a recipe pinned to a commit
    from before a flag existed would otherwise fail inside the first Job."""
    flags = sorted({arg for arg in args if arg.startswith("--")})
    missing = [flag for flag in flags if f'"{flag}"' not in recipe_text]
    if missing:
        raise RecipeTooOld(
            f"the recipe at {url} does not accept {', '.join(missing)}. "
            "Its Hub copy may predate these flags; pass --recipe-url with a newer copy."
        )

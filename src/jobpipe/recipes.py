"""Recipes: the uv scripts this layer launches. It generates no code.

A recipe is a single-file uv script. jobpipe ships its own copy of each (in
jobpipe/scripts/, copied from uv-scripts), so the two repos change independently.
The script's PEP 723 header carries its dependencies, and its `[tool.hf-jobs]`
table the Job's image, flavor, timeout and secrets. run.json records the sha256
of the exact script that ran.
"""

from __future__ import annotations

import pathlib
import tomllib
import urllib.request
from dataclasses import dataclass, field

HF_JOBS_DEFAULT_TIMEOUT = "2h"


@dataclass(frozen=True)
class Recipe:
    task: str
    script: str  # file name under jobpipe/scripts/
    upstream: str  # where the original is maintained


RECIPES = {
    "embed": Recipe(
        task="embed",
        script="embed.py",
        upstream="https://github.com/davanstrien/uv-scripts/blob/main/embeddings/"
        "generate-embeddings-saturate.py",
    ),
    "ocr": Recipe(
        task="ocr",
        script="ocr.py",
        upstream="https://github.com/davanstrien/uv-scripts/blob/main/ocr/lighton-ocr3-saturate.py",
    ),
}


def script_path(recipe: Recipe) -> str:
    """Local path of jobpipe's copy of the recipe (run_uv_job uploads it per launch)."""
    return str(pathlib.Path(__file__).parent / "scripts" / recipe.script)


def read_script(location: str, token: str | None = None) -> str:
    """The script's text, from a local path or a URL."""
    if location.startswith(("http://", "https://")):
        return fetch_text(location, token=token)
    return pathlib.Path(location).read_text()


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


def _is_hub_host(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    on_hub = host == "huggingface.co" or host.endswith(".huggingface.co")
    return parsed.scheme == "https" and on_hub


def fetch_text(url: str, token: str | None = None) -> str:
    request = urllib.request.Request(url)
    if token and _is_hub_host(url):  # never send the token to another host
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
            "It may be an older copy than this jobpipe expects."
        )

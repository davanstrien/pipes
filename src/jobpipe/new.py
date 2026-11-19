"""`hf pipe new NAME`: a new saturate recipe from the template.

The template (templates/recipe.py) is a whole single-file uv script. Its plumbing (inputs,
output, shards, kept columns, resume, the stats line) is the same in every recipe made from
it; the part between the TASK markers is each recipe's own. A recipe is generated once and
then edited between the markers. `plumbing_of` lets a test check that a recipe's plumbing
still matches the template.
"""

from __future__ import annotations

import pathlib
import re

TEMPLATE = pathlib.Path(__file__).parent / "templates" / "recipe.py"
START = "# ==== TASK: settings"
END = "# ==== END TASK"
PLACEHOLDER = "__TASK__"


def check_name(name: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ValueError(f"a recipe name is lower case letters, digits and '-': {name!r}")


def render(name: str, task_block: str | None = None) -> str:
    """The template with NAME filled in; task_block replaces the TASK part when given."""
    check_name(name)
    text = TEMPLATE.read_text().replace(PLACEHOLDER, name)
    if task_block is None:
        return text
    before, _, end_and_after = split(text)
    return before + task_block + end_and_after


def split(text: str) -> tuple[str, str, str]:
    """(before the TASK block, the TASK block, END marker and after)."""
    start = text.index(START)
    end = text.index(END)
    return text[:start], text[start:end], text[end:]


def plumbing_of(text: str, name: str) -> str:
    """The recipe with its TASK block taken out and its name put back to the placeholder."""
    before, _, after = split(text)
    return (before + after).replace(name, PLACEHOLDER)

"""jobpipe's copies of the recipes: still parse, still carry a Jobs header, still
accept every flag the resolver passes."""

import ast
import pathlib

from jobpipe.recipes import RECIPES, check_flags, parse_jobs_header, script_path
from jobpipe.resolve import resolve_embed


def test_copies_parse_and_have_a_jobs_header():
    for recipe in RECIPES.values():
        text = pathlib.Path(script_path(recipe)).read_text()
        ast.parse(text)
        header = parse_jobs_header(text)
        assert header.image and header.flavor and "HF_TOKEN" in header.secrets


def test_embed_copy_accepts_every_flag_we_pass(offline, monkeypatch, tmp_path):
    import jobpipe.resolve as res

    monkeypatch.setattr(res, "script_path", script_path)  # the real copy, not the stub
    text = pathlib.Path(script_path(RECIPES["embed"])).read_text()
    (tmp_path / "a.md").write_text("x")
    plans = [
        resolve_embed("org/ds", keep=["a"], prefix="p: ", limit=5, world=2),
        resolve_embed("org/ds", keep_text=True),
        resolve_embed("hf://buckets/me/b/*.parquet", column="body", id_column="doc"),
        resolve_embed(str(tmp_path)),
    ]
    for plan in plans:
        check_flags(text, plan.rank_args(0), plan.recipe_url)

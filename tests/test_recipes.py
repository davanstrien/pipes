from jobpipe.recipes import parse_jobs_header

SCRIPT = '''# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.1",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# env = { VLLM_USE_RUST_FRONTEND = 1 }
# ///
"""docstring"""
'''


def test_parse_jobs_header():
    header = parse_jobs_header(SCRIPT)
    assert header.image == "vllm/vllm-openai:v0.30.0"
    assert (header.flavor, header.timeout) == ("a10g-small", "2h")
    assert header.secrets == ["HF_TOKEN"] and header.env == {"VLLM_USE_RUST_FRONTEND": "1"}


def test_no_header():
    header = parse_jobs_header("print('hi')\n")
    assert header.image is None and header.secrets == []


def test_check_flags_names_the_missing_ones():
    import pytest

    from jobpipe.recipes import RecipeTooOld, check_flags

    text = 'parser.add_argument("--model")\nparser.add_argument("--keep", nargs="+")\n'
    check_flags(text, ["org/ds", "out", "--model", "m", "--keep", "a"], "url")
    with pytest.raises(RecipeTooOld, match="--no-text, --shard"):
        check_flags(text, ["--model", "m", "--no-text", "--shard", "0/1"], "url")

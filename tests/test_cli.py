"""CLI: zero-config verb path + --dump eject."""

import jobpipe.introspect as intro
import jobpipe.resolve as res
from jobpipe.cli import main


def _stub(monkeypatch):
    monkeypatch.setattr(intro, "num_examples", lambda *a, **k: 10_000)
    monkeypatch.setattr(intro, "dataset_layout",
                        lambda *a, **k: [{"config": "default", "split": "train"}])
    monkeypatch.setattr(intro, "string_columns",
                        lambda *a, **k: [{"name": "text", "avg_len": 500.0}])
    monkeypatch.setattr(intro, "model_metadata",
                        lambda *a, **k: {"params": 568_000_000, "model_type": "xlm-roberta"})
    monkeypatch.setattr(res, "_whoami", lambda token: "tester")


def test_zero_config_compile_only(monkeypatch, capsys):
    _stub(monkeypatch)
    assert main(["embed", "org/ds", "--compile-only"]) == 0
    out, err = capsys.readouterr()
    # driver on stdout: TEI arm (encoder), auto-named output, PEP 723 header
    assert "# /// script" in out and "[tool.hf-jobs]" in out
    assert "text-embeddings-router" in out
    assert "hf://datasets/tester/ds-embeddings/data" in out
    # provenance receipts on stderr
    assert "resolved embeddings spec" in err
    assert "[default]" in err and "[detected]" in err


def test_dump_writes_editable_uv_script(monkeypatch, capsys, tmp_path):
    _stub(monkeypatch)
    target = tmp_path / "my-embed.py"
    assert main(["embed", "org/ds", "--dump", str(target)]) == 0
    text = target.read_text()
    assert text.startswith("# /// script")
    assert 'saturate[hf]' in text and "[tool.hf-jobs]" in text
    _, err = capsys.readouterr()
    assert "hf jobs uv run" in err and str(target) in err


def test_generate_still_requires_model(monkeypatch, capsys):
    _stub(monkeypatch)
    try:
        main(["generate", "org/ds", "--compile-only"])
        raise AssertionError("argparse should have exited")
    except SystemExit as e:
        assert e.code == 2

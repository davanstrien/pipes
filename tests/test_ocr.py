"""resolve_ocr(): the second verb on the shared machinery."""

import pytest

import jobpipe.introspect as intro
from jobpipe.resolve import resolve_ocr


def arg(plan, flag):
    return plan.args[plan.args.index(flag) + 1]


def test_bucket_glob_of_images(offline):
    plan = resolve_ocr("hf://buckets/me/scans/books/**/*.jpg", limit=50, world=2)
    assert plan.task == "ocr" and plan.script_name == "ocr"
    assert plan.args[:2] == ["hf://buckets/me/scans/books/**/*.jpg",
                             "hf://buckets/tester/jobpipe-outputs/scans-LightOnOCR-3-0-8B"]
    assert arg(plan, "--model") == "lightonai/LightOnOCR-3-0.8B" and arg(plan, "--mode") == "plain"
    assert "--image-column" not in plan.args and "--config" not in plan.args
    assert plan.values["id_column"] == "(file path)" and plan.expected_items == 50
    assert plan.rank_args(1)[-2:] == ["--shard", "1/2"]
    assert plan.commands()[0].count("ocr.py") == 1


def test_refusals(offline):
    with pytest.raises(ValueError, match="PDFs"):
        resolve_ocr("hf://buckets/me/scans/*.pdf")
    with pytest.raises(ValueError, match="page images"):
        resolve_ocr("hf://buckets/me/scans/*.parquet")
    with pytest.raises(ValueError, match="no OCR script"):
        resolve_ocr("hf://buckets/me/scans/*.png", model="zai-org/GLM-OCR")
    with pytest.raises(ValueError, match="path"):
        resolve_ocr("hf://buckets/me/scans/*.png", id_column="doc")


def test_dataset_with_an_image_column(offline):
    offline.setattr(intro, "image_columns", lambda *a, **k: ["page", "thumb"])
    plan = resolve_ocr("org/scans")
    assert arg(plan, "--image-column") == "page" and plan.provenance["image_column"] == "detected"
    assert arg(plan, "--id-column") == "id"  # conftest's dataset has an 'id' string column
    assert (arg(plan, "--config"), arg(plan, "--split")) == ("default", "train")
    assert plan.expected_items == 10_000


def test_dataset_without_an_image_column(offline):
    offline.setattr(intro, "image_columns", lambda *a, **k: [])
    with pytest.raises(ValueError, match="--image-column"):
        resolve_ocr("org/texts")


def test_local_folder_of_images(offline, tmp_path):
    for name in ("p1.jpg", "p2.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    plan = resolve_ocr(str(tmp_path))
    assert plan.uploads[0][2] == "*.jpg" and plan.args[0].endswith("/**/*.jpg")


def test_fingerprint_tracks_mode_and_model(offline):
    base = resolve_ocr("hf://buckets/me/s/*.png").fingerprint
    grounding = resolve_ocr("hf://buckets/me/s/*.png", mode="grounding").fingerprint
    assert grounding["sha256"] != base["sha256"]
    assert base["recipe"] == "ocr.py"

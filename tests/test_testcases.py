"""Integrity checks for the labelled test-case sets in data/testcases/ (no models needed)."""

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "testcases"
LEVELS = {"none", "medium", "full"}


def _rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


TEXT = _rows("text_cases.jsonl")
IMAGES = _rows("image_cases.jsonl")


@pytest.mark.parametrize("rows", [TEXT, IMAGES], ids=["text", "image"])
def test_25_cases_spread_over_every_privacy_level(rows):
    assert len(rows) == 25
    assert len({r["id"] for r in rows}) == 25
    counts = Counter(r["privacy_level"] for r in rows)
    assert set(counts) == LEVELS
    assert min(counts.values()) >= 7


def test_text_labels_are_consistent():
    for r in TEXT:
        for item in r["sensitive"] + r["keep"]:
            assert item.lower() in r["prompt"].lower(), (r["id"], item)
        assert (r["privacy_level"] == "none") == (not r["sensitive"]), r["id"]
        assert r["edge_case"].strip()


def test_image_files_and_labels_are_consistent():
    for r in IMAGES:
        path = ROOT / r["file"]
        assert path.is_file(), r["file"]
        with Image.open(path) as im:
            w, h = im.size
        assert r["faces"] == len(r["face_boxes"])
        for left, top, right, bottom in r["face_boxes"]:
            assert 0 <= left < right <= w and 0 <= top < bottom <= h, r["id"]
        has_private = r["faces"] or r["sensitive_text"] or r["metadata"]
        assert (r["privacy_level"] == "none") == (not has_private), r["id"]


def test_image_metadata_labels_match_the_files():
    from sanitizer.image import strip_metadata

    for r in IMAGES:
        with Image.open(ROOT / r["file"]) as im:
            im.load()
            _, removed = strip_metadata(im)
        assert sorted(removed) == sorted(r["metadata"]), r["id"]

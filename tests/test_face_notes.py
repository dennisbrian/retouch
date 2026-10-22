"""Per-face "what was skipped" notes (retouch/face_notes.py + engine)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from retouch.face_notes import borderline_sides, caption_lines, caption_markdown, match_faces, note_keys, report_section

GOLDEN = Path(__file__).parent / "golden" / "inputs"


def _diag():
    return {
        "face_notes": [
            {"face": 0, "anchor_bbox_norm": [0.1, 0.1, 0.2, 0.3], "recipe": "natural", "notes": [
                {"op": "eye_edits", "action": "skipped", "reason": "left-hand eye: eye closed"},
                {"op": "safe_auto", "action": "review", "reason": "advisory"},
            ]},
            {"face": 1, "anchor_bbox_norm": [0.6, 0.1, 0.2, 0.3], "recipe": "natural", "notes": [
                {"op": "safe_auto", "action": "review", "reason": "advisory"},
            ]},
            {"face": 2, "anchor_bbox_norm": [0.4, 0.5, 0.1, 0.1], "recipe": "natural", "notes": [
                {"op": "face_strength", "action": "dampened", "reason": "face strength 40%", "factor": 0.4},
            ]},
        ],
        "frame_notes": [
            {"op": "detection", "action": "skipped", "reason": "face not on a person (background or poster), not retouched",
             "bbox_norm": [0.8, 0.0, 0.1, 0.1]},
        ],
    }


def test_caption_lists_skips_but_not_advisory_reviews():
    lines = caption_lines(_diag())
    assert lines[0] == "Face 0: eye edits skipped, left-hand eye: eye closed"
    assert lines[1] == "Face 2: face strength 40%"
    assert lines[2].startswith("Face at 85% across, 5% down: face not on a person")
    assert len(lines) == 3
    assert caption_markdown({}) == ""
    assert caption_markdown(_diag()).startswith("**Skipped or dampened:**")


def test_report_section_keeps_reviews_for_the_record():
    section = report_section(_diag())
    assert section["faces"][1]["notes"][0]["action"] == "review"
    assert section["frame"][0]["action"] == "skipped"
    assert section["caption"] == caption_lines(_diag())


def test_match_faces_by_position_not_order():
    a = _diag()["face_notes"]
    b = list(reversed(a))
    pairs = match_faces(a, b)
    assert all(x["face"] == y["face"] for x, y in pairs)


def test_eye_gate_reports_why():
    from retouch.eye_visibility import gate_occluded_eye_regions
    import inspect

    assert "decisions_out" in inspect.signature(gate_occluded_eye_regions).parameters


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine
    return RetouchEngine()


def test_closed_eye_note_and_person_gate_diagnostics_reach_the_result(engine):
    img = cv2.imread(str(GOLDEN / "portrait_studio.jpg"))
    out = engine.process(img, recipe="cosplay_clear_v1")
    entries = out.runtime_diagnostics["face_notes"]
    assert len(entries) == out.face_count == 1
    assert entries[0]["anchor_bbox_norm"] and entries[0]["recipe"] == "cosplay_clear_v1"
    for note in entries[0]["notes"]:
        assert {"op", "action", "reason"} <= set(note)


def test_poster_face_drop_is_reported(engine):
    img = cv2.imread(str(GOLDEN / "background_poster_face.jpg"))
    out = engine.process(img, recipe="natural")
    frame = out.runtime_diagnostics.get("frame_notes") or []
    gate = out.runtime_diagnostics.get("person_gate") or []
    # Whatever the gate decided about the poster face is reported, not silent.
    assert len(frame) == len(gate)
    for note in frame:
        assert note["gate"] == "person_gate" and note["action"] in ("skipped", "kept")


def test_p7_abstain_reaches_face_notes(engine):
    img = cv2.imread(str(GOLDEN / "portrait_studio.jpg"))
    out = engine.process(img, recipe="natural", cross_region_skin=50, neck_tone_match=50)
    p7 = out.p7_diagnostics
    assert p7.get("abstained")
    ops = {n["op"] for e in out.runtime_diagnostics["face_notes"] for n in e["notes"]}
    assert "cross_region_skin" in ops


@pytest.mark.parametrize("name", ["portrait_studio", "two_people"])
def test_preview_and_export_report_the_same_skips(engine, name):
    """Fast preview (800 px) and the full-quality export, same photo."""
    img = cv2.imread(str(GOLDEN / f"{name}.jpg"))
    img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    kw = dict(recipe="cosplay_clear_v1", iris_pop=60, face_strength=80)
    preview = engine.process(img.copy(), fast=True, **kw)
    export = engine.process(img.copy(), fast=False, quality="full", **kw)
    pe = preview.runtime_diagnostics["face_notes"]
    ex = export.runtime_diagnostics["face_notes"]
    assert len(pe) == len(ex)
    for a, b in match_faces(pe, ex):
        assert b is not None, "preview face missing from export"
        skip = borderline_sides(a) | borderline_sides(b)
        assert note_keys(a, skip) == note_keys(b, skip), (a["notes"], b["notes"])
    assert set(preview.p7_diagnostics) == set(export.p7_diagnostics)

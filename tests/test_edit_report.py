"""Tests for the per-photo "what was changed" report (retouch/edit_report.py)."""

import json

import numpy as np
import pytest

from retouch.edit_report import (
    CATEGORIES,
    active_edits,
    build_edit_report,
    credentials_note,
    pixel_change,
    report_path_for,
    sha256_file,
    write_edit_report,
)
from retouch.engine import ProcessingContext
from retouch.params import PROCESSING_PARAMS


class TestActiveEdits:
    def test_neutral_values_are_not_edits(self):
        edits = active_edits({
            "smooth": 0,
            "body_reshape_hip_width": 50.0,  # 50 = unchanged
            "white_balance_kelvin": 6500,
            "color_grade": "none",
            "lut": "none",
            "nose_blush": False,
            "ai_sr_scale": 1,
        })
        assert edits == {}

    def test_modifiers_never_count_on_their_own(self):
        edits = active_edits({
            "mid_reduction": 0.9,
            "sharpen_radius": 3.0,
            "film_gamma": 2.0,
            "mv2_eyeliner_color": "red",
            "grade_intensity": 0.5,
        })
        assert edits == {}

    def test_film_enable_is_an_edit(self):
        assert active_edits({"film_enable": True}) == {"colour": {"film_enable": True}}

    @pytest.mark.parametrize(
        "name,value,category",
        [
            ("slimming", 40, "shape"),
            ("reshape_jaw_width", -0.2, "shape"),
            ("body_reshape_hip_width", 40.0, "shape"),
            ("auto_body_reshape", True, "shape"),
            ("smooth", 30, "skin"),
            ("blemish", 30, "skin"),
            ("body_smooth", 20, "skin"),
            ("lip_tint", "cosplay", "makeup"),
            ("mv2_eyeliner", 30, "makeup"),
            ("teeth_whiten", 5, "eyes_teeth"),
            ("lens_glare", 50, "eyes_teeth"),
            ("hair_enhance", 5, "hair_costume"),
            ("relight", 20, "light"),
            ("background_blur", 10, "background"),
            ("ai_denoise", 50, "ai"),
            ("color_grade", "cosplay", "colour"),
            ("white_balance_kelvin", 5200, "colour"),
        ],
    )
    def test_categories(self, name, value, category):
        assert active_edits({name: value}) == {category: {name: value}}

    def test_every_registered_setting_classifies(self):
        # Pushing each setting off its neutral value must never raise and
        # must land in a known category (or be a modifier and drop out).
        known = {cat for cat, _ in CATEGORIES}
        for spec in PROCESSING_PARAMS:
            if isinstance(spec.default, str) or spec.choices:
                value = "custom-value"
            elif isinstance(spec.default, bool):
                value = not spec.default
            else:
                value = 37
            grouped = active_edits({spec.name: value})
            assert set(grouped) <= known, spec.name

    def test_accepts_processing_context(self):
        ctx = ProcessingContext()
        ctx.slimming = 25.0
        edits = active_edits(ctx)
        assert edits["shape"] == {"slimming": 25}

    def test_float_noise_is_rounded(self):
        edits = active_edits({"smooth": 55.00000000000001})
        assert edits["skin"]["smooth"] == 55


class TestBuildReport:
    def test_shape_flag_and_summary(self):
        report = build_edit_report({"slimming": 50, "smooth": 30}, recipe="x", face_count=1)
        assert report["shape_changed"] is True
        assert report["summary"][0].startswith("Face or body shape was changed")
        assert report["generative_fill"] is False
        assert report["recipe"] == "x"

    def test_no_shape_change(self):
        report = build_edit_report({"smooth": 30}, face_count=1)
        assert report["shape_changed"] is False
        assert report["summary"][0] == "No face or body shape change."

    def test_no_face_moves_face_edits_to_not_applied(self):
        report = build_edit_report(
            {"slimming": 50, "smooth": 30, "lip_enhance": 5, "body_smooth": 20,
             "color_grade": "cosplay"},
            face_count=0,
        )
        assert report["shape_changed"] is False
        assert set(report["edits"]) == {"skin", "colour"}
        assert report["edits"]["skin"]["settings"] == {"body_smooth": 20}
        assert report["not_applied_no_face"] == {"slimming": 50, "smooth": 30, "lip_enhance": 5}

    def test_unknown_face_count_keeps_everything(self):
        report = build_edit_report({"smooth": 30})
        assert "not_applied_no_face" not in report
        assert "skin" in report["edits"]

    def test_global_only_drops_face_categories(self):
        report = build_edit_report({"smooth": 30, "slimming": 50, "saturation": 10},
                                   global_only=True)
        assert set(report["edits"]) == {"colour"}

    def test_ai_use_is_listed(self):
        report = build_edit_report({"ai_denoise": 40}, face_count=1)
        assert report["ai_used"] and "NAFNet" in report["ai_used"][0]

    def test_recipe_falls_back_to_active_recipe(self):
        assert build_edit_report({"active_recipe": "natural"})["recipe"] == "natural"

    def test_hashes_and_pixels(self, tmp_path):
        src = tmp_path / "a.jpg"
        out = tmp_path / "b.jpg"
        src.write_bytes(b"source")
        out.write_bytes(b"output")
        before = np.zeros((40, 60, 3), np.uint8)
        after = before.copy()
        after[:20] = 100
        report = build_edit_report({}, source_path=src, output_path=out,
                                   before=before, after=after)
        assert report["source"]["sha256"] == sha256_file(src)
        assert report["output"]["sha256"] == sha256_file(out)
        assert report["pixels"]["changed_pct"] == pytest.approx(50.0, abs=1.0)

    def test_output_hash_left_out_for_embedding(self, tmp_path):
        out = tmp_path / "b.jpg"
        out.write_bytes(b"output")
        report = build_edit_report({}, output_path=out, include_output_hash=False)
        assert report["output"] == {"file": "b.jpg"}

    def test_report_is_json_serialisable_from_real_context(self):
        ctx = ProcessingContext()
        json.dumps(build_edit_report(ctx, face_count=1))


class TestPixelChange:
    def test_identical_is_zero(self):
        img = np.full((30, 30, 3), 90, np.uint8)
        assert pixel_change(img, img.copy())["changed_pct"] == 0.0

    def test_different_sizes_same_aspect(self):
        small = np.zeros((50, 100, 3), np.uint8)
        big = np.zeros((100, 200, 3), np.float32)
        assert pixel_change(small, big)["changed_pct"] == 0.0

    def test_crop_returns_none(self):
        assert pixel_change(np.zeros((50, 100, 3)), np.zeros((100, 100, 3))) is None

    def test_missing_returns_none(self):
        assert pixel_change(None, np.zeros((5, 5, 3))) is None


class TestWriteReport:
    def test_path_and_round_trip(self, tmp_path):
        out = tmp_path / "shoot" / "IMG_1.jpg"
        out.parent.mkdir()
        path = write_edit_report({"a": 1}, out)
        assert path == report_path_for(out) == tmp_path / "shoot" / "edit-reports" / "IMG_1.jpg.json"
        assert json.loads(path.read_text()) == {"a": 1}
        assert not list(path.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    "had,signed,fragment",
    [
        (False, False, "No Content Credentials"),
        (True, False, "would fail verification"),
        (False, True, "Signed with new Content Credentials"),
        (True, True, "as the parent"),
    ],
)
def test_credentials_note(had, signed, fragment):
    assert fragment in credentials_note(had, signed)

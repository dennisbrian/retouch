"""Regression tests for the 2026-09-25 QA flag recalibration.

banding, plastic_skin, seam and asymmetry used to flag every real render
(absolute measurements over the person mask), and color_drift's p99 tail
flagged most styled cosplay renders. These tests pin the new contract: a
render that changes nothing (or changes things in a benign way) stays clean,
while a planted defect of each kind still flags. Calibration data:
docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md.
"""

import cv2
import numpy as np
import pytest

from retouch import qa_detectors
from retouch.qa_detectors import (
    ASYMMETRY_THRESHOLD,
    BANDING_THRESHOLD,
    PLASTIC_SKIN_THRESHOLD,
    QA_STATUS_NOT_RUN,
    SEAM_THRESHOLD,
    detect_banding,
    detect_color_drift,
    detect_over_retouch_asymmetry,
    detect_plastic_skin,
    detect_seam,
    run_all,
)

H, W = 160, 200


def _photo(seed: int = 0) -> np.ndarray:
    """Photo-like BGR frame: skin-tone ramp + grain + fine texture, dark background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.stack([
        110 + 0.25 * xx,      # B
        140 + 0.20 * yy,      # G
        185 + 0.15 * xx,      # R
    ], axis=-1)
    grain = rng.normal(0.0, 3.0, (H, W, 1)).astype(np.float32)
    img = base + grain
    img[:, : W // 5] = 40 + rng.normal(0.0, 3.0, (H, W // 5, 3))  # background strip
    return np.clip(img, 0, 255).astype(np.uint8)


def _person_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.float32)
    mask[:, W // 5:] = 1.0
    return mask


def _face_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.float32)
    mask[30:130, 80:180] = 1.0
    return mask


def _posterize_face(img: np.ndarray, face: np.ndarray, step: int) -> np.ndarray:
    """Smooth the face then quantize to ``step``-level plateaus (8-bit banding)."""
    smooth = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 6)
    banded = np.clip(np.round(smooth / step) * step, 0, 255).astype(np.uint8)
    out = img.copy()
    out[face > 0.5] = banded[face > 0.5]
    return out


class TestBanding:
    def test_grainy_photo_not_flagged(self):
        img = _photo()
        assert detect_banding(img, _person_mask())["flagged"] is False

    def test_unchanged_render_scores_zero_against_reference(self):
        img = _photo()
        r = detect_banding(img, _person_mask(), img, face_mask=_face_mask())
        assert r["differential"] is True
        assert r["score"] == 0.0 and r["flagged"] is False

    def test_posterized_face_flagged_even_inside_large_person_mask(self):
        ref = _photo()
        out = _posterize_face(ref, _face_mask(), 3)
        r = detect_banding(out, _person_mask(), ref, face_mask=_face_mask())
        assert r["face_score"] > BANDING_THRESHOLD
        assert r["flagged"] is True

    def test_banding_already_in_source_not_blamed_on_retouch(self):
        ref = _posterize_face(_photo(), _face_mask(), 4)
        r = detect_banding(ref.copy(), _person_mask(), ref, face_mask=_face_mask())
        assert r["flagged"] is False

    def test_one_level_steps_are_not_banding(self):
        # A clean, noiseless gentle ramp is all 1-level steps with wide
        # plateaus: unavoidable 8-bit quantization, not visible bands.
        ramp = np.tile((np.arange(W) * 0.1 + 100).astype(np.uint8), (H, 1))
        img = np.dstack([ramp] * 3)
        assert detect_banding(img)["score"] == 0.0

    def test_coarse_posterized_ramp_flagged_without_reference(self):
        ramp = (np.arange(W) * 0.4 + 60).astype(np.float32)
        ramp = (np.round(ramp / 4) * 4).astype(np.uint8)
        img = np.dstack([np.tile(ramp, (H, 1))] * 3)
        assert detect_banding(img)["flagged"] is True


class TestPlasticSkin:
    def test_no_reference_is_not_run_and_never_flags(self):
        flat = np.full((64, 64, 3), 150, np.uint8)
        r = detect_plastic_skin(flat, np.ones((64, 64), np.float32))
        assert r["status"] == QA_STATUS_NOT_RUN
        assert r["flagged"] is False
        assert "hf_energy_ratio" in r

    def test_texture_kept_not_flagged(self):
        img = _photo()
        r = detect_plastic_skin(img, _face_mask(), img)
        assert r["texture_retention"] == pytest.approx(1.0)
        assert r["flagged"] is False

    def test_blurred_face_flagged(self):
        ref = _photo()
        out = ref.copy()
        face = _face_mask() > 0.5
        out[face] = cv2.GaussianBlur(ref, (0, 0), 2.0)[face]
        r = detect_plastic_skin(out, _face_mask(), ref)
        assert r["texture_retention"] < PLASTIC_SKIN_THRESHOLD
        assert r["flagged"] is True


def _outline(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8)
    return (cv2.dilate(m, np.ones((3, 3), np.uint8)) ^ m) > 0


class TestSeam:
    def test_natural_outline_not_flagged_against_reference(self):
        img = _photo()
        # The subject/background edge alone is a big absolute gradient...
        assert detect_seam(img, _person_mask())["seam_gradient"] > SEAM_THRESHOLD
        # ...but the retouch did not add it.
        r = detect_seam(img, _person_mask(), img)
        assert r["differential"] is True
        assert r["flagged"] is False

    def test_global_contrast_boost_not_a_seam(self):
        # A punchy-look contrast boost sharpens every edge, the outline too.
        ref = _photo()
        out = np.clip((ref.astype(np.float32) - 128) * 1.15 + 128, 0, 255).astype(np.uint8)
        assert detect_seam(out, _person_mask(), ref)["flagged"] is False

    def test_line_traced_along_outline_flagged(self):
        ref = _photo()
        out = ref.copy()
        out[_outline(_person_mask())] = 255
        assert detect_seam(out, _person_mask(), ref)["flagged"] is True


class TestAsymmetry:
    def _two_surface_reference(self) -> np.ndarray:
        # Left half of the region is naturally smooth (hair/fabric), right is
        # textured skin: raw energies differ though nothing was retouched.
        img = _photo()
        left = np.zeros((H, W), bool)
        left[:, : W // 2] = True
        img[left] = cv2.GaussianBlur(img, (0, 0), 3)[left]
        return img

    def test_unretouched_mixed_surfaces_not_flagged_with_reference(self):
        ref = self._two_surface_reference()
        mask = _person_mask()
        assert detect_over_retouch_asymmetry(ref, mask)["flagged"] is True  # legacy absolute
        r = detect_over_retouch_asymmetry(ref, mask, reference_img_bgr=ref)
        assert r["relative_to_reference"] is True
        assert r["flagged"] is False

    def test_one_zone_smoothed_flagged_with_reference(self):
        ref = _photo()
        face = _face_mask()
        out = ref.copy()
        zone = np.zeros((H, W), bool)
        zone[63:96, 80:113] = True  # left-middle cell of the face bbox grid
        out[zone] = cv2.GaussianBlur(ref, (0, 0), 3)[zone]
        r = detect_over_retouch_asymmetry(out, face, reference_img_bgr=ref)
        assert r["score"] > ASYMMETRY_THRESHOLD
        assert r["flagged"] is True


def _rotate_hue(img: np.ndarray, mask: np.ndarray, degrees: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    t = np.radians(degrees)
    sel = mask > 0.5
    lab[..., 1][sel] = (a * np.cos(t) - b * np.sin(t))[sel] + 128
    lab[..., 2][sel] = (a * np.sin(t) + b * np.cos(t))[sel] + 128
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class TestColorDrift:
    def test_small_intentional_edit_does_not_flag(self):
        # A lip-sized edit (3% of the region) with a big hue change moves the
        # p99 tail but not p95 or the mean: intentional makeup, not a cast.
        ref = _photo()
        region = _person_mask()
        lips = np.zeros((H, W), np.float32)
        lips[100:110, 110:170] = 1.0
        out = _rotate_hue(ref, lips, 40)
        r = detect_color_drift(out, region, ref)
        assert r["deltaH_p99_deg"] > qa_detectors.COLOR_DRIFT_HUE_P99_THRESHOLD
        assert r["flagged"] is False

    def test_whole_skin_cast_flags(self):
        ref = _photo()
        out = _rotate_hue(ref, _person_mask(), 25)
        r = detect_color_drift(out, _person_mask(), ref)
        assert r["deltaH_p95_deg"] is not None
        assert r["flagged"] is True


def test_run_all_clean_render_raises_no_calibrated_flags():
    ref = _photo()
    mask = _person_mask()
    out = run_all(
        ref.copy(), skin_mask=mask, reference_img_bgr=ref, img_before=ref,
        person_mask=mask, face_skin_mask=_face_mask(),
    )
    for name in ("banding", "plastic_skin", "seam", "asymmetry", "color_drift"):
        assert out[name]["flagged"] is False, name


def test_run_all_scores_plastic_skin_and_asymmetry_on_face_skin(monkeypatch):
    seen = {}

    def spy(name):
        def _f(img, mask, *args, **kwargs):
            seen[name] = mask
            return {"score": 0.0, "flagged": False}
        return _f

    monkeypatch.setattr(qa_detectors, "detect_plastic_skin", spy("plastic_skin"))
    monkeypatch.setattr(qa_detectors, "detect_over_retouch_asymmetry", spy("asymmetry"))
    ref = _photo()
    face = _face_mask()
    run_all(ref, skin_mask=_person_mask(), reference_img_bgr=ref,
            person_mask=_person_mask(), face_skin_mask=face)
    assert seen["plastic_skin"] is face
    assert seen["asymmetry"] is face

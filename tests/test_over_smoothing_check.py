"""Patch-scale over-smoothing and colour-cast checks (2026-09-27).

plastic_skin and color_drift averaged over the whole face (and color_drift
over the whole person), so one waxy cheek or a cast on the chin barely moved
them. Both now also scan cheek-sized patches of each face. These tests pin
that contract: a local defect flags and names where it is, while an even
edit, an untouched render and a small face do not flag through the patch
check. Calibration: docs/plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md.
"""

import cv2
import numpy as np
import pytest

from retouch import qa_detectors
from retouch.qa_detectors import (
    COLOR_DRIFT_PATCH_THRESHOLD,
    PLASTIC_SKIN_PATCH_MIN_FACE_SIDE,
    PLASTIC_SKIN_PATCH_THRESHOLD,
    PLASTIC_SKIN_THRESHOLD,
    detect_color_drift,
    detect_plastic_skin,
    face_zone_name,
    run_qa_with_evidence,
)

H, W = 420, 360
FACE = (60, 60, 240, 300)  # x, y, w, h of the face box


def _photo(seed: int = 0) -> np.ndarray:
    """Skin-toned frame with pore-scale grain (the fine texture the check measures)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.stack([120 + 0.05 * xx, 150 + 0.05 * yy, 200 + 0.03 * xx], axis=-1)
    grain = rng.normal(0.0, 5.0, (H, W, 1)).astype(np.float32)
    return np.clip(base + grain, 0, 255).astype(np.uint8)


def _face_mask(face=FACE) -> np.ndarray:
    x, y, w, h = face
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 1, -1)
    return mask.astype(np.float32)


def _cheek(face=FACE, fx=0.28, fy=0.58) -> np.ndarray:
    x, y, w, h = face
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (int(x + fx * w), int(y + fy * h)),
                (int(0.15 * w), int(0.13 * h)), 0, 0, 360, 1, -1)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)


def _blur_in(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    w = weight[..., None]
    return np.clip(img * (1 - w) + blurred * w, 0, 255).astype(np.uint8)


def _shift_ab(img: np.ndarray, weight: np.ndarray, da: float, db: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += da * weight
    lab[..., 2] += db * weight
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class TestWaxyPatch:
    def test_untouched_render_keeps_all_patch_texture(self):
        img, face = _photo(), _face_mask()
        r = detect_plastic_skin(img, face, img)
        assert r["flagged"] is False
        assert r["patch_retention"] == pytest.approx(1.0, abs=1e-3)

    def test_blurred_cheek_flags_although_whole_face_passes(self):
        ref, face = _photo(), _face_mask()
        out = _blur_in(ref, _cheek(), 1.5)
        r = detect_plastic_skin(out, face, ref)
        assert r["texture_retention"] >= PLASTIC_SKIN_THRESHOLD  # the old check alone passes
        assert r["patch_retention"] < PLASTIC_SKIN_PATCH_THRESHOLD
        assert r["flagged"] is True
        assert r["patch_zone"] == "cheek on the left of the photo"
        assert "Waxy patch on the cheek on the left of the photo" in r["finding"]
        assert r["score"] == pytest.approx(r["patch_retention"])

    def test_right_cheek_named_right(self):
        ref, face = _photo(), _face_mask()
        out = _blur_in(ref, _cheek(fx=0.72), 1.5)
        r = detect_plastic_skin(out, face, ref)
        assert r["flagged"] is True
        assert r["patch_zone"] == "cheek on the right of the photo"

    def test_even_light_smoothing_is_not_a_waxy_patch(self):
        ref, face = _photo(), _face_mask()
        # Keep ~75% of the grain everywhere: an even, moderate retouch.
        blurred = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 3)
        out = np.clip(0.75 * ref + 0.25 * blurred, 0, 255).astype(np.uint8)
        r = detect_plastic_skin(out, face, ref)
        assert r["flagged"] is False
        assert r["patch_retention"] > PLASTIC_SKIN_PATCH_THRESHOLD

    def test_flattened_nose_is_not_scored(self):
        ref, face = _photo(), _face_mask()
        out = _blur_in(ref, _cheek(fx=0.5, fy=0.52), 3.0)
        r = detect_plastic_skin(out, face, ref)
        assert r["flagged"] is False

    def test_small_face_gets_whole_face_check_only(self):
        side = PLASTIC_SKIN_PATCH_MIN_FACE_SIDE - 40
        small = (100, 100, side - 20, side)
        ref, face = _photo(), _face_mask(small)
        out = _blur_in(ref, _cheek(small), 1.5)
        r = detect_plastic_skin(out, face, ref)
        assert r["patch_retention"] is None
        assert r["texture_retention"] is not None

    def test_whole_face_blur_still_flags_with_whole_face_message(self):
        ref, face = _photo(), _face_mask()
        out = _blur_in(ref, face, 1.5)
        r = detect_plastic_skin(out, face, ref)
        assert r["flagged"] is True
        assert r["texture_retention"] < PLASTIC_SKIN_THRESHOLD
        assert r["finding"].startswith("Waxy skin: the face keeps")


class TestPatchColourCast:
    def test_cast_on_lower_face_flags_and_names_it(self):
        ref, face = _photo(), _face_mask()
        lower = face.copy()
        lower[: FACE[1] + FACE[3] * 2 // 3] = 0
        out = _shift_ab(ref, cv2.GaussianBlur(lower, (0, 0), 2.0), 0.0, 9.0)
        r = detect_color_drift(out, face, ref, face_mask=face)
        assert r["patch_delta_ab"] > COLOR_DRIFT_PATCH_THRESHOLD
        assert r["flagged"] is True
        assert "chin" in r["patch_zone"] or "jaw" in r["patch_zone"]
        assert r["finding"].startswith("Colour cast on the ")

    def test_cast_over_half_the_face_names_the_cast_half(self):
        ref, face = _photo(), _face_mask()
        lower = face.copy()
        lower[: FACE[1] + FACE[3] // 2] = 0  # about half the face: the median follows the cast
        out = _shift_ab(ref, lower, -9.0, 0.0)
        r = detect_color_drift(out, face, ref, face_mask=face)
        assert r["flagged"] is True
        assert r["patch_point"][1] > FACE[1] + FACE[3] // 2
        assert "forehead" not in r["patch_zone"]

    def test_even_shift_over_whole_face_is_not_a_patch_cast(self):
        ref, face = _photo(), _face_mask()
        out = _shift_ab(ref, face, 0.0, 4.0)
        r = detect_color_drift(out, face, ref, face_mask=face)
        assert r["patch_delta_ab"] < 1.0

    def test_untouched_render_scores_zero(self):
        ref, face = _photo(), _face_mask()
        r = detect_color_drift(ref, face, ref, face_mask=face)
        assert r["patch_delta_ab"] == pytest.approx(0.0, abs=1e-3)
        assert r["flagged"] is False

    def test_without_face_mask_the_patch_gate_does_not_run(self):
        ref, face = _photo(), _face_mask()
        out = _shift_ab(ref, _cheek(), 12.0, 0.0)
        r = detect_color_drift(out, face, ref)
        assert r["patch_delta_ab"] is None
        assert "finding" not in r


def test_run_qa_uses_the_finding_as_the_review_message():
    ref, face = _photo(), _face_mask()
    out = _blur_in(ref, _cheek(), 1.5)
    person = np.ones((H, W), np.float32)
    warnings, evidence = run_qa_with_evidence(out, person, ref, face_skin_mask=face)
    plastic = [w for w in warnings if w.detector == "plastic_skin"]
    assert plastic, [w.detector for w in warnings]
    assert plastic[0].message.startswith("Waxy patch on the cheek")
    assert evidence["plastic_skin"]["patch_zone"] == "cheek on the left of the photo"


def test_run_all_passes_face_skin_to_color_drift(monkeypatch):
    seen = {}

    def spy(img, mask, ref, face_mask=None):
        seen["face_mask"] = face_mask
        return {"score": 0.0, "flagged": False}

    monkeypatch.setattr(qa_detectors, "detect_color_drift", spy)
    ref, face = _photo(), _face_mask()
    qa_detectors.run_all(ref, skin_mask=np.ones((H, W), np.float32),
                         reference_img_bgr=ref, face_skin_mask=face)
    assert seen["face_mask"] is face


@pytest.mark.parametrize("point, name", [
    ((70, 80), "forehead on the left of the photo"),
    ((180, 80), "forehead"),
    ((70, 200), "cheek on the left of the photo"),
    ((180, 200), "nose"),
    ((280, 200), "cheek on the right of the photo"),
    ((180, 340), "chin"),
    ((280, 340), "jaw on the right of the photo"),
])
def test_face_zone_names(point, name):
    assert face_zone_name(point, FACE) == name


def test_mask_sliver_is_not_scanned_as_a_face():
    ref, face = _photo(), _face_mask()
    face[5:25, 5:40] = 1.0  # 700-px sliver of skin, e.g. between wig strands
    out = _shift_ab(ref, np.pad(np.ones((20, 35), np.float32), ((5, H - 25), (5, W - 40))), -10.0, 0.0)
    r = detect_color_drift(out, face, ref, face_mask=face)
    assert r["patch_delta_ab"] < 1.0
    assert r["patch_face_bbox"][2] > 200

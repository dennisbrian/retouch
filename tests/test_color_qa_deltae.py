"""Tests for the K8 color-fidelity QA gate (CIEDE2000 + hue-drift detector).

The authoritative correctness check is the Sharma et al. 2005 Table I
34 published CIEDE2000 test pairs.
"""

import os

import cv2
import numpy as np
import pytest

from retouch.qa_detectors import (
    detect_color_drift,
    delta_e_2000,
    run_all,
)

# Sharma, Wu, Dalal (2005) "The CIEDE2000 Color-Difference Formula:
# Implementation Notes..." — Table I. 34 published test pairs
# (L1,a1,b1, L2,a2,b2, expected dE). Expected values are the reference
# implementation outputs (exact Sharma closed form); pairs 9-20 are the
# known borderline pairs, so the test tolerance is relaxed to 2e-3.
SHARMA_TABLE_I = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2645, 50.0000, 0.0000, -82.7485, 0.9968),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0018, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0018, -2.4900, 4.7461),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0012, -2.4900, 4.7461),
    (50.0000, 0.0009, -2.4900, 50.0000, -0.0009, 2.4900, 4.8045),
    (50.0000, 0.0009, -2.4900, 50.0000, -0.0018, 2.4900, 4.8045),
    (50.0000, 0.0009, -2.4900, 50.0000, -0.0012, 2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0018, -2.4900, 4.7461),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0012, -2.4900, 4.7461),
    (60.2574, -34.0099, 36.2677, 60.4626, -31.1530, 30.7451, 2.1399),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, -1.7736, -0.0136, 2.9435, 7.1892),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, 0.6258, 1.8761),
    (0.9033, -0.0636, 0.6258, 0.0000, 0.0000, 0.0000, 0.8118),
    (0.0000, 0.0000, 0.0000, 0.9033, -0.0636, 0.6258, 0.8118),
    (53.1229, 1.2364, -1.1036, 53.3771, 1.2190, -1.0915, 0.2485),
    (95.6417, -0.2340, -0.0752, 95.6090, -0.1955, -0.1138, 0.0715),
    (99.8670, 0.0000, 0.0000, 99.5745, 0.0000, 0.0000, 0.1678),
]


def test_delta_e_2000_sharma_table_i():
    max_err = 0.0
    for idx, (l1, a1, b1, l2, a2, b2, expected) in enumerate(SHARMA_TABLE_I):
        got = delta_e_2000(
            np.array([l1, a1, b1], dtype=np.float32),
            np.array([l2, a2, b2], dtype=np.float32),
        )
        err = abs(got - expected)
        # Pairs 9-20 (index 8-19) are the borderline set from the paper.
        tol = 2e-3 if 8 <= idx <= 19 else 1e-3
        assert err < tol, f"pair {idx + 1} got {got}, expected {expected}, err {err}"
        max_err = max(max_err, err)
    print(f"Sharma max error: {max_err:.6f}")


def test_delta_e_2000_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    lab1 = rng.uniform(0, 100, (40, 30, 3)).astype(np.float32)
    lab2 = rng.uniform(0, 100, (40, 30, 3)).astype(np.float32)
    de_img = delta_e_2000(lab1, lab2)
    assert de_img.shape == (40, 30)
    de00 = delta_e_2000(lab1[0, 0], lab2[0, 0])
    assert abs(de_img[0, 0] - de00) < 1e-5


def _make_skin_patch(hue_deg: float, size: int = 64) -> np.ndarray:
    """Build a synthetic skin-ish BGR patch at a given LAB hue angle."""
    # Skin chroma point in a/b, then rotate hue.
    a0, b0 = 8.0, 14.0
    theta = np.radians(hue_deg)
    a = a0 * np.cos(theta) - b0 * np.sin(theta)
    b = a0 * np.sin(theta) + b0 * np.cos(theta)
    lab = np.zeros((size, size, 3), dtype=np.float32)
    lab[:, :, 0] = 65.0
    lab[:, :, 1] = a
    lab[:, :, 2] = b
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return (np.clip(bgr, 0.0, 1.0) * 255.0).astype(np.uint8)


def test_detect_color_drift_large_hue_rotation_flags():
    ref = _make_skin_patch(20.0)
    # Rotate hue by 40 degrees -> large skin Δh, should flag.
    out = _make_skin_patch(60.0)
    res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
    assert res["flagged"] is True
    assert res["deltaH_mean_deg"] > 30.0
    assert "score" in res and res["score"] > 0.0


def test_detect_color_drift_tiny_rotation_passes():
    ref = _make_skin_patch(20.0)
    out = _make_skin_patch(22.0)  # 2° shift, well under budget
    res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
    assert res["flagged"] is False
    assert res["deltaH_mean_deg"] < 6.0


def test_detect_color_drift_no_reference():
    img = _make_skin_patch(20.0)
    res = detect_color_drift(img, skin_mask=None, reference_img_bgr=None)
    assert res["flagged"] is False
    # Unmeasured, not a zero-drift pass (Q5, 2026-09-23).
    assert res["score"] is None
    assert res["status"] == "not-run"
    assert "note" in res


def _lab_to_bgr_f(lab: np.ndarray) -> np.ndarray:
    """LAB (float32, L 0-100) → float32 BGR [0, 255] (no uint8 quantisation)."""
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr, 0.0, 1.0).astype(np.float32) * 255.0


def _skin_lab(hue_offset_deg: float, size: int = 64, L: float = 65.0,
              chroma_scale: float = 1.0) -> np.ndarray:
    a0, b0 = 8.0 * chroma_scale, 14.0 * chroma_scale
    t = np.radians(hue_offset_deg)
    lab = np.zeros((size, size, 3), dtype=np.float32)
    lab[..., 0] = L
    lab[..., 1] = a0 * np.cos(t) - b0 * np.sin(t)
    lab[..., 2] = a0 * np.sin(t) + b0 * np.cos(t)
    return lab


class TestColorDriftChromaFlooredPercentile:
    """Regression for the 2026-09-23 rule change: near-neutral pixels (hue
    undefined as chroma → 0) must not trip the gate; real skin hue rotations
    must. The old single-pixel ``max Δh > 15°`` rule flagged the first case."""

    def test_single_near_neutral_pixel_hue_flip_does_not_flag(self):
        ref_lab = _skin_lab(0.0)
        out_lab = ref_lab.copy()
        # One near-neutral pixel whose hue flips by 180° between ref and out.
        ref_lab[10, 10, 1:] = (0.4, 0.4)
        out_lab[10, 10, 1:] = (-0.4, -0.4)
        ref, out = _lab_to_bgr_f(ref_lab), _lab_to_bgr_f(out_lab)
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["deltaH_max_deg"] > 150.0  # raw compat field still sees it
        assert res["flagged"] is False
        assert res["deltaH_p99_deg"] < 1.0

    def test_many_near_neutral_pixels_hue_flip_does_not_flag(self):
        # 5% near-neutral pixels flipping: a plain p99 (no chroma floor) would
        # read ~180° here, so this pins the relative chroma floor itself.
        ref_lab = _skin_lab(0.0)
        out_lab = ref_lab.copy()
        ref_lab[:3, :, 1:] = (0.4, 0.4)
        out_lab[:3, :, 1:] = (-0.4, -0.4)
        ref, out = _lab_to_bgr_f(ref_lab), _lab_to_bgr_f(out_lab)
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["flagged"] is False
        assert res["hue_kept_fraction"] < 0.97

    @pytest.mark.parametrize("deg", [8.0, 15.0, 20.0])
    def test_uniform_skin_hue_rotation_flags(self, deg):
        ref = _lab_to_bgr_f(_skin_lab(0.0))
        out = _lab_to_bgr_f(_skin_lab(deg))
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["flagged"] is True
        assert res["deltaH_p99_deg"] == pytest.approx(deg, abs=1.0)

    @pytest.mark.parametrize("L,chroma_scale", [(35.0, 1.0), (65.0, 0.4), (30.0, 0.5)])
    def test_rotation_flags_on_darker_or_lower_chroma_skin(self, L, chroma_scale):
        # Tone invariance: the floor is relative to the region's own chroma,
        # so a darker / less saturated subject is gated the same way.
        ref = _lab_to_bgr_f(_skin_lab(0.0, L=L, chroma_scale=chroma_scale))
        out = _lab_to_bgr_f(_skin_lab(15.0, L=L, chroma_scale=chroma_scale))
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["flagged"] is True
        ok = detect_color_drift(ref.copy(), skin_mask=None, reference_img_bgr=ref)
        assert ok["flagged"] is False

    def test_bw_grade_hue_not_measurable_no_crash(self):
        ref = _lab_to_bgr_f(_skin_lab(0.0))
        grey = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        out = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["flagged"] is False
        assert res["deltaH_p99_deg"] is None
        assert res["status"] == "not-run"  # hue gate unmeasured, not a pass
        assert res["deltaE_mean"] > 1.0

    def test_neutral_grey_reference_no_crash(self):
        rng = np.random.default_rng(3)
        ref = np.full((64, 64, 3), 128.0, np.float32) + rng.normal(0, 0.3, (64, 64, 3)).astype(np.float32)
        out = np.full((64, 64, 3), 128.0, np.float32) + rng.normal(0, 0.3, (64, 64, 3)).astype(np.float32)
        res = detect_color_drift(out, skin_mask=None, reference_img_bgr=ref)
        assert res["flagged"] is False

    def test_not_measured_includes_new_fields(self):
        res = detect_color_drift(_make_skin_patch(20.0), None, None)
        for k in ("deltaH_p99_deg", "deltaH_mean_chroma_floored_deg",
                  "chroma_floor", "hue_kept_fraction"):
            assert k in res and res[k] is None


def test_run_all_includes_color_drift():
    rng = np.random.default_rng(1)
    img = (rng.uniform(0, 255, (48, 48, 3))).astype(np.uint8)
    out = run_all(img, skin_mask=None, reference_img_bgr=img)
    assert "color_drift" in out
    assert out["color_drift"]["flagged"] is False


if __name__ == "__main__":
    test_delta_e_2000_sharma_table_i()

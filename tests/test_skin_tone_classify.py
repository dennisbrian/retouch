import cv2
import numpy as np
import pytest

from retouch.color_science import classify_fitzpatrick, ita_value, tone_adaptation_params


def _lab_patch(L: float, a: float, b: float, n: int = 4) -> np.ndarray:
    """Build an (n,n,3) float32 CIELAB patch with constant L*,a*,b*."""
    patch = np.zeros((n, n, 3), dtype=np.float32)
    patch[..., 0] = L
    patch[..., 1] = a
    patch[..., 2] = b
    return patch


def _lab_to_bgr_patch(L: float, a: float, b: float, n: int = 4) -> np.ndarray:
    """Build a float32 BGR patch [0,255] from a CIELAB constant via cv2."""
    lab = _lab_patch(L, a, b, n)
    bgr = cv2_lab_to_bgr(lab)
    return (bgr * 255.0).astype(np.float32)


def cv2_lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)


def test_ita_value_known_angles():
    assert ita_value(_lab_patch(70.0, 0.0, 15.0)) == pytest.approx(53.13, abs=0.5)
    assert ita_value(_lab_patch(35.0, 0.0, 12.0)) == pytest.approx(-51.34, abs=0.5)


def test_ita_value_near_zero_bstar_masked():
    # b* exactly 0 -> unstable angle, must not explode; returns finite mean.
    out = ita_value(_lab_patch(70.0, 0.0, 0.0))
    assert np.isfinite(out)


def test_classify_light_patch():
    bgr = _lab_to_bgr_patch(70.0, 10.0, 15.0)
    res = classify_fitzpatrick(bgr)
    assert 1 <= res["type_index"] <= 3
    assert res["label"] in {"I", "II", "III"}
    assert res["ita"] == pytest.approx(53.13, abs=1.0)


def test_classify_dark_patch():
    bgr = _lab_to_bgr_patch(25.0, 12.0, 12.0)
    res = classify_fitzpatrick(bgr)
    assert res["type_index"] in (5, 6)
    assert res["ita"] < 0


def test_classify_uint8_input():
    lab = _lab_patch(60.0, 10.0, 15.0)
    bgr_f = (cv2_lab_to_bgr(lab) * 255.0).astype(np.float32)
    bgr_u8 = np.clip(np.round(bgr_f), 0, 255).astype(np.uint8)
    res_f = classify_fitzpatrick(bgr_f)
    res_u = classify_fitzpatrick(bgr_u8)
    assert res_f["type_index"] == res_u["type_index"]
    assert np.isfinite(res_u["ita"])


def test_ita_monotonic_darker():
    light = _lab_to_bgr_patch(72.0, 10.0, 18.0)
    medium = _lab_to_bgr_patch(55.0, 11.0, 14.0)
    dark = _lab_to_bgr_patch(30.0, 12.0, 12.0)
    itas = [classify_fitzpatrick(p)["ita"] for p in (light, medium, dark)]
    assert itas == sorted(itas, reverse=True)
    assert itas[0] > itas[1] > itas[2]


def test_tone_adaptation_keys_all_types():
    for idx in range(1, 7):
        params = tone_adaptation_params(idx)
        assert set(params.keys()) == {"smooth_scale", "highlight_scale", "locus_target"}
        assert 0.0 < params["smooth_scale"] <= 1.0
    # Darkest type has no hue locus target.
    assert tone_adaptation_params(6)["locus_target"] is None

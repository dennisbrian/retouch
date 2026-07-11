"""R15 QA extension detector tests (read-only analysis, no pixel mutation)."""

from __future__ import annotations

import numpy as np

from retouch.qa_detectors import (
    detect_pore_spectrum_distance,
    detect_over_retouch_asymmetry,
    gui_skin_score,
    PORE_SPECTRUM_THRESHOLD,
    ASYMMETRY_THRESHOLD,
    SKIN_SCORE_FLOOR,
)


def _make_skin(h: int = 256, w: int = 256, rng_seed: int = 0) -> np.ndarray:
    """Natural skin-like texture: gentle ramp + per-pixel value noise + chroma jitter."""
    rng = np.random.default_rng(rng_seed)
    ramp = np.linspace(90, 160, w, dtype=np.float32)
    base = np.tile(ramp, (h, 1))  # gray ramp
    noise = rng.normal(0.0, 14.0, size=(h, w)).astype(np.float32)
    lum = np.clip(base + noise, 0, 255)
    # chroma jitter (skin tone) so chroma variance is non-trivial
    a_jit = rng.normal(0.0, 6.0, size=(h, w)).astype(np.float32)
    b_jit = rng.normal(0.0, 5.0, size=(h, w)).astype(np.float32)
    g = lum
    r = np.clip(lum + 8.0 + a_jit, 0, 255)
    b = np.clip(lum - 6.0 + b_jit, 0, 255)
    img = np.stack([b, g, r], axis=2).astype(np.uint8)
    return img


def _blur_skin(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Heavily blur only the skin-masked region (simulates over-retouch)."""
    import cv2
    out = img.copy().astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), 12.0).astype(np.float32)
    m = (mask > 0.5).astype(np.float32)[..., None]
    out = out * (1.0 - m) + blurred * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _skin_mask(h: int = 256, w: int = 256) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[40:220, 40:220] = 1.0
    return m


def test_pore_spectrum_reference_flagged_and_self_zero():
    ref = _make_skin()
    mask = _skin_mask()
    import cv2
    blurred = cv2.GaussianBlur(ref, (0, 0), 14.0).astype(np.uint8)

    d_ref = detect_pore_spectrum_distance(ref, mask, reference_img_bgr=ref)
    assert d_ref["score"] < PORE_SPECTRUM_THRESHOLD
    assert d_ref["flagged"] is False
    assert "pore_energy_ratio" in d_ref and "pore_band_energy" in d_ref \
        and "ref_pore_band_energy" in d_ref

    d_blur = detect_pore_spectrum_distance(blurred, mask, reference_img_bgr=ref)
    assert d_blur["score"] > PORE_SPECTRUM_THRESHOLD
    assert d_blur["flagged"] is True


def test_pore_spectrum_no_reference_safe():
    ref = _make_skin()
    d = detect_pore_spectrum_distance(ref, None, reference_img_bgr=None)
    assert d["score"] == 0.0
    assert d["flagged"] is False
    assert d["note"] == "no reference"


def test_asymmetry_half_blur_flagged_uniform_not():
    ref = _make_skin()
    mask = _skin_mask()
    h, w = mask.shape

    # Uniform blur over whole skin -> symmetric -> should NOT flag.
    uniform = _blur_skin(ref, mask)
    d_uniform = detect_over_retouch_asymmetry(uniform, skin_mask=mask)
    assert d_uniform["flagged"] is False
    assert "zone_energies" in d_uniform and "face_mean_energy" in d_uniform \
        and "min_zone_ratio" in d_uniform

    # Blur only the left half of the skin -> asymmetric -> should flag.
    half = np.zeros_like(mask)
    half[:, :w // 2] = mask[:, :w // 2]
    asymmetric = _blur_skin(ref, half)
    d_asym = detect_over_retouch_asymmetry(asymmetric, skin_mask=mask)
    assert d_asym["score"] > ASYMMETRY_THRESHOLD
    assert d_asym["flagged"] is True


def test_gui_skin_score_blurred_lower():
    ref = _make_skin()
    mask = _skin_mask()
    blurred = _blur_skin(ref, mask)

    s_textured = gui_skin_score(ref, skin_mask=mask, reference_img_bgr=ref)
    s_blurred = gui_skin_score(blurred, skin_mask=mask, reference_img_bgr=ref)

    assert 0.0 <= s_textured["score"] <= 100.0
    assert 0.0 <= s_blurred["score"] <= 100.0
    assert "chroma_var" in s_textured and "texture_metric" in s_textured \
        and "flagged" in s_textured
    assert s_blurred["score"] < s_textured["score"] - 10.0


def test_detectors_do_not_mutate_inputs():
    ref = _make_skin()
    mask = _skin_mask()
    blurred = _blur_skin(ref, mask)

    for img, name in ((ref, "ref"), (blurred, "blurred")):
        before = img.copy()
        detect_pore_spectrum_distance(img, mask, reference_img_bgr=ref)
        detect_over_retouch_asymmetry(img, skin_mask=mask)
        gui_skin_score(img, skin_mask=mask, reference_img_bgr=ref)
        assert np.array_equal(img, before), f"input mutated: {name}"

    # reference must also be untouched
    ref_before = ref.copy()
    detect_pore_spectrum_distance(blurred, mask, reference_img_bgr=ref)
    assert np.array_equal(ref, ref_before)

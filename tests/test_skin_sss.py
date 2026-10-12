"""Subsurface-scattering skin-render approximation tests."""

import numpy as np
import pytest
import cv2

from retouch.skin import SkinProcessor


FITZPATRICK_BGR = {
    "I": (189, 208, 244),
    "II": (143, 180, 231),
    "III": (109, 152, 208),
    "IV": (87, 114, 165),
    "V": (62, 82, 122),
    "VI": (38, 51, 80),
}


def _shaded_face(bgr, seed=7, size=128):
    """Base tone under a multiplicative lit->shadow ramp, plus pore noise."""
    rng = np.random.RandomState(seed)
    ramp = np.linspace(1.0, 0.5, size, dtype=np.float32)[None, :, None]
    base = np.zeros((size, size, 3), np.float32) + np.asarray(bgr, np.float32)
    img = base * ramp + rng.normal(0, 3.0, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


_sp = SkinProcessor()


def test_strength_zero_identity():
    """apply_sss with strength=0.0 returns input unchanged."""
    img = np.random.RandomState(0).randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32)
    out = _sp.apply_sss(img, mask, 0.0, face_width=100.0)
    assert np.array_equal(out, img)


def test_none_mask_identity():
    """apply_sss with skin_mask=None returns input unchanged."""
    img = np.random.RandomState(0).randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = _sp.apply_sss(img, None, 0.7, face_width=100.0)
    assert np.array_equal(out, img)


def test_outside_mask_untouched():
    """Pixels outside the mask remain byte-identical to input."""
    size = 64
    img = np.random.RandomState(2).randint(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[32:96, 32:96] = 1.0  # Center square (will be clipped to size)
    mask = mask[:size, :size]  # Ensure mask is same size
    out = _sp.apply_sss(img, mask, 0.7, face_width=100.0)
    # Pixels where mask == 0 must be identical
    assert np.array_equal(out[mask == 0], img[mask == 0])


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    size = 64
    # Test uint8
    img_u8 = np.random.RandomState(3).randint(0, 255, (size, size, 3), dtype=np.uint8)
    mask = np.ones((size, size), dtype=np.float32)
    out_u8 = _sp.apply_sss(img_u8, mask, 0.5, face_width=100.0)
    assert out_u8.dtype == np.uint8

    # Test float32
    img_f32 = img_u8.astype(np.float32)
    out_f32 = _sp.apply_sss(img_f32, mask, 0.5, face_width=100.0)
    assert out_f32.dtype == np.float32


def test_monotonic_strength():
    """Effect increases monotonically with strength."""
    size = 64
    img = _shaded_face(FITZPATRICK_BGR["III"], seed=11, size=size)
    mask = np.ones((size, size), dtype=np.float32)

    # Compute mean absolute difference for each strength level
    diffs = []
    for strength in [0.2, 0.5, 0.9]:
        out = _sp.apply_sss(img, mask, strength, face_width=100.0)
        out_f32 = out.astype(np.float32)
        img_f32 = img.astype(np.float32)
        mean_diff = float(np.mean(np.abs(out_f32 - img_f32)))
        diffs.append(mean_diff)

    # Verify strictly increasing
    assert diffs[0] < diffs[1]
    assert diffs[1] < diffs[2]


def test_tone_invariant_relative_effect():
    """Relative effect (normalized by input level) is similar across all skin tones."""
    size = 64
    relative_effects = []

    for tone_name, tone_bgr in FITZPATRICK_BGR.items():
        img = _shaded_face(tone_bgr, seed=12, size=size)
        mask = np.ones((size, size), dtype=np.float32)
        out = _sp.apply_sss(img, mask, 0.8, face_width=100.0)

        img_f32 = img.astype(np.float32)
        out_f32 = out.astype(np.float32)
        mean_diff = float(np.mean(np.abs(out_f32 - img_f32)))
        baseline = float(np.mean(img_f32))
        normalizer = max(baseline, 1.0)
        rel = mean_diff / normalizer
        relative_effects.append(rel)

    # All relative effects should be > 0
    assert all(r > 0 for r in relative_effects)

    # Spread should be reasonable (max/min < 4.0)
    rel_spread = max(relative_effects) / (min(relative_effects) + 1e-6)
    assert rel_spread < 4.0, f"Relative effect spread {rel_spread} exceeds 4.0; effects per tone: {relative_effects}"


def test_terminator_warms():
    """Lit->shadow transition zone warms in LAB a/b."""
    size = 128
    img = _shaded_face(FITZPATRICK_BGR["III"], seed=13, size=size)
    mask = np.ones((size, size), dtype=np.float32)
    out = _sp.apply_sss(img, mask, 0.8, face_width=100.0)

    # Convert to LAB for analysis
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute a-channel shift
    a_shift = out_lab[..., 1] - img_lab[..., 1]

    # Define zones: the ramp goes 1.0 -> 0.5 across the width axis
    # Lit zone: columns 0 : size//10 (bright region)
    # Ramp/terminator zone: columns size*3//10 : size*7//10 (transition)
    lit_zone = a_shift[:, :size // 10]
    ramp_zone = a_shift[:, size * 3 // 10 : size * 7 // 10]

    mean_a_shift_lit = float(np.mean(lit_zone))
    mean_a_shift_ramp = float(np.mean(ramp_zone))

    # Ramp zone should be warmer (higher a) than lit zone
    assert mean_a_shift_ramp > mean_a_shift_lit
    # Ramp zone should have positive shift
    assert mean_a_shift_ramp > 0


def test_detail_preserved():
    """High-frequency detail (pores, texture) is largely preserved."""
    size = 128
    img = _shaded_face(FITZPATRICK_BGR["IV"], seed=14, size=size)
    mask = np.ones((size, size), dtype=np.float32)

    # Compute high-frequency detail before processing
    img_f32 = img.astype(np.float32)
    low_before = cv2.GaussianBlur(img_f32, (0, 0), 2.0)
    detail_before = img_f32 - low_before

    # Apply SSS
    out = _sp.apply_sss(img, mask, 0.7, face_width=100.0)

    # Compute high-frequency detail after processing
    out_f32 = out.astype(np.float32)
    low_after = cv2.GaussianBlur(out_f32, (0, 0), 2.0)
    detail_after = out_f32 - low_after

    # Compare over central region
    roi = slice(16, 112), slice(16, 112)
    lum_before = np.mean(detail_before[roi], axis=2)
    lum_after = np.mean(detail_after[roi], axis=2)

    energy_before = float(np.mean(lum_before ** 2))
    energy_after = float(np.mean(lum_after ** 2))

    # Ratio should be between 0.70 and 1.30 (detail somewhat attenuated, not removed)
    ratio = energy_after / (energy_before + 1e-6)
    assert 0.70 <= ratio <= 1.30, f"Detail energy ratio {ratio} out of range [0.70, 1.30]"


def test_tiny_mask_no_crash():
    """Tiny mask (6x6 non-zero region) does not crash."""
    size = 64
    img = _shaded_face(FITZPATRICK_BGR["V"], seed=15, size=size)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[32:38, 32:38] = 1.0  # 6x6 block
    out = _sp.apply_sss(img, mask, 0.7, face_width=100.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype

"""P4 makeup unmix — full product slice tests."""

import numpy as np
import pytest

from retouch.makeup_unmix import (
    apply_makeup_coverage_even,
    apply_makeup_unmix,
    cake_reduce,
    closed_form_alpha,
    estimate_makeup_alpha,
    even_coverage,
    recompose,
    unmix_makeup,
)


FITZPATRICK_BGR = {
    "I": (189, 208, 244),
    "II": (143, 180, 231),
    "III": (109, 152, 208),
    "IV": (87, 114, 165),
    "V": (62, 82, 122),
    "VI": (38, 51, 80),
}

# Match the headroom-qualified palette in test_specular_finish.py. A clipped
# highlight cannot be identified from its lost additive intensity alone.
SPECULAR_BGR = {
    "I": (180, 195, 220),
    "II": (143, 180, 210),
    "III": (109, 152, 190),
    "IV": (87, 114, 165),
    "V": (62, 82, 122),
    "VI": (38, 51, 80),
}


def _foundation_disk(bgr, seed=42):
    """Return a noisy bare swatch and a known broad foundation coverage area."""
    h = w = 96
    rng = np.random.RandomState(seed)
    base = np.zeros((h, w, 3), np.float32) + np.asarray(bgr, np.float32)
    bare = np.clip(base + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= 24 ** 2
    makeup = np.clip(
        np.asarray(bgr, np.float32) * 1.18 + np.array([0, 8, 12]), 0, 255,
    )
    foundation = bare.astype(np.float32)
    foundation[disk] = 0.5 * foundation[disk] + 0.5 * makeup
    return bare, np.clip(foundation, 0, 255).astype(np.uint8), disk


def test_strength_zero_identity():
    img = np.random.RandomState(0).randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32)
    out = apply_makeup_coverage_even(img, mask, 0.0)
    assert out is img or np.array_equal(out, img)
    out2 = apply_makeup_unmix(img, mask, coverage_even=0.0, cake_reduce_strength=0.0)
    assert out2 is img or np.array_equal(out2, img)


def test_even_coverage_reduces_variance_on_paint():
    h = w = 40
    img = np.full((h, w, 3), 200.0, dtype=np.float32)
    rng = np.random.RandomState(1)
    img[:, :, 0] = np.clip(img[:, :, 0] + rng.normal(0, 40, (h, w)), 0, 255)
    alpha = np.ones((h, w), dtype=np.float32)
    out = even_coverage(img, alpha, 0.9)
    assert out[:, :, 0].var() < img[:, :, 0].var()


def test_alpha_on_white_higher_than_mid_gray():
    h = w = 32
    mask = np.ones((h, w), dtype=np.float32)
    white = np.full((h, w, 3), 245, dtype=np.uint8)
    mid = np.full((h, w, 3), 128, dtype=np.uint8)
    a_w = estimate_makeup_alpha(white, mask)
    a_m = estimate_makeup_alpha(mid, mask)
    assert float(a_w.mean()) >= float(a_m.mean()) - 0.05


def test_closed_form_alpha_disk():
    h = w = 64
    S = np.full((h, w, 3), 160.0, dtype=np.float32)
    Mcol = np.array([200.0, 180.0, 220.0], dtype=np.float32)
    M = np.broadcast_to(Mcol, (h, w, 3)).copy()
    yy, xx = np.ogrid[:h, :w]
    disk = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 18 ** 2
    a_gt = disk.astype(np.float32) * 0.6
    I = recompose(S, M, a_gt)
    a_hat = closed_form_alpha(I, S, M)
    err = float(np.abs(a_hat[disk] - 0.6).mean())
    assert err < 0.08, err


def test_recompose_roundtrip_approx():
    h = w = 48
    S = np.full((h, w, 3), 140.0, dtype=np.float32)
    M = np.full((h, w, 3), 220.0, dtype=np.float32)
    a = np.full((h, w), 0.5, dtype=np.float32)
    I = recompose(S, M, a)
    S2, M2, a2 = unmix_makeup(I.astype(np.uint8), np.ones((h, w), dtype=np.float32))
    I2 = recompose(S2, M2, a2)
    err = float(np.abs(I - I2).mean())
    assert err < 25.0, err


def test_cake_reduce_lowers_hf():
    a = np.zeros((64, 64), dtype=np.float32)
    a[::2, ::2] = 1.0
    out = cake_reduce(a, 1.0)
    # high-freq checkerboard should be attenuated toward low-pass
    assert float(out.std()) < float(a.std())


def test_unmix_bare_skin_low_alpha():
    img = np.full((40, 40, 3), 170, dtype=np.uint8)
    img[:, :, 0] = 150
    img[:, :, 1] = 165
    img[:, :, 2] = 200
    _, _, alpha = unmix_makeup(img, np.ones((40, 40), dtype=np.float32))
    assert float(alpha.mean()) < 0.35


def test_user_alpha_overrides_automatic_specular_exclusion():
    """An explicit user paint mask remains authoritative over auto detection."""
    img = np.full((48, 48, 3), (180, 190, 210), dtype=np.uint8)
    user_alpha = np.zeros((48, 48), dtype=np.float32)
    user_alpha[16:32, 16:32] = 1.0
    _, _, alpha = unmix_makeup(
        img,
        np.ones((48, 48), dtype=np.float32),
        user_alpha=user_alpha,
    )
    assert float(alpha[20:28, 20:28].mean()) > 0.9


@pytest.mark.parametrize("tone", FITZPATRICK_BGR.values(), ids=list(FITZPATRICK_BGR))
def test_tone_relative_alpha_detects_foundation_across_skin_tones(tone):
    """A comparable density change must not disappear on darker skin."""
    _, foundation, disk = _foundation_disk(tone)
    alpha = estimate_makeup_alpha(foundation, np.ones(foundation.shape[:2], np.float32))
    assert float(alpha[disk].mean()) > 0.5
    assert float(alpha[~disk].mean()) < 0.05


@pytest.mark.parametrize("tone", FITZPATRICK_BGR.values(), ids=list(FITZPATRICK_BGR))
def test_tone_relative_alpha_keeps_bare_skin_below_false_positive_gate(tone):
    """Noisy bare skin remains a no-op for every Fitzpatrick reference tone."""
    bare, _, _ = _foundation_disk(tone)
    _, _, alpha = unmix_makeup(bare, np.ones(bare.shape[:2], np.float32))
    assert float(alpha.mean()) < 0.05


@pytest.mark.parametrize("tone", SPECULAR_BGR.values(), ids=list(SPECULAR_BGR))
def test_compact_specular_highlight_cannot_bleed_coverage_even(tone):
    """A real additive highlight is excluded before alpha smoothing can spread it."""
    bare, _, _ = _foundation_disk(tone)
    h, w = bare.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    spot = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= 8 ** 2
    highlighted = bare.astype(np.float32)
    highlighted[spot] = np.clip(highlighted[spot] + 90.0, 0, 255)
    highlighted = highlighted.astype(np.uint8)

    _, _, alpha = unmix_makeup(highlighted, np.ones((h, w), np.float32))
    out = apply_makeup_coverage_even(highlighted, np.ones((h, w), np.float32), 0.7)
    assert float(alpha[spot].mean()) < 0.05
    assert float(np.abs(out.astype(np.float32) - highlighted)[~spot].max()) <= 1.0

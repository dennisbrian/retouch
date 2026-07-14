"""P4 makeup_unmix minimal product slice."""

import numpy as np

from retouch.makeup_unmix import (
    apply_makeup_coverage_even,
    estimate_makeup_alpha,
    even_coverage,
)


def test_strength_zero_identity():
    img = np.random.RandomState(0).randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32)
    out = apply_makeup_coverage_even(img, mask, 0.0)
    assert out is img or np.array_equal(out, img)


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

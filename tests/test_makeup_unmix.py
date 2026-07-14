"""P4 makeup unmix — full product slice tests."""

import numpy as np

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

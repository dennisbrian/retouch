"""Automatic spot healing (retouch/spot_heal_auto.py).

Synthetic skin with pore-scale texture on light, medium and dark tones. Spots
are planted multiplicatively in linear light (a hemoglobin-like transmittance
for pimples, a brown one for moles), so the same spot reads the way it would
on each skin tone. No detector or model is needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.params import PROCESSING_PARAMS
from retouch.spot_heal_auto import detect_spots, feature_mask_from_regions, heal_spots
from retouch.utils import bgr_f32_to_lab_f32

SIZE = 640
FACE_W = 600.0

SKIN_TONES = {
    "light": (170, 190, 230),
    "medium": (105, 140, 190),
    "dark": (45, 65, 95),
}
PIMPLE = np.array([0.80, 0.70, 0.93])   # BGR transmittance at the centre
MOLE = np.array([0.30, 0.32, 0.40])


def _srgb_to_lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return 255.0 * np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


def _skin(tone="medium", seed=0):
    """(img float32, skin mask) with multiplicative pore texture."""
    rng = np.random.default_rng(seed)
    base = _srgb_to_lin(np.array(SKIN_TONES[tone], np.float64))
    tex = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 1.2)
    tex = tex / tex.std() * 0.035
    lin = base[None, None, :] * (1.0 + tex[:, :, None])
    img = _lin_to_srgb(lin).astype(np.float32)
    mask = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(mask, (SIZE // 2, SIZE // 2), (290, 300), 0, 0, 360, 1.0, -1)
    return img, mask


def _plant(img, cx, cy, r, trans, sharp=False):
    lin = _srgb_to_lin(img.astype(np.float64))
    R = int(r * 4)
    yy, xx = np.mgrid[cy - R:cy + R + 1, cx - R:cx + R + 1]
    d = np.hypot(xx - cx, yy - cy)
    prof = np.clip((r - d) / 1.5 + 0.5, 0, 1) if sharp else np.exp(-0.5 * (d / (0.6 * r)) ** 2)
    lin[cy - R:cy + R + 1, cx - R:cx + R + 1] *= trans[None, None, :] ** prof[:, :, None]
    return _lin_to_srgb(lin).astype(np.float32)


def _residual(out, clean, cx, cy, r):
    """Mean low-pass LAB distance to the unspotted image around a spot."""
    R = int(r * 3)
    sl = (slice(cy - R, cy + R + 1), slice(cx - R, cx + R + 1))
    sg = max(0.35 * r, 1.0)
    a = cv2.GaussianBlur(bgr_f32_to_lab_f32(out[sl].astype(np.float32)), (0, 0), sg)
    b = cv2.GaussianBlur(bgr_f32_to_lab_f32(clean[sl].astype(np.float32)), (0, 0), sg)
    q = int(r)
    c = R
    return float(np.sqrt(((a - b)[c - q:c + q + 1, c - q:c + q + 1] ** 2).sum(-1)).mean())


@pytest.mark.parametrize("tone", list(SKIN_TONES))
def test_pimple_is_healed_on_every_tone(tone):
    clean, mask = _skin(tone)
    spotted = _plant(clean, 300, 330, 9, PIMPLE)
    out, diags = heal_spots(spotted, mask, FACE_W, 60)
    before = _residual(spotted, clean, 300, 330, 9)
    after = _residual(out, clean, 300, 330, 9)
    assert diags["healed"] >= 1
    assert after < 0.4 * before, (tone, before, after)


@pytest.mark.parametrize("tone", list(SKIN_TONES))
def test_mole_is_kept(tone):
    clean, mask = _skin(tone)
    spotted = _plant(clean, 280, 300, 8, MOLE, sharp=True)
    out, diags = heal_spots(spotted, mask, FACE_W, 100)
    assert diags["kept_moles"] >= 1
    assert _residual(out, spotted, 280, 300, 8) < 0.5


def test_untouched_pixels_are_bit_identical():
    clean, mask = _skin("light")
    spotted = np.clip(np.round(_plant(clean, 320, 320, 9, PIMPLE)), 0, 255).astype(np.uint8)
    out, diags = heal_spots(spotted, mask, FACE_W, 60)
    assert out.dtype == np.uint8 and diags["healed"] >= 1
    changed = np.abs(out.astype(int) - spotted.astype(int)).max(-1) > 0
    yy, xx = np.nonzero(changed)
    # Every edit stays within a few spot radii of the spot.
    assert np.hypot(xx - 320, yy - 320).max() < 9 * 4


def test_zero_strength_is_a_no_op():
    clean, mask = _skin()
    spotted = _plant(clean, 300, 300, 9, PIMPLE)
    out, diags = heal_spots(spotted, mask, FACE_W, 0)
    assert out is spotted
    assert diags["healed"] == 0


def test_float_input_stays_float():
    clean, mask = _skin()
    spotted = _plant(clean, 300, 300, 9, PIMPLE)
    out, _ = heal_spots(spotted, mask, FACE_W, 60)
    assert out.dtype == np.float32


@pytest.mark.parametrize("tone", list(SKIN_TONES))
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_clean_skin_is_left_alone(tone, seed):
    clean, mask = _skin(tone, seed=seed)
    out, diags = heal_spots(clean, mask, FACE_W, 60)
    assert diags["healed"] == 0
    assert np.array_equal(out, clean)


def test_texture_continues_across_the_heal():
    clean, mask = _skin("medium", seed=5)
    spotted = _plant(clean, 300, 300, 12, PIMPLE)
    out, diags = heal_spots(spotted, mask, FACE_W, 60)
    assert diags["healed"] >= 1

    def hp_std(img, r):
        L = bgr_f32_to_lab_f32(img.astype(np.float32))[..., 0]
        hp = L - cv2.GaussianBlur(L, (0, 0), 3.0)
        return float(hp[300 - r:300 + r, 300 - r:300 + r].std())

    # A flat fill would drop the pore texture inside the spot to near zero.
    assert hp_std(out, 8) > 0.6 * hp_std(clean, 8)


def test_bright_spot_is_not_a_candidate():
    """Piercings, catchlights and specular dots are brighter, never healed."""
    clean, mask = _skin("medium")
    spotted = clean.copy()
    cv2.circle(spotted, (300, 300), 7, (235, 235, 235), -1)
    out, diags = heal_spots(spotted, mask, FACE_W, 60)
    assert diags["healed"] == 0
    assert np.array_equal(out, spotted)


def test_line_is_not_a_spot():
    """A wrinkle, lash shadow or hair strand is long and thin, not a spot."""
    clean, mask = _skin("light")
    spotted = clean.copy()
    cv2.line(spotted, (220, 300), (380, 310), (110, 120, 150), 3)
    out, diags = heal_spots(spotted, mask, FACE_W, 60)
    assert diags["healed"] == 0


def test_feature_margin_and_protect_mask():
    clean, mask = _skin("light")
    spotted = _plant(clean, 300, 300, 9, PIMPLE)
    feature = np.zeros_like(mask)
    cv2.circle(feature, (300, 318), 6, 1.0, -1)           # a feature right next to it
    assert detect_spots(spotted, mask, FACE_W, 60)[0]
    spots, _ = detect_spots(spotted, mask, FACE_W, 60, feature_mask=feature)
    assert not spots
    protect = np.zeros_like(mask)
    cv2.circle(protect, (300, 300), 20, 1.0, -1)
    out, diags = heal_spots(spotted, mask, FACE_W, 60, protect_mask=protect)
    assert diags["healed"] == 0 and np.array_equal(out, spotted)


def test_higher_strength_finds_at_least_as_many_spots():
    clean, mask = _skin("medium", seed=7)
    img = clean
    for i, (x, y) in enumerate([(200, 220), (400, 230), (220, 420), (410, 410), (300, 320)]):
        trans = 1.0 - (1.0 - PIMPLE) * (0.35 + 0.15 * i)
        img = _plant(img, x, y, 8, trans)
    counts = [len(detect_spots(img, mask, FACE_W, s)[0]) for s in (20, 60, 100)]
    assert counts == sorted(counts) and counts[-1] >= 3


def test_feature_mask_from_regions():
    class R:
        left_eye = np.zeros((10, 10), np.uint8)
        lips = np.full((10, 10), 255, np.uint8)
        hair = None

    m = feature_mask_from_regions(R())
    assert m.shape == (10, 10) and float(m.max()) == 1.0
    assert feature_mask_from_regions(object()) is None


def test_param_spec():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "spot_heal")
    assert spec.default == 0
    assert spec.cli_flag == "spot-heal"
    assert spec.recipe_key == "skin.spot_heal"
    assert (spec.min_val, spec.max_val) == (0, 100)

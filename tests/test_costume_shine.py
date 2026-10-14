"""Tests for Costume Shine (retouch/costume_shine.py).

Synthetic scenes with hand-built multiclass confidences, so no model is
needed. Exposure changes are simulated with a linear-light gain; that is a
stand-in for real darker or brighter photos, not corpus validation.
"""

import cv2
import numpy as np
import pytest

from retouch.costume_shine import apply_costume_shine
from retouch.params import PROCESSING_PARAMS

H, W = 360, 480
SKIN_BGR = np.array([150, 170, 215], np.float32) / 255.0
FACE = [(10, 10, 150, 60)]  # face width 150 px -> disc ~9 px, glint area ~9 px
STREAK_ROWS = (100, 160, 220)
STREAK_COLS = slice(300, 440)


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def _scene(material_bgr=(0.06, 0.05, 0.05), gain=1.0, bow=True, glints=True, fishnet=True, print_=False):
    """Left: skin. Right: glossy material with specular streaks, a white bow,
    tiny glints, a fishnet patch over skin and optionally a green print line."""
    rng = np.random.default_rng(3)
    lin = np.empty((H, W, 3), np.float32)
    lin[:, : W // 4] = _lin(SKIN_BGR)
    lin[:, W // 4:] = _lin(np.array(material_bgr, np.float32))
    lin *= 1.0 + 0.03 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.0)[..., None]
    yy = np.arange(H, dtype=np.float32)[:, None]
    for row in STREAK_ROWS:
        # A soft specular lobe of neutral light, about 0.03 face widths across
        # (the median width of the gated streaks on the two bunny photos), and
        # below clipping even at the 1.5x exposure test.
        lobe = np.exp(-0.5 * ((yy - row) / 1.5) ** 2) * 0.5
        lin[:, STREAK_COLS] += lobe[:, :, None] * np.ones((1, STREAK_COLS.stop - STREAK_COLS.start, 3))
    if bow:
        lin[250:320, 160:240] = 0.80  # white bow, 70 x 80 px
        lin[285:290, 240:290] = 0.80  # its ribbon tail, 5 px wide
    if glints:
        for x in range(170, 290, 30):
            lin[40:42, x:x + 2] = 0.9  # 2 x 2 studs
    if fishnet:
        net = np.zeros((H, W), bool)
        net[20:90, 330:450] = True
        threads = ((np.arange(W)[None, :] % 8) < 2) | ((np.arange(H)[:, None] % 8) < 2)
        cell = net & ~threads
        lin[cell] = _lin(SKIN_BGR)
    if print_:
        lin[330:334, 300:440] = (0.05, 0.6, 0.05)  # green print line (BGR), 4 px
    img = _srgb(np.clip(lin * gain, 0, 1)).astype(np.float32)
    probs = np.zeros((H, W, 6), np.float32)
    probs[:, : W // 4, 2] = 0.95
    probs[:, W // 4:, 4] = 0.95
    probs[..., 0] = 0.05
    return img, probs


def _run(img, probs, strength=1.0):
    return apply_costume_shine(
        img,
        strength,
        FACE,
        segment_classes=lambda u8: cv2.resize(probs, (u8.shape[1], u8.shape[0])),
        hair_full=None,
        person_mask=np.ones((H, W), np.float32),
    )


def _Y(img):
    return _lin(np.clip(img, 0, 1)).mean(axis=2)


def _streak_light(img, reference):
    """Light the streaks add over the same scene without them."""
    rows = np.concatenate([np.arange(r - 3, r + 4) for r in STREAK_ROWS])
    return float((_Y(img)[rows][:, STREAK_COLS] - _Y(reference)[rows][:, STREAK_COLS]).sum())


def _no_streaks(**kw):
    img, probs = _scene(**kw)
    import tests.test_costume_shine as me
    saved = me.STREAK_ROWS
    try:
        me.STREAK_ROWS = ()
        clean, _ = _scene(**kw)
    finally:
        me.STREAK_ROWS = saved
    return img, probs, clean


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "costume_shine")
    assert spec.default == 0
    assert spec.cli_flag == "costume-shine"
    assert spec.recipe_key == "fabric.costume_shine"


def test_zero_strength_is_identity():
    img, probs = _scene()
    out, diag = _run(img, probs, 0.0)
    assert out is img
    assert diag["reason"] == "off"


def test_streaks_are_calmed_and_scale_with_strength():
    img, probs, clean = _no_streaks()
    before = _streak_light(img, clean)
    full, diag = _run(img, probs, 1.0)
    half, _ = _run(img, probs, 0.5)
    assert diag["applied"]
    kept_full = _streak_light(full, clean) / before
    kept_half = _streak_light(half, clean) / before
    assert kept_full < 0.5, kept_full
    assert kept_full < kept_half < 0.85
    # Some gloss always stays.
    assert kept_full > 0.1


def test_skin_bow_ribbon_glints_and_fishnet_are_left_alone():
    img, probs = _scene()
    out, _ = _run(img, probs, 1.0)
    np.testing.assert_array_equal(out[:, : W // 4 - 20], img[:, : W // 4 - 20])  # skin
    assert np.abs(out[250:320, 160:290] - img[250:320, 160:290]).max() < 1e-6  # bow + ribbon
    assert np.abs(out[35:48, 165:295] - img[35:48, 165:295]).max() < 1e-6  # studs
    net = (slice(24, 86), slice(334, 446))
    assert np.abs(out[net] - img[net]).mean() < 0.002


def test_coloured_print_is_not_shine():
    img, probs = _scene(print_=True)
    out, _ = _run(img, probs, 1.0)
    assert np.abs(out[328:336, 300:440] - img[328:336, 300:440]).max() < 0.01


def test_streak_on_red_vinyl_is_calmed():
    img, probs, clean = _no_streaks(material_bgr=(0.08, 0.06, 0.55))
    out, _ = _run(img, probs, 1.0)
    assert _streak_light(out, clean) / _streak_light(img, clean) < 0.6


@pytest.mark.parametrize("gain", [0.45, 1.5])
def test_same_share_removed_at_other_exposures(gain):
    """Linear-light exposure change (simulated darker / brighter photo).

    The scene has almost no grain, so its texture spread sits at the gate's
    floor and a darker copy gates a slightly wider band; on the two bunny
    photos (grain above the floor) the share moved by 0.04 at most.
    """
    def share(g):
        img, probs, clean = _no_streaks(gain=g)
        out, _ = _run(img, probs, 1.0)
        return 1.0 - _streak_light(out, clean) / _streak_light(img, clean)
    assert abs(share(gain) - share(1.0)) < 0.16


def test_matte_costume_is_untouched():
    img, probs = _scene(bow=False, glints=False, fishnet=False)
    import tests.test_costume_shine as me
    saved = me.STREAK_ROWS
    try:
        me.STREAK_ROWS = ()
        matte, probs = _scene(bow=False, glints=False, fishnet=False)
    finally:
        me.STREAK_ROWS = saved
    out, diag = _run(matte, probs, 1.0)
    assert np.abs(out - matte).max() < 0.004


def test_engine_stage_is_registered_before_costume_clarity():
    from retouch import stage_wrappers
    from retouch.engine import ProcessingContext

    assert ProcessingContext().costume_shine == 0.0
    names = [cls.name for cls in (stage_wrappers.CostumeShineStage, stage_wrappers.CostumeClarityStage)]
    assert names == ["costume_shine", "costume_clarity"]
    src = open(stage_wrappers.__file__).read()
    assert src.index("registry.add(CostumeShineStage") < src.index("registry.add(CostumeClarityStage")

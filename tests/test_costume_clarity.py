"""Tests for Costume Clarity (retouch/costume_clarity.py).

Synthetic scenes with hand-built multiclass confidences, so no model is
needed. Darker skin and darker costumes are simulated with a linear-light
gain and a recolour (no real darker-skin cosplay photo in the corpus yet);
that is a stand-in, not real-corpus validation.
"""

import cv2
import numpy as np
import pytest

from retouch.costume_clarity import apply_costume_clarity
from retouch.params import PROCESSING_PARAMS

H, W = 240, 320
SKIN_BGR = np.array([150, 170, 215], np.float32) / 255.0  # light skin
COSTUME_BGR = np.array([60, 40, 140], np.float32) / 255.0  # dark red fabric


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def _weave(seed=0):
    """Fabric-like texture: a fine weave plus mid-scale panels, in [-1, 1]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng(seed)
    t = 0.5 * np.sin(xx * 1.3) * np.sin(yy * 1.3) + 0.5 * np.sign(np.sin(xx / 9.0))
    return t + 0.05 * rng.standard_normal((H, W)).astype(np.float32)


def _scene(skin_bgr=SKIN_BGR, costume_bgr=COSTUME_BGR, gain=1.0):
    """Left half skin (soft texture), right half costume (weave)."""
    img = np.empty((H, W, 3), np.float32)
    img[:, : W // 2] = skin_bgr
    img[:, W // 2:] = costume_bgr
    rng = np.random.default_rng(1)
    skin_tex = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.5)
    tex = np.where(np.arange(W)[None, :] < W // 2, 0.02 * skin_tex, 0.06 * _weave())
    img = img * (1.0 + tex[..., None])
    img = _srgb(_lin(np.clip(img, 0, 1)) * gain).astype(np.float32)
    probs = np.zeros((H, W, 6), np.float32)
    probs[:, : W // 2, 2] = 0.95
    probs[:, W // 2:, 4] = 0.95
    probs[..., 0] = 0.05
    return img, probs


def _run(img, probs, strength=0.6, person=None, face_skin=None):
    return apply_costume_clarity(
        img,
        strength,
        [],
        segment_classes=lambda u8: cv2.resize(probs, (u8.shape[1], u8.shape[0])),
        hair_full=None,
        person_mask=np.ones((H, W), np.float32) if person is None else person,
        face_skin=face_skin,
    )


def _L(img):
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)[..., 0]


def _detail(L, sl):
    hp = L - cv2.GaussianBlur(L, (0, 0), 4)
    return float(hp[sl].std())


INNER_COSTUME = (slice(40, H - 40), slice(W // 2 + 40, W - 40))
INNER_SKIN = (slice(40, H - 40), slice(20, W // 2 - 40))


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "costume_clarity")
    assert spec.default == 0
    assert spec.cli_flag == "costume-clarity"
    assert spec.recipe_key == "fabric.costume_clarity"


def test_zero_strength_is_identity():
    img, probs = _scene()
    out, diag = _run(img, probs, strength=0.0)
    assert out is img
    assert diag["reason"] == "off"


def test_no_segmentation_is_identity():
    img, _ = _scene()
    out, diag = apply_costume_clarity(img, 0.6, [], segment_classes=None, person_mask=None)
    assert out is img
    assert diag["reason"] == "no_segmentation"


def test_costume_gains_detail_skin_untouched():
    img, probs = _scene()
    out, diag = _run(img, probs)
    assert diag["applied"]
    L0, L1 = _L(img), _L(out)
    assert _detail(L1, INNER_COSTUME) > 1.2 * _detail(L0, INNER_COSTUME)
    # Skin is bit-identical away from the feathered boundary.
    assert np.array_equal(out[INNER_SKIN], img[INNER_SKIN])
    # And nothing on the skin side moves more than a hair even at the edge.
    assert np.abs(L1[:, : W // 2] - L0[:, : W // 2]).max() < 0.5


def test_costume_colour_kept():
    img, probs = _scene()
    out, _ = _run(img, probs, strength=1.0)
    lab0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[INNER_COSTUME]
    lab1 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[INNER_COSTUME]
    # Mean a*/b* move by well under 1 unit: lightness detail, not a recolour.
    assert np.abs(lab1[..., 1:].mean(axis=(0, 1)) - lab0[..., 1:].mean(axis=(0, 1))).max() < 1.0


@pytest.mark.parametrize("gain", [0.25, 2.0])
def test_tone_invariance_relative_effect(gain):
    """Simulated darker (and lighter) exposure of skin and costume alike."""
    img, probs = _scene()
    out, _ = _run(img, probs)
    ref = _detail(_L(out), INNER_COSTUME) / _detail(_L(img), INNER_COSTUME)
    img2, probs2 = _scene(gain=gain)
    out2, diag2 = _run(img2, probs2)
    rel = _detail(_L(out2), INNER_COSTUME) / _detail(_L(img2), INNER_COSTUME)
    assert diag2["applied"]
    assert rel > 1.15
    assert abs(rel - ref) / (ref - 1.0) < 0.5
    assert np.array_equal(out2[INNER_SKIN], img2[INNER_SKIN])


def test_darker_skin_left_alone():
    """Simulated deep skin tone (linear gain 0.12 on the skin only)."""
    dark_skin = _srgb(_lin(SKIN_BGR) * np.array([0.10, 0.12, 0.16])).astype(np.float32)
    img, probs = _scene(skin_bgr=dark_skin)
    out, diag = _run(img, probs)
    assert diag["applied"]
    assert np.array_equal(out[INNER_SKIN], img[INNER_SKIN])
    assert _detail(_L(out), INNER_COSTUME) > 1.2 * _detail(_L(img), INNER_COSTUME)


def test_black_glove_labelled_skin_still_gets_clarity():
    """Segmenter calls a black glove 'body skin'; its colour says otherwise."""
    img, probs = _scene()
    glove = (slice(60, 180), slice(W // 2 + 20, W - 20))
    img[glove] = 0.06 * (1.0 + 0.3 * _weave(3)[glove][..., None])
    probs[glove] = 0
    probs[glove][..., 2] = 0.95
    probs[glove + (2,)] = 0.95
    out, _ = _run(img, probs)
    inner = (slice(80, 160), slice(W // 2 + 50, W - 50))
    assert _detail(_L(out), inner) > 1.15 * _detail(_L(img), inner)


def test_skin_through_fishnet_kept():
    """Costume-labelled fishnet over skin: mesh lines change, holes don't."""
    img, probs = _scene()
    yy, xx = np.mgrid[0:H, 0:W]
    mesh = ((xx + yy) % 12 < 2) | ((xx - yy) % 12 < 2)
    right = xx >= W // 2
    rng = np.random.default_rng(5)
    skin_tex = 0.03 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.0)
    img[right] = (SKIN_BGR * (1.0 + skin_tex[..., None]))[right]
    img[mesh & right] = 0.05
    out, _ = _run(img, probs, strength=1.0)
    d = np.abs(_L(out) - _L(img))
    inner = np.zeros((H, W), bool)
    inner[INNER_COSTUME] = True
    holes = inner & ~cv2.dilate(mesh.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lines = inner & mesh
    assert d[holes].mean() < 0.3
    assert d[lines].mean() > 4 * d[holes].mean()


def test_noise_not_amplified():
    """A flat costume with only sensor noise: the texture band is cored."""
    img, probs = _scene()
    rng = np.random.default_rng(9)
    img[:, W // 2:] = COSTUME_BGR
    # Sensor noise covers the whole frame, skin included.
    img = img + 0.01 * rng.standard_normal((H, W, 1)).astype(np.float32)
    out, _ = _run(img, probs, strength=1.0)
    fine = lambda L: (L - cv2.GaussianBlur(L, (0, 0), 1.0))[INNER_COSTUME].std()
    assert fine(_L(out)) < 1.2 * fine(_L(img))


def test_painted_neutral_skin_skips_colour_veto():
    """White face paint as the skin model would match a white costume."""
    white = np.array([0.93, 0.93, 0.93], np.float32)
    img, probs = _scene(skin_bgr=white, costume_bgr=white * 0.98)
    out, diag = _run(img, probs)
    assert diag.get("skin_colour_veto") == "skipped_neutral_skin"
    assert _detail(_L(out), INNER_COSTUME) > 1.1 * _detail(_L(img), INNER_COSTUME)
    assert np.array_equal(out[INNER_SKIN], img[INNER_SKIN])


def test_engine_stage_wired(monkeypatch):
    import retouch.costume_clarity as cc
    from retouch.engine import ProcessingContext, RetouchEngine
    from retouch.stage_wrappers import CostumeClarityStage, build_global_registry
    from retouch.stages import PipelineState

    eng = RetouchEngine.__new__(RetouchEngine)
    names = [s.name for s in build_global_registry(eng)]
    assert names.index("costume_clarity") < names.index("grading")

    calls = {}

    def fake(img, strength, boxes, **kw):
        calls["strength"] = strength
        return img + 0.01, {"applied": True}

    monkeypatch.setattr(cc, "apply_costume_clarity", fake)

    class _Parser:
        _segment_classes = staticmethod(lambda x: None)
        parse_hair_full_image = staticmethod(lambda x: None)

    eng._parser = _Parser()
    img = np.full((32, 32, 3), 0.5, np.float32)
    ctx = ProcessingContext()
    stage = CostumeClarityStage(eng)
    state = PipelineState(img=img, ctx=ctx, h_img=32, w_img=32, person_mask=None,
                          acc_skin=None, acc_skin_hair=None, acc_lips=None,
                          acc_sharpen=None, acc_hair_only=None, faces=[], style_ref=None)
    assert not stage.enabled(state)
    ctx.costume_clarity = 60
    assert stage.enabled(state)
    out = eng._stage_costume_clarity(img, ctx, [], None, None)
    assert calls["strength"] == pytest.approx(0.6)
    assert np.allclose(out, img + 0.01)
    assert ctx._runtime_diagnostics["costume_clarity"]["applied"]

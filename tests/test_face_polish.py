"""Face polish (porcelain reference look) and the two face ops it rides on.

Covers:
* ``perf_optimizations._face_polish_ctx`` — the one-control macro only raises
  component floors, keeps explicit stronger values and a user-chosen
  specular finish, and never mutates the shared context.
* ``SkinProcessor.shine_removal`` v2 — acts on realistic highlights (15-25 L
  above the local skin, keeping ~85% of its chroma), which v1's
  "chroma < 40% of skin" gate never matched on real portraits. Checked on
  lighter and darker skin.
* ``SkinProcessor.apply_specular_finish`` — measures the specular baseline
  over the skin mask, not the whole crop, so a dark background no longer
  turns ordinary skin into "specular" and darkens the face.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.params import PROCESSING_PARAMS
from retouch.perf_optimizations import _face_polish_ctx
from retouch.skin import SkinProcessor


def _ctx(**kw):
    base = dict(
        face_polish=0,
        shine_removal=0,
        shadow_lift=0,
        face_exposure=0.0,
        specular_finish="matte",
        specular_finish_strength=0.5,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestFacePolishCtx:
    def test_zero_returns_same_object(self):
        ctx = _ctx()
        assert _face_polish_ctx(ctx) is ctx

    def test_raises_component_floors_and_scales_linearly(self):
        half = _face_polish_ctx(_ctx(face_polish=50))
        full = _face_polish_ctx(_ctx(face_polish=100))
        for name in ("shine_removal", "shadow_lift", "face_exposure"):
            assert getattr(half, name) > 0
            assert getattr(full, name) == pytest.approx(2 * getattr(half, name), abs=1)
        # The powder finish greyed broad lit cheeks on pale skin; not used.
        assert full.specular_finish == "matte"

    def test_explicit_stronger_value_wins(self):
        out = _face_polish_ctx(_ctx(face_polish=100, shadow_lift=95))
        assert out.shadow_lift == 95

    def test_context_is_not_mutated(self):
        ctx = _ctx(face_polish=100)
        _face_polish_ctx(ctx)
        assert ctx.shine_removal == 0 and ctx.specular_finish == "matte"

    def test_int_components_stay_int(self):
        out = _face_polish_ctx(_ctx(face_polish=37))
        assert isinstance(out.shine_removal, int)

    def test_registered_param(self):
        spec = next(p for p in PROCESSING_PARAMS if p.name == "face_polish")
        assert spec.default == 0
        assert spec.cli_flag == "face-polish"
        assert spec.recipe_key == "skin.face_polish"


def _pale_face_with_hotspot(skin_bgr, hot_gain=22.0, size=512):
    """Soft, slightly desaturated highlight on a lit skin gradient.

    The highlight is additive near-white light (dichromatic model): it
    brightens the skin by ``hot_gain`` L levels at its peak and pulls the
    colour ~15% toward grey, as measured on real portrait highlights. A
    gentle left-right lighting gradient stands in for the lit side of a face,
    which must not count as shine. Scale follows a real face ROI (~2x the
    face width): a hot spot a few percent of the crop wide.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.asarray(skin_bgr, np.float32)
    img *= (0.92 + 0.16 * xx / w)[..., None]  # lit side
    g = np.exp(-(((xx - 256) ** 2 + (yy - 220) ** 2) / (2 * 10.0 ** 2)))[..., None]
    mean = img.mean(axis=2, keepdims=True)
    grey = img + (mean - img) * 0.15
    img = img * (1 - g) + (grey + hot_gain) * g
    mask = np.zeros((h, w), np.float32)
    mask[80:432, 80:432] = 1.0
    return np.clip(img, 0, 255), mask


def _L(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


class TestShineRemovalRealisticHighlight:
    @pytest.mark.parametrize(
        "skin_bgr",
        [(170, 185, 205), (60, 80, 110)],
        ids=["lighter", "darker"],
    )
    def test_realistic_highlight_is_toned_down(self, skin_bgr):
        img, mask = _pale_face_with_hotspot(skin_bgr)
        out = SkinProcessor().shine_removal(img, mask, strength=70)
        L0, L1 = _L(img), _L(out)
        peak = (slice(216, 225), slice(252, 261))
        drop = float(L0[peak].mean() - L1[peak].mean())
        assert drop > 5.0, f"highlight barely touched: {drop:.2f} L"
        # Satin, not flat: some of the highlight remains.
        assert L1[peak].mean() > L1[150:170, 248:264].mean()

    def test_lit_side_of_face_is_not_shine(self):
        img, mask = _pale_face_with_hotspot((170, 185, 205))
        out = SkinProcessor().shine_removal(img, mask, strength=100)
        lit = (slice(360, 420), slice(380, 425))
        assert float(np.abs(_L(out)[lit] - _L(img)[lit]).mean()) < 0.5


class TestSpecularFinishUsesSkinBaseline:
    def test_matte_does_not_darken_skin_next_to_dark_background(self):
        h = w = 256
        img = np.full((h, w, 3), 20.0, np.float32)  # dark hair/background
        img[48:208, 64:192] = (175, 190, 210)  # plain skin, no highlight
        mask = np.zeros((h, w), np.float32)
        mask[48:208, 64:192] = 1.0
        out = SkinProcessor().apply_specular_finish(img, mask, mode="matte", strength=0.5)
        change = float(np.abs(out - img)[56:200, 72:184].mean())
        assert change < 2.0, f"plain skin darkened by {change:.1f}"

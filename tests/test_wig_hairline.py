"""Wig hairline blend: hairline detection, rejections, feather, band tone, wiring.

Synthetic scenes only (a skin-toned head with a wig block above a hairline).
None of the cosplay photos at hand shows a hairline (every wig has full
bangs), so darker and lighter skin are simulated by scaling the scene's
lightness, not measured on real photos.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch import wig_hairline as wh
from retouch.color_science import oklab_to_bgr
from retouch.prosthetic_blend import _BROWS, _FACE_OVAL

H, W = 460, 420
CX, CY = 210.0, 270.0
RX, RY = 95.0, 125.0


class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Landmarks:
    def __init__(self, pts):
        self.landmark = [_P(x / W, y / H) for x, y in pts]


class _Face:
    def __init__(self):
        pts = np.tile([[CX, CY]], (478, 1)).astype(np.float32)
        for k, idx in enumerate(_FACE_OVAL):
            t = np.radians(-90.0 + 10.0 * k)
            pts[idx] = (CX + RX * np.cos(t), CY + RY * np.sin(t))
        brow_x = np.linspace(0.15, 0.6, 10) * RX
        for i, idx in enumerate(_BROWS[:10]):
            pts[idx] = (CX - brow_x[i], CY - 0.35 * RY)
        for i, idx in enumerate(_BROWS[10:]):
            pts[idx] = (CX + brow_x[i], CY - 0.35 * RY)
        self.landmarks = _Landmarks(pts)
        self.bbox = (int(CX - RX), int(CY - RY), int(2 * RX), int(2 * RY))


def _ellipse(cx, cy, rx, ry):
    yy, xx = np.ogrid[:H, :W]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


YY, XX = np.mgrid[:H, :W]
FACE = _ellipse(CX, CY, RX, RY)
HEAD = FACE | _ellipse(CX, CY - 0.55 * RY, 1.05 * RX, 0.95 * RY)
# Wig front: a hard line across the upper forehead, well above the brows.
HAIRLINE_Y = CY - 0.62 * RY
WIG = HEAD & (YY < HAIRLINE_Y)
# Full bangs: the wig comes down over the brows.
BANGS = HEAD & (YY < CY - 0.30 * RY)
SKIN = HEAD & ~WIG


def _scene(wig=WIG, L=0.74, a=0.035, b=0.05, wig_lab=(0.80, 0.03, -0.04),
           lace=0.0, backdrop=(0.45, -0.02, -0.09), seed=0):
    """OKLab scene: textured skin head, strand-textured wig, optional lace band."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W, 3), np.float32)
    lab[...] = backdrop
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
    head = HEAD & ~wig
    lab[..., 0][head] = L * (1.0 + 0.03 * tex[head])
    lab[..., 1][head] = a * (1.0 + 0.03 * tex[head])
    lab[..., 2][head] = b * (1.0 + 0.03 * tex[head])
    strands = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=0.7, sigmaY=6)
    strands /= max(float(strands.std()), 1e-6)
    scale = L / 0.74
    lab[..., 0][wig] = wig_lab[0] * scale * (1.0 + 0.04 * strands[wig])
    lab[..., 1][wig] = wig_lab[1]
    lab[..., 2][wig] = wig_lab[2]
    if lace:
        d = cv2.distanceTransform((~wig).astype(np.uint8), cv2.DIST_L2, 5)
        band = head & (d > 0) & (d <= 4)
        lab[..., 0][band] += lace * scale
        lab[..., 2][band] -= 0.2 * lace
    return lab


def _bgr01(lab):
    return oklab_to_bgr(lab, float32_out=True) / 255.0


def _run(img, strength=1.0, wig=WIG, hair=None):
    skin = (FACE & ~wig).astype(np.float32)
    person = HEAD.astype(np.float32)
    hair = wig.astype(np.float32) if hair is None else hair
    return wh.apply_wig_hairline(img, [_Face()], skin, hair, strength, person_mask=person)


def _edge_width(img):
    """Median 10-90% rise (px) of the OKLab colour across the hairline."""
    lab = wh._to_oklab(img)
    y0 = int(HAIRLINE_Y)
    widths = []
    for x in range(int(CX - 0.5 * RX), int(CX + 0.5 * RX), 4):
        col = lab[y0 - 25:y0 + 25, x]
        hair = col[:8].mean(0)
        skin = col[-8:].mean(0)
        t = (col - hair) @ (skin - hair) / max(float(np.sum((skin - hair) ** 2)), 1e-9)
        i10 = int(np.argmax(t > 0.1))
        i90 = int(np.argmax(t > 0.9))
        widths.append(i90 - i10)
    return float(np.median(widths))


def _lace_offset(img):
    """Lightness of the 4 px band below the hairline minus the forehead below it."""
    lab = wh._to_oklab(img)
    y0 = int(np.ceil(HAIRLINE_Y))
    xs = slice(int(CX - 0.5 * RX), int(CX + 0.5 * RX))
    return float(lab[y0:y0 + 4, xs, 0].mean() - lab[y0 + 14:y0 + 24, xs, 0].mean())


class TestOff:
    def test_zero_strength_is_identity(self):
        img = _bgr01(_scene(lace=0.06))
        out, diag = _run(img, strength=0.0)
        assert out is img
        assert diag["reason"] == "off"

    def test_no_hair_mask_is_identity(self):
        img = _bgr01(_scene())
        out, diag = wh.apply_wig_hairline(img, [_Face()], FACE.astype(np.float32), None, 1.0)
        assert out is img
        assert diag["reason"] == "no_hair_mask"


class TestFindsHairline:
    def test_hard_wig_front_found(self):
        img = _bgr01(_scene())
        _, diag = _run(img)
        assert diag["applied"]
        assert diag["faces"][0]["hairline_len_fw"] > 0.4

    def test_bangs_to_the_brows_left_alone(self):
        img = _bgr01(_scene(wig=BANGS))
        out, diag = _run(img, wig=BANGS)
        assert not diag["applied"]
        assert np.array_equal(out, img)

    def test_skin_coloured_wall_beside_the_wig_is_not_a_hairline(self):
        # Backdrop the colour of the skin: the wig's outer edge against it
        # faces away from the face and must stay crisp.
        img = _bgr01(_scene(backdrop=(0.74, 0.035, 0.05)))
        out, diag = _run(img)
        assert diag["applied"]
        outer = HEAD & (YY < HAIRLINE_Y - 40)
        ring = cv2.dilate(outer.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ring &= ~cv2.erode(outer.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        assert float(np.abs(out - img)[ring].max()) < 1e-6

    def test_coarse_mask_snaps_to_the_real_edge(self):
        # A segmenter mask that stops 5 px short of the wig front still finds
        # it, and nothing deep inside the wig changes.
        img = _bgr01(_scene())
        coarse = WIG & (YY < HAIRLINE_Y - 5)
        out, diag = _run(img, hair=coarse.astype(np.float32))
        assert diag["applied"]
        assert _edge_width(out) > _edge_width(img) + 2
        deep = WIG & (YY < HAIRLINE_Y - 30)
        assert float(np.abs(out - img)[deep].max()) < 1e-6


class TestBlend:
    def test_hard_edge_is_feathered(self):
        img = _bgr01(_scene())
        before = _edge_width(img)
        mid = _edge_width(_run(img, 0.5)[0])
        full = _edge_width(_run(img, 1.0)[0])
        assert before <= 5
        assert before < mid < full
        assert full >= 6

    def test_lace_band_toned_to_forehead(self):
        img = _bgr01(_scene(lace=0.06))
        before = _lace_offset(img)
        after = _lace_offset(_run(img, 1.0)[0])
        assert before > 0.04
        assert abs(after) < 0.6 * before

    def test_change_stays_near_the_hairline(self):
        img = _bgr01(_scene(lace=0.06))
        out, _ = _run(img, 1.0)
        far = np.abs(YY - HAIRLINE_Y) > 0.12 * 2 * RX
        assert float(np.abs(out - img)[far].max()) < 1e-6

    def test_fine_texture_kept(self):
        img = _bgr01(_scene())
        out, _ = _run(img, 1.0)
        g0 = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_BGR2GRAY)
        hp = lambda g: g - cv2.GaussianBlur(g, (0, 0), 1.0)
        band = (np.abs(YY - HAIRLINE_Y) < 10) & (np.abs(XX - CX) < 0.5 * RX)
        band &= np.abs(YY - HAIRLINE_Y) > 3  # the step itself is meant to soften
        assert hp(g1)[band].std() > 0.7 * hp(g0)[band].std()

    @pytest.mark.parametrize("L", [0.42, 0.74, 0.86])
    def test_tone_invariant(self, L):
        # Simulated darker and lighter skin (same scene scaled in lightness):
        # the hairline is found and feathered the same way.
        img = _bgr01(_scene(L=L))
        out, diag = _run(img, 1.0)
        assert diag["applied"]
        assert _edge_width(out) >= _edge_width(img) + 3
        laced = _bgr01(_scene(L=L, lace=0.06 * L / 0.74))
        assert abs(_lace_offset(_run(laced, 1.0)[0])) < 0.6 * _lace_offset(laced)

    def test_uint8_input_round_trips(self):
        img = (np.clip(_bgr01(_scene()), 0, 1) * 255 + 0.5).astype(np.uint8)
        out, diag = _run(img, 1.0)
        assert out.dtype == np.uint8 and diag["applied"]


class TestEngineWiring:
    def test_param_registered(self):
        from retouch.params import param_names
        assert "cosplay_wig_lace_blend" in param_names()

    def test_cosplay_moat_stage_calls_blend(self, monkeypatch):
        from retouch.engine import ProcessingContext, RetouchEngine
        called = {}

        def fake(img, faces, acc_skin, hair, strength, person_mask=None):
            called["strength"] = strength
            called["hair"] = hair
            return img, {"applied": False, "faces": []}

        monkeypatch.setattr(wh, "apply_wig_hairline", fake)
        eng = RetouchEngine.__new__(RetouchEngine)
        ctx = ProcessingContext()
        ctx.cosplay_wig_lace_blend = 40.0
        img = _bgr01(_scene())
        out = RetouchEngine._stage_cosplay_moat(
            eng, img, ctx, (WIG | FACE).astype(np.float32), HEAD.astype(np.float32),
            faces=[_Face()], acc_skin=FACE.astype(np.float32),
            acc_hair_only=WIG.astype(np.float32),
        )
        assert called["strength"] == pytest.approx(0.4)
        assert np.array_equal(called["hair"], WIG.astype(np.float32))
        # With only the wig-lace blend on there is no 8-bit round trip.
        assert out is img

    def test_engine_end_to_end_on_synthetic_face(self):
        from retouch.engine import ProcessingContext, RetouchEngine
        eng = RetouchEngine.__new__(RetouchEngine)
        ctx = ProcessingContext()
        ctx.cosplay_wig_lace_blend = 100.0
        img = _bgr01(_scene())
        out = RetouchEngine._stage_cosplay_moat(
            eng, img, ctx, None, HEAD.astype(np.float32), faces=[_Face()],
            acc_skin=(FACE & ~WIG).astype(np.float32), acc_hair_only=WIG.astype(np.float32),
        )
        assert ctx._runtime_diagnostics["wig_hairline"]["applied"]
        assert _edge_width(out) > _edge_width(img) + 3

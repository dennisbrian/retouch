"""Prosthetic edge blending: seam detection, rejections, blend, engine wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch import prosthetic_blend as pb
from retouch.color_science import oklab_to_bgr

H, W = 420, 420
CX, CY = 210.0, 250.0
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
        # Face oval clockwise from the forehead top (index 10) round an
        # ellipse; 454 and 234 land at the sides, 152 at the chin.
        for k, idx in enumerate(pb._FACE_OVAL):
            t = np.radians(-90.0 + 10.0 * k)
            pts[idx] = (CX + RX * np.cos(t), CY + RY * np.sin(t))
        brow_x = np.linspace(0.15, 0.6, 10) * RX
        for i, idx in enumerate(pb._BROWS[:10]):
            pts[idx] = (CX - brow_x[i], CY - 0.35 * RY)
        for i, idx in enumerate(pb._BROWS[10:]):
            pts[idx] = (CX + brow_x[i], CY - 0.35 * RY)
        self.landmarks = _Landmarks(pts)
        self.bbox = (int(CX - RX), int(CY - RY), int(2 * RX), int(2 * RY))


def _ellipse(cx, cy, rx, ry):
    yy, xx = np.ogrid[:H, :W]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


FACE = _ellipse(CX, CY, RX, RY)
# The head reaches above the face oval: the forehead up to the hairline.
HEAD = FACE | _ellipse(CX, CY - 0.55 * RY, 0.95 * RX, 0.9 * RY)
PLATE = _ellipse(CX - 0.1 * RX, CY - 0.8 * RY, 0.4 * RX, 0.17 * RY)


def _skin_lab(L=0.72, a=0.045, b=0.055, seed=0, chroma_tex=0.03):
    """Skin-toned head with fine texture on a blue backdrop (OKLab)."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = 0.45, -0.02, -0.09
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
    lab[..., 0][HEAD] = L * (1.0 + 0.03 * tex[HEAD])
    lab[..., 1][HEAD] = a * (1.0 + chroma_tex * tex[HEAD])
    lab[..., 2][HEAD] = b * (1.0 + chroma_tex * tex[HEAD])
    return lab


def _plant_plate(lab, offset=(0.025, -0.006, 0.014), lip=True):
    """A skin-toned appliance on the forehead: slightly off colour, sharp edge."""
    out = lab.copy()
    L = float(np.median(lab[..., 0][FACE]))
    scale = L / 0.72  # the same visual mismatch on lighter or deeper skin
    out[PLATE] += np.array(offset, np.float32) * scale
    if lip:
        p = PLATE.astype(np.uint8)
        ring_out = (cv2.dilate(p, np.ones((5, 5), np.uint8)) > 0) & ~PLATE
        ring_in = PLATE & ~(cv2.erode(p, np.ones((5, 5), np.uint8)) > 0)
        out[..., 0][ring_out] -= 0.04 * scale
        out[..., 0][ring_in] += 0.02 * scale
    return out


def _bgr01(lab):
    return oklab_to_bgr(lab, float32_out=True) / 255.0


def _run(img, strength=1.0, hair=None, ref=None):
    skin = FACE.astype(np.float32)
    person = HEAD.astype(np.float32)
    return pb.apply_prosthetic_blend(
        img, [_Face()], skin, strength, person_mask=person, hair_mask=hair, ref=ref
    )


def _edge_step(img):
    """Colour difference just inside vs just outside the plate edge (x100)."""
    p = PLATE.astype(np.uint8)
    inner = (cv2.erode(p, np.ones((5, 5))) > 0) & ~(cv2.erode(p, np.ones((17, 17))) > 0)
    outer = (cv2.dilate(p, np.ones((17, 17))) > 0) & ~(cv2.dilate(p, np.ones((5, 5))) > 0)
    lab = pb._to_oklab(img)
    return float(np.linalg.norm(lab[inner].mean(0) - lab[outer].mean(0))) * 100


@pytest.mark.parametrize("L,a,b", [(0.78, 0.040, 0.050), (0.45, 0.050, 0.055)])
def test_finds_and_softens_forehead_seam_on_light_and_deep_skin(L, a, b):
    clean = _bgr01(_skin_lab(L, a, b))
    img = _bgr01(_plant_plate(_skin_lab(L, a, b)))
    out, diag = _run(img)
    face = diag["faces"][0]
    assert diag["applied"] is True
    assert face["seams"] >= 1
    before, after, base = _edge_step(img), _edge_step(out), _edge_step(clean)
    # Most of the step across the appliance edge is gone.
    assert after - base < 0.5 * (before - base)
    # Pixels far from the plate are untouched.
    far = ~(cv2.dilate(PLATE.astype(np.uint8), np.ones((101, 101), np.uint8)) > 0)
    assert np.array_equal(out[far], img[far])


def test_clean_skin_is_left_alone():
    img = _bgr01(_skin_lab())
    out, diag = _run(img)
    assert out is img
    assert diag["applied"] is False
    assert diag["faces"][0]["seams"] == 0


def test_strength_zero_is_identity():
    img = _bgr01(_plant_plate(_skin_lab()))
    out, diag = _run(img, strength=0.0)
    assert out is img
    assert diag["reason"] == "off"


def test_soft_blush_is_not_a_seam():
    lab = _skin_lab()
    yy, xx = np.mgrid[:H, :W]
    blob = np.exp(-(((xx - CX) / 40.0) ** 2 + ((yy - (CY - 0.8 * RY)) / 20.0) ** 2)).astype(np.float32)
    lab[..., 1] += 0.03 * blob * HEAD
    out, diag = _run(_bgr01(lab))
    assert diag["faces"][0]["seams"] == 0


def test_lightness_only_line_is_not_a_seam():
    # A long sharp wrinkle-like line: lightness changes, colour does not.
    lab = _skin_lab()
    line = np.zeros((H, W), np.uint8)
    cv2.ellipse(line, (int(CX), int(CY - 0.7 * RY)), (60, 12), 0, 200, 340, 1, 3)
    m = line.astype(bool) & HEAD
    lab[..., 0][m] *= 0.8
    lab[..., 1][m] *= 0.8
    lab[..., 2][m] *= 0.8
    out, diag = _run(_bgr01(lab))
    assert diag["faces"][0]["seams"] == 0


def test_grey_hairline_is_not_a_seam():
    # Pale blond hair the hair mask missed, with a sharp edge onto the
    # forehead: close to skin in hue and lightness, but much less saturated.
    lab = _skin_lab(chroma_tex=0.9)
    hair = HEAD & ~FACE & (np.mgrid[:H, :W][0] < CY - 0.85 * RY)
    lab[..., 1][hair] = 0.015
    lab[..., 2][hair] = 0.020
    out, diag = _run(_bgr01(lab))
    assert diag["faces"][0]["seams"] == 0


def test_seams_found_on_reference_blended_on_image():
    # Seams are found on the pre-retouch reference; the blend lands on the
    # retouched image passed as ``img``.
    ref = _bgr01(_plant_plate(_skin_lab()))
    img = np.clip(ref * 0.98, 0.0, 1.0).astype(np.float32)
    out, diag = _run(img, ref=ref)
    assert diag["applied"] is True
    assert _edge_step(out) < _edge_step(img)


def test_param_registered_off_by_default():
    from retouch.params import PROCESSING_PARAMS

    spec = next(p for p in PROCESSING_PARAMS if p.name == "prosthetic_blend")
    assert spec.default == 0
    assert spec.recipe_key == "skin.prosthetic_blend"
    assert spec.cli_flag == "prosthetic-blend"


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    return RetouchEngine()


def test_engine_prosthetic_blend_off_by_default_and_noop_without_seams(engine):
    from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

    img = make_synthetic_face_image()
    h, w = img.shape[:2]
    base = engine.process(img, recipe="cosplay_clear_v1", face_contexts=[make_face_context(w, h, img)])
    on = engine.process(
        img, recipe="cosplay_clear_v1", face_contexts=[make_face_context(w, h, img)], prosthetic_blend=80
    )
    assert np.array_equal(np.asarray(base), np.asarray(on))
    assert "prosthetic_blend" not in base.runtime_diagnostics
    assert on.runtime_diagnostics["prosthetic_blend"]["applied"] is False

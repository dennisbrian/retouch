"""Body paint support: detection, colour lock, patchy-paint evening, engine wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch import body_paint as bp
from retouch.color_science import bgr_to_oklab, oklab_to_bgr

H, W = 160, 160


def _face_mask(h=H, w=W):
    yy, xx = np.ogrid[:h, :w]
    return (((xx - w / 2) / (w * 0.3)) ** 2 + ((yy - h / 2) / (h * 0.38)) ** 2) <= 1.0


def _lab_image(L, C, hue_deg, mask=None, base=None, seed=0):
    """Uniform OKLab patch (L, C, hue) inside ``mask`` over ``base`` elsewhere."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W, 3), np.float32)
    if base is not None:
        lab[:] = base
    else:
        lab[..., 0] = 0.5
        lab[..., 1] = 0.08  # colourful backdrop: the photo is not monochrome
        lab[..., 2] = -0.05
    m = np.ones((H, W), bool) if mask is None else mask
    lab[..., 0][m] = L + rng.normal(0, 0.01, m.sum())
    lab[..., 1][m] = C * np.cos(np.radians(hue_deg)) + rng.normal(0, 0.003, m.sum())
    lab[..., 2][m] = C * np.sin(np.radians(hue_deg)) + rng.normal(0, 0.003, m.sum())
    return lab


def _bgr01(lab):
    return oklab_to_bgr(lab, float32_out=True) / 255.0


@pytest.mark.parametrize("hue,label", [(250.0, "blue"), (150.0, "green"), (300.0, "purple")])
def test_detects_off_hue_paint(hue, label):
    face = _face_mask()
    lab = _lab_image(0.6, 0.11, hue, face)
    found = bp.detect_painted_face(lab, face)
    assert found is not None, label
    assert found.kind == "colour"
    assert abs(((found.paint_hue_deg - hue + 180) % 360) - 180) < 5


def test_detects_grey_paint_in_colour_photo():
    face = _face_mask()
    lab = _lab_image(0.6, 0.005, 0.0, face)
    found = bp.detect_painted_face(lab, face)
    assert found is not None and found.kind == "grey"


@pytest.mark.parametrize(
    "L,C,hue",
    [
        (0.85, 0.035, 40.0),  # pale skin, low chroma
        (0.72, 0.06, 50.0),   # medium skin
        (0.35, 0.045, 55.0),  # deep skin: low L, chroma relative to L is high
        (0.80, 0.035, 356.0),  # pink-toned skin across the 0/360 wrap
        (0.60, 0.06, 105.0),  # green venue-light cast still inside the band
    ],
)
def test_human_skin_is_not_paint(L, C, hue):
    face = _face_mask()
    lab = _lab_image(L, C, hue, face)
    assert bp.detect_painted_face(lab, face) is None


def test_black_and_white_photo_is_not_grey_paint():
    face = _face_mask()
    lab = _lab_image(0.6, 0.004, 0.0, face, base=np.array([0.4, 0.0, 0.0], np.float32))
    assert bp.is_monochrome(lab)
    assert bp.detect_painted_face(lab, face) is None


def test_lock_restores_paint_colour_but_keeps_lightness_edit():
    face = _face_mask()
    ref_lab = _lab_image(0.6, 0.004, 0.0, face)
    edited_lab = ref_lab.copy()
    edited_lab[..., 0][face] += 0.05          # a lightness edit (relight)
    edited_lab[..., 1][face] += 0.02          # a pink cast from skin-tone ops
    ref, edited = _bgr01(ref_lab), _bgr01(edited_lab)
    out = bp.lock_paint_colour(edited, ref, face.astype(np.float32))
    out_lab = bgr_to_oklab(out * 255.0)
    inner = cv2.erode(face.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert np.abs(out_lab[..., 1][inner] - ref_lab[..., 1][inner]).mean() < 0.003
    assert np.abs(out_lab[..., 0][inner] - edited_lab[..., 0][inner]).mean() < 0.003


def _patchy_blue(seed=1):
    """Blue paint with sponge blotches in L and thin spots where skin shows."""
    rng = np.random.default_rng(seed)
    face = _face_mask()
    lab = _lab_image(0.6, 0.11, 250.0, face, seed=seed)
    blot = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3.0)
    blot /= blot[face].std()
    lab[..., 0] += np.where(face, 0.03 * blot, 0.0)
    thin = np.zeros((H, W), np.uint8)
    for cx, cy in ((60, 60), (100, 90), (75, 110)):
        cv2.circle(thin, (cx, cy), 6, 1, -1)
    thin = thin.astype(bool) & face
    skin_a, skin_b = 0.06 * np.cos(np.radians(50)), 0.06 * np.sin(np.radians(50))
    lab[..., 1][thin] = skin_a
    lab[..., 2][thin] = skin_b
    return lab, face, thin


def test_even_paint_reduces_blotches_and_show_through():
    lab, face, thin = _patchy_blue()
    img = _bgr01(lab)
    found = bp.detect_painted_face(lab, face)
    assert found is not None
    out = bp.even_paint(img, face.astype(np.float32), 1.0, face_width=W * 0.6, faces=[found])
    out_lab = bgr_to_oklab(out * 255.0)
    inner = cv2.erode(face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    def band(L):
        return L - cv2.GaussianBlur(L, (0, 0), 6.0)

    before = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    after = cv2.GaussianBlur(out_lab[..., 0], (0, 0), 1.0)
    assert band(after)[inner].std() < 0.75 * band(before)[inner].std()

    def dist_to_paint(x):
        return np.hypot(x[..., 1] - found.paint_a, x[..., 2] - found.paint_b)

    assert dist_to_paint(out_lab)[thin].mean() < 0.6 * dist_to_paint(lab)[thin].mean()


def test_even_paint_keeps_strong_features():
    lab, face, _ = _patchy_blue(seed=2)
    lab[..., 0][70:73, 40:120] -= 0.25  # a dark crease line (brow, lash line)
    img = _bgr01(lab)
    found = bp.detect_painted_face(lab, face)
    out = bp.even_paint(img, face.astype(np.float32), 1.0, face_width=W * 0.6, faces=[found])
    out_lab = bgr_to_oklab(out * 255.0)
    line_before = lab[..., 0][70:73, 50:110].mean() - lab[..., 0][80:83, 50:110].mean()
    line_after = out_lab[..., 0][70:73, 50:110].mean() - out_lab[..., 0][80:83, 50:110].mean()
    assert line_after < 0.8 * line_before  # still clearly darker (both negative)


def test_strength_zero_is_identity():
    lab, face, _ = _patchy_blue()
    img = _bgr01(lab)
    assert bp.even_paint(img, face.astype(np.float32), 0.0, face_width=90) is img
    out, diag = bp.apply_body_paint(img, img, face.astype(np.float32), [(40, 20, 80, 120)], 0.0)
    assert out is img and diag["applied"] is False


def test_apply_body_paint_skips_human_skin():
    face = _face_mask()
    lab = _lab_image(0.7, 0.06, 50.0, face)
    img = _bgr01(lab)
    out, diag = bp.apply_body_paint(img, img, face.astype(np.float32), [(40, 20, 80, 120)], 0.8)
    assert out is img
    assert diag["reason"] == "no_paint_found"


def test_region_extends_to_connected_body_paint_only():
    face = _face_mask()
    lab = _lab_image(0.6, 0.11, 250.0, face)
    # painted neck/shoulder connected below the face, and a separate blue
    # object elsewhere on the "person" that does not touch the face.
    connected = np.zeros((H, W), bool)
    connected[135:160, 55:105] = True
    island = np.zeros((H, W), bool)
    island[5:20, 5:20] = True
    for m in (connected, island):
        lab[..., 0][m] = 0.6
        lab[..., 1][m] = 0.11 * np.cos(np.radians(250))
        lab[..., 2][m] = 0.11 * np.sin(np.radians(250))
    found = bp.detect_painted_face(lab, face)
    region = bp.paint_region_mask(lab, [found], [face], person_mask=np.ones((H, W), np.float32))
    assert region[145:155, 65:95].mean() > 0.9
    assert region[8:17, 8:17].max() < 0.1


# ---------------------------------------------------------------------------
# Engine wiring (landmark-only face context, no ONNX)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    return RetouchEngine()


def _painted_synthetic():
    from tests.golden_face_fixture import make_synthetic_face_image

    img = make_synthetic_face_image()
    lab = bgr_to_oklab(img)
    h, w = img.shape[:2]
    cx, cy = int(0.47 * w), int(0.41 * h)
    ax, ay = int(0.20 * w), int(0.28 * h)
    yv, xv = np.ogrid[:h, :w]
    ell = (((xv - cx) / ax) ** 2 + ((yv - cy) / ay) ** 2) <= 1.0
    lab[..., 1][ell] = 0.004
    lab[..., 2][ell] = 0.0
    return oklab_to_bgr(lab), ell


def test_engine_body_paint_off_by_default_and_noop_without_paint(engine):
    from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

    img = make_synthetic_face_image()
    h, w = img.shape[:2]
    fc = make_face_context(w, h, img)
    base = engine.process(img, recipe="cosplay_clear_v1", face_contexts=[fc])
    fc = make_face_context(w, h, img)
    on = engine.process(img, recipe="cosplay_clear_v1", face_contexts=[fc], body_paint=80)
    assert np.array_equal(np.asarray(base), np.asarray(on))
    assert "body_paint" not in base.runtime_diagnostics
    assert on.runtime_diagnostics["body_paint"]["applied"] is False


def test_engine_keeps_grey_paint_grey(engine):
    from tests.golden_face_fixture import make_face_context

    img, ell = _painted_synthetic()
    h, w = img.shape[:2]
    inner = cv2.erode(ell.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)

    def chroma(x):
        lab = bgr_to_oklab(np.asarray(x))
        return float(np.hypot(lab[..., 1], lab[..., 2])[inner].mean())

    on = engine.process(
        img, recipe="cosplay_clear_v1", face_contexts=[make_face_context(w, h, img)], body_paint=50
    )
    diag = on.runtime_diagnostics["body_paint"]
    assert diag["applied"] is True
    assert diag["painted_faces"][0]["kind"] == "grey"
    off = engine.process(
        img, recipe="cosplay_clear_v1", face_contexts=[make_face_context(w, h, img)]
    )
    # Without the lock cosplay_clear_v1's skin ops tint grey paint (chroma
    # ~0.004 -> ~0.008); with it the paint stays near the input. The grade
    # (applied to the whole photo afterwards) may still move it a little.
    drift_off = abs(chroma(off) - chroma(img))
    drift_on = abs(chroma(on) - chroma(img))
    assert drift_off > 0.002
    assert drift_on < 0.5 * drift_off

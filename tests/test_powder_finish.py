"""Tests for retouch/powder_finish.py (opt-in ``powder_finish``)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.params import PROCESSING_PARAMS
from retouch.powder_finish import powder_finish
from retouch.utils import bgr_f32_to_lab_f32

IED = 100.0


def _scene(scale: float = 1.0, paint: bool = False):
    """A face-sized skin patch with a lit side, fine texture and shine spots.

    ``scale`` multiplies linear light, so 0.12 simulates much darker skin
    under the same lighting. ``paint`` makes the skin near-white and
    low-chroma, like white face paint, so shine keeps the skin's chroma.
    """
    rng = np.random.default_rng(3)
    h, w = 260, 260
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([0.93, 0.92, 0.90] if paint else [0.45, 0.55, 0.75], np.float32)
    lit = 0.75 + 0.35 * (xx / w)  # broad lit side: form, not shine
    # A lit cheekbone: broad (0.35 x IED) and bright, still form, not shine.
    lit = lit * (1.0 + 0.5 * np.exp(-((yy - 200) ** 2 + (xx - 200) ** 2) / (2 * 35.0**2)))
    tex = 1.0 + 0.03 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
    lin = base[None, None, :] * (lit * tex)[..., None] * 0.6
    shine = np.zeros((h, w), np.float32)
    for cy, cx, r in [(90, 90, 9), (150, 170, 12), (190, 110, 7)]:
        shine += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    # Additive highlight that keeps the skin's own colour (as on paint and
    # on pale skin), so a chroma-drop detector would miss it.
    lin = lin * (1.0 + 0.9 * shine[..., None])
    img = np.clip((lin * scale) ** (1 / 2.2) * 255.0, 0, 255).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (130, 130), (115, 120), 0, 0, 360, 1.0, -1)
    return img, mask, shine > 0.5


def _L(img):
    return bgr_f32_to_lab_f32(img.astype(np.float32))[..., 0]


def _shine_contrast(img, mask, spots):
    L = _L(img)
    ring = (mask > 0.5) & ~cv2.dilate(spots.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    return float(L[spots].mean() - np.median(L[ring]))


def test_strength_zero_is_identity():
    img, mask, _ = _scene()
    assert powder_finish(img, mask, 0, IED) is img


def test_missing_or_empty_mask_is_identity():
    img, mask, _ = _scene()
    assert powder_finish(img, None, 60, IED) is img
    assert powder_finish(img, np.zeros_like(mask), 60, IED) is img


def test_softens_shine_and_scales_with_strength():
    img, mask, spots = _scene()
    before = _shine_contrast(img, mask, spots)
    mid = _shine_contrast(powder_finish(img, mask, 50, IED), mask, spots)
    full = _shine_contrast(powder_finish(img, mask, 100, IED), mask, spots)
    assert before > 8
    assert full < mid < before * 0.85
    assert full > 0, "a soft residue of the highlight should remain"


def test_works_on_white_face_paint():
    img, mask, spots = _scene(paint=True)
    before = _shine_contrast(img, mask, spots)
    after = _shine_contrast(powder_finish(img, mask, 60, IED), mask, spots)
    assert after < before * 0.75


def test_keeps_skin_tone_not_grey():
    """The old flat powder pulled lit skin toward grey; this must not."""
    img, mask, spots = _scene()
    out = powder_finish(img, mask, 100, IED)
    lab_in = bgr_f32_to_lab_f32(img)
    lab_out = bgr_f32_to_lab_f32(out)
    sel = mask > 0.5
    c_in = np.hypot(lab_in[..., 1] - 128, lab_in[..., 2] - 128)[sel]
    c_out = np.hypot(lab_out[..., 1] - 128, lab_out[..., 2] - 128)[sel]
    assert np.median(c_out) >= np.median(c_in) - 0.5
    # Overall brightness is kept (within 2 L levels).
    assert abs(float(np.median(_L(out)[sel])) - float(np.median(_L(img)[sel]))) < 2.0


def test_lit_side_of_face_is_not_flattened():
    img, mask, spots = _scene()
    out = powder_finish(img, mask, 100, IED)
    far = cv2.dilate(spots.astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    sel = (mask > 0.5) & far
    L_in, L_out = _L(img), _L(out)
    cols = [slice(40, 70), slice(190, 220)]
    grad_in = np.median(L_in[:, cols[1]][sel[:, cols[1]]]) - np.median(L_in[:, cols[0]][sel[:, cols[0]]])
    grad_out = np.median(L_out[:, cols[1]][sel[:, cols[1]]]) - np.median(L_out[:, cols[0]][sel[:, cols[0]]])
    assert grad_in > 10
    assert grad_out > 0.85 * grad_in


def test_lit_cheekbone_keeps_its_brightness():
    """A broad lit area is form; the finish must not darken it into a patch."""
    img, mask, spots = _scene()
    out = powder_finish(img, mask, 100, IED)
    near_spot = cv2.dilate(spots.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    yy, xx = np.mgrid[0:260, 0:260]
    cheek = (((yy - 200) ** 2 + (xx - 200) ** 2) < 20**2) & ~near_spot & (mask > 0.5)
    assert cheek.sum() > 200
    drop = float(np.median(_L(img)[cheek]) - np.median(_L(out)[cheek]))
    assert drop < 3.0


def test_pixels_outside_skin_untouched():
    img, mask, _ = _scene()
    out = powder_finish(img, mask, 100, IED)
    assert np.array_equal(out[mask == 0], img[mask == 0])


def test_eyes_protected():
    img, mask, spots = _scene()
    eyes = np.zeros_like(mask)
    cv2.circle(eyes, (90, 90), 12, 1.0, -1)  # covers the first shine spot
    out = powder_finish(img, mask, 100, IED, eyes_mask=eyes)
    core = np.zeros_like(mask, bool)
    cv2.circle(core.view(np.uint8), (90, 90), 8, 1, -1)
    assert np.abs(out - img)[core].max() < 1.0


@pytest.mark.parametrize("dark", [0.12, 0.25])
def test_tone_invariant(dark):
    img, mask, spots = _scene()
    dimg, _, _ = _scene(scale=dark)
    r_light = _shine_contrast(powder_finish(img, mask, 60, IED), mask, spots) / _shine_contrast(img, mask, spots)
    r_dark = _shine_contrast(powder_finish(dimg, mask, 60, IED), mask, spots) / _shine_contrast(dimg, mask, spots)
    assert r_dark < 0.85, "darker skin must get the finish too"
    assert abs(r_light - r_dark) < 0.15


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "powder_finish")
    assert spec.default == 0
    assert spec.cli_flag == "powder-finish"
    assert spec.recipe_key == "skin.powder_finish"


class TestEngineWiring:
    @pytest.fixture(scope="class")
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        eng = RetouchEngine()
        off = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc]))
        on = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc], powder_finish=100))
        return img, fc, off, on

    def test_default_off_changes_nothing(self, run):
        from retouch.engine import RetouchEngine

        img, fc, off, _ = run
        again = np.asarray(RetouchEngine().process(img, recipe="natural", face_contexts=[fc], powder_finish=0))
        assert np.array_equal(off, again)

    def test_on_reaches_the_face(self, run):
        img, fc, off, on = run
        diff = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
        assert diff.max() > 0, "powder_finish=100 had no effect: op not wired into the face path"
        x, y, fw, fh = fc.face_data.bbox
        ys, xs = np.nonzero(diff > 1)
        pad = 0.5 * fw
        assert xs.min() >= x - pad and xs.max() <= x + fw + pad
        assert ys.min() >= y - pad and ys.max() <= y + fh + pad

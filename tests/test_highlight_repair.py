"""Tests for retouch/highlight_repair.py (opt-in ``highlight_repair``)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.highlight_repair import _tone_curve, repair_blown_highlights
from retouch.params import PROCESSING_PARAMS
from retouch.utils import bgr_f32_to_lab_f32

IED = 100.0
CENTER = (130, 130)


def _scene(scale: float = 1.0, paint: bool = False, gain: float = 1.2, radius: float = 16.0):
    """A face-sized skin patch with fine texture and one flash hot spot.

    The hot spot adds up to ``gain`` of white light (specular reflection is
    the light's colour, not the skin's) and is then clipped at the file
    ceiling, like a sensor. ``scale`` multiplies the skin's own linear light
    only, so 0.15 simulates much darker skin under the same flash.
    ``paint`` makes the skin near-white and low-chroma.
    """
    rng = np.random.default_rng(5)
    h, w = 260, 260
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([0.93, 0.92, 0.90] if paint else [0.45, 0.55, 0.75], np.float32)
    lit = 0.8 + 0.3 * (xx / w)
    tex = 1.0 + 0.05 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
    lin = base[None, None, :] * (lit * tex)[..., None] * 0.55 * scale
    # A specular lobe: flat-topped with a fast fall-off, like a real hot spot.
    r2 = ((yy - CENTER[1]) ** 2 + (xx - CENTER[0]) ** 2) / (2 * radius**2)
    spot = np.exp(-(r2**2))
    lin = lin + gain * spot[..., None]
    img = (np.clip(lin, 0, 1) ** (1 / 2.2) * 255.0).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (130, 130), (115, 120), 0, 0, 360, 1.0, -1)
    return img, mask


def _blown(img):
    """Any channel at the ceiling."""
    return img.max(axis=2) >= 254.0


def _white(img):
    """Every channel at the ceiling: no colour and no texture left."""
    return img.min(axis=2) >= 254.0


def _lab(img):
    return bgr_f32_to_lab_f32(img.astype(np.float32))


def _ring(img, mask, r0=40, r1=60):
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - CENTER[1], xx - CENTER[0])
    return (d >= r0) & (d <= r1) & (mask > 0.5)


def test_strength_zero_is_identity():
    img, mask = _scene()
    out, _ = repair_blown_highlights(img, mask, 0, IED)
    assert out is img


def test_no_blown_pixels_is_identity():
    img, mask = _scene(gain=0.0)
    assert not _blown(img).any()
    out, diags = repair_blown_highlights(img, mask, 100, IED)
    assert out is img and diags["repaired"] == 0


def test_blown_spot_is_rebuilt_below_white():
    img, mask = _scene()
    core = _blown(img)
    assert core.sum() > 200, "scene must have a real blown patch"
    out, diags = repair_blown_highlights(img, mask, 60, IED)
    assert diags["repaired"] == 1
    assert not _blown(out)[core].any(), "blown pixels must come off the ceiling"
    L0 = _lab(img)[..., 0]
    L1 = _lab(out)[..., 0]
    ring = _ring(img, mask)
    # Still a highlight: brighter than the skin around it.
    assert float(L1[core].mean()) > float(np.median(L0[ring])) + 20


def test_blown_spot_gets_skin_colour_back():
    img, mask = _scene()
    core = _white(img)
    assert core.sum() > 100
    out, _ = repair_blown_highlights(img, mask, 60, IED)
    lab0, lab1 = _lab(img), _lab(out)
    ring = _ring(img, mask)
    ra, rb = np.median(lab0[..., 1][ring]) - 128, np.median(lab0[..., 2][ring]) - 128
    ca, cb = lab1[..., 1][core].mean() - 128, lab1[..., 2][core].mean() - 128
    # Chroma back to a good share of the skin's, at the skin's hue.
    assert np.hypot(ca, cb) > 0.35 * np.hypot(ra, rb)
    hue_diff = abs(np.degrees(np.arctan2(cb, ca) - np.arctan2(rb, ra)))
    assert hue_diff < 12.0
    # Before: the blown core was neutral white.
    assert np.hypot(lab0[..., 1][core].mean() - 128, lab0[..., 2][core].mean() - 128) < 0.2 * np.hypot(ra, rb)


def test_blown_core_gets_texture_not_a_flat_patch():
    img, mask = _scene()
    core = cv2.erode(_white(img).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out, _ = repair_blown_highlights(img, mask, 60, IED)
    L0, L1 = _lab(img)[..., 0], _lab(out)[..., 0]

    def fine_std(L, m):
        hp = L - cv2.GaussianBlur(L, (0, 0), 2.0)
        return float(hp[m].std())

    assert fine_std(L0, core) < 0.3
    ring = _ring(img, mask)
    assert fine_std(L1, core) > 0.3 * fine_std(L0, ring)


def test_no_halo_around_the_spot():
    """Brightness falls from the spot's centre outward: no dark core in a bright ring."""
    img, mask = _scene()
    out, _ = repair_blown_highlights(img, mask, 100, IED)
    L = cv2.GaussianBlur(_lab(out)[..., 0], (0, 0), 2.0)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(yy - CENTER[1], xx - CENTER[0])
    prof = [float(L[(d >= r) & (d < r + 3)].mean()) for r in range(0, 60, 3)]
    rises = [b - a for a, b in zip(prof, prof[1:])]
    assert max(rises) < 1.5, prof


def test_scales_with_strength():
    img, mask = _scene()
    core = _blown(img)
    L = [float(_lab(repair_blown_highlights(img, mask, s, IED)[0])[..., 0][core].mean()) for s in (30, 60, 100)]
    assert L[0] > L[1] > L[2]


def test_white_face_paint_left_alone():
    img, mask = _scene(paint=True, gain=0.6)
    assert _blown(img).sum() > 100
    out, diags = repair_blown_highlights(img, mask, 100, IED)
    assert diags["painted"] and diags["repaired"] == 0
    assert np.array_equal(out, img)


def test_cool_white_paint_with_some_chroma_left_alone():
    """White paint under cool light reads blue-magenta (Alex's bunny photos:
    CIELab hue about -30 deg, chroma / L* 0.12-0.14): outside natural skin."""
    img, mask = _scene(paint=True, gain=0.6)
    # Tint the paint toward blue-magenta: more blue and red, less green.
    img = np.clip(img * np.array([1.0, 0.93, 0.97], np.float32), 0, 255)
    lab = _lab(img)
    hue = np.degrees(np.arctan2(np.median(lab[..., 2][mask > 0.5]) - 128, np.median(lab[..., 1][mask > 0.5]) - 128))
    assert hue < -5.0 and _blown(img).sum() > 50
    out, diags = repair_blown_highlights(img, mask, 100, IED)
    assert diags["painted"]
    assert np.array_equal(out, img)


def test_blown_prop_not_ringed_by_skin_left_alone():
    """A white prop held over the face: the skin mask has a hole around it."""
    img, mask = _scene()
    m = mask.copy()
    cv2.circle(m, CENTER, 45, 0.0, -1)
    out, diags = repair_blown_highlights(img, m, 100, IED)
    assert diags["repaired"] == 0 and diags["skipped_ring"] >= 1
    assert np.array_equal(out, img)


def test_features_left_alone():
    img, mask = _scene()
    feat = np.zeros_like(mask)
    cv2.circle(feat, CENTER, 40, 1.0, -1)
    out, diags = repair_blown_highlights(img, mask, 100, IED, feature_mask=feat)
    assert diags["repaired"] == 0
    assert np.array_equal(out, img)


def test_mostly_blown_face_left_alone():
    img, mask = _scene(gain=3.0, radius=45.0)
    out, diags = repair_blown_highlights(img, mask, 100, IED)
    assert diags["blown_share"] > 0.35
    assert np.array_equal(out, img)


def test_pixels_far_from_the_spot_untouched():
    img, mask = _scene()
    out, _ = repair_blown_highlights(img, mask, 100, IED)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    far = np.hypot(yy - CENTER[1], xx - CENTER[0]) > 75
    assert np.array_equal(out[far], img[far])


@pytest.mark.parametrize("dark", [0.15])
def test_tone_invariant(dark):
    """Darker skin under the same flash: found, rebuilt, and coloured alike.

    The same white light is a much larger share of a darker face's light, so
    its unclipped shoulder is paler; the rebuilt core is judged against that
    shoulder (it must continue it, not stay neutral white) and against the
    skin's hue.
    """
    results = []
    for scale in (1.0, dark):
        img, mask = _scene(scale=scale)
        core = _white(img)
        assert core.sum() > 100
        out, diags = repair_blown_highlights(img, mask, 60, IED)
        assert diags["repaired"] == 1
        assert not _blown(out)[_blown(img)].any()
        lab0, lab1 = _lab(img), _lab(out)
        blown = _blown(img).astype(np.uint8)
        shoulder = (cv2.dilate(blown, np.ones((13, 13), np.uint8)) > 0) & (blown == 0)
        sa, sb = lab0[..., 1][shoulder].mean() - 128, lab0[..., 2][shoulder].mean() - 128
        ca, cb = lab1[..., 1][core].mean() - 128, lab1[..., 2][core].mean() - 128
        ring = _ring(img, mask)
        ra, rb = np.median(lab0[..., 1][ring]) - 128, np.median(lab0[..., 2][ring]) - 128
        hue_diff = abs(np.degrees(np.arctan2(cb, ca) - np.arctan2(rb, ra)))
        assert hue_diff < 15.0
        results.append(np.hypot(ca, cb) / np.hypot(sa, sb))
    # At least as rich as the shoulder on both tones (on darker skin the
    # middle of the spot also takes some diffuse skin colour, so it ends
    # richer than its pale shoulder rather than grey).
    assert min(results) > 0.8, results


def test_tone_curve_monotone_and_lands_below_white():
    x = np.linspace(0, 1, 501, dtype=np.float32)
    for t in (0.55, 0.7, 0.9):
        y = _tone_curve(x, t)
        assert np.all(np.diff(y) >= -1e-6)
        assert abs(float(y[-1]) - t) < 1e-3
        assert np.allclose(y[x <= 0.3], x[x <= 0.3])


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "highlight_repair")
    assert spec.default == 0
    assert spec.cli_flag == "highlight-repair"
    assert spec.recipe_key == "skin.highlight_repair"


class TestEngineWiring:
    @pytest.fixture(scope="class")
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        skin = fc.regions.skin
        skin = skin.astype(np.float32) / (255.0 if skin.max() > 1.5 else 1.0)
        # Plant a blown patch at the skin point deepest inside the mask.
        dt = cv2.distanceTransform((skin > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        r = int(max(4, min(dt.max() * 0.4, fc.face_data.ied * 0.12)))
        img = img.copy()
        cv2.circle(img, (int(cx), int(cy)), r, (255, 255, 255), -1)
        eng = RetouchEngine()
        off = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc]))
        on = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc], highlight_repair=100))
        return img, fc, off, on, (cx, cy, r)

    def test_default_off_changes_nothing(self, run):
        from retouch.engine import RetouchEngine

        img, fc, off, _, _ = run
        again = np.asarray(RetouchEngine().process(img, recipe="natural", face_contexts=[fc], highlight_repair=0))
        assert np.array_equal(off, again)

    def test_on_reaches_the_blown_patch(self, run):
        img, fc, off, on, (cx, cy, r) = run
        diff = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
        assert diff[cy, cx] > 5, "highlight_repair=100 had no effect: op not wired into the face path"
        ys, xs = np.nonzero(diff > 1)
        x, y, fw, fh = fc.face_data.bbox
        pad = 0.5 * fw
        assert xs.min() >= x - pad and xs.max() <= x + fw + pad
        assert ys.min() >= y - pad and ys.max() <= y + fh + pad

"""Tests for the bloodshot eye-whites fix (retouch/bloodshot_eyes.py).

The eye here is synthetic: an almond eye opening with a shaded white, a dark
iris wearing a red contact lens wider than the landmark iris, a catchlight,
lashes on the upper rim and a pink waterline on the lower rim, then planted
vessels and a diffuse pink. Real bloodshot eyes were not available; the
darker-exposure case is a linear-light gain on the same image (simulated,
not a real darker-skin or low-light photo).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.bloodshot_eyes import calm_bloodshot_eye, calm_bloodshot_eyes
from retouch.eyes import EyeEnhancer

H, W = 160, 260
CY, CX = 80, 130
IRIS_R = 24          # landmark iris radius
CONTACT_R = 30       # red contact lens is wider than the landmark iris


def _lab(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                        cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR).astype(np.float32)


def _build(seed=0):
    """Return (img uint8, eye mask, iris mask, sclera mask, truth dict)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lab = np.zeros((H, W, 3), np.float32)
    lab[...] = (175, 140, 145)                    # peach skin
    eye = ((xx - CX) / 105.0) ** 2 + ((yy - CY) / 42.0) ** 2 <= 1.0
    # White with the eyeball's own shading: darker toward the corners.
    shade = 1.0 - 0.18 * ((xx - CX) / 105.0) ** 2
    white_l = 215.0 * shade
    lab[eye, 0] = white_l[eye]
    lab[eye, 1] = 129.0
    lab[eye, 2] = 132.0                            # faintly warm white
    dist_c = np.hypot(yy - CY, xx - CX)
    contact = eye & (dist_c <= CONTACT_R)
    lab[contact] = (95, 168, 140)                  # red contact lens
    iris = dist_c <= IRIS_R
    lab[iris & eye] = (70, 160, 138)
    lab[(dist_c <= 9) & eye] = (20, 128, 128)      # pupil
    catch = np.hypot(yy - (CY - 8), xx - (CX + 7)) <= 3.5
    lab[catch] = (250, 128, 124)
    # Rims: lashes on top, pink waterline at the bottom.
    edge = eye & ~cv2.erode(eye.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    top = edge & (yy < CY)
    bottom = edge & (yy >= CY)
    lab[top] = (40, 130, 130)
    lab[bottom] = (150, 158, 140)
    lab[..., 0] += rng.normal(0, 1.5, (H, W)).astype(np.float32)
    img = np.clip(_from_lab(lab), 0, 255).astype(np.uint8)

    iris_m = (iris & eye).astype(np.float32)
    eye_m = eye.astype(np.float32)
    sclera_m = np.clip(eye_m - iris_m, 0, 1)
    white = eye & (dist_c > CONTACT_R + 3) & ~edge
    white = cv2.erode(white.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return img, eye_m, iris_m, sclera_m, dict(
        white=white, contact=contact & ~iris, iris=iris & eye, catch=catch,
        top=top, bottom=bottom, eye=eye)


def _plant(img, truth, seed=1, diffuse=8.0):
    """Vessels (thin, red, a bit darker) plus a diffuse pink on the white."""
    rng = np.random.default_rng(seed)
    vm = np.zeros((H, W), np.float32)
    ys, xs = np.where(truth["white"])
    for _ in range(12):
        i = rng.integers(len(ys))
        y, x = float(ys[i]), float(xs[i])
        ang = np.arctan2(CY - y, CX - x) + rng.normal(0, 0.5)
        pts = []
        for _ in range(40):
            ang += rng.normal(0, 0.2)
            y += np.sin(ang)
            x += np.cos(ang)
            pts.append((x * 8, y * 8))
        cv2.polylines(vm, [np.array(pts, np.int32)], False, 1.0,
                      thickness=2, lineType=cv2.LINE_AA, shift=3)
    vm = np.clip(vm, 0, 1) * truth["white"]
    lab = _lab(img)
    lab[..., 1] += diffuse * truth["white"] + 20.0 * vm
    lab[..., 0] -= 14.0 * vm
    out = np.where((truth["white"] | (vm > 0))[..., None],
                   _from_lab(lab), img.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8), vm > 0.3


def _removed(clean, planted, out, where):
    a0, ap, ao = (_lab(x)[..., 1][where] for x in (clean, planted, out))
    return float(((ap - a0).sum() - (ao - a0).sum()) / max((ap - a0).sum(), 1e-6))


class TestNoOp:
    def test_zero_strength_is_identity(self):
        img, _, iris, sclera, _ = _build()
        assert calm_bloodshot_eye(img, sclera, iris, 0) is img

    def test_missing_masks(self):
        img, _, iris, sclera, _ = _build()
        assert calm_bloodshot_eye(img, None, iris, 80) is img
        assert calm_bloodshot_eye(img, sclera, None, 80) is img
        assert calm_bloodshot_eye(img, np.zeros_like(sclera), iris, 80) is img

    def test_tiny_iris_skipped(self):
        img, _, _, sclera, _ = _build()
        tiny = np.zeros_like(sclera)
        tiny[CY, CX] = 1.0
        assert calm_bloodshot_eye(img, sclera, tiny, 80) is img

    def test_accepts_0_255_masks(self):
        img, _, iris, sclera, truth = _build()
        planted, _ = _plant(img, truth)
        a = calm_bloodshot_eye(planted, sclera, iris, 80)
        b = calm_bloodshot_eye(planted, (sclera * 255).astype(np.uint8),
                               (iris * 255).astype(np.uint8), 80)
        assert np.array_equal(a, b)


class TestCalmsRedness:
    def test_vessels_removed_and_monotone(self):
        img, _, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth, diffuse=0.0)
        r50 = _removed(img, planted, calm_bloodshot_eye(planted, sclera, iris, 50), vessels)
        r100 = _removed(img, planted, calm_bloodshot_eye(planted, sclera, iris, 100), vessels)
        assert r50 >= 0.30
        assert r100 >= 0.60
        assert r100 > r50

    def test_vessel_darkening_restored(self):
        img, _, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth, diffuse=0.0)
        out = calm_bloodshot_eye(planted, sclera, iris, 100)
        dl_before = (_lab(img)[..., 0] - _lab(planted)[..., 0])[vessels].mean()
        dl_after = (_lab(img)[..., 0] - _lab(out)[..., 0])[vessels].mean()
        assert dl_after < 0.5 * dl_before

    def test_diffuse_pink_calmed(self):
        img, _, iris, sclera, truth = _build()
        planted, _ = _plant(img, truth, diffuse=10.0)
        out = calm_bloodshot_eye(planted, sclera, iris, 100)
        # Pulled to the natural-white cap (a* ~ +4 here), not to the clean
        # input (+1), and ramped at the rims: about half of the planted pink.
        assert _removed(img, planted, out, truth["white"]) >= 0.5

    def test_clean_white_barely_changes(self):
        img, _, iris, sclera, truth = _build()
        out = calm_bloodshot_eye(img, sclera, iris, 100)
        da = np.abs(_lab(out)[..., 1] - _lab(img)[..., 1])[truth["white"]]
        assert float(da.mean()) < 1.0


@pytest.fixture(scope="module")
def run():
    img, _, iris, sclera, truth = _build()
    planted, _ = _plant(img, truth)
    return planted, calm_bloodshot_eye(planted, sclera, iris, 100), truth, sclera


class TestLeavesTheRestAlone:

    def test_outside_sclera_mask_identical(self, run):
        planted, out, _, sclera = run
        assert np.array_equal(out[sclera <= 0], planted[sclera <= 0])

    def test_contact_lens_wider_than_iris_kept(self, run):
        planted, out, truth, _ = run
        ring = truth["contact"] & (np.hypot(*np.mgrid[0:H, 0:W] - np.array([CY, CX])[:, None, None]) <= CONTACT_R - 1)
        d = np.abs(out.astype(int) - planted.astype(int)).max(-1)[ring]
        assert d.max() <= 2

    def test_iris_and_catchlight_kept(self, run):
        planted, out, truth, _ = run
        for key in ("iris", "catch"):
            d = np.abs(out.astype(int) - planted.astype(int)).max(-1)[truth[key]]
            assert d.max() <= 1, key

    def test_lashes_and_waterline_kept(self, run):
        planted, out, truth, _ = run
        for key in ("top", "bottom"):
            d = np.abs(_lab(out) - _lab(planted)).max(-1)[truth[key]]
            assert float(np.percentile(d, 95)) <= 2.0, key

    def test_eyeball_shading_kept(self, run):
        planted, out, truth, _ = run
        yy, xx = np.mgrid[0:H, 0:W]
        w = truth["white"]
        centre = w & (np.abs(xx - CX) < 55)
        corner = w & (np.abs(xx - CX) > 75)
        lp, lo = _lab(planted)[..., 0], _lab(out)[..., 0]
        ratio_before = lp[corner].mean() / lp[centre].mean()
        ratio_after = lo[corner].mean() / lo[centre].mean()
        assert abs(ratio_after - ratio_before) < 0.02
        # A gentle lift, not a flat paint-over.
        assert 0.0 <= lo[centre].mean() - lp[centre].mean() <= 0.05 * lp[centre].mean()


class TestToneInvariance:
    """Simulated darker exposure: linear-light gain on the same eye."""

    @staticmethod
    def _gain(img, g):
        x = img.astype(np.float32) / 255.0
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * g
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def test_darker_exposure_gets_comparable_fix(self):
        img, _, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth, diffuse=0.0)
        bright = _removed(img, planted, calm_bloodshot_eye(planted, sclera, iris, 100), vessels)
        dimg = self._gain(img, 0.45)
        dplanted, dvessels = _plant(dimg, truth, diffuse=0.0)
        dark = _removed(dimg, dplanted, calm_bloodshot_eye(dplanted, sclera, iris, 100), dvessels)
        assert dark >= 0.75 * bright


class TestDtypesAndWiring:
    def test_float32_in_float32_out(self):
        img, _, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth)
        out = calm_bloodshot_eye(planted.astype(np.float32), sclera, iris, 100)
        assert out.dtype == np.float32
        ref = calm_bloodshot_eye(planted, sclera, iris, 100)
        assert np.abs(out - ref.astype(np.float32)).max() <= 1.0

    def _regions(self, eye, iris, sclera):
        class R:
            pass
        r = R()
        z = np.zeros_like(eye)
        r.left_eye, r.left_iris, r.left_sclera = eye, iris, sclera
        r.right_eye, r.right_iris, r.right_sclera = z, z, z
        return r

    def test_regions_helper_and_eye_enhancer_route(self):
        img, eye, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth, diffuse=0.0)
        r = self._regions(eye, iris, sclera)
        direct = calm_bloodshot_eyes(planted, r, 100)
        assert _removed(img, planted, direct, vessels) >= 0.6
        # eye_enhance 0 + vessel strength 100 must still run the fix.
        via = EyeEnhancer().enhance(planted, r, strength=0, vessel_strength=100)
        assert _removed(img, planted, via, vessels) >= 0.6

    def test_regions_helper_derives_sclera(self):
        img, eye, iris, sclera, truth = _build()
        planted, vessels = _plant(img, truth, diffuse=0.0)
        r = self._regions(eye, iris, None)
        out = calm_bloodshot_eyes(planted, r, 100)
        assert _removed(img, planted, out, vessels) >= 0.6

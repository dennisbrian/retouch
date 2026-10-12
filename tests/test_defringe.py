"""Tests for retouch.defringe (``purple_fringing``, GUI "Defringe").

Synthetic scenes: a dark object against a blown backdrop with a planted
lens-style fringe band, plus real coloured objects at the same kind of edge.
Darker exposure is simulated with a linear-light gain, which is a stand-in
for real darker scenes, not a validation on them.
"""

import cv2
import numpy as np
import pytest

from retouch.defringe import band_radius, fringe_weight, remove_fringes


def _chroma(img_u8, mask):
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)[mask].mean())


def _scene(obj_bgr, fringe_bgr=None, width=5, size=(900, 3000)):
    """Bright left half, object on the right, optional fringe band on the
    object side of the edge. Returns image and the fringe-band mask."""
    h, w = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = obj_bgr
    edge = w // 2
    img[:, :edge] = (252, 252, 252)
    band = np.zeros((h, w), bool)
    band[:, edge:edge + width] = True
    if fringe_bgr is not None:
        img[band] = fringe_bgr
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return img, band


PURPLE = (200, 40, 150)   # BGR
GREEN = (60, 170, 40)
CYAN = (230, 200, 40)


class TestNoOp:
    def test_strength_zero_returns_input(self):
        img, _ = _scene((20, 20, 20), PURPLE)
        assert remove_fringes(img, 0) is img

    def test_flat_image_unchanged(self):
        img = np.full((200, 300, 3), (200, 40, 150), np.uint8)
        np.testing.assert_array_equal(remove_fringes(img, 100), img)

    def test_pixels_away_from_edge_bit_identical(self):
        img, band = _scene((20, 20, 20), PURPLE)
        out = remove_fringes(img, 100)
        r = band_radius(img.shape)
        far = np.ones(band.shape, bool)
        far[:, 1500 - 4 * r:1500 + 4 * r] = False
        np.testing.assert_array_equal(out[far], img[far])


class TestRemovesFringe:
    @pytest.mark.parametrize("fringe", [PURPLE, GREEN, CYAN], ids=["purple", "green", "cyan"])
    def test_fringe_on_black_object(self, fringe):
        img, band = _scene((20, 20, 20), fringe)
        before = _chroma(img, band)
        after = _chroma(remove_fringes(img, 100), band)
        assert before > 15
        assert after < 0.35 * before, (before, after)

    def test_old_function_removed_less(self):
        from retouch.utils import remove_purple_fringing
        img, band = _scene((20, 20, 20), PURPLE)
        old = _chroma(remove_purple_fringing(img, 1.0), band)
        new = _chroma(remove_fringes(img, 100), band)
        assert new < 0.5 * old

    def test_strength_is_monotone(self):
        img, band = _scene((20, 20, 20), PURPLE)
        c = [_chroma(remove_fringes(img, s), band) for s in (0, 50, 100)]
        assert c[0] > c[1] > c[2]

    def test_float_input_keeps_dtype_and_range(self):
        img, band = _scene((20, 20, 20), PURPLE)
        f = img.astype(np.float32) / 255.0
        out = remove_fringes(f, 100)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        u8 = np.clip(np.round(out * 255), 0, 255).astype(np.uint8)
        assert _chroma(u8, band) < 0.35 * _chroma(img, band)


class TestKeepsRealColour:
    @pytest.mark.parametrize("obj", [PURPLE, GREEN, CYAN, (90, 30, 20)],
                             ids=["purple", "green", "cyan", "navy"])
    def test_solid_coloured_object_edge_kept(self, obj):
        """A real coloured object against a blown backdrop: its edge colour
        is the same as its interior, so it must stay."""
        img, band = _scene(obj)
        before = _chroma(img, band)
        after = _chroma(remove_fringes(img, 100), band)
        assert after > 0.85 * before, (before, after)

    def test_skin_and_red_never_touched(self):
        for obj in [(150, 170, 220), (60, 90, 140), (40, 40, 200)]:  # light skin, dark skin, red
            img, _ = _scene(obj)
            assert fringe_weight(img).max() < 0.01

    def test_low_contrast_edge_ignored(self):
        """A purple band next to a mid-grey area is not a lens fringe."""
        img = np.full((400, 600, 3), 110, np.uint8)
        img[:, 300:306] = PURPLE
        img[:, 306:] = 60
        np.testing.assert_array_equal(remove_fringes(img, 100), img)


class TestToneInvariance:
    def test_darker_exposure_same_relative_effect(self):
        """Simulated darker exposure (linear gain 0.45): the gates follow the
        photo's own highlight level, so the share removed stays similar."""
        img, band = _scene((20, 20, 20), PURPLE)
        lin = (img.astype(np.float32) / 255.0) ** 2.2
        dark = np.clip(((lin * 0.45) ** (1 / 2.2)) * 255, 0, 255).astype(np.uint8)
        share = lambda x: 1 - _chroma(remove_fringes(x, 100), band) / _chroma(x, band)
        assert share(dark) > 0.6
        assert abs(share(dark) - share(img)) < 0.2


def test_band_radius_scales_with_image():
    assert band_radius((600, 800)) == 2
    assert band_radius((6240, 4160)) == 14
    assert band_radius((20000, 20000)) == 24

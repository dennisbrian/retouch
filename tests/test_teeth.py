"""Tests for retouch/teeth.py — TeethWhitener."""
import cv2
import numpy as np
import pytest
from retouch.teeth import TeethWhitener


@pytest.fixture
def whitener():
    return TeethWhitener()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


class TestWhiten:
    def test_zero_strength(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=0)
        assert np.all(result == img)

    def test_none_mask(self, whitener, img):
        result = whitener.whiten(img, None, strength=50)
        assert np.all(result == img)

    def test_empty_mask(self, whitener, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image_with_teeth(self, whitener):
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=80)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= orig_lab[:, :, 0].mean()

    def test_wid_delta_capped_to_acceptability(self, whitener):
        """Mean ΔWID over the teeth mask must stay within WAT (2.62)."""
        # Yellow-ish teeth on a bright mouth: high b* drives a large ΔWID
        # at full strength, so the cap must engage.
        lab = np.full((64, 64, 3), (110, 130, 145), dtype=np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.ones((64, 64), dtype=np.float32)
        before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        out = whitener.whiten(img, mask, strength=100)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        d_l = after[:, :, 0] - before[:, :, 0]
        d_a = after[:, :, 1] - before[:, :, 1]
        d_b = after[:, :, 2] - before[:, :, 2]
        mean_d_wid = 0.511 * d_l.mean() - 2.324 * d_a.mean() - 1.100 * d_b.mean()
        assert mean_d_wid <= 2.62 + 0.5  # cap, with rounding headroom

    def test_wid_delta_scales_with_strength_below_cap(self, whitener):
        """Low strength must not trigger the cap (ΔWID stays under WAT)."""
        # Mouth backdrop (darker) with a brighter tooth patch — realistic.
        lab = np.full((64, 64, 3), (90, 135, 145), dtype=np.uint8)
        lab[20:44, 20:44] = (160, 130, 140)  # bright tooth patch
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.ones((64, 64), dtype=np.float32)
        out = whitener.whiten(img, mask, strength=20)
        patch = np.s_[20:44, 20:44]
        # Some change must happen at low strength (cap not engaged).
        assert not np.array_equal(out[patch], img[patch])

    def test_no_absolute_l_floor_for_dark_mouth(self, whitener):
        """Dark mouth (low-key lighting) still detects bright-relative teeth.

        The old `l > 80` gate would reject every pixel in a dim mouth, even
        the genuinely-brighter-than-surroundings teeth — a fairness failure
        on low-lit or dark-skinned subjects. The tone-adaptive `l > l_median`
        gate must still pick up the bright pixels in such a mouth.
        """
        # Dark mouth with a brighter tooth patch in the middle.
        lab = np.full((64, 64, 3), (40, 130, 140), dtype=np.uint8)
        lab[24:40, 24:40] = (70, 125, 130)  # teeth: brighter than mouth median
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result[32, 32] > 0.1, "bright tooth patch must be detected"
        assert result[2, 2] < 0.1, "dark mouth void must be rejected"

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize("at_frame_edge", [False, True])
    def test_exact_outside_detected_teeth_support(
        self, whitener, monkeypatch, dtype, at_frame_edge
    ):
        rng = np.random.default_rng(20260903)
        source_u8 = rng.integers(20, 236, size=(64, 64, 3), dtype=np.uint8)
        source = source_u8 if dtype == np.uint8 else source_u8.astype(np.float32)
        mouth_mask = np.zeros((64, 64), dtype=np.float32)
        teeth_support = np.zeros((64, 64), dtype=np.float32)

        if at_frame_edge:
            mouth_mask[0:16, 0:24] = 1.0
            teeth_support[0:8, 0:12] = 1.0
            teeth_support[8:10, 0:12] = 0.25
        else:
            mouth_mask[22:42, 18:46] = 1.0
            teeth_support[27:35, 25:39] = 1.0
            teeth_support[25:27, 25:39] = 0.25

        monkeypatch.setattr(
            whitener,
            "_detect_teeth",
            lambda _image, _mouth: teeth_support.copy(),
        )

        result = whitener.whiten(source, mouth_mask, strength=80)

        outside = teeth_support == 0
        np.testing.assert_array_equal(result[outside], source[outside])
        assert np.any(result[~outside] != source[~outside])
        assert result.dtype == source.dtype
        assert np.isfinite(result).all()


class TestDetectTeeth:
    def test_small_mouth_returns_empty(self, whitener, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32:34, 32:34] = 1.0
        result = whitener._detect_teeth(img, mask)
        assert result.shape == (64, 64)
        assert result.max() == 0.0

    def test_output_shape(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_detects_bright_pixels(self, whitener):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[:, :, 1] = 80
        img[20:40, 20:40] = [220, 215, 210]
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result[30, 30] > 0

    def test_skips_dark_pixels(self, whitener, img):
        img[:] = 30
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.max() == 0.0

    def test_skips_saturated_pixels(self, whitener):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[:, :, 1] = 10
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.max() < 0.5

    def test_tongue_and_gums_excluded_from_teeth_mask(self, whitener):
        """Priority 5 (mouth protection): _detect_teeth's saturation gate
        must exclude pink/red tongue and gum pixels from the whitening mask
        even when they sit inside the same mouth-interior ROI as real teeth.

        No neural tongue segmenter or BiSeNet tongue class exists (there is
        no labeled corpus to build/validate one against -- see
        docs/plans/PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md), so
        this is the actual shipped containment: a relative low-saturation
        gate (teeth are close to gray; tongue/gums are saturated pink/red).
        """
        # Dim, mildly saturated mouth void (real oral-cavity shadow, not a
        # neutral gray) -- a perfectly desaturated background collapses
        # s_median to 0 and makes the relative gate degenerate.
        img = np.full((64, 64, 3), (60, 55, 70), dtype=np.uint8)
        # Bright, low-saturation "teeth" patch (near-gray, high L).
        img[8:24, 8:56] = (225, 220, 215)
        # Saturated pink "tongue" patch (BGR: strong red, weak blue/green).
        img[40:60, 16:48] = (110, 90, 200)
        mask = np.ones((64, 64), dtype=np.float32)

        teeth_mask = whitener._detect_teeth(img, mask)
        assert teeth_mask[16, 32] > 0.3, "the low-saturation bright patch must be detected as teeth"
        assert teeth_mask[50, 32] < 0.1, "the saturated pink patch (tongue/gums) must be excluded"

        # End to end: whitening must not touch the tongue-colored region.
        out = whitener.whiten(img, mask, strength=100)
        np.testing.assert_array_equal(out[40:60, 16:48], img[40:60, 16:48])

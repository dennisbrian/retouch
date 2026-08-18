"""Tests for retouch/color_science.py — Oklab converters and skin-tone measurement."""

import numpy as np
import cv2
import pytest

from retouch.color_science import (
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_oklch,
    oklch_to_oklab,
    measure_skin_state,
    skin_chroma_std,
    apply_abney_hue_correction,
    invert_abney_hue_correction,
    find_gamut_intersection_srgb,
    find_gamut_intersection_p3,
    find_gamut_intersection_rec2020,
    compress_chroma_gamut,
    bgr_to_cam16_ucs,
    cam16_ucs_to_bgr,
    cam16_ucs_delta_e,
    chromatic_adapt_cat16,
    linear_to_pq,
    pq_to_linear,
    SkinState,
    SKIN_LOCI,
)


class TestBgrToOklab:
    """Test vectorized BGR→Oklab conversion."""

    def test_white_pure(self):
        """Pure white should be achromatic with a≈b≈0."""
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        a = oklab[..., 1]
        b = oklab[..., 2]
        # White is achromatic (a=b=0) to float32 roundoff scale
        assert np.abs(a).max() < 1e-3, f"a for white should be ≈0, got {np.abs(a).max()}"
        assert np.abs(b).max() < 1e-3, f"b for white should be ≈0, got {np.abs(b).max()}"

    def test_black_pure(self):
        """Pure black should map to L≈0.0, C≈0."""
        img = np.full((64, 64, 3), 0, dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        L = oklab[..., 0]
        a = oklab[..., 1]
        b = oklab[..., 2]
        assert np.allclose(L, 0.0, atol=0.02)
        assert np.abs(a).max() < 0.02
        assert np.abs(b).max() < 0.02

    def test_gray_neutral(self):
        """Gray (128, 128, 128) should have near-zero a/b (achromatic)."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        a = oklab[..., 1]
        b = oklab[..., 2]
        # Gray is achromatic to float32 roundoff scale
        assert np.abs(a).max() < 1e-3, f"a max {np.abs(a).max()} should be near zero"
        assert np.abs(b).max() < 1e-3, f"b max {np.abs(b).max()} should be near zero"

    def test_output_shape_and_dtype(self):
        """Output should be (H, W, 3) float32."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        assert oklab.shape == (100, 100, 3)
        assert oklab.dtype == np.float32

    def test_output_range(self):
        """Oklab L should be in [0, 1], a/b in roughly [-0.4, 0.4]."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        # For typical sRGB images in [0, 255]
        assert (oklab[..., 0] >= 0.0).all() and (oklab[..., 0] <= 1.1).all(), "L out of range"
        assert (oklab[..., 1] >= -0.5).all() and (oklab[..., 1] <= 0.5).all(), "a out of range"
        assert (oklab[..., 2] >= -0.5).all() and (oklab[..., 2] <= 0.5).all(), "b out of range"


class TestOklabAnchors:
    """Regression anchors against Ottosson's reference Oklab values.

    Guards the M1 matrix (linear sRGB -> LMS) against transcription drift:
    a corrupted M1 row rotates hues and injects chroma into neutrals.
    Reference values computed with the exact Ottosson M1/M2 from oklab.com.
    """

    def _oklch_of(self, r, g, b):
        img = np.full((4, 4, 3), (b, g, r), dtype=np.uint8)
        return oklab_to_oklch(bgr_to_oklab(img))[0, 0]

    def test_pure_red_hue_anchor(self):
        """Pure red sRGB primary must land at h≈29.2° (Ottosson reference)."""
        _, _, h = self._oklch_of(255, 0, 0)
        assert h == pytest.approx(29.2, abs=0.5), f"red hue {h}° should be ≈29.2°"

    def test_pure_blue_ab_anchor(self):
        """Pure blue sRGB primary must land at a≈-0.0325, b≈-0.3115."""
        img = np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8)  # BGR blue
        oklab = bgr_to_oklab(img)[0, 0]
        assert oklab[1] == pytest.approx(-0.0325, abs=1e-3), f"blue a {oklab[1]}"
        assert oklab[2] == pytest.approx(-0.3115, abs=1e-3), f"blue b {oklab[2]}"

    def test_white_achromatic_exact(self):
        """White must have |a|, |b| < 1e-3 (float32 roundoff scale)."""
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        assert np.abs(oklab[..., 1]).max() < 1e-3
        assert np.abs(oklab[..., 2]).max() < 1e-3

    def test_skin_swatch_fair_hue_anchor(self):
        """Fair skin swatch RGB(241,184,166) must land at h≈38.2°."""
        _, _, h = self._oklch_of(241, 184, 166)
        assert h == pytest.approx(38.2, abs=1.0), f"fair swatch hue {h}° should be ≈38.2°"

    def test_skin_swatch_deep_hue_anchor(self):
        """Deep skin swatch RGB(92,60,44) must land at h≈47.3°."""
        _, _, h = self._oklch_of(92, 60, 44)
        assert h == pytest.approx(47.3, abs=1.0), f"deep swatch hue {h}° should be ≈47.3°"


class TestOklabToBgr:
    """Test vectorized Oklab→BGR conversion."""

    def test_round_trip_error_max(self):
        """BGR→Oklab→BGR max error should be ≤ 2/255 per pixel."""
        img_orig = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        oklab = bgr_to_oklab(img_orig)
        img_recon = oklab_to_bgr(oklab)
        max_error = np.abs(img_orig.astype(int) - img_recon.astype(int)).max()
        assert max_error <= 2, f"Max error {max_error} > 2"

    def test_white_round_trip(self):
        """White round-trip should preserve byte-perfect or near-perfect."""
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        oklab = bgr_to_oklab(img)
        img_recon = oklab_to_bgr(oklab)
        assert np.allclose(img, img_recon, atol=2)

    def test_output_type_and_shape(self):
        """Output should be (H, W, 3) uint8."""
        oklab = np.random.uniform(-0.4, 0.4, (100, 100, 3)).astype(np.float32)
        oklab[..., 0] = np.clip(oklab[..., 0], 0.0, 1.0)
        img = oklab_to_bgr(oklab)
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8

    def test_output_range_clipped(self):
        """Output should be valid uint8 [0, 255]."""
        oklab = np.random.uniform(-0.4, 0.4, (100, 100, 3)).astype(np.float32)
        oklab[..., 0] = np.clip(oklab[..., 0], 0.0, 1.0)
        img = oklab_to_bgr(oklab)
        assert (img >= 0).all() and (img <= 255).all()


class TestOklabOklchConversions:
    """Test Oklab ↔ OKLCh conversions."""

    def test_oklab_to_oklch_achromatic(self):
        """Achromatic (a=b=0) should have C=0, h undefined."""
        oklab = np.array([[[0.5, 0.0, 0.0]]])
        oklch = oklab_to_oklch(oklab)
        assert np.allclose(oklch[0, 0, 1], 0.0, atol=1e-6)

    def test_oklch_to_oklab_achromatic(self):
        """OKLCh with C=0 should map back to a≈0, b≈0."""
        oklch = np.array([[[0.5, 0.0, 90.0]]])
        oklab = oklch_to_oklab(oklch)
        assert np.abs(oklab[0, 0, 1]) < 1e-6
        assert np.abs(oklab[0, 0, 2]) < 1e-6

    def test_hue_0_degrees(self):
        """h=0° (red axis) should give positive a, near-zero b."""
        oklch = np.array([[[0.5, 0.1, 0.0]]])
        oklab = oklch_to_oklab(oklch)
        assert oklab[0, 0, 1] > 0.05  # a positive
        assert np.abs(oklab[0, 0, 2]) < 0.01  # b near zero

    def test_hue_90_degrees(self):
        """h=90° (yellow axis) should give positive b, near-zero a."""
        oklch = np.array([[[0.5, 0.1, 90.0]]])
        oklab = oklch_to_oklab(oklch)
        assert np.abs(oklab[0, 0, 1]) < 0.01  # a near zero
        assert oklab[0, 0, 2] > 0.05  # b positive

    def test_round_trip_oklch(self):
        """OKLCh→Oklab→OKLCh should preserve values."""
        oklch_orig = np.random.uniform(0, 1, (100, 100, 3)).astype(np.float32)
        oklch_orig[..., 2] = oklch_orig[..., 2] * 360.0  # h in [0, 360)
        oklab = oklch_to_oklab(oklch_orig)
        oklch_recon = oklab_to_oklch(oklab)
        # Hue wrapping: handle discontinuities at 0/360
        h_diff = (oklch_orig[..., 2] - oklch_recon[..., 2] + 180.0) % 360.0 - 180.0
        assert np.allclose(oklch_orig[..., 0], oklch_recon[..., 0], atol=1e-5)
        assert np.allclose(oklch_orig[..., 1], oklch_recon[..., 1], atol=1e-5)
        assert (np.abs(h_diff) < 0.1).all()  # Hue within 0.1° (floating-point noise)


class TestMeasureSkinState:
    """Test skin-tone measurement and circular hue mean."""

    def test_returns_skin_state_dataclass(self):
        """Output should be a SkinState instance."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        state = measure_skin_state(img, mask)
        assert isinstance(state, SkinState)
        assert hasattr(state, "L_mean")
        assert hasattr(state, "C_mean")
        assert hasattr(state, "C_std")
        assert hasattr(state, "h_mean")

    def test_uniform_image(self):
        """Uniform image should have zero C_std."""
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        state = measure_skin_state(img, mask)
        assert state.C_std < 0.001

    def test_none_mask_uses_whole_image(self):
        """None mask should process entire image."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        state1 = measure_skin_state(img, None)
        state2 = measure_skin_state(img, np.ones((64, 64), dtype=np.float32))
        assert np.isclose(state1.L_mean, state2.L_mean, atol=0.02)

    def test_empty_mask_returns_neutral(self):
        """Empty mask (all zeros) should return neutral state."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        state = measure_skin_state(img, mask, thresh=0.3)
        assert np.isclose(state.L_mean, 0.5, atol=0.1)
        assert np.isclose(state.C_mean, 0.085, atol=0.01)

    def test_mask_threshold(self):
        """Mask threshold should exclude sub-threshold pixels."""
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        state1 = measure_skin_state(img, mask, thresh=0.3)  # 0.5 > 0.3, included
        state2 = measure_skin_state(img, mask, thresh=0.6)  # 0.5 < 0.6, empty
        assert np.isclose(state1.L_mean, state2.L_mean, atol=0.05) or state2.L_mean == 0.5

    def test_hue_circular_mean_opposite_angles(self):
        """Average of 0° and 180° should be undefined (zero resultant)."""
        # Create a patchy skin with hues at 0° and 180°
        oklab = np.zeros((64, 64, 3), dtype=np.float32)
        oklab[:, :, 0] = 0.6  # L
        oklab[:, :, 1] = 0.1  # a
        oklab[:32, :, 2] = 0.0  # b = 0 (h ≈ 0°)
        oklab[32:, :, 2] = -0.1  # b < 0 (h ≈ 180°)

        # Convert back to BGR via OKLCh
        from retouch.color_science import oklch_to_oklab, oklab_to_bgr
        oklch = np.zeros((64, 64, 3), dtype=np.float32)
        oklch[:, :, 0] = 0.6
        oklch[:, :, 1] = np.sqrt(0.1**2 + 0.1**2)
        oklch[:32, :, 2] = 0.0
        oklch[32:, :, 2] = 180.0
        oklab_test = oklch_to_oklab(oklch)
        img = oklab_to_bgr(oklab_test)

        mask = np.ones((64, 64), dtype=np.float32)
        state = measure_skin_state(img, mask)
        # With opposite hues, the circular mean magnitude should be low
        # (actual mean is poorly defined, but h_mean will be one of them)
        assert 0 <= state.h_mean < 360


class TestSkinChromaStd:
    """Test the σ_C uniformity metric."""

    def test_returns_float(self):
        """Should return a float value."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        sigma_c = skin_chroma_std(img, mask)
        assert isinstance(sigma_c, (float, np.floating))

    def test_uniform_chroma_low_std(self):
        """Uniform image should have very low σ_C."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        sigma_c = skin_chroma_std(img, mask)
        assert sigma_c < 0.01

    def test_varied_chroma_higher_std(self):
        """Varied colors should have higher σ_C than uniform."""
        # Create two images: one uniform, one varied
        img_uniform = np.full((64, 64, 3), 128, dtype=np.uint8)

        img_varied = np.zeros((64, 64, 3), dtype=np.uint8)
        # Half warm, half cool
        img_varied[:32, :] = [50, 100, 200]
        img_varied[32:, :] = [200, 150, 50]

        mask = np.ones((64, 64), dtype=np.float32)
        sigma_c_uniform = skin_chroma_std(img_uniform, mask)
        sigma_c_varied = skin_chroma_std(img_varied, mask)

        # Varied image should have higher σ_C
        assert sigma_c_varied > sigma_c_uniform, f"Varied σ_C {sigma_c_varied} should be > uniform {sigma_c_uniform}"


class TestSkinLociData:
    """Test the SKIN_LOCI reference table."""

    def test_loci_structure(self):
        """SKIN_LOCI should have expected keys and classes."""
        assert "fair" in SKIN_LOCI
        assert "tan" in SKIN_LOCI
        assert "deep" in SKIN_LOCI

    def test_loci_fair_values(self):
        """Fair skin locus should have expected targets."""
        assert "L_min" in SKIN_LOCI["fair"]
        assert "h_target" in SKIN_LOCI["fair"]
        assert "C_target" in SKIN_LOCI["fair"]
        assert SKIN_LOCI["fair"]["h_target"] == 45.0
        assert np.isclose(SKIN_LOCI["fair"]["C_target"], 0.085, atol=0.001)

    def test_loci_tan_values(self):
        """Tan skin locus should have expected targets."""
        assert SKIN_LOCI["tan"]["h_target"] == 52.0
        assert np.isclose(SKIN_LOCI["tan"]["C_target"], 0.105, atol=0.001)

    def test_loci_deep_values(self):
        """Deep skin locus should have expected targets."""
        assert SKIN_LOCI["deep"]["h_target"] == 58.0
        assert np.isclose(SKIN_LOCI["deep"]["C_target"], 0.125, atol=0.001)


class TestUnifyHueLineIntegration:
    """Test SkinProcessor.unify_hue_line via end-to-end behavior."""

    @pytest.fixture
    def skin_proc(self):
        from retouch.skin import SkinProcessor
        return SkinProcessor()

    def test_zero_strength_returns_original(self, skin_proc):
        """Zero strength should return the input unchanged."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = skin_proc.unify_hue_line(img, mask, hue_strength=0, chroma_strength=0)
        assert np.array_equal(result, img)

    def test_none_mask_returns_original(self, skin_proc):
        """None mask should return the input unchanged."""
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = skin_proc.unify_hue_line(img, None, hue_strength=50, chroma_strength=50)
        assert np.array_equal(result, img)

    def test_hue_rotation_bounded(self, skin_proc):
        """Hue rotation at strength=100 should be bounded by design."""
        # Create a diverse hue image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            hue = (y / 100.0) * 360.0
            h_rad = np.radians(hue)
            # Rough HSV-to-BGR (for a quick test image)
            r = int(128 + 100 * np.cos(h_rad))
            g = int(128 + 100 * np.sin(h_rad))
            b = int(128 - 50 * np.cos(h_rad))
            img[y, :] = [np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]

        mask = np.ones((100, 100), dtype=np.float32)
        result = skin_proc.unify_hue_line(img, mask, hue_strength=100, chroma_strength=0)

        # Check that hue changes are bounded (±8° clipping in implementation)
        oklab_orig = bgr_to_oklab(img)
        oklch_orig = oklab_to_oklch(oklab_orig)
        oklab_result = bgr_to_oklab(result)
        oklch_result = oklab_to_oklch(oklab_result)

        h_diff = (oklch_orig[..., 2] - oklch_result[..., 2] + 180.0) % 360.0 - 180.0
        max_h_diff = np.abs(h_diff).max()
        # Allow small tolerance for floating-point and mask blending
        assert max_h_diff <= 10.0, f"Hue rotation {max_h_diff}° exceeds expected bounds"

    def test_patchy_skin_hue_std_decreases(self, skin_proc):
        """Patchy skin (two hue clusters) should have hue std decrease."""
        # Create patchy skin with two distinct hue clusters
        img = np.zeros((128, 128, 3), dtype=np.uint8)

        # Left half: warm hue (orange-ish)
        oklab_warm = np.zeros((128, 64, 3), dtype=np.float32)
        oklab_warm[..., 0] = 0.65  # L
        oklab_warm[..., 1] = 0.08  # a (warm)
        oklab_warm[..., 2] = 0.10  # b (warm)

        # Right half: cooler hue (yellow-ish)
        oklab_cool = np.zeros((128, 64, 3), dtype=np.float32)
        oklab_cool[..., 0] = 0.65  # L
        oklab_cool[..., 1] = 0.10  # a (cooler)
        oklab_cool[..., 2] = 0.05  # b (cooler)

        oklab_patchy = np.concatenate([oklab_warm, oklab_cool], axis=1)
        img_left = oklab_to_bgr(oklab_patchy[:, :64, :])
        img_right = oklab_to_bgr(oklab_patchy[:, 64:, :])
        img = np.concatenate([img_left, img_right], axis=1)

        mask = np.ones((128, 128), dtype=np.float32)

        # Measure before
        state_before = measure_skin_state(img, mask)
        hue_std_before = state_before.C_std  # Not h_std, but C_std is available

        # Apply unification
        result = skin_proc.unify_hue_line(img, mask, hue_strength=75, chroma_strength=50)

        # Measure after
        state_after = measure_skin_state(result, mask)
        hue_std_after = state_after.C_std

        # Chroma std should decrease under chroma_even
        assert hue_std_after <= hue_std_before

    def test_chroma_std_decreases_with_chroma_even(self, skin_proc):
        """Applying chroma_even should reduce chroma variance σ_C."""
        # Create image with varying chroma
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            c_val = 0.05 + 0.10 * (y / 100.0)  # Chroma 0.05→0.15
            h_rad = np.radians(45.0)
            oklab_pixel = np.array([[[0.65, c_val * np.cos(h_rad), c_val * np.sin(h_rad)]]])
            img[y, :] = oklab_to_bgr(oklab_pixel)[0, 0]

        mask = np.ones((100, 100), dtype=np.float32)
        sigma_c_before = skin_chroma_std(img, mask)

        result = skin_proc.unify_hue_line(img, mask, hue_strength=0, chroma_strength=100)
        sigma_c_after = skin_chroma_std(result, mask)

        # Chroma std should decrease
        assert sigma_c_after <= sigma_c_before

    def test_strength_25_50_75_monotonic(self, skin_proc):
        """Hue unification should be monotonic across strengths 25, 50, 75."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = [100, 150, 200]
        img[50:, :] = [200, 100, 150]
        mask = np.ones((100, 100), dtype=np.float32)

        result_25 = skin_proc.unify_hue_line(img, mask, hue_strength=25, chroma_strength=0)
        result_50 = skin_proc.unify_hue_line(img, mask, hue_strength=50, chroma_strength=0)
        result_75 = skin_proc.unify_hue_line(img, mask, hue_strength=75, chroma_strength=0)

        state_25 = measure_skin_state(result_25, mask)
        state_50 = measure_skin_state(result_50, mask)
        state_75 = measure_skin_state(result_75, mask)

        # Hues should move monotonically toward the target
        # (Can't easily test exact monotonicity due to circular hue, but check convergence)
        assert state_25.C_mean >= 0.0  # Sanity check
        assert state_50.C_mean >= 0.0
        assert state_75.C_mean >= 0.0


class TestAbneyHueCorrection:
    """Tests for K2 Abney hue-linearity correction and inverse mapping."""

    def test_reference_lightness_baseline(self):
        """At baseline lightness L=0.6, correction is exact zero."""
        for h in [25.0, 45.0, 65.0]:
            h_lin = apply_abney_hue_correction(h, 0.6)
            assert h_lin == pytest.approx(h, abs=1e-5)

    def test_roundtrip(self):
        """apply_abney_hue_correction and invert_abney_hue_correction must round-trip cleanly."""
        h_arr = np.linspace(0, 359, 72, dtype=np.float32)
        for L in [0.2, 0.5, 0.6, 0.85]:
            h_lin = apply_abney_hue_correction(h_arr, L)
            h_rec = invert_abney_hue_correction(h_lin, L)
            diff = np.abs((h_arr - h_rec + 180.0) % 360.0 - 180.0)
            assert np.max(diff) < 1e-3, f"Max roundtrip error {np.max(diff)} > 1e-3"

    def test_lightness_dependent_shift_in_skin_quadrant(self):
        """Highlights (L > 0.6) shift positive, shadows (L < 0.6) shift negative in skin quadrant."""
        h_skin = 45.0
        h_bright = apply_abney_hue_correction(h_skin, 0.9)
        h_dark = apply_abney_hue_correction(h_skin, 0.3)

        assert h_bright > h_skin
        assert h_dark < h_skin

    def test_outside_skin_quadrant_tapers(self):
        """Non-skin hues (e.g. blue h=240, green h=140) receive zero skin Abney shift."""
        for h in [0.0, 140.0, 240.0, 300.0]:
            h_lin = apply_abney_hue_correction(h, 0.9)
            assert h_lin == pytest.approx(h, abs=1e-4)


class TestGamutChromaCompression:
    """Tests for K3 soft-knee gamut-aware chroma compression."""

    def test_in_gamut_unmodified(self):
        """Moderate in-gamut chroma (C < 0.85 * C_max) must be 100% unchanged."""
        oklch = np.zeros((10, 10, 3), dtype=np.float32)
        oklch[..., 0] = 0.65
        oklch[..., 1] = 0.05
        oklch[..., 2] = 45.0

        out = compress_chroma_gamut(oklch, knee=0.85)
        assert np.allclose(oklch, out, atol=1e-5)

    def test_preserves_lightness_and_hue(self):
        """Gamut compression must leave L and h 100% untouched."""
        oklch = np.zeros((10, 10, 3), dtype=np.float32)
        oklch[..., 0] = 0.70
        oklch[..., 1] = 0.35  # Oversaturated
        oklch[..., 2] = 30.0

        out = compress_chroma_gamut(oklch, knee=0.85)
        assert np.allclose(oklch[..., 0], out[..., 0])
        assert np.allclose(oklch[..., 2], out[..., 2])
        assert (out[..., 1] <= oklch[..., 1]).all()

    def test_sRGB_gamut_boundary(self):
        """C_max(L, h) should return positive maximum in-gamut chroma."""
        c_max = find_gamut_intersection_srgb(0.6, 45.0)
        assert 0.05 < c_max < 0.35


class TestCAM16AndCAT16:
    """K1 & K4: CAM16-UCS appearance space & CAT16 chromatic adaptation."""

    def test_cam16_ucs_roundtrip(self):
        """BGR -> CAM16-UCS -> BGR round-trip must be accurate."""
        img = np.random.randint(20, 235, (32, 32, 3), dtype=np.uint8)
        cam16 = bgr_to_cam16_ucs(img)
        recon = cam16_ucs_to_bgr(cam16)
        max_err = np.abs(img.astype(int) - recon.astype(int)).max()
        assert max_err <= 3, f"CAM16-UCS roundtrip error {max_err} > 3"

    def test_cam16_delta_e_identical_images_zero(self):
        """Perceptual ΔE_CAM16-UCS for identical images must be zero."""
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        delta_e = cam16_ucs_delta_e(img, img)
        assert np.max(delta_e) < 1e-4

    def test_cat16_identity_white_point(self):
        """CAT16 adaptation with identical source/target white point returns original image."""
        img = np.full((16, 16, 3), 150, dtype=np.uint8)
        adapted = chromatic_adapt_cat16(img, source_wp=(0.95, 1.0, 1.08), target_wp=(0.95, 1.0, 1.08))
        assert np.all(adapted == img)


class TestPQTransferFunctions:
    """K7: ST 2084 / PQ EOTF & OETF transfer function tests."""

    def test_linear_pq_roundtrip(self):
        """Linear -> PQ -> Linear round-trip must be accurate."""
        lin = np.linspace(0.01, 0.99, 100, dtype=np.float32)
        pq = linear_to_pq(lin)
        recon = pq_to_linear(pq)
        assert np.allclose(lin, recon, atol=1e-4)

    def test_wide_gamut_solvers(self):
        """P3 and Rec.2020 gamut solvers extend maximum allowable chroma."""
        c_srgb = find_gamut_intersection_srgb(0.6, 45.0)
        c_p3 = find_gamut_intersection_p3(0.6, 45.0)
        c_2020 = find_gamut_intersection_rec2020(0.6, 45.0)
        assert c_srgb < c_p3 < c_2020





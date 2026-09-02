"""Tests for retouch/undereye.py — UnderEyeRepairer, UndereyeProcessor, and friends."""
import numpy as np
import pytest
import cv2

from retouch.undereye import (
    UnderEyeRepairer,
    UndereyeProcessor,
    UndereyeAnalyzer,
    UndereyeRemover,
)
from retouch.chromophore import decompose_chromophores
from retouch.utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32


class MockFaceRegions:
    def __init__(self):
        self.left_under_eye = None
        self.right_under_eye = None


@pytest.fixture
def repairer():
    return UnderEyeRepairer()


@pytest.fixture
def processor():
    return UndereyeProcessor()


@pytest.fixture
def analyzer():
    return UndereyeAnalyzer()


@pytest.fixture
def remover():
    return UndereyeRemover()


@pytest.fixture
def img_uint8():
    """Create a uint8 image with simulated dark circles."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    # Add darker region to simulate dark circles
    img[30:40, 20:44] = [80, 85, 90]  # Dark but not pure
    return img


@pytest.fixture
def img_float32():
    """Create a float32 image with simulated dark circles."""
    img = np.full((64, 64, 3), 128.0, dtype=np.float32)
    img[30:40, 20:44] = [80.0, 85.0, 90.0]
    return img


@pytest.fixture
def undereye_mask():
    """Create a simple under-eye region mask."""
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[30:40, 20:44] = 1.0
    return mask


@pytest.fixture
def cheek_reference_img():
    """Create an image where cheeks are bright and under-eye is dark."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    img[:25, :] = [180, 175, 170]  # Bright cheek area
    img[30:40, 20:44] = [60, 65, 70]  # Dark under-eye area
    return img


# ---- Test UndereyeAnalyzer ----


class TestUndereyeAnalyzer:
    """Test dark-circle detection via LAB L-channel analysis."""

    def test_detect_dark_circles_empty_mask(self, analyzer, img_uint8):
        """Empty mask should return zero detection and NaN reference."""
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
        empty_mask = np.zeros((64, 64), dtype=np.float32)
        detection, ref_l = analyzer.detect_dark_circles(lab, empty_mask)
        assert detection.max() < 0.01
        assert np.isnan(ref_l)

    def test_detect_dark_circles_uniform_image(self, analyzer):
        """Uniform image (no dark circles) should return minimal detection."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.float32)
        detection, ref_l = analyzer.detect_dark_circles(lab, mask)
        # No real dark circles, so detection should be minimal or empty
        assert detection.max() <= 0.5  # Some noise possible, but mostly empty
        assert not np.isnan(ref_l)

    def test_detect_dark_circles_with_real_dark_area(self, analyzer, cheek_reference_img):
        """Image with actual dark circles should detect them."""
        lab = cv2.cvtColor(cheek_reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:40, 20:44] = 1.0  # Under-eye region
        detection, ref_l = analyzer.detect_dark_circles(lab, mask, threshold_offset=15.0)
        assert not np.isnan(ref_l)
        assert ref_l > 0  # Should have a valid reference
        # Detection should exist in the dark region
        assert detection[30:40, 20:44].max() > 0.5

    def test_detect_dark_circles_returns_tuple(self, analyzer, img_uint8, undereye_mask):
        """Method should return (mask, float) tuple."""
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
        result = analyzer.detect_dark_circles(lab, undereye_mask)
        assert isinstance(result, tuple)
        assert len(result) == 2
        detection_mask, ref_l = result
        assert isinstance(detection_mask, np.ndarray)
        assert detection_mask.dtype in (np.float32, np.float64)


# ---- Test UndereyeRemover ----


class TestUndereyeRemover:
    """Test selective L-brightening and chroma reduction."""

    def test_brighten_dark_circles_zero_strength(self, remover):
        """Zero strength should return unchanged LAB."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_orig = lab.copy()
        dark_mask = np.ones((64, 64), dtype=np.float32)
        result = remover.brighten_dark_circles(lab, dark_mask, 120.0, strength=0.0)
        assert np.allclose(result, lab_orig)

    def test_brighten_dark_circles_modifies_l_channel(self, remover):
        """With strength > 0, L-channel should increase in dark areas."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Create a region with lower L than reference
        lab[30:40, 20:44, 0] = 80.0
        lab_orig_l = lab[:, :, 0].copy()
        dark_mask = np.zeros((64, 64), dtype=np.float32)
        dark_mask[30:40, 20:44] = 1.0
        result = remover.brighten_dark_circles(lab, dark_mask, 120.0, strength=0.8, max_lift=30.0)
        # Dark region L-values should have increased
        assert result[30:40, 20:44, 0].mean() > lab_orig_l[30:40, 20:44].mean()

    def test_reduce_puffiness_chroma_zero_strength(self, remover):
        """Zero strength should return unchanged LAB."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_orig = lab.copy()
        dark_mask = np.ones((64, 64), dtype=np.float32)
        result = remover.reduce_puffiness_chroma(lab, dark_mask, strength=0.0)
        assert np.allclose(result, lab_orig)

    def test_reduce_puffiness_chroma_desaturates(self, remover):
        """Puffiness reduction should reduce chroma in masked regions."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Make a red-tinted region (high chroma)
        img[30:40, 20:44] = [100, 150, 128]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_ab = np.sqrt((lab[30:40, 20:44, 1] - 128.0) ** 2 + (lab[30:40, 20:44, 2] - 128.0) ** 2)
        dark_mask = np.zeros((64, 64), dtype=np.float32)
        dark_mask[30:40, 20:44] = 1.0
        result = remover.reduce_puffiness_chroma(lab, dark_mask, strength=0.8)
        new_ab = np.sqrt((result[30:40, 20:44, 1] - 128.0) ** 2 + (result[30:40, 20:44, 2] - 128.0) ** 2)
        # Chroma should be reduced
        assert new_ab.mean() < orig_ab.mean()

    def test_feather_edges_softens_mask(self, remover):
        """Feathering should soften sharp edges."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0  # Hard square
        feathered = remover.feather_edges(mask, feather_radius=5)
        # Edges should be softened (values between 0 and 1)
        edge_pixels = feathered[19, 20:44]  # Top edge
        assert (edge_pixels > 0.0).any() and (edge_pixels < 1.0).any()

    def test_feather_edges_preserves_center(self, remover):
        """Feathering should keep center region at ~1.0."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        feathered = remover.feather_edges(mask, feather_radius=5)
        # Center should still be close to 1.0
        assert feathered[30:40, 30:40].min() > 0.9


# ---- Test UndereyeProcessor ----


class TestUndereyeProcessor:
    """Test the orchestrated under-eye processing pipeline."""

    def test_process_zero_strength_uint8(self, processor, img_uint8, undereye_mask):
        """Zero strength should return unchanged image (uint8)."""
        result = processor.process(
            img_uint8, undereye_mask,
            darken_removal_strength=0.0,
            puffiness_reduction_strength=0.0
        )
        assert np.array_equal(result, img_uint8)
        assert result.dtype == np.uint8

    def test_process_zero_strength_float32(self, processor, img_float32, undereye_mask):
        """Zero strength should return unchanged image (float32)."""
        result = processor.process(
            img_float32, undereye_mask,
            darken_removal_strength=0.0,
            puffiness_reduction_strength=0.0
        )
        assert np.allclose(result, img_float32)
        assert result.dtype == np.float32

    def test_process_empty_mask(self, processor, img_uint8):
        """Empty mask should return unchanged image."""
        empty_mask = np.zeros((64, 64), dtype=np.float32)
        result = processor.process(
            img_uint8, empty_mask,
            darken_removal_strength=0.5,
            puffiness_reduction_strength=0.5
        )
        assert np.array_equal(result, img_uint8)

    def test_process_darken_removal_brightens_darks(self, processor, cheek_reference_img, undereye_mask):
        """Darken removal should brighten dark under-eye regions."""
        result = processor.process(
            cheek_reference_img, undereye_mask,
            darken_removal_strength=0.8,
            puffiness_reduction_strength=0.0
        )
        lab_orig = cv2.cvtColor(cheek_reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        # L-channel in under-eye region should have increased
        orig_l = lab_orig[30:40, 20:44, 0].mean()
        result_l = lab_result[30:40, 20:44, 0].mean()
        assert result_l > orig_l

    def test_process_dtype_preservation_uint8(self, processor, img_uint8, undereye_mask):
        """Uint8 input should return uint8."""
        result = processor.process(
            img_uint8, undereye_mask,
            darken_removal_strength=0.5,
            puffiness_reduction_strength=0.0
        )
        assert result.dtype == np.uint8

    def test_process_dtype_preservation_float32(self, processor, img_float32, undereye_mask):
        """Float32 input should return float32."""
        result = processor.process(
            img_float32, undereye_mask,
            darken_removal_strength=0.5,
            puffiness_reduction_strength=0.0
        )
        assert result.dtype == np.float32

    def test_process_shape_preserved(self, processor, img_uint8, undereye_mask):
        """Output shape should match input."""
        result = processor.process(
            img_uint8, undereye_mask,
            darken_removal_strength=0.5,
            puffiness_reduction_strength=0.0
        )
        assert result.shape == img_uint8.shape

    def test_process_puffiness_reduction_desaturates(self, processor, cheek_reference_img, undereye_mask):
        """Puffiness reduction should reduce chroma under the eyes."""
        result = processor.process(
            cheek_reference_img, undereye_mask,
            darken_removal_strength=0.0,
            puffiness_reduction_strength=0.8
        )
        lab_orig = cv2.cvtColor(cheek_reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Chroma in under-eye should be reduced
        orig_chroma = np.sqrt(
            (lab_orig[30:40, 20:44, 1] - 128.0) ** 2 + (lab_orig[30:40, 20:44, 2] - 128.0) ** 2
        ).mean()
        result_chroma = np.sqrt(
            (lab_result[30:40, 20:44, 1] - 128.0) ** 2 + (lab_result[30:40, 20:44, 2] - 128.0) ** 2
        ).mean()
        assert result_chroma < orig_chroma

    def test_hemoglobin_spike_zero_strength_is_identity(self, processor, cheek_reference_img, undereye_mask):
        result = processor.attenuate_hemoglobin(
            cheek_reference_img, undereye_mask, strength=0.0
        )
        assert np.array_equal(result, cheek_reference_img)

    def test_hemoglobin_spike_reduces_vascular_excess_without_lifting_luminance(self, processor):
        """E-EYE-4 should remove excess vascular color, not brighten a shadow."""
        img = np.full((80, 80, 3), (110, 130, 160), dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[30:50, 24:56] = 1.0
        # Strong red-purple vascular signal inside a bounded under-eye area.
        img[30:50, 24:56] = (80, 85, 180)

        hb_before = decompose_chromophores(img)[1]
        out = processor.attenuate_hemoglobin(img, mask, strength=0.8)
        hb_after = decompose_chromophores(out)[1]
        source_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        output_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        inside = mask > 0.5

        assert float(hb_after[inside].mean()) < float(hb_before[inside].mean())
        assert abs(float(output_lab[inside, 0].mean() - source_lab[inside, 0].mean())) <= 2.0
        assert np.array_equal(out[~inside], img[~inside])

    def test_hemoglobin_spike_preserves_float32_contract(self, processor):
        img = np.full((64, 64, 3), (110.0, 130.0, 160.0), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[24:40, 20:44] = 1.0
        img[24:40, 20:44] = (80.0, 85.0, 180.0)
        out = processor.attenuate_hemoglobin(img, mask, strength=0.7)
        assert out.dtype == np.float32
        assert out.shape == img.shape


# ---- Test UnderEyeRepairer (legacy interface) ----


class TestUnderEyeRepairer:
    """Test backward-compatible legacy interface."""

    def test_repair_zero_strength(self, repairer, img_uint8):
        """Zero strength should return unchanged image."""
        regions = MockFaceRegions()
        result = repairer.repair(img_uint8, regions, strength=0)
        assert np.all(result == img_uint8)

    def test_repair_none_masks(self, repairer, img_uint8):
        """None masks should return unchanged image."""
        regions = MockFaceRegions()
        regions.left_under_eye = None
        regions.right_under_eye = None
        result = repairer.repair(img_uint8, regions, strength=50)
        assert np.all(result == img_uint8)

    def test_repair_empty_masks(self, repairer, img_uint8):
        """Empty masks should return unchanged image."""
        regions = MockFaceRegions()
        regions.left_under_eye = np.zeros((64, 64), dtype=np.float32)
        regions.right_under_eye = np.zeros((64, 64), dtype=np.float32)
        result = repairer.repair(img_uint8, regions, strength=50)
        assert np.all(result == img_uint8)

    def test_repair_output_shape(self, repairer, img_uint8):
        """Output shape should match input."""
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
        result = repairer.repair(img_uint8, regions, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_repair_modifies_dark_region(self, repairer, cheek_reference_img):
        """Repair should brighten dark under-eye regions."""
        regions = MockFaceRegions()
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:40, 20:44] = 1.0
        regions.left_under_eye = mask.copy()
        regions.right_under_eye = np.zeros_like(mask)
        result = repairer.repair(cheek_reference_img, regions, strength=80)
        # Result should differ from input (brightening occurred)
        assert not np.array_equal(result, cheek_reference_img)

    def test_repair_output_dtype(self, repairer, img_uint8):
        """Output should be uint8."""
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32) * 0.5
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32) * 0.5
        result = repairer.repair(img_uint8, regions, strength=75)
        assert result.dtype == np.uint8

    def test_repair_strength_zero_with_masks(self, repairer, img_uint8):
        """Zero strength with valid masks should still return unchanged."""
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
        result = repairer.repair(img_uint8, regions, strength=0)
        assert np.array_equal(result, img_uint8)


# ---- Edge case tests ----


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multi_face_handling(self, processor):
        """Test processing multiple under-eye regions in sequence."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[20:30, 10:20] = [80, 85, 90]  # Left eye
        img[20:30, 44:54] = [80, 85, 90]  # Right eye

        left_mask = np.zeros((64, 64), dtype=np.float32)
        left_mask[20:30, 10:20] = 1.0

        result = processor.process(img, left_mask, darken_removal_strength=0.5)
        # Both eyes should be unmodified (only left is masked), but processing should complete
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_extreme_darkness(self, processor):
        """Test with very dark under-eye region."""
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        img[30:40, 20:44] = [20, 25, 30]  # Very dark
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:40, 20:44] = 1.0
        result = processor.process(img, mask, darken_removal_strength=1.0)
        # Should brighten the dark region (conservative cap at 30L increase)
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_mean_l = lab_orig[30:40, 20:44, 0].mean()
        result_mean_l = lab_result[30:40, 20:44, 0].mean()
        assert result_mean_l > orig_mean_l  # Brightened
        assert result_mean_l < orig_mean_l + 35  # Conservative cap

    def test_large_image(self, processor):
        """Test with larger image dimensions."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        img[100:120, 80:120] = [80, 85, 90]
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[100:120, 80:120] = 1.0
        result = processor.process(img, mask, darken_removal_strength=0.5)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_small_mask_region(self, processor):
        """Test with tiny under-eye mask region."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[30, 30] = [80, 85, 90]
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30, 30] = 1.0
        result = processor.process(img, mask, darken_removal_strength=0.5)
        # Should handle gracefully without crashing
        assert result.dtype == np.uint8


# ---- v2 (2026-09-02) — support geometry, lash exclusion, tone invariance ----


def _v2_scene(scale=1.0, ied=100):
    """Uniform skin with a blurred dark blob under a fake eye opening.

    Returns (img_uint8, under_eye_polygon_mask, eye_hull, shadow_core_mask).
    """
    rng = np.random.default_rng(0)
    base = np.array([150, 170, 200], np.float32) * scale
    img = np.clip(base + rng.normal(0, 3, (300, 300, 1)), 0, 255)
    eye = np.zeros((300, 300), np.float32)
    cv2.ellipse(eye, (150, 100), (28, 12), 0, 0, 360, 1, -1)
    ue = np.zeros((300, 300), np.float32)
    cv2.ellipse(ue, (150, 115), (30, 10), 0, 0, 360, 1, -1)  # top edge = lash line
    shadow = np.zeros((300, 300), np.float32)
    cv2.ellipse(shadow, (150, 132), (28, 12), 0, 0, 360, 1, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 5)
    img = np.clip(img * (1.0 - 0.25 * shadow[..., None]), 0, 255).astype(np.uint8)
    return img, ue, eye, shadow > 0.8


def _L(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


class TestUndereyeV2:
    def test_dilate_down_only_extends_downward(self):
        from retouch.undereye import _dilate_down
        m = np.zeros((40, 40), np.uint8)
        m[10:15, 10:30] = 1
        d = _dilate_down(m, 5, 2)
        rows = np.nonzero(d.any(1))[0]
        cols = np.nonzero(d.any(0))[0]
        assert (rows.min(), rows.max()) == (10, 19)      # 5 px down, none up
        assert (cols.min(), cols.max()) == (8, 31)       # 2 px sideways

    def test_support_excludes_eye_and_lash_margin(self):
        from retouch.undereye import build_undereye_support
        img, ue, eye, _ = _v2_scene()
        sup, ring, valid = build_undereye_support(ue, 100.0, exclude=eye)
        assert float((sup * eye).max()) == 0.0
        lash = cv2.dilate((eye > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        assert float(sup[lash].max()) < 0.5   # feathered, but never full weight on lashes
        assert ring.sum() > 200 and not (ring & (eye > 0.5)).any()
        # no hard step anywhere (the first v2 draft cut the lash margin after
        # feathering and left a 0.76 step against the lash band, DSCF4576)
        gy = np.abs(np.diff(sup, axis=0)).max()
        gx = np.abs(np.diff(sup, axis=1)).max()
        assert max(gy, gx) < 0.3

    def test_lift_lands_on_shadow_not_lashes_and_preserves_texture(self, processor):
        img, ue, eye, core = _v2_scene()
        out = processor.process(img, ue, darken_removal_strength=1.0, ied=100.0, exclude=eye)
        L0, L1 = _L(img), _L(out)
        assert (L1[core] - L0[core]).mean() > 15.0          # shadow lifted
        assert np.array_equal(out[eye > 0.5], img[eye > 0.5])  # eye bit-exact
        hp0 = L0 - cv2.GaussianBlur(L0, (0, 0), 3)
        hp1 = L1 - cv2.GaussianBlur(L1, (0, 0), 3)
        assert abs(hp1[core].std() - hp0[core].std()) < 1.0   # texture kept

    def test_outside_support_bit_exact_uint8_and_float32(self, processor):
        img, ue, eye, _ = _v2_scene()
        sup = processor.last_support(ue, 100.0, exclude=eye)
        out = processor.process(img, ue, darken_removal_strength=0.7, ied=100.0, exclude=eye)
        assert out.dtype == np.uint8
        assert np.array_equal(out[sup == 0], img[sup == 0])
        imgf = img.astype(np.float32)
        outf = processor.process(imgf, ue, darken_removal_strength=0.7, ied=100.0, exclude=eye)
        assert outf.dtype == np.float32
        assert np.array_equal(outf[sup == 0], imgf[sup == 0])
        assert (outf != imgf).any()

    def test_strength_is_monotonic(self, processor):
        img, ue, eye, core = _v2_scene()
        L0 = _L(img)
        lifts = []
        for s in (0.2, 0.5, 0.8):
            out = processor.process(img, ue, darken_removal_strength=s, ied=100.0, exclude=eye)
            lifts.append((_L(out)[core] - L0[core]).mean())
        assert lifts[0] > 1.0 and lifts[0] < lifts[1] < lifts[2]

    def test_tone_invariance_relative_threshold(self, processor):
        """The same relative shadow on a darker complexion must still fire
        (the v1 absolute 15-L threshold did not)."""
        lifts = {}
        for scale in (1.0, 0.45):
            img, ue, eye, core = _v2_scene(scale=scale)
            out = processor.process(img, ue, darken_removal_strength=1.0, ied=100.0, exclude=eye)
            lifts[scale] = (_L(out)[core] - _L(img)[core]).mean() / max(_L(img)[core].mean(), 1)
        # relative lift within 35% of each other across a ~2.2x luminance change
        assert lifts[0.45] > 0.05
        assert abs(lifts[0.45] - lifts[1.0]) / lifts[1.0] < 0.35

    def test_no_shadow_means_no_change(self, processor):
        rng = np.random.default_rng(1)
        img = np.clip(np.array([150, 170, 200], np.float32) + rng.normal(0, 3, (300, 300, 1)), 0, 255).astype(np.uint8)
        _, ue, eye, _ = _v2_scene()
        out = processor.process(img, ue, darken_removal_strength=1.0, ied=100.0, exclude=eye)
        assert np.abs(_L(out) - _L(img)).max() <= 1.0

    def test_no_exclude_fallback_keeps_thin_polygon(self, processor):
        """Without an eye contour the lash shave must not erase a ~0.12-IED
        tall polygon (the real landmark polygon height)."""
        from retouch.undereye import build_undereye_support
        ue = np.zeros((300, 300), np.float32)
        ue[100:112, 100:180] = 1.0   # 12 px tall at IED 100
        sup, ring, valid = build_undereye_support(ue, 100.0)
        assert (sup > 0.5).sum() > 500
        assert ring.sum() > 100

    def test_legacy_brighten_single_cap(self, remover):
        """max_lift is the only cap: darkness 20 at strength 1 lifts by 20."""
        lab = np.full((32, 32, 3), 128.0, np.float32)
        lab[8:24, 8:24, 0] = 100.0
        mask = np.zeros((32, 32), np.float32)
        mask[8:24, 8:24] = 1.0
        out = remover.brighten_dark_circles(lab.copy(), mask, 120.0, strength=1.0, max_lift=30.0)
        assert abs(out[16, 16, 0] - 120.0) < 1e-4

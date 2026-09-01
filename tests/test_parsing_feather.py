"""Tests for mask feathering refactor (Stage 0-1 of PLAN_C3_SOFT_MASKS)."""

import numpy as np
import pytest
import cv2
from retouch.parsing import _masks_from_label_map


class TestMasksFromLabelMapGaussian:
    """Test that _masks_from_label_map with mode='gaussian' works correctly."""

    @staticmethod
    def _create_synthetic_label_map(h: int = 256, w: int = 256) -> np.ndarray:
        """Create a synthetic label map for testing.

        Maps:
        - Background: 0
        - Skin: 1 (large central region)
        - Hair: 17 (top and sides)
        - Left eye: 4 (top-left)
        - Right eye: 5 (top-right)
        - Lips: 12/13 (bottom center)
        - Eyebrows: 2/3 (above eyes)
        """
        label_map = np.zeros((h, w), dtype=np.uint8)

        # Skin: large central region (not a perfect circle, just a blob)
        y, x = np.ogrid[:h, :w]
        mask = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= (80 ** 2)
        label_map[mask] = 1

        # Hair: top and sides
        hair_mask = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= (110 ** 2)
        hair_mask &= (y < h // 3)
        label_map[hair_mask] = 17

        # Left eye: small circle top-left
        eye_y, eye_x = h // 3, w // 3
        left_eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= (15 ** 2)
        label_map[left_eye_mask] = 4

        # Right eye: small circle top-right
        eye_x = 2 * w // 3
        right_eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= (15 ** 2)
        label_map[right_eye_mask] = 5

        # Left eyebrow
        brow_y, brow_x = h // 3 - 20, w // 3
        left_brow_mask = ((x - brow_x) ** 2 + (y - brow_y) ** 2) <= (12 ** 2)
        label_map[left_brow_mask] = 2

        # Right eyebrow
        brow_x = 2 * w // 3
        right_brow_mask = ((x - brow_x) ** 2 + (y - brow_y) ** 2) <= (12 ** 2)
        label_map[right_brow_mask] = 3

        # Lips: small rectangle at bottom center
        lips_y1, lips_y2 = 2 * h // 3, 2 * h // 3 + 20
        lips_x1, lips_x2 = w // 2 - 15, w // 2 + 15
        label_map[lips_y1:lips_y2, lips_x1:lips_x2] = 12

        return label_map

    def test_gaussian_feather_output_is_float32(self):
        """Gaussian feathering should produce float32 [0, 1] masks."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result = _masks_from_label_map(
            label_map, img_bgr, feather, mode="gaussian", include_cloth=True
        )

        # Check all expected keys
        expected_keys = {
            'skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye',
            'mouth_interior', 'lips', 'neck', 'hair', 'cloth', 'face_oval'
        }
        assert set(result.keys()) == expected_keys

        # Check dtype and range for non-None masks
        for k, mask in result.items():
            if mask is not None:
                assert mask.dtype == np.float32
                assert mask.min() >= 0.0
                assert mask.max() <= 1.0
                assert mask.shape == (256, 256)

    def test_gaussian_interior_saturation(self):
        """Gaussian feathering should produce saturated interior (≈1.0) in mask interior."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result = _masks_from_label_map(
            label_map, img_bgr, feather, mode="gaussian", include_cloth=True
        )

        # Skin mask should have saturated interior
        skin_mask = result['skin']
        # Erode the mask to get well-inside interior
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded = cv2.erode(skin_mask, kernel, iterations=1)
        interior = eroded[eroded > 0.01]  # Get non-background

        if len(interior) > 0:
            # Median should be very close to 1.0 for gaussian feather
            assert np.median(interior) >= 0.95

    def test_exclude_cloth_path(self):
        """When include_cloth=False, cloth should not be in result."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result_no_cloth = _masks_from_label_map(
            label_map, img_bgr, feather, mode="gaussian", include_cloth=False
        )

        # 'cloth' should not be in the dict
        assert 'cloth' not in result_no_cloth

        result_with_cloth = _masks_from_label_map(
            label_map, img_bgr, feather, mode="gaussian", include_cloth=True
        )

        # 'cloth' should be in the dict
        assert 'cloth' in result_with_cloth
        assert result_with_cloth['cloth'].dtype == np.float32

    def test_skin_minus_exclusions(self):
        """Skin mask should have exclusions (eyes, eyebrows, lips) subtracted."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result = _masks_from_label_map(
            label_map, img_bgr, feather, mode="gaussian", include_cloth=True
        )

        skin = result['skin']
        left_eye = result['left_eye']
        right_eye = result['right_eye']
        lips = result['lips']

        # At peak eye/lip positions, skin should be significantly lower than those masks
        # (because they were subtracted)
        # Find a spot with high eye value
        if left_eye.max() > 0.5:
            y, x = np.unravel_index(left_eye.argmax(), left_eye.shape)
            # Skin at this location should be much lower than eye
            assert skin[y, x] < left_eye[y, x] * 0.5  # At least 50% reduction


class TestMasksFromLabelMapGuided:
    """Test guided filter feathering mode."""

    @staticmethod
    def _create_synthetic_label_map(h: int = 256, w: int = 256) -> np.ndarray:
        """Create a synthetic label map with some edge structure."""
        label_map = np.zeros((h, w), dtype=np.uint8)

        # Central skin region
        y, x = np.ogrid[:h, :w]
        skin_mask = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= (80 ** 2)
        label_map[skin_mask] = 1

        # Hair at top with some strands (simulate with alternating regions)
        hair_y = h // 3
        for i in range(0, w, 10):
            label_map[max(0, hair_y - 30):hair_y, i:min(w, i + 5)] = 17

        return label_map

    def test_guided_filter_output_dtype(self):
        """Guided filtering should produce float32 [0, 1] masks."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result = _masks_from_label_map(
            label_map, img_bgr, feather, mode="guided", include_cloth=True
        )

        # Check dtype and range
        for k, mask in result.items():
            if mask is not None:
                assert mask.dtype == np.float32
                assert mask.min() >= 0.0
                assert mask.max() <= 1.0

    def test_guided_filter_interior_saturation(self):
        """Guided filter should also maintain saturated interior."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feather = 5

        result = _masks_from_label_map(
            label_map, img_bgr, feather, mode="guided", include_cloth=True
        )

        skin_mask = result['skin']
        # Define the interior from the original binary class map. Eroding the
        # already-soft output and selecting every value above 0.01 also samples
        # the feather halo, so its median varies with the OpenCV guided-filter
        # implementation rather than measuring interior saturation.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hard_skin = (label_map == 1).astype(np.uint8)
        interior_mask = cv2.erode(hard_skin, kernel, iterations=1).astype(bool)
        interior = skin_mask[interior_mask]

        if len(interior) > 0:
            # Median should be close to 1.0
            assert np.median(interior) >= 0.99

    def test_guided_filter_unsupported_mode_raises(self):
        """Invalid feather mode should raise ValueError."""
        label_map = self._create_synthetic_label_map()
        img_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unknown feather mode"):
            _masks_from_label_map(
                label_map, img_bgr, 5, mode="invalid_mode", include_cloth=True
            )


class TestGoldenByteIdentity:
    """Gaussian mode must preserve the old post-processing output.

    The eye and eyebrow keys intentionally changed to camera-viewer
    handedness in the current parser, so those four semantic keys are
    compared against their opposite old keys.  All unrelated masks remain
    byte-identical; the composite skin mask is allowed only floating-point
    subtraction-order noise from that key swap.
    """

    @staticmethod
    def _reference_old_block(full_label_map, feather, include_cloth):
        from retouch.utils import feather_mask

        m = {}
        m['skin'] = (full_label_map == 1).astype(np.float32)
        m['left_eyebrow'] = (full_label_map == 2).astype(np.float32)
        m['right_eyebrow'] = (full_label_map == 3).astype(np.float32)
        m['left_eye'] = (full_label_map == 4).astype(np.float32)
        m['right_eye'] = (full_label_map == 5).astype(np.float32)
        m['mouth_interior'] = (full_label_map == 11).astype(np.float32)
        m['lips'] = ((full_label_map == 12) | (full_label_map == 13)).astype(np.float32)
        m['neck'] = (full_label_map == 14).astype(np.float32)
        m['hair'] = (full_label_map == 17).astype(np.float32)
        if include_cloth:
            m['cloth'] = (full_label_map == 16).astype(np.float32)
        m['face_oval'] = (
            (full_label_map == 1) | (full_label_map == 2) | (full_label_map == 3) |
            (full_label_map == 4) | (full_label_map == 5) | (full_label_map == 10) |
            (full_label_map == 11) | (full_label_map == 12) | (full_label_map == 13)
        ).astype(np.float32)
        for k in ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye',
                  'lips', 'face_oval', 'neck', 'hair', 'cloth']:
            if k in m:
                r = feather // 2 if k in ['left_eye', 'right_eye', 'lips',
                                          'left_eyebrow', 'right_eyebrow', 'cloth'] else feather
                m[k] = feather_mask(m[k], radius=r)
        for excl_k in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye',
                       'lips', 'mouth_interior']:
            if excl_k in m and m[excl_k] is not None:
                m['skin'] = np.clip(m['skin'] - m[excl_k], 0.0, 1.0)
        return m

    @pytest.mark.parametrize("feather", [3, 7, 8, 15])
    @pytest.mark.parametrize("include_cloth", [True, False])
    def test_gaussian_mode_byte_identical(self, feather, include_cloth):
        rng = np.random.default_rng(7)
        label = rng.integers(0, 19, (240, 200)).astype(np.uint8)
        img = rng.integers(0, 256, (240, 200, 3)).astype(np.uint8)

        old = self._reference_old_block(label, feather, include_cloth)
        new = _masks_from_label_map(label, img, feather, mode="gaussian",
                                    include_cloth=include_cloth)

        assert set(old.keys()) == set(new.keys())
        handedness_swap = {
            'left_eyebrow': 'right_eyebrow',
            'right_eyebrow': 'left_eyebrow',
            'left_eye': 'right_eye',
            'right_eye': 'left_eye',
        }
        for k in old:
            expected = old[handedness_swap.get(k, k)]
            if k == 'skin':
                np.testing.assert_allclose(
                    new[k], expected, rtol=0.0, atol=2e-7,
                    err_msg=f"mask '{k}' drifted from pre-refactor output",
                )
            else:
                assert np.array_equal(
                    expected, new[k],
                ), f"mask '{k}' drifted from pre-refactor output"

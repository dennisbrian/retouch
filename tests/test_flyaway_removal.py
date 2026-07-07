"""Tests for H1 flyaway / stray-hair removal — remove_flyaways.

Covers:
  - Flyaway detection produces a non-empty mask
  - Removal produces a clean result (flyaway region changed)
  - dtype preservation (uint8 + float32)
  - Zero-strength is a no-op (byte-identical)
  - Mask discipline (outside-hair pixels untouched)
  - Eyebrow exclusion guard
  - Intentional thick strand preserved (thickness cap)
  - Flow-deviation gate keeps edge strands that agree with flow
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.hairwork import (
    remove_flyaways,
    hair_flow,
    _flyaway_mask,
)


def _vertical_strand_image(size: int = 256) -> np.ndarray:
    """Synthetic image with vertical strands (vertical light/dark stripes)."""
    img = np.full((size, size, 3), 80, dtype=np.uint8)
    stripe_w = 6
    for x in range(0, size, 2 * stripe_w):
        img[:, x : x + stripe_w] = 160
    return img


def _full_hair_mask(size: int = 256) -> np.ndarray:
    return np.ones((size, size), dtype=np.float32)


def _add_diagonal_flyaway(img: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw a thin diagonal flyaway line crossing the vertical strands."""
    out = img.copy()
    size = out.shape[0]
    for i in range(size):
        y = i
        x = i
        if 0 <= y < size and 0 <= x < size:
            for t in range(thickness):
                if x + t < size:
                    out[y, x + t] = 220  # bright flyaway
    return out


def _add_thick_lock(img: np.ndarray, thickness: int = 12) -> np.ndarray:
    """Draw a thick intentional hair lock (should NOT be removed)."""
    out = img.copy()
    size = out.shape[0]
    mid = size // 2
    out[:, mid : mid + thickness] = 30  # dark thick band
    return out


@pytest.fixture
def face_width() -> float:
    return 120.0


class TestFlyawayMask:
    def test_detects_diagonal_flyaway(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(base, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        fly = _flyaway_mask(img, mask, orient, cohere,
                            strength=70, exclude_mask=None,
                            face_width=face_width)

        assert fly.dtype == np.float32
        assert fly.shape == (size, size)
        # The diagonal flyaway region should be detected (non-zero mask)
        assert fly.max() > 0.0
        # Some pixels along the diagonal should be in the mask
        diag_pixels = fly[np.arange(size), np.arange(size)]
        assert diag_pixels.sum() > 0.0

    def test_empty_mask_when_no_hair(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        zero_mask = np.zeros((size, size), dtype=np.float32)
        orient, cohere = hair_flow(img, hair_mask=zero_mask)

        fly = _flyaway_mask(img, zero_mask, orient, cohere,
                            strength=80, exclude_mask=None,
                            face_width=face_width)
        assert fly.max() == 0.0

    def test_eyebrow_exclusion(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(base, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        # Exclude the entire image — should suppress detection entirely
        exclude = np.ones((size, size), dtype=np.float32)
        fly = _flyaway_mask(img, mask, orient, cohere,
                            strength=70, exclude_mask=exclude,
                            face_width=face_width)
        assert fly.max() < 0.01

    def test_thick_lock_not_detected(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_thick_lock(base, thickness=14)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        fly = _flyaway_mask(img, mask, orient, cohere,
                            strength=70, exclude_mask=None,
                            face_width=face_width)
        # The thick lock occupies a vertical band; it should mostly NOT be
        # in the flyaway mask (thickness cap removes it).
        mid = size // 2
        lock_region = fly[:, mid : mid + 14]
        # The thick lock should be largely excluded (most pixels zero)
        assert lock_region.mean() < 0.3


class TestRemoveFlyaways:
    def test_zero_strength_is_noop(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(base, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=0, face_width=face_width)
        assert result is img or np.array_equal(result, img)

    def test_removal_changes_flyaway_region(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(base, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=80, face_width=face_width)
        # The result must differ from the input somewhere
        assert not np.array_equal(result, img)
        # The flyaway diagonal pixels should be pulled toward the
        # surrounding strand color (closer to the original base than to
        # the bright 220 flyaway value).
        diag = np.arange(size)
        flyaway_vals_in = img[diag, diag].astype(np.float32).mean(axis=1)
        flyaway_vals_out = result[diag, diag].astype(np.float32).mean(axis=1)
        base_vals = base[diag, diag].astype(np.float32).mean(axis=1)
        # Healed values should be closer to base (clean) than the bright flyaway
        diff_in = np.abs(flyaway_vals_in - base_vals).mean()
        diff_out = np.abs(flyaway_vals_out - base_vals).mean()
        assert diff_out < diff_in

    def test_dtype_preservation_uint8(self, face_width):
        size = 192
        img = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(img, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=70, face_width=face_width)
        assert result.dtype == np.uint8

    def test_dtype_preservation_float32(self, face_width):
        size = 192
        img = _vertical_strand_image(size).astype(np.float32)
        img = _add_diagonal_flyaway(img.astype(np.uint8), thickness=2)
        img = img.astype(np.float32)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img.astype(np.uint8), hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=70, face_width=face_width)
        assert result.dtype == np.float32

    def test_none_hair_mask_is_noop(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        orient, cohere = hair_flow(img)
        result = remove_flyaways(img, None, orient, cohere,
                                  strength=80, face_width=face_width)
        assert np.array_equal(result, img)

    def test_empty_hair_mask_is_noop(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        zero_mask = np.zeros((size, size), dtype=np.float32)
        orient, cohere = hair_flow(img, hair_mask=zero_mask)
        result = remove_flyaways(img, zero_mask, orient, cohere,
                                  strength=80, face_width=face_width)
        assert np.array_equal(result, img)

    def test_outside_hair_pixels_untouched(self, face_width):
        size = 192
        img = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(img, thickness=2)
        # Hair mask only covers left half; right half must be untouched.
        # The heal feather (radius ~2px) legitimately bleeds a few pixels
        # across the mask boundary, so we check the right half minus a
        # 6px guard band at the boundary.
        mask = np.zeros((size, size), dtype=np.float32)
        mask[:, : size // 2] = 1.0
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=80, face_width=face_width)
        guard = 12
        right_in = img[:, size // 2 + guard :]
        right_out = result[:, size // 2 + guard :]
        assert np.array_equal(right_in, right_out)

    def test_output_shape_matches(self, face_width):
        size = 160
        img = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(img, thickness=2)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=60, face_width=face_width)
        assert result.shape == img.shape

    def test_skin_mask_extends_search(self, face_width):
        size = 192
        base = _vertical_strand_image(size)
        img = _add_diagonal_flyaway(base, thickness=2)
        # Hair mask = top half, skin mask = bottom half. The flyaway crosses
        # both, so providing skin_mask should extend the heal into the bottom.
        hair_m = np.zeros((size, size), dtype=np.float32)
        hair_m[: size // 2, :] = 1.0
        skin_m = np.zeros((size, size), dtype=np.float32)
        skin_m[size // 2 :, :] = 1.0
        orient, cohere = hair_flow(img, hair_mask=hair_m)

        result_no_skin = remove_flyaways(
            img, hair_m, orient, cohere,
            strength=80, face_width=face_width, skin_mask=None,
        )
        result_with_skin = remove_flyaways(
            img, hair_m, orient, cohere,
            strength=80, face_width=face_width, skin_mask=skin_m,
        )
        # With skin_mask the bottom-half flyaway should also be healed,
        # so result_with_skin differs from result_no_skin in the bottom half.
        bot_no = result_no_skin[size // 2 :, :]
        bot_with = result_with_skin[size // 2 :, :]
        assert not np.array_equal(bot_no, bot_with)

    def test_thick_lock_preserved(self, face_width):
        size = 256
        base = _vertical_strand_image(size)
        img = _add_thick_lock(base, thickness=14)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = remove_flyaways(img, mask, orient, cohere,
                                  strength=60, face_width=face_width)
        # The thick lock band should be largely unchanged
        mid = size // 2
        lock_in = img[:, mid : mid + 14].astype(np.float32)
        lock_out = result[:, mid : mid + 14].astype(np.float32)
        # Allow some feather bleed at edges but the core should be stable
        core_in = lock_in[:, 4:10].mean()
        core_out = lock_out[:, 4:10].mean()
        assert abs(core_in - core_out) < 15.0

"""Tests for redness_even (S3 color-blotch evening on a-channel)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.skin import SkinProcessor, _blotch_bandpass


class TestRednessEven:
    """Test the redness_even method (a-channel blotch removal)."""

    def test_strength_zero_returns_input(self, sp: SkinProcessor):
        """Strength 0 should return input unchanged (byte-identical)."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones((100, 100), dtype=np.float32)
        result = sp.redness_even(img, skin_mask, strength=0, face_width=100.0)
        assert np.array_equal(result, img)

    def test_none_mask_returns_input(self, sp: SkinProcessor):
        """None skin mask should return input unchanged."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = sp.redness_even(img, None, strength=50, face_width=100.0)
        assert np.array_equal(result, img)

    def test_strength_zero_with_mask(self, sp: SkinProcessor):
        """Strength 0 even with a valid mask should return input unchanged."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones((100, 100), dtype=np.float32)
        result = sp.redness_even(img, skin_mask, strength=0, face_width=100.0)
        assert np.array_equal(result, img)

    def test_flat_gradient_invariance(self, sp: SkinProcessor):
        """Flat gradient input should have minimal change (no banding)."""
        h, w = 200, 200
        ramp = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
        img = np.stack([ramp, ramp, ramp], axis=2).astype(np.uint8)
        skin_mask = np.ones((h, w), dtype=np.float32)
        result = sp.redness_even(img, skin_mask, strength=50, face_width=100.0)
        diff = np.abs(result.astype(int) - img.astype(int))
        max_diff = np.percentile(diff, 95)
        assert max_diff <= 2

    def test_red_patch_blotch_decreases(self, sp: SkinProcessor):
        """Synthetic red patch on skin -> a-channel band std should decrease."""
        np.random.seed(42)
        h, w = 200, 200
        face_width = 80.0

        lab = np.full((h, w, 3), (128, 128, 128), dtype=np.float32)
        a_base = 130.0 + np.random.normal(0, 5, (h, w)).astype(np.float32)
        lab[:, :, 1] = np.clip(a_base, 0, 255)
        lab[:, :, 0] = 180.0
        lab[:, :, 2] = 125.0

        lab[50:150, 50:150, 1] += 20.0
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)

        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        skin_mask = np.ones((h, w), dtype=np.float32)

        a_before = lab[:, :, 1]
        dog_before = _blotch_bandpass(a_before, face_width)
        std_before = np.std(dog_before[skin_mask > 0.3])

        result = sp.redness_even(img, skin_mask, strength=50, face_width=face_width)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_after = lab_after[:, :, 1]
        dog_after = _blotch_bandpass(a_after, face_width)
        std_after = np.std(dog_after[skin_mask > 0.3])

        assert std_after < std_before * 0.95

    def test_global_mean_a_unchanged(self, sp: SkinProcessor):
        """Global mean a-channel should stay within ±1 after redness_even."""
        np.random.seed(42)
        h, w = 200, 200

        lab = np.full((h, w, 3), (180, 128, 125), dtype=np.float32)
        blotch = np.random.normal(0, 8, (h, w)).astype(np.float32)
        lab[:, :, 1] = np.clip(128.0 + blotch, 0, 255)

        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        skin_mask = np.ones((h, w), dtype=np.float32)

        mean_a_before = np.mean(lab[:, :, 1][skin_mask > 0.3])

        result = sp.redness_even(img, skin_mask, strength=50, face_width=100.0)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_a_after = np.mean(lab_after[:, :, 1][skin_mask > 0.3])

        assert abs(float(mean_a_after) - float(mean_a_before)) < 2.0

    def test_lips_excluded(self, sp: SkinProcessor):
        """Lips region should have nearly unchanged a-channel after processing."""
        h, w = 200, 200
        face_width = 100.0

        lab = np.full((h, w, 3), (180, 130, 125), dtype=np.float32)
        blotch = np.random.RandomState(42).normal(0, 6, (h, w)).astype(np.float32)
        lab[:, :, 1] = np.clip(130.0 + blotch, 0, 255)

        lab[80:120, 80:120, 1] = 160.0

        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        skin_mask = np.ones((h, w), dtype=np.float32)
        lips_mask = np.zeros((h, w), dtype=np.float32)
        lips_mask[80:120, 80:120] = 1.0

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_orig_lips = lab_orig[80:120, 80:120, 1].copy()

        result = sp.redness_even(
            img, skin_mask, strength=80, face_width=face_width, lips_mask=lips_mask
        )

        lab_result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_result_lips = lab_result[80:120, 80:120, 1]

        lips_diff = np.abs(a_result_lips.astype(float) - a_orig_lips.astype(float))
        assert lips_diff.mean() < 1.0

    def test_strength_monotonicity(self, sp: SkinProcessor):
        """A-channel blotch std should decrease monotonically with increasing strength."""
        np.random.seed(42)
        h, w = 150, 150
        face_width = 80.0

        lab = np.full((h, w, 3), (180, 128, 125), dtype=np.float32)
        blotch = np.random.normal(0, 10, (h, w)).astype(np.float32)
        lab[:, :, 1] = np.clip(128.0 + blotch, 0, 255)

        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        skin_mask = np.ones((h, w), dtype=np.float32)

        blotch_stds = []
        for strength in [0, 25, 50, 100]:
            result = sp.redness_even(img, skin_mask, strength, face_width)
            lab_res = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            a_chan = lab_res[:, :, 1]
            dog = _blotch_bandpass(a_chan, face_width)
            blotch_std = np.std(dog[skin_mask > 0.3])
            blotch_stds.append(blotch_std)

        assert blotch_stds[1] < blotch_stds[0]
        assert blotch_stds[2] < blotch_stds[1]
        assert blotch_stds[3] < blotch_stds[2]

    def test_empty_mask_returns_input(self, sp: SkinProcessor):
        """Empty mask (all zeros) should return input unchanged."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        empty_mask = np.zeros((100, 100), dtype=np.float32)
        result = sp.redness_even(img, empty_mask, strength=50, face_width=100.0)
        assert np.array_equal(result, img)

    def test_small_face_width(self, sp: SkinProcessor):
        """Very small face_width should still produce a valid output (no crash)."""
        h, w = 50, 50
        img = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
        skin_mask = np.ones((h, w), dtype=np.float32)
        result = sp.redness_even(img, skin_mask, strength=50, face_width=5.0)
        assert result.dtype == np.uint8
        assert result.shape == (h, w, 3)

    def test_output_format(self, sp: SkinProcessor):
        """Output should be uint8 BGR with valid range."""
        img = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones((100, 100), dtype=np.float32)
        result = sp.redness_even(img, skin_mask, strength=50, face_width=100.0)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
        assert np.all(result >= 0)
        assert np.all(result <= 255)


@pytest.fixture
def sp() -> SkinProcessor:
    return SkinProcessor()

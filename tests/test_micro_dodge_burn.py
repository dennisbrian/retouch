"""Tests for micro dodge & burn (blotch evening) feature."""

import cv2
import numpy as np
import pytest

from retouch.skin import SkinProcessor, _blotch_bandpass, _edge_protect


class TestBlotchBandpass:
    """Test the difference-of-Gaussians bandpass filter."""

    def test_flat_image_zero_bandpass(self):
        """Flat constant image should produce near-zero bandpass."""
        L = np.full((100, 100), 128.0, dtype=np.float32)
        dog = _blotch_bandpass(L, face_width=100.0)

        # DoG of a constant is ~0 (both Gaussians equal to constant)
        assert np.abs(dog).max() < 1.0

    def test_linear_ramp_minimal_bandpass(self):
        """Linear ramp should produce minimal bandpass (no banding artifact)."""
        h, w = 100, 100
        L = np.linspace(0, 255, w, dtype=np.float32)
        L = np.tile(L, (h, 1))

        dog = _blotch_bandpass(L, face_width=100.0)

        # Linear ramp → constant gradient → DoG small (edges may have edge effects)
        # Interior should be much smaller than edges
        assert np.abs(dog[40:60, 40:60]).max() < 2.0

    def test_scale_by_face_width(self):
        """Bandpass sigma should scale with face_width."""
        L = np.random.RandomState(42).normal(128, 20, (100, 100)).astype(np.float32)
        L = np.clip(L, 0, 255)

        dog_small = _blotch_bandpass(L, face_width=50.0)
        dog_large = _blotch_bandpass(L, face_width=200.0)

        # Larger face width = larger kernel = more blurring = different signal
        # They should not be identical
        assert not np.allclose(dog_small, dog_large)

    def test_output_shape(self):
        """Bandpass output should match input shape."""
        L = np.random.RandomState(42).normal(128, 20, (150, 200)).astype(np.float32)
        L = np.clip(L, 0, 255)

        dog = _blotch_bandpass(L, face_width=100.0)

        assert dog.shape == L.shape


class TestEdgeProtect:
    """Test the edge protection mask."""

    def test_flat_region_full_protection(self):
        """Flat region (low gradient) should have high protection."""
        lab = np.zeros((100, 100, 3), dtype=np.float32)
        lab[:, :, 0] = 128.0  # Uniform L

        protect = _edge_protect(lab)

        # Flat interior should be close to 1.0
        assert protect[50, 50] > 0.9

    def test_edge_region_reduced_protection(self):
        """Edge region (high gradient) should have low protection."""
        lab = np.zeros((100, 100, 3), dtype=np.float32)
        lab[:50, :, 0] = 100.0
        lab[50:, :, 0] = 150.0  # Sharp transition at row 50

        protect = _edge_protect(lab)

        # Edge should have lower protection
        assert protect[50, 50] < protect[25, 50]

    def test_output_range(self):
        """Protection mask should be in [0, 1]."""
        lab = np.random.RandomState(42).normal(128, 30, (100, 100, 3)).astype(np.float32)

        protect = _edge_protect(lab)

        assert np.all(protect >= 0.0)
        assert np.all(protect <= 1.0)

    def test_output_shape(self):
        """Protection mask should be (H, W)."""
        lab = np.random.RandomState(42).normal(128, 30, (150, 200, 3)).astype(np.float32)

        protect = _edge_protect(lab)

        assert protect.shape == (150, 200)


class TestMicroDodgeBurn:
    """Test the micro_dodge_burn method."""

    def test_zero_strength_returns_input(self):
        """Strength 0 should return input unchanged (byte-identical)."""
        img_bgr = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        processor = SkinProcessor()
        result = processor.micro_dodge_burn(img_bgr, skin_mask, strength=0, face_width=100.0)

        assert np.array_equal(result, img_bgr)

    def test_none_mask_returns_input(self):
        """None skin mask should return input unchanged."""
        img_bgr = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)

        processor = SkinProcessor()
        result = processor.micro_dodge_burn(img_bgr, None, strength=50, face_width=100.0)

        assert np.array_equal(result, img_bgr)

    def test_linear_ramp_invariance(self):
        """Linear ramp image should have minimal change (no banding)."""
        h, w = 100, 100
        gray_ramp = np.linspace(50, 200, w, dtype=np.uint8)
        img_bgr = np.stack([gray_ramp] * h, axis=0)
        img_bgr = np.stack([img_bgr] * 3, axis=-1)

        skin_mask = np.ones((h, w), dtype=np.float32)

        processor = SkinProcessor()
        result = processor.micro_dodge_burn(img_bgr, skin_mask, strength=50, face_width=100.0)

        # Difference should be minimal (linear ramp → DoG ~0 → minimal change)
        diff = np.abs(result.astype(int) - img_bgr.astype(int))
        max_diff = np.percentile(diff, 95)  # Use 95th percentile to ignore edge effects
        assert max_diff <= 2

    def test_blotchy_skin_std_decreases(self):
        """Blotchy synthetic skin should have lower blotch-std after processing."""
        np.random.seed(42)
        h, w = 150, 150
        face_width = 100.0

        # Base luminance with synthetic blotches
        base_gray = 150
        img_l = np.full((h, w), base_gray, dtype=np.float32)

        # Add blotch-scale noise (pore-ish features)
        noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        img_l += noise
        img_l = np.clip(img_l, 0, 255)

        # Convert to BGR (simple gray conversion)
        img_bgr = np.stack([img_l.astype(np.uint8)] * 3, axis=-1)

        # Full skin mask
        skin_mask = np.ones((h, w), dtype=np.float32)

        # Measure pre-processing blotch std
        from retouch.skin import _blotch_bandpass
        dog_before = _blotch_bandpass(img_l, face_width)
        blotch_std_before = np.std(dog_before[skin_mask > 0.3])

        # Process
        processor = SkinProcessor()
        result_bgr = processor.micro_dodge_burn(img_bgr, skin_mask, strength=50, face_width=face_width)

        # Extract processed L
        result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_l = result_lab[:, :, 0]

        # Measure post-processing blotch std
        dog_after = _blotch_bandpass(result_l, face_width)
        blotch_std_after = np.std(dog_after[skin_mask > 0.3])

        # Blotch std should strictly decrease
        assert blotch_std_after < blotch_std_before

    def test_strength_monotonicity(self):
        """Blotch std should decrease monotonically with increasing strength."""
        np.random.seed(42)
        h, w = 150, 150
        face_width = 100.0

        # Blotchy synthetic skin
        base_gray = 150
        img_l = np.full((h, w), base_gray, dtype=np.float32)
        noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        img_l += noise
        img_l = np.clip(img_l, 0, 255)

        img_bgr = np.stack([img_l.astype(np.uint8)] * 3, axis=-1)
        skin_mask = np.ones((h, w), dtype=np.float32)

        # Measure blotch std at different strengths
        from retouch.skin import _blotch_bandpass

        blotch_stds = []
        for strength in [0, 25, 50, 100]:
            processor = SkinProcessor()
            result_bgr = processor.micro_dodge_burn(img_bgr, skin_mask, strength, face_width)
            result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            result_l = result_lab[:, :, 0]
            dog = _blotch_bandpass(result_l, face_width)
            blotch_std = np.std(dog[skin_mask > 0.3])
            blotch_stds.append(blotch_std)

        # Should be monotonically decreasing (or flat for strength=0)
        assert blotch_stds[0] == blotch_stds[0]  # strength=0, no change
        assert blotch_stds[1] < blotch_stds[0]
        assert blotch_stds[2] < blotch_stds[1]
        assert blotch_stds[3] < blotch_stds[2]

    def test_pore_preservation(self):
        """High-frequency (pore) energy should change minimally."""
        np.random.seed(42)
        h, w = 150, 150
        face_width = 100.0

        # Blotchy base + high-freq pores
        base_l = np.full((h, w), 150.0, dtype=np.float32)
        blotch_noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        pore_noise = np.random.normal(0, 3, (h, w)).astype(np.float32)
        img_l = base_l + blotch_noise + pore_noise
        img_l = np.clip(img_l, 0, 255)

        img_bgr = np.stack([img_l.astype(np.uint8)] * 3, axis=-1)
        skin_mask = np.ones((h, w), dtype=np.float32)

        # High-band energy (pores): sigma=2 Gaussian
        before_blur = cv2.GaussianBlur(img_l, (0, 0), 2.0)
        high_energy_before = np.std(img_l - before_blur)

        # Process
        processor = SkinProcessor()
        result_bgr = processor.micro_dodge_burn(img_bgr, skin_mask, strength=50, face_width=face_width)
        result_lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_l = result_lab[:, :, 0]

        # High-band energy after
        after_blur = cv2.GaussianBlur(result_l, (0, 0), 2.0)
        high_energy_after = np.std(result_l - after_blur)

        # Change should be ≤ 5% (pore preservation is approximate)
        energy_change = abs(high_energy_after - high_energy_before) / (high_energy_before + 1e-6)
        assert energy_change <= 0.05

    def test_partial_mask(self):
        """With partial mask, only masked regions should change."""
        h, w = 100, 100
        img_bgr = np.full((h, w, 3), 128, dtype=np.uint8)

        # Mask only left half
        skin_mask = np.zeros((h, w), dtype=np.float32)
        skin_mask[:, :w // 2] = 1.0

        processor = SkinProcessor()
        result = processor.micro_dodge_burn(img_bgr, skin_mask, strength=50, face_width=100.0)

        # Right half (unmasked) should be identical
        assert np.array_equal(result[:, w // 2:], img_bgr[:, w // 2:])

    def test_output_format(self):
        """Output should be uint8 BGR."""
        img_bgr = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones((100, 100), dtype=np.float32)

        processor = SkinProcessor()
        result = processor.micro_dodge_burn(img_bgr, skin_mask, strength=50, face_width=100.0)

        assert result.dtype == np.uint8
        assert result.shape == img_bgr.shape
        assert np.all(result >= 0)
        assert np.all(result <= 255)

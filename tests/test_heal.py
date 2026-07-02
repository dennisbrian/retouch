"""Tests for retouch/heal.py — heal_region, mask_to_b64, b64_to_mask."""
import base64

import cv2
import numpy as np
import pytest

from retouch.heal import heal_region, mask_to_b64, b64_to_mask, _auto_radius


class TestHealRegion:
    def test_flat_region_no_seam(self):
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[50:78, 50:78] = 255
        result = heal_region(img, mask, method="telea")
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        roi = result[55:73, 55:73]
        assert np.allclose(roi, 128, atol=5)

    def test_gradient_region_no_seam(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(128):
            img[:, i] = int(i * 255 / 127)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:88, 40:88] = 255
        result = heal_region(img, mask, method="telea")
        assert result.shape == img.shape
        center_vals = result[64, 55:75, 0]
        diffs = np.abs(np.diff(center_vals.astype(int)))
        assert np.all(diffs < 20)

    def test_textured_region(self):
        rng = np.random.RandomState(42)
        img = rng.randint(100, 160, (128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[50:78, 50:78] = 255
        img[50:78, 50:78] = [20, 20, 20]
        result = heal_region(img, mask, method="telea")
        healed_roi = result[55:73, 55:73]
        assert healed_roi.mean() > 50

    def test_ns_method(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        result = heal_region(img, mask, method="ns")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_empty_mask_returns_original(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = heal_region(img, mask, method="telea")
        assert np.array_equal(result, img)

    def test_float_mask_accepted(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = heal_region(img, mask, method="telea")
        assert result.shape == img.shape

    def test_explicit_radius(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255
        result = heal_region(img, mask, method="telea", radius=10)
        assert result.shape == img.shape


class TestAutoRadius:
    def test_small_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:34, 30:34] = 255
        r = _auto_radius(mask)
        assert r >= 2

    def test_large_mask(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:200, 50:200] = 255
        r = _auto_radius(mask)
        assert r > 5

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        r = _auto_radius(mask)
        assert r == 3


class TestMaskRoundTrip:
    def test_uint8_roundtrip(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        b64 = mask_to_b64(mask)
        assert isinstance(b64, str)
        assert len(b64) > 0
        decoded = b64_to_mask(b64)
        assert decoded.shape == mask.shape
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, mask)

    def test_float32_roundtrip(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        b64 = mask_to_b64(mask)
        decoded = b64_to_mask(b64)
        assert decoded.shape == mask.shape
        assert decoded.dtype == np.uint8
        assert np.all(decoded[10:50, 10:50] == 255)
        assert np.all(decoded[0:5, 0:5] == 0)

    def test_roundtrip_with_resize(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        b64 = mask_to_b64(mask)
        decoded = b64_to_mask(b64, target_shape=(128, 128))
        assert decoded.shape == (128, 128)

    def test_valid_base64(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:25, 5:25] = 255
        b64 = mask_to_b64(mask)
        decoded_bytes = base64.b64decode(b64)
        assert len(decoded_bytes) > 0


class TestHealBeforePipeline:
    def test_heal_persists_through_processing(self):
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        img[50:78, 50:78] = [20, 20, 20]
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[50:78, 50:78] = 255
        healed = heal_region(img, mask, method="telea")
        center = healed[64, 64]
        assert center[0] > 80
        gray_healed = cv2.cvtColor(healed, cv2.COLOR_BGR2GRAY)
        assert gray_healed[64, 64] > 80

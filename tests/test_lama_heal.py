"""Tests for retouch/spot_heal.py — LamaHealer (F4.b large-region removal).

These tests exercise the fallback path (LaMa ONNX model not bundled) plus
dtype/no-op/shape invariants that hold for both the LaMa and fallback
paths. The real LaMa model is not downloaded in CI; the fallback path
(multi-pass Telea) is what these tests verify end-to-end.
"""
import cv2
import numpy as np
import pytest

from retouch.spot_heal import LamaHealer, SpotHealer, _coerce_mask_uint8


@pytest.fixture
def flat_img() -> np.ndarray:
    return np.full((96, 96, 3), 128, dtype=np.uint8)


@pytest.fixture
def object_img() -> np.ndarray:
    """128x128 flat field with a large green block in the center."""
    img = np.full((128, 128, 3), 150, dtype=np.uint8)
    img[40:88, 40:88] = [10, 200, 10]
    return img


@pytest.fixture
def large_object_mask() -> np.ndarray:
    m = np.zeros((128, 128), dtype=np.uint8)
    m[40:88, 40:88] = 255
    return m


@pytest.fixture
def healer() -> LamaHealer:
    # Reset any cached session so each test starts clean (fallback path).
    LamaHealer.reset_session()
    return LamaHealer()


class TestHealLargeFallback:
    """Fallback path (LaMa model absent) → multi-pass Telea."""

    def test_fallback_removes_large_object(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer
    ) -> None:
        result = healer.heal_large(object_img, large_object_mask)
        assert result.dtype == np.uint8
        assert result.shape == object_img.shape
        center = result[64, 64]
        # The big green block should be replaced with something near the
        # surrounding 150 field, not the original [10, 200, 10].
        assert abs(int(center[0]) - 150) < 60
        assert abs(int(center[1]) - 150) < 80

    def test_fallback_used_when_model_absent(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer, monkeypatch
    ) -> None:
        """When model_exists returns False, SpotHealer fallback is exercised."""
        import retouch.spot_heal as sh

        called = {"fallback": False}
        original = SpotHealer.heal_object_removal

        def spy(self, img, mask, method="telea"):
            called["fallback"] = True
            return original(self, img, mask, method=method)

        monkeypatch.setattr(SpotHealer, "heal_object_removal", spy)
        monkeypatch.setattr(sh, "model_exists", lambda name: False)
        LamaHealer.reset_session()
        h = LamaHealer()
        h.heal_large(object_img, large_object_mask)
        assert called["fallback"] is True

    def test_fallback_produces_reasonable_result(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer
    ) -> None:
        """Result should be visually close to the surrounding background."""
        result = healer.heal_large(object_img, large_object_mask)
        # Sample a corner region (unaffected) and the healed center.
        corner = result[5:15, 5:15].astype(np.float32)
        center = result[60:68, 60:68].astype(np.float32)
        # Mean difference between healed center and original field should be modest.
        assert abs(center.mean() - 150.0) < 40.0
        # Corner must be untouched.
        assert abs(corner.mean() - 150.0) < 1.0


class TestDtypePreservation:
    def test_uint8_preserved(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer
    ) -> None:
        result = healer.heal_large(object_img, large_object_mask)
        assert result.dtype == np.uint8

    def test_float32_preserved(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer
    ) -> None:
        img_f = object_img.astype(np.float32)
        result = healer.heal_large(img_f, large_object_mask)
        assert result.dtype == np.float32
        assert result.max() <= 255.0

    def test_float32_mask_accepted(
        self, object_img: np.ndarray, large_object_mask: np.ndarray, healer: LamaHealer
    ) -> None:
        m_f = (large_object_mask.astype(np.float32) / 255.0)
        result = healer.heal_large(object_img, m_f)
        assert result.shape == object_img.shape
        assert result.dtype == np.uint8

    def test_invalid_dtype_raises(self, healer: LamaHealer) -> None:
        img_i16 = np.full((64, 64, 3), 100, dtype=np.int16)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        with pytest.raises(ValueError):
            healer.heal_large(img_i16, mask)

    def test_invalid_shape_raises(self, healer: LamaHealer) -> None:
        img = np.full((64, 64), 100, dtype=np.uint8)  # 2D, not (H,W,3)
        mask = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError):
            healer.heal_large(img, mask)


class TestZeroMaskNoOp:
    def test_zero_mask_uint8_noop(
        self, flat_img: np.ndarray, healer: LamaHealer
    ) -> None:
        zero = np.zeros((96, 96), dtype=np.uint8)
        result = healer.heal_large(flat_img, zero)
        assert np.array_equal(result, flat_img)

    def test_zero_mask_float32_noop(
        self, flat_img: np.ndarray, healer: LamaHealer
    ) -> None:
        img_f = flat_img.astype(np.float32)
        zero = np.zeros((96, 96), dtype=np.float32)
        result = healer.heal_large(img_f, zero)
        assert np.array_equal(result, img_f)

    def test_zero_mask_float32_returns_copy(self, flat_img: np.ndarray, healer: LamaHealer) -> None:
        img_f = flat_img.astype(np.float32)
        zero = np.zeros((96, 96), dtype=np.float32)
        result = healer.heal_large(img_f, zero)
        assert result is not img_f


class TestConstructorValidation:
    def test_tile_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            LamaHealer(tile_size=128)

    def test_tile_overlap_too_large_raises(self) -> None:
        with pytest.raises(ValueError):
            LamaHealer(tile_size=512, tile_overlap=300)

    def test_invalid_fallback_method_raises(self) -> None:
        with pytest.raises(ValueError):
            LamaHealer(fallback_method="bogus")

    def test_ns_fallback_method_ok(self) -> None:
        h = LamaHealer(fallback_method="ns")
        assert h.fallback_method == "ns"


class TestTilingGrid:
    def test_small_image_no_tiles(self, healer: LamaHealer) -> None:
        # 96x96 with default 1024 tile → empty grid (single-pass path).
        assert healer._tile_grid(96, 96) == []

    def test_large_image_produces_tiles(self) -> None:
        h = LamaHealer(tile_size=512, tile_overlap=64)
        tiles = h._tile_grid(1200, 1200)
        assert len(tiles) >= 4
        # First tile starts at origin.
        assert tiles[0][0] == 0 and tiles[0][1] == 0
        # Tiles never exceed image bounds.
        for (y0, x0, th, tw) in tiles:
            assert y0 + th <= 1200
            assert x0 + tw <= 1200


class TestSessionCaching:
    def test_session_none_when_model_absent(self, monkeypatch) -> None:
        import retouch.spot_heal as sh

        monkeypatch.setattr(sh, "model_exists", lambda name: False)
        LamaHealer.reset_session()
        assert LamaHealer._get_session() is None

    def test_reset_session_clears_cache(self, monkeypatch) -> None:
        import retouch.spot_heal as sh

        monkeypatch.setattr(sh, "model_exists", lambda name: False)
        LamaHealer.reset_session()
        _ = LamaHealer._get_session()
        LamaHealer.reset_session()
        assert LamaHealer._session is None

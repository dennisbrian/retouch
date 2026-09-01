"""Tests for retouch/lips.py — LipEnhancer."""
import numpy as np
import pytest
from retouch.lips import LipEnhancer


@pytest.fixture
def enhancer():
    return LipEnhancer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def lip_mask():
    m = np.zeros((64, 64), dtype=np.float32)
    m[24:40, 20:44] = 1.0
    return m


@pytest.fixture
def lip_img():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    img[24:40, 20:44] = [100, 60, 180]
    return img


class TestEnhance:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=0)
        assert np.all(result == img)

    def test_none_mask(self, enhancer, img):
        result = enhancer.enhance(img, None, strength=50)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer.enhance(img, mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image_default(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50)
        assert not np.allclose(result, lip_img)

    def test_changes_image_matte(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, finish="matte")
        assert not np.allclose(result, lip_img)

    def test_changes_image_velvet(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, finish="velvet")
        assert not np.allclose(result, lip_img)

    def test_with_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint="pink")
        assert not np.allclose(result, lip_img)

    def test_with_cosplay_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint="cosplay")
        assert not np.allclose(result, lip_img)

    def test_tint_as_tuple(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint=(100, 150, 200))
        assert not np.allclose(result, lip_img)


class TestAddLipGloss:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0)
        assert np.all(result == img)

    def test_no_lip_pixels_returns_original(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._add_lip_gloss(img, mask, 0.5)
        assert np.all(result == img)

    def test_with_lip_highlights(self, enhancer):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[24:40, 20:44] = [180, 130, 200]
        img[28:32, 28:32] = [240, 230, 250]
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[24:40, 20:44] = 1.0
        result = enhancer._add_lip_gloss(img, mask, 0.8)
        assert not np.allclose(result, img)

    def test_output_type(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0.3)
        assert result.dtype == np.uint8


class TestLipGlossToneInvariance:
    """_add_lip_gloss's specular floor must be relative to the lip's own
    diffuse baseline, not an absolute 0-255 constant -- an absolute floor
    systematically under-detects (or misses) real specular gloss highlights
    on darker-toned lips, since a specular reflection is additive on top of
    a lower diffuse base (dichromatic model). Regression coverage for the
    2026-07-15 fix (see docs/plans/PLAN_P4_MAKEUP_UNMIX.md Sec 16 -- same
    bug class as the specular.py and makeup_unmix.py findings there).
    """

    # Fitzpatrick-representative lip BGR swatches with headroom below 255.
    TONES = {
        "I": (150, 120, 180),
        "III": (90, 70, 140),
        "V": (55, 40, 95),
        "VI": (30, 20, 55),
    }

    @staticmethod
    def _spot_mask(h: int = 60, w: int = 60, radius: int = 8):
        yy, xx = np.ogrid[:h, :w]
        return ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= radius ** 2

    @pytest.mark.parametrize("tone", list(TONES.keys()))
    def test_gloss_highlight_detected_even_on_dark_lips(self, tone):
        """A +25 L-channel gloss highlight must be detected at every tone.
        Pre-fix, the absolute L > 130.0 floor read exactly 0 pixels changed
        for Fitzpatrick V-VI at this boost -- i.e. lip gloss finish was
        silently inert on darker-toned lips."""
        import cv2

        bgr = self.TONES[tone]
        h = w = 60
        spot = self._spot_mask(h, w)
        mask = np.ones((h, w), dtype=np.float32)
        lip = np.zeros((h, w, 3), np.float32) + np.array(bgr, np.float32)
        rng = np.random.RandomState(1)
        lip += rng.normal(0, 3, (h, w, 3))
        lip_lab = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lip_lab[spot, 0] = np.clip(lip_lab[spot, 0] + 25.0, 0, 255)
        lip_bgr = cv2.cvtColor(lip_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        enhancer = LipEnhancer()
        out = enhancer._add_lip_gloss(lip_bgr, mask, 1.0)
        delta = np.abs(out.astype(np.float32) - lip_bgr.astype(np.float32))
        assert float(delta[spot].max()) > 20.0, (tone, float(delta[spot].max()))

    @pytest.mark.parametrize("tone", list(TONES.keys()))
    def test_flat_lips_no_false_positive_all_tones(self, tone):
        """No-op invariant: flat lips with only natural sensor noise (no real
        highlight) must not trigger the gloss boost at any tone, across
        multiple noise seeds."""
        bgr = self.TONES[tone]
        h = w = 60
        mask = np.ones((h, w), dtype=np.float32)
        enhancer = LipEnhancer()
        for seed in range(20):
            lip = np.zeros((h, w, 3), np.float32) + np.array(bgr, np.float32)
            rng = np.random.RandomState(seed)
            lip = np.clip(lip + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
            out = enhancer._add_lip_gloss(lip, mask, 1.0)
            delta = float(np.abs(out.astype(np.float32) - lip.astype(np.float32)).max())
            assert delta < 1.0, (tone, seed, delta)

    def test_detection_no_longer_collapses_on_darkest_tone(self):
        """Direct regression for the worst pre-fix case: Fitzpatrick VI used
        to read exactly 0 delta for a +25 gloss highlight (total
        non-detection). Confirms the relative floor closes that gap."""
        import cv2

        bgr = self.TONES["VI"]
        h = w = 60
        spot = self._spot_mask(h, w)
        mask = np.ones((h, w), dtype=np.float32)
        lip = np.zeros((h, w, 3), np.float32) + np.array(bgr, np.float32)
        rng = np.random.RandomState(1)
        lip += rng.normal(0, 3, (h, w, 3))
        lip_lab = cv2.cvtColor(np.clip(lip, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        lip_lab[spot, 0] = np.clip(lip_lab[spot, 0] + 25.0, 0, 255)
        lip_bgr = cv2.cvtColor(lip_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        enhancer = LipEnhancer()
        out = enhancer._add_lip_gloss(lip_bgr, mask, 1.0)
        delta = np.abs(out.astype(np.float32) - lip_bgr.astype(np.float32))
        assert float(delta[spot].max()) > 20.0


class TestExtractTexture:
    def test_output_shape(self, enhancer, lip_img, lip_mask):
        result = enhancer._extract_texture(lip_img, lip_mask, face_width=500)
        assert result.shape == (64, 64)

    def test_zero_output_with_flat_image(self, enhancer, img, lip_mask):
        result = enhancer._extract_texture(img, lip_mask, face_width=500)
        assert np.all(result == 0)

    def test_nonzero_with_varied_image(self, enhancer):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[:, :, 0] = np.tile(np.arange(0, 64, dtype=np.uint8), (64, 1))
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._extract_texture(img, mask, face_width=500)
        assert np.abs(result).sum() > 0


class TestReapplyTexture:
    def test_opacity_zero(self, enhancer, lip_img, lip_mask):
        texture = np.random.rand(64, 64).astype(np.float32) * 30
        result = enhancer._reapply_texture(lip_img, texture, lip_mask, face_width=500, opacity=0)
        np.testing.assert_array_equal(result, lip_img)

    def test_with_texture(self, enhancer, lip_img, lip_mask):
        texture = np.random.rand(64, 64).astype(np.float32) * 20
        result = enhancer._reapply_texture(lip_img, texture, lip_mask, face_width=500, opacity=0.5)
        assert result.shape == (64, 64, 3)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_exact_outside_support(self, enhancer, lip_mask, dtype):
        rng = np.random.default_rng(20260901)
        source_u8 = rng.integers(20, 236, size=(64, 64, 3), dtype=np.uint8)
        source = source_u8 if dtype == np.uint8 else source_u8.astype(np.float32)
        texture = rng.normal(0.0, 8.0, size=(64, 64)).astype(np.float32)

        result = enhancer._reapply_texture(
            source, texture, lip_mask, face_width=500, opacity=0.75
        )

        outside = lip_mask == 0
        np.testing.assert_array_equal(result[outside], source[outside])
        assert np.any(result[~outside] != source[~outside])
        assert result.dtype == source.dtype
        assert np.isfinite(result).all()


class TestLipGlossContainment:
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_blur_cannot_cross_permitted_lip_support(self, enhancer, dtype):
        rng = np.random.default_rng(20260902)
        source_u8 = rng.integers(35, 105, size=(64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[22:42, 18:46] = 1.0

        # Put a compact highlight against the permitted support boundary so
        # the gloss Gaussian blur would otherwise spill onto the canvas.
        source_u8[27:35, 18:23] = 235
        source = source_u8 if dtype == np.uint8 else source_u8.astype(np.float32)

        result = enhancer._add_lip_gloss(source, mask, strength=1.0)

        outside = mask == 0
        np.testing.assert_array_equal(result[outside], source[outside])
        assert np.any(result[~outside] != source[~outside])
        assert result.dtype == source.dtype
        assert np.isfinite(result).all()


class TestSmooth:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._smooth(img, lip_mask, 0)
        assert np.all(result == img)

    def test_changes_image(self, enhancer):
        x = np.tile(np.linspace(100, 200, 64, dtype=np.uint8), (64, 1))
        img = np.stack([x, x, x], axis=-1)
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._smooth(img, mask, 0.5)
        assert not np.allclose(result, img)


class TestApplyTint:
    def test_unknown_tint_falls_back_to_nude(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "nonexistent", 0.25)
        assert result.shape == (64, 64, 3)

    def test_changes_with_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "pink", 0.25)
        assert not np.allclose(result, lip_img)

    def test_tuple_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, (100, 150, 200), 0.25)
        assert not np.allclose(result, lip_img)

    def test_zero_strength(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "pink", 0)
        assert np.all(result == lip_img)

    def test_all_tints(self, enhancer, lip_img, lip_mask):
        for tint in ["nude", "pink", "rose", "coral", "berry", "red", "cosplay"]:
            result = enhancer._apply_tint(lip_img, lip_mask, tint, 0.25)
            assert not np.allclose(result, lip_img), f"Tint {tint} did not change image"

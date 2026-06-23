"""Tests for retouch/highlight.py — film-like highlight roll-off."""

import numpy as np

from retouch.highlight import (
    apply_highlight_rolloff,
    recover_highlights,
    soft_clip_highlights,
    tone_map_filmic,
    tone_map_reinhard,
)


class TestSoftClipHighlights:
    def test_no_value_exceeds_threshold(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        out = soft_clip_highlights(img, threshold=230.0, rolloff_start=200.0)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        assert int(out.max()) <= 230

    def test_linear_region_unchanged(self):
        values = list(range(0, 200, 20))
        img = np.array(values, dtype=np.uint8).reshape(-1, 1, 1)
        img = np.repeat(img, 3, axis=-1)
        out = soft_clip_highlights(img, threshold=230.0, rolloff_start=200.0)
        for i in range(len(values)):
            np.testing.assert_array_equal(out[i], img[i])

    def test_highlights_compressed_below_threshold(self):
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = soft_clip_highlights(img, threshold=230.0, rolloff_start=200.0)
        assert out[0, 0, 0] < 255
        assert int(out[0, 0, 0]) <= 230

    def test_smooth_at_rolloff_boundary(self):
        img = np.array([[[199, 200, 201]]], dtype=np.uint8)
        out = soft_clip_highlights(img, threshold=230.0, rolloff_start=200.0)
        assert out[0, 0, 0] == 199
        assert out[0, 0, 1] == 200
        assert int(out[0, 0, 2]) <= 201
        assert abs(int(out[0, 0, 2]) - 201) <= 1

    def test_returns_uint8_bgr(self):
        img = np.full((8, 8, 3), 200, dtype=np.uint8)
        out = soft_clip_highlights(img)
        assert out.dtype == np.uint8
        assert out.shape == (8, 8, 3)

    def test_full_white_input_below_threshold(self):
        # At x=255 with default params, output should be ~225, well below 230
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = soft_clip_highlights(img)
        assert 220 <= out[0, 0, 0] <= 230


class TestApplyHighlightRolloff:
    def test_strength_zero_is_passthrough(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        out = apply_highlight_rolloff(img, strength=0.0)
        np.testing.assert_array_equal(out, img)

    def test_strength_one_full_applies(self):
        img = np.full((4, 4, 3), 250, dtype=np.uint8)
        out = apply_highlight_rolloff(img, strength=1.0)
        expected = soft_clip_highlights(img)
        np.testing.assert_array_equal(out, expected)

    def test_negative_strength_clamps_to_zero(self):
        img = np.full((1, 1, 3), 100, dtype=np.uint8)
        out = apply_highlight_rolloff(img, strength=-0.5)
        np.testing.assert_array_equal(out, img)

    def test_excess_strength_clamps_to_one(self):
        img = np.full((1, 1, 3), 100, dtype=np.uint8)
        out = apply_highlight_rolloff(img, strength=1.5)
        expected = soft_clip_highlights(img)
        np.testing.assert_array_equal(out, expected)

    def test_intermediate_strength_partial(self):
        img = np.full((1, 1, 3), 250, dtype=np.uint8)
        out = apply_highlight_rolloff(img, strength=0.5)
        soft = soft_clip_highlights(img)
        # Should be strictly between input and soft-clipped value
        assert soft[0, 0, 0] <= out[0, 0, 0] <= img[0, 0, 0]


class TestRecoverHighlights:
    def test_below_threshold_gets_brighter(self):
        img = np.array([[[100, 100, 100], [200, 200, 200]]], dtype=np.uint8)
        out = recover_highlights(img, threshold=240.0, amount=0.3)
        assert out[0, 0, 0] > 100
        assert out[0, 1, 0] > 200
        assert out[0, 0, 0] == 142
        assert out[0, 1, 0] == 212

    def test_above_threshold_unchanged(self):
        img = np.array([[[250, 250, 250]]], dtype=np.uint8)
        out = recover_highlights(img, threshold=240.0, amount=0.3)
        np.testing.assert_array_equal(out, img)

    def test_zero_amount_is_passthrough(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        out = recover_highlights(img, threshold=240.0, amount=0.0)
        np.testing.assert_array_equal(out, img)

    def test_at_threshold_unchanged(self):
        img = np.array([[[240, 240, 240]]], dtype=np.uint8)
        out = recover_highlights(img, threshold=240.0, amount=0.3)
        np.testing.assert_array_equal(out, img)


class TestToneMapReinhard:
    def test_output_in_range(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        out = tone_map_reinhard(img)
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_black_stays_black(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = tone_map_reinhard(img)
        np.testing.assert_array_equal(out, img)

    def test_white_is_compressed(self):
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = tone_map_reinhard(img)
        # 255 -> 1.0 / (1 + 1.0) = 0.5 -> 127 (truncation in uint8 cast)
        assert out[0, 0, 0] == 127

    def test_monotonic_increasing(self):
        values = np.arange(0, 256, 16, dtype=np.uint8)
        img = values.reshape(-1, 1, 1).repeat(3, axis=-1)
        out = tone_map_reinhard(img)
        flat = out[:, 0, 0].astype(int)
        assert np.all(np.diff(flat) >= 0)

    def test_exposure_scales_up(self):
        img = np.full((1, 1, 3), 128, dtype=np.uint8)
        out_low = tone_map_reinhard(img, exposure=1.0)
        out_high = tone_map_reinhard(img, exposure=2.0)
        assert out_high[0, 0, 0] > out_low[0, 0, 0]


class TestToneMapFilmic:
    def test_output_in_range(self):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        out = tone_map_filmic(img)
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_black_stays_black(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = tone_map_filmic(img)
        np.testing.assert_array_equal(out, img)

    def test_monotonic_tonal(self):
        values = np.arange(0, 256, dtype=np.uint8)
        img = values.reshape(-1, 1, 1).repeat(3, axis=-1)
        out = tone_map_filmic(img)
        flat = out[:, 0, 0].astype(int)
        assert np.all(np.diff(flat) >= 0)

    def test_uses_dynamic_range(self):
        # Filmic should produce a curve that reaches into the upper range
        values = np.arange(0, 256, dtype=np.uint8)
        img = values.reshape(-1, 1, 1).repeat(3, axis=-1)
        out = tone_map_filmic(img)
        assert int(out.max()) > 200

    def test_has_soft_toe(self):
        # Shadow lift: input 0.25 (64) should map higher than linear 64
        # under the Hable curve with white-point normalization
        values = np.arange(0, 256, dtype=np.uint8)
        img = values.reshape(-1, 1, 1).repeat(3, axis=-1)
        out = tone_map_filmic(img)
        # Pick a dark-mid value (index 32 ~ input 32)
        # Filmic toe raises the value relative to identity
        assert out[32, 0, 0] >= 32

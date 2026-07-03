"""Tests for exposure-locked frequency smoothing (smooth_exposure_lock).

Covers both the ``build_context`` wiring (recipe/override → ctx field) and the
pure compensation math in ``_apply_exposure_lock``.
"""

import numpy as np
import pytest

from retouch.engine import build_context, ProcessingContext
from retouch.perf_optimizations import _apply_exposure_lock


class TestBuildContextExposureLock:
    """Test smooth_exposure_lock extraction from recipes and overrides."""

    def test_recipe_with_exposure_lock(self):
        """Recipe dict with {"skin": {"exposure_lock": 0.7}} → ctx.smooth_exposure_lock == 0.7."""
        rec = {
            "skin": {"exposure_lock": 0.7, "smooth": 0.3},
            "frequency": {"smooth": 0.3},
        }
        ctx = build_context("test_recipe", rec, {})
        assert ctx.smooth_exposure_lock == 0.7

    def test_recipe_without_exposure_lock(self):
        """Recipe dict without skin.exposure_lock → ctx.smooth_exposure_lock == 0.0."""
        rec = {"skin": {"smooth": 0.3}, "frequency": {"smooth": 0.3}}
        ctx = build_context("test_recipe", rec, {})
        assert ctx.smooth_exposure_lock == 0.0

    def test_empty_recipe_no_exposure_lock(self):
        """Empty recipe → ctx.smooth_exposure_lock == 0.0."""
        ctx = build_context("test_recipe", {}, {})
        assert ctx.smooth_exposure_lock == 0.0

    def test_caller_override_wins_over_recipe(self):
        """Explicit caller override wins over recipe value."""
        rec = {"skin": {"exposure_lock": 0.7}}
        overrides = {"smooth_exposure_lock": 0.3}
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.smooth_exposure_lock == 0.3

    def test_caller_none_override_uses_recipe(self):
        """Caller override absent → falls back to recipe value."""
        rec = {"skin": {"exposure_lock": 0.7}}
        ctx = build_context("test_recipe", rec, {})
        assert ctx.smooth_exposure_lock == 0.7

    def test_default_is_zero(self):
        """ProcessingContext default field value is 0.0."""
        assert ProcessingContext.smooth_exposure_lock == 0.0


class TestApplyExposureLock:
    """Test the pure compensation math in _apply_exposure_lock."""

    @staticmethod
    def _luma(img):
        return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    def _make_scene(self):
        """Skin-toned pre-smoothing canvas w/ bright specular noise + full mask."""
        h, w = 64, 64
        rng = np.random.default_rng(0)
        base = np.empty((h, w, 3), dtype=np.float32)
        base[:, :, 0] = 150.0  # B
        base[:, :, 1] = 170.0  # G
        base[:, :, 2] = 200.0  # R (skin-toned, warm)
        # Bright micro-speculars sprinkled in.
        pre = base.copy()
        spec = rng.random((h, w)) > 0.85
        pre[spec] += 55.0
        pre = np.clip(pre, 0, 255)
        skin_n = np.ones((h, w), dtype=np.float32)
        return pre, skin_n

    def test_lock_zero_is_identity(self):
        """lock=0 leaves the canvas exactly unchanged."""
        pre, skin_n = self._make_scene()
        smoothed = pre * 0.9  # darker post-smoothing
        out = _apply_exposure_lock(smoothed, pre, skin_n, 0.0)
        np.testing.assert_array_equal(out, smoothed)

    def test_lock_one_recovers_mean(self):
        """Darkened canvas + lock=1.0 recovers pre-smoothing mean luma within 0.5."""
        pre, skin_n = self._make_scene()
        smoothed = pre * 0.9  # ~10% darker — simulates specular damping
        out = _apply_exposure_lock(smoothed, pre, skin_n, 1.0)
        mean_pre = float(self._luma(pre).mean())
        mean_out = float(self._luma(out).mean())
        assert abs(mean_out - mean_pre) < 0.5

    def test_brightened_canvas_untouched(self):
        """A canvas that got brighter than pre is left unchanged (no darkening)."""
        pre, skin_n = self._make_scene()
        brighter = np.clip(pre * 1.05, 0, 255)
        out = _apply_exposure_lock(brighter, pre, skin_n, 1.0)
        np.testing.assert_array_equal(out, brighter)

    def test_none_mask_is_identity(self):
        """skin_n=None short-circuits to identity."""
        pre, _ = self._make_scene()
        smoothed = pre * 0.9
        out = _apply_exposure_lock(smoothed, pre, None, 1.0)
        np.testing.assert_array_equal(out, smoothed)

    def test_output_clipped(self):
        """Output stays within [0, 255]."""
        pre, skin_n = self._make_scene()
        smoothed = pre * 0.5
        out = _apply_exposure_lock(smoothed, pre, skin_n, 1.0)
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_partial_lock_undercompensates(self):
        """lock=0.5 recovers less than lock=1.0 (monotone in lock)."""
        pre, skin_n = self._make_scene()
        smoothed = pre * 0.9
        mean_pre = float(self._luma(pre).mean())
        m_half = float(self._luma(_apply_exposure_lock(smoothed, pre, skin_n, 0.5)).mean())
        m_full = float(self._luma(_apply_exposure_lock(smoothed, pre, skin_n, 1.0)).mean())
        m_smoothed = float(self._luma(smoothed).mean())
        assert m_smoothed < m_half < m_full <= mean_pre + 0.5

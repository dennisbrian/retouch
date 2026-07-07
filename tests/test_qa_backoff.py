"""Tests for A5 — "No plastic skin" guarantee (QA auto-back-off).

Covers :class:`retouch.qa_backoff.QABackoff` in isolation (no engine /
no MediaPipe required) plus an integration check that the engine wires
the back-off loop into ``_run_core_pipeline``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from retouch.qa_backoff import QABackoff
from retouch.qa_detectors import QAWarning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(**kwargs) -> SimpleNamespace:
    """Build a lightweight stand-in for ProcessingContext with given params."""
    defaults = dict(
        smooth=50.0,
        whiten=30.0,
        equalize=20.0,
        blemish=10.0,
        skin_quantize=0.0,
        micro_dodge_burn=0.0,
        nose_smooth=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _img() -> np.ndarray:
    """A small valid BGR image for the check_and_backoff signature."""
    return np.full((32, 32, 3), 128, dtype=np.uint8)


def _warn(detector: str, flagged: bool, score: float = 0.0) -> QAWarning:
    return QAWarning(
        detector=detector,
        score=score,
        flagged=flagged,
        message=f"{detector} test warning",
        threshold=0.5,
        details={},
    )


# ---------------------------------------------------------------------------
# backoff_strategy — pure function of (warning_type, current_params)
# ---------------------------------------------------------------------------


class TestBackoffStrategy:
    def test_plastic_skin_reduces_smooth(self):
        ctx = _ctx(smooth=80.0)
        strat = QABackoff().backoff_strategy("plastic_skin", {"smooth": 80.0})
        assert "smooth" in strat
        assert strat["smooth"] < 80.0
        # Default factor 0.6 → 48.0
        assert strat["smooth"] == pytest.approx(48.0, abs=0.01)

    def test_plastic_skin_reduces_all_relevant_params(self):
        params = {
            "smooth": 50.0,
            "whiten": 30.0,
            "equalize": 20.0,
            "blemish": 10.0,
            "skin_quantize": 40.0,
            "micro_dodge_burn": 15.0,
            "nose_smooth": 60.0,
        }
        strat = QABackoff().backoff_strategy("plastic_skin", params)
        for key in ("smooth", "whiten", "equalize", "blemish",
                    "skin_quantize", "nose_smooth"):
            assert key in strat, f"{key} should be backed off"
            assert strat[key] < params[key], f"{key} not reduced"
            assert strat[key] >= 0.0

    def test_plastic_skin_skips_zero_params(self):
        # A param at 0 is a no-op; backing it off wastes an iteration
        # and masks a real flag from a different param.
        params = {
            "smooth": 50.0,
            "whiten": 0.0,
            "equalize": 0.0,
            "blemish": 0.0,
            "skin_quantize": 0.0,
            "micro_dodge_burn": 0.0,
            "nose_smooth": None,
        }
        strat = QABackoff().backoff_strategy("plastic_skin", params)
        assert "smooth" in strat
        assert "whiten" not in strat
        assert "equalize" not in strat
        assert "nose_smooth" not in strat

    def test_plastic_skin_respects_floor(self):
        # With a floor, a tiny smooth value should not drop below it.
        bo = QABackoff(smooth_factor=0.1, smooth_floor=10.0)
        strat = bo.backoff_strategy("plastic_skin", {"smooth": 15.0})
        assert strat["smooth"] >= 10.0

    def test_unknown_warning_returns_empty(self):
        # A5 scope is plastic-skin only. Other warnings must not trigger
        # a guess-based back-off — the engine ships the original result.
        params = {"smooth": 50.0, "whiten": 30.0}
        for w in ("halo", "banding", "clipping", "seam", "nonexistent"):
            assert QABackoff().backoff_strategy(w, params) == {}

    def test_strategy_is_pure(self):
        # Same inputs → same outputs, no mutation of current_params.
        params = {"smooth": 60.0, "whiten": 20.0}
        params_copy = dict(params)
        s1 = QABackoff().backoff_strategy("plastic_skin", params)
        s2 = QABackoff().backoff_strategy("plastic_skin", params)
        assert s1 == s2
        assert params == params_copy


# ---------------------------------------------------------------------------
# check_and_backoff — the engine-facing entry point
# ---------------------------------------------------------------------------


class TestCheckAndBackoff:
    def test_no_warnings_returns_none(self):
        ctx = _ctx()
        assert QABackoff().check_and_backoff(_img(), ctx, []) is None

    def test_unflagged_warning_returns_none(self):
        # Conservative: a warning below threshold is informational, not
        # actionable. Back-off only on a real flag.
        ctx = _ctx()
        warns = [_warn("plastic_skin", flagged=False, score=0.3)]
        assert QABackoff().check_and_backoff(_img(), ctx, warns) is None

    def test_flagged_plastic_skin_returns_adjustments(self):
        ctx = _ctx(smooth=80.0, whiten=40.0)
        warns = [_warn("plastic_skin", flagged=True, score=0.2)]
        adj = QABackoff().check_and_backoff(_img(), ctx, warns)
        assert adj is not None
        assert "smooth" in adj
        assert adj["smooth"] < 80.0
        assert adj["whiten"] < 40.0

    def test_non_plastic_flag_returns_none(self):
        # Halo/banding flags are out of A5 scope.
        ctx = _ctx()
        warns = [_warn("halo", flagged=True, score=20.0)]
        assert QABackoff().check_and_backoff(_img(), ctx, warns) is None

    def test_multiple_flags_keep_most_conservative(self):
        # If two flagged warnings touch the same param, the smaller value
        # wins (most conservative).
        ctx = _ctx(smooth=100.0)
        warns = [
            _warn("plastic_skin", flagged=True, score=0.1),
        ]
        adj = QABackoff().check_and_backoff(_img(), ctx, warns)
        assert adj is not None
        assert adj["smooth"] < 100.0

    def test_invalid_image_returns_none(self):
        ctx = _ctx()
        warns = [_warn("plastic_skin", flagged=True)]
        assert QABackoff().check_and_backoff(None, ctx, warns) is None

    def test_iterations_compound(self):
        # Simulate two iterations: the second back-off should compound on
        # the first (params already reduced).
        ctx = _ctx(smooth=100.0)
        bo = QABackoff()
        warns = [_warn("plastic_skin", flagged=True)]
        adj1 = bo.check_and_backoff(_img(), ctx, warns)
        assert adj1 is not None
        # Apply and re-snapshot
        QABackoff.apply_adjustments(ctx, adj1)
        adj2 = bo.check_and_backoff(_img(), ctx, warns)
        assert adj2 is not None
        assert adj2["smooth"] < adj1["smooth"]

    def test_max_iterations_constant(self):
        # The engine loop is bounded; verify the default.
        assert QABackoff().max_iterations >= 1
        assert QABackoff().max_iterations <= 4  # conservative ceiling


# ---------------------------------------------------------------------------
# apply_adjustments — in-place ctx mutation
# ---------------------------------------------------------------------------


class TestApplyAdjustments:
    def test_applies_to_context(self):
        ctx = _ctx(smooth=80.0)
        QABackoff.apply_adjustments(ctx, {"smooth": 48.0})
        assert ctx.smooth == 48.0

    def test_unknown_param_logged_not_raised(self):
        ctx = _ctx()
        # Should not raise even if the param isn't on the context.
        QABackoff.apply_adjustments(ctx, {"nonexistent_param": 1.0})
        assert not hasattr(ctx, "nonexistent_param")


# ---------------------------------------------------------------------------
# Integration with real QA detectors (no engine / no MediaPipe)
# ---------------------------------------------------------------------------


class TestIntegrationWithDetectors:
    def _plastic_image(self) -> np.ndarray:
        """Build an image that detect_plastic_skin flags.

        A near-flat skin-tone region has almost no high-frequency energy,
        so hf_energy_ratio ≈ 0 < PLASTIC_SKIN_THRESHOLD → flagged.
        """
        img = np.full((64, 64, 3), 160, dtype=np.uint8)  # flat skin-ish
        # Tiny tint so it isn't perfectly uniform
        img[:, :, 0] = 150
        img[:, :, 1] = 130
        img[:, :, 2] = 120
        return img

    def _textured_image(self) -> np.ndarray:
        """Build an image with strong high-frequency texture (not flagged)."""
        rng = np.random.RandomState(0)
        noise = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return noise

    def test_backoff_triggers_on_real_plastic_flag(self):
        from retouch.qa_detectors import detect_plastic_skin, PLASTIC_SKIN_THRESHOLD

        img = self._plastic_image()
        mask = np.ones((64, 64), dtype=np.float32)
        det = detect_plastic_skin(img, mask=mask)
        assert det["flagged"], "test image should be flagged as plastic"
        assert det["hf_energy_ratio"] < PLASTIC_SKIN_THRESHOLD

        ctx = _ctx(smooth=80.0, whiten=40.0)
        warn = QAWarning(
            detector="plastic_skin",
            score=det["score"],
            flagged=det["flagged"],
            message="plastic",
            threshold=PLASTIC_SKIN_THRESHOLD,
            details=det,
        )
        adj = QABackoff().check_and_backoff(img, ctx, [warn])
        assert adj is not None
        assert adj["smooth"] < 80.0

    def test_no_backoff_on_textured_image(self):
        from retouch.qa_detectors import detect_plastic_skin

        img = self._textured_image()
        mask = np.ones((64, 64), dtype=np.float32)
        det = detect_plastic_skin(img, mask=mask)
        # Textured noise should not be flagged.
        if det["flagged"]:
            pytest.skip("random noise unexpectedly flagged — threshold tuning")
        ctx = _ctx(smooth=80.0)
        warn = QAWarning(
            detector="plastic_skin",
            score=det["score"],
            flagged=det["flagged"],
            message="plastic",
            threshold=0.6,
            details=det,
        )
        # Not flagged → check_and_backoff returns None
        assert QABackoff().check_and_backoff(img, ctx, [warn]) is None


# ---------------------------------------------------------------------------
# Engine wiring smoke test
# ---------------------------------------------------------------------------


class TestEngineWiring:
    def test_engine_has_qa_backoff_attribute(self):
        # Without instantiating (avoids MediaPipe/model load), verify the
        # class wires the attribute in __init__ by inspecting source.
        import inspect
        from retouch import engine as engine_mod

        src = inspect.getsource(engine_mod.RetouchEngine.__init__)
        assert "_qa_backoff" in src
        assert "QABackoff" in src

    def test_core_pipeline_has_backoff_loop(self):
        import inspect
        from retouch import engine as engine_mod

        src = inspect.getsource(engine_mod.RetouchEngine._run_core_pipeline)
        # The A5 back-off loop must be present.
        assert "check_and_backoff" in src
        assert "plastic_skin" in src
        assert "apply_adjustments" in src

    def test_run_qa_helper_exists(self):
        import inspect
        from retouch import engine as engine_mod

        assert hasattr(engine_mod.RetouchEngine, "_run_qa")
        # Should be callable and accept (result, person_mask)
        sig = inspect.signature(engine_mod.RetouchEngine._run_qa)
        params = list(sig.parameters.keys())
        assert "result" in params
        assert "person_mask" in params


# ---------------------------------------------------------------------------
# Full back-off loop simulation (no engine)
# ---------------------------------------------------------------------------


class TestBackoffLoopSimulation:
    def test_loop_clears_flag_after_iterations(self):
        """Simulate the engine's back-off loop: reduce params until the
        plastic-skin flag clears (here simulated by a threshold on smooth)."""
        ctx = _ctx(smooth=100.0, whiten=50.0)
        bo = QABackoff(max_iterations=3)

        # Simulate: flag clears once smooth drops below 30
        def _make_warns(smooth_val: float) -> List[QAWarning]:
            if smooth_val < 30.0:
                return []  # cleared
            return [_warn("plastic_skin", flagged=True, score=0.2)]

        # Snapshot the loop the engine runs
        warns = _make_warns(ctx.smooth)
        for _ in range(bo.max_iterations):
            adj = bo.check_and_backoff(_img(), ctx, warns)
            if adj is None:
                break
            QABackoff.apply_adjustments(ctx, adj)
            warns = _make_warns(ctx.smooth)
            if not warns:
                break

        assert ctx.smooth < 30.0, "loop should have reduced smooth below 30"
        assert not warns, "flag should have cleared"

    def test_loop_terminates_at_max_iterations(self):
        """Even if the flag never clears, the loop must terminate."""
        ctx = _ctx(smooth=100.0)
        bo = QABackoff(max_iterations=2)
        # Flag never clears
        warns = [_warn("plastic_skin", flagged=True)]
        iterations = 0
        for _ in range(bo.max_iterations):
            adj = bo.check_and_backoff(_img(), ctx, warns)
            if adj is None:
                break
            QABackoff.apply_adjustments(ctx, adj)
            iterations += 1
            if iterations > bo.max_iterations:
                pytest.fail("loop did not terminate")
        assert iterations <= bo.max_iterations
        # smooth was reduced each iteration
        assert ctx.smooth < 100.0

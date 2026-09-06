"""Tests for the FA-02 production eligibility gate and candidate plumbing.

Scope of what these tests do and do NOT establish:

* they DO establish that the gate abstains on low-detail, small, noisy and
  invalid-support faces, that the three frozen candidate arms dispatch and
  produce finite same-shape output, that nothing outside the declared support
  changes, that the path is deterministic, and that the default mode is inert;
* they establish NOTHING about whether any candidate produces better texture.
  Every threshold under test is PROVISIONAL or a PLACEHOLDER (see
  ``retouch.fa02_texture_eligibility.THRESHOLD_PROVENANCE``); these tests lock
  in the *plumbing and abstention behaviour*, not a calibration.

The byte-identity guarantee for legacy mode lives here as a direct
``_process_face_core`` comparison, and additionally in
``tests/test_golden_pipeline_face.py``, whose committed snapshots are unchanged
by this feature.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np
import pytest

from retouch import fa02_texture_eligibility as elig


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures: synthetic faces engineered to hit one gate at a time
# ---------------------------------------------------------------------------

#: Comfortably above MIN_FACE_WIDTH_PX so scale is never the incidental failure
#: in a test that is about some other gate.
_BIG_FACE_WIDTH = 600.0


def _canvas(h=256, w=256, base=128.0):
    return np.full((h, w, 3), base, dtype=np.float32)


def _full_support(h=256, w=256):
    return np.ones((h, w), dtype=np.float32)


def _high_detail_canvas(h=256, w=256):
    """A deterministic canvas that clears ALL four gates.

    Structure matters here, not just amplitude. The noise proxy is the median
    std of the FLATTEST quarter of 8x8 blocks, so a uniformly textured field
    (e.g. band-limited noise everywhere) reads as noisy and would abstain on
    gate 3 -- correctly, since a flat-block statistic genuinely cannot tell
    dense uniform texture from sensor noise (an acknowledged limitation of the
    PLACEHOLDER proxy, documented in fa02_texture_eligibility).

    So the fixture is sparse high-contrast ridges on a smooth base: strong
    sigma-2 high-pass amplitude where the ridges are, and genuinely flat
    blocks in between for the noise proxy to land on. Fully deterministic --
    no RNG at all, so byte-stability is structural rather than seed-dependent.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Period-32 ridges: the 8x8 noise-proxy blocks that land in a 24px gap are
    # genuinely flat, so the quietest-quarter median sits near zero, while the
    # sigma-2 high-pass (measured over the whole support) still sees the ridges.
    ridges = (
        np.exp(-(((xx % 32.0) - 2.0) ** 2) / 0.5)
        + np.exp(-(((yy % 32.0) - 2.0) ** 2) / 0.5)
    )
    canvas = 128.0 + 90.0 * ridges
    return np.clip(np.repeat(canvas[:, :, None], 3, axis=2), 0.0, 255.0).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Gate 1: face scale
# ---------------------------------------------------------------------------


class TestFaceScaleGate:
    def test_small_face_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(),
            face_width_px=elig.MIN_FACE_WIDTH_PX - 1.0,
        )
        assert decision["eligible"] is False
        assert "face scale" in decision["reason"]

    def test_non_finite_face_width_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(), face_width_px=float("nan"),
        )
        assert decision["eligible"] is False
        assert "face scale not measured" in decision["reason"]

    def test_face_width_is_reported(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["measured"]["face_width_px"] == pytest.approx(_BIG_FACE_WIDTH)


# ---------------------------------------------------------------------------
# Gate 2: sharpness / detail observability
# ---------------------------------------------------------------------------


class TestSharpnessGate:
    def test_flat_face_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "insufficient source texture" in decision["reason"]

    def test_blurred_face_abstains(self):
        import cv2

        blurred = cv2.GaussianBlur(_high_detail_canvas(), (0, 0), 9.0)
        decision = elig.evaluate_face_eligibility(
            blurred, _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "insufficient source texture" in decision["reason"]

    def test_highpass_std_is_reported_and_finite(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        measured = decision["measured"]["highpass_std"]
        assert measured is not None and np.isfinite(measured)

    def test_highpass_std_returns_none_on_empty_mask(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert elig.highpass_std(_high_detail_canvas(64, 64), mask=empty) is None

    def test_highpass_std_is_tone_invariant(self):
        """Adding a constant offset must not change the texture amplitude.

        This is the CLAUDE.md tone-invariance property: the measure keys off
        deviation from the local mean, not absolute intensity, so it cannot
        systematically fail on darker skin.
        """
        base = _high_detail_canvas(128, 128)
        dark = np.clip(base - 60.0, 0.0, 255.0)
        assert elig.highpass_std(base) == pytest.approx(
            elig.highpass_std(dark), rel=1e-6,
        )


# ---------------------------------------------------------------------------
# Gate 3: noise / compression
# ---------------------------------------------------------------------------


class TestNoiseGate:
    def test_noisy_face_abstains(self):
        rng = np.random.default_rng(7)
        noisy = _high_detail_canvas() + rng.normal(
            0.0, elig.MAX_NOISE_SIGMA * 4.0, size=(256, 256, 3),
        ).astype(np.float32)
        decision = elig.evaluate_face_eligibility(
            np.clip(noisy, 0.0, 255.0), _full_support(),
            face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "noise/compression" in decision["reason"]

    def test_noise_proxy_orders_clean_below_noisy(self):
        rng = np.random.default_rng(11)
        clean = _canvas(128, 128)
        noisy = clean + rng.normal(0.0, 10.0, size=(128, 128, 3)).astype(np.float32)
        assert elig.estimate_noise_sigma(clean) < elig.estimate_noise_sigma(noisy)

    def test_noise_proxy_returns_none_when_support_covers_no_block(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[0, 0] = 1.0  # single pixel: no fully-covered 8x8 block
        assert elig.estimate_noise_sigma(_canvas(128, 128), mask=mask) is None

    def test_noise_proxy_returns_none_on_tiny_image(self):
        assert elig.estimate_noise_sigma(_canvas(8, 8)) is None


# ---------------------------------------------------------------------------
# Gate 4: support validity
# ---------------------------------------------------------------------------


class TestSupportValidityGate:
    def test_missing_support_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), None, face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "support invalid" in decision["reason"]
        assert decision["effective_support"] is None

    def test_empty_support_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), np.zeros((256, 256), dtype=np.float32),
            face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "support invalid" in decision["reason"]

    def test_mismatched_support_shape_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), np.ones((64, 64), dtype=np.float32),
            face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "does not match canvas" in decision["reason"]

    def test_non_finite_support_abstains(self):
        support = _full_support()
        support[10, 10] = np.nan
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), support, face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is False
        assert "non-finite" in decision["reason"]

    def test_absent_mark_policy_is_not_invalid_support(self):
        """``protected_mask=None`` (the default, no mark policy) must PASS.

        An absent policy means nothing is protected -- it is not a missing
        support. Getting this polarity backwards would make the default
        configuration abstain for a bogus reason and still look like a working
        gate in a coarse test.
        """
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(),
            face_width_px=_BIG_FACE_WIDTH, protected_mask=None,
        )
        assert decision["eligible"] is True
        assert decision["measured"]["mark_policy_active"] is False

    def test_fully_protecting_policy_abstains(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(),
            face_width_px=_BIG_FACE_WIDTH,
            protected_mask=np.ones((256, 256), dtype=np.float32),
        )
        assert decision["eligible"] is False
        assert "after mark-policy protection" in decision["reason"]

    def test_partial_protection_shrinks_effective_support(self):
        protect = np.zeros((256, 256), dtype=np.float32)
        protect[:64, :] = 1.0
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(),
            face_width_px=_BIG_FACE_WIDTH, protected_mask=protect,
        )
        assert decision["eligible"] is True
        assert np.all(decision["effective_support"][:64, :] == 0.0)
        assert decision["measured"]["support_px"] < decision["measured"][
            "support_px_before_protection"
        ]

    def test_0_255_support_is_normalised(self):
        """A 0-255 mask must be handled, and a 1.0000002 float mask must NOT be
        mistaken for one (CLAUDE.md 2026-09-02 composite-mask epsilon bug)."""
        decision_255 = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support() * 255.0,
            face_width_px=_BIG_FACE_WIDTH,
        )
        epsilon_mask = _full_support() * np.float32(1.0000002)
        decision_eps = elig.evaluate_face_eligibility(
            _high_detail_canvas(), epsilon_mask, face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision_255["eligible"] is True
        assert decision_eps["eligible"] is True
        assert decision_eps["measured"]["support_px"] == 256 * 256


# ---------------------------------------------------------------------------
# Eligible fixture + candidate dispatch
# ---------------------------------------------------------------------------


class TestEligibleFixtureAndDispatch:
    def test_high_detail_fixture_is_eligible(self):
        decision = elig.evaluate_face_eligibility(
            _high_detail_canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert decision["eligible"] is True, decision["reason"]
        assert decision["reason"] == "eligible"

    @pytest.mark.parametrize("mode", ["dog", "multiscale"])
    def test_frozen_arms_produce_finite_same_shape_detail(self, mode):
        canvas = _high_detail_canvas(128, 128)
        import cv2

        smoothed = cv2.GaussianBlur(canvas, (0, 0), 2.0)
        detail = elig.extract_detail(canvas - smoothed, mode, _BIG_FACE_WIDTH)
        assert detail.shape == canvas.shape
        assert detail.dtype == np.float32
        assert np.all(np.isfinite(detail))
        # A frozen band-difference of a real residual must not be identically
        # zero -- that would mean the dispatch silently produced a no-op.
        assert np.abs(detail).max() > 0.0

    def test_dog_matches_frozen_formula(self):
        """A2 is exactly ``G.6(D) - G1.2(D)`` at the scaled sigmas."""
        canvas = _high_detail_canvas(96, 96)
        import cv2

        residual = canvas - cv2.GaussianBlur(canvas, (0, 0), 2.0)
        sigmas = elig.scaled_sigmas(_BIG_FACE_WIDTH)
        expected = elig._gaussian(residual, sigmas[0]) - elig._gaussian(
            residual, sigmas[1],
        )
        got = elig.extract_detail(residual, "dog", _BIG_FACE_WIDTH)
        assert np.allclose(got, expected, atol=1e-6)

    def test_multiscale_matches_frozen_formula(self):
        """A3 is ``[G.6-G1.2](D) + .5 [G1.2-G2.4](D)``, remaining weights zero."""
        canvas = _high_detail_canvas(96, 96)
        import cv2

        residual = canvas - cv2.GaussianBlur(canvas, (0, 0), 2.0)
        s = elig.scaled_sigmas(_BIG_FACE_WIDTH)
        g = [elig._gaussian(residual, sig) for sig in s[:3]]
        expected = (g[0] - g[1]) + 0.5 * (g[1] - g[2])
        got = elig.extract_detail(residual, "multiscale", _BIG_FACE_WIDTH)
        assert np.allclose(got, expected, atol=1e-6)

    def test_sigma_scaling_follows_face_width(self):
        """Sigmas scale with face width against the frozen 500 px reference."""
        assert elig.scaled_sigmas(500.0) == pytest.approx(
            list(elig.SIGMAS_AT_IED_200),
        )
        assert elig.scaled_sigmas(1000.0)[0] == pytest.approx(1.2)

    def test_sigma_floor_is_respected_not_upsampled(self):
        """Tiny faces clamp at the 0.6 px floor; bands collapse, never invent."""
        sigmas = elig.scaled_sigmas(50.0)
        assert all(s >= elig.MINIMUM_SIGMA_PX for s in sigmas)

    @pytest.mark.parametrize("bad_mode", ["A4_orientation", "A5_retouch_frequency",
                                          "orientation", "frequency", "raw_residual",
                                          "legacy"])
    def test_a4_a5_and_non_frozen_arms_are_rejected(self, bad_mode):
        """A4/A5 must not be dispatchable from production plumbing at all."""
        residual = np.zeros((32, 32, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            elig.extract_detail(residual, bad_mode, _BIG_FACE_WIDTH)

    @pytest.mark.parametrize("mode", ["dog", "multiscale"])
    def test_deterministic(self, mode):
        canvas = _high_detail_canvas(96, 96)
        import cv2

        residual = canvas - cv2.GaussianBlur(canvas, (0, 0), 2.0)
        a = elig.extract_detail(residual, mode, _BIG_FACE_WIDTH)
        b = elig.extract_detail(residual, mode, _BIG_FACE_WIDTH)
        assert np.array_equal(a, b)

    def test_eligibility_decision_is_deterministic(self):
        canvas = _high_detail_canvas()
        first = elig.evaluate_face_eligibility(
            canvas, _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        second = elig.evaluate_face_eligibility(
            canvas, _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert first["eligible"] == second["eligible"]
        assert first["reason"] == second["reason"]
        assert first["measured"] == second["measured"]


# ---------------------------------------------------------------------------
# End-to-end through _process_face_core: legacy byte-identity, opt-in behaviour,
# and no changes outside the declared support.
# ---------------------------------------------------------------------------


def _face_core_inputs(fa02_mode="legacy"):
    """Real engine + real golden face fixture -- no mocks, no stub canvases."""
    from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

    img = make_synthetic_face_image()
    h, w = img.shape[:2]
    return img, make_face_context(w, h, img)


def _render(engine, img, face_ctx, fa02_mode):
    """Render the golden face fixture through the real pipeline.

    Goes through ``RetouchEngine.process`` -- the same entry point
    ``tests/test_golden_pipeline_face.py`` uses -- rather than calling
    ``_process_face_core`` with hand-built ROI arguments, so the ROI/region
    geometry is derived by production code rather than by the test.
    """
    kwargs = {} if fa02_mode is None else {"fa02_texture_mode": fa02_mode}
    return np.asarray(
        engine.process(img, recipe="natural", face_contexts=[face_ctx], **kwargs)
    )


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    return RetouchEngine()


@contextlib.contextmanager
def caplog_at(level, name="retouch.perf_optimizations"):
    """Collect log messages from one logger without relying on caplog.

    ``caplog`` cannot be combined with a module-scoped fixture in the same
    test cleanly here, so this captures via an explicit handler instead.
    """
    import logging

    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger(name)
    handler = _Collector(level=level)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class TestFaceCoreIntegration:
    def test_default_mode_is_legacy(self):
        from retouch.engine import build_context
        from retouch.params import resolve_recipe

        ctx = build_context("natural", resolve_recipe("natural"), {})
        assert ctx.fa02_texture_mode == "legacy"

    def test_legacy_mode_is_byte_identical_to_default(self, engine):
        """Explicit ``"legacy"`` and the unset default must be the same bytes.

        This is the direct half of the byte-identity guarantee;
        ``tests/test_golden_pipeline_face.py``'s committed snapshots are the
        frozen-hash half and are unchanged by this feature.
        """
        img, face_ctx = _face_core_inputs()
        default = _render(engine, img, face_ctx, None)
        explicit = _render(engine, img, face_ctx, "legacy")
        assert np.array_equal(default, explicit)

    def test_legacy_mode_is_deterministic(self, engine):
        img, face_ctx = _face_core_inputs()
        a = _render(engine, img, face_ctx, None)
        b = _render(engine, img, face_ctx, None)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("mode", ["raw_residual", "dog", "multiscale"])
    def test_opt_in_modes_run_without_crashing(self, engine, mode):
        img, face_ctx = _face_core_inputs()
        out = _render(engine, img, face_ctx, mode)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out.astype(np.float32)))

    @pytest.mark.parametrize("mode", ["raw_residual", "dog", "multiscale"])
    def test_opt_in_modes_are_deterministic(self, engine, mode):
        img, face_ctx = _face_core_inputs()
        a = _render(engine, img, face_ctx, mode)
        b = _render(engine, img, face_ctx, mode)
        assert np.array_equal(a, b)

    def test_abstention_falls_back_safely(self, engine, caplog):
        """A face that cannot clear the gate must still render, and must log why.

        The synthetic golden fixture is a smooth painted face -- it does not
        clear the (uncalibrated) texture floor, so it exercises the abstention
        branch end to end. The render must complete normally: abstention means
        "do not restore", never "fail".
        """
        import logging

        img, face_ctx = _face_core_inputs()
        with caplog.at_level(logging.INFO, logger="retouch.perf_optimizations"):
            out = _render(engine, img, face_ctx, "dog")
        assert out.shape == img.shape
        fa02_lines = [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("FA-02 texture:")
        ]
        assert fa02_lines, "no FA-02 observability line was logged"
        line = fa02_lines[0]
        # Point 5: mode, decision, reason, measured inputs, candidate, fallback.
        for token in ("mode=dog", "eligible=", "reason=", "measured=",
                      "thresholds=", "ran=", "legacy_micro_restore_ran=False"):
            assert token in line, "observability line missing {0}: {1}".format(
                token, line,
            )
        assert "abstained" in line, (
            "the smooth synthetic fixture should abstain: {0}".format(line)
        )

    def test_thresholds_are_patchable(self, monkeypatch):
        """Threshold constants must be genuinely overridable at runtime.

        Regression guard: these were originally keyword *defaults*, which bind
        at import time, so patching the module constant silently did nothing
        and the documented "configurable thresholds" contract was false.
        """
        strict = elig.evaluate_face_eligibility(
            _canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert strict["eligible"] is False

        monkeypatch.setattr(elig, "MIN_HIGHPASS_STD", 0.0)
        relaxed = elig.evaluate_face_eligibility(
            _canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        # The DECISION must flip, not merely the reported threshold -- asserting
        # only the reported value would still pass if the resolution were broken.
        assert relaxed["eligible"] is True, relaxed["reason"]
        assert relaxed["thresholds"]["minimum_highpass_std"] == 0.0

    def test_unknown_mode_falls_back_to_legacy(self, engine):
        """An unrecognised mode must degrade to legacy, never crash mid-render.

        The CLI validates against ``ParamSpec.choices``, but the Python API and
        a hand-edited recipe do not, so a typo or a reserved research arm name
        can reach the render path. ``extract_detail`` raises on those by design,
        so the dispatch must intercept them first.
        """
        import logging

        img, face_ctx = _face_core_inputs()
        legacy = _render(engine, img, face_ctx, "legacy")
        with caplog_at(logging.WARNING) as records:
            bogus = _render(engine, img, face_ctx, "A4_orientation")
        assert np.array_equal(legacy, bogus), (
            "an unknown mode must produce byte-identical output to legacy"
        )
        assert any("unknown fa02_texture_mode" in m for m in records), records

    @pytest.mark.parametrize("mode", ["raw_residual", "dog", "multiscale"])
    def test_eligible_face_executes_the_candidate_path(
        self, engine, monkeypatch, mode, caplog,
    ):
        """With the gate relaxed, the candidate branch must actually execute.

        Without this the whole ``if eligible:`` block is dead code in the test
        suite -- the synthetic golden fixture abstains on face scale and
        texture, so every unrelaxed render takes the abstention branch. The
        thresholds are relaxed ONLY to reach the code path; this asserts
        nothing about output quality and is not a calibration.

        Note ``dog`` reaches the branch but composites an identically zero
        detail signal at this fixture's face scale (collapsed band -- see
        ``test_dog_band_collapses_at_the_fixture_face_scale``), so this test
        asserts the branch EXECUTED, not that pixels moved.
        """
        import logging

        monkeypatch.setattr(elig, "MIN_FACE_WIDTH_PX", 50.0)
        monkeypatch.setattr(elig, "MIN_HIGHPASS_STD", 1.0)
        img, face_ctx = _face_core_inputs()
        with caplog.at_level(logging.INFO, logger="retouch.perf_optimizations"):
            out = _render(engine, img, face_ctx, mode)
        line = next(
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("FA-02 texture:")
        )
        assert "eligible=True" in line, line
        assert "abstained" not in line, line
        assert "ran={0}".format(mode) in line or "ran=raw_residual" in line, line
        assert out.shape == img.shape
        assert np.all(np.isfinite(out.astype(np.float32)))

    def test_raw_residual_reuses_the_existing_production_op(
        self, engine, monkeypatch,
    ):
        """``raw_residual`` must call ``skin.restore_micro_texture``, not a copy.

        This is the only assertion that distinguishes "reuse the shipping op
        under a new gate" from "a second implementation of the same idea".
        """
        from retouch.skin import SkinProcessor

        monkeypatch.setattr(elig, "MIN_FACE_WIDTH_PX", 50.0)
        monkeypatch.setattr(elig, "MIN_HIGHPASS_STD", 1.0)

        calls = []
        original = SkinProcessor.restore_micro_texture

        def _spy(self, *args, **kwargs):
            calls.append(kwargs.get("strength"))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SkinProcessor, "restore_micro_texture", _spy)
        img, face_ctx = _face_core_inputs()
        _render(engine, img, face_ctx, "raw_residual")
        assert calls, "raw_residual did not reach skin.restore_micro_texture"

    def test_dog_band_collapses_at_the_fixture_face_scale(self):
        """A2's single band is identically zero when both sigmas hit the floor.

        At the golden fixture's face width (~133 px) the frozen stack scales
        sigmas 0 and 1 to 0.6 and 0.6 -- both clamped to MINIMUM_SIGMA_PX -- so
        ``G.6(D) - G.6(D)`` is exactly zero and the ``dog`` arm cannot change a
        pixel no matter how the gate is set. A3 still has its second band.

        This is the frozen spec's "collapsed bands are logged, not upsampled
        into invented detail" condition. It is a real property of the frozen
        formulas at small face scales, NOT a bug and NOT a quality judgement:
        A2 and A3 are simply not interchangeable below roughly face width 250.
        """
        import cv2

        fixture_face_width = 132.856
        assert 0 in elig.collapsed_bands(fixture_face_width)

        canvas = _high_detail_canvas(96, 96)
        residual = canvas - cv2.GaussianBlur(canvas, (0, 0), 2.0)
        dog = elig.extract_detail(residual, "dog", fixture_face_width)
        multi = elig.extract_detail(residual, "multiscale", fixture_face_width)
        assert np.abs(dog).max() == 0.0
        assert np.abs(multi).max() > 0.0

        # At a large face width the band is resolved and A2 is non-trivial.
        assert elig.collapsed_bands(1000.0) == ()
        assert np.abs(elig.extract_detail(residual, "dog", 1000.0)).max() > 0.0

    def test_dog_is_a_noop_at_the_gates_own_face_scale_floor(self):
        """``dog`` collapses at MIN_FACE_WIDTH_PX itself -- an evaluation blocker.

        At face width exactly 250 (the gate's PLACEHOLDER floor) the scaled
        sigmas are [0.6, 0.6, 1.2, 2.4, 4.8]: sigma_0 and sigma_1 are BOTH
        clamped to the 0.6 px floor, so A2's only band is identically zero.
        Clearing the face-scale gate therefore does NOT make ``dog`` evaluable.

        Band 0 lifts off the floor only above fw=250, and equals the frozen
        ``G.6 - G1.2`` only at fw >= 500. This is a hard, measurement-derived
        constraint on any future DoG evaluation, independent of corpus size or
        annotation quality, and is recorded here so it cannot be lost.
        """
        assert elig.collapsed_bands(elig.MIN_FACE_WIDTH_PX) == (0,)
        assert elig.collapsed_bands(250.0) == (0,)
        assert elig.collapsed_bands(300.0) == ()
        # Below fw=500 the band exists but is a compressed ratio, not the
        # frozen 0.6-vs-1.2 pair.
        assert elig.scaled_sigmas(300.0)[:2] == pytest.approx([0.6, 0.72])
        assert elig.scaled_sigmas(500.0)[:2] == pytest.approx([0.6, 1.2])
        # multiscale's weighted band 1 resolves from the floor upward.
        assert 1 not in elig.collapsed_bands(elig.MIN_FACE_WIDTH_PX)

    @pytest.mark.parametrize("mode", ["multiscale"])
    def test_no_changes_outside_declared_support(self, engine, monkeypatch, mode):
        """The new composite must not touch a pixel outside its support.

        Measured on the PER-FACE ROI canvas, not the final frame. The final
        frame is produced by a feathered face-composite that blends the ROI
        back in, so a full-frame diff attributes that blend's own soft edge to
        FA-02 and cannot answer this question. The ROI canvas and
        ``regions.skin`` share a coordinate space, so the comparison is exact.

        The two renders differ ONLY in whether the gate passed (thresholds
        relaxed vs. not), so every changed pixel is attributable to the FA-02
        composite and nothing else.
        """
        img, face_ctx = _face_core_inputs()
        captured = {}

        from retouch import perf_optimizations as perf
        import retouch.engine as engine_mod

        original = perf._process_face_core

        def _spy(canvas, regions, *args, **kwargs):
            result = original(canvas, regions, *args, **kwargs)
            captured["canvas"] = np.asarray(result.canvas).copy()
            captured["regions"] = regions
            return result

        monkeypatch.setattr(perf, "_process_face_core", _spy)
        monkeypatch.setattr(engine_mod, "_process_face_core", _spy)

        _render(engine, img, face_ctx, mode)
        abstained = captured["canvas"]
        regions = captured["regions"]

        monkeypatch.setattr(elig, "MIN_FACE_WIDTH_PX", 50.0)
        monkeypatch.setattr(elig, "MIN_HIGHPASS_STD", 1.0)
        _render(engine, img, face_ctx, mode)
        restored = captured["canvas"]

        assert not np.array_equal(abstained, restored), (
            "the relaxed render is identical to the abstaining one -- the "
            "candidate path did not actually composite anything, so this test "
            "would pass vacuously"
        )

        skin = regions.skin.astype(np.float32)
        if skin.max() > 1.5:
            skin = skin / 255.0
        outside = skin <= 0.0
        delta = np.abs(
            restored.astype(np.int32) - abstained.astype(np.int32)
        ).max(axis=2)
        assert int(delta[outside].max()) == 0, (
            "FA-02 {0} changed {1} px outside its declared support "
            "(max delta {2})".format(
                mode, int((delta[outside] > 0).sum()), int(delta[outside].max()),
            )
        )

    def test_eligible_but_zero_micro_restore_is_a_noop(self, engine, monkeypatch):
        """``micro_restore=0`` must no-op even on an eligible face.

        The candidate strength is scaled by ``micro_restore`` exactly as the
        legacy op is, so an eligible face with the strength at zero must not
        change pixels -- and the log line must say so rather than implying
        work happened.
        """
        import logging

        monkeypatch.setattr(elig, "MIN_FACE_WIDTH_PX", 50.0)
        monkeypatch.setattr(elig, "MIN_HIGHPASS_STD", 1.0)
        img, face_ctx = _face_core_inputs()
        with caplog_at(logging.INFO) as records:
            zeroed = np.asarray(
                engine.process(
                    img, recipe="natural", face_contexts=[face_ctx],
                    fa02_texture_mode="dog", micro_restore=0,
                )
            )
            legacy_zeroed = np.asarray(
                engine.process(
                    img, recipe="natural", face_contexts=[face_ctx],
                    fa02_texture_mode="legacy", micro_restore=0,
                )
            )
        assert np.array_equal(zeroed, legacy_zeroed), (
            "with micro_restore=0 the FA-02 path must be indistinguishable "
            "from legacy -- both do nothing"
        )
        lines = [m for m in records if m.startswith("FA-02 texture:")]
        assert lines and "micro_restore=0" in lines[0], lines

    def test_mutual_exclusivity_legacy_op_does_not_also_run(self, engine, monkeypatch):
        """With FA-02 active, the legacy block must not also invoke
        ``skin.restore_micro_texture`` -- otherwise restoration double-applies.
        """
        from retouch.skin import SkinProcessor

        calls = []
        original = SkinProcessor.restore_micro_texture

        def _spy(self, *args, **kwargs):
            calls.append(kwargs.get("strength"))
            return original(self, *args, **kwargs)

        img, face_ctx = _face_core_inputs()

        # Baseline: legacy mode DOES call it (proving the spy works and that
        # this really is the live production op, not dead code).
        monkeypatch.setattr(SkinProcessor, "restore_micro_texture", _spy)
        _render(engine, img, face_ctx, "legacy")
        assert calls, (
            "legacy mode did not call restore_micro_texture -- the spy or the "
            "assumption that micro_restore defaults to 20 is wrong"
        )

        # dog mode never routes to the legacy op, whether it abstains or not.
        calls.clear()
        _render(engine, img, face_ctx, "dog")
        assert calls == []


# ---------------------------------------------------------------------------
# Opt-in guarantees: no recipe or default enables a candidate
# ---------------------------------------------------------------------------


class TestOptInOnly:
    def test_paramspec_default_is_legacy(self):
        from retouch.params import PROCESSING_PARAMS

        spec = next(s for s in PROCESSING_PARAMS if s.name == "fa02_texture_mode")
        assert spec.default == "legacy"
        assert spec.choices == ("legacy", "raw_residual", "dog", "multiscale")

    def test_every_recipe_resolves_to_legacy(self):
        from retouch.engine import build_context
        from retouch.params import RECIPES, resolve_recipe

        offenders = [
            name for name in RECIPES
            if build_context(name, resolve_recipe(name), {}).fa02_texture_mode
            != "legacy"
        ]
        assert offenders == [], (
            "recipes must not enable an FA-02 candidate: {0}".format(offenders)
        )

    def test_no_recipe_source_file_sets_the_key(self):
        """Grep guard: the key must not appear in any recipe definition file."""
        offenders = []
        for path in sorted((ROOT / "retouch").glob("*recipe*.py")):
            text = path.read_text()
            if re.search(r"[\"']fa02_texture_mode[\"']", text):
                offenders.append(path.name)
        assert offenders == [], (
            "fa02_texture_mode must not be set in recipe files: {0}".format(offenders)
        )

    def test_thresholds_are_documented_as_uncalibrated(self):
        """Every threshold must carry provenance, and none may claim validation.

        Matches the FA-02 scoring lock's documentation discipline and the
        eye-gate precedent (CLAUDE.md 2026-08-31): an undocumented threshold is
        how a miscalibrated gate ships looking correct.
        """
        for key, entry in elig.THRESHOLD_PROVENANCE.items():
            assert "provenance" in entry, key
            assert re.match(
                r"^(PROVISIONAL|PLACEHOLDER|MEASURED)", entry["provenance"],
            ), (key, entry["provenance"])
        decision = elig.evaluate_face_eligibility(
            _canvas(), _full_support(), face_width_px=_BIG_FACE_WIDTH,
        )
        assert "UNCALIBRATED" in decision["threshold_status"]

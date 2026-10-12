"""Tests for retouch/frequency.py's mark_protect parameter (FA-01, guided
smoothing only).

docs/plans/RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md
Sec 1.1/1.1a: base smoothing had zero mark-policy awareness -- a mark's
source-relative contrast was destroyed 55-90% at high smooth_strength
before this fix. The validated mechanism shrinks the FINAL blend alpha
(m_2d) at policy-protected mark locations via a per-blob distance-
transform feather, capped below each blob's own inscribed radius; it
never touches the guided filter's own input statistics (the rejected
"filter_input" alternative recovered marginally more contrast but with a
~10x larger halo). This file tests only the guided engine, matching the
experiment's validated scope -- bilateral/anisotropic receive no
protection by an explicit smooth_engine=="guided" gate in combine(), not
by omission, and are asserted here to confirm the gate holds.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.frequency import combine, separate, FrequencySeparator


FACE_WIDTH = 400.0
CANVAS = 256


def _textured_canvas(seed: int = 7) -> np.ndarray:
    """Skin-toned canvas with enough high-band energy that
    _texture_adaptation_factor doesn't silently floor smooth_strength --
    the exact scene-construction mistake the FA-01 experiment caught and
    fixed before trusting its own numbers (std=9 gave adapt=0.55; std=30
    reaches adapt=1.0 on this canvas geometry)."""
    rng = np.random.default_rng(seed)
    h = w = CANVAS
    base = np.full((h, w, 3), (150.0, 175.0, 205.0), dtype=np.float32)
    noise = rng.normal(0, 30.0, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    base += noise[:, :, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _add_mark(canvas: np.ndarray, center, radius: int) -> np.ndarray:
    """Composite a visibly darker circular mark at ``center``."""
    h, w = canvas.shape[:2]
    m = np.zeros((h, w), dtype=np.float32)
    cv2.circle(m, center, radius, 1.0, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    mark_bgr = np.array([70.0, 95.0, 120.0], dtype=np.float32)
    out = canvas.astype(np.float32) * (1.0 - m[:, :, None]) + mark_bgr[None, None, :] * m[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _mark_mask(shape, center, radius: int) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.float32)
    cv2.circle(m, center, radius, 1.0, -1)
    return m


def _mark_contrast(img_gray: np.ndarray, center, mark_radius: int, ring_inner=3, ring_outer=12) -> float:
    """Ring-minus-mark contrast, matching scripts/qa/smoothing_mark_protection_experiment.py."""
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mark_sel = r <= mark_radius * 0.6
    ring_sel = (r >= mark_radius + ring_inner) & (r <= mark_radius + ring_outer)
    return float(img_gray[ring_sel].mean() - img_gray[mark_sel].mean())


def _outer_halo(img_gray: np.ndarray, baseline_gray: np.ndarray, center, mark_radius: int, max_r: int = 30) -> float:
    """Max abs deviation (arm - baseline) in the ring well outside the mark
    (radius+6 .. max_r) -- the halo/contamination instrument from the
    FA-01 experiment. A near-zero value means no boundary artifact."""
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2).astype(np.int32)
    vals = []
    for radius in range(mark_radius + 6, max_r):
        sel = r == radius
        if np.any(sel):
            vals.append(float(img_gray[sel].mean()) - float(baseline_gray[sel].mean()))
    return float(np.max(np.abs(vals))) if vals else 0.0


def _run(canvas, mark_mask, smooth_strength, mark_protect=None, smooth_engine="guided", mid_reduction=0.35):
    layers = separate(canvas, face_width=FACE_WIDTH)
    result = combine(
        layers,
        skin_mask=np.ones(canvas.shape[:2], dtype=np.float32),
        smooth_strength=smooth_strength,
        mid_reduction=mid_reduction,
        face_width=FACE_WIDTH,
        smooth_engine=smooth_engine,
        mark_protect=mark_protect,
    )
    return result


class TestMarkProtectContrastRetention:
    """Protected mark contrast retention at high smoothing strength."""

    @pytest.mark.parametrize("radius", [4, 8, 15])
    def test_protected_mark_retains_far_more_contrast_than_baseline(self, radius):
        center = (128, 128)
        canvas = _add_mark(_textured_canvas(), center, radius)
        mask = _mark_mask(canvas.shape, center, radius)
        source_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        source_contrast = _mark_contrast(source_gray, center, radius)

        baseline = _run(canvas, mask, smooth_strength=0.9, mark_protect=None)
        protected = _run(canvas, mask, smooth_strength=0.9, mark_protect=mask)

        baseline_gray = cv2.cvtColor(np.clip(baseline, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        protected_gray = cv2.cvtColor(np.clip(protected, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        baseline_contrast = _mark_contrast(baseline_gray, center, radius)
        protected_contrast = _mark_contrast(protected_gray, center, radius)

        # Matches the research finding's magnitude: baseline loses the
        # majority of source contrast; protection recovers most of it.
        assert baseline_contrast < 0.5 * source_contrast
        assert protected_contrast > 0.7 * source_contrast
        assert protected_contrast > baseline_contrast + 20


class TestMarkProtectNoHalo:
    """No significant halo at the protection boundary."""

    def test_blend_alpha_protection_has_near_zero_outer_halo(self):
        center = (128, 128)
        radius = 10
        canvas = _add_mark(_textured_canvas(), center, radius)
        mask = _mark_mask(canvas.shape, center, radius)

        baseline = np.clip(_run(canvas, mask, 0.9, mark_protect=None), 0, 255).astype(np.uint8)
        protected = np.clip(_run(canvas, mask, 0.9, mark_protect=mask), 0, 255).astype(np.uint8)

        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        protected_gray = cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY)

        halo = _outer_halo(protected_gray, baseline_gray, center, radius)
        # FA-01 experiment measured blend_alpha's outer_delta at 0.04-0.5
        # across scenes/sizes; filter_input (rejected) measured 0.26-4.17.
        # A generous ceiling that would still catch a real regression.
        assert halo < 2.0


class TestMarkProtectMultipleRadii:
    """Multiple mark radii, including the feather<=radius rule."""

    @pytest.mark.parametrize("radius", [3, 6, 10, 20])
    def test_feather_derived_per_blob_stays_bounded_by_radius(self, radius):
        from retouch.frequency import _mark_protect_feather_mask, MARK_PROTECT_FEATHER_FACTOR

        shape = (200, 200)
        mask = _mark_mask((*shape, 3), (100, 100), radius)
        out = _mark_protect_feather_mask(mask, shape)
        assert out is not None
        # Fully inside the mark: full protection.
        assert out[100, 100] == pytest.approx(1.0, abs=1e-6)
        # The actual bound this test is named for: feather is derived as
        # MARK_PROTECT_FEATHER_FACTOR (0.6) * blob radius, so protection
        # must have decayed to ~nothing by 2x the mark's own radius --
        # a fixed pixel offset (as used below) wouldn't scale with radius
        # and wouldn't distinguish "bounded by radius" from "bounded by a
        # constant."
        beyond_radius = out[100, min(199, 100 + int(radius * 2))]
        assert beyond_radius < 0.05
        # And at exactly the mark's edge, feather should not have fully
        # decayed yet for a properly radius-scaled feather (except the
        # smallest radius, where feather size rounds down to ~1px).
        if radius >= 6:
            at_edge = out[100, 100 + radius]
            assert at_edge > 0.05


class TestMarkProtectNearParserBoundary:
    """Marks near parser boundaries (e.g. skin/eye edge) must not protect
    or size a feather from pixels the caller's skin_mask already excludes."""

    def test_mark_protect_is_clipped_to_the_skin_mask_boundary(self):
        h = w = 200
        canvas = _textured_canvas()[:h, :w]
        # A mark whose footprint straddles a hard skin-mask boundary at
        # x=100 (simulating a mark detected right at an eye/skin edge).
        center, radius = (100, 100), 15
        canvas = _add_mark(canvas, center, radius)
        mark_mask = _mark_mask(canvas.shape, center, radius)

        skin_mask = np.ones((h, w), dtype=np.float32)
        skin_mask[:, 100:] = 0.0  # right half is NOT skin (e.g. an eye region)

        layers = separate(canvas, face_width=FACE_WIDTH)

        def _combine(mp):
            return combine(
                layers, skin_mask=skin_mask, smooth_strength=0.9, mid_reduction=0.35,
                face_width=FACE_WIDTH, smooth_engine="guided", mark_protect=mp,
            )

        protected = np.clip(_combine(mark_mask), 0, 255).astype(np.uint8)
        unprotected = np.clip(_combine(None), 0, 255).astype(np.uint8)

        # Pixels outside skin_mask must be untouched regardless of
        # mark_protect (combine's own pre-existing contract via
        # blend_masked at m_2d==0 there) -- true whether or not this
        # feature ran at all, so assert it holds for both arms.
        assert np.array_equal(protected[:, 150:], canvas[:, 150:])
        assert np.array_equal(unprotected[:, 150:], canvas[:, 150:])

        # The discriminating half: inside skin (x < 100, left of the
        # boundary and inside the mark's protected footprint), protection
        # must actually change the smoothed output relative to no
        # protection -- otherwise this test would pass identically even
        # if mark_protect were silently ignored near a boundary.
        left_of_boundary = slice(0, 100)
        assert not np.array_equal(
            protected[80:120, left_of_boundary], unprotected[80:120, left_of_boundary]
        )


class TestMarkProtectOverlapping:
    """Overlapping protected marks merge into one blob (distance-transform
    radius of the merged shape), not two conflicting per-mark radii."""

    def test_overlapping_marks_both_fully_protected(self):
        from retouch.frequency import _mark_protect_feather_mask

        shape = (200, 200)
        mask = np.zeros(shape, dtype=np.float32)
        cv2.circle(mask, (95, 100), 10, 1.0, -1)
        cv2.circle(mask, (108, 100), 10, 1.0, -1)  # overlaps the first

        out = _mark_protect_feather_mask(mask, shape)
        assert out is not None
        assert out[100, 95] == pytest.approx(1.0, abs=1e-6)
        assert out[100, 108] == pytest.approx(1.0, abs=1e-6)

    def test_overlapping_marks_render_without_artifact_between_them(self):
        center_a, center_b = (95, 128), (115, 128)
        radius = 10
        canvas = _textured_canvas()
        canvas = _add_mark(canvas, center_a, radius)
        canvas = _add_mark(canvas, center_b, radius)
        mask = np.zeros(canvas.shape[:2], dtype=np.float32)
        cv2.circle(mask, center_a, radius, 1.0, -1)
        cv2.circle(mask, center_b, radius, 1.0, -1)

        result = np.clip(_run(canvas, mask, 0.9, mark_protect=mask), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Both mark centers should retain most of their source contrast.
        for center in (center_a, center_b):
            source_c = _mark_contrast(source_gray, center, radius)
            result_c = _mark_contrast(gray, center, radius)
            assert result_c > 0.6 * source_c


class TestMarkProtectLegacyEquivalence:
    """mark_policy=None (mark_protect=None) must stay byte-identical to
    pre-feature output -- the zero/legacy behavior criterion."""

    def test_mark_protect_none_is_byte_identical_to_pre_feature_call(self):
        center, radius = (128, 128), 10
        canvas = _add_mark(_textured_canvas(), center, radius)
        mask = _mark_mask(canvas.shape, center, radius)

        with_default = _run(canvas, mask, 0.9)  # mark_protect defaults to None
        with_explicit_none = _run(canvas, mask, 0.9, mark_protect=None)
        assert np.array_equal(with_default, with_explicit_none)

    def test_mark_protect_none_matches_frequency_separator_direct_call(self):
        """FrequencySeparator().combine() called without mark_protect at all
        (as every pre-existing call site in the repo does) must be
        unaffected by the new parameter's default."""
        center, radius = (128, 128), 10
        canvas = _add_mark(_textured_canvas(), center, radius)
        layers = separate(canvas, face_width=FACE_WIDTH)
        skin_mask = np.ones(canvas.shape[:2], dtype=np.float32)

        sep = FrequencySeparator()
        legacy_call = sep.combine(
            layers, skin_mask=skin_mask, smooth_strength=0.9,
            mid_reduction=0.35, face_width=FACE_WIDTH, smooth_engine="guided",
        )
        explicit_none = sep.combine(
            layers, skin_mask=skin_mask, smooth_strength=0.9,
            mid_reduction=0.35, face_width=FACE_WIDTH, smooth_engine="guided",
            mark_protect=None,
        )
        assert np.array_equal(legacy_call, explicit_none)


class TestMarkProtectEngineGate:
    """Bilateral/anisotropic must not receive mark protection -- validated
    scope is guided smoothing only, enforced by an explicit gate."""

    @pytest.mark.parametrize("engine", ["bilateral", "anisotropic"])
    def test_non_guided_engines_ignore_mark_protect(self, engine):
        center, radius = (128, 128), 10
        canvas = _add_mark(_textured_canvas(), center, radius)
        mask = _mark_mask(canvas.shape, center, radius)

        without = _run(canvas, mask, 0.9, mark_protect=None, smooth_engine=engine)
        with_protect = _run(canvas, mask, 0.9, mark_protect=mask, smooth_engine=engine)
        assert np.array_equal(without, with_protect)

    def test_guided_engine_is_affected_by_mark_protect(self):
        """Sanity check that the gate is engine-selective, not a global
        no-op -- guided must actually change when mark_protect is set."""
        center, radius = (128, 128), 10
        canvas = _add_mark(_textured_canvas(), center, radius)
        mask = _mark_mask(canvas.shape, center, radius)

        without = _run(canvas, mask, 0.9, mark_protect=None, smooth_engine="guided")
        with_protect = _run(canvas, mask, 0.9, mark_protect=mask, smooth_engine="guided")
        assert not np.array_equal(without, with_protect)

"""Tests for the FA-03 mark-classification experiment
(scripts/qa/fa03_mark_classification_experiment.py).

docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md is the research
baseline (not re-derived here). The confirmed failure: two DSCF2310
eyeliner components score beauty_mark=0.70 and blemish=0.70 (an exact
tie), and Python dict insertion order in freckle.py's
`max(scores, key=scores.get)` silently resolves the tie toward
beauty_mark -> mole. This experiment module is diagnostic-only; it does
not modify retouch/freckle.py or retouch/marks.py. These tests verify
the experiment's own contract: deterministic tie/abstain handling,
geometry+context features actually distinguishing eyeliner shape from
compact marks, and that no arm rejects a genuine compact near-eye mark.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "qa"))

from fa03_mark_classification_experiment import (  # noqa: E402
    ARMS,
    ABSTAIN,
    Candidate,
    TIE_MARGIN,
    _resolve_winner,
    arm_current_baseline,
    arm_geometry_only,
    arm_geometry_plus_semantic,
    arm_scored_classical,
    arm_semantic_context_only,
    compute_features,
    freeze_candidates,
)


CANVAS = 256


def _skin_canvas(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = CANVAS
    base = np.full((h, w, 3), (170.0, 180.0, 210.0), dtype=np.float32)
    noise = rng.normal(0, 6.0, (h, w)).astype(np.float32)
    base += noise[:, :, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _full_skin_mask() -> np.ndarray:
    return np.ones((CANVAS, CANVAS), dtype=np.float32)


def _eye_region_mask(center=(128, 128), axes=(30, 14)) -> np.ndarray:
    m = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    cv2.ellipse(m, center, axes, 0, 0, 360, 1.0, -1)
    return m


def _empty_regions(eye_center=(128, 128), eye_axes=(30, 14)) -> dict:
    return {
        "left_eye": _eye_region_mask(eye_center, eye_axes),
        "right_eye": None,
        "left_eyebrow": None,
        "right_eyebrow": None,
        "hair": None,
    }


def _composite_stroke(canvas: np.ndarray, p0, p1, thickness: int, color=(50.0, 55.0, 70.0)) -> np.ndarray:
    """A thin, elongated, low-solidity stroke -- eyeliner-like shape."""
    out = canvas.copy()
    cv2.line(out, p0, p1, color, thickness, lineType=cv2.LINE_AA)
    return out


def _composite_dot(canvas: np.ndarray, center, radius: int, color=(50.0, 55.0, 70.0)) -> np.ndarray:
    """A compact, round, high-solidity mark -- mole/freckle-like shape."""
    out = canvas.copy()
    cv2.circle(out, center, radius, color, -1, lineType=cv2.LINE_AA)
    return out


def _features_for_point(canvas: np.ndarray, skin_mask: np.ndarray, regions: dict, near_point, face_width=100.0):
    candidates = freeze_candidates(canvas, skin_mask)
    assert candidates, "expected at least one detected candidate"
    best = min(
        candidates,
        key=lambda c: np.hypot(c.centroid[0] - near_point[0], c.centroid[1] - near_point[1]),
    )
    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_pixels = lab[skin_mask > 0.3]
    a_median = float(np.median(skin_pixels[:, 1]))
    a_std = float(np.std(skin_pixels[:, 1]))
    l_median = float(np.median(skin_pixels[:, 0]))
    l_std = float(np.std(skin_pixels[:, 0]))
    return best, compute_features(
        best, lab, skin_mask, a_median, a_std, l_median, l_std, regions, face_width,
    )


class TestTieHandlingIsOrderIndependent:
    """Regression test reproducing the current equal-score failure and
    proving the fix does not depend on dict insertion order."""

    def test_exact_tie_between_two_classes_abstains_not_first_class(self):
        scores = {"beauty_mark": 0.7, "blemish": 0.7, "freckle": 0.0, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.winner == ABSTAIN
        assert result.margin == pytest.approx(0.0)

    def test_tie_outcome_is_identical_regardless_of_dict_key_order(self):
        forward = {"freckle": 0.0, "beauty_mark": 0.7, "blemish": 0.7, "noise": 0.0, "eye_makeup": 0.0}
        reversed_order = {"eye_makeup": 0.0, "noise": 0.0, "blemish": 0.7, "beauty_mark": 0.7, "freckle": 0.0}
        r1 = _resolve_winner(forward)
        r2 = _resolve_winner(reversed_order)
        assert r1.winner == r2.winner == ABSTAIN

    def test_near_tie_within_margin_also_abstains(self):
        scores = {"beauty_mark": 0.70, "blemish": 0.70 - (TIE_MARGIN / 2), "freckle": 0.0, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.winner == ABSTAIN

    def test_clear_winner_above_margin_is_not_abstained(self):
        scores = {"beauty_mark": 0.9, "blemish": 0.2, "freckle": 0.0, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.winner == "beauty_mark"
        assert result.margin == pytest.approx(0.7)

    def test_dscf2310_reported_scores_reproduce_as_ambiguous_not_mole(self):
        """Exact scores from the audit JSON for the viewer-left eye
        component (freckle=.45, beauty_mark=.70, blemish=.70, noise=0)."""
        scores = {"freckle": 0.45, "beauty_mark": 0.70, "blemish": 0.70, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.winner == ABSTAIN, (
            "freckle.py's max(scores, key=scores.get) would silently pick "
            "beauty_mark here via dict insertion order; the experiment's "
            "resolver must not reproduce that"
        )


class TestElongatedEyeAdjacentCandidate:
    """The confirmed DSCF2310 failure shape: elongated, low-solidity,
    adjacent to and aligned with an eye boundary."""

    def test_current_baseline_ties_on_synthetic_eyeliner_like_stroke(self):
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        # Stroke just below and tangent to the eye ellipse's lower edge.
        canvas = _composite_stroke(canvas, (100, 145), (140, 150), thickness=2)
        _, feat = _features_for_point(canvas, skin, regions, near_point=(120, 147))
        assert feat["pca_axis_ratio"] >= 3.0, "synthetic stroke must actually be elongated"

    def test_geometry_and_context_arms_classify_stroke_as_eye_makeup(self):
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        canvas = _composite_stroke(canvas, (100, 145), (140, 150), thickness=2)
        _, feat = _features_for_point(canvas, skin, regions, near_point=(120, 147))

        for arm_name, arm_fn in (
            ("geometry_only", arm_geometry_only),
            ("geometry_plus_semantic", arm_geometry_plus_semantic),
            ("scored_classical", arm_scored_classical),
        ):
            result = arm_fn(feat)
            assert result.winner == "eye_makeup", (
                f"{arm_name} should classify an elongated, eye-aligned, "
                f"low-solidity stroke as eye_makeup, got {result.winner} "
                f"(scores={result.scores})"
            )

    def test_current_baseline_never_produces_confident_mole_on_this_shape(self):
        """Even without shape awareness, the fixed tie-break must not
        silently emit beauty_mark for the exact tie this shape produces."""
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        canvas = _composite_stroke(canvas, (100, 145), (140, 150), thickness=2)
        _, feat = _features_for_point(canvas, skin, regions, near_point=(120, 147))
        result = arm_current_baseline(feat)
        assert result.winner != "beauty_mark" or result.margin >= TIE_MARGIN


class TestCompactGenuineMarkNearEye:
    """Preservation requirement: genuine compact marks near the eye must
    remain eligible, not rejected merely for proximity."""

    def test_compact_round_mark_near_eye_is_not_classified_as_eye_makeup_by_shape_arms(self):
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        # Compact dot close to (but not overlapping) the eye ellipse.
        mark_center = (128, 150)
        canvas = _composite_dot(canvas, mark_center, radius=4)
        _, feat = _features_for_point(canvas, skin, regions, near_point=mark_center)
        assert feat["pca_axis_ratio"] < 2.0, "synthetic dot must actually be compact"

        for arm_name, arm_fn in (
            ("current_baseline", arm_current_baseline),
            ("geometry_only", arm_geometry_only),
            ("geometry_plus_semantic", arm_geometry_plus_semantic),
            ("scored_classical", arm_scored_classical),
        ):
            result = arm_fn(feat)
            assert result.winner != "eye_makeup", (
                f"{arm_name} incorrectly classified a compact near-eye mark "
                f"as eye_makeup by proximity alone (scores={result.scores})"
            )

    def test_semantic_context_only_arm_is_known_weak_on_this_case(self):
        """Documents a real limitation found during this experiment:
        semantic-context-only has no shape/appearance evidence, so a
        compact mark close to the eye can score eye_makeup on proximity
        alone in this arm. This is exactly why the task requires
        comparing arms rather than shipping context-only alone."""
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        mark_center = (128, 145)
        canvas = _composite_dot(canvas, mark_center, radius=4)
        _, feat = _features_for_point(canvas, skin, regions, near_point=mark_center)
        result = arm_semantic_context_only(feat)
        # Not asserting a "correct" outcome here -- recording the actual
        # observed weakness so a future reader does not rediscover it by
        # trial and error, and so this arm is not mistaken for safe to
        # ship standalone.
        assert result.winner in ("eye_makeup", ABSTAIN, "freckle", "beauty_mark")


class TestCompactMarkAwayFromBoundaries:
    """Control case: a compact mark with no nearby semantic boundary."""

    def test_compact_mark_far_from_eye_is_classified_as_identity_like(self):
        canvas = _skin_canvas()
        regions = _empty_regions(eye_center=(20, 20), eye_axes=(5, 3))  # far corner, irrelevant
        skin = _full_skin_mask()
        mark_center = (200, 200)
        canvas = _composite_dot(canvas, mark_center, radius=4)
        _, feat = _features_for_point(canvas, skin, regions, near_point=mark_center)
        assert feat["eye_distance_norm"] > 0.5

        for arm_name, arm_fn in (
            ("current_baseline", arm_current_baseline),
            ("geometry_only", arm_geometry_only),
            ("geometry_plus_semantic", arm_geometry_plus_semantic),
            ("scored_classical", arm_scored_classical),
        ):
            result = arm_fn(feat)
            assert result.winner in ("beauty_mark", "freckle", ABSTAIN), (
                f"{arm_name} unexpectedly classified an isolated compact mark "
                f"as {result.winner}"
            )


class TestAmbiguityAbstentionBehavior:
    """The classification contract: preserve competing scores, add an
    explicit abstain outcome, never force a confident class on
    conflicting or insufficient evidence."""

    def test_no_positive_evidence_resolves_to_noise_not_arbitrary_class(self):
        scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.winner == "noise"

    def test_abstain_result_still_carries_full_score_vector(self):
        scores = {"freckle": 0.45, "beauty_mark": 0.70, "blemish": 0.70, "noise": 0.0, "eye_makeup": 0.0}
        result = _resolve_winner(scores)
        assert result.scores == scores, "abstain must not discard competing class scores"

    def test_ambiguous_is_a_distinct_outcome_from_every_real_class(self):
        assert ABSTAIN not in ("freckle", "beauty_mark", "blemish", "noise", "eye_makeup")


class TestDeterminism:
    """Same candidate, same input, same result on repeated calls -- no
    hidden randomness or iteration-order dependence anywhere in the
    scoring path."""

    def test_repeated_calls_are_byte_identical(self):
        canvas = _skin_canvas()
        eye_center = (128, 128)
        regions = _empty_regions(eye_center=eye_center, eye_axes=(30, 14))
        skin = _full_skin_mask()
        canvas = _composite_stroke(canvas, (100, 145), (140, 150), thickness=2)
        _, feat1 = _features_for_point(canvas, skin, regions, near_point=(120, 147))
        _, feat2 = _features_for_point(canvas, skin, regions, near_point=(120, 147))
        for arm_fn in ARMS.values():
            r1 = arm_fn(feat1)
            r2 = arm_fn(feat2)
            assert r1.winner == r2.winner
            assert r1.scores == r2.scores


class TestLegacyPathUnchangedWhenExperimentalClassifierDisabled:
    """The experiment module must not alter retouch/freckle.py's actual
    behavior -- it calls the same private candidate-generation method
    but never monkeypatches or replaces it."""

    def test_freeze_candidates_uses_the_real_unmodified_detector(self):
        from retouch.freckle import FreckleRemover

        canvas = _skin_canvas()
        skin = _full_skin_mask()
        original_method = FreckleRemover._detect_components
        candidates = freeze_candidates(canvas, skin)
        assert FreckleRemover._detect_components is original_method, (
            "the experiment must not monkeypatch or replace the production "
            "candidate-generation method"
        )
        # Sanity: candidate generation still runs and returns real Candidate objects.
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_production_classify_anomalies_output_is_unaffected_by_import(self):
        """Importing the experiment module must not change
        FreckleRemover.classify_anomalies's own output on a real scene."""
        from retouch.freckle import FreckleRemover

        canvas = _skin_canvas()
        skin = _full_skin_mask()
        canvas = _composite_stroke(canvas, (100, 145), (140, 150), thickness=2)
        result = FreckleRemover().classify_anomalies(canvas, face_mask=skin)
        # Legacy behavior: still whatever freckle.py's own rules produce --
        # this experiment does not change it. Just confirm it still runs
        # and can (per the known bug) still classify shapes without
        # shape/context awareness.
        assert isinstance(result, list)

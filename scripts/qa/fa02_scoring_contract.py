"""FA-02 scoring/promotion CONTRACT: frozen BEFORE any candidate arm is scored.

    python3 scripts/qa/fa02_scoring_contract.py freeze test_output/fa02_scoring_lock.json

This module answers one question and deliberately no other: **under exactly what
frozen, pre-registered rule would an FA-02 texture representation be allowed to
be called a production candidate?** It does not choose a winner, does not tune
anything, and does not touch ``restore_micro_texture()``, ``FrequencySeparator``,
``DEFAULT_CONFIG`` or the six frozen arms.

The narrow production claim this contract exists to certify (owner's wording):

    "Micro-texture restoration is enabled only for faces with sufficient
    native-resolution source texture and validated supports; otherwise the
    pipeline abstains/falls back safely."

Note what that claim is: a claim about an ABSTENTION POLICY, not a claim that
any representation is good. The eligibility contract (:func:`evaluate_eligibility`)
is therefore the operative half; the margin test is what would let a specific arm
be switched on for the faces that clear eligibility.

Design commitments, each of which is the point rather than incidental
--------------------------------------------------------------------

**1. A missing input is never a zero.** Every scoring term returns a
:class:`TermResult` that is either ``available`` with a number, or ``unavailable``
with the name of the input that was missing. A weighted sum silently dropping an
unavailable term and renormalizing around it is precisely how "the candidate
wins" gets manufactured out of safety terms with zero texture evidence. The
aggregator (:func:`aggregate_case_score`) therefore REFUSES to emit a score for
a case when any *required* term is unavailable, rather than scoring what is left.

**2. Disqualifiers are three-valued, and "no data" is not a pass.** A hard gate
evaluated against absent data returns ``not_applicable``, never ``pass``. A case
whose gates are ALL ``not_applicable`` has demonstrated nothing and is refused
admission to the margin test (:func:`evaluate_disqualifiers`). This matters
concretely: on the ten committed real pilot patches ``forbidden_max_abs_delta``
is ``None`` on nine of them (there are no forbidden pixels to measure), and a
two-valued gate would have reported "all arms clear all safety gates" from an
empty measurement.

**3. Disqualifiers are not part of the weighted sum.** They are per-case hard
gates evaluated independently of any score. An arm failing any disqualifier on
any admissible case is out, regardless of how well it scores.

**4. Beating A0 requires strictly positive recovery evidence.** ``A0_disabled``
emits an all-zero delta, so it scores exactly 0 on every recovery term AND 0 on
every penalty term. Its aggregate is structurally 0. Therefore no combination of
safety/penalty terms can beat it -- only positive pore or fine-hair recovery can.
This is a proved property of the rule, not an accident, and it is what makes the
contract impossible to satisfy with the corpus as it stands. See
:data:`CONTRACT_PROPERTIES`.

**5. Provenance is recorded for every constant.** Each numeric constant below is
tagged MEASURED, PLACEHOLDER or BORROWED. Nothing here is validated. The eye-gate
recalibration of 2026-08-31 (CLAUDE.md, Outstanding Fixes) is the precedent this
file follows deliberately: thresholds shipped against a mislabelled anchor and
had to be corrected against real study data later. Every threshold in this file
carries that caveat from day one rather than acquiring it after a regression.

Python 3.9 compatible. JSON-safe returns throughout. Fail-closed validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fa02_texture_representation_experiment as exp  # noqa: E402

SCORING_SCHEMA_VERSION = 1

# The two reference arms the candidate must beat. Both are present in every
# committed run, so this is not an aspirational reference.
BASELINE_ARMS = ("A0_disabled", "A1_raw")

# ---------------------------------------------------------------------------
# Provenance vocabulary for every numeric constant in this file
# ---------------------------------------------------------------------------
# MEASURED    -- a number read off a real measurement recorded in this
#                repository, with the citation in the comment.
# PLACEHOLDER -- an arbitrary starting value that has NEVER been calibrated.
#                It exists so the rule is complete and executable, not because
#                any evidence supports it.
# BORROWED    -- taken from an existing convention already used elsewhere in
#                this project, cited in the comment.
PROVENANCE_KINDS = ("MEASURED", "PLACEHOLDER", "BORROWED")


def _c(value, provenance, note):
    """One frozen constant carrying its own provenance. JSON-safe."""
    if provenance not in PROVENANCE_KINDS:
        raise ValueError("Unknown provenance kind: " + str(provenance))
    return {"value": value, "provenance": provenance, "note": note}


# ---------------------------------------------------------------------------
# THE FROZEN SCORING RULE (data, not inline weights)
# ---------------------------------------------------------------------------
# Weights sum to 1.0 across the seven terms. The split is 0.60 recovery /
# 0.40 penalty: a representation exists to RESTORE texture, so recovery must
# dominate, but not so far that a noise amplifier can buy a win.
#
# EVERY WEIGHT BELOW IS A PLACEHOLDER. None was fitted, and none can be
# fitted, because no arm has ever been scored under this rule. They are frozen
# in this state deliberately: the whole point of pre-registration is that the
# numbers are chosen while the results are unknown.
SCORING_RULE = {
    "schema_version": SCORING_SCHEMA_VERSION,
    "rule_id": "fa02_texture_candidate_v1",
    "claim": (
        "Micro-texture restoration is enabled only for faces with sufficient "
        "native-resolution source texture and validated supports; otherwise the "
        "pipeline abstains/falls back safely."
    ),
    "terms": {
        # ---------------- Recovery terms (positive is good) ----------------
        "pore_recovery": {
            "weight": _c(0.35, "PLACEHOLDER",
                         "Largest single weight: pores are the signal FA-02 exists to "
                         "restore. Never calibrated -- no pore has ever been annotated."),
            "sign": "higher_is_better",
            "required": True,
            "source": "summary.json results[].features[] (kind=pore_contrast).signed_recovery",
            "input_shape": 'case["pores"] entries {id, center:[x,y], radius}',
            "artifact_class": "committed_metadata",
            "statistic": "median of per-feature signed_recovery, clipped to [-1, 1]",
            "clip": _c([-1.0, 1.0], "BORROWED",
                       "signed_recovery is a RATIO of contrast recovered to contrast lost; "
                       "1.0 is exact restoration. Clipping bounds an overshoot ratio from "
                       "dominating the median the same way exp.score() bounds its signed "
                       "display range."),
        },
        "fine_hair_recovery": {
            "weight": _c(0.25, "PLACEHOLDER",
                         "Second recovery signal. Weighted below pores because a strand is "
                         "a sparser, more fragile measurement. Never calibrated."),
            "sign": "higher_is_better",
            "required": True,
            "source": "summary.json results[].hair_trace_samples[]",
            "input_shape": 'case["profiles"] entries {id, trace_id, kind, p0:[x,y], p1:[x,y]}',
            "artifact_class": "committed_metadata",
            "statistic": (
                "sum(retained_sections_at_half_source_contrast) / sum(observable_sections) "
                "across traces; undefined when no trace has an observable section"
            ),
        },
        # ---------------- Penalty terms (positive magnitude is bad) --------
        "noise_amplification": {
            "weight": _c(0.15, "PLACEHOLDER",
                         "Returning sensor noise/JPEG blocking as 'texture' is the primary "
                         "failure mode of every arm here. Never calibrated."),
            "sign": "lower_is_better",
            "required": True,
            "source": (
                "summary.json results[].increment_projection_onto_nuisance_confoundable "
                "and .jpeg_block_excess_change_proxy"
            ),
            "input_shape": "a paired-clean control case (nuisance array or jpeg_origin_xy)",
            "artifact_class": "committed_metadata",
            "statistic": "max(|nuisance projection|, |jpeg block excess|) over whichever exist",
        },
        "defect_reintroduction": {
            "weight": _c(0.10, "PLACEHOLDER",
                         "Bringing back a defect a human already repaired. Weighted below "
                         "noise only because the hard disqualifier below already catches the "
                         "severe case; this term prices the sub-threshold remainder."),
            "sign": "lower_is_better",
            "required": True,
            "source": "summary.json results[].corrected_signed_contrast_recovery",
            "input_shape": "a non-empty corrected mask in the case NPZ",
            "artifact_class": "committed_metadata",
            "statistic": "max(0, corrected_signed_contrast_recovery) -- only reintroduction is penalised",
        },
        "makeup_edge_contamination": {
            "weight": _c(0.05, "PLACEHOLDER",
                         "A makeup edge is a nuisance edge an arm may over-sharpen. Lowest "
                         "penalty weight because it is a cosmetic artifact, not a safety "
                         "failure. Never calibrated."),
            "sign": "lower_is_better",
            "required": True,
            "source": "run/<case>/<arm>/signed_arrays.npz delta, inside annotation makeup_edge_regions",
            "input_shape": "annotation makeup_edge_regions[] rectangles + the delta float arrays",
            # This is the ONE term whose input is not committed metadata. The
            # float arrays are gitignored (.gitignore:7 test_output/), so on a
            # fresh clone this term is unavailable even where labels exist.
            "artifact_class": "local_float_arrays",
            "statistic": "RMS of delta luma inside the union of makeup_edge_regions rectangles",
        },
        "halo": {
            "weight": _c(0.05, "PLACEHOLDER",
                         "Ring overshoot around a protected/corrected support. Never calibrated."),
            "sign": "lower_is_better",
            "required": True,
            # Chosen: the RINGS mechanism, not profiles' edge_shape overshoot.
            # Rationale: rings are computed for every case that has any
            # corrected/protected support, and are already keyed by distance
            # from that support, which is exactly the halo geometry. edge_shape
            # overshoot only exists on case["profiles"] entries of kind="edge",
            # which is the SAME missing input as fine_hair_recovery -- using it
            # would make two terms fail for one reason and overstate coverage.
            "source": "summary.json results[].rings[] with outer_distance_px > 0",
            "input_shape": "a non-empty corrected or protected mask in the case NPZ",
            "artifact_class": "committed_metadata",
            "statistic": "max |delta_mean| over rings strictly outside the support",
        },
        "broad_tone_chroma_leakage": {
            "weight": _c(0.05, "PLACEHOLDER",
                         "A texture arm must not shift overall tone or colour. Low weight "
                         "because the band-limited arms reject DC by construction; this is a "
                         "regression tripwire, not a discriminator. Never calibrated."),
            "sign": "lower_is_better",
            "required": True,
            "source": "summary.json results[].broad_luma_rms and .chroma_delta_rms",
            "input_shape": "always present -- computed for every case/arm",
            "artifact_class": "committed_metadata",
            "statistic": "broad_luma_rms + chroma_delta_rms, in encoded 0-255 levels",
            "normalizer": _c(2.0, "BORROWED",
                             "exp.DEFAULT_CONFIG['signed_display_range'] = 2.0, the project's "
                             "existing 'this much change is fully visible' scale. Reused so "
                             "the penalty is expressed on a scale the project already uses."),
        },
    },
    # A penalty term's raw magnitude is divided by its scale before weighting,
    # so a penalty equal to its scale costs exactly its full weight.
    "penalty_scales": {
        "noise_amplification": _c(1.0, "PLACEHOLDER",
                                  "A projection of 1.0 means the arm returned 100% of the "
                                  "injected nuisance. Never calibrated against a real face."),
        "defect_reintroduction": _c(1.0, "PLACEHOLDER",
                                    "1.0 = the full original defect contrast is back. "
                                    "Never calibrated."),
        "makeup_edge_contamination": _c(2.0, "BORROWED",
                                        "signed_display_range, as above."),
        "halo": _c(2.0, "BORROWED", "signed_display_range, as above."),
        "broad_tone_chroma_leakage": _c(2.0, "BORROWED", "signed_display_range, as above."),
    },
    "aggregation": {
        "case_score": "sum(weight * term_value) over recovery terms minus sum(weight * min(1, |penalty| / scale))",
        "required_terms_policy": (
            "ALL terms marked required=true must be available for a case to receive a "
            "score. A case missing any required term is scored as null and excluded "
            "from the margin test. Weights are NEVER renormalized around a missing term."
        ),
        "baseline_comparison": "per_case",
        "baseline_comparison_note": (
            "The candidate must show a strictly positive margin against BOTH A0_disabled "
            "and A1_raw on EVERY admissible case. Per-case, not median: the production "
            "claim is per-face ('enabled only for faces with sufficient texture'), so the "
            "test matches the claim's granularity. A median over ~10 patches also flips on "
            "one or two cases, which is not a basis for a production switch."
        ),
        "minimum_margin": _c(0.0, "PLACEHOLDER",
                             "Strictly greater than zero. No minimum effect size is set "
                             "because no effect has ever been measured; a real one must be "
                             "chosen from calibration data before a locked_test run."),
        "minimum_admissible_cases": _c(8, "PLACEHOLDER",
                                       "Refuse to emit a verdict below this many admissible "
                                       "cases. Mirrors CANDIDATE_SUBJECT_TARGET[0]=8 in "
                                       "fa02_readiness_report.py so the scoring rule cannot "
                                       "declare a winner on a corpus the readiness report "
                                       "already calls too small. Never validated."),
    },
    "eligibility": {
        "note": (
            "The operative half of the production claim. A face/patch that does not clear "
            "BOTH gates must ABSTAIN: micro-texture restoration does not run and the "
            "pipeline falls back to its existing non-restored output."
        ),
        "texture_metric": "highpass_std",
        "texture_metric_definition": (
            "std( luma(X) - GaussianBlur(luma(X), sigma) ) measured on the NATIVE X canvas "
            "inside the allow support only. This is a texture AMPLITUDE measure relative to "
            "the patch's own local mean, NOT an absolute intensity threshold, so it does not "
            "violate the tone-invariance rule in CLAUDE.md (it does not scale with skin tone)."
        ),
        "highpass_sigma_px": _c(2.0, "MEASURED",
                                "The exact sigma used by this session's ff47-event visual "
                                "measurement (commit 427ae49): 'highpass std over a 2px "
                                "Gaussian recorded per tile'. Matching the sigma is required "
                                "for the thresholds below to mean anything -- Laplacian "
                                "variance (retouch/image_analyzer.py:284) is a DIFFERENT "
                                "quantity and its numbers are not interchangeable with these."),
        "minimum_highpass_std": _c(3.5, "PLACEHOLDER",
                                   "PROVISIONAL, N=10 patches, NOT a validated production "
                                   "threshold. Bracketed by this session's ff47-event visual "
                                   "measurement (commit 427ae49): unresolvable skin measured "
                                   "1.3-1.9 and the one genuinely separable fine-hair region "
                                   "(DSCF1058) measured 6.75. 3.5 is placed between those two "
                                   "populations; it is NOT a decision boundary derived from a "
                                   "sweep, and there are exactly two anchor populations with "
                                   "no overlap sample between them. Needs recalibration "
                                   "against a corpus that actually contains resolvable pores. "
                                   "Compare the eye-gate precedent (CLAUDE.md 2026-08-31): a "
                                   "threshold set from a single mislabelled anchor missed 84% "
                                   "of true positives until it was re-derived from study data."),
        "requires_validated_supports": _c(True, "BORROWED",
                                          "fa02_portrait_manifest.ACCEPTED_ANNOTATION_SOURCES: "
                                          "a support is validated only when it is manual or "
                                          "owner_approved. A detector-derived or synthetic "
                                          "support is recordable but never accepted."),
        "required_ground_truth_categories": _c(["pore_regions", "fine_hair_regions"], "BORROWED",
                                               "fa02_portrait_manifest.GROUND_TRUTH_CATEGORIES. "
                                               "At least one must be populated: a face with no "
                                               "labelled recoverable texture cannot demonstrate "
                                               "recovery, so it must abstain."),
        "abstain_action": (
            "Do not run micro-texture restoration on this face. Fall back to the existing "
            "pipeline output (equivalent to arm A0_disabled). Abstention is the SAFE default "
            "and is what the pipeline does today; eligibility is a gate to switch restoration "
            "ON, never a gate to switch a safety behaviour off."
        ),
    },
    "disqualifiers": {
        "note": (
            "Per-case HARD GATES, evaluated independently of any weighted score. An arm "
            "failing ANY disqualifier on ANY admissible case is disqualified outright. "
            "Three-valued: a gate with no data returns not_applicable, NEVER pass."
        ),
        "forbidden_pixel_change": {
            "field": "forbidden_max_abs_delta",
            "threshold": _c(0.0, "BORROWED",
                            "Exactly zero. The harness's common guard already makes this "
                            "structurally zero (EXPERIMENT report: 'All 66 outputs have "
                            "exactly zero change in externally forbidden pixels'). Re-asserted "
                            "here as an explicit gate rather than trusted as an invariant."),
            "comparison": "must_equal",
        },
        "innermost_ring_leakage": {
            "field": "rings[] where outer_distance_px == 0",
            "threshold": _c(0.0, "BORROWED",
                            "Same zero-leakage guarantee as above, measured at the support "
                            "itself rather than as a whole-image maximum."),
            "comparison": "abs_must_equal",
        },
        "defect_reintroduction": {
            "field": "corrected_signed_contrast_recovery",
            "threshold": _c(0.25, "PLACEHOLDER",
                            "PROVISIONAL. Bringing back more than a quarter of a defect a "
                            "human already repaired is a visible regression. No study "
                            "supports 0.25 specifically -- it is a quarter, chosen as an "
                            "obviously-too-much fraction, and the corpus contains ZERO "
                            "corrected masks against which to calibrate it."),
            "comparison": "must_be_at_most",
        },
        "makeup_edge_contamination": {
            "field": "RMS delta luma inside makeup_edge_regions",
            "threshold": _c(1.0, "PLACEHOLDER",
                            "PROVISIONAL, in encoded 0-255 levels: half of the project's "
                            "signed_display_range (2.0, exp.DEFAULT_CONFIG). Never calibrated "
                            "against a real over-sharpened makeup edge."),
            "comparison": "must_be_at_most",
        },
    },
    "explicitly_not_disqualifiers": {
        "protected_signed_contrast_recovery": (
            "An OBSERVATION, not a gate. forbidden_max_abs_delta already covers 'protected "
            "pixels changed', and a contrast-recovery ratio measured in a ring AROUND a "
            "protected mark is a halo measurement, which the halo term already prices. "
            "Making it a second gate would double-count one physical effect. Recorded in "
            "the report as an observation so an owner can see the number."
        ),
    },
    "out_of_scope": (
        "This contract does NOT advance the corpus's promotion stage. "
        "fa02_readiness_report.py's production_candidate_ready and production_ready criteria "
        "apply unchanged and most remain unmet (locked_test split empty, calibration split "
        "empty, five non-computable human sign-offs outstanding). This file only makes the "
        "'candidate beats A0/A1' criterion mechanically checkable ONCE its inputs exist."
    ),
}

# Properties of the rule that follow from its structure rather than from any
# measurement. Stated here so an auditor can verify them by reading, and so the
# report can cite them instead of re-deriving them.
CONTRACT_PROPERTIES = {
    "a0_is_structurally_zero": (
        "A0_disabled emits an all-zero delta, so every recovery term is 0 and every "
        "penalty term is 0; its aggregate score is exactly 0.0."
    ),
    "beating_a0_requires_positive_recovery": (
        "Because A0 scores 0.0 and penalties only subtract, a candidate can exceed A0 "
        "ONLY via strictly positive pore_recovery or fine_hair_recovery. No combination "
        "of safety/penalty terms can produce a win. Consequently a corpus with zero pore "
        "and zero usable fine-hair ground truth CANNOT satisfy this contract, by "
        "construction and not by accident."
    ),
    "missing_input_never_scores_zero": (
        "Unavailable terms make the whole case score null. They are never coerced to 0, "
        "which would be indistinguishable from a perfect penalty score."
    ),
    "empty_gates_are_not_a_pass": (
        "A case whose disqualifiers are all not_applicable is inadmissible, so 'no safety "
        "data' can never be reported as 'safe'."
    ),
}


# ---------------------------------------------------------------------------
# Term results: available number, or explicit unavailability. Never a silent 0.
# ---------------------------------------------------------------------------


def term_available(name, value, detail):
    """A computed term contribution."""
    return {"term": name, "available": True, "value": float(value), "detail": detail}


def term_unavailable(name, missing_input, detail):
    """An explicitly missing term. ``value`` is None, never 0."""
    return {
        "term": name,
        "available": False,
        "value": None,
        "missing_input": missing_input,
        "detail": detail,
    }


def _finite(value):
    """True when ``value`` is a real, finite number (bool is not a number here)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return bool(np.isfinite(value))


def _weight(term):
    return SCORING_RULE["terms"][term]["weight"]["value"]


def _scale(term):
    return SCORING_RULE["penalty_scales"][term]["value"]


# ---------------------------------------------------------------------------
# The seven scoring terms. Each is pure: (result entry [, extras]) -> TermResult
# ---------------------------------------------------------------------------


def term_pore_recovery(result):
    """Median signed contrast recovery over annotated pore features.

    Consumes ``result["features"]`` entries of kind ``pore_contrast``, which
    ``exp.score()`` emits one-per-entry for ``case["pores"]``. That list is
    populated only from human-annotated pore centres/radii; the harness never
    finds a pore itself ("not candidate generation", exp.score).

    Unavailable when no pore feature exists, or when every feature's
    ``signed_recovery`` is null (``exp.score`` returns None when the X-to-S
    contrast loss is below 1e-5, i.e. nothing was lost so recovery is undefined).
    """
    features = [f for f in (result.get("features") or [])
                if f.get("kind") == "pore_contrast"]
    if not features:
        return term_unavailable(
            "pore_recovery", 'case["pores"] -> results[].features[]',
            "No annotated pore features. exp.score() emits one feature per "
            'case["pores"] entry {id, center, radius}; an empty list means no '
            "human has annotated a pore centre on this patch.",
        )
    values = [f.get("signed_recovery") for f in features]
    usable = [v for v in values if _finite(v)]
    if not usable:
        return term_unavailable(
            "pore_recovery", "features[].signed_recovery (all null)",
            "{0} pore feature(s) present but every signed_recovery is null "
            "(undefined denominator: no contrast was lost from X to S).".format(len(features)),
        )
    low, high = SCORING_RULE["terms"]["pore_recovery"]["clip"]["value"]
    clipped = [min(high, max(low, float(v))) for v in usable]
    return term_available(
        "pore_recovery", float(np.median(clipped)),
        "median of {0} usable signed_recovery value(s) of {1} feature(s), "
        "clipped to [{2}, {3}]".format(len(clipped), len(features), low, high),
    )


def term_fine_hair_recovery(result):
    """Fraction of observable hair cross-sections retaining half their source contrast.

    Consumes ``result["hair_trace_samples"]``, which ``exp.score()`` derives from
    ``case["profiles"]`` entries carrying a ``trace_id``. "Observable" is the
    harness's own definition: the section had contrast > 0.1 in X. A trace with
    no observable section contributes nothing, because retaining 0 of 0 sections
    is not evidence of anything.
    """
    traces = result.get("hair_trace_samples") or []
    if not traces:
        return term_unavailable(
            "fine_hair_recovery", 'case["profiles"] -> results[].hair_trace_samples[]',
            "No hair trace samples. These require case[\"profiles\"] entries with a "
            "trace_id, i.e. human-annotated line segments across a strand.",
        )
    observable = sum(int(t.get("observable_sections") or 0) for t in traces)
    retained = sum(int(t.get("retained_sections_at_half_source_contrast") or 0) for t in traces)
    if observable <= 0:
        return term_unavailable(
            "fine_hair_recovery", "hair_trace_samples[].observable_sections (all zero)",
            "{0} trace(s) present but none had an observable section (source contrast "
            "> 0.1); there is no strand visible in X to retain.".format(len(traces)),
        )
    return term_available(
        "fine_hair_recovery", retained / float(observable),
        "{0}/{1} observable section(s) retained at half source contrast "
        "across {2} trace(s)".format(retained, observable, len(traces)),
    )


def term_noise_amplification(result):
    """Nuisance returned as if it were texture.

    Consumes the two paired-control fields ``exp.score()`` emits only when the
    case carries them: ``increment_projection_onto_nuisance_confoundable``
    (present iff the NPZ has a ``nuisance`` array) and
    ``jpeg_block_excess_change_proxy`` (present iff the case declares
    ``jpeg_origin_xy``). Takes the worse of whichever exist.
    """
    projection = result.get("increment_projection_onto_nuisance_confoundable")
    jpeg = result.get("jpeg_block_excess_change_proxy")
    magnitudes = []
    if _finite(projection):
        magnitudes.append(("nuisance_projection", abs(float(projection))))
    if _finite(jpeg):
        magnitudes.append(("jpeg_block_excess", abs(float(jpeg))))
    if not magnitudes:
        return term_unavailable(
            "noise_amplification",
            "nuisance array or jpeg_origin_xy (paired-clean control)",
            "Neither increment_projection_onto_nuisance_confoundable nor "
            "jpeg_block_excess_change_proxy is present. Both require a paired-clean "
            "control case: identical S and supports with a KNOWN injected nuisance. "
            "A real portrait has no clean twin, so this needs a deliberately "
            "constructed control pair.",
        )
    worst = max(magnitudes, key=lambda item: item[1])
    return term_available(
        "noise_amplification", worst[1],
        "worst of {0}: {1}".format([m[0] for m in magnitudes], worst[0]),
    )


def term_defect_reintroduction(result):
    """How much of an already-repaired defect the arm brought back.

    Consumes ``corrected_signed_contrast_recovery``, which ``exp.score()`` emits
    only when the case's ``corrected`` mask is non-empty. Only positive recovery
    is penalised: pushing the defect FURTHER down is not this term's concern
    (the forbidden-pixel disqualifier already forbids touching it at all).
    """
    value = result.get("corrected_signed_contrast_recovery")
    if not _finite(value):
        return term_unavailable(
            "defect_reintroduction", "corrected mask (non-empty) in the case NPZ",
            "corrected_signed_contrast_recovery is absent or null. exp.score() emits "
            "it only when arr['corrected'] has non-zero pixels, i.e. when a human "
            "recorded a defect that was repaired before S was captured.",
        )
    return term_available(
        "defect_reintroduction", max(0.0, float(value)),
        "corrected_signed_contrast_recovery={0:.6g} (negative clamped to 0)".format(float(value)),
    )


def term_makeup_edge_contamination(result, makeup_regions, delta_luma=None):
    """RMS delta luma inside annotated makeup-edge rectangles.

    This is a NEW measurement: ``exp.score()`` has no makeup-edge metric, and the
    semantic ``makeup_edge_regions`` category is deliberately kept out of
    ``case["rois"]`` by ``fa02_portrait_manifest`` (merging would let cross-category
    ID collisions through), so the per-ROI statistics cannot supply it either.

    ``delta_luma`` is the case/arm's luma delta plane, read from
    ``run/<case>/<arm>/signed_arrays.npz``. Those float arrays are GITIGNORED
    (.gitignore:7 ``test_output/``), so on a fresh clone they do not exist and
    this term is unavailable even on the three patches that DO carry makeup-edge
    labels. That asymmetry is reported, not hidden: this is the only term whose
    artifact_class is ``local_float_arrays``.
    """
    if not makeup_regions:
        return term_unavailable(
            "makeup_edge_contamination", "annotation makeup_edge_regions[]",
            "No makeup_edge_regions annotated on this patch.",
        )
    if delta_luma is None:
        return term_unavailable(
            "makeup_edge_contamination", "run/<case>/<arm>/signed_arrays.npz delta",
            "{0} makeup_edge_region(s) are annotated, but the signed delta arrays are "
            "not available. They are gitignored (.gitignore:7 test_output/), so this "
            "term cannot be computed from committed metadata alone -- it needs a local "
            "run directory.".format(len(makeup_regions)),
        )
    height, width = delta_luma.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    used = 0
    for region in makeup_regions:
        x, y, w, h = region["xywh"]
        if x + w > width or y + h > height:
            # Bounds are already gated by check_promotion_gates; refuse rather
            # than silently clipping a rectangle into a different measurement.
            return term_unavailable(
                "makeup_edge_contamination", "in-bounds makeup_edge_regions",
                "region {0!r} {1} is outside the {2}x{3} delta plane".format(
                    region.get("id"), region["xywh"], width, height),
            )
        mask[y:y + h, x:x + w] = True
        used += 1
    if not mask.any():
        return term_unavailable(
            "makeup_edge_contamination", "non-empty makeup_edge_regions union",
            "The annotated rectangles cover zero pixels.",
        )
    value = float(np.sqrt(np.mean(np.square(delta_luma[mask], dtype=np.float64))))
    return term_available(
        "makeup_edge_contamination", value,
        "RMS delta luma over {0} px in {1} makeup_edge_region(s)".format(
            int(mask.sum()), used),
    )


def term_halo(result):
    """Worst mean ring delta strictly outside a corrected/protected support.

    Consumes ``result["rings"]`` with ``outer_distance_px > 0``. Ring 0 IS the
    support itself and is a leakage disqualifier, not a halo, so it is excluded
    here to keep the two measurements from double-counting one effect.

    See SCORING_RULE["terms"]["halo"]["source"] for why rings were chosen over
    the ``edge_shape`` overshoot/undershoot alternative.
    """
    rings = [r for r in (result.get("rings") or [])
             if _finite(r.get("outer_distance_px")) and r["outer_distance_px"] > 0]
    if not rings:
        return term_unavailable(
            "halo", "rings[] with outer_distance_px > 0",
            "No rings outside a support. exp.score() emits rings only when a "
            "corrected or protected mask is non-empty on this case.",
        )
    magnitudes = [abs(float(r["delta_mean"])) for r in rings if _finite(r.get("delta_mean"))]
    if not magnitudes:
        return term_unavailable(
            "halo", "rings[].delta_mean", "Rings present but no finite delta_mean.")
    return term_available(
        "halo", max(magnitudes),
        "max |delta_mean| over {0} ring(s) outside the support".format(len(magnitudes)),
    )


def term_broad_tone_chroma_leakage(result):
    """Broad luminance plus chroma drift, in encoded 0-255 levels.

    Both fields are computed for every case/arm by ``exp.score()``, so this is
    the one term expected to be available on real data today. It is also the one
    term that cannot, alone, win anything -- it is a penalty.
    """
    broad = result.get("broad_luma_rms")
    chroma = result.get("chroma_delta_rms")
    missing = [name for name, value in (("broad_luma_rms", broad),
                                        ("chroma_delta_rms", chroma))
               if not _finite(value)]
    if missing:
        return term_unavailable(
            "broad_tone_chroma_leakage", ", ".join(missing),
            "Absent or null; exp.score() returns null when no pixel is eligible.",
        )
    return term_available(
        "broad_tone_chroma_leakage", float(broad) + float(chroma),
        "broad_luma_rms={0:.6g} + chroma_delta_rms={1:.6g}".format(float(broad), float(chroma)),
    )


RECOVERY_TERMS = ("pore_recovery", "fine_hair_recovery")
PENALTY_TERMS = (
    "noise_amplification",
    "defect_reintroduction",
    "makeup_edge_contamination",
    "halo",
    "broad_tone_chroma_leakage",
)


def compute_terms(result, makeup_regions=None, delta_luma=None):
    """Compute all seven terms for one case/arm result entry. JSON-safe."""
    return {
        "pore_recovery": term_pore_recovery(result),
        "fine_hair_recovery": term_fine_hair_recovery(result),
        "noise_amplification": term_noise_amplification(result),
        "defect_reintroduction": term_defect_reintroduction(result),
        "makeup_edge_contamination": term_makeup_edge_contamination(
            result, makeup_regions or [], delta_luma),
        "halo": term_halo(result),
        "broad_tone_chroma_leakage": term_broad_tone_chroma_leakage(result),
    }


def aggregate_case_score(terms):
    """Combine terms into one case/arm score, or refuse.

    Refuses (returns ``score=None``) when ANY required term is unavailable.
    Weights are never renormalized around a missing term -- see
    SCORING_RULE["aggregation"]["required_terms_policy"] for why that rule is
    the load-bearing one.
    """
    missing = sorted(
        name for name, term in terms.items()
        if SCORING_RULE["terms"][name]["required"] and not term["available"]
    )
    if missing:
        return {
            "score": None,
            "scorable": False,
            "missing_required_terms": missing,
            "contributions": {},
            "detail": (
                "Refusing to score: {0} required term(s) unavailable ({1}). Weights are "
                "NOT renormalized around a missing term; a partial score built from the "
                "available safety terms would be a win manufactured from absent texture "
                "evidence.".format(len(missing), ", ".join(missing))
            ),
        }
    contributions = {}
    total = 0.0
    for name in RECOVERY_TERMS:
        contribution = _weight(name) * terms[name]["value"]
        contributions[name] = contribution
        total += contribution
    for name in PENALTY_TERMS:
        # min(1, ...) caps one catastrophic penalty at its own weight so it
        # cannot swamp the whole score into meaninglessness; the hard
        # disqualifiers, not the score, are what stop a catastrophic arm.
        normalized = min(1.0, abs(terms[name]["value"]) / _scale(name))
        contribution = -_weight(name) * normalized
        contributions[name] = contribution
        total += contribution
    return {
        "score": float(total),
        "scorable": True,
        "missing_required_terms": [],
        "contributions": {k: float(v) for k, v in contributions.items()},
        "detail": "weighted sum of {0} recovery and {1} penalty term(s)".format(
            len(RECOVERY_TERMS), len(PENALTY_TERMS)),
    }


# ---------------------------------------------------------------------------
# Hard safety disqualifiers -- three-valued, independent of any score
# ---------------------------------------------------------------------------

GATE_PASS = "pass"
GATE_FAIL = "fail"
GATE_NOT_APPLICABLE = "not_applicable"


def _gate(name, status, detail, value=None):
    return {"gate": name, "status": status, "detail": detail, "value": value}


def evaluate_disqualifiers(result, makeup_regions=None, delta_luma=None):
    """Evaluate every hard gate for one case/arm. Returns a JSON-safe verdict.

    Three-valued by design. ``not_applicable`` means the gate had no data, which
    is NOT a pass: a case whose gates are all ``not_applicable`` is reported as
    ``admissible=False`` and must be excluded from the margin test. Without that
    rule this function would certify an arm as safe on a patch that contained
    nothing to be unsafe about.
    """
    spec = SCORING_RULE["disqualifiers"]
    gates = []

    # 1. Any change at all in corrected/protected pixels.
    forbidden = result.get("forbidden_max_abs_delta")
    if not _finite(forbidden):
        gates.append(_gate(
            "forbidden_pixel_change", GATE_NOT_APPLICABLE,
            "forbidden_max_abs_delta is null: this case has NO corrected or protected "
            "pixels, so there is nothing to have changed. Absence of forbidden pixels "
            "is not evidence of safety."))
    else:
        limit = spec["forbidden_pixel_change"]["threshold"]["value"]
        passed = float(forbidden) <= limit
        gates.append(_gate(
            "forbidden_pixel_change", GATE_PASS if passed else GATE_FAIL,
            "forbidden_max_abs_delta={0:.6g}, must equal {1}".format(float(forbidden), limit),
            float(forbidden)))

    # 2. Post-gate leakage at the innermost ring (the support itself).
    inner = [r for r in (result.get("rings") or [])
             if r.get("outer_distance_px") == 0 and _finite(r.get("delta_rms"))]
    if not inner:
        gates.append(_gate(
            "innermost_ring_leakage", GATE_NOT_APPLICABLE,
            "No ring at outer_distance_px == 0: no corrected/protected support on "
            "this case, so no innermost ring exists to measure."))
    else:
        limit = spec["innermost_ring_leakage"]["threshold"]["value"]
        worst = max(abs(float(r["delta_rms"])) for r in inner)
        passed = worst <= limit
        gates.append(_gate(
            "innermost_ring_leakage", GATE_PASS if passed else GATE_FAIL,
            "max |delta_rms| at ring 0 over {0} support(s) = {1:.6g}, must equal "
            "{2}".format(len(inner), worst, limit), worst))

    # 3. Defect reintroduction beyond the frozen fraction.
    recovery = result.get("corrected_signed_contrast_recovery")
    if not _finite(recovery):
        gates.append(_gate(
            "defect_reintroduction", GATE_NOT_APPLICABLE,
            "corrected_signed_contrast_recovery is absent: no corrected mask on this "
            "case, so no repaired defect exists that could be reintroduced."))
    else:
        limit = spec["defect_reintroduction"]["threshold"]["value"]
        passed = float(recovery) <= limit
        gates.append(_gate(
            "defect_reintroduction", GATE_PASS if passed else GATE_FAIL,
            "corrected_signed_contrast_recovery={0:.6g}, must be <= {1}".format(
                float(recovery), limit), float(recovery)))

    # 4. Makeup-edge contamination beyond the frozen tolerance.
    contamination = term_makeup_edge_contamination(result, makeup_regions or [], delta_luma)
    if not contamination["available"]:
        gates.append(_gate(
            "makeup_edge_contamination", GATE_NOT_APPLICABLE,
            "Not measurable: " + contamination["detail"]))
    else:
        limit = spec["makeup_edge_contamination"]["threshold"]["value"]
        passed = contamination["value"] <= limit
        gates.append(_gate(
            "makeup_edge_contamination", GATE_PASS if passed else GATE_FAIL,
            "makeup-edge delta RMS={0:.6g}, must be <= {1}".format(
                contamination["value"], limit), contamination["value"]))

    failed = [g for g in gates if g["status"] == GATE_FAIL]
    evaluated = [g for g in gates if g["status"] != GATE_NOT_APPLICABLE]
    return {
        "gates": gates,
        "failed": [g["gate"] for g in failed],
        "evaluated_gate_count": len(evaluated),
        "not_applicable_gate_count": len(gates) - len(evaluated),
        "disqualified": bool(failed),
        # The rule that stops "no data" from reading as "safe".
        "admissible": bool(evaluated) and not failed,
        "detail": (
            "{0}/{1} gate(s) had data; {2} failed.".format(
                len(evaluated), len(gates), len(failed))
            + ("" if evaluated else
               " NO gate had any data, so this case demonstrates nothing about safety "
               "and is INADMISSIBLE to the margin test.")
        ),
    }


# ---------------------------------------------------------------------------
# Baseline-beating requirement
# ---------------------------------------------------------------------------


def compare_against_baselines(candidate_scores, baseline_scores, *, minimum_cases=None):
    """Per-case margin test of a candidate arm against A0 and A1.

    ``candidate_scores`` maps case_id -> score (float) or None. ``baseline_scores``
    maps arm -> {case_id: score or None}. A case counts only when the candidate
    AND both baselines produced a real score for it.

    Verdict rules, all frozen:

    * every admissible case must show a strictly positive margin against BOTH
      baselines (per-case, not median -- see the aggregation note in
      SCORING_RULE for why the claim's granularity decides this);
    * below ``minimum_cases`` admissible cases the function REFUSES a verdict
      rather than declaring a win on a handful of patches.
    """
    if minimum_cases is None:
        minimum_cases = SCORING_RULE["aggregation"]["minimum_admissible_cases"]["value"]
    margin_floor = SCORING_RULE["aggregation"]["minimum_margin"]["value"]

    missing_baselines = [arm for arm in BASELINE_ARMS if arm not in baseline_scores]
    if missing_baselines:
        return {
            "verdict": "not_computable",
            "reason": "missing baseline arm(s): " + ", ".join(missing_baselines),
            "comparable_cases": 0,
            "per_case": [],
            "minimum_cases_required": minimum_cases,
        }

    per_case = []
    for case_id in sorted(candidate_scores):
        candidate = candidate_scores[case_id]
        baselines = {arm: baseline_scores[arm].get(case_id) for arm in BASELINE_ARMS}
        if candidate is None or any(v is None for v in baselines.values()):
            per_case.append({
                "case_id": case_id,
                "comparable": False,
                "candidate_score": candidate,
                "baseline_scores": baselines,
                "detail": "candidate or a baseline produced no score for this case",
            })
            continue
        margins = {arm: float(candidate) - float(value) for arm, value in baselines.items()}
        beats = all(margin > margin_floor for margin in margins.values())
        per_case.append({
            "case_id": case_id,
            "comparable": True,
            "candidate_score": float(candidate),
            "baseline_scores": {k: float(v) for k, v in baselines.items()},
            "margins": margins,
            "beats_both": bool(beats),
            "detail": "margins vs {0}".format(", ".join(
                "{0}={1:+.6g}".format(arm, margins[arm]) for arm in BASELINE_ARMS)),
        })

    comparable = [entry for entry in per_case if entry["comparable"]]
    if not comparable:
        return {
            "verdict": "not_computable",
            "reason": (
                "No case produced a score for the candidate and both baselines. With "
                "zero comparable cases there is nothing to compare; this is NOT a loss "
                "and NOT a win."
            ),
            "comparable_cases": 0,
            "per_case": per_case,
            "minimum_cases_required": minimum_cases,
        }
    if len(comparable) < minimum_cases:
        return {
            "verdict": "not_computable",
            "reason": (
                "Only {0} comparable case(s); the frozen rule requires at least {1} "
                "before any verdict is emitted. Declaring a winner below this count "
                "would be a result read off a handful of patches.".format(
                    len(comparable), minimum_cases)
            ),
            "comparable_cases": len(comparable),
            "per_case": per_case,
            "minimum_cases_required": minimum_cases,
        }

    losers = [entry["case_id"] for entry in comparable if not entry["beats_both"]]
    return {
        "verdict": "beats_baselines" if not losers else "fails",
        "reason": (
            "every one of {0} comparable case(s) shows a positive margin against both "
            "{1}".format(len(comparable), " and ".join(BASELINE_ARMS))
            if not losers else
            "{0} of {1} comparable case(s) failed to beat both baselines: {2}".format(
                len(losers), len(comparable), ", ".join(losers))
        ),
        "comparable_cases": len(comparable),
        "failing_cases": losers,
        "per_case": per_case,
        "minimum_cases_required": minimum_cases,
        "aggregation": SCORING_RULE["aggregation"]["baseline_comparison"],
    }


# ---------------------------------------------------------------------------
# Eligibility / abstention contract
# ---------------------------------------------------------------------------


def highpass_std(image_bgr, mask=None, sigma=None):
    """Texture amplitude: std of luma minus its Gaussian low-pass.

    Deliberately matches the metric and sigma of this session's ff47-event
    visual measurement (commit 427ae49: "highpass std over a 2px Gaussian"),
    because the only anchors this project has (1.3-1.9 unresolvable skin,
    6.75 separable fine hair) are in THAT unit. ``retouch/image_analyzer.py``
    computes Laplacian std instead -- a different quantity whose numbers are
    NOT interchangeable with these anchors, which is why it is not reused here.

    Tone-invariance: this is an amplitude measure of deviation from the local
    mean, not an absolute intensity threshold, so it does not scale with the
    subject's skin tone (CLAUDE.md, Tone-Invariance & Fairness).

    ``mask`` restricts measurement to the allow support; measurement is on the
    NATIVE X canvas, never on the smoothed S.
    """
    if sigma is None:
        sigma = SCORING_RULE["eligibility"]["highpass_sigma_px"]["value"]
    array = np.asarray(image_bgr, dtype=np.float32)
    plane = exp.luma(array) if array.ndim == 3 else array
    highpass = plane - exp.gaussian(plane, sigma)
    if mask is not None:
        selected = highpass[np.asarray(mask) > 0]
        if selected.size == 0:
            return None
        return float(np.std(selected.astype(np.float64)))
    return float(np.std(highpass.astype(np.float64)))


def evaluate_eligibility(texture_std, annotation, *, case_id=None):
    """Is this face/patch eligible for micro-texture restoration at all?

    Encodes the owner's narrow production claim. BOTH gates must pass:

    1. **Sufficient native-resolution source texture** -- ``texture_std``
       (:func:`highpass_std` on native X inside the allow support) at or above
       the PROVISIONAL floor. Below it there is no texture information present
       to restore, so restoring is amplifying noise.
    2. **Validated supports** -- an accepted (``manual``/``owner_approved``)
       annotation carrying at least one populated ground-truth texture category.

    Failing either => ABSTAIN. Abstention means micro-texture restoration does
    not run and the pipeline keeps its existing output; it is the safe default
    and the behaviour today, never a relaxation of a safety check.
    """
    spec = SCORING_RULE["eligibility"]
    floor = spec["minimum_highpass_std"]["value"]
    reasons = []

    if not _finite(texture_std):
        texture_ok = False
        reasons.append(
            "source texture not measured (needs highpass_std on the native X canvas "
            "inside the allow support)")
    else:
        texture_ok = float(texture_std) >= floor
        if not texture_ok:
            reasons.append(
                "insufficient native source texture: highpass_std={0:.4g} < {1} "
                "(PROVISIONAL floor)".format(float(texture_std), floor))

    source = (annotation or {}).get("annotation_source")
    accepted = ("manual", "owner_approved")
    supports_ok = source in accepted
    if not supports_ok:
        reasons.append(
            "supports not validated: annotation_source={0!r} is not one of {1}".format(
                source, ", ".join(accepted)))

    categories = spec["required_ground_truth_categories"]["value"]
    populated = [c for c in categories if (annotation or {}).get(c)]
    labels_ok = bool(populated)
    if not labels_ok:
        reasons.append(
            "no populated ground-truth texture category ({0}); a face with no labelled "
            "recoverable texture cannot demonstrate recovery".format("/".join(categories)))

    eligible = bool(texture_ok and supports_ok and labels_ok)
    return {
        "case_id": case_id,
        "eligible": eligible,
        "action": "restore" if eligible else "abstain",
        "texture_std": None if not _finite(texture_std) else float(texture_std),
        "texture_floor": floor,
        "texture_sufficient": bool(texture_ok),
        "supports_validated": bool(supports_ok),
        "ground_truth_categories_populated": populated,
        "reasons": reasons,
        "abstain_action": spec["abstain_action"],
        "threshold_status": (
            "PROVISIONAL: calibrated from 10 patches against two non-overlapping anchor "
            "populations (this session's ff47-event visual measurement, commit 427ae49). "
            "NOT a validated production threshold; needs recalibration against a corpus "
            "containing resolvable pores."
        ),
    }


# ---------------------------------------------------------------------------
# Sequencing gate: the scoring rule must not change between stages
# ---------------------------------------------------------------------------


def check_scoring_sequencing(lock, dev_run_references, locked_test_run_reference=None):
    """The gate specific to a SCORING-based promotion.

    Subject separation is NOT reimplemented here: ``retouch.corpus_manifest``'s
    ``person_crosses_split`` and ``fa02_portrait_manifest.check_promotion_gates``
    already own it, and the readiness report already unions across files. This
    adds the one guarantee neither of them can express:

        the SAME scoring lock (and the same config lock) must govern the
        dev-stage candidate selection AND the eventual locked_test evaluation.

    A scoring rule edited after seeing dev results and before the locked-test run
    is the exact failure this whole contract exists to prevent, and it leaves no
    trace in any subject/split check.
    """
    violations = []
    scoring_sha = lock.get("scoring_sha256")
    if not scoring_sha:
        violations.append({"code": "lock_missing_scoring_sha256"})
    if lock.get("scoring_sha256") != canonical_scoring_hash(lock.get("scoring_rule") or {}):
        violations.append({"code": "scoring_lock_hash_mismatch",
                           "detail": "scoring_sha256 does not match its own scoring_rule"})

    references = list(dev_run_references or [])
    if not references:
        violations.append({
            "code": "no_dev_run_referenced",
            "detail": "A locked-test evaluation must name the dev runs whose candidate it "
                      "is confirming; without them there is no selection to confirm.",
        })

    config_hashes = sorted({r.get("config_sha256") for r in references if r.get("config_sha256")})
    if len(config_hashes) > 1:
        violations.append({"code": "dev_runs_used_multiple_configs", "hashes": config_hashes})

    scoring_hashes = sorted({r.get("scoring_sha256") for r in references
                             if r.get("scoring_sha256")})
    if any(value != scoring_sha for value in scoring_hashes):
        violations.append({
            "code": "dev_run_scored_under_a_different_scoring_lock",
            "detail": "A dev run was scored under a scoring rule that is not this lock.",
        })

    if locked_test_run_reference is not None:
        if locked_test_run_reference.get("scoring_sha256") != scoring_sha:
            violations.append({
                "code": "locked_test_scoring_lock_differs_from_dev",
                "detail": "The scoring rule changed between dev selection and the "
                          "locked_test evaluation. The locked-test result is void.",
            })
        if config_hashes and locked_test_run_reference.get("config_sha256") not in config_hashes:
            violations.append({
                "code": "locked_test_config_differs_from_dev",
                "detail": "The arm config changed between dev selection and locked_test.",
            })
        frozen_at = lock.get("frozen_at_utc")
        run_at = locked_test_run_reference.get("run_at_utc")
        if frozen_at and run_at and run_at < frozen_at:
            violations.append({
                "code": "locked_test_run_predates_scoring_lock",
                "detail": "The locked_test run happened BEFORE the scoring rule was "
                          "frozen, so the rule cannot be said to have been pre-registered.",
            })

    return {
        "valid": not violations,
        "violations": violations,
        "scoring_sha256": scoring_sha,
        "dev_run_count": len(references),
        "note": (
            "Ordering evidence, not proof: a hash plus a timestamp shows the rule is "
            "unchanged and that the lock file existed, but only the owner's attestation "
            "can establish that no result informed the rule before it was written."
        ),
    }


# ---------------------------------------------------------------------------
# Freezing the scoring lock
# ---------------------------------------------------------------------------


def canonical_scoring_hash(rule):
    """Stable hash of the scoring rule. Same convention as exp.canonical_hash."""
    return hashlib.sha256(
        json.dumps(rule, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def build_scoring_lock(*, declaration="a_priori_pre_registration", references=None):
    """Assemble the lock payload (without writing it)."""
    rule = json.loads(json.dumps(SCORING_RULE, sort_keys=True))
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "lock_type": "fa02_scoring_contract",
        "scoring_rule": rule,
        "scoring_sha256": canonical_scoring_hash(rule),
        "contract_properties": CONTRACT_PROPERTIES,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": declaration,
        "development_references": list(references or []),
        "module_sha256": exp.digest(Path(__file__).resolve()),
        "baseline_arms": list(BASELINE_ARMS),
        "arms": list(exp.ARMS),
        "note": (
            "A timestamp and a hash audit the RULE; they do not prove no prior exposure "
            "to results. Ordering evidence comes from this lock being referenced by every "
            "scored run (see check_scoring_sequencing). Every threshold in this rule is "
            "PROVISIONAL and none has been validated -- see each constant's provenance."
        ),
    }


def freeze_scoring(output, *, declaration="a_priori_pre_registration", references=None):
    """Write the scoring lock, refusing to overwrite -- mirrors exp.freeze_config."""
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing scoring lock")
    lock = build_scoring_lock(declaration=declaration, references=references)
    output.parent.mkdir(parents=True, exist_ok=True)
    exp.write_json(output, lock)
    return lock


def load_scoring_lock(path):
    """Read a scoring lock and verify it hashes to what it claims."""
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != SCORING_SCHEMA_VERSION:
        raise ValueError("Expected FA-02 scoring lock schema_version=1")
    if lock.get("lock_type") != "fa02_scoring_contract":
        raise ValueError("Not an FA-02 scoring contract lock")
    if canonical_scoring_hash(lock.get("scoring_rule") or {}) != lock.get("scoring_sha256"):
        raise ValueError("Scoring lock hash mismatch: the rule was edited after freezing")
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze the scoring rule BEFORE scoring anything")
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--selection", default="a_priori_pre_registration",
                        choices=("a_priori_pre_registration", "development_selected"))
    freeze.add_argument("--development-reference", action="append", default=[])
    show = commands.add_parser("show", help="Print the current rule and its hash")
    show.add_argument("--json", type=Path)
    verify = commands.add_parser("verify", help="Verify an existing scoring lock")
    verify.add_argument("lock", type=Path)
    args = parser.parse_args(argv)

    if args.command == "freeze":
        lock = freeze_scoring(args.output, declaration=args.selection,
                              references=args.development_reference)
        print("scoring_sha256: " + lock["scoring_sha256"])
        print("frozen_at_utc:  " + lock["frozen_at_utc"])
        print("wrote " + str(args.output))
        return 0
    if args.command == "show":
        lock = build_scoring_lock()
        print("scoring_sha256: " + lock["scoring_sha256"])
        if args.json:
            exp.write_json(args.json, lock)
            print("wrote " + str(args.json))
        return 0
    lock = load_scoring_lock(args.lock)
    print("scoring_sha256: " + lock["scoring_sha256"])
    print("frozen_at_utc:  " + lock.get("frozen_at_utc", "?"))
    print("hash verified against the rule it contains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

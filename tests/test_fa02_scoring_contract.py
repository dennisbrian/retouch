"""FA-02 frozen scoring contract: term availability, hard gates, margin test.

Every fixture in this file is SYNTHETIC and clearly marked. The scoring rule's
weights and thresholds were frozen before any of these ran, and nothing here
may be used to adjust them -- these tests check the scorer's MECHANICS, not
whether any arm is good.

The one exception is
:func:`test_real_pilot_directories_report_blocked_not_a_baseline_win`, which
runs the report against the ten REAL committed pilot directories. Its job is to
assert the report stays HONEST on real data: that it reports blocked/inadmissible
rather than manufacturing a baseline-beat verdict out of absent inputs.

The load-bearing assertions -- the ones that would otherwise let a wrong answer
through silently rather than fail -- are:

* a term with a missing input is ``available=False`` with ``value is None``,
  never ``value == 0`` (a zeroed penalty term is a PERFECT score);
* a case missing any required term scores ``None``, and weights are NOT
  renormalized around it;
* a hard gate with no data is ``not_applicable``, never ``pass``, and a case
  whose gates are all not_applicable is INADMISSIBLE;
* the margin test refuses a verdict below the frozen minimum case count instead
  of declaring a win on a handful of patches.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/qa"))

import fa02_scoring_contract as contract  # noqa: E402
import fa02_scoring_readiness_report as scoring_report  # noqa: E402
import fa02_texture_representation_experiment as exp  # noqa: E402

PILOT_ROOT = ROOT / "test_output"


# ---------------------------------------------------------------------------
# Synthetic fixtures. None of this is real portrait evidence.
# ---------------------------------------------------------------------------


def _bare_result(**overrides):
    """A minimal synthetic summary.json result entry with NO optional inputs.

    Mirrors the shape ``exp.score()`` emits for a case that has no pores, no
    profiles, no corrected/protected mask and no paired-clean control -- which
    is exactly the shape all ten committed real patches have.
    """
    result = {
        "case_id": "synthetic_case",
        "arm": "A2_dog",
        "eligible_pixels": 1000,
        "weighted_coverage": 0.5,
        "delta_rms": 0.4,
        "delta_luma_mean": 0.01,
        "broad_luma_rms": 0.1,
        "chroma_delta_rms": 0.05,
        "forbidden_max_abs_delta": None,
        "clipped_pixel_count": 0,
        "rois": [],
        "features": [],
        "profiles": [],
        "rings": [],
        "hair_trace_samples": [],
    }
    result.update(overrides)
    return result


def _full_result(**overrides):
    """A synthetic result entry where EVERY scoring input is present."""
    result = _bare_result(
        forbidden_max_abs_delta=0.0,
        features=[
            {"id": "p1", "kind": "pore_contrast", "signed_recovery": 0.40},
            {"id": "p2", "kind": "pore_contrast", "signed_recovery": 0.60},
        ],
        hair_trace_samples=[
            {"trace_id": "t1", "observable_sections": 4,
             "retained_sections_at_half_source_contrast": 3},
        ],
        rings=[
            {"support": "protected", "outer_distance_px": 0, "delta_rms": 0.0,
             "delta_mean": 0.0},
            {"support": "protected", "outer_distance_px": 4, "delta_rms": 0.2,
             "delta_mean": 0.1},
        ],
        corrected_signed_contrast_recovery=0.05,
        increment_projection_onto_nuisance_confoundable=0.10,
    )
    result.update(overrides)
    return result


def _makeup_regions():
    return [{"id": "edge_a", "xywh": [2, 2, 6, 6], "tags": ["makeup_edge"]}]


# ---------------------------------------------------------------------------
# Each term: unavailable when its input is missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term_name,call",
    [
        ("pore_recovery", lambda r: contract.term_pore_recovery(r)),
        ("fine_hair_recovery", lambda r: contract.term_fine_hair_recovery(r)),
        ("noise_amplification", lambda r: contract.term_noise_amplification(r)),
        ("defect_reintroduction", lambda r: contract.term_defect_reintroduction(r)),
        ("halo", lambda r: contract.term_halo(r)),
    ],
)
def test_term_is_unavailable_when_its_input_is_missing(term_name, call):
    """A missing input must NEVER be coerced to 0.

    This is the assertion the whole contract rests on: a penalty term silently
    reading 0 is indistinguishable from a perfect score, and a recovery term
    reading 0 lets the case still be scored on safety terms alone.
    """
    result = call(_bare_result())
    assert result["term"] == term_name
    assert result["available"] is False
    assert result["value"] is None, "a missing input must not become a number"
    assert result["missing_input"], "an unavailable term must name what is missing"


def test_makeup_edge_term_unavailable_without_regions_and_without_arrays():
    """Two distinct missing inputs, each named separately."""
    no_regions = contract.term_makeup_edge_contamination(_bare_result(), [], None)
    assert no_regions["available"] is False
    assert no_regions["value"] is None
    assert "makeup_edge_regions" in no_regions["missing_input"]

    # Labels exist, but the gitignored float arrays do not -- the real
    # situation on a fresh clone for the three labelled ff47 patches.
    no_arrays = contract.term_makeup_edge_contamination(
        _bare_result(), _makeup_regions(), None)
    assert no_arrays["available"] is False
    assert no_arrays["value"] is None
    assert "signed_arrays.npz" in no_arrays["missing_input"]


def test_broad_tone_term_is_available_on_the_bare_result():
    """The one term computable everywhere -- and it can only ever subtract."""
    result = contract.term_broad_tone_chroma_leakage(_bare_result())
    assert result["available"] is True
    assert result["value"] == pytest.approx(0.15)


def test_broad_tone_term_unavailable_when_the_run_had_no_eligible_pixel():
    result = contract.term_broad_tone_chroma_leakage(
        _bare_result(broad_luma_rms=None, chroma_delta_rms=None))
    assert result["available"] is False
    assert result["value"] is None


# ---------------------------------------------------------------------------
# Each term: a real number when synthetic input IS present
# ---------------------------------------------------------------------------


def test_every_term_returns_a_number_when_all_inputs_are_present():
    delta = np.zeros((16, 16), dtype=np.float32)
    delta[2:8, 2:8] = 0.5
    terms = contract.compute_terms(_full_result(), _makeup_regions(), delta)
    for name, term in terms.items():
        assert term["available"] is True, "{0} should be available: {1}".format(
            name, term.get("detail"))
        assert isinstance(term["value"], float)
        assert np.isfinite(term["value"])

    assert terms["pore_recovery"]["value"] == pytest.approx(0.5)      # median(0.4, 0.6)
    assert terms["fine_hair_recovery"]["value"] == pytest.approx(0.75)  # 3/4
    assert terms["noise_amplification"]["value"] == pytest.approx(0.10)
    assert terms["defect_reintroduction"]["value"] == pytest.approx(0.05)
    assert terms["halo"]["value"] == pytest.approx(0.1)   # ring 0 excluded
    assert terms["makeup_edge_contamination"]["value"] == pytest.approx(0.5)


def test_pore_recovery_clips_and_takes_the_median():
    result = _bare_result(features=[
        {"id": "a", "kind": "pore_contrast", "signed_recovery": 0.2},
        {"id": "b", "kind": "pore_contrast", "signed_recovery": 0.4},
        # An absurd overshoot must not drag the median through the roof.
        {"id": "c", "kind": "pore_contrast", "signed_recovery": 99.0},
    ])
    term = contract.term_pore_recovery(result)
    assert term["available"] is True
    assert term["value"] == pytest.approx(0.4)


def test_pore_features_present_but_all_null_recovery_is_unavailable():
    """Features with an undefined denominator are not evidence of recovery."""
    result = _bare_result(features=[
        {"id": "a", "kind": "pore_contrast", "signed_recovery": None},
    ])
    term = contract.term_pore_recovery(result)
    assert term["available"] is False
    assert term["value"] is None


def test_hair_traces_with_no_observable_section_are_unavailable():
    result = _bare_result(hair_trace_samples=[
        {"trace_id": "t", "observable_sections": 0,
         "retained_sections_at_half_source_contrast": 0},
    ])
    term = contract.term_fine_hair_recovery(result)
    assert term["available"] is False
    assert term["value"] is None


def test_defect_reintroduction_clamps_negative_recovery_to_zero():
    """Pushing a defect further down is not reintroduction."""
    term = contract.term_defect_reintroduction(
        _bare_result(corrected_signed_contrast_recovery=-0.4))
    assert term["available"] is True
    assert term["value"] == 0.0


def test_halo_excludes_the_innermost_ring():
    """Ring 0 IS the support: it is a leakage gate, not a halo measurement."""
    result = _bare_result(rings=[
        {"support": "protected", "outer_distance_px": 0, "delta_mean": 9.0, "delta_rms": 9.0},
    ])
    assert contract.term_halo(result)["available"] is False


# ---------------------------------------------------------------------------
# Aggregation: refuses rather than renormalizing
# ---------------------------------------------------------------------------


def test_case_with_a_missing_required_term_is_not_scored():
    aggregate = contract.aggregate_case_score(
        contract.compute_terms(_bare_result(), [], None))
    assert aggregate["scorable"] is False
    assert aggregate["score"] is None
    assert "pore_recovery" in aggregate["missing_required_terms"]
    assert aggregate["contributions"] == {}


def test_weights_are_not_renormalized_around_a_missing_term():
    """The exact failure the contract exists to prevent.

    A case with strong safety numbers and NO texture evidence must not produce
    a score at all. If weights were renormalized this would return a high
    number built entirely from penalty terms that happen to be near zero.
    """
    safe_but_blind = _bare_result(
        forbidden_max_abs_delta=0.0, broad_luma_rms=0.0, chroma_delta_rms=0.0)
    aggregate = contract.aggregate_case_score(
        contract.compute_terms(safe_but_blind, [], None))
    assert aggregate["score"] is None
    assert set(aggregate["missing_required_terms"]) >= {
        "pore_recovery", "fine_hair_recovery", "noise_amplification",
        "defect_reintroduction", "makeup_edge_contamination", "halo"}


def test_full_case_scores_and_a0_is_structurally_zero():
    """CONTRACT_PROPERTIES: A0 scores exactly 0, so beating it needs recovery."""
    delta = np.zeros((16, 16), dtype=np.float32)
    aggregate = contract.aggregate_case_score(
        contract.compute_terms(_full_result(), _makeup_regions(), delta))
    assert aggregate["scorable"] is True
    assert isinstance(aggregate["score"], float)

    # A0_disabled emits an all-zero delta: every term is 0, so the score is 0.
    zero_terms = contract.compute_terms(
        _full_result(
            features=[{"id": "p", "kind": "pore_contrast", "signed_recovery": 0.0}],
            hair_trace_samples=[{"trace_id": "t", "observable_sections": 2,
                                 "retained_sections_at_half_source_contrast": 0}],
            increment_projection_onto_nuisance_confoundable=0.0,
            corrected_signed_contrast_recovery=0.0,
            broad_luma_rms=0.0, chroma_delta_rms=0.0,
            rings=[{"support": "protected", "outer_distance_px": 4,
                    "delta_mean": 0.0, "delta_rms": 0.0}]),
        _makeup_regions(), np.zeros((16, 16), dtype=np.float32))
    assert contract.aggregate_case_score(zero_terms)["score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hard disqualifiers: three-valued, and "no data" is not a pass
# ---------------------------------------------------------------------------


def test_gates_with_no_data_are_not_applicable_and_the_case_is_inadmissible():
    """The vacuous-pass trap. Absence of forbidden pixels is not safety."""
    verdict = contract.evaluate_disqualifiers(_bare_result(), [], None)
    statuses = {g["gate"]: g["status"] for g in verdict["gates"]}
    assert set(statuses.values()) == {contract.GATE_NOT_APPLICABLE}
    assert contract.GATE_PASS not in statuses.values()
    assert verdict["disqualified"] is False
    assert verdict["admissible"] is False, "no safety data must never read as safe"
    assert verdict["evaluated_gate_count"] == 0


def test_clean_synthetic_data_passes_every_gate_and_is_admissible():
    delta = np.zeros((16, 16), dtype=np.float32)
    verdict = contract.evaluate_disqualifiers(_full_result(), _makeup_regions(), delta)
    assert verdict["failed"] == []
    assert verdict["disqualified"] is False
    assert verdict["admissible"] is True
    assert verdict["evaluated_gate_count"] == 4


def test_forbidden_pixel_change_fails():
    verdict = contract.evaluate_disqualifiers(
        _full_result(forbidden_max_abs_delta=0.001), [], None)
    assert "forbidden_pixel_change" in verdict["failed"]
    assert verdict["disqualified"] is True
    assert verdict["admissible"] is False


def test_innermost_ring_leakage_fails():
    verdict = contract.evaluate_disqualifiers(
        _full_result(rings=[
            {"support": "corrected", "outer_distance_px": 0, "delta_rms": 0.3,
             "delta_mean": 0.3}]), [], None)
    assert "innermost_ring_leakage" in verdict["failed"]


def test_defect_reintroduction_beyond_the_frozen_fraction_fails():
    limit = contract.SCORING_RULE["disqualifiers"]["defect_reintroduction"]["threshold"]["value"]
    verdict = contract.evaluate_disqualifiers(
        _full_result(corrected_signed_contrast_recovery=limit + 0.01), [], None)
    assert "defect_reintroduction" in verdict["failed"]
    # Just under the line must still pass, so the threshold is a real boundary.
    ok = contract.evaluate_disqualifiers(
        _full_result(corrected_signed_contrast_recovery=limit - 0.01), [], None)
    assert "defect_reintroduction" not in ok["failed"]


def test_makeup_edge_contamination_beyond_tolerance_fails():
    limit = contract.SCORING_RULE["disqualifiers"]["makeup_edge_contamination"]["threshold"]["value"]
    delta = np.zeros((16, 16), dtype=np.float32)
    delta[2:8, 2:8] = limit + 1.0
    verdict = contract.evaluate_disqualifiers(_full_result(), _makeup_regions(), delta)
    assert "makeup_edge_contamination" in verdict["failed"]


def test_protected_recovery_is_an_observation_not_a_gate():
    """Deliberate design choice, asserted so it cannot drift into a gate."""
    verdict = contract.evaluate_disqualifiers(
        _full_result(protected_signed_contrast_recovery=0.99), [], None)
    assert "protected_signed_contrast_recovery" not in [g["gate"] for g in verdict["gates"]]
    assert "protected_signed_contrast_recovery" in contract.SCORING_RULE[
        "explicitly_not_disqualifiers"]


# ---------------------------------------------------------------------------
# Baseline-beat: per-case aggregation, with win/lose/tie fixtures
# ---------------------------------------------------------------------------


def _scores(values):
    return {"case_{0}".format(i): v for i, v in enumerate(values)}


def test_baseline_beat_requires_every_case_to_win():
    """Per-case aggregation: one losing case sinks the whole verdict."""
    n = contract.SCORING_RULE["aggregation"]["minimum_admissible_cases"]["value"]
    candidate = _scores([0.5] * n)
    baselines = {"A0_disabled": _scores([0.0] * n), "A1_raw": _scores([0.1] * n)}
    win = contract.compare_against_baselines(candidate, baselines)
    assert win["verdict"] == "beats_baselines"
    assert win["comparable_cases"] == n
    assert win["aggregation"] == "per_case"

    # Exactly one case loses to A1 -- a MEDIAN rule would still call this a win.
    losing = dict(candidate)
    losing["case_0"] = 0.05
    lose = contract.compare_against_baselines(losing, baselines)
    assert lose["verdict"] == "fails"
    assert lose["failing_cases"] == ["case_0"]


def test_baseline_beat_treats_a_tie_as_a_loss():
    n = contract.SCORING_RULE["aggregation"]["minimum_admissible_cases"]["value"]
    candidate = _scores([0.2] * n)
    baselines = {"A0_disabled": _scores([0.0] * n), "A1_raw": _scores([0.2] * n)}
    result = contract.compare_against_baselines(candidate, baselines)
    assert result["verdict"] == "fails", "a tie is not a win"


def test_baseline_beat_refuses_a_verdict_below_the_minimum_case_count():
    """No declaring a production candidate off a handful of patches."""
    n = contract.SCORING_RULE["aggregation"]["minimum_admissible_cases"]["value"]
    candidate = _scores([0.9] * (n - 1))
    baselines = {"A0_disabled": _scores([0.0] * (n - 1)),
                 "A1_raw": _scores([0.1] * (n - 1))}
    result = contract.compare_against_baselines(candidate, baselines)
    assert result["verdict"] == "not_computable"
    assert result["comparable_cases"] == n - 1


def test_baseline_beat_is_not_computable_when_nothing_scored():
    """All-null scores must NOT read as a loss or a win."""
    candidate = {"case_0": None, "case_1": None}
    baselines = {"A0_disabled": {"case_0": None, "case_1": None},
                 "A1_raw": {"case_0": None, "case_1": None}}
    result = contract.compare_against_baselines(candidate, baselines)
    assert result["verdict"] == "not_computable"
    assert result["comparable_cases"] == 0


def test_baseline_beat_needs_both_baselines():
    result = contract.compare_against_baselines(
        _scores([0.5]), {"A0_disabled": _scores([0.0])})
    assert result["verdict"] == "not_computable"
    assert "A1_raw" in result["reason"]


# ---------------------------------------------------------------------------
# Eligibility / abstention contract
# ---------------------------------------------------------------------------


def _accepted_annotation(**overrides):
    annotation = {"annotation_source": "owner_approved",
                  "fine_hair_regions": [{"id": "h", "xywh": [0, 0, 4, 4], "tags": []}]}
    annotation.update(overrides)
    return annotation


def test_eligibility_requires_texture_supports_and_labels():
    floor = contract.SCORING_RULE["eligibility"]["minimum_highpass_std"]["value"]
    ok = contract.evaluate_eligibility(floor + 1.0, _accepted_annotation())
    assert ok["eligible"] is True
    assert ok["action"] == "restore"

    # 1. Below the texture floor -> abstain (the ff47 skin measurement range).
    low = contract.evaluate_eligibility(1.5, _accepted_annotation())
    assert low["eligible"] is False and low["action"] == "abstain"
    assert low["texture_sufficient"] is False

    # 2. Unvalidated supports -> abstain even with plenty of texture.
    unvalidated = contract.evaluate_eligibility(
        floor + 5.0, _accepted_annotation(annotation_source="detector_derived"))
    assert unvalidated["eligible"] is False
    assert unvalidated["supports_validated"] is False

    # 3. No ground-truth label -> abstain.
    unlabelled = contract.evaluate_eligibility(
        floor + 5.0, {"annotation_source": "owner_approved"})
    assert unlabelled["eligible"] is False
    assert unlabelled["ground_truth_categories_populated"] == []


def test_eligibility_abstains_when_texture_was_never_measured():
    result = contract.evaluate_eligibility(None, _accepted_annotation())
    assert result["eligible"] is False
    assert result["action"] == "abstain"


def test_eligibility_threshold_is_declared_provisional():
    """Matches the eye-gate precedent: honesty from day one, not after a regression."""
    result = contract.evaluate_eligibility(10.0, _accepted_annotation())
    assert "PROVISIONAL" in result["threshold_status"]


def test_highpass_std_matches_the_ff47_metric_and_separates_flat_from_textured():
    """Sigma must match the anchor's metric; Laplacian variance is NOT this."""
    assert contract.SCORING_RULE["eligibility"]["highpass_sigma_px"]["value"] == 2.0
    assert contract.SCORING_RULE["eligibility"]["highpass_sigma_px"]["provenance"] == "MEASURED"

    rng = np.random.default_rng(1701)
    flat = np.full((64, 64, 3), 128.0, dtype=np.float32)
    textured = flat + rng.normal(0, 12, size=flat.shape).astype(np.float32)
    assert contract.highpass_std(flat) == pytest.approx(0.0, abs=1e-4)
    assert contract.highpass_std(textured) > contract.highpass_std(flat)

    # A mask restricts measurement to the allow support.
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:32] = 1.0
    assert contract.highpass_std(textured, mask) > 0.0
    assert contract.highpass_std(textured, np.zeros((64, 64), np.float32)) is None


# ---------------------------------------------------------------------------
# Scoring lock: stable, reproducible, refuses to overwrite
# ---------------------------------------------------------------------------


def test_scoring_hash_is_stable_and_reproducible():
    first = contract.canonical_scoring_hash(contract.SCORING_RULE)
    second = contract.canonical_scoring_hash(
        json.loads(json.dumps(contract.SCORING_RULE, sort_keys=True)))
    assert first == second
    assert len(first) == 64


def test_scoring_hash_changes_when_any_weight_changes():
    """Freeze proof: an edited rule cannot keep its hash."""
    mutated = json.loads(json.dumps(contract.SCORING_RULE))
    mutated["terms"]["pore_recovery"]["weight"]["value"] = 0.34
    assert contract.canonical_scoring_hash(mutated) != contract.canonical_scoring_hash(
        contract.SCORING_RULE)


def test_candidate_set_narrowing_did_not_change_the_scoring_rule():
    """The load-bearing proof that 2026-09-06 was a SCOPE change, not a re-tune.

    The candidate-set narrowing must leave every weight, threshold, term,
    disqualifier and eligibility rule untouched. That is mechanically checkable:
    ``scoring_sha256`` is a hash of ``scoring_rule`` alone, and the arm lists are
    recorded OUTSIDE it, so the hash must still equal the value frozen at
    652ee53. If this assertion ever fails, someone tuned something while claiming
    to narrow scope.
    """
    frozen_at_652ee53 = "0bbca63e3425bb95d4937aa6af11d1498ed4cdaaed2beb9c6c66b6ba18dea490"
    assert contract.canonical_scoring_hash(contract.SCORING_RULE) == frozen_at_652ee53
    lock = contract.build_scoring_lock()
    assert lock["scoring_sha256"] == frozen_at_652ee53
    # The arm scope lives at top level, never inside the hashed rule.
    for key in ("candidate_arms_this_round", "evaluated_candidate_arms",
                "reserved_arms_pending_failure"):
        assert key in lock
        assert key not in lock["scoring_rule"]


def test_frozen_candidate_set_is_consistent_with_the_harness():
    """The four-arm candidate set, its baselines and its reserves must partition ARMS."""
    assert contract.FROZEN_CANDIDATE_SET == (
        "A0_disabled", "A1_raw", "A2_dog", "A3_multiscale")
    # A candidate set naming an arm the harness lacks would be unrunnable.
    assert set(contract.FROZEN_CANDIDATE_SET).issubset(set(exp.ARMS))
    # Baselines are inside the frozen set but are not themselves candidates:
    # a candidate must BEAT them, so testing them would compare A0 to A0.
    assert set(contract.BASELINE_ARMS).issubset(set(contract.FROZEN_CANDIDATE_SET))
    assert contract.EVALUATED_CANDIDATE_ARMS == ("A2_dog", "A3_multiscale")
    # Reserved arms are the exact complement -- derived, so no arm can be lost.
    assert contract.RESERVED_ARMS_PENDING_FAILURE == (
        "A4_orientation", "A5_retouch_frequency")
    assert (set(contract.FROZEN_CANDIDATE_SET) | set(contract.RESERVED_ARMS_PENDING_FAILURE)
            == set(exp.ARMS))
    assert not (set(contract.FROZEN_CANDIDATE_SET)
                & set(contract.RESERVED_ARMS_PENDING_FAILURE))
    # Reserved means excluded from a DECISION, not deleted: both arms are still
    # implemented and still run in the harness.
    for arm in contract.RESERVED_ARMS_PENDING_FAILURE:
        assert arm in exp.ARMS


def test_reserved_arms_are_reported_not_silently_dropped():
    """A reserved arm gets a distinct verdict, never omission or a fake result."""
    results = scoring_report.run_baseline_tests([])
    for arm in contract.RESERVED_ARMS_PENDING_FAILURE:
        assert arm in results, "a reserved arm must still appear in the report"
        assert results[arm]["verdict"] == "reserved_not_evaluated"
        assert results[arm]["evaluated_this_round"] is False
    for arm in contract.EVALUATED_CANDIDATE_ARMS:
        # Evaluated arms with no data must still say not_computable, never a win.
        assert results[arm]["verdict"] == "not_computable"


def test_freeze_writes_a_lock_and_refuses_to_overwrite(tmp_path):
    target = tmp_path / "scoring_lock.json"
    lock = contract.freeze_scoring(target)
    assert target.is_file()
    assert lock["scoring_sha256"] == contract.canonical_scoring_hash(lock["scoring_rule"])
    assert lock["frozen_at_utc"].endswith("Z")
    assert lock["baseline_arms"] == list(contract.BASELINE_ARMS)
    # The harness's full arm set is recorded unchanged, alongside the narrower
    # set this round actually decides on.
    assert lock["arms"] == list(exp.ARMS)
    assert lock["candidate_arms_this_round"] == list(contract.FROZEN_CANDIDATE_SET)
    assert lock["evaluated_candidate_arms"] == list(contract.EVALUATED_CANDIDATE_ARMS)
    assert lock["reserved_arms_pending_failure"] == list(
        contract.RESERVED_ARMS_PENDING_FAILURE)

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        contract.freeze_scoring(target)

    reloaded = contract.load_scoring_lock(target)
    assert reloaded["scoring_sha256"] == lock["scoring_sha256"]


def test_load_scoring_lock_detects_a_post_freeze_edit(tmp_path):
    target = tmp_path / "scoring_lock.json"
    contract.freeze_scoring(target)
    tampered = json.loads(target.read_text())
    tampered["scoring_rule"]["terms"]["halo"]["weight"]["value"] = 0.99
    target.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash mismatch"):
        contract.load_scoring_lock(target)


def test_every_numeric_constant_declares_its_provenance():
    """No unexplained magic number may enter the frozen rule."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if set(node) >= {"value", "provenance", "note"}:
                found.append(node)
                assert node["provenance"] in contract.PROVENANCE_KINDS
                assert node["note"].strip(), "a constant needs a provenance note"
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(contract.SCORING_RULE)
    assert len(found) >= 15
    # The eligibility floor in particular must never quietly become "validated".
    floor = contract.SCORING_RULE["eligibility"]["minimum_highpass_std"]
    assert floor["provenance"] == "PLACEHOLDER"
    assert "PROVISIONAL" in floor["note"]


def test_weights_sum_to_one():
    total = sum(contract.SCORING_RULE["terms"][t]["weight"]["value"]
                for t in scoring_report.ALL_TERMS)
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sequencing gate
# ---------------------------------------------------------------------------


def test_sequencing_gate_accepts_a_consistent_dev_then_locked_test_sequence(tmp_path):
    lock = contract.freeze_scoring(tmp_path / "lock.json")
    dev = [{"config_sha256": "cfg", "scoring_sha256": lock["scoring_sha256"]}]
    report = contract.check_scoring_sequencing(
        lock, dev, {"config_sha256": "cfg", "scoring_sha256": lock["scoring_sha256"],
                    "run_at_utc": "2099-01-01T00:00:00Z"})
    assert report["valid"] is True


def test_sequencing_gate_catches_a_scoring_rule_changed_between_stages(tmp_path):
    """The failure no subject/split check can ever see."""
    lock = contract.freeze_scoring(tmp_path / "lock.json")
    dev = [{"config_sha256": "cfg", "scoring_sha256": lock["scoring_sha256"]}]
    report = contract.check_scoring_sequencing(
        lock, dev, {"config_sha256": "cfg", "scoring_sha256": "a_different_rule",
                    "run_at_utc": "2099-01-01T00:00:00Z"})
    assert report["valid"] is False
    assert "locked_test_scoring_lock_differs_from_dev" in [
        v["code"] for v in report["violations"]]


def test_sequencing_gate_catches_a_locked_test_run_predating_the_lock(tmp_path):
    lock = contract.freeze_scoring(tmp_path / "lock.json")
    dev = [{"config_sha256": "cfg", "scoring_sha256": lock["scoring_sha256"]}]
    report = contract.check_scoring_sequencing(
        lock, dev, {"config_sha256": "cfg", "scoring_sha256": lock["scoring_sha256"],
                    "run_at_utc": "1999-01-01T00:00:00Z"})
    assert "locked_test_run_predates_scoring_lock" in [
        v["code"] for v in report["violations"]]


def test_sequencing_gate_requires_a_referenced_dev_run(tmp_path):
    lock = contract.freeze_scoring(tmp_path / "lock.json")
    report = contract.check_scoring_sequencing(lock, [])
    assert report["valid"] is False
    assert "no_dev_run_referenced" in [v["code"] for v in report["violations"]]


# ---------------------------------------------------------------------------
# Integration against the TEN REAL committed pilot directories
# ---------------------------------------------------------------------------


def _real_pilot_directories():
    return sorted(PILOT_ROOT.glob("fa02_pilot_*"))


def test_real_pilot_directories_report_blocked_not_a_baseline_win(tmp_path):
    """The honesty test. Real committed metadata, asserted to stay honest.

    This must NOT claim a baseline-beat result: the pore/hair inputs that the
    scoring rule requires are absent from every one of the ten patches, so the
    only correct outputs are "blocked" terms and a "not_computable" verdict.
    """
    directories = _real_pilot_directories()
    if len(directories) < 10:
        pytest.skip("real FA-02 pilot directories are not present")

    lock_path = tmp_path / "scoring_lock.json"
    contract.freeze_scoring(lock_path)
    report = scoring_report.build_report(lock_path, directories)

    # The structural gap, asserted against real data rather than assumed.
    assert report["totals"]["pore_scoring_inputs"] == 0
    assert report["totals"]["profile_scoring_inputs"] == 0
    assert report["totals"]["case_count"] == 10

    # The two recovery terms -- the only ones that could ever beat A0 -- are
    # blocked on every case/arm.
    for term in ("pore_recovery", "fine_hair_recovery"):
        record = report["terms"][term]
        assert record["status"] == "blocked"
        assert record["computable_case_arm_count"] == 0
        assert record["blocked_case_arm_count"] > 0

    # Nothing is scorable, so no arm may be declared a winner. The candidate-set
    # narrowing (2026-09-06) means the two arms this round DECIDES on must report
    # not_computable, while the reserved arms report that they were not asked --
    # a distinct verdict, so "excluded by scope" can never be misread as "tested
    # and inconclusive". Neither verdict may ever be a win.
    assert report["totals"]["scorable_case_arms"] == 0
    assert report["totals"]["evaluated_candidate_arms"] == ["A2_dog", "A3_multiscale"]
    assert report["totals"]["reserved_arms_pending_failure"] == [
        "A4_orientation", "A5_retouch_frequency"]
    # The harness is untouched by the narrowing: all six arms still ran.
    assert report["totals"]["arm_count"] == 6

    for arm, verdict in report["baseline_beat_tests"].items():
        assert verdict["verdict"] != "beats_baselines", (
            "{0} must never be declared a winner without texture evidence".format(arm))
        assert verdict["comparable_cases"] == 0
    for arm in report["totals"]["evaluated_candidate_arms"]:
        assert report["baseline_beat_tests"][arm]["verdict"] == "not_computable", (
            "{0} is a candidate this round and must report not_computable, not "
            "a verdict, without texture evidence".format(arm))
    for arm in report["totals"]["reserved_arms_pending_failure"]:
        entry = report["baseline_beat_tests"][arm]
        assert entry["verdict"] == "reserved_not_evaluated"
        assert entry["evaluated_this_round"] is False
        # Reported, never silently dropped.
        assert "reserve" in entry["reason"].lower()

    # Safety gates: no arm FAILS, but nothing is disqualified either, and the
    # coverage is thin. Measured on this machine, where the gitignored float
    # arrays happen to be present:
    #   * makeup_edge_contamination is evaluable on 3 labelled patches x 6 arms
    #   * forbidden_pixel_change / innermost_ring_leakage only on DSCF2709,
    #     the single patch carrying a protected mask, x 6 arms
    #   * defect_reintroduction is NEVER evaluable: zero corrected masks corpus-wide
    # so most case/arms remain inadmissible for lack of any safety data at all.
    gates = report["disqualifiers"]
    assert gates["disqualified_count"] == 0
    assert gates["by_gate"]["defect_reintroduction"]["pass"] == 0
    assert gates["by_gate"]["defect_reintroduction"]["not_applicable"] == 60
    assert gates["inadmissible_count"] > 0, (
        "most case/arms have no safety data and must stay inadmissible")
    # Admissibility is NOT scorability: clearing a safety gate never substitutes
    # for the missing texture evidence, which is what the assertions above pin.
    assert report["totals"]["scorable_case_arms"] == 0

    # The abstention contract abstains everywhere.
    assert report["totals"]["eligible_cases"] == []
    assert len(report["totals"]["abstaining_cases"]) == 10

    assert report["blocking_for_first_production_candidate_comparison"]


def test_real_pilot_report_records_the_run_identity_for_ordering(tmp_path):
    """Ordering evidence: the report ties the scoring lock to each run's hashes."""
    directories = _real_pilot_directories()
    if len(directories) < 10:
        pytest.skip("real FA-02 pilot directories are not present")
    lock_path = tmp_path / "scoring_lock.json"
    lock = contract.freeze_scoring(lock_path)
    report = scoring_report.build_report(lock_path, directories,
                                         measure_texture=False)
    assert report["run_references"], "each scored run must be identifiable"
    for reference in report["run_references"]:
        assert reference["config_sha256"]
        assert reference["manifest_sha256"]
        assert reference["scoring_sha256"] == lock["scoring_sha256"]


def test_real_pilot_report_does_not_promote_the_corpus(tmp_path):
    """This contract advances the RULE, never the corpus's promotion stage."""
    directories = _real_pilot_directories()
    if len(directories) < 10:
        pytest.skip("real FA-02 pilot directories are not present")
    lock_path = tmp_path / "scoring_lock.json"
    contract.freeze_scoring(lock_path)
    report = scoring_report.build_report(lock_path, directories, measure_texture=False)
    assert "does not promote the corpus" in report["disclaimer"]
    assert "production_ready" not in json.dumps(report["totals"])

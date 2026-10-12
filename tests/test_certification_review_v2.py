import importlib.util
import json
import sys
from pathlib import Path

from retouch.certification_review_v2 import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    finalize_matrix_review,
    make_blinded_presentation,
    make_review_payload,
    make_review_record,
    review_item_id,
    review_payload_is_certifying,
    validate_review_payload,
)


FINALIZER_PATH = Path(__file__).resolve().parents[1] / "scripts/recipes/finalize_core_recipe_certification.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location("finalize_core_recipe_certification_v2", FINALIZER_PATH)
assert FINALIZER_SPEC and FINALIZER_SPEC.loader
finalizer = importlib.util.module_from_spec(FINALIZER_SPEC)
FINALIZER_SPEC.loader.exec_module(finalizer)


def _record(
    reviewer,
    outcome="approved",
    *,
    recipe="natural",
    defects=None,
    seed=None,
):
    return make_review_record(
        "portrait",
        recipe,
        reviewer,
        outcome,
        0.9,
        source_artifact_id="sha256-source",
        output_artifact_id="sha256-output",
        defects=defects,
        seed=seed or reviewer,
    )


def _payload(first="approved", second="approved", *, third=None, recipe="natural", defects=None):
    reviews = [
        _record("reviewer-a", first, recipe=recipe, defects=defects if first == "approved" else None),
        _record("reviewer-b", second, recipe=recipe),
    ]
    third_reviews = [] if third is None else [_record("reviewer-c", third, recipe=recipe)]
    return make_review_payload(reviews, third_reviews=third_reviews, matrix_id="fixture-matrix")


def _matrix(*, automatic=True, face_aware=True):
    return {
        "version": 1,
        "recipe_count": 1,
        "corpus_complete": True,
        "automatic_pass": automatic,
        "face_aware_run": face_aware,
        # Deliberately contradictory legacy values are repaired by the v2
        # finalizer rather than being trusted.
        "final_certified": True,
        "human_review": "pending",
        "face_aware_certification": False,
        "compatibility": {"final_certified": True, "face_aware_certification": False},
        "cases": [{
            "case": "portrait",
            "recipes": ["natural"],
            "face_aware_run": face_aware,
            "automatic_pass": automatic,
            "final_certified": True,
            "human_review": "pending",
            "face_aware_certification": False,
        }],
    }


def _write_fixture(tmp_path, matrix, payload):
    output = tmp_path / "core-cert"
    output.mkdir(exist_ok=True)
    (output / "matrix_manifest.json").write_text(json.dumps(matrix), encoding="utf-8")
    (output / "human_review.json").write_text(json.dumps(payload), encoding="utf-8")
    return output


def test_blinded_presentation_is_opaque_and_reproducibly_randomized():
    first = make_blinded_presentation("portrait::natural", "source-hash", "output-hash", seed="a")
    repeat = make_blinded_presentation("portrait::natural", "source-hash", "output-hash", seed="a")
    assert first == repeat
    assert first["left_id"] != first["right_id"]
    assert first["randomized"] is True
    assert "source" not in first["left_id"]
    assert "output" not in first["right_id"]

    other = make_blinded_presentation("portrait::natural", "source-hash", "output-hash", seed="b")
    assert {other["left_id"], other["right_id"]} != {first["left_id"], first["right_id"]}
    assert other["left_id"] != other["right_id"]


def test_two_independent_approved_blinded_reviews_are_certifying():
    payload = _payload()
    result = validate_review_payload(payload, [("portrait", "natural")])
    assert result["schema_valid"] is True
    assert result["valid"] is True
    assert result["complete"] is True
    assert result["human_review"] == "approved"
    assert result["items"][review_item_id("portrait", "natural")]["third_review_required"] is False
    assert review_payload_is_certifying(payload, [review_item_id("portrait", "natural")]) is True


def test_uncertain_is_explicit_and_never_becomes_approval():
    payload = _payload(first="uncertain", second="uncertain")
    result = validate_review_payload(payload, [("portrait", "natural")])
    assert result["valid"] is True
    assert result["human_review"] == "uncertain"
    assert review_payload_is_certifying(payload, [("portrait", "natural")]) is False


def test_disagreement_requires_a_distinct_third_review_and_majority_resolves_it():
    incomplete = validate_review_payload(
        _payload(first="approved", second="rejected"),
        [("portrait", "natural")],
    )
    item = incomplete["items"][review_item_id("portrait", "natural")]
    assert incomplete["valid"] is False
    assert item["disagreement"] is True
    assert item["third_review_required"] is True
    assert item["third_review_complete"] is False
    assert incomplete["human_review"] == "pending"

    resolved = validate_review_payload(
        _payload(first="approved", second="rejected", third="approved"),
        [("portrait", "natural")],
    )
    assert resolved["valid"] is True
    assert resolved["items"][review_item_id("portrait", "natural")]["resolved_outcome"] == "approved"
    assert resolved["human_review"] == "approved"

    no_majority = validate_review_payload(
        _payload(first="approved", second="rejected", third="uncertain"),
        [("portrait", "natural")],
    )
    assert no_majority["valid"] is True
    assert no_majority["items"][review_item_id("portrait", "natural")]["resolved_outcome"] == "uncertain"
    assert no_majority["human_review"] == "uncertain"


def test_critical_defect_requires_adjudication_and_two_reports_confirm_it():
    critical = [{"label": "severe_boundary_bleed", "severity": "critical", "region": "jaw", "note": "visible"}]
    without_third = validate_review_payload(
        _payload(defects=critical),
        [("portrait", "natural")],
    )
    item = without_third["items"][review_item_id("portrait", "natural")]
    assert without_third["valid"] is False
    assert item["third_review_required"] is True
    assert item["critical_defects_reported"] == ["severe_boundary_bleed"]

    confirmed = validate_review_payload(
        make_review_payload(
            [_record("reviewer-a", defects=critical), _record("reviewer-b")],
            third_reviews=[_record("reviewer-c", defects=critical)],
        ),
        [("portrait", "natural")],
    )
    item = confirmed["items"][review_item_id("portrait", "natural")]
    assert confirmed["valid"] is True
    assert item["critical_defects_confirmed"] is True
    assert item["status"] == "rejected"
    assert confirmed["human_review"] == "rejected"


def test_duplicate_reviewer_is_inconsistent_even_when_outcomes_match():
    payload = make_review_payload([_record("reviewer-a"), _record("reviewer-a")])
    result = validate_review_payload(payload, [("portrait", "natural")])
    assert result["valid"] is False
    assert result["human_review"] == "inconsistent"
    assert result["items"][review_item_id("portrait", "natural")]["status"] == "inconsistent"


def test_finalizer_fixture_derives_all_fields_without_contradiction(tmp_path, monkeypatch):
    output = _write_fixture(tmp_path, _matrix(), _payload())
    monkeypatch.setattr(sys, "argv", ["finalize", str(output)])
    assert finalizer.main() == 0

    finalized = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert finalized["final_certified"] is True
    assert finalized["human_review"] == "approved"
    assert finalized["face_aware_certification"] is True
    assert finalized["compatibility"]["final_certified"] is True
    assert finalized["compatibility"]["face_aware_certification"] is True
    row = finalized["cases"][0]
    assert row["human_review"] == "approved"
    assert row["final_certified"] is True
    assert row["face_aware_certification"] is True
    assert row["compatibility"]["final_certified"] is True
    assert finalized["review_evidence_v2"]["valid"] is True


def test_finalizer_fails_closed_for_legacy_or_missing_review_evidence(tmp_path, monkeypatch):
    legacy_payload = {"version": 1, "rows": [{"case": "portrait", "recipe": "natural", "human_natural_output": "approved"}]}
    output = _write_fixture(tmp_path, _matrix(), legacy_payload)
    monkeypatch.setattr(sys, "argv", ["finalize", str(output)])
    assert finalizer.main() == 1

    finalized = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert finalized["final_certified"] is False
    assert finalized["face_aware_certification"] is False
    assert finalized["compatibility"]["final_certified"] is False
    assert finalized["human_review"] == "inconsistent"
    assert finalized["cases"][0]["final_certified"] is False
    assert finalized["cases"][0]["face_aware_certification"] is False


def test_finalizer_does_not_certify_uncertain_or_missing_third_review(tmp_path, monkeypatch):
    output = _write_fixture(tmp_path, _matrix(), _payload(first="approved", second="rejected"))
    monkeypatch.setattr(sys, "argv", ["finalize", str(output)])
    assert finalizer.main() == 1
    incomplete = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert incomplete["final_certified"] is False
    assert incomplete["cases"][0]["final_certified"] is False

    output = _write_fixture(tmp_path, _matrix(), _payload(first="uncertain", second="uncertain"))
    monkeypatch.setattr(sys, "argv", ["finalize", str(output)])
    assert finalizer.main() == 1
    uncertain = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    assert uncertain["human_review"] == "uncertain"
    assert uncertain["final_certified"] is False

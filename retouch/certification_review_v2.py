"""Pure Certification Evidence v2 human-review contract.

The v2 contract deliberately keeps reviewer evidence separate from render and
automatic-QA evidence.  A review payload is certifying only when it contains
two distinct blinded reviews for every expected case/recipe item.  A third,
distinct reviewer is required whenever the first two reviewers disagree or
either reports a critical defect.

The functions in this module operate on JSON-compatible mappings and return
new mappings.  They do not read files, mutate caller-owned data, or depend on
the GUI, which makes the contract suitable for fixture and release tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_NAME = "retouch.certification.review"
SCHEMA_VERSION = 2

OUTCOMES = frozenset({"approved", "rejected", "uncertain"})
HUMAN_REVIEW_STATES = frozenset({
    "pending",
    "approved",
    "rejected",
    "uncertain",
    "inconsistent",
})
DEFECT_SEVERITIES = frozenset({"minor", "major", "critical"})

# The vocabulary is intentionally small and actionable.  Unknown labels are
# still accepted when they have the required structured label/severity/region
# shape, because a locked corpus may discover a product-specific defect that
# should not be silently dropped by an older reader.
CRITICAL_DEFECT_LABELS = frozenset({
    "face_missing",
    "face_swap",
    "identity_change",
    "likeness_change",
    "geometry_failure",
    "unsafe_mask",
    "severe_boundary_bleed",
    "severe_color_shift",
    "critical_artifact",
})

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,255}$")
_ROLE_WORDS = frozenset({"source", "output", "original", "retouched", "input"})


class ReviewContractError(ValueError):
    """Raised when a v2 review payload cannot be trusted."""


def review_item_id(case: Any, recipe: Any) -> str:
    """Return the canonical, stable identifier for one case/recipe render."""
    case_text = str(case or "").strip()
    recipe_text = str(recipe or "").strip()
    if not case_text or not recipe_text:
        raise ValueError("case and recipe are required")
    return f"{case_text}::{recipe_text}"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _opaque_presentation_id(pair_digest: str, side: str) -> str:
    return f"p2-{pair_digest[:24]}-{side}"


def make_blinded_presentation(
    item_id: str,
    source_artifact_id: Any,
    output_artifact_id: Any,
    *,
    seed: Any = None,
) -> Dict[str, Any]:
    """Create opaque randomized left/right identifiers for a review pair.

    The source/output identifiers are used only to create a stable pair
    commitment; they are not copied into the presentation object.  Reviewers
    therefore receive two opaque identifiers and cannot infer which side is
    the source from the JSON contract.  ``seed`` makes fixture generation
    reproducible while still allowing independent assignments to use distinct
    randomization inputs.
    """
    if not str(item_id or "").strip():
        raise ValueError("item_id is required")
    source = str(source_artifact_id or "").strip()
    output = str(output_artifact_id or "").strip()
    if not source or not output or source == output:
        raise ValueError("source and output artifact identifiers must be distinct")
    commitment = hashlib.sha256(
        _canonical_json({
            "item_id": str(item_id),
            "source": source,
            "output": output,
            "seed": seed,
        }).encode("utf-8")
    ).hexdigest()
    # The parity is part of the commitment, not exposed as source-left or
    # output-left.  The presentation IDs themselves are role-neutral.
    left_side = "a" if int(commitment[-2:], 16) % 2 == 0 else "b"
    right_side = "b" if left_side == "a" else "a"
    return {
        "left_id": _opaque_presentation_id(commitment, left_side),
        "right_id": _opaque_presentation_id(commitment, right_side),
        "randomized": True,
        "pair_commitment": commitment,
    }


def make_review_record(
    case: Any,
    recipe: Any,
    reviewer_id: Any,
    outcome: Any,
    confidence: Any,
    *,
    source_artifact_id: Any,
    output_artifact_id: Any,
    defects: Optional[Sequence[Mapping[str, Any]]] = None,
    seed: Any = None,
    review_id: Optional[str] = None,
    presentation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one completed, blinded review record.

    This helper is intentionally strict about the values it creates, so test
    fixtures and small internal tools do not accidentally emit a legacy or
    partially structured row.
    """
    item_id = review_item_id(case, recipe)
    reviewer_text = str(reviewer_id or "").strip()
    if not reviewer_text:
        raise ValueError("reviewer_id is required")
    outcome_text = str(outcome or "").strip().lower()
    if outcome_text not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    if isinstance(confidence, bool):
        raise ValueError("confidence must be a number between 0 and 1")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    review_identifier = review_id or _review_id(item_id, reviewer_text, seed)
    pair = dict(presentation or make_blinded_presentation(
        item_id,
        source_artifact_id,
        output_artifact_id,
        seed=seed,
    ))
    return {
        "review_id": str(review_identifier),
        "item_id": item_id,
        "case": str(case),
        "recipe": str(recipe),
        "reviewer_id": reviewer_text,
        "reviewer_type": "independent",
        "blinded": True,
        "presentation": pair,
        "outcome": outcome_text,
        "confidence": confidence_value,
        "defects": [dict(defect) for defect in (defects or [])],
    }


def _review_id(item_id: str, reviewer_id: str, seed: Any) -> str:
    digest = hashlib.sha256(
        _canonical_json({"item_id": item_id, "reviewer": reviewer_id, "seed": seed}).encode("utf-8")
    ).hexdigest()
    return f"review-{digest[:24]}"


def make_review_payload(
    reviews: Sequence[Mapping[str, Any]],
    *,
    third_reviews: Optional[Sequence[Mapping[str, Any]]] = None,
    matrix_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap completed records in the v2 JSON envelope."""
    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "protocol": {
            "blinded": True,
            "randomized_left_right": True,
            "independent_reviewers": 2,
            "third_review_on_disagreement_or_critical_defect": True,
        },
        "reviews": [copy.deepcopy(dict(row)) for row in reviews],
        "third_reviews": [copy.deepcopy(dict(row)) for row in (third_reviews or [])],
    }


def expected_items_from_matrix(matrix: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Extract unique case/recipe review items from a matrix manifest."""
    items: List[Dict[str, str]] = []
    errors: List[str] = []
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        return [], ["matrix.cases must be a non-empty list"]
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            errors.append(f"matrix.cases[{index}] must be an object")
            continue
        case_name = str(case.get("case") or "").strip()
        recipes = case.get("recipes")
        if not case_name or not isinstance(recipes, list) or not recipes:
            errors.append(f"matrix.cases[{index}] requires case and non-empty recipes")
            continue
        for recipe in recipes:
            recipe_name = str(recipe or "").strip()
            if not recipe_name:
                errors.append(f"matrix.cases[{index}] contains an empty recipe")
                continue
            item_id = review_item_id(case_name, recipe_name)
            if item_id in seen:
                errors.append(f"duplicate matrix review item: {item_id}")
                continue
            seen.add(item_id)
            items.append({"item_id": item_id, "case": case_name, "recipe": recipe_name})
    return items, errors


def _record_list(payload: Mapping[str, Any], primary: str, legacy_alias: Optional[str] = None) -> Tuple[Any, str]:
    if primary in payload and legacy_alias in payload if legacy_alias else False:
        return None, f"payload cannot contain both {primary} and {legacy_alias}"
    if primary in payload:
        return payload.get(primary), primary
    if legacy_alias and legacy_alias in payload:
        return payload.get(legacy_alias), legacy_alias
    return None, f"payload.{primary} is required"


def _is_opaque_id(value: Any) -> bool:
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        return False
    tokens = {token.lower() for token in re.split(r"[-_.:]+", value) if token}
    return not bool(tokens & _ROLE_WORDS)


def _validate_defects(value: Any, path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    normalized: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return [], [f"{path} must be a list"]
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{item_path} must be an object")
            continue
        label = str(raw.get("label") or "").strip().lower()
        severity = str(raw.get("severity") or "").strip().lower()
        if not label:
            errors.append(f"{item_path}.label is required")
        if severity not in DEFECT_SEVERITIES:
            errors.append(f"{item_path}.severity must be one of {sorted(DEFECT_SEVERITIES)}")
        region = raw.get("region", "global")
        if not isinstance(region, str) or not region.strip():
            errors.append(f"{item_path}.region must be a non-empty string")
        note = raw.get("note", "")
        if not isinstance(note, str):
            errors.append(f"{item_path}.note must be a string")
        critical = severity == "critical" or label in CRITICAL_DEFECT_LABELS
        if "critical" in raw and not isinstance(raw.get("critical"), bool):
            errors.append(f"{item_path}.critical must be boolean when supplied")
        if isinstance(raw.get("critical"), bool) and raw["critical"] != critical:
            errors.append(f"{item_path}.critical contradicts severity/label")
        normalized.append({
            "label": label,
            "severity": severity,
            "region": region.strip() if isinstance(region, str) else region,
            "note": note,
            "critical": critical,
        })
    return normalized, errors


def _validate_record(
    raw: Any,
    *,
    path: str,
    expected_by_id: Mapping[str, Mapping[str, str]],
    seen_review_ids: set,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(raw, Mapping):
        return None, [f"{path} must be an object"]
    required = (
        "review_id", "item_id", "case", "recipe", "reviewer_id",
        "blinded", "presentation", "outcome", "confidence", "defects",
    )
    for field in required:
        if field not in raw:
            errors.append(f"{path}.{field} is required")
    review_id = str(raw.get("review_id") or "").strip()
    item_id = str(raw.get("item_id") or "").strip()
    if not review_id:
        errors.append(f"{path}.review_id must be non-empty")
    elif review_id in seen_review_ids:
        errors.append(f"duplicate review_id: {review_id}")
    else:
        seen_review_ids.add(review_id)
    expected = expected_by_id.get(item_id)
    if expected is None:
        errors.append(f"{path}.item_id is not an expected matrix item: {item_id!r}")
    else:
        if str(raw.get("case") or "") != expected["case"]:
            errors.append(f"{path}.case does not match item_id")
        if str(raw.get("recipe") or "") != expected["recipe"]:
            errors.append(f"{path}.recipe does not match item_id")
    reviewer_id = str(raw.get("reviewer_id") or "").strip()
    if not reviewer_id:
        errors.append(f"{path}.reviewer_id must be non-empty")
    if raw.get("reviewer_type", "independent") != "independent":
        errors.append(f"{path}.reviewer_type must be independent")
    if raw.get("blinded") is not True:
        errors.append(f"{path}.blinded must be true")
    presentation = raw.get("presentation")
    if not isinstance(presentation, Mapping):
        errors.append(f"{path}.presentation must be an object")
        presentation = {}
    left_id = presentation.get("left_id")
    right_id = presentation.get("right_id")
    if not _is_opaque_id(left_id):
        errors.append(f"{path}.presentation.left_id must be an opaque identifier")
    if not _is_opaque_id(right_id):
        errors.append(f"{path}.presentation.right_id must be an opaque identifier")
    if left_id == right_id:
        errors.append(f"{path}.presentation left and right identifiers must differ")
    if presentation.get("randomized") is not True:
        errors.append(f"{path}.presentation.randomized must be true")
    outcome = str(raw.get("outcome") or "").strip().lower()
    if outcome not in OUTCOMES:
        errors.append(f"{path}.outcome must be one of {sorted(OUTCOMES)}")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"{path}.confidence must be a number between 0 and 1")
    else:
        if not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{path}.confidence must be a number between 0 and 1")
    defects, defect_errors = _validate_defects(raw.get("defects"), f"{path}.defects")
    errors.extend(defect_errors)
    normalized = {
        "review_id": review_id,
        "item_id": item_id,
        "case": str(raw.get("case") or ""),
        "recipe": str(raw.get("recipe") or ""),
        "reviewer_id": reviewer_id,
        "reviewer_type": "independent",
        "blinded": True,
        "presentation": dict(presentation),
        "outcome": outcome,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        "defects": defects,
    }
    return normalized, errors


def validate_review_payload(
    payload: Any,
    expected_items: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Validate v2 evidence and return per-item review decisions.

    ``expected_items`` accepts canonical item IDs, ``(case, recipe)`` pairs,
    or mappings containing ``item_id``/``case``/``recipe``.  Supplying it is
    strongly recommended; without it the function can validate reviewer
    structure but cannot detect a missing matrix item.
    """
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return {
            "schema_valid": False,
            "valid": False,
            "complete": False,
            "human_review": "inconsistent",
            "errors": ["review payload must be an object"],
            "items": {},
            "disagreements": [],
        }
    if payload.get("schema") != SCHEMA_NAME:
        errors.append(f"payload.schema must equal {SCHEMA_NAME!r}")
    if payload.get("version") != SCHEMA_VERSION:
        errors.append(f"payload.version must equal {SCHEMA_VERSION}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        errors.append("payload.protocol is required")
        protocol = {}
    protocol_requirements = {
        "blinded": True,
        "randomized_left_right": True,
        "independent_reviewers": 2,
        "third_review_on_disagreement_or_critical_defect": True,
    }
    for field, expected in protocol_requirements.items():
        if protocol.get(field) != expected:
            errors.append(f"payload.protocol.{field} must be {expected!r}")

    raw_reviews, review_field = _record_list(payload, "reviews", "rows")
    raw_third, third_field = _record_list(payload, "third_reviews", "adjudications")
    if raw_reviews is None:
        errors.append(review_field)
        raw_reviews = []
    if raw_third is None:
        # Third reviews are optional at the envelope level; they become
        # mandatory per item when disagreement/critical evidence requires it.
        raw_third = []
    if not isinstance(raw_reviews, list):
        errors.append(f"payload.{review_field} must be a list")
        raw_reviews = []
    if not isinstance(raw_third, list):
        errors.append(f"payload.{third_field} must be a list")
        raw_third = []

    expected: List[Dict[str, str]] = []
    if expected_items is not None:
        for item in expected_items:
            if isinstance(item, Mapping):
                item_id = str(item.get("item_id") or "").strip()
                case = str(item.get("case") or "").strip()
                recipe = str(item.get("recipe") or "").strip()
                if not item_id and case and recipe:
                    item_id = review_item_id(case, recipe)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                case, recipe = str(item[0]), str(item[1])
                item_id = review_item_id(case, recipe)
            else:
                item_id = str(item or "").strip()
                if "::" in item_id:
                    case, recipe = item_id.split("::", 1)
                else:
                    case, recipe = "", ""
            if not item_id or not case or not recipe:
                errors.append(f"invalid expected review item: {item!r}")
                continue
            expected.append({"item_id": item_id, "case": case, "recipe": recipe})
    else:
        inferred = {}
        for raw in list(raw_reviews) + list(raw_third):
            if isinstance(raw, Mapping):
                item_id = str(raw.get("item_id") or "").strip()
                case = str(raw.get("case") or "").strip()
                recipe = str(raw.get("recipe") or "").strip()
                if not item_id and case and recipe:
                    item_id = review_item_id(case, recipe)
                if item_id and case and recipe:
                    inferred[item_id] = {"item_id": item_id, "case": case, "recipe": recipe}
        expected = list(inferred.values())
        if not expected:
            errors.append("no expected review items were supplied or inferable")
    expected_by_id = {item["item_id"]: item for item in expected}
    if len(expected_by_id) != len(expected):
        errors.append("expected review items must be unique")

    seen_review_ids: set = set()
    primary_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    third_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(raw_reviews):
        normalized, record_errors = _validate_record(
            raw,
            path=f"payload.{review_field}[{index}]",
            expected_by_id=expected_by_id,
            seen_review_ids=seen_review_ids,
        )
        errors.extend(record_errors)
        if normalized is not None and normalized["item_id"] in expected_by_id:
            primary_by_item[normalized["item_id"]].append(normalized)
    for index, raw in enumerate(raw_third):
        normalized, record_errors = _validate_record(
            raw,
            path=f"payload.{third_field}[{index}]",
            expected_by_id=expected_by_id,
            seen_review_ids=seen_review_ids,
        )
        errors.extend(record_errors)
        if normalized is not None and normalized["item_id"] in expected_by_id:
            third_by_item[normalized["item_id"]].append(normalized)

    expected_ids = set(expected_by_id)
    observed_ids = set(primary_by_item) | set(third_by_item)
    for unexpected in sorted(observed_ids - expected_ids):
        errors.append(f"unexpected review item: {unexpected}")

    item_results: Dict[str, Dict[str, Any]] = {}
    disagreement_ids: List[str] = []
    item_errors: Dict[str, List[str]] = defaultdict(list)
    for item in expected:
        item_id = item["item_id"]
        primary = primary_by_item.get(item_id, [])
        third = third_by_item.get(item_id, [])
        local_errors: List[str] = []
        if len(primary) != 2:
            local_errors.append(f"requires exactly two independent reviews; found {len(primary)}")
        reviewer_ids = [row["reviewer_id"] for row in primary]
        if len(reviewer_ids) == 2 and len(set(reviewer_ids)) != 2:
            local_errors.append("the two independent reviewers must be distinct")
        outcomes = [row["outcome"] for row in primary]
        disagreement = len(outcomes) == 2 and outcomes[0] != outcomes[1]
        critical_labels = [
            defect["label"]
            for row in primary
            for defect in row["defects"]
            if defect.get("critical")
        ]
        third_required = disagreement or bool(critical_labels)
        if disagreement:
            disagreement_ids.append(item_id)
        if third_required:
            if len(third) != 1:
                local_errors.append(
                    "exactly one distinct third review is required for disagreement or critical defects"
                )
            elif third[0]["reviewer_id"] in set(reviewer_ids):
                local_errors.append("third reviewer must be distinct from both independent reviewers")
        elif third:
            local_errors.append("third review is present although no adjudication trigger exists")

        all_reviews = primary + third
        all_outcomes = [row["outcome"] for row in all_reviews]
        critical_all = [
            defect["label"]
            for row in all_reviews
            for defect in row["defects"]
            if defect.get("critical")
        ]
        critical_counts = Counter(critical_all)
        confirmed_critical = any(count >= 2 for count in critical_counts.values())
        if third_required and len(third) == 1 and third[0]["outcome"] == "rejected" and critical_labels:
            # An adjudicating rejection confirms that a critical issue is not
            # merely a single-reviewer observation.
            confirmed_critical = True

        complete = not local_errors
        resolved_outcome: Optional[str] = None
        if complete:
            if third_required:
                outcome_counts = Counter(all_outcomes)
                winner, winner_count = outcome_counts.most_common(1)[0]
                # Three distinct outcomes have no majority.  Treating the
                # first record as the winner would silently convert an
                # uncertainty into approval.
                resolved_outcome = winner if winner_count >= 2 else "uncertain"
            elif len(outcomes) == 2 and outcomes[0] == outcomes[1]:
                resolved_outcome = outcomes[0]
        if not complete:
            status = "pending" if third_required and len(third) == 0 else "inconsistent"
        elif resolved_outcome == "rejected" or confirmed_critical:
            status = "rejected"
        elif resolved_outcome == "uncertain":
            status = "uncertain"
        elif resolved_outcome == "approved":
            status = "approved"
        else:
            status = "inconsistent"
            local_errors.append("review outcome could not be resolved")
            complete = False
        item_errors[item_id].extend(local_errors)
        item_results[item_id] = {
            "item_id": item_id,
            "case": item["case"],
            "recipe": item["recipe"],
            "review_count": len(primary),
            "third_review_count": len(third),
            "reviewer_ids": reviewer_ids + [row["reviewer_id"] for row in third],
            "outcomes": outcomes,
            "resolved_outcome": resolved_outcome,
            "status": status,
            "complete": complete,
            "disagreement": disagreement,
            "third_review_required": third_required,
            "third_review_complete": bool(third) and len(third) == 1,
            "critical_defects_reported": sorted(set(critical_all)),
            "critical_defects_confirmed": confirmed_critical,
            "errors": list(local_errors),
        }

    for item_id, local in item_errors.items():
        errors.extend(f"{item_id}: {message}" for message in local)
    complete = bool(expected) and all(item["complete"] for item in item_results.values()) and not errors
    statuses = [item["status"] for item in item_results.values()]
    if not complete:
        # A missing adjudicator is an incomplete, reviewable state rather than
        # malformed evidence. Schema/record/duplicate errors remain explicitly
        # inconsistent and can never pass the finalizer.
        structural_error = any(item["status"] == "inconsistent" for item in item_results.values()) or any(
            error.startswith("payload.")
            or "duplicate" in error
            or "unexpected review item" in error
            or "does not match" in error
            for error in errors
        )
        human_review = "inconsistent" if structural_error else "pending"
    elif any(status == "rejected" for status in statuses):
        human_review = "rejected"
    elif any(status == "uncertain" for status in statuses):
        human_review = "uncertain"
    elif statuses and all(status == "approved" for status in statuses):
        human_review = "approved"
    else:
        human_review = "inconsistent"
    return {
        "schema_valid": not any(
            error.startswith("payload.schema") or error.startswith("payload.version") or error.startswith("payload.protocol")
            for error in errors
        ),
        "valid": not errors,
        "complete": complete,
        "human_review": human_review,
        "errors": errors,
        "items": item_results,
        "disagreements": disagreement_ids,
        "third_review_required": sorted(
            item_id for item_id, item in item_results.items() if item["third_review_required"]
        ),
        "critical_defects": sorted({
            label
            for item in item_results.values()
            for label in item["critical_defects_reported"]
        }),
    }


def _matrix_gate_state(matrix: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    """Compute release prerequisites without trusting stale final flags."""
    rows_present = bool(rows)
    face_aware = rows_present and all(row.get("face_aware_run") is True for row in rows)
    automatic = (
        rows_present
        and matrix.get("automatic_pass") is True
        and matrix.get("corpus_complete") is True
        and all(row.get("automatic_pass") is True for row in rows)
    )
    return {
        "face_aware_run": face_aware,
        "automatic_pass": automatic,
        "corpus_complete": matrix.get("corpus_complete") is True,
    }


def _compatibility_fields(existing: Any, final_certified: bool) -> Dict[str, Any]:
    result = dict(existing) if isinstance(existing, Mapping) else {}
    result["final_certified"] = bool(final_certified)
    result["face_aware_certification"] = bool(final_certified)
    # These aliases are only populated when they already existed or when the
    # compatibility object is newly created.  No stale true alias survives.
    for alias in ("certified", "is_certified"):
        if alias in result:
            result[alias] = bool(final_certified)
    return result


def finalize_matrix_review(
    matrix: Mapping[str, Any],
    review_payload: Any,
) -> Dict[str, Any]:
    """Return a consistent, fail-closed matrix with v2 review evidence.

    The input matrix and review payload are never modified.  Missing or
    inconsistent review evidence yields ``final_certified == False`` and
    explicit error/state fields; it never preserves a stale true flag.
    """
    output = copy.deepcopy(dict(matrix)) if isinstance(matrix, Mapping) else {}
    rows_raw = output.get("cases")
    rows = rows_raw if isinstance(rows_raw, list) else []
    expected, matrix_errors = expected_items_from_matrix(output)
    review = validate_review_payload(review_payload, expected)
    errors = list(matrix_errors) + list(review.get("errors") or [])
    item_results = review.get("items") or {}
    row_results: Dict[str, Dict[str, Any]] = {}
    normalized_rows: List[Dict[str, Any]] = []
    gates = _matrix_gate_state(output, [row for row in rows if isinstance(row, Mapping)])
    for index, raw_row in enumerate(rows):
        row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
        case_name = str(row.get("case") or "")
        recipes = row.get("recipes") if isinstance(row.get("recipes"), list) else []
        item_ids = [review_item_id(case_name, recipe) for recipe in recipes if case_name and str(recipe or "").strip()]
        case_items = [item_results[item_id] for item_id in item_ids if item_id in item_results]
        case_errors = [item_error for item_id in item_ids for item_error in item_results.get(item_id, {}).get("errors", [])]
        if not case_items or len(case_items) != len(item_ids):
            case_errors.append("missing review evidence for one or more case recipes")
        if not review.get("schema_valid") or matrix_errors:
            case_errors.append("review or matrix envelope is structurally invalid")
            case_status = "inconsistent"
        elif case_errors:
            # Item-level pending means a required adjudicator or review is
            # still absent. Item-level inconsistent means supplied evidence
            # contradicts the v2 contract.
            case_status = (
                "inconsistent"
                if any(item["status"] == "inconsistent" for item in case_items)
                else "pending"
            )
        elif all(item["status"] == "approved" for item in case_items):
            case_status = "approved"
        elif any(item["status"] == "rejected" for item in case_items):
            case_status = "rejected"
        elif any(item["status"] == "uncertain" for item in case_items):
            case_status = "uncertain"
        else:
            case_status = "pending"
        row_certified = bool(
            row.get("face_aware_run") is True
            and row.get("automatic_pass") is True
            and output.get("corpus_complete") is True
            and output.get("automatic_pass") is True
            and case_status == "approved"
            and not case_errors
        )
        row["human_review"] = case_status
        row["final_certified"] = row_certified
        row["face_aware_certification"] = row_certified
        row["compatibility"] = _compatibility_fields(row.get("compatibility"), row_certified)
        row["certification_state"] = "certified" if row_certified else case_status
        row["review_evidence_v2"] = {
            "status": case_status,
            "item_ids": item_ids,
            "errors": case_errors,
        }
        normalized_rows.append(row)
        row_results[case_name or f"row-{index}"] = {
            "human_review": case_status,
            "final_certified": row_certified,
            "errors": case_errors,
        }
        errors.extend(f"case {case_name or index}: {error}" for error in case_errors)

    review_status = str(review.get("human_review") or "inconsistent")
    if errors and review_status == "approved":
        review_status = "inconsistent"
    all_rows_certified = bool(normalized_rows) and all(row.get("final_certified") is True for row in normalized_rows)
    final_certified = bool(
        gates["face_aware_run"]
        and gates["automatic_pass"]
        and review_status == "approved"
        and all_rows_certified
        and not errors
    )
    # A rejected/uncertain result is meaningful evidence; missing or malformed
    # structure is explicitly inconsistent.  Neither state can pass.
    if not review.get("valid") and review_status == "approved":
        review_status = "inconsistent"
    output["cases"] = normalized_rows
    output["face_aware_run"] = gates["face_aware_run"]
    output["automatic_pass"] = gates["automatic_pass"]
    output["human_review"] = review_status
    output["final_certified"] = final_certified
    output["face_aware_certification"] = final_certified
    output["compatibility"] = _compatibility_fields(output.get("compatibility"), final_certified)
    output["certification_state"] = "certified" if final_certified else review_status
    output["review_evidence_v2"] = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "valid": bool(review.get("valid")) and not matrix_errors,
        "complete": bool(review.get("complete")) and not matrix_errors,
        "human_review": review_status,
        "disagreements": list(review.get("disagreements") or []),
        "third_review_required": list(review.get("third_review_required") or []),
        "critical_defects": list(review.get("critical_defects") or []),
        "errors": errors,
        "rows": row_results,
    }
    # Synchronize only known legacy aliases if they were present.  This
    # prevents an old true alias from contradicting the new final gate.
    for alias in ("certified", "is_certified"):
        if alias in output:
            output[alias] = final_certified
    return {
        "matrix": output,
        "review": review,
        "errors": errors,
        "final_certified": final_certified,
        "human_review": review_status,
    }


def review_payload_is_certifying(
    payload: Any,
    expected_items: Iterable[Any],
) -> bool:
    """Small predicate for callers that only need the human gate."""
    result = validate_review_payload(payload, expected_items)
    return bool(result.get("valid") and result.get("human_review") == "approved")


__all__ = [
    "CRITICAL_DEFECT_LABELS",
    "DEFECT_SEVERITIES",
    "HUMAN_REVIEW_STATES",
    "OUTCOMES",
    "ReviewContractError",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "expected_items_from_matrix",
    "finalize_matrix_review",
    "make_blinded_presentation",
    "make_review_payload",
    "make_review_record",
    "review_item_id",
    "review_payload_is_certifying",
    "validate_review_payload",
]

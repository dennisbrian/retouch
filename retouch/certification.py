"""Shared truth contract for automatic and human visual certification."""

from __future__ import annotations

from typing import Any, Mapping


HUMAN_REVIEW_STATES = {"pending", "approved", "rejected", "not_applicable"}


def certification_fields(
    *,
    face_aware_run: bool,
    automatic_pass: bool,
    human_review: str = "pending",
) -> dict[str, Any]:
    """Return independent evidence fields; never infer success from a flag."""
    human_review = str(human_review or "pending").lower()
    if human_review not in HUMAN_REVIEW_STATES:
        raise ValueError(f"unknown human review state: {human_review!r}")
    final_certified = bool(
        face_aware_run
        and automatic_pass
        and human_review in {"approved", "not_applicable"}
    )
    return {
        "face_aware_run": bool(face_aware_run),
        "automatic_pass": bool(automatic_pass),
        "human_review": human_review,
        "final_certified": final_certified,
        # Compatibility alias for older reports. It intentionally means the
        # final gate, never merely "the flag was absent".
        "face_aware_certification": final_certified,
    }


def core_case_automatic_pass(
    case_manifest: Mapping[str, Any],
    case_output: Any,
    expected_recipes: int,
    *,
    returncode: int = 0,
) -> bool:
    """Validate one recipe-sweep case, including real output artifacts."""
    if returncode != 0 or case_manifest.get("global_only") is not False:
        return False
    outputs = case_manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != expected_recipes:
        return False
    output_root = case_output
    for row in outputs:
        if not isinstance(row, Mapping) or row.get("status") != "done":
            return False
        output_path = output_root / str(row.get("output", ""))
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            return False
    contact_sheet = output_root / str(case_manifest.get("contact_sheet", ""))
    return contact_sheet.is_file() and contact_sheet.stat().st_size > 0


def core_matrix_final_certified(
    rows: list[Mapping[str, Any]],
    human_rows: list[Mapping[str, Any]],
    expected_recipes: int,
) -> bool:
    """Reject incomplete cases, non-face-aware evidence, and pending review."""
    if not rows or not human_rows:
        return False
    if any(
        not bool(row.get("face_aware_run"))
        or not bool(row.get("automatic_pass"))
        or row.get("human_review") != "approved"
        for row in rows
    ):
        return False
    required = {
        (str(case.get("case")), str(recipe))
        for case in rows
        for recipe in case.get("recipes", [])
    }
    reviewed = {
        (str(row.get("case")), str(row.get("recipe")))
        for row in human_rows
        if row.get("human_natural_output") == "approved" and str(row.get("reviewer", "")).strip()
    }
    return bool(required) and required <= reviewed and all(
        int(row.get("recipe_count", 0)) == expected_recipes for row in rows
    )

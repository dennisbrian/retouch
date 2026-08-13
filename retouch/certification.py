"""Shared truth contract for automatic and human visual certification."""

from __future__ import annotations

from typing import Any, Mapping


HUMAN_REVIEW_STATES = {"pending", "approved", "rejected", "not_applicable"}

# A single portrait is not representative enough for release certification.
# Case names are intentionally semantic so the corpus can remain outside the
# repository while the gate still proves that the required strata were run.
CORE_CORPUS_DIMENSIONS = (
    "skin_tone",
    "lighting",
    "glasses",
    "wig",
    "hands_on_face",
    "group_portrait",
)

_CORE_CORPUS_ALIASES = {
    "skin_tone": ("skin_tone", "skin-tone", "dark_skin", "deep_skin", "light_skin", "tone"),
    "lighting": ("lighting", "light", "lowlight", "mixed_light", "venue"),
    "glasses": ("glasses", "eyeglass", "spectacle"),
    "wig": ("wig", "cosplay", "hair"),
    "hands_on_face": ("hands_on_face", "hands-on-face", "hand_face", "handsonface"),
    "group_portrait": ("group", "family", "multi_face", "multiface"),
}


def core_corpus_coverage(case_names: Any) -> dict[str, Any]:
    """Report whether named cases cover every required certification stratum."""
    names = [str(name).strip().lower() for name in case_names]
    covered = {
        dimension: sorted(
            name for name in names
            if any(alias in name for alias in aliases)
        )
        for dimension, aliases in _CORE_CORPUS_ALIASES.items()
    }
    missing = [dimension for dimension in CORE_CORPUS_DIMENSIONS if not covered[dimension]]
    return {
        "required": list(CORE_CORPUS_DIMENSIONS),
        "covered": covered,
        "missing": missing,
        "complete": not missing,
    }


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


def core_case_human_review_status(
    case_name: str,
    recipe_names: Any,
    human_rows: Any,
) -> str:
    """Return the independent reviewer state for one completed corpus case."""
    expected = {str(recipe) for recipe in recipe_names}
    matching = {
        str(row.get("recipe")): row
        for row in human_rows
        if isinstance(row, Mapping) and str(row.get("case")) == str(case_name)
    }
    if not expected or not expected <= set(matching):
        return "pending"
    states = {
        recipe: str(matching[recipe].get("human_natural_output", "pending")).lower()
        for recipe in expected
    }
    if "rejected" in states.values():
        return "rejected"
    if all(
        states[recipe] == "approved" and str(matching[recipe].get("reviewer", "")).strip()
        for recipe in expected
    ):
        return "approved"
    return "pending"


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
    *,
    corpus_complete: bool = True,
) -> bool:
    """Reject incomplete cases, non-face-aware evidence, and pending review."""
    if not corpus_complete or not rows or not human_rows:
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

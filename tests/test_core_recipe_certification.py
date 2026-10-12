from retouch.certification import (
    core_case_automatic_pass,
    core_case_human_review_status,
    core_corpus_coverage,
    core_matrix_final_certified,
)
from retouch.core_recipes import CORE_RECIPE_NAMES, CORE_RECIPE_REVIEW_DIMENSIONS
from retouch.recipes import RECIPES


def test_core_recipe_scope_is_stable_and_valid():
    assert 10 <= len(CORE_RECIPE_NAMES) <= 15
    assert len(CORE_RECIPE_NAMES) == len(set(CORE_RECIPE_NAMES))
    assert set(CORE_RECIPE_NAMES) <= set(RECIPES)
    assert set(CORE_RECIPE_NAMES) == set(CORE_RECIPE_REVIEW_DIMENSIONS)


def _valid_case(tmp_path, *, global_only=False, contact=True):
    case_output = tmp_path / "case"
    case_output.mkdir(exist_ok=True)
    outputs = []
    for index, recipe in enumerate(CORE_RECIPE_NAMES):
        filename = f"{index:02d}_{recipe}.jpg"
        (case_output / filename).write_bytes(b"rendered")
        outputs.append({"recipe": recipe, "output": filename, "status": "done"})
    if contact:
        (case_output / "contact_sheet.jpg").write_bytes(b"sheet")
    return case_output, {
        "global_only": global_only,
        "outputs": outputs,
        "contact_sheet": "contact_sheet.jpg",
    }


def test_core_automatic_aggregator_accepts_only_complete_face_aware_case(tmp_path):
    case_output, manifest = _valid_case(tmp_path)
    assert core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES))


def test_core_automatic_aggregator_rejects_missing_output(tmp_path):
    case_output, manifest = _valid_case(tmp_path)
    (case_output / manifest["outputs"][0]["output"]).unlink()
    assert not core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES))


def test_core_automatic_aggregator_rejects_failed_case(tmp_path):
    case_output, manifest = _valid_case(tmp_path)
    assert not core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES), returncode=1)


def test_core_automatic_aggregator_rejects_global_only_case(tmp_path):
    case_output, manifest = _valid_case(tmp_path, global_only=True)
    assert not core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES))


def test_core_automatic_aggregator_rejects_empty_contact_sheet(tmp_path):
    case_output, manifest = _valid_case(tmp_path, contact=False)
    assert not core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES))


def test_core_final_aggregator_rejects_pending_human_review(tmp_path):
    case_output, manifest = _valid_case(tmp_path)
    row = {
        "case": "portrait",
        "recipes": list(CORE_RECIPE_NAMES),
        "recipe_count": len(CORE_RECIPE_NAMES),
        "face_aware_run": True,
        "automatic_pass": core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES)),
        "human_review": "pending",
    }
    human_rows = [
        {"case": "portrait", "recipe": recipe, "human_natural_output": "pending", "reviewer": ""}
        for recipe in CORE_RECIPE_NAMES
    ]
    assert not core_matrix_final_certified([row], human_rows, len(CORE_RECIPE_NAMES))


def test_core_final_aggregator_requires_approved_human_review(tmp_path):
    case_output, manifest = _valid_case(tmp_path)
    row = {
        "case": "portrait",
        "recipes": list(CORE_RECIPE_NAMES),
        "recipe_count": len(CORE_RECIPE_NAMES),
        "face_aware_run": True,
        "automatic_pass": core_case_automatic_pass(manifest, case_output, len(CORE_RECIPE_NAMES)),
        "human_review": "approved",
    }
    human_rows = [
        {"case": "portrait", "recipe": recipe, "human_natural_output": "approved", "reviewer": "qa"}
        for recipe in CORE_RECIPE_NAMES
    ]
    assert core_matrix_final_certified([row], human_rows, len(CORE_RECIPE_NAMES))


def test_core_corpus_requires_representative_conditions():
    coverage = core_corpus_coverage([
        "skin_tone_dark",
        "mixed_lighting",
        "glasses_portrait",
        "cosplay_wig",
        "hands_on_face",
        "group_portrait",
    ])

    assert coverage["complete"] is True
    assert coverage["missing"] == []


def test_core_final_aggregator_rejects_incomplete_corpus_even_if_reviewed():
    human_rows = [
        {"case": "portrait", "recipe": recipe, "human_natural_output": "approved", "reviewer": "qa"}
        for recipe in CORE_RECIPE_NAMES
    ]
    row = {
        "case": "portrait",
        "recipes": list(CORE_RECIPE_NAMES),
        "recipe_count": len(CORE_RECIPE_NAMES),
        "face_aware_run": True,
        "automatic_pass": True,
        "human_review": "approved",
    }
    assert not core_matrix_final_certified(
        [row], human_rows, len(CORE_RECIPE_NAMES), corpus_complete=False
    )


def test_core_case_human_review_requires_every_recipe_and_reviewer():
    rows = [
        {"case": "portrait", "recipe": recipe, "human_natural_output": "approved", "reviewer": "qa"}
        for recipe in CORE_RECIPE_NAMES
    ]

    assert core_case_human_review_status("portrait", CORE_RECIPE_NAMES, rows) == "approved"
    rows[0]["reviewer"] = ""
    assert core_case_human_review_status("portrait", CORE_RECIPE_NAMES, rows) == "pending"

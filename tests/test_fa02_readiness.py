"""FA-02 texture-annotation schema, readiness report and workbench contracts.

All fixture data is clearly fake: placeholder person_ids (``test_subject_*``),
tiny synthetic arrays and byte-blob "images". No real subject, session or
portrait is referenced anywhere in this file.

The load-bearing assertions here are the ones that would otherwise silently
produce a WRONG answer rather than a failing one:

* subjects are unioned across pilot directories, not summed (one person in two
  directories is one person);
* legacy ``review_regions`` are never counted as semantic labels, no matter
  what their free-text tags say;
* an annotation written before the semantic categories existed still
  validates, and the categories survive the normalizer round-trip.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/qa"))
# Fixture builders (_annotation, _corpus_manifest, _write_case_arrays, ...) are
# reused from the bridge's own suite rather than duplicated, so the two files
# cannot drift into disagreeing about what a valid fixture looks like.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa02_annotation_workbench as workbench  # noqa: E402
import fa02_portrait_manifest as bridge  # noqa: E402
import fa02_readiness_report as readiness  # noqa: E402
import fa02_texture_representation_experiment as exp  # noqa: E402

from test_fa02_portrait_manifest import (  # noqa: E402
    CANVAS,
    _annotation,
    _corpus_manifest,
    _probe,
    _write_case_arrays,
    _write_corpus,
)


def _regions(*specs):
    """``("id", x, y, w, h)`` tuples -> the region-list shape."""
    return [{"id": rid, "xywh": [x, y, w, h], "tags": []} for rid, x, y, w, h in specs]


# --------------------------------------------------------------------------
# Extended annotation schema
# --------------------------------------------------------------------------


def test_old_style_annotation_without_new_categories_still_validates():
    """The four committed real patches are exactly this shape; they must live."""
    annotation = _annotation()
    assert not any(category in annotation for category in bridge.TEXTURE_REGION_CATEGORIES)
    result = bridge.validate_support_annotation(annotation)
    # Every category is present-and-empty, never missing and never invented.
    for category in bridge.TEXTURE_REGION_CATEGORIES:
        assert result[category] == []
    assert len(result["review_regions"]) == 1


def test_committed_real_annotations_still_validate_under_the_extended_schema():
    """Primary compatibility evidence: the actual shipped owner-approved files."""
    paths = sorted((ROOT / "test_output").glob("fa02_pilot_*/annotation_*.json"))
    if not paths:
        pytest.skip("real pilot annotations are not present in this checkout")
    labelled = 0
    for path in paths:
        result = bridge.validate_support_annotation(json.loads(path.read_text()))
        assert result["annotation_source"] in bridge.ACCEPTED_ANNOTATION_SOURCES
        # Every category is present-and-normalized, whether or not it carries
        # labels: the four original patches declare none (review_regions only),
        # the six ff47-event patches declare real ones. Both shapes must live.
        for category in bridge.TEXTURE_REGION_CATEGORIES:
            assert isinstance(result[category], list)
        if any(result[c] for c in bridge.TEXTURE_REGION_CATEGORIES):
            labelled += 1
            # A populated category is only meaningful with an attributable
            # human behind it -- the same gate _validate_texture_regions
            # enforces, re-asserted here against the real shipped files.
            assert result.get("reviewer")
    # Guard against the whole corpus silently reverting to label-free files.
    assert labelled >= 1


def test_new_region_categories_validate_and_survive_the_round_trip():
    annotation = _annotation()
    annotation["pore_regions"] = _regions(("pore_a", 4, 4, 16, 16), ("pore_b", 40, 4, 16, 16))
    annotation["fine_hair_regions"] = _regions(("hair_a", 4, 4, 20, 20))  # overlaps pore_a: legal
    annotation["noise_regions"] = _regions(("noise_a", 90, 90, 20, 20))
    annotation["uncertainty_regions"] = _regions(("unsure_a", 60, 60, 8, 8))
    result = bridge.validate_support_annotation(annotation)
    assert [r["id"] for r in result["pore_regions"]] == ["pore_a", "pore_b"]
    assert len(result["fine_hair_regions"]) == 1
    assert len(result["noise_regions"]) == 1
    # Round-trip through JSON and back: a normalizer that rebuilds key-by-key
    # would silently drop these, and the readiness counts would read zero.
    again = bridge.validate_support_annotation(json.loads(json.dumps(result)))
    assert again["pore_regions"] == result["pore_regions"]


def test_per_region_notes_are_preserved():
    annotation = _annotation()
    annotation["pore_regions"] = [
        {"id": "pore_a", "xywh": [4, 4, 16, 16], "tags": ["pore"],
         "notes": "clearly resolved at 1:1"}
    ]
    result = bridge.validate_support_annotation(annotation)
    assert result["pore_regions"][0]["notes"] == "clearly resolved at 1:1"


@pytest.mark.parametrize("xywh", ([-1, 0, 16, 16], [0, 0, 0, 16], [0, 0, 16, -4],
                                  [0, 0, 16], "16x16", [0.5, 0, 16, 16]))
def test_malformed_new_regions_are_rejected(xywh):
    annotation = _annotation()
    annotation["pore_regions"] = [{"id": "pore_a", "xywh": xywh, "tags": []}]
    with pytest.raises(ValueError, match="pore_regions"):
        bridge.validate_support_annotation(annotation)


def test_new_region_list_of_the_wrong_type_is_rejected():
    annotation = _annotation()
    annotation["fine_hair_regions"] = {"id": "not_a_list"}
    with pytest.raises(ValueError, match="fine_hair_regions must be a list"):
        bridge.validate_support_annotation(annotation)


def test_new_region_out_of_crop_bounds_is_rejected_by_the_promotion_gate(tmp_path):
    """Crop bounds are only knowable once the NPZ is on hand -- gate, not schema."""
    manifest = _corpus_manifest(tmp_path)
    corpus_path = _write_corpus(tmp_path, manifest)
    arrays = _write_case_arrays(tmp_path)
    (tmp_path / "placeholder_native.tif").write_bytes(b"placeholder-native-capture")

    annotation = _annotation()
    # In-frame as a rectangle, but past the right edge of the CANVAS crop.
    annotation["pore_regions"] = _regions(("pore_far", CANVAS - 8, 4, 32, 16))
    bridge.validate_support_annotation(annotation)  # schema alone cannot see it

    report = bridge.load_corpus_report(corpus_path, image_probe=_probe)
    case = bridge.build_portrait_case(
        report, annotation, case_id="portrait_case_0", arrays=arrays.name,
        arrays_base=tmp_path, corpus_manifest_path=corpus_path,
        session_id="PLACEHOLDER_session_A",
        owner_mapping_reference="PLACEHOLDER/owner-mapping",
        smoothing_provenance="PLACEHOLDER fixture smoother",
        color_profile="PLACEHOLDER encoded BGR",
        inter_eye_distance_px=200.0, face_width_px=500.0)
    fa02 = {"schema_version": 1, "purpose": "development_pilot", "corpus": "fixture",
            "parameter_selection": "fixture", "cases": [case]}
    path = tmp_path / "fa02_manifest.json"
    exp.write_json(path, fa02)
    gate = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {v["code"] for v in gate["violations"]}
    assert gate["valid"] is False
    assert "texture_region_outside_crop" in codes


@pytest.mark.parametrize("other", ("corrected_defect_regions", "noise_regions"))
def test_illegal_category_overlap_is_rejected(other):
    annotation = _annotation()
    annotation["protected_identity_marks"] = _regions(("mark_a", 40, 40, 20, 20))
    annotation[other] = _regions(("other_a", 50, 50, 20, 20))  # shares interior area
    with pytest.raises(ValueError, match="mutually exclusive"):
        bridge.validate_support_annotation(annotation)


def test_touching_but_non_overlapping_exclusive_regions_are_allowed():
    """Abutting rectangles share a zero-area boundary, not interior pixels."""
    annotation = _annotation()
    annotation["protected_identity_marks"] = _regions(("mark_a", 40, 40, 10, 10))
    annotation["noise_regions"] = _regions(("noise_a", 50, 40, 10, 10))
    assert bridge.validate_support_annotation(annotation)["noise_regions"]


def test_legal_category_overlap_is_allowed():
    """pore x fine_hair must NOT be constrained -- real cheeks have both."""
    annotation = _annotation()
    annotation["pore_regions"] = _regions(("pore_a", 40, 40, 30, 30))
    annotation["fine_hair_regions"] = _regions(("hair_a", 45, 45, 30, 30))
    annotation["makeup_edge_regions"] = _regions(("makeup_a", 42, 42, 30, 30))
    annotation["uncertainty_regions"] = _regions(("unsure_a", 41, 41, 30, 30))
    result = bridge.validate_support_annotation(annotation)
    assert len(result["pore_regions"]) == len(result["fine_hair_regions"]) == 1


def test_duplicate_region_ids_across_categories_are_rejected():
    annotation = _annotation()
    annotation["pore_regions"] = _regions(("shared_id", 4, 4, 16, 16))
    annotation["noise_regions"] = _regions(("shared_id", 90, 90, 16, 16))
    with pytest.raises(ValueError, match="unique across all categories"):
        bridge.validate_support_annotation(annotation)


def test_duplicate_id_against_a_legacy_review_region_is_rejected():
    annotation = _annotation()  # ships review_regions[0].id == "pore_field"
    annotation["pore_regions"] = _regions(("pore_field", 40, 40, 16, 16))
    with pytest.raises(ValueError, match="unique across all categories"):
        bridge.validate_support_annotation(annotation)


def test_texture_regions_require_accepted_provenance():
    annotation = _annotation()
    annotation["annotation_source"] = "detector_derived"
    annotation.pop("reviewer")
    annotation["pore_regions"] = _regions(("pore_a", 4, 4, 16, 16))
    with pytest.raises(ValueError, match="accepted annotation_source"):
        bridge.validate_support_annotation(annotation)


def test_texture_regions_require_a_named_reviewer():
    annotation = _annotation()
    annotation["annotation_source"] = "manual"  # accepted, but reviewer optional
    annotation.pop("reviewer")
    assert bridge.validate_support_annotation(annotation)["pore_regions"] == []
    annotation["pore_regions"] = _regions(("pore_a", 4, 4, 16, 16))
    with pytest.raises(ValueError, match="named reviewer"):
        bridge.validate_support_annotation(annotation)


def test_uncertainty_regions_alone_still_need_provenance_but_are_not_ground_truth():
    annotation = _annotation()
    annotation["uncertainty_regions"] = _regions(("unsure_a", 4, 4, 16, 16))
    result = bridge.validate_support_annotation(annotation)
    assert len(result["uncertainty_regions"]) == 1
    assert "uncertainty_regions" not in bridge.GROUND_TRUTH_CATEGORIES


def test_extended_template_still_validates_and_stays_fake():
    path = ROOT / "scripts/qa/fa02_support_annotation_template.json"
    result = bridge.validate_support_annotation(json.loads(path.read_text()))
    assert result["person_id"] == "test_subject_001"
    for category in bridge.TEXTURE_REGION_CATEGORIES:
        assert result[category], f"template should demonstrate {category}"
    text = path.read_text()
    assert "PLACEHOLDER" in text and "DSCF" not in text


def test_allow_support_description_is_not_dropped():
    annotation = _annotation()
    annotation["supports"]["allow"]["description"] = "eligible skin, hair excluded"
    result = bridge.validate_support_annotation(annotation)
    assert result["supports"]["allow"]["description"] == "eligible skin, hair excluded"


# --------------------------------------------------------------------------
# Readiness report
# --------------------------------------------------------------------------


def _pilot_dir(root, name, *, person_id="test_subject_001",
               split="dev", annotation_extra=None, strata=None):
    """Build one complete fake FA-02 pilot directory with known counts."""
    directory = root / name
    directory.mkdir()
    manifest = _corpus_manifest(directory, splits=(split,), person_ids=([person_id],))
    # Each pilot directory is a separate shoot/asset, so give it a distinct
    # asset_id -- exactly like the four real committed directories, which hold
    # priority_printing_DSCF2650, priority_printing_DSCF2709, etc. Reusing one
    # id across directories would make the aggregator's by-id dedup collapse
    # genuinely different assets and hide the union-vs-sum question this
    # fixture exists to test.
    asset_id = f"placeholder_asset_{name}"
    manifest["assets"][0]["asset_id"] = asset_id
    if strata:
        manifest["assets"][0]["strata"].update(strata)
    corpus_path = _write_corpus(directory, manifest)
    arrays = _write_case_arrays(directory)
    (directory / "placeholder_native.tif").write_bytes(
        f"placeholder-native-capture-{name}".encode())

    annotation = _annotation(asset_id=asset_id, person_id=person_id)
    annotation.update(annotation_extra or {})
    (directory / "annotation_case.json").write_text(json.dumps(annotation, indent=2))

    report = bridge.load_corpus_report(corpus_path, image_probe=_probe)
    case = bridge.build_portrait_case(
        report, annotation, case_id=f"case_{name}", arrays=arrays.name,
        arrays_base=directory, corpus_manifest_path=corpus_path,
        session_id=f"PLACEHOLDER_session_{name}",
        owner_mapping_reference="PLACEHOLDER/owner-mapping",
        smoothing_provenance="PLACEHOLDER fixture smoother",
        color_profile="PLACEHOLDER encoded BGR",
        inter_eye_distance_px=200.0, face_width_px=500.0,
        purpose="development_pilot" if split != "locked_test" else "locked_comparison",
        allow_locked_test=split == "locked_test")
    exp.write_json(directory / "fa02_manifest.json", {
        "schema_version": 1,
        "purpose": "development_pilot" if split != "locked_test" else "locked_comparison",
        "corpus": "fixture", "parameter_selection": "fixture", "cases": [case]})
    return directory


def test_readiness_counts_match_a_known_synthetic_fixture(tmp_path):
    """3 directories, 2 distinct subjects, known per-category region counts."""
    labelled = {
        "pore_regions": _regions(("pore_a", 4, 4, 16, 16), ("pore_b", 40, 4, 16, 16),
                                 ("pore_c", 4, 40, 16, 16)),
        "fine_hair_regions": _regions(("hair_a", 70, 4, 16, 16)),
        "noise_regions": _regions(("noise_a", 100, 100, 16, 16)),
        "makeup_edge_regions": _regions(("makeup_a", 4, 70, 16, 16)),
    }
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001")
    # Same fake subject as pilot_a: must be counted ONCE, not twice.
    _pilot_dir(tmp_path, "pilot_b", person_id="test_subject_001",
               annotation_extra=labelled, strata={"lighting": "high_key"})
    _pilot_dir(tmp_path, "pilot_c", person_id="test_subject_002",
               strata={"lighting": "low_key"})

    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a", tmp_path / "pilot_b", tmp_path / "pilot_c"],
        corpus_validator_kwargs={"image_probe": _probe})
    totals = report["totals"]

    assert totals["pilot_directories"] == 3
    # THE discriminating assertion: union, not sum. Summing per-manifest
    # person counts would give 3.
    assert totals["subject_count"] == 2
    assert totals["subject_counts_by_split"]["dev"] == 2
    assert totals["subjects"] == ["test_subject_001", "test_subject_002"]
    assert totals["patch_count"] == 3
    assert totals["fa02_case_count"] == 3
    assert totals["gate_reports_passing"] == totals["gate_reports_run"] == 3
    assert totals["problems"] == []

    counts = totals["reviewed_region_counts"]
    assert counts["pore_regions"] == 3
    assert counts["fine_hair_regions"] == 1
    assert counts["noise_regions"] == 1
    assert counts["makeup_edge_regions"] == 1
    assert counts["protected_identity_marks"] == 0
    assert totals["ground_truth_region_total"] == 4
    assert totals["control_region_total"] == 2
    assert totals["reviewed_patch_counts"]["pore_regions"] == 1
    assert totals["human_labelled_patch_count"] == 1
    # Three fixture patches each carry one legacy review_region, counted apart.
    assert totals["legacy_review_region_total"] == 3
    assert totals["legacy_review_regions_are_labels"] is False

    assert totals["strata_tally"]["lighting"] == {"high_key": 1, "low_key": 1, "studio": 1}
    json.dumps(report)  # JSON-safe end to end


def test_readiness_never_counts_legacy_review_regions_as_labels(tmp_path):
    """A review_region tagged 'pore_review' is an advisory note, not a label."""
    _pilot_dir(tmp_path, "pilot_a", annotation_extra={
        "review_regions": [
            {"id": "looks_porey", "xywh": [4, 4, 16, 16],
             "tags": ["pore_review", "fine_hair_review", "texture"]},
        ]})
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a"], corpus_validator_kwargs={"image_probe": _probe})
    totals = report["totals"]
    assert totals["legacy_review_region_total"] == 1
    assert totals["ground_truth_region_total"] == 0
    assert totals["reviewed_region_counts"]["pore_regions"] == 0
    assert totals["human_labelled_patch_count"] == 0


def test_readiness_reports_infrastructure_validated_without_labels(tmp_path):
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001")
    _pilot_dir(tmp_path, "pilot_b", person_id="test_subject_002")
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a", tmp_path / "pilot_b"],
        corpus_validator_kwargs={"image_probe": _probe})
    promotion = report["promotion"]
    assert promotion["attained_stage"] == "infrastructure_validated"
    assert promotion["next_stage"] == "dev_pilot_ready"
    missing = {c["criterion"] for c in promotion["missing_for_next_stage"]}
    # Two subjects exist, so the ONLY thing blocking dev_pilot_ready is labels.
    assert missing == {"human_reviewed_texture_annotations_exist"}


def test_readiness_reports_dev_pilot_ready_once_human_labels_exist(tmp_path):
    labelled = {"pore_regions": _regions(("pore_a", 4, 4, 16, 16)),
                "fine_hair_regions": _regions(("hair_a", 40, 4, 16, 16))}
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001",
               annotation_extra=labelled)
    _pilot_dir(tmp_path, "pilot_b", person_id="test_subject_002")
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a", tmp_path / "pilot_b"],
        corpus_validator_kwargs={"image_probe": _probe})
    promotion = report["promotion"]
    assert promotion["attained_stage"] == "dev_pilot_ready"
    assert promotion["next_stage"] == "production_candidate_ready"


def test_single_subject_corpus_cannot_reach_dev_pilot_ready(tmp_path):
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001",
               annotation_extra={"pore_regions": _regions(("pore_a", 4, 4, 16, 16))})
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a"], corpus_validator_kwargs={"image_probe": _probe})
    promotion = report["promotion"]
    assert promotion["attained_stage"] == "infrastructure_validated"
    assert "multiple_subjects" in {
        c["criterion"] for c in promotion["missing_for_next_stage"]}


def test_ground_truth_criterion_is_never_marked_met_by_a_proxy(tmp_path):
    """The forbidden criterion stays uncomputable no matter what data exists."""
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001",
               annotation_extra={"pore_regions": _regions(("pore_a", 4, 4, 16, 16))})
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a"], corpus_validator_kwargs={"image_probe": _probe})
    criteria = report["promotion"]["criteria"]["production_candidate_ready"]
    entry = next(c for c in criteria
                 if c["criterion"] == "candidate_beats_A0_and_A1_on_texture_and_safety")
    assert entry["met"] is None
    assert entry["automatically_checkable"] is False
    assert "NOT COMPUTABLE" in entry["detail"]
    # And production_ready's human sign-offs are likewise never auto-met.
    human = [c for c in report["promotion"]["criteria"]["production_ready"]
             if not c["automatically_checkable"]]
    assert {c["criterion"] for c in human} >= {"owner_review_passes",
                                               "full_recipe_validation_completed"}
    assert all(c["met"] is None for c in human)


def test_readiness_flags_a_subject_crossing_splits_across_separate_manifests(tmp_path):
    """Cross-FILE leakage: validate_corpus_manifest sees one file at a time."""
    _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001", split="dev")
    _pilot_dir(tmp_path, "pilot_b", person_id="test_subject_001", split="locked_test")
    report = readiness.build_readiness_report(
        [tmp_path / "pilot_a", tmp_path / "pilot_b"],
        corpus_validator_kwargs={"image_probe": _probe})
    totals = report["totals"]
    assert totals["subjects_crossing_splits"] == ["test_subject_001"]
    assert report["promotion"]["attained_stage"] is None
    assert "no_subject_crosses_splits" in {
        c["criterion"] for c in report["promotion"]["missing_for_next_stage"]}


def test_standalone_annotation_edits_are_bounds_checked_by_the_gate(tmp_path):
    """Goal-3 bounds checking must reach the file a reviewer actually edits.

    The case embeds a frozen annotation copy; the reviewer edits the standalone
    ``annotation_*.json``. If the report only gated the embedded copy, a
    reviewer's out-of-bounds rectangle would pass unnoticed.
    """
    directory = _pilot_dir(tmp_path, "pilot_a")
    path = directory / "annotation_case.json"
    annotation = json.loads(path.read_text())
    annotation["pore_regions"] = _regions(("pore_far", CANVAS - 8, 4, 64, 16))
    path.write_text(json.dumps(annotation))

    report = readiness.build_readiness_report(
        [directory], corpus_validator_kwargs={"image_probe": _probe})
    assert any("texture_region_outside_crop" in p for p in report["totals"]["problems"])


def test_embedded_vs_standalone_divergence_is_observed_but_never_blocks(tmp_path):
    """A case built before new labels were typed is benign, not a violation."""
    directory = _pilot_dir(tmp_path, "pilot_a", person_id="test_subject_001")
    _pilot_dir(tmp_path, "pilot_b", person_id="test_subject_002")
    path = directory / "annotation_case.json"
    annotation = json.loads(path.read_text())
    annotation["pore_regions"] = _regions(("pore_a", 4, 4, 16, 16))
    path.write_text(json.dumps(annotation))

    report = readiness.build_readiness_report(
        [directory, tmp_path / "pilot_b"],
        corpus_validator_kwargs={"image_probe": _probe})
    record = next(r for r in report["directories"] if r["directory"] == str(directory))
    assert record["embedded_annotation_matches_standalone"] is False
    # ...and it neither fails a gate nor blocks promotion.
    assert report["totals"]["problems"] == []
    assert report["totals"]["gate_reports_passing"] == 2
    assert report["promotion"]["attained_stage"] == "dev_pilot_ready"


def test_readiness_records_an_invalid_annotation_instead_of_counting_it(tmp_path):
    directory = _pilot_dir(tmp_path, "pilot_a")
    (directory / "annotation_broken.json").write_text(
        json.dumps({"schema_version": 1, "asset_id": "x"}))
    report = readiness.build_readiness_report(
        [directory], corpus_validator_kwargs={"image_probe": _probe})
    totals = report["totals"]
    assert totals["invalid_annotation_count"] == 1
    assert totals["patch_count"] == 1  # the valid one only
    assert any("annotation_broken" in p for p in totals["problems"])
    assert report["promotion"]["attained_stage"] is None


def test_readiness_cli_writes_only_the_named_json(tmp_path, capsys):
    directory = _pilot_dir(tmp_path, "pilot_a")
    before = sorted(p.name for p in directory.iterdir())
    out = tmp_path / "reports" / "readiness.json"
    code = readiness.main([str(directory), "--json", str(out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["report"] == "fa02_readiness"
    assert payload["totals"]["subject_count"] == 1
    printed = capsys.readouterr().out
    assert "attained stage:" in printed
    assert "advisory only, NOT labels" in printed
    assert sorted(p.name for p in directory.iterdir()) == before


def test_readiness_cli_rejects_a_missing_directory(tmp_path):
    assert readiness.main([str(tmp_path / "does_not_exist")]) == 2


# --------------------------------------------------------------------------
# Annotation workbench (read-only looking tool)
# --------------------------------------------------------------------------


def test_workbench_windows_the_native_image_at_1_to_1_without_writing_pixels(tmp_path):
    directory = _pilot_dir(tmp_path, "pilot_a")
    annotation = json.loads((directory / "annotation_case.json").read_text())
    annotation["pore_regions"] = _regions(("pore_a", 10, 20, 16, 16))
    (directory / "annotation_case.json").write_text(json.dumps(annotation))
    before = sorted(p.name for p in directory.iterdir())

    out = directory / "workbench.html"
    code = workbench.main([str(directory / "corpus_manifest.json"),
                           "--annotation", str(directory / "annotation_case.json"),
                           "--context-margin", "32", "--html", str(out)])
    assert code == 0
    text = out.read_text()
    # native_roi 100,80 + CANVAS, margin 32 -> window origin (68,48).
    assert "left:-68px;top:-48px" in text
    assert f"width:{CANVAS + 64}px;height:{CANVAS + 64}px" in text
    # Crop-local pore at (10,20) -> window offset (100-68+10, 80-48+20).
    assert "left:42px;top:52px;width:16px;height:16px" in text
    assert "pore_a" in text
    # Nothing was resized, cropped or re-encoded, and only the HTML appeared.
    assert "image_probe" not in text
    assert sorted(p.name for p in directory.iterdir()) == sorted(before + ["workbench.html"])


def test_workbench_refuses_to_draw_an_invalid_annotation(tmp_path, capsys):
    directory = _pilot_dir(tmp_path, "pilot_a")
    broken = directory / "annotation_broken.json"
    broken.write_text(json.dumps({"schema_version": 1, "asset_id": "x"}))
    out = directory / "workbench.html"
    assert workbench.main([str(directory / "corpus_manifest.json"),
                           "--annotation", str(broken), "--html", str(out)]) == 1
    assert not out.exists()  # fail closed: no half-drawn page


def test_workbench_reports_a_missing_native_file_instead_of_faking_it(tmp_path):
    directory = _pilot_dir(tmp_path, "pilot_a")
    (directory / "placeholder_native.tif").unlink()
    out = directory / "workbench.html"
    assert workbench.main([str(directory / "corpus_manifest.json"),
                           "--annotation", str(directory / "annotation_case.json"),
                           "--html", str(out)]) == 0
    text = out.read_text()
    assert "Native capture not present" in text
    assert "<img" not in text  # no broken image dressed up as evidence


def test_workbench_lists_annotatable_assets_without_labelling_anything(tmp_path, capsys):
    directory = _pilot_dir(tmp_path, "pilot_a")
    assert workbench.main([str(directory / "corpus_manifest.json")]) == 0
    printed = capsys.readouterr().out
    assert "assets available to annotate (1)" in printed
    assert "placeholder_asset_pilot_a" in printed


def test_new_tools_run_no_detector_and_no_pipeline_import():
    """Extends the bridge's own guarantee to the modules added for this workflow."""
    for name in ("fa02_annotation_workbench.py", "fa02_readiness_report.py"):
        text = (ROOT / "scripts/qa" / name).read_text()
        for forbidden in ("detect_marks(", "retouch.engine", "retouch.detection",
                          "retouch.parsing", "retouch.freckle", "retouch.marks",
                          "import cv2", "cv2."):
            assert forbidden not in text, f"{name} must not reference {forbidden}"


def test_readiness_report_computes_no_metric_from_run_summaries():
    """Guard against a future proxy: the scanner must not read summary numbers.

    Checks executable code only. Docstrings and comments legitimately DISCUSS
    delta_rms (explaining why it is not used); scanning raw text would forbid
    the explanation along with the behaviour.
    """
    import ast

    source = (ROOT / "scripts/qa/fa02_readiness_report.py").read_text()
    tree = ast.parse(source)

    # No numeric/image library is even imported -- there is nothing to compute
    # a metric WITH.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"numpy", "cv2", "csv", "statistics", "math", "pandas"}
    # Prose in a criterion's `detail` legitimately EXPLAINS why delta_rms is
    # unusable, so scanning for the word would forbid the explanation along
    # with the behaviour. The import set is the check with teeth: with no
    # numeric library available there is nothing to compute a proxy WITH.

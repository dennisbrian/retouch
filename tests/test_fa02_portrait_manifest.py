"""Corpus-manifest-v3 <-> FA-02 portrait bridge contracts.

All fixture data is clearly fake: placeholder person_ids (``test_subject_*``),
tiny synthetic arrays, and byte-blob "images". No real subject, session or
portrait is referenced anywhere in this file.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/qa"))
import fa02_portrait_manifest as bridge  # noqa: E402
import fa02_portrait_review as review  # noqa: E402
import fa02_texture_representation_experiment as exp  # noqa: E402


# The harness's conservative common guard has a ~30px radius at the reference
# scale plus a 4px feather, measured from every forbidden pixel AND from the
# canvas border. A 64px fixture canvas has zero eligible pixels as a result --
# use a canvas large enough that a real eligible region survives.
CANVAS = 160


def _probe(_path):
    return {"probe_status": "ok", "decodable": True, "width": 800, "height": 600,
            "format": "PNG", "mode": "RGB"}


def _write_asset(root: Path, name: str, content: bytes):
    """Write a tiny real PNG so Pillow can probe it without a fixture probe."""
    import cv2
    path = root / name
    seed = hashlib.sha256(content).digest()[0]
    cv2.imwrite(str(path), np.full((12, 16, 3), int(seed) % 200 + 20, np.uint8))
    return path.name, hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_manifest(root: Path, *, splits=("dev", "calibration"), person_ids=None):
    """A fake two-asset v3 manifest with placeholder subjects."""
    person_ids = person_ids or (["test_subject_001"], ["test_subject_002"])
    assets = []
    for index, (split, people) in enumerate(zip(splits, person_ids)):
        name, sha = _write_asset(root, f"placeholder_{index}.png", f"placeholder-{index}".encode())
        assets.append({
            "asset_id": f"placeholder_asset_{index}",
            "path": name,
            "sha256": sha,
            "split": split,
            "person_ids": list(people),
            "tags": ["marks"],
            "strata": {"skin_tone": "medium", "lighting": "studio",
                       "face_scale": "close", "pose": "frontal"},
            "label_validation": {"status": "validated", "method": "human",
                                 "reference": "placeholder-labels"},
        })
    return {
        "schema_version": 3,
        "consent_reference": "PLACEHOLDER/consent/not-real",
        "assets": assets,
    }


def _write_corpus(root: Path, manifest):
    path = root / "corpus_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _write_case_arrays(root: Path, name="case_arrays.npz", size=CANVAS):
    """Tiny synthetic X/S plus disjoint allow/corrected/protected supports."""
    rng = np.random.default_rng(4242)
    x = rng.uniform(90, 170, (size, size, 3)).astype(np.float32)
    s = exp.gaussian(x, 2.0).astype(np.float32)
    allow = np.ones((size, size), np.float32)
    corrected = np.zeros((size, size), np.float32)
    corrected[20:24, 20:24] = 1
    protected = np.zeros((size, size), np.float32)
    protected[130:138, 130:138] = 1
    path = root / name
    np.savez_compressed(path, X=x, S=s, allow=allow, corrected=corrected, protected=protected)
    return path


def _annotation(asset_id="placeholder_asset_0", person_id="test_subject_001",
                *, size=CANVAS, source="placeholder_native.tif"):
    return {
        "schema_version": 1,
        "asset_id": asset_id,
        "person_id": person_id,
        "native_dimensions": [800, 600],
        "native_roi_xywh": [100, 80, size, size],
        "native_source_reference": source,
        "supports": {
            "allow": {"reference": "PLACEHOLDER/allow", "provenance": "manual"},
            "corrected": {"reference": "PLACEHOLDER/corrected", "provenance": "manual",
                          "description": "placeholder corrected defect",
                          "human_reference": "PLACEHOLDER/reviewer-note"},
            "protected": {"reference": "PLACEHOLDER/protected", "provenance": "manual",
                          "description": "placeholder protected mark"},
        },
        "support_acceptance_reference": "PLACEHOLDER/acceptance-record",
        "annotation_source": "owner_approved",
        "reviewer": "PLACEHOLDER_reviewer",
        "review_regions": [{"id": "pore_field", "xywh": [4, 4, 16, 16], "tags": ["pore_review"]}],
        "notes": "fixture only",
    }


@pytest.fixture
def workspace(tmp_path):
    """Fake corpus manifest + FA-02 case arrays + native source, one directory."""
    manifest = _corpus_manifest(tmp_path)
    corpus_path = _write_corpus(tmp_path, manifest)
    arrays = _write_case_arrays(tmp_path)
    (tmp_path / "placeholder_native.tif").write_bytes(b"placeholder-native-capture")
    return tmp_path, corpus_path, arrays


def _build(workspace, *, annotation=None, **overrides):
    root, corpus_path, arrays = workspace
    report = bridge.load_corpus_report(corpus_path, image_probe=_probe)
    kwargs = dict(
        case_id="portrait_case_0",
        arrays=arrays.name,
        arrays_base=root,
        corpus_manifest_path=corpus_path,
        session_id="PLACEHOLDER_session_A",
        owner_mapping_reference="PLACEHOLDER/owner-mapping",
        smoothing_provenance="PLACEHOLDER: fixture Gaussian sigma=2, not a real smoother capture",
        color_profile="PLACEHOLDER encoded BGR",
        inter_eye_distance_px=200.0,
        face_width_px=500.0,
    )
    kwargs.update(overrides)
    return bridge.build_portrait_case(report, annotation or _annotation(), **kwargs)


# --------------------------------------------------------------------------
# Annotation schema
# --------------------------------------------------------------------------


def test_annotation_template_file_validates():
    template = json.loads((ROOT / "scripts/qa/fa02_support_annotation_template.json").read_text())
    result = bridge.validate_support_annotation(template)
    assert result["annotation_source"] == "owner_approved"
    assert result["person_id"] == "test_subject_001"  # clearly fake placeholder


def test_missing_person_id_is_rejected():
    annotation = _annotation()
    del annotation["person_id"]
    with pytest.raises(ValueError, match="person_id"):
        bridge.validate_support_annotation(annotation)


def test_missing_annotation_source_is_rejected():
    annotation = _annotation()
    del annotation["annotation_source"]
    with pytest.raises(ValueError, match="annotation_source"):
        bridge.validate_support_annotation(annotation)


def test_owner_approved_without_reviewer_is_rejected():
    annotation = _annotation()
    del annotation["reviewer"]
    with pytest.raises(ValueError, match="reviewer"):
        bridge.validate_support_annotation(annotation)


def test_unknown_vocabulary_value_is_rejected():
    annotation = _annotation()
    annotation["annotation_source"] = "guessed"
    with pytest.raises(ValueError, match="annotation_source"):
        bridge.validate_support_annotation(annotation)
    annotation = _annotation()
    annotation["supports"]["allow"]["provenance"] = "vibes"
    with pytest.raises(ValueError, match="supports.allow.provenance"):
        bridge.validate_support_annotation(annotation)


@pytest.mark.parametrize("roi", ([-1, 0, 64, 64], [0, 0, 0, 64], [790, 0, 64, 64],
                                 [0, 560, 64, 64], [0, 0, 64], "64x64"))
def test_malformed_or_out_of_bounds_roi_is_rejected(roi):
    annotation = _annotation()
    annotation["native_roi_xywh"] = roi
    with pytest.raises(ValueError, match="native_roi_xywh"):
        bridge.validate_support_annotation(annotation)


def test_detector_derived_annotation_is_recordable_but_not_promotable(workspace):
    annotation = _annotation()
    annotation["annotation_source"] = "detector_derived"
    del annotation["reviewer"]
    # The schema accepts it as a record of provenance...
    assert bridge.validate_support_annotation(annotation)["annotation_source"] == "detector_derived"
    # ...but it can never become an externally accepted support.
    with pytest.raises(ValueError, match="accepted provenance"):
        _build(workspace, annotation=annotation)


# --------------------------------------------------------------------------
# Portrait case construction from the registry
# --------------------------------------------------------------------------


def test_case_built_from_corpus_asset_carries_full_provenance(workspace):
    case = _build(workspace)
    assert case["kind"] == "portrait"
    assert case["split"] == "dev"
    assert case["person_ids"] == ["test_subject_001"]
    assert case["support_status"] == "externally_accepted"
    assert case["masks_provenance"] in bridge.ACCEPTED_ANNOTATION_SOURCES
    assert case["corpus_source"]["asset_id"] == "placeholder_asset_0"
    assert case["corpus_source"]["asset_split"] == "dev"
    assert case["crop_xywh"] == [100, 80, CANVAS, CANVAS]
    assert case["resampled"] is False


def test_case_missing_person_id_in_corpus_asset_is_rejected(tmp_path):
    manifest = _corpus_manifest(tmp_path)
    manifest["assets"][0].pop("person_ids")
    corpus_path = _write_corpus(tmp_path, manifest)
    # corpus_manifest v3 itself refuses an asset with no person_ids, so the
    # bridge fails closed at load time rather than building anything.
    with pytest.raises(ValueError, match="person_id_required"):
        bridge.load_corpus_report(corpus_path, image_probe=_probe)


def test_annotation_person_not_in_asset_is_rejected(workspace):
    with pytest.raises(ValueError, match="not among the corpus asset"):
        _build(workspace, annotation=_annotation(person_id="test_subject_999"))


def test_cross_split_person_leakage_is_rejected(tmp_path):
    """Same fake person in two corpus splits: v3 refuses the whole manifest."""
    manifest = _corpus_manifest(
        tmp_path, splits=("dev", "locked_test"),
        person_ids=(["test_subject_001"], ["test_subject_001"]))
    corpus_path = _write_corpus(tmp_path, manifest)
    with pytest.raises(ValueError, match="person_crosses_split"):
        bridge.load_corpus_report(corpus_path, image_probe=_probe)


def test_locked_test_subject_refused_for_development_purpose(tmp_path):
    manifest = _corpus_manifest(tmp_path, splits=("locked_test", "dev"))
    corpus_path = _write_corpus(tmp_path, manifest)
    arrays = _write_case_arrays(tmp_path)
    (tmp_path / "placeholder_native.tif").write_bytes(b"placeholder-native-capture")
    workspace = (tmp_path, corpus_path, arrays)
    with pytest.raises(ValueError, match="not permitted for purpose 'development_pilot'"):
        _build(workspace)


def test_locked_test_refused_by_default_even_for_locked_comparison(tmp_path):
    manifest = _corpus_manifest(tmp_path, splits=("locked_test", "dev"))
    corpus_path = _write_corpus(tmp_path, manifest)
    arrays = _write_case_arrays(tmp_path)
    (tmp_path / "placeholder_native.tif").write_bytes(b"placeholder-native-capture")
    workspace = (tmp_path, corpus_path, arrays)
    with pytest.raises(ValueError, match="Refusing locked_test material by default"):
        _build(workspace, purpose="locked_comparison")
    case = _build(workspace, purpose="locked_comparison", allow_locked_test=True)
    assert case["split"] == "locked_test"


def test_declared_native_dimensions_vs_saved_array_shape_mismatch_is_rejected(workspace):
    annotation = _annotation()
    annotation["native_roi_xywh"] = [100, 80, 32, 32]  # not the 64x64 saved canvas
    with pytest.raises(ValueError, match="do not match the saved canvas shape"):
        _build(workspace, annotation=annotation)


def test_invalid_corpus_manifest_refuses_to_source_a_case(tmp_path):
    manifest = _corpus_manifest(tmp_path)
    manifest["assets"][0]["sha256"] = "0" * 64  # deliberate checksum mismatch
    corpus_path = _write_corpus(tmp_path, manifest)
    with pytest.raises(ValueError, match="sha256_mismatch"):
        bridge.load_corpus_report(corpus_path, image_probe=_probe)


# --------------------------------------------------------------------------
# Manifest assembly + promotion gates
# --------------------------------------------------------------------------


def _portrait_manifest(workspace, **overrides):
    root, corpus_path, _ = workspace
    case = _build(workspace, **overrides)
    manifest = bridge.build_portrait_manifest([case], manifest_dir=root)
    path = root / "fa02_portrait_manifest.json"
    exp.write_json(path, manifest)
    return path, manifest


def test_built_manifest_passes_the_harness_validator_and_all_gates(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    exp.validate_manifest(manifest, root)  # inherited, not reimplemented
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    assert report["valid"] is True, report["violations"]
    assert report["violations"] == []
    json.dumps(report)  # JSON-safe


def test_gate_rejects_case_split_disagreeing_with_corpus(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    # Case claims calibration; the registry says this person/asset is dev.
    manifest["cases"][0]["split"] = "calibration"
    exp.write_json(path, manifest)
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "split_disagrees_with_corpus" in codes
    assert "case_split_contradicts_person_split" in codes


def test_gate_rejects_locked_test_asset_in_development_manifest(tmp_path):
    manifest = _corpus_manifest(tmp_path, splits=("locked_test", "dev"))
    corpus_path = _write_corpus(tmp_path, manifest)
    arrays = _write_case_arrays(tmp_path)
    (tmp_path / "placeholder_native.tif").write_bytes(b"placeholder-native-capture")
    workspace = (tmp_path, corpus_path, arrays)
    # Build against locked_test with the escape hatch, then mislabel the run.
    case = _build(workspace, purpose="locked_comparison", allow_locked_test=True)
    fa02 = {"schema_version": 1, "purpose": "development_pilot",
            "corpus": "fixture", "parameter_selection": "fixture", "cases": [case]}
    path = tmp_path / "leaky_manifest.json"
    exp.write_json(path, fa02)
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "locked_test_used_for_development" in codes
    assert "development_purpose_non_development_split" in codes


def test_gate_rejects_missing_support_provenance(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    del manifest["cases"][0]["masks_provenance"]
    manifest["cases"][0]["annotation"].pop("reviewer")
    manifest["cases"][0]["annotation"]["annotation_source"] = "owner_approved"
    exp.write_json(path, manifest)
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "missing_masks_provenance" in codes
    assert "annotation_invalid" in codes


def test_gate_rejects_crop_array_shape_mismatch(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    manifest["cases"][0]["crop_xywh"] = [100, 80, 32, 32]
    exp.write_json(path, manifest)
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "crop_array_shape_mismatch" in codes


def test_gate_rejects_native_source_hash_mismatch(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    (root / "placeholder_native.tif").write_bytes(b"a different native capture")
    exp.write_json(path, manifest)
    report = bridge.check_promotion_gates(
        corpus_path, path, corpus_validator_kwargs={"image_probe": _probe})
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "native_source_hash_mismatch" in codes


def test_identical_supports_and_config_propagate_to_every_arm(workspace):
    """Every arm is driven by one shared support map and the frozen config."""
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    case = manifest["cases"][0]

    # Recompute the shared eligibility map the way worker() does, then confirm
    # every one of the six arms, driven by exactly that map and exactly the
    # frozen config, respects it identically: zero delta outside the support
    # and zero delta in externally forbidden pixels.
    arr = exp.load_arrays(root / case["arrays"])
    geo = exp.geometry(case, exp.DEFAULT_CONFIG)
    geo["face_width_px"] = case["face_width_px"]
    eligibility = exp.map_external_supports(arr, geo, exp.DEFAULT_CONFIG)
    assert np.count_nonzero(eligibility) > 0
    forbidden = (arr["corrected"] > 0) | (arr["protected"] > 0)
    for arm in exp.ARMS:
        out = exp.restore(arr["X"], arr["S"], arm, geo, exp.DEFAULT_CONFIG, eligibility)
        assert np.all(out["delta"][eligibility == 0] == 0)
        assert np.max(np.abs(out["delta"][forbidden])) == 0

    result = bridge.check_arm_support_uniformity(manifest, root)
    assert result["uniform"] is True, result["violations"]
    entry = result["cases"][0]
    assert entry["arms"] == list(exp.ARMS) and len(entry["arms"]) == 6
    assert entry["eligible_pixels"] == int(np.count_nonzero(eligibility))


@pytest.mark.parametrize("override_key", ("arm_overrides", "A2_dog", "per_arm"))
def test_arm_uniformity_gate_rejects_a_per_arm_override(workspace, override_key):
    """Falsifiability: a manifest implying per-arm asymmetry must be refused.

    The runner would silently IGNORE such a key, so a manifest carrying one is
    misleading evidence -- which is precisely what a gate exists to catch.
    """
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    manifest["cases"][0][override_key] = {"A4_orientation": {"gain": 0.9}}
    result = bridge.check_arm_support_uniformity(manifest, root)
    assert result["uniform"] is False
    assert any(override_key in v["detail"] for v in result["violations"])


def test_arm_uniformity_gate_rejects_a_tampered_shared_canvas(workspace):
    """A second/altered canvas breaks 'all arms see the same X/S and supports'."""
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    _write_case_arrays(root, name=manifest["cases"][0]["arrays"])  # different RNG draw? same seed
    manifest["cases"][0]["arrays_sha256"] = "0" * 64
    result = bridge.check_arm_support_uniformity(manifest, root)
    assert result["uniform"] is False
    assert any("hash differs" in v["detail"] for v in result["violations"])
    manifest["cases"][0].pop("arrays")
    result = bridge.check_arm_support_uniformity(manifest, root)
    assert result["uniform"] is False
    assert any("exactly one arrays entry" in v["detail"] for v in result["violations"])


def test_arm_uniformity_gate_rejects_a_config_off_the_frozen_lock(workspace):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    frozen = exp.canonical_hash(exp.DEFAULT_CONFIG)
    assert bridge.check_arm_support_uniformity(
        manifest, root, config_sha256=frozen)["uniform"] is True
    tuned = copy.deepcopy(exp.DEFAULT_CONFIG)
    tuned["gain"] = 0.2  # not a tuning proposal: a deliberate lock violation
    result = bridge.check_arm_support_uniformity(manifest, root, config=tuned, config_sha256=frozen)
    assert result["uniform"] is False
    assert any("frozen lock hash" in v["detail"] for v in result["violations"])


def test_gate_rejects_native_dimensions_disagreeing_with_the_corpus_probe(workspace):
    """A ROI declared inside a frame that does not exist is caught."""
    root, corpus_path, _ = workspace
    # Point the case's native source at the registered corpus asset itself, so
    # the declared native_dimensions and the probed image are comparable.
    asset_png = "placeholder_0.png"
    annotation = _annotation(source=asset_png)
    path, manifest = _portrait_manifest(workspace, annotation=annotation,
                                        native_source_reference=asset_png)
    report = bridge.check_promotion_gates(corpus_path, path)
    codes = {violation["code"] for violation in report["violations"]}
    assert report["valid"] is False
    assert "native_dimensions_disagree_with_corpus_probe" in codes


# --------------------------------------------------------------------------
# Read-only review CLI
# --------------------------------------------------------------------------


def test_review_cli_runs_read_only_and_writes_only_the_named_html(tmp_path, capsys):
    manifest = _corpus_manifest(tmp_path)  # writes tiny real PNGs
    corpus_path = _write_corpus(tmp_path, manifest)
    before = sorted(p.name for p in tmp_path.iterdir())

    html_out = tmp_path / "review" / "contact_sheet.html"
    code = review.main([str(corpus_path), "--split", "dev", "--html", str(html_out)])
    assert code == 0
    text = html_out.read_text()
    assert "placeholder_asset_0" in text and "<img" in text
    assert "placeholder_asset_1" not in text  # calibration filtered out
    printed = capsys.readouterr().out
    assert "candidates (1)" in printed

    after = sorted(p.name for p in tmp_path.iterdir())
    assert after == sorted(before + ["review"])  # nothing else created/modified
    assert json.loads(corpus_path.read_text()) == manifest  # manifest untouched


def test_review_cli_runs_with_empty_calibration_and_locked_test(tmp_path, capsys):
    manifest = _corpus_manifest(tmp_path, splits=("dev", "dev"),
                                person_ids=(["test_subject_001"], ["test_subject_002"]))
    corpus_path = _write_corpus(tmp_path, manifest)
    assert review.main([str(corpus_path), "--split", "locked_test"]) == 0
    printed = capsys.readouterr().out
    assert "candidates (0)" in printed
    assert "locked_test  assets=   0" in printed


def test_review_cli_exits_nonzero_on_invalid_manifest(tmp_path, capsys):
    manifest = _corpus_manifest(tmp_path)
    manifest["assets"][0]["split"] = "holdout"  # v2 vocabulary is gone in v3
    corpus_path = _write_corpus(tmp_path, manifest)
    assert review.main([str(corpus_path)]) == 1


def test_gate_cli_exits_nonzero_on_violation(workspace, capsys):
    root, corpus_path, _ = workspace
    path, manifest = _portrait_manifest(workspace)
    assert bridge.main([str(corpus_path), str(path)]) == 0
    manifest["cases"][0]["split"] = "calibration"
    exp.write_json(path, manifest)
    assert bridge.main([str(corpus_path), str(path)]) == 1


def test_bridge_runs_no_detector_and_no_pipeline_import():
    for name in ("fa02_portrait_manifest.py", "fa02_portrait_review.py",
                 "fa02_annotation_workbench.py", "fa02_readiness_report.py"):
        text = (ROOT / "scripts/qa" / name).read_text()
        for forbidden in ("detect_marks(", "retouch.engine", "retouch.detection",
                          "retouch.parsing", "retouch.freckle", "retouch.marks"):
            assert forbidden not in text


def test_no_real_subject_data_in_fixtures_or_template():
    """Everything shipped here must be an obvious placeholder."""
    template = (ROOT / "scripts/qa/fa02_support_annotation_template.json").read_text()
    assert "_TEMPLATE" in template and "PLACEHOLDER" in template
    assert "test_subject_001" in template
    assert "DSCF" not in template  # no real corpus filenames

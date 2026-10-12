"""FA-02 readiness: what evidence exists, what stage it supports, what is missing.

    python3 scripts/qa/fa02_readiness_report.py test_output/fa02_pilot_* \
        --json test_output/fa02_readiness.json

Scans a set of FA-02 pilot directories (each holding a ``corpus_manifest.json``,
one or more ``annotation_*.json``, and an ``fa02_manifest.json``) and answers
one question: **is there enough reviewed evidence to move FA-02 from
infrastructure validation toward a production candidate, and if not, exactly
what is missing.**

Design commitments, all of which are the point rather than incidental:

* **Counts come from declared human labels only.** The semantic categories in
  ``fa02_portrait_manifest.TEXTURE_REGION_CATEGORIES`` are counted; the legacy
  ``review_regions`` list is counted SEPARATELY and never folded in. A legacy
  region tagged ``"pore_review"`` means "someone should look here", not "pores
  are present here" -- reading its free-text tag as a label would manufacture
  ground truth out of an advisory note and is precisely the proxy this file
  refuses to compute.
* **No metric is ever substituted for a missing label.** The
  ``production_candidate_ready`` criterion "one representation beats A0 and A1
  on texture/safety" is UNCOMPUTABLE without labelled pore/hair ground truth.
  It is reported as unmet with the reason, never approximated from the run
  summaries' unlabelled ``delta_rms`` numbers.
* **Human-only criteria are named as human-only.** "Owner review passes" is a
  sign-off, not a file property. Such criteria are listed under
  ``requires_owner_action`` and are never marked met by a proxy check.
* **Subjects are unioned, not summed.** Each pilot directory carries its own
  single-asset corpus manifest, and one person legitimately appears in several
  of them. Summing per-manifest ``person_counts_by_split`` would double-count.
  The union is also the only place cross-FILE subject/split leakage can be
  detected at all -- ``validate_corpus_manifest`` sees one file at a time.

Read-only. The only file written is the JSON the caller explicitly names.

Exit codes: 0 report produced, 1 an input directory was unreadable/invalid,
2 usage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fa02_portrait_manifest import (  # noqa: E402
    ACCEPTED_ANNOTATION_SOURCES,
    CONTROL_CATEGORIES,
    GROUND_TRUTH_CATEGORIES,
    TEXTURE_REGION_CATEGORIES,
    check_promotion_gates,
    validate_support_annotation,
)
from retouch.corpus_manifest import (  # noqa: E402
    ALLOWED_SPLITS,
    corpus_split_report,
    validate_corpus_manifest,
)

# The four named stages. Ordered weakest-first; the report assigns the highest
# stage whose criteria are all met AND whose predecessors are all met.
PROMOTION_STAGES = (
    "infrastructure_validated",
    "dev_pilot_ready",
    "production_candidate_ready",
    "production_ready",
)

# Numeric targets for production_candidate_ready, from the owner's brief.
# Ranges, not single numbers: the lower bound is the gate, the upper bound is
# recorded so the report can say "8 of 8-12" rather than implying 8 is plenty.
CANDIDATE_SUBJECT_TARGET = (8, 12)
CANDIDATE_PATCH_TARGET = (30, 50)

# Strata dimensions the report tallies. Not required to be present -- a
# dimension absent from every asset is reported as a gap, never guessed.
REPORTED_STRATA = (
    "lighting", "face_scale", "pose", "skin_tone",
    "occlusion", "texture", "compression", "exposure",
    "lighting_difficulty", "faces",
)


def _criterion(name, met, detail, *, automatable=True):
    """One promotion criterion. ``met`` is None when it cannot be computed."""
    return {
        "criterion": name,
        "met": met,
        "automatically_checkable": bool(automatable),
        "detail": detail,
    }


def scan_pilot_directory(directory, *, corpus_validator_kwargs=None):
    """Read one FA-02 pilot directory into a JSON-safe evidence record.

    Never raises for a merely incomplete directory (a pilot mid-construction
    is a legitimate state to report on); it records what it found and what it
    could not read. It DOES surface validation failures, because an invalid
    corpus manifest or annotation must not be silently counted as evidence.
    """
    directory = Path(directory).expanduser().resolve()
    record = {
        "directory": str(directory),
        "problems": [],
        "corpus_valid": None,
        "assets": [],
        "annotations": [],
        "fa02_manifest_present": False,
        "gate_report": None,
    }

    corpus_path = directory / "corpus_manifest.json"
    if not corpus_path.is_file():
        record["problems"].append("corpus_manifest.json missing")
    else:
        try:
            manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
            corpus = validate_corpus_manifest(
                manifest, root=directory, **(corpus_validator_kwargs or {})
            )
        except (ValueError, OSError) as error:
            record["problems"].append(f"corpus_manifest unreadable: {error}")
        else:
            record["corpus_valid"] = bool(corpus.get("valid"))
            if not corpus["valid"]:
                record["problems"].append(
                    "corpus_manifest invalid: "
                    + ", ".join(sorted({str(i.get("code")) for i in corpus["errors"]}))
                )
            # Split counts come from the shared module, not a reimplementation.
            record["split_report"] = corpus_split_report(corpus)
            for asset in corpus.get("assets", []):
                record["assets"].append(
                    {
                        "asset_id": asset.get("asset_id"),
                        "split": asset.get("split"),
                        "person_ids": list(asset.get("person_ids") or []),
                        "strata": dict(asset.get("strata") or {}),
                        "tags": list(asset.get("tags") or []),
                    }
                )

    for annotation_path in sorted(directory.glob("annotation_*.json")):
        entry = {"path": str(annotation_path), "valid": False}
        try:
            annotation = validate_support_annotation(
                json.loads(annotation_path.read_text(encoding="utf-8"))
            )
        except (ValueError, OSError) as error:
            entry["error"] = str(error)
            record["problems"].append(f"{annotation_path.name}: {error}")
        else:
            entry.update(
                valid=True,
                asset_id=annotation["asset_id"],
                person_id=annotation["person_id"],
                annotation_source=annotation["annotation_source"],
                reviewer=annotation.get("reviewer"),
                # Advisory-only legacy list, kept strictly apart from the
                # semantic categories below. Its tags are NOT labels.
                legacy_review_regions=len(annotation["review_regions"]),
                region_counts={
                    category: len(annotation.get(category) or [])
                    for category in TEXTURE_REGION_CATEGORIES
                },
            )
        record["annotations"].append(entry)

    fa02_path = directory / "fa02_manifest.json"
    record["fa02_manifest_present"] = fa02_path.is_file()
    if fa02_path.is_file() and corpus_path.is_file():
        # Feed the standalone annotation files in by case id so the gate can
        # detect drift between them and the copy frozen into the case at build
        # time. This report counts regions from the standalone files, so a
        # silent fork between the two would mean the readiness numbers and the
        # harness's evidence describe different annotations.
        external = {}
        try:
            cases = json.loads(fa02_path.read_text(encoding="utf-8")).get("cases") or []
        except (ValueError, OSError):
            cases = []
        by_asset = {}
        for entry in record["annotations"]:
            if entry.get("valid"):
                by_asset[entry["asset_id"]] = entry["path"]
        matches_embedded = True
        for case in cases:
            if not isinstance(case, dict):
                continue
            asset_id = (case.get("corpus_source") or {}).get("asset_id")
            path = by_asset.get(asset_id)
            if path and case.get("id"):
                try:
                    raw = json.loads(Path(path).read_text(encoding="utf-8"))
                    external[case["id"]] = raw
                    embedded = case.get("annotation")
                    if embedded is not None and validate_support_annotation(
                            embedded) != validate_support_annotation(raw):
                        matches_embedded = False
                except (ValueError, OSError):
                    pass
        # Non-blocking observation, deliberately NOT a promotion criterion: a
        # case built before the standalone file gained new labels differs from
        # it for a benign build-order reason, not a provenance violation. It
        # is recorded so an owner knows to rebuild the case, nothing more.
        record["embedded_annotation_matches_standalone"] = matches_embedded

        # Re-run the gates live rather than trusting a gate_report.json that
        # may predate a schema change. A stale pass is not a pass.
        try:
            gate = check_promotion_gates(
                corpus_path, fa02_path,
                annotations=external or None,
                corpus_validator_kwargs=corpus_validator_kwargs or None,
            )
        except (ValueError, OSError, KeyError) as error:
            record["problems"].append(f"promotion gates could not run: {error}")
        else:
            record["gate_report"] = {
                "valid": bool(gate.get("valid")),
                "purpose": gate.get("purpose"),
                "case_count": int(gate.get("case_count") or 0),
                "violation_codes": sorted(
                    {str(v.get("code")) for v in gate.get("violations") or []}
                ),
                "case_ids": [
                    entry.get("case_id")
                    for entry in (gate.get("arm_support_uniformity") or {}).get("cases", [])
                ],
            }
            if not gate.get("valid"):
                record["problems"].append(
                    "promotion gates failed: "
                    + ", ".join(record["gate_report"]["violation_codes"])
                )
    return record


def aggregate(records):
    """Union per-directory evidence into corpus-wide totals.

    Subjects and assets are UNIONED by id, never summed: the four committed
    pilot directories hold four single-asset manifests covering only two
    people, and summing ``person_counts_by_split`` would report four subjects.
    """
    subjects_by_split = {split: set() for split in ALLOWED_SPLITS}
    all_subjects = set()
    assets_by_id = {}
    strata_tally = {dimension: {} for dimension in REPORTED_STRATA}
    strata_missing = {dimension: [] for dimension in REPORTED_STRATA}

    for record in records:
        for asset in record["assets"]:
            asset_id = asset["asset_id"]
            if asset_id in assets_by_id:
                continue
            assets_by_id[asset_id] = asset
            split = asset.get("split")
            for person_id in asset["person_ids"]:
                all_subjects.add(person_id)
                if split in subjects_by_split:
                    subjects_by_split[split].add(person_id)
            for dimension in REPORTED_STRATA:
                value = asset["strata"].get(dimension)
                if value is None:
                    strata_missing[dimension].append(asset_id)
                else:
                    strata_tally[dimension][value] = strata_tally[dimension].get(value, 0) + 1

    # Cross-FILE subject/split leakage. validate_corpus_manifest enforces
    # person_crosses_split within ONE manifest; with one manifest per pilot
    # directory, only this union can see a person who is dev here and
    # locked_test there. It is the guarantee the project actually cares about.
    cross_split = sorted(
        person_id
        for person_id in all_subjects
        if sum(1 for split in ALLOWED_SPLITS if person_id in subjects_by_split[split]) > 1
    )

    patches = []
    region_totals = {category: 0 for category in TEXTURE_REGION_CATEGORIES}
    legacy_total = 0
    reviewed_patches = {category: 0 for category in TEXTURE_REGION_CATEGORIES}
    human_labelled_patches = 0
    invalid_annotations = 0

    for record in records:
        for entry in record["annotations"]:
            if not entry.get("valid"):
                invalid_annotations += 1
                continue
            patches.append(entry["path"])
            legacy_total += entry["legacy_review_regions"]
            populated_ground_truth = False
            for category, count in entry["region_counts"].items():
                region_totals[category] += count
                if count:
                    reviewed_patches[category] += 1
                    if category in GROUND_TRUTH_CATEGORIES:
                        populated_ground_truth = True
            if populated_ground_truth and entry["annotation_source"] in ACCEPTED_ANNOTATION_SOURCES:
                human_labelled_patches += 1

    gate_reports = [r["gate_report"] for r in records if r.get("gate_report")]
    return {
        "pilot_directories": len(records),
        "subject_count": len(all_subjects),
        "subjects": sorted(all_subjects),
        "subjects_by_split": {
            split: sorted(subjects_by_split[split]) for split in ALLOWED_SPLITS
        },
        "subject_counts_by_split": {
            split: len(subjects_by_split[split]) for split in ALLOWED_SPLITS
        },
        "subjects_crossing_splits": cross_split,
        "asset_count": len(assets_by_id),
        "patch_count": len(patches),
        "invalid_annotation_count": invalid_annotations,
        "strata_tally": {
            dimension: dict(sorted(values.items()))
            for dimension, values in sorted(strata_tally.items())
        },
        "strata_missing_assets": {
            dimension: sorted(ids)
            for dimension, ids in sorted(strata_missing.items())
            if ids
        },
        "reviewed_region_counts": dict(sorted(region_totals.items())),
        "reviewed_patch_counts": dict(sorted(reviewed_patches.items())),
        "ground_truth_region_total": sum(
            region_totals[category] for category in GROUND_TRUTH_CATEGORIES
        ),
        "control_region_total": sum(
            region_totals[category] for category in CONTROL_CATEGORIES
        ),
        "human_labelled_patch_count": human_labelled_patches,
        # Deliberately separate from every count above. Advisory notes, not
        # labels; folding them in would fabricate ground truth.
        "legacy_review_region_total": legacy_total,
        "legacy_review_regions_are_labels": False,
        "gate_reports_run": len(gate_reports),
        "gate_reports_passing": sum(1 for g in gate_reports if g["valid"]),
        "fa02_case_count": sum(g["case_count"] for g in gate_reports),
        "problems": sorted(
            problem for record in records for problem in record["problems"]
        ),
    }


def classify_stage(totals):
    """Evaluate every stage's criteria; return criteria + the attained stage.

    A stage is attained only when every automatically-checkable criterion is
    met AND every earlier stage is attained. A criterion whose ``met`` is
    ``None`` (not computable from files) blocks attainment: the report says
    "an owner must confirm this", it never assumes yes.
    """
    dev_subjects = totals["subject_counts_by_split"]["dev"]
    calibration_subjects = totals["subject_counts_by_split"]["calibration"]
    ground_truth = totals["ground_truth_region_total"]
    noise_regions = totals["reviewed_region_counts"]["noise_regions"]

    stages = {}

    # -- infrastructure_validated ------------------------------------------
    stages["infrastructure_validated"] = [
        _criterion(
            "at_least_one_pilot_case",
            totals["fa02_case_count"] >= 1,
            f"{totals['fa02_case_count']} FA-02 case(s) across "
            f"{totals['pilot_directories']} pilot director(ies)",
        ),
        _criterion(
            "all_promotion_gates_pass",
            totals["gate_reports_run"] > 0
            and totals["gate_reports_passing"] == totals["gate_reports_run"],
            f"{totals['gate_reports_passing']}/{totals['gate_reports_run']} "
            "gate report(s) pass (re-run live, not read from disk)",
        ),
        _criterion(
            "all_annotations_valid",
            totals["invalid_annotation_count"] == 0 and totals["patch_count"] > 0,
            f"{totals['patch_count']} valid annotation(s), "
            f"{totals['invalid_annotation_count']} invalid",
        ),
        _criterion(
            "no_subject_crosses_splits",
            not totals["subjects_crossing_splits"],
            "cross-file union check: "
            + (", ".join(totals["subjects_crossing_splits"]) or "no subject appears in two splits"),
        ),
    ]

    # -- dev_pilot_ready ----------------------------------------------------
    stages["dev_pilot_ready"] = [
        _criterion(
            "multiple_subjects",
            totals["subject_count"] >= 2,
            f"{totals['subject_count']} distinct person_id(s): "
            + (", ".join(totals["subjects"]) or "none"),
        ),
        _criterion(
            "human_reviewed_texture_annotations_exist",
            totals["human_labelled_patch_count"] >= 1,
            f"{totals['human_labelled_patch_count']} patch(es) carry a populated "
            f"{'/'.join(GROUND_TRUTH_CATEGORIES)} list from a manual/owner_approved "
            f"annotation; {ground_truth} such region(s) corpus-wide. "
            f"({totals['legacy_review_region_total']} legacy review_regions exist but "
            "are advisory 'look here' notes, NOT labels, and are not counted.)",
        ),
        _criterion(
            "all_provenance_and_promotion_gates_pass",
            totals["gate_reports_run"] > 0
            and totals["gate_reports_passing"] == totals["gate_reports_run"],
            f"{totals['gate_reports_passing']}/{totals['gate_reports_run']} pass",
        ),
    ]

    # -- production_candidate_ready ----------------------------------------
    stages["production_candidate_ready"] = [
        _criterion(
            "dev_subject_count",
            dev_subjects >= CANDIDATE_SUBJECT_TARGET[0],
            f"{dev_subjects} dev subject(s); target "
            f"{CANDIDATE_SUBJECT_TARGET[0]}-{CANDIDATE_SUBJECT_TARGET[1]}",
        ),
        _criterion(
            "reviewed_patch_count",
            totals["patch_count"] >= CANDIDATE_PATCH_TARGET[0],
            f"{totals['patch_count']} reviewed patch(es); target "
            f"{CANDIDATE_PATCH_TARGET[0]}-{CANDIDATE_PATCH_TARGET[1]}",
        ),
        _criterion(
            "representative_nuisance_cases",
            noise_regions >= 1,
            f"{noise_regions} labelled noise/compression region(s); a "
            "representation cannot be shown safe against nuisance that was "
            "never annotated",
        ),
        # The one the owner explicitly forbade approximating.
        _criterion(
            "candidate_beats_A0_and_A1_on_texture_and_safety",
            None,
            "NOT COMPUTABLE. This criterion requires labelled pore/fine-hair "
            "ground truth to score 'useful texture recovered' against; the "
            "corpus currently has "
            f"{ground_truth} such label(s). The existing run summaries contain "
            "only unlabelled delta_rms/eligible-pixel numbers, which measure "
            "how much an arm CHANGED, not whether it restored real texture or "
            "amplified noise. No proxy is computed here on purpose: an arm "
            "ranking derived from unlabelled deltas would be indistinguishable "
            "from a ranking of raw gain. Populate pore_regions/fine_hair_regions "
            "and noise_regions first, then define the scoring rule BEFORE "
            "looking at the numbers.",
            automatable=False,
        ),
    ]

    # -- production_ready ---------------------------------------------------
    # Almost all of these are human sign-offs or full-pipeline runs that leave
    # no file this scanner can read. They are listed, honestly, as owner
    # actions rather than given a fabricated automatic check.
    stages["production_ready"] = [
        _criterion(
            "locked_test_split_populated_and_untouched",
            totals["subject_counts_by_split"]["locked_test"] >= 1,
            f"{totals['subject_counts_by_split']['locked_test']} locked_test "
            "subject(s). Structure only: no check can prove an owner has never "
            "seen a subject -- that requires the owner's confirmed-untouched "
            "person mapping.",
        ),
        _criterion(
            "subject_separated_calibration_completed",
            calibration_subjects >= 1,
            f"{calibration_subjects} calibration subject(s) (structural presence "
            "only; that tuning actually happened on them is not file-visible)",
        ),
        _criterion(
            "full_recipe_validation_completed", None,
            "Requires an end-to-end RetouchEngine recipe run with the candidate "
            "representation wired in. No such integration exists (FA-02 is "
            "offline-harness only) and none is authorized by this workflow.",
            automatable=False,
        ),
        _criterion(
            "locked_test_evaluated_only_after_parameters_frozen", None,
            "A sequencing property of how the work is done, not a file "
            "property. Requires a config lock dated before the locked_test run "
            "plus the owner's attestation that no locked_test result informed "
            "any parameter.",
            automatable=False,
        ),
        _criterion(
            "no_critical_identity_makeup_halo_or_defect_regressions", None,
            "Requires labelled protected/makeup/corrected regions AND a human "
            "looking at real renders. The structural guard (zero delta in "
            "forbidden pixels) is already proven and is NOT this criterion.",
            automatable=False,
        ),
        _criterion(
            "runtime_and_memory_acceptable", None,
            "Requires a benchmark of the candidate at native resolution inside "
            "the real pipeline. The harness's 512x512 offline timings do not "
            "answer it.",
            automatable=False,
        ),
        _criterion(
            "owner_review_passes", None,
            "Inherently a human sign-off. There is no proxy for it and none is "
            "computed.",
            automatable=False,
        ),
    ]

    attained = None
    for stage in PROMOTION_STAGES:
        if all(criterion["met"] is True for criterion in stages[stage]):
            attained = stage
        else:
            break

    next_stage = None
    if attained is None:
        next_stage = PROMOTION_STAGES[0]
    else:
        index = PROMOTION_STAGES.index(attained)
        if index + 1 < len(PROMOTION_STAGES):
            next_stage = PROMOTION_STAGES[index + 1]

    missing = []
    requires_owner_action = []
    if next_stage:
        for criterion in stages[next_stage]:
            if criterion["met"] is True:
                continue
            if criterion["automatically_checkable"]:
                missing.append(criterion)
            else:
                requires_owner_action.append(criterion)

    return {
        "stages": PROMOTION_STAGES,
        "criteria": stages,
        "attained_stage": attained,
        "next_stage": next_stage,
        "missing_for_next_stage": missing,
        "next_stage_requires_owner_action": requires_owner_action,
    }


def build_readiness_report(directories, *, corpus_validator_kwargs=None):
    """Full JSON-safe readiness report over a set of pilot directories."""
    records = [
        scan_pilot_directory(directory, corpus_validator_kwargs=corpus_validator_kwargs)
        for directory in directories
    ]
    totals = aggregate(records)
    classification = classify_stage(totals)
    report = {
        "report": "fa02_readiness",
        "schema_version": 1,
        "totals": totals,
        "promotion": classification,
        "directories": records,
        "disclaimer": (
            "Counts reflect DECLARED human annotations only. No region here was "
            "produced by image analysis, and no unlabelled metric was substituted "
            "for a missing label. Criteria marked automatically_checkable=false "
            "cannot be settled by any file on disk and require owner action."
        ),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="+", type=Path,
                        help="FA-02 pilot directories to scan")
    parser.add_argument("--json", type=Path, help="Write the full report here")
    args = parser.parse_args(argv)

    missing = [str(d) for d in args.directories if not Path(d).expanduser().is_dir()]
    if missing:
        print("not a directory: " + ", ".join(missing), file=sys.stderr)
        return 2

    report = build_readiness_report(args.directories)
    totals = report["totals"]
    promotion = report["promotion"]

    print(f"pilot directories: {totals['pilot_directories']}")
    print(f"subjects:          {totals['subject_count']} "
          f"({', '.join(totals['subjects']) or 'none'})")
    for split in ALLOWED_SPLITS:
        print(f"  {split:12s} subjects={totals['subject_counts_by_split'][split]}")
    print(f"patches:           {totals['patch_count']}")
    print(f"FA-02 cases:       {totals['fa02_case_count']}")
    print(f"gates:             {totals['gate_reports_passing']}/{totals['gate_reports_run']} pass")
    print("reviewed regions by category:")
    for category, count in totals["reviewed_region_counts"].items():
        print(f"  {category:28s} {count}")
    print(f"  {'(legacy review_regions)':28s} {totals['legacy_review_region_total']}"
          "  <- advisory only, NOT labels")
    print(f"\nattained stage:    {promotion['attained_stage']}")
    print(f"next stage:        {promotion['next_stage']}")
    for criterion in promotion["missing_for_next_stage"]:
        print(f"  MISSING  {criterion['criterion']}: {criterion['detail']}")
    for criterion in promotion["next_stage_requires_owner_action"]:
        print(f"  OWNER    {criterion['criterion']}: {criterion['detail']}")
    if totals["problems"]:
        print("\nproblems:")
        for problem in totals["problems"]:
            print(f"  {problem}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if totals["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FA-02 scoring readiness: can the frozen scoring contract even RUN on this corpus?

    python3 scripts/qa/fa02_scoring_readiness_report.py test_output/fa02_scoring_lock.json \
        test_output/fa02_pilot_* --json test_output/fa02_scoring_readiness.json

This is the companion to ``fa02_readiness_report.py`` and deliberately does NOT
replace it. That report answers "does enough reviewed evidence exist to promote
the corpus". This one answers a narrower, mechanical question:

    Given the frozen scoring lock, which scoring terms are COMPUTABLE on the
    committed evidence, which are blocked by a missing input, what do the hard
    safety disqualifiers actually say, and is the baseline-beat test runnable at
    all?

The expected honest answer today is "almost nothing is computable", and this
script is built to say that loudly rather than emit all-null output that looks
like a bug. Specifically:

* A term with no input reports ``blocked`` and NAMES the missing input. It never
  reports 0.
* A disqualifier with no data reports ``not_applicable``. It never reports pass.
  A case where every gate is not_applicable is INADMISSIBLE, so "no safety data"
  can never be laundered into "safe".
* The baseline-beat test reports ``not_computable`` with the reason, never a
  win or a loss, when nothing is scorable.
* Only THIS ROUND'S frozen candidate arms are put to the baseline-beat test
  (read from the lock's ``evaluated_candidate_arms``). Arms held in reserve are
  still listed, with verdict ``reserved_not_evaluated`` and the reason, so an
  excluded arm is never indistinguishable from an unimplemented one.

Artifact classes matter here and are reported per term. Six of the seven terms
read ``run/summary.json``, which is COMMITTED. The makeup-edge term reads
``run/<case>/<arm>/signed_arrays.npz``, which is GITIGNORED (.gitignore:7
``test_output/``) -- so on a fresh clone it is unavailable even on the three
patches that carry makeup-edge labels, while on the authoring machine it is
computable. The report states which situation produced it.

Read-only apart from the JSON the caller names.

Exit codes: 0 report produced, 1 an input was unreadable, 2 usage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fa02_scoring_contract as contract  # noqa: E402
import fa02_texture_representation_experiment as exp  # noqa: E402
from fa02_portrait_manifest import (  # noqa: E402
    GROUND_TRUTH_CATEGORIES,
    validate_support_annotation,
)
# Reuse the existing criterion SHAPE so both reports speak the same dialect and
# this file cannot drift from it. Importing rather than redefining also means
# the existing report's tests keep covering the shape.
from fa02_readiness_report import _criterion  # noqa: E402

ALL_TERMS = tuple(contract.RECOVERY_TERMS) + tuple(contract.PENALTY_TERMS)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta_luma(run_dir, case_id, arm):
    """Luma delta plane for one case/arm, or None when the float arrays are absent.

    These arrays are gitignored, so None is the NORMAL result on a fresh clone
    and must not be treated as an error. Returning None makes the makeup-edge
    term report ``blocked`` with the right reason instead of crashing.
    """
    path = Path(run_dir) / case_id / arm / "signed_arrays.npz"
    if not path.is_file():
        return None, "signed_arrays.npz not present (gitignored float arrays)"
    try:
        with np.load(path, allow_pickle=False) as data:
            if "delta" not in data.files:
                return None, "signed_arrays.npz has no 'delta' array"
            return exp.luma(np.asarray(data["delta"], dtype=np.float32)), "read from local run"
    except (OSError, ValueError) as error:
        return None, "unreadable signed_arrays.npz: {0}".format(error)


def scan_run_directory(pilot_dir):
    """Collect one pilot directory's committed scoring inputs. Never raises."""
    pilot_dir = Path(pilot_dir).expanduser().resolve()
    record = {
        "directory": str(pilot_dir),
        "problems": [],
        "run_present": False,
        "complete": None,
        "config_sha256": None,
        "manifest_sha256": None,
        "results": [],
        "annotations_by_case": {},
        "cases": [],
    }

    run_dir = pilot_dir / "run"
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        record["problems"].append("run/summary.json missing")
        return record
    record["run_present"] = True

    try:
        summary = _load_json(summary_path)
    except (OSError, ValueError) as error:
        record["problems"].append("run/summary.json unreadable: {0}".format(error))
        return record
    record["results"] = summary.get("results") or []
    record["purpose"] = summary.get("purpose")

    # Read the run's identity from COMPLETE.json, which already carries both
    # hashes, rather than recomputing them from the manifest. This is the
    # ordering evidence the scoring lock must be tied to.
    complete_path = run_dir / "COMPLETE.json"
    if complete_path.is_file():
        try:
            complete = _load_json(complete_path)
        except (OSError, ValueError) as error:
            record["problems"].append("run/COMPLETE.json unreadable: {0}".format(error))
        else:
            record["complete"] = complete
            record["config_sha256"] = complete.get("config_sha256")
            record["manifest_sha256"] = complete.get("manifest_sha256")
    else:
        record["problems"].append("run/COMPLETE.json missing: the run may be partial")

    # Case metadata: whether pore/profile inputs exist at all, plus the
    # annotation that carries the semantic texture categories.
    manifest_path = pilot_dir / "fa02_manifest.json"
    annotations = {}
    for annotation_path in sorted(pilot_dir.glob("annotation_*.json")):
        try:
            annotations[annotation_path.name] = validate_support_annotation(
                _load_json(annotation_path))
        except (OSError, ValueError) as error:
            record["problems"].append("{0}: {1}".format(annotation_path.name, error))

    if manifest_path.is_file():
        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError) as error:
            record["problems"].append("fa02_manifest.json unreadable: {0}".format(error))
        else:
            for case in manifest.get("cases") or []:
                if not isinstance(case, dict):
                    continue
                case_id = case.get("id")
                asset_id = (case.get("corpus_source") or {}).get("asset_id")
                annotation = None
                for value in annotations.values():
                    if value.get("asset_id") == asset_id:
                        annotation = value
                        break
                if annotation is None:
                    annotation = case.get("annotation")
                    if annotation is not None:
                        try:
                            annotation = validate_support_annotation(annotation)
                        except ValueError:
                            annotation = None
                record["annotations_by_case"][case_id] = annotation
                record["cases"].append({
                    "case_id": case_id,
                    "split": case.get("split"),
                    "person_ids": list(case.get("person_ids") or []),
                    "arrays": case.get("arrays"),
                    # THE structural gap, measured rather than assumed.
                    "pore_input_count": len(case.get("pores") or []),
                    "profile_input_count": len(case.get("profiles") or []),
                    "roi_count": len(case.get("rois") or []),
                    "makeup_edge_region_count": len(
                        (annotation or {}).get("makeup_edge_regions") or []),
                    "ground_truth_region_count": sum(
                        len((annotation or {}).get(category) or [])
                        for category in GROUND_TRUTH_CATEGORIES),
                })
    else:
        record["problems"].append("fa02_manifest.json missing")
    return record


def score_directory(record, *, measure_texture=True):
    """Apply the frozen contract to one pilot directory's results."""
    run_dir = Path(record["directory"]) / "run"
    annotations = record["annotations_by_case"]
    by_case = {}

    for result in record["results"]:
        case_id = result.get("case_id")
        arm = result.get("arm")
        annotation = annotations.get(case_id) or {}
        makeup_regions = annotation.get("makeup_edge_regions") or []
        delta_luma, delta_note = (None, "not needed: no makeup_edge_regions")
        if makeup_regions:
            delta_luma, delta_note = _delta_luma(run_dir, case_id, arm)

        terms = contract.compute_terms(result, makeup_regions, delta_luma)
        aggregate = contract.aggregate_case_score(terms)
        gates = contract.evaluate_disqualifiers(result, makeup_regions, delta_luma)

        by_case.setdefault(case_id, {})[arm] = {
            "arm": arm,
            "terms": terms,
            "aggregate": aggregate,
            "disqualifiers": gates,
            "makeup_delta_source": delta_note,
            # Explicitly an observation, never a gate -- see
            # SCORING_RULE["explicitly_not_disqualifiers"].
            "observation_protected_signed_contrast_recovery":
                result.get("protected_signed_contrast_recovery"),
        }

    # Eligibility is a property of the CASE (the face/patch), not of an arm.
    eligibility = {}
    for case in record["cases"]:
        case_id = case["case_id"]
        annotation = annotations.get(case_id) or {}
        texture_std = None
        texture_note = "not measured"
        if measure_texture and case.get("arrays"):
            arrays_path = Path(record["directory"]) / case["arrays"]
            if arrays_path.is_file():
                try:
                    arr = exp.load_arrays(arrays_path)
                    texture_std = contract.highpass_std(arr["X"], arr.get("allow"))
                    texture_note = (
                        "highpass_std(sigma={0}) on native X inside the allow support".format(
                            contract.SCORING_RULE["eligibility"]["highpass_sigma_px"]["value"])
                    )
                except (OSError, ValueError, KeyError) as error:
                    texture_note = "arrays unreadable: {0}".format(error)
            else:
                texture_note = (
                    "case arrays NPZ not present (gitignored float arrays); source "
                    "texture cannot be measured from committed metadata alone")
        eligibility[case_id] = dict(
            contract.evaluate_eligibility(texture_std, annotation, case_id=case_id),
            measurement_note=texture_note,
        )

    return {"by_case": by_case, "eligibility": eligibility}


def summarize_terms(scored_directories):
    """Corpus-wide: which terms are computable, which are blocked and why."""
    tally = {}
    for term in ALL_TERMS:
        spec = contract.SCORING_RULE["terms"][term]
        tally[term] = {
            "term": term,
            "weight": spec["weight"]["value"],
            "required": spec["required"],
            "artifact_class": spec["artifact_class"],
            "expected_input": spec["input_shape"],
            "computable_case_arm_count": 0,
            "blocked_case_arm_count": 0,
            "missing_inputs": {},
            "status": "blocked",
        }
    for scored in scored_directories:
        for arms in scored["by_case"].values():
            for entry in arms.values():
                for term, result in entry["terms"].items():
                    record = tally[term]
                    if result["available"]:
                        record["computable_case_arm_count"] += 1
                    else:
                        record["blocked_case_arm_count"] += 1
                        key = result["missing_input"]
                        record["missing_inputs"][key] = record["missing_inputs"].get(key, 0) + 1
    for record in tally.values():
        if record["computable_case_arm_count"] and not record["blocked_case_arm_count"]:
            record["status"] = "computable"
        elif record["computable_case_arm_count"]:
            record["status"] = "partially_computable"
        record["missing_inputs"] = dict(sorted(record["missing_inputs"].items()))
    return tally


def summarize_disqualifiers(scored_directories):
    """Per-gate corpus tally plus the admissibility conclusion."""
    gate_tally = {}
    admissible = []
    inadmissible = []
    disqualified = []
    for scored in scored_directories:
        for case_id, arms in sorted(scored["by_case"].items()):
            for arm, entry in sorted(arms.items()):
                gates = entry["disqualifiers"]
                label = {"case_id": case_id, "arm": arm}
                if gates["disqualified"]:
                    disqualified.append(dict(label, failed=gates["failed"]))
                elif gates["admissible"]:
                    admissible.append(label)
                else:
                    inadmissible.append(dict(
                        label, reason="every gate not_applicable: no safety data"))
                for gate in gates["gates"]:
                    bucket = gate_tally.setdefault(
                        gate["gate"], {"pass": 0, "fail": 0, "not_applicable": 0})
                    bucket[gate["status"]] += 1
    return {
        "by_gate": dict(sorted(gate_tally.items())),
        "admissible_case_arms": admissible,
        "inadmissible_case_arms": inadmissible,
        "disqualified_case_arms": disqualified,
        "admissible_count": len(admissible),
        "inadmissible_count": len(inadmissible),
        "disqualified_count": len(disqualified),
    }


def run_baseline_tests(scored_directories, candidate_arms=None, reserved_arms=None):
    """Run the margin test for THIS ROUND'S candidate arms across the corpus.

    ``candidate_arms`` is the frozen candidate set minus the baselines, i.e. the
    arms a verdict is actually being sought for. It defaults to the module
    constant but is normally supplied by :func:`build_report` from the LOCK, so
    the report evaluates what the frozen contract says rather than whatever the
    harness happens to implement today. That distinction is the whole point of a
    lock: ``exp.ARMS`` can grow without silently widening this round's decision.

    ``reserved_arms`` are still REPORTED -- with verdict ``reserved_not_evaluated``
    and the owner instruction as the reason -- rather than omitted. Dropping them
    silently would leave a reader unable to tell an excluded arm from one that
    was never implemented.
    """
    if candidate_arms is None:
        candidate_arms = contract.EVALUATED_CANDIDATE_ARMS
    if reserved_arms is None:
        reserved_arms = contract.RESERVED_ARMS_PENDING_FAILURE

    scores = {}
    for scored in scored_directories:
        for case_id, arms in scored["by_case"].items():
            for arm, entry in arms.items():
                scores.setdefault(arm, {})[case_id] = entry["aggregate"]["score"]
    baselines = {arm: scores.get(arm, {}) for arm in contract.BASELINE_ARMS}

    results = {}
    for arm in candidate_arms:
        results[arm] = contract.compare_against_baselines(scores.get(arm, {}), baselines)
    for arm in reserved_arms:
        # Deliberately NOT run through compare_against_baselines: a reserved arm
        # has no verdict this round, and emitting "not_computable" for it would
        # conflate "we lack the data" with "we chose not to ask".
        results[arm] = {
            "verdict": "reserved_not_evaluated",
            "reason": (
                "Excluded from this round's candidate set by the frozen scoring lock. "
                + contract.RESERVED_ARMS_NOTE
            ),
            "comparable_cases": 0,
            "per_case": [],
            "evaluated_this_round": False,
        }
    return results


def build_report(lock_path, directories, *, measure_texture=True):
    """Full JSON-safe scoring-readiness report."""
    lock = contract.load_scoring_lock(lock_path)
    records = [scan_run_directory(directory) for directory in directories]
    scored = [score_directory(record, measure_texture=measure_texture) for record in records]

    # This round's decision scope comes from the LOCK, falling back to the module
    # constants for a lock frozen before the candidate set was narrowed.
    candidate_arms = lock.get("evaluated_candidate_arms")
    if candidate_arms is None:
        candidate_arms = list(contract.EVALUATED_CANDIDATE_ARMS)
    reserved_arms = lock.get("reserved_arms_pending_failure")
    if reserved_arms is None:
        reserved_arms = list(contract.RESERVED_ARMS_PENDING_FAILURE)

    terms = summarize_terms(scored)
    gates = summarize_disqualifiers(scored)
    baseline = run_baseline_tests(scored, candidate_arms, reserved_arms)

    case_rows = []
    for record, result in zip(records, scored):
        for case in record["cases"]:
            case_id = case["case_id"]
            case_rows.append(dict(
                case,
                directory=record["directory"],
                eligibility=result["eligibility"].get(case_id),
            ))

    total_pore_inputs = sum(c["pore_input_count"] for c in case_rows)
    total_profile_inputs = sum(c["profile_input_count"] for c in case_rows)
    eligible_cases = [c["case_id"] for c in case_rows
                      if (c["eligibility"] or {}).get("eligible")]

    # Runs whose identity the scoring lock would be tied to, for the
    # sequencing gate. Read from COMPLETE.json, not recomputed.
    run_references = [
        {
            "directory": record["directory"],
            "config_sha256": record["config_sha256"],
            "manifest_sha256": record["manifest_sha256"],
            "scoring_sha256": lock["scoring_sha256"],
        }
        for record in records if record["config_sha256"]
    ]
    sequencing = contract.check_scoring_sequencing(lock, run_references)

    blocking = []
    if total_pore_inputs == 0:
        blocking.append(_criterion(
            "pore_scoring_inputs_exist", False,
            'ZERO case["pores"] entries corpus-wide across {0} case(s). The scoring '
            "rule's highest-weighted term (pore_recovery, weight {1}) cannot be "
            "computed for any arm. Note the shape gap: the annotation schema carries "
            "pore_regions as RECTANGLES {{id, xywh, tags}}, while exp.score() consumes "
            'case["pores"] as POINTS {{id, center, radius}}. Nothing converts one to '
            "the other, and pore_regions is itself empty corpus-wide (commit 427ae49 "
            "found no resolvable pores in this source material), so a converter would "
            "have nothing to convert.".format(
                len(case_rows), contract.SCORING_RULE["terms"]["pore_recovery"]["weight"]["value"])))
    if total_profile_inputs == 0:
        blocking.append(_criterion(
            "fine_hair_scoring_inputs_exist", False,
            'ZERO case["profiles"] entries corpus-wide. fine_hair_recovery (weight {0}) '
            "cannot be computed. One fine_hair_regions RECTANGLE exists (DSCF1058), but "
            'case["profiles"] needs LINE SEGMENTS {{p0, p1}} perpendicular to the '
            "strand; a bounding box does not encode strand orientation, so deriving the "
            "segment from it would invent the one fact the measurement depends on.".format(
                contract.SCORING_RULE["terms"]["fine_hair_recovery"]["weight"]["value"])))
    if gates["admissible_count"] == 0:
        blocking.append(_criterion(
            "any_case_arm_cleared_the_safety_gates", False,
            "{0} case/arm combination(s) are INADMISSIBLE because every hard gate "
            "returned not_applicable (no forbidden pixels, no corrected mask, no rings, "
            "no measurable makeup edge). No case/arm has demonstrated safety; absence "
            "of data is not a pass.".format(gates["inadmissible_count"])))
    if not eligible_cases:
        blocking.append(_criterion(
            "any_case_is_eligible_for_restoration", False,
            "0 of {0} case(s) clear the eligibility contract, so the frozen policy says "
            "ABSTAIN on every one of them. This is the contract working as written, not "
            "a failure to compute.".format(len(case_rows))))

    computable = [t for t in terms.values() if t["status"] != "blocked"]
    return {
        "report": "fa02_scoring_readiness",
        "schema_version": 1,
        "scoring_lock": {
            "path": str(Path(lock_path).resolve()),
            "scoring_sha256": lock["scoring_sha256"],
            "frozen_at_utc": lock.get("frozen_at_utc"),
            "selection": lock.get("selection"),
            "rule_id": lock["scoring_rule"]["rule_id"],
        },
        "contract_properties": lock.get("contract_properties"),
        "totals": {
            "pilot_directories": len(records),
            "case_count": len(case_rows),
            # Still all six: the harness is unchanged by the candidate narrowing.
            "arm_count": len(exp.ARMS),
            "candidate_arms_this_round": list(lock.get("candidate_arms_this_round")
                                              or contract.FROZEN_CANDIDATE_SET),
            "evaluated_candidate_arms": list(candidate_arms),
            "reserved_arms_pending_failure": list(reserved_arms),
            "case_arm_results": sum(len(r["results"]) for r in records),
            "pore_scoring_inputs": total_pore_inputs,
            "profile_scoring_inputs": total_profile_inputs,
            "computable_terms": sorted(t["term"] for t in computable),
            "blocked_terms": sorted(t["term"] for t in terms.values()
                                    if t["status"] == "blocked"),
            "scorable_case_arms": sum(
                1 for s in scored for arms in s["by_case"].values()
                for e in arms.values() if e["aggregate"]["scorable"]),
            "eligible_cases": eligible_cases,
            "abstaining_cases": [c["case_id"] for c in case_rows
                                 if not (c["eligibility"] or {}).get("eligible")],
        },
        "terms": terms,
        "disqualifiers": gates,
        "baseline_beat_tests": baseline,
        "eligibility": [
            {"case_id": c["case_id"], "directory": c["directory"], **(c["eligibility"] or {})}
            for c in case_rows
        ],
        "cases": case_rows,
        "sequencing_gate": sequencing,
        "run_references": run_references,
        "blocking_for_first_production_candidate_comparison": blocking,
        "problems": sorted(p for r in records for p in r["problems"]),
        "disclaimer": (
            "This report certifies MECHANICAL COMPUTABILITY of a frozen scoring rule. It "
            "does not promote the corpus: fa02_readiness_report.py's "
            "production_candidate_ready / production_ready criteria apply unchanged and "
            "most remain unmet. Every threshold in the scoring lock is PROVISIONAL and "
            "none has been validated against study data."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lock", type=Path, help="Frozen scoring lock JSON")
    parser.add_argument("directories", nargs="+", type=Path,
                        help="FA-02 pilot directories to score")
    parser.add_argument("--json", type=Path, help="Write the full report here")
    parser.add_argument("--no-texture-measurement", action="store_true",
                        help="Skip reading case NPZ arrays (metadata-only mode)")
    args = parser.parse_args(argv)

    if not Path(args.lock).is_file():
        print("not a file: {0}".format(args.lock), file=sys.stderr)
        return 2
    missing = [str(d) for d in args.directories if not Path(d).expanduser().is_dir()]
    if missing:
        print("not a directory: " + ", ".join(missing), file=sys.stderr)
        return 2

    try:
        report = build_report(args.lock, args.directories,
                              measure_texture=not args.no_texture_measurement)
    except (OSError, ValueError) as error:
        print("could not build report: {0}".format(error), file=sys.stderr)
        return 1

    totals = report["totals"]
    print("scoring_sha256:    " + report["scoring_lock"]["scoring_sha256"])
    print("pilot directories: {0}".format(totals["pilot_directories"]))
    print("cases / case-arms: {0} / {1}".format(totals["case_count"], totals["case_arm_results"]))
    print('case["pores"] inputs:    {0}'.format(totals["pore_scoring_inputs"]))
    print('case["profiles"] inputs: {0}'.format(totals["profile_scoring_inputs"]))
    print("\nscoring terms:")
    for term in ALL_TERMS:
        record = report["terms"][term]
        print("  {0:32s} {1:22s} computable={2:3d} blocked={3:3d}".format(
            term, record["status"], record["computable_case_arm_count"],
            record["blocked_case_arm_count"]))
    print("\ndisqualifier gates (three-valued):")
    for gate, counts in report["disqualifiers"]["by_gate"].items():
        print("  {0:32s} pass={1:3d} fail={2:3d} not_applicable={3:3d}".format(
            gate, counts["pass"], counts["fail"], counts["not_applicable"]))
    print("  admissible case/arms:   {0}".format(report["disqualifiers"]["admissible_count"]))
    print("  INADMISSIBLE (no data): {0}".format(report["disqualifiers"]["inadmissible_count"]))
    print("\ncandidate set (frozen for this round):")
    print("  baselines:            " + ", ".join(contract.BASELINE_ARMS))
    print("  evaluated candidates: " + ", ".join(totals["evaluated_candidate_arms"]))
    print("  RESERVED (not evaluated this round): "
          + (", ".join(totals["reserved_arms_pending_failure"]) or "none"))
    print("    " + contract.RESERVED_ARMS_NOTE)
    print("\nbaseline-beat tests:")
    for arm, result in report["baseline_beat_tests"].items():
        print("  {0:22s} {1:22s} {2}".format(arm, result["verdict"], result["reason"]))
    print("\neligibility (abstention contract):")
    print("  eligible:  {0}".format(", ".join(totals["eligible_cases"]) or "NONE"))
    print("  abstaining: {0} case(s)".format(len(totals["abstaining_cases"])))
    print("\nblocking for a first production-candidate comparison:")
    for criterion in report["blocking_for_first_production_candidate_comparison"]:
        print("  MISSING  {0}".format(criterion["criterion"]))
    if report["problems"]:
        print("\nproblems:")
        for problem in report["problems"]:
            print("  " + problem)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print("\nwrote " + str(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

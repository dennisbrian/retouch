"""Step 2 of docs/plans/PLAN_RETOUCH_GAP_RESEARCH_2026_09_15.md: freeze the
current production engine's *decisions* on the step-1 inventory.

    .venv/bin/python scripts/qa/baseline_freeze.py test_output/merged_inventory_2026_09_15.json \\
        --recipe natural --out test_output/baseline_freeze_2026_09_15.json

This is explicitly NOT a mistakes table. Step 1 established that no
per-region ground-truth labels exist for these assets (corpus_manifest v3's
identity_marks is free text, not region geometry) -- there is nothing to
score a decision against yet. What this script freezes is what the pipeline
actually did and why, keyed by source sha256, so a later reviewed-label pass
has something concrete to compare against:

  - code/model provenance (git commit, recipe, quality mode, resolution)
  - QA detector scores + thresholds (signals, not adjudicated defects)
  - Safe Auto stage decisions (apply/dampen/skip + reason + evidence)
  - P7/FA02 diagnostics, recorded as explicitly present or explicitly absent
  - per-stage timings

Every row is stamped dev/fa02-tuning provenance and an "unknowns" list
describing what could NOT be answered (e.g. no ground truth, P7 diagnostics
empty). Do not read a QA flag or Safe Auto skip as a confirmed error --
report the score, not a verdict.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from retouch import RetouchEngine  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def freeze_one_asset(
    asset: Mapping[str, Any],
    *,
    root: Path,
    recipe: str,
    quality: str,
    engine: RetouchEngine,
) -> Dict[str, Any]:
    image_path = root / str(asset["path"])
    image = cv2.imread(str(image_path))
    unknowns: List[str] = []

    row: Dict[str, Any] = {
        "asset_id": asset.get("asset_id"),
        "sha256": asset.get("sha256"),
        "person_ids": asset.get("person_ids", []),
        "split": asset.get("split"),
        "source_provenance_note": "all assets are dev-split, previously used for FA-02 tuning -- "
        "not independent qualification evidence",
        "code_provenance": {
            "git_commit": _git_commit(),
            "recipe": recipe,
            "quality": quality,
        },
    }

    if image is None:
        row["status"] = "image_unreadable"
        row["unknowns"] = ["source image failed to decode -- no decision recorded"]
        return row

    row["image_shape"] = list(image.shape)
    row["resolution_note"] = (
        "captured at native resolution (quality='full' branches to native for "
        "reshaping/per-face/global stages; only detection+segmentation run at "
        "the 2048px proxy) -- ops known to no-op below full res would not show "
        "here even if they no-op at 2048px screening"
    )

    started = time.time()
    result = engine.process(image, recipe=recipe, quality=quality)
    wall_seconds = time.time() - started

    row["status"] = "processed"
    row["face_count"] = int(result.face_count)
    if result.face_count == 0:
        unknowns.append("zero faces detected -- per-face pipeline did not run on this asset")

    # result.qa is FLAGGED-ONLY (see qa_detectors.run_qa_with_evidence's own
    # docstring) -- do not use it to compute a per-detector flag rate, since a
    # detector absent from every asset's qa list would then be silently
    # dropped from the denominator instead of counted as flagged=False.
    # result.qa_evidence is the complete map (every ALL_DETECTOR_NAMES entry,
    # every asset), which is what summarize_qa_signals below actually needs.
    row["qa_signals"] = [_json_safe(warning) for warning in (result.qa or [])]
    row["qa_signals_note"] = (
        "QA detector scores/flags are heuristic signals against fixed thresholds, "
        "not reviewed defect labels -- a flagged=True entry is not a confirmed mistake. "
        "This list is FLAGGED-ONLY; use qa_evidence for the complete per-detector record."
    )

    row["safe_auto_decisions"] = _json_safe(result.safe_auto_decisions or [])

    p7 = result.p7_diagnostics or {}
    row["p7_diagnostics"] = _json_safe(p7)
    if not p7:
        unknowns.append("p7_diagnostics empty -- no P7 ownership decision recorded for this asset")

    fa02 = result.fa02_diagnostics or []
    row["fa02_diagnostics"] = _json_safe(fa02)
    if not fa02 or all(entry is None for entry in fa02):
        unknowns.append("fa02_diagnostics all None -- texture-eligibility path did not record a decision")

    row["qa_evidence_keys"] = sorted((result.qa_evidence or {}).keys())
    row["qa_evidence"] = _json_safe(result.qa_evidence or {})
    row["qa_evidence_note"] = (
        "includes provisional support-size heuristics (e.g. P8 count/800, "
        "min(feature,surround)/500) documented as uncalibrated confidence, not "
        "correctness probability -- see RESEARCH_RETOUCH_REMAINING_GAPS_2026_09_15.md #4"
    )

    row["timings_seconds"] = dict(result.timings or {})
    row["wall_seconds"] = round(wall_seconds, 3)

    row["ground_truth_available"] = False
    unknowns.append(
        "no reviewed per-region ground-truth labels exist for this asset (v3 "
        "identity_marks is free text, not region geometry) -- accuracy cannot "
        "be scored from this row alone"
    )
    row["unknowns"] = unknowns
    return row


def summarize_qa_signals(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-detector flag rate across assets -- a cross-asset pattern, not a verdict.

    Reads ``row["qa_evidence"]`` (the complete per-detector map, every asset),
    NOT ``row["qa_signals"]`` -- the latter is the engine's flagged-only list
    (qa_detectors.run_qa_with_evidence's own contract), so a detector that is
    never flagged would silently vanish from both numerator and denominator
    instead of counting as a real "not flagged" observation. Only detectors
    with status "checked-pass"/"checked-flagged" count toward total_count;
    "not-run"/"unavailable" entries are excluded rather than treated as a
    flagged=False vote, since that status means the detector was not actually
    evaluated on that asset.

    A detector flagged on every asset is worth surfacing loudly (either a real
    shared defect at this recipe/resolution, or a miscalibrated/broken
    detector) but this function does not decide which; that needs reviewed
    labels, which step 1 established do not exist yet.
    """
    per_detector: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for detector, evidence in (row.get("qa_evidence") or {}).items():
            status = evidence.get("status")
            if status not in ("checked-pass", "checked-flagged"):
                continue
            entry = per_detector.setdefault(detector, {"flagged_count": 0, "total_count": 0, "scores": []})
            entry["total_count"] += 1
            entry["scores"].append(evidence.get("score"))
            if evidence.get("flagged"):
                entry["flagged_count"] += 1
    for detector, entry in per_detector.items():
        entry["flagged_fraction"] = round(entry["flagged_count"] / entry["total_count"], 3) if entry["total_count"] else None
    return dict(sorted(per_detector.items()))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="merged corpus_manifest.json from corpus_inventory.py")
    parser.add_argument("--recipe", default="natural")
    parser.add_argument("--quality", default="full", choices=["full", "draft"])
    parser.add_argument("--out", default=None, help="write the freeze JSON here (not committed)")
    args = parser.parse_args(argv[1:])

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets", [])
    if not assets:
        print("manifest has no assets", file=sys.stderr)
        return 2

    print(f"freezing {len(assets)} asset(s) at recipe={args.recipe!r} quality={args.quality!r}")
    print(f"git commit: {_git_commit()}")
    print("NOTE: this records decisions, not mistakes -- no ground truth exists yet\n")

    engine = RetouchEngine()
    rows: List[Dict[str, Any]] = []
    try:
        for index, asset in enumerate(assets, start=1):
            print(f"[{index}/{len(assets)}] {asset.get('asset_id')} ...", flush=True)
            row = freeze_one_asset(
                asset, root=manifest_path.parent, recipe=args.recipe, quality=args.quality, engine=engine
            )
            rows.append(row)
            status = row.get("status")
            faces = row.get("face_count", "-")
            wall = row.get("wall_seconds", "-")
            print(f"    status={status} faces={faces} wall={wall}s unknowns={len(row.get('unknowns', []))}")
    finally:
        engine.close()

    qa_summary = summarize_qa_signals(rows)
    multi_face_assets = [row["asset_id"] for row in rows if row.get("face_count", 0) > 1]

    freeze = {
        "schema": "retouch_baseline_freeze_v1",
        "generated_from_manifest": str(manifest_path),
        "recipe": args.recipe,
        "quality": args.quality,
        "git_commit": _git_commit(),
        "asset_count": len(rows),
        "qa_signal_summary": qa_summary,
        "qa_signal_summary_note": "per-detector flag rate across assets -- a shared-pattern signal, "
        "not a verdict; a detector flagged on every asset needs review, not automatic distrust "
        "of either the detector or the images",
        "multi_face_assets": multi_face_assets,
        "multi_face_assets_note": "manifest treats these as single-subject assets but the pipeline "
        "detected >1 face -- exactly the P7 ownership case step 3 needs to look at; "
        "p7_diagnostics was empty on all rows in this freeze, so no ownership decision is recorded",
        "rows": rows,
    }

    print()
    processed = sum(1 for row in rows if row.get("status") == "processed")
    zero_face = sum(1 for row in rows if row.get("face_count") == 0)
    flagged_qa = sum(1 for row in rows for w in row.get("qa_signals", []) if w.get("flagged"))
    print(f"summary: {processed}/{len(rows)} processed, {zero_face} zero-face, "
          f"{flagged_qa} QA signals flagged across all assets (unadjudicated)")
    print(f"multi-face assets (manifest says single-subject): {multi_face_assets}")
    for detector, entry in qa_summary.items():
        if entry["flagged_fraction"] == 1.0:
            print(f"  ALL assets flagged by detector={detector!r} -- review before trusting or dismissing")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(freeze, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote baseline freeze: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Split "detector flags the photograph" from "detector flags pipeline damage".

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \\
        scripts/qa/qa_signal_source_check.py test_output/merged_inventory_2026_09_15.json \\
        --recipe natural --out test_output/qa_signal_source_check_2026_09_15.json

The 2026-09-15 baseline freeze found banding, plastic_skin, and asymmetry
flagged on 100% of the 10-asset dev inventory -- unadjudicated at the time,
since flag rate alone cannot say whether that is a real shared defect or a
detector reacting to something already present in the unprocessed source.

This needs no reviewed ground-truth labels (unlike step 3/4's error-rate
questions): detect_banding/detect_plastic_skin/detect_over_retouch_asymmetry
all compute their score/flagged fields from the single image they're given
plus a mask, independent of whether that image has been retouched. Running
the same detector, same mask, on the source image answers a narrower but
fully decidable question: does this detector's flagged state change between
source and processed output? If not, it is measuring the photograph, not the
retouch, regardless of what its flag rate looks like.

This is diagnostic evidence about the detectors, not a correctness ruling on
either the images or the pipeline -- an unchanged flag does not mean the flag
is wrong, only that this detector alone cannot attribute it to the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from retouch import RetouchEngine  # noqa: E402
from retouch.qa_detectors import run_qa_with_evidence  # noqa: E402

CHECKED_DETECTORS = ("banding", "plastic_skin", "asymmetry")


def check_one_asset(
    asset: Mapping[str, Any],
    *,
    root: Path,
    recipe: str,
    quality: str,
    engine: RetouchEngine,
) -> Dict[str, Any]:
    image_path = root / str(asset["path"])
    source_bgr = cv2.imread(str(image_path))
    if source_bgr is None:
        return {"asset_id": asset.get("asset_id"), "status": "image_unreadable"}

    result = engine.process(source_bgr, recipe=recipe, quality=quality)
    skin_mask = result.skin_mask

    if skin_mask is None:
        return {"asset_id": asset.get("asset_id"), "status": "no_skin_mask"}

    _, output_evidence = run_qa_with_evidence(
        np.asarray(result), skin_mask, reference_img_bgr=source_bgr, face_skin_mask=skin_mask,
    )
    _, source_evidence = run_qa_with_evidence(
        source_bgr, skin_mask, reference_img_bgr=None, face_skin_mask=skin_mask,
    )

    per_detector: Dict[str, Any] = {}
    for name in CHECKED_DETECTORS:
        out_e = output_evidence.get(name, {})
        src_e = source_evidence.get(name, {})
        out_flagged = bool(out_e.get("flagged"))
        src_flagged = bool(src_e.get("flagged"))
        per_detector[name] = {
            "output_score": out_e.get("score"),
            "output_flagged": out_flagged,
            "source_score": src_e.get("score"),
            "source_flagged": src_flagged,
            "flag_state_changed": out_flagged != src_flagged,
            "interpretation": (
                "flag state differs between source and output -- consistent with "
                "the pipeline changing this property (not proof of harm or benefit)"
                if out_flagged != src_flagged
                else "flag state identical on unprocessed source -- this detector "
                "alone cannot attribute the flag to the pipeline on this asset"
            ),
        }

    return {
        "asset_id": asset.get("asset_id"),
        "status": "checked",
        "face_count": int(result.face_count),
        "detectors": per_detector,
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--recipe", default="natural")
    parser.add_argument("--quality", default="full", choices=["full", "draft"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv[1:])

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets", [])
    if not assets:
        print("manifest has no assets", file=sys.stderr)
        return 2

    print(f"checking {len(assets)} asset(s) against detectors: {CHECKED_DETECTORS}\n")

    engine = RetouchEngine()
    rows: List[Dict[str, Any]] = []
    try:
        for index, asset in enumerate(assets, start=1):
            print(f"[{index}/{len(assets)}] {asset.get('asset_id')} ...", flush=True)
            row = check_one_asset(asset, root=manifest_path.parent, recipe=args.recipe, quality=args.quality, engine=engine)
            rows.append(row)
            if row.get("status") == "checked":
                for name, d in row["detectors"].items():
                    changed = "CHANGED" if d["flag_state_changed"] else "unchanged"
                    print(f"    {name:14s} output={d['output_flagged']!s:5s} source={d['source_flagged']!s:5s} ({changed})")
    finally:
        engine.close()

    summary: Dict[str, Any] = {}
    for name in CHECKED_DETECTORS:
        checked_rows = [r for r in rows if r.get("status") == "checked"]
        unchanged = sum(1 for r in checked_rows if not r["detectors"][name]["flag_state_changed"])
        summary[name] = {
            "assets_checked": len(checked_rows),
            "flag_state_unchanged_count": unchanged,
            "flag_state_unchanged_fraction": round(unchanged / len(checked_rows), 3) if checked_rows else None,
        }

    print("\nsummary (fraction of assets where flag state is identical source vs output):")
    for name, s in summary.items():
        print(f"  {name:14s} {s['flag_state_unchanged_fraction']}")

    report = {
        "schema": "retouch_qa_signal_source_check_v1",
        "generated_from_manifest": str(manifest_path),
        "recipe": args.recipe,
        "quality": args.quality,
        "checked_detectors": list(CHECKED_DETECTORS),
        "summary": summary,
        "rows": rows,
    }

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

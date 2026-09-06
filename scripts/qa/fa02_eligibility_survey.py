"""FA-02 eligibility distribution/search survey (diagnostic, not tuning).

Runs the REAL production engine (``fa02_texture_experimental_v1``, mode
``multiscale``) over a fixed corpus of real photos at NATIVE resolution and
reports, per detected face:

* the exact ``fa02_texture_eligibility.evaluate_face_eligibility`` measurement
  (``highpass_std``, face width, noise, support size)
* the eligible/abstained verdict against the frozen 3.5 threshold
* a rank ordering by ``highpass_std``, for locating the least-abstained faces

This is characterization + search, NOT calibration. It must never:

* modify, sharpen, resize-to-game, or otherwise alter an input image before
  measurement (the engine's own native-resolution proxy/reshape stages are
  the only transformation applied, same as any other production render);
* change ``MIN_HIGHPASS_STD`` or any other threshold in
  ``retouch/fa02_texture_eligibility.py``;
* run over ``locked_test``-split assets (this would contaminate that split for
  the eventual A2/A3 production-candidate comparison).

Corpus: the 10 dev-split assets already registered across the
``test_output/fa02_pilot_*/corpus_manifest.json`` files (all confirmed
``split: "dev"``). Native source paths are resolved from each manifest's
``source`` field against ``~/Desktop/<source folder>/<stem>.jpg`` -- the same
"find the true unretouched source" convention documented in the
retouch-engine-rerun skill.

Usage:
    .venv/bin/python scripts/qa/fa02_eligibility_survey.py \
        --out test_output/fa02_eligibility_survey.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Manifest asset_id -> (source folder under ~/Desktop, native filename).
# Resolved by hand from each test_output/fa02_pilot_<STEM>/corpus_manifest.json
# (all ten confirmed split="dev" -- see this script's module docstring).
DEV_CORPUS: List[Dict[str, str]] = [
    {"asset_id": "ff47_DSCF1058", "folder": "ff47 event", "file": "DSCF1058.jpg"},
    {"asset_id": "ff47_DSCF1423", "folder": "ff47 event", "file": "DSCF1423.jpg"},
    {"asset_id": "ff47_DSCF1606", "folder": "ff47 event", "file": "DSCF1606.jpg"},
    {"asset_id": "ff47_DSCF2183", "folder": "ff47 event", "file": "DSCF2183.jpg"},
    {"asset_id": "ff47_DSCF2258", "folder": "ff47 event", "file": "DSCF2258.jpg"},
    {"asset_id": "ff47_DSCF2274", "folder": "ff47 event", "file": "DSCF2274.jpg"},
    {"asset_id": "priority_printing_DSCF2650", "folder": "Priority for Printing", "file": "DSCF2650.jpg"},
    {"asset_id": "priority_printing_DSCF2709", "folder": "Priority for Printing", "file": "DSCF2709.jpg"},
    {"asset_id": "nahida_DSCF6754", "folder": "Nahida", "file": "DSCF6754.jpg"},
    {"asset_id": "nahida_DSCF9559", "folder": "Nahida", "file": "DSCF9559edited.jpg"},
]


def _resolve_desktop_path(folder: str, filename: str) -> Optional[str]:
    home = os.path.expanduser("~")
    candidate = os.path.join(home, "Desktop", folder, filename)
    if os.path.isfile(candidate):
        return candidate
    matches = glob.glob(os.path.join(home, "Desktop", folder, "**", filename), recursive=True)
    matches = [m for m in matches if "_retouched" not in m and "/Edited/" not in m and "compare" not in m]
    return matches[0] if matches else None


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_survey(threshold: float = 3.5) -> Dict[str, Any]:
    import cv2
    from retouch.engine import RetouchEngine
    from retouch.fa02_texture_eligibility import MIN_HIGHPASS_STD

    if abs(MIN_HIGHPASS_STD - threshold) > 1e-9:
        raise RuntimeError(
            "Threshold drift: module MIN_HIGHPASS_STD=%r != survey's frozen "
            "%r. This script asserts the threshold rather than silently "
            "measuring against a different one." % (MIN_HIGHPASS_STD, threshold)
        )

    engine = RetouchEngine()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for entry in DEV_CORPUS:
        path = _resolve_desktop_path(entry["folder"], entry["file"])
        if path is None:
            missing.append(entry["asset_id"])
            continue
        img = cv2.imread(path)
        if img is None:
            missing.append(entry["asset_id"])
            continue

        # NATIVE resolution -- no max_dim cap. The batch this survey follows
        # up on measured everything at max_dim=2048, and a 6240px source
        # downscaled to 2048 can destroy exactly the 2px-scale high-frequency
        # energy the gate measures. This is the discriminating check the
        # survey exists to run.
        result = engine.process(img, recipe="fa02_texture_experimental_v1")

        for face_idx, diag in enumerate(result.fa02_diagnostics or []):
            if diag is None:
                continue
            row = {
                "asset_id": entry["asset_id"],
                "source_path": path,
                "face_index": face_idx,
                "image_shape": list(img.shape[:2]),
            }
            row.update(diag)
            rows.append(row)

    values = [
        r["measured"]["highpass_std"]
        for r in rows
        if r.get("measured", {}).get("highpass_std") is not None
    ]
    ranked = sorted(
        (r for r in rows if r.get("measured", {}).get("highpass_std") is not None),
        key=lambda r: r["measured"]["highpass_std"],
        reverse=True,
    )
    eligible_count = sum(1 for r in rows if r.get("eligible"))

    report = {
        "threshold": threshold,
        "corpus_split": "dev",
        "corpus_size_requested": len(DEV_CORPUS),
        "corpus_size_measured": len(rows),
        "missing_assets": missing,
        "eligible_face_count": eligible_count,
        "abstained_face_count": len(rows) - eligible_count,
        "distribution": {
            "n": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "median": _percentile(values, 50) if values else None,
            "p25": _percentile(values, 25) if values else None,
            "p75": _percentile(values, 75) if values else None,
            "p90": _percentile(values, 90) if values else None,
            "distance_from_threshold_at_max": (
                (threshold - max(values)) if values else None
            ),
        },
        "ranked_by_highpass_std_desc": [
            {
                "asset_id": r["asset_id"],
                "face_index": r["face_index"],
                "highpass_std": r["measured"]["highpass_std"],
                "face_width_px": r["measured"]["face_width_px"],
                "noise_sigma": r["measured"]["noise_sigma"],
                "eligible": r["eligible"],
                "reason": r["reason"],
            }
            for r in ranked
        ],
        "all_rows": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "test_output" / "fa02_eligibility_survey.json"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    report = run_survey()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    dist = report["distribution"]
    print(f"Corpus: {report['corpus_size_measured']} faces measured "
          f"({report['corpus_size_requested']} assets requested, "
          f"{len(report['missing_assets'])} missing)")
    print(f"Eligible: {report['eligible_face_count']} / "
          f"{report['corpus_size_measured']}")
    print(f"highpass_std distribution: min={dist['min']:.3f} "
          f"p25={dist['p25']:.3f} median={dist['median']:.3f} "
          f"p75={dist['p75']:.3f} p90={dist['p90']:.3f} max={dist['max']:.3f} "
          f"(threshold={report['threshold']})")
    if dist["max"] is not None:
        print(f"Closest face to threshold: {dist['max']:.3f}, "
              f"{dist['distance_from_threshold_at_max']:.3f} below {report['threshold']}")
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

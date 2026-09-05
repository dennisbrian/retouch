"""FA-03 pilot comparison across the 5 real corpus faces plus a small
synthetic grid, per docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md
Sec 5's "recommended first authorized experiment" (scoped down: this is
the small reviewed pilot the task asks for, explicitly NOT the 30-portrait
/192-synthetic benchmark -- that corpus is not available).

Real images: DSCF2306, DSCF2308, DSCF2310, DSCF2362, DSCF2365 (all
existing cosplay/heavy-makeup portraits already used across this
session's FA-01/FA-02 work; unreviewed for confirmed mole/freckle ground
truth, consistent with this repo's known corpus limitation -- see
CLAUDE.md). DSCF2310 carries the two independently confirmed eyeliner
false positives from the research baseline.

Synthetic: a small grid of compact marks at varying eye-distance
(near/far) and one elongated eyeliner-like stroke, composited onto a real
face crop, so the near-eye preservation claim ("genuine compact marks
near the eye must remain eligible") is tested on something more than one
hand-placed point.

For each real face: captures the true (canvas_original, regions) pair via
the same monkeypatch pattern used in prior FA-01/FA-02 sessions
(patching retouch.engine._process_face_core, not perf_optimizations's
own reference -- engine.py imports the name directly, so patching the
module-level perf_optimizations symbol does not intercept engine's call).

Reports, per the task's required breakdown: candidate coverage, per-arm
class distribution, abstention rate, and the specific DSCF2310 outcome
(do the two confirmed eyeliner components stop being accepted as moles).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retouch.engine import RetouchEngine  # noqa: E402
from retouch import engine as engine_mod  # noqa: E402

from fa03_mark_classification_experiment import (  # noqa: E402
    ARMS,
    ABSTAIN,
    classify_frozen_candidates,
)


CORPUS_ROOT = "/private/tmp/retouch-meitu-final-0071447"
FACES = ("DSCF2306", "DSCF2308", "DSCF2310", "DSCF2362", "DSCF2365")
DSCF2310_TARGET_BBOXES = {(236, 381, 18, 12), (367, 428, 22, 13)}
OUT_DIR = os.path.join(os.path.dirname(__file__), "fa03_pilot_out")


def capture_real_face(image_path: str, recipe: str = "xiaohongshu"):
    captured = {}
    orig = engine_mod._process_face_core

    def spy(canvas, regions, shifted_face, ctx, *a, **kw):
        if "canvas_original" not in captured:
            captured["canvas_original"] = canvas.copy()
            captured["regions"] = regions
        return orig(canvas, regions, shifted_face, ctx, *a, **kw)

    engine_mod._process_face_core = spy
    try:
        img = cv2.imread(image_path)
        engine = RetouchEngine()
        engine.process(img, recipe=recipe)
    finally:
        engine_mod._process_face_core = orig

    if "regions" not in captured:
        return None, None
    return captured["canvas_original"], captured["regions"]


def regions_to_dict(regions) -> dict:
    names = (
        "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "hair",
        "skin", "left_under_eye", "right_under_eye", "crows_feet_l", "crows_feet_r",
    )
    return {n: getattr(regions, n, None) for n in names}


def run_one_face(name: str) -> dict:
    path = os.path.join(CORPUS_ROOT, name, "00_source.jpg")
    if not os.path.exists(path):
        return {"error": f"missing {path}"}

    canvas, regions = capture_real_face(path)
    if canvas is None:
        return {"error": "restore_micro_texture/_process_face_core never captured regions"}

    img_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    regions_dict = regions_to_dict(regions)
    skin = regions_dict["skin"]
    skin_norm = skin.astype(np.float32)
    if skin_norm.max() > 1.5:
        skin_norm = skin_norm / 255.0

    candidates, features, results = classify_frozen_candidates(
        img_u8, skin_norm, regions_dict, face_width=200.0,
    )

    per_arm_summary = {}
    dscf2310_outcome = {}
    for arm_name in ARMS:
        winners = [r.winner for r in results[arm_name]]
        counts = Counter(winners)
        per_arm_summary[arm_name] = dict(counts)

        if name == "DSCF2310":
            target_results = []
            for c, r in zip(candidates, results[arm_name]):
                if c.bbox in DSCF2310_TARGET_BBOXES:
                    target_results.append({"bbox": c.bbox, "winner": r.winner, "scores": r.scores})
            dscf2310_outcome[arm_name] = target_results

    out = {
        "face": name,
        "num_candidates": len(candidates),
        "per_arm_class_distribution": per_arm_summary,
        "abstention_rate": {
            arm_name: sum(1 for r in results[arm_name] if r.winner == ABSTAIN) / max(len(results[arm_name]), 1)
            for arm_name in ARMS
        },
    }
    if dscf2310_outcome:
        out["dscf2310_target_outcome"] = dscf2310_outcome
    return out


def run_synthetic_grid() -> dict:
    """Compact marks at 3 eye-distances + 1 eyeliner-like stroke,
    composited onto the DSCF2310 canvas (real skin/lighting), to check
    the near-eye preservation claim beyond a single hand-placed point."""
    path = os.path.join(CORPUS_ROOT, "DSCF2310", "00_source.jpg")
    canvas, regions = capture_real_face(path)
    if canvas is None:
        return {"error": "could not capture DSCF2310 for synthetic grid"}
    img_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    regions_dict = regions_to_dict(regions)
    skin = regions_dict["skin"]
    skin_norm = skin.astype(np.float32)
    if skin_norm.max() > 1.5:
        skin_norm = skin_norm / 255.0

    left_eye = regions_dict["left_eye"]
    ys, xs = np.where(left_eye > 0.3)
    eye_cx, eye_cy = float(xs.mean()), float(ys.mean())
    eye_bottom_y = int(ys.max())

    grid_points = {
        "compact_near_eye_5px": (int(eye_cx - 20), eye_bottom_y + 5),
        "compact_near_eye_15px": (int(eye_cx - 20), eye_bottom_y + 15),
        "compact_far_from_eye": (int(eye_cx - 20), eye_bottom_y + 60),
    }

    results_out = {}
    for label, center in grid_points.items():
        synth = img_u8.copy()
        cv2.circle(synth, center, 4, (55.0, 60.0, 80.0), -1, lineType=cv2.LINE_AA)
        candidates, features, results = classify_frozen_candidates(
            synth, skin_norm, regions_dict, face_width=200.0,
        )
        if not candidates:
            results_out[label] = {"error": "no candidate detected at this point"}
            continue
        best_idx = min(
            range(len(candidates)),
            key=lambda i: np.hypot(candidates[i].centroid[0] - center[0], candidates[i].centroid[1] - center[1]),
        )
        results_out[label] = {
            "bbox": candidates[best_idx].bbox,
            "area": candidates[best_idx].area,
            "eye_distance_norm": features[best_idx]["eye_distance_norm"],
            "pca_axis_ratio": features[best_idx]["pca_axis_ratio"],
            "per_arm_winner": {arm_name: results[arm_name][best_idx].winner for arm_name in ARMS},
        }

    return results_out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for name in FACES:
        print(f"--- {name} ---")
        result = run_one_face(name)
        all_results[name] = result
        print(json.dumps(result, indent=2, default=str))

    print("\n--- synthetic near-eye grid (composited on DSCF2310) ---")
    synthetic = run_synthetic_grid()
    all_results["synthetic_grid"] = synthetic
    print(json.dumps(synthetic, indent=2, default=str))

    with open(os.path.join(OUT_DIR, "pilot_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

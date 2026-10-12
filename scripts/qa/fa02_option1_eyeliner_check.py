"""FA-02 Sec 6: does the eyeliner-misclassification confound (first found
in 154d854 on this same corpus) reach restore_micro_texture's dimensional
zone if detect_marks were run unconditionally (the "option 1" fix)?

Captures the REAL FaceRegions object and pre/post-smoothing canvases from
a live RetouchEngine.process() call (monkeypatching
SkinProcessor.restore_micro_texture to record its arguments, then letting
the real call proceed) -- not a synthetic regions stand-in -- and runs
detect_marks directly against them. Read-only: does not modify
retouch/skin.py or perf_optimizations.py.

Requires the DSCF2310 corpus image used throughout this document
(docs/plans/RESEARCH_FA02_MICRO_TEXTURE_RESTORE_2026_09_05.md) at
/private/tmp/retouch-meitu-bakeoff-20260904/DSCF2310/00_source.jpg.

Run:
    .venv/bin/python scripts/qa/fa02_option1_eyeliner_check.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retouch.engine import RetouchEngine  # noqa: E402
from retouch import skin as skin_mod  # noqa: E402
from retouch.marks import detect_marks  # noqa: E402


CORPUS_IMAGE = "/private/tmp/retouch-meitu-bakeoff-20260904/DSCF2310/00_source.jpg"
DIM_ATTRS = (
    "nose_bridge", "cheek_highlights_l", "cheek_highlights_r",
    "left_under_eye", "right_under_eye", "left_eye", "right_eye",
    "crows_feet_l", "crows_feet_r",
)


def _norm01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    return m / 255.0 if m.max() > 1.5 else m


def main():
    if not os.path.exists(CORPUS_IMAGE):
        print(f"Corpus image not found at {CORPUS_IMAGE} -- skipping.")
        return

    captured = {}
    orig = skin_mod.SkinProcessor.restore_micro_texture

    def spy(self, smoothed_bgr, original_bgr, regions, strength=20, smooth_strength=0.5):
        captured["original_bgr"] = original_bgr.copy()
        captured["regions"] = regions
        return orig(self, smoothed_bgr, original_bgr, regions, strength, smooth_strength)

    skin_mod.SkinProcessor.restore_micro_texture = spy
    try:
        img = cv2.imread(CORPUS_IMAGE)
        engine = RetouchEngine()
        engine.process(img, recipe="xiaohongshu")
    finally:
        skin_mod.SkinProcessor.restore_micro_texture = orig

    if "regions" not in captured:
        print("restore_micro_texture was never called (micro_restore<=0?) -- nothing to check.")
        return

    regions = captured["regions"]
    original = captured["original_bgr"]
    skin = _norm01(getattr(regions, "skin"))

    records = detect_marks(np.clip(original, 0, 255).astype(np.uint8), face_mask=skin)
    print(f"detect_marks found {len(records)} records")

    dim_mask = np.zeros(skin.shape[:2], dtype=np.float32)
    for attr in DIM_ATTRS:
        m = getattr(regions, attr, None)
        if m is not None:
            dim_mask = np.clip(dim_mask + _norm01(m), 0.0, 1.0)
    print(f"dimensional-zone union: {int((dim_mask > 0.5).sum())} px")

    overlapping = []
    for r in records:
        cx, cy = int(r.centroid[0]), int(r.centroid[1])
        if 0 <= cy < dim_mask.shape[0] and 0 <= cx < dim_mask.shape[1] and dim_mask[cy, cx] > 0.5:
            overlapping.append(r)

    print(f"records with centroid inside dimensional zone: {len(overlapping)}")
    for r in sorted(overlapping, key=lambda r: -r.bbox[2] * r.bbox[3]):
        x, y, w, h = r.bbox
        print(f"  class={r.mark_class:16s} conf={r.confidence:.2f} bbox={r.bbox} area_px={w*h}")

    print(
        "\nLargest-by-area overlapping detections are the eyeliner wings "
        "(elongated bbox, ~18x12 / 22x13px, class=mole conf=0.70) -- an "
        "order of magnitude larger than the genuine freckle detections "
        "(3x3 to 5x5px) in the same list. Area, not count, is what would "
        "be subtracted from restore_micro_texture's dimensional mask "
        "under an unconditional detect_marks call."
    )


if __name__ == "__main__":
    main()

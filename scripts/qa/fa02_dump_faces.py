"""One-time extraction of the exact (pre_smooth_canvas, support_mask,
face_width_px) triples the FA-02 gate measured, for the 11-face dev-split
survey corpus (see fa02_eligibility_survey.py).

This does NOT modify retouch/fa02_texture_eligibility.py or any production
code path. It monkeypatches evaluate_face_eligibility for the duration of
one engine run per photo, purely to intercept and save its own real inputs
before delegating to the unmodified function -- the measurement that reaches
the gate is untouched.

Output: /tmp/fa02_scale_noise/crops/<asset_id>_face<i>.npz, each containing
    canvas         float32 (H, W, 3) BGR, the exact pre_smooth_canvas crop
    support        float32 (H, W) in [0, 1], the exact effective support
    face_width_px  float

These are diagnostic-only extraction artifacts (never eligible production
inputs) and MUST stay under /tmp, never committed or copied to ~/Desktop.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DUMP_DIR = Path("/tmp/fa02_scale_noise/crops")


def main() -> int:
    import logging

    import cv2
    import numpy as np

    logging.basicConfig(level=logging.WARNING)

    from fa02_eligibility_survey import DEV_CORPUS, _resolve_desktop_path
    import retouch.fa02_texture_eligibility as elig_mod
    from retouch.engine import RetouchEngine

    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    real_evaluate = elig_mod.evaluate_face_eligibility
    _state = {"asset_id": None, "face_counter": 0}

    def _capturing_evaluate(pre_smooth_canvas, support_mask, face_width_px,
                             protected_mask=None, **kwargs):
        decision = real_evaluate(
            pre_smooth_canvas, support_mask, face_width_px,
            protected_mask=protected_mask, **kwargs,
        )
        effective_support = decision.get("effective_support")
        if effective_support is not None:
            asset_id = _state["asset_id"]
            idx = _state["face_counter"]
            _state["face_counter"] += 1
            out_path = DUMP_DIR / f"{asset_id}_face{idx}.npz"
            np.savez_compressed(
                out_path,
                canvas=np.asarray(pre_smooth_canvas, dtype=np.float32),
                support=np.asarray(effective_support, dtype=np.float32),
                face_width_px=np.float64(face_width_px),
            )
            print(f"  saved {out_path} canvas={pre_smooth_canvas.shape} "
                  f"support_px={int((effective_support > 0.5).sum())} "
                  f"face_width_px={face_width_px:.2f}")
        else:
            print(f"  no effective_support for {_state['asset_id']} face "
                  f"{_state['face_counter']} (reason={decision.get('reason')})")
            _state["face_counter"] += 1
        return decision

    # perf_optimizations does `from .fa02_texture_eligibility import
    # evaluate_face_eligibility` INSIDE the elif branch (a lazy import
    # inside the function body, re-executed on every call, not a
    # module-level binding) -- so patching elig_mod.evaluate_face_eligibility
    # before each engine.process() call is sufficient; no separate patch of
    # perf_optimizations' own namespace is needed. Verified by reading
    # perf_optimizations.py:725.
    engine = RetouchEngine()

    for entry in DEV_CORPUS:
        path = _resolve_desktop_path(entry["folder"], entry["file"])
        if path is None:
            print(f"=== {entry['asset_id']} === SOURCE NOT FOUND, skipping")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"=== {entry['asset_id']} === UNREADABLE, skipping")
            continue

        _state["asset_id"] = entry["asset_id"]
        _state["face_counter"] = 0

        elig_mod.evaluate_face_eligibility = _capturing_evaluate
        try:
            print(f"=== {entry['asset_id']} ===")
            engine.process(img, recipe="fa02_texture_experimental_v1")
        finally:
            elig_mod.evaluate_face_eligibility = real_evaluate

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

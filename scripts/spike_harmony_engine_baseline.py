#!/usr/bin/env python3
"""Re-baseline harmony gates through the real engine path — blemish plan S5.

The harmony spike (`spike_harmony_metrics.py`) validated H1/H3/H4/H5 with a
guided-filter *proxy* for smoothing. Its provisional gates (D_TPR <= 0.45,
banding delta <= +0.03) must be re-measured on real `engine.process()` renders
before they may gate or auto-tune anything (PLAN_BLEMISH_POLICY_HARMONY.md
S5 QA gate). This script renders every asset two ways:

  face-only    smooth=70, body_smooth=0   (the seam failure the metric exists
                                           to catch — also the engine default
                                           shape: body_smooth defaults to 0)
  consistent   smooth=70, body_smooth=70  (porcelain parity treatment)

then evaluates `retouch.harmony.evaluate_harmony` against the original and
reports whether the provisional gates discriminate the two regimes.

A second regime does the same A/B at recipe strength (cosplay_portrait_polish_v1
with its body ops zeroed vs the full recipe), because the bare `smooth` op is
frequency-separated with a texture floor and may not produce the seam that a
full porcelain treatment stack does.

Run: python3 scripts/spike_harmony_engine_baseline.py [image ...]
Writes test_output/harmony_engine_baseline/<name>_panel.jpg contact sheets.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.engine import RetouchEngine  # noqa: E402
from retouch.harmony import build_face_anchored_body_mask, evaluate_harmony  # noqa: E402

DEFAULTS = [
    ROOT / "test_output" / "DSCF4550.jpg",
    ROOT / "test_output" / "DSCF4454.jpg",
    ROOT / "test_output" / "DSCF4463.jpg",
    ROOT / "test_output" / "DSCF4503.jpg",
    ROOT / "test_output" / "DSCF7011.jpg",
    ROOT / "test_output" / "DSCF7142.jpg",
    ROOT / "test_output" / "DSCF7204.jpg",
]

D_TPR_GATE = 0.45          # provisional, spike §3.1
BANDING_DELTA_GATE = 0.03  # provisional, spike §3.5

_BODY_OFF = dict(body_smooth=0, body_equalize=0, body_whiten=0,
                 body_match_face=0, body_relight=0, body_dodge_burn=0,
                 body_shadow_lift=0)

# regime name -> (face-only kwargs, consistent kwargs)
REGIMES = {
    "smooth70": (dict(smooth=70, **_BODY_OFF),
                 dict(smooth=70, body_smooth=70)),
    "recipe": (dict(recipe="cosplay_portrait_polish_v1", **_BODY_OFF),
               dict(recipe="cosplay_portrait_polish_v1")),
}


def _load(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    scale = 1600.0 / max(img.shape[:2])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


def _mask_vis(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros((*shape, 3), dtype=np.uint8)
    m = mask[..., 0] if mask.ndim == 3 else mask
    m = np.clip(m.astype(np.float32), 0, 1)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]))
    return cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULTS
    out_dir = ROOT / "test_output" / "harmony_engine_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = RetouchEngine()

    rows = {name: [] for name in REGIMES}
    for path in paths:
        img = _load(path)
        if img is None:
            print(f"{path.name:<12} unreadable, skipped")
            continue

        probe = engine.process(img.copy(), smooth=0, **_BODY_OFF)
        if probe.face_count == 0 or probe.skin_mask is None:
            print(f"{path.name:<12} no face, skipped")
            continue
        face_mask = probe.skin_mask
        person = engine._detector.segment_person(img)
        if person is None:
            print(f"{path.name:<12} no person mask, skipped")
            continue
        hair = engine._parser.parse_hair_full_image(img)
        body_mask = build_face_anchored_body_mask(img, face_mask, person,
                                                  hair_mask=hair)

        for regime, (kw_fo, kw_co) in REGIMES.items():
            t0 = time.time()
            face_only = engine.process(img.copy(),
                                       face_contexts=probe.face_contexts, **kw_fo)
            consistent = engine.process(img.copy(),
                                        face_contexts=probe.face_contexts, **kw_co)

            h_fo = evaluate_harmony(np.asarray(face_only), face_skin_mask=face_mask,
                                    body_skin_mask=body_mask, reference_img_bgr=img)
            h_co = evaluate_harmony(np.asarray(consistent), face_skin_mask=face_mask,
                                    body_skin_mask=body_mask, reference_img_bgr=img)
            if not h_fo["available"]:
                print(f"{path.stem:<12} {regime:<8} harmony unavailable "
                      f"(face_px={h_fo['face_pixels']}, body_px={h_fo['body_pixels']})")
                continue

            h_orig = evaluate_harmony(img, face_skin_mask=face_mask,
                                      body_skin_mask=body_mask,
                                      reference_img_bgr=img)
            flag_fo = h_fo["texture_parity_drift"] > D_TPR_GATE
            flag_co = h_co["texture_parity_drift"] > D_TPR_GATE
            verdict = ("OK: f/o flags, con passes"
                       if flag_fo and not flag_co else
                       "MISS: f/o quiet" if not flag_fo else
                       "MISS: con flags")
            rows[regime].append((path.name, h_orig, h_fo, h_co, verdict))
            print(f"{path.stem:<12} {regime:<8} "
                  f"TPRorig={h_orig['texture_parity_ratio']:5.2f} "
                  f"D_f/o={h_fo['texture_parity_drift']:5.2f} "
                  f"D_con={h_co['texture_parity_drift']:5.2f} "
                  f"specD_f/o={h_fo['specular_parity_drift']:5.2f} "
                  f"specD_con={h_co['specular_parity_drift']:5.2f} "
                  f"bandD_f/o={h_fo['body_banding_delta']:6.3f} "
                  f"bandD_con={h_co['body_banding_delta']:6.3f} "
                  f"marks={h_fo['marks_kept']}/{h_fo['marks_before']} "
                  f"{verdict} [{time.time() - t0:.1f}s]")

            panel_top = np.hstack([
                _label(img, "original"),
                _label(np.asarray(face_only), f"face-only ({regime})"),
                _label(np.asarray(consistent), f"consistent ({regime})")])
            panel_bot = np.hstack([
                _mask_vis(person, img.shape[:2]),
                _mask_vis(face_mask, img.shape[:2]),
                _mask_vis(body_mask, img.shape[:2])])
            panel = np.vstack([panel_top,
                               _label(panel_bot,
                                      "person | face skin | face-anchored body")])
            scale = 2400.0 / panel.shape[1]
            panel = cv2.resize(panel, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{path.stem}_{regime}_panel.jpg"), panel,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\npanels: {out_dir}")
    for regime, rws in rows.items():
        n_ok = sum(1 for r in rws if r[4].startswith("OK"))
        print(f"{regime}: gate discrimination at D_TPR<={D_TPR_GATE}: "
              f"{n_ok}/{len(rws)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

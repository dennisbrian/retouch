"""Visual-QA sign-off harness.

Runs the full RetouchEngine pipeline on reference portraits with two configs
(SmartProcessor suggestion + fixed ``natural_polish_v1`` baseline) and measures
the objective Visual-QA gates defined in ``docs/VISUAL_QA.md``.

Outputs (into ``test_output/qa_signoff_2026-07-10/``):
    * ``<stem>_smart_compare.jpg`` / ``<stem>_baseline_compare.jpg``  — before/after
    * ``<stem>_skin_texture.png``  — 100% cheek crop before/after
    * ``<stem>_edge_halo.png``     — face/feature-edge crop before/after
    * ``<stem>_diff_x8.png``       — 8x amplified abs diff

Gates that cannot be measured programmatically (Natural Output, No Edge
Tearing, Skin Tone Uniformity) are reported as PENDING with the inspection
image paths, never as PASS.

Reuses existing tooling only:
    retouch.io.imread_engine / resize_for_processing / make_comparison
    retouch.engine.RetouchEngine / ProcessingResult
    retouch.smart_default.SmartProcessor
    retouch.params.gui_values_to_engine_kwargs
    retouch.frequency.FrequencySeparator
    retouch.qa_detectors.detect_halo
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qa_visual_signoff")

# --- repo root on path -------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from retouch.io import (  # noqa: E402
    imread_engine,
    resize_for_processing,
    make_comparison,
)
from retouch.engine import RetouchEngine, ProcessingResult  # noqa: E402
from retouch.smart_default import SmartProcessor  # noqa: E402
from retouch.params import gui_values_to_engine_kwargs  # noqa: E402
from retouch.frequency import FrequencySeparator  # noqa: E402
from retouch.qa_detectors import detect_halo  # noqa: E402

# --- configuration ----------------------------------------------------------
REF_IMAGES = [
    "test_output/DSCF4454.jpg",
    "test_output/DSCF4463.jpg",
    "test_output/DSCF4503.jpg",
    "test_output/DSCF4550.jpg",
]
OUT_DIR = "test_output/qa_signoff_2026-07-10"
BASELINE_RECIPE = "natural_polish_v1"
PROC_MAX_DIM = 1500
SKIN_PATCH = 200  # px square for texture gate + skin crop
EDGE_PATCH = 220  # px square for halo crop

GATE_THRESHOLDS = {
    "texture_ssim": 0.92,
    "color_drift_deltaE": 2.0,
    "highlight_clip_pct": 0.5,
    "shadow_clip_pct": 0.5,
}


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Cast any image-like array to uint8 BGR via float32 [0,255] clip."""
    if img.dtype == np.uint8:
        return img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = np.clip(img.astype(np.float32), 0.0, 255.0)
    return out.astype(np.uint8)


def mean_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-SSIM on two same-shaped float arrays (grayscale or single channel).

    Replaces skimage (not installed). Adequate for a local skin-patch texture
    comparison where both images are aligned and equally scaled.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    data_range = float(max(hi - lo, 1e-6))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = ((a - mu_a) ** 2).mean()
    var_b = ((b - mu_b) ** 2).mean()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def _skin_mask_center(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
    """Centroid of largest skin-blob in a float [0,1] mask, or None."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    ys, xs = np.where(m > 0.5)
    if len(xs) == 0:
        ys, xs = np.where(m > 0.05)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def _skin_bbox_top(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    """Top sub-region of the skin mask (hairline / forehead edge band)."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    ys, xs = np.where(m > 0.5)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _safe_crop(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Center-crop ``size`` square from ``img`` at (cx,cy), clamped."""
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, min(cx - half, w - size))
    y1 = max(0, min(cy - half, h - size))
    if size > w:
        x1 = 0
    if size > h:
        y1 = 0
    crop = img[y1 : y1 + min(size, h), x1 : x1 + min(size, w)]
    if crop.shape[0] < size or crop.shape[1] < size:
        crop = cv2.copyMakeBorder(
            crop,
            0,
            max(0, size - crop.shape[0]),
            0,
            max(0, size - crop.shape[1]),
            cv2.BORDER_REFLECT,
        )
    return crop


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(
        img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        cv2.LINE_AA,
    )


def _clip_pct(img_u8: np.ndarray, level: int) -> float:
    """Fraction of pixels saturating exactly at *level* (floor 0 / ceiling 255).

    Uses exact equality — clipping means a channel pinned at the boundary, not
    merely ``>=`` it (which would always be true for level 0).
    """
    return float(100.0 * float(np.mean(img_u8 == level)))


# ---------------------------------------------------------------------------
# per-image pipeline
# ---------------------------------------------------------------------------
def process_one(
    engine: RetouchEngine,
    smart: SmartProcessor,
    path: str,
    out_dir: str,
) -> Dict[str, Any]:
    stem = os.path.splitext(os.path.basename(path))[0]
    logger.info("Processing %s", stem)

    img_raw = imread_engine(path)
    img, _ = resize_for_processing(_to_uint8(img_raw), PROC_MAX_DIM)
    orig_u8 = img

    # --- run 1: SmartProcessor suggestion -----------------------------------
    suggestion = smart.analyze_and_suggest(img)
    smart_kwargs: Dict[str, Any] = gui_values_to_engine_kwargs(suggestion.params)
    result_smart = engine.process(img, recipe=suggestion.recipe, **smart_kwargs)
    qa_smart = list(result_smart.qa) if getattr(result_smart, "qa", None) else []

    # --- run 2: fixed baseline recipe ---------------------------------------
    result_base = engine.process(img, recipe=BASELINE_RECIPE)
    qa_base = list(result_base.qa) if getattr(result_base, "qa", None) else []

    smart_u8 = _to_uint8(result_smart)
    base_u8 = _to_uint8(result_base)

    # --- emit comparison images ---------------------------------------------
    make_comparison(
        orig_u8, smart_u8,
        os.path.join(out_dir, f"{stem}_smart_compare.jpg"), "jpg", 90,
    )
    make_comparison(
        orig_u8, base_u8,
        os.path.join(out_dir, f"{stem}_baseline_compare.jpg"), "jpg", 90,
    )

    # --- skin / edge crops ---------------------------------------------------
    mask_smart = getattr(result_smart, "skin_mask", None)
    center = _skin_mask_center(mask_smart)
    if center is None:
        center = (orig_u8.shape[1] // 2, int(orig_u8.shape[0] * 0.55))
    skin_before = _safe_crop(orig_u8, *center, SKIN_PATCH)
    skin_after = _safe_crop(smart_u8, *center, SKIN_PATCH)
    skin_stack = np.hstack([skin_before, skin_after])
    _label(skin_stack[:SKIN_PATCH, :SKIN_PATCH], "BEFORE")
    _label(skin_stack[:SKIN_PATCH, SKIN_PATCH:], "AFTER")
    cv2.imwrite(os.path.join(out_dir, f"{stem}_skin_texture.png"), skin_stack)

    bbox = _skin_bbox_top(mask_smart)
    if bbox is not None:
        edge_cx = (bbox[0] + bbox[2]) // 2
        edge_cy = bbox[1] + EDGE_PATCH // 2
    else:
        edge_cx, edge_cy = center[0], min(center[1] - SKIN_PATCH // 2, orig_u8.shape[0] - 1)
    edge_before = _safe_crop(orig_u8, edge_cx, edge_cy, EDGE_PATCH)
    edge_after = _safe_crop(smart_u8, edge_cx, edge_cy, EDGE_PATCH)
    edge_stack = np.hstack([edge_before, edge_after])
    _label(edge_stack[:EDGE_PATCH, :EDGE_PATCH], "BEFORE")
    _label(edge_stack[:EDGE_PATCH, EDGE_PATCH:], "AFTER")
    cv2.imwrite(os.path.join(out_dir, f"{stem}_edge_halo.png"), edge_stack)

    # --- 8x amplified diff ---------------------------------------------------
    diff = np.abs(orig_u8.astype(np.int16) - smart_u8.astype(np.int16))
    diff_x8 = np.clip(diff * 8, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_diff_x8.png"), diff_x8)

    # --- measurable gates ----------------------------------------------------
    gates = {}

    # Texture Preservation: SSIM on high-freq band of skin patch.
    try:
        sep = FrequencySeparator()
        face_width = float(max(orig_u8.shape[:2]) * 0.3)
        hf_b = sep.separate(skin_before, face_width).high.mean(axis=2)
        hf_a = sep.separate(skin_after, face_width).high.mean(axis=2)
        ssim = mean_ssim(hf_b, hf_a)
        gates["texture"] = (
            "PASS" if ssim >= GATE_THRESHOLDS["texture_ssim"]
            else "FAIL",
            f"SSIM {ssim:.3f} on skin high-freq band (thr {GATE_THRESHOLDS['texture_ssim']})",
        )
    except Exception as exc:  # noqa: BLE001 — measurement must not crash the batch
        logger.warning("texture gate failed for %s: %s", stem, exc)
        gates["texture"] = ("PENDING", f"measurement error: {exc}")

    # No Color Drift: ΔE in LAB on neutral-gray or stable background pixels.
    try:
        orig_lab = cv2.cvtColor(orig_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        after_lab = cv2.cvtColor(smart_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        b, g, r = orig_u8[:, :, 0], orig_u8[:, :, 1], orig_u8[:, :, 2]
        neutral = (
            (np.abs(r.astype(int) - g.astype(int)) < 12)
            & (np.abs(g.astype(int) - b.astype(int)) < 12)
            & (orig_lab[:, :, 0] > 25)
            & (orig_lab[:, :, 0] < 225)
        )
        if neutral.sum() >= 100:
            ys, xs = np.where(neutral)
            sel = np.random.RandomState(0).choice(len(xs), 100, replace=False)
            idx = (ys[sel], xs[sel])
            src = orig_lab[idx]
            dst = after_lab[idx]
            de = float(np.mean(np.sqrt(np.sum((src - dst) ** 2, axis=1))))
            used = "neutral gray"
        else:
            # fall back to stable background corners
            samples = [
                orig_lab[5, 5], orig_lab[5, -6],
                orig_lab[-6, 5], orig_lab[-6, -6],
                orig_lab[orig_lab.shape[0] // 2, 5],
            ]
            samples_dst = [
                after_lab[5, 5], after_lab[5, -6],
                after_lab[-6, 5], after_lab[-6, -6],
                after_lab[after_lab.shape[0] // 2, 5],
            ]
            de = float(np.mean([
                np.sqrt(np.sum((s - d) ** 2))
                for s, d in zip(samples, samples_dst)
            ]))
            used = "stable background corners"
        gates["color_drift"] = (
            "PASS" if de <= GATE_THRESHOLDS["color_drift_deltaE"]
            else "FAIL",
            f"ΔE {de:.2f} ({used}, thr {GATE_THRESHOLDS['color_drift_deltaE']})",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("color drift gate failed for %s: %s", stem, exc)
        gates["color_drift"] = ("PENDING", f"measurement error: {exc}")

    # No Highlight Clipping / No Shadow Crushing
    hi_pct = _clip_pct(smart_u8, 255)
    lo_pct = _clip_pct(smart_u8, 0)
    gates["highlight"] = (
        "PASS" if hi_pct <= GATE_THRESHOLDS["highlight_clip_pct"] else "FAIL",
        f"{hi_pct:.3f}% pixels at 255 (thr {GATE_THRESHOLDS['highlight_clip_pct']}%)",
    )
    gates["shadow"] = (
        "PASS" if lo_pct <= GATE_THRESHOLDS["shadow_clip_pct"] else "FAIL",
        f"{lo_pct:.3f}% pixels at 0 (thr {GATE_THRESHOLDS['shadow_clip_pct']}%)",
    )

    # No Halo: reuse qa_detectors.detect_halo on the smart output.
    try:
        halo = detect_halo(smart_u8, mask_smart)
        gates["halo"] = (
            "FAIL" if halo.get("flagged", False) else "PASS",
            f"score {halo.get('score', 0.0):.2f}, flagged={halo.get('flagged', False)}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("halo gate failed for %s: %s", stem, exc)
        gates["halo"] = ("PENDING", f"measurement error: {exc}")

    # Human-review gates — never auto-PASS.
    gates["natural"] = (
        "PENDING",
        f"human review: {out_dir}/{stem}_smart_compare.jpg, "
        f"{stem}_skin_texture.png, {stem}_diff_x8.png",
    )
    gates["edge_tearing"] = (
        "PENDING",
        f"human review: {out_dir}/{stem}_edge_halo.png, {stem}_skin_texture.png",
    )
    gates["skin_uniformity"] = (
        "PENDING",
        f"human review: {out_dir}/{stem}_skin_texture.png, {stem}_smart_compare.jpg",
    )

    return {
        "stem": stem,
        "gates": gates,
        "qa_smart": qa_smart,
        "qa_base": qa_base,
        "smart_recipe": suggestion.recipe,
        "out_dir": out_dir,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
GATE_LABELS = {
    "texture": "Texture Preservation",
    "halo": "No Halo",
    "edge_tearing": "No Edge Tearing",
    "color_drift": "No Color Drift",
    "highlight": "No Highlight Clipping",
    "shadow": "No Shadow Crushing",
    "skin_uniformity": "Skin Tone Uniformity",
    "natural": "Natural Output",
}


def print_report(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("VISUAL-QA SIGN-OFF BATCH — 2026-07-10")
    print("=" * 78)
    for row in rows:
        print(f"\n### {row['stem']}  (smart recipe: {row['smart_recipe']})")
        for key, label in GATE_LABELS.items():
            status, detail = row["gates"][key]
            print(f"  [{status:>7}] {label}: {detail}")
        if row["qa_smart"]:
            print("  engine.qa flags (smart):")
            for w in row["qa_smart"]:
                print(f"    - {w.detector}: flagged={w.flagged} score={w.score:.3f} "
                      f"msg={w.message}")
    print("\n" + "=" * 78)


def append_report(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "",
        "## 9. Visual-QA Sign-off Batch — 2026-07-10",
        "",
        "Harness: `scripts/qa_visual_signoff.py`. Two configs per image: "
        "SmartProcessor suggestion and fixed `natural_polish_v1` baseline. "
        "Measured gates run on the SmartProcessor output (the shipping path); "
        "side-by-side comparisons emitted for both configs.",
        "",
        "**Honesty note:** Texture / Halo / Color-Drift / Highlight / Shadow "
        "are measured programmatically. Natural Output, No Edge Tearing, and "
        "Skin Tone Uniformity require a human and are reported PENDING with the "
        "inspection image paths — never as PASS.",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['stem']}")
        lines.append("")
        for key, label in GATE_LABELS.items():
            status, detail = row["gates"][key]
            lines.append(f"- **{label}**: {status} — {detail}")
        flags = [
            f"{w.detector}(flagged={w.flagged},score={w.score:.3f})"
            for w in row["qa_smart"]
        ]
        if flags:
            lines.append(f"- Engine QA flags (smart): {', '.join(flags)}")
        lines.append("")

    doc_path = os.path.join(ROOT, "docs", "VISUAL_QA.md")
    with open(doc_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Appended gate report to %s", doc_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    engine = RetouchEngine()
    smart = SmartProcessor(engine=engine)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for rel in REF_IMAGES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            logger.warning("Missing reference image: %s — skipped", path)
            failures.append(f"missing: {rel}")
            continue
        try:
            rows.append(process_one(engine, smart, path, OUT_DIR))
        except Exception as exc:  # noqa: BLE001 — continue batch on per-image crash
            logger.exception("Failed to process %s: %s", rel, exc)
            failures.append(f"crash: {rel}: {exc}")

    if not rows:
        logger.error("No images processed; aborting report.")
        return 1

    print_report(rows)
    append_report(rows)

    if failures:
        logger.warning("Non-fatal failures: %s", "; ".join(failures))
    logger.info("Done. %d image(s) processed, %d failure(s).", len(rows), len(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())

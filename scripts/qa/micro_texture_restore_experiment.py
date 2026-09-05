"""FA-02: does restore_micro_texture bring back a defect that smoothing/mark
protection just suppressed?

docs/plans/PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md
Sec "FA-02 -- texture preservation evidence" asks to compare current texture
restoration against band-selective restoration that excludes approved
defect supports, with restoration disabled as an ablation, and warns that
"a high-frequency energy increase is not evidence of authentic texture by
itself."

Source-reading (this session) found the current texture-restoration
surface is actually TWO mechanisms, not one:

  1. frequency.combine()'s texture_opacity / pore_synthesis -- applied
     BEFORE blemish.remove runs (perf_optimizations.py: frequency.combine
     at line ~479-572, blemish.remove at line ~885), so it cannot be
     reprojecting an already-repaired defect's old texture; that ordering
     rules mechanism 1 out as the FA-02 subject.
  2. skin.SkinProcessor.restore_micro_texture -- runs AFTER blemish.remove
     and the base smoothing stage (perf_optimizations.py:670), and
     computes `detail = pre_smooth_canvas - smoothed_canvas`, then adds a
     fraction of that detail back inside a fixed "dimensional" zone
     (nose_bridge, cheek_highlights_l/r, left/right_under_eye, left/
     right_eye, crows_feet_l/r). It has ZERO mark-policy or defect
     awareness: if a mole, blemish, or protected mark's footprint
     overlaps a dimensional zone, this mechanism restores the PRE-
     smoothing pixel values there -- reintroducing exactly the texture
     that smoothing/mark-protection/blemish-removal just worked to
     suppress or protect from being altered. This is the real "restored
     defect" case the plan describes. `micro_restore` is a live,
     registered ParamSpec, and 4 shipped recipes set it (8-35 strength).

This script calls the REAL production `skin.SkinProcessor.restore_micro_texture`
directly (not a local reimplementation -- it's a small, pure function of
plain arrays + a regions object, so there is no reason to duplicate it).
No other retouch/ code path is touched; this is read-only evidence
gathering. `regions` is a minimal SimpleNamespace exposing only the
dimensional-zone masks restore_micro_texture actually reads.

Three arms per the plan's required ablation:
  - "current"        : call restore_micro_texture exactly as production does today
  - "defect_excluded": subtract the mark's protected footprint from the
                        dimensional mask before calling the SAME function
                        (band-selective restoration with defect exclusion)
  - "disabled"        : strength=0 (restoration ablation)

Discriminator, per the plan's own warning that "energy increase is not
evidence of authentic texture": high-frequency energy alone cannot tell
authentic pore texture (mechanism 1, correlated with the source, outside
any mark) apart from a restored defect (also correlated with the source,
but INSIDE a mark's footprint). This script therefore measures energy
INSIDE the mark footprint specifically, and separately confirms whether
the visual defect signal (mark contrast, same ring-minus-mark instrument
FA-01 used) comes back inside a dimensional zone.

Run:
    .venv/bin/python scripts/qa/micro_texture_restore_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retouch.skin import SkinProcessor  # noqa: E402
from retouch.frequency import combine, separate  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "micro_texture_restore_out")
FACE_WIDTH = 400.0
CANVAS = 256


def _textured_canvas(seed: int = 7) -> np.ndarray:
    """Same construction as the FA-01 smoothing experiment (std=30 avoids
    silently flooring _texture_adaptation_factor)."""
    rng = np.random.default_rng(seed)
    h = w = CANVAS
    base = np.full((h, w, 3), (150.0, 175.0, 205.0), dtype=np.float32)
    noise = rng.normal(0, 30.0, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    base += noise[:, :, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _add_mark(canvas: np.ndarray, center, radius: int) -> np.ndarray:
    h, w = canvas.shape[:2]
    m = np.zeros((h, w), dtype=np.float32)
    cv2.circle(m, center, radius, 1.0, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    mark_bgr = np.array([70.0, 95.0, 120.0], dtype=np.float32)
    out = canvas.astype(np.float32) * (1.0 - m[:, :, None]) + mark_bgr[None, None, :] * m[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _mark_mask(shape, center, radius: int) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.float32)
    cv2.circle(m, center, radius, 1.0, -1)
    return m


def _mark_contrast(img_gray: np.ndarray, center, mark_radius: int, ring_inner=3, ring_outer=12) -> float:
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mark_sel = r <= mark_radius * 0.6
    ring_sel = (r >= mark_radius + ring_inner) & (r <= mark_radius + ring_outer)
    return float(img_gray[ring_sel].mean() - img_gray[mark_sel].mean())


def _make_regions(shape, cheek_center):
    """Minimal FaceRegions-like object: only the dimensional-zone attrs
    restore_micro_texture actually reads. cheek_highlights_l is placed at
    the mark's own location so the mark sits INSIDE a dimensional zone --
    the failure condition the plan's "repaired-defect exclusion" concerns."""
    h, w = shape[:2]
    cheek = np.zeros((h, w), dtype=np.float32)
    cv2.circle(cheek, cheek_center, 30, 1.0, -1)
    empty = np.zeros((h, w), dtype=np.float32)
    return SimpleNamespace(
        nose_bridge=empty,
        cheek_highlights_l=cheek,
        cheek_highlights_r=empty,
        left_under_eye=empty,
        right_under_eye=empty,
        left_eye=empty,
        right_eye=empty,
        crows_feet_l=empty,
        crows_feet_r=empty,
    )


def _smooth(canvas: np.ndarray, smooth_strength: float, mark_protect=None) -> np.ndarray:
    """Real production smoothing path: separate() + combine(), including
    the FA-01 mark_protect mechanism already shipped (c665da1) so this
    experiment measures restore_micro_texture's interaction with the
    CURRENT protected pipeline, not a stripped-down one."""
    layers = separate(canvas, face_width=FACE_WIDTH)
    result = combine(
        layers,
        skin_mask=np.ones(canvas.shape[:2], dtype=np.float32),
        smooth_strength=smooth_strength,
        mid_reduction=0.35,
        face_width=FACE_WIDTH,
        smooth_engine="guided",
        mark_protect=mark_protect,
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def run_arm(pre_smooth, smoothed, regions, mode: str, mark_mask, micro_strength=30, smooth_strength=0.9):
    proc = SkinProcessor()
    if mode == "disabled":
        return smoothed.copy()

    regions_arg = regions
    if mode == "defect_excluded":
        excluded_cheek = np.clip(regions.cheek_highlights_l - mark_mask, 0.0, 1.0)
        regions_arg = SimpleNamespace(**{**vars(regions), "cheek_highlights_l": excluded_cheek})

    return proc.restore_micro_texture(
        smoothed, pre_smooth, regions_arg,
        strength=micro_strength, smooth_strength=smooth_strength,
    )


def build_scene(radius: int, center=(128, 128)):
    canvas = _add_mark(_textured_canvas(), center, radius)
    mark_mask = _mark_mask(canvas.shape, center, radius)
    regions = _make_regions(canvas.shape, cheek_center=center)
    return canvas, mark_mask, regions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for radius in (4, 8, 15):
        center = (128, 128)
        canvas, mark_mask, regions = build_scene(radius, center)
        source_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        source_contrast = _mark_contrast(source_gray, center, radius)

        for policy_label, mark_protect in (("no_mark_policy", None), ("mark_protected", mark_mask)):
            smoothed = _smooth(canvas, smooth_strength=0.9, mark_protect=mark_protect)
            smoothed_gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
            smoothed_contrast = _mark_contrast(smoothed_gray, center, radius)

            for mode in ("disabled", "current", "defect_excluded"):
                out = run_arm(canvas, smoothed, regions, mode, mark_mask)
                out_gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                out_contrast = _mark_contrast(out_gray, center, radius)

                # Energy inside the mark footprint specifically (the
                # discriminator the plan's warning demands -- NOT whole-face
                # high-band energy, which can't tell restored-defect from
                # authentic pore texture apart).
                mark_sel = mark_mask > 0.5
                in_mark_energy = float(np.abs(
                    out_gray[mark_sel].astype(np.float32) - smoothed_gray[mark_sel].astype(np.float32)
                ).mean())

                key = f"r{radius}_{policy_label}_{mode}"
                results[key] = {
                    "source_contrast": source_contrast,
                    "smoothed_contrast": smoothed_contrast,
                    "restored_contrast": out_contrast,
                    "contrast_recovered_vs_smoothed": out_contrast - smoothed_contrast,
                    "in_mark_mean_abs_delta_vs_smoothed": in_mark_energy,
                }
                cv2.imwrite(os.path.join(OUT_DIR, f"{key}.png"), out)

        cv2.imwrite(os.path.join(OUT_DIR, f"r{radius}_00_source.png"), canvas)

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    check_mark_independent_discriminator()


def check_mark_independent_discriminator():
    """FA-02 Sec 5: can detail = pre_smooth - smoothed's own magnitude/shape
    distinguish a mark from authentic texture WITHOUT mark detection? Looked
    promising on the noise-only synthetic canvas (30x magnitude separation)
    and failed on a real face (top connected high-detail components are
    the same size/magnitude with or without a mark present) -- rejected,
    not just untuned: the residual carries no information about *why*
    smoothing removed a signal, so no threshold on it can reliably tell
    "mark" from "hair strand/fold shadow" on real skin.
    """
    print("\n--- mark-independent discriminator check (FA-02 Sec 5) ---")

    # Synthetic: looks strong (no coherent structure other than the mark).
    canvas = _add_mark(_textured_canvas(), (128, 128), 8)
    smoothed = _smooth(canvas, smooth_strength=0.9, mark_protect=None)
    detail_mag = np.abs(canvas.astype(np.float32) - smoothed.astype(np.float32)).mean(axis=2)
    mark_sel = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mark_sel, (128, 128), 8, 1, -1)
    other_sel = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(other_sel, (60, 60), 8, 1, -1)
    print(f"synthetic mark region:    mean={detail_mag[mark_sel.astype(bool)].mean():.2f}")
    print(f"synthetic texture region: mean={detail_mag[other_sel.astype(bool)].mean():.2f}")

    # Real face: fails (needs the real DSCF2306 corpus image on disk).
    real_path = "/private/tmp/retouch-meitu-bakeoff-20260904/DSCF2306/00_source.jpg"
    if not os.path.exists(real_path):
        print("(real-face check skipped -- corpus image not found at", real_path, ")")
        return

    img = cv2.imread(real_path)
    crop = img[363:363 + 219, 729:729 + 191]
    mark_center = (int(crop.shape[1] * 0.72), int(crop.shape[0] * 0.62))
    crop_with_mark = _add_mark(crop, mark_center, 6)

    for label, im in (("clean", crop), ("with_mark", crop_with_mark)):
        sm = _smooth(im, smooth_strength=0.9, mark_protect=None)
        dm = np.abs(im.astype(np.float32) - sm.astype(np.float32)).mean(axis=2)
        thresh = np.percentile(dm, 90)
        high_mask = (dm > thresh).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(high_mask, connectivity=8)
        comps = sorted(
            ((stats[i, cv2.CC_STAT_AREA], float(dm[labels == i].mean())) for i in range(1, n)),
            key=lambda x: -x[0],
        )
        print(f"real face ({label}): top 5 components (area, mean_mag) = {comps[:5]}")


if __name__ == "__main__":
    main()

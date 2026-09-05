"""FA-01: compare smoothing-vs-mark-protection mechanisms.

Investigates the unresolved interaction named in
docs/plans/RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md:
base smoothing (retouch.frequency.FrequencySeparator.combine) has zero
mark-policy awareness (verified: mark_policy='preserve_all' + smooth=90 is
byte-identical to mark_policy=None on a real render). This script builds a
falsifiable synthetic scene and compares candidate protection mechanisms
against six criteria. It does NOT change any production code path —
every mechanism below is implemented locally in this script, called
directly against retouch.frequency internals, never against the shipped
FrequencySeparator.combine.

Source-reading narrowed the originally-proposed five candidates to three
distinct mechanisms (see docs/plans/RESEARCH_FA01_SMOOTHING_MARK_PROTECTION_2026_09_05.md
Sec 1 for the full argument):

  - `combine()`'s skin_mask only ever controls the FINAL blend alpha
    (`blend_masked(orig_crop, processed_crop, m_2d)`); it is never passed
    to the smoothing filter itself.
  - `_guided_smooth` -> `utils.guided_filter` computes `mean_I = blur(src)`
    with cv2.blur (a plain box filter) over the WHOLE crop, no mask
    parameter exists. A mark's pixel values leak into neighboring pixels'
    local statistics regardless of what the caller excludes from the
    final blend.
  - Consequence: "output-mask exclusion" (candidate 2), "feathered/soft
    attenuation" (candidate 3), and "post-smoothing restoration of source
    detail" (candidate 5) are the SAME mechanism (excluding/restoring at
    the blend-alpha stage) at different alpha profiles or timing — encoded
    below as one BLEND_ALPHA arm swept over feather width. Only
    "exclusion from filter/reference statistics" (candidate 4) is
    mechanistically distinct, since it changes what the filter itself
    reads. Implemented here via normalized convolution (mask-weighted
    moments), the same technique retouch/undereye.py's analyzer already
    uses — NOT added to utils.guided_filter, which stays untouched.

Run:
    .venv/bin/python scripts/qa/smoothing_mark_protection_experiment.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retouch.frequency import (  # noqa: E402
    FrequencyLayers,
    GAUSSIAN_BLEND_FACTOR,
    SMOOTH_K_FACTOR,
    SMOOTH_K_MIN,
    SIGMA_BASE,
    SIGMA_STRENGTH_FACTOR,
    _texture_adaptation_factor,
)

# Matches params.py's registered default (ParamSpec "mid_reduction",
# default=0.35) -- not a guess.
MID_REDUCTION = 0.35
from retouch.utils import adaptive_ksize, blend_masked  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "smoothing_mark_protection_out")
os.makedirs(OUT_DIR, exist_ok=True)

FACE_WIDTH = 400.0
CANVAS = 512


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(seed: int = 7):
    """Skin-toned canvas with real pore-scale texture (so smoothing has
    something to preserve/destroy and _texture_adaptation_factor doesn't
    silently floor smooth_strength on a flat image) plus two dark marks
    sized on either side of k_mid, so the small mark lives mostly in the
    HIGH band (frequency.separate) and the large one mostly in LOW/MID.
    """
    rng = np.random.default_rng(seed)
    h = w = CANVAS
    base = np.full((h, w, 3), (150.0, 175.0, 205.0), dtype=np.float32)  # BGR skin tone

    # Pore-scale texture: band-limited noise so the high band carries real
    # energy. std=30 is calibrated (not guessed) against
    # retouch.frequency._texture_adaptation_factor on this exact scene
    # geometry: std=9 gave adapt=0.55 (silently halving every requested
    # smooth_strength below), std=30 reaches adapt=1.0 so smooth_strength
    # sweeps (0.5, 0.9) actually run at their requested values.
    fine_noise = rng.normal(0, 30.0, (h, w)).astype(np.float32)
    fine_noise = cv2.GaussianBlur(fine_noise, (0, 0), 1.2)
    base += fine_noise[:, :, None]

    # Gentle low-frequency shading so LOW/MID bands aren't degenerate either.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base[:, :, 0] += 8.0 * np.sin(yy / 90.0)
    base[:, :, 1] += 6.0 * np.sin(xx / 110.0)

    canvas = np.clip(base, 0, 255)

    small_center = (180, 220)   # ~8px diameter mark: sub-k_mid (k_mid=17 @ FACE_WIDTH=400)
    small_radius = 4
    large_center = (330, 280)   # ~30px diameter mark: between k_mid(17) and k_low(49)
    large_radius = 15

    marks_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(marks_mask, small_center, small_radius, 255, -1)
    cv2.circle(marks_mask, large_center, large_radius, 255, -1)

    mark_bgr = np.array([70.0, 95.0, 120.0], dtype=np.float32)  # visibly darker than skin
    for center, radius in ((small_center, small_radius), (large_center, large_radius)):
        m = np.zeros((h, w), dtype=np.float32)
        cv2.circle(m, center, radius, 1.0, -1)
        m = cv2.GaussianBlur(m, (5, 5), 0)  # tiny edge softening, still a real mark not a hard disk
        canvas = canvas * (1.0 - m[:, :, None]) + mark_bgr[None, None, :] * m[:, :, None]

    skin_mask = np.ones((h, w), dtype=np.float32)  # whole crop is "skin" for this experiment

    return {
        "canvas": np.clip(canvas, 0, 255).astype(np.float32),
        "skin_mask": skin_mask,
        "marks_binary": marks_mask,  # 0/255, ground truth, bypasses detect_marks entirely
        "small_center": small_center, "small_radius": small_radius,
        "large_center": large_center, "large_radius": large_radius,
    }


# ---------------------------------------------------------------------------
# Frequency separation (verbatim algorithm from FrequencySeparator.separate)
# ---------------------------------------------------------------------------

def separate(img_bgr: np.ndarray, face_width: float) -> FrequencyLayers:
    img_f = img_bgr.astype(np.float32)
    k_low = adaptive_ksize(face_width, factor=0.12, minimum=5)
    k_mid = adaptive_ksize(face_width, factor=0.04, minimum=3)
    low = cv2.GaussianBlur(img_f, (k_low, k_low), 0)
    med_blur = cv2.GaussianBlur(img_f, (k_mid, k_mid), 0)
    mid = med_blur - low
    high = img_f - med_blur
    return FrequencyLayers(low, mid, high), k_low, k_mid


# ---------------------------------------------------------------------------
# Guided filter variants
# ---------------------------------------------------------------------------

def guided_filter_plain(src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Verbatim self-guided branch of utils.guided_filter (no downsample path
    needed at this crop size). Reproduced locally so the experiment does not
    depend on internal helper availability and stays obviously read-only."""
    mean_I = cv2.blur(src, (radius, radius))
    mean_II = cv2.blur(src * src, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I * (1 - a)
    mean_a = cv2.blur(a, (radius, radius))
    mean_b = cv2.blur(b, (radius, radius))
    return mean_a * src + mean_b


def guided_filter_normalized(src: np.ndarray, weight: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Mask-weighted (normalized-convolution) guided filter: excluded pixels
    (weight=0) contribute nothing to any local moment. Same normalized-
    convolution technique retouch/undereye.py already uses for its masked
    low-pass estimate — not a new primitive invented for this experiment.
    """
    w = weight.astype(np.float32)
    wsum = cv2.blur(w, (radius, radius))
    wsum_safe = np.maximum(wsum, 1e-6)

    def wmean(x):
        return cv2.blur(x * w, (radius, radius)) / wsum_safe

    mean_I = wmean(src)
    mean_II = wmean(src * src)
    var_I = mean_II - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I * (1 - a)
    mean_a = wmean(a)
    mean_b = wmean(b)
    out = mean_a * src + mean_b
    # Where local support is ~0 (e.g. deep inside a large excluded mark),
    # normalized convolution is undefined; fall back to the unweighted
    # filter output there rather than propagating garbage.
    starved = wsum < 0.05
    if np.any(starved):
        plain = guided_filter_plain(src, radius, eps)
        out = np.where(starved, plain, out)
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def smooth_strength_to_sigma(smooth_strength: float):
    sigma_color = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
    sigma_space = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
    return sigma_color, sigma_space


def run_arm(scene, smooth_strength: float, mechanism: str, feather_px: int = 0):
    """mechanism in {"baseline", "blend_alpha", "filter_input"}.

    baseline: current shipped behavior — skin_mask covers everything,
        marks get smoothed exactly like any other skin pixel.
    blend_alpha: marks excluded from the FINAL blend only (candidates
        2/3/5 collapsed into one arm, feather_px sweeps hard->soft).
    filter_input: marks excluded from the guided filter's OWN local
        statistics via normalized convolution (candidate 4), in addition
        to being excluded from the final blend at the same feather.
    """
    canvas = scene["canvas"]
    marks = scene["marks_binary"].astype(np.float32) / 255.0
    face_width = scene.get("face_width", FACE_WIDTH)

    layers, k_low, k_mid = separate(canvas, face_width)
    low, mid_original, high = layers.low, layers.mid, layers.high

    if mechanism == "baseline":
        protect_mask = np.zeros_like(marks)
    else:
        protect_mask = marks

    m_raw = scene["skin_mask"] * (1.0 - protect_mask)
    if feather_px > 0 and mechanism != "baseline":
        # Re-add the excluded region back in AFTER a soft feather, so the
        # "protection" tapers rather than hard-cutting — this is exactly
        # what candidates 2 (hard) vs 3 (soft) differ on.
        soft_protect = cv2.GaussianBlur(protect_mask, (feather_px | 1, feather_px | 1), 0)
        m_raw = scene["skin_mask"] * (1.0 - soft_protect)

    adapt = _texture_adaptation_factor(high, m_raw)
    eff_smooth_strength = smooth_strength * adapt

    feather_r = max(9, int(face_width * 0.04) | 1)
    m_2d = cv2.GaussianBlur(m_raw, (feather_r, feather_r), 0)
    m_3d = m_2d[:, :, np.newaxis]

    mid = mid_original * (1.0 - m_3d * MID_REDUCTION)

    k_smooth = adaptive_ksize(face_width, factor=SMOOTH_K_FACTOR, minimum=SMOOTH_K_MIN)
    smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

    low_mid_f32 = np.clip(low + mid_original, 0, 255)
    sigma_color, sigma_space = smooth_strength_to_sigma(eff_smooth_strength)
    radius = max(2, int(round(sigma_space)))
    eps = sigma_color ** 2

    if mechanism == "filter_input":
        filter_weight = 1.0 - protect_mask  # exclude marks from the FILTER's own statistics
        smoothed_f32 = np.zeros_like(low_mid_f32)
        for c in range(3):
            smoothed_f32[:, :, c] = guided_filter_normalized(low_mid_f32[:, :, c], filter_weight, radius, eps)
    else:
        smoothed_f32 = np.zeros_like(low_mid_f32)
        for c in range(3):
            smoothed_f32[:, :, c] = guided_filter_plain(low_mid_f32[:, :, c], radius, eps)

    smoothed_low_bilateral = smoothed_f32 - mid_original
    blend_gaussian = min(1.0, eff_smooth_strength * GAUSSIAN_BLEND_FACTOR)
    smoothed_low_final = cv2.addWeighted(
        smoothed_low_bilateral, 1.0 - blend_gaussian, smoothed_low_gaussian, blend_gaussian, 0.0,
    )

    low_out = low * (1.0 - m_3d) + smoothed_low_final * m_3d
    processed = low_out + mid + high
    orig = layers.low + layers.mid + layers.high
    result = blend_masked(orig, processed, m_2d)
    return np.clip(result, 0, 255).astype(np.uint8), adapt, m_2d


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def radial_profile(img_gray: np.ndarray, center, max_r: int = 40):
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2).astype(np.int32)
    profile = []
    for radius in range(max_r):
        sel = r == radius
        if np.any(sel):
            profile.append(float(img_gray[sel].mean()))
        else:
            profile.append(float("nan"))
    return profile


def mark_contrast(img_gray: np.ndarray, center, mark_radius: int, ring_inner: int = 3, ring_outer: int = 12):
    """Mark level vs. surrounding-ring level, source-relative contrast."""
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mark_sel = r <= mark_radius * 0.6  # interior only, avoid edge feather
    ring_sel = (r >= mark_radius + ring_inner) & (r <= mark_radius + ring_outer)
    mark_level = float(img_gray[mark_sel].mean())
    ring_level = float(img_gray[ring_sel].mean())
    return ring_level - mark_level  # positive = mark darker than surround


def halo_score(img_gray: np.ndarray, baseline_gray: np.ndarray, center, mark_radius: int, max_r: int = 30):
    """Non-monotonic bump in (arm - baseline) radial profile just outside the
    mark = halo/island signature. Returns max abs deviation in the
    ring_outer..max_r band relative to the deviation right at the mark edge."""
    prof_arm = np.array(radial_profile(img_gray, center, max_r))
    prof_base = np.array(radial_profile(baseline_gray, center, max_r))
    diff = prof_arm - prof_base
    edge_band = diff[mark_radius:mark_radius + 4]
    outer_band = diff[mark_radius + 6:max_r]
    edge_val = float(np.nanmax(np.abs(edge_band))) if edge_band.size else 0.0
    outer_val = float(np.nanmax(np.abs(outer_band))) if outer_band.size else 0.0
    return {"edge_delta": edge_val, "outer_delta": outer_val}


def neighbor_continuity(img_gray: np.ndarray, center, mark_radius: int, probe_r: int = 25):
    """Std-dev of a ring well outside the mark: continuity of neighboring
    skin (not flattened, not artificially textured by the protection edge)."""
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    sel = (r >= mark_radius + 15) & (r <= mark_radius + probe_r)
    return float(img_gray[sel].std())


# ---------------------------------------------------------------------------
# Semi-synthetic real-face scene
# ---------------------------------------------------------------------------

REAL_FACE_PATH = "/private/tmp/retouch-meitu-bakeoff-20260904/DSCF2306/00_source.jpg"


def build_real_face_scene():
    """A real face crop (genuine skin texture/lighting/noise) with a
    synthetic mark composited at a KNOWN, hand-picked coordinate on clean
    cheek skin -- away from makeup/eyeliner, so this arm tests the
    smoothing-protection mechanisms against real skin statistics without
    the detect_marks eyeliner-misclassification confound already
    documented for this corpus (docs/plans/RESEARCH_FA01_PROTECTION_AND_ABSTENTION_AUDIT_2026_09_05.md
    used the same DSCF2306 family). Ground truth is the hand-picked
    coordinate, not a detector output -- ``detect_marks`` is never called
    in this arm.
    """
    if not os.path.exists(REAL_FACE_PATH):
        return None
    img = cv2.imread(REAL_FACE_PATH)
    if img is None:
        return None

    # Crop to a face-scale region so face_width-derived kernels are
    # comparable to the synthetic scene's FACE_WIDTH.
    x, y, w, h = 729, 363, 191, 219
    pad_x, pad_y = 30, 60
    crop = img[max(0, y - 20):y + h + pad_y, max(0, x - pad_x):x + w + pad_x].astype(np.float32)
    ch, cw = crop.shape[:2]

    # Clean right cheek, below the eye, above the jaw, no makeup in this
    # corpus face at this coordinate (visually confirmed).
    mark_center = (int(cw * 0.72), int(ch * 0.62))
    mark_radius = 6

    mark_bgr = crop[mark_center[1], mark_center[0]] * 0.45  # visibly darker than local skin
    m = np.zeros((ch, cw), dtype=np.float32)
    cv2.circle(m, mark_center, mark_radius, 1.0, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    canvas = crop * (1.0 - m[:, :, None]) + mark_bgr[None, None, :] * m[:, :, None]

    marks_mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(marks_mask, mark_center, mark_radius, 255, -1)

    return {
        "canvas": np.clip(canvas, 0, 255).astype(np.float32),
        "skin_mask": np.ones((ch, cw), dtype=np.float32),
        "marks_binary": marks_mask,
        "small_center": mark_center, "small_radius": mark_radius,
        "large_center": mark_center, "large_radius": mark_radius,  # single mark, reuse both slots
        "face_width": 191.0 * 1.3,  # bbox width with a margin, ballpark for k_low/k_mid sizing
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(scene, prefix: str):
    source_gray = cv2.cvtColor(scene["canvas"].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_00_source.png"), scene["canvas"].astype(np.uint8))
    cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_00_marks_mask.png"), scene["marks_binary"])

    strengths = [0.5, 0.9]  # moderate and high smoothing strength (criterion: high strength)
    feather_widths = [0, 9, 25]  # 0=hard exclusion, 9/25 = soft/wide feather

    results = {}
    baseline_by_strength = {}

    for smooth_strength in strengths:
        arm_key = f"baseline_s{smooth_strength}"
        out, adapt, _ = run_arm(scene, smooth_strength, "baseline")
        baseline_by_strength[smooth_strength] = out
        cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_{arm_key}.png"), out)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        results[arm_key] = {
            "mechanism": "baseline", "smooth_strength": smooth_strength, "feather_px": None,
            "adapt_factor": adapt,
            "small_mark_contrast": mark_contrast(gray, scene["small_center"], scene["small_radius"]),
            "large_mark_contrast": mark_contrast(gray, scene["large_center"], scene["large_radius"]),
            "small_neighbor_std": neighbor_continuity(gray, scene["small_center"], scene["small_radius"]),
            "large_neighbor_std": neighbor_continuity(gray, scene["large_center"], scene["large_radius"]),
        }

        for feather_px in feather_widths:
            for mechanism in ("blend_alpha", "filter_input"):
                arm_key = f"{mechanism}_s{smooth_strength}_f{feather_px}"
                out, adapt, _ = run_arm(scene, smooth_strength, mechanism, feather_px)
                cv2.imwrite(os.path.join(OUT_DIR, f"{prefix}_{arm_key}.png"), out)
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                base_gray = cv2.cvtColor(baseline_by_strength[smooth_strength], cv2.COLOR_BGR2GRAY)

                results[arm_key] = {
                    "mechanism": mechanism, "smooth_strength": smooth_strength, "feather_px": feather_px,
                    "adapt_factor": adapt,
                    "small_mark_contrast": mark_contrast(gray, scene["small_center"], scene["small_radius"]),
                    "large_mark_contrast": mark_contrast(gray, scene["large_center"], scene["large_radius"]),
                    "small_neighbor_std": neighbor_continuity(gray, scene["small_center"], scene["small_radius"]),
                    "large_neighbor_std": neighbor_continuity(gray, scene["large_center"], scene["large_radius"]),
                    "small_halo": halo_score(gray, base_gray, scene["small_center"], scene["small_radius"]),
                    "large_halo": halo_score(gray, base_gray, scene["large_center"], scene["large_radius"]),
                }

    results["source_reference"] = {
        "small_mark_contrast": mark_contrast(source_gray, scene["small_center"], scene["small_radius"]),
        "large_mark_contrast": mark_contrast(source_gray, scene["large_center"], scene["large_radius"]),
    }
    return results


def main():
    synthetic_scene = build_scene()
    synthetic_results = run_sweep(synthetic_scene, "synth")

    all_results = {"synthetic": synthetic_results}

    real_scene = build_real_face_scene()
    if real_scene is not None:
        real_results = run_sweep(real_scene, "real")
        all_results["real_face_semisynthetic"] = real_results
    else:
        all_results["real_face_semisynthetic"] = {"skipped": f"source not found: {REAL_FACE_PATH}"}

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)

    print(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"\nImages + results.json written to {OUT_DIR}")


if __name__ == "__main__":
    main()

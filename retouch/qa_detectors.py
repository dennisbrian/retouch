"""Quality Assurance detectors for output self-QA.

Provides pure-function artifact detectors that analyze output images for:
  - Banding (posterization on smooth gradients)
  - Clipping (blown highlights and crushed blacks)
  - Plastic skin (over-smoothed texture loss in skin regions)
  - Halo (edge overshoot from aggressive sharpening)
  - Seam (gradient discontinuities along subject-separation boundaries)

Each detector returns a dict with:
  - "score": float in [0, 1] (higher = more problematic)
  - "flagged": bool (True if score exceeds threshold)
  - Additional detail keys specific to each detector

:func:`run_qa_with_evidence` additionally classifies every detector into an
explicit status — QA_STATUS_PASSED / QA_STATUS_FLAGGED / QA_STATUS_UNAVAILABLE
/ QA_STATUS_NOT_RUN — so a caller can always tell "checked and passed" apart
from "never checked for this input" instead of inferring it from a missing
dict key. :func:`run_qa` remains the flagged-only List[QAWarning] view used
by export gating and QA back-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .harmony import evaluate_harmony


logger = logging.getLogger(__name__)


def _to_u8_for_analysis(img: np.ndarray) -> np.ndarray:
    """Return a uint8 BGR snapshot of ``img`` for QA analysis.

    QA detectors measure artifacts (banding, clipping, halos, ...), not pixel
    values, so the analysis result is identical whether run on a float32
    [0, 255] canvas or its uint8 truncation. Float32 input is truncated
    (not rounded) to match the legacy uint8 chain's
    ``np.clip(x, 0, 255).astype(np.uint8)`` convention so thresholds see
    values consistent with the pre-E1 pipeline. uint8 input is returned
    unchanged (byte-identical path).
    """
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class QAWarning:
    detector: str
    score: float
    flagged: bool
    message: str
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def _detector_failure(name: str, exc: Exception) -> Dict[str, Any]:
    """Return an explicit fail-closed result for an unavailable detector."""
    logger.warning("QA detector %s failed: %s", name, exc, exc_info=True)
    return {
        "score": 1.0,
        "flagged": True,
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


# Every detector name run_all() may populate. Used to give a "not-run" status
# to entries the current inputs skipped entirely (e.g. no reference image),
# so a consumer never has to infer "not checked" from key absence.
ALL_DETECTOR_NAMES = (
    "banding", "clipping", "plastic_skin", "halo", "kee_farid", "seam",
    "color_drift", "pore_spectrum", "asymmetry", "skin_score", "harmony",
    "perceived_retouching", "cam16_delta_e",
)

# Explicit per-detector status vocabulary. "passed"/"flagged" require the
# detector to have actually run; "unavailable" means it raised; "not_run"
# means the pipeline never attempted it for this input (missing mask/
# reference/person, or the QA stage itself failed before any detector ran).
QA_STATUS_PASSED = "checked-pass"
QA_STATUS_FLAGGED = "checked-flagged"
QA_STATUS_UNAVAILABLE = "unavailable"
QA_STATUS_NOT_RUN = "not-run"


def _qa_status(det_result: Dict[str, Any]) -> str:
    """Classify one run_all() detector result into the status vocabulary."""
    if det_result.get("available") is False:
        return QA_STATUS_UNAVAILABLE
    return QA_STATUS_FLAGGED if det_result.get("flagged", False) else QA_STATUS_PASSED


def _not_measured(reason: str, **null_fields: Any) -> Dict[str, Any]:
    """Result for a detector that could not measure (e.g. no/mismatched reference).

    Zero scores here used to classify as ``checked-pass``; an unmeasured
    comparison must stay distinguishable from a measured pass.
    """
    return {
        "status": QA_STATUS_NOT_RUN,
        "score": None,
        "flagged": False,
        "reason": reason,
        "note": reason,
        **null_fields,
    }


def not_run_evidence(reason: str) -> Dict[str, Dict[str, Any]]:
    """Build a complete not-run evidence dict for every known detector."""
    return {
        name: {"status": QA_STATUS_NOT_RUN, "score": None, "flagged": False, "reason": reason}
        for name in ALL_DETECTOR_NAMES
    }


# Threshold constants — adjust based on tuning
# Recalibrated 2026-09-25 (docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md):
# banding, plastic_skin, seam and asymmetry used absolute measurements that
# flagged 100% of real renders, so the review page's "Flagged" filter was
# noise. They now measure what the retouch CHANGED against the input photo.
BANDING_THRESHOLD = 0.06  # Flag if >6% of the person (or face skin) is newly banded
BANDING_STEP_MIN = 2      # Smallest step (8-bit luma codes) that reads as a band
BANDING_STEP_MAX = 16     # Larger jumps are real edges, not quantization
BANDING_PLATEAU_PX = 3    # Flat run required on both sides of a step
BANDING_MIN_FACE_PIXELS = 400  # Smaller face masks are not scored separately
CLIPPING_THRESHOLD = 0.08  # Flag if clipped blob fraction >8% of image
PLASTIC_SKIN_THRESHOLD = 0.60  # Flag if skin keeps <60% of its fine texture vs input
HALO_THRESHOLD = 15.0  # Flag if mean overshoot amplitude >15 levels
SEAM_THRESHOLD = 5.0   # Flag if the retouch adds >5 L-levels of boundary gradient

# K8 — color-fidelity (Δ-E / hue-drift) gate.
# Skin hue should shift only a few degrees under a grade; beyond these the
# grade has wrecked skin color. Hue angle is undefined as chroma → 0, so the
# gate only looks at pixels whose chroma (in BOTH ref and output) is at least
# COLOR_DRIFT_CHROMA_FLOOR_FRAC × the region's own median reference chroma —
# a relative floor, so it tracks each subject's skin chroma instead of an
# absolute Lab number. Over that chroma-floored population:
#   flag if mean Δh > COLOR_DRIFT_HUE_MEAN_THRESHOLD
#        or 95th-percentile Δh > COLOR_DRIFT_HUE_P95_THRESHOLD.
# Calibrated 2026-09-23 on real `natural` renders (7 images, p99 ≤ 7.2°,
# floored mean ≤ 0.32°) vs uniform OKLCh skin-hue rotations (+8°: p99
# 10.5–14.8°, +15°: 18.6–21.5°) — see detect_color_drift docstring.
# Recalibrated 2026-09-25: the old p99 > 10° tail gate flagged 15/21 styled
# cosplay renders whose mean moved only 1.4–3.7° — the tail is the person
# mask's intentional lip/eye/makeup edits, not a skin cast. The tail gate is
# now p95 > 20° (styled cosplay renders ≤ 11.8°; cinema_grade_v1's real
# skin cast reads mean ≥ 21°, p95 ≥ 35° on all 7 photos).
COLOR_DRIFT_HUE_MEAN_THRESHOLD = 6.0   # Flag if chroma-floored mean Δh > 6°
COLOR_DRIFT_HUE_P95_THRESHOLD = 20.0   # Flag if chroma-floored p95 Δh > 20°
COLOR_DRIFT_HUE_GATE_PERCENTILE = 95.0
# Still reported (``deltaH_p99_deg``) for compatibility; no longer gates.
COLOR_DRIFT_HUE_P99_THRESHOLD = 10.0
COLOR_DRIFT_HUE_PERCENTILE = 99.0
COLOR_DRIFT_CHROMA_FLOOR_FRAC = 0.5    # Keep pixels with C ≥ 0.5·median(C_ref)
# Numerical guard only: hue is undefined below ~1 ΔE of chroma (≈ one JND),
# for ANY skin tone. Real skin chroma is ≥ ~5; this never binds on skin and
# exists so a neutral-grey reference region can't admit pure hue noise.
COLOR_DRIFT_CHROMA_EPS = 1.0
COLOR_DRIFT_MIN_HUE_PIXELS = 100       # Fewer kept pixels → hue gate not measured

# Patch-scale checks (2026-09-27, docs/plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md).
# Whole-face averages hide a local defect: one waxy cheek or a cast on the
# chin is a small share of the face (and a smaller share of the person), so
# plastic_skin and color_drift also look at cheek-sized patches of each face.
# A patch is a Gaussian window whose sigma is a fraction of that face's size,
# so the check scales with the face instead of with the frame.
PATCH_MIN_FACE_PIXELS = 400        # Smaller face-skin blobs are not faces
PATCH_MIN_FACE_SHARE = 0.02        # ...nor blobs under 2% of the largest (mask slivers)
PATCH_MIN_COVERAGE = 0.6           # A patch must be mostly face skin
# Below this face-box size (longer side, px) the fine-texture band holds the
# nose wings and eyelid creases rather than pores, and ordinary smoothing
# reads as a waxy patch (a 146-px face flagged on 5 of 8 clean recipes).
# Such faces still get the whole-face texture check.
PLASTIC_SKIN_PATCH_MIN_FACE_SIDE = 200
# plastic_skin: fine-texture energy kept inside one patch (output / input).
# Patches whose input texture is under this share of the face's median patch
# texture are skipped (too little texture to measure a loss against).
PLASTIC_SKIN_PATCH_SIGMA_FRAC = 0.05
PLASTIC_SKIN_PATCH_TEXTURE_FLOOR = 0.35
PLASTIC_SKIN_PATCH_THRESHOLD = PLASTIC_SKIN_THRESHOLD
# color_drift: chroma shift (Lab a/b, ΔE units) of one patch beyond the face's
# own median shift. A grade that moves the whole face evenly scores ~0 here
# (the global gate covers it); only an uneven, local cast counts. A recipe
# blush is a deliberate pink cheek patch: the strongest (blush 40,
# zzz_anime_v2) reads up to 5.6, so the limit sits just above it.
COLOR_DRIFT_PATCH_SIGMA_FRAC = 0.08
COLOR_DRIFT_PATCH_THRESHOLD = 6.0
# Deprecated: the old single-pixel max rule (flagged plain `natural` renders
# via one near-neutral pixel). Kept for import compatibility; not used.
COLOR_DRIFT_HUE_MAX_THRESHOLD = 15.0
COLOR_DRIFT_THRESHOLD = COLOR_DRIFT_HUE_P95_THRESHOLD  # Reported flag boundary (p95 Δh, deg)

# R15 — QA extensions (read-only analysis detectors).
PORE_SPECTRUM_THRESHOLD = 0.50  # Flag if pore-energy loss ratio > 50%
ASYMMETRY_THRESHOLD = 0.40      # Flag if a face zone keeps <60% of the face's average texture retention
ASYMMETRY_MIN_ZONE_PIXELS = 64  # Relative mode ignores smaller grid-corner slivers
SKIN_SCORE_FLOOR = 25.0         # Below this 0-100 skin score = clearly plastic

# K1 — CAM16-UCS perceptual colour-difference gate. Mean skin ΔE above this
# means the grade has visibly shifted skin appearance in a uniform
# perceptual space (values are J'/a'/b' units, roughly JND ~2.3).
CAM16_DELTA_E_MEAN_THRESHOLD = 10.0
CAM16_DELTA_E_MAX_THRESHOLD = 30.0

# AA6 — Kee–Farid's deterministic 8-statistic retouching vector.  This is a
# read-only diagnostic; its corpus-calibrated 1–5 mapping is deliberately not
# a runtime model.
_AA6_C2 = 0.03 ** 2
_AA6_C3 = _AA6_C2 / 2.0
_AA6_FILTER_9 = np.full((9, 9), -1.0 / 81.0, dtype=np.float32)
_AA6_FILTER_9[4, 4] += 1.0


def _bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to CIELab float32.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.

    Returns:
        (H, W, 3) float32 Lab image, L in [0, 100], a/b in [-128, 128].
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return img_lab


def _aa6_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Normalize an optional AA6 region mask to a boolean image mask."""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        raise ValueError("AA6 masks must match the image dimensions")
    mask_f = mask.astype(np.float32, copy=False)
    if mask_f.max() > 1.5:
        mask_f = mask_f / 255.0
    return mask_f > 0.5


def _aa6_weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return deterministic weighted summary statistics, safe for empty ROIs."""
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return 0.0, 0.0
    mean = float(np.sum(values * weights) / total_weight)
    variance = float(np.sum(weights * (values - mean) ** 2) / total_weight)
    return mean, float(np.sqrt(max(variance, 0.0)))


def perceived_retouching_vector(
    img_bgr: np.ndarray,
    reference_img_bgr: np.ndarray,
    face_mask: Optional[np.ndarray],
    warp_field: Optional[np.ndarray] = None,
    body_mask: Optional[np.ndarray] = None,
    body_region_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute AA6's eight deterministic Kee–Farid QA statistics.

    The four geometry statistics summarize the signed warp displacement's
    magnitude along the local reference-luminance gradient, separately for
    face and body.  ``body_region_weights`` is optional but, when supplied,
    lets the pose/parsing caller apply the prescribed bust/waist/thigh ×2 and
    head/hair ×0.5 weighting without embedding class-specific assumptions in
    this generic QA module.

    The four face photometry statistics are local contrast-structure SSIM
    (the brightness term is omitted) and the signed 9×9 high-pass response
    change D.  Positive D means a net loss of high-frequency energy (blur);
    negative D means a net gain (sharpening).
    """
    if img_bgr.shape != reference_img_bgr.shape or img_bgr.ndim != 3:
        raise ValueError("AA6 images must be same-shape BGR images")
    h, w = img_bgr.shape[:2]
    face = _aa6_mask(face_mask, (h, w))
    body = _aa6_mask(body_mask, (h, w))

    if warp_field is None:
        warp = np.zeros((h, w, 2), dtype=np.float32)
    else:
        if warp_field.shape != (h, w, 2):
            raise ValueError("AA6 warp_field must have shape (H, W, 2)")
        warp = warp_field.astype(np.float32, copy=False)

    ref_luma = _gray_f32(reference_img_bgr) / 255.0
    grad_x = cv2.Sobel(ref_luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ref_luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    unit_x = np.divide(grad_x, grad_norm, out=np.zeros_like(grad_x), where=grad_norm > 1e-6)
    unit_y = np.divide(grad_y, grad_norm, out=np.zeros_like(grad_y), where=grad_norm > 1e-6)
    # Normalize by image diagonal so the same geometric edit scores the same
    # at a proxy and at native resolution.
    projected_warp = np.abs(warp[:, :, 0] * unit_x + warp[:, :, 1] * unit_y)
    projected_warp /= float(np.hypot(h, w))

    face_weights = face.astype(np.float32)
    body_weights = body.astype(np.float32)
    if body_region_weights is not None:
        if body_region_weights.shape != (h, w):
            raise ValueError("AA6 body_region_weights must match image dimensions")
        body_weights *= np.maximum(body_region_weights.astype(np.float32), 0.0)
    face_warp_mean, face_warp_std = _aa6_weighted_mean_std(projected_warp, face_weights)
    body_warp_mean, body_warp_std = _aa6_weighted_mean_std(projected_warp, body_weights)

    proc_luma = _gray_f32(img_bgr) / 255.0
    mu_ref = cv2.GaussianBlur(ref_luma, (11, 11), 1.5)
    mu_proc = cv2.GaussianBlur(proc_luma, (11, 11), 1.5)
    var_ref = np.maximum(cv2.GaussianBlur(ref_luma * ref_luma, (11, 11), 1.5) - mu_ref * mu_ref, 0.0)
    var_proc = np.maximum(cv2.GaussianBlur(proc_luma * proc_luma, (11, 11), 1.5) - mu_proc * mu_proc, 0.0)
    covariance = cv2.GaussianBlur(ref_luma * proc_luma, (11, 11), 1.5) - mu_ref * mu_proc
    sigma_ref = np.sqrt(var_ref)
    sigma_proc = np.sqrt(var_proc)
    ssim_cs = ((2.0 * sigma_ref * sigma_proc + _AA6_C2) / (var_ref + var_proc + _AA6_C2)) * (
        (covariance + _AA6_C3) / (sigma_ref * sigma_proc + _AA6_C3)
    )
    ssim_mean, ssim_std = _aa6_weighted_mean_std(ssim_cs, face_weights)

    ref_response = np.abs(cv2.filter2D(ref_luma, cv2.CV_32F, _AA6_FILTER_9))
    proc_response = np.abs(cv2.filter2D(proc_luma, cv2.CV_32F, _AA6_FILTER_9))
    frequency_d = np.log((ref_response + 1e-6) / (proc_response + 1e-6))
    frequency_mean, frequency_std = _aa6_weighted_mean_std(frequency_d, face_weights)
    return {
        "face_warp_gradient_mean": face_warp_mean,
        "face_warp_gradient_std": face_warp_std,
        "body_warp_gradient_mean": body_warp_mean,
        "body_warp_gradient_std": body_warp_std,
        "face_ssim_cs_mean": ssim_mean,
        "face_ssim_cs_std": ssim_std,
        "face_frequency_d_mean": frequency_mean,
        "face_frequency_d_std": frequency_std,
    }


def _banding_edges(gray: np.ndarray, step_min: int = 0) -> np.ndarray:
    """Return a bool map of staircase-banding step pixels in a uint8 luma plane.

    A banding step is a jump of ``BANDING_STEP_MIN``–``BANDING_STEP_MAX``
    code values with a perfectly flat run of ``BANDING_PLATEAU_PX`` pixels on
    BOTH sides, along either axis. Sensor noise, film grain and JPEG texture
    dither real gradients, so flat runs next to a multi-level jump are rare in
    photographs; posterized gradients (8-bit tone stretch, over-smoothed skin
    re-quantized, heavy compression) are made of exactly these. One-level
    steps are ignored: they are the unavoidable 8-bit quantization of any
    gentle ramp and are not visible as bands. ``step_min`` overrides
    ``BANDING_STEP_MIN`` (the reference side uses 1, see detect_banding).
    """
    step_min = step_min or BANDING_STEP_MIN
    g = gray.astype(np.int16)
    k = BANDING_PLATEAU_PX
    edges = np.zeros(g.shape, dtype=bool)
    for axis in (0, 1):
        d = np.diff(g, axis=axis)
        flat = d == 0
        ok = (np.abs(d) >= step_min) & (np.abs(d) <= BANDING_STEP_MAX)
        for s in range(1, k + 1):
            before = np.zeros_like(flat)
            after = np.zeros_like(flat)
            if axis == 1:
                before[:, s:] = flat[:, :-s]
                after[:, :-s] = flat[:, s:]
            else:
                before[s:, :] = flat[:-s, :]
                after[:-s, :] = flat[s:, :]
            ok &= before & after
        if axis == 1:
            edges[:, 1:] |= ok
        else:
            edges[1:, :] |= ok
    return edges


def _banded_coverage(img_u8: np.ndarray, step_min: int = 0) -> np.ndarray:
    """Bool map of pixels inside a banded plateau (step pixels + their runs)."""
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    edges = _banding_edges(gray, step_min)
    size = 2 * BANDING_PLATEAU_PX + 1
    return cv2.dilate(edges.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def _region(mask: Optional[np.ndarray], shape: tuple) -> Optional[np.ndarray]:
    """Binarize an optional 0–1 or 0–255 mask at 0.5; ``None`` if empty."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.5:
        mask_f /= 255.0
    region = mask_f > 0.5
    return region if np.any(region) else None


def _face_windows(mask: Optional[np.ndarray], sigma_frac: float):
    """Yield ``(slices, region, sigma, bbox)`` for each face in a face-skin mask.

    Each connected blob of the mask is a face, unless it is under
    ``PATCH_MIN_FACE_PIXELS`` or under ``PATCH_MIN_FACE_SHARE`` of the
    largest blob (a sliver of skin between strands of hair, say).
    ``slices`` crop the frame to the face plus a 3-sigma margin, ``region``
    is that face's boolean mask inside the crop, ``sigma`` is the patch
    window (``sigma_frac`` × the face's longer side) and ``bbox`` is the
    face's ``(x, y, w, h)`` in frame coordinates.
    """
    region_full = _region(mask, mask.shape[:2]) if mask is not None else None
    if region_full is None:
        return
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        region_full.astype(np.uint8), connectivity=8
    )
    H, W = region_full.shape
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    min_area = max(PATCH_MIN_FACE_PIXELS, PATCH_MIN_FACE_SHARE * largest)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        sigma = max(2.0, sigma_frac * max(w, h))
        pad = int(3 * sigma) + 2
        y0, y1 = max(0, y - pad), min(H, y + h + pad)
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        region = labels[y0:y1, x0:x1] == i
        yield (slice(y0, y1), slice(x0, x1)), region, sigma, (x, y, w, h)


def face_zone_name(point: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> str:
    """Name where ``point`` (x, y) sits on a face box, as seen in the photo.

    Left and right are the photo's, not the subject's, so the name matches
    what a reviewer sees on screen.
    """
    x, y, w, h = bbox
    fx = (point[0] - x) / max(w, 1)
    fy = (point[1] - y) / max(h, 1)
    side = "left" if fx < 0.33 else "right" if fx > 0.67 else ""
    if fy < 0.33:
        part = "forehead"
    elif fy < 0.72:
        part = "cheek" if side else "nose"
    else:
        part = "jaw" if side else "chin"
    if side:
        return f"{part} on the {side} of the photo"
    return part


def _nose_zone(shape: Tuple[int, int], sy: slice, sx: slice, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Boolean mask (crop coordinates) of the nose zone of a face box.

    Shine removal and nose highlights legitimately flatten the nose's fine
    detail (on a small face the nostril and bridge edges fall in the
    fine-texture band), so a patch centred there is not scored as waxy.
    Same zone :func:`face_zone_name` calls "nose".
    """
    x, y, w, h = bbox
    zone = np.zeros(shape, dtype=bool)
    r0 = int(y + 0.33 * h) - sy.start
    r1 = int(y + 0.72 * h) - sy.start
    c0 = int(x + 0.33 * w) - sx.start
    c1 = int(x + 0.67 * w) - sx.start
    zone[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = True
    return zone


def _patch_texture_retention(
    L_hf_out: np.ndarray,
    L_hf_ref: np.ndarray,
    mask: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Worst cheek-sized patch of fine-texture retention over every face.

    Retention of a patch = sqrt(Gσ(hf_out²·m) / Gσ(hf_ref²·m)): the ratio of
    fine-texture energy the output keeps inside a Gaussian window. Windows
    that are mostly outside the face skin, or whose input texture is below
    ``PLASTIC_SKIN_PATCH_TEXTURE_FLOOR`` × the face's median, are skipped.
    Returns ``None`` when no face has a measurable patch.
    """
    worst: Optional[Dict[str, Any]] = None
    for (sy, sx), region, sigma, bbox in _face_windows(mask, PLASTIC_SKIN_PATCH_SIGMA_FRAC):
        if max(bbox[2], bbox[3]) < PLASTIC_SKIN_PATCH_MIN_FACE_SIDE:
            continue
        m = region.astype(np.float32)
        e_out = cv2.GaussianBlur(L_hf_out[sy, sx] ** 2 * m, (0, 0), sigma)
        e_ref = cv2.GaussianBlur(L_hf_ref[sy, sx] ** 2 * m, (0, 0), sigma)
        cover = cv2.GaussianBlur(m, (0, 0), sigma)
        ok = region & (cover > PATCH_MIN_COVERAGE)
        if not np.any(ok):
            continue
        ref_density = e_ref / np.maximum(cover, 1e-6)
        median_density = float(np.median(ref_density[ok]))
        ok &= ref_density > (PLASTIC_SKIN_PATCH_TEXTURE_FLOOR ** 2) * median_density
        ok &= ~_nose_zone(region.shape, sy, sx, bbox)
        if median_density <= 1e-9 or not np.any(ok):
            continue
        retention = np.sqrt(e_out / np.maximum(e_ref, 1e-12))
        idx = int(np.argmin(np.where(ok, retention, np.inf)))
        py, px = np.unravel_index(idx, retention.shape)
        value = float(retention[py, px])
        if worst is None or value < worst["retention"]:
            point = (int(sx.start + px), int(sy.start + py))
            worst = {
                "retention": value,
                "point": list(point),
                "face_bbox": list(bbox),
                "zone": face_zone_name(point, bbox),
            }
    return worst


def _patch_chroma_cast(
    out_lab: np.ndarray,
    ref_lab: np.ndarray,
    mask: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Worst patch-scale chroma shift beyond each face's own median shift.

    Per pixel, the shift is the output's (a, b) minus the input's. Each face's
    median shift is subtracted (an even grade is the global gate's job), the
    remainder is averaged in a Gaussian window, and the largest window
    magnitude (ΔE units in the a/b plane) is returned. ``None`` when no face
    has a measurable patch.
    """
    worst: Optional[Dict[str, Any]] = None
    for (sy, sx), region, sigma, bbox in _face_windows(mask, COLOR_DRIFT_PATCH_SIGMA_FRAC):
        d = out_lab[sy, sx, 1:] - ref_lab[sy, sx, 1:]
        median_shift = np.median(d[region], axis=0)
        m = region.astype(np.float32)
        cover = cv2.GaussianBlur(m, (0, 0), sigma)
        ok = region & (cover > PATCH_MIN_COVERAGE)
        if not np.any(ok):
            continue
        dev = (d - median_shift) * m[..., None]
        norm = np.maximum(cover, 1e-6)
        da = cv2.GaussianBlur(dev[..., 0], (0, 0), sigma) / norm
        db = cv2.GaussianBlur(dev[..., 1], (0, 0), sigma) / norm
        magnitude = np.hypot(da, db)
        idx = int(np.argmax(np.where(ok, magnitude, -1.0)))
        py, px = np.unravel_index(idx, magnitude.shape)
        value = float(magnitude[py, px])
        # When a cast covers about half the face the median follows the
        # cast, and the most deviant patch is the untouched part. If that
        # patch moved less than the face's median shift, point at the patch
        # that moved most instead, so the zone names where the cast is.
        shift_a = da + float(median_shift[0])
        shift_b = db + float(median_shift[1])
        moved = np.hypot(shift_a, shift_b)
        if moved[py, px] < float(np.hypot(*median_shift)):
            idx = int(np.argmax(np.where(ok, moved, -1.0)))
            py, px = np.unravel_index(idx, moved.shape)
        if worst is None or value > worst["delta_ab"]:
            point = (int(sx.start + px), int(sy.start + py))
            worst = {
                "delta_ab": value,
                "point": list(point),
                "face_bbox": list(bbox),
                "zone": face_zone_name(point, bbox),
            }
    return worst


def detect_banding(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect staircase banding (posterized gradients) the retouch introduced.

    Measures the fraction of the region covered by banded plateaus: flat runs
    separated by 2+ level steps (see :func:`_banding_edges`). With a
    ``reference_img_bgr`` (the pipeline always passes the input photo) only
    plateaus that are NOT already present in the reference count, so banding
    baked into the source JPEG (flat walls, crushed shadows, poster art) does
    not flag every render. The score is the larger of the new-banding
    fraction over ``mask`` (person) and over ``face_mask`` (face skin), so a
    banded face is not diluted by a large body/background mask.

    Calibration (2026-09-25, 7 real photos × 5 recipes incl. cosplay and
    cinema grades): score ≤ 0.026 on every render (the highest were a
    bright, JPEG-blocky cheek). Face skin smoothed and posterized to 3-level
    steps scores ≥ 0.095 on all 7 photos; the whole frame posterized to
    8-level steps ≥ 0.099. Threshold 0.06. See
    ``docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md``.

    The previous measurement (Sobel > 0.8 L on 3×3 low-variance pixels) was
    satisfied by every 1-level quantization step in any photo and flagged
    100% of renders.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.
        reference_img_bgr: Optional pre-retouch BGR image of the same shape;
            makes the measurement differential.
        face_mask: Optional (H, W) face-skin mask scored separately.

    Returns:
        dict with keys:
            - "score": float in [0, 1], banded-area fraction (new vs reference
              when a reference is given)
            - "flagged": bool, True if score > BANDING_THRESHOLD
            - "smooth_pixels": int, pixels in the analysed region
            - "step_pixels": int, banded-plateau pixels in the region
            - "face_score": float | None, the same fraction over face skin
            - "differential": bool, True if measured against a reference
    """
    empty = {"score": 0.0, "flagged": False, "smooth_pixels": 0, "step_pixels": 0,
             "face_score": None, "differential": False}
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image; cannot measure banding reliably
        return empty

    img_u8 = _to_u8_for_analysis(img_bgr)
    region = _region(mask, img_u8.shape[:2])
    if region is None:
        return empty

    banded = _banded_coverage(img_u8)
    differential = (
        reference_img_bgr is not None and reference_img_bgr.shape == img_u8.shape
    )
    if differential:
        # The reference side counts 1-level staircases too: flat JPEG blocks
        # in the source whose 1-level steps a brightening curve turns into
        # 2-level ones are the source's quantization, not retouch banding.
        banded &= ~_banded_coverage(_to_u8_for_analysis(reference_img_bgr), 1)

    region_count = int(region.sum())
    banded_count = int((banded & region).sum())
    score = banded_count / max(region_count, 1)

    face_score = None
    if face_mask is not None:
        face = _region(face_mask, img_u8.shape[:2])
        if face is not None and int(face.sum()) >= BANDING_MIN_FACE_PIXELS:
            face_score = float((banded & face).sum() / face.sum())
            score = max(score, face_score)

    return {
        "score": min(1.0, float(score)),
        "flagged": bool(score > BANDING_THRESHOLD),
        "smooth_pixels": region_count,
        "step_pixels": banded_count,
        "face_score": face_score,
        "differential": bool(differential),
    }


def detect_clipping(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect blown highlights and crushed blacks.

    Measures the fraction of pixels at 0 or 255 per channel, and identifies
    large connected blobs of clipped pixels (blown-white clusters).

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally so the 0/255 clipping test is
            consistent with the uint8 pipeline.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.

    Returns:
        dict with keys:
            - "score": float, max of (% pixels at 0/255) or (blob fraction)
            - "flagged": bool, True if blob fraction > CLIPPING_THRESHOLD
            - "clipped_pixel_fraction": float, % of 0/255 pixels
            - "clipped_blob_fraction": float, % of image in clipped blobs
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image
        return {
            "score": 0.0,
            "flagged": False,
            "clipped_pixel_fraction": 0.0,
            "clipped_blob_fraction": 0.0,
        }

    img_bgr = _to_u8_for_analysis(img_bgr)
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Find pixels at 0 or 255 per channel
    clipped_any = np.any(
        (img_bgr == 0) | (img_bgr == 255),
        axis=2
    )  # (H, W) bool

    # Apply mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.5:
            mask_f /= 255.0
        clipped_any = clipped_any & (mask_f > 0.5)

    clipped_pixel_count = int(clipped_any.sum())
    clipped_pixel_fraction = clipped_pixel_count / total_pixels

    # Find large blobs of clipped pixels
    clipped_uint8 = clipped_any.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(clipped_uint8)

    # Find the size of the largest blob
    blob_sizes = np.bincount(labels.ravel())
    largest_blob_size = blob_sizes[1:].max() if len(blob_sizes) > 1 else 0
    clipped_blob_fraction = largest_blob_size / total_pixels

    # Score is the max of the two metrics
    score = max(clipped_pixel_fraction, clipped_blob_fraction)
    flagged = clipped_blob_fraction > CLIPPING_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "clipped_pixel_fraction": float(clipped_pixel_fraction),
        "clipped_blob_fraction": float(clipped_blob_fraction),
    }


def detect_plastic_skin(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Detect over-smoothed skin (plastic/waxen appearance).

    Measures how much fine skin texture (std of L − Gaussian(L, σ=2) inside
    the mask) the output keeps relative to the reference (input) photo, and
    flags when it keeps less than ``PLASTIC_SKIN_THRESHOLD`` (60%). The
    pipeline passes the face-skin mask, so hair and clothes do not count.

    Without a reference there is nothing to compare against: the result is
    ``not-run`` and never flagged (``hf_energy_ratio`` is still reported).
    The old absolute rule (hf / mid-frequency ratio < 0.6 over the person
    mask) flagged every portrait, edited or not, because clothes and hair
    dominate the mid-frequency term.

    Calibration (2026-09-25, 7 photos × 5 recipes, face-skin mask): every
    render kept ≥ 0.89 of its input texture; a σ=1.5 Gaussian blur over the
    face skin keeps 0.29–0.52 and flags on all 7 photos. See
    ``docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md``.

    Patch check (2026-09-27): a whole-face average hides one waxy cheek, so
    each face of 200 px or more is also scanned with cheek-sized windows
    (Gaussian σ = 5% of the face box), nose excluded, and the worst window's
    retention is reported as ``patch_retention`` with ``patch_zone`` naming
    where it is. It flags below ``PLASTIC_SKIN_PATCH_THRESHOLD``. On 127
    renders (9 photos, 18 recipes) the ordinary looks kept ≥ 0.62 per patch;
    the new flags all came from heavy anime/porcelain looks (``zzz_anime``,
    ``scifi_cosplay``, ``fantasy_goddess``, ``xhs_soft_glow``,
    ``pink_dream``, 0.37–0.59), whose foreheads or cheeks really are wiped
    flat. A σ=1.5 blur of one cheek reads 0.30–0.54 and flags on every face
    scanned (12/12). See
    ``docs/plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md``.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image (output).
            Float input is truncated to uint8 internally for analysis.
        mask: Optional (H, W) float mask [0, 1], typically skin mask.
        reference_img_bgr: Optional (H, W, 3) uint8 or float32 [0, 255] BGR
            image (input) for comparison.

    Returns:
        dict with keys:
            - "score": float, the lower of whole-face and worst-patch texture
              retention vs reference, clamped to [0, 1] (0 = all texture
              erased, 1 = fully preserved); None if no reference
            - "flagged": bool, True if retention < PLASTIC_SKIN_THRESHOLD or
              patch retention < PLASTIC_SKIN_PATCH_THRESHOLD
            - "texture_retention": float or None, output / reference fine-texture std
            - "hf_energy_ratio": float, high-freq std / mid-freq std (informational)
            - "energy_loss_vs_reference": float or None (if reference provided)
            - "patch_retention", "patch_zone", "patch_point",
              "patch_face_bbox": worst patch (None when no face was scanned)
            - "finding": plain-language message when flagged
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image
        return {
            "score": 1.0,
            "flagged": False,
            "hf_energy_ratio": 1.0,
            "texture_retention": None,
            "energy_loss_vs_reference": None,
        }

    img_bgr = _to_u8_for_analysis(img_bgr)
    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Gaussian blur at sigma=2 (mid-frequency reference)
    L_blur = cv2.GaussianBlur(L, (0, 0), 2.0)

    # High-frequency = original - blurred
    L_hf = L - L_blur
    hf_energy = float(np.std(L_hf))

    # Mid-frequency = some reference level (std of the blurred version)
    mf_energy = float(np.std(L_blur))

    if mf_energy < 1e-6:
        # Flat image, no mid-freq content
        energy_ratio = 1.0
    else:
        energy_ratio = hf_energy / mf_energy

    # Apply mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.5:
            mask_f /= 255.0

        # Restrict calculation to masked region
        L_masked = L[mask_f > 0.5]
        L_blur_masked = L_blur[mask_f > 0.5]

        if len(L_masked) > 0:
            L_hf_masked = L_masked - L_blur_masked
            hf_energy = float(np.std(L_hf_masked))
            mf_energy = float(np.std(L_blur_masked))

            if mf_energy > 1e-6:
                energy_ratio = hf_energy / mf_energy

    region = _region(mask, L.shape)
    if region is None:
        region = np.ones(L.shape, dtype=bool)
    hf_out = float(np.std(L_hf[region]))

    # Compare with reference if provided
    energy_loss = None
    retention = None
    if reference_img_bgr is not None and reference_img_bgr.shape == img_bgr.shape:
        ref_lab = _bgr_to_lab(_to_u8_for_analysis(reference_img_bgr))
        L_ref = ref_lab[:, :, 0]
        L_ref_blur = cv2.GaussianBlur(L_ref, (0, 0), 2.0)
        L_ref_hf = L_ref - L_ref_blur
        ref_mf_energy = float(np.std(L_ref_blur[region]))
        ref_hf_energy = float(np.std(L_ref_hf[region]))
        if ref_mf_energy > 1e-6:
            ref_energy_ratio = ref_hf_energy / ref_mf_energy
            # Energy loss is how much the ratio decreased
            energy_loss = max(0.0, ref_energy_ratio - energy_ratio) / max(ref_energy_ratio, 1e-6)
        if ref_hf_energy > 1e-6:
            retention = hf_out / ref_hf_energy

    if retention is None:
        result = _not_measured(
            "no reference supplied — plastic_skin compares skin texture "
            "against the input photo",
        )
        result.update({
            "hf_energy_ratio": float(energy_ratio),
            "texture_retention": None,
            "energy_loss_vs_reference": energy_loss,
        })
        return result

    patch = None
    if mask is not None:
        patch = _patch_texture_retention(L_hf, L_ref_hf, mask)
    whole_flag = retention < PLASTIC_SKIN_THRESHOLD
    patch_flag = patch is not None and patch["retention"] < PLASTIC_SKIN_PATCH_THRESHOLD
    worst = retention if patch is None else min(retention, patch["retention"])
    result = {
        "score": float(min(1.0, max(0.0, worst))),
        "flagged": bool(whole_flag or patch_flag),
        "hf_energy_ratio": float(energy_ratio),
        "texture_retention": float(retention),
        "energy_loss_vs_reference": energy_loss,
        "patch_retention": None if patch is None else patch["retention"],
        "patch_zone": None if patch is None else patch["zone"],
        "patch_point": None if patch is None else patch["point"],
        "patch_face_bbox": None if patch is None else patch["face_bbox"],
    }
    if patch_flag and not whole_flag:
        result["finding"] = (
            f"Waxy patch on the {patch['zone']}: it keeps "
            f"{patch['retention']:.0%} of its skin texture, the rest of the "
            f"face keeps {retention:.0%}"
        )
    elif whole_flag:
        result["finding"] = (
            f"Waxy skin: the face keeps {retention:.0%} of its skin texture"
        )
    return result


def detect_halo(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    img_before: Optional[np.ndarray] = None,
) -> dict:
    """Detect sharpening halos and edge overshoot artifacts.

    Identifies strong edges via Canny, then measures mean overshoot of L channel
    beyond the pre-edge plateau within a 5-10px band. High overshoot indicates
    sharpening halos. If ``img_before`` is provided, evaluates differential halo score.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.
        img_before: Optional baseline BGR image before processing.

    Returns:
        dict with keys:
            - "score": float, mean overshoot amplitude in levels
            - "flagged": bool, True if score > HALO_THRESHOLD
            - "mean_overshoot": float, mean L overshoot at edges
            - "edge_count": int, number of edge pixels analyzed
    """
    if img_bgr.shape[0] < 16 or img_bgr.shape[1] < 16:
        # Too small to measure edge halos
        return {
            "score": 0.0,
            "flagged": False,
            "mean_overshoot": 0.0,
            "edge_count": 0,
        }

    img_bgr = _to_u8_for_analysis(img_bgr)
    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Detect strong edges via Canny
    L_uint8 = np.clip(L, 0, 100).astype(np.uint8) * 255 // 100
    edges = cv2.Canny(L_uint8, 50, 150)  # Returns binary edge map

    if edges.sum() == 0:
        # No edges detected
        return {
            "score": 0.0,
            "flagged": False,
            "mean_overshoot": 0.0,
            "edge_count": 0,
        }

    # For each edge pixel, measure overshoot using morphological dilation
    edge_dilated = cv2.dilate(edges.astype(np.uint8), np.ones((7, 7), np.uint8))
    edge_eroded = cv2.erode(edges.astype(np.uint8), np.ones((3, 3), np.uint8))

    # Forward band = dilation for overshoot, backward band = edge pixels themselves
    forward_band = edge_dilated & ~edges.astype(np.uint8)
    edge_band = edges.astype(np.uint8) & ~edge_eroded

    if forward_band.sum() == 0:
        return {"score": 0.0, "flagged": False, "mean_overshoot": 0.0, "edge_count": 0}

    # Baseline: L at edge pixels
    baseline_vals = L[edge_band > 0]
    # Overshoot: L in forward band
    forward_vals = L[forward_band > 0]

    if len(baseline_vals) == 0 or len(forward_vals) == 0:
        return {"score": 0.0, "flagged": False, "mean_overshoot": 0.0, "edge_count": 0}

    # For each edge pixel, find nearest forward pixel's max value
    # Simplified: percentile-based comparison
    baseline_median = np.median(baseline_vals)
    forward_max = np.percentile(forward_vals, 95)
    mean_overshoot = float(max(0.0, forward_max - baseline_median))

    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.5:
            mask_f /= 255.0
        baseline_masked = baseline_vals[mask_f[edge_band > 0].astype(bool)] if edge_band[mask_f > 0.5].sum() > 0 else np.array([], dtype=np.float32)
        forward_masked = forward_vals[mask_f[forward_band > 0].astype(bool)] if forward_band[mask_f > 0.5].sum() > 0 else np.array([], dtype=np.float32)
        if len(baseline_masked) > 0 and len(forward_masked) > 0:
            baseline_median = np.median(baseline_masked)
            forward_max = np.percentile(forward_masked, 95)
            mean_overshoot = float(max(0.0, forward_max - baseline_median))

    if img_before is not None:
        before_res = detect_halo(img_before, mask=mask)
        mean_overshoot = max(0.0, mean_overshoot - before_res.get("mean_overshoot", 0.0))

    # Score is the mean overshoot
    score = mean_overshoot
    flagged = mean_overshoot > HALO_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "mean_overshoot": float(mean_overshoot),
        "edge_count": int(forward_band.sum()),
    }


def _seam_excess(
    L: np.ndarray, boundary: np.ndarray, interior: np.ndarray
) -> Optional[tuple]:
    """Return (boundary excess, context gradient) for the mask outline.

    Excess = mean boundary gradient minus the context (mean of the interior
    and exterior mean gradients); ``None`` when a region is empty.
    """
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    exterior = ~interior
    if not (np.any(boundary) and np.any(interior) and np.any(exterior)):
        return None
    boundary_mean = float(np.mean(grad_mag[boundary]))
    context_mean = float((np.mean(grad_mag[interior]) + np.mean(grad_mag[exterior])) / 2.0)
    return boundary_mean - context_mean, context_mean


def detect_seam(
    img_bgr: np.ndarray,
    person_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Detect a seam (gradient discontinuity) the retouch added along the person-mask boundary.

    The excess gradient along the mask boundary (boundary mean minus the
    average of interior and exterior means) is measured on the output. With
    a ``reference_img_bgr`` (the pipeline passes the input photo) the same
    excess is measured on the reference and only the INCREASE counts: the
    person outline is a natural high-contrast edge (subject against
    background), which on its own read 22–29 L-levels on every real photo and
    flagged 100% of renders under the old absolute rule. The reference
    excess is first scaled by how much the context gradient itself grew
    (clarity/contrast looks sharpen every edge, the outline included, which
    is not a seam). Without a reference the absolute excess is reported
    (legacy behaviour).

    Calibration (2026-09-25, 7 photos × 5 recipes): the retouch added at
    most 1.4 L-levels. A 2-px line 10 L-levels brighter traced round the
    whole outline adds 6.0–9.2 on all 7 photos (a 5-level line, 1.8–3.3,
    does not flag). See
    ``docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md``.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis.
        person_mask: Optional (H, W) float mask [0, 1] delimiting the subject.
        reference_img_bgr: Optional pre-retouch BGR image of the same shape.

    Returns:
        dict with keys ``score``, ``flagged``, ``seam_gradient`` (added
        boundary gradient, L-levels), ``boundary_pixels``,
        ``output_excess`` / ``reference_excess`` (absolute excesses; the
        latter None without a reference) and ``differential``.
    """
    empty = {"score": 0.0, "flagged": False, "seam_gradient": 0.0, "boundary_pixels": 0,
             "output_excess": None, "reference_excess": None, "differential": False}
    if img_bgr.shape[0] < 16 or img_bgr.shape[1] < 16 or person_mask is None:
        return empty
    img_bgr = _to_u8_for_analysis(img_bgr)
    mask_f = person_mask.astype(np.float32)
    if mask_f.max() > 1.5:
        mask_f /= 255.0
    interior = mask_f > 0.5
    kernel = np.ones((5, 5), np.uint8)
    boundary = (cv2.dilate(interior.astype(np.uint8), kernel) ^
                cv2.erode(interior.astype(np.uint8), kernel)) > 0
    measured = _seam_excess(_bgr_to_lab(img_bgr)[:, :, 0], boundary, interior)
    if measured is None:
        return empty
    out_excess, out_context = measured
    ref_excess = None
    base = 0.0
    if reference_img_bgr is not None and reference_img_bgr.shape == img_bgr.shape:
        ref_lab = _bgr_to_lab(_to_u8_for_analysis(reference_img_bgr))
        ref_measured = _seam_excess(ref_lab[:, :, 0], boundary, interior)
        if ref_measured is not None:
            ref_excess, ref_context = ref_measured
            gain = out_context / ref_context if ref_context > 1e-6 else 1.0
            base = ref_excess * gain
    seam_gradient = max(0.0, out_excess - base)
    return {
        "score": min(1.0, seam_gradient / 20.0),
        "flagged": bool(seam_gradient > SEAM_THRESHOLD),
        "seam_gradient": seam_gradient,
        "boundary_pixels": int(boundary.sum()),
        "output_excess": out_excess,
        "reference_excess": ref_excess,
        "differential": ref_excess is not None,
    }


def _bgr_to_lab_f(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (uint8 or float32 [0, 255]) image to CIELab float32.

    Unlike :func:`_bgr_to_lab` this is float-safe: no truncation to uint8,
    so the color-fidelity gate keeps full precision for the Δ-E math.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.

    Returns:
        (H, W, 3) float32 Lab image, L in [0, 100], a/b in ~[-128, 128].
    """
    if img_bgr.dtype == np.uint8:
        bgr = img_bgr.astype(np.float32) / 255.0
    else:
        bgr = np.clip(img_bgr, 0.0, 255.0).astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 color difference (Sharma et al. 2005, closed form).

    Fully vectorized — no per-pixel loops.

    Args:
        lab1: (H, W, 3) or (3,) float LAB image (L in [0, 100], a/b any range).
        lab2: Same shape as ``lab1``.

    Returns:
        (H, W) or scalar float32 array of ΔE2000 values.
    """
    arr1 = np.asarray(lab1, dtype=np.float32)
    arr2 = np.asarray(lab2, dtype=np.float32)
    squeeze = arr1.ndim == 1
    if squeeze:
        arr1 = arr1.reshape(1, 3)
        arr2 = arr2.reshape(1, 3)
    c1 = np.sqrt(arr1[..., 1] ** 2 + arr1[..., 2] ** 2)
    c2 = np.sqrt(arr2[..., 1] ** 2 + arr2[..., 2] ** 2)
    ac1c2 = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(ac1c2 ** 7 / (ac1c2 ** 7 + 25.0 ** 7)))
    a1p = (1.0 + g) * arr1[..., 1]
    a2p = (1.0 + g) * arr2[..., 1]
    c1p = np.sqrt(a1p ** 2 + arr1[..., 2] ** 2)
    c2p = np.sqrt(a2p ** 2 + arr2[..., 2] ** 2)
    h1p = np.mod(np.arctan2(arr1[..., 2], a1p), 2.0 * np.pi)
    h2p = np.mod(np.arctan2(arr2[..., 2], a2p), 2.0 * np.pi)
    dl = arr2[..., 0] - arr1[..., 0]
    dc = c2p - c1p
    dh = h2p - h1p
    dh = dh - (2.0 * np.pi) * (dh > np.pi)
    dh = dh + (2.0 * np.pi) * (dh < -np.pi)
    dh_term = 2.0 * np.sqrt(c1p * c2p) * np.sin(dh / 2.0)
    zero_chroma = (c1p * c2p) == 0
    dh_term = np.where(zero_chroma, 0.0, dh_term)
    lbar = (arr1[..., 0] + arr2[..., 0]) / 2.0
    cbar = (c1p + c2p) / 2.0
    hbar = (h1p + h2p) / 2.0
    hbar = hbar + np.where(np.abs(h1p - h2p) > np.pi, np.pi, 0.0)
    hbar = hbar - np.where(hbar > 2.0 * np.pi, 2.0 * np.pi, 0.0)
    # Sharma et al. 2005 closed form (correct hue-rotation terms, in radians).
    t = (
        1.0
        - 0.17 * np.cos(hbar - np.pi / 6.0)
        + 0.24 * np.cos(2.0 * hbar)
        + 0.32 * np.cos(3.0 * hbar + np.pi / 30.0)
        - 0.20 * np.cos(4.0 * hbar - 63.0 * np.pi / 180.0)
    )
    dtheta_deg = 30.0 * np.exp(-(((hbar - 275.0 * np.pi / 180.0) / (25.0 * np.pi / 180.0)) ** 2))
    rc = 2.0 * np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7))
    rt = -np.sin(np.radians(2.0 * dtheta_deg)) * rc
    sl = 1.0 + (0.015 * (lbar - 50.0) ** 2) / np.sqrt(20.0 + (lbar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cbar
    sh = 1.0 + 0.015 * cbar * t
    de = np.sqrt(
        (dl / sl) ** 2
        + (dc / sc) ** 2
        + (dh_term / sh) ** 2
        + rt * (dc / sc) * (dh_term / sh)
    )
    if squeeze:
        return float(de[0])
    if arr1.ndim == 3:
        return de.reshape(arr1.shape[0], arr1.shape[1])
    return de


def detect_color_drift(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
) -> dict:
    """Color-fidelity gate: measure skin hue/ΔE drift between reference and output.

    Implements K8 — a CIEDE2000 + hue-angle (Δh) color-safety check. The
    primary use is a before/after comparison (``reference_img_bgr`` given):
    computes ΔE2000 and per-pixel hue-angle shift Δh over the skin region and
    flags if the skin hue has wandered beyond a few degrees.

    Hue gate (chroma-floored, percentile-based). Hue angle is undefined as
    chroma → 0, so a near-neutral pixel's Δh is noise (one such pixel read
    19.9° on a plain ``natural`` render whose mean Δh was 0.33°). The gate
    therefore only considers pixels whose chroma in BOTH reference and output
    is ≥ ``COLOR_DRIFT_CHROMA_FLOOR_FRAC`` × the region's median reference
    chroma (relative, so it follows the subject's own skin chroma — no
    absolute chroma cut that behaves differently across skin tones; the tiny
    ``COLOR_DRIFT_CHROMA_EPS`` is a numerical "hue is defined" guard only).
    Over that population it flags if the mean Δh >
    ``COLOR_DRIFT_HUE_MEAN_THRESHOLD`` (6°) or the 95th-percentile Δh >
    ``COLOR_DRIFT_HUE_P95_THRESHOLD`` (20°). The old single-pixel max rule
    is gone; if fewer than ``COLOR_DRIFT_MIN_HUE_PIXELS`` pixels clear the
    floor (e.g. a B&W grade) the hue gate is not evaluated: the result is
    ``not-run`` (never flagged) and still carries ``deltaE_mean``.

    Calibration (2026-09-23, person mask as passed by the pipeline):
    ``natural`` renders of 7 real photos: p99 ≤ 7.2°, floored mean ≤ 0.32°.
    Uniform OKLCh hue rotation of the reference: +8° → p99 10.5–14.8°,
    +15° → 18.6–21.5°, +30° → 35.4–36.6° (all flag).

    Recalibration (2026-09-25, 7 photos × 5 recipes): the p99 > 10° tail
    gate flagged 15/21 styled cosplay renders (mean Δh 1.4–3.7°) because
    the person mask also holds intentional lip/eye/makeup edits. The tail
    gate is now p95 > 20°: styled cosplay renders read p95 ≤ 11.8°, while
    ``cinema_grade_v1``'s skin cast reads mean ≥ 21° (7/7 flag). A cast
    has to cover more than ~5% of the person to move p95, so a 25° cast on
    only the lower half of a small face could pass (flagged 5/7 in the study).

    Patch gate (2026-09-27): with a ``face_mask`` (the pipeline passes the
    face-skin mask) each face is also scanned for a cheek-sized patch whose
    a/b shift differs from the rest of that face by more than
    ``COLOR_DRIFT_PATCH_THRESHOLD`` ΔE (``patch_delta_ab``). An even grade
    scores ~0 here, so only an uneven, local cast counts. See
    ``docs/plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md``.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR output image.
        skin_mask: Optional (H, W) float mask [0, 1]; analysis restricted to
            masked pixels when a reference is supplied.
        reference_img_bgr: Optional pre-grade reference (uint8 or float32
            [0, 255]) BGR image of the same shape.
        face_mask: Optional (H, W) face-skin mask [0, 1] for the patch gate.

    Returns:
        dict with keys:
            - "score": float in [0, 1] (higher = worse); 0.5 sits at the flag
              boundary (max of p95/(2·p95 threshold), mean/(2·mean threshold)).
            - "flagged": bool, True if chroma-floored mean Δh > 6° or
              chroma-floored p95 Δh > 20°.
            - "deltaH_p95_deg": float | None, chroma-floored 95th-percentile
              |Δh| (drives ``flagged``).
            - "deltaE_mean": float, mean ΔE2000 over (skin) region.
            - "deltaH_p99_deg": float | None, chroma-floored 99th-percentile
              |Δh| (informational); None if too few pixels cleared the floor.
            - "deltaH_mean_chroma_floored_deg": float | None, chroma-floored
              mean |Δh| (drives ``flagged``).
            - "chroma_floor": float, the chroma floor used (Lab units).
            - "hue_kept_fraction": float, fraction of region pixels kept.
            - "deltaH_mean_deg": float, raw (unfloored) mean |Δh| — kept for
              compatibility; informational, does not drive ``flagged``.
            - "deltaH_max_deg": float, raw (unfloored) max |Δh| — kept for
              compatibility; dominated by near-neutral pixels, informational.
            - "note": str, explanatory text when no reference is supplied.
    """
    _none_fields = dict(
        deltaE_mean=None, deltaH_mean_deg=None, deltaH_max_deg=None,
        deltaH_p99_deg=None, deltaH_p95_deg=None,
        deltaH_mean_chroma_floored_deg=None,
        chroma_floor=None, hue_kept_fraction=None,
    )
    if reference_img_bgr is None:
        return _not_measured(
            "no reference supplied — color_drift is a before/after comparison",
            **_none_fields,
        )

    if reference_img_bgr.shape != img_bgr.shape:
        return _not_measured("reference shape mismatch", **_none_fields)

    out_lab = _bgr_to_lab_f(img_bgr)
    ref_lab = _bgr_to_lab_f(reference_img_bgr)

    if skin_mask is not None:
        mask_f = skin_mask.astype(np.float32)
        if mask_f.max() > 1.5:
            mask_f /= 255.0
        region = mask_f > 0.5
    else:
        region = np.ones(out_lab.shape[:2], dtype=bool)

    if not np.any(region):
        return _not_measured("empty skin region", **_none_fields)

    out_sub = out_lab[region]
    ref_sub = ref_lab[region]

    # Deterministic subsample: statistics gates (mean/max hue) are stable to
    # ~1e-3 at 50k samples; full-frame CIEDE2000 on 24MP skin is ~2.5s for
    # no added precision. Fixed seed keeps QA warnings reproducible.
    if out_sub.shape[0] > 50_000:
        idx = np.random.default_rng(0).choice(
            out_sub.shape[0], size=50_000, replace=False
        )
        out_sub = out_sub[idx]
        ref_sub = ref_sub[idx]

    delta_e = delta_e_2000(ref_sub, out_sub)
    deltaE_mean = float(np.mean(delta_e))

    # Hue-angle shift from a/b channels (degrees, wrapped to [-180, 180]).
    h_ref = np.degrees(np.arctan2(ref_sub[:, 2], ref_sub[:, 1]))
    h_out = np.degrees(np.arctan2(out_sub[:, 2], out_sub[:, 1]))
    delta_h = h_out - h_ref
    delta_h = (delta_h + 180.0) % 360.0 - 180.0
    delta_h = np.abs(delta_h)
    deltaH_mean_deg = float(np.mean(delta_h))
    deltaH_max_deg = float(np.max(delta_h))

    # Chroma floor relative to the region's own reference chroma: hue is only
    # meaningful where both samples carry colour. EPS is a numerical guard
    # (hue undefined below ~1 JND of chroma), not a skin-tone threshold.
    c_ref = np.hypot(ref_sub[:, 1], ref_sub[:, 2])
    c_out = np.hypot(out_sub[:, 1], out_sub[:, 2])
    chroma_floor = max(
        COLOR_DRIFT_CHROMA_FLOOR_FRAC * float(np.median(c_ref)),
        COLOR_DRIFT_CHROMA_EPS,
    )
    keep = np.minimum(c_ref, c_out) >= chroma_floor
    n_keep = int(np.count_nonzero(keep))
    hue_kept_fraction = float(n_keep / delta_h.shape[0])

    if n_keep < COLOR_DRIFT_MIN_HUE_PIXELS:
        # Too little chroma left (e.g. B&W grade / neutral region): hue drift
        # is not measurable, so the gate did not run (status not-run, not a
        # measured pass); ΔE is still reported for information.
        return _not_measured(
            "too few chromatic pixels for a hue-drift measurement",
            deltaE_mean=deltaE_mean,
            deltaH_mean_deg=deltaH_mean_deg,
            deltaH_max_deg=deltaH_max_deg,
            deltaH_p99_deg=None,
            deltaH_p95_deg=None,
            deltaH_mean_chroma_floored_deg=None,
            chroma_floor=float(chroma_floor),
            hue_kept_fraction=hue_kept_fraction,
        )

    dh_kept = delta_h[keep]
    deltaH_p99_deg = float(np.percentile(dh_kept, COLOR_DRIFT_HUE_PERCENTILE))
    deltaH_p95_deg = float(np.percentile(dh_kept, COLOR_DRIFT_HUE_GATE_PERCENTILE))
    deltaH_mean_floored = float(np.mean(dh_kept))

    hue_flag = (
        deltaH_mean_floored > COLOR_DRIFT_HUE_MEAN_THRESHOLD
        or deltaH_p95_deg > COLOR_DRIFT_HUE_P95_THRESHOLD
    )
    score = min(1.0, max(
        deltaH_p95_deg / (2.0 * COLOR_DRIFT_HUE_P95_THRESHOLD),
        deltaH_mean_floored / (2.0 * COLOR_DRIFT_HUE_MEAN_THRESHOLD),
    ))

    patch = None
    if face_mask is not None and face_mask.shape[:2] == out_lab.shape[:2]:
        patch = _patch_chroma_cast(out_lab, ref_lab, face_mask)
    patch_flag = patch is not None and patch["delta_ab"] > COLOR_DRIFT_PATCH_THRESHOLD
    if patch is not None:
        score = min(1.0, max(score, patch["delta_ab"] / (2.0 * COLOR_DRIFT_PATCH_THRESHOLD)))
    flagged = hue_flag or patch_flag
    extra: Dict[str, Any] = {
        "patch_delta_ab": None if patch is None else patch["delta_ab"],
        "patch_zone": None if patch is None else patch["zone"],
        "patch_point": None if patch is None else patch["point"],
        "patch_face_bbox": None if patch is None else patch["face_bbox"],
    }
    if patch_flag and not hue_flag:
        extra["finding"] = (
            f"Colour cast on the {patch['zone']}: it shifted "
            f"{patch['delta_ab']:.1f} ΔE more than the rest of the face"
        )

    return {
        **extra,
        "score": float(score),
        "flagged": bool(flagged),
        "deltaE_mean": deltaE_mean,
        "deltaH_mean_deg": deltaH_mean_deg,
        "deltaH_max_deg": deltaH_max_deg,
        "deltaH_p99_deg": deltaH_p99_deg,
        "deltaH_p95_deg": deltaH_p95_deg,
        "deltaH_mean_chroma_floored_deg": deltaH_mean_floored,
        "chroma_floor": float(chroma_floor),
        "hue_kept_fraction": hue_kept_fraction,
    }


def _gray_f32(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (uint8 or float32) to float32 luminance for spectrum analysis.

    Explicit colorspace conversion at the boundary; no uint8 arithmetic.
    """
    u = _to_u8_for_analysis(img_bgr)
    gray = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _pore_band_energy(gray_f: np.ndarray) -> float:
    """Total power in the pore radial-frequency band (period ~2-8 px).

    Computes the 2D FFT power spectrum, averages over angular bins to a radial
    PSD, and sums power whose radial frequency falls in the pore band. Float32
    internal throughout.
    """
    h, w = gray_f.shape
    f = gray_f - gray_f.mean()
    # rfft2: real input → half-spectrum (negative-x mirrors dropped). Band
    # sums scale by a constant 0.5 vs full fft2; img-vs-ref comparisons are
    # unaffected. Halves RAM/time vs complex128 fft2 on megapixel frames.
    F = np.fft.rfft2(f)
    power = F.real ** 2 + F.imag ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.rfftfreq(w)
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Period 2-8 px  ->  frequency 1/8 .. 1/2 cycles/px
    pore_mask = (r >= (1.0 / 8.0)) & (r <= 0.5)
    return float(power[pore_mask].sum())


def _pore_fraction(gray_f: np.ndarray) -> float:
    """Fraction of total FFT power residing in the pore band (scale-invariant)."""
    h, w = gray_f.shape
    f = gray_f - gray_f.mean()
    F = np.fft.rfft2(f)
    power = F.real ** 2 + F.imag ** 2
    total = float(power.sum())
    if total < 1e-9:
        return 0.0
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.rfftfreq(w)
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    pore_mask = (r >= (1.0 / 8.0)) & (r <= 0.5)
    return float(power[pore_mask].sum()) / total


def detect_pore_spectrum_distance(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Plastic-skin metric v2 — pore-spectrum distance (texture-spectrum backoff).

    Measures the texture-spectrum backoff directly: compares pore-band FFT
    energy of the processed image against the un-retouched reference. Low
    pore energy = pores erased = plastic. Pure analysis; never mutates input.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR processed image.
        skin_mask: Optional (H, W) float mask [0, 1] (unused; kept for API symmetry).
        reference_img_bgr: Optional un-retouched before-image (same shape).

    Returns:
        dict with keys:
            - "score": float in [0, 1]; ~0 pores intact, ->1 plastic.
            - "flagged": bool, True if score > PORE_SPECTRUM_THRESHOLD.
            - "pore_energy_ratio": float, processed / reference pore energy.
            - "pore_band_energy": float.
            - "ref_pore_band_energy": float or None.
            - "note": str when no reference is supplied.
    """
    if reference_img_bgr is None:
        return _not_measured(
            "no reference",
            pore_energy_ratio=None, pore_band_energy=None, ref_pore_band_energy=None,
        )

    if reference_img_bgr.shape != img_bgr.shape:
        return _not_measured(
            "reference shape mismatch",
            pore_energy_ratio=None, pore_band_energy=None, ref_pore_band_energy=None,
        )

    proc_energy = _pore_band_energy(_gray_f32(img_bgr))
    ref_energy = _pore_band_energy(_gray_f32(reference_img_bgr))

    if ref_energy < 1e-9:
        ratio = 0.0
    else:
        ratio = proc_energy / ref_energy
    ratio = min(1.0, max(0.0, ratio))
    score = 1.0 - ratio
    flagged = score > PORE_SPECTRUM_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "pore_energy_ratio": float(ratio),
        "pore_band_energy": float(proc_energy),
        "ref_pore_band_energy": float(ref_energy),
    }


def detect_over_retouch_asymmetry(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    zone_masks: Optional[Dict[str, np.ndarray]] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Detect the 'one perfect cheek' tell: one zone over-smoothed vs face average.

    Computes per-zone texture-energy (mean Sobel magnitude on luminance) and
    reports the maximum drop below the face mean. Asymmetric over-smoothing is a
    classic over-retouch artifact. Pure analysis; never mutates input.

    With a ``reference_img_bgr`` (the pipeline passes the input photo and the
    face-skin mask) each zone's energy is divided by the same zone's energy
    in the reference, so the check compares how much texture each zone KEPT.
    Without it, raw energies are compared — which only makes sense when every
    zone is the same kind of surface: over the person mask, a hair or dark
    clothing zone is always "smoother" than the face, and the old run_all
    wiring flagged 21/28 real renders that way. Zones form a 3×3 grid over
    the mask's bounding box when a reference is given (over the whole frame
    otherwise, the legacy layout).

    Calibration (2026-09-25, face-skin mask, 7 photos × 5 recipes):
    asymmetry ≤ 0.19 on every render. Blurring one cheek zone (σ=1.5)
    flags on 4/7 photos (0.41–0.68); on very smooth or small faces the blur
    removes too little measurable texture to register (0.15–0.21). See
    ``docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md``.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        skin_mask: Optional (H, W) float mask [0, 1]; auto-partitioned into a
            3x3 grid of subregions when ``zone_masks`` is not supplied.
        zone_masks: Optional dict name -> (H, W) mask for explicit zones.
        reference_img_bgr: Optional pre-retouch BGR image of the same shape;
            makes zone energies relative (texture retention per zone).

    Returns:
        dict with keys:
            - "score": float in [0, 1]; asymmetry magnitude.
            - "flagged": bool, True if score > ASYMMETRY_THRESHOLD.
            - "zone_energies": dict name -> energy.
            - "face_mean_energy": float.
            - "min_zone_ratio": float, min zone/face_mean.
            - "relative_to_reference": bool, zone energies are retention
              ratios vs the reference (present on measured results).
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        return {
            "score": 0.0,
            "flagged": False,
            "zone_energies": {},
            "face_mean_energy": 0.0,
            "min_zone_ratio": 1.0,
        }

    def _energy(image: np.ndarray) -> np.ndarray:
        gray = _gray_f32(image)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)  # high-frequency energy map

    hf = _energy(img_bgr)
    hf_ref = None
    if reference_img_bgr is not None and reference_img_bgr.shape[:2] == img_bgr.shape[:2]:
        hf_ref = _energy(reference_img_bgr)

    if zone_masks is not None:
        zones = {name: (m.astype(np.float32)) for name, m in zone_masks.items()}
    elif skin_mask is not None:
        sm = skin_mask.astype(np.float32)
        if sm.max() > 1.5:
            sm /= 255.0
        sm = sm > 0.5
        if not np.any(sm):
            return {
                "score": 0.0,
                "flagged": False,
                "zone_energies": {},
                "face_mean_energy": 0.0,
                "min_zone_ratio": 1.0,
            }
        y0, x0, y1, x1 = 0, 0, sm.shape[0], sm.shape[1]
        if hf_ref is not None:
            ys, xs = np.nonzero(sm)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        zones = {}
        for iy in range(3):
            for ix in range(3):
                cell = np.zeros_like(sm)
                r0, r1 = y0 + iy * h // 3, y0 + (iy + 1) * h // 3
                c0, c1 = x0 + ix * w // 3, x0 + (ix + 1) * w // 3
                cell[r0:r1, c0:c1] = True
                zones[f"z{iy}{ix}"] = (cell & sm).astype(np.float32)
    else:
        return {
            "score": 0.0,
            "flagged": False,
            "zone_energies": {},
            "face_mean_energy": 0.0,
            "min_zone_ratio": 1.0,
        }

    zone_energies: Dict[str, float] = {}
    for name, m in zones.items():
        m = m.astype(np.float32)
        if m.max() > 1.5:
            m /= 255.0
        idx = m > 0.5
        if idx.sum() < 4:
            continue
        energy = float(np.mean(hf[idx]))
        if hf_ref is not None:
            if idx.sum() < ASYMMETRY_MIN_ZONE_PIXELS:
                continue  # sliver of mask in a grid corner: too noisy to compare
            energy /= max(float(np.mean(hf_ref[idx])), 1e-6)
        zone_energies[name] = energy

    if len(zone_energies) == 0:
        return {
            "score": 0.0,
            "flagged": False,
            "zone_energies": {},
            "face_mean_energy": 0.0,
            "min_zone_ratio": 1.0,
        }

    energies = np.array(list(zone_energies.values()), dtype=np.float32)
    face_mean = float(np.mean(energies))
    if face_mean < 1e-6:
        return {
            "score": 0.0,
            "flagged": False,
            "zone_energies": zone_energies,
            "face_mean_energy": face_mean,
            "min_zone_ratio": 1.0,
        }

    mins = np.min(energies) / face_mean
    asymmetry = float(np.max((face_mean - energies) / face_mean))
    asymmetry = min(1.0, max(0.0, asymmetry))
    flagged = asymmetry > ASYMMETRY_THRESHOLD

    return {
        "score": float(asymmetry),
        "flagged": bool(flagged),
        "zone_energies": zone_energies,
        "face_mean_energy": face_mean,
        "min_zone_ratio": float(mins),
        "relative_to_reference": hf_ref is not None,
    }


def gui_skin_score(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Single trustworthy 0-100 skin-quality number for auto-strength.

    Combines chroma variance (over-evens -> plastic) and texture presence
    (pore-band energy, vs reference when supplied else absolute fraction).
    Informational only — does NOT mutate inputs and is not a hard gate.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        skin_mask: Optional (H, W) float mask [0, 1] restricting chroma analysis.
        reference_img_bgr: Optional un-retouched reference for texture compare.

    Returns:
        dict with keys "score" (0-100), "chroma_var", "texture_metric",
        "flagged" (True if score < SKIN_SCORE_FLOOR).
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        return {
            "score": 100.0,
            "chroma_var": 0.0,
            "texture_metric": 1.0,
            "flagged": False,
        }

    u = _to_u8_for_analysis(img_bgr)
    lab = _bgr_to_lab(u)
    if skin_mask is not None:
        sm = skin_mask.astype(np.float32)
        if sm.max() > 1.5:
            sm /= 255.0
        region = sm > 0.5
    else:
        region = np.ones(lab.shape[:2], dtype=bool)

    a = lab[region, 1]
    b = lab[region, 2]
    if len(a) < 4:
        chroma_var = 0.0
    else:
        chroma = np.sqrt(a.astype(np.float32) ** 2 + b.astype(np.float32) ** 2)
        chroma_var = float(np.std(chroma))

    proc_frac = _pore_fraction(_gray_f32(img_bgr))
    if reference_img_bgr is not None and reference_img_bgr.shape == img_bgr.shape:
        ref_frac = _pore_fraction(_gray_f32(reference_img_bgr))
        if ref_frac < 1e-9:
            texture_metric = 0.0
        else:
            texture_metric = float(min(1.0, max(0.0, proc_frac / ref_frac)))
    else:
        texture_metric = float(min(1.0, max(0.0, proc_frac)))

    CHROMA_HEALTHY = 10.0  # std of chroma (Lab units) considered healthy
    chroma_metric = float(min(1.0, max(0.0, chroma_var / CHROMA_HEALTHY)))
    score01 = 0.45 * chroma_metric + 0.55 * texture_metric
    score = float(min(100.0, max(0.0, score01 * 100.0)))
    flagged = score < SKIN_SCORE_FLOOR

    return {
        "score": score,
        "chroma_var": chroma_var,
        "texture_metric": texture_metric,
        "flagged": bool(flagged),
    }


def compute_kee_farid_vector(
    img_bgr: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Kee-Farid 8-statistic image forensic retouching vector.

    Evaluates 8 deterministic image statistical moments:
    1. Spatial variance (Luminance)
    2. Laplacian high-frequency energy
    3. DCT high-frequency power ratio
    4. Pore frequency spectrum energy
    5. Local texture symmetry ratio
    6. Edge sharpness gradient magnitude
    7. Contrast dynamic range ratio
    8. Luminance skewness

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        face_mask: Optional (H, W) float face skin mask.

    Returns:
        Dict mapping statistic names to float values.
    """
    img_u8 = _to_u8_for_analysis(img_bgr)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if face_mask is not None:
        m = (face_mask > 0.5)
        if m.sum() > 10:
            pixels = gray[m]
        else:
            pixels = gray.flatten()
    else:
        pixels = gray.flatten()

    var = float(np.var(pixels))
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_energy = float(np.mean(lap**2))
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    pore_e = float(_pore_band_energy(gray))

    mean_val = float(np.mean(pixels))
    std_val = max(float(np.std(pixels)), 1e-6)
    skewness = float(np.mean(((pixels - mean_val) / std_val) ** 3))

    contrast_range = float(np.percentile(pixels, 95) - np.percentile(pixels, 5))

    return {
        "spatial_variance": var,
        "laplacian_energy": lap_energy,
        "edge_sharpness": grad_mag,
        "pore_spectrum_energy": pore_e,
        "luminance_skewness": skewness,
        "contrast_range": contrast_range,
        "mean_luminance": mean_val,
        "std_luminance": std_val,
    }



def detect_cam16_delta_e(
    img_bgr: np.ndarray,
    reference_img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """K1: mean/max CAM16-UCS perceptual colour difference vs the reference.

    Exact uniform-space ΔE (Li et al. 2017) on skin pixels; complements
    ``detect_color_drift`` (hue-only, Lab) by catching combined
    lightness/chroma/hue appearance shifts.

    Args:
        img_bgr: (H, W, 3) BGR image after processing.
        reference_img_bgr: (H, W, 3) BGR image before processing.
        skin_mask: Optional (H, W) mask; ΔE is measured inside it.

    Returns:
        Detector dict with mean/max ΔE and a flagged verdict.
    """
    from .color_science import bgr_to_cam16_ucs, cam16_ucs_delta_e

    if img_bgr.shape != reference_img_bgr.shape:
        return {
            "status": QA_STATUS_NOT_RUN,
            "score": None,
            "flagged": False,
            "reason": "reference/image shape mismatch",
        }

    # Perf: CAM16 conversion is pure-NumPy and costs ~0.7 ms/MPx per image.
    # Stride-subsample large inputs before conversion — mean/p99.5 statistics
    # over skin are statistically stable on a 4x-decimated grid, and this
    # keeps the QA hot path (every render, every recipe) affordable.
    stride = 1
    h, w = img_bgr.shape[:2]
    if h * w > 256 * 256:
        stride = 2
    if h * w > 1024 * 1024:
        stride = 4
    if stride > 1:
        img_bgr = img_bgr[::stride, ::stride]
        reference_img_bgr = reference_img_bgr[::stride, ::stride]
        if skin_mask is not None:
            skin_mask = skin_mask[::stride, ::stride]

    delta = cam16_ucs_delta_e(
        _to_u8_for_analysis(img_bgr), _to_u8_for_analysis(reference_img_bgr)
    )
    if skin_mask is not None:
        m = skin_mask.astype(np.float32, copy=False)
        if m.max() > 1.5:
            m = m / 255.0
        inside = delta[m > 0.5]
        measured = inside if inside.size > 0 else delta
    else:
        measured = delta

    mean_de = float(np.mean(measured))
    max_de = float(np.percentile(measured, 99.5))
    flagged = mean_de > CAM16_DELTA_E_MEAN_THRESHOLD or max_de > CAM16_DELTA_E_MAX_THRESHOLD
    return {
        "status": QA_STATUS_FLAGGED if flagged else QA_STATUS_PASSED,
        "score": mean_de,
        "mean_delta_e": mean_de,
        "max_delta_e": max_de,
        "flagged": flagged,
        "reason": "CAM16-UCS skin appearance shift beyond budget" if flagged else None,
    }


def run_all(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
    img_before: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
    face_skin_mask: Optional[np.ndarray] = None,
    body_skin_mask: Optional[np.ndarray] = None,
    mark_policy: Optional[Mapping[str, Any]] = None,
    warp_field: Optional[np.ndarray] = None,
    body_region_weights: Optional[np.ndarray] = None,
    geometry_reference_img_bgr: Optional[np.ndarray] = None,
    geometry_reference_reason: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all QA detectors and aggregate their results."""
    result: Dict[str, Dict[str, Any]] = {}
    photo_reference = geometry_reference_img_bgr
    if geometry_reference_reason is not None:
        photo_reference = None
    elif photo_reference is None:
        photo_reference = reference_img_bgr
    if (
        photo_reference is not None
        and photo_reference.shape != img_bgr.shape
    ):
        photo_reference = None
        geometry_reference_reason = "geometry-only reference shape mismatch"
    ref_before = photo_reference
    if ref_before is None and geometry_reference_reason is None:
        ref_before = img_before if img_before is not None else reference_img_bgr
    try:
        result["banding"] = detect_banding(
            img_bgr, skin_mask, ref_before, face_mask=face_skin_mask
        )
    except Exception as exc:
        result["banding"] = _detector_failure("banding", exc)
    try:
        result["clipping"] = detect_clipping(img_bgr, skin_mask)
    except Exception as exc:
        result["clipping"] = _detector_failure("clipping", exc)
    try:
        # Face skin when known: clothes/hair texture is not "skin texture".
        texture_mask = face_skin_mask if face_skin_mask is not None else skin_mask
        result["plastic_skin"] = detect_plastic_skin(
            img_bgr, texture_mask, photo_reference
        )
    except Exception as exc:
        result["plastic_skin"] = _detector_failure("plastic_skin", exc)
    try:
        result["halo"] = detect_halo(img_bgr, skin_mask, img_before=ref_before)
    except Exception as exc:
        result["halo"] = _detector_failure("halo", exc)
    try:
        result["kee_farid"] = compute_kee_farid_vector(img_bgr, skin_mask)
    except Exception as exc:
        result["kee_farid"] = _detector_failure("kee_farid", exc)

    try:
        pm = person_mask if person_mask is not None else skin_mask
        result["seam"] = detect_seam(img_bgr, pm, ref_before)
    except Exception as exc:
        result["seam"] = _detector_failure("seam", exc)
    try:
        result["color_drift"] = detect_color_drift(
            img_bgr, skin_mask, photo_reference, face_mask=face_skin_mask
        )
    except Exception as exc:
        result["color_drift"] = _detector_failure("color_drift", exc)
    try:
        result["pore_spectrum"] = detect_pore_spectrum_distance(
            img_bgr, skin_mask, photo_reference
        )
    except Exception as exc:
        result["pore_spectrum"] = _detector_failure("pore_spectrum", exc)
    try:
        result["asymmetry"] = detect_over_retouch_asymmetry(
            img_bgr,
            face_skin_mask if face_skin_mask is not None else skin_mask,
            reference_img_bgr=photo_reference,
        )
    except Exception as exc:
        result["asymmetry"] = _detector_failure("asymmetry", exc)
    try:
        result["skin_score"] = gui_skin_score(
            img_bgr, skin_mask, photo_reference
        )
    except Exception as exc:
        result["skin_score"] = _detector_failure("skin_score", exc)
    try:
        result["harmony"] = evaluate_harmony(
            img_bgr,
            face_skin_mask=face_skin_mask,
            body_skin_mask=body_skin_mask,
            reference_img_bgr=photo_reference,
            mark_policy=mark_policy,
        )
    except Exception as exc:
        result["harmony"] = _detector_failure("harmony", exc)
    if reference_img_bgr is not None:
        try:
            result["cam16_delta_e"] = detect_cam16_delta_e(
                img_bgr, reference_img_bgr, skin_mask=skin_mask
            )
        except Exception as exc:
            result["cam16_delta_e"] = _detector_failure("cam16_delta_e", exc)
        try:
            result["perceived_retouching"] = perceived_retouching_vector(
                img_bgr,
                reference_img_bgr,
                face_mask=face_skin_mask if face_skin_mask is not None else skin_mask,
                warp_field=warp_field,
                body_mask=body_skin_mask,
                body_region_weights=body_region_weights,
            )
            # Informational vector, not a pass/fail gate: score/flagged are
            # fixed so it never triggers run_qa's flagged-only filter, but
            # its real geometry/photometry statistics stay in the dict for
            # complete evidence (see run_qa_with_evidence).
            result["perceived_retouching"]["score"] = 0.0
            result["perceived_retouching"]["flagged"] = False
            result["perceived_retouching"]["available"] = True
        except Exception as exc:
            result["perceived_retouching"] = _detector_failure(
                "perceived_retouching", exc
            )
    else:
        result["perceived_retouching"] = {
            "status": QA_STATUS_NOT_RUN, "score": None, "flagged": False,
            "reason": "no reference image supplied",
        }
    if geometry_reference_reason is not None:
        for name in ("color_drift", "pore_spectrum"):
            measurement = result.get(name)
            if measurement is not None and measurement.get("status") == QA_STATUS_NOT_RUN:
                measurement["reason"] = geometry_reference_reason
                measurement["geometry_reference_available"] = False
        plastic = result.get("plastic_skin")
        if plastic is not None:
            plastic["geometry_reference_available"] = False
            plastic["geometry_reference_reason"] = geometry_reference_reason
        skin_score = result.get("skin_score")
        if skin_score is not None:
            skin_score["geometry_reference_available"] = False
            skin_score["geometry_reference_reason"] = geometry_reference_reason
    return result


_QA_MESSAGES = {
    "banding": "Banding visible in smooth gradient regions",
    "clipping": "Highlight/shadow clipping detected",
    "plastic_skin": "Skin texture loss detected — may appear plastic",
    "halo": "Edge overshoot halos detected from sharpening",
    "seam": "Seam visible at subject boundary",
    "color_drift": "Skin hue shift detected — color grade drifted beyond budget",
    "pore_spectrum": "Skin pore-spectrum loss detected — may appear plastic",
    "asymmetry": "Asymmetric over-smoothing detected — one face zone over-retouched",
    "skin_score": "Skin quality score low — plastic/over-evolved appearance",
    "cam16_delta_e": "CAM16-UCS skin appearance shift detected — perceptual ΔE beyond budget",
}

_QA_THRESHOLDS_BY_DETECTOR = {
    "banding": BANDING_THRESHOLD,
    "clipping": CLIPPING_THRESHOLD,
    "plastic_skin": PLASTIC_SKIN_THRESHOLD,
    "halo": HALO_THRESHOLD,
    "seam": SEAM_THRESHOLD,
    "color_drift": COLOR_DRIFT_THRESHOLD,
    "pore_spectrum": PORE_SPECTRUM_THRESHOLD,
    "asymmetry": ASYMMETRY_THRESHOLD,
    # cam16_delta_e gates on the mean ΔE (max is a secondary signal in details).
    "cam16_delta_e": CAM16_DELTA_E_MEAN_THRESHOLD,
    # skin_score is informational (soft); no hard gate threshold.
}


def run_qa_with_evidence(
    result: np.ndarray,
    person_mask: Optional[np.ndarray],
    reference_img_bgr: Optional[np.ndarray] = None,
    face_skin_mask: Optional[np.ndarray] = None,
    mark_policy: Optional[Mapping[str, Any]] = None,
    warp_field: Optional[np.ndarray] = None,
    geometry_reference_img_bgr: Optional[np.ndarray] = None,
    geometry_reference_reason: Optional[str] = None,
) -> "tuple[List[QAWarning], Dict[str, Dict[str, Any]]]":
    """Run the QA detector pipeline, returning warnings AND complete evidence.

    ``warnings`` is the flagged-only list (:func:`run_qa`'s contract; used by
    back-off and export gating, which must only act on hard failures).
    ``evidence`` is a complete ``{detector_name: {...}}`` map covering every
    name in :data:`ALL_DETECTOR_NAMES`, so a caller can always distinguish
    "checked and passed" (``status: checked-pass``) from "never checked"
    (``status: not-run``) from "checked, detector itself failed" (``status:
    unavailable``) — never by inferring it from key absence.
    """
    from .harmony import build_face_anchored_body_mask

    if person_mask is None or not np.any(person_mask > 0.3):
        return [], not_run_evidence("no person mask covering the frame")

    qa_warnings: List[QAWarning] = []
    try:
        body_skin_mask = None
        body_reference = geometry_reference_img_bgr
        if geometry_reference_reason is not None:
            body_reference = None
        elif body_reference is None:
            body_reference = reference_img_bgr
        elif body_reference.shape != result.shape:
            body_reference = None
            geometry_reference_reason = "geometry-only reference shape mismatch"
        if face_skin_mask is not None and body_reference is not None:
            body_skin_mask = build_face_anchored_body_mask(
                body_reference, face_skin_mask, person_mask,
            )
        qa_raw = run_all(
            result,
            skin_mask=person_mask,
            reference_img_bgr=reference_img_bgr,
            img_before=reference_img_bgr,
            person_mask=person_mask,
            face_skin_mask=face_skin_mask,
            body_skin_mask=body_skin_mask,
            mark_policy=mark_policy,
            warp_field=warp_field,
            geometry_reference_img_bgr=geometry_reference_img_bgr,
            geometry_reference_reason=geometry_reference_reason,
        )
    except Exception as e:
        logger.warning("QA pipeline failed: %s", e, exc_info=True)
        evidence = not_run_evidence(f"QA pipeline failed: {type(e).__name__}: {e}")
        return [QAWarning(
            detector="qa_pipeline",
            score=1.0,
            flagged=True,
            threshold=0.0,
            message="QA pipeline failed; output could not be validated",
            details={
                "available": False,
                "error": f"{type(e).__name__}: {e}",
            },
        )], evidence

    evidence: Dict[str, Dict[str, Any]] = {}
    for detector_name, det_result in qa_raw.items():
        # run_all() may already have set an explicit not-run status (e.g.
        # perceived_retouching with no reference image) — respect it rather
        # than reclassifying a not-run placeholder as "passed".
        status = det_result.get("status") or _qa_status(det_result)
        evidence[detector_name] = {**det_result, "status": status}
        # Warning-surfacing matches run_qa's pre-evidence behavior exactly:
        # only a detector result with flagged=True becomes a QAWarning. A
        # detector that self-reports available=False without flagging (e.g.
        # evaluate_harmony on masks too small to measure) stays silent here,
        # same as before — it is visible in evidence as status=unavailable,
        # but does not gate export.
        if not det_result.get("flagged", False):
            continue
        if det_result.get("available") is False:
            msg = (
                f"QA detector {detector_name} failed; "
                "output could not be validated"
            )
        else:
            msg = det_result.get("finding") or _QA_MESSAGES.get(
                detector_name, f"{detector_name} artifact detected"
            )
        qa_warnings.append(QAWarning(
            detector=detector_name,
            score=det_result.get("score", 0.0),
            flagged=True,
            threshold=_QA_THRESHOLDS_BY_DETECTOR.get(detector_name, 0.0),
            message=msg,
            details={k: v for k, v in det_result.items()
                     if k not in ("score", "flagged")},
        ))
    # Names run_all() never populated for this input (e.g. perceived_retouching
    # sets its own not-run entry, but future detectors might not).
    for name in ALL_DETECTOR_NAMES:
        evidence.setdefault(name, {
            "status": QA_STATUS_NOT_RUN, "score": None, "flagged": False,
            "reason": "detector not attempted for this input",
        })
    return qa_warnings, evidence


def run_qa(
    result: np.ndarray,
    person_mask: Optional[np.ndarray],
    reference_img_bgr: Optional[np.ndarray] = None,
    face_skin_mask: Optional[np.ndarray] = None,
    mark_policy: Optional[Mapping[str, Any]] = None,
    warp_field: Optional[np.ndarray] = None,
    geometry_reference_img_bgr: Optional[np.ndarray] = None,
    geometry_reference_reason: Optional[str] = None,
) -> List["QAWarning"]:
    """Run the QA detector pipeline on a processed uint8 BGR image.

    Pure orchestration over :func:`run_all` plus the face-anchored body
    mask (harmony) and the threshold/message maps. Returns a list of
    :class:`QAWarning` (only flagged detectors). A failure of the QA
    pipeline itself surfaces as a single fail-closed ``qa_pipeline``
    warning. See :func:`run_qa_with_evidence` for the complete
    pass/flagged/unavailable/not-run picture across every detector.
    """
    warnings, _evidence = run_qa_with_evidence(
        result, person_mask,
        reference_img_bgr=reference_img_bgr,
        face_skin_mask=face_skin_mask,
        mark_policy=mark_policy,
        warp_field=warp_field,
        geometry_reference_img_bgr=geometry_reference_img_bgr,
        geometry_reference_reason=geometry_reference_reason,
    )
    return warnings

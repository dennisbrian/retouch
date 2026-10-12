"""FA-03: mark classification experiment with frozen candidate generation.

docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md (Astra's report,
treated as the research baseline; not re-derived here) documents the
confirmed failure: on DSCF2310, two drawn-eyeliner components score
beauty_mark=0.70 and blemish=0.70 (an exact tie), and Python dict
insertion order in `max(scores, key=scores.get)`
(retouch/freckle.py:204) resolves the tie toward beauty_mark, which
`detect_marks()` renames to `mole`. Shape (PCA major/minor ratio 6.89 and
8.14) and eye-context (~4px from the parsed eye) are computed nowhere in
the current decision path.

This script isolates classification from candidate generation, per the
report's mandate: "Candidate generation must remain identical across all
arms so classification quality is isolated from proposal recall." It does
NOT modify retouch/freckle.py, retouch/marks.py, retouch/blemish.py,
retouch/perf_optimizations.py, or any production call site -- it calls
FreckleRemover._detect_components (the private candidate-proposal step)
directly and does its own scoring downstream, entirely in this file.

Architecture (three passes, so all 5 arms compare on identical input):

  1. freeze_candidates() -- runs the real, unmodified
     FreckleRemover._detect_components on a real face, and unlike
     freckle.py's own classify_anomalies (which discards the true
     component mask after computing bbox stats), keeps the boolean
     component mask itself. This is what makes real PCA/contour shape
     features possible; bbox aspect ratio is deliberately never used as a
     substitute (the report calls this out explicitly: 1.50/1.69 bbox
     aspect looks mild, true PCA ratio 6.89/8.14 does not).

  2. compute_features() -- one pass over the frozen candidates producing
     a plain dict of features per candidate: current-rule inputs (area,
     a_norm, L_norm, chroma, L_std) plus new shape features (PCA
     major/minor ratio, contour circularity, solidity/fill, skeleton
     length-to-width) plus new semantic-context features (distance to
     eye/brow/hair masks in face-normalized units, boundary alignment of
     the component's major axis against the nearest eye-mask tangent,
     whether the component crosses a semantic boundary) plus local-ring
     appearance (skin-relative luminance/chroma from a dilated local
     annulus, ring completeness/support quality). This pass touches
     `regions` (a real FaceRegions from a live engine capture) but not
     any production classifier code.

  3. Five scoring arms, each a pure function of one candidate's feature
     dict, returning a full class-score vector (never collapsed to a
     single winner without a margin check) plus an explicit ambiguous/
     abstain outcome. Ties or near-ties never depend on dict insertion
     order -- decided deterministically by an explicit rule stated in
     each arm, with an ambiguous fallback when that rule cannot resolve
     the tie either.

Run:
    .venv/bin/python scripts/qa/fa03_mark_classification_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retouch.freckle import FreckleRemover  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "fa03_mark_classification_out")

# Classes carried through every arm. "ambiguous" is new relative to
# freckle.py's four-class taxonomy and is never silently dropped or
# folded into an existing class -- it is the explicit abstain outcome
# the report's classification contract requires.
CLASSES = ("freckle", "beauty_mark", "blemish", "noise", "eye_makeup")
ABSTAIN = "ambiguous"

# Margin below which the top two scores are treated as a tie/near-tie,
# regardless of which arm produced them. This single constant is the
# fix for freckle.py:204's dict-order-dependent max(): whenever the
# top-two gap is below this margin, the outcome is ABSTAIN, not
# whichever class happens to be enumerated first.
TIE_MARGIN = 0.05


# ----------------------------------------------------------------------
# Pass 1: frozen candidate generation (unmodified FreckleRemover call)
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    """One frozen candidate: true component mask plus current-rule stats.

    ``comp_mask`` is the actual (H, W) boolean support from
    connectedComponentsWithStats -- not reconstructed from the bbox --
    so every shape feature below reflects the real, possibly highly
    non-convex or fragmented, component footprint.
    """

    index: int
    comp_mask: np.ndarray  # (H, W) bool, full-image coordinates
    bbox: Tuple[int, int, int, int]
    area: int
    centroid: Tuple[float, float]


def freeze_candidates(img_bgr: np.ndarray, skin_mask: np.ndarray) -> List[Candidate]:
    """Run the real, unmodified candidate-proposal step and keep true masks.

    Calls FreckleRemover._detect_components directly (candidate
    generation is explicitly frozen and shared across all 5 arms per the
    report's requirement) rather than classify_anomalies, because the
    latter discards the component mask after computing scores.
    """
    fr = FreckleRemover()
    labels, stats, centroids, n_labels = fr._detect_components(img_bgr, skin_mask)
    if labels is None or n_labels <= 1:
        return []

    candidates = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < FreckleRemover._MIN_AREA:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        candidates.append(Candidate(
            index=len(candidates),
            comp_mask=(labels == i),
            bbox=(x, y, w, h),
            area=area,
            centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
        ))
    return candidates


# ----------------------------------------------------------------------
# Pass 2: feature computation (shared by every arm)
# ----------------------------------------------------------------------

def _pca_axis_ratio(comp_mask: np.ndarray) -> float:
    """Major/minor axis ratio from the TRUE component's pixel covariance
    (not axis-aligned bbox width/height, which the report explicitly
    warns understates elongation: bbox aspect 1.50/1.69 vs PCA 6.89/8.14
    on the two DSCF2310 components)."""
    ys, xs = np.where(comp_mask)
    if len(xs) < 3:
        return 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-9, None)
    major, minor = float(np.sqrt(eigvals[-1])), float(np.sqrt(eigvals[0]))
    return major / max(minor, 1e-6)


def _contour_circularity(comp_mask: np.ndarray) -> float:
    """4*pi*Area / Perimeter^2 from the actual contour (1.0 = perfect
    circle, near 0 = thin line). Matches the report's diagnostic metric."""
    mask_u8 = comp_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if perim <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def _solidity(comp_mask: np.ndarray) -> float:
    """Area / convex-hull area. A thin stroke has low solidity relative
    to a compact blob of the same area, independent of PCA ratio (a
    diagonal line and a fragmented dashed line can share a PCA ratio but
    differ in solidity)."""
    mask_u8 = comp_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 1.0
    return float(np.clip(area / hull_area, 0.0, 1.0))


def _skeleton_length_to_width(comp_mask: np.ndarray) -> float:
    """Approximate skeleton length (via thinning-free distance-transform
    ridge count) divided by mean width (area / length). Cheap proxy: for
    a thin stroke, PCA major-axis length approximates skeleton length,
    and area/major_length approximates mean width. Ratio is large for a
    long, narrow stroke and small for a compact blob."""
    ys, xs = np.where(comp_mask)
    if len(xs) < 3:
        return 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ major_vec
    length = float(proj.max() - proj.min()) if len(proj) else 1.0
    length = max(length, 1.0)
    mean_width = len(xs) / length
    return length / max(mean_width, 1e-6)


def _boundary_distance_and_alignment(
    comp_mask: np.ndarray,
    eye_mask: Optional[np.ndarray],
    face_width: float,
) -> Tuple[float, float, bool]:
    """(normalized min-distance to eye mask, tangent-alignment score in
    [0,1], crosses_boundary bool).

    Alignment: compares the component's PCA major-axis orientation
    against the local tangent of the nearest eye-mask boundary point
    (finite difference along that boundary's contour), per the report's
    ILoveEye-motivated feature (eye-contour-relative construction of
    drawn eyeliner). 1.0 = perfectly aligned (component continues the
    eye's own boundary direction, i.e. looks like a drawn extension of
    the lash line); 0.0 = perpendicular (unrelated orientation, more
    consistent with an isolated mark than a continuation stroke).
    """
    if eye_mask is None or not np.any(eye_mask > 0.3):
        return float("inf"), 0.0, False

    eye_binary = (eye_mask > 0.3).astype(np.uint8) * 255
    contours, _ = cv2.findContours(eye_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("inf"), 0.0, False
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    ys, xs = np.where(comp_mask)
    comp_pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = comp_pts.mean(axis=0)

    dists = np.linalg.norm(boundary - centroid, axis=1)
    nearest_idx = int(np.argmin(dists))
    min_dist = float(dists[nearest_idx]) / max(face_width, 1.0)

    crosses = bool(np.any(comp_mask & (eye_mask > 0.3)))

    # Local tangent via finite difference a few contour points either side.
    n = len(boundary)
    if n < 5:
        return min_dist, 0.0, crosses
    span = max(2, n // 40)
    p_before = boundary[(nearest_idx - span) % n]
    p_after = boundary[(nearest_idx + span) % n]
    tangent = p_after - p_before
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6 or len(comp_pts) < 3:
        return min_dist, 0.0, crosses
    tangent = tangent / tangent_norm

    centered = comp_pts - centroid
    cov = (centered.T @ centered) / max(len(comp_pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    major_vec = major_vec / max(np.linalg.norm(major_vec), 1e-9)

    cos_angle = abs(float(np.dot(tangent, major_vec)))
    alignment = float(np.clip(cos_angle, 0.0, 1.0))
    return min_dist, alignment, crosses


def compute_features(
    candidate: Candidate,
    lab: np.ndarray,
    skin_mask: np.ndarray,
    a_median: float,
    a_std: float,
    l_median: float,
    l_std: float,
    regions: Dict[str, Optional[np.ndarray]],
    face_width: float,
) -> Dict[str, Any]:
    """One feature dict per candidate, shared read-only input to all arms."""
    comp_mask = candidate.comp_mask
    pixels = lab[comp_mask]
    area = float(pixels.shape[0])
    l_mean = float(np.mean(pixels[:, 0]))
    a_mean = float(np.mean(pixels[:, 1]))
    b_mean = float(np.mean(pixels[:, 2]))
    l_component_std = float(np.std(pixels[:, 0]))
    a_norm = (a_mean - a_median) / (a_std + 1e-6)
    l_scale = max(l_std, l_median * 0.10, 1.0)
    l_norm = (l_mean - l_median) / l_scale
    chroma = float(np.sqrt((a_mean - 128.0) ** 2 + (b_mean - 128.0) ** 2))

    # Local reference ring: dilate the true component mask, exclude the
    # component itself and require skin>0.8 support (report Sec 2: "A
    # diagnostic local reference ring... other candidates excluded").
    comp_u8 = comp_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(comp_u8, kernel) > 0
    ring = dilated & (~comp_mask) & (skin_mask > 0.8)
    ring_pixels = int(np.count_nonzero(ring))
    if ring_pixels >= 8:
        ring_lab = lab[ring]
        ring_l_med = float(np.median(ring_lab[:, 0]))
        ring_a_med = float(np.median(ring_lab[:, 1]))
        ring_b_med = float(np.median(ring_lab[:, 2]))
        ring_dl = l_mean - ring_l_med
        ring_da = a_mean - ring_a_med
        ring_db = b_mean - ring_b_med
        ring_quality = 1.0
    else:
        ring_dl = ring_da = ring_db = 0.0
        ring_quality = 0.0  # insufficient local support -- evidence of uncertainty

    pca_ratio = _pca_axis_ratio(comp_mask)
    circularity = _contour_circularity(comp_mask)
    solidity = _solidity(comp_mask)
    skel_ratio = _skeleton_length_to_width(comp_mask)

    eye_mask = _nearest_eye_mask(candidate, regions)
    eye_dist, eye_alignment, crosses_eye = _boundary_distance_and_alignment(
        comp_mask, eye_mask, face_width,
    )
    brow_mask = _nearest_of(candidate, regions, ("left_eyebrow", "right_eyebrow"))
    brow_dist, brow_alignment, crosses_brow = _boundary_distance_and_alignment(
        comp_mask, brow_mask, face_width,
    )
    hair_mask = regions.get("hair")
    hair_dist = _min_distance_to_mask(comp_mask, hair_mask, face_width)
    crosses_hair = bool(hair_mask is not None and np.any(comp_mask & (hair_mask > 0.3)))

    return {
        "area": area,
        "a_norm": a_norm,
        "l_norm": l_norm,
        "chroma": chroma,
        "l_component_std": l_component_std,
        "pca_axis_ratio": pca_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "skeleton_length_to_width": skel_ratio,
        "eye_distance_norm": eye_dist,
        "eye_alignment": eye_alignment,
        "crosses_eye_boundary": crosses_eye,
        "brow_distance_norm": brow_dist,
        "brow_alignment": brow_alignment,
        "crosses_brow_boundary": crosses_brow,
        "hair_distance_norm": hair_dist,
        "crosses_hair_boundary": crosses_hair,
        "ring_dl": ring_dl,
        "ring_da": ring_da,
        "ring_db": ring_db,
        "ring_pixels": ring_pixels,
        "ring_quality": ring_quality,
    }


def _nearest_eye_mask(candidate: Candidate, regions: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    return _nearest_of(candidate, regions, ("left_eye", "right_eye"))


def _nearest_of(candidate: Candidate, regions: Dict[str, Optional[np.ndarray]], names: Tuple[str, ...]) -> Optional[np.ndarray]:
    best_mask, best_dist = None, float("inf")
    cx, cy = candidate.centroid
    for name in names:
        m = regions.get(name)
        if m is None or not np.any(m > 0.3):
            continue
        ys, xs = np.where(m > 0.3)
        d = float(np.min(np.hypot(xs - cx, ys - cy)))
        if d < best_dist:
            best_dist, best_mask = d, m
    return best_mask


def _min_distance_to_mask(comp_mask: np.ndarray, mask: Optional[np.ndarray], face_width: float) -> float:
    if mask is None or not np.any(mask > 0.3):
        return float("inf")
    ys, xs = np.where(mask > 0.3)
    cys, cxs = np.where(comp_mask)
    if len(cxs) == 0 or len(xs) == 0:
        return float("inf")
    cx, cy = cxs.mean(), cys.mean()
    d = float(np.min(np.hypot(xs - cx, ys - cy)))
    return d / max(face_width, 1.0)


# ----------------------------------------------------------------------
# Pass 3: five scoring arms
# ----------------------------------------------------------------------

@dataclass
class ArmResult:
    scores: Dict[str, float]
    winner: str  # one of CLASSES or ABSTAIN
    margin: float
    reason: str


def _resolve_winner(scores: Dict[str, float], reason_extra: str = "") -> ArmResult:
    """Deterministic, order-independent winner selection with an explicit
    abstain outcome on ties/near-ties. This is the fix for
    freckle.py:204's `max(scores, key=scores.get)`, which silently
    favors whichever class is enumerated first in the dict on an exact
    tie (the DSCF2310 .70/.70 case)."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked or ranked[0][1] <= 0.0:
        return ArmResult(scores, "noise", 0.0, "no positive evidence" + reason_extra)
    top_class, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    if margin < TIE_MARGIN:
        tied = [k for k, v in ranked if top_score - v < TIE_MARGIN]
        return ArmResult(
            scores, ABSTAIN, margin,
            f"tie/near-tie among {sorted(tied)} (margin={margin:.3f})" + reason_extra,
        )
    return ArmResult(scores, top_class, margin, f"{top_class}={top_score:.2f}" + reason_extra)


def arm_current_baseline(f: Dict[str, Any]) -> ArmResult:
    """Arm 1: exact port of freckle.py:156-206's additive rules, but with
    the order-independent winner/abstain resolution from _resolve_winner
    instead of freckle.py's raw max(). This isolates "does adding
    deterministic tie-breaking alone fix DSCF2310" from every other
    change -- if this arm alone resolves it, the shape/context features
    below are not required for the minimal fix."""
    area = f["area"]
    scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
    if 4 <= area <= 25:
        scores["freckle"] += 0.5
    if f["a_norm"] > 0.3:
        scores["freckle"] += 0.45
    if f["l_norm"] >= -2.0:
        scores["freckle"] += 0.05
    if 10 <= area <= 70:
        scores["beauty_mark"] += 0.2
    if f["a_norm"] < -0.3:
        scores["beauty_mark"] += 0.4
    if f["l_norm"] < -2.0:
        scores["beauty_mark"] += 0.5
    if 6 <= area <= 80:
        scores["blemish"] += 0.2
    if f["chroma"] > 25.0:
        scores["blemish"] += 0.3
    if f["l_component_std"] > 8.0:
        scores["blemish"] += 0.5
    if area < 4:
        scores["noise"] = 0.9
    return _resolve_winner(scores)


def arm_geometry_only(f: Dict[str, Any]) -> ArmResult:
    """Arm 2: shape features alone, no appearance, no semantic context.
    A thin, low-solidity, low-circularity, high-PCA-ratio, high skeleton
    ratio component scores as eye_makeup regardless of color; a compact
    high-circularity, high-solidity component scores toward
    beauty_mark/freckle by shape alone (area still separates freckle
    from beauty_mark, since geometry-only must still explain the size
    axis the report keeps from the current rules)."""
    scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
    area = f["area"]
    if area < 4:
        return _resolve_winner({**scores, "noise": 0.9})

    elongated = f["pca_axis_ratio"] >= 3.5
    thin = f["solidity"] <= 0.55 or f["circularity"] <= 0.35
    if elongated and thin:
        scores["eye_makeup"] += 0.7
    else:
        # compact: split freckle vs beauty_mark by size only, as before,
        # scaled down since geometry alone cannot see color.
        if 4 <= area <= 25:
            scores["freckle"] += 0.4
        if 10 <= area <= 70:
            scores["beauty_mark"] += 0.4
        if 6 <= area <= 80:
            scores["blemish"] += 0.2
    return _resolve_winner(scores)


def arm_semantic_context_only(f: Dict[str, Any]) -> ArmResult:
    """Arm 3: eye/brow/hair proximity and boundary alignment/crossing
    alone, no shape, no appearance. The report is explicit that
    proximity alone must never reject a candidate -- only alignment
    (does the candidate continue the eye's own boundary direction)
    combined with closeness pushes toward eye_makeup."""
    scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
    area = f["area"]
    if area < 4:
        return _resolve_winner({**scores, "noise": 0.9})

    near_eye = f["eye_distance_norm"] < 0.05
    near_brow = f["brow_distance_norm"] < 0.05
    aligned = f["eye_alignment"] > 0.7 or f["brow_alignment"] > 0.7
    if (near_eye or near_brow) and aligned:
        scores["eye_makeup"] += 0.7
    elif f["crosses_hair_boundary"]:
        scores["eye_makeup"] += 0.4  # hair/brow contamination, weaker evidence alone
    else:
        # No adjacency evidence either way -- fall back to a flat prior
        # split so this arm does not simply abstain on everything;
        # semantic-only cannot see color/shape so freckle vs beauty_mark
        # is genuinely undetermined here.
        scores["freckle"] += 0.3
        scores["beauty_mark"] += 0.3
    return _resolve_winner(scores)


def arm_geometry_plus_semantic(f: Dict[str, Any]) -> ArmResult:
    """Arm 4: shape AND context combined, appearance still excluded.
    High elongation PLUS boundary alignment is the report's explicit
    "supports makeup/hair" rule (Sec 4, strategy 1) -- neither alone is
    sufficient in this arm; both must agree."""
    scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
    area = f["area"]
    if area < 4:
        return _resolve_winner({**scores, "noise": 0.9})

    elongated = f["pca_axis_ratio"] >= 3.5
    thin = f["solidity"] <= 0.55 or f["circularity"] <= 0.35
    near_boundary = f["eye_distance_norm"] < 0.08 or f["brow_distance_norm"] < 0.08
    aligned = f["eye_alignment"] > 0.6 or f["brow_alignment"] > 0.6

    if elongated and thin and near_boundary and aligned:
        scores["eye_makeup"] += 0.9
    elif elongated and thin and near_boundary:
        # Elongated and close, but not clearly aligned -- weaker,
        # ambiguous evidence rather than a confident makeup call.
        scores["eye_makeup"] += 0.5
        scores["beauty_mark"] += 0.3
    else:
        if 4 <= area <= 25:
            scores["freckle"] += 0.4
        if 10 <= area <= 70:
            scores["beauty_mark"] += 0.4
        if 6 <= area <= 80:
            scores["blemish"] += 0.15
    return _resolve_winner(scores)


def arm_scored_classical(f: Dict[str, Any]) -> ArmResult:
    """Arm 5: combines the strongest available signals from every
    category (appearance + shape + context + local ring), each
    contributing an independent, capped vote, matching the report's
    "scored classical classifier combining the strongest available
    signals" (Sec 4, strategy 2, rule-based approximation -- a full
    logistic fit needs more labels than this pilot has, per the report's
    own caution against claiming calibrated probabilities from a
    handful of examples). Ring quality gates how much the appearance
    vote is trusted."""
    scores = {"freckle": 0.0, "beauty_mark": 0.0, "blemish": 0.0, "noise": 0.0, "eye_makeup": 0.0}
    area = f["area"]
    if area < 4:
        return _resolve_winner({**scores, "noise": 0.9})

    # Appearance votes (as before), down-weighted when the local ring
    # could not be measured (ring_quality==0 -- insufficient support).
    ring_trust = 0.5 + 0.5 * f["ring_quality"]
    if 4 <= area <= 25:
        scores["freckle"] += 0.3 * ring_trust
    if f["a_norm"] > 0.3:
        scores["freckle"] += 0.25 * ring_trust
    if 10 <= area <= 70:
        scores["beauty_mark"] += 0.15 * ring_trust
    if f["a_norm"] < -0.3:
        scores["beauty_mark"] += 0.2 * ring_trust
    if f["l_norm"] < -2.0:
        scores["beauty_mark"] += 0.25 * ring_trust
    if 6 <= area <= 80:
        scores["blemish"] += 0.15 * ring_trust
    if f["chroma"] > 25.0:
        scores["blemish"] += 0.15 * ring_trust
    if f["l_component_std"] > 8.0:
        scores["blemish"] += 0.25 * ring_trust

    # Shape votes.
    elongated = f["pca_axis_ratio"] >= 3.5
    thin = f["solidity"] <= 0.55 or f["circularity"] <= 0.35
    if elongated and thin:
        scores["eye_makeup"] += 0.35

    # Context votes -- only strengthen eye_makeup when BOTH proximity
    # AND alignment agree (proximity alone never contributes, per the
    # report's explicit constraint that adjacency alone must not reject
    # a candidate).
    near_boundary = f["eye_distance_norm"] < 0.08 or f["brow_distance_norm"] < 0.08
    aligned = f["eye_alignment"] > 0.6 or f["brow_alignment"] > 0.6
    if near_boundary and aligned:
        scores["eye_makeup"] += 0.35
    if f["crosses_hair_boundary"]:
        scores["eye_makeup"] += 0.15

    # A compact, high-circularity, non-elongated candidate gets a small
    # bonus toward identity classes even when close to the eye -- this
    # is the explicit "genuine compact marks near the eye must remain
    # eligible" requirement, implemented as a positive signal rather
    # than only the absence of a penalty.
    compact = f["pca_axis_ratio"] < 2.0 and f["circularity"] > 0.5
    if compact:
        scores["beauty_mark"] += 0.1
        scores["freckle"] += 0.1

    return _resolve_winner(scores)


ARMS = {
    "1_current_baseline": arm_current_baseline,
    "2_geometry_only": arm_geometry_only,
    "3_semantic_context_only": arm_semantic_context_only,
    "4_geometry_plus_semantic": arm_geometry_plus_semantic,
    "5_scored_classical": arm_scored_classical,
}


def classify_frozen_candidates(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    regions: Dict[str, Optional[np.ndarray]],
    face_width: float,
) -> Tuple[List[Candidate], List[Dict[str, Any]], Dict[str, List[ArmResult]]]:
    """Full pipeline: freeze candidates once, compute features once,
    score with every arm. Returns (candidates, features, {arm_name:
    [ArmResult, ...]}) all index-aligned to the candidate list."""
    candidates = freeze_candidates(img_bgr, skin_mask)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_pixels = lab[skin_mask > 0.3] if np.any(skin_mask > 0.3) else lab.reshape(-1, 3)
    a_median = float(np.median(skin_pixels[:, 1]))
    a_std = float(np.std(skin_pixels[:, 1]))
    l_median = float(np.median(skin_pixels[:, 0]))
    l_std = float(np.std(skin_pixels[:, 0]))

    features = [
        compute_features(c, lab, skin_mask, a_median, a_std, l_median, l_std, regions, face_width)
        for c in candidates
    ]

    results: Dict[str, List[ArmResult]] = {name: [] for name in ARMS}
    for feat in features:
        for name, fn in ARMS.items():
            results[name].append(fn(feat))

    return candidates, features, results


if __name__ == "__main__":
    print(__doc__)

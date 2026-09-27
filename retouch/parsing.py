"""Face region parsing — generates per-region masks from MediaPipe landmarks.

Regions produced:
    skin, forehead, left_cheek, right_cheek, nose,
    left_eye, right_eye, left_eyebrow, right_eyebrow,
    left_iris, right_iris, lips, mouth_interior (for teeth),
    left_under_eye, right_under_eye, face_oval.

All masks are float32 (H, W) in [0, 1] with soft feathered edges.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import logging
import numpy as np
import os
import threading
import onnxruntime as ort
from .model_fetch import model_status
from .utils import create_polygon_mask, feather_mask, get_points, normalize_mask

logger = logging.getLogger(__name__)

# One-shot flag so the guided-filter fallback notice doesn't repeat per mask/face.
_GUIDED_FALLBACK_WARNED = False

# BiSeNet contract: 19-class CHW logits at fixed 512x512 spatial resolution.
_BISENET_CLASSES = 19
_BISENET_SPATIAL = (512, 512)

# FA-01 diagnostic default (docs/plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md
# §2.1). Checked against 5 real BiSeNet parses (DSCF2306/2308/2310/2362/2365):
# ring_px=2 already drops right_eyebrow on one of those five (DSCF2362) when
# the region erodes to nothing (expected per _bisenet_boundary_confidence_by_region's
# docstring, not a bug); ring_px=3 drops it on more faces. Kept at 2 as the
# smaller, less lossy choice. Keep in sync with any call site that overrides
# ring_px.
_BOUNDARY_RING_PX = 2

# MediaPipe multiclass selfie segmenter (``selfie_multiclass_256x256``,
# Apache-2.0 per its model card). Confidence-mask channel order from the
# model card appendix. Used only on the landmark-fallback path, i.e. when
# BiSeNet is not installed, to give hair/neck/bangs a real per-pixel signal.
_MC_HAIR, _MC_BODY_SKIN, _MC_FACE_SKIN, _MC_CLOTHES, _MC_OTHERS = 1, 2, 3, 4, 5
_MC_NUM_CLASSES = 6
# If removing segmenter hair/clothes/accessories from the landmark skin would
# drop more than this share of it, the segmenter is disagreeing with the
# landmarks (face paint, heavy makeup, a mask) rather than finding bangs; keep
# the landmark skin instead of gutting it.
_MC_MAX_SKIN_LOSS = 0.4
_MC_DISABLE_ENV = "RETOUCH_CLASS_SEGMENTER"
# Wig growth (see _grow_wig_hair). Lab histogram bins: 8 lightness x 16 x 16.
_WIG_L_BIN, _WIG_AB_BIN = 32, 16
# Need at least this many confident hair pixels to learn a wig colour from.
_WIG_MIN_SEED_PX = 100
# A colour bin counts as wig colour when it holds this share of the seed
# pixels and at least this multiple of its share among costume pixels.
_WIG_MIN_SEED_SHARE = 0.004
_WIG_SEED_OVER_COSTUME = 2.0
# Lighting spreads one wig colour across lightness; a wig chroma bin may
# reach this many lightness bins either way while the costume barely has it.
_WIG_L_SPREAD, _WIG_SPREAD_MAX_COSTUME = 2, 0.002
# Accessory pixels within this many face widths of the face count as
# possibly wig, not costume, when learning the costume colours.
_WIG_HEAD_RADIUS_FW = 1.0
# Growth larger than this multiple of the face oval means the colour model
# matched something that is not a wig (a same-coloured wall or outfit).
_WIG_MAX_FACE_MULTIPLE = 5.0
# Working size for the whole-frame hair pass (parse_hair_full_image fallback).
_FULL_HAIR_WORK_DIM = 2048


# Segmenter confidences below/above these count as absent/present (a few
# percent of every class leaks everywhere; true regions sit near 0.9).
_MC_RAMP_LO, _MC_RAMP_HI = 0.2, 0.6


def _confidence_ramp(p: np.ndarray) -> np.ndarray:
    """Smoothstep a soft class confidence to a 0-1 presence map."""
    t = np.clip((p.astype(np.float32) - _MC_RAMP_LO) / (_MC_RAMP_HI - _MC_RAMP_LO), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _smooth_hist3(h: np.ndarray) -> np.ndarray:
    """[1, 2, 1] / 4 smoothing along each axis of a 3-D histogram (wrapping)."""
    for ax in range(3):
        h = 0.5 * h + 0.25 * np.roll(h, 1, ax) + 0.25 * np.roll(h, -1, ax)
    return h


def _grow_wig_hair(
    img_bgr: np.ndarray,
    class_probs: np.ndarray,
    face_oval: np.ndarray,
    face_width: int,
) -> np.ndarray:
    """Hair confidence with wig pixels the segmenter mislabelled added back.

    The multiclass selfie segmenter was trained on natural hair. On pale,
    long or brightly coloured cosplay wigs it calls most of the wig an
    accessory ("others") or clothing and keeps the hair class for a patch of
    fringe, so hair ops and the hair exclusions miss the rest of the wig.

    The pixels it did call hair tell us what this wig looks like. This learns
    their colours (a Lab histogram) and compares them with the costume's
    colours (clothes pixels, plus accessory pixels away from the head). An
    accessory or clothes pixel joins the hair when its colour is common in
    the hair and rare in the costume, and when it is connected to the
    confident hair. So the white wig grows back while the black hat, the red
    collar and a white prop held away from the head stay out. Everything is
    relative to the photo's own hair and costume colours, so it does not
    depend on skin tone or exposure. When the hair and costume share a
    colour (dark hair, dark outfit), nothing grows and the segmenter's hair
    is returned unchanged.

    Args:
        img_bgr: (H, W, 3) BGR image, uint8 or float in [0, 1] / [0, 255].
        class_probs: (H, W, 6) multiclass confidences.
        face_oval: (H, W) bool face-oval mask.
        face_width: face width in pixels.

    Returns:
        (H, W) float32 hair confidence in [0, 1], never below the segmenter's.
    """
    hair_p = class_probs[:, :, _MC_HAIR]
    seeds = hair_p > 0.5
    n_seed = int(seeds.sum())
    if n_seed < _WIG_MIN_SEED_PX:
        return hair_p
    img_u8 = img_bgr
    if img_u8.dtype != np.uint8:
        scale = 255.0 if float(np.max(img_u8)) <= 1.5 else 1.0
        img_u8 = np.clip(img_u8.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(np.ascontiguousarray(img_u8), cv2.COLOR_BGR2LAB)
    n_l, n_ab = 256 // _WIG_L_BIN, 256 // _WIG_AB_BIN
    idx = (
        (lab[:, :, 0] // _WIG_L_BIN).astype(np.int32) * (n_ab * n_ab)
        + (lab[:, :, 1] // _WIG_AB_BIN).astype(np.int32) * n_ab
        + (lab[:, :, 2] // _WIG_AB_BIN).astype(np.int32)
    )
    n_bins = n_l * n_ab * n_ab

    label = class_probs.argmax(axis=2)
    candidate = np.isin(label, (_MC_HAIR, _MC_CLOTHES, _MC_OTHERS)) & ~seeds
    r = max(int(_WIG_HEAD_RADIUS_FW * face_width), 3)
    # Distance transform, not a dilation: a face-width elliptical kernel on
    # a full frame costs seconds.
    near_head = cv2.distanceTransform(
        (~face_oval).astype(np.uint8), cv2.DIST_L2, 5
    ) <= r
    costume = ((label == _MC_CLOTHES) | ((label == _MC_OTHERS) & ~near_head)) & ~seeds

    def _hist(sel: np.ndarray) -> np.ndarray:
        n = max(int(sel.sum()), 1)
        h = np.bincount(idx[sel], minlength=n_bins).reshape(n_l, n_ab, n_ab)
        return _smooth_hist3(h.astype(np.float64)) / n

    h_seed, h_costume = _hist(seeds), _hist(costume)
    wig = (h_seed >= _WIG_MIN_SEED_SHARE) & (h_seed >= _WIG_SEED_OVER_COSTUME * h_costume)
    spread = wig.copy()
    for d in range(1, _WIG_L_SPREAD + 1):
        spread[d:] |= wig[:-d]
        spread[:-d] |= wig[d:]
    wig |= spread & (h_costume < _WIG_SPREAD_MAX_COSTUME)

    match = candidate & wig.reshape(-1)[idx]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    match = cv2.morphologyEx(match.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    _, lbl = cv2.connectedComponents((match | seeds).astype(np.uint8))
    keep = np.unique(lbl[seeds])
    grown = np.isin(lbl, keep[keep > 0]) & match
    n_grown = int(grown.sum())
    if n_grown == 0:
        return hair_p
    if n_grown > _WIG_MAX_FACE_MULTIPLE * max(int(face_oval.sum()), 1):
        logger.info(
            "Wig growth would add %d px (%.1fx the face); keeping segmenter hair",
            n_grown, n_grown / max(int(face_oval.sum()), 1),
        )
        return hair_p
    not_hair = (
        class_probs[:, :, 0] + class_probs[:, :, _MC_FACE_SKIN] + class_probs[:, :, _MC_BODY_SKIN]
    )
    grown_p = grown.astype(np.float32) * np.clip(1.0 - not_hair, 0.0, 1.0)
    logger.debug("Wig growth: %d seed px, %d px added", n_seed, n_grown)
    return np.maximum(hair_p, grown_p).astype(np.float32)


def _sanitize_bisenet_logits(logits: np.ndarray, source: str) -> np.ndarray:
    """Validate BiSeNet output layout and scrub NaNs.

    A model swap/re-export returning NHWC layout or a different class
    count would otherwise argmax/softmax over the wrong axis and silently
    corrupt every mask. Raises ValueError so callers take their existing
    fallback paths instead. NaNs are replaced with -1e4 (argmax then picks
    a valid class; softmax saturates safely).
    """
    if logits.ndim == 4:
        ok = logits.shape[1] >= _BISENET_CLASSES and logits.shape[2:] == _BISENET_SPATIAL
    else:
        ok = logits.ndim == 3 and logits.shape[0] >= _BISENET_CLASSES and logits.shape[1:] == _BISENET_SPATIAL
    if not ok:
        raise ValueError(
            f"{source}: unexpected BiSeNet logits shape {logits.shape}; "
            f"expected CHW (>= {_BISENET_CLASSES}, {_BISENET_SPATIAL[0]}, {_BISENET_SPATIAL[1]})"
        )
    return np.nan_to_num(logits.astype(np.float32, copy=False), nan=-1e4)


# CelebAMask-HQ label -> region name, subject-anatomical labels swapped to
# camera-viewer convention (see the eye-handedness note in
# _masks_from_label_map). Observation-only companion to that function; keep
# the label numbers in sync if that mapping ever changes.
_CONFIDENCE_LABEL_TO_REGION = {
    1: "skin", 2: "right_eyebrow", 3: "left_eyebrow",
    4: "right_eye", 5: "left_eye",
    11: "mouth_interior", 12: "lips", 13: "lips",
    14: "neck", 17: "hair", 16: "cloth",
}


def _bisenet_confidence_by_region(
    logits: np.ndarray, pred_crop: np.ndarray,
) -> "dict[str, float]":
    """Summarize BiSeNet's per-pixel confidence into one scalar per region.

    Observational metadata only (see FaceRegions.parse_confidence) — not a
    calibrated error bound, and no op may read this to change edit strength.

    Uses the softmax top1-minus-top2 margin (in probability units) rather
    than max(softmax): margin answers "how much could this pixel's label
    have been the runner-up class", which is the boundary-quality question
    this field exists to surface. Computed once at the native 512x512 crop
    resolution the model actually predicted at, before any resize/paste/
    feather blurs the region boundaries.
    """
    # Guard against a re-exported model with extra channels: softmax must
    # normalize only over the 19 classes this mapping understands, or the
    # margin silently shifts as unmapped channels are folded into the sum.
    logits = logits[:_BISENET_CLASSES]
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    top2 = np.partition(softmax, -2, axis=0)[-2:]
    margin = top2[1] - top2[0]  # top1 - top2, both >= 0

    # Some regions are the union of multiple labels (lips = 12 + 13). Pool
    # the pixels before averaging so both sub-labels contribute — averaging
    # each label separately and keeping min()/max() would silently discard
    # whichever sub-label happened to lose the comparison.
    region_to_labels: "dict[str, list[int]]" = {}
    for label, region in _CONFIDENCE_LABEL_TO_REGION.items():
        region_to_labels.setdefault(region, []).append(label)

    out: "dict[str, float]" = {}
    for region, labels in region_to_labels.items():
        sel = np.isin(pred_crop, labels)
        if np.any(sel):
            out[region] = float(np.mean(margin[sel]))
    return out


def _log_confidence_evidence(
    parse_confidence: "dict[str, float]",
    parse_boundary_confidence: "dict[str, float]",
    ring_px: int,
) -> None:
    """FA-01 diagnostic observation: emit whole-region vs boundary-ring
    margins side by side so the gap §2.1 describes is actually inspectable,
    not just computed and discarded. Observational logging only — nothing
    reads these dicts back to change behavior (see FaceRegions.parse_confidence).
    """
    if not parse_confidence and not parse_boundary_confidence:
        return
    logger.debug(
        "FA-01 parse confidence (ring_px=%d): whole=%s boundary=%s",
        ring_px,
        {k: round(v, 3) for k, v in parse_confidence.items()},
        {k: round(v, 3) for k, v in parse_boundary_confidence.items()},
    )


def _bisenet_boundary_confidence_by_region(
    logits: np.ndarray, pred_crop: np.ndarray, ring_px: int = _BOUNDARY_RING_PX,
) -> "dict[str, float]":
    """Same margin signal as _bisenet_confidence_by_region, but averaged only
    over each region's boundary ring instead of its whole area.

    Diagnostic companion for FA-01 (docs/plans/RESEARCH_FACE_RETOUCH_ALGORITHMS_2026_09_05.md
    §2.1): a region-wide average can stay high while its edge is uncertain
    ("a high average can coexist with an uncertain eyelid or hairline").
    Observational metadata only — same non-consumption contract as
    parse_confidence; see FaceRegions.parse_boundary_confidence.

    Pools multi-label regions (e.g. lips = 12 | 13) into one binary mask
    *before* computing the ring, so an internal seam between two sub-labels
    (upper/lower lip contact line) is never mistaken for the region's outer
    boundary.

    A region whose mask is smaller than the ring (thin eyebrows/eyes at
    512x512 can erode to nothing) has no interior to subtract, so the ring
    equals the full region and the entry is omitted rather than silently
    degrading to the plain regional average.
    """
    logits = logits[:_BISENET_CLASSES]
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    top2 = np.partition(softmax, -2, axis=0)[-2:]
    margin = top2[1] - top2[0]

    region_to_labels: "dict[str, list[int]]" = {}
    for label, region in _CONFIDENCE_LABEL_TO_REGION.items():
        region_to_labels.setdefault(region, []).append(label)

    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), dtype=np.uint8)
    out: "dict[str, float]" = {}
    for region, labels in region_to_labels.items():
        binary = np.isin(pred_crop, labels).astype(np.uint8)
        if not np.any(binary):
            continue
        eroded = cv2.erode(binary, kernel)
        if not np.any(eroded):
            continue  # no interior left at this ring width; see docstring
        dilated = cv2.dilate(binary, kernel)
        ring = (dilated.astype(bool)) & (~eroded.astype(bool))
        out[region] = float(np.mean(margin[ring]))
    return out


def _masks_from_label_map(
    full_label_map: np.ndarray,
    img_bgr: np.ndarray,
    feather: int,
    mode: str = "gaussian",
    include_cloth: bool = True,
) -> dict:
    """Convert BiSeNet label map to feathered soft masks.

    Generates binary masks for each class, applies feathering, and performs
    skin-exclusion subtraction. The output is a dict of float32 masks in [0, 1].

    Args:
        full_label_map: (H, W) uint8 BiSeNet class label map.
        img_bgr: (H, W, 3) uint8 BGR image (used as guide for guided filter).
        feather: Base feather radius in pixels.
        mode: "gaussian" (default, Gaussian blur) or "guided" (guided filter).
        include_cloth: If True, initialize cloth mask; if False, skip it.
                      Used to preserve byte-identical behavior between parse()
                      and parse_batch() which differ in cloth initialization.

    Returns:
        Dict mapping class names to (H, W) float32 masks in [0, 1].
    """
    bisenet_masks = {}

    # BiSeNet is trained on CelebAMask-HQ, which labels eyes/eyebrows in
    # *subject-anatomical* convention: class 4 = the subject's left eye,
    # which appears on the camera's RIGHT (higher x in a frontal crop).  The
    # rest of this codebase (MediaPipe landmark clusters LEFT_EYE/RIGHT_EYE
    # below, and the regions.left_*/right_* fields consumers read) uses
    # camera-viewer convention: left = camera-left = lower x.  Swap the
    # BiSeNet eye/eyebrow classes here so the region fields describe the same
    # physical eye as the landmark-derived iris masks.  Without this swap
    # `left_sclera = clip(left_eye - left_iris)` subtracts two disjoint eyes
    # and is a total no-op (see docs/review/REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md §P0/P1).
    # The synthetic feather tests place label 4 at camera-left, which happens
    # to match the post-swap 'right_eye' key assignment, so they stay green.
    bisenet_masks['skin'] = (full_label_map == 1).astype(np.float32)
    bisenet_masks['right_eyebrow'] = (full_label_map == 2).astype(np.float32)
    bisenet_masks['left_eyebrow'] = (full_label_map == 3).astype(np.float32)
    bisenet_masks['right_eye'] = (full_label_map == 4).astype(np.float32)
    bisenet_masks['left_eye'] = (full_label_map == 5).astype(np.float32)
    bisenet_masks['mouth_interior'] = (full_label_map == 11).astype(np.float32)
    bisenet_masks['lips'] = ((full_label_map == 12) | (full_label_map == 13)).astype(np.float32)
    bisenet_masks['neck'] = (full_label_map == 14).astype(np.float32)
    bisenet_masks['hair'] = (full_label_map == 17).astype(np.float32)
    if include_cloth:
        bisenet_masks['cloth'] = (full_label_map == 16).astype(np.float32)

    bisenet_masks['face_oval'] = (
        (full_label_map == 1) | (full_label_map == 2) | (full_label_map == 3) |
        (full_label_map == 4) | (full_label_map == 5) | (full_label_map == 10) |
        (full_label_map == 11) | (full_label_map == 12) | (full_label_map == 13)
    ).astype(np.float32)

    # Apply feathering (or guided filtering)
    feather_keys = ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'face_oval', 'neck', 'hair', 'cloth']
    for k in feather_keys:
        if k not in bisenet_masks:
            continue
        r = feather // 2 if k in ['left_eye', 'right_eye', 'lips', 'left_eyebrow', 'right_eyebrow', 'cloth'] else feather

        if mode == "gaussian":
            bisenet_masks[k] = feather_mask(bisenet_masks[k], radius=r)
        elif mode == "guided":
            # Guided filter: binary mask refined by image edges
            # Ensure guide is float32 [0, 1]
            guide = img_bgr.astype(np.float32) / 255.0
            if guide.ndim == 3:
                # Convert BGR to grayscale for guided filtering
                guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            try:
                # Try cv2.ximgproc.guidedFilter
                import cv2.ximgproc as xp
                filtered = xp.guidedFilter(guide, bisenet_masks[k], radius=r, eps=1e-3)
            except (ImportError, AttributeError, cv2.error) as e:
                # Fallback to utils.guided_filter (logged once per process,
                # never a silent degrade to Gaussian)
                global _GUIDED_FALLBACK_WARNED
                if not _GUIDED_FALLBACK_WARNED:
                    logger.warning(
                        "cv2.ximgproc.guidedFilter unavailable or failed, using "
                        "utils.guided_filter for mask feathering: %s", e
                    )
                    _GUIDED_FALLBACK_WARNED = True
                from .utils import guided_filter
                filtered = guided_filter(bisenet_masks[k], radius=r, eps=1e-3, guide=guide)
            filtered = np.clip(filtered, 0.0, 1.0)
            # Residual Gaussian so edges aren't over-crisp on skin-to-skin
            # boundaries — keep/tune per Stage 2 visual QA (PLAN_C3 risks).
            residual_r = max(r // 3, 1)
            bisenet_masks[k] = feather_mask(filtered, radius=residual_r)
        else:
            raise ValueError(f"Unknown feather mode: {mode!r}")

    # Clean skin mask after feathering: subtract excluded regions
    for excl_k in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'mouth_interior']:
        if excl_k in bisenet_masks and bisenet_masks[excl_k] is not None:
            bisenet_masks['skin'] = np.clip(bisenet_masks['skin'] - bisenet_masks[excl_k], 0.0, 1.0)

    return bisenet_masks

# =====================================================================
# MediaPipe Face Mesh landmark indices (stable across versions)
# =====================================================================

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# MediaPipe face-mesh eye/eyebrow contours, in **camera-viewer** convention:
# LEFT_EYE / LEFT_IRIS describe the eye on the camera's left (lower x in a
# frontal crop).  This matches the regions.left_*/right_* field convention
# consumed downstream, but is the OPPOSITE of the subject-anatomical
# convention used by BiSeNet (CelebAMask-HQ) and by face_quality.py's
# LEFT_EYE_INDICES.  The BiSeNet label→key swap in _masks_from_label_map
# reconciles the two so regions.left_iris and regions.left_eye describe the
# same physical eye.  See docs/review/REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md §P1.
LEFT_EYE = [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153,
    145, 144, 163, 7,
]
RIGHT_EYE = [
    263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380,
    374, 373, 390, 249,
]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Nose region
NOSE = [
    168, 6, 197, 195, 5, 4, 1, 2, 98, 327,
    294, 278, 344, 440, 275, 45, 220, 115, 48, 64,
]

# Under-eye: lower lid → upper cheek
LEFT_UNDER_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    130, 25, 110, 24, 23, 22, 26, 112, 243,
]
RIGHT_UNDER_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    359, 255, 339, 254, 253, 252, 256, 341, 463,
]

# Forehead: top of face oval → above eyebrows
FOREHEAD_TOP = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
FOREHEAD_BOTTOM = [
    70, 63, 105, 66, 107,  # left eyebrow
    9,                       # mid brow bridge
    336, 296, 334, 293, 300, # right eyebrow
]

# Cheek regions (approximate polygons)
LEFT_CHEEK = [
    117, 118, 119, 120, 121, 128, 245, 193, 55,
    65, 52, 53, 46, 124, 35, 111, 117,
]
RIGHT_CHEEK = [
    346, 347, 348, 349, 350, 357, 465, 417, 285,
    295, 282, 283, 276, 353, 265, 340, 346,
]

# Nasolabial folds (wrinkle zones)
LEFT_NASOLABIAL = [206, 216, 92, 165, 167]
RIGHT_NASOLABIAL = [426, 436, 322, 391, 393]

# Crow's feet / outer eye wrinkle zones — lateral (temple-side) skin outward
# from the outer eye corner (33 / 263), NOT the eyelid/lash contour itself.
# Verified 2026-07-03 by rendering these points on two real detected faces
# (test_output/DSCF4550.jpg and grace_hopper.jpg) and visually confirming
# the cluster sits outside the LEFT_EYE/RIGHT_EYE lash contour, in skin,
# toward the temple/hairline — see fix notes for the render technique used.
LEFT_CROWS_FEET = [34, 227, 116, 137]
RIGHT_CROWS_FEET = [264, 447, 345, 366]


class FaceRegions:
    """Holds all per-region masks for one face."""

    __slots__ = [
        "face_oval", "skin", "forehead",
        "left_cheek", "right_cheek", "nose",
        "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow",
        "left_iris", "right_iris",
        "left_sclera", "right_sclera",
        "lips", "mouth_interior",
        "left_under_eye", "right_under_eye",
        "nose_bridge", "forehead_center",
        "cheek_highlights_l", "cheek_highlights_r",
        "jawline_contour", "hair", "neck",
        "nasolabial_l", "nasolabial_r",
        "crows_feet_l", "crows_feet_r",
        "cloth",
        # Cached once per face so downstream eye consumers reuse the same
        # visibility decision instead of recomputing EAR/contrast.
        "_eye_gate_cache",
        # Observational metadata, NOT a mask: per-region BiSeNet confidence
        # summary (see parse_confidence()). Exposed for inspection/QA only —
        # no op may use it to gate or scale edit strength (soft-probability
        # confidence is not a calibrated per-pixel error bound; see
        # docs/plans/RESEARCH_COLOR_SCIENCE_2026_09_04.md-style caution and
        # Guo et al. 2017 on neural-network calibration).
        "parse_confidence",
        # Same contract as parse_confidence, but averaged over each region's
        # boundary ring only (see _bisenet_boundary_confidence_by_region) —
        # surfaces edge uncertainty a whole-region average can hide.
        "parse_boundary_confidence",
    ]

    def __init__(self) -> None:
        for attr in self.__slots__:
            setattr(self, attr, None)


class FaceParser:
    """Generate precise per-region masks using BiSeNet ONNX + MediaPipe Face Mesh coordinates."""

    def __init__(self) -> None:
        self._model_path = None
        self._sess = None
        # Lazily-built MediaPipe multiclass segmenter for the landmark
        # fallback path (see _get_class_segmenter). Never pickled.
        self._class_segmenter = None
        self._class_segmenter_failed = False
        self._class_segmenter_lock = threading.Lock()
        try:
            status = model_status("resnet18_bisenet")
            if status.get("available"):
                self._model_path = status.get("path")
            else:
                logger.info(
                    "BiSeNet face parsing unavailable; using landmark-only masks (%s)",
                    status.get("reason") or status.get("description", "model not verified"),
                )
        except Exception as exc:  # optional model must never block landmark parsing
            logger.info("BiSeNet face parsing unavailable: %s", exc)

        if self._model_path and os.path.exists(self._model_path):
            try:
                # BiSeNet exposes dynamic batch/output shapes and currently
                # fails CoreML MLProgram compilation on the supported macOS
                # runtime. Keep this parser deterministic on CPU while other
                # ONNX models remain eligible for hardware acceleration.
                providers = ["CPUExecutionProvider"]

                self._sess = ort.InferenceSession(self._model_path, providers=providers)
                logger.info("ONNX Runtime initialized with active providers: %s", self._sess.get_providers())
            except Exception as e:
                logger.warning("Failed to initialize ONNX with providers: %s. Error: %s. Falling back to default.", providers if 'providers' in locals() else 'None', e)
                try:
                    self._sess = ort.InferenceSession(self._model_path)
                    logger.info("ONNX Runtime fallback initialized. Providers: %s", self._sess.get_providers())
                except Exception as fallback_err:
                    logger.error("ONNX Runtime failed completely: %s", fallback_err)

    def parse(
        self,
        landmarks: Any,
        img_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        person_mask: Optional[np.ndarray] = None,
        ied: float = 100.0,
        mask_feather_mode: str = "gaussian",
    ) -> "FaceRegions":
        """Parse a single face into region masks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList (468+ landmarks).
            img_bgr: BGR input image.
            face_bbox: RetinaFace bounding box (x, y, w, h).
            person_mask: Optional float mask from selfie segmentation.
            ied: Inter-eye distance in pixels.

        Returns:
            FaceRegions with all masks populated.
        """
        h_img, w_img = img_bgr.shape[:2]
        feather = max(int(ied * 0.08), 3)  # adaptive feather radius
        regions = FaceRegions()

        # ---- Run BiSeNet face parsing ONNX if session exists ----
        bisenet_masks = {}
        parse_confidence: Optional["dict[str, float]"] = None
        parse_boundary_confidence: Optional["dict[str, float]"] = None
        if self._sess is not None and face_bbox is not None:
            try:
                x_face, y_face, w_face, h_face = face_bbox
                # Pad crop by 30% to capture hair and neck
                pad_x = int(w_face * 0.3)
                pad_y = int(h_face * 0.3)
                cx1 = max(0, x_face - pad_x)
                cy1 = max(0, y_face - pad_y)
                cx2 = min(w_img, x_face + w_face + pad_x)
                cy2 = min(h_img, y_face + h_face + pad_y)
                cw = cx2 - cx1
                ch = cy2 - cy1

                if cw >= 4 and ch >= 4:
                    crop = img_bgr[cy1:cy2, cx1:cx2]
                    # Preprocess for BiSeNet
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    crop_f = crop_resized.astype(np.float32) / 255.0

                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    crop_norm = (crop_f - mean) / std
                    crop_input = np.transpose(crop_norm, (2, 0, 1))[np.newaxis, :, :, :]

                    # Run inference
                    outs = self._sess.run(None, {'input': crop_input})
                    logits = _sanitize_bisenet_logits(outs[0][0], "parse")  # (19, 512, 512)
                    pred_crop = np.argmax(logits, axis=0).astype(np.uint8)

                    # Resize back to crop size
                    pred_crop_resized = cv2.resize(pred_crop, (cw, ch), interpolation=cv2.INTER_NEAREST)

                    # Paste to full image size class label map
                    full_label_map = np.zeros((h_img, w_img), dtype=np.uint8)
                    full_label_map[cy1:cy2, cx1:cx2] = pred_crop_resized

                    # Generate feathered masks from label map
                    bisenet_masks = _masks_from_label_map(
                        full_label_map, img_bgr, feather, mode=mask_feather_mode, include_cloth=True
                    )
                    parse_confidence = _bisenet_confidence_by_region(logits, pred_crop)
                    parse_boundary_confidence = _bisenet_boundary_confidence_by_region(logits, pred_crop)
                    _log_confidence_evidence(parse_confidence, parse_boundary_confidence, _BOUNDARY_RING_PX)
            except Exception as e:
                input_shape = crop_input.shape if 'crop_input' in locals() else None
                logger.warning(
                    "BiSeNet face parsing failed for face_bbox=%s on image shape=%s (model=%s, input_shape=%s): %s. Falling back to landmark-only regions.",
                    face_bbox, img_bgr.shape, self._model_path, input_shape, e
                )

        # Populate regions from BiSeNet masks
        regions.skin = bisenet_masks.get('skin')
        regions.lips = bisenet_masks.get('lips')
        regions.mouth_interior = bisenet_masks.get('mouth_interior')
        regions.left_eye = bisenet_masks.get('left_eye')
        regions.right_eye = bisenet_masks.get('right_eye')
        regions.left_eyebrow = bisenet_masks.get('left_eyebrow')
        regions.right_eyebrow = bisenet_masks.get('right_eyebrow')
        regions.face_oval = bisenet_masks.get('face_oval')
        regions.neck = bisenet_masks.get('neck')
        regions.hair = bisenet_masks.get('hair')
        regions.cloth = bisenet_masks.get('cloth')
        regions.parse_confidence = parse_confidence
        regions.parse_boundary_confidence = parse_boundary_confidence

        # BiSeNet has no hand/limb class; clip its unconstrained per-pixel
        # skin/face_oval to the landmark face-oval boundary before anything
        # downstream derives sub-regions from them (see
        # _clip_bisenet_skin_to_landmark_oval docstring).
        self._clip_bisenet_skin_to_landmark_oval(regions, landmarks, w_img, h_img, feather)

        # Fallback to landmarks if BiSeNet failed or has empty skin.
        # _landmark_fallback_only sets its own parse_confidence={} sentinel
        # (no BiSeNet involved on that path).
        if regions.skin is None or regions.skin.max() < 0.01:
            regions = self._landmark_fallback(landmarks, img_bgr, person_mask, ied)
        else:
            self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)

        return regions

    def parse_hair_full_image(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run BiSeNet on the *whole* image (no face-bbox crop) to get a
        body-wide hair mask, including wig hair draping past the face crop
        onto shoulders/chest that ``parse()`` never sees.

        This is coarser than the face-crop path (512x512 covers the entire
        frame instead of just a padded face region), so it returns the
        model's *hair confidence*, not a hard argmax label.  Callers use it
        as a soft exclusion signal for body-skin masking; a hard full-frame
        label can otherwise erase large skin candidates when a colourful wig
        is classified ambiguously. Returns None if the model isn't available.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image, full frame.

        Without BiSeNet (fresh installs cannot ship it) the multiclass selfie
        segmenter stands in, with wig growth (``_grow_wig_hair``) so a long
        pale or coloured wig over the chest is still excluded from body skin.

        Returns:
            (H, W) float32 mask in [0, 1], or None if neither model is available.
        """
        if self._sess is None:
            return self._class_hair_full_image(img_bgr)

        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_f = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_f - mean) / std
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :, :, :]

        try:
            outs = self._sess.run(None, {"input": img_input})
            logits = _sanitize_bisenet_logits(outs[0][0], "parse_hair_full_image")
        except Exception:
            return None

        # Stable softmax.  The full-frame model is deliberately a weak,
        # coarse cue; retaining probability lets downstream masking reduce
        # its influence where the hair class is uncertain.
        logits -= np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        hair_512 = exp_logits[17] / np.maximum(
            np.sum(exp_logits, axis=0), np.finfo(np.float32).tiny
        )
        hair_full = cv2.resize(hair_512, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        return np.clip(hair_full, 0.0, 1.0).astype(np.float32)

    def _class_hair_full_image(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Whole-frame hair confidence from the multiclass segmenter, or None.

        The segmenter sees 256 px, so the wig growth runs on a copy no larger
        than ``_FULL_HAIR_WORK_DIM`` and is upsampled: at 26 MP it would
        otherwise take seconds for no extra detail.
        """
        h, w = img_bgr.shape[:2]
        work = img_bgr
        scale = _FULL_HAIR_WORK_DIM / max(h, w)
        if scale < 1.0:
            work = cv2.resize(
                img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        probs = self._segment_classes(work)
        if probs is None:
            return None
        face = (probs.argmax(axis=2) == _MC_FACE_SKIN).astype(np.uint8)
        n_lbl, _lbl, stats, _c = cv2.connectedComponentsWithStats(face)
        if n_lbl <= 1:
            hair = probs[:, :, _MC_HAIR].copy()
        else:
            # Scale the head neighbourhood by the largest face in frame.
            face_width = int(stats[1:, cv2.CC_STAT_WIDTH][np.argmax(stats[1:, cv2.CC_STAT_AREA])])
            hair = _grow_wig_hair(work, probs, face > 0, max(face_width, 1))
        if hair.shape != (h, w):
            hair = cv2.resize(hair, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(hair, 0.0, 1.0).astype(np.float32)

    def parse_batch(
        self,
        crop_list: List[np.ndarray],
        landmarks_compat_list: List[Any],
        face_bbox_list: List[Tuple[int, int, int, int]],
        person_masks: List[Optional[np.ndarray]],
        ieds: List[float],
        mask_feather_mode: str = "gaussian",
    ) -> List["FaceRegions"]:
        """Parse multiple face crops in a single batch ONNX call.

        Args:
            crop_list: List of BGR crop images (canvases).
            landmarks_compat_list: List of landmark compat objects.
            face_bbox_list: List of crop-relative bounding boxes.
            person_masks: List of crop-relative person masks.
            ieds: List of crop-relative IED values.

        Returns:
            List of FaceRegions.
        """
        num_faces = len(crop_list)
        results = [None] * num_faces

        # If ONNX session is not loaded, fall back to sequential single-face parse
        if self._sess is None:
            for i in range(num_faces):
                results[i] = self.parse(
                    landmarks_compat_list[i],
                    crop_list[i],
                    face_bbox_list[i],
                    person_masks[i],
                    ieds[i]
                )
            return results

        # 1. Prepare and preprocess all crops
        inputs = []
        crop_coords = []  # Store (cx1, cy1, cx2, cy2, cw, ch, h_img, w_img) for paste-back
        valid_indices = []

        for i in range(num_faces):
            img_bgr = crop_list[i]
            face_bbox = face_bbox_list[i]
            h_img, w_img = img_bgr.shape[:2]

            try:
                x_face, y_face, w_face, h_face = face_bbox
                pad_x = int(w_face * 0.3)
                pad_y = int(h_face * 0.3)
                cx1 = max(0, x_face - pad_x)
                cy1 = max(0, y_face - pad_y)
                cx2 = min(w_img, x_face + w_face + pad_x)
                cy2 = min(h_img, y_face + h_face + pad_y)
                cw = cx2 - cx1
                ch = cy2 - cy1

                if cw >= 4 and ch >= 4:
                    crop = img_bgr[cy1:cy2, cx1:cx2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    crop_f = crop_resized.astype(np.float32) / 255.0

                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    crop_norm = (crop_f - mean) / std
                    crop_input = np.transpose(crop_norm, (2, 0, 1))

                    inputs.append(crop_input)
                    crop_coords.append((cx1, cy1, cx2, cy2, cw, ch, h_img, w_img))
                    valid_indices.append(i)
                else:
                    results[i] = self._landmark_fallback(
                        landmarks_compat_list[i], img_bgr, person_masks[i], ieds[i]
                    )
            except Exception as e:
                crop_shape = getattr(img_bgr, 'shape', None)
                logger.warning(
                    "BiSeNet preprocessing failed for face index=%d (face_bbox=%s, crop_shape=%s, model=%s): %s. Using landmark fallback.",
                    i, face_bbox, crop_shape, self._model_path, e
                )
                results[i] = self._landmark_fallback(
                    landmarks_compat_list[i], img_bgr, person_masks[i], ieds[i]
                )

        # 2. Run inference in sub-batches of size 4 (preserves memory)
        try:
            _MAX_BATCH_SIZE = 4
            logits_list = []
            num_valid = len(inputs)

            for i in range(0, num_valid, _MAX_BATCH_SIZE):
                sub_inputs = inputs[i:i+_MAX_BATCH_SIZE]
                sub_batch = np.stack(sub_inputs, axis=0) # shape (B, 3, 512, 512)

                try:
                    outs = self._sess.run(None, {'input': sub_batch})
                    logits = _sanitize_bisenet_logits(outs[0], "parse_batch")  # (B, 19, 512, 512)
                    for b in range(len(sub_inputs)):
                        logits_list.append(logits[b])
                except Exception as e:
                    logger.warning(
                        "Batch BiSeNet inference failed (likely due to CoreML dynamic batch limits) for sub_batch shape=%s, batch_size=%d (model=%s): %s. "
                        "Falling back to sequential inference (batch size 1).",
                        sub_batch.shape, len(sub_inputs), self._model_path, e
                    )
                    for b_in in sub_inputs:
                        single_batch = b_in[np.newaxis, :, :, :]
                        outs_single = self._sess.run(None, {'input': single_batch})
                        logits_list.append(
                            _sanitize_bisenet_logits(outs_single[0][0], "parse_batch(single)")
                        )

            # 3. Postprocess and paste back in order (preserves input index positioning)
            for idx_valid, idx_face in enumerate(valid_indices):
                cx1, cy1, cx2, cy2, cw, ch, h_img, w_img = crop_coords[idx_valid]
                logits = _sanitize_bisenet_logits(logits_list[idx_valid], "parse_batch(post)")

                pred_crop = np.argmax(logits, axis=0).astype(np.uint8)
                parse_confidence = _bisenet_confidence_by_region(logits, pred_crop)
                parse_boundary_confidence = _bisenet_boundary_confidence_by_region(logits, pred_crop)
                _log_confidence_evidence(parse_confidence, parse_boundary_confidence, _BOUNDARY_RING_PX)
                pred_crop_resized = cv2.resize(pred_crop, (cw, ch), interpolation=cv2.INTER_NEAREST)

                full_label_map = np.zeros((h_img, w_img), dtype=np.uint8)
                full_label_map[cy1:cy2, cx1:cx2] = pred_crop_resized

                # Compute feather for this face and generate masks
                feather = max(int(ieds[idx_face] * 0.08), 3)
                bisenet_masks = _masks_from_label_map(
                    full_label_map, crop_list[idx_face], feather, mode=mask_feather_mode, include_cloth=False
                )

                # Build FaceRegions
                regions = FaceRegions()
                regions.skin = bisenet_masks.get('skin')
                regions.lips = bisenet_masks.get('lips')
                regions.mouth_interior = bisenet_masks.get('mouth_interior')
                regions.left_eye = bisenet_masks.get('left_eye')
                regions.right_eye = bisenet_masks.get('right_eye')
                regions.left_eyebrow = bisenet_masks.get('left_eyebrow')
                regions.right_eyebrow = bisenet_masks.get('right_eyebrow')
                regions.face_oval = bisenet_masks.get('face_oval')
                regions.neck = bisenet_masks.get('neck')
                regions.hair = bisenet_masks.get('hair')
                regions.cloth = bisenet_masks.get('cloth')
                regions.parse_confidence = parse_confidence
                regions.parse_boundary_confidence = parse_boundary_confidence

                # BiSeNet has no hand/limb class; clip its unconstrained
                # per-pixel skin/face_oval to the landmark face-oval
                # boundary before anything downstream derives sub-regions
                # from them (see _clip_bisenet_skin_to_landmark_oval).
                self._clip_bisenet_skin_to_landmark_oval(
                    regions, landmarks_compat_list[idx_face], w_img, h_img, feather
                )

                # Handle fallback if skin is empty. _landmark_fallback_only
                # sets its own parse_confidence={} sentinel.
                if regions.skin is None or regions.skin.max() < 0.01:
                    regions = self._landmark_fallback(
                        landmarks_compat_list[idx_face], crop_list[idx_face], person_masks[idx_face], ieds[idx_face]
                    )
                else:
                    self._add_landmark_subregions(regions, landmarks_compat_list[idx_face], h_img, w_img, ieds[idx_face], feather)

                results[idx_face] = regions
        except Exception as batch_err:
            logger.error(
                "Critical failure in batch parsing pipeline for %d faces (model=%s): %s. Falling back to sequential single-face parse.",
                num_faces, self._model_path, batch_err
            )
            for i in range(num_faces):
                try:
                    results[i] = self.parse(
                        landmarks_compat_list[i],
                        crop_list[i],
                        face_bbox_list[i],
                        person_masks[i],
                        ieds[i]
                    )
                except Exception as parse_err:
                    crop_shape = getattr(crop_list[i], 'shape', None)
                    logger.error(
                        "Sequential single-face fallback also failed for face %d/%d (face_bbox=%s, crop_shape=%s, model=%s): %s. Using landmark fallback.",
                        i, num_faces, face_bbox_list[i], crop_shape, self._model_path, parse_err
                    )
                    results[i] = self._landmark_fallback(
                        landmarks_compat_list[i], crop_list[i], person_masks[i], ieds[i]
                    )

        return results

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_class_segmenter"] = None
        state["_class_segmenter_failed"] = False
        state["_class_segmenter_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._class_segmenter_lock = threading.Lock()

    def close(self) -> None:
        """Release the multiclass segmenter, if one was built.

        Like ``FaceDetector.close`` this must run on the main thread while the
        MediaPipe dispatcher is alive; leaving tasks to the garbage collector
        can hang interpreter shutdown. Idempotent.
        """
        seg = getattr(self, "_class_segmenter", None)
        self._class_segmenter = None
        if seg is not None:
            try:
                seg.close()
            except Exception as exc:
                logger.debug("Multiclass segmenter close raised: %s", exc)

    def _get_class_segmenter(self):
        """Return the multiclass segmenter, building it on first use.

        Resolves ``selfie_multiclass`` through the verified model cache
        (downloading it on first use unless offline mode is on). Any failure
        is logged once and remembered, and the caller falls back to
        landmark-only masks. ``RETOUCH_CLASS_SEGMENTER=0`` disables it.
        """
        if self._class_segmenter is not None or self._class_segmenter_failed:
            return self._class_segmenter
        with self._class_segmenter_lock:
            if self._class_segmenter is not None or self._class_segmenter_failed:
                return self._class_segmenter
            if os.environ.get(_MC_DISABLE_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
                self._class_segmenter_failed = True
                return None
            try:
                import mediapipe as mp
                from .model_fetch import get_model_path

                path = get_model_path("selfie_multiclass")
                vision = mp.tasks.vision
                base = mp.tasks.BaseOptions
                self._class_segmenter = vision.ImageSegmenter.create_from_options(
                    vision.ImageSegmenterOptions(
                        base_options=base(model_asset_path=path, delegate=base.Delegate.CPU),
                        running_mode=vision.RunningMode.IMAGE,
                        output_category_mask=False,
                        output_confidence_masks=True,
                    )
                )
                logger.info("Face parsing: BiSeNet absent, using MediaPipe multiclass segmenter for hair/neck masks")
            except Exception as exc:
                self._class_segmenter_failed = True
                logger.info(
                    "Multiclass segmenter unavailable; using landmark-only hair/neck masks: %s", exc
                )
        return self._class_segmenter

    def _segment_classes(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Per-pixel class confidences ``(H, W, 6)`` float32, or None.

        Channels follow the model card: background, hair, body-skin,
        face-skin, clothes, others (accessories).
        """
        seg = self._get_class_segmenter()
        if seg is None or img_bgr is None or img_bgr.ndim != 3 or min(img_bgr.shape[:2]) < 8:
            return None
        try:
            import mediapipe as mp

            img_u8 = img_bgr
            if img_u8.dtype != np.uint8:
                img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
            with self._class_segmenter_lock:
                result = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            masks = result.confidence_masks or []
            if len(masks) != _MC_NUM_CLASSES:
                return None
            h, w = img_bgr.shape[:2]
            probs = np.empty((h, w, _MC_NUM_CLASSES), dtype=np.float32)
            for i, m in enumerate(masks):
                arr = np.squeeze(np.asarray(m.numpy_view(), dtype=np.float32))
                if arr.shape != (h, w):
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
                probs[:, :, i] = arr
            return np.clip(probs, 0.0, 1.0)
        except Exception as exc:
            logger.warning("Multiclass segmentation failed on shape=%s: %s", img_bgr.shape, exc)
            return None

    def _landmark_fallback(
        self,
        landmarks: Any,
        img_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        ied: float,
    ) -> "FaceRegions":
        """Landmark fallback, refined by the multiclass segmenter when available."""
        return self._landmark_fallback_only(
            landmarks, img_bgr, person_mask, ied,
            class_probs=self._segment_classes(img_bgr),
        )

    def _clip_bisenet_skin_to_landmark_oval(
        self,
        regions: "FaceRegions",
        landmarks: Any,
        w_img: int,
        h_img: int,
        feather: int,
    ) -> None:
        """Clip BiSeNet's ``skin``/``face_oval`` to the landmark face-oval.

        BiSeNet's per-pixel classifier has no hand/limb class, so any
        skin-toned pixel it associates with "face skin" by local
        texture/context — including a hand resting near the jaw — is
        labeled skin. Left unconstrained, this bleeds face-retouch
        treatment onto an occluding hand (see
        docs/plans/RESEARCH_FRONTIER_AD_2026_07_19.md). The landmark face
        oval is pure polygon geometry over the actual jawline/hairline
        contour, independent of pixel color, so intersecting against it
        excludes anything spatially outside the face regardless of skin
        tone. For an unoccluded face this is a near-no-op (BiSeNet's mask
        is already inside the landmark boundary), so this must run before
        ``_add_landmark_subregions`` derives every sub-region from
        ``regions.skin``.
        """
        landmark_oval = self._mask(landmarks, FACE_OVAL, w_img, h_img, feather)
        if regions.skin is not None:
            regions.skin = regions.skin * landmark_oval
        if regions.face_oval is not None:
            regions.face_oval = regions.face_oval * landmark_oval

    def _landmark_fallback_only(
        self,
        landmarks: Any,
        img_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        ied: float,
        class_probs: Optional[np.ndarray] = None,
    ) -> "FaceRegions":
        """Construct face region masks using landmarks when BiSeNet fails or is bypassed.

        ``class_probs`` is the optional ``(H, W, 6)`` output of
        ``_segment_classes``. With it, hair and neck come from the segmenter
        and bangs/accessories are cut out of skin. Without it, hair is a
        head-hugging band inside the person mask and neck stays empty.
        """
        h_img, w_img = img_bgr.shape[:2]
        feather = max(int(ied * 0.08), 3)
        regions = FaceRegions()
        # No BiSeNet inference on this path: empty dict (not None) means
        # "no model confidence data available", distinct from an attribute
        # that was simply never populated.
        regions.parse_confidence = {}
        regions.parse_boundary_confidence = {}

        regions.face_oval = self._mask(landmarks, FACE_OVAL, w_img, h_img, feather)
        regions.left_eye = self._mask(landmarks, LEFT_EYE, w_img, h_img, feather // 2)
        regions.right_eye = self._mask(landmarks, RIGHT_EYE, w_img, h_img, feather // 2)
        regions.left_eyebrow = self._mask(landmarks, LEFT_EYEBROW, w_img, h_img, feather // 2)
        regions.right_eyebrow = self._mask(landmarks, RIGHT_EYEBROW, w_img, h_img, feather // 2)
        regions.lips = self._mask(landmarks, LIPS_OUTER, w_img, h_img, feather // 2)
        regions.mouth_interior = self._mask(landmarks, LIPS_INNER, w_img, h_img, 0)

        # Synthesize skin
        skin = regions.face_oval.copy()
        for exclusion in [
            regions.left_eye, regions.right_eye,
            regions.left_eyebrow, regions.right_eyebrow,
            regions.lips,
        ]:
            skin = np.clip(skin - exclusion, 0, 1)

        have_classes = class_probs is not None and class_probs.shape[:2] == (h_img, w_img)
        pm = None
        if person_mask is not None:
            pm = normalize_mask(person_mask)
            from .utils import squeeze_mask
            pm = squeeze_mask(pm)
            if not have_classes:
                skin *= pm
            # With classes, _refine_fallback_with_classes applies the person
            # gate together with the segmenter's own face-skin vote.
        regions.skin = skin
        regions.neck = np.zeros((h_img, w_img), dtype=np.float32)
        if person_mask is not None:
            regions.hair = self._landmark_hair_band(regions.face_oval, pm)
        else:
            regions.hair = np.zeros((h_img, w_img), dtype=np.float32)

        if have_classes:
            self._refine_fallback_with_classes(regions, class_probs, img_bgr, feather, person_mask=pm)

        self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)
        return regions

    @staticmethod
    def _face_oval_extent(face_oval: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(face_oval > 0.5)
        if ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _landmark_hair_band(self, face_oval: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Model-free hair estimate: person pixels in a band around the head.

        The old fallback used ``person_mask - face_oval``, which is the whole
        body (clothes, hands, props) and 10-20x BiSeNet's hair area. This keeps
        only the part of the person within ~0.35 face widths of the face oval
        and above the chin, which is where hair almost always is. Long hair
        below the chin is missed; that is the safe direction to be wrong.
        """
        h, w = face_oval.shape[:2]
        ext = self._face_oval_extent(face_oval)
        if ext is None:
            return np.zeros((h, w), dtype=np.float32)
        x0, _y0, x1, y1 = ext
        r = max(int(0.35 * (x1 - x0 + 1)), 3)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        oval_bin = (face_oval > 0.5).astype(np.uint8)
        band = cv2.dilate(oval_bin, k).astype(np.float32) - oval_bin.astype(np.float32)
        band[y1:, :] = 0.0
        band = feather_mask(band, radius=max(r // 4, 3))
        return np.clip(band * person_mask, 0.0, 1.0).astype(np.float32)

    def _refine_fallback_with_classes(
        self,
        regions: "FaceRegions",
        class_probs: np.ndarray,
        img_bgr: np.ndarray,
        feather: int,
        person_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Replace landmark hair/neck with segmenter output and cut bangs out of skin.

        ``person_mask`` (already normalised) gates skin together with the
        segmenter's face-skin class, which vouches for heads the person
        segmenter loses under pale wigs against bright windows.
        """
        h, w = class_probs.shape[:2]
        body_p = class_probs[:, :, _MC_BODY_SKIN]
        ext = self._face_oval_extent(regions.face_oval)
        hair_p = class_probs[:, :, _MC_HAIR]
        if ext is not None:
            hair_p = _grow_wig_hair(
                img_bgr, class_probs, regions.face_oval > 0.5, ext[2] - ext[0] + 1
            )

        # Hair: segmenter confidence, edges snapped to the image with a
        # guided filter (the model runs at 256 px, so raw edges are soft).
        guide = img_bgr.astype(np.float32)
        if guide.max() > 1.5:
            guide = guide / 255.0
        guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY) if guide.ndim == 3 else guide
        r = max(int(feather), 2)
        try:
            import cv2.ximgproc as xp
            hair = xp.guidedFilter(guide, hair_p, radius=r, eps=1e-3)
        except (ImportError, AttributeError, cv2.error):
            from .utils import guided_filter
            hair = guided_filter(hair_p, radius=r, eps=1e-3, guide=guide)
        regions.hair = np.clip(hair, 0.0, 1.0).astype(np.float32)

        # Skin: remove what the segmenter says is hair (bangs, fringe),
        # clothes (masks, high collars) or accessories (glasses, headdress)
        # inside the landmark face oval, and gate by person / face-skin
        # presence. The segmenter's confidences are soft (a few percent of
        # every class leaks everywhere), so multiplying them in raw capped
        # confident skin at ~0.87-0.95; ramp them to hard 0/1 instead.
        skin = regions.skin
        if person_mask is not None:
            gate = _confidence_ramp(np.maximum(person_mask, class_probs[:, :, _MC_FACE_SKIN]))
        else:
            gate = np.ones((h, w), dtype=np.float32)
        hair_cut = _confidence_ramp(hair_p)
        other_cut = _confidence_ramp(
            np.clip(class_probs[:, :, _MC_CLOTHES] + class_probs[:, :, _MC_OTHERS], 0.0, 1.0)
        )
        occluder = np.maximum(hair_cut, other_cut)

        # Guard: when the segmenter calls much of the face clothes or an
        # accessory (face paint, heavy makeup, a mask) or calls the lower face
        # hair, it is disagreeing with the landmarks; keep the landmark skin.
        # Hair above the eye line is bangs, which may legitimately cover most
        # of a forehead, so it does not count towards the guard (it used to,
        # and full bangs then threw the whole refinement away).
        base = skin * gate
        eye_row = self._eye_line_row(regions, h)
        lower = np.zeros((h, 1), dtype=np.float32)
        lower[eye_row:] = 1.0
        guard_cut = np.maximum(other_cut, hair_cut * lower)
        before = float(base.sum())
        loss = float((base * guard_cut).sum()) / before if before > 0 else 0.0
        if loss <= _MC_MAX_SKIN_LOSS:
            factor = gate * (1.0 - occluder)
        else:
            logger.info(
                "Multiclass segmenter would remove %.0f%% of landmark skin below the "
                "eye line; keeping landmark skin",
                100.0 * loss,
            )
            factor = gate

        # The segmenter runs at 256 px, so its edges are blocky and loose;
        # snap them to the photo's own edges (the landmark geometry and the
        # eye/brow/lip holes stay as they are).
        try:
            import cv2.ximgproc as xp
            factor = xp.guidedFilter(guide, factor.astype(np.float32), radius=r, eps=1e-3)
        except (ImportError, AttributeError, cv2.error):
            from .utils import guided_filter
            factor = guided_filter(factor.astype(np.float32), radius=r, eps=1e-3, guide=guide)
        regions.skin = np.clip(skin * np.clip(factor, 0.0, 1.0), 0.0, 1.0).astype(np.float32)

        # Neck: body-skin in a band under the jaw, no wider than the face
        # (keeps chest, shoulders and hands out, like BiSeNet's neck label).
        if ext is not None:
            x0, y0, x1, y1 = ext
            fw, fh = x1 - x0 + 1, y1 - y0 + 1
            band = np.zeros((h, w), dtype=np.float32)
            bx0 = max(0, int(x0 + 0.05 * fw))
            bx1 = min(w, int(x1 - 0.05 * fw) + 1)
            by0 = max(0, int(y1 - 0.25 * fh))
            by1 = min(h, int(y1 + 0.5 * fh) + 1)
            band[by0:by1, bx0:bx1] = 1.0
            band = feather_mask(band, radius=max(int(0.08 * fw), 3))
            neck = body_p * band * (1.0 - np.clip(regions.face_oval, 0.0, 1.0))
            # A hand raised to the chin is body-skin too; keep only the blobs
            # that reach up to the jaw.
            n_lbl, lbl = cv2.connectedComponents((neck > 0.5).astype(np.uint8))
            if n_lbl > 1:
                jaw_rows = lbl[by0:min(h, y1 + 1)]
                keep = np.isin(lbl, np.unique(jaw_rows[jaw_rows > 0]))
                keep = feather_mask(keep.astype(np.float32), radius=max(int(0.03 * fw), 2))
                neck = neck * np.clip(keep, 0.0, 1.0)
            regions.neck = np.clip(neck, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _eye_line_row(regions: "FaceRegions", h: int) -> int:
        """Row of the eye centres (the face oval's middle if eyes are empty)."""
        rows = []
        for eye in (regions.left_eye, regions.right_eye):
            if eye is not None:
                ys = np.nonzero(eye > 0.5)[0]
                if ys.size:
                    rows.append(float(ys.mean()))
        if rows:
            return int(np.clip(round(float(np.mean(rows))), 0, h))
        ys = np.nonzero(regions.face_oval > 0.5)[0] if regions.face_oval is not None else np.array([])
        return int((ys.min() + ys.max()) // 2) if ys.size else h // 2

    def _add_landmark_subregions(
        self,
        regions: "FaceRegions",
        landmarks: Any,
        h_img: int,
        w_img: int,
        ied: float,
        feather: int,
    ) -> None:
        """Construct internal/sub-region masks using landmarks."""
        try:
            regions.left_iris = self._iris_mask(landmarks, LEFT_IRIS, w_img, h_img, ied)
            regions.right_iris = self._iris_mask(landmarks, RIGHT_IRIS, w_img, h_img, ied)
        except (IndexError, AttributeError):
            regions.left_iris = np.zeros((h_img, w_img), dtype=np.float32)
            regions.right_iris = np.zeros((h_img, w_img), dtype=np.float32)

        regions.nose = self._mask(landmarks, NOSE, w_img, h_img, feather)
        regions.left_under_eye = self._mask(landmarks, LEFT_UNDER_EYE, w_img, h_img, feather)
        regions.right_under_eye = self._mask(landmarks, RIGHT_UNDER_EYE, w_img, h_img, feather)
        regions.forehead = self._forehead_mask(landmarks, w_img, h_img, feather)
        regions.left_cheek = self._mask(landmarks, LEFT_CHEEK, w_img, h_img, feather)
        regions.right_cheek = self._mask(landmarks, RIGHT_CHEEK, w_img, h_img, feather)

        regions.nose_bridge = self._mask(landmarks, [168, 6, 197, 195], w_img, h_img, feather)
        regions.forehead_center = self._circle_mask(landmarks, 151, ied * 0.25, w_img, h_img, feather)
        regions.cheek_highlights_l = self._circle_mask(landmarks, 117, ied * 0.22, w_img, h_img, feather)
        regions.cheek_highlights_r = self._circle_mask(landmarks, 346, ied * 0.22, w_img, h_img, feather)
        regions.nasolabial_l = self._mask(landmarks, LEFT_NASOLABIAL, w_img, h_img, feather)
        regions.nasolabial_r = self._mask(landmarks, RIGHT_NASOLABIAL, w_img, h_img, feather)
        regions.crows_feet_l = self._mask(landmarks, LEFT_CROWS_FEET, w_img, h_img, feather)
        regions.crows_feet_r = self._mask(landmarks, RIGHT_CROWS_FEET, w_img, h_img, feather)
        regions.jawline_contour = np.clip(
            self._mask(landmarks, [234, 127, 93, 132, 58, 172, 136, 150], w_img, h_img, feather) +
            self._mask(landmarks, [454, 323, 361, 288, 397, 365, 379, 378], w_img, h_img, feather),
            0, 1
        )

        # For sub-skin highlights/contours, intersect them with the skin mask to be perfectly clean.
        # The forehead band sits above the skin region (hairline), so it must be clipped to the
        # broader face_oval (which includes the upper-face label) rather than skin, else it vanishes.
        _face_oval = getattr(regions, 'face_oval', None)
        for attr in ['nose_bridge', 'forehead_center', 'cheek_highlights_l', 'cheek_highlights_r', 'jawline_contour', 'left_under_eye', 'right_under_eye', 'left_cheek', 'right_cheek', 'forehead', 'nasolabial_l', 'nasolabial_r', 'crows_feet_l', 'crows_feet_r']:
            val = getattr(regions, attr)
            if val is None:
                continue
            if attr == 'forehead':
                # Forehead band sits above the skin region; the landmark polygon is
                # already a clean shape, so keep it unclipped (wrinkle_soften excludes
                # hair/eyebrows/eyes internally).
                setattr(regions, attr, val)
            else:
                setattr(regions, attr, val * regions.skin)

        # Compute sclera masks (eye regions minus iris)
        if regions.left_eye is not None and regions.left_iris is not None:
            regions.left_sclera = np.clip(regions.left_eye - regions.left_iris, 0.0, 1.0)
        else:
            regions.left_sclera = np.zeros((h_img, w_img), dtype=np.float32)

        if regions.right_eye is not None and regions.right_iris is not None:
            regions.right_sclera = np.clip(regions.right_eye - regions.right_iris, 0.0, 1.0)
        else:
            regions.right_sclera = np.zeros((h_img, w_img), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask(
        self,
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        pts = get_points(landmarks, indices, w, h)
        return create_polygon_mask(pts, (h, w), feather_radius=feather)

    def _circle_mask(
        self,
        landmarks: Any,
        index: int,
        radius_px: float,
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        """Create a circular feathered mask around a single landmark."""
        lm = landmarks.landmark[index]
        cx, cy = int(lm.x * w), int(lm.y * h)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), int(radius_px), 1.0, -1)
        return feather_mask(mask, radius=feather)

    def _iris_mask(
        self,
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
        ied: float,
    ) -> np.ndarray:
        """Circle mask for iris based on iris landmarks."""
        pts = get_points(landmarks, indices, w, h)
        center = pts.mean(axis=0).astype(int)
        radius = max(int(ied * 0.07), 4)  # iris ≈ 7% of IED
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, tuple(center), radius, 1.0, -1)
        return feather_mask(mask, radius=max(radius // 3, 2))

    def _forehead_mask(
        self,
        landmarks: Any,
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        """Forehead: area between face oval top and eyebrow line."""
        top_pts = get_points(landmarks, FOREHEAD_TOP, w, h)
        bot_pts = get_points(landmarks, FOREHEAD_BOTTOM, w, h)
        # Combine into a closed polygon: top_pts forward + bot_pts reversed
        poly = np.vstack([top_pts, bot_pts[::-1]])
        return create_polygon_mask(poly, (h, w), feather_radius=feather)

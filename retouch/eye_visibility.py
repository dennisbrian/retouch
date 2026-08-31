"""Per-eye visibility guards for eye enhancement.

Landmark-derived iris masks are intentionally stable, even when an eye is
covered by hair, a wig, a hand, or a closed lid.  That stability is useful
for face geometry, but it is unsafe for pixel edits: an enhancer can
otherwise paint a convincing iris over the occluder.

This module turns two per-eye visibility signals into a conservative gate:

1. **EAR (eyelid aspect ratio)** — primary.  Computed from the 6-point
   MediaPipe eye contour (indices in :mod:`retouch.parsing`:
   ``LEFT_EYE`` / ``RIGHT_EYE``).  EAR is resolution-invariant, always
   available on the landmark path (independent of BiSeNet), and independent
   of the iris circle.  A genuinely closed eye (lid onto lid) collapses EAR
   toward ~0.10; the measured open-eye floor on the calibration corpus is
   0.194 (see ``docs/plans/RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md``
   for the full calibration).  Threshold sits just above the measured
   occluded ceiling (0.283) so the full occluded distribution gates.

2. **Tone-adaptive iris-vs-sclera contrast** — secondary.  Computed from the
   parsed ``*_eye`` and ``*_iris`` masks against the eye's own 95th
   percentile L (never an absolute luminance threshold, per the
   tone-invariance rule).  Closed eyes collapse the sclera→iris luminance
   gap; open eyes maintain it.  This signal is only meaningful when the
   BiSeNet eye mask is non-degenerate — BiSeNet has a documented class-5
   (camera-left eye) collapse on ~65% of real portraits, and ``eye_px == 0``
   is a *model fragility*, not an occlusion signal.  Contrast therefore
   never fires on a collapsed mask; the gate fails open there.

3. **Hair-overlap** — tertiary, retained for the case where BiSeNet reports
   a healthy eye mask but a hair strand blankets the iris.  Note this
   signal is structurally dead on the landmark-fallback path (``regions.hair``
   is ``person_mask - face_oval`` there, exactly zero inside the face oval
   where eyes live) — see §B1 of the review.  Documented, not removed.

The gate is deliberately mask-only at the plumbing level: a shallow copy of
the regions object is made and only the affected side's ``*_eye`` /
``*_iris`` / ``*_sclera`` arrays are zeroed.  Non-gated eyes and the
caller's original object are untouched.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Calibrated thresholds.  See docs/plans/RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md
# for the full study (83-image DSCF corpus, 166 eyes, both BiSeNet-live and
# landmark-fallback arms). Still provisional pending hard-case sourcing
# (hair/wig/sunglasses — the corpus has none), a Fitzpatrick IV-VI stratum,
# and a second labeler pass (§6 of that doc) — but the DSCF-corpus arm's
# acceptance criterion (missed-gate <= 5%) is met and the prior shipped
# values were not: `_MIN_EAR = 0.12` missed 16/19 (84%) genuinely closed
# eyes, calibrated against a mislabeled anchor (DSCF4454-R was recorded as
# "plainly open" at EAR 0.136; at 300px it is unambiguously closed).
#
# EAR: measured occluded range is 0.101-0.283 (19/19 eyes, both arms
# identical by construction); measured visible floor is 0.194 (DSCF7204-R).
# The two ranges overlap between 0.194 and 0.283 (DSCF4454-L, occluded,
# reads 0.2833) — no single EAR threshold gets 100% recall with zero
# false-gates in that band. Per the cost asymmetry (missed gate is
# catastrophic; false gate is mild, see the study's §2), the threshold is
# placed just above the occluded ceiling: gates the full occluded
# distribution while adding ~5% false-gates on narrow-but-open eyes.
# 0.285 also clears the golden-face fixture's frozen right-eye landmarks
# (EAR 0.288 — inside the measured overlap band above, but this is a
# synthetic image with no real eye to inspect, so treat 0.288 as a
# plumbing constraint, not a label). Do not raise _MIN_EAR without
# re-checking that fixture (tests/golden_face_fixture.py), or its eye ops
# go untested again (see CLAUDE.md's 2026-08-19 golden-face entry).
_MIN_EAR = 0.285

# Contrast: full-coverage (landmark-fallback) arm measured occluded range
# 0.012-0.863, visible range 0.064-0.949 — wide overlap, so this stays
# secondary to EAR. Grid-searched jointly with EAR across both arms at the
# occluded-recall=1.0 constraint: 0.55 contributes no additional recall
# over EAR alone on this corpus (BiSeNet-live arm's contrast coverage is
# broken — 35% of visible eyes have no usable eye mask, a documented
# class-5 collapse) but is retained for future hard cases (e.g. hair/wig)
# where EAR won't fire but contrast might. Eye-mask floor keeps BiSeNet
# class-collapse (eye_px==0) from firing the contrast arm.
_MIN_EYE_MASK_PIXELS = 100
_MIN_CONTRAST = 0.55

# Hair overlap — retained from the prior gate; structurally dead on the
# landmark-fallback path (see module docstring).  Kept conservative.
_HAIR_OVER_IRIS = 0.50
_HAIR_OVER_EYE = 0.75

_EYE_FIELDS = {
    "left": ("left_eye", "left_iris", "left_sclera"),
    "right": ("right_eye", "right_iris", "right_sclera"),
}

# Six-point EAR subsets for the two MediaPipe eye contours. ``parsing.LEFT_EYE``
# and ``RIGHT_EYE`` are 16-point polygons ordered around the contour, not the
# six points required by the standard EAR formula. The project uses
# camera-viewer names: the 33 cluster is ``left`` and the 263 cluster is
# ``right``. Each tuple is [outer canthus, inner canthus, upper-1, upper-2,
# lower-1, lower-2], the order expected by the formula below.
# Verified against the DSCF corpus: DSCF6961 (open) reads 0.475/0.500;
# DSCF4576's genuinely-closed right eye reads 0.088.  (An earlier draft
# used (33,160,158,133,153,144) / (263,385,387,362,373,380), which reads
# 0.603 open / 0.271 closed on the same reference eyes — inverted pairs
# put upper-lid points in the canthus slots.  Don't reorder these.)
_EAR_INDICES = {
    "left": (33, 133, 159, 158, 145, 144),
    "right": (263, 362, 386, 385, 374, 380),
}


# ──────────────────────────────────────────────────────────────────────────
# Signal 1: EAR (eyelid aspect ratio)
# ──────────────────────────────────────────────────────────────────────────

def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ear_for_side(
    landmarks: Any,
    side: Any,
    width: int,
    height: int,
) -> Optional[float]:
    """Return the 6-point EAR for one eye, or None if landmarks are missing.

    ``side`` may be ``"left"``/``"right"`` or an explicit six-index tuple.
    The latter keeps this helper tolerant of callers that already resolved the
    canonical map. Standard formula: EAR = (||upper1-lower1|| +
    ||upper2-lower2||) / (2 * ||outer-inner||), where outer/inner are the
    horizontal canthus corners and the two upper/lower pairs span the lid
    aperture.
    """
    if isinstance(side, str):
        indices = _EAR_INDICES.get(side)
    else:
        try:
            indices = tuple(side)[:6]
        except (TypeError, ValueError):
            return None
    if indices is None or len(indices) < 6:
        return None
    pts = []
    for i in indices:
        try:
            if not isinstance(i, (int, np.integer)):
                return None
            lm = landmarks.landmark[i] if hasattr(landmarks, "landmark") else landmarks[i]
        except (IndexError, AttributeError, TypeError):
            return None
        if not hasattr(lm, "x") or not hasattr(lm, "y"):
            return None
        try:
            x = float(lm.x) * width
            y = float(lm.y) * height
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        pts.append((x, y))
    outer, inner, u1, u2, l1, l2 = pts
    horiz = _euclid(outer, inner)
    if horiz < 1e-3:
        return None
    return (_euclid(u1, l1) + _euclid(u2, l2)) / (2.0 * horiz)


# ──────────────────────────────────────────────────────────────────────────
# Signal 2: tone-adaptive iris-vs-sclera contrast
# ──────────────────────────────────────────────────────────────────────────

def _as_mask(value: Any) -> Optional[np.ndarray]:
    """Return a finite float mask in ``[0, 1]`` or ``None``.

    Region stubs used by callers and tests are not always full
    ``FaceRegions`` instances, so this helper intentionally accepts any
    array-like and treats malformed masks as unavailable rather than
    raising in the eye stage.
    """
    if value is None:
        return None
    try:
        mask = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if mask.ndim != 2 or mask.size == 0:
        return None
    return np.clip(np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _weighted_overlap(occluder: np.ndarray, region: np.ndarray) -> float:
    if occluder.shape != region.shape:
        return 0.0
    denominator = float(region.sum())
    if denominator <= 1e-6:
        return 0.0
    return float(np.sum(occluder * region) / denominator)


def _contrast_for_side(
    img_bgr: Optional[np.ndarray],
    eye_mask: Optional[np.ndarray],
    iris_mask: Optional[np.ndarray],
) -> Optional[float]:
    """Return the tone-adaptive iris-vs-sclera contrast, or None.

    Contrast is (eye_hi - iris_median) / max(eye_hi, 1e-6) where eye_hi is
    the 95th-percentile L inside the eye mask.  This is a *relative*
    measure against the eye's own brightest pixels — never an absolute
    luminance threshold (per the tone-invariance/fairness rule).  Returns
    None when the eye mask is too small to trust (BiSeNet class-collapse).
    """
    if img_bgr is None or eye_mask is None or iris_mask is None:
        return None
    if not isinstance(eye_mask, np.ndarray) or not isinstance(iris_mask, np.ndarray):
        return None
    if eye_mask.shape != iris_mask.shape:
        return None
    if eye_mask.ndim != 2:
        return None
    try:
        image = np.asarray(img_bgr)
    except (TypeError, ValueError):
        return None
    if image.ndim != 3 or image.shape[:2] != eye_mask.shape or image.shape[2] != 3:
        return None
    eye_support = eye_mask > 0.25
    if int(np.count_nonzero(eye_support)) < _MIN_EYE_MASK_PIXELS:
        # BiSeNet fragility (class-5 collapse on ~65% of real portraits).
        # A missing eye mask is NOT an occlusion signal; fail open.
        return None
    iris_support = iris_mask > 0.25
    if int(np.count_nonzero(iris_support)) < 5:
        return None
    try:
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    except Exception:
        return None
    l_chan = lab[:, :, 0]
    eye_l = l_chan[eye_support]
    iris_l = l_chan[iris_support & eye_support]
    if len(iris_l) == 0:
        return None
    eye_hi = float(np.percentile(eye_l, 95))
    eye_lo = float(np.percentile(eye_l, 5))
    iris_med = float(np.median(iris_l))
    if eye_hi < 1e-6:
        return None

    # A flat synthetic patch (or a very low-information crop) is not
    # evidence of an occluded eye. In the full face path EAR remains the
    # primary signal, while direct callers fail open until contrast is
    # meaningful.
    if (eye_hi - eye_lo) / eye_hi < 0.05:
        return None
    return (eye_hi - iris_med) / eye_hi


# ──────────────────────────────────────────────────────────────────────────
# Per-eye decision
# ──────────────────────────────────────────────────────────────────────────

def _should_gate_eye(
    ear: Optional[float],
    contrast: Optional[float],
    eye_value: Any,
    iris_value: Any,
    hair_value: Any,
) -> bool:
    """Return whether one eye lacks enough visible evidence for enhancement."""
    iris = _as_mask(iris_value)
    if iris is None or float(iris.max()) < 0.01:
        return False

    iris_support = iris > 0.25
    if int(np.count_nonzero(iris_support)) < 5:
        return False

    eye = _as_mask(eye_value)
    # Direct enhancer callers do not provide landmarks.  If their parser
    # result contains a usable iris but no eye-contour evidence at all, keep
    # the conservative legacy safety behavior and suppress that side.  The
    # full per-face path does provide EAR; there, an empty BiSeNet eye mask is
    # treated as model fragility and fails open unless another signal fires.
    if (
        ear is None
        and contrast is None
        and (eye is None or float(eye.max()) < 0.01)
    ):
        return True

    if ear is not None and ear < _MIN_EAR:
        return True

    if contrast is not None and contrast < _MIN_CONTRAST:
        return True

    hair = _as_mask(hair_value)
    if hair is not None and hair.shape == iris.shape:
        if (_weighted_overlap(hair, iris) >= _HAIR_OVER_IRIS
                or _weighted_overlap(hair, eye if eye is not None else iris) >= _HAIR_OVER_EYE):
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def gate_occluded_eye_regions(
    regions: Any,
    landmarks: Any = None,
    img_bgr: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Any:
    """Return regions with unreliable eye masks disabled per eye.

    The return value is the original object when no eye is gated.  If one or
    both eyes are gated, a shallow copy is made and only the affected
    ``*_eye``, ``*_iris``, and ``*_sclera`` arrays are replaced with zeros.

    Args:
        regions: ``FaceRegions`` (or compatible namespace) for one face.
        landmarks: MediaPipe NormalizedLandmarkList (468+).  Required for
            the EAR signal; when None, only the contrast/hair signals are
            available (the gate degrades gracefully — never gates on
            missing EAR alone, which is the safe direction).
        img_bgr: (H, W, 3) uint8 BGR.  Required for the tone-adaptive
            contrast signal; when None, contrast is skipped.
        width, height: image dimensions for EAR.  Derived from ``img_bgr``
            when None.

    Memoization: when ``regions`` is a ``FaceRegions`` with an
    ``_eye_gate_cache`` slot, the gated result is cached on both the original
    regions object and its shallow gated copy. The re-application sites in
    ``EyeEnhancer.enhance`` (which pass only ``img_bgr``, no landmarks) reuse
    a decision made with landmark evidence, even after the image becomes a
    new downstream array. A cache miss with new landmark evidence recomputes
    from scratch.
    """
    # Memoization fast path: if a prior call cached a gated result on this
    # regions object and we are not being asked to recompute with fresh
    # landmark evidence, return the cached value.
    cache = getattr(regions, "_eye_gate_cache", None)
    if cache is not None and landmarks is None:
        cached_result = cached_img_id = None
        cached_with_landmarks = False
        if isinstance(cache, tuple) and len(cache) >= 3:
            cached_result, cached_img_id, cached_with_landmarks = cache[:3]
        elif isinstance(cache, tuple) and len(cache) == 2:
            cached_result, cached_img_id = cache
        if cached_result is not None and (
            cached_with_landmarks
            or cached_img_id == (id(img_bgr) if img_bgr is not None else None)
        ):
            return cached_result

    if img_bgr is not None and (width is None or height is None):
        height, width = img_bgr.shape[:2]

    ear_left = ear_right = None
    if landmarks is not None and width is not None and height is not None:
        ear_left = _ear_for_side(landmarks, "left", width, height)
        ear_right = _ear_for_side(landmarks, "right", width, height)

    contrast_left = _contrast_for_side(
        img_bgr,
        _as_mask(getattr(regions, "left_eye", None)),
        _as_mask(getattr(regions, "left_iris", None)),
    )
    contrast_right = _contrast_for_side(
        img_bgr,
        _as_mask(getattr(regions, "right_eye", None)),
        _as_mask(getattr(regions, "right_iris", None)),
    )

    hair = getattr(regions, "hair", None)
    gated_sides = []
    if _should_gate_eye(ear_left, contrast_left,
                        getattr(regions, "left_eye", None),
                        getattr(regions, "left_iris", None), hair):
        gated_sides.append("left")
    if _should_gate_eye(ear_right, contrast_right,
                        getattr(regions, "right_eye", None),
                        getattr(regions, "right_iris", None), hair):
        gated_sides.append("right")

    if not gated_sides:
        result = regions
    else:
        logger.info(
            "eye_visibility_gate: gating %s (EAR L=%.3f R=%.3f; contrast L=%s R=%s)",
            ",".join(gated_sides),
            ear_left if ear_left is not None else float("nan"),
            ear_right if ear_right is not None else float("nan"),
            f"{contrast_left:.3f}" if contrast_left is not None else "—",
            f"{contrast_right:.3f}" if contrast_right is not None else "—",
        )
        try:
            gated = copy.copy(regions)
        except Exception:
            return regions
        for side in gated_sides:
            for field in _EYE_FIELDS[side]:
                value = getattr(gated, field, None)
                if value is None:
                    continue
                try:
                    setattr(gated, field, np.zeros_like(value, dtype=np.float32))
                except (TypeError, ValueError, AttributeError):
                    continue
        result = gated

    # Cache on both the original object and a shallow gated copy so downstream
    # enhancer re-application sites hit the same decision. Guard for region
    # stubs without the slot.
    cache_entry = (
        result,
        id(img_bgr) if img_bgr is not None else None,
        landmarks is not None,
    )
    for cache_target in (regions, result):
        try:
            setattr(cache_target, "_eye_gate_cache", cache_entry)
        except (AttributeError, TypeError):
            pass

    return result


__all__ = ["gate_occluded_eye_regions"]

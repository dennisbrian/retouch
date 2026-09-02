"""Under-eye dark circle repair module.

Targeted darkness/shadow reduction under the eyes via LAB L-channel selective
brightening, with optional chroma desaturation for puffiness reduction.

2026-09-02 (v2, ``UndereyeProcessor.process``): the original detector kept the
darkest connected component of the landmark under-eye polygon that sat more
than an absolute 15 L below its surround. On real portraits that component is
the lower lash line / eye corner (the polygon's top edge *is* the lash
contour), so the lift landed on lashes and liner while genuine tear-trough
shadow was untouched; and the lift was capped twice (clip to ``max_lift`` and
a second ``max_lift / 100`` factor), so strength 100 could move at most 9 L.
See ``docs/plans/RESEARCH_DARK_CIRCLE_OP_2026_09_02.md``.

The v2 path builds its own support (polygon extended downward into the
tear trough, eye contour + lash margin excluded), measures darkness on a
masked low-pass of L relative to a cheek ring median (a *fraction* of the
face's own reference, per the tone-invariance rule), and lifts only the
low-frequency component so skin texture is preserved. ``strength`` is the
fraction of the shadow removed (1.0 = flattened to the cheek reference).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .parsing import FaceRegions
from .chromophore import decompose_chromophores, reconstruct_from_chromophores
from .utils import (
    blend_masked, normalize_mask, feather_mask as _feather_mask,
    bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, restore_outside_support,
)


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert BGR to LAB, handling both uint8 and float32."""
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert LAB back to BGR, handling both uint8 and float32."""
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


# --- v2 geometry / signal helpers ------------------------------------------
# All radii are fractions of the inter-eye distance (IED) so behaviour is
# resolution-invariant. Calibrated on the 83-face DSCF corpus (2026-09-02).
_UE_EXTEND_DOWN = 0.28     # extend the landmark polygon downward into the tear trough
_UE_EXTEND_SIDE = 0.06     # ... with only this much sideways reach (temple / nose bridge)
_UE_TAPER_FLOOR = 0.35     # lift weight at the bottom of the extension (1.0 at the lid)
_UE_LASH_MARGIN = 0.08     # exclusion margin around the eye contour (lashes, liner)
_UE_FEATHER = 0.06         # support feather radius
_UE_RING_OUTER = 0.10      # cheek reference ring width beyond the support
_UE_RING_GAP = 0.02        # gap between support and ring
_UE_EYE_RING_EXCL = 0.12   # keep the ring this far from the eye contour
_UE_LOWPASS_SIGMA = 0.04   # low-pass sigma for the shadow estimate
_UE_REL_T0 = 0.04          # relative darkness where the weight starts rising
_UE_REL_T1 = 0.12          # relative darkness where the weight saturates
_UE_MAX_LIFT = 30.0        # absolute L cap on the lift (single cap)
_UE_MAX_AB = 8.0           # absolute a/b cap on the chroma pull
_UE_AB_FACTOR = 0.75       # fraction of the a/b gap to close (concealer warmth)
# Under-eye polygon area / IED^2, corpus median: used to recover IED when the
# caller cannot supply it (legacy `repair` interface).
_UE_AREA_PER_IED2 = 0.02


def _ellipse(r: int) -> np.ndarray:
    r = max(int(r), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dilate_down(hard: np.ndarray, r: int, r_side: Optional[int] = None) -> np.ndarray:
    """Dilate a binary mask downward by ``r`` px with ``r_side`` sideways reach.

    Uses the upper half of an ellipse anchored at its bottom row, so a pixel
    at row y is set when any source pixel in rows y-r..y (within the ellipse)
    is set. Roughly-upright faces only; feathering absorbs small roll.
    """
    r = max(int(r), 1)
    rs = r if r_side is None else max(int(r_side), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * r + 1))[: r + 1, :]
    return cv2.dilate(hard, k, anchor=(rs, r))


def estimate_ied_from_mask(mask: np.ndarray) -> float:
    """Recover an IED estimate from a landmark under-eye mask's area."""
    area = float((mask > 0.5).sum())
    return max(float(np.sqrt(area / _UE_AREA_PER_IED2)), 20.0)


def build_undereye_support(
    mask: np.ndarray,
    ied: float,
    exclude: Optional[np.ndarray] = None,
    skin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the v2 (support, ring, valid) triple for one eye.

    Args:
        mask: landmark under-eye polygon mask (feathered or hard, 0-1 or 0-255).
        ied: inter-eye distance in px.
        exclude: eye-opening mask (landmark contour or BiSeNet). Dilated by the
            lash margin and removed from the support and the ring. When None,
            the support is eroded by the lash margin from its own top edge.
        skin: optional skin mask; the ring is intersected with it.

    Returns:
        support (float32 0-1, feathered, exactly 0 outside), ring (bool),
        valid (bool: pixels the low-pass estimate may sample from).
    """
    m = normalize_mask(mask)
    assert m is not None
    hard = (m > 0.5).astype(np.uint8)
    if not hard.any():
        z = np.zeros(m.shape, np.float32)
        return z, np.zeros(m.shape, bool), np.ones(m.shape, bool)

    lash_r = max(int(ied * _UE_LASH_MARGIN), 1)
    poly = hard.copy()
    ext_px = max(int(ied * _UE_EXTEND_DOWN), 1)
    hard = _dilate_down(hard, ext_px, max(int(ied * _UE_EXTEND_SIDE), 1))

    if exclude is not None:
        ex = (normalize_mask(exclude) > 0.3).astype(np.uint8)
        ex_lash = cv2.dilate(ex, _ellipse(lash_r))
        ex_ring = cv2.dilate(ex, _ellipse(max(int(ied * _UE_EYE_RING_EXCL), 1)))
    else:
        # No eye contour: the polygon's top edge is the lash line, so shave the
        # lash margin off the top rows of the polygon instead.
        # (Only the top ``lash_r`` rows of the polygon - the polygon is just
        # ~0.12 IED tall, so a further dilation would erase it.)
        top = (m > 0.5).astype(np.uint8)
        shifted = np.zeros_like(top)
        shifted[lash_r:] = top[:-lash_r]
        ex_lash = top & (1 - shifted)
        ex_ring = cv2.dilate(ex_lash, _ellipse(max(int(ied * _UE_EYE_RING_EXCL), 1)))
    hard[ex_lash > 0] = 0

    feather_r = max(int(ied * _UE_FEATHER), 2)
    support = _feather_mask(hard.astype(np.float32), radius=feather_r)
    support = np.clip(support, 0.0, 1.0).astype(np.float32)
    # keep the eye itself and everything beyond the feathered halo exactly 0;
    # the lash margin is removed with a *feathered* edge (a hard cut here left
    # a visible step where the lift met the un-lifted lash band, DSCF4576).
    halo = cv2.dilate(hard, _ellipse(feather_r * 3))
    support[halo == 0] = 0.0
    ex_soft = _feather_mask(ex_lash.astype(np.float32), radius=feather_r)
    support *= np.clip(1.0 - ex_soft, 0.0, 1.0).astype(np.float32)
    if exclude is not None:
        support[ex > 0] = 0.0
    # Taper with distance below the landmark polygon: dark circles are darkest
    # at the lid and fade into the cheek; contour makeup / hair shadow lower
    # down should not be flattened at full weight.
    dist = cv2.distanceTransform((1 - poly).astype(np.uint8), cv2.DIST_L2, 3)
    taper = 1.0 - (1.0 - _UE_TAPER_FLOOR) * np.clip(dist / float(ext_px), 0.0, 1.0)
    support *= taper.astype(np.float32)
    if skin is not None:
        sk = normalize_mask(skin)
        if sk is not None and sk.max() > 0.01:
            # keep the op on skin (hair / brows / background at the temple)
            support *= sk

    outer = cv2.dilate(hard, _ellipse(max(int(ied * (_UE_RING_OUTER + _UE_RING_GAP)), 2)))
    inner = cv2.dilate(hard, _ellipse(max(int(ied * _UE_RING_GAP), 1)))
    ring = (outer > 0) & (inner == 0) & (ex_ring == 0)
    if skin is not None:
        sk = normalize_mask(skin)
        if sk is not None and sk.max() > 0.01:
            ring &= sk > 0.5
    # The low-pass shadow estimate may sample everything but the lash margin
    # (the wider ``ex_ring`` exclusion is for the reference ring only;
    # using it here starved the estimate right under the lid).
    valid = ex_lash == 0
    return support, ring, valid


class UndereyeAnalyzer:
    """Detect dark-circle regions via LAB L-channel analysis."""

    def analyze(
        self,
        lab: np.ndarray,
        support: np.ndarray,
        ring: np.ndarray,
        valid: np.ndarray,
        ied: float,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """v2 shadow estimate.

        Returns ``(weight, lab_lp, ref)``: a 0-1 weight map (relative darkness
        of the masked low-pass L versus the ring median, smoothstepped between
        ``_UE_REL_T0`` and ``_UE_REL_T1``), the masked low-pass LAB, and the
        ring reference ``[L, a, b]`` (None when no usable reference exists).
        """
        sigma = max(float(ied) * _UE_LOWPASS_SIGMA, 1.0)
        v = valid.astype(np.float32)
        den = cv2.GaussianBlur(v, (0, 0), sigma)
        lab_lp = np.empty_like(lab)
        for c in range(3):
            num = cv2.GaussianBlur(lab[:, :, c] * v, (0, 0), sigma)
            lab_lp[:, :, c] = num / np.maximum(den, 1e-3)
        lab_lp[den < 1e-3] = lab[den < 1e-3]

        samples = lab_lp[ring]
        if samples.shape[0] < 50:
            samples = lab_lp[support > 0.5]
            if samples.shape[0] < 50:
                return np.zeros(support.shape, np.float32), lab_lp, None
        ref = np.median(samples, axis=0).astype(np.float32)
        rel = (ref[0] - lab_lp[:, :, 0]) / max(float(ref[0]), 1.0)
        weight = _smoothstep(rel, _UE_REL_T0, _UE_REL_T1).astype(np.float32)
        return weight, lab_lp, ref

    def detect_dark_circles(
        self,
        lab: np.ndarray,
        mask: np.ndarray,
        threshold_offset: float = 15.0,
    ) -> Tuple[np.ndarray, float]:
        """Detect dark circles by comparing L-channel to surrounding skin.

        Args:
            lab: (H, W, 3) LAB image in float32.
            mask: (H, W) under-eye region mask [0, 1].
            threshold_offset: L-darkening threshold (default 15 units).

        Returns:
            (detection_mask, local_median_l): Binary detection mask and reference L value.
        """
        if mask.max() < 0.01:
            return np.zeros_like(mask), np.nan

        L = lab[:, :, 0]

        # Get local skin reference: dilate mask to get surrounding cheek area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate((mask > 0.3).astype(np.uint8), kernel, iterations=1).astype(np.float32)
        surround = np.clip(dilated - mask, 0.0, 1.0)

        surround_pixels = L[surround > 0.3]
        # If no surround found, fall back to median of the masked region itself
        if len(surround_pixels) < 20:
            mask_pixels = L[mask > 0.3]
            if len(mask_pixels) < 20:
                return np.zeros_like(mask), np.nan
            local_median_l = np.median(mask_pixels)
        else:
            local_median_l = np.median(surround_pixels)

        # Flag regions darker than median - threshold_offset
        dark_threshold = local_median_l - threshold_offset
        dark_mask = ((L < dark_threshold) * mask).astype(np.float32)

        # Filter noise via connected components (min area ~100 pixels)
        dark_u8 = (dark_mask > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_u8)

        filtered_dark = np.zeros_like(dark_u8)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= 100:
                filtered_dark[labels == i] = 1

        return filtered_dark.astype(np.float32), float(local_median_l)


class UndereyeRemover:
    """Apply selective L-brightening to dark areas with edge preservation."""

    def brighten_dark_circles(
        self,
        lab: np.ndarray,
        dark_circle_mask: np.ndarray,
        local_median_l: float,
        strength: float = 0.5,
        max_lift: float = 30.0,
    ) -> np.ndarray:
        """Brighten dark circle regions via selective L-channel lifting.

        Args:
            lab: (H, W, 3) LAB image in float32.
            dark_circle_mask: (H, W) detection mask from UndereyeAnalyzer.
            local_median_l: Reference L value from surrounding skin.
            strength: 0–1 strength factor.
            max_lift: Maximum L increase (default 30, conservative).

        Returns:
            Modified LAB image.
        """
        if dark_circle_mask.max() < 0.01 or strength <= 0:
            return lab

        L = lab[:, :, 0].copy()
        darkness = np.clip(local_median_l - L, 0.0, max_lift)

        # Selective brightening: only brighten pixels darker than median.
        # ``max_lift`` is the single cap (the clip above); the pre-2026-09-02
        # code multiplied by ``max_lift / 100`` as well, which limited the lift
        # to 9 L at strength 1.0.
        l_lift = darkness * dark_circle_mask * strength
        lab[:, :, 0] = np.clip(L + l_lift, 0.0, 255.0)

        return lab

    def reduce_puffiness_chroma(
        self,
        lab: np.ndarray,
        dark_circle_mask: np.ndarray,
        strength: float = 0.5,
        chroma_reduction: float = 0.7,
    ) -> np.ndarray:
        """Desaturate under-eye via chroma reduction for puffiness minimization.

        Args:
            lab: (H, W, 3) LAB image in float32.
            dark_circle_mask: (H, W) under-eye region mask.
            strength: 0–1 strength factor.
            chroma_reduction: Target chroma ratio (0.7 = 30% desaturation).

        Returns:
            Modified LAB image with reduced chroma in dark-circle regions.
        """
        if dark_circle_mask.max() < 0.01 or strength <= 0:
            return lab

        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        # Compute chroma (offset-adjusted per cv2 LAB convention)
        chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
        hue = np.arctan2(b - 128.0, a - 128.0)

        # Reduce chroma selectively
        new_chroma = chroma * (1.0 - (1.0 - chroma_reduction) * strength * dark_circle_mask)

        # Reconstruct a/b
        a_new = 128.0 + new_chroma * np.cos(hue)
        b_new = 128.0 + new_chroma * np.sin(hue)

        lab[:, :, 1] = np.clip(a_new, 0.0, 255.0)
        lab[:, :, 2] = np.clip(b_new, 0.0, 255.0)

        return lab

    def feather_edges(
        self,
        mask: np.ndarray,
        feather_radius: int = 5,
    ) -> np.ndarray:
        """Apply morphological opening (dilate then erode) for soft edge blending.

        Args:
            mask: (H, W) binary mask.
            feather_radius: Feather kernel size in pixels.

        Returns:
            Feathered mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_radius, feather_radius))
        # Opening: erode then dilate (shrink then expand, softens edges)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Gaussian blur for further softening
        feathered = cv2.GaussianBlur(opened.astype(np.float32), (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        return np.clip(feathered, 0.0, 1.0).astype(np.float32)


class UndereyeProcessor:
    """Orchestrate under-eye dark circle analysis + removal."""

    def __init__(self):
        """Initialize analyzer and remover."""
        self.analyzer = UndereyeAnalyzer()
        self.remover = UndereyeRemover()

    def process(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        darken_removal_strength: float = 0.0,
        puffiness_reduction_strength: float = 0.0,
        ied: Optional[float] = None,
        exclude: Optional[np.ndarray] = None,
        skin: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one under-eye region (v2): low-frequency shadow lift +
        optional chroma reduction.

        Supports both uint8 and float32 input. Output dtype matches input dtype.
        Pixels outside the final support are returned bit-exact (the LAB
        roundtrip is confined to a ROI and restored outside support).

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            mask: (H, W) landmark under-eye polygon mask.
            darken_removal_strength: 0-1 fraction of the shadow to remove.
            puffiness_reduction_strength: 0-1 chroma desaturation strength.
            ied: inter-eye distance in px (estimated from ``mask`` when None).
            exclude: eye-opening mask (landmark contour preferred); lashes and
                the eye are kept out of the support.
            skin: optional skin mask restricting the cheek reference ring.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if (darken_removal_strength <= 0 and puffiness_reduction_strength <= 0) or mask is None or mask.max() < 0.01:
            return img_bgr

        if ied is None:
            ied = estimate_ied_from_mask(normalize_mask(mask))
        ied = float(ied)

        support, ring, valid = build_undereye_support(mask, ied, exclude=exclude, skin=skin)
        if support.max() < 0.01:
            return img_bgr

        # ROI: support + ring + low-pass reach
        ys, xs = np.nonzero((support > 0) | ring)
        pad = int(ied * (_UE_LOWPASS_SIGMA * 3 + _UE_RING_OUTER)) + 2
        h, w = support.shape
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)

        is_float = img_bgr.dtype == np.float32
        roi = img_bgr[y0:y1, x0:x1]
        roi_f = roi.astype(np.float32)
        lab = bgr_f32_to_lab_f32(roi_f)
        sup = support[y0:y1, x0:x1]
        weight, lab_lp, ref = self.analyzer.analyze(lab, sup, ring[y0:y1, x0:x1], valid[y0:y1, x0:x1], ied)
        if ref is None:
            return img_bgr
        eff = weight * sup

        if darken_removal_strength > 0:
            s = float(np.clip(darken_removal_strength, 0.0, 1.0))
            d_l = np.clip((ref[0] - lab_lp[:, :, 0]) * s * eff, 0.0, _UE_MAX_LIFT)
            d_a = np.clip((ref[1] - lab_lp[:, :, 1]) * s * eff * _UE_AB_FACTOR, -_UE_MAX_AB, _UE_MAX_AB)
            d_b = np.clip((ref[2] - lab_lp[:, :, 2]) * s * eff * _UE_AB_FACTOR, -_UE_MAX_AB, _UE_MAX_AB)
            lab[:, :, 0] = np.clip(lab[:, :, 0] + d_l, 0.0, 255.0)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + d_a, 0.0, 255.0)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + d_b, 0.0, 255.0)

        if puffiness_reduction_strength > 0:
            lab = self.remover.reduce_puffiness_chroma(
                lab, eff, strength=puffiness_reduction_strength, chroma_reduction=0.7
            )

        out_roi = lab_f32_to_bgr_f32(np.clip(lab, 0.0, 255.0))
        if not is_float:
            out_roi = np.clip(np.rint(out_roi), 0, 255).astype(np.uint8)
        out_roi = restore_outside_support(roi, out_roi, sup)
        result = img_bgr.copy()
        result[y0:y1, x0:x1] = out_roi
        return result

    def last_support(self, mask: np.ndarray, ied: Optional[float] = None,
                     exclude: Optional[np.ndarray] = None,
                     skin: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the v2 effective support for ``mask`` (diagnostics / QA)."""
        if ied is None:
            ied = estimate_ied_from_mask(normalize_mask(mask))
        return build_undereye_support(mask, float(ied), exclude=exclude, skin=skin)[0]

    def attenuate_hemoglobin(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        strength: float = 0.0,
    ) -> np.ndarray:
        """Experimental E-EYE-4 spike: reduce vascular color, not luminance.

        The existing under-eye path can brighten and desaturate a dark region,
        but neither operation distinguishes vascular color from a true shadow.
        This leaf operator attenuates only hemoglobin that exceeds the nearby
        cheek baseline, then restores the source L channel before compositing.
        It is intentionally not wired to engine parameters or recipes until
        real-image visual QA establishes a safe product control.
        """
        if strength <= 0.0 or mask is None or mask.max() < 0.01:
            return img_bgr

        s = float(np.clip(strength, 0.0, 1.0))
        is_float = img_bgr.dtype == np.float32
        region = normalize_mask(mask)
        assert region is not None
        if region.max() < 0.01:
            return img_bgr

        # Estimate a robust cheek baseline from a ring surrounding the
        # landmark-defined under-eye region. A minimum scale rejects normal
        # sensor variation on otherwise uniform skin.
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        ring = np.clip(
            cv2.dilate((region > 0.3).astype(np.uint8), ring_kernel).astype(np.float32)
            - region,
            0.0,
            1.0,
        )
        work = img_bgr.astype(np.float32)
        melanin, hemoglobin = decompose_chromophores(work)
        samples = hemoglobin[ring > 0.3]
        if samples.size < 20:
            samples = hemoglobin[region > 0.3]
        if samples.size < 20:
            return img_bgr

        baseline = float(np.median(samples))
        mad = float(np.median(np.abs(samples - baseline)))
        sigma = max(1.4826 * mad, 0.03)
        excess = np.clip(hemoglobin - baseline, 0.0, None)
        confidence = np.clip((excess - 2.0 * sigma) / (3.0 * sigma), 0.0, 1.0)
        effective_mask = region * confidence
        if effective_mask.max() < 1e-4:
            return img_bgr

        corrected_hb = hemoglobin - s * excess
        reconstructed = reconstruct_from_chromophores(
            melanin,
            corrected_hb,
            img_bgr=work,
        )

        # Chromophore reconstruction changes both color and brightness. Keep
        # source lightness so this spike cannot flatten an anatomical shadow.
        # Use the float-native LAB convention for both images. Mixing OpenCV's
        # uint8 LAB scale with float LAB would reintroduce a luminance shift.
        source_lab = _to_lab(work, True)
        corrected_lab = _to_lab(reconstructed, True)
        corrected_lab[:, :, 0] = source_lab[:, :, 0]
        corrected = _from_lab(corrected_lab, True)
        if not is_float:
            corrected = np.clip(corrected, 0.0, 255.0).astype(np.uint8)
        return blend_masked(img_bgr, corrected, effective_mask)


class UnderEyeRepairer:
    """Legacy interface for backward compatibility with existing engine wiring."""

    def __init__(self):
        """Initialize the processor."""
        self._processor = UndereyeProcessor()

    def repair(
        self,
        img_bgr: np.ndarray,
        regions: FaceRegions,
        strength: int = 40,
    ) -> np.ndarray:
        """Repair dark circles under both eyes (legacy interface).

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            regions: FaceRegions from parser.
            strength: 0–100 darkness removal intensity.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        result = img_bgr.copy()

        for mask, eye in ((regions.left_under_eye, getattr(regions, "left_eye", None)),
                          (regions.right_under_eye, getattr(regions, "right_eye", None))):
            if mask is not None and mask.max() > 0.01:
                result = self._processor.process(result, mask, darken_removal_strength=s, exclude=eye)

        return result

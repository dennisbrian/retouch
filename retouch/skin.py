"""Skin processing — smoothing, whitening, and CLAHE tone equalization.

All operations work within the skin mask to never affect hair, eyes, or background.
"""

from __future__ import annotations

import logging
import math

from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

import cv2
import numpy as np

from .utils import (
    blend_masked,
    normalize_mask,
    squeeze_mask,
    guided_filter,
    apply_u8_op_float,
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
)
from .color_space import bgr_to_lch, lch_to_bgr, skin_mask_lch
from .color_science import (
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_oklch,
    oklch_to_oklab,
    measure_skin_state,
    SKIN_LOCI,
)


def _blotch_bandpass(L: np.ndarray, face_width: float) -> np.ndarray:
    """Difference-of-Gaussians bandpass of luminance channel.

    Extracts the blotch frequency range (between pores and shading) by
    subtracting a small-scale Gaussian from a large-scale Gaussian.

    Args:
        L: (H, W) float32 luminance channel [0, 255].
        face_width: Face width in pixels. Sigma parameters scale as face_width/N.

    Returns:
        (H, W) float32 bandpass signal (can be negative).
    """
    sigma_small = max(1.0, face_width / 40.0)
    sigma_large = max(1.0, face_width / 12.0)

    # cv2.GaussianBlur with ksize=(0,0) uses sigma to determine kernel size
    low_small = cv2.GaussianBlur(L, (0, 0), sigma_small)
    low_large = cv2.GaussianBlur(L, (0, 0), sigma_large)

    return low_small - low_large


def _edge_protect(lab: np.ndarray) -> np.ndarray:
    """Continuous edge protection mask based on gradient magnitude.

    Protects feature lines (high gradients) from dodge & burn by returning
    a mask that is 1.0 in smooth areas and 0.0 near edges. Combined with
    highlight protection to avoid specular clipping.

    Args:
        lab: (H, W, 3) float32 LAB image.

    Returns:
        (H, W) float32 protection mask in [0, 1].
    """
    L = lab[:, :, 0]

    # Compute gradient magnitude of L using Sobel
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Inverse normalized gradient: 1.0 where flat, 0.0 where edges
    # Threshold of 30 protects features with strong gradients
    grad_protect = 1.0 - np.clip(grad_mag / 30.0, 0.0, 1.0)

    return grad_protect


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class SkinProcessor:
    """Skin smoothing, whitening, and tone equalization."""

    MEDIAPIPE_CHIN_IDX = 152

    @staticmethod
    def _build_dimensional_mask(
        regions: Any,
        shape: Tuple[int, int],
        attrs: Tuple[str, ...] = (
            "nose_bridge", "cheek_highlights_l", "cheek_highlights_r",
        ),
        feather: int = 0,
    ) -> np.ndarray:
        """Union of the named FaceRegions attrs, with optional Gaussian feather.

        Returns a float32 mask in [0, 1] with the same shape as the target.
        Feather is the Gaussian kernel size (odd int). 0 = no feather.
        """
        dim_mask: Optional[np.ndarray] = None
        for attr in attrs:
            m = getattr(regions, attr, None)
            if m is None:
                continue
            m_f = squeeze_mask(m.astype(np.float32, copy=False) if m.dtype != np.float32 else m)
            if dim_mask is None:
                dim_mask = m_f.copy()
                continue
            dim_mask = np.clip(dim_mask + m_f, 0.0, 1.0)

        if dim_mask is None:
            return np.zeros(shape, dtype=np.float32)

        if feather > 0:
            dim_mask = cv2.GaussianBlur(dim_mask, (feather, feather), 0)

        return dim_mask

    def whiten(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 30,
        tone: str = "rosy",
        hue_stable: bool = False,
    ) -> np.ndarray:
        """Adaptive Rosy Foundation: LAB-based skin whitening and rosy/porcelain cosmetic shift.

        Supports both uint8 and float32 input. If input is float32 [0, 255], output is float32.
        If input is uint8, output is uint8.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            skin_mask: (H, W) float mask 0–1. May be None to skip processing.
            strength: -100–100 signed whitening strength. Positive lifts, negative
                deepens shadows. 0 returns the input unchanged.
            tone: Cosmetic tone: "rosy", "porcelain", or "neutral".
            hue_stable: If True, preserve hue using OKLab/OKLCh.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength == 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        s = strength / 100.0
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        if hue_stable:
            oklab = bgr_to_oklab(img_u8)
            oklch = oklab_to_oklch(oklab)
            L = oklch[:, :, 0]
            protection = self._get_highlight_protection(lab)
            l_val = L.copy()
            shadow_protection = (np.clip((l_val * 255.0 - 80.0) / 40.0, 0.0, 1.0)
                                 if s >= 0
                                 else np.clip((l_val * 255.0 - 10.0) / 20.0, 0.0, 1.0))
            skin_color_mask = skin_mask * shadow_protection
            lift_factor = 0.16 * abs(s)
            if s >= 0:
                L_new = L + (1.0 - L) * skin_color_mask * lift_factor * protection
            else:
                L_new = L + L * skin_color_mask * lift_factor * shadow_protection
            oklch[:, :, 0] = np.clip(L_new, 0.0, 1.0)
            oklab_out = oklch_to_oklab(oklch)
            out = oklab_to_bgr(oklab_out, float32_out=is_float)
            return blend_masked(img_bgr, out, skin_mask)

        protection = self._get_highlight_protection(lab)

        # Fix #2: Copy l_val to prevent stale reads on subsequent passes
        l_val = lab[:, :, 0].copy()
        shadow_protection = np.clip((l_val - 80.0) / 40.0, 0.0, 1.0)
        skin_color_mask = skin_mask * shadow_protection

        # Fix #7: Calculate has_skin once to avoid redundant np.any calls
        skin_indices = skin_mask > 0.3
        has_skin = np.any(skin_indices)

        median_a = 128.0
        median_b = 128.0
        if has_skin:
            median_a = np.median(lab[:, :, 1][skin_indices])
            median_b = np.median(lab[:, :, 2][skin_indices])

        # Soft-clipping luminance lift/reduction using skin_color_mask
        if s >= 0:
            lift_factor = 0.16 * s
            lab[:, :, 0] = l_val + (255.0 - l_val) * skin_color_mask * lift_factor * protection
        else:
            shadow_decay = np.clip((l_val - 10.0) / 20.0, 0.0, 1.0)
            lift_factor = 0.16 * s
            lab[:, :, 0] = l_val + l_val * skin_color_mask * lift_factor * shadow_decay

        abs_s = abs(s)
        if tone == "porcelain":
            lab[:, :, 2] = lab[:, :, 2] - 5.0 * abs_s * skin_color_mask
        elif tone == "neutral":
            pass
        else:  # rosy
            lab[:, :, 1] = lab[:, :, 1] + 3.0 * abs_s * skin_color_mask
            lab[:, :, 2] = lab[:, :, 2] - 4.0 * abs_s * skin_color_mask

        if has_skin:
            a_target = 128.0 + (median_a - 128.0) * 0.85
            b_target = 128.0 + (median_b - 128.0) * 0.85
            blend_factor = 0.12 * abs_s
            lab[:, :, 1] = lab[:, :, 1] + (a_target - lab[:, :, 1]) * skin_mask * blend_factor
            lab[:, :, 2] = lab[:, :, 2] + (b_target - lab[:, :, 2]) * skin_mask * blend_factor

        if is_float:
            whitened = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, whitened, skin_mask)

    def smooth_undereye_shadow(
        self,
        img_bgr: np.ndarray,
        under_eye_masks: Sequence[Optional[np.ndarray]],
        skin_mask: Optional[np.ndarray] = None,
        strength: float = 0.5,
        feather_radius: int = 3,
    ) -> np.ndarray:
        """Selectively soften under-eye shadows (dark circles) at conservative strength.

        Detects shadow pixels *inside* the under-eye region masks and applies a
        targeted edge-preserving guided filter, blended back via a feathered mask.
        The sclera is never touched because processing is confined to the
        under-eye masks. Preserves shadow depth (only ~50% of ``strength`` is used).

        Supports both uint8 and float32 input ([0, 255] range). Returns the same
        dtype as input. Pixel math stays float32; no uint8 intermediates in arithmetic.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            under_eye_masks: Tuple/list of one or two float32 masks
                (``regions.left_under_eye``, ``regions.right_under_eye``). A mask
                that is None is skipped.
            skin_mask: Optional (H, W) float mask 0–1. Non-skin pixels are excluded.
            strength: 0–1 shadow smoothing strength. ``<= 0`` is a no-op (input
                returned byte-identical).
            feather_radius: Odd Gaussian kernel size for mask feathering. Larger =
                softer blend boundary.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if strength <= 0 or under_eye_masks is None:
            return img_bgr

        # Collect valid under-eye masks (None entries skipped).
        zones: List[np.ndarray] = []
        for m in under_eye_masks:
            if m is None:
                continue
            m_f = squeeze_mask(m.astype(np.float32, copy=False) if getattr(m, "dtype", None) != np.float32 else m)
            if m_f.max() < 0.01:
                continue
            zones.append(m_f)
        if not zones:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        # Explicit colorspace boundary: BGR -> uint8 for cvtColor, then float32 LAB.
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        img_f = img_bgr.astype(np.float32) if not is_float else img_bgr
        result = img_f.copy()

        # Feather kernel must be an odd diameter.
        fr = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        if fr < 1:
            fr = 1

        skin_gate = squeeze_mask(skin_mask.astype(np.float32, copy=False)) if skin_mask is not None else None

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        processed_any = False
        for ue in zones:
            zone = ue > 0.5
            zone_px = int(np.count_nonzero(zone))
            if zone_px == 0:
                continue

            # Local median computed only over the under-eye zone (robust baseline).
            local_median = float(np.median(L[zone]))
            shadow = (
                (L < local_median - 15.0) & (L > 40.0) & (L < 200.0) & zone
            ).astype(np.float32)

            # Morphological open strips sub-3x3 noise from the shadow map.
            shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, morph_kernel)

            coverage = float(np.count_nonzero(shadow > 0.5)) / float(zone_px)
            # Strength-aware gate: at higher strength we treat milder dark circles
            # (lower coverage needed), keeping low strength conservative so the
            # feature never hollows bright, shadow-free under-eyes.
            min_coverage = max(0.1, 0.35 - 0.25 * strength)
            if coverage < min_coverage:
                continue  # Too few shadow pixels: this eye has no dark circle.

            # Feather the shadow mask to avoid hard seams at zone edges.
            shadow_feathered = cv2.GaussianBlur(shadow, (fr, fr), 0)
            shadow_feathered = np.clip(shadow_feathered, 0.0, 1.0)

            # Gate by skin so non-skin (e.g. eye/sclera-adjacent) pixels are excluded.
            if skin_gate is not None:
                shadow_feathered = shadow_feathered * skin_gate

            if shadow_feathered.max() < 0.01:
                continue

            blend = shadow_feathered * 0.5 * strength

            # Conservative guided filter per channel (edge-preserving smoothing).
            smoothed = np.empty_like(img_f)
            try:
                for c in range(3):
                    smoothed[:, :, c] = guided_filter(
                        img_f[:, :, c], radius=4, eps=1.0, guide=None
                    )
            except cv2.error as e:
                logger.warning("guided_filter failed in smooth_undereye_shadow: %s; skipping eye.", e)
                continue

            blend_3d = blend[:, :, np.newaxis]
            result = result * (1.0 - blend_3d) + smoothed * blend_3d
            processed_any = True

        if not processed_any:
            return img_bgr

        if is_float:
            return np.clip(result, 0, 255).astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def equalize(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 40,
        ref_lab: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """CLAHE + local average color equalization to unify skin tone.

        Supports both uint8 and float32 input. If input is float32 [0, 255], output is float32.
        If input is uint8, output is uint8.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100 equalization intensity.
            ref_lab: Optional reference LAB image whose a/b statistics
                are used as the colour target.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = math.sqrt(strength / 100.0)
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            # Fix #1: Safely snapshot the reference before mutations, keeping channel indexing correct
            if ref_lab is not None:
                ref_ab = ref_lab.astype(np.float32)[:, :, 1:3].copy()
            else:
                ref_ab = lab[:, :, 1:3].copy()
            median_a = np.median(ref_ab[:, :, 0][skin_indices])
            median_b = np.median(ref_ab[:, :, 1][skin_indices])
            pull = (0.20 * s * skin_mask)
            lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * pull
            lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * pull

        # PERF: Run CLAHE only on the L channel to save memory.
        # CLAHE is a uint8 primitive (8-bit histogram bins), so L is quantized
        # to uint8 here regardless of input dtype.
        l_chan_u8 = np.clip(lab[:, :, 0], 0, 255).astype(np.uint8)
        clip_limit = 1.0 + s * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_chan_u8)

        # Snapshot original L over skin before CLAHE blend (a/b pull above did
        # not touch L, so lab[:, :, 0] here is still the original luminance).
        l_orig = lab[:, :, 0].copy()

        protection = self._get_highlight_protection(lab)
        l_clahe_f = l_clahe.astype(np.float32)
        l_final_f = l_clahe_f * protection + lab[:, :, 0] * (1.0 - protection)

        # Preserve mean skin luminance: CLAHE redistributes a bright, low-contrast
        # histogram downward, shifting exposure. equalize should even out tone,
        # not change exposure — so re-center the mean over skin with a global shift.
        if np.any(skin_indices):
            mean_orig = float(l_orig[skin_indices].mean())
            mean_new = float(l_final_f[skin_indices].mean())
            l_final_f = l_final_f + (mean_orig - mean_new)

            # Never let CLAHE make skin LESS even: on pale, already-even faces
            # it can amplify luminance variance (gray mottling on cheeks). If
            # skin L std increased, rescale deviations around the preserved
            # mean back to the original std. Where CLAHE genuinely evens tone
            # (std decreases), behavior is unchanged.
            sd_orig = float(l_orig[skin_indices].std())
            sd_new = float(l_final_f[skin_indices].std())
            if sd_new > sd_orig and sd_new > 1e-6:
                l_final_f = mean_orig + (l_final_f - mean_orig) * (sd_orig / sd_new)

        lab[:, :, 0] = np.clip(l_final_f, 0, 255)

        # Fix #6: Explicitly clip all channels to prevent AB wrap-around artifacts
        if is_float:
            equalized = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            equalized = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, equalized, skin_mask * s)

    def dodge_burn(self, img_bgr: np.ndarray, regions: Any, strength: int = 40) -> np.ndarray:
        """Subtle 3-5% sculpting (brighten nose bridge, forehead center, cheeks;
        darken jawline/edges), scaled by highlight protection.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            regions: FaceRegions object with brightening/darkening sub-masks.
            strength: 0–100 dodge & burn intensity.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or regions is None or getattr(regions, "skin", None) is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        s = strength / 100.0
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        protection = self._get_highlight_protection(lab)

        brighten_mask = self._build_dimensional_mask(
            regions,
            shape=img_bgr.shape[:2],
            attrs=(
                "nose_bridge", "forehead_center",
                "cheek_highlights_l", "cheek_highlights_r",
            ),
        )

        darken_mask = (regions.jawline_contour.astype(np.float32, copy=False)
                       if regions.jawline_contour is not None
                       else np.zeros(img_bgr.shape[:2], dtype=np.float32))
        darken_mask = np.clip(darken_mask - brighten_mask, 0.0, 1.0)

        l_val = lab[:, :, 0]
        lab[:, :, 0] = l_val + (255.0 - l_val) * brighten_mask * 0.05 * s * protection
        lab[:, :, 0] = np.clip(lab[:, :, 0] - lab[:, :, 0] * darken_mask * 0.04 * s, 0, 255)

        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Exclude hair and eyebrows from dodge/burn output so we never
        # sculpt inside hair roots or scrub pigment off brow hairs.
        blend = regions.skin
        for excl_attr in ("hair", "left_eyebrow", "right_eyebrow"):
            excl = getattr(regions, excl_attr, None)
            if excl is not None:
                blend = np.clip(
                    blend.astype(np.float32, copy=False)
                    - excl.astype(np.float32),
                    0.0,
                    1.0,
                )
        return blend_masked(img_bgr, result, blend)

    def wrinkle_soften(
        self,
        img_bgr: np.ndarray,
        regions: Any,
        strength: float = 0,
    ) -> np.ndarray:
        """Ridge-aware wrinkle and line softening via black-hat morphology.

        Detects directional ridge patterns (wrinkles) in landmark-defined zones
        using black-hat morphology at multiple orientations, then attenuates
        the ridge signal from the L channel. Depth is capped at 60% reduction
        to preserve natural facial geometry.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            regions: FaceRegions object with wrinkle zones.
            strength: 0–100 wrinkle softening intensity.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or regions is None or getattr(regions, "skin", None) is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        h_img, w_img = img_bgr.shape[:2]
        s = strength / 100.0

        # Estimate wrinkle scale from image dimensions
        # Typical wrinkle width is ~ied*0.15-0.3; approximate ied as face height
        face_height = h_img * 0.6  # Approximate face height as 60% of image
        wrinkle_length = int(face_height * 0.2)
        wrinkle_length = max(wrinkle_length, 8)

        # Build wrinkle-eligible mask: union of all wrinkle zones
        wrinkle_mask_attrs = []
        for attr in ("forehead", "nasolabial_l", "nasolabial_r",
                     "crows_feet_l", "crows_feet_r", "neck"):
            if hasattr(regions, attr):
                wrinkle_mask_attrs.append(attr)

        wrinkle_mask = self._build_dimensional_mask(
            regions,
            shape=(h_img, w_img),
            attrs=tuple(wrinkle_mask_attrs),
            feather=0,
        )

        # Exclude hair, eyebrows, and eyes explicitly. Eyes are a defense-in-depth
        # guard: even if a wrinkle zone's landmark geometry is imperfect, the eye
        # itself (lash/lid) must never be touched by this op.
        for excl_attr in ("hair", "left_eyebrow", "right_eyebrow", "left_eye", "right_eye"):
            excl = getattr(regions, excl_attr, None)
            if excl is not None:
                wrinkle_mask = np.clip(
                    wrinkle_mask.astype(np.float32, copy=False)
                    - excl.astype(np.float32),
                    0.0,
                    1.0,
                )

        if wrinkle_mask.max() < 0.01:
            return img_bgr

        # Convert to LAB and extract L channel
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Detect ridges using difference-of-Gaussians at wrinkle scale
        # This isolates mid-frequency structures (wrinkles, fine lines)
        sigma_small = max(1.0, wrinkle_length / 6.0)
        sigma_large = max(1.0, wrinkle_length / 2.0)

        gaussian_small = cv2.GaussianBlur(L, (0, 0), sigma_small)
        gaussian_large = cv2.GaussianBlur(L, (0, 0), sigma_large)

        # DoG isolates mid-frequency details (wrinkles live here)
        dog = gaussian_small - gaussian_large

        # Wrinkles are dark ridges, so they appear as negative DoG values
        # Clip to get absolute magnitude of negative features only
        ridge_signal = np.clip(-dog, 0.0, 255.0)

        # Attenuate ridge signal: subtract from L, capped at 60% reduction
        ridge_signal_masked = ridge_signal * wrinkle_mask

        # Compute scale for normalization
        ridge_nonzero = ridge_signal_masked[ridge_signal_masked > 0.1]
        if ridge_nonzero.size > 0:
            # Normalize by 95th percentile for stability
            ridge_scale = np.percentile(ridge_nonzero, 95)
            if ridge_scale < 1.0:
                ridge_scale = 1.0
        else:
            ridge_scale = 1.0

        # Attenuation: strength scales the effect, capped at 60% of ridge depth
        # ridge_signal is in [0, 255], scale it by strength
        # At strength=100, reduce ridge signal by max 60% (preserve 40% residual depth)
        max_reduction_frac = 0.6  # 60% cap on depth reduction

        # Scale attenuation directly from ridge signal
        # ridge_signal is already in magnitude units (0-255)
        # strength is normalized (0-100), so we use it directly as a fraction
        attenuation = ridge_signal_masked * (s / 100.0) * max_reduction_frac * 100.0

        # Apply to L channel: ADD attenuation to lighten dark wrinkles
        # (dark wrinkles have high negative DoG values, so we lighten them)
        lab[:, :, 0] = np.clip(L + attenuation, 0.0, 255.0)

        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Blend using the wrinkle mask
        return blend_masked(img_bgr, result, wrinkle_mask)

    def harmonize_neck(
        self,
        img_bgr: np.ndarray,
        face_landmarks: Any,
        person_mask: Optional[np.ndarray],
        face_skin_mask: Optional[np.ndarray],
        neck_mask: Optional[np.ndarray] = None,
        strength: int = 40,
    ) -> np.ndarray:
        """Unify the neck and chest skin color and brightness with the face skin.

        Supports both uint8 and float32 input. If input is float32 [0, 255], output is float32.
        If input is uint8, output is uint8.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            face_landmarks: MediaPipe NormalizedLandmarkList.
            person_mask: (H, W) float person segmentation mask. May be None.
            face_skin_mask: (H, W) float skin mask of the face.
            neck_mask: Optional pre-computed neck mask.
            strength: 0–100 harmonization intensity.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength <= 0 or person_mask is None or face_skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = strength / 100.0
        h_img, w_img = img_bgr.shape[:2]
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        face_skin_indices = face_skin_mask > 0.3
        if not np.any(face_skin_indices):
            return img_bgr

        face_median_l = np.median(lab[:, :, 0][face_skin_indices])
        face_median_a = np.median(lab[:, :, 1][face_skin_indices])
        face_median_b = np.median(lab[:, :, 2][face_skin_indices])

        coords = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in face_landmarks.landmark]
        if not coords:
            return img_bgr
        xs, ys = zip(*coords)
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)

        if neck_mask is not None and neck_mask.max() > 0.01:
            neck_mask_final = neck_mask.copy()
        else:
            chin_y = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].y * h_img)
            chin_x = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].x * w_img)

            neck_y1 = chin_y
            neck_y2 = min(int(chin_y + face_h * 1.5), h_img)
            neck_x1 = max(int(chin_x - face_w * 0.8), 0)
            neck_x2 = min(int(chin_x + face_w * 0.8), w_img)

            if neck_y2 <= neck_y1 or neck_x2 <= neck_x1:
                return img_bgr

            pm = normalize_mask(person_mask)
            pm = squeeze_mask(pm)

            neck_mask_est = np.zeros((h_img, w_img), dtype=np.float32)
            neck_mask_est[neck_y1:neck_y2, neck_x1:neck_x2] = pm[neck_y1:neck_y2, neck_x1:neck_x2]

            dist_ab = np.sqrt((lab[:, :, 1] - face_median_a) ** 2 + (lab[:, :, 2] - face_median_b) ** 2)
            face_std_a = np.std(lab[:, :, 1][face_skin_indices])
            face_std_b = np.std(lab[:, :, 2][face_skin_indices])
            ab_threshold = max(12.0, (face_std_a + face_std_b) * 2.0)
            skin_match = (dist_ab < ab_threshold).astype(np.float32)
            
            # Fix #5: Erode face skin mask to create a smoother boundary exclusion
            face_skin_eroded = cv2.erode(face_skin_mask, np.ones((5, 5), np.uint8))
            neck_mask_final = np.clip(neck_mask_est * skin_match - face_skin_eroded, 0, 1)

        lm = face_landmarks.landmark
        d_left = abs(lm[6].x - lm[234].x)
        d_right = abs(lm[454].x - lm[6].x)
        yaw_ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)

        if yaw_ratio <= 1.3:
            p1 = np.array([lm[33].x * w_img, lm[33].y * h_img, lm[33].z * face_w])
            p2 = np.array([lm[263].x * w_img, lm[263].y * h_img, lm[263].z * face_w])
            p3 = np.array([lm[6].x * w_img, lm[6].y * h_img, lm[6].z * face_w])
            
            n = np.cross(p2 - p1, p3 - p1)
            n = n / (np.linalg.norm(n) + 1e-5)
            
            # Approximate plane distance in screen-space (XY only) to avoid geometric flaws of substitute Z
            Y, X = np.ogrid[:h_img, :w_img]
            dist_to_plane = np.abs(n[0] * (X - p1[0]) + n[1] * (Y - p1[1]))
            
            threshold = face_w * 0.15
            depth_gate = (dist_to_plane < threshold).astype(np.float32)
            neck_mask_final *= depth_gate

        if neck_mask_final.max() < 0.01:
            return img_bgr

        k_blur = max(15, int(face_w * 0.03)) | 1
        neck_mask_final = cv2.GaussianBlur(neck_mask_final, (k_blur, k_blur), 0)

        neck_indices = neck_mask_final > 0.3
        if not np.any(neck_indices):
            return img_bgr

        neck_median_l = np.median(lab[:, :, 0][neck_indices])
        neck_median_a = np.median(lab[:, :, 1][neck_indices])
        neck_median_b = np.median(lab[:, :, 2][neck_indices])

        l_diff = (face_median_l - neck_median_l) * 0.75 * s
        lab[:, :, 0] = np.clip(lab[:, :, 0] + neck_mask_final * l_diff, 0, 255)

        a_diff = (face_median_a - neck_median_a) * 0.7 * s
        b_diff = (face_median_b - neck_median_b) * 0.7 * s
        lab[:, :, 1] = np.clip(lab[:, :, 1] + neck_mask_final * a_diff, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + neck_mask_final * b_diff, 0, 255)

        if is_float:
            return lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def apply_specular_bloom(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 40,
        tone: str = "rosy",
    ) -> np.ndarray:
        """Generates a soft pink/lavender or neutral halo around skin highlights where L > 220.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float skin mask 0–1.
            strength: 0–100 bloom intensity.
            tone: Cosmetic tone shift: "rosy", "porcelain", or "neutral".

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]

        highlight_mask = np.clip((l_chan - 220.0) / 20.0, 0.0, 1.0) * (skin_mask > 0.1).astype(np.float32)

        if highlight_mask.max() < 0.01:
            return img_bgr

        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)
        dilate_k = max(3, int(min_dim * 0.005)) | 1
        blur_k = max(15, int(min_dim * 0.03)) | 1

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        highlight_mask = cv2.dilate(highlight_mask, kernel_dilate)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (blur_k, blur_k), 0)
        highlight_mask *= (skin_mask > 0.1).astype(np.float32)
        highlight_mask = np.clip(highlight_mask, 0.0, 1.0)

        # Fix #1: Explicitly copy channels to avoid passing live views into cv2.merge, clipping to prevent wraps
        l_shifted = np.clip(l_chan + 15.0, 0, 255)
        a_shifted = np.clip(lab[:, :, 1], 0, 255)
        b_shifted = np.clip(lab[:, :, 2], 0, 255)
        
        if tone == "porcelain":
            b_shifted = np.clip(b_shifted - 10.0, 0, 255)
        elif tone == "rosy":
            a_shifted = np.clip(a_shifted + 10.0, 0, 255)
            b_shifted = np.clip(b_shifted - 5.0, 0, 255)
        
        lab_shifted = cv2.merge([l_shifted, a_shifted, b_shifted])
        if is_float:
            img_shifted_bgr = lab_f32_to_bgr_f32(lab_shifted)
        else:
            img_shifted_bgr = cv2.cvtColor(lab_shifted.astype(np.uint8), cv2.COLOR_LAB2BGR)

        s = strength / 100.0
        return blend_masked(img_bgr, img_shifted_bgr, highlight_mask * s)

    def restore_micro_texture(
        self,
        smoothed_bgr: np.ndarray,
        original_bgr: np.ndarray,
        regions: Any,
        strength: int = 20,
        smooth_strength: float = 0.5,
    ) -> np.ndarray:
        """Re-inject dimensional micro-contrast lost during frequency-based smoothing.

        Computes the per-pixel difference between the original and smoothed
        canvases and adds it back selectively in dimensional face zones
        (nose bridge, cheek highlights, under-eye transition). This keeps
        the skin looking "naturally good" rather than "retouched" — the
        subtle micro-contrast that gives a face its dimensionality.

        Args:
            smoothed_bgr: (H, W, 3) uint8 BGR canvas after frequency-based
                smoothing (output of ``FrequencySeparator.combine``).
            original_bgr: (H, W, 3) uint8 BGR canvas before smoothing.
            regions: ``FaceRegions`` with ``nose_bridge``, ``cheek_highlights_l``,
                ``cheek_highlights_r``, ``left_under_eye``, ``right_under_eye``
                sub-masks.
            strength: 0–50 restore amount. 0 = off, 25 = subtle, 50 = strong.
            smooth_strength: 0–1 strength of the smoothing that was applied.
                Restoration scales with this so it has zero effect at
                ``smooth_strength=0`` (no smoothing ⇒ no lost detail to
                restore).

        Returns:
            (H, W, 3) uint8 BGR canvas with restored micro-contrast.
        """
        if strength <= 0 or smooth_strength <= 0 or regions is None:
            return smoothed_bgr

        h, w = smoothed_bgr.shape[:2]
        feather = max(3, int(min(h, w) * 0.01)) | 1

        # Build a dimensional mask: where micro-contrast actually matters
        # (cheek highlights, nose bridge, under-eye transition). These are
        # the zones the bilateral+mid_reduction step is most likely to
        # flatten, and where restoration is perceptually most valuable.
        dim_mask = self._build_dimensional_mask(
            regions,
            shape=smoothed_bgr.shape[:2],
            attrs=(
                "nose_bridge",
                "cheek_highlights_l",
                "cheek_highlights_r",
                "left_under_eye",
                "right_under_eye",
            ),
            feather=feather,
        )

        if dim_mask.max() < 0.01:
            return smoothed_bgr

        # Detail = what smoothing killed (signed).
        detail = original_bgr.astype(np.float32) - smoothed_bgr.astype(np.float32)

        # Effective restore amount:
        #   strength/100  → 0..0.5 for the GUI's 0..50 range
        #   * smooth_strength  → tapers to 0 when smoothing was off
        #   * dim_mask         → confined to dimensional zones
        # At default (strength=20, smooth_strength=0.5): 0.10 × dim_mask,
        # which is right in the 0.15–0.25 sweet spot the user asked for
        # when smoothing is heavier.
        restore_amount = (strength / 100.0) * float(smooth_strength)
        dim_mask_3d = dim_mask[:, :, np.newaxis]

        restored = smoothed_bgr.astype(np.float32) + detail * restore_amount * dim_mask_3d
        restored = np.clip(restored, 0, 255)
        if smoothed_bgr.dtype == np.float32:
            # E1 float path: pure arithmetic op — stay float32, no quantization.
            return restored
        return restored.astype(np.uint8)

    def local_clarity(
        self,
        img_bgr: np.ndarray,
        regions: Any,
        strength: float = 0.10,
        radius: int = 20,
    ) -> np.ndarray:
        """Local clarity — apply a high-pass boost only to nose/lips/eyes.

        High-pass = original - GaussianBlur(original, radius).
        Result = original + high_pass * strength * local_mask.

        This gives "pop" to dimensional features (nose, lips, eye area) without
        the global clarity side effect of accentuating skin texture everywhere.
        Modulated by the global clarity slider in the engine (0-100 → 0.0-0.3).

        Args:
            img_bgr: (H, W, 3) uint8 BGR canvas.
            regions: ``FaceRegions`` with ``nose``, ``lips``, ``left_eye``,
                ``right_eye``, ``nose_bridge`` sub-masks. Missing entries are
                skipped.
            strength: 0.0–0.3 boost factor on the high-pass signal. 0 = no-op.
            radius: Gaussian radius (in pixels) used to extract the
                low-frequency base for the high-pass.

        Returns:
            (H, W, 3) uint8 BGR canvas with localized clarity applied.
        """
        if strength <= 0 or regions is None:
            return img_bgr

        h, w = img_bgr.shape[:2]
        feather = max(3, int(min(h, w) * 0.01)) | 1

        local_mask = self._build_dimensional_mask(
            regions,
            shape=img_bgr.shape[:2],
            attrs=("nose", "lips", "left_eye", "right_eye", "nose_bridge"),
            feather=feather,
        )

        if local_mask.max() < 0.01:
            return img_bgr

        ksize = max(radius * 2 + 1, 3)
        img_f = img_bgr.astype(np.float32)
        low = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
        high = img_f - low

        mask_3d = local_mask[:, :, np.newaxis]
        boosted = img_f + high * float(strength) * mask_3d
        if img_bgr.dtype == np.float32:
            return np.clip(boosted, 0, 255)
        return np.clip(boosted, 0, 255).astype(np.uint8)

    def flatten(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
    ) -> np.ndarray:
        """Edge-preserving cel flatten via guided filter on LAB L.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100 flattening intensity.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = strength / 100.0
        h, w = img_bgr.shape[:2]

        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        r = max(8, int(min(h, w) * 0.04))
        eps = (6.0 + 14.0 * s) ** 2

        # Apply guided filter (self-guided with max_dim=1200 for proxy resolution)
        q = guided_filter(L, radius=r, eps=eps, guide=None, max_dim=1200)

        skin_3d = skin_mask[:, :, np.newaxis]
        lab[:, :, 0] = L + (q - L) * s * skin_3d[:, :, 0]
        if is_float:
            out = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, out, skin_mask)

    def quantize_tones(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
        bands: int = 3,
        softness: float = 0.35,
    ) -> np.ndarray:
        """Soft cel shading via band-based L quantization on skin pixels.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100 quantization intensity.
            bands: Number of tone bands (2 or 3).
            softness: Transition width factor [0, 1].

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = strength / 100.0
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        skin_px = L[skin_mask > 0.3]
        if skin_px.size < 50:
            return img_bgr

        if bands == 3:
            centers = np.percentile(skin_px, [15.0, 55.0, 92.0])
        else:
            centers = np.percentile(skin_px, [25.0, 80.0])

        diffs = np.diff(centers)
        if diffs.min() < 4.0:
            return img_bgr

        edges = (centers[:-1] + centers[1:]) / 2.0

        q = np.full_like(L, centers[0].item())
        q += np.clip(L - centers[-1], 0.0, None)
        q -= np.clip(centers[0] - L, 0.0, None)

        for i in range(len(edges)):
            e_i = float(edges[i])
            c_delta = float(centers[i + 1] - centers[i])
            w_i = max(2.0, softness * (float(centers[i + 1]) - float(centers[i])))
            step = c_delta * np.clip(0.5 + 0.5 * np.tanh((L - e_i) / w_i), 0.0, 1.0)
            q += step

        protection = self._get_highlight_protection(lab)
        lab[:, :, 0] = L + (q - L) * s * protection * skin_mask
        if is_float:
            out = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, out, skin_mask)

    def unify_tone(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
        target_hue: float = -1.0,
        chroma_compress: float = 0.5,
    ) -> np.ndarray:
        """Hue and chroma unification for anime-style uniform skin tone.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100 unification intensity.
            target_hue: Target hue in degrees [0, 360). -1 = auto (chroma-weighted mean).
            chroma_compress: Chroma compression factor toward mean [0, 1].

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        s = strength / 100.0
        lch = bgr_to_lch(img_bgr)
        H = lch[:, :, 2]
        C = lch[:, :, 1]

        lch_skin = skin_mask_lch(lch)
        w = (skin_mask * lch_skin).astype(np.float32)
        w_sum = w.sum()

        if w_sum < 1e-6:
            return img_bgr

        if target_hue < 0.0:
            H_rad = np.deg2rad(H)
            sin_sum = (w * C * np.sin(H_rad)).sum()
            cos_sum = (w * C * np.cos(H_rad)).sum()
            t = float(np.rad2deg(np.arctan2(sin_sum, cos_sum))) % 360.0
        else:
            t = target_hue

        d = ((t - H + 180.0) % 360.0) - 180.0
        near = np.clip(1.0 - np.abs(d) / 60.0, 0.0, 1.0)

        H_new = (H + d * s * 0.8 * near) % 360.0

        c_mean = float((w * C).sum() / max(w_sum, 1e-6))
        C_new = C + (c_mean - C) * s * chroma_compress * near

        lch_out = lch.copy()
        lch_out[:, :, 2] = H_new
        lch_out[:, :, 1] = C_new

        out = lch_to_bgr(lch_out)
        return blend_masked(img_bgr, out, skin_mask * lch_skin)

    def unify_hue_line(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        hue_strength: int = 0,
        chroma_strength: int = 0,
        locus: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Hue-line skin tone unification via OKLCh constant-hue-line pull.

        Rotates hue toward a target (up to ±8°) and compresses chroma variance
        (up to ±0.04) while leaving luminance untouched. Protects high-chroma
        regions (makeup/tattoo), dark features (hair/shadow), and specular highlights.

        Supports both uint8 and float32 input. If input is float32 [0, 255], output is float32.
        If input is uint8, output is uint8.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            skin_mask: (H, W) float mask 0–1. May be None to skip processing.
            hue_strength: 0–100 hue rotation intensity.
            chroma_strength: 0–100 chroma variance compression intensity.
            locus: Optional target locus dict with 'h_target' and 'C_target' keys.
                If None, auto-determined from skin state.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if (hue_strength <= 0 and chroma_strength <= 0) or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        # Convert to OKLCh
        oklab = bgr_to_oklab(img_u8)
        oklch = oklab_to_oklch(oklab)

        # Measure skin state to pick locus class if not provided
        if locus is None:
            state = measure_skin_state(img_u8, skin_mask, thresh=0.3)
            L_mean = state.L_mean

            # Choose locus class by L_mean
            if L_mean >= SKIN_LOCI["fair"].get("L_min", 0.72):
                locus = SKIN_LOCI["fair"]
            elif L_mean >= SKIN_LOCI["tan"].get("L_min", 0.55):
                locus = SKIN_LOCI["tan"]
            else:
                locus = SKIN_LOCI["deep"]

        h_target = locus["h_target"]
        C_target = locus["C_target"]

        # Extract channels
        L = oklch[..., 0]
        C = oklch[..., 1]
        h = oklch[..., 2]

        # Eligibility mask: exclude high chroma (makeup/tattoo), very dark (hair/shadow), specular
        eligible = np.ones(img_bgr.shape[:2], dtype=np.float32)
        # Smoothstep gates instead of hard binary (prevents contour seams)
        eligible *= (1.0 - _smoothstep(0.0, 0.18, C))  # 1->0 as C goes 0->0.18 (high-chroma = ineligible)
        eligible *= _smoothstep(0.25, 0.50, L)       # 0->1 as L goes 0.25->0.50
        eligible *= (1.0 - _smoothstep(0.90, 0.95, L))  # 1->0 as L goes 0.90->0.95
        eligible *= skin_mask  # Within skin mask

        # Hue rotation: shortest arc to target, clipped to ±8°
        delta_h = (h_target - h + 180.0) % 360.0 - 180.0  # Shortest arc
        delta_h_clipped = np.clip(delta_h, -8.0, 8.0)
        h_new = (h + delta_h_clipped * (hue_strength / 100.0) * eligible) % 360.0

        # Chroma pull: toward target, clipped to ±0.04
        delta_C = np.clip(C_target - C, -0.04, 0.04)
        C_new = C + delta_C * (chroma_strength / 100.0) * eligible

        # Reconstruct OKLCh
        oklch_out = oklch.copy()
        oklch_out[..., 0] = L
        oklch_out[..., 1] = C_new
        oklch_out[..., 2] = h_new

        # Convert back to BGR
        oklab_out = oklch_to_oklab(oklch_out)
        out = oklab_to_bgr(oklab_out, float32_out=is_float)

        return blend_masked(img_bgr, out, skin_mask)

    def micro_dodge_burn(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
        face_width: float = 100.0,
    ) -> np.ndarray:
        """Auto micro dodge & burn — evening luminance blotches without blur.

        Applies a band-passed correction to the L channel that evens out
        luminance unevenness at the blotch frequency (between pores and shading).
        No blurring of the image; operates only on the band-passed signal.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1. May be None to skip processing.
            strength: 0–100 correction strength. 0 returns input unchanged.
            face_width: Face width in pixels for bandpass frequency scaling.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Band-pass the L channel at blotch scale
        dog = _blotch_bandpass(L, face_width)

        # Edge protection (preserve feature lines)
        edge_protect = _edge_protect(lab)

        # Highlight protection (preserve specular)
        highlight_protect = self._get_highlight_protection(lab)

        # Combined protection
        protection = edge_protect * highlight_protect

        # Correction: subtract band signal (negative DoG brightens, positive darkens)
        # scaled by strength and mask
        strength_factor = (strength / 100.0) * 0.85
        L_new = L - dog * strength_factor * skin_mask * protection

        # Clip to valid range
        lab[:, :, 0] = np.clip(L_new, 0, 255)

        # Convert back and blend
        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, skin_mask)

    def redness_even(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
        face_width: float = 100.0,
        lips_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        dog_a = _blotch_bandpass(a, face_width)
        dog_b = _blotch_bandpass(b, face_width)

        edge_protect = _edge_protect(lab)
        highlight_protect = self._get_highlight_protection(lab)
        protection = edge_protect * highlight_protect

        strength_factor = strength / 100.0
        correction_a = dog_a * strength_factor * skin_mask * protection
        correction_b = dog_b * strength_factor * 0.5 * skin_mask * protection

        if lips_mask is not None:
            correction_a *= (1.0 - lips_mask)
            correction_b *= (1.0 - lips_mask)

        lab[:, :, 1] = a - correction_a
        lab[:, :, 2] = b - correction_b

        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, skin_mask)

    def shine_removal(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 0,
        eyes_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Shine / oil removal via specular detection and local median reconstruction.

        Detects bright, low-chroma regions (shine), reconstructs their chroma from
        surrounding skin via guided filtering, and compresses their luminance toward
        a local non-shine median using exponential rolloff (adapted from soft_clip_highlights).
        Eye catchlights are excluded via eyes_mask to preserve their natural appearance.

        Supports both uint8 and float32 input. If input is float32 [0, 255], output is float32.
        If input is uint8, output is uint8.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            skin_mask: (H, W) float mask 0–1. May be None to skip processing.
            strength: 0–100 shine removal intensity. 0 returns input unchanged.
            eyes_mask: Optional (H, W) float mask marking eye regions to exclude.
                       Regions marked here will not be modified.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.

        Algorithm:
        1. Detect shine: pixels with L > adaptive_threshold AND low chroma (desaturated)
           within skin_mask. Feather the mask with Gaussian blur for soft edges.
        2. Exclude eyes: subtract dilated eyes_mask from shine mask to protect catchlights.
        3. Reconstruct chroma: guided-filter a/b channels to inpaint plausible chroma
           in shine regions, pulling from surrounding skin.
        4. Compress L: compute per-pixel targets as local non-shine L median (via
           guided filter), then apply soft exponential compression toward the target,
           scaled by strength. Over-removal guard ensures residual shine (≥30% prominence).
        5. Blend into skin_mask to avoid edge artifacts.
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h_img, w_img = lab.shape[:2]

        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # Compute chroma (C = sqrt((a-128)^2 + (b-128)^2)).
        # cv2's LAB convention offsets a/b by 128 (a neutral/gray pixel is
        # a=128, b=128), so the offset MUST be subtracted before computing
        # chroma — matching retouch/color_space.py's bgr_to_lab convention.
        # Using raw a/b here would give ~181 for a perfectly neutral pixel,
        # swamping real skin-vs-shine chroma differences (~10-40 units).
        chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)

        # --- Detect shine: L > adaptive_threshold AND low chroma ---
        # Adaptive threshold based on local context: median L in skin region
        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            median_L_skin = np.median(L[skin_indices])
            median_chroma_skin = np.median(chroma[skin_indices])
        else:
            return img_bgr

        # Shine threshold: adaptive, typically median_L + 20-30 L levels (oily shine usually 180-230 range)
        # This is relative to actual skin L, not a fixed value
        shine_L_threshold = median_L_skin + 20.0

        # Chroma threshold: use the median chroma of normal skin as reference
        # Shine is detected where chroma << normal skin chroma (e.g., < 40% of normal)
        # This accounts for the fact that normal skin has color and shine doesn't
        chroma_threshold_for_shine = max(5.0, median_chroma_skin * 0.4)

        # Luminance gate: soft ramp for pixels above shine_L_threshold
        L_gate = np.clip((L - shine_L_threshold) / 15.0, 0.0, 1.0)  # Smooth ramp over 15 L units

        # Chroma gate: soft ramp for pixels below chroma_threshold (low chroma = likely shine)
        # Gate is 1.0 when chroma is low, 0.0 when chroma is high
        chroma_gate = 1.0 - np.clip(chroma / chroma_threshold_for_shine, 0.0, 1.0)

        # Combine gates: both conditions must be met
        shine_mask = L_gate * chroma_gate * skin_mask

        # Feather shine mask with Gaussian blur for soft edges
        shine_mask = cv2.GaussianBlur(shine_mask, (11, 11), 0)
        shine_mask = np.clip(shine_mask, 0.0, 1.0)

        # --- Exclude eyes ---
        if eyes_mask is not None:
            # Dilate eyes_mask slightly to ensure catchlight protection
            eyes_mask_norm = normalize_mask(eyes_mask.astype(np.float32, copy=False) if eyes_mask.dtype != np.float32 else eyes_mask)
            eye_dilation_kernel = np.ones((5, 5), np.uint8)
            eyes_dilated = cv2.dilate(eyes_mask_norm, eye_dilation_kernel)
            shine_mask = shine_mask * (1.0 - eyes_dilated)

        # Early return if shine_mask is negligible
        if shine_mask.max() < 0.01:
            return img_bgr

        # --- Compute local non-shine median as per-pixel target ---
        # Compute a smooth baseline L by filtering only non-shine pixels
        # Use a large-radius Gaussian blur of L masked to non-shine regions
        non_shine_mask = 1.0 - shine_mask

        # Create masked version: set shine regions to a neutral value that won't affect blur
        L_masked = L * non_shine_mask + np.median(L[skin_indices]) * shine_mask

        # Blur the masked image to get a smooth local baseline
        # Large radius so the target is a local average of surrounding non-shine L
        L_target_smooth = cv2.GaussianBlur(L_masked, (31, 31), 0)

        # In shine regions, use the smoothed target; elsewhere use original L
        # This ensures shine pixels are pulled toward their local non-shine neighbors
        L_target = L * (1.0 - shine_mask) + L_target_smooth * shine_mask

        # --- Reconstruct chroma via guided filtering ---
        # Guided-filter a and b channels to inpaint plausible color in shine regions
        # Use L as the guide so the filter preserves luminance structure
        a_inpainted = guided_filter(a, radius=30, eps=100.0, guide=None)
        b_inpainted = guided_filter(b, radius=30, eps=100.0, guide=None)

        # Blend inpainted chroma into shine regions
        a_new = a * (1.0 - shine_mask) + a_inpainted * shine_mask
        b_new = b * (1.0 - shine_mask) + b_inpainted * shine_mask

        # --- Compress L toward local target using soft exponential rolloff ---
        # Adapted from soft_clip_highlights: instead of compressing toward a fixed
        # ceiling, we compress each pixel toward its per-pixel target.
        # Formula: out = current + (target - current) * (1 - exp(-k * excess))
        # where excess = max(current - target, 0) normalized.

        s = strength / 100.0

        # For pixels where current L > target L, apply soft compression downward
        L_excess = np.maximum(L - L_target, 0.0)
        L_excess_max = np.max(L_excess) if np.max(L_excess) > 0 else 1.0
        normalized_excess = L_excess / (L_excess_max + 1e-6)

        # Exponential approach: compress down by a fraction dependent on normalized excess
        # k chosen so the curve is smooth and the reduction tops out around 70% at strength=100
        k_rolloff = 1.5  # Controls curvature; higher = more aggressive compression
        compression_factor = 1.0 - np.exp(-k_rolloff * normalized_excess)

        # Over-removal guard: never reduce more than ~70% of the way to target at full strength
        max_reduction = 0.70
        compression_factor = np.clip(compression_factor * s, 0.0, max_reduction)

        # Apply compression only in shine regions
        L_new = L - L_excess * compression_factor * shine_mask

        # --- Assemble result ---
        lab[:, :, 0] = np.clip(L_new, 0, 255)
        lab[:, :, 1] = a_new
        lab[:, :, 2] = b_new

        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, skin_mask)

    def texture_transplant(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: float = 0,
        face_width: float = 100.0,
    ) -> np.ndarray:
        """Texture transplant: clone high-frequency texture from clean skin to over-smoothed zones.

        Implements same-face texture cloning by:
        1. Finding the cleanest (lowest-blotch) skin region via _blotch_bandpass variance
        2. Extracting tileable high-band texture from the donor patch
        3. Detecting texture-poor (over-smoothed/inpainted) zones via local high-band energy
        4. Blending donor texture into target zones with rotation jitter and luminance matching

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1. May be None to skip processing.
            strength: 0–100 texture transplant intensity. 0 returns input unchanged.
            face_width: Face width in pixels. Used to scale patch sizes and frequency bands.

        Returns:
            (H, W, 3) uint8 BGR image.

        Algorithm overview:
        - Early return if strength <= 0 or skin_mask is None
        - Compute blotch-band metric within skin regions via _blotch_bandpass
        - Divide skin-masked area into candidate patches, score by local blotch variance
        - Pick lowest-variance patch as donor (cleanest skin, usually cheeks)
        - If insufficient candidate patches exist, return input unchanged (no clean donor found)
        - Extract high-band residual from donor patch (L - gaussian_blur(L, sigma≈2))
        - Apply soft radial falloff to donor patch edges for tileable seaming
        - Detect target zones: regions where local high-band energy < threshold (smoothed/inpainted)
        - For each target zone, blend donor texture with:
          * Random rotation jitter (±15-45°, seeded by position for reproducibility)
          * Luminance rescaling to match target zone's local contrast
          * Soft masking to avoid harsh edges
        - Final blend into skin_mask via blend_masked
        - Donor region itself remains unmodified (read-only)
        """
        if strength <= 0 or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = strength / 100.0
        h_img, w_img = img_bgr.shape[:2]

        # Convert to LAB for processing
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # --- Step 1: Compute blotch-band signal (for donor region detection) ---
        blotch_band = _blotch_bandpass(L, face_width)

        # --- Step 2: Find cleanest donor patch ---
        skin_indices = skin_mask > 0.3
        if np.sum(skin_indices) < 100:
            return img_bgr  # Insufficient skin area

        # Divide skin-masked region into candidate patches
        patch_size = max(int(face_width * 0.10), 20)  # 10% of face width, min 20px
        grid_step = patch_size // 2  # 50% overlap for smoother scoring

        # Precompute pore-scale high-band residual for the whole image so we can
        # reject "clean by blotch metric but actually textureless" patches (bug
        # found in review 2026-07-03: a perfectly flat patch has zero variance at
        # EVERY frequency, including the blotch band, so it was scoring as the
        # "cleanest" donor while having literally nothing to transplant — donor_
        # highband std was 0.0). A valid donor must have real pore-scale texture,
        # not just low blotchiness.
        L_blurred_full = cv2.GaussianBlur(L, (0, 0), 2.0)
        highband_full = L - L_blurred_full

        candidate_patches = []
        for y in range(0, h_img - patch_size, grid_step):
            for x in range(0, w_img - patch_size, grid_step):
                patch_skin_mask = skin_mask[y:y + patch_size, x:x + patch_size]
                if np.mean(patch_skin_mask) < 0.5:
                    continue  # Skip patches that aren't mostly skin

                patch_blotch = blotch_band[y:y + patch_size, x:x + patch_size]
                patch_blotch_masked = patch_blotch * patch_skin_mask
                blotch_variance = np.std(patch_blotch_masked[patch_skin_mask > 0.3])

                # Donor-quality guard: the patch must actually contain pore-scale
                # high-band texture (std > a small floor) to be worth cloning from.
                patch_highband = highband_full[y:y + patch_size, x:x + patch_size]
                patch_highband_energy = np.std(patch_highband[patch_skin_mask > 0.3])

                if not np.isnan(blotch_variance) and patch_highband_energy > 0.5:
                    candidate_patches.append({
                        'x': x, 'y': y,
                        'variance': blotch_variance,
                        'highband_energy': patch_highband_energy,
                    })

        if len(candidate_patches) < 3:
            return img_bgr  # Not enough textured candidate patches to find a clean donor

        # Sort by blotch variance (ascending) and pick the cleanest *textured* patch
        candidate_patches.sort(key=lambda p: p['variance'])
        donor_patch = candidate_patches[0]
        donor_x, donor_y = donor_patch['x'], donor_patch['y']

        # --- Step 3: Extract tileable high-band texture from donor ---
        donor_L = L[donor_y:donor_y + patch_size, donor_x:donor_x + patch_size]
        donor_skin = skin_mask[donor_y:donor_y + patch_size, donor_x:donor_x + patch_size]

        # High-band residual (pore-scale texture)
        donor_L_blurred = cv2.GaussianBlur(donor_L, (0, 0), 2.0)
        donor_highband = donor_L - donor_L_blurred

        # Apply soft radial falloff to patch edges for seamless tiling
        # Create a radial gradient mask: 1.0 at center, 0.0 at edges
        center_y, center_x = patch_size / 2.0, patch_size / 2.0
        yy, xx = np.ogrid[:patch_size, :patch_size]
        dist_to_center = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
        radial_falloff = 1.0 - np.clip(dist_to_center / (max_dist * 0.8), 0.0, 1.0)
        radial_falloff = np.power(radial_falloff, 2.0)  # Smooth curve

        donor_highband = donor_highband * radial_falloff

        # --- Step 4: Detect texture-poor target zones ---
        # Compute high-band residual for the entire image
        L_blurred = cv2.GaussianBlur(L, (0, 0), 2.0)
        L_highband = L - L_blurred

        # Compute local high-band energy via standard deviation in sliding windows
        window_size = max(int(face_width * 0.08), 16)
        window_size = window_size if window_size % 2 == 1 else window_size + 1  # Ensure odd

        # Use moving window via morphological approach: compute std in local regions
        # Simple approximation: use highband squared as proxy for energy
        L_highband_sq = L_highband ** 2
        # Low-pass filter to get local energy estimate
        local_energy_approx = cv2.GaussianBlur(L_highband_sq, (window_size, window_size), 0)
        local_highband_energy = np.sqrt(np.clip(local_energy_approx, 0, None))

        # Target zones: low energy relative to face median energy
        valid_energy = local_highband_energy[skin_indices]
        median_energy = np.median(valid_energy) if len(valid_energy) > 0 else 1.0
        energy_threshold = median_energy * 0.4  # Zones with <40% of median energy

        target_mask = np.logical_and(
            local_highband_energy < energy_threshold,
            skin_mask > 0.3
        ).astype(np.float32)

        if np.sum(target_mask) < 50:
            return img_bgr  # Insufficient target area

        # Feather target mask for smooth blending
        target_mask = cv2.GaussianBlur(target_mask, (11, 11), 0)
        target_mask = np.clip(target_mask, 0.0, 1.0)

        # --- Step 5: Blend transplanted texture with rotation jitter and luminance matching ---
        # Place donor texture tiles across target regions
        result_L = L.copy()

        # Deterministic seeding based on position for reproducibility
        random_seed_base = int(donor_x + donor_y * w_img)

        for tile_y in range(0, h_img, patch_size):
            for tile_x in range(0, w_img, patch_size):
                tile_target_mask = target_mask[
                    tile_y:min(tile_y + patch_size, h_img),
                    tile_x:min(tile_x + patch_size, w_img)
                ]

                if np.max(tile_target_mask) < 0.01:
                    continue

                # Deterministic random seed from tile position
                np.random.seed((random_seed_base + tile_x + tile_y * 1000) % (2 ** 31))

                # Random rotation jitter (±15 to ±45 degrees)
                angle = np.random.uniform(-45, 45)
                rotation_matrix = cv2.getRotationMatrix2D(
                    (patch_size / 2, patch_size / 2), angle, 1.0
                )
                donor_highband_rotated = cv2.warpAffine(
                    donor_highband,
                    rotation_matrix,
                    (patch_size, patch_size),
                    borderMode=cv2.BORDER_REFLECT,
                )

                # Tile the rotated donor texture to cover the target area
                tile_h = min(patch_size, h_img - tile_y)
                tile_w = min(patch_size, w_img - tile_x)

                # Crop rotated donor to match tile dimensions
                donor_tile = donor_highband_rotated[:tile_h, :tile_w]
                target_tile_mask = tile_target_mask[:tile_h, :tile_w]

                if np.max(target_tile_mask) < 0.01:
                    continue

                # Luminance matching: rescale donor texture to match the face's own
                # HEALTHY high-band energy level (median_energy, computed in Step 4
                # from the whole skin region), NOT the target zone's own pre-transplant
                # energy. The target zone was selected specifically because its energy
                # is far below median (< 40% of median_energy, see energy_threshold
                # above) — scaling toward the target's own near-zero energy would be
                # circular and self-defeating (on a perfectly flat target, energy_scale
                # would collapse to ~0, making the transplant a no-op on exactly the
                # zones it exists to repair). Bug caught in review 2026-07-03: the
                # original implementation measured tile_L_energy from the target zone
                # itself, producing a 0/0-ish ratio on flat regions. Fixed by reusing
                # median_energy as the target amplitude instead.
                tile_L = result_L[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]

                donor_tile_energy = np.std(donor_tile[target_tile_mask > 0.1])
                if donor_tile_energy > 0.1:
                    # Scale donor texture to match healthy face-wide texture amplitude
                    energy_scale = median_energy / (donor_tile_energy + 1e-6)
                    donor_tile = donor_tile * energy_scale

                # Blend into result with target mask
                blend_factor = target_tile_mask * s
                result_L[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = np.clip(
                    tile_L + donor_tile * blend_factor,
                    0.0, 255.0
                )

        # --- Step 6: Assemble result and verify donor region unchanged ---
        lab[:, :, 0] = result_L

        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab, 0, 255))
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, skin_mask)

    @staticmethod
    def _get_highlight_protection(lab: np.ndarray) -> np.ndarray:
        """Linearly decays adjustments for bright pixels (L > 220) to prevent specular clipping.

        Args:
            lab: (H, W, 3) float32 LAB image.

        Returns:
            (H, W) float32 protection mask in [0, 1].
        """
        l_val = lab[:, :, 0]
        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        return protection

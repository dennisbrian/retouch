"""C5 — Skin-anchored color harmonization.

After skin correction (C1) the corrected skin tone becomes a stable anchor.
This module shifts the *background* colors to complement that anchor — a
split-tone-like grade applied only outside the person, so the subject reads
as the hero of the frame without the background fighting it.

Design notes
------------
* All pixel arithmetic is float32. uint8 inputs are converted at the
  boundary and converted back on output (dtype preserved).
* Skin tone is measured in OKLCh (perceptually uniform, matches the
  C1 locus system in ``color_science.py``). The background grade is
  applied in LCH (LAB-cylindrical) so the hue/chroma moves are perceptually
  uniform and the L* key for split-toning is the same one
  ``color_space.split_tone_lch`` uses.
* Complementarity is computed on the hue wheel: the background is nudged
  toward the hue that is ``~150°`` from the skin hue (split-complementary,
  softer than a hard 180° complement). An analogous bias term can pull a
  fraction of the shift toward an adjacent hue for a warmer/cool palette.
* The grade is a masked split-tone (shadow tint + highlight tint) plus a
  gentle chroma compression on the background so the subject's skin
  chroma reads as the chroma anchor. All moves are masked by
  ``person_mask`` so the subject (skin, hair, costume) is never touched.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .color_science import (
    SkinState,
    bgr_to_oklab,
    measure_skin_state,
    oklab_to_oklch,
)
from .color_space import (
    bgr_f32_to_lch_f32,
    lch_f32_to_bgr_f32,
)
from .utils import blend_masked, normalize_mask, squeeze_mask

logger = logging.getLogger(__name__)


# Conversion constant: OKLCh hue is in degrees [0, 360), same wheel
# orientation as LAB-LCH hue (both derive hue from atan2(b, a)). A
# complement computed from the OKLCh skin hue is therefore directly
# comparable to the LCH background hue used for the split-tone grade.
_HUE_WHEEL = 360.0


def _circular_distance(a: float, b: float) -> float:
    """Shortest unsigned angular distance between two hues in degrees."""
    d = abs(a - b) % _HUE_WHEEL
    return min(d, _HUE_WHEEL - d)


def _hue_target(skin_hue: float, mode: str) -> float:
    """Return the background target hue for a given harmony ``mode``.

    Args:
        skin_hue: OKLCh hue of the corrected skin anchor, in degrees.
        mode: One of ``"complementary"``, ``"split"``, ``"analogous_warm"``,
            ``"analogous_cool"``.

    Returns:
        Target hue in degrees [0, 360).
    """
    if mode == "complementary":
        return (skin_hue + 180.0) % _HUE_WHEEL
    if mode == "split":
        # Split-complementary: ±150° from skin — softer than 180°.
        return (skin_hue + 150.0) % _HUE_WHEEL
    if mode == "analogous_warm":
        return (skin_hue + 30.0) % _HUE_WHEEL
    if mode == "analogous_cool":
        return (skin_hue - 30.0) % _HUE_WHEEL
    # Unknown mode → fall back to split-complementary (the default).
    logger.warning("Unknown harmony mode %r, falling back to 'split'", mode)
    return (skin_hue + 150.0) % _HUE_WHEEL


class BackgroundHarmonizer:
    """Shift background colors to complement a corrected skin-tone anchor.

    The harmonizer is a self-contained stage: given the corrected image, a
    skin mask (the C1 anchor region) and a person mask (the subject to
    protect), it derives a split-tone grade for the background and applies
    it through a feathered inverse-person mask.

    dtype-aware: accepts uint8 BGR or float32 BGR [0, 255] and returns the
    same dtype. The float32 path stays float-native end-to-end (no uint8
    round-trip) via the ``bgr_f32_to_lch_f32`` / ``lch_f32_to_bgr_f32``
    helpers, matching the engine's E1 float convention.
    """

    def __init__(self) -> None:
        # Stateless instance — kept as a class so the engine can hold one
        # alongside the other stage processors (SkinProcessor, ColorGrader).
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harmonize(
        self,
        img: np.ndarray,
        skin_mask: Optional[np.ndarray],
        person_mask: Optional[np.ndarray],
        strength: float = 0.5,
        mode: str = "split",
        chroma_compress: float = 0.25,
        shadow_tint: float = 0.35,
        highlight_tint: float = 0.25,
    ) -> np.ndarray:
        """Adjust background colors to complement the corrected skin tone.

        Args:
            img: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            skin_mask: (H, W) float mask [0, 1] of the C1-corrected skin
                region. Used only to *measure* the skin-tone anchor; it is
                never graded. If None or empty, the image is returned
                unchanged (no anchor → no harmonization).
            person_mask: (H, W) float mask [0, 1] of the whole subject
                (skin + hair + clothing). The grade is applied only where
                this mask is low. If None, the whole image is treated as
                background (rarely what you want — a warning is logged).
            strength: Overall effect strength in [0, 1]. 0 = no-op, 1 =
                full split-tone + chroma compression on the background.
            mode: Harmony mode — ``"complementary"``, ``"split"`` (default),
                ``"analogous_warm"``, ``"analogous_cool"``.
            chroma_compress: Fraction [0, 1] by which background chroma is
                pulled toward the skin chroma anchor. 0 = leave background
                chroma alone; 1 = force background chroma to the skin
                chroma. Default 0.25 gives a subtle "subject pops" effect.
            shadow_tint: Strength [0, 1] of the complementary tint applied
                to background shadows (L* < 50).
            highlight_tint: Strength [0, 1] of the complementary tint
                applied to background highlights (L* > 50).

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        if strength <= 0.0:
            return img
        if skin_mask is None:
            return img

        is_float = img.dtype == np.float32
        if not is_float and img.dtype != np.uint8:
            # Only uint8 and float32 are supported input dtypes.
            raise TypeError(
                f"BackgroundHarmonizer.harmonize expects uint8 or float32 BGR, "
                f"got {img.dtype}"
            )

        # --- Build a normalized background mask (1 outside person, 0 inside) ---
        bg_mask = self._build_background_mask(person_mask, img.shape[:2])
        if bg_mask is None:
            return img

        # --- Measure the skin-tone anchor from the corrected skin region ---
        skin_state = measure_skin_state(
            img if not is_float else np.clip(img, 0, 255).astype(np.uint8),
            skin_mask,
            thresh=0.3,
        )
        # measure_skin_state returns a neutral fallback when no skin is found;
        # only harmonize when there is a real anchor.
        if skin_mask.max() <= 0.3:
            return img

        anchor_hue = float(skin_state.h_mean)
        anchor_chroma = float(skin_state.C_mean)
        target_hue = _hue_target(anchor_hue, mode)

        # --- Convert to LCH for the background grade (float-native) ---
        if is_float:
            img_255 = np.clip(img, 0.0, 255.0).astype(np.float32)
        else:
            img_255 = img.astype(np.float32)

        lch = bgr_f32_to_lch_f32(img_255)
        lch = self._apply_background_grade(
            lch=lch,
            bg_mask=bg_mask,
            target_hue=target_hue,
            anchor_chroma=anchor_chroma,
            strength=float(strength),
            chroma_compress=float(chroma_compress),
            shadow_tint=float(shadow_tint),
            highlight_tint=float(highlight_tint),
        )

        graded_255 = lch_f32_to_bgr_f32(lch)

        if is_float:
            out = np.clip(graded_255, 0.0, 255.0).astype(np.float32)
        else:
            out = np.clip(graded_255, 0.0, 255.0).astype(np.uint8)

        # Blend the graded result onto the original through the background
        # mask. blend_masked works in [0, 255] for both dtypes and preserves
        # the input dtype.
        return blend_masked(img, out, bg_mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_background_mask(
        self,
        person_mask: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Return a float32 [0, 1] background mask (1 = background).

        Feathers the person-mask boundary so the grade does not produce a
        hard seam at the subject edge. Returns None when there is no
        subject at all (degenerate — caller should no-op).
        """
        h, w = shape
        if person_mask is None:
            logger.warning(
                "BackgroundHarmonizer: person_mask is None — grading the "
                "whole image as background. This is rarely intended."
            )
            return np.ones((h, w), dtype=np.float32)

        pm = person_mask.astype(np.float32, copy=False)
        pm = squeeze_mask(pm)
        if pm.max() > 1.0:
            pm = pm / 255.0
        pm = normalize_mask(pm)
        if pm is None or float(pm.max()) < 1e-3:
            # No detectable subject → no background to grade distinctly.
            return None

        # Feather the person mask so the inverse (background) mask has a
        # soft edge. Kernel scales with image size (never hardcoded).
        feather = max(3, int(min(h, w) * 0.015)) | 1
        pm_blur = cv2.GaussianBlur(pm, (feather, feather), 0)
        bg = np.clip(1.0 - pm_blur, 0.0, 1.0).astype(np.float32)
        return bg

    def _apply_background_grade(
        self,
        lch: np.ndarray,
        bg_mask: np.ndarray,
        target_hue: float,
        anchor_chroma: float,
        strength: float,
        chroma_compress: float,
        shadow_tint: float,
        highlight_tint: float,
    ) -> np.ndarray:
        """Apply the split-tone + chroma-compression grade to background LCH.

        All operations are vectorized NumPy on float32 arrays. The
        background mask gates every move so the subject is untouched.

        Args:
            lch: (H, W, 3) float32 LCH (L in [0,100], C in [0,~180], H in [0,360)).
            bg_mask: (H, W) float32 [0,1] background mask.
            target_hue: Complementary hue to push background toward, degrees.
            anchor_chroma: Skin anchor chroma (OKLCh C, ~0.05–0.15).
            strength: Overall effect strength [0, 1].
            chroma_compress: Background chroma pull fraction [0, 1].
            shadow_tint: Shadow tint strength [0, 1].
            highlight_tint: Highlight tint strength [0, 1].

        Returns:
            New (H, W, 3) float32 LCH.
        """
        out = lch.copy()
        L = out[:, :, 0]
        C = out[:, :, 1]
        H = out[:, :, 2]

        m3 = bg_mask[:, :, np.newaxis]
        m2 = bg_mask

        # --- 1. Split-tone: tint shadows and highlights toward target_hue ---
        # L*-keyed weights, matching color_space.split_tone_lch convention.
        l_norm = L / 100.0
        shadow_w = np.clip(0.5 - l_norm, 0.0, 1.0) * 2.0  # 1 at L=0, 0 at L=50
        highlight_w = np.clip(l_norm - 0.5, 0.0, 1.0) * 2.0  # 0 at L=50, 1 at L=100

        # Combined tint weight: shadow_tint in shadows, highlight_tint in
        # highlights, both gated by strength and the background mask.
        tint_s = shadow_w * shadow_tint * strength * m2
        tint_h = highlight_w * highlight_tint * strength * m2

        # Rotate hue toward target_hue proportional to the tint weight.
        # Shortest-arc rotation.
        d_hue = ((target_hue - H + 180.0) % 360.0) - 180.0
        h_shift = d_hue * (tint_s + tint_h)
        H_new = (H + h_shift) % 360.0

        # Add a small chroma lift in the tinted regions so the tint is
        # actually visible (a pure hue rotation on a near-grey pixel is
        # invisible). Lift is anchored to a modest chroma so we don't
        # over-saturate the background.
        tint_chroma_target = max(anchor_chroma * 200.0, 12.0)  # LCH C scale
        chroma_lift = tint_chroma_target * (tint_s + tint_h)
        C_new = C + chroma_lift

        # --- 2. Chroma compression: pull background chroma toward the anchor ---
        if chroma_compress > 0.0:
            # Map OKLCh anchor chroma (~0.05–0.15) to the LCH C scale
            # (~10–30) so the pull target is comparable to background C.
            anchor_c_lch = max(anchor_chroma * 200.0, 8.0)
            pull = (anchor_c_lch - C) * chroma_compress * strength * m2
            C_new = C_new + pull

        out[:, :, 1] = np.clip(C_new, 0.0, 200.0)
        out[:, :, 2] = H_new
        return out

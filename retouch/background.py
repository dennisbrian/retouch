"""T1 — Background replace & scene relight.

Provides the ``BackgroundReplacer`` stage: a self-contained set of
subject/background compositing operations that all share a feathered
person-mask boundary for seamless integration.  This module finally
wires the 7 historically-dead ``anime_crystal_void`` recipe keys
(``background_blur``, ``background_desaturation``, ``light_wrap``,
``blue_shadow_grade``, ``cyan_midtone_grade``, ``subject_sharpen``,
``matte_black``) — they had been aspirational no-ops silently ignored by
the engine since the recipe was first authored.

Design notes
------------
* All pixel arithmetic is float32. uint8 inputs are converted at the
  boundary and converted back on output (dtype preserved).  float32
  inputs in [0, 255] stay float-native end-to-end (E1 convention).
* Subject protection is absolute: every operation that touches the
  background is gated by a *feathered* inverse person mask so the
  subject (skin, hair, costume) is never altered, and the seam is
  invisible.  ``subject_sharpen`` is the one operation that targets the
  *subject* — it uses the positive person mask.
* Light direction for ``relight_scene`` is specified in spherical
  coordinates (azimuth/elevation in degrees) and applied as a
  Blinn-Phong-style shading term derived from the image's own luminance
  gradient — a landmark-free approximation that works on any subject,
  not just faces (matching the ``body_relight.py`` precedent).
* Colour grades (``blue_shadow_grade``, ``cyan_midtone_grade``,
  ``matte_black``) operate in LCH so hue/chroma moves are perceptually
  uniform and the L*-keyed tonal masks match the
  ``color_space.split_tone_lch`` convention used by the harmonizer.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .color_space import (
    bgr_f32_to_lch_f32,
    lch_f32_to_bgr_f32,
)
from .utils import blend_masked, normalize_mask, squeeze_mask

logger = logging.getLogger(__name__)


_HUE_WHEEL = 360.0


def _circular_signed(a: float, b: float) -> float:
    """Shortest signed angular delta from *a* to *b* in degrees, in (-180, 180]."""
    return ((b - a + 180.0) % _HUE_WHEEL) - 180.0


class BackgroundReplacer:
    """Subject/background compositing stage (T1).

    A stateless stage processor that holds alongside
    ``BackgroundHarmonizer`` in the engine.  Every public method is
    dtype-aware: accepts uint8 BGR or float32 BGR [0, 255] and returns
    the same dtype.  The float32 path stays float-native end-to-end.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _feathered_person_mask(
        self,
        person_mask: Optional[np.ndarray],
        shape: Tuple[int, int],
        feather_frac: float = 0.015,
    ) -> np.ndarray:
        """Return a float32 [0, 1] feathered person mask.

        ``person_mask`` may be uint8/float, 2D or (H, W, 1).  An all-zero
        mask yields an all-zero return (caller decides what that means).
        The feather radius scales with image size — never hardcoded.
        """
        h, w = shape
        if person_mask is None:
            logger.warning(
                "BackgroundReplacer: person_mask is None — treating the "
                "whole image as subject. This is rarely intended."
            )
            return np.ones((h, w), dtype=np.float32)

        pm = person_mask.astype(np.float32, copy=False)
        pm = squeeze_mask(pm)
        if pm.max() > 1.0:
            pm = pm / 255.0
        pm = normalize_mask(pm)
        if pm is None:
            return np.zeros((h, w), dtype=np.float32)
        feather = max(3, int(min(h, w) * feather_frac)) | 1
        return cv2.GaussianBlur(pm, (feather, feather), 0).astype(np.float32)

    def _background_mask(
        self,
        person_mask: Optional[np.ndarray],
        shape: Tuple[int, int],
        feather_frac: float = 0.015,
    ) -> np.ndarray:
        """Return a float32 [0, 1] background mask (1 = background)."""
        pm = self._feathered_person_mask(person_mask, shape, feather_frac)
        return np.clip(1.0 - pm, 0.0, 1.0).astype(np.float32)

    def _to_f32_255(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Return (img_as_float32_[0,255], was_float_input)."""
        is_float = img.dtype == np.float32
        if not is_float and img.dtype != np.uint8:
            raise TypeError(
                f"BackgroundReplacer expects uint8 or float32 BGR, got {img.dtype}"
            )
        if is_float:
            return np.clip(img, 0.0, 255.0).astype(np.float32), True
        return img.astype(np.float32), False

    def _restore_dtype(self, img_f32_255: np.ndarray, was_float: bool) -> np.ndarray:
        if was_float:
            return np.clip(img_f32_255, 0.0, 255.0).astype(np.float32)
        return np.clip(img_f32_255, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replace(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        background_img: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Composite the subject over a new background image.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255] source.
            person_mask: (H, W) float mask [0, 1] of the subject.
            background_img: (H, W, 3) uint8 or float32 BGR [0, 255]
                replacement background.  Resized to match ``img`` if the
                shapes differ.
            strength: 0.0 = no-op (return original), 1.0 = full replace.
                Intermediate values blend the new background in through
                the background mask.

        Returns:
            (H, W, 3) image, same dtype as ``img``.
        """
        if strength <= 0.0:
            return img

        h, w = img.shape[:2]
        bg = background_img
        if bg.shape[:2] != (h, w):
            interp = cv2.INTER_AREA if bg.shape[0] * bg.shape[1] > h * w else cv2.INTER_CUBIC
            bg = cv2.resize(bg, (w, h), interpolation=interp)

        src_f, src_was_float = self._to_f32_255(img)
        bg_f, bg_was_float = self._to_f32_255(bg)
        bg_f = bg_f.astype(np.float32, copy=False)

        bg_mask = self._background_mask(person_mask, (h, w))
        # strength gates the background swap; the subject is always kept.
        m = (bg_mask * float(strength))[:, :, np.newaxis]
        out = src_f * (1.0 - m) + bg_f * m
        return self._restore_dtype(out, src_was_float)

    def relight_scene(
        self,
        img: np.ndarray,
        person_mask: Optional[np.ndarray],
        azimuth: float,
        elevation: float,
        strength: float,
    ) -> np.ndarray:
        """Relight the whole scene with a new directional light.

        A landmark-free Blinn-Phong-style shading term is derived from
        the image's own luminance gradient (Sobel on the L channel),
        which approximates surface orientation without a face mesh —
        matching the ``body_relight.py`` precedent so this works on any
        subject, not just faces.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255].
            person_mask: Optional (H, W) float mask [0, 1].  When None,
                the whole image is relit.
            azimuth: Light azimuth in degrees (0 = right, 90 = top,
                180 = left, 270 = bottom).
            elevation: Light elevation in degrees (0 = horizon, 90 =
                overhead).  Clamped to [0, 90].
            strength: 0–100.  0 = no-op, 100 = full relight.

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        if strength <= 0.0:
            return img

        src_f, was_float = self._to_f32_255(img)
        h, w = src_f.shape[:2]

        # We only need L for the gradient; use the LAB helper directly.
        from .utils import bgr_f32_to_lab_f32  # local import avoids cycle in docs

        lab_img = bgr_f32_to_lab_f32(src_f)
        L = lab_img[:, :, 0]  # [0, 255] uint8-scale

        # Surface-normal proxy from the luminance gradient.
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)

        az = np.deg2rad(azimuth)
        el = np.deg2rad(max(0.0, min(90.0, elevation)))
        # Light direction in image space (x right, y down, z toward viewer).
        lx = np.cos(el) * np.cos(az)
        ly = np.cos(el) * np.sin(az)
        lz = np.sin(el)

        # Gradient → pseudo-normal (flip sign so lit side brightens).
        nz = np.full_like(L, 1.0, dtype=np.float32)
        nx = -gx
        ny = -gy
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx /= nlen
        ny /= nlen
        nz /= nlen

        # Lambertian + small specular (Blinn-Phong with viewer = +z).
        diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        half_z = lz + 1.0
        spec_factor = 1.0
        spec = np.clip(nz * (half_z / (np.sqrt(lx * lx + ly * ly + half_z * half_z) + 1e-6)), 0.0, 1.0) ** 32.0 * spec_factor

        # Shading gain centred at 1.0 (neutral at diffuse=1).
        s = float(strength) / 100.0
        shading = (0.5 + 0.5 * diffuse) + 0.25 * spec
        shading = 1.0 + (shading - 1.0) * s
        shading = np.clip(shading, 0.2, 2.5).astype(np.float32)

        out = src_f * shading[:, :, np.newaxis]

        # Gate by person mask if provided (relight subject + background together,
        # but a mask lets the caller restrict the effect to one region).
        if person_mask is not None:
            m = self._feathered_person_mask(person_mask, (h, w))
            out = src_f * (1.0 - m[:, :, np.newaxis]) + out * m[:, :, np.newaxis]

        return self._restore_dtype(out, was_float)

    def blur_background(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Apply bokeh / background blur, subject stays sharp.

        Uses a large Gaussian blur on the background, then composites
        through the feathered background mask.  The radius scales with
        image size; ``radius`` is a 0–100 strength mapping to a
        fraction of the smaller image dimension.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255].
            person_mask: (H, W) float mask [0, 1] of the subject.
            radius: 0–100.  0 = no-op, 100 = strong bokeh (~4% of min
                dimension blur sigma).

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        if radius <= 0.0:
            return img

        src_f, was_float = self._to_f32_255(img)
        h, w = src_f.shape[:2]
        min_dim = float(min(h, w))

        # Map 0–100 to a blur sigma in [0, min_dim*0.04].
        sigma = (float(radius) / 100.0) * (min_dim * 0.04)
        ksize = max(3, int(sigma * 3.0)) | 1

        blurred = cv2.GaussianBlur(src_f, (ksize, ksize), sigma)
        bg_mask = self._background_mask(person_mask, (h, w))
        out = blend_masked(src_f, blurred, bg_mask)
        return self._restore_dtype(out, was_float)

    def grade_background(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """Apply a colour grade to the background only.

        ``params`` is a dict of optional keys (any subset may be
        present; absent keys are no-ops):

        * ``desaturation`` (0–100): pull background chroma toward 0.
        * ``blue_shadow_grade`` (0–100): rotate shadow hue toward blue
          (~240°) and lift shadow chroma.
        * ``cyan_midtone_grade`` (0–100): rotate midtone hue toward
          cyan (~180°) and lift midtone chroma.
        * ``matte_black`` (0–100): crush background blacks for a matte
          look (lifts the toe so shadows go flat-grey instead of
          crushed-black).

        All moves are LCH-based, masked by the feathered background
        mask, and subject-safe.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255].
            person_mask: (H, W) float mask [0, 1] of the subject.
            params: grade parameter dict (see above).

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        # No-op if every grade key is zero/absent.
        if not any(float(params.get(k, 0.0)) > 0.0 for k in (
            "desaturation", "blue_shadow_grade", "cyan_midtone_grade", "matte_black"
        )):
            return img

        src_f, was_float = self._to_f32_255(img)
        h, w = src_f.shape[:2]
        bg_mask = self._background_mask(person_mask, (h, w))

        lch = bgr_f32_to_lch_f32(src_f)
        out_lch = lch.copy()
        L = out_lch[:, :, 0]
        C = out_lch[:, :, 1]
        H = out_lch[:, :, 2]
        m2 = bg_mask  # (H, W)

        # --- Desaturation ---
        desat = float(params.get("desaturation", 0.0))
        if desat > 0.0:
            d = (desat / 100.0) * m2
            C = C * (1.0 - d)

        # --- Blue shadow grade ---
        bsg = float(params.get("blue_shadow_grade", 0.0))
        if bsg > 0.0:
            l_norm = np.clip(L / 100.0, 0.0, 1.0)
            shadow_w = np.clip(0.5 - l_norm, 0.0, 1.0) * 2.0  # 1 at L=0
            w_s = shadow_w * (bsg / 100.0) * m2
            d_hue = _circular_signed(0.0, 240.0)  # rotate toward blue
            H = (H + d_hue * w_s) % _HUE_WHEEL
            C = C + 12.0 * w_s

        # --- Cyan midtone grade ---
        cmg = float(params.get("cyan_midtone_grade", 0.0))
        if cmg > 0.0:
            l_norm = np.clip(L / 100.0, 0.0, 1.0)
            mid_w = 1.0 - np.abs(l_norm - 0.5) * 2.0  # 1 at L=50, 0 at 0/100
            mid_w = np.clip(mid_w, 0.0, 1.0)
            w_m = mid_w * (cmg / 100.0) * m2
            d_hue = _circular_signed(0.0, 180.0)  # rotate toward cyan
            H = (H + d_hue * w_m) % _HUE_WHEEL
            C = C + 10.0 * w_m

        # --- Matte black: crush blacks but lift the toe for a flat matte look ---
        mb = float(params.get("matte_black", 0.0))
        if mb > 0.0:
            l_norm = np.clip(L / 100.0, 0.0, 1.0)
            shadow_w = np.clip(0.5 - l_norm, 0.0, 1.0) * 2.0
            w_mb = shadow_w * (mb / 100.0) * m2
            # Crush: pull L down; matte: lift the crushed floor so it
            # reads as flat dark grey (~8% L) rather than 0.
            crush = -25.0 * w_mb
            toe_lift = 8.0 * w_mb
            L = L + crush + toe_lift

        out_lch[:, :, 0] = np.clip(L, 0.0, 100.0)
        out_lch[:, :, 1] = np.clip(C, 0.0, 200.0)
        out_lch[:, :, 2] = H

        graded = lch_f32_to_bgr_f32(out_lch)
        out = blend_masked(src_f, graded, bg_mask)
        return self._restore_dtype(out, was_float)

    def light_wrap(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        strength: float,
        wrap_color: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Apply a light-wrap rim around the subject edge.

        Spills a soft glow from the background onto the subject's
        silhouette edge — the signature composite integration effect
        that sells a subject-in-new-background composite.  The wrap
        colour defaults to the average background colour so it reads as
        ambient spill.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255].
            person_mask: (H, W) float mask [0, 1] of the subject.
            strength: 0–100.  0 = no-op.
            wrap_color: Optional (B, G, R) colour in [0, 255].  When
                None, the mean background colour is used.

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        if strength <= 0.0:
            return img

        src_f, was_float = self._to_f32_255(img)
        h, w = src_f.shape[:2]

        pm = self._feathered_person_mask(person_mask, (h, w))
        bg_mask = np.clip(1.0 - pm, 0.0, 1.0)

        # Wrap colour: average background pixel, or explicit override.
        if wrap_color is None:
            bg_pixels = src_f[bg_mask > 0.5]
            if bg_pixels.shape[0] > 0:
                wrap_color = tuple(float(v) for v in bg_pixels.mean(axis=0))
            else:
                wrap_color = (255.0, 255.0, 255.0)
        wrap = np.array(wrap_color, dtype=np.float32)

        # Edge band = dilated subject minus eroded subject (boundary zone).
        subj_bin = (pm > 0.5).astype(np.uint8)
        k = max(3, int(min(h, w) * 0.01)) | 1
        dilated = cv2.dilate(subj_bin, np.ones((k, k), np.uint8))
        eroded = cv2.erode(subj_bin, np.ones((k, k), np.uint8))
        edge = (dilated & ~eroded).astype(np.float32)

        # Feather the edge band into a soft glow.
        glow_radius = max(5, int(min(h, w) * 0.02)) | 1
        edge_soft = cv2.GaussianBlur(edge, (glow_radius, glow_radius), 0.0)
        edge_soft = np.clip(edge_soft, 0.0, 1.0).astype(np.float32)

        s = float(strength) / 100.0
        m = (edge_soft * s)[:, :, np.newaxis]
        # Screen-blend the wrap colour onto the edge.
        screen = 255.0 - ((255.0 - src_f) * (255.0 - wrap) / 255.0)
        out = src_f * (1.0 - m) + screen * m
        return self._restore_dtype(out, was_float)

    def sharpen_subject(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        strength: float,
        radius: float = 1.0,
    ) -> np.ndarray:
        """Sharpen the subject only (background untouched).

        Unsharp-mask through the feathered person mask — the
        subject-side counterpart to ``blur_background``.

        Args:
            img: (H, W, 3) uint8 or float32 BGR [0, 255].
            person_mask: (H, W) float mask [0, 1] of the subject.
            strength: 0–100.  0 = no-op.
            radius: Unsharp Gaussian sigma in pixels.

        Returns:
            (H, W, 3) image, same dtype as input.
        """
        if strength <= 0.0:
            return img

        src_f, was_float = self._to_f32_255(img)
        h, w = src_f.shape[:2]
        ksize = max(3, int(radius * 3.0)) | 1
        blurred = cv2.GaussianBlur(src_f, (ksize, ksize), radius)
        high = src_f - blurred
        s = float(strength) / 100.0
        sharpened = src_f + high * s
        pm = self._feathered_person_mask(person_mask, (h, w))
        out = blend_masked(src_f, sharpened, pm)
        return self._restore_dtype(out, was_float)


__all__ = ["BackgroundReplacer"]

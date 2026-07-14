"""Colour grading presets — final output tone and mood.

Provides the :class:`ColorGrader` class with a wide collection of cinematic,
analog, and corrective effects, plus preset loading from the bundled JSON
library.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from . import lut as _lut_mod
from .lut import CubeLUT, list_available_luts, load_cube, luts_dir
from .utils import apply_curve, blend_masked, normalize_mask, screen_blend, squeeze_mask, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32

from . import skin_protect
from .color_science import apply_subtractive_saturation
from .precision import ensure_float, to_uint8

logger = logging.getLogger(__name__)


def _apply_curve_f(channel: np.ndarray, curve_points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Float counterpart of ``apply_curve`` for use inside float pipelines.

    Interpolates ``channel`` (float, range [0, 255]) through the piecewise
    linear curve defined by ``curve_points`` (in 0-255 sample space) and
    returns a float32 result. Used by ``_F_apply_luminance_curve`` so the
    curve is applied without an intermediate uint8 LUT.

    Args:
        channel: Single-channel float32 array with values in [0, 255].
        curve_points: Sequence of (input, output) sample points in 0-255.

    Returns:
        Float32 array, same shape as ``channel``, values in [0, 255].
    """
    xs = np.array([p[0] for p in curve_points], dtype=np.float32)
    ys = np.array([p[1] for p in curve_points], dtype=np.float32)
    return np.clip(np.interp(channel, xs, ys), 0.0, 255.0).astype(np.float32)


def _bgr_f_to_lab_u8_conv(img_f: np.ndarray) -> np.ndarray:
    """Convert float32 BGR [0,1] to float32 LAB in uint8 convention.

    OpenCV's ``COLOR_BGR2LAB`` with float input produces L in [0,100] and
    a/b in [-128,127].  This helper rescales the result to match the uint8
    convention (L in [0,255], a/b in [0,255] with 128 = neutral) so that
    existing codebase math (e.g. ``l / 128.0``, ``lab[:,:,2] + warmth*30``)
    works identically without any uint8 round-trip.
    """
    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * 2.55
    lab[:, :, 1] = lab[:, :, 1] + 128.0
    lab[:, :, 2] = lab[:, :, 2] + 128.0
    return lab


def _lab_u8_conv_to_bgr_f(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_bgr_f_to_lab_u8_conv`: uint8-convention LAB to float32 BGR [0,1]."""
    lab_f = lab.astype(np.float32, copy=True)
    lab_f[:, :, 0] = lab_f[:, :, 0] / 2.55
    lab_f[:, :, 1] = lab_f[:, :, 1] - 128.0
    lab_f[:, :, 2] = lab_f[:, :, 2] - 128.0
    bgr = cv2.cvtColor(lab_f, cv2.COLOR_LAB2BGR).astype(np.float32)
    return bgr


def _bgr_f_to_hsv_u8_conv(img_f: np.ndarray) -> np.ndarray:
    """Convert float32 BGR [0,1] to float32 HSV in uint8 convention.

    OpenCV's ``COLOR_BGR2HSV`` with float input produces H in [0,360] and
    S/V in [0,1].  This helper rescales to match the uint8 convention
    (H in [0,180], S/V in [0,255]).
    """
    hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = hsv[:, :, 0] * 0.5
    hsv[:, :, 1] = hsv[:, :, 1] * 255.0
    hsv[:, :, 2] = hsv[:, :, 2] * 255.0
    return hsv


def _hsv_u8_conv_to_bgr_f(hsv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_bgr_f_to_hsv_u8_conv`: uint8-convention HSV to float32 BGR [0,1]."""
    hsv_f = hsv.astype(np.float32, copy=True)
    hsv_f[:, :, 0] = hsv_f[:, :, 0] * 2.0
    hsv_f[:, :, 1] = hsv_f[:, :, 1] / 255.0
    hsv_f[:, :, 2] = hsv_f[:, :, 2] / 255.0
    bgr = cv2.cvtColor(hsv_f, cv2.COLOR_HSV2BGR).astype(np.float32)
    return bgr


def _large_sigma_blur(
    img: np.ndarray,
    ksize: int,
    max_compute_dim: int = 1400,
) -> np.ndarray:
    """GaussianBlur with compute-downsample guard for large-σ kernels.

    For images where ``min(h, w) <= max_compute_dim`` the call is byte-identical
    to ``cv2.GaussianBlur(img, (ksize, ksize), 0)`` — the exact same OpenCV call
    is made on the exact same input. This is the safety property relied on by
    the byte-identity tests.

    For larger images the blur is computed on a downsampled copy
    (INTER_AREA down / INTER_LINEAR up) with a proportionally-scaled kernel.
    This is visually lossless because all call sites produce low-frequency
    outputs (glow / haze / bloom / soft-light base) that are later composited,
    never read as-is for detail. Mirrors the ``guided_filter max_dim`` pattern.

    Args:
        img: (H, W) or (H, W, C) float32 (or uint8) image. dtype preserved.
        ksize: Odd kernel size in pixels (full kernel, not radius).
        max_compute_dim: Threshold on ``min(h, w)`` above which the
            downsampled path is taken. Default 1400 (sub-1400px == byte-identical).

    Returns:
        Blurred image, same shape and dtype as input.
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim <= max_compute_dim:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    scale = float(max_compute_dim) / min_dim
    h_small = max(1, int(round(h * scale)))
    w_small = max(1, int(round(w * scale)))
    img_small = cv2.resize(img, (w_small, h_small), interpolation=cv2.INTER_AREA)
    ksize_small = max(3, int(round(ksize * scale))) | 1
    blurred_small = cv2.GaussianBlur(img_small, (ksize_small, ksize_small), 0)
    return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

_DEFAULT_PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"
_USER_PRESETS_DIRS: List[Path] = []


def register_presets_dir(directory: str | Path) -> None:
    _USER_PRESETS_DIRS.append(Path(directory))


def _find_preset_file(name: str) -> Optional[Path]:
    for d in [_DEFAULT_PRESETS_DIR] + _USER_PRESETS_DIRS:
        for ext in ("", ".json"):
            p = d / f"{name}{ext}"
            if p.exists():
                return p
    return None


def load_preset(name: str) -> Dict[str, Any]:
    """Load a single preset JSON file by name.

    Args:
        name: Preset name (without extension).

    Returns:
        Decoded preset settings dictionary.

    Raises:
        FileNotFoundError: If the preset cannot be located in any search dir.
    """
    fpath = _find_preset_file(name)
    if fpath is None:
        raise FileNotFoundError(f"Preset '{name}' not found")
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    if not _DEFAULT_PRESETS_DIR.exists():
        return presets
    for fpath in sorted(_DEFAULT_PRESETS_DIR.glob("*.json")):
        name = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                presets[name] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            import warnings
            warnings.warn(f"Failed to load preset '{name}': {e}")
    return presets


def list_available_presets() -> List[str]:
    names: List[str] = []
    seen: set = set()
    for d in [_DEFAULT_PRESETS_DIR] + _USER_PRESETS_DIRS:
        if d.exists():
            for fpath in sorted(d.glob("*.json")):
                name = fpath.stem
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


PRESETS: Dict[str, Dict[str, Any]] = load_all_presets()


# ---------------------------------------------------------------------------
# ColorGrader
# ---------------------------------------------------------------------------

class ColorGrader:
    """Apply colour grading presets to finalise image tone/mood."""

    def __init__(self):
        self._lut_cache: Dict[str, CubeLUT] = {}
        self._lock: threading.Lock = threading.Lock()

    def grade(
        self,
        img_bgr: np.ndarray,
        preset: Union[str, Dict[str, Any]] = "natural",
        intensity: float = 1.0,
        split_tone_mask: Optional[np.ndarray] = None,
        glow_mask: Optional[np.ndarray] = None,
        haze_mask: Optional[np.ndarray] = None,
        skin_mask: Optional[np.ndarray] = None,
        skip_glows: bool = False,
        skip_post_effects: bool = False,
        skin_protect_strength: float = 0.0,
        return_float: bool = False,
    ) -> np.ndarray:
        """Apply a colour grading preset to an image.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 [0,1] BGR image.
            preset: Preset name (loaded from JSON) or a settings dict.
            intensity: 0.0–1.0 blend strength against the original.
            split_tone_mask: Optional mask to restrict split-toning.
            glow_mask: Optional mask to restrict glow effects.
            haze_mask: Optional mask to restrict haze effect.
            skin_mask: Optional (H, W) float32 mask in [0, 1] identifying skin
                pixels. When provided, the warmth (b/a-channel) shift is clamped
                on skin pixels so the global warmth setting cannot push skin
                too far from neutral.
            skip_glows: Skip glow/orton effects for batch blending.
            skip_post_effects: Skip halation, chromatic aberration, LUT and grain
                for batch blending.
            skin_protect_strength: 0.0–1.0 strength of skin-tone protection during
                the color operations. Other Fuji foundation effects (tonal curve,
                highlight rolloff, film grain) are now applied as global stages in
                ``engine.py``, not as kwargs to ``grade()``.
            return_float: If True, return float32 [0,1] instead of uint8.

        Returns:
            (H, W, 3) uint8 or float32 [0,1] BGR image.
        """
        if isinstance(preset, str):
            settings = PRESETS.get(preset)
            if settings is None:
                try:
                    settings = load_preset(preset)
                except FileNotFoundError:
                    settings = PRESETS.get("natural", {})
        else:
            settings = preset

        is_float_input = img_bgr.dtype == np.float32
        original_for_blend = img_bgr.copy()
        result = img_bgr.copy()

        def _color_ops(img: np.ndarray) -> np.ndarray:
            r = ensure_float(img)

            if "white_balance" in settings:
                r = self._F_adjust_white_balance(r, settings["white_balance"])
            if "curves" in settings and "L" in settings["curves"]:
                r = self._F_apply_luminance_curve(r, settings["curves"]["L"])
            if "rgb_curves" in settings:
                r = self._F_apply_rgb_curves(r, settings["rgb_curves"])

            tone_rgb = {}
            if "tone_curve_red" in settings: tone_rgb["R"] = settings["tone_curve_red"]
            if "tone_curve_green" in settings: tone_rgb["G"] = settings["tone_curve_green"]
            if "tone_curve_blue" in settings: tone_rgb["B"] = settings["tone_curve_blue"]
            if tone_rgb:
                r = self._F_apply_rgb_curves(r, tone_rgb)

            if settings.get("shadow_lift", 0) > 0:
                r = self._F_lift_shadows(r, settings["shadow_lift"])
            if "calibration" in settings:
                r = self._F_apply_calibration(r, settings["calibration"])
            if abs(settings.get("warmth", 0)) > 0.001:
                warmth = settings["warmth"]
                r = self._F_adjust_warmth(r, warmth)
                if skin_mask is not None and skin_mask.max() > 0.01:
                    sign = 1.0 if warmth > 0 else -1.0
                    b_target = sign * min(abs(warmth) * 30.0, 10.0)
                    a_target = sign * min(abs(warmth) * 10.0, 4.0)
                    b_delta = b_target - (warmth * 30.0)
                    a_delta = a_target - (warmth * 10.0)
                    lab = _bgr_f_to_lab_u8_conv(r)
                    skin_bool = skin_mask > 0.3
                    lab[:, :, 2] = np.where(skin_bool, lab[:, :, 2] + b_delta, lab[:, :, 2])
                    lab[:, :, 1] = np.where(skin_bool, lab[:, :, 1] + a_delta, lab[:, :, 1])
                    lab = np.clip(lab, 0.0, 255.0).astype(np.float32)
                    r = np.clip(_lab_u8_conv_to_bgr_f(lab), 0.0, 1.0).astype(np.float32)
            if abs(settings.get("saturation_boost", 0)) > 0.001:
                if settings.get("saturation_mode", "additive") == "subtractive":
                    r = apply_subtractive_saturation(r, settings["saturation_boost"] / 100.0)
                else:
                    r = self._F_adjust_saturation(r, settings["saturation_boost"])

            if "hsl_adjustments" in settings:
                r = self._F_apply_hsl_adjustments(r, settings["hsl_adjustments"])
            elif "hsl_hue_shift" in settings:
                r = self._F_hsl_hue_shift(r, settings["hsl_hue_shift"])

            if "split_tone_three_way" in settings:
                r = self._F_split_tone_three_way(r, settings["split_tone_three_way"], split_tone_mask)
            elif "split_tone" in settings:
                r = self._F_split_tone(r, settings["split_tone"], split_tone_mask)

            if settings.get("clarity", 0) != 0:
                r = self._F_add_clarity(r, settings["clarity"])
            if settings.get("haze", 0) > 0:
                r = self._F_add_haze(r, settings["haze"], mask=haze_mask)

            if return_float or is_float_input:
                return np.clip(r, 0.0, 1.0).astype(np.float32)
            return to_uint8(r)

        if skin_protect_strength > 0:
            if is_float_input or return_float:
                def _color_ops_u8(img_u8: np.ndarray) -> np.ndarray:
                    r = _color_ops(img_u8)
                    if r.dtype == np.float32:
                        return np.clip(r * 255.0, 0, 255).astype(np.uint8)
                    return r
                result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
                result_u8 = skin_protect.protect_skin(result_u8, _color_ops_u8, skin_protect_strength)
                result = result_u8.astype(np.float32) / 255.0
            else:
                result = skin_protect.protect_skin(result, _color_ops, skin_protect_strength)
        else:
            result = _color_ops(result)

        # K3 — gamut-aware chroma compression. No-op when the graded result is
        # fully in-gamut (byte-identical golden path); otherwise rolls over-saturated
        # chroma back toward the sRGB boundary along a constant-hue, constant-L line.
        if settings.get("gamut_compress", True):
            result = self._apply_gamut_compress(result)

        # Track if we need float output
        want_float = return_float or is_float_input

        if "halation" in settings and not skip_post_effects:
            h_conf = settings["halation"]
            if isinstance(h_conf, (int, float)):
                result = self._add_halation(result, intensity=float(h_conf))
            else:
                result = self._add_halation(
                    result, threshold=h_conf.get("threshold", 200),
                    radius=h_conf.get("radius", 15), intensity=h_conf.get("intensity", 0.3)
                )

        if settings.get("vignette", 0) > 0 and not skip_post_effects:
            result = self._add_vignette(result, settings["vignette"])
        if settings.get("glow", 0) > 0 and not skip_glows:
            result = self._add_glow(result, settings["glow"], tint=settings.get("glow_tint", None), mask=glow_mask)
        if settings.get("orton_glow", 0) > 0 and not skip_glows:
            result = self._add_orton_glow(result, settings["orton_glow"], mask=glow_mask)
        if settings.get("sparkles", 0) > 0:
            result = self._add_sparkles(result, settings["sparkles"])
        if settings.get("chromatic_aberration", 0) > 0 and not skip_post_effects:
            result = self._add_chromatic_aberration(result, settings["chromatic_aberration"])
        if "lut" in settings and not skip_post_effects:
            result = self._add_film_emulation(result, settings["lut"])
        if settings.get("grain", 0) > 0 and not skip_post_effects:
            result = self._add_grain(result, settings["grain"])

        # Convert back to float if needed (post-effects return uint8)
        if want_float and result.dtype == np.uint8:
            result = result.astype(np.float32) / 255.0

        if intensity < 1.0:
            # Ensure both arrays have the same dtype for blending
            if result.dtype != original_for_blend.dtype:
                if result.dtype == np.float32:
                    original_for_blend = ensure_float(original_for_blend)
                else:
                    result = to_uint8(result)
            if result.dtype == np.float32:
                result = original_for_blend * (1.0 - intensity) + result * intensity
                result = np.clip(result, 0.0, 1.0).astype(np.float32)
            else:
                result = cv2.addWeighted(original_for_blend, 1.0 - intensity, result, intensity, 0)

        return result

    def _apply_gamut_compress(self, img: np.ndarray) -> np.ndarray:
        """Apply :func:`color_science.gamut_compress` to a BGR image.

        Short-circuits to a true no-op (returns ``img`` untouched) when the
        image is fully in-gamut, guaranteeing byte-identical output on the
        golden path. Converts via Oklab/OKLCh and back; the round-trip is only
        taken when at least one pixel is out of gamut.
        """
        from .color_science import (
            bgr_to_oklab,
            find_gamut_intersection,
            gamut_compress,
            oklab_to_bgr,
            oklab_to_oklch,
            oklch_to_oklab,
        )

        is_float = img.dtype == np.float32
        # A uint8 BGR image is, by definition, already inside the sRGB gamut,
        # so there is nothing to compress (and the OKLab round-trip would only
        # introduce rounding). Short-circuit to a true no-op — this keeps the
        # golden path byte-identical. Only the float path (where colors can be
        # out of gamut before the final clip) actually needs gamut mapping.
        if not is_float:
            return img
        # Cheap guard: a float image whose samples already lie inside [0, 1] is
        # sRGB-in-gamut, so there is nothing to map. This avoids the (expensive)
        # per-pixel gamut bisection on the common in-gamut float path.
        if bool(np.all(img >= -1e-4) and np.all(img <= 1.0 + 1e-4)):
            return img
        oklab = bgr_to_oklab(np.clip(img, 0.0, 1.0).astype(np.float32) * 255.0)
        oklch = oklab_to_oklch(oklab)
        C = oklch[..., 1]
        Cmax = find_gamut_intersection(oklab)
        if bool(np.all(C <= Cmax + 1e-4)):
            return img
        oklch = gamut_compress(oklch)
        oklab2 = oklch_to_oklab(oklch)
        out = oklab_to_bgr(oklab2, float32_out=is_float)
        return np.ascontiguousarray((out / 255.0).astype(np.float32))

    def add_impact_finish(
        self,
        img_bgr: np.ndarray,
        strength: float,
        subject_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply a punchy "impact" finish — contrast + clarity + glow composite.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            strength: 0–100 intensity of the finish.
            subject_mask: Optional (H, W) float32 mask in [0, 1]. When provided,
                the clarity/contrast effects are restricted to the subject so the
                background does not get noisy.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0:
            return img_bgr
        s = np.clip(strength / 100.0, 0.0, 1.0)
        original = img_bgr.copy()
        result = img_bgr.copy()

        result = self._apply_luminance_curve(
            result, [(0, 0), (45, int(45 - 18 * s)), (128, int(128 + 7 * s)), (210, int(210 + 20 * s)), (255, 255)]
        )
        result = self._adjust_saturation(result, 0.10 * s)
        result = self._add_clarity(result, 0.18 * s)
        result = self._add_glow(result, 0.06 * s, tint=(235, 140, 210))

        blended = cv2.addWeighted(original, 1.0 - s, result, s, 0)

        if subject_mask is not None:
            mask_3d = subject_mask[:, :, np.newaxis].astype(np.float32)
            blended = (blended.astype(np.float32) * mask_3d
                       + original.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)

        return blended

    def grade_stack(
        self,
        img_bgr: np.ndarray,
        preset_weights: Dict[str, float],
    ) -> np.ndarray:
        """Blend multiple presets with custom weights — true independent mixing.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            preset_weights: Mapping of preset name to weight.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if not preset_weights:
            return img_bgr
        total_w = sum(preset_weights.values())
        if total_w <= 0:
            return img_bgr

        base = img_bgr.astype(np.float32)
        accum = np.zeros_like(base)
        for preset_name, weight in preset_weights.items():
            if weight <= 0:
                continue
            # CRITICAL FIX: Skip glows and post-effects during blending to prevent
            # compounding noise, grain, and bloom artifacts.
            graded = self.grade(
                img_bgr, preset_name, 1.0,
                skip_glows=True, skip_post_effects=True,
            )
            accum += graded.astype(np.float32) * weight

        result = accum / total_w
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_glow(
        self,
        img_bgr: np.ndarray,
        opacity: float,
        tint: Optional[Tuple[int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if opacity <= 0:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        img_f = img_bgr.astype(np.float32) if not is_float else img_bgr
        h, w = img_bgr.shape[:2]
        ksize = max(int(min(h, w) * 0.025), 7) | 1
        blurred = _large_sigma_blur(img_f, ksize)

        if tint is not None:
            tint_arr = np.array(tint, dtype=np.float32)
            if tint_arr.max() <= 1.0: tint_arr = tint_arr * 255.0
            tint_layer = np.ones_like(blurred) * tint_arr
            blurred = cv2.addWeighted(blurred, 0.7, tint_layer, 0.3, 0)

        screen = screen_blend(img_f, blurred)
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        highlight_mask = np.clip((l_chan - 225.0) / 20.0, 0, 1)[:, :, np.newaxis]

        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2: m_f = m_f[:, :, np.newaxis]
            highlight_mask = highlight_mask * m_f

        result = img_f * (1.0 - highlight_mask * opacity) + screen * (highlight_mask * opacity)
        if is_float:
            return np.clip(result, 0, 255).astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_luminance_curve(
        self,
        img: np.ndarray,
        curve_points: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        out_f = self._F_apply_luminance_curve(ensure_float(img), curve_points)
        return to_uint8(out_f)

    def _F_apply_luminance_curve(
        self,
        img_f: np.ndarray,
        curve_points: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        """Float version of :meth:`_apply_luminance_curve`.

        Operates entirely in float32 [0, 1] in/out; the curve is applied
        via linear interpolation on the L* channel (no uint8 LUT step),
        which avoids the ``apply_curve`` -> ``cv2.LUT`` -> ``astype(uint8)``
        quantization that the uint8 path goes through.

        Args:
            img_f: (H, W, 3) float32 BGR image with values in [0, 1].
            curve_points: Sequence of (input, output) sample points in
                0-255 sample space.

        Returns:
            (H, W, 3) float32 BGR image in [0, 1].
        """
        is_identity = all(int(p[0]) == int(p[1]) for p in curve_points)
        if is_identity:
            return img_f.copy()
        lab = _bgr_f_to_lab_u8_conv(img_f)
        lab[:, :, 0] = _apply_curve_f(lab[:, :, 0], curve_points)
        lab = np.clip(lab, 0.0, 255.0).astype(np.float32)
        out_f = _lab_u8_conv_to_bgr_f(lab)
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _lift_shadows(self, img: np.ndarray, lift: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        lab[:, :, 0] = np.clip(l + shadow_mask * lift, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_lift_shadows(self, img_f: np.ndarray, lift: float) -> np.ndarray:
        lab = _bgr_f_to_lab_u8_conv(img_f)
        l = lab[:, :, 0]
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        lab[:, :, 0] = np.clip(l + shadow_mask * lift, 0, 255)
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _adjust_warmth(self, img: np.ndarray, warmth: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth * 30, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth * 10, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_adjust_warmth(self, img_f: np.ndarray, warmth: float) -> np.ndarray:
        lab = _bgr_f_to_lab_u8_conv(img_f)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth * 30, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth * 10, 0, 255)
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _adjust_saturation(self, img: np.ndarray, boost: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        factor = 1.0 + boost * (1.0 - s / 255.0)
        hsv[:, :, 1] = np.clip(s * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _F_adjust_saturation(self, img_f: np.ndarray, boost: float) -> np.ndarray:
        hsv = _bgr_f_to_hsv_u8_conv(img_f)
        s = hsv[:, :, 1]
        factor = 1.0 + boost * (1.0 - s / 255.0)
        hsv[:, :, 1] = np.clip(s * factor, 0, 255)
        out_f = _hsv_u8_conv_to_bgr_f(np.clip(hsv, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _split_tone(
        self,
        img: np.ndarray,
        tones: Dict[str, Tuple[int, int]],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_weight = np.clip(1.0 - l / 128.0, 0, 1)[:, :, np.newaxis]
        highlight_weight = np.clip((l - 128.0) / 128.0, 0, 1)[:, :, np.newaxis]
        shadow_ab = np.array(tones["shadows"], dtype=np.float32)
        highlight_ab = np.array(tones["highlights"], dtype=np.float32)
        ab = lab[:, :, 1:3]
        ab = ab + (shadow_ab - 128) * shadow_weight * 0.15
        ab = ab + (highlight_ab - 128) * highlight_weight * 0.15
        lab[:, :, 1:3] = np.clip(ab, 0, 255)
        split_toned = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        if mask is not None: return blend_masked(img, split_toned, mask)
        return split_toned

    def _F_split_tone(
        self,
        img_f: np.ndarray,
        tones: Dict[str, Tuple[int, int]],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        lab = _bgr_f_to_lab_u8_conv(img_f)
        l = lab[:, :, 0]
        shadow_weight = np.clip(1.0 - l / 128.0, 0, 1)[:, :, np.newaxis]
        highlight_weight = np.clip((l - 128.0) / 128.0, 0, 1)[:, :, np.newaxis]
        shadow_ab = np.array(tones["shadows"], dtype=np.float32)
        highlight_ab = np.array(tones["highlights"], dtype=np.float32)
        ab = lab[:, :, 1:3]
        ab = ab + (shadow_ab - 128) * shadow_weight * 0.15
        ab = ab + (highlight_ab - 128) * highlight_weight * 0.15
        lab[:, :, 1:3] = np.clip(ab, 0, 255)
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        out_f = np.clip(out_f, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            m = ensure_float(mask)
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
            out_f = img_f * (1.0 - m) + out_f * m
            out_f = np.clip(out_f, 0.0, 1.0)
        return out_f

    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        r: int,
        eps: float,
    ) -> np.ndarray:
        mean_I = cv2.boxFilter(guide, -1, (r, r))
        mean_p = cv2.boxFilter(src, -1, (r, r))
        mean_Ip = cv2.boxFilter(guide * src, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide * guide, -1, (r, r))
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))
        return mean_a * guide + mean_b

    def _add_clarity(self, img: np.ndarray, strength: float) -> np.ndarray:
        if strength == 0:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        
        # MAJOR FIX: Removed useless clarity cache. np.array_equal takes ~20ms on 4K 
        # and always fails because earlier grading steps modify the L channel.
        h, w = img.shape[:2]
        r = max(int(min(h, w) * 0.015), 5)
        eps = 0.02
        l_norm = l_chan / 255.0
        base = self._guided_filter(l_norm, l_norm, r, eps) * 255.0
        detail = l_chan - base
        
        l_new = np.clip(base + detail * (1.0 + strength), 0, 255)
        lab[:, :, 0] = l_new.astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _F_add_clarity(self, img_f: np.ndarray, strength: float) -> np.ndarray:
        if strength == 0:
            return img_f
        lab = _bgr_f_to_lab_u8_conv(img_f)
        l_chan = lab[:, :, 0]
        h, w = img_f.shape[:2]
        r = max(int(min(h, w) * 0.015), 5)
        eps = 0.02
        l_norm = l_chan / 255.0
        base = self._guided_filter(l_norm, l_norm, r, eps) * 255.0
        detail = l_chan - base
        l_new = np.clip(base + detail * (1.0 + strength), 0, 255)
        lab[:, :, 0] = l_new
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _add_vignette(self, img: np.ndarray, strength: float) -> np.ndarray:
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - (dist / max_dist) ** 2 * strength
        vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
        is_float = img.dtype == np.float32
        if is_float:
            return np.clip(img * vignette, 0.0, 1.0).astype(np.float32)
        return np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    def _apply_rgb_curves(
        self,
        img_bgr: np.ndarray,
        curves_dict: Dict[str, Sequence[Tuple[int, int]]],
    ) -> np.ndarray:
        b, g, r = cv2.split(img_bgr)
        if "R" in curves_dict: r = apply_curve(r, curves_dict["R"])
        if "G" in curves_dict: g = apply_curve(g, curves_dict["G"])
        if "B" in curves_dict: b = apply_curve(b, curves_dict["B"])
        return cv2.merge([b, g, r])

    def _F_apply_rgb_curves(
        self,
        img_f: np.ndarray,
        curves_dict: Dict[str, Sequence[Tuple[int, int]]],
    ) -> np.ndarray:
        b, g, r = cv2.split(img_f)
        if "R" in curves_dict:
            r = _apply_curve_f(r * 255.0, curves_dict["R"]) / 255.0
        if "G" in curves_dict:
            g = _apply_curve_f(g * 255.0, curves_dict["G"]) / 255.0
        if "B" in curves_dict:
            b = _apply_curve_f(b * 255.0, curves_dict["B"]) / 255.0
        return np.clip(cv2.merge([b, g, r]), 0.0, 1.0)

    def _hsl_hue_shift(
        self,
        img_bgr: np.ndarray,
        shifts_dict: Dict[str, float],
    ) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        ranges = {"red": [(0, 10), (170, 180)], "orange": [(10, 25)], "yellow": [(25, 35)],
                  "green": [(35, 80)], "cyan": [(80, 105)], "blue": [(105, 140)], "magenta": [(140, 170)]}
        for color, shift_deg in shifts_dict.items():
            if color not in ranges or shift_deg == 0: continue
            shift_cv = float(shift_deg) / 2.0
            mask = np.zeros_like(h, dtype=bool)
            for low, high in ranges[color]: mask |= (h >= low) & (h <= high)
            hsv[:, :, 0] = np.where(mask, (h + shift_cv) % 180, h)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _F_hsl_hue_shift(
        self,
        img_f: np.ndarray,
        shifts_dict: Dict[str, float],
    ) -> np.ndarray:
        hsv = _bgr_f_to_hsv_u8_conv(img_f)
        h = hsv[:, :, 0]
        ranges = {"red": [(0, 10), (170, 180)], "orange": [(10, 25)], "yellow": [(25, 35)],
                  "green": [(35, 80)], "cyan": [(80, 105)], "blue": [(105, 140)], "magenta": [(140, 170)]}
        for color, shift_deg in shifts_dict.items():
            if color not in ranges or shift_deg == 0: continue
            shift_cv = float(shift_deg) / 2.0
            mask = np.zeros_like(h, dtype=bool)
            for low, high in ranges[color]: mask |= (h >= low) & (h <= high)
            hsv[:, :, 0] = np.where(mask, (h + shift_cv) % 180, h)
        out_f = _hsv_u8_conv_to_bgr_f(np.clip(hsv, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _add_chromatic_aberration(self, img_bgr: np.ndarray, max_disp: float) -> np.ndarray:
        if max_disp <= 0: return img_bgr
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = x - cx; dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        max_r = np.sqrt(cx*cx + cy*cy)
        factor = (r / max_r) * max_disp
        map_x_r = (x + (dx / (r + 1e-5)) * factor).astype(np.float32)
        map_y_r = (y + (dy / (r + 1e-5)) * factor).astype(np.float32)
        map_x_b = (x - (dx / (r + 1e-5)) * factor).astype(np.float32)
        map_y_b = (y - (dy / (r + 1e-5)) * factor).astype(np.float32)
        b, g, r_chan = cv2.split(img_bgr)
        r_new = cv2.remap(r_chan, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        b_new = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return cv2.merge([b_new, g, r_new])

    def _add_halation(
        self,
        img_bgr: np.ndarray,
        threshold: int = 200,
        radius: int = 15,
        intensity: float = 0.3,
    ) -> np.ndarray:
        if intensity <= 0: return img_bgr
        # E1 float-canvas adapter: grade()'s float path hands us float32 [0,1].
        # The uint8-scale threshold (200) and the final astype(np.uint8) would
        # otherwise floor the whole [0,1] canvas to black.
        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        halation_color = np.zeros_like(img_u8, dtype=np.uint8)
        halation_color[:, :, 2] = thresh
        halation_color[:, :, 1] = (thresh * 0.45).astype(np.uint8)
        ksize = radius | 1
        blurred_halation = cv2.GaussianBlur(halation_color, (ksize, ksize), 0).astype(np.float32)
        img_f = img_u8.astype(np.float32)
        screen = screen_blend(img_f, blurred_halation * intensity)
        result = np.clip(screen, 0, 255).astype(np.uint8)
        if is_float:
            return result.astype(np.float32) / 255.0
        return result

    def _add_grain(self, img_bgr: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0: return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr
        h, w = img_u8.shape[:2]
        gw, gh = max(w // 2, 64), max(h // 2, 64)
        noise = np.random.normal(0, 255.0 * strength, (gh, gw)).astype(np.float32)
        noise_scaled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noise_blurred = cv2.GaussianBlur(noise_scaled, (3, 3), 0.5)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        weight = np.exp(-((l_chan - 128.0) ** 2) / (2.0 * 64.0 ** 2))
        weighted_noise = (noise_blurred * weight)[:, :, np.newaxis]
        img_f = img_u8.astype(np.float32)
        result = np.clip(img_f + weighted_noise, 0, 255).astype(np.uint8)
        if is_float:
            return result.astype(np.float32) / 255.0
        return result

    def _adjust_white_balance(
        self,
        img_bgr: np.ndarray,
        multipliers: Dict[str, float],
    ) -> np.ndarray:
        b_mult = multipliers.get("B", 1.0)
        g_mult = multipliers.get("G", 1.0)
        r_mult = multipliers.get("R", 1.0)
        if all(abs(m - 1.0) < 0.001 for m in (r_mult, g_mult, b_mult)):
            return img_bgr
        gray = np.array([128, 128, 128], dtype=np.float32)
        wb = np.clip(gray * np.array([b_mult, g_mult, r_mult]), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        gray_u8 = gray.astype(np.uint8).reshape(1, 1, 3)
        gray_lab = cv2.cvtColor(gray_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        wb_lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_off = wb_lab[0, 0, 1] - gray_lab[0, 0, 1]
        b_off = wb_lab[0, 0, 2] - gray_lab[0, 0, 2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_off, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_off, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_adjust_white_balance(
        self,
        img_f: np.ndarray,
        multipliers: Dict[str, float],
    ) -> np.ndarray:
        b_mult = multipliers.get("B", 1.0)
        g_mult = multipliers.get("G", 1.0)
        r_mult = multipliers.get("R", 1.0)
        if all(abs(m - 1.0) < 0.001 for m in (r_mult, g_mult, b_mult)):
            return img_f
        gray = np.array([128, 128, 128], dtype=np.float32) / 255.0
        wb = np.clip(gray * np.array([b_mult, g_mult, r_mult]), 0.0, 1.0).astype(np.float32)
        gray_lab = _bgr_f_to_lab_u8_conv(gray.reshape(1, 1, 3))
        wb_lab = _bgr_f_to_lab_u8_conv(wb.reshape(1, 1, 3))
        a_off = wb_lab[0, 0, 1] - gray_lab[0, 0, 1]
        b_off = wb_lab[0, 0, 2] - gray_lab[0, 0, 2]
        lab = _bgr_f_to_lab_u8_conv(img_f)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_off, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_off, 0, 255)
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _apply_calibration(
        self,
        img_bgr: np.ndarray,
        calibration: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        out_f = self._F_apply_calibration(ensure_float(img_bgr), calibration)
        return to_uint8(out_f)

    def _F_apply_calibration(
        self,
        img_f: np.ndarray,
        calibration: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """Float version of :meth:`_apply_calibration`.

        Performs the per-hue calibration math in float32, avoiding the
        intermediate ``astype(uint8)`` clipping on H/S that the uint8
        path applies. Operates in HSV space; H is in [0, 180] and S in
        [0, 255] (OpenCV conventions).

        Args:
            img_f: (H, W, 3) float32 BGR image with values in [0, 1].
            calibration: Dict keyed by color name (``"red"``, ``"green"``,
                ``"blue"``) with ``hue`` and ``sat`` shifts.

        Returns:
            (H, W, 3) float32 BGR image in [0, 1].
        """
        if not calibration:
            return img_f
        hsv = _bgr_f_to_hsv_u8_conv(img_f)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        r_hue_shift = calibration.get("red", {}).get("hue", 0) * 0.15 / 2.0
        r_sat_shift = calibration.get("red", {}).get("sat", 0) * 0.005
        g_hue_shift = calibration.get("green", {}).get("hue", 0) * 0.15 / 2.0
        g_sat_shift = calibration.get("green", {}).get("sat", 0) * 0.005
        b_hue_shift = calibration.get("blue", {}).get("hue", 0) * 0.15 / 2.0
        b_sat_shift = calibration.get("blue", {}).get("sat", 0) * 0.005
        if (r_hue_shift == 0 and r_sat_shift == 0 and g_hue_shift == 0
                and g_sat_shift == 0 and b_hue_shift == 0 and b_sat_shift == 0):
            return img_f
        sigma = 15.0
        dist_r = np.minimum(np.abs(h - 0.0), np.abs(h - 180.0))
        w_r = np.exp(-(dist_r ** 2) / (2.0 * sigma ** 2))
        dist_g = np.abs(h - 60.0)
        w_g = np.exp(-(dist_g ** 2) / (2.0 * sigma ** 2))
        dist_b = np.abs(h - 120.0)
        w_b = np.exp(-(dist_b ** 2) / (2.0 * sigma ** 2))
        h_new = (h + w_r * r_hue_shift + w_g * g_hue_shift + w_b * b_hue_shift) % 180.0
        s_new = np.clip(s * (1.0 + w_r * r_sat_shift + w_g * g_sat_shift + w_b * b_sat_shift), 0, 255)
        hsv[:, :, 0] = h_new
        hsv[:, :, 1] = s_new
        out_f = _hsv_u8_conv_to_bgr_f(np.clip(hsv, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _add_orton_glow(
        self,
        img_bgr: np.ndarray,
        opacity: float,
        blur_radius: int = 35,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if opacity <= 0: return img_bgr
        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float32) / 255.0
        ksize = max(blur_radius, int(min(h, w) * 0.015)) | 1
        blurred = _large_sigma_blur(img_f, ksize)
        soft_light = (1.0 - 2.0 * blurred) * (img_f ** 2) + 2.0 * blurred * img_f
        result = img_f * (1.0 - opacity) + soft_light * opacity
        orton_img = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        if mask is not None: return blend_masked(img_bgr, orton_img, mask)
        return orton_img

    def _split_tone_three_way(
        self,
        img_bgr: np.ndarray,
        tones: Dict[str, Any],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out_f = self._F_split_tone_three_way(ensure_float(img_bgr), tones, mask)
        return to_uint8(out_f)

    def _F_split_tone_three_way(
        self,
        img_f: np.ndarray,
        tones: Dict[str, Any],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Float version of :meth:`_split_tone_three_way`.

        The shadow / midtone / highlight sigmoid weights and the
        a/b channel offsets are computed in float32 [0, 1]; the
        per-pixel ``np.clip(ab, 0, 255)`` that the uint8 path applies
        inside the ab channels is deferred to the final BGR conversion,
        so cumulative offsets don't prematurely quantize.

        Args:
            img_f: (H, W, 3) float32 BGR image with values in [0, 1].
            tones: Dict with ``shadows``, ``midtones``, ``highlights``
                (each ``{hue, sat}``) and optional ``balance``.
            mask: Optional (H, W) or (H, W, 1) float mask in [0, 1]
                restricting the effect.

        Returns:
            (H, W, 3) float32 BGR image in [0, 1].
        """
        if not tones:
            return img_f
        shadows = tones.get("shadows", {})
        midtones = tones.get("midtones", {})
        highlights = tones.get("highlights", {})
        balance = tones.get("balance", 0.0)
        lab = _bgr_f_to_lab_u8_conv(img_f)
        l_val = lab[:, :, 0]
        l_norm = l_val / 255.0

        def hsl_to_lab_offsets(hue: Optional[float], sat: Optional[float]) -> Tuple[float, float]:
            if hue is None or sat is None or sat == 0:
                return 0.0, 0.0
            theta = np.radians(hue)
            chroma = (sat / 100.0) * 25.0
            return chroma * np.cos(theta), chroma * np.sin(theta)

        s_a, s_b = hsl_to_lab_offsets(shadows.get("hue"), shadows.get("sat"))
        m_a, m_b = hsl_to_lab_offsets(midtones.get("hue"), midtones.get("sat"))
        h_a, h_b = hsl_to_lab_offsets(highlights.get("hue"), highlights.get("sat"))
        if (abs(s_a) < 0.01 and abs(s_b) < 0.01 and abs(m_a) < 0.01
                and abs(m_b) < 0.01 and abs(h_a) < 0.01 and abs(h_b) < 0.01):
            return img_f

        shift = (balance / 100.0) * 0.2
        w_shadow = 1.0 - 1.0 / (1.0 + np.exp(-(l_norm - (0.3 + shift)) * 10.0))
        w_highlight = 1.0 / (1.0 + np.exp(-(l_norm - (0.7 + shift)) * 10.0))
        w_midtone = np.clip(1.0 - w_shadow - w_highlight, 0.0, 1.0)

        ab = lab[:, :, 1:3].copy()
        ab[:, :, 0] += (s_a * w_shadow + m_a * w_midtone + h_a * w_highlight)
        ab[:, :, 1] += (s_b * w_shadow + m_b * w_midtone + h_b * w_highlight)
        lab[:, :, 1:3] = ab
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        out_f = np.clip(out_f, 0.0, 1.0).astype(np.float32)
        if mask is not None:
            m = ensure_float(mask)
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
            out_f = img_f * (1.0 - m) + out_f * m
            out_f = np.clip(out_f, 0.0, 1.0)
        return out_f

    def _add_film_emulation(
        self,
        img_bgr: np.ndarray,
        lut_input: Optional[Union[str, Path, CubeLUT]] = None,
        strength: float = 1.0,
    ) -> np.ndarray:
        if lut_input is None:
            return img_bgr
        if isinstance(lut_input, CubeLUT):
            cube = lut_input
        else:
            text = str(lut_input).strip()
            if not text or text.lower() == "none":
                return img_bgr
            cube = self._resolve_cube_lut(text)
        if strength <= 0.0:
            return img_bgr
        out = cube.apply(img_bgr)
        if strength >= 1.0:
            return out
        return cv2.addWeighted(img_bgr, 1.0 - strength, out, strength, 0)

    def _resolve_cube_lut(self, lut_input: str) -> CubeLUT:
        p = Path(lut_input)
        candidates: List[Path] = [p]
        if p.suffix.lower() != ".cube":
            candidates.append(_lut_mod.luts_dir() / f"{p.name}.cube")
        chosen: Optional[Path] = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            available = ", ".join(list_available_luts()) or "<none>"
            raise FileNotFoundError(
                f"LUT {lut_input!r} not found. Searched: "
                + ", ".join(str(c) for c in candidates)
                + f". Available LUTs in luts/: {available}"
            )
        return self._get_cached_lut(chosen)

    def _get_cached_lut(self, path: Path) -> CubeLUT:
        key = str(path.resolve())
        with self._lock:
            cached = self._lut_cache.get(key)
        if cached is not None:
            return cached
        cube = load_cube(path)
        with self._lock:
            self._lut_cache[key] = cube
        return cube

    def _apply_hsl_adjustments(
        self,
        img_bgr: np.ndarray,
        adjustments: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]; s = hsv[:, :, 1]; v = hsv[:, :, 2]
        color_centers = {"red": 0.0, "orange": 14.0, "yellow": 27.0, "green": 57.0, 
                          "cyan": 92.0, "blue": 122.0, "purple": 148.0, "magenta": 156.0}
        sigma = 10.0
        hue_adj = adjustments.get("hue", {}); sat_adj = adjustments.get("saturation", {}); lum_adj = adjustments.get("luminance", {})
        for color in color_centers:
            h_shift = hue_adj.get(color, 0); s_shift = sat_adj.get(color, 0); l_shift = lum_adj.get(color, 0)
            if h_shift == 0 and s_shift == 0 and l_shift == 0: continue
            center = color_centers[color]
            dist = np.abs(h - center)
            dist = np.minimum(dist, 180.0 - dist)
            weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
            if h_shift != 0:
                shift_cv = float(h_shift) / 2.0
                h = (h + weight * shift_cv) % 180
            if s_shift != 0:
                s = np.where(weight > 0.01, np.clip(s + weight * s_shift, 0, 255), s)
            if l_shift != 0:
                v = np.where(weight > 0.01, np.clip(v + weight * l_shift, 0, 255), v)
        hsv[:, :, 0] = h; hsv[:, :, 1] = s; hsv[:, :, 2] = v
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _F_apply_hsl_adjustments(
        self,
        img_f: np.ndarray,
        adjustments: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        hsv = _bgr_f_to_hsv_u8_conv(img_f)
        h = hsv[:, :, 0]; s = hsv[:, :, 1]; v = hsv[:, :, 2]
        color_centers = {"red": 0.0, "orange": 14.0, "yellow": 27.0, "green": 57.0,
                          "cyan": 92.0, "blue": 122.0, "purple": 148.0, "magenta": 156.0}
        sigma = 10.0
        hue_adj = adjustments.get("hue", {}); sat_adj = adjustments.get("saturation", {}); lum_adj = adjustments.get("luminance", {})
        for color in color_centers:
            h_shift = hue_adj.get(color, 0); s_shift = sat_adj.get(color, 0); l_shift = lum_adj.get(color, 0)
            if h_shift == 0 and s_shift == 0 and l_shift == 0: continue
            center = color_centers[color]
            dist = np.abs(h - center)
            dist = np.minimum(dist, 180.0 - dist)
            weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
            if h_shift != 0:
                shift_cv = float(h_shift) / 2.0
                h = (h + weight * shift_cv) % 180
            if s_shift != 0:
                s = np.where(weight > 0.01, np.clip(s + weight * s_shift, 0, 255), s)
            if l_shift != 0:
                v = np.where(weight > 0.01, np.clip(v + weight * l_shift, 0, 255), v)
        hsv[:, :, 0] = h; hsv[:, :, 1] = s; hsv[:, :, 2] = v
        out_f = _hsv_u8_conv_to_bgr_f(np.clip(hsv, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # LCH HSL panel methods (Phase 1.d — perceptual uniformity replaces HSV)
    # ------------------------------------------------------------------

    _HSL_CHANNELS: Dict[str, Tuple[float, float]] = {
        "red":       (0.0, 30.0),
        "orange":    (30.0, 30.0),
        "yellow":    (60.0, 30.0),
        "green":     (120.0, 30.0),
        "aqua":      (180.0, 30.0),
        "blue":      (240.0, 30.0),
        "purple":    (280.0, 30.0),
        "magenta":   (320.0, 30.0),
    }

    def adjust_hsl_lch(
        self,
        img_bgr: np.ndarray,
        hue_shift: float = 0.0,
        sat_scale: float = 1.0,
        lum_shift: float = 0.0,
    ) -> np.ndarray:
        """Global perceptual HSL adjustment in LCH space.

        Unlike HSV (where equal numeric sat/lum steps don't equal
        perceived changes), LCH gives perceptually uniform results.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            hue_shift: Hue rotation in degrees (-180 to 180).
            sat_scale: Saturation multiplier (1.0 = no change,
                       0.0 = desaturate, 2.0 = double saturation).
            lum_shift: Lightness shift in [-100, 100] L* units.
                       Positive brightens, negative darkens.

        Returns:
            (H, W, 3) uint8 BGR.
        """
        if abs(hue_shift) < 1e-4 and abs(sat_scale - 1.0) < 1e-4 and abs(lum_shift) < 1e-4:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import bgr_f32_to_lch_f32, lch_f32_to_bgr_f32
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import bgr_to_lch, lch_to_bgr
            lch = bgr_to_lch(img_bgr)
        if abs(hue_shift) > 1e-4:
            lch[:, :, 2] = np.mod(lch[:, :, 2] + hue_shift, 360.0)
        if abs(sat_scale - 1.0) > 1e-4:
            lch[:, :, 1] = lch[:, :, 1] * sat_scale
        if abs(lum_shift) > 1e-4:
            lch[:, :, 0] = np.clip(lch[:, :, 0] + lum_shift, 0.0, 100.0)
        if is_float:
            return lch_f32_to_bgr_f32(lch)
        return lch_to_bgr(lch)

    def adjust_per_channel_lch(
        self,
        img_bgr: np.ndarray,
        channel: str,
        hue_shift: float = 0.0,
        sat_scale: float = 1.0,
        lum_shift: float = 0.0,
    ) -> np.ndarray:
        """Perceptual HSL adjustment for a single colour channel.

        Operates on a soft hue range (e.g. ``"red"`` = hues near 0°,
        ``"blue"`` = hues near 240°).  This is the L*‑keyed equivalent
        of the Lightroom HSL/Color panel.

        Valid channel names: red, orange, yellow, green, aqua, blue,
        purple, magenta.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            channel: Colour channel name.
            hue_shift: Hue shift within the range (-60 to 60 typical).
            sat_scale: Chroma scale for pixels in the range.
            lum_shift: L* shift for pixels in the range.

        Returns:
            (H, W, 3) uint8 BGR.
        """
        if channel not in self._HSL_CHANNELS:
            raise ValueError(
                f"Unknown channel {channel!r}. Valid: {list(self._HSL_CHANNELS)}"
            )
        if abs(hue_shift) < 1e-4 and abs(sat_scale - 1.0) < 1e-4 and abs(lum_shift) < 1e-4:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import (
                bgr_f32_to_lch_f32,
                lch_f32_to_bgr_f32,
                adjust_hue_range,
                adjust_chroma_range,
                adjust_luminance_range,
            )
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import (
                bgr_to_lch,
                lch_to_bgr,
                adjust_hue_range,
                adjust_chroma_range,
                adjust_luminance_range,
            )
            lch = bgr_to_lch(img_bgr)
        hue_center, hue_width = self._HSL_CHANNELS[channel]
        if abs(hue_shift) > 1e-4:
            lch = adjust_hue_range(lch, hue_center, hue_width, hue_shift)
        if abs(sat_scale - 1.0) > 1e-4:
            lch = adjust_chroma_range(lch, hue_center, hue_width, sat_scale)
        if abs(lum_shift) > 1e-4:
            lch = adjust_luminance_range(lch, hue_center, hue_width, lum_shift)
        if is_float:
            return lch_f32_to_bgr_f32(lch)
        return lch_to_bgr(lch)

    def split_tone_lch(
        self,
        img_bgr: np.ndarray,
        shadow_hue: float = 0.0,
        shadow_sat: float = 0.0,
        highlight_hue: float = 0.0,
        highlight_sat: float = 0.0,
        balance: float = 0.0,
    ) -> np.ndarray:
        """L*‑keyed split toning — perceptually uniform shadow/highlight tint.

        Shadows and highlights are separated by the L* channel (not HSV
        Value), so the boundary matches where the eye sees dark vs light.
        This produces more natural transitions than HSV‑based split toning.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            shadow_hue: Hue for shadows (degrees, 0–360).
            shadow_sat: Saturation strength for shadows (0–1).
            highlight_hue: Hue for highlights.
            highlight_sat: Saturation strength for highlights.
            balance: Crossover shift in [-100, 100].

        Returns:
            (H, W, 3) uint8 BGR.
        """
        if shadow_sat <= 1e-6 and highlight_sat <= 1e-6:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import (
                bgr_f32_to_lch_f32,
                lch_f32_to_bgr_f32,
                split_tone_lch as _st,
            )
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import bgr_to_lch, lch_to_bgr, split_tone_lch as _st
            lch = bgr_to_lch(img_bgr)
        toned = _st(lch, shadow_hue, shadow_sat, highlight_hue, highlight_sat, balance)
        if is_float:
            return lch_f32_to_bgr_f32(toned)
        return lch_to_bgr(toned)

    def negative_split_tone(
        self,
        img_bgr: np.ndarray,
        shadow_desat: float = 0.0,
        highlight_desat: float = 0.0,
    ) -> np.ndarray:
        """Desaturate shadows and/or highlights — 'faded film' look.

        Unlike :meth:`split_tone_lch` which adds colour, this removes
        it from the targeted luminance zones.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            shadow_desat: Shadow desaturation strength (0–1).
            highlight_desat: Highlight desaturation strength (0–1).

        Returns:
            (H, W, 3) uint8 BGR.
        """
        if shadow_desat <= 0.0 and highlight_desat <= 0.0:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import (
                bgr_f32_to_lch_f32,
                lch_f32_to_bgr_f32,
                negative_split_tone_lch as _nst,
            )
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import bgr_to_lch, lch_to_bgr, negative_split_tone_lch as _nst
            lch = bgr_to_lch(img_bgr)
        toned = _nst(lch, shadow_desat, highlight_desat)
        if is_float:
            return lch_f32_to_bgr_f32(toned)
        return lch_to_bgr(toned)

    def white_balance_lch(
        self,
        img_bgr: np.ndarray,
        temperature: float = 6500.0,
        tint: float = 0.0,
    ) -> np.ndarray:
        """White balance via LCH hue-shift on near‑neutral pixels.

        Maps Kelvin temperature to a warm/cool hue shift and applies
        ``tint`` as a green‑magenta axis shift. Operates primarily on
        low‑chroma pixels so saturated areas are not over‑corrected.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            temperature: Kelvin colour temperature (2000–50000).
                         6500 = neutral daylight. Lower = warmer.
            tint: Green‑magenta shift in [-100, 100]. Negative = green,
                  positive = magenta.

        Returns:
            (H, W, 3) uint8 BGR.
        """
        if abs(temperature - 6500.0) < 1e-3 and abs(tint) < 1e-4:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import bgr_f32_to_lch_f32, lch_f32_to_bgr_f32
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import bgr_to_lch, lch_to_bgr
            lch = bgr_to_lch(img_bgr)
        t = float(np.clip(temperature, 2000.0, 50000.0))
        tint_val = float(np.clip(tint, -100.0, 100.0))
        # Mired-based white balance correction
        # Mired = 1e6 / temperature (reciprocal Kelvin)
        # Neutral at 6500K ≈ 153.85 mired
        # Negative mired_delta → cool (high K), positive → warm (low K)
        MIRED_NEUTRAL = 1e6 / 6500.0  # ≈ 153.85
        mired_delta = (1e6 / max(t, 100.0)) - MIRED_NEUTRAL
        # Scale mired_delta to hue shift: ~30 mireds per ±50° hue
        # This gives smooth, monotonic correction with correct sign flip at 6500K
        hue_delta = mired_delta / MIRED_NEUTRAL * 100.0 * 0.3
        chroma = lch[:, :, 1]
        chroma_weight = np.clip(1.0 - chroma / 60.0, 0.0, 1.0)
        hue_delta = hue_delta * chroma_weight
        lch[:, :, 2] = np.mod(lch[:, :, 2] + hue_delta, 360.0)
        if abs(tint_val) > 1e-4:
            tint_hue = 150.0 if tint_val < 0 else 330.0
            tint_strength = abs(tint_val) / 100.0 * chroma_weight * 0.2
            delta = (tint_hue - lch[:, :, 2] + 180.0) % 360.0 - 180.0
            lch[:, :, 2] = np.mod(
                lch[:, :, 2] + delta * tint_strength, 360.0
            )
        if is_float:
            return lch_f32_to_bgr_f32(lch)
        return lch_to_bgr(lch)

    def channel_mixer_bw(
        self,
        img_bgr: np.ndarray,
        r_weight: float = 0.3,
        g_weight: float = 0.59,
        b_weight: float = 0.11,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ) -> np.ndarray:
        """Per‑channel black & white conversion with luminance controls.

        Standard luminance weights: R=0.299, G=0.587, B=0.114 (BT.601).
        Adjustable for creative B&W — boost reds for darker skies, boost
        green for lighter foliage.

        Args:
            img_bgr: (H, W, 3) uint8 BGR.
            r_weight: Red weight (BGR channel 2). Default 0.30.
            g_weight: Green weight (BGR channel 1). Default 0.59.
            b_weight: Blue weight (BGR channel 0). Default 0.11.
            brightness: Additive luminance shift in [-50, 50].
            contrast: Multiplicative contrast in [-50, 50].

        Returns:
            (H, W, 3) uint8 BGR (grayscale tri‑channel for compatibility).
        """
        f = img_bgr.astype(np.float32)
        total = r_weight + g_weight + b_weight
        if total < 1e-6:
            return np.zeros_like(img_bgr)
        r_w, g_w, b_w = r_weight / total, g_weight / total, b_weight / total
        gray = f[:, :, 2] * r_w + f[:, :, 1] * g_w + f[:, :, 0] * b_w
        if contrast != 0.0:
            c = float(np.clip(contrast, -50.0, 50.0)) / 50.0
            mid = 128.0
            gray = mid + (gray - mid) * (1.0 + c)
        gray = np.clip(gray + brightness, 0, 255)
        out_dtype = np.float32 if img_bgr.dtype == np.float32 else np.uint8
        return np.stack([gray, gray, gray], axis=-1).astype(out_dtype)

    # ------------------------------------------------------------------
    # Color transfer
    # ------------------------------------------------------------------

    @staticmethod
    def _reinhard_chroma(
        src: np.ndarray,
        mean_src: float,
        std_src: float,
        mean_ref: float,
        std_ref: float,
    ) -> np.ndarray:
        """Single-channel Reinhard chroma match (A/B only), ratio-clamped.

        Mirrors ``style_transfer.reinhard_transfer_masked``'s safety policy:
        skip near-flat source channels (std < 1e-3) to leave them unchanged,
        otherwise scale by std ratio clamped to [0.3, 3.0] to avoid crush.
        """
        if std_src < 1e-3:
            return src
        ratio = np.clip(std_ref / (std_src + 1e-6), 0.3, 3.0)
        return (src - mean_src) * ratio + mean_ref

    def color_transfer(
        self,
        img_bgr: np.ndarray,
        ref_bgr: Optional[np.ndarray],
        intensity: float = 1.0,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if ref_bgr is None: return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab_src = bgr_f32_to_lab_f32(img_bgr)
            ref_u8 = np.clip(ref_bgr * 255.0, 0, 255).astype(np.uint8) if ref_bgr.dtype == np.float32 else ref_bgr
            lab_ref = bgr_f32_to_lab_f32(ref_u8.astype(np.float32)) if ref_bgr.dtype != np.uint8 else cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        else:
            lab_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_l = lab_src[:, :, 0].copy(); src_a = lab_src[:, :, 1]; src_b = lab_src[:, :, 2]
        ref_a = lab_ref[:, :, 1]; ref_b = lab_ref[:, :, 2]
        mean_a_src = src_a.mean(); std_a_src = src_a.std()
        mean_b_src = src_b.mean(); std_b_src = src_b.std()
        mean_a_ref = ref_a.mean(); std_a_ref = ref_a.std()
        mean_b_ref = ref_b.mean(); std_b_ref = ref_b.std()
        # Match A/B (chroma) channels only; L (lightness) is copied verbatim.
        # Symmetric with style_transfer.reinhard_transfer_masked: clamp the
        # std ratio to [0.3, 3.0] and skip a near-flat source channel to avoid
        # channel crush (previously unbounded -> hard clip at [0,255]).
        a_result = self._reinhard_chroma(src_a, mean_a_src, std_a_src, mean_a_ref, std_a_ref)
        b_result = self._reinhard_chroma(src_b, mean_b_src, std_b_src, mean_b_ref, std_b_ref)
        lab_result = np.stack([src_l, a_result, b_result], axis=2)
        if is_float:
            result = lab_f32_to_bgr_f32(np.clip(lab_result, 0, 255))
            if intensity < 1.0:
                result = img_bgr * (1.0 - intensity) + result * intensity
            if mask is not None:
                return blend_masked(img_bgr, result, mask)
            return result
        lab_result = np.clip(lab_result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        if intensity < 1.0: result = cv2.addWeighted(img_bgr, 1.0 - intensity, result, intensity, 0)
        if mask is not None: return blend_masked(img_bgr, result, mask)
        return result

    def apply_soft_light_layer(
        self,
        img_bgr: np.ndarray,
        layer_bgr: Optional[np.ndarray] = None,
        opacity: float = 0.5,
        blur_radius: int = 0,
    ) -> np.ndarray:
        """Non-destructive grading via soft-light blend.

        If ``layer_bgr`` is given, it is blended over ``img_bgr`` using
        soft-light at ``opacity`` strength.  If ``layer_bgr`` is None,
        a self-blended soft-light layer is used (same image blurred at
        ``blur_radius``) — this is the classic "detail-preserving
        contrast" technique used in DaVinci Resolve and Photoshop.

        Args:
            img_bgr: (H, W, 3) uint8 BGR base image.
            layer_bgr: Optional overlay layer.  If None, the base is
                Gaussian-blurred to create the layer.
            opacity: Blend strength in [0, 1].
            blur_radius: Gaussian blur radius for self-blend mode.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if opacity <= 0.0:
            return img_bgr
        from .utils import soft_light_blend
        if layer_bgr is not None:
            sl = soft_light_blend(img_bgr, layer_bgr)
        else:
            h, w = img_bgr.shape[:2]
            ksize = max(blur_radius, int(min(h, w) * 0.02)) | 1
            blurred = _large_sigma_blur(img_bgr, ksize)
            sl = soft_light_blend(img_bgr, blurred)
        if opacity >= 1.0:
            return sl
        return cv2.addWeighted(img_bgr, 1.0 - opacity, sl, opacity, 0)

    def _add_haze(
        self,
        img_bgr: np.ndarray,
        strength: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if strength <= 0: return img_bgr
        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float32)
        ksize = max(int(min(h, w) * 0.08), 25) | 1
        blurred = _large_sigma_blur(img_f, ksize)
        haze_tint = np.array([245.0, 230.0, 240.0], dtype=np.float32) / 255.0
        haze_layer = blurred * haze_tint
        screen = screen_blend(img_f, haze_layer)
        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2: m_f = m_f[:, :, np.newaxis]
            haze_factor = m_f * strength
        else:
            haze_factor = strength
        result = img_f * (1.0 - haze_factor) + screen * haze_factor
        lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_lift = 25.0 * strength
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        if mask is not None:
            m_2d = squeeze_mask(m_f)
            lab[:, :, 0] = np.clip(l + shadow_mask * (shadow_lift * m_2d), 0, 255)
        else:
            lab[:, :, 0] = np.clip(l + shadow_mask * shadow_lift, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_add_haze(
        self,
        img_f: np.ndarray,
        strength: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if strength <= 0:
            return img_f
        h, w = img_f.shape[:2]
        img_255 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.float32)
        ksize = max(int(min(h, w) * 0.08), 25) | 1
        blurred = _large_sigma_blur(img_255, ksize)
        haze_tint = np.array([245.0, 230.0, 240.0], dtype=np.float32) / 255.0
        haze_layer = blurred * haze_tint
        screen = screen_blend(img_255, haze_layer)
        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            haze_factor = m_f * strength
        else:
            haze_factor = strength
        result_255 = img_255 * (1.0 - haze_factor) + screen * haze_factor
        result_255 = np.clip(result_255, 0.0, 255.0).astype(np.float32)
        result_f = np.clip(result_255 / 255.0, 0.0, 1.0).astype(np.float32)
        lab = _bgr_f_to_lab_u8_conv(result_f)
        l = lab[:, :, 0]
        shadow_lift = 25.0 * strength
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        if mask is not None:
            m_2d = squeeze_mask(m_f)
            lab[:, :, 0] = np.clip(l + shadow_mask * (shadow_lift * m_2d), 0, 255)
        else:
            lab[:, :, 0] = np.clip(l + shadow_mask * shadow_lift, 0, 255)
        out_f = _lab_u8_conv_to_bgr_f(np.clip(lab, 0.0, 255.0).astype(np.float32))
        return np.clip(out_f, 0.0, 1.0).astype(np.float32)

    def _add_sparkles(self, img_bgr: np.ndarray, opacity: float) -> np.ndarray:
        if opacity <= 0: return img_bgr
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        local_max = cv2.dilate(gray, kernel)
        peaks = (gray == local_max) & (thresh > 0)
        y_indices, x_indices = np.where(peaks)
        if len(x_indices) == 0: return img_bgr
        num_sparkles = min(len(x_indices), 50)
        indices = np.random.choice(len(x_indices), num_sparkles, replace=False)
        sparkle_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        for idx in indices:
            cx, cy = x_indices[idx], y_indices[idx]
            size = np.random.randint(6, 13)
            color = (255, 255, 255)
            cv2.circle(sparkle_overlay, (cx, cy), 1, color, -1)
            cv2.line(sparkle_overlay, (cx - size, cy), (cx + size, cy), (240, 245, 255), 1)
            cv2.line(sparkle_overlay, (cx, cy - size), (cx, cy + size), (240, 245, 255), 1)
            diag_size = int(size * 0.6)
            cv2.line(sparkle_overlay, (cx - diag_size, cy - diag_size), (cx + diag_size, cy + diag_size), (220, 230, 255), 1)
            cv2.line(sparkle_overlay, (cx - diag_size, cy + diag_size), (cx + diag_size, cy - diag_size), (220, 230, 255), 1)
        sparkle_overlay = cv2.GaussianBlur(sparkle_overlay, (3, 3), 0)
        img_f = img_bgr.astype(np.float32)
        sparkle_f = sparkle_overlay.astype(np.float32)
        screened = screen_blend(img_f, sparkle_f * opacity)
        return np.clip(screened, 0, 255).astype(np.uint8)

    # ---- Stage C4: 透明感 / 空気感 Finish Pack ----

    def fade_toe(
        self,
        img_bgr: np.ndarray,
        strength: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Lifted-black with hue-locked toe (L-only fade in LAB).

        Lifts shadows while preserving hue by modifying only the L channel in LAB.
        Classic RGB-channel fade shifts shadows blue-green unintentionally; this
        approach avoids that by locking a/b untouched.

        Args:
            img_bgr: uint8 BGR image.
            strength: Fade strength, 0-1 (0=no-op, 1=full lift).
            mask: Optional per-pixel mask to scope the effect (1=apply, 0=preserve).

        Returns:
            uint8 BGR image with lifted shadows.
        """
        if strength <= 0:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        # Shadow threshold and weight: pixels with L < 50 get max lift, linear falloff to L=50
        shadow_weight = np.clip(1.0 - l / 50.0, 0, 1)
        # Lift amount: ~20 on 0-255 L scale (matches _lift_shadows magnitude)
        lift_amount = 20.0
        l_new = np.clip(l + shadow_weight * strength * lift_amount, 0, 255)
        lab[:, :, 0] = l_new
        if is_float:
            result = lab_f32_to_bgr_f32(lab)
        else:
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            result = blend_masked(img_bgr, result, m_f)
        return result

    def highlight_drift(
        self,
        img_bgr: np.ndarray,
        strength: float,
        skin_state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Bounded hue rotation of highlights toward cyan, skin-protected.

        Rotates hue of bright pixels (L > 75 on 0-100 L* scale) toward cyan (180°)
        by up to 10-15° (scaled by strength). If skin_state or mask is provided,
        skin regions (high chroma) are protected from the rotation.

        Args:
            img_bgr: uint8 BGR image.
            strength: Rotation strength, 0-1.
            skin_state: Optional SkinState object for chroma-based skin detection.
            mask: Optional per-pixel mask (1=skin=protect, 0=non-skin=full effect).

        Returns:
            uint8 BGR image with hue-rotated highlights.
        """
        if strength <= 0:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            from .color_space import bgr_f32_to_lch_f32, lch_f32_to_bgr_f32
            lch = bgr_f32_to_lch_f32(img_bgr)
        else:
            from .color_space import bgr_to_lch, lch_to_bgr
            lch = bgr_to_lch(img_bgr)
        l_chan = lch[:, :, 0]
        h_chan = lch[:, :, 2]
        c_chan = lch[:, :, 1]
        # Highlight threshold on 0-100 L* scale; pixels with L > 75 get rotation
        highlight_mask = np.clip((l_chan - 75.0) / 25.0, 0, 1)
        # Max hue rotation: 10-15°, scaled by strength
        max_rotation = 12.0  # degrees
        rotation = max_rotation * strength
        # Protect skin: if mask provided, use it; else use a chroma-based ramp.
        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            protect_factor = m_f  # mask=1 (skin) -> protect_factor=1 (fully protected)
        else:
            # Chroma-based skin ramp in LCh(ab) C units (0-~130 scale for sRGB gamut).
            # Below chroma_floor: achromatic/background, unprotected (protect_factor=0).
            # At/above chroma_ceiling: clearly chromatic (skin-range and beyond),
            # fully protected (protect_factor=1). This codebase's own
            # `color_space.skin_mask_lch` uses chroma_min=8.0 as its skin-chroma
            # gate, so 8.0 is reused here as the floor (below it = not skin).
            # chroma_ceiling=20.0 chosen because typical measured skin chroma in
            # LCh(ab) for well-lit portraits runs roughly 15-35 (moderately
            # saturated warm tones); by C=20 a pixel is unambiguously chromatic
            # (skin or a strongly colored object), so full protection is safe —
            # ramping over [8, 20] avoids a hard step at the skin_mask_lch floor.
            chroma_floor = 8.0
            chroma_ceiling = 20.0
            protect_factor = np.clip(
                (c_chan - chroma_floor) / (chroma_ceiling - chroma_floor), 0, 1
            )[:, :, np.newaxis]
        # Apply rotation, attenuated by protection
        h_rotated = h_chan + rotation * highlight_mask * (1.0 - protect_factor[:, :, 0])
        h_rotated = np.clip(h_rotated, 0, 360)
        lch[:, :, 2] = h_rotated
        if is_float:
            return lch_f32_to_bgr_f32(lch)
        return lch_to_bgr(lch)

    def airy_haze(
        self,
        img_bgr: np.ndarray,
        strength: float,
        l_threshold: float = 200.0,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """L-threshold-scoped haze with person_mask-aware distance falloff.

        Computes a soft glow from pixels above L_threshold, screen-blends it with
        the image, and scales opacity by person_mask (background gets more air,
        subject gets less for sharpness preservation). Distance falloff ensures
        the glow fades naturally from highlights outward.

        Args:
            img_bgr: uint8 BGR image.
            strength: Haze strength, 0-1.
            l_threshold: LAB L threshold (0-255 scale) above which haze is computed.
                         Default 200 creates haze only in very bright areas.
            person_mask: Optional per-pixel mask (1=subject, 0=background).
                         Scales glow opacity: background gets 100%, subject gets 40%.

        Returns:
            uint8 BGR image with L-threshold-scoped haze effect.
        """
        if strength <= 0:
            return img_bgr
        h, w = img_bgr.shape[:2]
        is_float = img_bgr.dtype == np.float32
        img_f = img_bgr.astype(np.float32) if not is_float else img_bgr
        # Compute haze source from blurred image
        ksize = max(int(min(h, w) * 0.02), 7) | 1
        blurred = _large_sigma_blur(img_f, ksize)
        # Extract LAB L to threshold
        if is_float:
            lab_u8 = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab_u8[:, :, 0].astype(np.float32)
        # Create highlight mask from L_threshold
        hl_mask = np.clip((l_chan - l_threshold) / 20.0, 0, 1)[:, :, np.newaxis]
        # Screen-blend the blurred image where highlights exist
        screen = screen_blend(img_f, blurred)
        result = img_f * (1.0 - hl_mask * strength) + screen * (hl_mask * strength)
        # Apply person_mask: background (person_mask~0) gets full effect,
        # subject (person_mask~1) gets attenuated (40% opacity)
        if person_mask is not None:
            pm_f = normalize_mask(person_mask)
            if pm_f.ndim == 2:
                pm_f = pm_f[:, :, np.newaxis]
            # glow_opacity = (1.0 - pm_f * 0.6) means:
            # person_mask=0 (bg) -> opacity=1.0 (full)
            # person_mask=1 (fg) -> opacity=0.4 (attenuated)
            glow_opacity = 1.0 - pm_f * 0.6
            result = img_f + (result - img_f) * glow_opacity * strength
        if is_float:
            return np.clip(result, 0, 255).astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def clarity_split(
        self,
        img_bgr: np.ndarray,
        negative_strength: float,
        positive_strength: float,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Negative clarity on form band + positive micro-contrast on texture band.

        Uses two guided-filter passes at different radii:
        - Large radius (form band): guided-filter base, inverted (subtracted) for
          negative local-contrast reduction.
        - Small radius (texture band): high-pass detail, boosted for positive
          micro-contrast enhancement.

        The result is soft-yet-detailed, a signature of JP portrait finishing.

        Args:
            img_bgr: uint8 BGR image.
            negative_strength: Form-band clarity reduction, 0-1.
            positive_strength: Texture-band clarity boost, 0-1.
            mask: Optional per-pixel mask to scope the effect.

        Returns:
            uint8 BGR image with split-clarity processing.
        """
        if negative_strength <= 0 and positive_strength <= 0:
            return img_bgr
        is_float = img_bgr.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        h, w = img_bgr.shape[:2]
        # Large radius for form band (low-frequency local contrast).
        # NOTE: 0.04*min(h,w) (matching _add_clarity's positive-clarity radius
        # convention) was measured to leave the guided-filter base nearly
        # identical to l_chan for genuine form-scale structure (~30-80px
        # period) — at that radius detail_form's std was ~5% of l_chan's std,
        # so no amount of scaling the correction term could produce a
        # meaningful reduction (this was the actual root cause of the
        # near-inert negative-clarity bug, not just an undersized constant).
        # 0.25*min(h,w) (min 40px) was verified empirically to isolate
        # genuine form-band content and, combined with form_reduction_k=1.0,
        # delivers 40-90%+ form-band local-contrast reduction across test
        # patterns (52.9% on a period-30px sine-blob probe measured with a
        # sigma=15 Gaussian, matching the plan's "negative clarity on the
        # form band" intent).
        r_form = max(int(min(h, w) * 0.25), 40)
        eps = 0.05
        l_norm = l_chan / 255.0
        # Guided filter on form band
        base_form = self._guided_filter(l_norm, l_norm, r_form, eps) * 255.0
        detail_form = l_chan - base_form
        # Small radius for texture band (high-frequency micro-contrast)
        r_texture = max(int(min(h, w) * 0.008), 2)
        base_texture = self._guided_filter(l_norm, l_norm, r_texture, eps) * 255.0
        detail_texture = l_chan - base_texture
        # Composite: reduce form-band contrast, boost texture-band detail.
        # Negative pass is a direct subtractive correction (mirrors
        # _add_clarity's positive `base + detail * (1.0 + strength)` pattern,
        # but shrinking detail_form's contribution instead of growing it).
        # form_reduction_k=1.0 means negative_strength=1.0 fully removes the
        # form-band detail signal (l_new == base_form), the maximal "flatten
        # the form band" reading of the spec; negative_strength=0 leaves
        # detail_form fully intact (l_new == l_chan, a true no-op).
        form_reduction_k = 1.0
        l_new = base_form + detail_form * (1.0 - negative_strength * form_reduction_k)
        # Texture-band boost mirrors _add_clarity's `detail * (1.0 + strength)`
        # pattern: at positive_strength=1.0 the texture-band detail signal is
        # doubled, a strength on par with _add_clarity's own scaling.
        l_new = l_new + detail_texture * positive_strength
        l_new = np.clip(l_new, 0, 255)
        lab[:, :, 0] = l_new
        if is_float:
            result = lab_f32_to_bgr_f32(lab)
        else:
            lab_u8 = lab.astype(np.uint8)
            result = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            result = blend_masked(img_bgr, result, m_f)
        return result

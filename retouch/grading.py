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
from .utils import apply_curve, blend_masked, normalize_mask, screen_blend, squeeze_mask

from . import skin_protect
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
    ) -> np.ndarray:
        """Apply a colour grading preset to an image.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
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

        Returns:
            (H, W, 3) uint8 BGR image.
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

        original_u8 = img_bgr
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
                    bgr_u8 = np.clip(r * 255.0, 0, 255).astype(np.uint8)
                    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
                    skin_bool = skin_mask > 0.3
                    lab[:, :, 2] = np.where(skin_bool, lab[:, :, 2] + b_delta, lab[:, :, 2])
                    lab[:, :, 1] = np.where(skin_bool, lab[:, :, 1] + a_delta, lab[:, :, 1])
                    lab = np.clip(lab, 0, 255)
                    r = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
            if abs(settings.get("saturation_boost", 0)) > 0.001:
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

            return to_uint8(r)

        if skin_protect_strength > 0:
            result = skin_protect.protect_skin(result, _color_ops, skin_protect_strength)
        else:
            result = _color_ops(result)

        if "halation" in settings and not skip_post_effects:
            h_conf = settings["halation"]
            if isinstance(h_conf, (int, float)):
                result = self._add_halation(result, intensity=float(h_conf))
            else:
                result = self._add_halation(
                    result, threshold=h_conf.get("threshold", 200),
                    radius=h_conf.get("radius", 15), intensity=h_conf.get("intensity", 0.3)
                )

        if settings.get("vignette", 0) > 0:
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

        if intensity < 1.0:
            result = cv2.addWeighted(original_u8, 1.0 - intensity, result, intensity, 0)

        return result

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
        img_f = img_bgr.astype(np.float32)
        h, w = img_bgr.shape[:2]
        ksize = max(int(min(h, w) * 0.025), 7) | 1
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)

        if tint is not None:
            tint_arr = np.array(tint, dtype=np.float32)
            if tint_arr.max() <= 1.0: tint_arr = tint_arr * 255.0
            tint_layer = np.ones_like(blurred) * tint_arr
            blurred = cv2.addWeighted(blurred, 0.7, tint_layer, 0.3, 0)

        screen = screen_blend(img_f, blurred)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        highlight_mask = np.clip((l_chan - 225.0) / 20.0, 0, 1)[:, :, np.newaxis]

        if mask is not None:
            m_f = normalize_mask(mask)
            if m_f.ndim == 2: m_f = m_f[:, :, np.newaxis]
            highlight_mask = highlight_mask * m_f

        result = img_f * (1.0 - highlight_mask * opacity) + screen * (highlight_mask * opacity)
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = _apply_curve_f(lab[:, :, 0], curve_points)
        out_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

    def _lift_shadows(self, img: np.ndarray, lift: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        lab[:, :, 0] = np.clip(l + shadow_mask * lift, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_lift_shadows(self, img_f: np.ndarray, lift: float) -> np.ndarray:
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        lab[:, :, 0] = np.clip(l + shadow_mask * lift, 0, 255)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

    def _adjust_warmth(self, img: np.ndarray, warmth: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth * 30, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth * 10, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _F_adjust_warmth(self, img_f: np.ndarray, warmth: float) -> np.ndarray:
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth * 30, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth * 10, 0, 255)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

    def _adjust_saturation(self, img: np.ndarray, boost: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        factor = 1.0 + boost * (1.0 - s / 255.0)
        hsv[:, :, 1] = np.clip(s * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _F_adjust_saturation(self, img_f: np.ndarray, boost: float) -> np.ndarray:
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        factor = 1.0 + boost * (1.0 - s / 255.0)
        hsv[:, :, 1] = np.clip(s * factor, 0, 255)
        out_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out_u8.astype(np.float32) / 255.0

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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_weight = np.clip(1.0 - l / 128.0, 0, 1)[:, :, np.newaxis]
        highlight_weight = np.clip((l - 128.0) / 128.0, 0, 1)[:, :, np.newaxis]
        shadow_ab = np.array(tones["shadows"], dtype=np.float32)
        highlight_ab = np.array(tones["highlights"], dtype=np.float32)
        ab = lab[:, :, 1:3]
        ab = ab + (shadow_ab - 128) * shadow_weight * 0.15
        ab = ab + (highlight_ab - 128) * highlight_weight * 0.15
        lab[:, :, 1:3] = np.clip(ab, 0, 255)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        out_f = out_u8.astype(np.float32) / 255.0
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        h, w = bgr_u8.shape[:2]
        r = max(int(min(h, w) * 0.015), 5)
        eps = 0.02
        l_norm = l_chan / 255.0
        base = self._guided_filter(l_norm, l_norm, r, eps) * 255.0
        detail = l_chan - base
        l_new = np.clip(base + detail * (1.0 + strength), 0, 255)
        lab[:, :, 0] = l_new.astype(np.uint8)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

    def _add_vignette(self, img: np.ndarray, strength: float) -> np.ndarray:
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - (dist / max_dist) ** 2 * strength
        vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        ranges = {"red": [(0, 10), (170, 180)], "orange": [(10, 25)], "yellow": [(25, 35)],
                  "green": [(35, 80)], "cyan": [(80, 105)], "blue": [(105, 140)], "magenta": [(140, 170)]}
        for color, shift_deg in shifts_dict.items():
            if color not in ranges or shift_deg == 0: continue
            shift_cv = float(shift_deg) / 2.0
            mask = np.zeros_like(h, dtype=bool)
            for low, high in ranges[color]: mask |= (h >= low) & (h <= high)
            hsv[:, :, 0] = np.where(mask, (h + shift_cv) % 180, h)
        out_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out_u8.astype(np.float32) / 255.0

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
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        halation_color = np.zeros_like(img_bgr, dtype=np.uint8)
        halation_color[:, :, 2] = thresh
        halation_color[:, :, 1] = (thresh * 0.45).astype(np.uint8)
        ksize = radius | 1
        blurred_halation = cv2.GaussianBlur(halation_color, (ksize, ksize), 0).astype(np.float32)
        img_f = img_bgr.astype(np.float32)
        screen = screen_blend(img_f, blurred_halation * intensity)
        return np.clip(screen, 0, 255).astype(np.uint8)

    def _add_grain(self, img_bgr: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0: return img_bgr
        h, w = img_bgr.shape[:2]
        gw, gh = max(w // 2, 64), max(h // 2, 64)
        noise = np.random.normal(0, 255.0 * strength, (gh, gw)).astype(np.float32)
        noise_scaled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noise_blurred = cv2.GaussianBlur(noise_scaled, (3, 3), 0.5)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        weight = np.exp(-((l_chan - 128.0) ** 2) / (2.0 * 64.0 ** 2))
        weighted_noise = (noise_blurred * weight)[:, :, np.newaxis]
        img_f = img_bgr.astype(np.float32)
        return np.clip(img_f + weighted_noise, 0, 255).astype(np.uint8)

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
        gray = np.array([128, 128, 128], dtype=np.float32)
        wb = np.clip(gray * np.array([b_mult, g_mult, r_mult]), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        gray_u8 = gray.astype(np.uint8).reshape(1, 1, 3)
        gray_lab = cv2.cvtColor(gray_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        wb_lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_off = wb_lab[0, 0, 1] - gray_lab[0, 0, 1]
        b_off = wb_lab[0, 0, 2] - gray_lab[0, 0, 2]
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_off, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_off, 0, 255)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
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
        out_u8 = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out_u8.astype(np.float32) / 255.0

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
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
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
        out_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out_f = out_u8.astype(np.float32) / 255.0
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
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
        out_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out_u8.astype(np.float32) / 255.0

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
        from .color_space import bgr_to_lch, lch_to_bgr
        lch = bgr_to_lch(img_bgr)
        if abs(hue_shift) > 1e-4:
            lch[:, :, 2] = np.mod(lch[:, :, 2] + hue_shift, 360.0)
        if abs(sat_scale - 1.0) > 1e-4:
            lch[:, :, 1] = lch[:, :, 1] * sat_scale
        if abs(lum_shift) > 1e-4:
            lch[:, :, 0] = np.clip(lch[:, :, 0] + lum_shift, 0.0, 100.0)
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
        from .color_space import (
            bgr_to_lch,
            lch_to_bgr,
            adjust_hue_range,
            adjust_chroma_range,
            adjust_luminance_range,
        )
        hue_center, hue_width = self._HSL_CHANNELS[channel]
        lch = bgr_to_lch(img_bgr)
        if abs(hue_shift) > 1e-4:
            lch = adjust_hue_range(lch, hue_center, hue_width, hue_shift)
        if abs(sat_scale - 1.0) > 1e-4:
            lch = adjust_chroma_range(lch, hue_center, hue_width, sat_scale)
        if abs(lum_shift) > 1e-4:
            lch = adjust_luminance_range(lch, hue_center, hue_width, lum_shift)
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
        from .color_space import bgr_to_lch, lch_to_bgr, split_tone_lch as _st
        lch = bgr_to_lch(img_bgr)
        toned = _st(lch, shadow_hue, shadow_sat, highlight_hue, highlight_sat, balance)
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
        from .color_space import bgr_to_lch, lch_to_bgr
        lch = bgr_to_lch(img_bgr)
        t = float(np.clip(temperature, 2000.0, 50000.0))
        tint_val = float(np.clip(tint, -100.0, 100.0))
        kelvin_hue = 60.0 - (t - 2000.0) / 48000.0 * 90.0
        kelvin_hue = np.mod(kelvin_hue, 360.0)
        chroma = lch[:, :, 1]
        chroma_weight = np.clip(1.0 - chroma / 60.0, 0.0, 1.0)
        hue_delta = (kelvin_hue - 30.0) * chroma_weight * 0.3
        lch[:, :, 2] = np.mod(lch[:, :, 2] + hue_delta, 360.0)
        if abs(tint_val) > 1e-4:
            tint_hue = 150.0 if tint_val < 0 else 330.0
            tint_strength = abs(tint_val) / 100.0 * chroma_weight * 0.2
            lch[:, :, 2] = np.mod(
                lch[:, :, 2] + (tint_hue - lch[:, :, 2]) * tint_strength, 360.0
            )
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
        return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Color transfer
    # ------------------------------------------------------------------

    def color_transfer(
        self,
        img_bgr: np.ndarray,
        ref_bgr: Optional[np.ndarray],
        intensity: float = 1.0,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if ref_bgr is None: return img_bgr
        lab_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_l = lab_src[:, :, 0].copy(); src_a = lab_src[:, :, 1]; src_b = lab_src[:, :, 2]
        ref_a = lab_ref[:, :, 1]; ref_b = lab_ref[:, :, 2]
        mean_a_src = src_a.mean(); std_a_src = src_a.std()
        mean_b_src = src_b.mean(); std_b_src = src_b.std()
        mean_a_ref = ref_a.mean(); std_a_ref = ref_a.std()
        mean_b_ref = ref_b.mean(); std_b_ref = ref_b.std()
        a_result = (src_a - mean_a_src) * (std_a_ref / (std_a_src + 1e-6)) + mean_a_ref
        b_result = (src_b - mean_b_src) * (std_b_ref / (std_b_src + 1e-6)) + mean_b_ref
        lab_result = np.stack([src_l, a_result, b_result], axis=2)
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
            blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)
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
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
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
        bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        img_255 = bgr_u8.astype(np.float32)
        ksize = max(int(min(h, w) * 0.08), 25) | 1
        blurred = cv2.GaussianBlur(img_255, (ksize, ksize), 0)
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
        lab = cv2.cvtColor(np.clip(result_255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_lift = 25.0 * strength
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        if mask is not None:
            m_2d = squeeze_mask(m_f)
            lab[:, :, 0] = np.clip(l + shadow_mask * (shadow_lift * m_2d), 0, 255)
        else:
            lab[:, :, 0] = np.clip(l + shadow_mask * shadow_lift, 0, 255)
        out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out_u8.astype(np.float32) / 255.0

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

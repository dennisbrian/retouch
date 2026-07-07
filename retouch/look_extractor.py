"""F6 — Look-from-reference → editable preset extraction.

Reverse-engineer a finishing "look" from a reference image into a set of
processing parameters that reproduce the look when applied to a base image.
The extracted params live in two forms:

  * ``engine_params`` — a flat dict of kwargs that can be passed directly to
    ``RetouchEngine.process(**params)`` (tone, color, film, grain, vignette).
  * ``preset`` — a ColorGrader-style preset dict (curves / white_balance /
    split_tone_three_way / hsl_adjustments) suitable for saving to
    ``presets/`` and re-use as a normal grading preset.

The extractor is purely histogram/statistics-driven — no face detection, no
iterative optimisation — so it runs in milliseconds and is deterministic. It
is intentionally a *starting point*: the plan doc (F6 §4) frames the output
as a fit to be refined by the user, not an exact inverse.

Internal dtype is float32 throughout; uint8 inputs are converted at the
boundary and never used in pixel arithmetic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _to_f32_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"LookExtractor expects HxWx3 BGR, got shape {img.shape}"
        )
    if img.dtype == np.float32:
        return np.clip(img, 0.0, 255.0).astype(np.float32)
    return img.astype(np.float32)


def _bgr_f32_to_lab_u8_conv(img_f: np.ndarray) -> np.ndarray:
    """Convert float32 BGR [0,255] to float32 LAB in uint8 convention.

    OpenCV's ``COLOR_BGR2LAB`` on float input expects BGR in [0,1] and
    returns L in [0,100], a/b in [-128,127]. This normalizes the input,
    converts, then rescales to the uint8 convention (L in [0,255],
    a/b in [0,255] with 128=neutral) so the rest of the codebase math
    (thresholds at L<85, L>170, etc.) works correctly.
    """
    img01 = np.clip(img_f * (1.0 / 255.0), 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(img01, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * 2.55
    lab[:, :, 1] = lab[:, :, 1] + 128.0
    lab[:, :, 2] = lab[:, :, 2] + 128.0
    return lab


def _bgr_to_lab_f32(img_f: np.ndarray) -> np.ndarray:
    return _bgr_f32_to_lab_u8_conv(img_f)


class LookExtractor:
    """Extract an editable look preset from a reference image.

    Two extraction modes:
      * **Unpaired** (reference only): measure absolute look characteristics
        against neutral expectations — produces an absolute grade.
      * **Paired** (base + reference): measure the delta between an original
        and an edited image — produces a relative grade that, when applied to
        the base, approximates the reference.

    The extractor never touches skin/face machinery; it works on global tone
    and color statistics only. This keeps it fast and side-effect-free.
    """

    NEUTRAL_L_CURVE: List[List[int]] = [
        [0, 0], [64, 64], [128, 128], [192, 192], [255, 255],
    ]

    def __init__(self, downsample_dim: int = 1024) -> None:
        self.downsample_dim = downsample_dim

    def extract(
        self,
        reference_img: np.ndarray,
        base_img: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract look parameters from a reference image.

        Args:
            reference_img: (H, W, 3) BGR uint8 or float32 [0, 255] image
                representing the target look.
            base_img: Optional (H, W, 3) BGR original image. When provided,
                the extractor measures the delta between base and reference;
                when omitted, it measures absolute statistics against a
                neutral mid-gray expectation.

        Returns:
            Dict with keys:
              - ``engine_params``: flat kwargs for ``engine.process(**...)``
              - ``preset``: ColorGrader-style preset dict
              - ``mode``: "paired" or "unpaired"
        """
        ref_f = self._prepare(reference_img)
        if base_img is not None:
            base_f = self._prepare(base_img)
            if base_f.shape[:2] != ref_f.shape[:2]:
                ref_f = cv2.resize(
                    ref_f,
                    (base_f.shape[1], base_f.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            mode = "paired"
        else:
            base_f = None
            mode = "unpaired"

        tone = self._extract_tone(base_f, ref_f)
        color = self._extract_color(base_f, ref_f)
        film = self._extract_film_density(base_f, ref_f)
        grain = self._extract_grain(ref_f)
        vignette = self._extract_vignette(ref_f)

        engine_params: Dict[str, Any] = {}
        engine_params.update(tone)
        engine_params.update(color)
        engine_params.update(film)
        engine_params.update(grain)
        engine_params.update(vignette)

        preset = self._build_preset(base_f, ref_f)

        return {
            "engine_params": engine_params,
            "preset": preset,
            "mode": mode,
        }

    def extract_to_preset(
        self,
        reference_img: np.ndarray,
        base_img: Optional[np.ndarray] = None,
        name: str = "extracted_look",
        save: bool = True,
    ) -> Dict[str, Any]:
        """Extract and optionally save a preset JSON to the presets directory.

        Convenience wrapper around :meth:`extract` for the common case where
        only the grading preset is wanted on disk.
        """
        result = self.extract(reference_img, base_img)
        preset = result["preset"]
        preset["description"] = f"extracted from {name}"
        if save:
            self._save_preset(preset, name)
        return preset

    # ------------------------------------------------------------------
    # Internal: preparation
    # ------------------------------------------------------------------

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        img_f = _to_f32_bgr(img)
        h, w = img_f.shape[:2]
        m = min(h, w)
        if m > self.downsample_dim:
            scale = float(self.downsample_dim) / m
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img_f = cv2.resize(img_f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img_f, dtype=np.float32)

    # ------------------------------------------------------------------
    # Tone: shadows / highlights / whites / blacks
    # ------------------------------------------------------------------

    def _extract_tone(
        self,
        base_f: Optional[np.ndarray],
        ref_f: np.ndarray,
    ) -> Dict[str, float]:
        ref_lab = _bgr_to_lab_f32(ref_f)
        ref_l = ref_lab[:, :, 0]

        if base_f is not None:
            base_lab = _bgr_to_lab_f32(base_f)
            base_l = base_lab[:, :, 0]
        else:
            base_l = np.array([128.0], dtype=np.float32)

        ref_p = np.percentile(ref_l, [5, 25, 50, 75, 95]).astype(np.float32)
        base_p = np.percentile(base_l, [5, 25, 50, 75, 95]).astype(np.float32)

        def _shift(ref_val: float, base_val: float, gain: float = 1.0) -> float:
            return float(np.clip((ref_val - base_val) * gain, -100.0, 100.0))

        shadows = _shift(ref_p[0], base_p[0], 0.5)
        blacks = _shift(ref_p[0], base_p[0], 0.3)
        highlights = _shift(ref_p[4], base_p[4], 0.5)
        whites = _shift(ref_p[4], base_p[4], 0.3)

        if base_f is not None:
            ref_range = float(ref_p[4] - ref_p[0])
            base_range = float(base_p[4] - base_p[0])
            contrast = float(
                np.clip((ref_range / max(base_range, _EPS) - 1.0) * 50.0, -50.0, 50.0)
            )
            brightness = float(np.clip(ref_p[2] - base_p[2], -100.0, 100.0))
        else:
            contrast = float(
                np.clip((ref_p[4] - ref_p[0] - 180.0) * 0.2, -50.0, 50.0)
            )
            brightness = float(np.clip(ref_p[2] - 128.0, -100.0, 100.0))

        return {
            "shadows": round(shadows, 2),
            "highlights": round(highlights, 2),
            "whites": round(whites, 2),
            "blacks": round(blacks, 2),
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
        }

    # ------------------------------------------------------------------
    # Color: warmth / saturation / split tone
    # ------------------------------------------------------------------

    def _extract_color(
        self,
        base_f: Optional[np.ndarray],
        ref_f: np.ndarray,
    ) -> Dict[str, float]:
        ref_lab = _bgr_to_lab_f32(ref_f)
        ref_l = ref_lab[:, :, 0]
        ref_a = ref_lab[:, :, 1] - 128.0
        ref_b = ref_lab[:, :, 2] - 128.0

        if base_f is not None:
            base_lab = _bgr_to_lab_f32(base_f)
            base_b_mean = float(np.mean(base_lab[:, :, 2] - 128.0))
            base_a_mean = float(np.mean(base_lab[:, :, 1] - 128.0))

            base_hsv = cv2.cvtColor(base_f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            base_sat_mean = float(np.mean(base_hsv[:, :, 1]))
        else:
            base_b_mean = 0.0
            base_a_mean = 0.0
            base_sat_mean = 128.0

        ref_b_mean = float(np.mean(ref_b))
        ref_a_mean = float(np.mean(ref_a))

        ref_hsv = cv2.cvtColor(ref_f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_sat_mean = float(np.mean(ref_hsv[:, :, 1]))

        warmth = float(np.clip((ref_b_mean - base_b_mean) / 30.0, -1.0, 1.0))
        saturation = float(np.clip(ref_sat_mean - base_sat_mean, -100.0, 100.0))

        engine_params: Dict[str, float] = {
            "saturation": round(saturation, 2),
        }

        if abs(warmth) > 0.05:
            engine_params["white_balance_kelvin"] = int(
                np.clip(round(6500 - warmth * 1500), 2000, 12000)
            )

        if base_f is not None:
            base_lab = _bgr_to_lab_f32(base_f)
            base_l = base_lab[:, :, 0]
            base_a = base_lab[:, :, 1] - 128.0
            base_b = base_lab[:, :, 2] - 128.0
            split = self._split_tone_delta_from_arrays(
                ref_l, ref_a, ref_b, base_l, base_a, base_b
            )
        else:
            split = None

        if split is not None:
            engine_params["_split_tone_three_way"] = split
            if split["shadows"]["sat"] > 1.0:
                engine_params["shadow_hue"] = split["shadows"]["hue"]
                engine_params["shadow_sat"] = split["shadows"]["sat"]
            if split["highlights"]["sat"] > 1.0:
                engine_params["highlight_hue"] = split["highlights"]["hue"]
                engine_params["highlight_sat"] = split["highlights"]["sat"]
            if split["midtones"]["sat"] > 1.0:
                engine_params["midtone_hue"] = split["midtones"]["hue"]
                engine_params["midtone_sat"] = split["midtones"]["sat"]

        return engine_params

    # ------------------------------------------------------------------
    # Film density: toe / shoulder / crosstalk
    # ------------------------------------------------------------------

    def _extract_film_density(
        self,
        base_f: Optional[np.ndarray],
        ref_f: np.ndarray,
    ) -> Dict[str, Any]:
        ref_lin = self._to_scene_linear(ref_f)

        luma = (
            0.2126 * ref_lin[..., 2]
            + 0.7152 * ref_lin[..., 1]
            + 0.0722 * ref_lin[..., 0]
        )
        luma = np.clip(luma, _EPS, 1.0)

        ref_p = np.percentile(luma, [1, 50, 99])
        shadow_density = float(1.0 - ref_p[0])
        highlight_compression = float(1.0 - ref_p[2])
        midpoint = float(np.clip(ref_p[1], 0.2, 0.8))

        toe = float(np.clip(shadow_density * 0.4, 0.0, 0.4))
        shoulder = float(np.clip(highlight_compression * 0.4, 0.0, 0.4))

        mean_rgb = np.mean(ref_lin, axis=(0, 1))
        r_m, g_m, b_m = float(mean_rgb[2]), float(mean_rgb[1]), float(mean_rgb[0])
        mean_val = (r_m + g_m + b_m) / 3.0
        if mean_val > _EPS:
            cy_mg = float(np.clip((g_m - r_m) / mean_val * 0.1, -0.15, 0.15))
            cy_ye = float(np.clip((b_m - r_m) / mean_val * 0.1, -0.15, 0.15))
            mg_ye = float(np.clip((b_m - g_m) / mean_val * 0.1, -0.15, 0.15))
        else:
            cy_mg = cy_ye = mg_ye = 0.0

        gamma = 1.0
        if base_f is not None:
            base_lin = self._to_scene_linear(base_f)
            base_luma = (
                0.2126 * base_lin[..., 2]
                + 0.7152 * base_lin[..., 1]
                + 0.0722 * base_lin[..., 0]
            )
            base_luma = np.clip(base_luma, _EPS, 1.0)
            ref_luma_mean = float(np.mean(luma))
            base_luma_mean = float(np.mean(base_luma))
            if base_luma_mean > _EPS and ref_luma_mean > _EPS:
                gamma = float(np.clip(np.log(ref_luma_mean) / np.log(base_luma_mean), 0.7, 1.3))

        density_active = (
            abs(cy_mg) > 0.02 or abs(cy_ye) > 0.02 or abs(mg_ye) > 0.02
            or abs(gamma - 1.0) > 0.05
        )
        if not density_active:
            return {"film_enable": False}

        toe = toe if toe > 0.12 else 0.0
        shoulder = shoulder if shoulder > 0.12 else 0.0

        return {
            "film_enable": True,
            "film_strength": 0.7,
            "film_toe_r": round(toe, 3),
            "film_toe_g": round(toe, 3),
            "film_toe_b": round(toe, 3),
            "film_shoulder_r": round(shoulder, 3),
            "film_shoulder_g": round(shoulder, 3),
            "film_shoulder_b": round(shoulder, 3),
            "film_midpoint": round(midpoint, 3),
            "film_gamma": round(gamma, 3),
            "film_crosstalk_cy_mg": round(cy_mg, 3),
            "film_crosstalk_cy_ye": round(cy_ye, 3),
            "film_crosstalk_mg_ye": round(mg_ye, 3),
            "film_tonemap_strength": 0.6,
            "film_tonemap_toe": round(toe, 3),
            "film_tonemap_shoulder": round(shoulder, 3),
            "film_skew": 0.3,
        }

    def _to_scene_linear(self, img_f: np.ndarray) -> np.ndarray:
        rgb01 = np.clip(img_f[..., ::-1] * (1.0 / 255.0), 0.0, 1.0)
        lin = np.where(
            rgb01 <= 0.04045,
            rgb01 / 12.92,
            np.power((rgb01 + 0.055) / 1.055, 2.4),
        )
        return np.clip(lin, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Grain
    # ------------------------------------------------------------------

    def _extract_grain(self, ref_f: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(ref_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        if min(h, w) < 16:
            return {}

        ksize = max(3, (min(h, w) // 32) | 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        residual = gray - blurred
        noise_std = float(np.std(residual))

        grain_strength = float(np.clip(noise_std / 25.0, 0.0, 1.0))
        if grain_strength < 0.02:
            return {}

        return {
            "grain": round(grain_strength, 3),
            "grain_strength": round(grain_strength, 3),
        }

    # ------------------------------------------------------------------
    # Vignette
    # ------------------------------------------------------------------

    def _extract_vignette(self, ref_f: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(ref_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        if min(h, w) < 32:
            return {}

        cy, cx = h / 2.0, w / 2.0
        max_r = float(np.sqrt(cx * cx + cy * cy))
        if max_r < _EPS:
            return {}

        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r

        n_rings = 5
        ring_edges = np.linspace(0.0, 1.0, n_rings + 1)
        center_means: List[float] = []
        edge_means: List[float] = []
        for i in range(n_rings):
            lo, hi = ring_edges[i], ring_edges[i + 1]
            mask = (dist >= lo) & (dist < hi)
            if np.sum(mask) < 10:
                continue
            mean_val = float(np.mean(gray[mask]))
            if i < n_rings // 2:
                center_means.append(mean_val)
            else:
                edge_means.append(mean_val)

        if not center_means or not edge_means:
            return {}

        center_mean = float(np.mean(center_means))
        edge_mean = float(np.mean(edge_means))
        if center_mean < _EPS:
            return {}

        fall = (center_mean - edge_mean) / center_mean
        vignette_strength = float(np.clip(fall * 2.0, 0.0, 1.0))
        if vignette_strength < 0.03:
            return {}

        return {"vignette": round(vignette_strength * 100.0, 2)}

    # ------------------------------------------------------------------
    # Preset assembly
    # ------------------------------------------------------------------

    def _build_preset(
        self,
        base_f: Optional[np.ndarray],
        ref_f: np.ndarray,
    ) -> Dict[str, Any]:
        source = base_f if base_f is not None else ref_f
        curves = self._extract_l_curve(source, ref_f)
        white_balance = self._extract_white_balance(ref_f)
        hsl = self._extract_hsl(source, ref_f)

        preset: Dict[str, Any] = {}
        preset["curves"] = curves
        preset["white_balance"] = white_balance

        if base_f is not None:
            split_tone = self._extract_split_tone_delta(base_f, ref_f)
            if split_tone is not None:
                preset["split_tone_three_way"] = split_tone

        if hsl:
            preset["hsl_adjustments"] = hsl
        return preset

    def _extract_l_curve(
        self,
        base_f: np.ndarray,
        ref_f: np.ndarray,
    ) -> Dict[str, List[List[int]]]:
        base_lab = _bgr_to_lab_f32(base_f)
        ref_lab = _bgr_to_lab_f32(ref_f)

        base_l = base_lab[:, :, 0].ravel()
        ref_l = ref_lab[:, :, 0].ravel()

        percentiles = [0, 5, 25, 50, 75, 95, 100]
        base_pts = np.percentile(base_l, percentiles)
        ref_pts = np.percentile(ref_l, percentiles)

        curve_points: List[List[int]] = []
        for b, r in zip(base_pts, ref_pts):
            curve_points.append([int(round(b)), int(round(r))])

        curve_points = self._ensure_monotonic(curve_points)
        return {"L": curve_points}

    def _ensure_monotonic(self, points: List[List[int]]) -> List[List[int]]:
        if len(points) < 2:
            return points
        result = [points[0][:]]
        for i in range(1, len(points)):
            x, y = points[i]
            prev_y = result[-1][1]
            if y < prev_y:
                y = prev_y
            result.append([x, y])
        return result

    def _extract_white_balance(self, ref_f: np.ndarray) -> Dict[str, float]:
        img_f = ref_f
        b_mean = float(np.mean(img_f[:, :, 0]))
        g_mean = float(np.mean(img_f[:, :, 1]))
        r_mean = float(np.mean(img_f[:, :, 2]))

        gray = (b_mean + g_mean + r_mean) / 3.0
        if gray < 1.0:
            return {"R": 1.0, "G": 1.0, "B": 1.0}

        channel_spread = max(
            abs(b_mean - gray), abs(g_mean - gray), abs(r_mean - gray)
        ) / gray
        if channel_spread > 0.35:
            return {"R": 1.0, "G": 1.0, "B": 1.0}

        r_gain = gray / max(r_mean, _EPS)
        g_gain = gray / max(g_mean, _EPS)
        b_gain = gray / max(b_mean, _EPS)

        max_gain = max(r_gain, g_gain, b_gain)
        if max_gain > 0:
            r_gain /= max_gain
            g_gain /= max_gain
            b_gain /= max_gain

        r_gain = float(np.clip(r_gain, 0.7, 1.0))
        g_gain = float(np.clip(g_gain, 0.7, 1.0))
        b_gain = float(np.clip(b_gain, 0.7, 1.0))

        return {
            "R": round(r_gain, 3),
            "G": round(g_gain, 3),
            "B": round(b_gain, 3),
        }

    def _extract_split_tone_delta(
        self,
        base_f: np.ndarray,
        ref_f: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        base_lab = _bgr_to_lab_f32(base_f)
        ref_lab = _bgr_to_lab_f32(ref_f)
        return self._split_tone_delta_from_arrays(
            ref_lab[:, :, 0], ref_lab[:, :, 1] - 128.0, ref_lab[:, :, 2] - 128.0,
            base_lab[:, :, 0], base_lab[:, :, 1] - 128.0, base_lab[:, :, 2] - 128.0,
        )

    def _split_tone_delta_from_arrays(
        self,
        ref_l: np.ndarray,
        ref_a: np.ndarray,
        ref_b: np.ndarray,
        base_l: np.ndarray,
        base_a: np.ndarray,
        base_b: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        shadow_mask = ref_l < 85.0
        midtone_mask = (ref_l >= 85.0) & (ref_l <= 170.0)
        highlight_mask = ref_l > 170.0

        def _band_delta(mask: np.ndarray) -> Tuple[float, float]:
            if np.sum(mask) < 100:
                return 0.0, 0.0
            da = float(np.mean(ref_a[mask]) - np.mean(base_a[mask]))
            db = float(np.mean(ref_b[mask]) - np.mean(base_b[mask]))
            chroma = float(np.sqrt(da * da + db * db))
            if chroma < 1.5:
                return 0.0, 0.0
            hue_deg = float(np.degrees(np.arctan2(db, da)) % 360.0)
            sat = min(chroma / 60.0 * 100.0, 40.0)
            return round(hue_deg, 1), round(sat, 1)

        s_hue, s_sat = _band_delta(shadow_mask)
        m_hue, m_sat = _band_delta(midtone_mask)
        h_hue, h_sat = _band_delta(highlight_mask)

        if s_sat < 1.0 and m_sat < 1.0 and h_sat < 1.0:
            return None

        l_norm = ref_l / 255.0
        balance = float(np.mean(l_norm) - 0.5) * 100.0

        return {
            "shadows": {"hue": s_hue, "sat": s_sat},
            "midtones": {"hue": m_hue, "sat": m_sat},
            "highlights": {"hue": h_hue, "sat": h_sat},
            "balance": round(balance, 1),
        }

    def _extract_hsl(
        self,
        base_f: np.ndarray,
        ref_f: np.ndarray,
    ) -> Dict[str, Dict[str, int]]:
        color_centers = {
            "red": 0.0, "orange": 14.0, "yellow": 27.0, "green": 57.0,
            "cyan": 92.0, "blue": 122.0, "purple": 148.0, "magenta": 156.0,
        }
        sigma = 10.0

        base_hsv = cv2.cvtColor(base_f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(ref_f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

        base_h = base_hsv[:, :, 0]
        base_s = base_hsv[:, :, 1]
        ref_s = ref_hsv[:, :, 1]

        sat_adjusts: Dict[str, int] = {}
        for color, center in color_centers.items():
            dist = np.abs(base_h - center)
            dist = np.minimum(dist, 180.0 - dist)
            weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

            if np.sum(weight) < 100:
                continue

            base_sat_w = float(np.sum(base_s * weight) / np.sum(weight))
            ref_sat_w = float(np.sum(ref_s * weight) / np.sum(weight))
            delta = ref_sat_w - base_sat_w

            if abs(delta) > 8.0:
                sat_adjusts[color] = int(round(delta))

        result: Dict[str, Dict[str, int]] = {}
        if sat_adjusts:
            result["saturation"] = sat_adjusts
        return result

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_preset(self, preset: Dict[str, Any], name: str) -> Path:
        presets_dir = Path(__file__).resolve().parent.parent / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)

        filename = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
        filename = filename.strip("_") or "extracted_look"
        filepath = presets_dir / f"{filename}.json"

        suffix = 2
        while filepath.exists():
            filepath = presets_dir / f"{filename}_{suffix}.json"
            suffix += 1

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)

        logger.info("LookExtractor saved preset to %s", filepath)
        return filepath


__all__ = ["LookExtractor"]

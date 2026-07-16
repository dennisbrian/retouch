"""F9 — Image Analyzer + adaptive recipes ("targets, not deltas").

Analyze an image's lighting, dynamic range, noise, white balance, and (when
a face + skin mask are available) skin condition, then suggest processing
parameters and a starting recipe that would land the image on a neutral
*target* rather than applying a blind delta.

The analyzer is pure-vectorized NumPy/OpenCV on a downsampled proxy — no ML
model of its own. It accepts optional pre-computed face detections and skin
masks (BiSeNet or MediaPipe) so it can measure skin-region statistics; when
those are absent it falls back to global statistics only.

Internal dtype is float32 throughout; uint8 inputs are converted at the
boundary and never used in pixel arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _recipes_table() -> Dict[str, Any]:
    """Lazily import and return the RECIPES dict.

    Imported lazily so this module has no hard dependency on the recipe
    loader at module-load time (avoids a cycle when this module is
    consumed from build_context in params/engine).
    """
    from .recipes import RECIPES
    return RECIPES

# Lighting-type categories the analyzer classifies into. Kept as a module
# constant so callers (and tests) can assert against the literal set without
# importing a StrEnum (Python 3.9 compatibility).
LIGHTING_TYPES: Tuple[str, ...] = (
    "studio",
    "outdoor",
    "convention",
    "low_light",
)

# Default recipe when nothing more specific fits.
_DEFAULT_RECIPE = "natural"


# ---------------------------------------------------------------------------
# dtype / colorspace boundary helpers
# ---------------------------------------------------------------------------


def _to_f32_bgr(img: np.ndarray) -> np.ndarray:
    """Coerce an image to float32 BGR in [0, 255] without uint8 arithmetic.

    Raises ValueError for non-HxWx3 inputs.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"ImageAnalyzer expects HxWx3 BGR, got shape {img.shape}"
        )
    if img.dtype == np.float32:
        return np.clip(img, 0.0, 255.0).astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32)
    # float64 / float16 / int — coerce via float32
    return np.clip(img.astype(np.float32), 0.0, 255.0)


def _bgr_f32_to_lab_u8_conv(img_f: np.ndarray) -> np.ndarray:
    """Convert float32 BGR [0,255] to float32 LAB in uint8 convention.

    OpenCV's ``COLOR_BGR2LAB`` on float input expects BGR in [0,1] and
    returns L in [0,100], a/b in [-128,127]. This normalizes the input,
    converts, then rescales to the uint8 convention (L in [0,255],
    a/b in [0,255] with 128=neutral) so threshold math is consistent with
    the rest of the codebase (look_extractor, qa_detectors).
    """
    img01 = np.clip(img_f * (1.0 / 255.0), 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(img01, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * 2.55
    lab[:, :, 1] = lab[:, :, 1] + 128.0
    lab[:, :, 2] = lab[:, :, 2] + 128.0
    return lab


def _bgr_f32_to_hsv(img_f: np.ndarray) -> np.ndarray:
    """float32 BGR [0,255] → float32 HSV (H in [0,180], S/V in [0,255])."""
    return cv2.cvtColor(img_f.astype(np.float32), cv2.COLOR_BGR2HSV).astype(
        np.float32
    )


def _normalize_mask(mask: Optional[np.ndarray], ref_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Coerce a mask to float32 [0,1] matching ``ref_shape`` (H, W).

    Returns None when the mask is None or empty. Resizes to ``ref_shape``
    when shapes differ (masks are smooth; INTER_AREA downsample is safe).
    """
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    if m.shape != ref_shape:
        m = cv2.resize(m, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
        m = m.astype(np.float32)
    if m.max() < _EPS:
        return None
    return np.clip(m, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ImageAnalysis dataclass
# ---------------------------------------------------------------------------


@dataclass
class SkinCondition:
    """Per-image skin assessment. Populated only when a face + skin mask exist.

    All Lab values use the uint8 convention (L in [0,255], a/b in [0,255]
    with 128=neutral) so they line up with the rest of the codebase.

    Attributes:
        present: True if skin pixels were found.
        l_mean: Mean L of the skin region.
        a_mean: Mean a* of the skin region (positive = red).
        b_mean: Mean b* of the skin region (positive = yellow).
        l_std: L std — unevenness of skin lightness.
        c_std: chroma std (σ_C) — unevenness of skin chroma (lower = more uniform).
        redness: 0-1 score indicating excess redness (a* above expected locus).
        yellowness: 0-1 score indicating excess yellow (b* above expected locus).
        uniformity: 0-1 score (1 = perfectly uniform chroma).
    """

    present: bool = False
    l_mean: float = 0.0
    a_mean: float = 0.0
    b_mean: float = 0.0
    l_std: float = 0.0
    c_std: float = 0.0
    redness: float = 0.0
    yellowness: float = 0.0
    uniformity: float = 0.0


@dataclass
class ImageAnalysis:
    """Structured assessment of a single image.

    All numeric fields are plain Python floats (or lists of floats) so the
    dataclass is JSON-serializable and picklable across a process boundary.

    Attributes:
        lighting_type: One of LIGHTING_TYPES — "studio" / "outdoor" /
            "convention" / "low_light".
        dynamic_range: Estimate of usable dynamic range — 95th-5th L
            percentile, in [0, 255].
        noise_level: Estimated sensor-noise sigma (0-1 normalized).
        white_balance_estimate: Dict with ``kelvin`` (estimated CCT),
            ``tint`` (green-magenta offset, -100..100), and ``cast_strength``
            (0-1; how far from neutral the gray-world estimate is).
        skin_condition: SkinCondition dataclass (present=False when no face).
        dominant_colors: Up to 5 representative BGR colors as [B, G, R] lists.
        contrast_level: 0-1 normalized RMS-contrast measure.
        sharpness_estimate: 0-1 normalized high-frequency-energy measure.
        key: "low" / "normal" / "high" — exposure key classification.
        mean_luminance: Mean L of the whole image (uint8 convention, [0,255]).
        l_percentiles: List of L at [1, 5, 25, 50, 75, 95, 99] percentiles.
    """

    lighting_type: str = "studio"
    dynamic_range: float = 0.0
    noise_level: float = 0.0
    white_balance_estimate: Dict[str, float] = field(
        default_factory=lambda: {"kelvin": 6500.0, "tint": 0.0, "cast_strength": 0.0}
    )
    skin_condition: SkinCondition = field(default_factory=SkinCondition)
    dominant_colors: List[List[int]] = field(default_factory=list)
    contrast_level: float = 0.0
    sharpness_estimate: float = 0.0
    key: str = "normal"
    mean_luminance: float = 0.0
    l_percentiles: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ImageAnalyzer
# ---------------------------------------------------------------------------


class ImageAnalyzer:
    """Analyze an image and suggest processing parameters / recipe.

    The analyzer is constructed with an optional ``downsample_dim`` (the
    longest side the image is resized to before measurement — keeps the
    per-image cost in milliseconds). It accepts optional pre-computed face
    detections and skin masks via ``analyze()`` so it can measure
    skin-region statistics; without them it falls back to global-only stats.
    """

    # Luminance thresholds (uint8 Lab L convention) for key classification.
    KEY_LOW_THRESHOLD: float = 90.0
    KEY_HIGH_THRESHOLD: float = 170.0

    # Noise: residual std at which we consider the image "noisy".
    NOISE_HIGH: float = 8.0

    # Default target skin L (uint8 convention). Mirrors the "fair" locus L_min
    # of ~0.72 in Oklab, which is ~184 in uint8-Lab-L — but we use a slightly
    # lower target so we don't push already-bright skin further.
    TARGET_SKIN_L: float = 175.0

    # Expected skin locus in uint8 Lab (a/b centered on 128). These are the
    # "fair" reference values from color_science.SKIN_LOCI mapped into the
    # uint8-Lab convention: a ≈ 128 + 12, b ≈ 128 + 18.
    EXPECTED_SKIN_A: float = 140.0
    EXPECTED_SKIN_B: float = 146.0

    def __init__(self, downsample_dim: int = 512) -> None:
        if downsample_dim < 64:
            raise ValueError(f"downsample_dim must be >= 64, got {downsample_dim}")
        self.downsample_dim = downsample_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        img: np.ndarray,
        face_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        skin_mask: Optional[np.ndarray] = None,
        person_mask: Optional[np.ndarray] = None,
    ) -> ImageAnalysis:
        """Analyze ``img`` and return a structured :class:`ImageAnalysis`.

        Args:
            img: (H, W, 3) BGR uint8 or float32 [0, 255].
            face_bboxes: Optional list of (x, y, w, h) face boxes. When
                provided, the analyzer treats the image as a portrait and
                measures skin stats when ``skin_mask`` is also given.
            skin_mask: Optional (H, W) float mask [0, 1] (or [0, 255]).
                When provided alongside ``face_bboxes``, populates
                :attr:`ImageAnalysis.skin_condition`.
            person_mask: Optional (H, W) float mask. Currently informational;
                reserved for subject/background ratio in a later iteration.

        Returns:
            ImageAnalysis dataclass.
        """
        img_f = self._prepare(img)
        h, w = img_f.shape[:2]

        lab = _bgr_f32_to_lab_u8_conv(img_f)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]

        # Luminance / exposure
        percentiles = np.percentile(l_chan, [1, 5, 25, 50, 75, 95, 99]).astype(
            np.float32
        )
        mean_l = float(np.mean(l_chan))
        key = self._classify_key(mean_l)
        dynamic_range = float(percentiles[5] - percentiles[1])  # p95 - p5

        # Contrast: RMS contrast normalized to [0, 1]
        l_norm = l_chan / 255.0
        rms_contrast = float(np.std(l_norm))
        contrast_level = float(np.clip(rms_contrast / 0.35, 0.0, 1.0))

        # Sharpness: high-frequency energy via Laplacian std, normalized
        gray_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
        gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
        lap_std = float(np.std(lap))
        sharpness_estimate = float(np.clip(lap_std / 50.0, 0.0, 1.0))

        # Noise: MAD on low-gradient blocks (Laplacian-quiet regions)
        noise_sigma = self._estimate_noise(gray_u8)
        noise_level = float(np.clip(noise_sigma / self.NOISE_HIGH, 0.0, 1.0))

        # White balance
        wb = self._estimate_white_balance(img_f)

        # Dominant colors via k-means on a small sample
        dominant = self._dominant_colors(img_f, k=5)

        # Skin condition (only if both face + skin mask provided)
        skin = SkinCondition()
        if face_bboxes and skin_mask is not None:
            sm = _normalize_mask(skin_mask, (h, w))
            if sm is not None:
                skin = self._measure_skin(lab, sm)

        # Lighting-type classification
        lighting_type = self._classify_lighting(
            mean_l=mean_l,
            noise_level=noise_level,
            cast_strength=wb["cast_strength"],
            has_face=bool(face_bboxes),
            skin_present=skin.present,
            skin_l=skin.l_mean,
            p99=float(percentiles[6]),
        )

        return ImageAnalysis(
            lighting_type=lighting_type,
            dynamic_range=round(dynamic_range, 3),
            noise_level=round(noise_level, 4),
            white_balance_estimate={
                "kelvin": round(wb["kelvin"], 1),
                "tint": round(wb["tint"], 2),
                "cast_strength": round(wb["cast_strength"], 4),
            },
            skin_condition=skin,
            dominant_colors=dominant,
            contrast_level=round(contrast_level, 4),
            sharpness_estimate=round(sharpness_estimate, 4),
            key=key,
            mean_luminance=round(mean_l, 3),
            l_percentiles=[round(float(p), 3) for p in percentiles],
        )

    def suggest_params(self, analysis: ImageAnalysis) -> Dict[str, Any]:
        """Suggest processing parameters that move the image toward a target.

        Returns a flat dict of GUI-side parameter names → values. Only
        parameters that should *change* from the recipe default are emitted;
        absent keys mean "leave at default". All values are clamped to the
        valid slider range defined in :mod:`retouch.params`.

        Args:
            analysis: An :class:`ImageAnalysis` produced by :meth:`analyze`.
        """
        out: Dict[str, Any] = {}

        # Exposure / brightness — soft target mean L ~128, but never flatten
        # intentional low-key / high-key portraits. Large |brightness| also
        # inflates full-ΔE "color drift" gates that are mostly ΔL.
        target_global_l = 128.0
        delta_l = target_global_l - analysis.mean_luminance
        mean_l = analysis.mean_luminance
        if mean_l < 80.0:
            # Dark scene / dark-bg portrait: open a little, keep key.
            brightness = int(np.clip(delta_l * 0.15, 0, 12))
        elif mean_l > 180.0:
            brightness = int(np.clip(delta_l * 0.3, -20, 0))
        else:
            brightness = int(np.clip(delta_l * 0.5, -30, 30))
        if abs(brightness) >= 1:
            out["brightness"] = brightness

        # Dynamic-range targets: pull blacks up / whites down when DR is
        # clipped at either end. p1 < 10 → blacks crushed; p99 > 240 → whites hot.
        p = analysis.l_percentiles
        if len(p) >= 7:
            p1, _, _, _, _, p95, p99 = p
            if p1 < 10.0:
                # Lift blacks toward a target of ~12
                out["blacks"] = int(np.clip((12.0 - p1) * 0.5, 0, 60))
            if p99 > 240.0:
                # Pull whites down hard enough to engage (old thr 248 + gain 0.5
                # collapsed to whites=0 on real outdoor p99≈249).
                out["whites"] = int(np.clip(-(p99 - 240.0) * 1.5, -60, 0))
            if p95 > 235.0:
                out.setdefault(
                    "highlights",
                    int(np.clip(-(p95 - 235.0) * 1.0, -40, 0)),
                )
            # Contrast: if DR is narrow, add contrast; if wide, reduce slightly.
            dr = p95 - p1
            if dr < 120.0:
                # Narrow dynamic range — boost contrast
                out["contrast"] = int(np.clip((120.0 - dr) * 0.3, 0, 30))
            elif dr > 230.0 and "contrast" not in out:
                # Very wide DR — gentle contrast reduction
                out["contrast"] = int(np.clip(-(dr - 230.0) * 0.3, -20, 0))

        # White balance — convert estimated kelvin/tint to slider values.
        wb = analysis.white_balance_estimate
        kelvin = float(wb.get("kelvin", 6500.0))
        tint = float(wb.get("tint", 0.0))
        if abs(kelvin - 6500.0) > 200.0:
            out["white_balance_kelvin"] = int(np.clip(round(kelvin), 2000, 12000))
        if abs(tint) > 1.0:
            out["white_balance_tint"] = float(np.clip(tint, -100.0, 100.0))

        # Noise → denoise strength (F7 ai_denoise, 0-100). Only suggest when
        # noise_level is non-trivial.
        if analysis.noise_level > 0.25:
            out["ai_denoise"] = int(np.clip(round(analysis.noise_level * 60.0), 0, 100))

        # Sharpness — when the image is soft, suggest a gentle sharpen.
        if analysis.sharpness_estimate < 0.20:
            out["sharpen"] = int(np.clip((0.20 - analysis.sharpness_estimate) * 200.0, 5, 40))

        # Skin condition → skin params (only when skin present)
        skin = analysis.skin_condition
        if skin.present:
            # Brightness target for skin L
            skin_delta = self.TARGET_SKIN_L - skin.l_mean
            # Map to `whiten` (rosy brightening, 0-100) — positive when skin is dark.
            if skin_delta > 5.0:
                out["whiten"] = int(np.clip(skin_delta * 0.4, 0, 60))
            # Redness even: when skin a* is above expected, suggest redness_even
            if skin.a_mean > self.EXPECTED_SKIN_A + 4.0:
                out["redness_even"] = int(
                    np.clip((skin.a_mean - self.EXPECTED_SKIN_A) * 1.5, 0, 60)
                )
            # Yellow / sallowness: when b* is high, a touch of whiten already
            # covers it; emit equalize when chroma is uneven.
            if skin.c_std > 8.0:
                out["equalize"] = int(np.clip((skin.c_std - 8.0) * 2.0, 0, 50))

        # Lighting-type specific adjustments
        if analysis.lighting_type == "low_light":
            # Low-light: lift shadows, gentle clarity boost
            out.setdefault("shadows", 25)
            out.setdefault("clarity", 10)
        elif analysis.lighting_type == "outdoor":
            # Outdoor: a touch of vibrance to counter flat lighting
            out.setdefault("vibrance", 8)

        return out

    def suggest_recipe(self, analysis: ImageAnalysis) -> str:
        """Suggest the best starting recipe name for ``analysis``.

        The selection is rule-based on lighting type, key, WB cast, noise,
        and skin condition. Always returns a key present in
        ``retouch.recipes.RECIPES``; falls back to :data:`_DEFAULT_RECIPE`.
        """
        wb = analysis.white_balance_estimate
        cast = float(wb.get("cast_strength", 0.0))
        kelvin = float(wb.get("kelvin", 6500.0))
        skin = analysis.skin_condition

        # Ordered candidate lists — first one that exists in RECIPES wins.
        # This decouples the suggester from the exact recipe inventory: a
        # candidate listed here but absent from RECIPES is silently skipped.
        def _pick(candidates: Sequence[str]) -> str:
            for c in candidates:
                if c in _recipes_table():
                    return c
            return _DEFAULT_RECIPE if _DEFAULT_RECIPE in _recipes_table() else next(iter(_recipes_table()))

        # High noise → prefer a recipe with built-in smoothing.
        if analysis.noise_level > 0.5:
            return _pick(["milk_skin_v1", "xhs_ultrasoft", "porcelain_unified_v1"])

        # Dark-bg but subject-lit portraits (common cosplay/studio): high p99 +
        # low noise means the face is fine — do NOT pick heavy soft recipes
        # (xhs_ultrasoft smooth=0.85 kills pore SSIM).
        p99 = (
            float(analysis.l_percentiles[6])
            if len(analysis.l_percentiles) >= 7
            else 128.0
        )
        if (
            (analysis.key == "low" or analysis.lighting_type == "low_light")
            and p99 > 200.0
            and analysis.noise_level < 0.25
        ):
            return _pick(["beauty", "portrait", "natural", "natural_polish_v1"])

        # Genuine low-key / low-light scenes → cool porcelain / soft look
        if analysis.key == "low" or analysis.lighting_type == "low_light":
            if kelvin > 7000.0 or cast < 0.15:
                return _pick(["fuji_porcelain", "porcelain_unified_v1", "xhs_ultrasoft"])
            return _pick(["xhs_ultrasoft", "milk_skin_v1"])

        # Studio lighting → recipes tuned for controlled light.
        # Margin above the subject's own TARGET_SKIN_L (not an absolute L
        # level): skin already at/near the porcelain target benefits from
        # porcelain_unified_v1, regardless of the subject's own skin tone.
        # An absolute cutoff (e.g. "L > 160") is reachable near-for-free on
        # light skin but requires a much stronger relit adjustment on dark
        # skin to cross the same line, so it would systematically steer
        # darker-skinned subjects away from the porcelain recipes.
        if analysis.lighting_type == "studio":
            if skin.present and (self.TARGET_SKIN_L - skin.l_mean) < 15.0:
                return _pick(["porcelain_unified_v1", "beauty"])
            return _pick(["beauty", "portrait"])

        # Convention (mixed-temp / fluorescent) → convention repair recipes
        if analysis.lighting_type == "convention":
            if kelvin < 4500.0:
                return _pick(["con_fluorescent_v1", "con_mixed_temp_v1"])
            return _pick(["con_mixed_temp_v1", "con_fluorescent_v1"])

        # Outdoor — sub-classify by WB cast
        if analysis.lighting_type == "outdoor":
            if kelvin < 5000.0:
                return _pick(["outdoor_overcast_v1", "natural"])
            if kelvin > 6500.0 and analysis.dynamic_range > 180.0:
                return _pick(["outdoor_backlit_v1", "portrait"])
            # Warm golden-hour cast
            if kelvin < 6000.0 and cast < 0.20:
                return _pick(["outdoor_golden_hour_v1", "astia"])
            return _pick(["outdoor_harsh_sun_v1", "natural"])

        # High-key bright scenes → soft glowing recipes
        if analysis.key == "high":
            return _pick(["xhs_soft_glow", "xiaohongshu", "beauty"])

        return _pick([_DEFAULT_RECIPE, "natural", "portrait"])

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
    # Internal: measurement
    # ------------------------------------------------------------------

    def _classify_key(self, mean_l: float) -> str:
        if mean_l < self.KEY_LOW_THRESHOLD:
            return "low"
        if mean_l > self.KEY_HIGH_THRESHOLD:
            return "high"
        return "normal"

    def _estimate_noise(self, gray_u8: np.ndarray) -> float:
        """Estimate sensor noise sigma via MAD on low-gradient blocks.

        Splits the image into 8x8 blocks, keeps the 25% with the lowest
        Laplacian energy (flat regions), and returns the median absolute
        deviation of those block stds. Robust to high-frequency content
        (textures, edges) which would inflate a naive global std.
        """
        h, w = gray_u8.shape
        if h < 16 or w < 16:
            return 0.0

        gray_f = gray_u8.astype(np.float32)
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
        lap_energy = cv2.boxFilter(lap * lap, cv2.CV_32F, (8, 8))

        # Collect block-level stats on a stride-8 grid
        block_ys = np.arange(0, max(h - 8, 1), 8)
        block_xs = np.arange(0, max(w - 8, 1), 8)
        if len(block_ys) == 0 or len(block_xs) == 0:
            return float(np.std(gray_f))

        energies: List[float] = []
        stds: List[float] = []
        for by in block_ys:
            for bx in block_xs:
                patch = gray_f[by : by + 8, bx : bx + 8]
                if patch.size < 64:
                    continue
                energies.append(float(lap_energy[by + 4, bx + 4]))
                stds.append(float(np.std(patch)))

        if not stds:
            return float(np.std(gray_f))

        # Keep the quietest 25% of blocks (low Laplacian energy = flat regions)
        energies_arr = np.asarray(energies, dtype=np.float32)
        stds_arr = np.asarray(stds, dtype=np.float32)
        thresh = np.percentile(energies_arr, 25)
        quiet = stds_arr[energies_arr <= thresh]
        if quiet.size == 0:
            return float(np.median(stds_arr))
        # The median std of the quiet blocks IS the noise sigma estimate.
        # (MAD-of-stds would measure noise *uniformity*, not noise magnitude.)
        return float(np.median(quiet))

    def _estimate_white_balance(self, img_f: np.ndarray) -> Dict[str, float]:
        """Estimate white balance via gray-world + skin-tone prior.

        Returns kelvin (CCT estimate), tint (green-magenta), and
        cast_strength (0-1; how far from neutral the gray-world estimate is).
        """
        b_mean = float(np.mean(img_f[:, :, 0]))
        g_mean = float(np.mean(img_f[:, :, 1]))
        r_mean = float(np.mean(img_f[:, :, 2]))

        gray = (b_mean + g_mean + r_mean) / 3.0
        if gray < _EPS:
            return {"kelvin": 6500.0, "tint": 0.0, "cast_strength": 0.0}

        # Gray-world gains
        r_gain = gray / max(r_mean, _EPS)
        g_gain = gray / max(g_mean, _EPS)
        b_gain = gray / max(b_mean, _EPS)

        # Normalize so the largest gain is 1.0 (relative cooling/warming)
        max_gain = max(r_gain, g_gain, b_gain)
        if max_gain > _EPS:
            r_gain /= max_gain
            g_gain /= max_gain
            b_gain /= max_gain

        # Cast strength: max relative deviation from neutral (1.0)
        cast = float(max(abs(r_gain - 1.0), abs(g_gain - 1.0), abs(b_gain - 1.0)))

        # Map R/B gain ratio → kelvin. Higher R-gain (low R) → cooler light
        # (high kelvin); higher B-gain (low B) → warmer light (low kelvin).
        # Using a simple monotonic map: kelvin = 6500 * (b_gain / r_gain).
        if r_gain > _EPS:
            ratio = b_gain / r_gain
        else:
            ratio = 1.0
        # Clamp ratio to a sensible CCT range and map to [3000, 11000]
        kelvin = float(np.clip(6500.0 / max(ratio, _EPS), 3000.0, 11000.0))
        # Smooth: blend with neutral to avoid wild swings on near-neutral images
        neutral_weight = float(np.clip(1.0 - cast * 4.0, 0.0, 1.0))
        kelvin = kelvin * (1.0 - neutral_weight) + 6500.0 * neutral_weight

        # Tint: green-magenta from g_gain deviation
        tint = float(np.clip((g_gain - 1.0) * -100.0, -100.0, 100.0))

        return {
            "kelvin": kelvin,
            "tint": tint,
            "cast_strength": cast,
        }

    def _dominant_colors(self, img_f: np.ndarray, k: int = 5) -> List[List[int]]:
        """Return up to ``k`` dominant BGR colors as [B, G, R] int lists.

        Uses a coarse 3D histogram (4 bins per channel → 64 buckets) on a
        stride-sampled subset. This is fully deterministic (no RNG) and
        ~10× faster than k-means on the same pixel count.
        """
        h, w = img_f.shape[:2]
        # Stride sample — keep the pixel count bounded
        stride = max(1, int((h * w) ** 0.5 / 32))
        pixels = img_f[::stride, ::stride].reshape(-1, 3).astype(np.float32)

        if pixels.shape[0] == 0:
            return [[128, 128, 128]]

        # Quantize to 4 bins per channel (0-63, 64-127, 128-191, 192-255)
        bins = 4
        bin_size = 256.0 / bins
        quantized = np.clip(pixels // bin_size, 0, bins - 1).astype(np.int32)
        # Flatten the 3D bin index into a single key for histogramming
        flat_idx = quantized[:, 0] * (bins * bins) + quantized[:, 1] * bins + quantized[:, 2]

        # Count populations per bucket
        total_buckets = bins ** 3
        counts = np.bincount(flat_idx, minlength=total_buckets)
        # Only keep non-empty buckets, sorted by population (desc)
        non_empty = np.where(counts > 0)[0]
        order = non_empty[np.argsort(-counts[non_empty])]

        out: List[List[int]] = []
        for idx in order[:k]:
            # Recover the bin (b, g, r) and report the bucket center
            b_bin = int(idx // (bins * bins))
            g_bin = int((idx % (bins * bins)) // bins)
            r_bin = int(idx % bins)
            center_b = int(round((b_bin + 0.5) * bin_size))
            center_g = int(round((g_bin + 0.5) * bin_size))
            center_r = int(round((r_bin + 0.5) * bin_size))
            out.append([center_b, center_g, center_r])
        return out

    def _measure_skin(self, lab: np.ndarray, skin_mask: np.ndarray) -> SkinCondition:
        """Measure skin-region statistics in Lab space.

        ``lab`` is float32 in uint8 convention (L [0,255], a/b centered on 128).
        ``skin_mask`` is float32 [0, 1] matching lab.shape[:2].
        """
        mask_idx = skin_mask > 0.3
        if not np.any(mask_idx):
            return SkinCondition(present=False)

        l_skin = lab[:, :, 0][mask_idx]
        a_skin = lab[:, :, 1][mask_idx]
        b_skin = lab[:, :, 2][mask_idx]

        l_mean = float(np.mean(l_skin))
        a_mean = float(np.mean(a_skin))
        b_mean = float(np.mean(b_skin))
        l_std = float(np.std(l_skin))

        # Chroma in uint8 Lab: sqrt((a-128)^2 + (b-128)^2)
        chroma = np.sqrt((a_skin - 128.0) ** 2 + (b_skin - 128.0) ** 2)
        c_std = float(np.std(chroma))

        # Redness / yellowness: how far above the expected locus (clamped 0-1)
        red_excess = max(0.0, a_mean - self.EXPECTED_SKIN_A)
        yellow_excess = max(0.0, b_mean - self.EXPECTED_SKIN_B)
        redness = float(np.clip(red_excess / 20.0, 0.0, 1.0))
        yellowness = float(np.clip(yellow_excess / 25.0, 0.0, 1.0))

        # Uniformity: 1 - normalized c_std. Typical skin c_std is 0.01-0.15 in
        # Oklab; in uint8 Lab that scales to ~2-25, so normalize by 25.
        uniformity = float(np.clip(1.0 - c_std / 25.0, 0.0, 1.0))

        return SkinCondition(
            present=True,
            l_mean=round(l_mean, 3),
            a_mean=round(a_mean, 3),
            b_mean=round(b_mean, 3),
            l_std=round(l_std, 3),
            c_std=round(c_std, 3),
            redness=round(redness, 4),
            yellowness=round(yellowness, 4),
            uniformity=round(uniformity, 4),
        )

    # ------------------------------------------------------------------
    # Internal: lighting classification
    # ------------------------------------------------------------------

    def _classify_lighting(
        self,
        mean_l: float,
        noise_level: float,
        cast_strength: float,
        has_face: bool,
        skin_present: bool,
        skin_l: float,
        p99: float = 255.0,
    ) -> str:
        """Classify lighting into one of LIGHTING_TYPES.

        Heuristics:
          - low_light: high noise (high-ISO) OR both mean and highlights dark.
            Dark mean alone is NOT enough — cosplay/studio shots often have
            black backgrounds with a well-lit subject (high p99).
          - convention: strong WB cast (>0.12) + face present — typical of
            mixed-temperature convention-hall lighting.
          - studio: low noise + low cast + well-lit subject — clean controlled light.
          - outdoor: default when none of the above match.
        """
        if noise_level > 0.5:
            return "low_light"
        # Genuinely underexposed: dark mean AND crushed/dim highlights.
        if mean_l < 90.0 and p99 < 180.0:
            return "low_light"
        if cast_strength > 0.12 and has_face:
            return "convention"
        # Dark-bg lit-subject portraits still count as studio when clean.
        if noise_level < 0.15 and cast_strength < 0.06 and (mean_l > 120.0 or p99 > 200.0):
            return "studio"
        return "outdoor"


__all__ = [
    "ImageAnalysis",
    "ImageAnalyzer",
    "SkinCondition",
    "LIGHTING_TYPES",
]

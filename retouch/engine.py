"""RetouchEngine — main pipeline orchestrator.

Pipeline order:
    1.  Face Detection (MediaPipe + optional RetinaFace)
    2.  Person Segmentation (MediaPipe Selfie Segmentation)
    3.  Face Region Parsing (landmarks → per-region masks)
    4.  3-Level Frequency Separation
    5.  Skin Smoothing (frequency-based)
    6.  Blemish Removal (inpainting)
    7.  Under-Eye Dark Circle Repair
    8.  Eye Enhancement (whites, iris, catchlight)
    9.  Lip Enhancement (vibrance, tint)
    10. Teeth Whitening
    11. Skin Tone Equalization (CLAHE)
    12. Skin Whitening / Foundation (LAB)
    13. Contrast Adjustment
    14. Colour Grading (preset)
    15. Selective Sharpening
    16. Global Impact Finish

Changelog vs. previous version
--------------------------------
BUGFIX-1  s_mask multi-face scope leak: s_mask was referenced after the per-face
          loop using only the *last* face's value, causing wrong split-tone masks
          on multi-face images. Now accumulated correctly via
          accumulated_skin_mask (already computed) and passed as that.

BUGFIX-2  color_ref double-write: the `p` dict set "color_ref" to None and then
          re-assigned from the arg, but the first assignment silently shadowed the
          kwarg. Removed the erroneous early None assignment.

BUGFIX-3  r_dark_circles reused the "whites" eye recipe key instead of a
          dedicated "dark_circles" key, so dark-circle strength was always driven
          by teeth-whitening intensity. Now reads rec["eyes"].get("dark_circles")
          with a zero fallback.

BUGFIX-4  LUT name collision: the `lut` parameter variable was silently
          overwritten by the brightness LUT ndarray later in the function body,
          corrupting the LUT path for any call that set both brightness and lut.
          Renamed the internal ndarray to `_brightness_lut`.

ARCH-1    ProcessingContext dataclass replaces the raw `p` dict, giving type
          safety, IDE completion, and a single place to evolve parameter defaults.

ARCH-2    Monolithic process() decomposed into focused private stage methods:
          _stage_reshape, _stage_per_face, _stage_global, _stage_grade,
          _stage_finish. Each returns (result, context) and can be tested or
          bypassed independently.

ARCH-3    Multi-face processing is now parallelised with ThreadPoolExecutor when
          more than one face is detected, each face writing into its own canvas
          slice that is composited back. Falls back to serial on single-face.

ARCH-4    ProcessingResult returned instead of a bare ndarray, carrying the
          output image, per-face debug masks, applied parameters, and timing
          breakdowns so callers can introspect what happened.

PERF-1    Fast-preview path moved *before* detection so detection itself runs on
          the downscaled image (saves ~40 % on a 24 MP input at 800 px preview).

PERF-2    Region masks are now lazily normalised once via _norm_mask() helper
          instead of repeated inline max-checks.

PERF-3    Accumulated masks reuse the pre-normalised region copies so there are
          no redundant astype/divide passes in the face loop.

RECIPE-1  Recipe resolution extracted into resolve_recipe() — composable, unit-
          testable, and callable independently of a full engine instantiation.

RECIPE-2  Recipes now support an optional "extends" key for single-level
          inheritance: a recipe dict with {"extends": "natural", ...} inherits
          all keys from the parent before applying its own overrides.
"""

from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)

from .detection import FaceDetector, FaceData, FaceContext
from .parsing import FaceParser, FaceRegions
from .geometry import FaceReshaper
from .makeup import MakeupEngine
from .makeup_v2 import MakeupEngineV2
from .frequency import FrequencySeparator, _texture_adaptation_factor
from .perf_optimizations import (
    FaceProcessorPool,
    _accum,
    _FaceResult,
    _norm_mask,
    _process_face_core,
)
from .skin import SkinProcessor
from .blemish import BlemishRemover
from .eyes import EyeEnhancer
from .undereye import UnderEyeRepairer
from .lips import LipEnhancer
from .teeth import TeethWhitener
from .grading import ColorGrader, PRESETS
from .harmonizer import BackgroundHarmonizer
from .background import BackgroundReplacer
from . import grain, highlight, tonal, qa_detectors
from .qa_detectors import QAWarning
from .qa_backoff import QABackoff
from .hair import HairEnhancer
from .relight import Relighter
from .enhance import AIEnhancer
from .recipes import RECIPES
from .recipe_loader import load_user_recipes
from .style import StyleProfile
from .style_transfer import subject_aware_transfer
from .utils import correct_exposure, apply_global_bloom, apply_skin_diffusion, vibrance as _vibrance_fn, squeeze_mask, feather_mask, guided_filter, normalize_mask, blend_masked, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32
from .color_space import bgr_to_lch, skin_mask_lch
from .color_science import bgr_to_oklab, oklab_to_oklch
from .params import resolve_recipe, _deep_merge, PROCESSING_PARAMS  # noqa: F401  (re-export for backward compat)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROXY_MAX_DIM = 2048

# --- Proxy native-detail reinjection adaptation (F8.0) ---
# The proxy round-trip (downscale → retouch → upscale) discards pore-scale
# texture that only exists above proxy resolution. F8.0 reinjects the native
# high band into edit-mask regions, but discounts it by the recipe's smooth
# strength. On flat, front-lit skin the native texture is predominantly
# fine-scale (above proxy Nyquist) and intrinsically low-amplitude, so that
# discount is exactly what reads as "waxy". These thresholds map the robust
# (MAD) energy of the native high band inside acc_skin to an adaptation
# factor: low energy (flat lighting) → lift the discount and reinject fuller
# native detail; high energy (directional lighting, coarse texture) → keep
# the existing behavior unchanged. Calibrated on the duotian-nikke batch
# (natural_polish_v1, 4160px input, 2048px proxy): waxy front-lit faces
# measure native-skin high-band energy ~1.48-1.51, well-lit faces ~1.65-1.73;
# both sit inside [LOW, HIGH] so both recover proxy-lost pore detail, with the
# waxy faces (lower energy → smaller adapt) getting the larger lift. Set
# REINJECT_DEBUG=1 to log per-image energy/adapt. HIGH=3.0 leaves genuinely
# high-frequency faces untouched.
REINJECT_ENERGY_LOW = 1.0
REINJECT_ENERGY_HIGH = 3.0
REINJECT_ADAPT_FLOOR = 0.0
REINJECT_BASE_GAIN = 0.85

# Anime cinematic variants get standard nose blush, no slimming
# Mapped modularly in recipes config


# ---------------------------------------------------------------------------
# ProcessingContext — typed parameter bag (replaces raw dict)
# ---------------------------------------------------------------------------

# Single source of truth for the 26 fields whose dataclass default matches
# ``ParamSpec.default``.  Mismatches (33 fields) and caller-only fields
# (5 fields) keep their hardcoded defaults until resolved by the user.
_DEFAULTS: Dict[str, Any] = {spec.name: spec.default for spec in PROCESSING_PARAMS}


@dataclass
class ProcessingContext:
    """Fully-resolved, typed processing parameters for one engine.process() call."""

    # --- Skin ---
    smooth: float = _DEFAULTS["smooth"]
    whiten: float = 0.0
    whiten_tone: str = _DEFAULTS["whiten_tone"]
    equalize: float = 0.0
    blemish: float = 0.0
    nose_smooth: Optional[float] = None
    micro_restore: float = _DEFAULTS["micro_restore"]
    micro_dodge_burn: float = 0.0
    mid_reduction: float = _DEFAULTS["mid_reduction"]
    blotch_reduction: float = _DEFAULTS["blotch_reduction"]
    texture_opacity: float = _DEFAULTS["texture_opacity"]
    pore_synthesis: float = 0.0
    regional_modulation: float = 0.0
    smooth_engine: str = "guided"
    specular_bloom: float = 0.0
    specular_bloom_tone: str = _DEFAULTS["specular_bloom_tone"]
    specular_finish: str = _DEFAULTS["specular_finish"]
    specular_finish_strength: float = _DEFAULTS["specular_finish_strength"]
    specular_recolor: float = _DEFAULTS["specular_recolor"]
    albedo_even: float = _DEFAULTS["albedo_even"]
    makeup_coverage_even: float = _DEFAULTS["makeup_coverage_even"]
    makeup_cake_reduce: float = _DEFAULTS["makeup_cake_reduce"]
    hemoglobin_smooth: float = _DEFAULTS["hemoglobin_smooth"]
    mole_protect: float = _DEFAULTS["mole_protect"]
    vein_attenuate: float = _DEFAULTS["vein_attenuate"]
    dodge_burn: float = 0.0
    relight: float = 0.0
    relight_azimuth: float = _DEFAULTS["relight_azimuth"]
    relight_elevation: float = _DEFAULTS["relight_elevation"]
    face_exposure: float = _DEFAULTS["face_exposure"]
    sculpt: float = 0.0
    skin_flatten: float = 0.0
    skin_quantize: float = 0.0
    skin_unify: float = 0.0
    skin_unify_hue: float = -1.0
    skin_hue_unify: float = 0.0
    skin_chroma_even: float = 0.0
    redness_even: float = 0.0
    skin_glow: float = 0.0
    whiten_hue_stable: bool = False
    skin_locus: Optional[Dict[str, float]] = None
    smooth_exposure_lock: float = 0.0
    shine_removal: float = 0.0
    wrinkle_soften: float = 0.0
    wrinkle_soften_forehead: float = 0.0
    wrinkle_soften_nasolabial: float = 0.0
    wrinkle_soften_neck: float = 0.0
    texture_transplant: float = 0.0
    body_smooth: float = 0.0
    body_equalize: float = 0.0
    body_whiten: float = 0.0
    body_match_face: float = 0.0
    body_relight: float = 0.0
    body_dodge_burn: float = 0.0
    body_shadow_lift: float = 0.0
    shadow_lift: float = 0.0
    nose_restore: float = 0.0
    freckle_removal: float = 0.0
    freckle_preserve_mask: Optional[np.ndarray] = None

    # --- Eyes ---
    eye_enhance: float = 0.0
    eye_sclera_vessel_remove: float = 0.0
    backdrop_cleanup: float = 0.0
    fabric_wrinkle_smooth: float = 0.0
    dark_circles: float = 0.0
    undereye_darken_removal: float = 0.0
    undereye_puffiness_reduction: float = 0.0
    undereye_shadow_strength: float = 0.0
    catchlight: float = 0.0
    eye_sclera_brighten: float = 0.0
    eye_iris_saturate: float = 0.0
    eye_iris_hue_shift: float = 0.0
    eye_iris_brightness: float = 0.0

    # --- Lips ---
    lip_enhance: float = 0.0
    lip_tint: Optional[Any] = _DEFAULTS["lip_tint"]
    lip_finish: str = _DEFAULTS["lip_finish"]

    # --- Teeth ---
    teeth_whiten: float = 0.0

    # --- Makeup ---
    blush: float = 0.0
    slimming: float = 0.0

    # --- Face reshape (F5 liquify sliders) ---
    reshape_eye_size: float = 0.0
    reshape_eye_distance: float = 0.0
    reshape_nose_width: float = 0.0
    reshape_nose_length: float = 0.0
    reshape_jaw_width: float = 0.0
    reshape_chin_length: float = 0.0
    reshape_mouth_size: float = 0.0
    reshape_smile: float = 0.0
    reshape_forehead: float = 0.0
    reshape_jaw_width_l: float = 0.0
    reshape_jaw_width_r: float = 0.0
    reshape_nose_width_l: float = 0.0
    reshape_nose_width_r: float = 0.0
    reshape_eye_size_l: float = 0.0
    reshape_eye_size_r: float = 0.0
    reshape_neck_width: float = 0.0
    reshape_neck_length: float = 0.0

    # --- Hair ---
    hair_enhance: float = 0.0
    hair_deglare: float = 0.0
    hair_ring_position: float = 30.0
    hair_ring_tint: float = 40.0
    hair_remove_flyaways: float = 0.0

    # --- Tonal / global ---
    contrast: float = 0.0
    brightness: Optional[float] = _DEFAULTS["brightness"]
    highlights: Optional[float] = _DEFAULTS["highlights"]
    shadows: Optional[float] = _DEFAULTS["shadows"]
    whites: Optional[float] = _DEFAULTS["whites"]
    blacks: Optional[float] = _DEFAULTS["blacks"]
    clarity: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    # Color-science K3/K9 (PLAN_COLOR_SCIENCE.md): default-on gamut compression
    # is a no-op on in-gamut colors (golden path byte-identical); additive is the
    # legacy saturation mode.
    gamut_compress: bool = True
    saturation_mode: str = "additive"
    auto_exposure: bool = _DEFAULTS["auto_exposure"]

    # --- Colour grading ---
    color_grade: Optional[str] = None
    grade_intensity: float = _DEFAULTS["grade_intensity"]
    color_grade_stack: Optional[List[Dict]] = None
    color_ref: Optional[np.ndarray] = None
    color_transfer_intensity: float = 1.0
    chromatic_aberration: Optional[float] = None
    halation: Optional[Union[float, Dict[str, Any]]] = None
    grain: Optional[float] = None
    lut: Optional[str] = None
    tonal_curve_strength: float = _DEFAULTS["tonal_curve_strength"]
    skin_protect_strength: float = _DEFAULTS["skin_protect_strength"]
    grain_strength: float = _DEFAULTS["grain_strength"]
    highlight_rolloff_strength: float = _DEFAULTS["highlight_rolloff_strength"]

    # --- Film density engine (C3) ---
    film_enable: bool = _DEFAULTS["film_enable"]
    film_strength: float = _DEFAULTS["film_strength"]
    film_toe_r: float = _DEFAULTS["film_toe_r"]
    film_toe_g: float = _DEFAULTS["film_toe_g"]
    film_toe_b: float = _DEFAULTS["film_toe_b"]
    film_shoulder_r: float = _DEFAULTS["film_shoulder_r"]
    film_shoulder_g: float = _DEFAULTS["film_shoulder_g"]
    film_shoulder_b: float = _DEFAULTS["film_shoulder_b"]
    film_midpoint: float = _DEFAULTS["film_midpoint"]
    film_gamma: float = _DEFAULTS["film_gamma"]
    film_crosstalk_cy_mg: float = _DEFAULTS["film_crosstalk_cy_mg"]
    film_crosstalk_cy_ye: float = _DEFAULTS["film_crosstalk_cy_ye"]
    film_crosstalk_mg_ye: float = _DEFAULTS["film_crosstalk_mg_ye"]
    film_tonemap_strength: float = _DEFAULTS["film_tonemap_strength"]
    film_tonemap_toe: float = _DEFAULTS["film_tonemap_toe"]
    film_tonemap_shoulder: float = _DEFAULTS["film_tonemap_shoulder"]
    film_skew: float = _DEFAULTS["film_skew"]

    # --- Split toning ---
    shadow_hue: float = _DEFAULTS["shadow_hue"]
    shadow_sat: float = _DEFAULTS["shadow_sat"]
    midtone_hue: float = _DEFAULTS["midtone_hue"]
    midtone_sat: float = _DEFAULTS["midtone_sat"]
    highlight_hue: float = _DEFAULTS["highlight_hue"]
    highlight_sat: float = _DEFAULTS["highlight_sat"]

    # --- White balance / B&W mixer ---
    white_balance_kelvin: int = _DEFAULTS["white_balance_kelvin"]
    white_balance_tint: float = _DEFAULTS["white_balance_tint"]
    bw_channel_mixer_r: int = _DEFAULTS["bw_channel_mixer_r"]
    bw_channel_mixer_g: int = _DEFAULTS["bw_channel_mixer_g"]
    bw_channel_mixer_b: int = _DEFAULTS["bw_channel_mixer_b"]
    negative_split_tone_shadow: float = _DEFAULTS["negative_split_tone_shadow"]
    negative_split_tone_highlight: float = _DEFAULTS["negative_split_tone_highlight"]
    hsl_hue_global: int = _DEFAULTS["hsl_hue_global"]
    hsl_sat_global: int = _DEFAULTS["hsl_sat_global"]
    hsl_lum_global: int = _DEFAULTS["hsl_lum_global"]

    # --- Global Bloom (Oniric-Style Glow) ---
    bloom: float = 0.0
    bloom_threshold: float = _DEFAULTS["bloom_threshold"]
    bloom_softness: float = _DEFAULTS["bloom_softness"]
    glow: float = 0.0

    # --- Lens effects ---
    vignette: float = 0.0
    sharpen: float = 0.0
    sharpen_radius: float = _DEFAULTS["sharpen_radius"]

    # --- Subject separation ---
    subject_separation: float = 0.0

    # --- C5: Skin-anchored background color harmonization ---
    background_harmonize: float = 0.0
    background_harmonize_mode: str = "split"

    # --- T1: Background replace & scene relight (wires anime_crystal_void) ---
    background_blur: float = 0.0
    lens_blur: float = 0.0
    background_desaturation: float = 0.0
    light_wrap: float = 0.0
    blue_shadow_grade: float = 0.0
    cyan_midtone_grade: float = 0.0
    subject_sharpen: float = 0.0
    matte_black: float = 0.0

    # --- T2: Makeup v2 (eyeshadow, eyeliner, contour, brows, ombre lips) ---
    mv2_eyeshadow: int = _DEFAULTS["mv2_eyeshadow"]
    mv2_eyeshadow_color: str = _DEFAULTS["mv2_eyeshadow_color"]
    mv2_eyeshadow_style: str = _DEFAULTS["mv2_eyeshadow_style"]
    mv2_eyeliner: int = _DEFAULTS["mv2_eyeliner"]
    mv2_eyeliner_color: str = _DEFAULTS["mv2_eyeliner_color"]
    mv2_eyeliner_style: str = _DEFAULTS["mv2_eyeliner_style"]
    mv2_contour: int = _DEFAULTS["mv2_contour"]
    mv2_brows: int = _DEFAULTS["mv2_brows"]
    mv2_brows_color: str = _DEFAULTS["mv2_brows_color"]
    mv2_ombre: bool = _DEFAULTS["mv2_ombre"]
    mv2_ombre_color1: str = _DEFAULTS["mv2_ombre_color1"]
    mv2_ombre_color2: str = _DEFAULTS["mv2_ombre_color2"]

    # --- Finish ---
    impact: float = 0.0
    fade_toe: float = 0.0
    highlight_drift: float = 0.0
    airy_haze: float = 0.0
    clarity_split_neg: float = 0.0
    clarity_split_pos: float = 0.0

    # --- Internal recipe tag (used for conditional logic) ---
    active_recipe: str = "natural"

    # --- Modular flags ---
    nose_blush: bool = _DEFAULTS["nose_blush"]
    under_eye_blush: bool = _DEFAULTS["under_eye_blush"]
    white_costume_lift: bool = _DEFAULTS["white_costume_lift"]

    # F8.2: output quality tier. "full" (default) runs per-face work at native
    # resolution (detection+segmentation stay at proxy); "draft" keeps the
    # legacy all-proxy path for fast batch contact sheets.
    quality: str = "full"

    # Cached per-face detection + parsing; None ⇒ engine detects/parses.
    face_contexts: Optional[List["FaceContext"]] = None

    # Per-face recipe/param overrides keyed by face index (detection order).
    # None ⇒ every face uses this global context (golden path).
    face_params: Optional[Union[Dict[int, Dict[str, Any]], str]] = None

    # 16-bit RAF ingest: full-precision float32 [0,255] BGR source at the
    # native processing resolution. Set by process() when the input is
    # float32; threaded into global grading so the extra tonal headroom
    # survives inter-stage uint8 quantization. None ⇒ uint8 ingest.
    hi_ref: Optional[np.ndarray] = None

    # F4: Manual heal marks — list of {"mask_png_b64": str, "method": str}
    heals: Optional[List[Dict[str, Any]]] = None

    # F7: AI denoise + super-resolution.
    # ai_denoise: 0-100 opacity blend (0 = no-op).
    # ai_sr_scale: export-time upscale factor (1=off, 2, 4).
    ai_denoise: float = 0.0
    ai_sr_scale: int = 1

    # --- A4: Neural boosters (PARKED — await A1 evidence) ---
    # stray_hair_boost: 0-100 strength for flyaway hair detection/removal.
    #   Runs after F11 QA, gated on enabled=False by default.
    # defect_boost: 0-100 strength for blemish/pore defect detection.
    #   Runs after F11 QA, gated on enabled=False by default.
    # See MASTER_PLAN.md line 112 and retouch/neural_boosters.py for status.
    neural_stray_hair_boost: float = 0.0
    neural_defect_boost: float = 0.0

    # --- A3: Cosplay skin moat (makeup-agnostic enhancements) ---
    # wig_lace_blend: 0-100 strength for seamless wig-lace blending at hairline.
    # stockings_smooth: 0-100 strength for hosiery smoothing without over-blurring.
    # consistency_strength: 0-100 shoot-consistency lock (white-balance continuity).
    cosplay_wig_lace_blend: float = 0.0
    cosplay_stockings_smooth: float = 0.0
    cosplay_consistency_strength: float = 0.0

    # --- T3: Body reshape (MediaPipe Pose) ---
    # Proportional body editing via full-body pose detection.
    # Slider range 0-100 maps to ±15% displacement of body segment length.
    # 50 = neutral (no change), <50 = compress, >50 = expand.
    body_reshape_arm_length: float = 50.0
    body_reshape_leg_length: float = 50.0
    body_reshape_torso_width: float = 50.0
    body_reshape_shoulder_width: float = 50.0
    body_reshape_hip_width: float = 50.0
    auto_body_reshape: float = 0.0

    # --- Selective HSL Adjustments ---
    hsl_hue_red: float = 0.0
    hsl_sat_red: float = 0.0
    hsl_lum_red: float = 0.0
    hsl_hue_orange: float = 0.0
    hsl_sat_orange: float = 0.0
    hsl_lum_orange: float = 0.0
    hsl_hue_yellow: float = 0.0
    hsl_sat_yellow: float = 0.0
    hsl_lum_yellow: float = 0.0
    hsl_hue_green: float = 0.0
    hsl_sat_green: float = 0.0
    hsl_lum_green: float = 0.0
    hsl_hue_cyan: float = 0.0
    hsl_sat_cyan: float = 0.0
    hsl_lum_cyan: float = 0.0
    hsl_hue_blue: float = 0.0
    hsl_sat_blue: float = 0.0
    hsl_lum_blue: float = 0.0
    hsl_hue_purple: float = 0.0
    hsl_sat_purple: float = 0.0
    hsl_lum_purple: float = 0.0
    hsl_hue_magenta: float = 0.0
    hsl_sat_magenta: float = 0.0
    hsl_lum_magenta: float = 0.0

    # --- Calibration ---
    calibration_red_hue: float = 0.0
    calibration_red_sat: float = 0.0
    calibration_red_lum: float = 0.0
    calibration_green_hue: float = 0.0
    calibration_green_sat: float = 0.0
    calibration_green_lum: float = 0.0
    calibration_blue_hue: float = 0.0
    calibration_blue_sat: float = 0.0
    calibration_blue_lum: float = 0.0


# ---------------------------------------------------------------------------
# ProcessingResult — rich return value
# ---------------------------------------------------------------------------

class ProcessingResult(np.ndarray):
    """Return value of RetouchEngine.process().

    Inherits from np.ndarray so OpenCV, Pillow, and other downstream callers
    can treat it directly as a standard uint8 BGR image array.
    """

    def __new__(
        cls,
        image: np.ndarray,
        skin_mask: Optional[np.ndarray] = None,
        skin_hair_mask: Optional[np.ndarray] = None,
        sharpen_mask: Optional[np.ndarray] = None,
        lips_mask: Optional[np.ndarray] = None,
        face_count: int = 0,
        params: Optional[ProcessingContext] = None,
        timings: Optional[Dict[str, float]] = None,
        face_contexts: Optional[List["FaceContext"]] = None,
        qa: Optional[List[QAWarning]] = None,
        face_recipes: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        obj = np.asarray(image).view(cls)
        obj.image = image
        obj.skin_mask = skin_mask
        obj.skin_hair_mask = skin_hair_mask
        obj.sharpen_mask = sharpen_mask
        obj.lips_mask = lips_mask
        obj.face_count = face_count
        obj.params = params
        obj.timings = timings or {}
        obj.face_contexts = face_contexts
        obj.qa = qa or []
        obj.face_recipes = face_recipes
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.image = getattr(obj, "image", None)
        self.skin_mask = getattr(obj, "skin_mask", None)
        self.skin_hair_mask = getattr(obj, "skin_hair_mask", None)
        self.sharpen_mask = getattr(obj, "sharpen_mask", None)
        self.lips_mask = getattr(obj, "lips_mask", None)
        self.face_count = getattr(obj, "face_count", 0)
        self.params = getattr(obj, "params", None)
        self.timings = getattr(obj, "timings", {})
        self.face_contexts = getattr(obj, "face_contexts", None)
        self.qa = getattr(obj, "qa", None) or []
        self.face_recipes = getattr(obj, "face_recipes", None)


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------


def build_context(
    active_recipe: str,
    rec: Dict,
    overrides: Dict,
) -> ProcessingContext:
    """Translate a recipe dict + caller overrides into a ProcessingContext.

    Recipe values are treated as defaults; any non-None override wins.

    The per-parameter recipe lookup and unit conversion is data-driven from
    ``retouch.params.PROCESSING_PARAMS`` — see that module for the canonical
    list of every parameter the engine understands.
    """

    from .params import PROCESSING_PARAMS, _resolve_recipe_value, _lookup_recipe
    from .params import _resolve_dodge_burn, _engine_for_recipe_value

    def _ov(key, recipe_val):
        v = overrides.get(key)
        return v if v is not None else recipe_val

    # A handful of parameters have engine-side quirks that the data-driven
    # converter cannot express on its own.  These are the only hand-written
    # lookups remaining — the other 50+ come from PROCESSING_PARAMS.
    def _resolve_engine_value(spec) -> Any:
        # Specs that need a hand-written recipe lookup.  Each branch matches
        # the *spec name* (which is unique) rather than the conversion code
        # (which is shared with the GUI-side path).  Adding a new quirk is a
        # one-liner here plus one name in _ENGINE_QUIRK_CONVERSIONS.
        if spec.name == "dodge_burn":
            if "dodge_burn" not in rec:
                return spec.default
            return _resolve_dodge_burn(rec)
        if spec.name == "relight":
            # Engine mirrors the recipe's ``skin.relight`` (same path the GUI
            # uses, defined in ParamSpec.recipe_key). Returns 0 when the
            # recipe has no relight information.
            v = _lookup_recipe(rec, spec.recipe_key) if spec.recipe_key else None
            return 0.0 if v is None else min(float(v) * 100.0, 100.0)
        if spec.name == "whiten":
            # Historical engine quirk: ``rosy`` always wins if it is in the
            # recipe, even when ``porcelain`` is also set.  Reproduce that
            # exactly to keep behaviour identical.
            skin = rec.get("skin", {})
            r_rosy = skin.get("rosy", skin.get("porcelain", 0.0))
            return float(r_rosy) * 100.0
        if spec.name == "whiten_tone":
            return "porcelain" if "porcelain" in rec.get("skin", {}) else "rosy"
        if spec.name == "blemish":
            # Engine mirrors the recipe's ``frequency.smooth`` (the GUI does
            # the same and that is what the alias spec handles separately).
            return float(rec.get("frequency", {}).get("smooth", 0.5)) * 100.0
        if spec.name == "catchlight":
            eyes = rec.get("eyes", {})
            if "catchlight" in eyes:
                return float(eyes["catchlight"]) * 100.0
            if "iris" in eyes:
                return float(eyes["iris"]) * 100.0
            return spec.default
        if spec.name == "teeth_whiten":
            eyes = rec.get("eyes", {})
            if "teeth_whiten" in eyes:
                return float(eyes["teeth_whiten"]) * 100.0
            if "whites" in eyes:
                return float(eyes["whites"]) * 100.0
            return spec.default
        if spec.name == "eye_enhance":
            eyes = rec.get("eyes", {})
            val = eyes.get("iris", eyes.get("whites", 0.0))
            return float(val) * 100.0
        if spec.name == "lip_enhance":
            return float(rec.get("lips", {}).get("gloss", 0.0)) * 100.0
        if spec.name == "lip_tint":
            return rec.get("lips", {}).get("tint", None)
        if spec.name == "hair_enhance":
            return float(rec.get("hair", {}).get("shine", 0.0)) * 100.0
        if spec.name == "color_grade":
            return rec.get("color_harmony", {}).get("preset", None)
        if spec.name == "mid_reduction":
            # The default depends on the recipe's smooth value.
            rec_smooth_pct = float(rec.get("frequency", {}).get("smooth", 0.5)) * 100.0
            return float(rec.get("frequency", {}).get(
                "mid_reduction", 0.35 if rec_smooth_pct < 50 else 0.45))
        return _resolve_recipe_value(spec, rec)

    # Map spec names to their engine-side quirks.  Specs that need
    # spec-specific lookup logic land here; everything else uses the
    # generic converter via _resolve_recipe_value.
    _ENGINE_QUIRK_CONVERSIONS = {
        "dodge_burn",
        "relight",
        "whiten",
        "whiten_tone",
        "blemish",
        "catchlight",
        "teeth_whiten",
        "eye_enhance",
        "lip_enhance",
        "lip_tint",
        "hair_enhance",
        "color_grade",
        "mid_reduction",
    }

    # Compute every parameter's recipe-derived engine value, then apply
    # caller overrides on top.
    resolved: Dict[str, Any] = {}
    for spec in PROCESSING_PARAMS:
        if spec.name in _ENGINE_QUIRK_CONVERSIONS:
            recipe_val = _resolve_engine_value(spec)
        else:
            recipe_val = _resolve_recipe_value(spec, rec, use_engine_key=True)
        resolved[spec.name] = _ov(spec.name, recipe_val)

    # Default intensity to 1.0 when the caller supplies color_grade
    caller_grade = overrides.get("color_grade")
    caller_intensity = overrides.get("grade_intensity")
    resolved["grade_intensity"] = (
        caller_intensity if caller_intensity is not None
        else (1.0 if caller_grade is not None else resolved["grade_intensity"])
    )

    # Build the ProcessingContext.  Everything we resolved from the recipe
    # goes through the spec list; everything that comes purely from the
    # caller (color_ref, …) is forwarded verbatim.
    # The ``_CALLER_ONLY`` set lists spec names that are excluded from the
    # generic spec_kwargs loop and hand-wired below.  Some are truly
    # caller-only (color_ref, skin_locus); the post-effects family
    # (halation / grain / lut) plus auto_exposure and
    # color_transfer_intensity may also come from the recipe — the caller
    # override wins, and when neither supplies a value the historical
    # caller-only default is kept (so absent-from-recipe stays a no-op
    # instead of picking up the spec default).
    _CALLER_ONLY = {
        "nose_smooth",
        "auto_exposure",
        "chromatic_aberration",
        "halation",
        "grain",
        "lut",
        "color_grade_stack",
        "color_ref",
        "color_transfer_intensity",
        "skin_locus",
        "smooth_exposure_lock",
    }
    spec_kwargs = {
        spec.name: resolved[spec.name]
        for spec in PROCESSING_PARAMS
        if spec.name not in _CALLER_ONLY
    }

    # Recipe-or-caller resolution for the formerly caller-only params
    # (2026-07-12 recipe-coverage fix).  Caller override wins; otherwise the
    # recipe value (via the spec's recipe_key + conversion); otherwise the
    # historical fallback.
    _spec_by_name = {spec.name: spec for spec in PROCESSING_PARAMS}

    def _recipe_or_caller(name: str, fallback: Any) -> Any:
        v = overrides.get(name)
        if v is not None:
            return v
        spec = _spec_by_name[name]
        raw = _lookup_recipe(rec, spec.recipe_key) if spec.recipe_key else None
        if raw is None:
            return fallback
        return _engine_for_recipe_value(spec.conversion, raw)
    # Extract skin_locus from recipe (if present) or overrides (if caller-supplied).
    # Caller overrides win over recipe defaults.
    recipe_skin_locus = rec.get("skin", {}).get("locus")
    final_skin_locus = overrides.get("skin_locus") if overrides.get("skin_locus") is not None else recipe_skin_locus
    # Extract smooth_exposure_lock from recipe (skin.exposure_lock) or override.
    # Caller override wins over recipe default; absent → 0.0.
    recipe_exposure_lock = rec.get("skin", {}).get("exposure_lock")
    final_exposure_lock = (
        overrides.get("smooth_exposure_lock")
        if overrides.get("smooth_exposure_lock") is not None
        else recipe_exposure_lock
    )
    if final_exposure_lock is None:
        final_exposure_lock = 0.0

    return ProcessingContext(
        # Recipe-derived fields (data-driven) — the bulk of the context.
        **spec_kwargs,
        # Caller-only fields (no recipe source)
        auto_exposure=_recipe_or_caller("auto_exposure", False),
        color_grade_stack=overrides.get("color_grade_stack"),
        # BUGFIX-2: color_ref comes only from the caller override, never None-initialised twice
        color_ref=overrides.get("color_ref"),
        color_transfer_intensity=_recipe_or_caller("color_transfer_intensity", 1.0),
        chromatic_aberration=overrides.get("chromatic_aberration"),
        halation=_recipe_or_caller("halation", None),
        grain=_recipe_or_caller("grain", None),
        lut=_recipe_or_caller("lut", None),
        skin_locus=final_skin_locus,
        smooth_exposure_lock=final_exposure_lock,
        # ``nose_smooth`` reads the caller override first, then the recipe's
        # ``frequency.nose_smooth`` (0-1 fraction, converted to 0-100 like
        # ``frequency.smooth``); absent → None (nose smoothed with the face).
        nose_smooth=(
            overrides.get("nose_smooth")
            if overrides.get("nose_smooth") is not None
            else (
                float(rec.get("frequency", {}).get("nose_smooth")) * 100.0
                if rec.get("frequency", {}).get("nose_smooth") is not None
                else None
            )
        ),
        # Final fixed values
        active_recipe=active_recipe,
    )


# ---------------------------------------------------------------------------
# Mask utilities & per-face processing result
#
# These helpers and the per-face core pipeline live in retouch.perf_optimizations
# so the multiprocessing worker can call them without re-entering engine.py
# (which would create a circular import). They are re-exported from there
# at the top of this module.
# ---------------------------------------------------------------------------


@dataclass
class _CoreResult:
    """Output of the core pipeline (stages 0–6), returned by
    ``_run_core_pipeline`` and upscaled by ``_process_with_proxy``."""
    result: np.ndarray
    acc_skin: Optional[np.ndarray]
    acc_skin_hair: Optional[np.ndarray]
    acc_lips: Optional[np.ndarray]
    acc_sharpen: Optional[np.ndarray]
    faces: list
    person_mask: Optional[np.ndarray]
    no_face: bool = False
    face_contexts: Optional[List["FaceContext"]] = None
    qa: List[QAWarning] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RetouchEngine
# ---------------------------------------------------------------------------

class RetouchEngine:
    """Professional-grade automated face retouching engine.

    Usage::

        engine = RetouchEngine()
        result = engine.process(img_bgr, preset='cosplay')
        cv2.imwrite('out.jpg', result.image)

        # Or access debug info:
        result.face_count       # int
        result.timings          # {'detection': 42.1, 'per_face': 310.5, ...}
        result.params           # ProcessingContext

    The returned ProcessingResult behaves like an ndarray for legacy callers
    (``result.shape``, ``result[...]``, ``np.array(result)`` all work).
    """

    def __init__(self, max_faces: int = 10, min_confidence: float = 0.4):
        self._detector = FaceDetector(
            max_faces=max_faces,
            min_confidence=min_confidence,
            refine_landmarks=True,
        )
        self._parser = FaceParser()
        self._reshaper = FaceReshaper()
        self._makeup = MakeupEngine()
        self._makeup_v2 = MakeupEngineV2()
        self._frequency = FrequencySeparator()
        self._skin = SkinProcessor()
        self._blemish = BlemishRemover()
        self._eyes = EyeEnhancer()
        self._undereye = UnderEyeRepairer()
        self._lips = LipEnhancer()
        self._teeth = TeethWhitener()
        self._grader = ColorGrader()
        self._hair = HairEnhancer()
        self._relighter = Relighter()
        self._enhancer: Optional[AIEnhancer] = None
        self._harmonizer = BackgroundHarmonizer()
        self._background_replacer = BackgroundReplacer()

        # Persistent process pool for multi-face parallel processing.
        # Lazily started on first multi-face call; shut down in close().
        self._face_pool = FaceProcessorPool()

        # Load user-imported recipes from ~/.cache/retouch/user_recipes/
        loaded = load_user_recipes()
        if loaded:
            import logging
            logging.getLogger(__name__).info(f"Loaded {len(loaded)} user recipes: {loaded}")

        # T4: best-effort plugin discovery + init. A broken or missing plugin
        # must never crash engine startup; failures are logged and swallowed.
        try:
            from .plugin_api import PluginManager
            self._plugin_manager = PluginManager()
            self._plugin_manager.discover()
            self._plugin_manager.init_all(self)
        except Exception as exc:  # noqa: BLE001 — plugins are optional
            import logging as _logging
            _logging.getLogger(__name__).warning("Plugin discovery failed: %s", exc)

        # Warm up JIT kernels on engine startup (safe fallback if Numba is missing)

        # P3: Stage registry for global phases (stages 3-6).
        # Built once; used by _run_global_phases when use_registry=True.
        from .stage_wrappers import build_global_registry
        self._global_registry = build_global_registry(self)

        # A5: QA auto-back-off. Conservative param reduction driven by QA
        # flags (plastic-skin). Lazily reusable; stateless per image.
        self._qa_backoff = QABackoff()
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        img_bgr: np.ndarray,
        recipe: Optional[str] = None,
        preset: Optional[str] = None,
        # Individual controls (None → use recipe default)
        smooth: Optional[float] = None,
        whiten: Optional[float] = None,
        eye_enhance: Optional[float] = None,
        eye_sclera_vessel_remove: Optional[float] = None,
        eye_sclera_brighten: Optional[float] = None,
        eye_iris_saturate: Optional[float] = None,
        eye_iris_hue_shift: Optional[float] = None,
        eye_iris_brightness: Optional[float] = None,
        backdrop_cleanup: Optional[float] = None,
        fabric_wrinkle_smooth: Optional[float] = None,
        dark_circles: Optional[float] = None,
        undereye_darken_removal: Optional[float] = None,
        undereye_puffiness_reduction: Optional[float] = None,
        catchlight: Optional[float] = None,
        blemish: Optional[float] = None,
        blotch_reduction: Optional[float] = None,
        lip_enhance: Optional[float] = None,
        lip_tint: Optional[Any] = None,
        teeth_whiten: Optional[float] = None,
        equalize: Optional[float] = None,
        contrast: Optional[float] = None,
        brightness: Optional[float] = None,
        highlights: Optional[float] = None,
        shadows: Optional[float] = None,
        whites: Optional[float] = None,
        blacks: Optional[float] = None,
        clarity: Optional[float] = None,
        vibrance: Optional[float] = None,
        saturation: Optional[float] = None,
        saturation_mode: Optional[str] = None,
        gamut_compress: Optional[bool] = None,
        glow: Optional[float] = None,
        vignette: Optional[float] = None,
        sharpen: Optional[float] = None,
        sharpen_radius: Optional[float] = None,
        subject_separation: Optional[float] = None,
        color_grade: Optional[str] = None,
        grade_intensity: Optional[float] = None,
        texture_opacity: Optional[float] = None,
        pore_synthesis: Optional[float] = None,
        mid_reduction: Optional[float] = None,
        nose_smooth: Optional[float] = None,
        micro_restore: Optional[float] = None,
        micro_dodge_burn: Optional[float] = None,
        hair_enhance: Optional[float] = None,
        hair_deglare: Optional[float] = None,
        hair_ring_position: Optional[float] = None,
        hair_ring_tint: Optional[float] = None,
        hair_remove_flyaways: Optional[float] = None,
        dodge_burn: Optional[float] = None,
        relight: Optional[float] = None,
        relight_azimuth: Optional[float] = None,
        relight_elevation: Optional[float] = None,
        face_exposure: Optional[float] = None,
        sculpt: Optional[float] = None,
        fade_toe: Optional[float] = None,
        highlight_drift: Optional[float] = None,
        airy_haze: Optional[float] = None,
        clarity_split_neg: Optional[float] = None,
        clarity_split_pos: Optional[float] = None,
        slimming: Optional[float] = None,
        reshape_eye_size: Optional[float] = None,
        reshape_eye_distance: Optional[float] = None,
        reshape_nose_width: Optional[float] = None,
        reshape_nose_length: Optional[float] = None,
        reshape_jaw_width: Optional[float] = None,
        reshape_chin_length: Optional[float] = None,
        reshape_mouth_size: Optional[float] = None,
        reshape_smile: Optional[float] = None,
        reshape_forehead: Optional[float] = None,
        reshape_jaw_width_l: Optional[float] = None,
        reshape_jaw_width_r: Optional[float] = None,
        reshape_nose_width_l: Optional[float] = None,
        reshape_nose_width_r: Optional[float] = None,
        reshape_eye_size_l: Optional[float] = None,
        reshape_eye_size_r: Optional[float] = None,
        reshape_neck_width: Optional[float] = None,
        reshape_neck_length: Optional[float] = None,
        blush: Optional[float] = None,
        lip_finish: Optional[str] = None,
        specular_bloom: Optional[float] = None,
        specular_bloom_tone: Optional[str] = None,
        specular_finish: Optional[str] = None,
        specular_finish_strength: Optional[float] = None,
        specular_recolor: Optional[float] = None,
        albedo_even: Optional[float] = None,
        makeup_coverage_even: Optional[float] = None,
        makeup_cake_reduce: Optional[float] = None,
        hemoglobin_smooth: Optional[float] = None,
        mole_protect: Optional[float] = None,
        vein_attenuate: Optional[float] = None,
        whiten_tone: Optional[str] = None,
        nose_blush: Optional[bool] = None,
        under_eye_blush: Optional[bool] = None,
        white_costume_lift: Optional[bool] = None,
        auto_exposure: Optional[bool] = None,
        bloom: Optional[float] = None,
        bloom_threshold: Optional[float] = None,
        bloom_softness: Optional[float] = None,
        impact: Optional[float] = None,
        chromatic_aberration: Optional[float] = None,
        halation: Optional[Union[float, Dict[str, Any]]] = None,
        grain: Optional[float] = None,
        skin_flatten: Optional[float] = None,
        skin_quantize: Optional[float] = None,
        skin_unify: Optional[float] = None,
        skin_unify_hue: Optional[float] = None,
        skin_hue_unify: Optional[float] = None,
        skin_chroma_even: Optional[float] = None,
        redness_even: Optional[float] = None,
        whiten_hue_stable: Optional[bool] = None,
        skin_glow: Optional[float] = None,
        shine_removal: Optional[float] = None,
        wrinkle_soften: Optional[float] = None,
        wrinkle_soften_forehead: Optional[float] = None,
        wrinkle_soften_nasolabial: Optional[float] = None,
        wrinkle_soften_neck: Optional[float] = None,
        texture_transplant: Optional[float] = None,
        body_smooth: Optional[float] = None,
        body_equalize: Optional[float] = None,
        body_whiten: Optional[float] = None,
        body_match_face: Optional[float] = None,
        body_relight: Optional[float] = None,
        body_dodge_burn: Optional[float] = None,
        body_shadow_lift: Optional[float] = None,
        shadow_lift: Optional[float] = None,
        nose_restore: Optional[float] = None,
        regional_modulation: Optional[float] = None,
        smooth_engine: Optional[str] = None,
        undereye_shadow_strength: Optional[float] = None,
        freckle_removal: Optional[float] = None,
        freckle_preserve_mask: Optional[np.ndarray] = None,
        lut: Optional[str] = None,
        skin_locus: Optional[Dict[str, float]] = None,
        tonal_curve_strength: Optional[float] = None,
        skin_protect_strength: Optional[float] = None,
        grain_strength: Optional[float] = None,
        highlight_rolloff_strength: Optional[float] = None,
        shadow_hue: Optional[float] = None,
        shadow_sat: Optional[float] = None,
        midtone_hue: Optional[float] = None,
        midtone_sat: Optional[float] = None,
        highlight_hue: Optional[float] = None,
        highlight_sat: Optional[float] = None,
        white_balance_kelvin: Optional[int] = None,
        white_balance_tint: Optional[float] = None,
        bw_channel_mixer_r: Optional[int] = None,
        bw_channel_mixer_g: Optional[int] = None,
        bw_channel_mixer_b: Optional[int] = None,
        negative_split_tone_shadow: Optional[float] = None,
        negative_split_tone_highlight: Optional[float] = None,
        hsl_hue_global: Optional[int] = None,
        hsl_sat_global: Optional[int] = None,
        hsl_lum_global: Optional[int] = None,
        film_enable: Optional[bool] = None,
        film_strength: Optional[float] = None,
        film_toe_r: Optional[float] = None,
        film_toe_g: Optional[float] = None,
        film_toe_b: Optional[float] = None,
        film_shoulder_r: Optional[float] = None,
        film_shoulder_g: Optional[float] = None,
        film_shoulder_b: Optional[float] = None,
        film_midpoint: Optional[float] = None,
        film_gamma: Optional[float] = None,
        film_crosstalk_cy_mg: Optional[float] = None,
        film_crosstalk_cy_ye: Optional[float] = None,
        film_crosstalk_mg_ye: Optional[float] = None,
        film_tonemap_strength: Optional[float] = None,
        film_tonemap_toe: Optional[float] = None,
        film_tonemap_shoulder: Optional[float] = None,
        film_skew: Optional[float] = None,
        color_grade_stack=None,
        color_ref: Optional[np.ndarray] = None,
        color_transfer_intensity: Optional[float] = None,
        fast: bool = False,
        style_profile: Optional[StyleProfile] = None,
        style_ref: Optional[np.ndarray] = None,
        debug_dir: Optional[str] = None,
        face_contexts: Optional[List["FaceContext"]] = None,
        heals: Optional[List[Dict[str, Any]]] = None,
        quality: Optional[str] = None,
        local_adjustments: Optional[List[Dict[str, Any]]] = None,
        ai_denoise: Optional[float] = None,
        ai_sr_scale: Optional[int] = None,
        # --- C5: Skin-anchored background color harmonization ---
        background_harmonize: Optional[float] = None,
        background_harmonize_mode: Optional[str] = None,
        # --- T1: Background replace & scene relight ---
        background_blur: Optional[float] = None,
        lens_blur: Optional[float] = None,
        background_desaturation: Optional[float] = None,
        light_wrap: Optional[float] = None,
        blue_shadow_grade: Optional[float] = None,
        cyan_midtone_grade: Optional[float] = None,
        subject_sharpen: Optional[float] = None,
        matte_black: Optional[float] = None,
        # --- T2: Makeup v2 ---
        mv2_eyeshadow: Optional[int] = None,
        mv2_eyeshadow_color: Optional[str] = None,
        mv2_eyeshadow_style: Optional[str] = None,
        mv2_eyeliner: Optional[int] = None,
        mv2_eyeliner_color: Optional[str] = None,
        mv2_eyeliner_style: Optional[str] = None,
        mv2_contour: Optional[int] = None,
        mv2_brows: Optional[int] = None,
        mv2_brows_color: Optional[str] = None,
        mv2_ombre: Optional[bool] = None,
        mv2_ombre_color1: Optional[str] = None,
        mv2_ombre_color2: Optional[str] = None,
        # --- A3: Cosplay skin moat ---
        cosplay_wig_lace_blend: Optional[float] = None,
        cosplay_stockings_smooth: Optional[float] = None,
        cosplay_consistency_strength: Optional[float] = None,
        # --- A4: Neural boosters ---
        neural_stray_hair_boost: Optional[float] = None,
        neural_defect_boost: Optional[float] = None,
        # --- T3: Body reshape (MediaPipe Pose) ---
        body_reshape_arm_length: Optional[float] = None,
        body_reshape_leg_length: Optional[float] = None,
        body_reshape_torso_width: Optional[float] = None,
        body_reshape_shoulder_width: Optional[float] = None,
        body_reshape_hip_width: Optional[float] = None,
        # --- T3: Auto body reshape (one-click) ---
        auto_body_reshape: Optional[float] = None,
        # --- Per-face recipe / param overrides (detection-order index) ---
        face_params: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> ProcessingResult:
        """Process a single image through the full Retouch pipeline.

        Returns a ProcessingResult. Access ``.image`` for the BGR ndarray, or
        use the result directly as an ndarray (legacy-compatible).
        """
        timings: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # PERF-1: downscale *before* detection when fast=True
        # ------------------------------------------------------------------
        from .precision import to_uint8

        # ------------------------------------------------------------------
        # 16-bit ingest: accept float32 [0,255] BGR (from io.read_image_16bit /
        # imread_engine). The uint8-oriented detection/reshape/per-face stages
        # consume a rounded uint8 view; the full-precision source (hi_ref) is
        # threaded into global grading (see _run_global_phases) so the extra
        # tonal headroom survives inter-stage quantization instead of banding.
        # ------------------------------------------------------------------
        hi_ref: Optional[np.ndarray] = None
        if img_bgr.dtype != np.uint8:
            hi_ref = np.clip(img_bgr.astype(np.float32, copy=False), 0.0, 255.0)
            img_bgr = to_uint8(hi_ref / 255.0)

        orig_h, orig_w = img_bgr.shape[:2]
        scale = 1.0
        if fast:
            max_preview = 800
            scale = min(max_preview / orig_w, max_preview / orig_h, 1.0)
            if scale < 1.0:
                img_bgr = cv2.resize(
                    img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
                if hi_ref is not None:
                    hi_ref = cv2.resize(
                        hi_ref, (img_bgr.shape[1], img_bgr.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )

        # ------------------------------------------------------------------
        # Recipe resolution (RECIPE-1)
        # ------------------------------------------------------------------
        active_recipe = recipe or preset or "natural"
        if active_recipe not in RECIPES:
            active_recipe = "natural"
        rec = resolve_recipe(active_recipe)

        overrides = {
            "smooth": smooth,
            "whiten": whiten,
            "eye_enhance": eye_enhance,
            "eye_sclera_vessel_remove": eye_sclera_vessel_remove,
            "eye_sclera_brighten": eye_sclera_brighten,
            "eye_iris_saturate": eye_iris_saturate,
            "eye_iris_hue_shift": eye_iris_hue_shift,
            "eye_iris_brightness": eye_iris_brightness,
            "backdrop_cleanup": backdrop_cleanup,
            "fabric_wrinkle_smooth": fabric_wrinkle_smooth,
            "dark_circles": dark_circles,
            "undereye_darken_removal": undereye_darken_removal,
            "undereye_puffiness_reduction": undereye_puffiness_reduction,
            "catchlight": catchlight,
            "blemish": blemish,
            "blotch_reduction": blotch_reduction,
            "lip_enhance": lip_enhance,
            "lip_tint": lip_tint,
            "teeth_whiten": teeth_whiten,
            "equalize": equalize,
            "contrast": contrast,
            "brightness": brightness,
            "highlights": highlights,
            "shadows": shadows,
            "whites": whites,
            "blacks": blacks,
            "color_grade": color_grade,
            "grade_intensity": grade_intensity,
            "saturation_mode": saturation_mode,
            "gamut_compress": gamut_compress,
            "texture_opacity": texture_opacity,
            "pore_synthesis": pore_synthesis,
            "mid_reduction": mid_reduction,
            "nose_smooth": nose_smooth,
            "micro_restore": micro_restore,
            "micro_dodge_burn": micro_dodge_burn,
            "hair_enhance": hair_enhance,
            "hair_deglare": hair_deglare,
            "hair_ring_position": hair_ring_position,
            "hair_ring_tint": hair_ring_tint,
            "hair_remove_flyaways": hair_remove_flyaways,
            "dodge_burn": dodge_burn,
            "relight": relight,
            "relight_azimuth": relight_azimuth,
            "relight_elevation": relight_elevation,
            "face_exposure": face_exposure,
            "sculpt": sculpt,
            "fade_toe": fade_toe,
            "highlight_drift": highlight_drift,
            "airy_haze": airy_haze,
            "clarity_split_neg": clarity_split_neg,
            "clarity_split_pos": clarity_split_pos,
            "slimming": slimming,
            "reshape_eye_size": reshape_eye_size,
            "reshape_eye_distance": reshape_eye_distance,
            "reshape_nose_width": reshape_nose_width,
            "reshape_nose_length": reshape_nose_length,
            "reshape_jaw_width": reshape_jaw_width,
            "reshape_chin_length": reshape_chin_length,
            "reshape_mouth_size": reshape_mouth_size,
            "reshape_smile": reshape_smile,
            "reshape_forehead": reshape_forehead,
            "reshape_jaw_width_l": reshape_jaw_width_l,
            "reshape_jaw_width_r": reshape_jaw_width_r,
            "reshape_nose_width_l": reshape_nose_width_l,
            "reshape_nose_width_r": reshape_nose_width_r,
            "reshape_eye_size_l": reshape_eye_size_l,
            "reshape_eye_size_r": reshape_eye_size_r,
            "reshape_neck_width": reshape_neck_width,
            "reshape_neck_length": reshape_neck_length,
            "blush": blush,
            "lip_finish": lip_finish,
            "specular_bloom": specular_bloom,
            "specular_bloom_tone": specular_bloom_tone,
            "specular_finish": specular_finish,
            "specular_finish_strength": specular_finish_strength,
            "specular_recolor": specular_recolor,
            "albedo_even": albedo_even,
            "makeup_coverage_even": makeup_coverage_even,
            "makeup_cake_reduce": makeup_cake_reduce,
            "hemoglobin_smooth": hemoglobin_smooth,
            "mole_protect": mole_protect,
            "vein_attenuate": vein_attenuate,
            "whiten_tone": whiten_tone,
            "nose_blush": nose_blush,
            "under_eye_blush": under_eye_blush,
            "white_costume_lift": white_costume_lift,
            "auto_exposure": auto_exposure,
            "bloom": bloom,
            "bloom_threshold": bloom_threshold,
            "bloom_softness": bloom_softness,
            "impact": impact,
            "chromatic_aberration": chromatic_aberration,
            "halation": halation,
            "grain": grain,
            "skin_flatten": skin_flatten,
            "skin_quantize": skin_quantize,
            "skin_unify": skin_unify,
            "skin_unify_hue": skin_unify_hue,
            "skin_hue_unify": skin_hue_unify,
            "skin_chroma_even": skin_chroma_even,
            "redness_even": redness_even,
            "whiten_hue_stable": whiten_hue_stable,
            "skin_glow": skin_glow,
            "shine_removal": shine_removal,
            "wrinkle_soften": wrinkle_soften,
            "wrinkle_soften_forehead": wrinkle_soften_forehead,
            "wrinkle_soften_nasolabial": wrinkle_soften_nasolabial,
            "wrinkle_soften_neck": wrinkle_soften_neck,
            "texture_transplant": texture_transplant,
            "body_smooth": body_smooth,
            "body_equalize": body_equalize,
            "body_whiten": body_whiten,
            "body_match_face": body_match_face,
            "body_relight": body_relight,
            "body_dodge_burn": body_dodge_burn,
            "body_shadow_lift": body_shadow_lift,
            "shadow_lift": shadow_lift,
            "nose_restore": nose_restore,
            "lut": lut,
            "skin_locus": skin_locus,
            "tonal_curve_strength": tonal_curve_strength,
            "skin_protect_strength": skin_protect_strength,
            "grain_strength": grain_strength,
            "highlight_rolloff_strength": highlight_rolloff_strength,
            "shadow_hue": shadow_hue,
            "shadow_sat": shadow_sat,
            "midtone_hue": midtone_hue,
            "midtone_sat": midtone_sat,
            "highlight_hue": highlight_hue,
            "highlight_sat": highlight_sat,
            "white_balance_kelvin": white_balance_kelvin,
            "white_balance_tint": white_balance_tint,
            "bw_channel_mixer_r": bw_channel_mixer_r,
            "bw_channel_mixer_g": bw_channel_mixer_g,
            "bw_channel_mixer_b": bw_channel_mixer_b,
            "negative_split_tone_shadow": negative_split_tone_shadow,
            "negative_split_tone_highlight": negative_split_tone_highlight,
            "hsl_hue_global": hsl_hue_global,
            "hsl_sat_global": hsl_sat_global,
            "hsl_lum_global": hsl_lum_global,
            "film_enable": film_enable,
            "film_strength": film_strength,
            "film_toe_r": film_toe_r,
            "film_toe_g": film_toe_g,
            "film_toe_b": film_toe_b,
            "film_shoulder_r": film_shoulder_r,
            "film_shoulder_g": film_shoulder_g,
            "film_shoulder_b": film_shoulder_b,
            "film_midpoint": film_midpoint,
            "film_gamma": film_gamma,
            "film_crosstalk_cy_mg": film_crosstalk_cy_mg,
            "film_crosstalk_cy_ye": film_crosstalk_cy_ye,
            "film_crosstalk_mg_ye": film_crosstalk_mg_ye,
            "film_tonemap_strength": film_tonemap_strength,
            "film_tonemap_toe": film_tonemap_toe,
            "film_tonemap_shoulder": film_tonemap_shoulder,
            "film_skew": film_skew,
            "color_grade_stack": color_grade_stack,
            "color_ref": color_ref,
            "color_transfer_intensity": color_transfer_intensity,
            "clarity": clarity,
            "vibrance": vibrance,
            "saturation": saturation,
            "glow": glow,
            "vignette": vignette,
            "sharpen": sharpen,
            "sharpen_radius": sharpen_radius,
            "subject_separation": subject_separation,
            "ai_denoise": ai_denoise,
            "ai_sr_scale": ai_sr_scale,
            "background_harmonize": background_harmonize,
            "background_harmonize_mode": background_harmonize_mode,
            "background_blur": background_blur,
            "lens_blur": lens_blur,
            "background_desaturation": background_desaturation,
            "light_wrap": light_wrap,
            "blue_shadow_grade": blue_shadow_grade,
            "cyan_midtone_grade": cyan_midtone_grade,
            "subject_sharpen": subject_sharpen,
            "matte_black": matte_black,
            "cosplay_wig_lace_blend": cosplay_wig_lace_blend,
            "cosplay_stockings_smooth": cosplay_stockings_smooth,
            "cosplay_consistency_strength": cosplay_consistency_strength,
            "neural_stray_hair_boost": neural_stray_hair_boost,
            "neural_defect_boost": neural_defect_boost,
            "body_reshape_arm_length": body_reshape_arm_length,
            "body_reshape_leg_length": body_reshape_leg_length,
            "body_reshape_torso_width": body_reshape_torso_width,
            "body_reshape_shoulder_width": body_reshape_shoulder_width,
            "body_reshape_hip_width": body_reshape_hip_width,
            "auto_body_reshape": auto_body_reshape,
            "mv2_eyeshadow": mv2_eyeshadow,
            "mv2_eyeshadow_color": mv2_eyeshadow_color,
            "mv2_eyeshadow_style": mv2_eyeshadow_style,
            "mv2_eyeliner": mv2_eyeliner,
            "mv2_eyeliner_color": mv2_eyeliner_color,
            "mv2_eyeliner_style": mv2_eyeliner_style,
            "mv2_contour": mv2_contour,
            "mv2_brows": mv2_brows,
            "mv2_brows_color": mv2_brows_color,
            "mv2_ombre": mv2_ombre,
            "mv2_ombre_color1": mv2_ombre_color1,
            "mv2_ombre_color2": mv2_ombre_color2,
            "regional_modulation": regional_modulation,
            "smooth_engine": smooth_engine,
            "undereye_shadow_strength": undereye_shadow_strength,
            "freckle_removal": freckle_removal,
            "freckle_preserve_mask": freckle_preserve_mask,
        }
        overrides.update(kwargs)

        ctx = build_context(active_recipe, rec, overrides)
        ctx.hi_ref = hi_ref
        if face_contexts is not None:
            ctx.face_contexts = face_contexts
        if heals is not None:
            ctx.heals = heals
        if quality is not None:
            ctx.quality = quality
        if local_adjustments is not None:
            ctx._local_adjustments = local_adjustments
        if face_params is not None:
            from .face_params import coerce_face_params
            ctx.face_params = coerce_face_params(face_params)

        if style_profile is not None:
            if overrides["contrast"] is None:
                ctx.contrast = style_profile.contrast_delta
            if overrides["brightness"] is None:
                ctx.brightness = np.clip(style_profile.brightness_delta * 1.0, -100.0, 100.0)
            if overrides["smooth"] is None:
                ctx.smooth = np.clip(style_profile.skin_smooth_strength * 100.0, 0.0, 100.0)
            if overrides["mid_reduction"] is None:
                ctx.mid_reduction = np.clip(style_profile.skin_mid_reduction, 0.0, 1.0)
            if overrides["texture_opacity"] is None:
                ctx.texture_opacity = np.clip(style_profile.skin_texture_opacity, 0.0, 1.0)
            if overrides["whiten"] is None:
                ctx.whiten = np.clip(style_profile.skin_l_mean_delta * 4.0, -100.0, 100.0)
            if overrides["whiten_tone"] is None:
                if style_profile.skin_a_mean_delta > 1.0:
                    ctx.whiten_tone = "rosy"
                elif style_profile.skin_b_mean_delta < -1.0:
                    ctx.whiten_tone = "porcelain"
                else:
                    ctx.whiten_tone = "neutral"
            if overrides["saturation"] is None:
                ctx.saturation = np.clip(style_profile.saturation_delta, -100.0, 100.0)

        # ------------------------------------------------------------------
        # F7: Pre-pipeline AI denoise — runs before detection + heals so the
        # whole pipeline sees a cleaner input. strength is a 0-100 opacity
        # blend between the original and the denoised image.
        # ------------------------------------------------------------------
        if ctx.ai_denoise and ctx.ai_denoise > 0.0:
            if self._enhancer is None:
                self._enhancer = AIEnhancer()
            t_dn = time.perf_counter()
            strength = float(ctx.ai_denoise) / 100.0
            img_bgr = self._enhancer.denoise(img_bgr, strength=strength)
            timings["ai_denoise"] = (time.perf_counter() - t_dn) * 1000

        # ------------------------------------------------------------------
        # F4: Pre-pipeline heal hook — heals run BEFORE retouch/grade
        # ------------------------------------------------------------------
        if ctx.heals:
            from .heal import heal_region, b64_to_mask
            t_heal = time.perf_counter()
            for heal_entry in ctx.heals:
                mask_b64 = heal_entry.get("mask_png_b64", "")
                method = heal_entry.get("method", "telea")
                if not mask_b64:
                    continue
                try:
                    heal_mask = b64_to_mask(mask_b64, img_bgr.shape)
                    img_bgr = heal_region(img_bgr, heal_mask, method=method)
                except Exception as e:
                    logger.warning("Heal failed: %s", e)
            timings["heal"] = (time.perf_counter() - t_heal) * 1000

        # ------------------------------------------------------------------
        # Core pipeline (stages 0–6) with automatic proxy down/upscaling
        # for high-res inputs. The fast=True 800 px preview path is
        # independent and applied above; the proxy path triggers when the
        # (possibly already fast-downscaled) image still exceeds
        # PROXY_MAX_DIM, regardless of the ``fast`` flag.
        # ------------------------------------------------------------------
        core = self._process_with_proxy(img_bgr, ctx, style_ref, timings)

        result = core.result
        acc_skin = core.acc_skin
        acc_skin_hair = core.acc_skin_hair
        acc_lips = core.acc_lips
        acc_sharpen = core.acc_sharpen
        faces = core.faces
        person_mask = core.person_mask
        built_contexts = core.face_contexts

        # ------------------------------------------------------------------
        # No-face fallback: minimal global processing
        # ------------------------------------------------------------------
        if core.no_face:
            if fast and scale < 1.0:
                result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            result = self._apply_sr_export(result, ctx, timings)
            timings["total"] = sum(timings.values())
            return ProcessingResult(
                image=result,
                face_count=0,
                params=ctx,
                timings=timings,
                face_contexts=built_contexts,
                qa=core.qa,
            )

        # ------------------------------------------------------------------
        # Upscale if fast preview (result only — kept as-is for the 800px path)
        # ------------------------------------------------------------------
        if fast and scale < 1.0:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Direct LAB skin tone shift from style profile
        if style_profile is not None:
            if abs(style_profile.skin_a_mean_delta) > 0.5 or abs(style_profile.skin_b_mean_delta) > 0.5:
                if faces and acc_skin is not None and acc_skin.max() > 0.01:
                    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
                    lab[:, :, 1] += style_profile.skin_a_mean_delta * acc_skin
                    lab[:, :, 2] += style_profile.skin_b_mean_delta * acc_skin
                    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        timings["total"] = sum(timings.values())

        # ------------------------------------------------------------------
        # Debug Visualizations (masks and frequency layers)
        # ------------------------------------------------------------------
        if debug_dir is not None:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save accumulated masks
            cv2.imwrite(os.path.join(debug_dir, "skin_mask.png"), (acc_skin * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(debug_dir, "skin_hair_mask.png"), (acc_skin_hair * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(debug_dir, "lips_mask.png"), (acc_lips * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(debug_dir, "sharpen_mask.png"), (acc_sharpen * 255).astype(np.uint8))
            
            # Glow mask visualization
            pm_norm = _norm_mask(person_mask)
            if pm_norm is not None:
                pm_norm = squeeze_mask(pm_norm)
                sharp_fg = np.clip(pm_norm - acc_skin, 0.0, 1.0)
                g_mask = 1.0 - sharp_fg
            else:
                g_mask = np.ones(result.shape[:2], dtype=np.float32)
            cv2.imwrite(os.path.join(debug_dir, "glow_mask.png"), (g_mask * 255).astype(np.uint8))
            
            # Frequency layers of the primary face
            if faces:
                face_width = faces[0].ied * 2.5
                layers = self._frequency.separate(result, face_width)
                cv2.imwrite(os.path.join(debug_dir, "freq_low.png"), np.clip(layers.low, 0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_dir, "freq_mid.png"), np.clip(layers.mid + 128, 0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_dir, "freq_high.png"), np.clip(layers.high + 128, 0, 255).astype(np.uint8))

        # ------------------------------------------------------------------
        # F7: Export-time AI super-resolution. Runs after the full pipeline
        # (and after debug visualizations, which are at retouch resolution).
        # Masks are NOT upscaled — they stay at retouch resolution; the SR
        # output is the final image.
        # ------------------------------------------------------------------
        result = self._apply_sr_export(result, ctx, timings)

        face_recipes = None
        if ctx.face_params:
            face_recipes = {
                i: dict(ctx.face_params[i])
                for i in range(len(faces))
                if i in ctx.face_params
            }
        return ProcessingResult(
            image=result,
            skin_mask=acc_skin,
            skin_hair_mask=acc_skin_hair,
            sharpen_mask=acc_sharpen,
            lips_mask=acc_lips,
            face_count=len(faces),
            params=ctx,
            timings=timings,
            face_contexts=built_contexts,
            qa=core.qa,
            face_recipes=face_recipes or None,
        )

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def _apply_sr_export(
        self,
        result: np.ndarray,
        ctx: ProcessingContext,
        timings: Dict[str, float],
    ) -> np.ndarray:
        """F7 export-time super-resolution. No-op when ``ai_sr_scale <= 1``."""
        if not ctx.ai_sr_scale or ctx.ai_sr_scale <= 1:
            return result
        if self._enhancer is None:
            self._enhancer = AIEnhancer()
        t_sr = time.perf_counter()
        out = self._enhancer.super_resolve(result, scale=int(ctx.ai_sr_scale))
        timings["ai_sr"] = (time.perf_counter() - t_sr) * 1000
        return out

    def _process_with_proxy(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
    ) -> _CoreResult:
        """Wrap the core pipeline with automatic proxy down/upscaling.

        F8.2 (default, ``quality="full"``): detection+segmentation run at
        proxy resolution; face bboxes/ied scale back to native; reshape +
        per-face work (BiSeNet parsing, frequency separation, skin ops,
        composite) runs at native resolution. Global phases also run at
        native. Face texture/pore detail is never resampled — the proxy is
        only used to find faces and segment the person.

        F8.1 legacy (``quality="draft"``): detection+reshape+per-face all
        run at proxy, the result is upscaled and pasted onto the native
        background, and F8.0 detail reinjection recovers high-band texture
        in non-skin regions. Kept for fast batch contact sheets.

        If the longest image side ≤ ``PROXY_MAX_DIM``, no proxy is used and
        both paths collapse to the same native-resolution pipeline.
        """
        h, w = img_bgr.shape[:2]
        proxy_scale = 1.0
        native_img_bgr = img_bgr
        proxy_img_bgr = img_bgr

        if max(h, w) > PROXY_MAX_DIM:
            proxy_scale = PROXY_MAX_DIM / float(max(h, w))
            new_w = int(w * proxy_scale)
            new_h = int(h * proxy_scale)
            proxy_img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ------------------------------------------------------------------
        # F8.2 native face-crop path (default)
        # ------------------------------------------------------------------
        if proxy_scale < 1.0 and getattr(ctx, 'quality', 'full') == 'full':
            return self._process_native_faces(
                native_img_bgr, proxy_img_bgr, proxy_scale, ctx, style_ref, timings,
            )

        # ------------------------------------------------------------------
        # F8.1 legacy / no-proxy path
        # ------------------------------------------------------------------
        # Stages 0-2: detection+reshape+per-face at proxy resolution
        core = self._run_detection_and_faces(proxy_img_bgr, ctx, timings)

        if proxy_scale < 1.0:
            # Upscale face-region result + masks back to native resolution
            upscaled_core = self._upscale_core_result(core, h, w)

            # F8.1: Composite upscaled face edits onto native image
            # Only paste the face ROIs that were actually retouched
            composite_result = self._composite_upscaled_faces_onto_native(
                native_img_bgr, upscaled_core, core.faces
            )

            # Update core result with the native-resolution composite
            upscaled_core.result = composite_result
            core = upscaled_core

            # F8.0: reinject native detail — but post-F8.1, only the pasted
            # edit-mask regions come from the blurry upscaled proxy; the
            # background is already pristine native. Injecting the high band
            # frame-wide (the pre-F8.1 behavior) would ADD a second copy of
            # the high band on top of native pixels, over-sharpening the
            # background (~3.2x the input's Laplacian energy, measured).
            # Restrict reinjection to edit-mask regions, still excluding
            # deliberately-smoothed skin.
            if core.acc_skin is not None:
                edit_mask = core.acc_skin.copy()
                if core.acc_skin_hair is not None:
                    edit_mask = np.maximum(edit_mask, core.acc_skin_hair)
                if core.acc_lips is not None:
                    edit_mask = np.maximum(edit_mask, core.acc_lips)
                if edit_mask.max() > 0.01:
                    smooth_strength = ctx.smooth / 100.0 if ctx.smooth > 0 else 0.3
                    sigma = 2.0 * (1.0 / proxy_scale)
                    original_float = native_img_bgr.astype(np.float32)
                    blurred = cv2.GaussianBlur(original_float, (0, 0), sigma)
                    high_band = original_float - blurred
                    # Adaptive discount lift: flat/front-lit skin (low native
                    # high-band energy) needs fuller reinjection or the proxy
                    # round-trip reads as waxy. adapt==1.0 → byte-identical to
                    # the previous fixed behavior; adapt→0 → discount removed
                    # and gain raised toward 1.0. Strictly asymmetric: never
                    # reinjects less than before.
                    adapt = _texture_adaptation_factor(
                        high_band, core.acc_skin,
                        energy_low=REINJECT_ENERGY_LOW,
                        energy_high=REINJECT_ENERGY_HIGH,
                        floor=REINJECT_ADAPT_FLOOR,
                    )
                    if os.getenv("REINJECT_DEBUG"):
                        _hm = np.abs(high_band).mean(axis=2)[core.acc_skin > 0.5]
                        _med = float(np.median(_hm))
                        _energy = float(np.median(np.abs(_hm - _med))) * 1.4826
                        logger.warning(
                            "REINJECT_DEBUG energy=%.3f adapt=%.3f", _energy, adapt
                        )
                    weight = np.clip(
                        edit_mask * (1.0 - core.acc_skin * smooth_strength * adapt),
                        0.0, 1.0,
                    )[:, :, np.newaxis]
                    gain = REINJECT_BASE_GAIN + (1.0 - adapt) * (1.0 - REINJECT_BASE_GAIN)
                    result_f = core.result.astype(np.float32) + high_band * weight * gain
                    core.result = np.clip(result_f, 0, 255).astype(np.uint8)

        # Handle no-face case: run minimal global processing if needed
        if core.no_face:
            result = self._no_face_fallback(native_img_bgr if proxy_scale < 1.0 else proxy_img_bgr, ctx, core.person_mask)
            h_img, w_img = result.shape[:2]
            return _CoreResult(
                result=result,
                acc_skin=core.acc_skin,
                acc_skin_hair=core.acc_skin_hair,
                acc_lips=core.acc_lips,
                acc_sharpen=core.acc_sharpen,
                faces=core.faces,
                person_mask=core.person_mask,
                no_face=True,
                face_contexts=core.face_contexts,
                qa=core.qa,
            )

        # Stages 3+: global phases at native resolution (or proxy if no scaling needed)
        # When using proxy (proxy_scale < 1.0): use the composited upscaled result
        # When NOT using proxy (proxy_scale == 1.0): use the detection_and_faces result
        core = self._run_global_phases(
            core.result,
            ctx, style_ref, timings,
            acc_skin=core.acc_skin,
            acc_skin_hair=core.acc_skin_hair,
            acc_lips=core.acc_lips,
            acc_sharpen=core.acc_sharpen,
            faces=core.faces,
            person_mask=core.person_mask,
            face_contexts=core.face_contexts,
        )

        return core

    def _process_native_faces(
        self,
        native_img_bgr: np.ndarray,
        proxy_img_bgr: np.ndarray,
        proxy_scale: float,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
    ) -> _CoreResult:
        """F8.2: detection at proxy, per-face work at native resolution.

        Stages 0 (detection + person segmentation) run on the proxy image.
        Face bboxes and ied are then rescaled to native pixel coordinates
        (MediaPipe landmarks are normalized [0, 1] and resolution-independent).
        Stages 1–2 (reshape + per-face parsing/skin ops + composite) and
        stages 3+ (global phases) all run on the native image. Face texture
        and pore detail are never resampled through the proxy.
        """
        h_native, w_native = native_img_bgr.shape[:2]
        scale_up = 1.0 / proxy_scale  # proxy → native multiplier

        # ------------------------------------------------------------------
        # Stage 0 — Detection + segmentation at proxy resolution
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        cached_contexts = ctx.face_contexts
        if cached_contexts is not None:
            faces_proxy = [fc.face_data for fc in cached_contexts]
            if ctx.auto_exposure:
                bboxes = [f.bbox for f in faces_proxy] if faces_proxy else None
                proxy_img_bgr, corrected = correct_exposure(proxy_img_bgr, face_bboxes=bboxes)
                # Re-detect after exposure correction (bboxes may shift)
                if corrected:
                    faces_proxy = self._detector.detect(proxy_img_bgr)
                    cached_contexts = None  # cache stale after re-detect
        else:
            if ctx.auto_exposure:
                faces_proxy = self._detector.detect(proxy_img_bgr)
                bboxes = [f.bbox for f in faces_proxy] if faces_proxy else None
                proxy_img_bgr, corrected = correct_exposure(proxy_img_bgr, face_bboxes=bboxes)
                if corrected:
                    faces_proxy = self._detector.detect(proxy_img_bgr)
            else:
                faces_proxy = self._detector.detect(proxy_img_bgr)
        person_mask_proxy = self._detector.segment_person(proxy_img_bgr)
        timings["detection"] = (time.perf_counter() - t0) * 1000

        # ------------------------------------------------------------------
        # No-face fallback
        # ------------------------------------------------------------------
        if not faces_proxy:
            h_img, w_img = native_img_bgr.shape[:2]
            return _CoreResult(
                result=native_img_bgr.copy(),
                acc_skin=np.zeros((h_img, w_img), dtype=np.float32),
                acc_skin_hair=np.zeros((h_img, w_img), dtype=np.float32),
                acc_lips=np.zeros((h_img, w_img), dtype=np.float32),
                acc_sharpen=np.zeros((h_img, w_img), dtype=np.float32),
                faces=[],
                person_mask=cv2.resize(
                    person_mask_proxy, (w_native, h_native),
                    interpolation=cv2.INTER_LINEAR,
                ) if person_mask_proxy is not None else None,
                no_face=True,
                face_contexts=None,
                qa=[],
            )

        # ------------------------------------------------------------------
        # Scale face bboxes + ied to native (landmarks stay normalized [0,1])
        # ------------------------------------------------------------------
        def _scale_face(face: FaceData) -> FaceData:
            x, y, bw, bh = face.bbox
            return FaceData(
                landmarks=face.landmarks,  # normalized [0,1] — resolution-independent
                bbox=(
                    int(round(x * scale_up)),
                    int(round(y * scale_up)),
                    int(round(bw * scale_up)),
                    int(round(bh * scale_up)),
                ),
                ied=face.ied * scale_up,
                confidence=face.confidence,
            )

        faces_native = [_scale_face(f) for f in faces_proxy]

        # Upscale person mask to native (smooth, so INTER_LINEAR is fine)
        person_mask_native = (
            cv2.resize(
                person_mask_proxy, (w_native, h_native),
                interpolation=cv2.INTER_LINEAR,
            )
            if person_mask_proxy is not None
            else None
        )

        # Cached contexts were built at proxy-res and cannot be reused at
        # native — force re-parse by clearing the cache for this call only.
        # (The cache is most valuable for repeated calls on the same image,
        # where proxy_scale is stable; we preserve the original ctx and
        # only bypass the cache for this F8.2 invocation.)
        if cached_contexts is not None:
            logger.debug(
                "F8.2: discarding %d cached face_contexts (built at proxy res, "
                "invalid for native face crops)", len(cached_contexts),
            )
            # Mutate a copy so the caller's ctx isn't permanently altered.
            ctx = copy.copy(ctx)
            ctx.face_contexts = None

        # ------------------------------------------------------------------
        # Stages 1–2 — Reshape + per-face + composite at NATIVE resolution
        # ------------------------------------------------------------------
        if ctx.face_params == "auto":
            ctx.face_params = self.suggest_face_params(native_img_bgr, faces_data=faces_native)

        t1 = time.perf_counter()
        result_native = self._stage_reshape(native_img_bgr, faces_native, ctx)
        timings["reshape"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        h_img, w_img = result_native.shape[:2]
        face_results, built_contexts = self._stage_per_face(
            result_native, faces_native, person_mask_native, ctx, h_img, w_img,
        )
        timings["per_face"] = (time.perf_counter() - t2) * 1000

        result_native, acc_skin, acc_skin_hair, acc_lips, acc_sharpen = self._composite_faces(
            result_native, face_results, h_img, w_img,
        )

        core = _CoreResult(
            result=result_native,
            acc_skin=acc_skin,
            acc_skin_hair=acc_skin_hair,
            acc_lips=acc_lips,
            acc_sharpen=acc_sharpen,
            faces=faces_native,
            person_mask=person_mask_native,
            no_face=False,
            face_contexts=built_contexts,
            qa=[],
        )

        # ------------------------------------------------------------------
        # Stages 3+ — Global phases at native resolution
        # ------------------------------------------------------------------
        core = self._run_global_phases(
            core.result,
            ctx, style_ref, timings,
            acc_skin=core.acc_skin,
            acc_skin_hair=core.acc_skin_hair,
            acc_lips=core.acc_lips,
            acc_sharpen=core.acc_sharpen,
            faces=core.faces,
            person_mask=core.person_mask,
            face_contexts=core.face_contexts,
        )

        return core

    @staticmethod
    def _upscale_core_result(
        core: _CoreResult, target_h: int, target_w: int
    ) -> _CoreResult:
        """Upscale a core result (image + masks + person mask) back to the
        pre-proxy resolution."""
        core.result = cv2.resize(
            core.result, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
        if core.person_mask is not None:
            core.person_mask = cv2.resize(
                core.person_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        for attr in ("acc_skin", "acc_skin_hair", "acc_lips", "acc_sharpen"):
            m = getattr(core, attr)
            if m is not None:
                setattr(core, attr, cv2.resize(
                    m, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                ))
        return core

    @staticmethod
    def _composite_upscaled_faces_onto_native(
        native_img: np.ndarray,
        upscaled_core: _CoreResult,
        faces: list,
    ) -> np.ndarray:
        """F8.1: Paste upscaled face-region edits onto native image.

        Only the face ROI regions are taken from upscaled_core.result;
        everything outside face ROIs comes from the pristine native_img.
        This preserves native-resolution detail in non-face regions.
        """
        result = native_img.copy()
        h_native, w_native = result.shape[:2]

        # Upscaled core result has same dimensions as native
        upscaled_img = upscaled_core.result

        # For each detected face, paste its upscaled region onto native
        # The upscaled_core.result already has the upscaled per-face edits composited
        # We just use the minimal bounding region that actually changed
        if not faces or upscaled_core.acc_skin is None:
            # No faces detected or no skin work done — use upscaled result as-is
            return upscaled_img

        # Create a mask from accumulated skin mask to identify edited regions
        edit_mask = upscaled_core.acc_skin.copy()
        if upscaled_core.acc_skin_hair is not None:
            edit_mask = np.maximum(edit_mask, upscaled_core.acc_skin_hair)
        if upscaled_core.acc_lips is not None:
            edit_mask = np.maximum(edit_mask, upscaled_core.acc_lips)

        # Blend upscaled edits into native using the edit mask
        if edit_mask.max() > 0.01:
            alpha = np.clip(edit_mask, 0.0, 1.0)[:, :, np.newaxis]
            result_f = (
                upscaled_img.astype(np.float32) * alpha +
                native_img.astype(np.float32) * (1.0 - alpha)
            )
            result = np.clip(result_f, 0, 255).astype(np.uint8)

        return result

    def _run_detection_and_faces(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        timings: Dict[str, float],
    ) -> _CoreResult:
        """F8.1: Stages 0-2 only (detection + reshape + per-face processing).

        Returns the composited face-region result at the input image resolution,
        plus accumulated masks and face data. When no faces are detected, runs
        minimal no-face fallback and returns early.

        Does NOT run global phases (stages 3+); those are handled separately
        by _run_global_phases() at native resolution.
        """
        # ------------------------------------------------------------------
        # Stage 0 — Detection + segmentation (with FaceContext caching)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        cached_contexts = ctx.face_contexts
        if cached_contexts is not None:
            faces = [fc.face_data for fc in cached_contexts]
            if ctx.auto_exposure:
                bboxes = [f.bbox for f in faces] if faces else None
                img_bgr, corrected = correct_exposure(img_bgr, face_bboxes=bboxes)
        else:
            if ctx.auto_exposure:
                faces = self._detector.detect(img_bgr)
                bboxes = [f.bbox for f in faces] if faces else None
                img_bgr, corrected = correct_exposure(img_bgr, face_bboxes=bboxes)
                if corrected:
                    faces = self._detector.detect(img_bgr)
            else:
                faces = self._detector.detect(img_bgr)
        person_mask = self._detector.segment_person(img_bgr)
        timings["detection"] = (time.perf_counter() - t0) * 1000

        # ------------------------------------------------------------------
        # No-face fallback: return early with minimal data
        # (global phases will handle the image processing)
        # ------------------------------------------------------------------
        if not faces:
            h_img, w_img = img_bgr.shape[:2]
            return _CoreResult(
                result=img_bgr.copy(),
                acc_skin=np.zeros((h_img, w_img), dtype=np.float32),
                acc_skin_hair=np.zeros((h_img, w_img), dtype=np.float32),
                acc_lips=np.zeros((h_img, w_img), dtype=np.float32),
                acc_sharpen=np.zeros((h_img, w_img), dtype=np.float32),
                faces=[],
                person_mask=person_mask,
                no_face=True,
                face_contexts=None,
                qa=[],
            )

        # ------------------------------------------------------------------
        # Stage 1 — Face reshaping (global, applied once before per-face work)
        # ------------------------------------------------------------------
        if ctx.face_params == "auto":
            ctx.face_params = self.suggest_face_params(img_bgr, faces_data=faces)

        t1 = time.perf_counter()
        result = self._stage_reshape(img_bgr, faces, ctx)
        timings["reshape"] = (time.perf_counter() - t1) * 1000

        # ------------------------------------------------------------------
        # Stage 2 — Per-face processing (parallel when >1 face)
        # ------------------------------------------------------------------
        t2 = time.perf_counter()
        h_img, w_img = result.shape[:2]
        face_results, built_contexts = self._stage_per_face(
            result, faces, person_mask, ctx, h_img, w_img
        )
        timings["per_face"] = (time.perf_counter() - t2) * 1000

        # Composite per-face canvases back (or use single result directly)
        result, acc_skin, acc_skin_hair, acc_lips, acc_sharpen = self._composite_faces(
            result, face_results, h_img, w_img
        )

        final_contexts = built_contexts if built_contexts is not None else cached_contexts
        return _CoreResult(
            result=result,
            acc_skin=acc_skin,
            acc_skin_hair=acc_skin_hair,
            acc_lips=acc_lips,
            acc_sharpen=acc_sharpen,
            faces=faces,
            person_mask=person_mask,
            no_face=False,
            face_contexts=final_contexts,
            qa=[],
        )

    def _run_global_phases(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
        acc_skin: Optional[np.ndarray],
        acc_skin_hair: Optional[np.ndarray],
        acc_lips: Optional[np.ndarray],
        acc_sharpen: Optional[np.ndarray],
        faces: list,
        person_mask: Optional[np.ndarray],
        face_contexts: Optional[List["FaceContext"]],
    ) -> _CoreResult:
        """F8.1: Stages 3+ (global tonal, grading, sharpen/finish).

        Runs on the native-resolution image (either direct or composite of
        upscaled face edits + native background). Converts to float32 for
        global processing, then back to uint8, and runs QA detectors.

        Returns a complete _CoreResult with QA warnings populated.
        """
        from .precision import to_float, to_uint8

        # ------------------------------------------------------------------
        # F1: Convert to float32 [0,1] for Phase 3 global stages
        # This eliminates inter-stage uint8 quantization banding
        # ------------------------------------------------------------------
        # 16-bit ingest: outside the edited (face) regions the retouched
        # uint8 result is just a quantized copy of the full-precision source,
        # so restore hi_ref there before global grading. Edited regions keep
        # the retouched pixels. This lets the extra RAF headroom survive the
        # tonal moves (highlights/shadows/curves) without banding.
        hi_ref = getattr(ctx, "hi_ref", None)
        if hi_ref is not None and hi_ref.shape[:2] == img_bgr.shape[:2]:
            edited = to_float(img_bgr)
            base = np.clip(hi_ref.astype(np.float32) / 255.0, 0.0, 1.0)
            edit = np.zeros(img_bgr.shape[:2], dtype=np.float32)
            for _m in (acc_skin, acc_skin_hair, acc_lips, acc_sharpen):
                if _m is not None and _m.shape[:2] == img_bgr.shape[:2]:
                    edit = np.maximum(edit, _m.astype(np.float32))
            edit = np.clip(edit, 0.0, 1.0)[:, :, np.newaxis]
            result = (edited * edit + base * (1.0 - edit)).astype(np.float32)
        else:
            result = to_float(img_bgr)

        h_img, w_img = result.shape[:2]

        # ------------------------------------------------------------------
        # P3: Stage registry path (byte-identical alternative to the
        # hardcoded stage calls below). Toggle via use_registry to A/B test.
        # Once verified byte-identical via golden harness, this becomes
        # the default and the hardcoded path is removed.
        # ------------------------------------------------------------------
        use_registry = getattr(self, "_use_global_registry", True)
        if use_registry:
            from .stages import PipelineState as _PS
            state = _PS(
                img=result,
                ctx=ctx,
                h_img=h_img,
                w_img=w_img,
                person_mask=person_mask,
                acc_skin=acc_skin,
                acc_skin_hair=acc_skin_hair,
                acc_lips=acc_lips,
                acc_sharpen=acc_sharpen,
                faces=faces,
                style_ref=style_ref,
            )
            state = self._global_registry.run(state)
            result = state.img
            timings.update(state.timings)
        else:
            # ------------------------------------------------------------------
            # Stage 3 — Subject-background separation (now in float)
            # ------------------------------------------------------------------
            t_subj = time.perf_counter()
            if ctx.subject_separation > 0:
                result = self._stage_subject_separation(result, person_mask, ctx)
            timings["subject_separation"] = (time.perf_counter() - t_subj) * 1000

            # ------------------------------------------------------------------
            # Stage 3.1 — C5: Skin-anchored background harmonization
            # ------------------------------------------------------------------
            t_harm = time.perf_counter()
            if ctx.background_harmonize > 0:
                result = self._stage_harmonize(result, ctx, acc_skin, person_mask)
            timings["background_harmonize"] = (time.perf_counter() - t_harm) * 1000

            # ------------------------------------------------------------------
            # Stage 3.2 — T1: Background replace & scene relight
            # (wires the 7 historically-dead anime_crystal_void keys)
            # ------------------------------------------------------------------
            t_bg = time.perf_counter()
            result = self._stage_background(result, ctx, person_mask)
            timings["background"] = (time.perf_counter() - t_bg) * 1000

            # ------------------------------------------------------------------
            # Stage 3.5 — Body skin retouch (now in float)
            # ------------------------------------------------------------------
            t_body = time.perf_counter()
            result = self._stage_body_skin(result, ctx, person_mask, acc_skin, acc_skin_hair, acc_lips, faces, h_img, w_img)
            timings["body_skin"] = (time.perf_counter() - t_body) * 1000

            # ------------------------------------------------------------------
            # Stage 3.6 — A3: Cosplay skin moat (wig-lace, stockings, consistency)
            # ------------------------------------------------------------------
            t_cosplay = time.perf_counter()
            result = self._stage_cosplay_moat(result, ctx, acc_skin_hair, person_mask)
            timings["cosplay_moat"] = (time.perf_counter() - t_cosplay) * 1000

            # ------------------------------------------------------------------
            # Stage 4 — Global tonal adjustments (now in float)
            # ------------------------------------------------------------------
            t3 = time.perf_counter()
            result = self._stage_global(result, ctx)
            timings["global"] = (time.perf_counter() - t3) * 1000

            # ------------------------------------------------------------------
            # Stage 5 — Colour grading (now in float)
            # ------------------------------------------------------------------
            t4 = time.perf_counter()
            result = self._stage_grade(
                result, ctx, acc_skin, acc_lips, person_mask,
                style_ref=style_ref, faces=faces,
            )
            timings["grading"] = (time.perf_counter() - t4) * 1000

            # ------------------------------------------------------------------
            # Stage 5.5 — F3: Local adjustments (after grade, before finish)
            # ------------------------------------------------------------------
            t_local = time.perf_counter()
            local_adjs = getattr(ctx, "_local_adjustments", None)
            if local_adjs:
                sem_masks = {"skin": acc_skin, "person": person_mask} if acc_skin is not None else None
                result = self._stage_local_adjustments(result, ctx, local_adjustments=local_adjs, semantic_masks=sem_masks)
            timings["local_adjustments"] = (time.perf_counter() - t_local) * 1000

            # ------------------------------------------------------------------
            # Stage 6 — Selective sharpening + impact finish (now in float)
            # ------------------------------------------------------------------
            t5 = time.perf_counter()
            result = self._stage_finish(result, ctx, acc_sharpen, faces=faces, person_mask=person_mask)
            timings["finish"] = (time.perf_counter() - t5) * 1000

            # ------------------------------------------------------------------
            # Stage 7 — T3: Body reshape (now in float, native resolution)
            # ------------------------------------------------------------------
            t6 = time.perf_counter()
            result = self._stage_body_reshape(result, ctx, person_mask=person_mask)
            timings["body_reshape"] = (time.perf_counter() - t6) * 1000

        # ------------------------------------------------------------------
        # Stage 7.5 — Background cleanup (dust/specks/creases via inpaint)
        # Runs in both registry + hardcoded paths (inserted after stage run).
        # ------------------------------------------------------------------
        if ctx.backdrop_cleanup > 0 and person_mask is not None:
            from .backdrop import clean_backdrop

            t_bdc = time.perf_counter()
            result = clean_backdrop(result, person_mask, ctx.backdrop_cleanup)
            timings["backdrop_cleanup"] = (time.perf_counter() - t_bdc) * 1000

        # ------------------------------------------------------------------
        # Stage 7.6 — Fabric/clothing wrinkle smoothing (mid-frequency folds)
        # Runs in both registry + hardcoded paths (inserted after backdrop).
        # Cloth mask = person minus skin/hair/neck (acc_skin_hair accumulator).
        # ------------------------------------------------------------------
        if (
            ctx.fabric_wrinkle_smooth > 0
            and person_mask is not None
            and acc_skin_hair is not None
        ):
            cloth_mask = np.clip(person_mask - acc_skin_hair, 0.0, 1.0)
            if cloth_mask.max() > 0.01:
                from .fabric import smooth_fabric_wrinkles

                t_fab = time.perf_counter()
                result = smooth_fabric_wrinkles(result, cloth_mask, ctx.fabric_wrinkle_smooth)
                timings["fabric_wrinkle_smooth"] = (time.perf_counter() - t_fab) * 1000

        # ------------------------------------------------------------------
        # F1: Convert back to uint8 for output
        # ------------------------------------------------------------------
        result = to_uint8(result)

        # ------------------------------------------------------------------
        # QA detectors
        # ------------------------------------------------------------------
        qa_warnings: List[QAWarning] = self._run_qa(result, person_mask, img_bgr)
        ctx._qa_results = {w.detector: w.details for w in qa_warnings}

        # ------------------------------------------------------------------
        # A4: Neural boosters (PARKED — runs only if enabled, after QA)
        # ------------------------------------------------------------------
        t_neural = time.perf_counter()
        result = self._stage_neural_boosters(result, ctx, person_mask)
        timings["neural_boosters"] = (time.perf_counter() - t_neural) * 1000

        return _CoreResult(
            result=result,
            acc_skin=acc_skin,
            acc_skin_hair=acc_skin_hair,
            acc_lips=acc_lips,
            acc_sharpen=acc_sharpen,
            faces=faces,
            person_mask=person_mask,
            no_face=len(faces) == 0,
            face_contexts=face_contexts,
            qa=qa_warnings,
        )

    @staticmethod
    def _run_qa(
        result: np.ndarray,
        person_mask: Optional[np.ndarray],
        reference_img_bgr: Optional[np.ndarray] = None,
    ) -> List[QAWarning]:
        """Run QA detectors on a processed uint8 BGR image.

        Returns a list of :class:`QAWarning` (only flagged detectors).
        Encapsulated as a helper so :meth:`_run_core_pipeline` can re-run
        QA after a back-off iteration without re-entering
        :meth:`_run_global_phases`.
        """
        if person_mask is None or not np.any(person_mask > 0.3):
            return []
        qa_warnings: List[QAWarning] = []
        try:
            qa_raw = qa_detectors.run_all(
                result,
                skin_mask=person_mask,
                reference_img_bgr=reference_img_bgr,
                person_mask=person_mask,
            )
        except Exception as e:
            logger.warning("QA detectors raised, skipping QA: %s", e)
            return []
        for detector_name, det_result in qa_raw.items():
            if not det_result.get("flagged", False):
                continue
            msg = {
                "banding": "Banding visible in smooth gradient regions",
                "clipping": "Highlight/shadow clipping detected",
                "plastic_skin": "Skin texture loss detected — may appear plastic",
                "halo": "Edge overshoot halos detected from sharpening",
                "seam": "Seam visible at subject boundary",
                "color_drift": "Skin hue shift detected — color grade drifted beyond budget",
                "pore_spectrum": "Skin pore-spectrum loss detected — may appear plastic",
                "asymmetry": "Asymmetric over-smoothing detected — one face zone over-retouched",
                "skin_score": "Skin quality score low — plastic/over-evolved appearance",
            }.get(detector_name, f"{detector_name} artifact detected")
            _qa_thresholds = {
                "banding": qa_detectors.BANDING_THRESHOLD,
                "clipping": qa_detectors.CLIPPING_THRESHOLD,
                "plastic_skin": qa_detectors.PLASTIC_SKIN_THRESHOLD,
                "halo": qa_detectors.HALO_THRESHOLD,
                "seam": qa_detectors.SEAM_THRESHOLD,
                "color_drift": qa_detectors.COLOR_DRIFT_THRESHOLD,
                "pore_spectrum": qa_detectors.PORE_SPECTRUM_THRESHOLD,
                "asymmetry": qa_detectors.ASYMMETRY_THRESHOLD,
                # skin_score is informational (soft); no hard gate threshold.
            }
            qa_warnings.append(QAWarning(
                detector=detector_name,
                score=det_result.get("score", 0.0),
                flagged=True,
                threshold=_qa_thresholds.get(detector_name, 0.0),
                message=msg,
                details={k: v for k, v in det_result.items()
                         if k not in ("score", "flagged")},
            ))
        return qa_warnings

    def _run_core_pipeline(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
    ) -> _CoreResult:
        """F8.1: Wrapper that runs the full pipeline (stages 0–6).

        For backward compatibility, this delegates to the split methods:
        _run_detection_and_faces() for stages 0-2, then handles no-face
        fallback OR runs _run_global_phases() for stages 3+.

        Honours ``ctx.face_contexts``: when provided, detection and parsing
        are skipped and the cached face data / regions are reused. Otherwise
        detection + parsing run normally and ``FaceContext`` objects are
        built and returned for caller caching.
        """
        # Stages 0-2: detection + reshape + per-face
        core = self._run_detection_and_faces(img_bgr, ctx, timings)

        # No-face fallback: run minimal global processing
        if core.no_face:
            result = self._no_face_fallback(img_bgr, ctx, core.person_mask)
            h_img, w_img = result.shape[:2]
            return _CoreResult(
                result=result,
                acc_skin=np.zeros((h_img, w_img), dtype=np.float32),
                acc_skin_hair=np.zeros((h_img, w_img), dtype=np.float32),
                acc_lips=np.zeros((h_img, w_img), dtype=np.float32),
                acc_sharpen=np.zeros((h_img, w_img), dtype=np.float32),
                faces=core.faces,
                person_mask=core.person_mask,
                no_face=True,
                face_contexts=core.face_contexts,
                qa=core.qa,
            )

        # Stages 3+: global phases (tonal, grading, finish)
        core = self._run_global_phases(
            core.result,
            ctx, style_ref, timings,
            acc_skin=core.acc_skin,
            acc_skin_hair=core.acc_skin_hair,
            acc_lips=core.acc_lips,
            acc_sharpen=core.acc_sharpen,
            faces=core.faces,
            person_mask=core.person_mask,
            face_contexts=core.face_contexts,
        )

        # ------------------------------------------------------------------
        # A5: "No plastic skin" guarantee — QA auto-back-off.
        # If plastic-skin is flagged after the first pass, iteratively
        # reduce smoothing-related params and re-process. Re-detection is
        # skipped (face contexts are cached from pass 1) so only the
        # per-face skin work + global phases re-run. We keep the best
        # result: the first iteration that clears the flag wins; if none
        # clear, the most-reduced (last) result ships since it is the
        # least plastic.
        # ------------------------------------------------------------------
        backoff = getattr(self, "_qa_backoff", None)
        if backoff is not None and core.face_contexts and not core.no_face:
            best_core = core
            for iteration in range(backoff.max_iterations):
                plastic_flagged = any(
                    w.detector == "plastic_skin" and w.flagged
                    for w in best_core.qa
                )
                if not plastic_flagged:
                    break  # flag cleared — ship this result

                adjustments = backoff.check_and_backoff(
                    best_core.result, ctx, best_core.qa
                )
                if not adjustments:
                    break  # nothing to back off — give up

                logger.info(
                    "A5 back-off iteration %d: applying %s",
                    iteration + 1, adjustments,
                )
                # Preserve original ctx values so we can revert if the
                # back-off made things worse (e.g. a different artifact
                # appeared). The last iteration's ctx is what ships.
                QABackoff.apply_adjustments(ctx, adjustments)

                # Re-use cached face contexts to skip re-detection /
                # re-parsing — only the per-face skin work + global
                # phases re-run with the adjusted params.
                re_core = self._run_detection_and_faces(
                    img_bgr, ctx, timings,
                )
                if re_core.no_face:
                    break  # detection diverged — keep best_core
                re_core = self._run_global_phases(
                    re_core.result,
                    ctx, style_ref, timings,
                    acc_skin=re_core.acc_skin,
                    acc_skin_hair=re_core.acc_skin_hair,
                    acc_lips=re_core.acc_lips,
                    acc_sharpen=re_core.acc_sharpen,
                    faces=re_core.faces,
                    person_mask=re_core.person_mask,
                    face_contexts=re_core.face_contexts,
                )
                best_core = re_core

                # If the flag cleared on this iteration, stop early.
                still_plastic = any(
                    w.detector == "plastic_skin" and w.flagged
                    for w in best_core.qa
                )
                if not still_plastic:
                    break

            core = best_core

        return core

    @staticmethod
    def _assemble_post_effects(ctx: ProcessingContext) -> Dict[str, Any]:
        post_effects: Dict[str, Any] = {}
        if ctx.chromatic_aberration is not None:
            post_effects["chromatic_aberration"] = ctx.chromatic_aberration
        if ctx.halation is not None:
            post_effects["halation"] = ctx.halation
        if ctx.grain is not None:
            post_effects["grain"] = ctx.grain
        if ctx.lut is not None:
            post_effects["lut"] = ctx.lut
        return post_effects

    def _no_face_fallback(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """F1/E2: uint8 no-face path fixed — now converts to float32 [0,1]
        at the top (matching _run_global_phases) so --global-only batch runs
        get the same float-pipeline benefit as face-detected runs."""
        from .precision import to_float, to_uint8
        # 16-bit ingest: no faces ⇒ nothing is edited, so start global grading
        # from the full-precision source when available (no banding on skies/
        # gradients under heavy grade).
        hi_ref = getattr(ctx, "hi_ref", None)
        if hi_ref is not None and hi_ref.shape[:2] == img.shape[:2]:
            result = np.clip(hi_ref.astype(np.float32) / 255.0, 0.0, 1.0)
        else:
            result = to_float(img.copy())

        if ctx.subject_separation > 0:
            result = self._stage_subject_separation(result, person_mask, ctx)
        result = self._stage_global(result, ctx)

        film_tonemap_active = bool(ctx.film_enable and ctx.film_tonemap_strength > 0)

        if ctx.tonal_curve_strength > 0 and not film_tonemap_active:
            # tonal.apply_hd_curve expects uint8 — boundary conversion
            result_u8 = to_uint8(result)
            result_u8 = tonal.apply_hd_curve(result_u8, strength=ctx.tonal_curve_strength)
            result = to_float(result_u8)

        # --- White balance (LCH-based, Phase 1.d) ---
        if ctx.white_balance_kelvin != _DEFAULTS["white_balance_kelvin"] or ctx.white_balance_tint != _DEFAULTS["white_balance_tint"]:
            # white_balance_lch expects uint8 — boundary conversion
            result_u8 = to_uint8(result)
            result_u8 = self._grader.white_balance_lch(
                result_u8,
                temperature=ctx.white_balance_kelvin,
                tint=ctx.white_balance_tint,
            )
            result = to_float(result_u8)

        # --- Master HSL (Phase 1.d) — global LCH adjustments ---
        if ctx.hsl_hue_global != 0 or ctx.hsl_sat_global != 0 or ctx.hsl_lum_global != 0:
            result_u8 = to_uint8(result)
            result_u8 = self._grader.adjust_hsl_lch(
                result_u8,
                hue_shift=ctx.hsl_hue_global * 0.6,
                sat_scale=1.0 + ctx.hsl_sat_global / 100.0,
                lum_shift=ctx.hsl_lum_global * 0.5,
            )
            result = to_float(result_u8)

        # --- C3: Parametric film-density engine (no-face path) ---
        if ctx.film_enable:
            from .film import FilmDensityEngine
            _film_engine = FilmDensityEngine()
            result = _film_engine.apply_from_context(result, ctx)

        post_effects = self._assemble_post_effects(ctx)
        skip_glows = ctx.bloom > 0.0

        if ctx.film_enable and ctx.lut is not None and ctx.lut != "none":
            logger.warning(
                "film.enable=true with lut=%r: LUT skipped (mutually exclusive "
                "with C3 film engine)", ctx.lut,
            )
            post_effects.pop("lut", None)

        # grade_intensity == 0 means "no grade" — skip entirely so a preset set
        # with amount 0 (e.g. color_harmony.amount: 0.0) doesn't still leak the
        # preset's post-effects (grain/halation/CA/lut) onto the image.
        has_custom_hsl = any(getattr(ctx, f"hsl_hue_{c}", 0.0) != 0.0 or getattr(ctx, f"hsl_sat_{c}", 0.0) != 0.0 or getattr(ctx, f"hsl_lum_{c}", 0.0) != 0.0 for c in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"])
        has_custom_calib = any(getattr(ctx, f"calibration_{c}_hue", 0.0) != 0.0 or getattr(ctx, f"calibration_{c}_sat", 0.0) != 0.0 or getattr(ctx, f"calibration_{c}_lum", 0.0) != 0.0 for c in ["red", "green", "blue"])

        if (ctx.color_grade and ctx.grade_intensity > 0) or has_custom_hsl or has_custom_calib:
            active_grade = ctx.color_grade if (ctx.color_grade and ctx.color_grade != "none") else "natural"
            intensity = ctx.grade_intensity if (ctx.color_grade and ctx.color_grade != "none") else 1.0
            settings = PRESETS.get(active_grade, PRESETS["natural"]).copy()
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]
            settings["gamut_compress"] = ctx.gamut_compress
            settings["saturation_mode"] = ctx.saturation_mode

            h_adj = {}
            s_adj = {}
            l_adj = {}
            for color in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]:
                h_val = getattr(ctx, f"hsl_hue_{color}", 0.0)
                s_val = getattr(ctx, f"hsl_sat_{color}", 0.0)
                l_val = getattr(ctx, f"hsl_lum_{color}", 0.0)
                if h_val != 0.0: h_adj[color] = h_val
                if s_val != 0.0: s_adj[color] = s_val
                if l_val != 0.0: l_adj[color] = l_val
            
            if h_adj or s_adj or l_adj:
                if "hsl_adjustments" not in settings:
                    settings["hsl_adjustments"] = {}
                if "hue" not in settings["hsl_adjustments"]:
                    settings["hsl_adjustments"]["hue"] = {}
                if "saturation" not in settings["hsl_adjustments"]:
                    settings["hsl_adjustments"]["saturation"] = {}
                if "luminance" not in settings["hsl_adjustments"]:
                    settings["hsl_adjustments"]["luminance"] = {}
                
                settings["hsl_adjustments"]["hue"].update(h_adj)
                settings["hsl_adjustments"]["saturation"].update(s_adj)
                settings["hsl_adjustments"]["luminance"].update(l_adj)

            calib = {}
            for color in ["red", "green", "blue"]:
                h_val = getattr(ctx, f"calibration_{color}_hue", 0.0)
                s_val = getattr(ctx, f"calibration_{color}_sat", 0.0)
                l_val = getattr(ctx, f"calibration_{color}_lum", 0.0)
                if h_val != 0.0 or s_val != 0.0 or l_val != 0.0:
                    calib[color] = {"hue": h_val, "sat": s_val, "lum": l_val}
            
            if calib:
                if "calibration" not in settings:
                    settings["calibration"] = {}
                for color, vals in calib.items():
                    if color not in settings["calibration"]:
                        settings["calibration"][color] = {}
                    settings["calibration"][color].update(vals)

            result = self._grader.grade(
                result, settings, intensity,
                skip_glows=skip_glows, skip_post_effects=True,
                skin_protect_strength=ctx.skin_protect_strength,
                return_float=True,
            )
        # Remaining ops are uint8-contract — boundary conversions
        result_u8 = to_uint8(result)

        if ctx.highlight_rolloff_strength > 0:
            result_u8 = highlight.apply_highlight_rolloff(result_u8, ctx.highlight_rolloff_strength)

        if ctx.bloom > 0.0:
            result_u8 = apply_global_bloom(
                result_u8,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )

        if ctx.glow > 0:
            result_u8 = self._grader._add_glow(result_u8, ctx.glow / 100.0)

        if post_effects:
            result_u8 = self._grader.grade(
                result_u8, post_effects, 1.0,
                skin_protect_strength=ctx.skin_protect_strength,
            )

        if ctx.vignette > 0:
            result_u8 = self._grader._add_vignette(result_u8, ctx.vignette / 100.0)

        if ctx.impact > 0:
            result_u8 = self._grader.add_impact_finish(result_u8, ctx.impact, subject_mask=person_mask)

        if ctx.grain_strength > 0:
            result_u8 = grain.apply_film_grain(result_u8, ctx.grain_strength)

        # --- Negative split tone (Phase 1.d) — desaturate shadows/highlights ---
        if ctx.negative_split_tone_shadow > 0 or ctx.negative_split_tone_highlight > 0:
            result_u8 = self._grader.negative_split_tone(
                result_u8,
                shadow_desat=ctx.negative_split_tone_shadow / 100.0,
                highlight_desat=ctx.negative_split_tone_highlight / 100.0,
            )

        # --- B&W channel mixer (Phase 1.d) — applied last ---
        bw_active = (
            ctx.bw_channel_mixer_r != _DEFAULTS["bw_channel_mixer_r"]
            or ctx.bw_channel_mixer_g != _DEFAULTS["bw_channel_mixer_g"]
            or ctx.bw_channel_mixer_b != _DEFAULTS["bw_channel_mixer_b"]
        )
        if bw_active:
            result_u8 = self._grader.channel_mixer_bw(
                result_u8,
                r_weight=ctx.bw_channel_mixer_r / 100.0,
                g_weight=ctx.bw_channel_mixer_g / 100.0,
                b_weight=ctx.bw_channel_mixer_b / 100.0,
            )

        return result_u8

    def _ctx_for_face(self, ctx: ProcessingContext, face_index: int) -> ProcessingContext:
        """Resolve per-face overrides; identity when face_params empty/missing."""
        if not ctx.face_params or ctx.face_params == "auto":
            return ctx
        ov = ctx.face_params.get(face_index)
        if not ov:
            return ctx
        from .face_params import resolve_face_context
        return resolve_face_context(ctx, ov)

    def suggest_face_params(
        self,
        img_bgr: np.ndarray,
        face_contexts: Optional[List["FaceContext"]] = None,
        faces_data: Optional[List["FaceData"]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Suggest per-face recipe overrides based on classical demographics detection."""
        if faces_data is None:
            if not face_contexts:
                faces_data = self._detector.detect(img_bgr)
            else:
                faces_data = [f.face_data for f in face_contexts]

        from .face_params import suggest_face_recipe
        out = {}
        for i, face_data in enumerate(faces_data):
            recipe = suggest_face_recipe(img_bgr, face_data.landmarks, face_data.bbox, face_data.ied)
            out[i] = {"recipe": recipe}
        return out

    def _face_ctxs_for_reshape(self, ctx: ProcessingContext, n_faces: int):
        """Per-face contexts for reshape B-lite, or None when no face_params."""
        if not ctx.face_params or n_faces <= 0:
            return None
        return [self._ctx_for_face(ctx, i) for i in range(n_faces)]

    def _stage_reshape(self, img: np.ndarray, faces, ctx: ProcessingContext) -> np.ndarray:
        face_ctxs = self._face_ctxs_for_reshape(ctx, len(faces) if faces else 0)
        if self._any_reshape_active(ctx, face_ctxs):
            return self._reshaper.reshape(img, faces, ctx, face_ctxs=face_ctxs)
        return img.copy()

    @staticmethod
    def _any_reshape_active(
        ctx: ProcessingContext,
        face_ctxs: Optional[List[ProcessingContext]] = None,
    ) -> bool:
        def _active(c: ProcessingContext) -> bool:
            if c.slimming > 0:
                return True
            return any(
                getattr(c, attr, 0.0) != 0
                for attr in (
                    "reshape_eye_size",
                    "reshape_eye_distance",
                    "reshape_nose_width",
                    "reshape_nose_length",
                    "reshape_jaw_width",
                    "reshape_chin_length",
                    "reshape_mouth_size",
                    "reshape_smile",
                    "reshape_forehead",
                    "reshape_jaw_width_l",
                    "reshape_jaw_width_r",
                    "reshape_nose_width_l",
                    "reshape_nose_width_r",
                    "reshape_eye_size_l",
                    "reshape_eye_size_r",
                    "reshape_neck_width",
                    "reshape_neck_length",
                )
            )

        if face_ctxs:
            return any(_active(c) for c in face_ctxs) or _active(ctx)
        return _active(ctx)

    @staticmethod
    def _compute_face_roi_padding(
        face_w: int, face_h: int
    ) -> Tuple[int, int, int, int]:
        return (
            int(face_h * 0.8),
            int(face_h * 1.8),
            int(face_w * 0.6),
            int(face_w * 0.6),
        )

    def _stage_per_face(
        self,
        img: np.ndarray,
        faces,
        person_mask: np.ndarray,
        ctx: ProcessingContext,
        h_img: int,
        w_img: int,
    ) -> Tuple[List[_FaceResult], Optional[List["FaceContext"]]]:
        """Process each face. Cropping and region parsing are batched first,
        then smoothing and styling are parallelised.

        When ``ctx.face_contexts`` is provided, detection + parsing are
        skipped and the cached regions are reused; otherwise regions are
        parsed and ``FaceContext`` objects are built for caller caching.

        Returns ``(face_results, built_contexts)`` where ``built_contexts``
        is the freshly-built ``FaceContext`` list, or ``None`` when cached
        contexts were supplied (reuse path)."""
        crop_list = []
        landmarks_compat_list = []
        face_bbox_list = []
        person_masks = []
        ieds = []
        prepared_faces = []

        for face in faces:
            face_x, face_y, face_w, face_h = face.bbox
            pad_t, pad_b, pad_l, pad_r = self._compute_face_roi_padding(face_w, face_h)

            roi_y1 = max(0, face_y - pad_t)
            roi_y2 = min(h_img, face_y + face_h + pad_b)
            roi_x1 = max(0, face_x - pad_l)
            roi_x2 = min(w_img, face_x + face_w + pad_r)

            canvas = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            roi_person_mask = (
                person_mask[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                if person_mask is not None
                else None
            )

            shifted_landmarks = copy.deepcopy(face.landmarks)
            for lm in shifted_landmarks.landmark:
                lm.x = (lm.x * w_img - roi_x1) / (roi_x2 - roi_x1)
                lm.y = (lm.y * h_img - roi_y1) / (roi_y2 - roi_y1)

            shifted_bbox = (face_x - roi_x1, face_y - roi_y1, face_w, face_h)

            crop_list.append(canvas)
            landmarks_compat_list.append(shifted_landmarks)
            face_bbox_list.append(shifted_bbox)
            person_masks.append(roi_person_mask)
            ieds.append(face.ied)

            prepared_faces.append({
                'roi_box': (roi_x1, roi_y1, roi_x2, roi_y2),
                'canvas': canvas,
                'roi_person_mask': roi_person_mask,
                'shifted_landmarks': shifted_landmarks,
                'shifted_bbox': shifted_bbox
            })

        # Region parsing — or reuse cached regions from ctx.face_contexts
        cached_contexts = ctx.face_contexts
        if cached_contexts is not None:
            all_regions = [fc.regions for fc in cached_contexts]
            built_contexts: Optional[List["FaceContext"]] = None
        else:
            all_regions = self._parser.parse_batch(
                crop_list, landmarks_compat_list, face_bbox_list, person_masks, ieds
            )
            built_contexts = [
                FaceContext(
                    face_data=faces[i],
                    regions=all_regions[i],
                    index=i,
                    face_image=crop_list[i],
                )
                for i in range(len(faces))
            ]

        results: List[Optional[_FaceResult]] = [None] * len(faces)
        if len(faces) == 1:
            results[0] = self._process_one_face(
                img, faces[0], person_mask, self._ctx_for_face(ctx, 0), h_img, w_img,
                regions=all_regions[0], preprepared=prepared_faces[0]
            )
            return results, built_contexts  # type: ignore[return-value]

        # Multi-face: try ProcessPool (FaceProcessorPool) for true parallelism,
        # falling back to ThreadPoolExecutor (shares memory + GIL-released ops).
        proc_results: Optional[List[Optional[dict]]] = None
        try:
            def _slim_ctx(c):
                return {
                    k: v for k, v in c.__dict__.items()
                    if not isinstance(v, np.ndarray)
                }
            payloads = [
                (
                    prepared_faces[i]['canvas'],
                    all_regions[i],
                    prepared_faces[i]['shifted_bbox'],
                    prepared_faces[i]['shifted_landmarks'],
                    ieds[i],
                    _slim_ctx(self._ctx_for_face(ctx, i)),
                    prepared_faces[i]['roi_box'],
                    prepared_faces[i]['roi_person_mask'],
                    prepared_faces[i]['roi_box'][3] - prepared_faces[i]['roi_box'][1],
                    prepared_faces[i]['roi_box'][2] - prepared_faces[i]['roi_box'][0],
                )
                for i in range(len(faces))
            ]
            proc_results = self._face_pool.process_faces(payloads)
        except Exception:
            logger.exception("FaceProcessorPool failed; falling back to ThreadPoolExecutor")
            proc_results = None

        if proc_results is not None:
            for i, pr in enumerate(proc_results):
                if pr is None:
                    results[i] = self._process_one_face(
                        img, faces[i], person_mask, self._ctx_for_face(ctx, i),
                        h_img, w_img,
                        regions=all_regions[i], preprepared=prepared_faces[i]
                    )
                else:
                    results[i] = _FaceResult(
                        canvas=pr['canvas'],
                        skin_mask=pr['skin_mask'],
                        skin_hair_mask=pr['skin_hair_mask'],
                        lips_mask=pr['lips_mask'],
                        sharpen_mask=pr['sharpen_mask'],
                        roi_box=pr['roi_box'],
                    )
            return results, built_contexts  # type: ignore[return-value]

        # Fallback: ThreadPoolExecutor (engine instance shared via memory)
        with ThreadPoolExecutor(max_workers=min(len(faces), 4)) as pool:
            future_to_idx = {
                pool.submit(
                    self._process_one_face,
                    img, face, person_mask, self._ctx_for_face(ctx, i),
                    h_img, w_img,
                    regions=all_regions[i], preprepared=prepared_faces[i]
                ): i
                for i, face in enumerate(faces)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results, built_contexts  # type: ignore[return-value]

    def _process_one_face(
        self,
        img: np.ndarray,
        face,
        person_mask: np.ndarray,
        ctx: ProcessingContext,
        h_img: int,
        w_img: int,
        regions: Optional[FaceRegions] = None,
        preprepared: Optional[Dict] = None,
    ) -> _FaceResult:
        """Full per-face pipeline. Operates on a private copy of the cropped Portrait ROI canvas.

        Note: When preprepared is provided (batched ROI-first path),
        img, person_mask, h_img, and w_img parameters are bypassed
        and not used during per-face rendering. face is still used
        for face.ied to construct FaceData.
        """
        if preprepared is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = preprepared['roi_box']
            canvas = preprepared['canvas'].copy()
            roi_person_mask = preprepared['roi_person_mask']
            shifted_landmarks = preprepared['shifted_landmarks']
            shifted_bbox = preprepared['shifted_bbox']
            roi_h = roi_y2 - roi_y1
            roi_w = roi_x2 - roi_x1
        else:
            face_x, face_y, face_w, face_h = face.bbox

            # Safe expanded padding to cover hair (top) and neck/chest (bottom)
            pad_t, pad_b, pad_l, pad_r = self._compute_face_roi_padding(face_w, face_h)

            roi_y1 = max(0, face_y - pad_t)
            roi_y2 = min(h_img, face_y + face_h + pad_b)
            roi_x1 = max(0, face_x - pad_l)
            roi_x2 = min(w_img, face_x + face_w + pad_r)

            roi_h = roi_y2 - roi_y1
            roi_w = roi_x2 - roi_x1

            # Private copy of cropped canvas & segmentation masks
            canvas = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            roi_person_mask = (
                person_mask[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                if person_mask is not None
                else None
            )

            # Shift landmarks & bbox to be relative to the ROI crop
            shifted_landmarks = copy.deepcopy(face.landmarks)
            for lm in shifted_landmarks.landmark:
                lm.x = (lm.x * w_img - roi_x1) / roi_w
                lm.y = (lm.y * h_img - roi_y1) / roi_h

            shifted_bbox = (face_x - roi_x1, face_y - roi_y1, face_w, face_h)

        shifted_face = FaceData(
            bbox=shifted_bbox,
            landmarks=shifted_landmarks,
            ied=face.ied
        )

        # ---- Parse regions (only when not already provided by the batch path) ----
        if regions is None:
            regions = self._parser.parse(
                shifted_face.landmarks, canvas, shifted_face.bbox, roi_person_mask, shifted_face.ied
            )

        processors = {
            'skin': self._skin,
            'relighter': self._relighter,
            'blemish': self._blemish,
            'undereye': self._undereye,
            'eyes': self._eyes,
            'teeth': self._teeth,
            'lips': self._lips,
            'makeup': self._makeup,
            'makeup_v2': self._makeup_v2,
            'hair': self._hair,
            'frequency': self._frequency,
        }
        return _process_face_core(
            canvas, regions, shifted_face, ctx,
            roi_x1, roi_y1, roi_h, roi_w, roi_person_mask, processors,
        )

    def _composite_faces(
        self,
        base: np.ndarray,
        face_results: List[_FaceResult],
        h_img: int,
        w_img: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Merge cropped per-face canvases and masks back into a single full-resolution output image."""
        acc_skin = np.zeros((h_img, w_img), dtype=np.float32)
        acc_skin_hair = np.zeros((h_img, w_img), dtype=np.float32)
        acc_lips = np.zeros((h_img, w_img), dtype=np.float32)
        acc_sharpen = np.zeros((h_img, w_img), dtype=np.float32)

        result = base.copy()
        for fr in face_results:
            x1, y1, x2, y2 = fr.roi_box
            edit_mask = np.maximum.reduce((
                fr.skin_hair_mask,
                fr.lips_mask,
                fr.sharpen_mask,
            ))
            alpha = np.clip(edit_mask, 0.0, 1.0)[:, :, np.newaxis]
            
            # Blend cropped canvas back anywhere this face ROI was edited.
            roi_blend = np.clip(
                fr.canvas.astype(np.float32) * alpha
                + result[y1:y2, x1:x2].astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
            result[y1:y2, x1:x2] = roi_blend

            # Accumulate cropped masks into full-resolution canvas masks
            acc_skin[y1:y2, x1:x2] = np.maximum(acc_skin[y1:y2, x1:x2], fr.skin_mask)
            acc_skin_hair[y1:y2, x1:x2] = np.maximum(acc_skin_hair[y1:y2, x1:x2], fr.skin_hair_mask)
            acc_lips[y1:y2, x1:x2] = np.maximum(acc_lips[y1:y2, x1:x2], fr.lips_mask)
            acc_sharpen[y1:y2, x1:x2] = np.maximum(acc_sharpen[y1:y2, x1:x2], fr.sharpen_mask)

        return result, acc_skin, acc_skin_hair, acc_lips, acc_sharpen

    def _stage_subject_separation(
        self,
        img: np.ndarray,
        person_mask: Optional[np.ndarray],
        ctx: ProcessingContext,
    ) -> np.ndarray:
        """Brighten subject / darken background using the person mask.

        Maps subject_separation (0-100) to exposure deltas:
          subject += 0.3 EV  * strength%
          background -= 0.4 EV * strength%
        Applied in LAB L-channel for clean exposure shifts.
        Accepts uint8 or float32 [0,1] input, returns same dtype.

        F1/E2: float path is genuinely float-native (no uint8 round-trip)
        via ``bgr_f32_to_lab_f32`` / ``lab_f32_to_bgr_f32``.
        """
        strength = ctx.subject_separation
        if strength <= 0 or person_mask is None:
            return img

        is_float = img.dtype == np.float32
        if is_float:
            img_255 = img * 255.0
        else:
            img_255 = img.astype(np.float32)

        pm = person_mask.astype(np.float32)
        pm = squeeze_mask(pm)
        if pm.max() > 1.0:
            pm /= 255.0

        h_img, w_img = img.shape[:2]
        feather = max(3, int(min(h_img, w_img) * 0.02) | 1)
        pm = cv2.GaussianBlur(pm, (feather, feather), 0)

        if is_float:
            lab = bgr_f32_to_lab_f32(img_255)
        else:
            bgr_u8 = np.clip(img_255, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        s = strength / 100.0
        subject_gain = 1.0 + 0.23 * s
        background_gain = 1.0 - 0.33 * s

        mask_subject = pm
        mask_bg = np.clip(1.0 - pm, 0.0, 1.0)

        L_new = L * (mask_subject * subject_gain + mask_bg * background_gain)
        lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)

        if is_float:
            result_f255 = lab_f32_to_bgr_f32(lab)
            return np.clip(result_f255 / 255.0, 0.0, 1.0).astype(np.float32)
        else:
            result_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            return result_u8

    def _stage_harmonize(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_skin: Optional[np.ndarray],
        person_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """C5: Skin-anchored background color harmonization.

        Shifts background colors to complement the corrected skin tone
        (the C1 anchor). Runs after subject_separation and before the
        global tonal stage, so the global grade still applies on top of
        the harmonized background.

        Accepts uint8 or float32 [0,1] input, returns same dtype. The
        harmonizer itself works in [0, 255] float32 internally, so the
        [0,1] float path is scaled at the boundary.
        """
        strength = ctx.background_harmonize
        if strength <= 0 or acc_skin is None or person_mask is None:
            return img

        is_float = img.dtype == np.float32
        if is_float:
            img_255 = np.clip(img * 255.0, 0.0, 255.0).astype(np.float32)
        else:
            img_255 = img

        # background_harmonize is 0-100; harmonizer.harmonize expects [0,1].
        s = float(strength) / 100.0
        out_255 = self._harmonizer.harmonize(
            img_255,
            skin_mask=acc_skin,
            person_mask=person_mask,
            strength=s,
            mode=str(ctx.background_harmonize_mode),
        )

        if is_float:
            return np.clip(out_255 / 255.0, 0.0, 1.0).astype(np.float32)
        return out_255

    def _stage_background(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """T1: Background replace & scene relight.

        Runs the 7 ``anime_crystal_void`` background operations through
        ``BackgroundReplacer``: background blur, background desaturation,
        blue/cyan background grades, matte-black crush, light-wrap rim,
        and subject sharpen.  Each is gated by the feathered person mask
        so the subject is protected; the operations compose in a fixed
        order (grade → blur → relight-free → wrap → sharpen) that
        matches the "crystal void" intent.

        Accepts uint8 or float32 [0,1] input, returns same dtype. The
        replacer works in [0, 255] float32 internally, so the [0,1]
        float path is scaled at the boundary.
        """
        # Early-out: no-op if all 7 background params are zero or there
        # is no person mask to gate the operations.
        bg_active = (
            ctx.background_blur > 0
            or ctx.background_desaturation > 0
            or ctx.light_wrap > 0
            or ctx.blue_shadow_grade > 0
            or ctx.cyan_midtone_grade > 0
            or ctx.subject_sharpen > 0
            or ctx.matte_black > 0
        )
        if not bg_active or person_mask is None:
            return img

        is_float = img.dtype == np.float32
        if is_float:
            img_255 = np.clip(img * 255.0, 0.0, 255.0).astype(np.float32)
        else:
            img_255 = img

        out = img_255
        replacer = self._background_replacer

        # 1. Background colour grade (desat + blue shadow + cyan midtone +
        #    matte black) — applied first so the blur softens the grade.
        if (
            ctx.background_desaturation > 0
            or ctx.blue_shadow_grade > 0
            or ctx.cyan_midtone_grade > 0
            or ctx.matte_black > 0
        ):
            out = replacer.grade_background(
                out,
                person_mask,
                {
                    "desaturation": float(ctx.background_desaturation),
                    "blue_shadow_grade": float(ctx.blue_shadow_grade),
                    "cyan_midtone_grade": float(ctx.cyan_midtone_grade),
                    "matte_black": float(ctx.matte_black),
                },
            )

        # 2. Background blur (bokeh) — subject stays sharp.
        if ctx.background_blur > 0:
            out = replacer.blur_background(out, person_mask, float(ctx.background_blur))
        if ctx.lens_blur > 0:
            out = replacer.lens_blur(out, person_mask, float(ctx.lens_blur))

        # 3. Light-wrap rim — composite integration glow around the subject.
        if ctx.light_wrap > 0:
            out = replacer.light_wrap(out, person_mask, float(ctx.light_wrap))

        # 4. Subject sharpen — crisp the subject (background untouched).
        if ctx.subject_sharpen > 0:
            out = replacer.sharpen_subject(out, person_mask, float(ctx.subject_sharpen))

        if is_float:
            return np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
        return out

    def _stage_body_skin(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray],
        acc_skin: Optional[np.ndarray],
        acc_skin_hair: Optional[np.ndarray],
        acc_lips: Optional[np.ndarray],
        faces: Optional[List[FaceData]],
        h_img: int,
        w_img: int,
    ) -> np.ndarray:
        """Body skin retouch: smoothing, tone matching to face, equalization, blemish removal.

        Operates on full-image body skin (arms, legs, décolletage, etc.) detected via
        LCH-based skin color detection, intersected with person_mask and excluding face ROIs.
        Accepts float32 [0,1] input, returns same dtype.

        Args:
            img: (H, W, 3) float32 BGR image [0, 1].
            ctx: ProcessingContext with body_smooth, body_equalize, body_whiten, body_match_face.
            person_mask: (H, W) float32 person segmentation mask [0, 1].
            acc_skin: (H, W) float32 accumulated face skin mask (already retouched).
            acc_skin_hair: (H, W) float32 accumulated face skin+hair mask.
            acc_lips: (H, W) float32 accumulated lips mask (already retouched).
            faces: List of detected FaceData objects.
            h_img, w_img: Image height and width.

        Returns:
            (H, W, 3) float32 BGR image [0, 1].
        """
        # Early exit: all body params are zero
        if (ctx.body_smooth <= 0 and ctx.body_equalize <= 0 and
            ctx.body_whiten <= 0 and ctx.body_match_face <= 0 and
            ctx.body_relight <= 0 and ctx.body_dodge_burn <= 0 and
            ctx.body_shadow_lift <= 0):
            return img

        if person_mask is None or person_mask.max() < 0.01:
            return img

        # ------ Build body skin mask ------
        # Convert to LCH and apply skin color detection
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        lch = bgr_to_lch(img_u8)
        skin_mask_lch_result = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0, chroma_min=8.0)

        # Intersect with person mask (normalized)
        pm = normalize_mask(person_mask)
        pm = squeeze_mask(pm)
        body_skin_candidate = skin_mask_lch_result * pm

        # Exclude face skin region (already handled by per-face processing)
        if acc_skin is not None:
            acc_skin_norm = normalize_mask(acc_skin)
            acc_skin_norm = squeeze_mask(acc_skin_norm)
            body_skin_candidate = np.clip(body_skin_candidate - acc_skin_norm, 0, 1)

            # Exclude the whole FACE INTERIOR, not just the per-face skin mask.
            # The raw acc_skin mask has gaps between skin patches (nose bridge,
            # nasolabial folds, brow) — pixels that belong to the face but sit
            # outside the segmented skin regions. Those gaps leak into the body
            # skin mask, so body_relight / body_dodge_burn / body_equalize then
            # apply directional shading + CLAHE to e.g. the nose bridge. On a
            # face with a real (already-correct) nose-bridge shadow, that
            # darkens it ~-10 L and adds ~+4 L local contrast, making the
            # source photo's soft warm shading read as a harsh cool/gray
            # bridge shadow (user-flagged on DSCF7142 / aaa_photoreal recipes).
            # The convex hull of the face-skin mask fills those interior gaps
            # while staying within the face outline, so genuine body skin
            # (neck/chest/arms, outside the hull) is untouched.
            skin_bin = (acc_skin_norm > 0.3).astype(np.uint8)
            contours, _ = cv2.findContours(
                skin_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                all_pts = np.vstack(contours)
                if len(all_pts) >= 3:
                    hull = cv2.convexHull(all_pts)
                    face_interior = np.zeros_like(skin_bin)
                    cv2.fillConvexPoly(face_interior, hull, 1)
                    body_skin_candidate = body_skin_candidate * (
                        1.0 - face_interior.astype(np.float32)
                    )

        # Exclude face hair region if available
        if acc_skin_hair is not None:
            acc_hair = normalize_mask(acc_skin_hair)
            acc_hair = squeeze_mask(acc_hair)
            # Erode hair mask to avoid over-exclusion at edges
            acc_hair_eroded = cv2.erode(acc_hair, np.ones((3, 3), np.uint8), iterations=1)
            body_skin_candidate = np.clip(body_skin_candidate - acc_hair_eroded, 0, 1)

        # Exclude lips (already handled by per-face lip processing). High-chroma
        # red lip pixels can pass skin_mask_lch's hue/chroma heuristic and were
        # otherwise left unexcluded, letting body_smooth/body_equalize/body_relight/
        # body_dodge_burn/shadow_lift darken the mouth on close-up crops.
        if acc_lips is not None:
            acc_lips_norm = normalize_mask(acc_lips)
            acc_lips_norm = squeeze_mask(acc_lips_norm)
            acc_lips_dilated = cv2.dilate(acc_lips_norm, np.ones((5, 5), np.uint8), iterations=1)
            body_skin_candidate = np.clip(body_skin_candidate - acc_lips_dilated, 0, 1)

        # Exclude wig/hair draping past the face crop onto the body (e.g.
        # long hair over the chest/shoulders). acc_skin_hair above only
        # covers hair near the detected face bbox; skin_mask_lch's hue/
        # chroma heuristic otherwise misclassifies light-colored (pink,
        # blonde, silver) hair as skin, since it has no texture awareness.
        # A whole-image BiSeNet pass (coarser than the face-crop path, but
        # topologically correct — see parse_hair_full_image) catches this.
        body_hair_mask = self._parser.parse_hair_full_image(
            np.clip(img * 255.0, 0, 255).astype(np.uint8)
        )
        if body_hair_mask is not None:
            body_skin_candidate = body_skin_candidate * (1.0 - body_hair_mask)

        # Morphological cleaning: open (remove small noise) then close (fill small holes)
        # Scale kernel size to person size, not face size
        person_bbox_size = max(h_img, w_img) * 0.15  # Estimate person region size
        k_morph = max(3, int(person_bbox_size / 50.0) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        body_skin_candidate = cv2.morphologyEx(body_skin_candidate, cv2.MORPH_OPEN, kernel)
        body_skin_candidate = cv2.morphologyEx(body_skin_candidate, cv2.MORPH_CLOSE, kernel)

        # Contiguity check: drop disconnected components that don't touch the face region
        if acc_skin is not None and acc_skin.max() > 0.01:
            # Geodesic dilation of the face-skin region, constrained to stay
            # within person_mask at every step. A single large isotropic
            # dilation (even scaled to person_bbox_size) can fail on close-up
            # crops where the face bbox and a distant chest/décolletage skin
            # patch don't overlap on either axis within a sane radius — but
            # bridging them with a huge kernel risks bleeding sideways into
            # background/other-person skin. Instead, "walk" the dilation
            # along the body silhouette in small steps re-masked by pm each
            # iteration, so it follows the person's contour (neck -> chest)
            # rather than growing as a raw circle. Cheap: ~10 iterations of a
            # small kernel.
            step_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            face_skin_dilated = acc_skin_norm.copy()
            for _ in range(12):
                face_skin_dilated = cv2.dilate(face_skin_dilated, step_kernel, iterations=1)
                face_skin_dilated = face_skin_dilated * pm

            # Find connected components in body_skin_candidate
            n_labels, labels = cv2.connectedComponents((body_skin_candidate > 0.3).astype(np.uint8))

            # Build filtered mask: only keep components that overlap dilated face region
            body_skin_filtered = np.zeros_like(body_skin_candidate)
            for label_id in range(1, n_labels):  # 0 is background
                comp_mask = (labels == label_id).astype(np.float32)
                overlap = np.sum(comp_mask * face_skin_dilated)
                if overlap > 10:  # At least 10 pixels of overlap with face region
                    body_skin_filtered += comp_mask * body_skin_candidate
            body_skin_candidate = np.clip(body_skin_filtered, 0, 1)

        # Tattoo/body paint exclusion: exclude high-chroma zones within body skin.
        # Use the SAME gate as unify_hue_line (skin.py ~line 945): smoothstep at
        # C=0.18 in OKLCh space. NOTE: `lch` above is CIE-LAB-derived LCH from
        # bgr_to_lch(), whose chroma is unbounded (~0-130+ for real photos) —
        # NOT the same space or scale as OKLCh's ~[0,1]-range chroma. Reusing
        # the CIE-LCH chroma channel with a 0.18 threshold was a bug: it clipped
        # tattoo_gate to 0 everywhere on real images, zeroing the entire
        # body_skin_mask. Recompute chroma in OKLCh to match unify_hue_line's
        # actual units before applying the 0.18 threshold.
        oklab = bgr_to_oklab(img_u8)
        oklch = oklab_to_oklch(oklab)
        c_oklch = oklch[:, :, 1].astype(np.float32)
        tattoo_gate = np.clip((0.18 - c_oklch) / 0.02, 0.0, 1.0)  # Smooth transition at 0.18
        body_skin_candidate = body_skin_candidate * tattoo_gate

        # Feather the mask for smooth blending at edges
        k_feather = max(5, int(person_bbox_size / 40.0) | 1)
        body_skin_mask = feather_mask(body_skin_candidate, radius=k_feather, sigma=k_feather / 2.0)

        if body_skin_mask.max() < 0.01:
            return img

        # ------ Apply body skin operations ------
        result = img.copy()

        # 1. Body smoothing (guided filter at body scale)
        if ctx.body_smooth > 0:
            s = ctx.body_smooth / 100.0
            # Radius/blend caps raised to visually match face-level smoothing
            # intensity (face's frequency-separation path was noticeably
            # stronger at the same recipe strength — user-flagged gap
            # across arms/torso/legs). Not a literal port of face's
            # multi-layer frequency separation, but close enough in
            # perceived smoothness at equal strength values.
            person_diag = np.sqrt(h_img ** 2 + w_img ** 2)
            gf_radius = max(3, int(person_diag * 0.09 * s))
            gf_eps = 0.02 * s
            result_smoothed = np.zeros_like(result)
            for c in range(result.shape[2]):
                result_smoothed[:, :, c] = guided_filter(
                    result[:, :, c], gf_radius, gf_eps, guide=None, max_dim=1200
                )
            # blend_masked expects uint8 [0,255] images, not float32 [0,1]
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            result_smoothed_u8 = np.clip(result_smoothed * 255.0, 0, 255).astype(np.uint8)
            result = blend_masked(result_u8, result_smoothed_u8, body_skin_mask * s).astype(np.float32) / 255.0

        # 2. Body tone matching to face (body_match_face)
        if ctx.body_match_face > 0 and acc_skin is not None and acc_skin.max() > 0.01:
            s = ctx.body_match_face / 100.0
            # Measure median LAB of retouched face skin
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(result_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

            face_indices = acc_skin_norm > 0.3
            if np.any(face_indices):
                face_median_l = np.median(lab[:, :, 0][face_indices])
                face_median_a = np.median(lab[:, :, 1][face_indices])
                face_median_b = np.median(lab[:, :, 2][face_indices])

                body_indices = body_skin_mask > 0.3
                if np.any(body_indices):
                    body_median_l = np.median(lab[:, :, 0][body_indices])
                    body_median_a = np.median(lab[:, :, 1][body_indices])
                    body_median_b = np.median(lab[:, :, 2][body_indices])

                    # Compute bounded corrections (hard clamp at ±8 L, ±6 a/b per plan)
                    l_diff = np.clip((face_median_l - body_median_l) * s, -8.0, 8.0)
                    a_diff = np.clip((face_median_a - body_median_a) * s, -6.0, 6.0)
                    b_diff = np.clip((face_median_b - body_median_b) * s, -6.0, 6.0)

                    lab[:, :, 0] = np.clip(lab[:, :, 0] + body_skin_mask * l_diff, 0, 255)
                    lab[:, :, 1] = np.clip(lab[:, :, 1] + body_skin_mask * a_diff, 0, 255)
                    lab[:, :, 2] = np.clip(lab[:, :, 2] + body_skin_mask * b_diff, 0, 255)

                    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

        # 3. Body equalization (global median a/b pull + CLAHE on L, scaled to body)
        if ctx.body_equalize > 0:
            s = ctx.body_equalize / 100.0
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(result_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

            body_indices = body_skin_mask > 0.3
            if np.any(body_indices):
                # Pull a/b toward median. Factor raised 0.12 -> 0.20 to
                # match face's own pull strength (user-flagged: body read
                # visibly less evened-out than face at equal strength).
                median_a = np.median(lab[:, :, 1][body_indices])
                median_b = np.median(lab[:, :, 2][body_indices])
                pull = (0.20 * s * body_skin_mask)
                lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * pull
                lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * pull

                # CLAHE clip limit raised to match face-level contrast lift.
                l_chan_u8 = np.clip(lab[:, :, 0], 0, 255).astype(np.uint8)
                clip_limit = 1.0 + s * 2.5
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l_chan_u8)
                lab_new = lab.copy()
                lab_new[:, :, 0] = l_clahe.astype(np.float32)
                result_equalized_u8 = cv2.cvtColor(np.clip(lab_new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                # blend_masked expects uint8 [0,255] images, not float32 [0,1]
                result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
                result = blend_masked(result_u8, result_equalized_u8, body_skin_mask * s).astype(np.float32) / 255.0

        # 4. Body whitening (lighten L channel in body skin)
        if ctx.body_whiten > 0:
            s = ctx.body_whiten / 100.0
            # Use SkinProcessor's whiten method on body skin
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            skin_processor = SkinProcessor()
            result_whitened = skin_processor.whiten(result_u8, body_skin_mask, strength=int(ctx.body_whiten))
            result = result_whitened.astype(np.float32) / 255.0

        # 5. Body relight (landmark-free directional shading — see body_relight.py).
        # Isolated module: unlike face relight (relight.py), this has no
        # FaceMesh dependency, so it can run on any body-skin mask.
        if ctx.body_relight > 0:
            from .body_relight import BodyRelighter
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            body_relighter = BodyRelighter()
            result_relit = body_relighter.relight(result_u8, body_skin_mask, strength=ctx.body_relight)
            result = result_relit.astype(np.float32) / 255.0

        # 6. Body dodge/burn (local-contrast sculpting — see body_relight.py).
        if ctx.body_dodge_burn > 0:
            from .body_relight import BodyRelighter
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            body_relighter = BodyRelighter()
            result_sculpted = body_relighter.dodge_burn(result_u8, body_skin_mask, strength=ctx.body_dodge_burn)
            result = result_sculpted.astype(np.float32) / 255.0

        # 7. Body shadow lift (targeted local fill-light — see shadow_lift.py).
        # Opt-in only: real photographic shade (e.g. a jaw/chest area next to
        # a bright prop/light) is not a retouch defect, so this must be
        # explicitly requested via a recipe rather than applied by default.
        if ctx.body_shadow_lift > 0:
            from .shadow_lift import ShadowLifter
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            shadow_lifter = ShadowLifter()
            # Include body-wide hair (e.g. long wig hair crossing a dark
            # background) so genuine hair-lighting falloff gets the same
            # gentle local lift as body skin, not left untouched.
            lift_mask = body_skin_mask
            if body_hair_mask is not None:
                lift_mask = np.clip(body_skin_mask + body_hair_mask, 0, 1)
            result_lifted = shadow_lifter.lift(result_u8, lift_mask, strength=ctx.body_shadow_lift)
            result = result_lifted.astype(np.float32) / 255.0

        # 8. Blemish removal on body (conservative size threshold scaled to person)
        # Gated on the stage itself being active (any of the 7 body params nonzero),
        # NOT on body_smooth specifically — each param must independently unlock its
        # own sub-step, and blemish removal is a reasonable default whenever body skin
        # is being touched at all. Strength is a fixed conservative default scaled by
        # the strongest active param, rather than derived from body_smooth alone
        # (deriving it from one unrelated param was the bug: a user setting only
        # body_whiten/body_match_face/body_equalize got silently zero blemish removal).
        if (ctx.body_smooth > 0 or ctx.body_equalize > 0 or
            ctx.body_whiten > 0 or ctx.body_match_face > 0 or
            ctx.body_relight > 0 or ctx.body_dodge_burn > 0 or
            ctx.body_shadow_lift > 0):
            active_strength = max(
                ctx.body_smooth, ctx.body_equalize, ctx.body_whiten,
                ctx.body_match_face, ctx.body_relight, ctx.body_dodge_burn,
                ctx.body_shadow_lift,
            )
            blemish_strength = max(20.0, active_strength * 0.5)  # conservative floor of 20
            blemish_remover = BlemishRemover()
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            result_unblemished = blemish_remover.remove(result_u8, body_skin_mask, strength=blemish_strength)
            # blend_masked expects uint8 [0,255] images, not float32 [0,1]
            result = blend_masked(
                result_u8,
                result_unblemished,
                body_skin_mask * 0.3
            ).astype(np.float32) / 255.0

        return result

    def _stage_cosplay_moat(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        hair_mask: Optional[np.ndarray],
        person_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """A3: Cosplay skin moat — makeup-agnostic enhancements.

        Applies three independent operations:
        1. Wig-lace blend: seamless transition where wig meets skin at hairline
        2. Stockings: detect and smooth hosiery without over-blurring
        3. Shoot-consistency lock: maintain white-balance continuity across shots

        Accepts float32 [0,1] input, returns same dtype.

        Args:
            img: (H, W, 3) float32 BGR image [0, 1].
            ctx: ProcessingContext with cosplay_* parameters.
            hair_mask: (H, W) float32 hair region mask [0, 1].
            person_mask: (H, W) float32 person segmentation mask [0, 1].

        Returns:
            (H, W, 3) float32 BGR image [0, 1].
        """
        # Early exit: all cosplay params are zero
        if (ctx.cosplay_wig_lace_blend <= 0 and
            ctx.cosplay_stockings_smooth <= 0 and
            ctx.cosplay_consistency_strength <= 0):
            return img

        is_float = img.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img

        result = img_u8.astype(np.float32)

        # --- Stage 1: Wig-lace blending ---
        if ctx.cosplay_wig_lace_blend > 0 and hair_mask is not None:
            from .cosplay_moat import WigLaceBlender
            blender = WigLaceBlender()
            # Detect skin mask from the processed image (for blending with natural skin)
            lch = bgr_to_lch(img_u8)
            skin_mask_detected = skin_mask_lch(lch, hue_center=25.0, hue_tolerance=25.0, chroma_min=8.0)
            result_blended = blender.blend(
                result,
                hair_mask,
                skin_mask_detected,
                strength=ctx.cosplay_wig_lace_blend / 100.0,
            )
            result = result_blended.astype(np.float32)

        # --- Stage 2: Stockings smoothing ---
        if ctx.cosplay_stockings_smooth > 0 and person_mask is not None:
            from .cosplay_moat import HosierySmoother
            smoother = HosierySmoother()
            result_smoothed = smoother.smooth(
                result,
                person_mask,
                strength=ctx.cosplay_stockings_smooth / 100.0,
            )
            result = result_smoothed.astype(np.float32)

        # --- Stage 3: Shoot consistency lock ---
        # Note: This stage requires reference shot analysis. For now, it's wired
        # but not actively applied (needs multi-image input which is beyond
        # the single-image process() API). Future versions will integrate with
        # batch_processor.py to compare consecutive shots.
        if ctx.cosplay_consistency_strength > 0:
            # Placeholder: analyze current shot to support future multi-frame batching
            from .cosplay_moat import ShootConsistencyLock
            lock = ShootConsistencyLock()
            current_analysis = lock.analyze_shot(img_u8, face_data=None)
            # Store for later batch-level comparison (if multi-image batch available)
            # For now, log that we detected the parameters but need reference frame
            if current_analysis.get("valid"):
                logger.debug(
                    f"shoot_consistency_lock: detected skin tone "
                    f"L={current_analysis['lab_l']:.1f}, "
                    f"a={current_analysis['lab_a']:.1f}, "
                    f"b={current_analysis['lab_b']:.1f} "
                    f"(chroma={current_analysis['chroma']:.1f})"
                )

        # Return in original dtype
        if is_float:
            return np.clip(result / 255.0, 0.0, 1.0).astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _stage_global(self, img: np.ndarray, ctx: ProcessingContext) -> np.ndarray:
        """Global tonal operators (contrast, brightness, HSL tonal curve).
        Accepts uint8 or float32 [0,1] input, returns same dtype.
        """
        is_float = img.dtype == np.float32
        result = img

        if ctx.contrast:
            if is_float:
                result = _F_adjust_contrast(result, ctx.contrast)
            else:
                result = _adjust_contrast(result, ctx.contrast)

        if ctx.brightness is not None and ctx.brightness != 0:
            gamma = np.clip(1.0 - (ctx.brightness / 100.0), 0.1, 4.0)
            if is_float:
                result = np.power(np.clip(result, 0.0, 1.0), gamma).astype(np.float32)
            else:
                x = np.arange(256, dtype=np.float32) / 255.0
                _brightness_lut = (np.power(x, gamma) * 255.0).astype(np.uint8)
                result = cv2.LUT(result, _brightness_lut)

        if any(v is not None and v != 0 for v in (
            ctx.highlights, ctx.shadows, ctx.whites, ctx.blacks
        )):
            if is_float:
                result = _F_adjust_tonal(
                    result,
                    shadows=ctx.shadows or 0,
                    highlights=ctx.highlights or 0,
                    whites=ctx.whites or 0,
                    blacks=ctx.blacks or 0,
                )
            else:
                result = _adjust_tonal(
                    result,
                    shadows=ctx.shadows or 0,
                    highlights=ctx.highlights or 0,
                    whites=ctx.whites or 0,
                    blacks=ctx.blacks or 0,
                )

        if ctx.clarity:
            result = self._grader._add_clarity(result, ctx.clarity / 100.0)

        if ctx.vibrance:
            if is_float:
                result = _F_adjust_vibrance(result, ctx.vibrance)
            else:
                result = _adjust_vibrance(result, ctx.vibrance)

        if ctx.saturation:
            if is_float:
                result = _F_apply_uniform_saturation(result, ctx.saturation)
            else:
                result = _apply_uniform_saturation(result, ctx.saturation)

        return result

    def _stage_grade(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_skin: np.ndarray,
        acc_lips: np.ndarray,
        person_mask: Optional[np.ndarray],
        style_ref: Optional[np.ndarray] = None,
        faces=None,
    ) -> np.ndarray:
        """Colour transfer, grading, and white-costume lift.
        Accepts uint8 or float32 [0,1] input, returns same dtype.
        """
        is_float = img.dtype == np.float32
        result = img

        def _to_uint8_if_float(x):
            if x.dtype == np.float32:
                return np.clip(x * 255.0, 0, 255).astype(np.uint8)
            return x

        def _to_float_if_needed(x, was_float):
            if was_float and x.dtype == np.uint8:
                return x.astype(np.float32) / 255.0
            return x

        # Subject-Aware Color Transfer (Meitu/Xingtu-style portrait match)
        if style_ref is not None:
            # inherently-uint8: subject_aware_transfer returns uint8 and uses cvtColor BGR2LAB internally
            result_u8 = _to_uint8_if_float(result)
            result_u8 = subject_aware_transfer(self, result_u8, style_ref, target_faces=faces, target_person=person_mask)
            result = _to_float_if_needed(result_u8, is_float)

        # Colour transfer (reference-based)
        if ctx.color_ref is not None:
            # color_transfer runs the source through bgr_f32_to_lab_f32, which
            # expects float BGR in [0,255]; `result` is [0,1] here, so scale
            # around the call (the ref is handled by the method internally).
            if is_float:
                ct_in = np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
                ct_out = self._grader.color_transfer(
                    ct_in, ctx.color_ref, intensity=ctx.color_transfer_intensity
                )
                result = np.clip(ct_out / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                result = self._grader.color_transfer(
                    result, ctx.color_ref, intensity=ctx.color_transfer_intensity
                )

        # --- White balance (LCH-based, Phase 1.d) ---
        if ctx.white_balance_kelvin != _DEFAULTS["white_balance_kelvin"] or ctx.white_balance_tint != _DEFAULTS["white_balance_tint"]:
            # white_balance_lch runs through bgr_f32_to_lch_f32, which expects
            # float BGR in [0,255]; `result` is [0,1] here, so scale around the
            # call or the LCh L collapses and the image crushes to black.
            if is_float:
                wb_in = np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
                wb_out = self._grader.white_balance_lch(
                    wb_in, temperature=ctx.white_balance_kelvin, tint=ctx.white_balance_tint,
                )
                result = np.clip(wb_out / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                result = self._grader.white_balance_lch(
                    result,
                    temperature=ctx.white_balance_kelvin,
                    tint=ctx.white_balance_tint,
                )

        # --- Master HSL (Phase 1.d) — global LCH adjustments ---
        if ctx.hsl_hue_global != 0 or ctx.hsl_sat_global != 0 or ctx.hsl_lum_global != 0:
            # adjust_hsl_lch runs through bgr_f32_to_lch_f32, which expects float
            # BGR in [0,255]; `result` is [0,1] here, so scale around the call
            # (otherwise the LCh L collapses and the image crushes to black).
            if is_float:
                hsl_in = np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
                hsl_out = self._grader.adjust_hsl_lch(
                    hsl_in,
                    hue_shift=ctx.hsl_hue_global * 0.6,
                    sat_scale=1.0 + ctx.hsl_sat_global / 100.0,
                    lum_shift=ctx.hsl_lum_global * 0.5,
                )
                result = np.clip(hsl_out / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                result = self._grader.adjust_hsl_lch(
                    result,
                    hue_shift=ctx.hsl_hue_global * 0.6,
                    sat_scale=1.0 + ctx.hsl_sat_global / 100.0,
                    lum_shift=ctx.hsl_lum_global * 0.5,
                )

        # Build glow mask — allow glow on skin & background, preserve costume details
        pm_norm = _norm_mask(person_mask)
        if pm_norm is not None:
            pm_norm = squeeze_mask(pm_norm)
            sharp_fg = np.clip(pm_norm - acc_skin, 0.0, 1.0)
            g_mask = 1.0 - sharp_fg
        else:
            g_mask = np.ones(img.shape[:2], dtype=np.float32)

        skip_glows = ctx.bloom > 0.0
        post_effects = self._assemble_post_effects(ctx)

        # --- C3: Parametric film-density engine (between HSL and tonal curve) ---
        film_tonemap_active = False
        if ctx.film_enable:
            from .film import FilmDensityEngine
            if ctx.lut is not None and ctx.lut != "none":
                logger.warning(
                    "film.enable=true with lut=%r: LUT skipped (mutually exclusive "
                    "with C3 film engine)", ctx.lut,
                )
                post_effects.pop("lut", None)
            if ctx.film_tonemap_strength > 0:
                film_tonemap_active = True
            _film_engine = FilmDensityEngine()
            result = _film_engine.apply_from_context(result, ctx)

        if ctx.tonal_curve_strength > 0 and not film_tonemap_active:
            # inherently-uint8: tonal.apply_hd_curve uses cv2.LUT which requires uint8 input
            result_u8 = _to_uint8_if_float(result)
            result_u8 = tonal.apply_hd_curve(result_u8, strength=ctx.tonal_curve_strength)
            result = _to_float_if_needed(result_u8, is_float)

        # Run core color grading (grade() now supports float natively)
        if ctx.color_grade_stack:
            # inherently-uint8: grade_stack always returns uint8 (no return_float param)
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.grade_stack(result_u8, ctx.color_grade_stack)
            result = _to_float_if_needed(result_u8, is_float)
        elif ctx.color_grade and ctx.grade_intensity > 0:
            # grade_intensity == 0 -> skip entirely so a preset at amount 0
            # doesn't leak its grain/halation/CA/lut post-effects.
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]
            settings["gamut_compress"] = ctx.gamut_compress
            settings["saturation_mode"] = ctx.saturation_mode

            result = self._grader.grade(
                result, settings, ctx.grade_intensity,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
                skin_mask=acc_skin,
                skip_glows=skip_glows, skip_post_effects=True,
                skin_protect_strength=ctx.skin_protect_strength,
                return_float=is_float,
            )

        # Apply recipe-level split toning
        if any((ctx.shadow_hue, ctx.shadow_sat, ctx.midtone_hue, ctx.midtone_sat, ctx.highlight_hue, ctx.highlight_sat)):
            tones = {
                "shadows": {"hue": ctx.shadow_hue, "sat": ctx.shadow_sat},
                "midtones": {"hue": ctx.midtone_hue, "sat": ctx.midtone_sat},
                "highlights": {"hue": ctx.highlight_hue, "sat": ctx.highlight_sat},
            }
            if is_float:
                result = self._grader._F_split_tone_three_way(result, tones, mask=acc_skin)
            else:
                result = self._grader._split_tone_three_way(result, tones, mask=acc_skin)

        if ctx.highlight_rolloff_strength > 0:
            # F1/E2: apply_highlight_rolloff now dtype-aware (float32 [0,255] path)
            result = highlight.apply_highlight_rolloff(result, ctx.highlight_rolloff_strength)

        # ---- Skin light-wrap diffusion (anime) ----
        if ctx.skin_glow > 0 and acc_skin is not None and acc_skin.max() > 0.01:
            # F1/E2: apply_skin_diffusion now dtype-aware (float32 [0,255] path via bgr_f32_to_lab_f32)
            result = apply_skin_diffusion(result, acc_skin, strength=ctx.skin_glow)

        # Apply Global Cinematic Bloom (float-native: handles float32 [0,1] I/O)
        if ctx.bloom > 0.0:
            result = apply_global_bloom(
                result,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )

        # Apply ctx-level atmospheric glow
        if ctx.glow > 0:
            # F1/E2: _add_glow now dtype-aware (float32 [0,255] path via bgr_f32_to_lab_f32)
            result = self._grader._add_glow(result, ctx.glow / 100.0, mask=g_mask)

        # ---- Stage C4: 透明感 / 空気感 Finish Pack ----

        # Finish effects run through bgr_f32_to_lab_f32 / bgr_f32_to_lch_f32,
        # which expect float BGR in [0,255]. At this point in the float path
        # `result` is [0,1], so scale up for the whole finish group and back —
        # otherwise the LAB/LCh L collapses toward 0 and highlight_drift /
        # clarity_split crush the image to black.
        _finish_needs_scale = is_float and (
            ctx.fade_toe > 0 or ctx.highlight_drift > 0
            or ctx.airy_haze > 0 or ctx.clarity_split_neg > 0
            or ctx.clarity_split_pos > 0
        )
        if _finish_needs_scale:
            result = np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)

        # Fade toe: lifted-black with hue-locked toe (L-only in LAB)
        if ctx.fade_toe > 0:
            result = self._grader.fade_toe(result, ctx.fade_toe / 100.0, mask=acc_skin)

        # Highlight drift: hue rotation toward cyan in highlights, skin-protected
        if ctx.highlight_drift > 0:
            result = self._grader.highlight_drift(result, ctx.highlight_drift / 100.0, mask=acc_skin)

        # Airy haze: L-threshold-scoped glow with person_mask-aware distance falloff
        if ctx.airy_haze > 0:
            result = self._grader.airy_haze(result, ctx.airy_haze / 100.0, person_mask=person_mask)

        # Clarity split: negative form-band clarity + positive micro-contrast
        if ctx.clarity_split_neg > 0 or ctx.clarity_split_pos > 0:
            result = self._grader.clarity_split(
                result,
                ctx.clarity_split_neg / 100.0,
                ctx.clarity_split_pos / 100.0,
                mask=None
            )

        if _finish_needs_scale:
            result = np.clip(result / 255.0, 0.0, 1.0).astype(np.float32)

        # Apply post-effects
        if post_effects:
            result = self._grader.grade(
                result, post_effects, 1.0,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
                skin_mask=acc_skin,
                skin_protect_strength=ctx.skin_protect_strength,
                return_float=is_float,
            )

        # White costume pearl/lavender lift
        if ctx.white_costume_lift:
            # F1/E2: _apply_white_costume_lift now dtype-aware (float32 [0,255] path via bgr_f32_to_lab_f32)
            result = self._apply_white_costume_lift(result, acc_skin, acc_lips, ctx.grade_intensity)

        # Apply ctx-level vignette (float-native: _add_vignette handles float32 [0,1] I/O)
        if ctx.vignette > 0:
            result = self._grader._add_vignette(result, ctx.vignette / 100.0)

        if ctx.grain_strength > 0:
            # apply_film_grain's float32 branch expects [0,255] (it runs
            # bgr_f32_to_lab_f32, which assumes that scale). At this point in
            # the float path `result` is [0,1], so scale up around the call or
            # grain math swamps the signal into full-frame static.
            if is_float:
                grained = grain.apply_film_grain(result * 255.0, ctx.grain_strength)
                result = np.clip(grained / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                result = grain.apply_film_grain(result, ctx.grain_strength)

        # --- Negative split tone ---
        if ctx.negative_split_tone_shadow > 0 or ctx.negative_split_tone_highlight > 0:
            # negative_split_tone runs through bgr_f32_to_lch_f32, which expects
            # float BGR in [0,255]; `result` is [0,1] here, so scale around the
            # call (otherwise the LCh L collapses and it crushes to black).
            if is_float:
                nst_in = np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
                nst_out = self._grader.negative_split_tone(
                    nst_in,
                    shadow_desat=ctx.negative_split_tone_shadow / 100.0,
                    highlight_desat=ctx.negative_split_tone_highlight / 100.0,
                )
                result = np.clip(nst_out / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                result = self._grader.negative_split_tone(
                    result,
                    shadow_desat=ctx.negative_split_tone_shadow / 100.0,
                    highlight_desat=ctx.negative_split_tone_highlight / 100.0,
                )

        # --- B&W channel mixer ---
        bw_active = (
            ctx.bw_channel_mixer_r != _DEFAULTS["bw_channel_mixer_r"]
            or ctx.bw_channel_mixer_g != _DEFAULTS["bw_channel_mixer_g"]
            or ctx.bw_channel_mixer_b != _DEFAULTS["bw_channel_mixer_b"]
        )
        if bw_active:
            # F1/E2: channel_mixer_bw now dtype-aware (returns same dtype as input)
            result = self._grader.channel_mixer_bw(
                result,
                r_weight=ctx.bw_channel_mixer_r / 100.0,
                g_weight=ctx.bw_channel_mixer_g / 100.0,
                b_weight=ctx.bw_channel_mixer_b / 100.0,
            )

        return result

    def _stage_local_adjustments(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        local_adjustments: Optional[List[Dict[str, Any]]] = None,
        semantic_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """F3: Apply brush/radial/linear local adjustments.

        Runs after _stage_grade, before _stage_finish, so local edits
        sit on top of the global look (like Lightroom).

        Each adjustment is a dict: {mask, op, strength, semantic?}
        - mask: float32 [0,1] or uint8 [0,255] mask
        - op: one of LOCAL_ADJUSTMENT_OPS keys (exposure, warmth, etc.)
        - strength: float (typically -100 to 100)
        - semantic: optional key into semantic_masks (e.g. "skin", "hair")
        """
        if not local_adjustments:
            return img

        from .regions import apply_local_adjustment

        result = img
        for adj in local_adjustments:
            mask = adj.get("mask")
            op = adj.get("op", "exposure")
            strength = adj.get("strength", 0.0)
            semantic = adj.get("semantic")

            if mask is None or strength == 0:
                continue

            sem_mask = None
            if semantic and semantic_masks and semantic in semantic_masks:
                sem_mask = semantic_masks[semantic]

            result = apply_local_adjustment(result, mask, op, strength, semantic_mask=sem_mask)

        return result

    def _stage_finish(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_sharpen: np.ndarray,
        faces=None,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Finish stage: sharpening + impact. Accepts uint8 or float32 [0,1]."""
        is_float = img.dtype == np.float32
        result = img
        sharpen_mask = acc_sharpen
        if ctx.sharpen > 0 and sharpen_mask.max() <= 0.01:
            sharpen_mask = np.ones_like(sharpen_mask)
        if ctx.sharpen > 0:
            radius = ctx.sharpen_radius
            if faces:
                avg_ied = np.mean([f.ied for f in faces])
                radius = ctx.sharpen_radius * (avg_ied / 80.0)
                radius = max(0.5, min(4.0, radius))
            # Monotone sharpen map: 0 -> 0, 50 -> 1.0, 100 -> 3.0 (linear in between)
            if ctx.sharpen <= 50:
                amount = ctx.sharpen / 50.0 * 1.0
            else:
                amount = 1.0 + (ctx.sharpen - 50.0) / 50.0 * 2.0
            if is_float:
                result = _F_apply_selective_sharpening(
                    result, sharpen_mask, radius=radius, amount=amount, threshold=2.0
                )
            else:
                result = _apply_selective_sharpening(
                    result, sharpen_mask, radius=radius, amount=amount, threshold=2
                )
        if ctx.impact > 0:
            if is_float:
                result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
                result_u8 = self._grader.add_impact_finish(result_u8, ctx.impact, subject_mask=person_mask)
                result = result_u8.astype(np.float32) / 255.0
            else:
                result = self._grader.add_impact_finish(result, ctx.impact, subject_mask=person_mask)
        return result

    def _stage_body_reshape(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """T3: Body reshape stage via MediaPipe Pose (runs after finish, at native resolution).

        Applies proportional body editing (arm/leg length, torso/shoulder/hip width)
        using pose-aware warping. Accepts uint8 or float32 [0,1].

        Args:
            img: (H, W, 3) input image.
            ctx: ProcessingContext with body_reshape_* params.
            person_mask: optional (H, W) float32 [0,1] mask for boundary control.

        Returns:
            (H, W, 3) body-reshaped image, same dtype as input.
        """
        from .body_reshape import BodyReshaper, suggest_body_reshape

        # Check if any body_reshape params are active (non-default)
        arm_len = (ctx.body_reshape_arm_length - 50.0)  # Center at 50.0
        leg_len = (ctx.body_reshape_leg_length - 50.0)
        torso_w = (ctx.body_reshape_torso_width - 50.0)
        shoulder_w = (ctx.body_reshape_shoulder_width - 50.0)
        hip_w = (ctx.body_reshape_hip_width - 50.0)

        # Early-return: manual-only path when auto is off and all sliders neutral.
        if ctx.auto_body_reshape <= 0 and not any(
            [arm_len, leg_len, torso_w, shoulder_w, hip_w]
        ):
            return img

        # Ensure uint8 for reshaper (it will preserve dtype on output)
        is_float = img.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img

        reshaper = BodyReshaper()

        # One-click auto: detect pose, suggest balanced proportions, and blend
        # the suggestion (scaled by strength) with any manual slider offsets.
        if ctx.auto_body_reshape > 0:
            pose = reshaper.detector.detect(img_u8)
            if pose.landmarks is None:
                # No pose detected -> fall back to manual-only behavior.
                if not any([arm_len, leg_len, torso_w, shoulder_w, hip_w]):
                    return img
            else:
                scale = ctx.auto_body_reshape / 100.0
                sugg = suggest_body_reshape(pose)
                arm_len += (sugg["arm_length"] - 50.0) * scale
                leg_len += (sugg["leg_length"] - 50.0) * scale
                torso_w += (sugg["torso_width"] - 50.0) * scale
                shoulder_w += (sugg["shoulder_width"] - 50.0) * scale
                hip_w += (sugg["hip_width"] - 50.0) * scale
                reshaper.pose_ctx = pose

        result_u8 = reshaper.reshape(
            img_u8,
            arm_length=arm_len,
            leg_length=leg_len,
            torso_width=torso_w,
            shoulder_width=shoulder_w,
            hip_width=hip_w,
            person_mask=person_mask,
        )

        if is_float:
            return result_u8.astype(np.float32) / 255.0
        else:
            return result_u8

    @staticmethod
    def _apply_white_costume_lift(
        img: np.ndarray,
        acc_skin_hair: np.ndarray,
        acc_lips: np.ndarray,
        grade_intensity: float,
    ) -> np.ndarray:
        is_float = img.dtype == np.float32
        if is_float:
            lab = bgr_f32_to_lab_f32(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_val, a_val, b_val = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        non_face = 1.0 - np.clip(acc_skin_hair + acc_lips, 0.0, 1.0)
        white_cond = (l_val > 170) & (np.abs(a_val - 128.0) < 15) & (np.abs(b_val - 128.0) < 15)
        white_mask = white_cond.astype(np.float32) * non_face

        if white_mask.max() <= 0.01:
            return img

        wm_blur = cv2.GaussianBlur(white_mask, (15, 15), 0)
        s = grade_intensity
        lab[:, :, 0] = np.clip(l_val + 10.0 * s * wm_blur, 0, 255)
        lab[:, :, 1] = np.clip(a_val + 1.0 * s * wm_blur, 0, 255)
        lab[:, :, 2] = np.clip(b_val + 3.0 * s * wm_blur, 0, 255)
        if is_float:
            return lab_f32_to_bgr_f32(lab)
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # =========================================================================
    # A4: Neural boosters (PARKED — await A1 evidence)
    # =========================================================================

    def _stage_neural_boosters(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply neural booster stages (stray hair, defect segmentation).

        PARKED: Currently disabled (neural_stray_hair_boost=0, neural_defect_boost=0 by default).
        Runs only if explicitly enabled via ProcessingContext. Operates after QA
        detectors (final cosmetic pass before export).

        Owner decision 2026-07-03: stay classical-only for now. Generative capabilities
        (AI pore-detail synthesis, identity-preserving diffusion refinement) are
        explicitly parked here, not scope-crept into S6/C5. Revisit once A1 evidence
        exists. See MASTER_PLAN.md line 112 and retouch/neural_boosters.py.

        Parameters
        ----------
        img : np.ndarray
            Current result image, float32 [0,255] BGR or uint8 [0,255] BGR.
        ctx : ProcessingContext
            Processing context with neural_stray_hair_boost/neural_defect_boost.
        person_mask : np.ndarray, optional
            Binary mask of detected person (for region gating).

        Returns
        -------
        np.ndarray
            Image after neural boosters, same dtype/shape as input.
        """
        # Gate: only run if at least one booster is active
        stray_hair_strength = float(ctx.neural_stray_hair_boost or 0.0)
        defect_strength = float(ctx.neural_defect_boost or 0.0)

        if stray_hair_strength <= 0 and defect_strength <= 0:
            return img

        # Currently disabled: placeholders return empty masks
        from .neural_boosters import StrayHairSegmenter, DefectSegmenter

        # Normalize to uint8 for processing (or use float if already float32)
        is_float = img.dtype == np.float32
        if is_float:
            img_uint8 = to_uint8(img)
        else:
            img_uint8 = img

        # Apply stray hair removal if enabled
        if stray_hair_strength > 0:
            segmenter = StrayHairSegmenter()
            if segmenter.enabled:
                mask = segmenter.detect(img_uint8)
                # When real implementation added: apply masked healing/removal
                # For now: placeholder returns empty mask, no-op
                pass
            else:
                self._warn_neural_booster_disabled("stray_hair_boost")

        # Apply defect boosting if enabled
        if defect_strength > 0:
            segmenter = DefectSegmenter()
            if segmenter.enabled:
                mask = segmenter.detect(img_uint8)
                # When real implementation added: use mask to target additional
                # blemish/pore refinement, texture transplant targeting, etc.
                # For now: placeholder returns empty mask, no-op
                pass
            else:
                self._warn_neural_booster_disabled("defect_boost")

        return img

    @staticmethod
    def _warn_neural_booster_disabled(name: str) -> None:
        # The neural booster segmenters are PARKED (neural_boosters.py, enabled=False).
        # A non-zero strength would silently do nothing; warn once per process so the
        # caller is not misled (e.g. recipes.py ships neural.*=30 by default).
        warned = getattr(RetouchEngine, "_neural_booster_warned", None)
        if warned is None:
            warned = set()
            RetouchEngine._neural_booster_warned = warned
        if name in warned:
            return
        warned.add(name)
        logger.warning(
            "neural booster '%s' requested (strength>0) but its segmenter is PARKED "
            "(enabled=False) — no effect. See retouch/neural_boosters.py. Set the "
            "strength to 0 or implement the segmenter to remove this warning.",
            name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release detector and process-pool resources."""
        try:
            self._face_pool.shutdown()
        except Exception:
            logger.warning("FaceProcessorPool shutdown raised", exc_info=True)
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level pure helpers (also used by tests)
# ---------------------------------------------------------------------------

def _adjust_vibrance(img: np.ndarray, vibrance: float) -> np.ndarray:
    """Smart saturation boost — protects skin tones, boosts unsaturated areas more."""
    return _vibrance_fn(img, None, vibrance / 100.0)


def _F_adjust_vibrance(img_f: np.ndarray, vibrance: float) -> np.ndarray:
    """Float32 [0,1] variant of _adjust_vibrance — genuinely float-native.

    F1/E2: ``vibrance()`` in utils.py now accepts float32 [0,255] input
    (dtype-aware branches), so this scales to [0,255] float32 and calls
    it directly — no uint8 quantization.
    """
    bgr_f255 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.float32)
    out_f255 = _vibrance_fn(bgr_f255, None, vibrance / 100.0)
    if out_f255.dtype != np.float32:
        out_f255 = out_f255.astype(np.float32)
    return np.clip(out_f255 / 255.0, 0.0, 1.0).astype(np.float32)


def _apply_uniform_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Uniform saturation adjustment."""
    if saturation == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + saturation / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _F_apply_uniform_saturation(img_f: np.ndarray, saturation: float) -> np.ndarray:
    """Float32 [0,1] variant of _apply_uniform_saturation — genuinely float-native.

    F1/E2: the old version round-tripped through uint8 twice (fake float). This
    version scales to [0,255] float32 for cv2 HSV (which requires that scale),
    operates in float, and scales back — no uint8 quantization.
    """
    if saturation == 0:
        return img_f
    bgr_f255 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.float32)
    hsv = cv2.cvtColor(bgr_f255, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + saturation / 100.0
    # cv2's float HSV returns S in [0, 1] (unlike uint8's [0, 255]), so the
    # saturation channel must be clamped to 1.0, not 255, or scaled S exceeds
    # gamut and HSV2BGR produces negative BGR.
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 1.0)
    out_f255 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(out_f255 / 255.0, 0.0, 1.0).astype(np.float32)


def _adjust_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 0:
        return img
    f = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
    result = f * (img.astype(np.float32) - 128.0) + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


def _F_adjust_contrast(img_f: np.ndarray, contrast: float) -> np.ndarray:
    """Float32 [0,1] variant of _adjust_contrast."""
    if contrast == 0:
        return img_f
    f = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
    img_255 = img_f * 255.0
    result = f * (img_255 - 128.0) + 128.0
    return np.clip(result / 255.0, 0.0, 1.0).astype(np.float32)


def _adjust_tonal(
    img: np.ndarray,
    shadows: float = 0,
    highlights: float = 0,
    whites: float = 0,
    blacks: float = 0,
) -> np.ndarray:
    """Parametric tonal adjustment via LUT curve. All values are -100 to +100."""
    if not any([shadows, highlights, whites, blacks]):
        return img
    x = np.arange(256, dtype=np.float32)
    y = x.copy()
    if blacks:
        w = np.clip(1.0 - x / 64.0, 0, 1)
        y = y + (blacks / 100.0) * 48 * w
    if shadows:
        w = np.clip(1.0 - np.abs(x - 25.0) / 100.0, 0, 1)
        w = w * w * (3 - 2 * w)
        y = y + (shadows / 100.0) * 48 * w
    if highlights:
        w = np.clip(1.0 - np.abs(x - 230.0) / 100.0, 0, 1)
        w = w * w * (3 - 2 * w)
        y = y + (highlights / 100.0) * 48 * w
    if whites:
        w = np.clip(1.0 - (255.0 - x) / 64.0, 0, 1)
        y = y + (whites / 100.0) * 48 * w
    y = np.clip(y, 0, 255).astype(np.uint8)
    return cv2.LUT(img, y)


def _F_adjust_tonal(
    img_f: np.ndarray,
    shadows: float = 0,
    highlights: float = 0,
    whites: float = 0,
    blacks: float = 0,
) -> np.ndarray:
    """Float32 [0,1] variant of _adjust_tonal. Applies curve directly without LUT."""
    if not any([shadows, highlights, whites, blacks]):
        return img_f
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    y = x.copy()
    if blacks:
        w = np.clip(1.0 - x / (64.0 / 255.0), 0, 1)
        y = y + (blacks / 100.0) * (48.0 / 255.0) * w
    if shadows:
        w = np.clip(1.0 - np.abs(x - 25.0 / 255.0) / (100.0 / 255.0), 0, 1)
        w = w * w * (3 - 2 * w)
        y = y + (shadows / 100.0) * (48.0 / 255.0) * w
    if highlights:
        w = np.clip(1.0 - np.abs(x - 230.0 / 255.0) / (100.0 / 255.0), 0, 1)
        w = w * w * (3 - 2 * w)
        y = y + (highlights / 100.0) * (48.0 / 255.0) * w
    if whites:
        w = np.clip(1.0 - (1.0 - x) / (64.0 / 255.0), 0, 1)
        y = y + (whites / 100.0) * (48.0 / 255.0) * w
    y_clipped = np.clip(y, 0.0, 1.0)
    return np.interp(img_f, x, y_clipped).astype(np.float32)


def _apply_selective_sharpening(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    radius: float = 0.8,
    amount: float = 0.8,
    threshold: int = 2,
) -> np.ndarray:
    """Photoshop-style unsharp masking applied through a soft mask."""
    img_f = img_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), radius)
    high_freq = img_f - blurred
    mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    if threshold > 0:
        gray_high = 0.114 * high_freq[:, :, 0] + 0.587 * high_freq[:, :, 1] + 0.299 * high_freq[:, :, 2]
        threshold_mask = (np.abs(gray_high) >= threshold)[:, :, np.newaxis]
        sharpened_diff = threshold_mask * (high_freq * (amount * mask_3d))
    else:
        sharpened_diff = high_freq * (amount * mask_3d)
    return np.clip(img_f + sharpened_diff, 0, 255).astype(np.uint8)


def _F_apply_selective_sharpening(
    img_f: np.ndarray,
    mask: np.ndarray,
    radius: float = 0.8,
    amount: float = 0.8,
    threshold: float = 2.0,
) -> np.ndarray:
    """Float32 [0,1] variant of _apply_selective_sharpening."""
    img_255 = img_f * 255.0
    blurred = cv2.GaussianBlur(img_255, (0, 0), radius)
    high_freq = img_255 - blurred
    mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    if threshold > 0:
        gray_high = 0.114 * high_freq[:, :, 0] + 0.587 * high_freq[:, :, 1] + 0.299 * high_freq[:, :, 2]
        threshold_mask = (np.abs(gray_high) >= threshold)[:, :, np.newaxis]
        sharpened_diff = threshold_mask * (high_freq * (amount * mask_3d))
    else:
        sharpened_diff = high_freq * (amount * mask_3d)
    result = np.clip(img_255 + sharpened_diff, 0, 255)
    return (result / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible v1 API)
# ---------------------------------------------------------------------------

def retouch(
    img_bgr: np.ndarray,
    smooth: float = 50,
    whiten: float = 30,
    eye_enhance: float = 30,
    eye_sclera_vessel_remove: float = 0.0,
    backdrop_cleanup: float = 0.0,
    fabric_wrinkle_smooth: float = 0.0,
    wrinkle_soften_forehead: float = 0.0,
    wrinkle_soften_nasolabial: float = 0.0,
    wrinkle_soften_neck: float = 0.0,
    contrast: float = 0,
    preset: Optional[str] = None,
    **kwargs,
) -> ProcessingResult:
    """One-shot retouching function (backward-compatible with v1 API).

    Returns a ProcessingResult which behaves like an ndarray for legacy code.
    See RetouchEngine.process() for the full parameter list.
    """
    with RetouchEngine() as engine:
        return engine.process(
            img_bgr,
            smooth=smooth,
            whiten=whiten,
            eye_enhance=eye_enhance,
            contrast=contrast,
            preset=preset,
            fabric_wrinkle_smooth=fabric_wrinkle_smooth,
            wrinkle_soften_forehead=wrinkle_soften_forehead,
            wrinkle_soften_nasolabial=wrinkle_soften_nasolabial,
            wrinkle_soften_neck=wrinkle_soften_neck,
            **kwargs,
        )

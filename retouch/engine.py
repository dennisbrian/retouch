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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)

from .detection import FaceDetector, FaceData, FaceContext
from .parsing import FaceParser, FaceRegions
from .geometry import FaceReshaper
from .makeup import MakeupEngine
from .frequency import FrequencySeparator
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
from . import grain, highlight, tonal, qa_detectors
from .qa_detectors import QAWarning
from .hair import HairEnhancer
from .relight import Relighter
from .recipes import RECIPES
from .recipe_loader import load_user_recipes
from .style import StyleProfile
from .style_transfer import subject_aware_transfer
from .utils import correct_exposure, apply_global_bloom, apply_skin_diffusion, vibrance as _vibrance_fn, squeeze_mask, feather_mask, guided_filter, normalize_mask, blend_masked
from .color_space import bgr_to_lch, skin_mask_lch
from .color_science import bgr_to_oklab, oklab_to_oklch
from .params import resolve_recipe, _deep_merge, PROCESSING_PARAMS  # noqa: F401  (re-export for backward compat)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROXY_MAX_DIM = 2048

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
    texture_opacity: float = _DEFAULTS["texture_opacity"]
    pore_synthesis: float = 0.0
    specular_bloom: float = 0.0
    specular_bloom_tone: str = _DEFAULTS["specular_bloom_tone"]
    dodge_burn: float = 0.0
    relight: float = 0.0
    relight_azimuth: float = _DEFAULTS["relight_azimuth"]
    relight_elevation: float = _DEFAULTS["relight_elevation"]
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
    shine_removal: float = 0.0
    wrinkle_soften: float = 0.0
    texture_transplant: float = 0.0
    body_smooth: float = 0.0
    body_equalize: float = 0.0
    body_whiten: float = 0.0
    body_match_face: float = 0.0

    # --- Eyes ---
    eye_enhance: float = 0.0
    dark_circles: float = 0.0
    catchlight: float = 0.0

    # --- Lips ---
    lip_enhance: float = 0.0
    lip_tint: Optional[Any] = _DEFAULTS["lip_tint"]
    lip_finish: str = _DEFAULTS["lip_finish"]

    # --- Teeth ---
    teeth_whiten: float = 0.0

    # --- Makeup ---
    blush: float = 0.0
    slimming: float = 0.0

    # --- Hair ---
    hair_enhance: float = 0.0

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

    # Cached per-face detection + parsing; None ⇒ engine detects/parses.
    face_contexts: Optional[List["FaceContext"]] = None

    # F4: Manual heal marks — list of {"mask_png_b64": str, "method": str}
    heals: Optional[List[Dict[str, Any]]] = None


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
    from .params import _resolve_dodge_burn

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
            return 0.0 if v is None else float(v) * 100.0
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
    # caller (color_ref, auto_exposure, …) is forwarded verbatim.
    # The ``_CALLER_ONLY`` set lists spec names whose value is intentionally
    # NOT taken from the recipe — the engine reads them from the override
    # dict directly (with the recipe default as the implicit default).
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
    }
    spec_kwargs = {
        spec.name: resolved[spec.name]
        for spec in PROCESSING_PARAMS
        if spec.name not in _CALLER_ONLY
    }
    # Extract skin_locus from recipe (if present) or overrides (if caller-supplied).
    # Caller overrides win over recipe defaults.
    recipe_skin_locus = rec.get("skin", {}).get("locus")
    final_skin_locus = overrides.get("skin_locus") if overrides.get("skin_locus") is not None else recipe_skin_locus

    return ProcessingContext(
        # Recipe-derived fields (data-driven) — the bulk of the context.
        **spec_kwargs,
        # Caller-only fields (no recipe source)
        auto_exposure=overrides.get("auto_exposure", False),
        color_grade_stack=overrides.get("color_grade_stack"),
        # BUGFIX-2: color_ref comes only from the caller override, never None-initialised twice
        color_ref=overrides.get("color_ref"),
        color_transfer_intensity=overrides.get("color_transfer_intensity", 1.0),
        chromatic_aberration=overrides.get("chromatic_aberration"),
        halation=overrides.get("halation"),
        grain=overrides.get("grain"),
        lut=overrides.get("lut"),
        skin_locus=final_skin_locus,
        # ``nose_smooth`` is recipe-static (always 0 in the spec default) and
        # the only consumer of overrides is the engine call, so it lives
        # outside the spec loop and is set straight from the override.
        nose_smooth=overrides.get("nose_smooth"),
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

        # Persistent process pool for multi-face parallel processing.
        # Lazily started on first multi-face call; shut down in close().
        self._face_pool = FaceProcessorPool()

        # Load user-imported recipes from ~/.cache/retouch/user_recipes/
        loaded = load_user_recipes()
        if loaded:
            import logging
            logging.getLogger(__name__).info(f"Loaded {len(loaded)} user recipes: {loaded}")

        # Warm up JIT kernels on engine startup (safe fallback if Numba is missing)
        
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
        dark_circles: Optional[float] = None,
        catchlight: Optional[float] = None,
        blemish: Optional[float] = None,
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
        dodge_burn: Optional[float] = None,
        relight: Optional[float] = None,
        relight_azimuth: Optional[float] = None,
        relight_elevation: Optional[float] = None,
        sculpt: Optional[float] = None,
        fade_toe: Optional[float] = None,
        highlight_drift: Optional[float] = None,
        airy_haze: Optional[float] = None,
        clarity_split_neg: Optional[float] = None,
        clarity_split_pos: Optional[float] = None,
        slimming: Optional[float] = None,
        blush: Optional[float] = None,
        lip_finish: Optional[str] = None,
        specular_bloom: Optional[float] = None,
        specular_bloom_tone: Optional[str] = None,
        whiten_tone: Optional[str] = None,
        nose_blush: Optional[bool] = None,
        under_eye_blush: Optional[bool] = None,
        white_costume_lift: Optional[bool] = None,
        auto_exposure: bool = False,
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
        texture_transplant: Optional[float] = None,
        body_smooth: Optional[float] = None,
        body_equalize: Optional[float] = None,
        body_whiten: Optional[float] = None,
        body_match_face: Optional[float] = None,
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
        color_grade_stack=None,
        color_ref: Optional[np.ndarray] = None,
        color_transfer_intensity: float = 1.0,
        fast: bool = False,
        style_profile: Optional[StyleProfile] = None,
        style_ref: Optional[np.ndarray] = None,
        debug_dir: Optional[str] = None,
        face_contexts: Optional[List["FaceContext"]] = None,
        heals: Optional[List[Dict[str, Any]]] = None,
    ) -> ProcessingResult:
        """Process a single image through the full Retouch pipeline.

        Returns a ProcessingResult. Access ``.image`` for the BGR ndarray, or
        use the result directly as an ndarray (legacy-compatible).
        """
        timings: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # PERF-1: downscale *before* detection when fast=True
        # ------------------------------------------------------------------
        orig_h, orig_w = img_bgr.shape[:2]
        scale = 1.0
        if fast:
            max_preview = 800
            scale = min(max_preview / orig_w, max_preview / orig_h, 1.0)
            if scale < 1.0:
                img_bgr = cv2.resize(
                    img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
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
            "dark_circles": dark_circles,
            "catchlight": catchlight,
            "blemish": blemish,
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
            "texture_opacity": texture_opacity,
            "pore_synthesis": pore_synthesis,
            "mid_reduction": mid_reduction,
            "nose_smooth": nose_smooth,
            "micro_restore": micro_restore,
            "micro_dodge_burn": micro_dodge_burn,
            "hair_enhance": hair_enhance,
            "dodge_burn": dodge_burn,
            "relight": relight,
            "relight_azimuth": relight_azimuth,
            "relight_elevation": relight_elevation,
            "sculpt": sculpt,
            "fade_toe": fade_toe,
            "highlight_drift": highlight_drift,
            "airy_haze": airy_haze,
            "clarity_split_neg": clarity_split_neg,
            "clarity_split_pos": clarity_split_pos,
            "slimming": slimming,
            "blush": blush,
            "lip_finish": lip_finish,
            "specular_bloom": specular_bloom,
            "specular_bloom_tone": specular_bloom_tone,
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
            "texture_transplant": texture_transplant,
            "body_smooth": body_smooth,
            "body_equalize": body_equalize,
            "body_whiten": body_whiten,
            "body_match_face": body_match_face,
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
        }

        ctx = build_context(active_recipe, rec, overrides)
        if face_contexts is not None:
            ctx.face_contexts = face_contexts
        if heals is not None:
            ctx.heals = heals

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
        )

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def _process_with_proxy(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
    ) -> _CoreResult:
        """Wrap the core pipeline with automatic proxy down/upscaling.

        If the longest image side exceeds ``PROXY_MAX_DIM``, the image is
        downscaled (INTER_AREA) before the expensive detection / parsing /
        smoothing / grading stages, then the result and all accumulated
        masks are upscaled (INTER_LINEAR) back to the pre-proxy resolution.
        """
        h, w = img_bgr.shape[:2]
        proxy_scale = 1.0
        native_img_bgr = img_bgr
        if max(h, w) > PROXY_MAX_DIM:
            proxy_scale = PROXY_MAX_DIM / float(max(h, w))
            new_w = int(w * proxy_scale)
            new_h = int(h * proxy_scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        core = self._run_core_pipeline(img_bgr, ctx, style_ref, timings)

        if proxy_scale < 1.0:
            core = self._upscale_core_result(core, h, w)
            # F8.0: reinject native detail in non-skin regions
            smooth_strength = ctx.smooth / 100.0 if ctx.smooth > 0 else 0.3
            sigma = 2.0 * (1.0 / proxy_scale)
            original_float = native_img_bgr.astype(np.float32)
            blurred = cv2.GaussianBlur(original_float, (0, 0), sigma)
            high_band = original_float - blurred
            skin_weight = core.acc_skin * smooth_strength if core.acc_skin is not None else 0.0
            weight = np.clip(1.0 - skin_weight, 0.0, 1.0)[:, :, np.newaxis]
            result_f = core.result.astype(np.float32) + high_band * weight * 0.85
            core.result = np.clip(result_f, 0, 255).astype(np.uint8)

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

    def _run_core_pipeline(
        self,
        img_bgr: np.ndarray,
        ctx: ProcessingContext,
        style_ref: Optional[np.ndarray],
        timings: Dict[str, float],
    ) -> _CoreResult:
        """Run stages 0–6 (detection → finish) at the given resolution.

        Honours ``ctx.face_contexts``: when provided, detection and parsing
        are skipped and the cached face data / regions are reused. Otherwise
        detection + parsing run normally and ``FaceContext`` objects are
        built and returned for caller caching.
        
        Phase 3 stages (global, grade, finish) run in float32 [0,1] to avoid
        inter-stage quantization banding.
        """
        from .precision import to_float, to_uint8
        
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
        # No-face fallback: minimal global processing
        # ------------------------------------------------------------------
        if not faces:
            result = self._no_face_fallback(img_bgr, ctx, person_mask)
            h_img, w_img = result.shape[:2]
            return _CoreResult(
                result=result,
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

        # ------------------------------------------------------------------
        # F1: Convert to float32 [0,1] for Phase 3 global stages
        # This eliminates inter-stage uint8 quantization banding
        # ------------------------------------------------------------------
        result = to_float(result)

        # ------------------------------------------------------------------
        # Stage 3 — Subject-background separation (now in float)
        # ------------------------------------------------------------------
        t_subj = time.perf_counter()
        if ctx.subject_separation > 0:
            result = self._stage_subject_separation(result, person_mask, ctx)
        timings["subject_separation"] = (time.perf_counter() - t_subj) * 1000

        # ------------------------------------------------------------------
        # Stage 3.5 — Body skin retouch (now in float)
        # ------------------------------------------------------------------
        t_body = time.perf_counter()
        result = self._stage_body_skin(result, ctx, person_mask, acc_skin, acc_skin_hair, faces, h_img, w_img)
        timings["body_skin"] = (time.perf_counter() - t_body) * 1000

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
        # Stage 6 — Selective sharpening + impact finish (now in float)
        # ------------------------------------------------------------------
        t5 = time.perf_counter()
        result = self._stage_finish(result, ctx, acc_sharpen, faces=faces, person_mask=person_mask)
        timings["finish"] = (time.perf_counter() - t5) * 1000

        # ------------------------------------------------------------------
        # F1: Convert back to uint8 for output
        # ------------------------------------------------------------------
        result = to_uint8(result)

        qa_warnings: List[QAWarning] = []
        if person_mask is not None and np.any(person_mask > 0.3):
            qa_raw = qa_detectors.run_all(result, skin_mask=person_mask, person_mask=person_mask)
            for detector_name, det_result in qa_raw.items():
                if det_result.get("flagged", False):
                    msg = {
                        "banding": "Banding visible in smooth gradient regions",
                        "clipping": "Highlight/shadow clipping detected",
                        "plastic_skin": "Skin texture loss detected — may appear plastic",
                        "halo": "Edge overshoot halos detected from sharpening",
                        "seam": "Seam visible at subject boundary",
                    }.get(detector_name, f"{detector_name} artifact detected")
                    _qa_thresholds = {
                        "banding": qa_detectors.BANDING_THRESHOLD,
                        "clipping": qa_detectors.CLIPPING_THRESHOLD,
                        "plastic_skin": qa_detectors.PLASTIC_SKIN_THRESHOLD,
                        "halo": qa_detectors.HALO_THRESHOLD,
                        "seam": qa_detectors.SEAM_THRESHOLD,
                    }
                    qa_warnings.append(QAWarning(
                        detector=detector_name,
                        score=det_result.get("score", 0.0),
                        flagged=True,
                        threshold=_qa_thresholds.get(detector_name, 0.0),
                        message=msg,
                        details={k: v for k, v in det_result.items() if k not in ("score", "flagged")},
                    ))
        ctx._qa_results = {d: r for d, r in qa_raw.items()} if 'qa_raw' in locals() else {}

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
            qa=qa_warnings,
        )

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
        result = img.copy()
        if ctx.subject_separation > 0:
            result = self._stage_subject_separation(result, person_mask, ctx)
        result = self._stage_global(result, ctx)

        if ctx.tonal_curve_strength > 0:
            result = tonal.apply_hd_curve(result, strength=ctx.tonal_curve_strength)

        # --- White balance (LCH-based, Phase 1.d) ---
        if ctx.white_balance_kelvin != _DEFAULTS["white_balance_kelvin"] or ctx.white_balance_tint != _DEFAULTS["white_balance_tint"]:
            result = self._grader.white_balance_lch(
                result,
                temperature=ctx.white_balance_kelvin,
                tint=ctx.white_balance_tint,
            )

        # --- Master HSL (Phase 1.d) — global LCH adjustments ---
        if ctx.hsl_hue_global != 0 or ctx.hsl_sat_global != 0 or ctx.hsl_lum_global != 0:
            result = self._grader.adjust_hsl_lch(
                result,
                hue_shift=ctx.hsl_hue_global * 0.6,
                sat_scale=1.0 + ctx.hsl_sat_global / 100.0,
                lum_shift=ctx.hsl_lum_global * 0.5,
            )

        post_effects = self._assemble_post_effects(ctx)
        skip_glows = ctx.bloom > 0.0

        if ctx.color_grade:
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]
            result = self._grader.grade(
                result, settings, ctx.grade_intensity,
                skip_glows=skip_glows, skip_post_effects=True,
                skin_protect_strength=ctx.skin_protect_strength,
            )

        if ctx.highlight_rolloff_strength > 0:
            result = highlight.apply_highlight_rolloff(result, ctx.highlight_rolloff_strength)

        if ctx.bloom > 0.0:
            result = apply_global_bloom(
                result,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )

        if ctx.glow > 0:
            result = self._grader._add_glow(result, ctx.glow / 100.0)

        if post_effects:
            result = self._grader.grade(
                result, post_effects, 1.0,
                skin_protect_strength=ctx.skin_protect_strength,
            )

        if ctx.vignette > 0:
            result = self._grader._add_vignette(result, ctx.vignette / 100.0)

        if ctx.impact > 0:
            result = self._grader.add_impact_finish(result, ctx.impact, subject_mask=person_mask)

        if ctx.grain_strength > 0:
            result = grain.apply_film_grain(result, ctx.grain_strength)

        # --- Negative split tone (Phase 1.d) — desaturate shadows/highlights ---
        if ctx.negative_split_tone_shadow > 0 or ctx.negative_split_tone_highlight > 0:
            result = self._grader.negative_split_tone(
                result,
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
            result = self._grader.channel_mixer_bw(
                result,
                r_weight=ctx.bw_channel_mixer_r / 100.0,
                g_weight=ctx.bw_channel_mixer_g / 100.0,
                b_weight=ctx.bw_channel_mixer_b / 100.0,
            )

        return result

    def _stage_reshape(self, img: np.ndarray, faces, ctx: ProcessingContext) -> np.ndarray:
        if ctx.slimming > 0:
            return self._reshaper.reshape(img, faces, ctx.slimming)
        return img.copy()

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
                img, faces[0], person_mask, ctx, h_img, w_img,
                regions=all_regions[0], preprepared=prepared_faces[0]
            )
            return results, built_contexts  # type: ignore[return-value]

        # Multi-face: try ProcessPool (FaceProcessorPool) for true parallelism,
        # falling back to ThreadPoolExecutor (shares memory + GIL-released ops).
        proc_results: Optional[List[Optional[dict]]] = None
        try:
            def _slim_ctx(ctx):
                return {
                    k: v for k, v in ctx.__dict__.items()
                    if not isinstance(v, np.ndarray)
                }
            payloads = [
                (
                    prepared_faces[i]['canvas'],
                    all_regions[i],
                    prepared_faces[i]['shifted_bbox'],
                    prepared_faces[i]['shifted_landmarks'],
                    ieds[i],
                    _slim_ctx(ctx),
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
                        img, faces[i], person_mask, ctx, h_img, w_img,
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
                    img, face, person_mask, ctx, h_img, w_img,
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
        result_u8 = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        if is_float:
            return result_u8.astype(np.float32) / 255.0
        return result_u8

    def _stage_body_skin(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        person_mask: Optional[np.ndarray],
        acc_skin: Optional[np.ndarray],
        acc_skin_hair: Optional[np.ndarray],
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
            faces: List of detected FaceData objects.
            h_img, w_img: Image height and width.

        Returns:
            (H, W, 3) float32 BGR image [0, 1].
        """
        # Early exit: all body params are zero
        if (ctx.body_smooth <= 0 and ctx.body_equalize <= 0 and
            ctx.body_whiten <= 0 and ctx.body_match_face <= 0):
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

        # Exclude face hair region if available
        if acc_skin_hair is not None:
            acc_hair = normalize_mask(acc_skin_hair)
            acc_hair = squeeze_mask(acc_hair)
            # Erode hair mask to avoid over-exclusion at edges
            acc_hair_eroded = cv2.erode(acc_hair, np.ones((3, 3), np.uint8), iterations=1)
            body_skin_candidate = np.clip(body_skin_candidate - acc_hair_eroded, 0, 1)

        # Morphological cleaning: open (remove small noise) then close (fill small holes)
        # Scale kernel size to person size, not face size
        person_bbox_size = max(h_img, w_img) * 0.15  # Estimate person region size
        k_morph = max(3, int(person_bbox_size / 50.0) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        body_skin_candidate = cv2.morphologyEx(body_skin_candidate, cv2.MORPH_OPEN, kernel)
        body_skin_candidate = cv2.morphologyEx(body_skin_candidate, cv2.MORPH_CLOSE, kernel)

        # Contiguity check: drop disconnected components that don't touch the face region
        if acc_skin is not None and acc_skin.max() > 0.01:
            # Dilate face skin region for tolerance
            face_skin_dilated = cv2.dilate(acc_skin_norm, np.ones((21, 21), np.uint8), iterations=1)

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

        # 1. Body smoothing (guided filter at body scale, milder than face)
        if ctx.body_smooth > 0:
            s = ctx.body_smooth / 100.0
            # Guided filter radius scales from person size (milder curve than face smoothing)
            person_diag = np.sqrt(h_img ** 2 + w_img ** 2)
            gf_radius = max(3, int(person_diag * 0.04 * s))  # Gentler than face scale
            gf_eps = 0.01 * s
            result_smoothed = np.zeros_like(result)
            for c in range(result.shape[2]):
                result_smoothed[:, :, c] = guided_filter(
                    result[:, :, c], gf_radius, gf_eps, guide=None, max_dim=1200
                )
            # blend_masked expects uint8 [0,255] images, not float32 [0,1]
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            result_smoothed_u8 = np.clip(result_smoothed * 255.0, 0, 255).astype(np.uint8)
            result = blend_masked(result_u8, result_smoothed_u8, body_skin_mask * s * 0.7).astype(np.float32) / 255.0

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
                # Pull a/b toward median (gentler factor than face)
                median_a = np.median(lab[:, :, 1][body_indices])
                median_b = np.median(lab[:, :, 2][body_indices])
                pull = (0.12 * s * body_skin_mask)  # Gentler than face (0.20)
                lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * pull
                lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * pull

                # Light CLAHE on L channel
                l_chan_u8 = np.clip(lab[:, :, 0], 0, 255).astype(np.uint8)
                clip_limit = 1.0 + s * 1.5  # Gentler than face
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l_chan_u8)
                lab_new = lab.copy()
                lab_new[:, :, 0] = l_clahe.astype(np.float32)
                result_equalized_u8 = cv2.cvtColor(np.clip(lab_new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                # blend_masked expects uint8 [0,255] images, not float32 [0,1]
                result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
                result = blend_masked(result_u8, result_equalized_u8, body_skin_mask * s * 0.6).astype(np.float32) / 255.0

        # 4. Body whitening (lighten L channel in body skin)
        if ctx.body_whiten > 0:
            s = ctx.body_whiten / 100.0
            # Use SkinProcessor's whiten method on body skin
            result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            skin_processor = SkinProcessor()
            result_whitened = skin_processor.whiten(result_u8, body_skin_mask, strength=int(ctx.body_whiten))
            result = result_whitened.astype(np.float32) / 255.0

        # 5. Blemish removal on body (conservative size threshold scaled to person)
        # Gated on the stage itself being active (any of the 4 body params nonzero),
        # NOT on body_smooth specifically — each param must independently unlock its
        # own sub-step, and blemish removal is a reasonable default whenever body skin
        # is being touched at all. Strength is a fixed conservative default scaled by
        # the strongest active param, rather than derived from body_smooth alone
        # (deriving it from one unrelated param was the bug: a user setting only
        # body_whiten/body_match_face/body_equalize got silently zero blemish removal).
        if (ctx.body_smooth > 0 or ctx.body_equalize > 0 or
            ctx.body_whiten > 0 or ctx.body_match_face > 0):
            active_strength = max(ctx.body_smooth, ctx.body_equalize, ctx.body_whiten, ctx.body_match_face)
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

        # For functions that don't yet support float, convert temporarily
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
            result_u8 = _to_uint8_if_float(result)
            result_u8 = subject_aware_transfer(self, result_u8, style_ref, target_faces=faces, target_person=person_mask)
            result = _to_float_if_needed(result_u8, is_float)

        # Colour transfer (reference-based)
        if ctx.color_ref is not None:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.color_transfer(
                result_u8, ctx.color_ref, intensity=ctx.color_transfer_intensity
            )
            result = _to_float_if_needed(result_u8, is_float)

        # --- White balance (LCH-based, Phase 1.d) ---
        if ctx.white_balance_kelvin != _DEFAULTS["white_balance_kelvin"] or ctx.white_balance_tint != _DEFAULTS["white_balance_tint"]:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.white_balance_lch(
                result_u8,
                temperature=ctx.white_balance_kelvin,
                tint=ctx.white_balance_tint,
            )
            result = _to_float_if_needed(result_u8, is_float)

        # --- Master HSL (Phase 1.d) — global LCH adjustments ---
        if ctx.hsl_hue_global != 0 or ctx.hsl_sat_global != 0 or ctx.hsl_lum_global != 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.adjust_hsl_lch(
                result_u8,
                hue_shift=ctx.hsl_hue_global * 0.6,
                sat_scale=1.0 + ctx.hsl_sat_global / 100.0,
                lum_shift=ctx.hsl_lum_global * 0.5,
            )
            result = _to_float_if_needed(result_u8, is_float)

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

        if ctx.tonal_curve_strength > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = tonal.apply_hd_curve(result_u8, strength=ctx.tonal_curve_strength)
            result = _to_float_if_needed(result_u8, is_float)

        # Run core color grading (grade() now supports float natively)
        if ctx.color_grade_stack:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.grade_stack(result_u8, ctx.color_grade_stack)
            result = _to_float_if_needed(result_u8, is_float)
        elif ctx.color_grade:
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]

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
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader._split_tone_three_way(result_u8, tones, mask=acc_skin)
            result = _to_float_if_needed(result_u8, is_float)

        if ctx.highlight_rolloff_strength > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = highlight.apply_highlight_rolloff(result_u8, ctx.highlight_rolloff_strength)
            result = _to_float_if_needed(result_u8, is_float)

        # ---- Skin light-wrap diffusion (anime) ----
        if ctx.skin_glow > 0 and acc_skin is not None and acc_skin.max() > 0.01:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = apply_skin_diffusion(result_u8, acc_skin, strength=ctx.skin_glow)
            result = _to_float_if_needed(result_u8, is_float)

        # Apply Global Cinematic Bloom
        if ctx.bloom > 0.0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = apply_global_bloom(
                result_u8,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )
            result = _to_float_if_needed(result_u8, is_float)

        # Apply ctx-level atmospheric glow
        if ctx.glow > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader._add_glow(result_u8, ctx.glow / 100.0, mask=g_mask)
            result = _to_float_if_needed(result_u8, is_float)

        # ---- Stage C4: 透明感 / 空気感 Finish Pack ----

        # Fade toe: lifted-black with hue-locked toe (L-only in LAB)
        if ctx.fade_toe > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.fade_toe(result_u8, ctx.fade_toe / 100.0, mask=acc_skin)
            result = _to_float_if_needed(result_u8, is_float)

        # Highlight drift: hue rotation toward cyan in highlights, skin-protected
        if ctx.highlight_drift > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.highlight_drift(result_u8, ctx.highlight_drift / 100.0, mask=acc_skin)
            result = _to_float_if_needed(result_u8, is_float)

        # Airy haze: L-threshold-scoped glow with person_mask-aware distance falloff
        if ctx.airy_haze > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.airy_haze(result_u8, ctx.airy_haze / 100.0, person_mask=person_mask)
            result = _to_float_if_needed(result_u8, is_float)

        # Clarity split: negative form-band clarity + positive micro-contrast
        if ctx.clarity_split_neg > 0 or ctx.clarity_split_pos > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.clarity_split(
                result_u8,
                ctx.clarity_split_neg / 100.0,
                ctx.clarity_split_pos / 100.0,
                mask=None
            )
            result = _to_float_if_needed(result_u8, is_float)

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
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._apply_white_costume_lift(result_u8, acc_skin, acc_lips, ctx.grade_intensity)
            result = _to_float_if_needed(result_u8, is_float)

        # Apply ctx-level vignette
        if ctx.vignette > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader._add_vignette(result_u8, ctx.vignette / 100.0)
            result = _to_float_if_needed(result_u8, is_float)

        if ctx.grain_strength > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = grain.apply_film_grain(result_u8, ctx.grain_strength)
            result = _to_float_if_needed(result_u8, is_float)

        # --- Negative split tone ---
        if ctx.negative_split_tone_shadow > 0 or ctx.negative_split_tone_highlight > 0:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.negative_split_tone(
                result_u8,
                shadow_desat=ctx.negative_split_tone_shadow / 100.0,
                highlight_desat=ctx.negative_split_tone_highlight / 100.0,
            )
            result = _to_float_if_needed(result_u8, is_float)

        # --- B&W channel mixer ---
        bw_active = (
            ctx.bw_channel_mixer_r != _DEFAULTS["bw_channel_mixer_r"]
            or ctx.bw_channel_mixer_g != _DEFAULTS["bw_channel_mixer_g"]
            or ctx.bw_channel_mixer_b != _DEFAULTS["bw_channel_mixer_b"]
        )
        if bw_active:
            result_u8 = _to_uint8_if_float(result)
            result_u8 = self._grader.channel_mixer_bw(
                result_u8,
                r_weight=ctx.bw_channel_mixer_r / 100.0,
                g_weight=ctx.bw_channel_mixer_g / 100.0,
                b_weight=ctx.bw_channel_mixer_b / 100.0,
            )
            result = _to_float_if_needed(result_u8, is_float)

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

    @staticmethod
    def _apply_white_costume_lift(
        img: np.ndarray,
        acc_skin_hair: np.ndarray,
        acc_lips: np.ndarray,
        grade_intensity: float,
    ) -> np.ndarray:
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
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

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
    """Float32 [0,1] variant of _adjust_vibrance.

    ``vibrance()`` in utils.py requires uint8 [0,255] BGR input (it round-trips
    through cv2 HSV conversion); feeding it float32 [0,1] silently corrupts the
    image instead of raising, since cv2 accepts float32 as an already-normalized
    HSV range. Convert at the boundary instead.
    """
    bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
    out_u8 = _adjust_vibrance(bgr_u8, vibrance)
    return out_u8.astype(np.float32) / 255.0


def _apply_uniform_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Uniform saturation adjustment."""
    if saturation == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + saturation / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _F_apply_uniform_saturation(img_f: np.ndarray, saturation: float) -> np.ndarray:
    """Float32 [0,1] variant of _apply_uniform_saturation."""
    if saturation == 0:
        return img_f
    bgr_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + saturation / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    out_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out_u8.astype(np.float32) / 255.0


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
            **kwargs,
        )

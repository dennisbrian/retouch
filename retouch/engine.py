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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection import FaceDetector, FaceData, FaceContext
from .parsing import FaceParser, FaceRegions
from .geometry import FaceReshaper
from .makeup import MakeupEngine
from .frequency import separate as freq_separate, combine as freq_combine
from .skin import SkinProcessor
from .blemish import BlemishRemover
from .eyes import EyeEnhancer
from .undereye import UnderEyeRepairer
from .lips import LipEnhancer
from .teeth import TeethWhitener
from .grading import ColorGrader, PRESETS
from .hair import HairEnhancer
from .relight import Relighter
from .recipes import RECIPES
from .style import StyleProfile
from .utils import correct_exposure, apply_global_bloom


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROXY_MAX_DIM = 2048

# Anime cinematic variants get standard nose blush, no slimming
# Mapped modularly in recipes config


# ---------------------------------------------------------------------------
# ProcessingContext — typed parameter bag (replaces raw dict)
# ---------------------------------------------------------------------------

@dataclass
class ProcessingContext:
    """Fully-resolved, typed processing parameters for one engine.process() call."""

    # --- Skin ---
    smooth: float = 50.0
    whiten: float = 0.0
    whiten_tone: str = "rosy"
    equalize: float = 0.0
    blemish: float = 0.0
    nose_smooth: Optional[float] = None
    mid_reduction: float = 0.40
    texture_opacity: float = 1.0
    pore_synthesis: float = 0.0
    specular_bloom: float = 0.0
    specular_bloom_tone: str = "rosy"
    dodge_burn: float = 0.0
    relight: float = 0.0
    relight_azimuth: float = 0.0
    relight_elevation: float = 30.0

    # --- Eyes ---
    eye_enhance: float = 0.0
    dark_circles: float = 0.0

    # --- Lips ---
    lip_enhance: float = 0.0
    lip_tint: Optional[Any] = None
    lip_finish: str = "gloss"

    # --- Teeth ---
    teeth_whiten: float = 0.0

    # --- Makeup ---
    blush: float = 0.0
    slimming: float = 0.0

    # --- Hair ---
    hair_enhance: float = 0.0

    # --- Tonal / global ---
    contrast: float = 0.0
    brightness: Optional[float] = None
    highlights: Optional[float] = None
    shadows: Optional[float] = None
    whites: Optional[float] = None
    blacks: Optional[float] = None
    clarity: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    auto_exposure: bool = False

    # --- Colour grading ---
    color_grade: Optional[str] = None
    grade_intensity: float = 0.0
    color_grade_stack: Optional[List[Dict]] = None
    color_ref: Optional[np.ndarray] = None
    color_transfer_intensity: float = 1.0
    chromatic_aberration: Optional[float] = None
    halation: Optional[float] = None
    grain: Optional[float] = None
    lut: Optional[str] = None

    # --- Split toning ---
    shadow_hue: float = 0.0
    shadow_sat: float = 0.0
    midtone_hue: float = 0.0
    midtone_sat: float = 0.0
    highlight_hue: float = 0.0
    highlight_sat: float = 0.0

    # --- Global Bloom (Oniric-Style Glow) ---
    bloom: float = 0.0
    bloom_threshold: float = 210.0
    bloom_softness: float = 30.0
    glow: float = 0.0

    # --- Lens effects ---
    vignette: float = 0.0
    sharpen: float = 0.0
    sharpen_radius: float = 1.0

    # --- Subject separation ---
    subject_separation: float = 0.0

    # --- Finish ---
    impact: float = 0.0

    # --- Internal recipe tag (used for conditional logic) ---
    active_recipe: str = "natural"

    # --- Modular flags ---
    nose_blush: bool = False
    under_eye_blush: bool = False
    white_costume_lift: bool = False

    # Cached per-face detection + parsing; None ⇒ engine detects/parses.
    face_contexts: Optional[List["FaceContext"]] = None


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


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

def resolve_recipe(name: str, _seen: Optional[set] = None) -> Dict:
    """Return a fully-merged recipe dict, resolving 'extends' recursively."""
    if _seen is None:
        _seen = set()
    if name in _seen:
        return RECIPES.get("natural", {})
    _seen.add(name)
    rec = RECIPES.get(name)
    if rec is None:
        rec = RECIPES.get("natural", {})
    parent_name = rec.get("extends")
    if parent_name and parent_name in RECIPES:
        resolved_parent = resolve_recipe(parent_name, _seen)
        merged = copy.deepcopy(resolved_parent)
        _deep_merge(merged, rec)
        return merged
    return rec



def _deep_merge(base: Dict, override: Dict) -> None:
    """Merge *override* into *base* in-place, recursing into nested dicts."""
    for key, val in override.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def build_context(
    active_recipe: str,
    rec: Dict,
    overrides: Dict,
) -> ProcessingContext:
    """Translate a recipe dict + caller overrides into a ProcessingContext.

    Recipe values are treated as defaults; any non-None override wins.
    """

    def _pct(val: float) -> float:
        return float(val) * 100.0

    def _ov(key, recipe_val):
        v = overrides.get(key)
        return v if v is not None else recipe_val

    # --- Skin ---
    r_smooth = _pct(rec.get("frequency", {}).get("smooth", 0.5))
    r_equalize = _pct(rec.get("skin", {}).get("equalize", 0.0))
    r_rosy = rec.get("skin", {}).get("rosy", rec.get("skin", {}).get("porcelain", 0.0))
    r_whiten = _pct(r_rosy)
    r_blemish = r_smooth
    r_mid = rec.get("frequency", {}).get("mid_reduction", 0.35 if r_smooth < 50 else 0.45)
    r_texture = rec.get("texture", {}).get("opacity", 1.0)
    r_pore = _pct(rec.get("texture", {}).get("pore_synthesis", 0.0))
    r_specular_bloom = rec.get("specular_bloom", 0.0)
    r_specular_bloom_tone = rec.get("specular_bloom_tone", "rosy")
    if "porcelain" in rec.get("skin", {}):
        r_whiten_tone = "porcelain"
    else:
        r_whiten_tone = "rosy"
    r_dodge_burn_raw = rec.get("dodge_burn", {})
    if isinstance(r_dodge_burn_raw, dict):
        r_dodge_burn = _pct(r_dodge_burn_raw.get("amount", 0.0))
    else:
        r_dodge_burn = float(r_dodge_burn_raw)
    r_relight_raw = rec.get("skin", {}).get("relight")
    if r_relight_raw is not None:
        r_relight = _pct(r_relight_raw)
    else:
        r_relight = float(rec.get("relight_strength", 0.0))
    r_relight_azimuth = rec.get("skin", {}).get("relight_azimuth", rec.get("light_azimuth", 0.0))
    r_relight_elevation = rec.get("skin", {}).get("relight_elevation", rec.get("light_elevation", 30.0))

    # --- Eyes ---
    eyes = rec.get("eyes", {})
    r_eye = _pct(eyes.get("iris", eyes.get("whites", 0.0)))
    # BUGFIX-3: dark-circle repair must not inherit teeth/eye-white strength.
    r_dark_circles = _pct(eyes.get("dark_circles", 0.0))
    r_teeth = _pct(eyes.get("whites", 0.0))

    # --- Lips ---
    lips = rec.get("lips", {})
    r_lip = _pct(lips.get("gloss", 0.0))
    r_lip_tint = lips.get("tint", None)

    # --- Hair ---
    r_hair = _pct(rec.get("hair", {}).get("shine", 0.0))

    # --- Colour grading ---
    harmony = rec.get("color_harmony", {})
    r_color_grade = harmony.get("preset", None)
    r_grade_intensity = harmony.get("amount", 0.0)

    # If caller supplied color_grade override, default intensity to 1.0
    caller_grade = overrides.get("color_grade")
    caller_intensity = overrides.get("grade_intensity")
    resolved_intensity = (
        caller_intensity if caller_intensity is not None
        else (1.0 if caller_grade is not None else r_grade_intensity)
    )

    r_contrast = rec.get("contrast", 0.0)
    r_brightness = rec.get("brightness", None)
    r_highlights = rec.get("highlights", None)
    r_shadows = rec.get("shadows", None)
    r_whites = rec.get("whites", None)
    r_blacks = rec.get("blacks", None)
    r_clarity = rec.get("clarity", 0.0)
    r_vibrance = rec.get("vibrance", 0.0)
    r_saturation = rec.get("saturation", 0.0)
    r_shadow_hue = rec.get("shadow_hue", 0.0)
    r_shadow_sat = rec.get("shadow_sat", 0.0)
    r_midtone_hue = rec.get("midtone_hue", 0.0)
    r_midtone_sat = rec.get("midtone_sat", 0.0)
    r_highlight_hue = rec.get("highlight_hue", 0.0)
    r_highlight_sat = rec.get("highlight_sat", 0.0)
    r_glow = rec.get("glow", 0.0)
    r_vignette = rec.get("vignette", 0.0)
    r_sharpen = rec.get("sharpen", 0.0)
    r_sharpen_radius = rec.get("sharpen_radius", 1.0)
    r_subject_sep = rec.get("subject_separation", 0.0)
    r_impact = _pct(rec.get("finish", {}).get("impact", 0.0))
    r_bloom = _pct(rec.get("bloom", {}).get("opacity", 0.0))
    r_bloom_threshold = rec.get("bloom", {}).get("threshold", 210.0)
    r_bloom_softness = rec.get("bloom", {}).get("softness", 30.0)

    # --- Reshaping / makeup defaults ---
    r_slimming = rec.get("slimming", 0.0)
    r_blush = rec.get("blush", 0.0)
    r_lip_finish = rec.get("lip_finish", "gloss")
    r_nose_blush = rec.get("nose_blush", False)
    r_under_eye_blush = rec.get("under_eye_blush", False)
    r_white_costume_lift = rec.get("white_costume_lift", False)

    return ProcessingContext(
        smooth=_ov("smooth", r_smooth),
        whiten=_ov("whiten", r_whiten),
        whiten_tone=_ov("whiten_tone", r_whiten_tone),
        equalize=_ov("equalize", r_equalize),
        blemish=_ov("blemish", r_blemish),
        nose_smooth=overrides.get("nose_smooth"),
        mid_reduction=_ov("mid_reduction", r_mid),
        texture_opacity=_ov("texture_opacity", r_texture),
        pore_synthesis=_ov("pore_synthesis", r_pore),
        specular_bloom=_ov("specular_bloom", r_specular_bloom),
        specular_bloom_tone=_ov("specular_bloom_tone", r_specular_bloom_tone),
        dodge_burn=_ov("dodge_burn", r_dodge_burn),
        relight=_ov("relight", r_relight),
        relight_azimuth=_ov("relight_azimuth", r_relight_azimuth),
        relight_elevation=_ov("relight_elevation", r_relight_elevation),
        eye_enhance=_ov("eye_enhance", r_eye),
        dark_circles=_ov("dark_circles", r_dark_circles),
        lip_enhance=_ov("lip_enhance", r_lip),
        lip_tint=_ov("lip_tint", r_lip_tint),
        lip_finish=_ov("lip_finish", r_lip_finish),
        teeth_whiten=_ov("teeth_whiten", r_teeth),
        blush=_ov("blush", r_blush),
        slimming=_ov("slimming", r_slimming),
        hair_enhance=_ov("hair_enhance", r_hair),
        contrast=_ov("contrast", r_contrast),
        brightness=_ov("brightness", r_brightness),
        highlights=_ov("highlights", r_highlights),
        shadows=_ov("shadows", r_shadows),
        whites=_ov("whites", r_whites),
        blacks=_ov("blacks", r_blacks),
        clarity=_ov("clarity", r_clarity),
        vibrance=_ov("vibrance", r_vibrance),
        saturation=_ov("saturation", r_saturation),
        shadow_hue=_ov("shadow_hue", r_shadow_hue),
        shadow_sat=_ov("shadow_sat", r_shadow_sat),
        midtone_hue=_ov("midtone_hue", r_midtone_hue),
        midtone_sat=_ov("midtone_sat", r_midtone_sat),
        highlight_hue=_ov("highlight_hue", r_highlight_hue),
        highlight_sat=_ov("highlight_sat", r_highlight_sat),
        glow=_ov("glow", r_glow),
        vignette=_ov("vignette", r_vignette),
        sharpen=_ov("sharpen", r_sharpen),
        sharpen_radius=_ov("sharpen_radius", r_sharpen_radius),
        subject_separation=_ov("subject_separation", r_subject_sep),
        auto_exposure=overrides.get("auto_exposure", False),
        color_grade=_ov("color_grade", r_color_grade),
        grade_intensity=resolved_intensity,
        color_grade_stack=overrides.get("color_grade_stack"),
        # BUGFIX-2: color_ref comes only from the caller override, never None-initialised twice
        color_ref=overrides.get("color_ref"),
        color_transfer_intensity=overrides.get("color_transfer_intensity", 1.0),
        chromatic_aberration=overrides.get("chromatic_aberration"),
        halation=overrides.get("halation"),
        grain=overrides.get("grain"),
        lut=overrides.get("lut"),
        impact=_ov("impact", r_impact),
        bloom=_ov("bloom", r_bloom),
        bloom_threshold=_ov("bloom_threshold", r_bloom_threshold),
        bloom_softness=_ov("bloom_softness", r_bloom_softness),
        active_recipe=active_recipe,
        nose_blush=_ov("nose_blush", r_nose_blush),
        under_eye_blush=_ov("under_eye_blush", r_under_eye_blush),
        white_costume_lift=_ov("white_costume_lift", r_white_costume_lift),
    )


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def _norm_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a float32 mask in [0, 1]. Returns None if input is None."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return m


def _accum(acc: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Add normalised mask into accumulator, clamped to 1."""
    if mask is None:
        return acc
    return np.clip(acc + _norm_mask(mask), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-face processing result
# ---------------------------------------------------------------------------

@dataclass
class _FaceResult:
    canvas: np.ndarray
    skin_mask: np.ndarray
    skin_hair_mask: np.ndarray
    lips_mask: np.ndarray
    sharpen_mask: np.ndarray
    roi_box: Tuple[int, int, int, int]


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


# ---------------------------------------------------------------------------
# Per-face core pipeline (module-level so it is picklable / callable from
# worker processes without the RetouchEngine instance, which holds non-picklable
# ONNX sessions and MediaPipe tasks).
# ---------------------------------------------------------------------------

def _process_face_core(
    canvas: np.ndarray,
    regions: FaceRegions,
    shifted_face: FaceData,
    ctx: ProcessingContext,
    roi_x1: int,
    roi_y1: int,
    roi_h: int,
    roi_w: int,
    roi_person_mask: Optional[np.ndarray],
    processors: Dict[str, Any],
) -> _FaceResult:
    """Run the per-face rendering pipeline on a private ROI canvas.

    ``processors`` maps names to the processor instances used by the
    pipeline ('skin', 'relighter', 'blemish', 'undereye', 'eyes',
    'teeth', 'lips', 'makeup', 'hair'). This lets the same logic run
    either inside the engine (passing ``self._xxx``) or inside a worker
    process (passing freshly-instantiated processors).
    """
    skin = processors['skin']
    relighter = processors['relighter']
    blemish = processors['blemish']
    undereye = processors['undereye']
    eyes = processors['eyes']
    teeth = processors['teeth']
    lips = processors['lips']
    makeup = processors['makeup']
    hair = processors['hair']

    face_width = shifted_face.ied * 2.5

    # ---- Accumulate masks ----
    skin_n = _norm_mask(regions.skin)
    hair_n = _norm_mask(regions.hair)
    lips_n = _norm_mask(regions.lips)
    neck_n = _norm_mask(regions.neck)

    acc_skin = np.zeros((roi_h, roi_w), dtype=np.float32)
    acc_skin_hair = np.zeros((roi_h, roi_w), dtype=np.float32)
    if skin_n is not None:
        acc_skin = np.clip(acc_skin + skin_n, 0.0, 1.0)
        acc_skin_hair = np.clip(acc_skin_hair + skin_n, 0.0, 1.0)
    if hair_n is not None:
        acc_skin_hair = np.clip(acc_skin_hair + hair_n, 0.0, 1.0)
    if neck_n is not None:
        acc_skin_hair = np.clip(acc_skin_hair + neck_n, 0.0, 1.0)

    acc_lips = np.zeros((roi_h, roi_w), dtype=np.float32)
    if lips_n is not None:
        acc_lips = np.clip(acc_lips + lips_n, 0.0, 1.0)

    # ---- Frequency separation ----
    original_lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    layers = freq_separate(canvas, face_width)

    # ---- Build smooth mask (protect eyes/brows/lips) ----
    smooth_mask = skin_n.copy() if skin_n is not None else np.zeros((roi_h, roi_w), np.float32)

    if shifted_face.ied > 0:
        k_size = max(3, int(shifted_face.ied * 0.08) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated_left_eye = cv2.dilate(regions.left_eye, kernel) if regions.left_eye is not None else None
        dilated_right_eye = cv2.dilate(regions.right_eye, kernel) if regions.right_eye is not None else None
    else:
        dilated_left_eye = regions.left_eye
        dilated_right_eye = regions.right_eye

    for excl in (
        dilated_left_eye, dilated_right_eye,
        regions.left_under_eye, regions.right_under_eye,
        regions.left_eyebrow, regions.right_eyebrow,
        regions.lips,
    ):
        if excl is not None:
            smooth_mask = np.clip(smooth_mask - excl.astype(np.float32), 0.0, 1.0)

    # ---- Frequency-based smoothing ----
    if ctx.nose_smooth is not None:
        nose_mask = _norm_mask(regions.nose)
        if nose_mask is not None:
            face_without_nose = np.clip(smooth_mask - nose_mask * smooth_mask, 0.0, 1.0)
            canvas = freq_combine(
                layers,
                skin_mask=face_without_nose,
                smooth_strength=ctx.smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
            )
            nose_canvas = freq_combine(
                layers,
                skin_mask=nose_mask * smooth_mask,
                smooth_strength=ctx.nose_smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
            )
            nose_alpha = (nose_mask * smooth_mask)[:, :, np.newaxis]
            canvas = (
                nose_canvas.astype(np.float32) * nose_alpha
                + canvas.astype(np.float32) * (1.0 - nose_alpha)
            ).astype(np.uint8)
        else:
            canvas = freq_combine(
                layers,
                skin_mask=smooth_mask,
                smooth_strength=ctx.smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
            )
    else:
        canvas = freq_combine(
            layers,
            skin_mask=smooth_mask,
            smooth_strength=ctx.smooth / 100.0,
            mid_reduction=ctx.mid_reduction,
            texture_opacity=ctx.texture_opacity,
            face_width=face_width,
            pore_synthesis=ctx.pore_synthesis / 100.0,
            roi_coords=(roi_x1, roi_y1),
        )

    # ---- Skin equalization ----
    if ctx.equalize > 0:
        canvas = skin.equalize(canvas, regions.skin, ctx.equalize, ref_lab=original_lab)

    # ---- Foundation / whitening ----
    if ctx.whiten != 0:
        canvas = skin.whiten(canvas, regions.skin, ctx.whiten, tone=ctx.whiten_tone)

    # ---- Virtual studio relighting ----
    if ctx.relight > 0:
        canvas = relighter.relight(
            canvas,
            shifted_face.landmarks,
            regions.skin,
            face_width=face_width,
            strength=ctx.relight,
            azimuth=ctx.relight_azimuth,
            elevation=ctx.relight_elevation,
        )

    # ---- Specular bloom ----
    if ctx.specular_bloom > 0:
        canvas = skin.apply_specular_bloom(
            canvas, regions.skin, ctx.specular_bloom, tone=ctx.specular_bloom_tone
        )

    # ---- Blemish removal ----
    if ctx.blemish > 0:
        canvas = blemish.remove(canvas, regions.skin, ctx.blemish)

    # ---- Under-eye repair ----
    if ctx.dark_circles > 0:
        canvas = undereye.repair(canvas, regions, ctx.dark_circles)

    # ---- Neck harmonisation ----
    if ctx.whiten != 0 or ctx.equalize > 0:
        canvas = skin.harmonize_neck(
            canvas,
            shifted_face.landmarks,
            roi_person_mask,
            regions.skin,
            regions.neck,
            strength=max(abs(ctx.whiten), ctx.equalize),
        )

    # ---- Eye enhancement ----
    if ctx.eye_enhance > 0:
        canvas = eyes.enhance(canvas, regions, ctx.eye_enhance)

    # ---- Teeth whitening ----
    if ctx.teeth_whiten > 0:
        canvas = teeth.whiten(canvas, regions.mouth_interior, ctx.teeth_whiten)

    # ---- Lip enhancement ----
    if ctx.lip_enhance > 0:
        canvas = lips.enhance(
            canvas, regions.lips, ctx.lip_enhance,
            tint=ctx.lip_tint, finish=ctx.lip_finish,
        )

    # ---- Blush ----
    if ctx.blush > 0:
        canvas = makeup.apply_blush(
            canvas, shifted_face.landmarks, face_width, ctx.blush,
            regions=regions,
            nose_blush=ctx.nose_blush,
            under_eye_blush=ctx.under_eye_blush,
        )

    # ---- Hair shine ----
    if ctx.hair_enhance > 0:
        canvas = hair.enhance(
            canvas, roi_person_mask, regions.face_oval,
            shifted_face.bbox, ctx.hair_enhance, regions.hair,
        )

    # ---- Dodge & burn ----
    if ctx.dodge_burn > 0:
        canvas = skin.dodge_burn(canvas, regions, ctx.dodge_burn)

    # ---- Build sharpening mask ----
    acc_sharpen = np.zeros((roi_h, roi_w), dtype=np.float32)
    eye_sharpen = _accum(np.zeros((roi_h, roi_w), np.float32), regions.left_eye)
    eye_sharpen = _accum(eye_sharpen, regions.right_eye)

    other_sharpen = _accum(np.zeros((roi_h, roi_w), np.float32), regions.left_eyebrow)
    other_sharpen = _accum(other_sharpen, regions.right_eyebrow)

    if regions.hair is not None and _norm_mask(regions.hair).max() > 0.01:
        hair_n_clean = _norm_mask(regions.hair)
        eroded = cv2.erode(hair_n_clean, np.ones((5, 5), np.uint8))
        hair_edges = np.clip(hair_n_clean - eroded, 0.0, 1.0)
        other_sharpen = np.clip(other_sharpen + hair_edges, 0.0, 1.0)

    acc_sharpen = np.clip(
        np.maximum(eye_sharpen * 1.0, other_sharpen * 0.53), 0.0, 1.0
    )

    return _FaceResult(
        canvas=canvas,
        skin_mask=acc_skin,
        skin_hair_mask=acc_skin_hair,
        lips_mask=acc_lips,
        sharpen_mask=acc_sharpen,
        roi_box=(roi_x1, roi_y1, roi_x1 + roi_w, roi_y1 + roi_h)
    )


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
        from .perf_optimizations import FaceProcessorPool
        self._face_pool = FaceProcessorPool()

        # Warm up JIT kernels on engine startup (safe fallback if Numba is missing)
        from .perf_optimizations import warmup_jit_kernels
        warmup_jit_kernels()

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
        hair_enhance: Optional[float] = None,
        dodge_burn: Optional[float] = None,
        relight: Optional[float] = None,
        relight_azimuth: Optional[float] = None,
        relight_elevation: Optional[float] = None,
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
        halation: Optional[float] = None,
        grain: Optional[float] = None,
        lut: Optional[str] = None,
        shadow_hue: Optional[float] = None,
        shadow_sat: Optional[float] = None,
        midtone_hue: Optional[float] = None,
        midtone_sat: Optional[float] = None,
        highlight_hue: Optional[float] = None,
        highlight_sat: Optional[float] = None,
        color_grade_stack=None,
        color_ref: Optional[np.ndarray] = None,
        color_transfer_intensity: float = 1.0,
        fast: bool = False,
        style_profile: Optional[StyleProfile] = None,
        style_ref: Optional[np.ndarray] = None,
        debug_dir: Optional[str] = None,
        face_contexts: Optional[List["FaceContext"]] = None,
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
            "hair_enhance": hair_enhance,
            "dodge_burn": dodge_burn,
            "relight": relight,
            "relight_azimuth": relight_azimuth,
            "relight_elevation": relight_elevation,
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
            "lut": lut,
            "shadow_hue": shadow_hue,
            "shadow_sat": shadow_sat,
            "midtone_hue": midtone_hue,
            "midtone_sat": midtone_sat,
            "highlight_hue": highlight_hue,
            "highlight_sat": highlight_sat,
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
                if pm_norm.ndim == 3:
                    pm_norm = pm_norm.squeeze(-1)
                sharp_fg = np.clip(pm_norm - acc_skin, 0.0, 1.0)
                g_mask = 1.0 - sharp_fg
            else:
                g_mask = np.ones(result.shape[:2], dtype=np.float32)
            cv2.imwrite(os.path.join(debug_dir, "glow_mask.png"), (g_mask * 255).astype(np.uint8))
            
            # Frequency layers of the primary face
            if faces:
                face_width = faces[0].ied * 2.5
                layers = freq_separate(result, face_width)
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
        if max(h, w) > PROXY_MAX_DIM:
            proxy_scale = PROXY_MAX_DIM / float(max(h, w))
            new_w = int(w * proxy_scale)
            new_h = int(h * proxy_scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        core = self._run_core_pipeline(img_bgr, ctx, style_ref, timings)

        if proxy_scale < 1.0:
            core = self._upscale_core_result(core, h, w)

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
        # Stage 3 — Global tonal adjustments
        # ------------------------------------------------------------------
        t3 = time.perf_counter()
        result = self._stage_global(result, ctx)
        timings["global"] = (time.perf_counter() - t3) * 1000

        # ------------------------------------------------------------------
        # Stage 4 — Subject-background separation
        # ------------------------------------------------------------------
        t_subj = time.perf_counter()
        if ctx.subject_separation > 0:
            result = self._stage_subject_separation(result, person_mask, ctx)
        timings["subject_separation"] = (time.perf_counter() - t_subj) * 1000

        # ------------------------------------------------------------------
        # Stage 5 — Colour grading
        # ------------------------------------------------------------------
        t4 = time.perf_counter()
        result = self._stage_grade(
            result, ctx, acc_skin, acc_skin_hair, acc_lips, person_mask,
            style_ref=style_ref, faces=faces,
        )
        timings["grading"] = (time.perf_counter() - t4) * 1000

        # ------------------------------------------------------------------
        # Stage 6 — Selective sharpening + impact finish
        # ------------------------------------------------------------------
        t5 = time.perf_counter()
        result = self._stage_finish(result, ctx, acc_sharpen, faces=faces)
        timings["finish"] = (time.perf_counter() - t5) * 1000

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
        )

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

        # Assemble post-effects settings
        post_effects: Dict[str, Any] = {}
        if ctx.chromatic_aberration is not None:
            post_effects["chromatic_aberration"] = ctx.chromatic_aberration
        if ctx.halation is not None:
            post_effects["halation"] = ctx.halation
        if ctx.grain is not None:
            post_effects["grain"] = ctx.grain
        if ctx.lut is not None:
            post_effects["lut"] = ctx.lut

        skip_glows = ctx.bloom > 0.0

        if ctx.color_grade:
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]
            result = self._grader.grade(
                result, settings, ctx.grade_intensity,
                skip_glows=skip_glows, skip_post_effects=True
            )

        if ctx.bloom > 0.0:
            result = apply_global_bloom(
                result,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )

        if post_effects:
            result = self._grader.grade(result, post_effects, 1.0)

        if ctx.impact > 0:
            result = self._grader.add_impact_finish(result, ctx.impact)
        return result

    def _stage_reshape(self, img: np.ndarray, faces, ctx: ProcessingContext) -> np.ndarray:
        if ctx.slimming > 0:
            return self._reshaper.reshape(img, faces, ctx.slimming)
        return img.copy()

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
            pad_t = int(face_h * 0.8)
            pad_b = int(face_h * 1.8)
            pad_l = int(face_w * 0.6)
            pad_r = int(face_w * 0.6)

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
            payloads = [
                (
                    prepared_faces[i]['canvas'],
                    all_regions[i],
                    prepared_faces[i]['shifted_bbox'],
                    prepared_faces[i]['shifted_landmarks'],
                    ieds[i],
                    ctx,
                    prepared_faces[i]['roi_box'],
                    prepared_faces[i]['roi_person_mask'],
                    prepared_faces[i]['roi_box'][3] - prepared_faces[i]['roi_box'][1],
                    prepared_faces[i]['roi_box'][2] - prepared_faces[i]['roi_box'][0],
                )
                for i in range(len(faces))
            ]
            proc_results = self._face_pool.process_faces(payloads)
        except Exception:
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
            pad_t = int(face_h * 0.8)
            pad_b = int(face_h * 1.8)
            pad_l = int(face_w * 0.6)
            pad_r = int(face_w * 0.6)

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
        """
        strength = ctx.subject_separation
        if strength <= 0 or person_mask is None:
            return img

        pm = person_mask.astype(np.float32)
        if pm.ndim == 3:
            pm = pm.squeeze(-1)
        if pm.max() > 1.0:
            pm /= 255.0

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Subject brighten: +0.3 EV = multiply by 2^(0.3) ≈ 1.23
        s = strength / 100.0
        subject_gain = 1.0 + 0.23 * s
        background_gain = 1.0 - 0.33 * s  # -0.4 EV ≈ * 0.76

        mask_subject = pm
        mask_bg = np.clip(1.0 - pm, 0.0, 1.0)

        L_new = L * (mask_subject * subject_gain + mask_bg * background_gain)
        lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _stage_global(self, img: np.ndarray, ctx: ProcessingContext) -> np.ndarray:
        """Global tonal operators (contrast, brightness, HSL tonal curve)."""
        result = img

        if ctx.contrast:
            result = _adjust_contrast(result, ctx.contrast)

        if ctx.brightness is not None and ctx.brightness != 0:
            gamma = np.clip(1.0 - (ctx.brightness / 100.0), 0.1, 4.0)
            # BUGFIX-4 + PERF: Vectorized LUT calculation
            x = np.arange(256, dtype=np.float32) / 255.0
            _brightness_lut = (np.power(x, gamma) * 255.0).astype(np.uint8)
            result = cv2.LUT(result, _brightness_lut)

        if any(v is not None and v != 0 for v in (
            ctx.highlights, ctx.shadows, ctx.whites, ctx.blacks
        )):
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
            result = _adjust_vibrance(result, ctx.vibrance)

        if ctx.saturation:
            result = _adjust_saturation(result, ctx.saturation)

        return result

    def _stage_grade(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_skin: np.ndarray,
        acc_skin_hair: np.ndarray,
        acc_lips: np.ndarray,
        person_mask: Optional[np.ndarray],
        style_ref: Optional[np.ndarray] = None,
        faces=None,
    ) -> np.ndarray:
        """Colour transfer, grading, and white-costume lift."""
        result = img

        # Subject-Aware Color Transfer (Meitu/Xingtu-style portrait match)
        if style_ref is not None:
            from .style import subject_aware_transfer
            result = subject_aware_transfer(self, result, style_ref, target_faces=faces, target_person=person_mask)

        # Colour transfer (reference-based)
        if ctx.color_ref is not None:
            result = self._grader.color_transfer(
                result, ctx.color_ref, intensity=ctx.color_transfer_intensity
            )

        # Build glow mask — allow glow on skin & background, preserve costume details
        # BUGFIX-1: use accumulated acc_skin (all faces) instead of loop-scoped s_mask
        pm_norm = _norm_mask(person_mask)
        if pm_norm is not None:
            if pm_norm.ndim == 3:
                pm_norm = pm_norm.squeeze(-1)
            sharp_fg = np.clip(pm_norm - acc_skin, 0.0, 1.0)
            g_mask = 1.0 - sharp_fg
        else:
            g_mask = np.ones(img.shape[:2], dtype=np.float32)

        # Determine if we should skip built-in glows (Smart Glow & Orton Glow) in presets
        # because the new high-level bloom control is active
        skip_glows = ctx.bloom > 0.0

        # Assemble post-effects settings (to run after bloom)
        post_effects: Dict[str, Any] = {}
        if ctx.chromatic_aberration is not None:
            post_effects["chromatic_aberration"] = ctx.chromatic_aberration
        if ctx.halation is not None:
            post_effects["halation"] = ctx.halation
        if ctx.grain is not None:
            post_effects["grain"] = ctx.grain
        if ctx.lut is not None:
            post_effects["lut"] = ctx.lut

        # Run core color grading
        if ctx.color_grade_stack:
            result = self._grader.grade_stack(result, ctx.color_grade_stack)
        elif ctx.color_grade:
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            # Extract preset post-effects to run them later
            for k in ["halation", "grain", "chromatic_aberration", "lut"]:
                if k in settings and k not in post_effects:
                    post_effects[k] = settings[k]
            
            result = self._grader.grade(
                result, settings, ctx.grade_intensity,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
                skip_glows=skip_glows, skip_post_effects=True,
            )

        # Apply recipe-level split toning (from ProcessingContext, not preset)
        if any((ctx.shadow_hue, ctx.shadow_sat, ctx.midtone_hue, ctx.midtone_sat, ctx.highlight_hue, ctx.highlight_sat)):
            tones = {
                "shadows": {"hue": ctx.shadow_hue, "sat": ctx.shadow_sat},
                "midtones": {"hue": ctx.midtone_hue, "sat": ctx.midtone_sat},
                "highlights": {"hue": ctx.highlight_hue, "sat": ctx.highlight_sat},
            }
            result = self._grader._split_tone_three_way(result, tones, mask=acc_skin)

        # Apply Global Cinematic Bloom (runs after color grading, but before halation/grain)
        if ctx.bloom > 0.0:
            result = apply_global_bloom(
                result,
                strength=ctx.bloom,
                threshold=ctx.bloom_threshold,
                softness=ctx.bloom_softness,
            )

        # Apply post-effects (halation, lut, grain, chromatic aberration)
        if post_effects:
            result = self._grader.grade(
                result, post_effects, 1.0,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
            )

        # White costume pearl/lavender lift
        if ctx.white_costume_lift:
            result = self._apply_white_costume_lift(result, acc_skin, acc_lips, ctx.grade_intensity)

        return result

    def _stage_finish(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_sharpen: np.ndarray,
        faces=None,
    ) -> np.ndarray:
        result = img
        sharpen_mask = acc_sharpen
        if ctx.sharpen > 0 and sharpen_mask.max() <= 0.01:
            sharpen_mask = np.ones_like(sharpen_mask)
        if ctx.sharpen > 0 or acc_sharpen.max() > 0.01:
            radius = ctx.sharpen_radius
            if faces:
                avg_ied = np.mean([f.ied for f in faces])
                radius = ctx.sharpen_radius * (avg_ied / 80.0)
                radius = max(0.5, min(4.0, radius))
            amount = max(1.2, ctx.sharpen / 100.0 * 2.0) if ctx.sharpen > 0 else 1.2
            result = _apply_selective_sharpening(
                result, sharpen_mask, radius=radius, amount=amount, threshold=2
            )
        if ctx.impact > 0:
            result = self._grader.add_impact_finish(result, ctx.impact)
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
            pass
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
    if vibrance == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    factor = 1.0 + (vibrance / 100.0) * (1.0 - s / 255.0)
    # Optimized skin hue mask: red-orange hues (keep h > 0 to match original intent, only drop redundant h < 180)
    skin_hue = ((h > 0) & (h < 25)) | (h > 160)
    skin_factor = np.clip(1.0 - (vibrance / 100.0) * 0.5, 0.5, 1.0)
    factor = np.where(skin_hue, np.minimum(factor, skin_factor), factor)
    hsv[:, :, 1] = np.clip(s * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _adjust_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Uniform saturation adjustment."""
    if saturation == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + saturation / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _adjust_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 0:
        return img
    f = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
    result = f * (img.astype(np.float32) - 128.0) + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)


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
        # OPTIMIZATION: Compute grayscale directly from float arrays instead of uint8 casting.
        # OpenCV uses BGR ordering, so channels are 0: Blue, 1: Green, 2: Red.
        # Standard BT.601 weights are 0.114 * B + 0.587 * G + 0.299 * R.
        gray_high = 0.114 * high_freq[:, :, 0] + 0.587 * high_freq[:, :, 1] + 0.299 * high_freq[:, :, 2]
        threshold_mask = (np.abs(gray_high) >= threshold)[:, :, np.newaxis]
        sharpened_diff = threshold_mask * (high_freq * (amount * mask_3d))
    else:
        sharpened_diff = high_freq * (amount * mask_3d)
    return np.clip(img_f + sharpened_diff, 0, 255).astype(np.uint8)


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

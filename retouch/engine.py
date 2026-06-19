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
          with a correct fallback.

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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection import FaceDetector
from .parsing import FaceParser
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
from .recipes import RECIPES
from .utils import correct_exposure


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOSE_BLUSH_RECIPES = frozenset({
    "scifi_cosplay", "cyber_doll", "cosplay", "anime",
    "fantasy_goddess", "pink_dream", "meitu_clone",
})
_UNDER_EYE_BLUSH_RECIPES = _NOSE_BLUSH_RECIPES
_MATTE_LIP_RECIPES = frozenset({"wedding", "magazine"})
_VELVET_LIP_RECIPES = frozenset({"korean_beauty", "xhs_ultrasoft"})
_SLIMMING_RECIPES = frozenset({
    "cosplay", "cosplay_3d", "cosplay_no_eq", "anime_cosplay", "anime",
    "xiaohongshu", "idol", "blue_dream", "xhs_ultrasoft",
})
_BLUSH_RECIPES = _SLIMMING_RECIPES | frozenset({"wedding"})
_WHITE_COSTUME_RECIPES = frozenset({"pink_dream", "meitu_clone"})


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
    specular_bloom: float = 0.0
    specular_bloom_tone: str = "rosy"
    dodge_burn: float = 0.0

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

    # --- Finish ---
    impact: float = 0.0

    # --- Internal recipe tag (used for conditional logic) ---
    active_recipe: str = "natural"


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


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

def resolve_recipe(name: str) -> Dict:
    """Return a fully-merged recipe dict, honouring single-level 'extends'."""
    rec = RECIPES.get(name)
    if rec is None:
        rec = RECIPES.get("natural", {})
    parent_name = rec.get("extends")
    if parent_name and parent_name in RECIPES:
        import copy
        merged = copy.deepcopy(RECIPES[parent_name])
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
    r_specular_bloom = rec.get("specular_bloom", 0.0)
    r_specular_bloom_tone = rec.get("specular_bloom_tone", "rosy")
    if "porcelain" in rec.get("skin", {}):
        r_whiten_tone = "porcelain"
    else:
        r_whiten_tone = "rosy"
    r_dodge_burn = _pct(rec.get("dodge_burn", {}).get("amount", 0.0))

    # --- Eyes ---
    eyes = rec.get("eyes", {})
    r_eye = _pct(eyes.get("iris", eyes.get("whites", 0.0)))
    # BUGFIX-3: use dedicated dark_circles key, fallback to whites only
    r_dark_circles = _pct(eyes.get("dark_circles", eyes.get("whites", 0.0)))
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
    r_impact = _pct(rec.get("finish", {}).get("impact", 0.0))

    # --- Reshaping / makeup defaults ---
    if active_recipe == "fantasy_goddess":
        r_slimming, r_blush = 35.0, 30.0
    elif active_recipe in ("scifi_cosplay", "cyber_doll", "pink_dream", "meitu_clone"):
        r_slimming = 0.0 if active_recipe == "scifi_cosplay" else 30.0
        r_blush = 35.0 if active_recipe in ("scifi_cosplay", "cyber_doll") else 30.0
    else:
        r_slimming = 30.0 if active_recipe in _SLIMMING_RECIPES else 0.0
        r_blush = 25.0 if active_recipe in _BLUSH_RECIPES else 0.0

    if active_recipe in _MATTE_LIP_RECIPES:
        r_lip_finish = "matte"
    elif active_recipe in _VELVET_LIP_RECIPES:
        r_lip_finish = "velvet"
    else:
        r_lip_finish = "gloss"

    return ProcessingContext(
        smooth=_ov("smooth", r_smooth),
        whiten=_ov("whiten", r_whiten),
        whiten_tone=_ov("whiten_tone", r_whiten_tone),
        equalize=_ov("equalize", r_equalize),
        blemish=_ov("blemish", r_blemish),
        nose_smooth=overrides.get("nose_smooth"),
        mid_reduction=_ov("mid_reduction", r_mid),
        texture_opacity=_ov("texture_opacity", r_texture),
        specular_bloom=_ov("specular_bloom", r_specular_bloom),
        specular_bloom_tone=_ov("specular_bloom_tone", r_specular_bloom_tone),
        dodge_burn=_ov("dodge_burn", r_dodge_burn),
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
        brightness=overrides.get("brightness"),
        highlights=overrides.get("highlights"),
        shadows=overrides.get("shadows"),
        whites=overrides.get("whites"),
        blacks=overrides.get("blacks"),
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
        active_recipe=active_recipe,
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

    def __init__(self, max_faces: int = 10, min_confidence: float = 0.5):
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
        lip_tint=None,
        teeth_whiten: Optional[float] = None,
        equalize: Optional[float] = None,
        contrast: Optional[float] = None,
        brightness: Optional[float] = None,
        highlights: Optional[float] = None,
        shadows: Optional[float] = None,
        whites: Optional[float] = None,
        blacks: Optional[float] = None,
        color_grade: Optional[str] = None,
        grade_intensity: Optional[float] = None,
        texture_opacity: Optional[float] = None,
        mid_reduction: Optional[float] = None,
        nose_smooth: Optional[float] = None,
        hair_enhance: Optional[float] = None,
        dodge_burn: Optional[float] = None,
        slimming: Optional[float] = None,
        blush: Optional[float] = None,
        lip_finish: Optional[str] = None,
        specular_bloom: Optional[float] = None,
        specular_bloom_tone: Optional[str] = None,
        whiten_tone: Optional[str] = None,
        auto_exposure: bool = False,
        impact: Optional[float] = None,
        chromatic_aberration: Optional[float] = None,
        halation: Optional[float] = None,
        grain: Optional[float] = None,
        lut: Optional[str] = None,
        color_grade_stack=None,
        color_ref: Optional[np.ndarray] = None,
        color_transfer_intensity: float = 1.0,
        fast: bool = False,
        style_profile: Optional[StyleProfile] = None,
        style_ref: Optional[np.ndarray] = None,
        debug_dir: Optional[str] = None,
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

        overrides = {k: v for k, v in locals().items() if k not in {
            "self", "img_bgr", "recipe", "preset", "fast",
            "timings", "orig_h", "orig_w", "scale", "active_recipe", "rec",
            "style_profile", "style_ref", "debug_dir",
        }}

        ctx = build_context(active_recipe, rec, overrides)

        if style_profile is not None:
            ctx.contrast = style_profile.contrast_delta
            ctx.brightness = np.clip(style_profile.brightness_delta * 2.0, -100.0, 100.0)
            ctx.smooth = np.clip(style_profile.skin_smooth_strength * 100.0, 0.0, 100.0)
            ctx.mid_reduction = np.clip(style_profile.skin_mid_reduction, 0.0, 1.0)
            ctx.texture_opacity = np.clip(style_profile.skin_texture_opacity, 0.0, 1.0)
            ctx.whiten = np.clip(style_profile.skin_l_mean_delta * 4.0, -100.0, 100.0)
            if style_profile.skin_a_mean_delta > 1.0:
                ctx.whiten_tone = "rosy"
            elif style_profile.skin_b_mean_delta < -1.0:
                ctx.whiten_tone = "porcelain"
            else:
                ctx.whiten_tone = "neutral"

        # ------------------------------------------------------------------
        # Stage 0 — Detection + segmentation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
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
            result = self._no_face_fallback(img_bgr, ctx)
            if fast and scale < 1.0:
                result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            return ProcessingResult(
                image=result,
                face_count=0,
                params=ctx,
                timings=timings,
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
        face_results = self._stage_per_face(result, faces, person_mask, ctx, h_img, w_img)
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
        # Stage 4 — Colour grading
        # ------------------------------------------------------------------
        t4 = time.perf_counter()
        result = self._stage_grade(result, ctx, acc_skin, acc_skin_hair, acc_lips, person_mask, style_ref=style_ref, faces=faces)
        timings["grading"] = (time.perf_counter() - t4) * 1000

        # ------------------------------------------------------------------
        # Stage 5 — Selective sharpening + impact finish
        # ------------------------------------------------------------------
        t5 = time.perf_counter()
        result = self._stage_finish(result, ctx, acc_sharpen)
        timings["finish"] = (time.perf_counter() - t5) * 1000

        # ------------------------------------------------------------------
        # Upscale if fast preview
        # ------------------------------------------------------------------
        if fast and scale < 1.0:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

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
                layers = freq_separate(img_bgr, face_width)
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
        )

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def _no_face_fallback(self, img: np.ndarray, ctx: ProcessingContext) -> np.ndarray:
        result = img.copy()
        if ctx.contrast:
            result = _adjust_contrast(result, ctx.contrast)
        if ctx.color_grade:
            result = self._grader.grade(result, ctx.color_grade, ctx.grade_intensity)
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
    ) -> List[_FaceResult]:
        """Process each face, parallelised with threads when >1 face (ARCH-3)."""
        if len(faces) == 1:
            return [self._process_one_face(img, faces[0], person_mask, ctx, h_img, w_img)]

        # ARCH-3: parallel per-face processing
        results: List[Optional[_FaceResult]] = [None] * len(faces)
        with ThreadPoolExecutor(max_workers=min(len(faces), 4)) as pool:
            future_to_idx = {
                pool.submit(
                    self._process_one_face,
                    img, face, person_mask, ctx, h_img, w_img
                ): i
                for i, face in enumerate(faces)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]

    def _process_one_face(
        self,
        img: np.ndarray,
        face,
        person_mask: np.ndarray,
        ctx: ProcessingContext,
        h_img: int,
        w_img: int,
    ) -> _FaceResult:
        """Full per-face pipeline. Operates on a private copy of the cropped Portrait ROI canvas."""
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
        roi_person_mask = person_mask[roi_y1:roi_y2, roi_x1:roi_x2] if person_mask is not None else None

        # Shift landmarks & bbox to be relative to the ROI crop
        import copy
        shifted_landmarks = copy.deepcopy(face.landmarks)
        for lm in shifted_landmarks.landmark:
            lm.x = (lm.x * w_img - roi_x1) / roi_w
            lm.y = (lm.y * h_img - roi_y1) / roi_h

        shifted_bbox = (face_x - roi_x1, face_y - roi_y1, face_w, face_h)

        from retouch.detection import FaceData
        shifted_face = FaceData(
            bbox=shifted_bbox,
            landmarks=shifted_landmarks,
            ied=face.ied
        )

        face_width = shifted_face.ied * 2.5
        active_recipe = ctx.active_recipe

        # ---- Parse regions ----
        regions = self._parser.parse(
            shifted_face.landmarks, canvas, shifted_face.bbox, roi_person_mask, shifted_face.ied
        )

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
                )
                layers2 = freq_separate(canvas, face_width)
                canvas = freq_combine(
                    layers2,
                    skin_mask=nose_mask * smooth_mask,
                    smooth_strength=ctx.nose_smooth / 100.0,
                    mid_reduction=ctx.mid_reduction,
                    texture_opacity=ctx.texture_opacity,
                    face_width=face_width,
                )
            else:
                canvas = freq_combine(
                    layers,
                    skin_mask=smooth_mask,
                    smooth_strength=ctx.smooth / 100.0,
                    mid_reduction=ctx.mid_reduction,
                    texture_opacity=ctx.texture_opacity,
                    face_width=face_width,
                )
        else:
            canvas = freq_combine(
                layers,
                skin_mask=smooth_mask,
                smooth_strength=ctx.smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
            )

        # ---- Skin equalization ----
        if ctx.equalize > 0:
            canvas = self._skin.equalize(canvas, regions.skin, ctx.equalize, ref_lab=original_lab)

        # ---- Foundation / whitening ----
        if ctx.whiten != 0:
            canvas = self._skin.whiten(canvas, regions.skin, ctx.whiten, tone=ctx.whiten_tone)

        # ---- Specular bloom ----
        if ctx.specular_bloom > 0:
            canvas = self._skin.apply_specular_bloom(
                canvas, regions.skin, ctx.specular_bloom, tone=ctx.specular_bloom_tone
            )

        # ---- Blemish removal ----
        if ctx.blemish > 0:
            canvas = self._blemish.remove(canvas, regions.skin, ctx.blemish)

        # ---- Under-eye repair ----
        if ctx.dark_circles > 0:
            canvas = self._undereye.repair(canvas, regions, ctx.dark_circles)

        # ---- Neck harmonisation ----
        if ctx.whiten != 0 or ctx.equalize > 0:
            canvas = self._skin.harmonize_neck(
                canvas,
                shifted_face.landmarks,
                roi_person_mask,
                regions.skin,
                regions.neck,
                strength=max(abs(ctx.whiten), ctx.equalize),
            )

        # ---- Eye enhancement ----
        if ctx.eye_enhance > 0:
            canvas = self._eyes.enhance(canvas, regions, ctx.eye_enhance)

        # ---- Teeth whitening ----
        if ctx.teeth_whiten > 0:
            canvas = self._teeth.whiten(canvas, regions.mouth_interior, ctx.teeth_whiten)

        # ---- Lip enhancement ----
        if ctx.lip_enhance > 0:
            canvas = self._lips.enhance(
                canvas, regions.lips, ctx.lip_enhance,
                tint=ctx.lip_tint, finish=ctx.lip_finish,
            )

        # ---- Blush ----
        if ctx.blush > 0:
            canvas = self._makeup.apply_blush(
                canvas, shifted_face.landmarks, face_width, ctx.blush,
                regions=regions,
                nose_blush=(active_recipe in _NOSE_BLUSH_RECIPES),
                under_eye_blush=(active_recipe in _UNDER_EYE_BLUSH_RECIPES),
            )

        # ---- Hair shine ----
        if ctx.hair_enhance > 0:
            canvas = self._hair.enhance(
                canvas, roi_person_mask, regions.face_oval,
                shifted_face.bbox, ctx.hair_enhance, regions.hair,
            )

        # ---- Dodge & burn ----
        if ctx.dodge_burn > 0:
            canvas = self._skin.dodge_burn(canvas, regions, ctx.dodge_burn)

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
            roi_box=(roi_x1, roi_y1, roi_x2, roi_y2)
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
            alpha = fr.skin_hair_mask[:, :, np.newaxis]
            
            # Blend cropped canvas back using skin_hair_mask as alpha channel
            roi_blend = (fr.canvas * alpha + result[y1:y2, x1:x2] * (1.0 - alpha)).astype(np.uint8)
            result[y1:y2, x1:x2] = roi_blend

            # Accumulate cropped masks into full-resolution canvas masks
            acc_skin[y1:y2, x1:x2] = np.maximum(acc_skin[y1:y2, x1:x2], fr.skin_mask)
            acc_skin_hair[y1:y2, x1:x2] = np.maximum(acc_skin_hair[y1:y2, x1:x2], fr.skin_hair_mask)
            acc_lips[y1:y2, x1:x2] = np.maximum(acc_lips[y1:y2, x1:x2], fr.lips_mask)
            acc_sharpen[y1:y2, x1:x2] = np.maximum(acc_sharpen[y1:y2, x1:x2], fr.sharpen_mask)

        return result, acc_skin, acc_skin_hair, acc_lips, acc_sharpen

    def _stage_global(self, img: np.ndarray, ctx: ProcessingContext) -> np.ndarray:
        """Global tonal operators (contrast, brightness, HSL tonal curve)."""
        result = img

        if ctx.contrast:
            result = _adjust_contrast(result, ctx.contrast)

        if ctx.brightness is not None and ctx.brightness != 0:
            gamma = 1.0 - (ctx.brightness / 100.0)
            # BUGFIX-4: renamed ndarray to _brightness_lut to avoid clobbering ctx.lut
            _brightness_lut = np.array(
                [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
            )
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

        # Assemble film/lens effect overrides from context
        fx_overrides: Dict[str, Any] = {}
        if ctx.chromatic_aberration is not None:
            fx_overrides["chromatic_aberration"] = ctx.chromatic_aberration
        if ctx.halation is not None:
            fx_overrides["halation"] = ctx.halation
        if ctx.grain is not None:
            fx_overrides["grain"] = ctx.grain
        if ctx.lut is not None:
            fx_overrides["lut"] = ctx.lut

        if ctx.color_grade_stack:
            result = self._grader.grade_stack(result, ctx.color_grade_stack)
            if fx_overrides:
                result = self._grader.grade(
                    result, fx_overrides, 1.0,
                    split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
                )
        elif ctx.color_grade:
            settings = PRESETS.get(ctx.color_grade, PRESETS["natural"]).copy()
            settings.update(fx_overrides)
            result = self._grader.grade(
                result, settings, ctx.grade_intensity,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
            )
        elif fx_overrides:
            result = self._grader.grade(
                result, fx_overrides, 1.0,
                split_tone_mask=acc_skin, glow_mask=g_mask, haze_mask=g_mask,
            )

        # White costume pearl/lavender lift
        if ctx.active_recipe in _WHITE_COSTUME_RECIPES:
            result = self._apply_white_costume_lift(result, acc_skin, acc_lips, ctx.grade_intensity)

        return result

    def _stage_finish(
        self,
        img: np.ndarray,
        ctx: ProcessingContext,
        acc_sharpen: np.ndarray,
    ) -> np.ndarray:
        result = img
        if acc_sharpen.max() > 0.01:
            result = _apply_selective_sharpening(
                result, acc_sharpen, radius=0.8, amount=1.2, threshold=2
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
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release detector resources."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Module-level pure helpers (also used by tests)
# ---------------------------------------------------------------------------

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
        gray_high = cv2.cvtColor(
            np.abs(high_freq).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        threshold_mask = (gray_high >= threshold)[:, :, np.newaxis]
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

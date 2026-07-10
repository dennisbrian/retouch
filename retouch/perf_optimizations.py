"""
RetouchEngine — Corrected Performance Optimization Implementations
=================================================================
Covers all four proposals from performance_optimization_proposals.md
with every bug, warning, and improvement addressed.

Proposals:
  1. Multi-processing to bypass GIL
  2. Downscaled MediaPipe inference
  3. CoreML / Metal GPU ONNX execution providers
  4. Numba JIT pixel loop with prange parallelism
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .frequency import FrequencySeparator
from .freckle import FreckleRemover


def _build_smooth_mask(
    skin: Optional[np.ndarray],
    exclusions: Tuple[Optional[np.ndarray], ...],
    out_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Build the per-face smooth mask by subtracting exclusion masks from skin.

    The smooth mask tells the bilateral/median pipeline which pixels may be
    smoothed. Eyes, brows, lips, under-eyes and hair roots must be excluded
    so retouching never crosses those boundaries.

    Args:
        skin: ``(H, W)`` float32 skin mask, or ``None`` to zero-init.
        exclusions: Tuple of optional ``(H, W)`` masks to subtract. ``None``
            entries are skipped. Non-float dtypes are cast to float32.
        out_shape: Optional ``(H, W)`` shape. Required when ``skin`` is ``None``.

    Returns:
        ``(H, W)`` float32 mask clipped to ``[0, 1]``.
    """
    if skin is not None:
        smooth_mask = skin.copy()
    elif out_shape is not None:
        smooth_mask = np.zeros(out_shape, dtype=np.float32)
    else:
        raise ValueError("Either `skin` or `out_shape` must be provided")

    for excl in exclusions:
        if excl is not None:
            smooth_mask = np.clip(
                smooth_mask - excl.astype(np.float32), 0.0, 1.0
            )

    # Erode by 3px to protect hair/skin boundary pixels that the segmentation
    # model may have misclassified (common with white/gray hair). The erosion
    # ensures a thin safety margin so smoothing never bleeds into hair.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_mask = cv2.erode(smooth_mask, kernel, iterations=1)

    return smooth_mask

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    from numba import prange
    numba_module = numba
else:
    prange = range
    class DummyNumbaModule:
        prange = range
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    numba_module = DummyNumbaModule()
    numba = numba_module

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Mask utilities & per-face result container
# (kept here, next to the per-face core pipeline, so worker processes
#  can import them without dragging the rest of engine.py)
# ──────────────────────────────────────────────────────────────────────────────

def _norm_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a float32 mask in [0, 1]. Returns None if input is None."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return m


def _apply_exposure_lock(
    canvas: np.ndarray,
    pre_smooth_canvas: np.ndarray,
    skin_n: Optional[np.ndarray],
    lock: float,
) -> np.ndarray:
    """Re-center post-smoothing skin luminance toward the pre-smoothing mean.

    Frequency-separation smoothing damps bright micro-speculars and drops mean
    skin luminance, making retouched faces look duller than the source. This
    applies an optional, mask-feathered multiplicative gain that recovers the
    lost brightness — but only compensates *darkening* (a face that got brighter
    is left untouched).

    Args:
        canvas: ``(H, W, 3)`` float32 [0, 255] post-smoothing BGR canvas.
        pre_smooth_canvas: ``(H, W, 3)`` float32 [0, 255] pre-smoothing BGR canvas.
        skin_n: ``(H, W)`` float32 skin mask in [0, 1], or ``None`` to skip.
        lock: Fraction in [0, 1] — 0 is identity, 1 fully recovers the mean.

    Returns:
        ``(H, W, 3)`` float32 [0, 255] canvas.
    """
    lock = lock or 0.0
    if lock <= 0 or skin_n is None:
        return canvas

    skin_sel = skin_n > 0.3
    if not np.any(skin_sel):
        return canvas

    # Rec.601 luma on the float BGR canvas (no LAB round-trip).
    def _luma(img: np.ndarray) -> np.ndarray:
        return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    mean_pre = float(_luma(pre_smooth_canvas)[skin_sel].mean())
    mean_post = float(_luma(canvas)[skin_sel].mean())

    # Only compensate darkening — never darken a face that got brighter.
    if mean_post >= mean_pre:
        return canvas

    gain = 1.0 + lock * (mean_pre - mean_post) / max(mean_post, 1e-6)
    canvas = canvas * (1.0 + (gain - 1.0) * skin_n[:, :, np.newaxis])
    return np.clip(canvas, 0.0, 255.0)


def _accum(acc: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Add normalised mask into accumulator, clamped to 1."""
    if mask is None:
        return acc
    return np.clip(acc + _norm_mask(mask), 0.0, 1.0)


@dataclass
class _FaceResult:
    canvas: np.ndarray
    skin_mask: np.ndarray
    skin_hair_mask: np.ndarray
    lips_mask: np.ndarray
    sharpen_mask: np.ndarray
    roi_box: Tuple[int, int, int, int]


# ──────────────────────────────────────────────────────────────────────────────
# Per-face core pipeline (module-level so it is picklable / callable from
# worker processes without the RetouchEngine instance, which holds non-picklable
# ONNX sessions and MediaPipe tasks).
#
# Lives here (not in engine.py) so that perf_optimizations._process_single_face_worker
# can call it without creating a circular import: engine imports parsing imports
# perf_optimizations, so the worker cannot reach back into engine.
# ──────────────────────────────────────────────────────────────────────────────

# TEMPORARY E1 dtype tracing / quantization bisect (remove before finishing)
_E1_TRACE = os.environ.get("E1_TRACE") == "1"
_E1_QUANT = set(filter(None, os.environ.get("E1_QUANT", "").split(",")))
_E1_ROUND = os.environ.get("E1_ROUND", "trunc")  # trunc|rint


def _e1_to_u8(canvas: np.ndarray) -> np.ndarray:
    """Final float32 [0,255] -> uint8 conversion for the per-face canvas.

    'trunc' matches the legacy per-op convention (`np.clip(x,0,255).astype(uint8)`
    truncated at every op in the uint8 chain), keeping outputs maximally close
    to the pre-E1 baseline. 'rint' is available for comparison.
    """
    c = np.clip(canvas, 0, 255)
    if _E1_ROUND == "rint":
        c = np.rint(c)
    return c.astype(np.uint8)


_E1_U8_FROM = os.environ.get("E1_U8_FROM", "")
_E1_CHAIN_ORDER = ['post_freq', 'flatten', 'restore_micro_texture', 'micro_dodge_burn',
                   'redness_even', 'equalize', 'unify_hue_line', 'unify_tone', 'whiten',
                   'shine_removal', 'relight', 'sculpt', 'quantize_tones',
                   'apply_specular_bloom', 'blemish.remove', 'undereye.repair',
                   'harmonize_neck', 'eyes.enhance', 'teeth.whiten', 'lips.enhance',
                   'makeup.apply_blush', 'hair.enhance', 'dodge_burn', 'wrinkle_soften',
                   'texture_transplant', 'local_clarity']


def _tr(op: str, canvas: np.ndarray) -> np.ndarray:
    if _E1_TRACE:
        print(f"E1TRACE {op} canvas_dtype={canvas.dtype}", flush=True)
    if (op in _E1_QUANT or "all" in _E1_QUANT) and canvas.dtype == np.float32:
        canvas = _e1_to_u8(canvas).astype(np.float32)
    if _E1_U8_FROM and canvas.dtype == np.float32:
        try:
            if _E1_CHAIN_ORDER.index(op) >= _E1_CHAIN_ORDER.index(_E1_U8_FROM):
                canvas = _e1_to_u8(canvas)  # real uint8: legacy path downstream
        except ValueError:
            pass
    return canvas


def _process_face_core(
    canvas: np.ndarray,
    regions: "Any",
    shifted_face: "Any",
    ctx: "Any",
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
    'teeth', 'lips', 'makeup', 'hair', 'frequency'). This lets the same
    logic run either inside the engine (passing ``self._xxx``) or inside
    a worker process (passing freshly-instantiated processors). The
    'frequency' entry is optional: when absent, a transient
    ``FrequencySeparator`` is created for the duration of this call.

    Stage E1: Canvas is converted to float32 [0, 255] at the top and kept
    float throughout the skin operation chain to eliminate quantization noise.
    Single uint8 conversion at the end.
    """
    skin = processors['skin']
    relighter = processors['relighter']
    blemish = processors['blemish']
    undereye = processors['undereye']
    eyes = processors['eyes']
    teeth = processors['teeth']
    lips = processors['lips']
    makeup = processors['makeup']
    makeup_v2 = processors.get('makeup_v2')
    hair = processors['hair']
    frequency = processors.get('frequency') or FrequencySeparator()

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

    # ---- E1: Convert canvas to float32 once at the top ----
    # Float convention: [0, 255] matching OpenCV BGR scale. All skin ops stay
    # float until the final conversion to uint8 at the end.
    canvas = canvas.astype(np.float32)
    # Snapshot before any retouch ops, so nose_restore (below) can blend
    # back toward the camera-original nose bridge shading. User-flagged:
    # smoothing/brightening the surrounding skin can make the source
    # photo's own real nose-bridge shadow read as more sharply defined by
    # contrast, even though no single op deepens it — restoring the
    # original pixels there directly avoids chasing that indirect effect
    # through every op's parameters.
    canvas_original = canvas.copy()

    # ctx may be a dict (pickled across process boundary) — convert to object
    if isinstance(ctx, dict):
        ctx = SimpleNamespace(**ctx)

    # ---- Frequency separation ----
    # Note: frequency.separate expects uint8, so convert temporarily
    canvas_u8_for_freq = np.clip(canvas, 0, 255).astype(np.uint8)
    original_lab = cv2.cvtColor(canvas_u8_for_freq, cv2.COLOR_BGR2LAB)
    # Snapshot pre-smoothing canvas for adaptive micro-texture restoration.
    # The restoration step compares the smoothed result against this original
    # to recover dimensional detail the bilateral+mid_reduction can wash out.
    pre_smooth_canvas = canvas.copy()
    layers = frequency.separate(canvas_u8_for_freq, face_width)

    # ---- Build smooth mask (protect eyes/brows/lips/hair) ----
    smooth_mask = _build_smooth_mask(
        skin_n,
        exclusions=(),
        out_shape=(roi_h, roi_w),
    )

    if shifted_face.ied > 0:
        k_size = max(3, int(shifted_face.ied * 0.08) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated_left_eye = cv2.dilate(regions.left_eye, kernel) if regions.left_eye is not None else None
        dilated_right_eye = cv2.dilate(regions.right_eye, kernel) if regions.right_eye is not None else None
    else:
        dilated_left_eye = regions.left_eye
        dilated_right_eye = regions.right_eye

    smooth_mask = _build_smooth_mask(
        smooth_mask,
        exclusions=(
            dilated_left_eye, dilated_right_eye,
            regions.left_under_eye, regions.right_under_eye,
            regions.left_eyebrow, regions.right_eyebrow,
            regions.lips,
            regions.hair,
        ),
    )

    # ---- Frequency-based smoothing ----
    # E1: frequency.combine now returns float32 to avoid quantization
    if ctx.nose_smooth is not None:
        nose_mask = _norm_mask(regions.nose)
        if nose_mask is not None:
            face_without_nose = np.clip(smooth_mask - nose_mask * smooth_mask, 0.0, 1.0)
            canvas = frequency.combine(
                layers,
                skin_mask=face_without_nose,
                smooth_strength=ctx.smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                blotch_reduction=ctx.blotch_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
                float32_out=True,
                regions=regions,
                regional_modulation=getattr(ctx, 'regional_modulation', 0.0),
                smooth_engine=getattr(ctx, 'smooth_engine', 'guided'),
            )
            nose_canvas = frequency.combine(
                layers,
                skin_mask=nose_mask * smooth_mask,
                smooth_strength=ctx.nose_smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                blotch_reduction=ctx.blotch_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
                float32_out=True,
                regions=regions,
                regional_modulation=getattr(ctx, 'regional_modulation', 0.0),
                smooth_engine=getattr(ctx, 'smooth_engine', 'guided'),
            )
            nose_alpha = (nose_mask * smooth_mask)[:, :, np.newaxis]
            canvas = (
                nose_canvas * nose_alpha
                + canvas * (1.0 - nose_alpha)
            )
        else:
            canvas = frequency.combine(
                layers,
                skin_mask=smooth_mask,
                smooth_strength=ctx.smooth / 100.0,
                mid_reduction=ctx.mid_reduction,
                blotch_reduction=ctx.blotch_reduction,
                texture_opacity=ctx.texture_opacity,
                face_width=face_width,
                pore_synthesis=ctx.pore_synthesis / 100.0,
                roi_coords=(roi_x1, roi_y1),
                float32_out=True,
                regions=regions,
                regional_modulation=getattr(ctx, 'regional_modulation', 0.0),
                smooth_engine=getattr(ctx, 'smooth_engine', 'guided'),
            )
    else:
        canvas = frequency.combine(
            layers,
            skin_mask=smooth_mask,
            smooth_strength=ctx.smooth / 100.0,
            mid_reduction=ctx.mid_reduction,
                blotch_reduction=ctx.blotch_reduction,
            texture_opacity=ctx.texture_opacity,
            face_width=face_width,
            pore_synthesis=ctx.pore_synthesis / 100.0,
            roi_coords=(roi_x1, roi_y1),
            float32_out=True,
            regions=regions,
            regional_modulation=getattr(ctx, 'regional_modulation', 0.0),
            smooth_engine=getattr(ctx, 'smooth_engine', 'guided'),
            )

    # ---- Selective under-eye shadow smoothing (S5.5) ----
    # Conservative, region-limited; no-op when strength <= 0.
    _undereye_strength = getattr(ctx, 'undereye_shadow_strength', 0.0) or 0.0
    if _undereye_strength > 0 and regions is not None:
        canvas = skin.smooth_undereye_shadow(
            canvas,
            under_eye_masks=[regions.left_under_eye, regions.right_under_eye],
            skin_mask=skin_n,
            strength=_undereye_strength,
            feather_radius=3,
        )

    # ---- Selective freckle removal (preserve beauty marks) ----
    # Runs after smoothing so the heal stays crisp; no-op when freckle_removal <= 0.
    _freckle_removal = getattr(ctx, 'freckle_removal', 0.0) or 0.0
    if _freckle_removal > 0:
        canvas = FreckleRemover().remove(
            img_bgr=canvas,
            face_mask=skin_n,
            freckle_removal=_freckle_removal,
            freckle_preserve_mask=getattr(ctx, 'freckle_preserve_mask', None),
        )

    # ---- Exposure-locked smoothing (recipe-only) ----
    # Frequency smoothing damps bright micro-speculars and drops mean skin
    # luminance; re-center it toward the pre-smoothing mean (darkening only),
    # feathered by the skin mask. No-op when smooth_exposure_lock <= 0.
    lock = getattr(ctx, 'smooth_exposure_lock', 0.0) or 0.0
    if lock > 0 and skin_n is not None:
        canvas = _apply_exposure_lock(canvas, pre_smooth_canvas, skin_n, lock)

    canvas = _tr('post_freq', canvas)

    # ---- Edge-preserving cel flatten ----
    if ctx.skin_flatten > 0:
        canvas = _tr('flatten', canvas)
        canvas = skin.flatten(canvas, regions.skin, int(ctx.skin_flatten))

    # ---- Adaptive micro-texture restoration ----
    # Re-injects dimensional micro-contrast in cheek / nose / under-eye zones
    # that the bilateral+mid_reduction step washed out. Modulated by
    # smooth_strength so this is a no-op when smoothing is off.
    if ctx.micro_restore > 0:
        canvas = _tr('restore_micro_texture', canvas)
        canvas = skin.restore_micro_texture(
            canvas,
            pre_smooth_canvas,
            regions,
            strength=ctx.micro_restore,
            smooth_strength=ctx.smooth / 100.0,
        )

    # ---- Micro dodge & burn (blotch evening) ----
    if ctx.micro_dodge_burn > 0:
        canvas = _tr('micro_dodge_burn', canvas)
        canvas = skin.micro_dodge_burn(canvas, _norm_mask(regions.skin), ctx.micro_dodge_burn, face_width)

    # ---- Redness evening (color blotch on a/b channels) ----
    if ctx.redness_even > 0:
        canvas = _tr('redness_even', canvas)
        canvas = skin.redness_even(canvas, _norm_mask(regions.skin), ctx.redness_even, face_width, lips_mask=_norm_mask(regions.lips))

    # ---- Skin equalization ----
    if ctx.equalize > 0:
        canvas = _tr('equalize', canvas)
        canvas = skin.equalize(canvas, regions.skin, ctx.equalize, ref_lab=original_lab)

    # ---- Skin hue-line unification (preferred-locus pull) ----
    if ctx.skin_hue_unify > 0 or ctx.skin_chroma_even > 0:
        canvas = _tr('unify_hue_line', canvas)
        canvas = skin.unify_hue_line(canvas, regions.skin, int(ctx.skin_hue_unify), int(ctx.skin_chroma_even), locus=ctx.skin_locus)

    # ---- Skin hue/chroma unification (anime) ----
    if ctx.skin_unify > 0:
        canvas = _tr('unify_tone', canvas)
        canvas = skin.unify_tone(canvas, regions.skin, int(ctx.skin_unify), target_hue=ctx.skin_unify_hue)

    # ---- Foundation / whitening ----
    if ctx.whiten != 0:
        canvas = _tr('whiten', canvas)
        canvas = skin.whiten(canvas, regions.skin, ctx.whiten, tone=ctx.whiten_tone, hue_stable=bool(ctx.whiten_hue_stable))

    # ---- Shine / oil removal (before relight so intentional glow isn't removed) ----
    if ctx.shine_removal > 0:
        canvas = _tr('shine_removal', canvas)
        eyes_mask = np.maximum(
            _norm_mask(regions.left_eye) if regions.left_eye is not None else np.zeros((roi_h, roi_w), dtype=np.float32),
            _norm_mask(regions.right_eye) if regions.right_eye is not None else np.zeros((roi_h, roi_w), dtype=np.float32),
        )
        canvas = skin.shine_removal(canvas, regions.skin, int(ctx.shine_removal), eyes_mask=eyes_mask)

    # ---- Virtual studio relighting ----
    if ctx.relight > 0:
        canvas = _tr('relight', canvas)
        canvas = relighter.relight(
            canvas,
            shifted_face.landmarks,
            regions.skin,
            face_width=face_width,
            strength=ctx.relight,
            azimuth=ctx.relight_azimuth,
            elevation=ctx.relight_elevation,
        )

    # ---- Face exposure lift (flat skin-L luminance, beyond relight ceiling) ----
    if ctx.face_exposure > 0:
        canvas = _tr('face_exposure', canvas)
        canvas = skin.face_exposure_lift(canvas, regions.skin, ctx.face_exposure)

    # ---- Facial structure sculpting (C2: low-band shaping, form-frequency modulation) ----
    if ctx.sculpt > 0:
        canvas = _tr('sculpt', canvas)
        canvas = relighter.sculpt(
            canvas,
            shifted_face.landmarks,
            regions.skin,
            face_width=face_width,
            strength=ctx.sculpt,
        )

    # ---- Tone quantization (cel shading bands) ----
    if ctx.skin_quantize > 0:
        canvas = _tr('quantize_tones', canvas)
        canvas = skin.quantize_tones(canvas, regions.skin, int(ctx.skin_quantize))

    # ---- Specular bloom ----
    if ctx.specular_bloom > 0:
        canvas = _tr('apply_specular_bloom', canvas)
        canvas = skin.apply_specular_bloom(
            canvas, regions.skin, ctx.specular_bloom, tone=ctx.specular_bloom_tone
        )

    # ---- Blemish removal ----
    if ctx.blemish > 0:
        canvas = _tr('blemish.remove', canvas)
        canvas = blemish.remove(canvas, regions.skin, ctx.blemish)

    # ---- Under-eye repair ----
    if ctx.dark_circles > 0 or ctx.undereye_darken_removal > 0 or ctx.undereye_puffiness_reduction > 0:
        canvas = _tr('undereye.repair', canvas)
        # Legacy interface for backward compatibility
        if ctx.dark_circles > 0:
            canvas = undereye.repair(canvas, regions, ctx.dark_circles)
        # New advanced processing pipeline
        if ctx.undereye_darken_removal > 0 or ctx.undereye_puffiness_reduction > 0:
            s_darken = ctx.undereye_darken_removal / 100.0
            s_puffiness = ctx.undereye_puffiness_reduction / 100.0
            for mask in (regions.left_under_eye, regions.right_under_eye):
                if mask is not None and mask.max() > 0.01:
                    canvas = undereye._processor.process(
                        canvas, mask,
                        darken_removal_strength=s_darken,
                        puffiness_reduction_strength=s_puffiness
                    )

    # ---- Neck harmonisation ----
    if ctx.whiten != 0 or ctx.equalize > 0 or ctx.skin_hue_unify > 0 or ctx.skin_chroma_even > 0 or ctx.redness_even > 0:
        canvas = _tr('harmonize_neck', canvas)
        canvas = skin.harmonize_neck(
            canvas,
            shifted_face.landmarks,
            roi_person_mask,
            regions.skin,
            regions.neck,
            strength=max(abs(ctx.whiten), ctx.equalize, ctx.skin_hue_unify, ctx.skin_chroma_even, ctx.redness_even),
        )

    # ---- Eye enhancement ----
    if ctx.eye_enhance > 0 or ctx.eye_sclera_vessel_remove > 0:
        canvas = _tr('eyes.enhance', canvas)
        canvas = eyes.enhance(canvas, regions, ctx.eye_enhance,
                              catchlight_strength=ctx.catchlight if ctx.catchlight > 0 else None,
                              vessel_strength=ctx.eye_sclera_vessel_remove)

    # ---- Eye Enhancement v0 (sclera brightening + iris saturation/hue/brightness) ----
    eye_v0_active = (ctx.eye_sclera_brighten > 0 or ctx.eye_iris_saturate > 0 or
                     ctx.eye_iris_hue_shift != 0 or ctx.eye_iris_brightness > 0)
    if eye_v0_active:
        canvas = _tr('eye_enhancement.enhance', canvas)
        from .eye_enhancement import EyeEnhancer as EyeEnhancerV0
        eye_enhancer_v0 = EyeEnhancerV0()
        canvas_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
        canvas_uint8 = eye_enhancer_v0.enhance(
            canvas_uint8, regions,
            sclera_brighten=ctx.eye_sclera_brighten,
            iris_saturate=ctx.eye_iris_saturate,
            iris_hue_shift=ctx.eye_iris_hue_shift,
            iris_brightness=ctx.eye_iris_brightness,
        )
        canvas = canvas_uint8.astype(np.float32)

    # ---- Teeth whitening ----
    if ctx.teeth_whiten > 0:
        canvas = _tr('teeth.whiten', canvas)
        canvas = teeth.whiten(canvas, regions.mouth_interior, ctx.teeth_whiten)

    # ---- Lip enhancement ----
    if ctx.lip_enhance > 0:
        canvas = _tr('lips.enhance', canvas)
        canvas = lips.enhance(
            canvas, regions.lips, ctx.lip_enhance,
            tint=ctx.lip_tint, finish=ctx.lip_finish,
        )

    # ---- Blush ----
    if ctx.blush > 0:
        canvas = _tr('makeup.apply_blush', canvas)
        canvas = makeup.apply_blush(
            canvas, shifted_face.landmarks, face_width, ctx.blush,
            regions=regions,
            nose_blush=ctx.nose_blush,
            under_eye_blush=ctx.under_eye_blush,
        )

    # ---- Makeup v2 (eyeshadow, eyeliner, contour, brows, ombre lips) ----
    if makeup_v2 is not None:
        # Convert canvas to uint8 for makeup_v2 operations
        canvas_u8_for_makeup = np.clip(canvas, 0, 255).astype(np.uint8)

        # Eyeshadow
        if ctx.mv2_eyeshadow > 0:
            canvas = _tr('makeup_v2.apply_eyeshadow', canvas)
            canvas_u8_for_makeup = makeup_v2.apply_eyeshadow(
                canvas_u8_for_makeup, shifted_face.landmarks,
                color=ctx.mv2_eyeshadow_color,
                strength=ctx.mv2_eyeshadow,
                style=ctx.mv2_eyeshadow_style,
            )

        # Eyeliner
        if ctx.mv2_eyeliner > 0:
            canvas = _tr('makeup_v2.apply_eyeliner', canvas)
            canvas_u8_for_makeup = makeup_v2.apply_eyeliner(
                canvas_u8_for_makeup, shifted_face.landmarks,
                color=ctx.mv2_eyeliner_color,
                thickness=ctx.mv2_eyeliner,
                style=ctx.mv2_eyeliner_style,
            )

        # Contour
        if ctx.mv2_contour > 0:
            canvas = _tr('makeup_v2.apply_contour', canvas)
            canvas_u8_for_makeup = makeup_v2.apply_contour(
                canvas_u8_for_makeup, shifted_face.landmarks,
                strength=ctx.mv2_contour,
            )

        # Brows
        if ctx.mv2_brows > 0:
            canvas = _tr('makeup_v2.apply_brows', canvas)
            canvas_u8_for_makeup = makeup_v2.apply_brows(
                canvas_u8_for_makeup, shifted_face.landmarks,
                color=ctx.mv2_brows_color,
                thickness=ctx.mv2_brows,
            )

        # Ombre lips
        if ctx.mv2_ombre:
            canvas = _tr('makeup_v2.apply_ombre_lips', canvas)
            canvas_u8_for_makeup = makeup_v2.apply_ombre_lips(
                canvas_u8_for_makeup, shifted_face.landmarks,
                color1=ctx.mv2_ombre_color1,
                color2=ctx.mv2_ombre_color2,
            )

        # Convert back to float32 if any makeup_v2 ops were applied
        if ctx.mv2_eyeshadow > 0 or ctx.mv2_eyeliner > 0 or ctx.mv2_contour > 0 or ctx.mv2_brows > 0 or ctx.mv2_ombre:
            canvas = canvas_u8_for_makeup.astype(np.float32)

    # ---- Hair shine (H2: deglare + anisotropic angel ring) ----
    # hair_enhance is reinterpreted as the angel-ring strength; hair_deglare
    # is the synthetic-glare compression strength. Both are independent and
    # steered by the H0 strand flow field. The legacy HairEnhancer.enhance
    # isotropic path is kept as a fallback when no hair mask is available.
    hair_deglare_v = getattr(ctx, 'hair_deglare', 0) or 0
    hair_ring_strength = ctx.hair_enhance
    hair_flyaway_v = getattr(ctx, 'hair_remove_flyaways', 0) or 0
    if hair_deglare_v > 0 or hair_ring_strength > 0 or hair_flyaway_v > 0:
        if regions.hair is not None and _norm_mask(regions.hair).max() > 0.01:
            from .hairwork import hair_flow, deglare_wig, add_angel_ring, remove_flyaways
            canvas_u8_for_flow = (np.clip(canvas, 0, 255).astype(np.uint8)
                                 if canvas.dtype != np.uint8 else canvas)
            orientation, coherence = hair_flow(
                canvas_u8_for_flow, hair_mask=_norm_mask(regions.hair)
            )
            # H1 — flyaway removal runs first so the cleaned silhouette feeds
            # the H2 deglare/ring stages (and the flow field recomputed for
            # those stages stays clean). Eyebrow/eyelash exclusion mirrors
            # the deglare guard.
            if hair_flyaway_v > 0:
                canvas = _tr('hair.remove_flyaways', canvas)
                eb = None
                if regions.left_eyebrow is not None or regions.right_eyebrow is not None:
                    eb = np.zeros((roi_h, roi_w), dtype=np.float32)
                    if regions.left_eyebrow is not None:
                        eb = np.clip(eb + _norm_mask(regions.left_eyebrow), 0, 1)
                    if regions.right_eyebrow is not None:
                        eb = np.clip(eb + _norm_mask(regions.right_eyebrow), 0, 1)
                skin_m = _norm_mask(regions.skin) if regions.skin is not None else None
                canvas = remove_flyaways(
                    canvas, _norm_mask(regions.hair), orientation, coherence,
                    strength=int(hair_flyaway_v), face_width=face_width,
                    eyebrow_mask=eb, skin_mask=skin_m,
                )
            if hair_deglare_v > 0:
                canvas = _tr('hair.deglare', canvas)
                eb = None
                if regions.left_eyebrow is not None or regions.right_eyebrow is not None:
                    eb = np.zeros((roi_h, roi_w), dtype=np.float32)
                    if regions.left_eyebrow is not None:
                        eb = np.clip(eb + _norm_mask(regions.left_eyebrow), 0, 1)
                    if regions.right_eyebrow is not None:
                        eb = np.clip(eb + _norm_mask(regions.right_eyebrow), 0, 1)
                canvas = deglare_wig(
                    canvas, _norm_mask(regions.hair), orientation, coherence,
                    strength=int(hair_deglare_v), face_width=face_width,
                    eyebrow_mask=eb,
                )
            if hair_ring_strength > 0:
                canvas = _tr('hair.ring', canvas)
                ring_pos = int(getattr(ctx, 'hair_ring_position', 30) or 30)
                ring_tint = int(getattr(ctx, 'hair_ring_tint', 40) or 40)
                canvas = add_angel_ring(
                    canvas, _norm_mask(regions.hair), orientation, coherence,
                    strength=int(hair_ring_strength), position=ring_pos,
                    tint=ring_tint, face_width=face_width,
                    landmarks=shifted_face.landmarks,
                )
        else:
            canvas = _tr('hair.enhance', canvas)
            canvas = hair.enhance(
                canvas, roi_person_mask, regions.face_oval,
                shifted_face.bbox, hair_ring_strength, regions.hair,
            )

    # ---- Dodge & burn ----
    if ctx.dodge_burn > 0:
        canvas = _tr('dodge_burn', canvas)
        canvas = skin.dodge_burn(canvas, regions, ctx.dodge_burn)

    # ---- Shadow lift (opt-in targeted fill-light, see shadow_lift.py) ----
    # Real photographic shade (e.g. a jaw shadow next to a bright prop, or
    # hair passing in front of a dark background) is not a retouch defect,
    # so this only runs if explicitly enabled. Scope includes regions.hair
    # alongside regions.skin so genuine hair-lighting falloff (e.g. light
    # hair darkening where it crosses a dark backdrop) gets the same
    # gentle local lift as skin, instead of being left untouched.
    if ctx.shadow_lift > 0:
        from .shadow_lift import ShadowLifter
        canvas = _tr('shadow_lift', canvas)
        shadow_lifter = ShadowLifter()
        lift_mask = regions.skin
        if regions.hair is not None:
            lift_mask = np.clip(_norm_mask(regions.skin) + _norm_mask(regions.hair), 0, 1)
        canvas = shadow_lifter.lift(canvas, lift_mask, strength=ctx.shadow_lift)

    # ---- Wrinkle & line softening ----
    region_strengths = {
        "forehead": ctx.wrinkle_soften_forehead,
        "nasolabial": ctx.wrinkle_soften_nasolabial,
        "neck": ctx.wrinkle_soften_neck,
    }
    if ctx.wrinkle_soften > 0 or any(v > 0 for v in region_strengths.values()):
        canvas = _tr('wrinkle_soften', canvas)
        canvas = skin.wrinkle_soften(
            canvas, regions, ctx.wrinkle_soften, region_strengths=region_strengths
        )

    # ---- Texture transplant (pore realism v2) ----
    if ctx.texture_transplant > 0:
        canvas = _tr('texture_transplant', canvas)
        canvas = skin.texture_transplant(
            canvas, regions.skin, ctx.texture_transplant, face_width=face_width
        )

    # ---- Local clarity (nose/lips/eyes pop) ----
    if ctx.clarity > 0:
        canvas = _tr('local_clarity', canvas)
        # F8.2: radius scales with face_width so the high-pass baseband is at
        # the same physical feature scale at native res as at proxy res.
        # 20px was the original fixed value at ~500px proxy face width.
        clarity_radius = max(8, int(face_width * 0.04)) | 1
        canvas = skin.local_clarity(
            canvas, regions,
            strength=ctx.clarity / 100.0 * 0.20,
            radius=clarity_radius,
        )

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

    # ---- Nose restore (opt-in) ----
    # Blends the nose region back toward the pre-retouch original, so an
    # already-good real nose shadow isn't left reading over-defined against
    # now-smoother surrounding skin. See canvas_original above. Uses
    # regions.nose (full nose landmark polygon), not regions.nose_bridge
    # (a near-empty 4-point sliver meant for dodge_burn's thin highlight
    # strip, not broad enough to cover the actual shadow area).
    nose_restore = getattr(ctx, 'nose_restore', 0) or 0
    if nose_restore > 0 and regions.nose is not None:
        blend_amount = np.clip(nose_restore / 100.0, 0.0, 1.0) * _norm_mask(regions.nose)
        canvas = canvas * (1.0 - blend_amount[:, :, None]) + canvas_original * blend_amount[:, :, None]

    # ---- E1: Convert canvas back to uint8 once at the end ----
    if canvas.dtype == np.float32:
        canvas = _e1_to_u8(canvas)

    return _FaceResult(
        canvas=canvas,
        skin_mask=acc_skin,
        skin_hair_mask=acc_skin_hair,
        lips_mask=acc_lips,
        sharpen_mask=acc_sharpen,
        roi_box=(roi_x1, roi_y1, roi_x1 + roi_w, roi_y1 + roi_h)
    )


# ──────────────────────────────────────────────────────────────────────────────
# PROPOSAL 1 — Multi-processing (GIL bypass)
# ──────────────────────────────────────────────────────────────────────────────

# Per-worker-process processor cache. Created lazily on first face processed
# inside a child process, then reused for subsequent faces. These processors
# are cheap (no ONNX models); only FaceParser loads ONNX and we never need it
# here because region masks are pre-computed in the parent process.
_WORKER_PROCESSORS: dict[str, Any] | None = None


def _get_worker_processors() -> dict[str, Any]:
    """Lazily build and cache the per-face processors in a worker process."""
    global _WORKER_PROCESSORS
    if _WORKER_PROCESSORS is None:
        from .skin import SkinProcessor
        from .blemish import BlemishRemover
        from .eyes import EyeEnhancer
        from .undereye import UnderEyeRepairer
        from .lips import LipEnhancer
        from .teeth import TeethWhitener
        from .makeup import MakeupEngine
        from .makeup_v2 import MakeupEngineV2
        from .hair import HairEnhancer
        from .relight import Relighter
        _WORKER_PROCESSORS = {
            "skin": SkinProcessor(),
            "relighter": Relighter(),
            "blemish": BlemishRemover(),
            "undereye": UnderEyeRepairer(),
            "eyes": EyeEnhancer(),
            "teeth": TeethWhitener(),
            "lips": LipEnhancer(),
            "makeup": MakeupEngine(),
            "makeup_v2": MakeupEngineV2(),
            "hair": HairEnhancer(),
        }
    return _WORKER_PROCESSORS


def _process_single_face_worker(payload: tuple) -> Dict[str, Any]:
    """Process a single face crop inside a child process.

    ``payload`` is a picklable tuple:
        (canvas, regions, shifted_bbox, shifted_landmarks, ied, ctx,
         roi_box, roi_person_mask, roi_h, roi_w)

    All inputs are plain picklable Python objects (NumPy arrays, the
    ``FaceRegions`` slots-object, the ``ProcessingContext`` dataclass,
    the landmark compat object, floats/ints). No MediaPipe or ONNX
    session objects cross the process boundary.

    Returns a plain dict of NumPy arrays so the result is trivially
    picklable for IPC back to the parent process.
    """
    (
        canvas,
        regions,
        shifted_bbox,
        shifted_landmarks,
        ied,
        ctx,
        roi_box,
        roi_person_mask,
        roi_h,
        roi_w,
    ) = payload

    from .detection import FaceData

    processors = _get_worker_processors()
    shifted_face = FaceData(
        bbox=shifted_bbox,
        landmarks=shifted_landmarks,
        ied=ied,
    )
    roi_x1, roi_y1 = roi_box[0], roi_box[1]

    fr = _process_face_core(
        canvas,
        regions,
        shifted_face,
        ctx,
        roi_x1,
        roi_y1,
        roi_h,
        roi_w,
        roi_person_mask,
        processors,
    )

    return {
        "canvas": fr.canvas,
        "skin_mask": fr.skin_mask,
        "skin_hair_mask": fr.skin_hair_mask,
        "lips_mask": fr.lips_mask,
        "sharpen_mask": fr.sharpen_mask,
        "roi_box": fr.roi_box,
    }


class FaceProcessorPool:
    """
    Manages a persistent ProcessPoolExecutor for per-face parallel processing.

    Initialize once when RetouchEngine starts up; reuse across all calls.
    Avoids per-call process spawning overhead (~200–400 ms per process).

    Usage:
        pool = FaceProcessorPool()          # engine __init__
        results = pool.process(faces_data)  # each pipeline call
        pool.shutdown()                     # engine teardown / context exit
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._ctx = mp.get_context("spawn")
        self._max_workers = max_workers or 4
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "FaceProcessorPool":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=self._ctx,
            )
            logger.info(
                "FaceProcessorPool started with %d workers", self._max_workers
            )

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def process_faces(
        self,
        payloads: List[Tuple[Any, ...]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process all face crops in parallel. Falls back to sequential on error.

        Each entry in *payloads* is the picklable tuple expected by
        ``_process_single_face_worker``. Returns a list of result dicts (or
        ``None`` for a face whose worker failed) in the same order as input.
        The caller is responsible for reconstructing ``_FaceResult`` objects
        and for falling back to in-process processing when an entry is
        ``None``.
        """
        n = len(payloads)

        # ── fast path: single face — no IPC overhead ─────────────────────────
        if n == 1:
            logger.debug("Single face: bypassing IPC, processing inline")
            return [_process_single_face_worker(payloads[0])]

        # ── multi-face: dispatch to worker pool ───────────────────────────────
        if self._executor is None:
            self.start()

        assert self._executor is not None, "Call .start() or use as context manager"

        processed_crops: list[dict | None] = [None] * n

        futures = {
            self._executor.submit(_process_single_face_worker, payload): i
            for i, payload in enumerate(payloads)
        }

        for future in as_completed(futures):
            face_idx = futures[future]
            try:
                processed_crops[face_idx] = future.result()
            except Exception:
                logger.exception(
                    "Worker failed for face %d; returning None for caller fallback",
                    face_idx,
                )
                processed_crops[face_idx] = None

        return processed_crops


# ──────────────────────────────────────────────────────────────────────────────
# PROPOSAL 3 — CoreML / Metal GPU ONNX execution providers
# ──────────────────────────────────────────────────────────────────────────────

def build_ort_providers() -> list[str | tuple[str, dict]]:
    """
    Build an ordered list of ONNX Runtime execution providers.
    Priority: ANE/GPU > CUDA > DirectML > CPU.
    """
    available = ort.get_available_providers()
    providers: list[str | tuple[str, dict]] = []

    # ── Apple Silicon: Core ML (Neural Engine + GPU) ──────────────────────────
    if "CoreMLExecutionProvider" in available:
        providers.append((
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "0",
            },
        ))
        logger.info(
            "CoreML provider enabled (MLProgram, ALL compute units)."
        )

    # ── NVIDIA: CUDA ──────────────────────────────────────────────────────────
    elif "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 ** 3,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        ))
        logger.info("CUDA provider enabled on device 0")

    # ── Windows / DirectML (AMD, Intel, Qualcomm) ─────────────────────────────
    elif "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        logger.info("DirectML provider enabled")

    else:
        logger.info(
            "No hardware acceleration provider available; using CPU."
        )

    providers.append("CPUExecutionProvider")
    return providers




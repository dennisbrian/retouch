"""F7 — AI denoise + super-resolution.

Two-stage enhancement:
    * ``denoise(img, strength)`` — edge-preserving noise reduction. Uses an
      ONNX denoise model (e.g. NAFNet/SCUNet) when available, falling back to
      a guided-bilateral filter that preserves pore texture.
    * ``super_resolve(img, scale)`` — AI upscaling via Real-ESRGAN ONNX when
      available, falling back to Lanczos resampling.

Design notes
------------
* All pixel arithmetic is float32; uint8 is only used at the public boundary
  when the caller hands us a uint8 image (we mirror the input dtype on the
  way out so callers can mix us into either pipeline).
* ONNX sessions are loaded lazily and cached per-process behind a lock — the
  Gradio UI shares one ``AIEnhancer`` across worker threads.
* Inference is tiled (512 px, 32 px overlap, feathered merge) so 4K inputs do
  not OOM the GPU/CPU. Tile size scales with ``KERNEL_SCALE`` when set.
* Model availability is checked via ``model_fetch.model_exists()``; the
  placeholder manifest entry (``sr_real_esrgan``) has an empty sha256 so
  ``model_exists`` returns False until the real file is dropped in. A
  separate ``denoise`` manifest entry is consulted the same way.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from . import model_fetch
from .perf_optimizations import build_ort_providers

logger = logging.getLogger(__name__)

_DENOISE_MODEL_NAME = "denoise_nafnet"
_SR_MODEL_NAME = "sr_real_esrgan"

_DEFAULT_TILE = 512
_DEFAULT_OVERLAP = 32
_SR_NATIVE_SCALE = 4

logger = logging.getLogger(__name__)


def _to_f32(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32)


def _from_f32(img: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    if target_dtype == np.float32:
        return np.clip(img, 0.0, 255.0).astype(np.float32)
    return np.clip(img, 0.0, 255.0).astype(target_dtype)


def _tile_grid(h: int, w: int, tile: int, overlap: int) -> list[list[int]]:
    """Return [[x0,x1,y0,y1], ...] tile bounds covering the image with overlap.

    Adjacent tiles overlap by ``overlap`` px so a feathered merge has no seam.
    The grid is computed so every tile fits inside the image (no padding) —
    the last tile in each axis is anchored to the far edge.
    """
    if h <= tile and w <= tile:
        return [[0, w, 0, h]]
    tiles: list[list[int]] = []
    step = max(1, tile - overlap)
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not xs or xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if not ys or ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    seen: set[tuple[int, int]] = set()
    for y0 in ys:
        for x0 in xs:
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            tiles.append([x0, x0 + tile, y0, y0 + tile])
    return tiles


def _feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """Linear-ramp feather mask for one tile of shape (h, w, 1)."""
    if overlap <= 0 or (h <= overlap and w <= overlap):
        return np.ones((h, w, 1), dtype=np.float32)
    wx = np.ones((w,), dtype=np.float32)
    wy = np.ones((h,), dtype=np.float32)
    if w > overlap:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        wx[:overlap] = ramp
        wx[-overlap:] = ramp[::-1]
    if h > overlap:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        wy[:overlap] = ramp
        wy[-overlap:] = ramp[::-1]
    return (wy[:, None, None] * wx[None, :, None]).astype(np.float32)


class AIEnhancer:
    """AI denoise + super-resolution with ONNX models and classical fallbacks.

    Thread-safe: a single instance is safe to share across Gradio worker
    threads. ONNX sessions are loaded lazily on first use and cached for the
    life of the instance.
    """

    def __init__(
        self,
        tile: int = _DEFAULT_TILE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        scale = os.environ.get("KERNEL_SCALE", "")
        if scale:
            try:
                factor = float(scale)
                tile = max(128, int(tile * factor))
                overlap = max(8, int(overlap * factor))
            except ValueError:
                pass
        self._tile = tile
        self._overlap = overlap
        self._denoise_sess: Optional[object] = None
        self._sr_sess: Optional[object] = None
        self._denoise_lock = threading.Lock()
        self._sr_lock = threading.Lock()
        self._denoise_loaded = False
        self._sr_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_session(self, model_name: str) -> Optional[object]:
        """Return an ONNX InferenceSession for ``model_name`` or ``None``.

        Returns ``None`` (not an exception) when the model file is absent —
        callers fall back to a classical method. Any other failure is logged
        and also returns ``None`` so the pipeline never hard-fails on a
        broken model.
        """
        try:
            if not model_fetch.model_exists(model_name):
                return None
            path = model_fetch.get_model_path(model_name)
        except model_fetch.ModelFetchError as e:
            logger.warning("AIEnhancer: model %s unavailable (%s); using fallback.", model_name, e)
            return None
        try:
            import onnxruntime as ort
        except ImportError as e:
            logger.warning("AIEnhancer: onnxruntime missing (%s); using fallback.", e)
            return None
        try:
            providers = build_ort_providers()
            sess = ort.InferenceSession(path, providers=providers)
            logger.info(
                "AIEnhancer: %s loaded with providers %s", model_name, sess.get_providers()
            )
            return sess
        except Exception as e:
            logger.warning("AIEnhancer: %s session init failed (%s); using fallback.", model_name, e)
            try:
                sess = ort.InferenceSession(path)
                logger.info("AIEnhancer: %s loaded (default providers).", model_name)
                return sess
            except Exception as fallback_err:
                logger.error("AIEnhancer: %s fallback init failed: %s", model_name, fallback_err)
                return None

    def _denoise_model(self) -> Optional[object]:
        if self._denoise_loaded:
            return self._denoise_sess
        with self._denoise_lock:
            if self._denoise_loaded:
                return self._denoise_sess
            self._denoise_sess = self._load_session(_DENOISE_MODEL_NAME)
            self._denoise_loaded = True
            return self._denoise_sess

    def _sr_model(self) -> Optional[object]:
        if self._sr_loaded:
            return self._sr_sess
        with self._sr_lock:
            if self._sr_loaded:
                return self._sr_sess
            self._sr_sess = self._load_session(_SR_MODEL_NAME)
            self._sr_loaded = True
            return self._sr_sess

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def _tiled_inference(self, run: callable, img: np.ndarray, out_scale: int) -> np.ndarray:
        """Run ``run(tile_f32) -> tile_f32`` over the image with feathered merge.

        ``out_scale`` is the spatial upscaling factor applied by ``run`` (1 for
        denoise, 4 for Real-ESRGAN). The output has shape
        ``(H*out_scale, W*out_scale, C)``.
        """
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        oh, ow = h * out_scale, w * out_scale
        out = np.zeros((oh, ow, c), dtype=np.float32)
        acc = np.zeros((oh, ow, c), dtype=np.float32)

        tiles = _tile_grid(h, w, self._tile, self._overlap)
        for x0, x1, y0, y1 in tiles:
            tile = img[y0:y1, x0:x1, :]
            result = run(tile)
            th, tw = result.shape[:2]
            ox0, oy0 = x0 * out_scale, y0 * out_scale
            weight = _feather_weight(th, tw, self._overlap * out_scale)
            out[oy0:oy0 + th, ox0:ox0 + tw, :] += result * weight
            acc[oy0:oy0 + th, ox0:ox0 + tw, :] += weight
        acc = np.maximum(acc, 1e-6)
        return out / acc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Edge-preserving denoise. ``strength`` ∈ [0, 1]; 0 is a no-op.

        Accepts float32 or uint8 BGR; returns the same dtype.
        """
        if strength <= 0.0:
            return img
        strength = float(min(max(strength, 0.0), 1.0))
        in_dtype = img.dtype
        f = _to_f32(img)

        sess = self._denoise_model()
        if sess is not None:
            try:
                denoised = self._run_denoise_model(sess, f)
            except Exception as e:
                logger.warning("AIEnhancer.denoise: model inference failed (%s); fallback.", e)
                denoised = self._bilateral_denoise(f, strength)
        else:
            denoised = self._bilateral_denoise(f, strength)

        out = f * (1.0 - strength) + denoised * strength
        return _from_f32(out, in_dtype)

    def super_resolve(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        """Super-resolution upscale by ``scale`` (2 or 4).

        Accepts float32 or uint8 BGR; returns the same dtype. When the ONNX
        Real-ESRGAN model is unavailable, falls back to Lanczos resampling.
        """
        if scale <= 1:
            return img
        scale = int(scale)
        in_dtype = img.dtype
        f = _to_f32(img)

        sess = self._sr_model()
        if sess is not None:
            try:
                return _from_f32(self._run_sr_model(sess, f, scale), in_dtype)
            except Exception as e:
                logger.warning("AIEnhancer.super_resolve: model inference failed (%s); fallback.", e)
        return _from_f32(self._lanczos_upscale(f, scale), in_dtype)

    def enhance(
        self,
        img: np.ndarray,
        denoise_strength: float = 0.5,
        sr_scale: int = 2,
    ) -> np.ndarray:
        """Combined denoise + super-resolution (denoise runs first)."""
        out = self.denoise(img, denoise_strength)
        return self.super_resolve(out, sr_scale)

    # ------------------------------------------------------------------
    # Inference + fallback implementations
    # ------------------------------------------------------------------

    def _run_denoise_model(self, sess: object, img: np.ndarray) -> np.ndarray:
        """Run the ONNX denoise model with tiled inference.

        Most published denoise models (NAFNet, SCUNet, DnCNN) accept a
        float32 NCHW tensor in [0, 1] and return the same shape. We normalise
        to [0, 1], run tiled inference over HWC tiles, then map back to
        [0, 255].
        """
        hwc = (img.astype(np.float32) / 255.0)
        input_name = self._denoise_input_name(sess)

        def _run(tile01: np.ndarray) -> np.ndarray:
            t = np.transpose(tile01, (2, 0, 1))[None, ...].astype(np.float32)
            t = np.ascontiguousarray(t)
            out = sess.run(None, {input_name: t})[0]
            out = np.transpose(out[0], (1, 2, 0))
            return out.astype(np.float32)

        out01 = self._tiled_inference(_run, hwc, out_scale=1)
        return np.clip(out01 * 255.0, 0.0, 255.0).astype(np.float32)

    def _run_sr_model(self, sess: object, img: np.ndarray, scale: int) -> np.ndarray:
        """Run the ONNX Real-ESRGAN model with tiled inference.

        Real-ESRGAN takes float32 RGB NCHW [0, 1] and returns RGB upscaled by
        the model's native factor (4x for ``real_esrgan_x4.onnx``). If the
        requested ``scale`` differs from the native factor, we resize the
        model output to the requested scale.
        """
        bgr = img.astype(np.float32)
        rgb = bgr[..., ::-1].astype(np.float32) / 255.0

        def _run(tile01: np.ndarray) -> np.ndarray:
            t = np.transpose(tile01, (2, 0, 1))[None, ...].astype(np.float32)
            t = np.ascontiguousarray(t)
            out = sess.run(None, {self._sr_input_name(sess): t})[0]
            out = np.transpose(out[0], (1, 2, 0))
            return out.astype(np.float32)

        out01 = self._tiled_inference(_run, rgb, out_scale=_SR_NATIVE_SCALE)
        if out01.shape[0] != img.shape[0] * scale:
            target = (img.shape[1] * scale, img.shape[0] * scale)
            out01 = cv2.resize(out01, target, interpolation=cv2.INTER_CUBIC)
        bgr_out = out01[..., ::-1] * 255.0
        return np.clip(bgr_out, 0.0, 255.0).astype(np.float32)

    def _bilateral_denoise(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Edge-preserving fallback: bilateral filter sized by ``strength``.

        This is NOT a Gaussian blur on skin — bilateral preserves edges and
        pore-scale texture. Diameter and sigma scale with strength so the
        fallback degrades gracefully toward the no-op case.
        """
        d = int(5 + round(strength * 6.0))
        if d % 2 == 0:
            d += 1
        sigma_color = 25.0 + strength * 50.0
        sigma_space = 25.0 + strength * 50.0
        if img.dtype != np.uint8:
            work = np.clip(img, 0.0, 255.0).astype(np.uint8)
        else:
            work = img
        out = cv2.bilateralFilter(work, d, sigma_color, sigma_space)
        return out.astype(np.float32)

    def _lanczos_upscale(self, img: np.ndarray, scale: int) -> np.ndarray:
        h, w = img.shape[:2]
        out = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # ONNX input-name helpers
    # ------------------------------------------------------------------

    def _denoise_input_name(self, sess: object) -> str:
        try:
            return sess.get_inputs()[0].name
        except Exception:
            return "input"

    def _sr_input_name(self, sess: object) -> str:
        try:
            return sess.get_inputs()[0].name
        except Exception:
            return "input"

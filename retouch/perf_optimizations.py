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
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

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
# Shared data structures (must be importable by worker processes)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegionMasks:
    """
    Holds per-face region mask coordinates as plain Python / NumPy objects
    so they are always picklable for IPC.

    All coordinate arrays are in pixel space relative to the crop canvas,
    NOT normalized floats.
    """
    skin: np.ndarray | None = None        # uint8 mask, same HxW as canvas
    lips: np.ndarray | None = None
    eyes: np.ndarray | None = None
    # Add more regions as the engine grows.

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dict of (possibly None) NumPy arrays.
        NumPy arrays are natively picklable — no special handling needed.
        """
        return {
            "skin": self.skin,
            "lips": self.lips,
            "eyes": self.eyes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegionMasks":
        return cls(skin=d["skin"], lips=d["lips"], eyes=d["eyes"])


@dataclass
class FaceData:
    """
    Stores landmarks already remapped to original-resolution pixel coordinates.

    IMPORTANT: pixel_landmarks stores (x, y) in the original image's pixel
    space. Normalized [0,1] values from MediaPipe are converted at
    construction time so that no call site can accidentally use the wrong
    scale.
    """
    # Shape: (N, 2) array of (x_px, y_orig_px) coordinates
    pixel_landmarks: np.ndarray

    original_width: int
    original_height: int

    @classmethod
    def from_normalized(
        cls,
        normalized_landmarks: list,          # MediaPipe landmark list
        original_width: int,
        original_height: int,
    ) -> "FaceData":
        """
        Convert MediaPipe normalized [0,1] landmarks to absolute pixel coords
        immediately on construction, eliminating any ambiguity downstream.
        """
        coords = np.array(
            [(lm.x * original_width, lm.y * original_height)
             for lm in normalized_landmarks],
            dtype=np.float32,
        )
        return cls(
            pixel_landmarks=coords,
            original_width=original_width,
            original_height=original_height,
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
        from retouch.skin import SkinProcessor
        from retouch.blemish import BlemishRemover
        from retouch.eyes import EyeEnhancer
        from retouch.undereye import UnderEyeRepairer
        from retouch.lips import LipEnhancer
        from retouch.teeth import TeethWhitener
        from retouch.makeup import MakeupEngine
        from retouch.hair import HairEnhancer
        from retouch.relight import Relighter
        _WORKER_PROCESSORS = {
            "skin": SkinProcessor(),
            "relighter": Relighter(),
            "blemish": BlemishRemover(),
            "undereye": UnderEyeRepairer(),
            "eyes": EyeEnhancer(),
            "teeth": TeethWhitener(),
            "lips": LipEnhancer(),
            "makeup": MakeupEngine(),
            "hair": HairEnhancer(),
        }
    return _WORKER_PROCESSORS


def _process_single_face_worker(payload: tuple) -> dict:
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

    # Late imports avoid a circular import at module load time
    # (engine imports parsing imports perf_optimizations).
    from retouch.engine import _process_face_core
    from retouch.detection import FaceData

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
        self._max_workers = max_workers or os.cpu_count() or 2
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
        payloads: list[tuple],
    ) -> list[dict | None]:
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
# PROPOSAL 2 — Downscaled MediaPipe inference
# ──────────────────────────────────────────────────────────────────────────────

_MIN_FACE_SHORT_SIDE_PX = 64


def detect_faces_downscaled(
    landmarker: Any,          # mediapipe FaceLandmarker (or your wrapper)
    img_bgr: np.ndarray,
    max_dim: int = 1024,
) -> list[FaceData]:
    """
    Run face landmark detection on a downscaled copy of the image, then
    remap results to original-resolution pixel coordinates.

    Args:
        landmarker:  MediaPipe FaceLandmarker instance (lives in parent process).
        img_bgr:     Original BGR image at full resolution.
        max_dim:     Maximum dimension (px) for detection input.

    Returns:
        List of FaceData with pixel_landmarks in original-image coordinates.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    # ── downscale if needed ───────────────────────────────────────────────────
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / float(max(h_orig, w_orig))
        w_down = int(w_orig * scale)
        h_down = int(h_orig * scale)
        img_detect = cv2.resize(
            img_bgr, (w_down, h_down), interpolation=cv2.INTER_AREA
        )
        logger.debug(
            "Detection downscaled: (%d,%d) → (%d,%d)", w_orig, h_orig, w_down, h_down
        )
    else:
        img_detect = img_bgr.copy()
        scale = 1.0

    # ── run landmark detection ────────────────────────────────────────────────
    raw_results = landmarker.detect(img_detect)

    if not raw_results:
        return []

    # ── small-face guard ──────────────────────────────────────────────────────
    if scale < 1.0:
        for face_lms in raw_results:
            xs = [lm.x * w_orig for lm in face_lms]
            ys = [lm.y * h_orig for lm in face_lms]
            bbox_w = (max(xs) - min(xs))
            bbox_h = (max(ys) - min(ys))
            short_side = min(bbox_w, bbox_h)
            if short_side < _MIN_FACE_SHORT_SIDE_PX:
                logger.debug(
                    "Small face detected (short side %.1fpx); "
                    "re-running detection at full resolution",
                    short_side,
                )
                img_full_copy = img_bgr.copy()
                raw_results = landmarker.detect(img_full_copy)
                break

    return [
        FaceData.from_normalized(
            normalized_landmarks=face_lms,
            original_width=w_orig,
            original_height=h_orig,
        )
        for face_lms in raw_results
    ]


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


def init_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX Runtime InferenceSession with the best available provider.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_profiling = False

    if sys.platform == "darwin":
        opts.inter_op_num_threads = 1

    providers = build_ort_providers()
    logger.info("Loading ONNX model: %s  providers=%s", model_path, providers)

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def init_mediapipe_with_gpu() -> Any:
    """
    Initialize a MediaPipe FaceLandmarker with GPU delegation on Apple Silicon.
    """
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    if sys.platform == "darwin":
        delegate = BaseOptions.Delegate.GPU
        logger.info("MediaPipe FaceLandmarker: GPU delegate (Metal)")
    else:
        delegate = BaseOptions.Delegate.CPU
        logger.info("MediaPipe FaceLandmarker: CPU delegate")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="face_landmarker.task",
            delegate=delegate,
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


# ──────────────────────────────────────────────────────────────────────────────
# PROPOSAL 4 — Numba JIT pixel loops
# ──────────────────────────────────────────────────────────────────────────────

_TONAL_SIGNATURES = [
    "float32[:,:,:](float32[:,:,:], float32[:])",
]


@numba.jit(
    _TONAL_SIGNATURES,
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def _apply_tonal_lut(img_f: np.ndarray, y_lut: np.ndarray) -> np.ndarray:
    """
    JIT-compiled tonal LUT mapping over a float32 image.
    """
    h, w, _ = img_f.shape
    out = np.empty_like(img_f)

    for y in numba.prange(h):
        for x in range(w):
            v0 = img_f[y, x, 0]
            v1 = img_f[y, x, 1]
            v2 = img_f[y, x, 2]

            i0 = int(v0)
            i1 = int(v1)
            i2 = int(v2)

            if i0 < 0:   i0 = 0
            elif i0 > 255: i0 = 255
            if i1 < 0:   i1 = 0
            elif i1 > 255: i1 = 255
            if i2 < 0:   i2 = 0
            elif i2 > 255: i2 = 255

            out[y, x, 0] = y_lut[i0]
            out[y, x, 1] = y_lut[i1]
            out[y, x, 2] = y_lut[i2]

    return out


def apply_tonal_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Public wrapper: normalizes inputs, guards against NaN, calls the JIT kernel.
    """
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32)
    elif img.dtype == np.float32:
        if img.max() <= 1.0:
            img_f = img * 255.0
        else:
            img_f = img
    else:
        img_f = img.astype(np.float32)
        if img_f.max() <= 1.0:
            img_f = img_f * 255.0

    lut_f = np.nan_to_num(lut.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if lut_f.shape[0] != 256:
        raise ValueError(f"LUT must have exactly 256 entries; got {lut_f.shape[0]}")

    return _apply_tonal_lut(img_f, lut_f)


@numba.jit(
    ["float32[:,:,:](float32[:,:,:], float32[:,:,:], float32[:,:])"],
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def _blend_highpass(
    low: np.ndarray,
    original: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled high-pass blend for frequency separation.
    Computes: out = low + (original - low) * (1 - alpha_mask)
    """
    h, w, _ = low.shape
    out = np.empty_like(low)

    for y in numba.prange(h):
        for x in range(w):
            a = alpha_mask[y, x]
            one_minus_a = 1.0 - a

            out[y, x, 0] = low[y, x, 0] + (original[y, x, 0] - low[y, x, 0]) * one_minus_a
            out[y, x, 1] = low[y, x, 1] + (original[y, x, 1] - low[y, x, 1]) * one_minus_a
            out[y, x, 2] = low[y, x, 2] + (original[y, x, 2] - low[y, x, 2]) * one_minus_a

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Warm-up JIT compilation
# ──────────────────────────────────────────────────────────────────────────────

def warmup_jit_kernels() -> None:
    """
    Triggers Numba JIT compilation of kernels on dummy data.
    """
    logger.info("Warming up Numba JIT kernels...")
    dummy_img = np.zeros((4, 4, 3), dtype=np.float32)
    dummy_lut = np.arange(256, dtype=np.float32)
    dummy_mask = np.ones((4, 4), dtype=np.float32) * 0.5

    _apply_tonal_lut(dummy_img, dummy_lut)
    _blend_highpass(dummy_img, dummy_img, dummy_mask)
    logger.info("Numba JIT warm-up complete.")

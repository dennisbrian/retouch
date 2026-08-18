"""Face detection layer — MediaPipe Tasks with a legacy compatibility path.

Primary:   RetinaFace (pip) for bounding boxes + MediaPipe FaceLandmarker for 478 landmarks.
Fallback:  MediaPipe FaceLandmarker on full image (if RetinaFace unavailable or finds nothing).
Optional:  insightface RetinaFace (better side-profiles) — wired but not primary.

The detector returns a list of FaceData objects that downstream modules consume.
"""

from __future__ import annotations

import dataclasses
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .parsing import FaceRegions
from .lighting import LightDirection
from .utils import inter_eye_distance

# Resolve model paths relative to this package
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_FACE_LANDMARKER_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")
_SELFIE_SEGMENTER_MODEL = os.path.join(_MODELS_DIR, "selfie_segmenter.tflite")

# Crop padding (as a fraction of the RetinaFace box) tried in order when
# landmarking a detected face. 0.3 succeeds for typical forward-facing
# portraits and is kept first so the common case costs one landmarker call;
# wider pads are retried only when it returns no landmarks.
_CROP_PAD_FRACTIONS = (0.3, 0.6, 1.0)

# Small-crop upscale retry (F4): MediaPipe's landmarker fails once a crop
# falls below ~256px, so tiny crops are upscaled ×2 (capped at 1024) before
# the landmarker runs. Landmarks come back normalized [0,1], which is
# scale-invariant under the resize, so the remap still uses the original
# (pre-upscale) crop dims — no fold-back needed.
_MIN_CROP_DIM = 256
_MAX_CROP_DIM = 1024


@dataclasses.dataclass
class _Landmark:
    """Shim for remapped landmark coordinates."""
    x: float
    y: float
    z: float = 0.0


@dataclasses.dataclass
class FaceData:
    """Container for a single detected face."""
    landmarks: object                    # Landmark list (NormalizedLandmark[])
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) bounding box
    ied: float                           # inter-eye distance in pixels
    confidence: float = 1.0
    # A numeric default kept for compatibility is not automatically measured
    # confidence. Consumers must inspect this source before using the value as
    # evidence.
    confidence_source: str = "unknown"


@dataclasses.dataclass
class FaceContext:
    """Cached per-face detection + parsing results, carried through the
    pipeline via ProcessingContext so re-detection / re-parsing can be
    skipped. Picklable for ProcessPool IPC.
    """
    face_data: FaceData
    regions: FaceRegions
    index: int = 0
    face_image: Optional[np.ndarray] = None
    # Cached once after parsing. Consumers must honor confidence before use.
    light_direction: Optional[LightDirection] = None


class _LandmarkCompat:
    """Compatibility wrapper: makes the new FaceLandmarkerResult landmarks
    look like the old mp.solutions NormalizedLandmarkList so downstream
    parsing code works unchanged.
    """
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def _detach_landmarks(landmark_list: Any) -> List[_Landmark]:
    """Copy MediaPipe landmarks into Retouch-owned plain dataclasses.

    The legacy Solutions API exposes protobuf-backed landmark proxies whose
    owner is the short-lived ``FaceMesh.process()`` result. Retaining those
    proxies makes later ``deepcopy``/session caching fail with ``ReferenceError``
    after the native result is released. The engine only consumes x/y/z, so
    detach those values at the detection boundary.
    """
    return [
        _Landmark(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(getattr(landmark, "z", 0.0)),
        )
        for landmark in landmark_list
    ]


class FaceDetector:
    """Detect faces and extract 478 landmarks.

    Primary: RetinaFace (pip) for bounding boxes, then MediaPipe FaceLandmarker
    on each crop for precise 478 landmarks.  The pinned legacy wheel is also
    supported where its Solutions API is available.  ``allow_unavailable`` is
    reserved for application startup: it produces a clearly non-face-aware
    detector instead of allowing an incompatible native runtime to abort the
    host process.
    """

    def __init__(
        self,
        max_faces: int = 10,
        min_confidence: float = 0.4,
        refine_landmarks: bool = True,
        allow_unavailable: bool = False,
    ):
        self.max_faces = max_faces
        self.min_confidence = min_confidence
        self.available = True
        self.unavailable_reason: Optional[str] = None
        self.backend_name = "uninitialized"
        self._legacy_mesh = None
        self._legacy_segmenter = None
        self._landmarker = None
        self._segmenter = None

        backend = os.environ.get("RETOUCH_MEDIAPIPE_BACKEND", "auto").strip().lower()
        has_legacy_solutions = hasattr(mp, "solutions")
        # Unit tests replace the factory with a mock so downstream parsing can
        # be exercised without constructing native MediaPipe.  Keep that seam
        # intact while the real runtime capability check protects production
        # processes from the macOS native abort.
        factory_is_mocked = hasattr(self._create_tasks, "mock_calls")
        if backend not in {"auto", "tasks", "legacy"}:
            raise ValueError("RETOUCH_MEDIAPIPE_BACKEND must be auto, tasks, or legacy")
        if not factory_is_mocked and (backend == "legacy" or (backend == "auto" and has_legacy_solutions)):
            if not has_legacy_solutions:
                error = RuntimeError(
                    "Legacy MediaPipe backend requested, but this MediaPipe build removed mp.solutions. "
                    "Install mediapipe==0.10.5 and protobuf<5 from requirements/base.txt."
                )
                if allow_unavailable:
                    self._mark_unavailable(error)
                    return
                raise error
            try:
                self._init_legacy_backend(max_faces, min_confidence, refine_landmarks)
            except Exception as exc:
                if allow_unavailable:
                    self._mark_unavailable(exc)
                    return
                raise
            self.backend_name = "mediapipe_legacy"
            return

        if backend in {"auto", "tasks"} and not has_legacy_solutions and not factory_is_mocked:
            version = getattr(mp, "__version__", "unknown")
            if version == "0.10.35":
                error = RuntimeError(
                    "MediaPipe 0.10.35 Tasks FaceLandmarker is not supported by this macOS runtime; "
                    "install mediapipe==0.10.5 and protobuf<5, then use the legacy CPU backend."
                )
                if allow_unavailable:
                    self._mark_unavailable(error)
                    return
                raise error

        # Resolve task assets through the manifest/cache layer. This is
        # important for wheels and frozen apps, whose package directory may be
        # read-only and cannot receive a first-use download.
        try:
            from .model_fetch import ModelFetchError, get_model_path, model_status

            global _FACE_LANDMARKER_MODEL, _SELFIE_SEGMENTER_MODEL
            _FACE_LANDMARKER_MODEL = get_model_path("face_landmarker")
            selfie_status = model_status("selfie_segmenter")
            if selfie_status.get("available") and selfie_status.get("path"):
                _SELFIE_SEGMENTER_MODEL = selfie_status["path"]
        except ModelFetchError as exc:
            error = FileNotFoundError(
                f"Verified face landmarker model is unavailable: {exc}\n"
                f"Expected package/cache path: {_FACE_LANDMARKER_MODEL}"
            )
            if allow_unavailable:
                self._mark_unavailable(error)
                return
            raise error

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions

        delegate = self._resolve_delegate(base)
        try:
            self._landmarker, self._segmenter = self._create_tasks(
                base, vision, delegate, max_faces, min_confidence
            )
        except Exception as exc:
            if delegate is base.Delegate.GPU:
                import logging
                logging.getLogger(__name__).warning(
                    "MediaPipe GPU delegate failed (%s: %s); falling back to CPU",
                    type(exc).__name__, exc,
                )
                self._landmarker, self._segmenter = self._create_tasks(
                    base, vision, base.Delegate.CPU, max_faces, min_confidence
                )
            else:
                if allow_unavailable:
                    self._mark_unavailable(exc)
                    return
                raise
        self.backend_name = "mediapipe_tasks"

    def _mark_unavailable(self, error: BaseException) -> None:
        """Keep the host application usable without pretending faces were found."""
        import logging
        self.available = False
        self.backend_name = "unavailable"
        self.unavailable_reason = f"{type(error).__name__}: {error}"
        logging.getLogger(__name__).warning(
            "Face-aware detection unavailable; using global-only fallback: %s",
            self.unavailable_reason,
        )

    def runtime_status(self) -> dict[str, Any]:
        """Return the authoritative face-runtime capability snapshot.

        ``available`` describes whether the detector backend initialized; it
        intentionally does not depend on whether a particular image contains
        a face.  Consumers can therefore distinguish an initialized
        face-aware runtime from a global-only fallback without guessing from
        a zero-length detection result.
        """
        return {
            "mode": "face_aware" if self.available else "global_only",
            "available": bool(self.available),
            "backend": self.backend_name,
            "reason": self.unavailable_reason,
            "probe_state": "initialized" if self.available else "blocked",
        }

    def _init_legacy_backend(self, max_faces: int, min_confidence: float, refine_landmarks: bool) -> None:
        """Initialize the stable CPU FaceMesh backend used by the pinned runtime."""
        self._legacy_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        try:
            self._legacy_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        except Exception:
            self._legacy_segmenter = None

    @staticmethod
    def _create_tasks(
        base: Any,
        vision: Any,
        delegate: Any,
        max_faces: int,
        min_confidence: float,
    ) -> Tuple[Any, Any]:
        """Build the FaceLandmarker + ImageSegmenter for a given delegate."""
        base_options = base(
            model_asset_path=_FACE_LANDMARKER_MODEL,
            delegate=delegate,
        )
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

        segmenter = None
        if os.path.exists(_SELFIE_SEGMENTER_MODEL):
            segmenter_base_options = base(
                model_asset_path=_SELFIE_SEGMENTER_MODEL,
                delegate=delegate,
            )
            segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=segmenter_base_options,
                    output_category_mask=False,
                    output_confidence_masks=True,
                )
            )
        return landmarker, segmenter

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> List[FaceData]:
        """Detect faces using RetinaFace for boxes + MediaPipe for landmarks.

        Falls back to MediaPipe landmarker on full image if RetinaFace is
        unavailable or returns nothing.
        """
        h, w = img_bgr.shape[:2]
        if not self.available:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces: List[FaceData] = []

        if self._legacy_mesh is not None:
            result = self._legacy_mesh.process(img_rgb)
            for landmarks in getattr(result, "multi_face_landmarks", None) or []:
                compat = _LandmarkCompat(_detach_landmarks(landmarks.landmark))
                bbox = self._bbox_from_landmarks(compat, w, h)
                faces.append(FaceData(
                    landmarks=compat,
                    bbox=bbox,
                    ied=inter_eye_distance(compat, w, h),
                    confidence=1.0,
                    confidence_source="mediapipe_presence_unavailable",
                ))
            return faces

        # 1. RetinaFace (pip) for bounding-box detection
        # FIX F1: pass engine min_confidence — pip default threshold is 0.9,
        # which silently drops occluded/downcast faces scoring 0.4-0.89.
        # FIX F2: pass BGR — retinaface 0.0.18's preprocess.get_image
        # documents ndarray input as BGR and preprocess_image reverses
        # channels internally (`img[:, :, 2 - i]`); handing it RGB fed the
        # model swapped channels and degraded scores.
        try:
            from retinaface import RetinaFace
            resp = RetinaFace.detect_faces(
                img_bgr, threshold=self.min_confidence
            )
        except ImportError:
            resp = None
        except Exception as exc:
            import logging
            # Surfaced at ERROR (not WARNING): a RetinaFace failure silently
            # degrades detection to the MediaPipe-only fallback, which misses
            # downcast/occluded faces. Known trigger: importing tensorflow or
            # keras before `retinaface` resolves the package onto its Keras 3
            # path and raises ValueError, so this must not pass unnoticed.
            logging.getLogger(__name__).error(
                "RetinaFace raised %s: %s — falling back to MediaPipe-only "
                "detection (reduced recall on occluded faces)",
                type(exc).__name__, exc
            )
            resp = None

        retinaface_box_count = 0
        if isinstance(resp, dict):
            for face_id, face_info in resp.items():
                box = face_info.get("facial_area")
                score = face_info.get("score", 1.0)
                if not box:
                    continue
                retinaface_box_count += 1
                x1, y1, x2, y2 = box
                bw = x2 - x1
                bh = y2 - y1

                # Pad the crop to avoid clipping hair/neck, then landmark it.
                # RetinaFace boxes are tight; MediaPipe's landmarker needs
                # surrounding context and fails on a 0.3 pad for downcast or
                # occluded faces (measured: a 770x1095 downcast face yields 0
                # landmarks at 0.3 but succeeds at 0.6). Escalate the pad and
                # retry rather than dropping the face. Retry widens the *pad*,
                # never downscales — MediaPipe also fails once the crop falls
                # below ~256px, so shrinking would trade one failure for another.
                for pad_frac in _CROP_PAD_FRACTIONS:
                    # If the padded box exceeds image bounds, reduce padding
                    # symmetrically rather than shifting (shifting creates a
                    # larger crop, up to the full image, which kills perf).
                    pad_x = int(bw * pad_frac)
                    pad_y = int(bh * pad_frac)
                    if bw + 2 * pad_x > w:
                        pad_x = max(0, (w - bw) // 2)
                    if bh + 2 * pad_y > h:
                        pad_y = max(0, (h - bh) // 2)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w, x2 + pad_x)
                    cy2 = min(h, y2 + pad_y)
                    cw = cx2 - cx1
                    ch = cy2 - cy1

                    if cw < 4 or ch < 4:
                        continue

                    crop_rgb = img_rgb[cy1:cy2, cx1:cx2]

                    # MediaPipe's landmarker fails on crops below ~256px;
                    # upscale small crops ×2 (cap 1024) before detection and
                    # fold the scale back into the remap.
                    landmark_scale = 1.0
                    if cw < _MIN_CROP_DIM or ch < _MIN_CROP_DIM:
                        landmark_scale = min(
                            2.0,
                            _MAX_CROP_DIM / float(max(cw, ch)),
                        )
                        if landmark_scale > 1.0:
                            crop_rgb = cv2.resize(
                                crop_rgb,
                                (int(round(cw * landmark_scale)),
                                 int(round(ch * landmark_scale))),
                                interpolation=cv2.INTER_LINEAR,
                            )

                    mp_crop = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=crop_rgb.copy()
                    )
                    crop_result = self._landmarker.detect(mp_crop)

                    if crop_result.face_landmarks:
                        lm_list = crop_result.face_landmarks[0]
                        remapped = self._remap_landmarks(
                            lm_list, cx1, cy1, cw, ch, w, h,
                        )
                        compat = _LandmarkCompat(remapped)
                        ied = inter_eye_distance(compat, w, h)
                        faces.append(FaceData(
                            landmarks=compat,
                            bbox=(x1, y1, bw, bh),
                            ied=ied,
                            confidence=float(score),
                            confidence_source="retinaface",
                        ))
                        break

        # 2. Fallback: MediaPipe on full image
        # Run if RetinaFace found nothing, OR if some RetinaFace boxes
        # failed crop-landmarking (partial coverage).
        retinaface_lost_some = retinaface_box_count > 0 and len(faces) < retinaface_box_count
        if not faces or retinaface_lost_some:
            # OPTIMIZATION: Run detection on downscaled copy to speed up inference on 4K/high-res frames
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                w_down = int(w * scale)
                h_down = int(h * scale)
                img_down = cv2.resize(img_rgb, (w_down, h_down), interpolation=cv2.INTER_AREA)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_down)
                result = self._landmarker.detect(mp_image)
                # If downscaling finds no faces, fallback to full resolution as safety guard
                if not result.face_landmarks:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    result = self._landmarker.detect(mp_image)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = self._landmarker.detect(mp_image)

            if result.face_landmarks:
                existing = set()
                for f in faces:
                    existing.add((round(f.bbox[0] / 10) * 10, round(f.bbox[1] / 10) * 10))
                for lm_list in result.face_landmarks:
                    compat = _LandmarkCompat(lm_list)
                    bbox = self._bbox_from_landmarks(compat, w, h)
                    # Deduplicate against faces already found via RetinaFace crop.
                    # Key on both x and y buckets: x-only collides for faces
                    # sharing an x column (stacked faces in portrait shots).
                    key = (round(bbox[0] / 10) * 10, round(bbox[1] / 10) * 10)
                    if key in existing:
                        continue
                    existing.add(key)
                    ied = inter_eye_distance(compat, w, h)
                    faces.append(FaceData(
                        landmarks=compat,
                        bbox=bbox,
                        ied=ied,
                        confidence_source="mediapipe_presence_unavailable",
                    ))

        return faces

    def segment_person(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return a float mask (H, W) separating person from background.

        Values 0.0 = background, 1.0 = person.
        """
        if not self.available:
            return np.ones(img_bgr.shape[:2], dtype=np.float32)
        if self._legacy_segmenter is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = self._legacy_segmenter.process(img_rgb)
            mask = getattr(result, "segmentation_mask", None)
            if mask is not None:
                return np.asarray(mask, dtype=np.float32)
        if self._segmenter is None:
            return np.ones(img_bgr.shape[:2], dtype=np.float32)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = self._segmenter.segment(mp_image)

        if result.confidence_masks and len(result.confidence_masks) > 0:
            mask = result.confidence_masks[0].numpy_view().copy()
            mask = np.squeeze(mask)
            return mask.astype(np.float32)

        return np.ones(img_bgr.shape[:2], dtype=np.float32)

    def close(self) -> None:
        """Release resources.

        Idempotent: each task is dropped after closing, so a second call is a
        no-op. Must be called on the main thread while the dispatcher is still
        alive — if the tasks are instead left to the garbage collector,
        MediaPipe's own ``FaceLandmarker.__del__`` blocks forever on a pending
        serial-dispatcher future and hangs the process.
        """
        for attr in ("_legacy_mesh", "_legacy_segmenter", "_landmarker", "_segmenter"):
            task = getattr(self, attr, None)
            if task is not None:
                self._close_task(task)
                setattr(self, attr, None)

    @staticmethod
    def _close_task(task: Any) -> None:
        """Close one MediaPipe task, never propagating teardown errors.

        A task that is already closed (or whose native handle is gone at
        interpreter shutdown) raises rather than returning cleanly; that must
        not mask the caller's real error or abort the remaining closes.
        """
        try:
            task.close()
        except Exception:
            pass

    # NOTE: deliberately no __del__. MediaPipe's own FaceLandmarker.__del__
    # blocks on a serial-dispatcher future, so adding a finalizer here cannot
    # prevent the hang (it is a blocking call, not an exception) and would
    # itself risk blocking on any cyclic GC. Callers must use close() or the
    # context manager on the main thread while the dispatcher is alive.

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_delegate(base: Any) -> Any:
        # Default to CPU. GPU delegation is opt-in via RETUCH_GPU=1: on some
        # platforms MediaPipe's GPU delegate constructor HANGS (does not raise),
        # so we must not attempt it by default — the __init__ try/except only
        # catches exceptions, not hangs. When RETUCH_GPU is set and GPU works,
        # task creation succeeds; if it raises, __init__ falls back to CPU.
        gpu_requested = os.environ.get("RETUCH_GPU", "").strip().lower()
        if gpu_requested in {"1", "true", "yes", "on"}:
            return base.Delegate.GPU
        return base.Delegate.CPU

    @staticmethod
    def _bbox_from_landmarks(
        landmarks_compat: Any,
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        """Compute tight bounding box from landmarks."""
        xs, ys = [], []
        for lm in landmarks_compat.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
        x1 = max(min(xs), 0)
        y1 = max(min(ys), 0)
        x2 = min(max(xs), w)
        y2 = min(max(ys), h)
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _remap_landmarks(
        landmarks: List[Any],
        ox: int,
        oy: int,
        cw: int,
        ch: int,
        full_w: int,
        full_h: int,
    ) -> List["_Landmark"]:
        """Remap crop-relative landmarks to full-image coordinates."""
        remapped = []
        for lm in landmarks:
            remapped.append(_Landmark(
                x=(lm.x * cw + ox) / full_w,
                y=(lm.y * ch + oy) / full_h,
                z=lm.z,
            ))
        return remapped

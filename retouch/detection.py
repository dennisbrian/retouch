"""Face detection layer — MediaPipe Tasks API (v0.10.35+).

Primary:   RetinaFace (pip) for bounding boxes + MediaPipe FaceLandmarker for 478 landmarks.
Fallback:  MediaPipe FaceLandmarker on full image (if RetinaFace unavailable or finds nothing).
Optional:  insightface RetinaFace (better side-profiles) — wired but not primary.

The detector returns a list of FaceData objects that downstream modules consume.
"""

import dataclasses
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .parsing import FaceRegions
from .utils import inter_eye_distance

# Resolve model paths relative to this package
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_FACE_LANDMARKER_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")
_SELFIE_SEGMENTER_MODEL = os.path.join(_MODELS_DIR, "selfie_segmenter.tflite")


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


class _LandmarkCompat:
    """Compatibility wrapper: makes the new FaceLandmarkerResult landmarks
    look like the old mp.solutions NormalizedLandmarkList so downstream
    parsing code works unchanged.
    """
    def __init__(self, landmark_list):
        self.landmark = landmark_list


class FaceDetector:
    """Detect faces and extract 478 landmarks.

    Primary: RetinaFace (pip) for bounding boxes, then MediaPipe FaceLandmarker
    on each crop for precise 478 landmarks.
    Fallback: MediaPipe FaceLandmarker on the full image.
    """

    def __init__(
        self,
        max_faces: int = 10,
        min_confidence: float = 0.4,
        refine_landmarks: bool = True,
    ):
        self.max_faces = max_faces
        self.min_confidence = min_confidence

        if not os.path.exists(_FACE_LANDMARKER_MODEL):
            raise FileNotFoundError(
                f"Face landmarker model not found at {_FACE_LANDMARKER_MODEL}\n"
                f"Download it from:\n"
                f"  https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

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
                raise

    @staticmethod
    def _create_tasks(base, vision, delegate, max_faces, min_confidence):
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
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces: List[FaceData] = []

        # 1. RetinaFace (pip) for bounding-box detection
        try:
            from retinaface import RetinaFace
            resp = RetinaFace.detect_faces(img_rgb)
        except ImportError:
            resp = None
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "RetinaFace raised %s: %s", type(exc).__name__, exc
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

                # Pad the crop by 30% to avoid clipping hair/neck.
                # If the padded box exceeds image bounds, reduce padding
                # symmetrically rather than shifting (shifting creates a
                # larger crop, up to the full image, which kills perf).
                pad_x = int(bw * 0.3)
                pad_y = int(bh * 0.3)
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
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb.copy())
                crop_result = self._landmarker.detect(mp_crop)

                if crop_result.face_landmarks:
                    lm_list = crop_result.face_landmarks[0]
                    remapped = self._remap_landmarks(lm_list, cx1, cy1, cw, ch, w, h)
                    compat = _LandmarkCompat(remapped)
                    ied = inter_eye_distance(compat, w, h)
                    faces.append(FaceData(
                        landmarks=compat,
                        bbox=(x1, y1, bw, bh),
                        ied=ied,
                        confidence=float(score)
                    ))

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
                    existing.add(round(f.bbox[0] / 10) * 10)
                for lm_list in result.face_landmarks:
                    compat = _LandmarkCompat(lm_list)
                    bbox = self._bbox_from_landmarks(compat, w, h)
                    # Deduplicate against faces already found via RetinaFace crop
                    key = round(bbox[0] / 10) * 10
                    if key in existing:
                        continue
                    existing.add(key)
                    ied = inter_eye_distance(compat, w, h)
                    faces.append(FaceData(landmarks=compat, bbox=bbox, ied=ied))

        return faces

    def segment_person(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return a float mask (H, W) separating person from background.

        Values 0.0 = background, 1.0 = person.
        """
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

    def close(self):
        """Release resources."""
        if hasattr(self, '_landmarker') and self._landmarker:
            self._landmarker.close()
        if hasattr(self, '_segmenter') and self._segmenter:
            self._segmenter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_delegate(base):
        return base.Delegate.CPU

    @staticmethod
    def _bbox_from_landmarks(landmarks_compat, w, h):
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
    def _remap_landmarks(landmarks, ox, oy, cw, ch, full_w, full_h):
        """Remap crop-relative landmarks to full-image coordinates."""
        remapped = []
        for lm in landmarks:
            remapped.append(_Landmark(
                x=(lm.x * cw + ox) / full_w,
                y=(lm.y * ch + oy) / full_h,
                z=lm.z,
            ))
        return remapped

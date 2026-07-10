"""Face region parsing — generates per-region masks from MediaPipe landmarks.

Regions produced:
    skin, forehead, left_cheek, right_cheek, nose,
    left_eye, right_eye, left_eyebrow, right_eyebrow,
    left_iris, right_iris, lips, mouth_interior (for teeth),
    left_under_eye, right_under_eye, face_oval.

All masks are float32 (H, W) in [0, 1] with soft feathered edges.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import logging
import numpy as np
import os
import onnxruntime as ort
from .perf_optimizations import build_ort_providers

from .utils import create_polygon_mask, feather_mask, get_points, normalize_mask

logger = logging.getLogger(__name__)

# =====================================================================
# MediaPipe Face Mesh landmark indices (stable across versions)
# =====================================================================

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE = [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153,
    145, 144, 163, 7,
]
RIGHT_EYE = [
    263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380,
    374, 373, 390, 249,
]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Nose region
NOSE = [
    168, 6, 197, 195, 5, 4, 1, 2, 98, 327,
    294, 278, 344, 440, 275, 45, 220, 115, 48, 64,
]

# Under-eye: lower lid → upper cheek
LEFT_UNDER_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    130, 25, 110, 24, 23, 22, 26, 112, 243,
]
RIGHT_UNDER_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    359, 255, 339, 254, 253, 252, 256, 341, 463,
]

# Forehead: top of face oval → above eyebrows
FOREHEAD_TOP = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
FOREHEAD_BOTTOM = [
    70, 63, 105, 66, 107,  # left eyebrow
    9,                       # mid brow bridge
    336, 296, 334, 293, 300, # right eyebrow
]

# Cheek regions (approximate polygons)
LEFT_CHEEK = [
    117, 118, 119, 120, 121, 128, 245, 193, 55,
    65, 52, 53, 46, 124, 35, 111, 117,
]
RIGHT_CHEEK = [
    346, 347, 348, 349, 350, 357, 465, 417, 285,
    295, 282, 283, 276, 353, 265, 340, 346,
]

# Nasolabial folds (wrinkle zones)
LEFT_NASOLABIAL = [206, 216, 92, 165, 167]
RIGHT_NASOLABIAL = [426, 436, 322, 391, 393]

# Crow's feet / outer eye wrinkle zones — lateral (temple-side) skin outward
# from the outer eye corner (33 / 263), NOT the eyelid/lash contour itself.
# Verified 2026-07-03 by rendering these points on two real detected faces
# (test_output/DSCF4550.jpg and grace_hopper.jpg) and visually confirming
# the cluster sits outside the LEFT_EYE/RIGHT_EYE lash contour, in skin,
# toward the temple/hairline — see fix notes for the render technique used.
LEFT_CROWS_FEET = [34, 227, 116, 137]
RIGHT_CROWS_FEET = [264, 447, 345, 366]


class FaceRegions:
    """Holds all per-region masks for one face."""

    __slots__ = [
        "face_oval", "skin", "forehead",
        "left_cheek", "right_cheek", "nose",
        "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow",
        "left_iris", "right_iris",
        "lips", "mouth_interior",
        "left_under_eye", "right_under_eye",
        "nose_bridge", "forehead_center",
        "cheek_highlights_l", "cheek_highlights_r",
        "jawline_contour", "hair", "neck",
        "nasolabial_l", "nasolabial_r",
        "crows_feet_l", "crows_feet_r",
        "cloth",
    ]

    def __init__(self) -> None:
        for attr in self.__slots__:
            setattr(self, attr, None)


class FaceParser:
    """Generate precise per-region masks using BiSeNet ONNX + MediaPipe Face Mesh coordinates."""

    def __init__(self) -> None:
        self._model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "resnet18.onnx"
        )
        self._sess = None
        if os.path.exists(self._model_path):
            import logging
            logger = logging.getLogger(__name__)
            try:
                # Auto-discover execution providers in order of preference
                providers = build_ort_providers()
                
                self._sess = ort.InferenceSession(self._model_path, providers=providers)
                logger.info("ONNX Runtime initialized with active providers: %s", self._sess.get_providers())
            except Exception as e:
                logger.warning("Failed to initialize ONNX with providers: %s. Error: %s. Falling back to default.", providers if 'providers' in locals() else 'None', e)
                try:
                    self._sess = ort.InferenceSession(self._model_path)
                    logger.info("ONNX Runtime fallback initialized. Providers: %s", self._sess.get_providers())
                except Exception as fallback_err:
                    logger.error("ONNX Runtime failed completely: %s", fallback_err)

    def parse(
        self,
        landmarks: Any,
        img_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        person_mask: Optional[np.ndarray] = None,
        ied: float = 100.0,
    ) -> "FaceRegions":
        """Parse a single face into region masks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList (468+ landmarks).
            img_bgr: BGR input image.
            face_bbox: RetinaFace bounding box (x, y, w, h).
            person_mask: Optional float mask from selfie segmentation.
            ied: Inter-eye distance in pixels.

        Returns:
            FaceRegions with all masks populated.
        """
        h_img, w_img = img_bgr.shape[:2]
        feather = max(int(ied * 0.08), 3)  # adaptive feather radius
        regions = FaceRegions()

        # ---- Run BiSeNet face parsing ONNX if session exists ----
        bisenet_masks = {}
        if self._sess is not None and face_bbox is not None:
            try:
                x_face, y_face, w_face, h_face = face_bbox
                # Pad crop by 30% to capture hair and neck
                pad_x = int(w_face * 0.3)
                pad_y = int(h_face * 0.3)
                cx1 = max(0, x_face - pad_x)
                cy1 = max(0, y_face - pad_y)
                cx2 = min(w_img, x_face + w_face + pad_x)
                cy2 = min(h_img, y_face + h_face + pad_y)
                cw = cx2 - cx1
                ch = cy2 - cy1

                if cw >= 4 and ch >= 4:
                    crop = img_bgr[cy1:cy2, cx1:cx2]
                    # Preprocess for BiSeNet
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    crop_f = crop_resized.astype(np.float32) / 255.0

                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    crop_norm = (crop_f - mean) / std
                    crop_input = np.transpose(crop_norm, (2, 0, 1))[np.newaxis, :, :, :]

                    # Run inference
                    outs = self._sess.run(None, {'input': crop_input})
                    logits = outs[0][0]  # shape (19, 512, 512)
                    pred_crop = np.argmax(logits, axis=0).astype(np.uint8)

                    # Resize back to crop size
                    pred_crop_resized = cv2.resize(pred_crop, (cw, ch), interpolation=cv2.INTER_NEAREST)

                    # Paste to full image size class label map
                    full_label_map = np.zeros((h_img, w_img), dtype=np.uint8)
                    full_label_map[cy1:cy2, cx1:cx2] = pred_crop_resized

                    # Generate binary masks for required classes
                    bisenet_masks['skin'] = (full_label_map == 1).astype(np.float32)
                    bisenet_masks['left_eyebrow'] = (full_label_map == 2).astype(np.float32)
                    bisenet_masks['right_eyebrow'] = (full_label_map == 3).astype(np.float32)
                    bisenet_masks['left_eye'] = (full_label_map == 4).astype(np.float32)
                    bisenet_masks['right_eye'] = (full_label_map == 5).astype(np.float32)
                    bisenet_masks['mouth_interior'] = (full_label_map == 11).astype(np.float32)
                    bisenet_masks['lips'] = ((full_label_map == 12) | (full_label_map == 13)).astype(np.float32)
                    bisenet_masks['neck'] = (full_label_map == 14).astype(np.float32)
                    bisenet_masks['hair'] = (full_label_map == 17).astype(np.float32)
                    bisenet_masks['cloth'] = (full_label_map == 16).astype(np.float32)

                    bisenet_masks['face_oval'] = (
                        (full_label_map == 1) | (full_label_map == 2) | (full_label_map == 3) |
                        (full_label_map == 4) | (full_label_map == 5) | (full_label_map == 10) |
                        (full_label_map == 11) | (full_label_map == 12) | (full_label_map == 13)
                    ).astype(np.float32)

                    # Apply feathering
                    for k in ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'face_oval', 'neck', 'hair', 'cloth']:
                        if k in bisenet_masks:
                            r = feather // 2 if k in ['left_eye', 'right_eye', 'lips', 'left_eyebrow', 'right_eyebrow', 'cloth'] else feather
                            bisenet_masks[k] = feather_mask(bisenet_masks[k], radius=r)

                    # Clean skin mask after feathering
                    for excl_k in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'mouth_interior']:
                        if excl_k in bisenet_masks and bisenet_masks[excl_k] is not None:
                            bisenet_masks['skin'] = np.clip(bisenet_masks['skin'] - bisenet_masks[excl_k], 0.0, 1.0)
            except Exception as e:
                input_shape = crop_input.shape if 'crop_input' in locals() else None
                logger.warning(
                    "BiSeNet face parsing failed for face_bbox=%s on image shape=%s (model=%s, input_shape=%s): %s. Falling back to landmark-only regions.",
                    face_bbox, img_bgr.shape, self._model_path, input_shape, e
                )

        # Populate regions from BiSeNet masks
        regions.skin = bisenet_masks.get('skin')
        regions.lips = bisenet_masks.get('lips')
        regions.mouth_interior = bisenet_masks.get('mouth_interior')
        regions.left_eye = bisenet_masks.get('left_eye')
        regions.right_eye = bisenet_masks.get('right_eye')
        regions.left_eyebrow = bisenet_masks.get('left_eyebrow')
        regions.right_eyebrow = bisenet_masks.get('right_eyebrow')
        regions.face_oval = bisenet_masks.get('face_oval')
        regions.neck = bisenet_masks.get('neck')
        regions.hair = bisenet_masks.get('hair')
        regions.cloth = bisenet_masks.get('cloth')

        # Fallback to landmarks if BiSeNet failed or has empty skin
        if regions.skin is None or regions.skin.max() < 0.01:
            regions = self._landmark_fallback_only(landmarks, img_bgr, person_mask, ied)
        else:
            self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)

        return regions

    def parse_hair_full_image(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run BiSeNet on the *whole* image (no face-bbox crop) to get a
        body-wide hair mask, including wig hair draping past the face crop
        onto shoulders/chest that ``parse()`` never sees.

        This is coarser than the face-crop path (512x512 covers the entire
        frame instead of just a padded face region), so treat the result as
        a soft exclusion signal for body_skin masking, not a precise
        per-strand mask. Returns None if the model isn't available.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image, full frame.

        Returns:
            (H, W) float32 mask in [0, 1], or None if BiSeNet is unavailable.
        """
        if self._sess is None:
            return None

        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_f = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_f - mean) / std
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :, :, :]

        try:
            outs = self._sess.run(None, {"input": img_input})
        except Exception:
            return None

        logits = outs[0][0]
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        hair_512 = (pred == 17).astype(np.float32)
        hair_full = cv2.resize(hair_512, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        return hair_full

    def parse_batch(
        self,
        crop_list: List[np.ndarray],
        landmarks_compat_list: List[Any],
        face_bbox_list: List[Tuple[int, int, int, int]],
        person_masks: List[Optional[np.ndarray]],
        ieds: List[float],
    ) -> List["FaceRegions"]:
        """Parse multiple face crops in a single batch ONNX call.

        Args:
            crop_list: List of BGR crop images (canvases).
            landmarks_compat_list: List of landmark compat objects.
            face_bbox_list: List of crop-relative bounding boxes.
            person_masks: List of crop-relative person masks.
            ieds: List of crop-relative IED values.

        Returns:
            List of FaceRegions.
        """
        num_faces = len(crop_list)
        results = [None] * num_faces

        # If ONNX session is not loaded, fall back to sequential single-face parse
        if self._sess is None:
            for i in range(num_faces):
                results[i] = self.parse(
                    landmarks_compat_list[i],
                    crop_list[i],
                    face_bbox_list[i],
                    person_masks[i],
                    ieds[i]
                )
            return results

        # 1. Prepare and preprocess all crops
        inputs = []
        crop_coords = []  # Store (cx1, cy1, cx2, cy2, cw, ch, h_img, w_img) for paste-back
        valid_indices = []

        for i in range(num_faces):
            img_bgr = crop_list[i]
            face_bbox = face_bbox_list[i]
            h_img, w_img = img_bgr.shape[:2]

            try:
                x_face, y_face, w_face, h_face = face_bbox
                pad_x = int(w_face * 0.3)
                pad_y = int(h_face * 0.3)
                cx1 = max(0, x_face - pad_x)
                cy1 = max(0, y_face - pad_y)
                cx2 = min(w_img, x_face + w_face + pad_x)
                cy2 = min(h_img, y_face + h_face + pad_y)
                cw = cx2 - cx1
                ch = cy2 - cy1

                if cw >= 4 and ch >= 4:
                    crop = img_bgr[cy1:cy2, cx1:cx2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    crop_f = crop_resized.astype(np.float32) / 255.0

                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    crop_norm = (crop_f - mean) / std
                    crop_input = np.transpose(crop_norm, (2, 0, 1))

                    inputs.append(crop_input)
                    crop_coords.append((cx1, cy1, cx2, cy2, cw, ch, h_img, w_img))
                    valid_indices.append(i)
                else:
                    results[i] = self._landmark_fallback_only(
                        landmarks_compat_list[i], img_bgr, person_masks[i], ieds[i]
                    )
            except Exception as e:
                crop_shape = getattr(img_bgr, 'shape', None)
                logger.warning(
                    "BiSeNet preprocessing failed for face index=%d (face_bbox=%s, crop_shape=%s, model=%s): %s. Using landmark fallback.",
                    i, face_bbox, crop_shape, self._model_path, e
                )
                results[i] = self._landmark_fallback_only(
                    landmarks_compat_list[i], img_bgr, person_masks[i], ieds[i]
                )

        # 2. Run inference in sub-batches of size 4 (preserves memory)
        try:
            _MAX_BATCH_SIZE = 4
            logits_list = []
            num_valid = len(inputs)

            for i in range(0, num_valid, _MAX_BATCH_SIZE):
                sub_inputs = inputs[i:i+_MAX_BATCH_SIZE]
                sub_batch = np.stack(sub_inputs, axis=0) # shape (B, 3, 512, 512)

                try:
                    outs = self._sess.run(None, {'input': sub_batch})
                    logits = outs[0] # shape (B, 19, 512, 512)
                    for b in range(len(sub_inputs)):
                        logits_list.append(logits[b])
                except Exception as e:
                    logger.warning(
                        "Batch BiSeNet inference failed (likely due to CoreML dynamic batch limits) for sub_batch shape=%s, batch_size=%d (model=%s): %s. "
                        "Falling back to sequential inference (batch size 1).",
                        sub_batch.shape, len(sub_inputs), self._model_path, e
                    )
                    for b_in in sub_inputs:
                        single_batch = b_in[np.newaxis, :, :, :]
                        outs_single = self._sess.run(None, {'input': single_batch})
                        logits_list.append(outs_single[0][0])

            # 3. Postprocess and paste back in order (preserves input index positioning)
            for idx_valid, idx_face in enumerate(valid_indices):
                cx1, cy1, cx2, cy2, cw, ch, h_img, w_img = crop_coords[idx_valid]
                logits = logits_list[idx_valid]

                pred_crop = np.argmax(logits, axis=0).astype(np.uint8)
                pred_crop_resized = cv2.resize(pred_crop, (cw, ch), interpolation=cv2.INTER_NEAREST)

                full_label_map = np.zeros((h_img, w_img), dtype=np.uint8)
                full_label_map[cy1:cy2, cx1:cx2] = pred_crop_resized

                # Generate masks
                bisenet_masks = {}
                bisenet_masks['skin'] = (full_label_map == 1).astype(np.float32)
                bisenet_masks['left_eyebrow'] = (full_label_map == 2).astype(np.float32)
                bisenet_masks['right_eyebrow'] = (full_label_map == 3).astype(np.float32)
                bisenet_masks['left_eye'] = (full_label_map == 4).astype(np.float32)
                bisenet_masks['right_eye'] = (full_label_map == 5).astype(np.float32)
                bisenet_masks['mouth_interior'] = (full_label_map == 11).astype(np.float32)
                bisenet_masks['lips'] = ((full_label_map == 12) | (full_label_map == 13)).astype(np.float32)
                bisenet_masks['neck'] = (full_label_map == 14).astype(np.float32)
                bisenet_masks['hair'] = (full_label_map == 17).astype(np.float32)

                bisenet_masks['face_oval'] = (
                    (full_label_map == 1) | (full_label_map == 2) | (full_label_map == 3) |
                    (full_label_map == 4) | (full_label_map == 5) | (full_label_map == 10) |
                    (full_label_map == 11) | (full_label_map == 12) | (full_label_map == 13)
                ).astype(np.float32)

                # Feathering
                feather = max(int(ieds[idx_face] * 0.08), 3)
                for k in ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'face_oval', 'neck', 'hair', 'cloth']:
                    if k in bisenet_masks:
                        r = feather // 2 if k in ['left_eye', 'right_eye', 'lips', 'left_eyebrow', 'right_eyebrow', 'cloth'] else feather
                        bisenet_masks[k] = feather_mask(bisenet_masks[k], radius=r)

                # Clean skin
                for excl_k in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'mouth_interior']:
                    if excl_k in bisenet_masks and bisenet_masks[excl_k] is not None:
                        bisenet_masks['skin'] = np.clip(bisenet_masks['skin'] - bisenet_masks[excl_k], 0.0, 1.0)

                # Build FaceRegions
                regions = FaceRegions()
                regions.skin = bisenet_masks.get('skin')
                regions.lips = bisenet_masks.get('lips')
                regions.mouth_interior = bisenet_masks.get('mouth_interior')
                regions.left_eye = bisenet_masks.get('left_eye')
                regions.right_eye = bisenet_masks.get('right_eye')
                regions.left_eyebrow = bisenet_masks.get('left_eyebrow')
                regions.right_eyebrow = bisenet_masks.get('right_eyebrow')
                regions.face_oval = bisenet_masks.get('face_oval')
                regions.neck = bisenet_masks.get('neck')
                regions.hair = bisenet_masks.get('hair')
                regions.cloth = bisenet_masks.get('cloth')

                # Handle fallback if skin is empty
                if regions.skin is None or regions.skin.max() < 0.01:
                    regions = self._landmark_fallback_only(
                        landmarks_compat_list[idx_face], crop_list[idx_face], person_masks[idx_face], ieds[idx_face]
                    )
                else:
                    self._add_landmark_subregions(regions, landmarks_compat_list[idx_face], h_img, w_img, ieds[idx_face], feather)

                results[idx_face] = regions
        except Exception as batch_err:
            logger.error(
                "Critical failure in batch parsing pipeline for %d faces (model=%s): %s. Falling back to sequential single-face parse.",
                num_faces, self._model_path, batch_err
            )
            for i in range(num_faces):
                try:
                    results[i] = self.parse(
                        landmarks_compat_list[i],
                        crop_list[i],
                        face_bbox_list[i],
                        person_masks[i],
                        ieds[i]
                    )
                except Exception as parse_err:
                    crop_shape = getattr(crop_list[i], 'shape', None)
                    logger.error(
                        "Sequential single-face fallback also failed for face %d/%d (face_bbox=%s, crop_shape=%s, model=%s): %s. Using landmark fallback.",
                        i, num_faces, face_bbox_list[i], crop_shape, self._model_path, parse_err
                    )
                    results[i] = self._landmark_fallback_only(
                        landmarks_compat_list[i], crop_list[i], person_masks[i], ieds[i]
                    )

        return results

    def _landmark_fallback_only(
        self,
        landmarks: Any,
        img_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        ied: float,
    ) -> "FaceRegions":
        """Construct face region masks using landmarks when BiSeNet fails or is bypassed."""
        h_img, w_img = img_bgr.shape[:2]
        feather = max(int(ied * 0.08), 3)
        regions = FaceRegions()

        regions.face_oval = self._mask(landmarks, FACE_OVAL, w_img, h_img, feather)
        regions.left_eye = self._mask(landmarks, LEFT_EYE, w_img, h_img, feather // 2)
        regions.right_eye = self._mask(landmarks, RIGHT_EYE, w_img, h_img, feather // 2)
        regions.left_eyebrow = self._mask(landmarks, LEFT_EYEBROW, w_img, h_img, feather // 2)
        regions.right_eyebrow = self._mask(landmarks, RIGHT_EYEBROW, w_img, h_img, feather // 2)
        regions.lips = self._mask(landmarks, LIPS_OUTER, w_img, h_img, feather // 2)
        regions.mouth_interior = self._mask(landmarks, LIPS_INNER, w_img, h_img, 0)

        # Synthesize skin
        skin = regions.face_oval.copy()
        for exclusion in [
            regions.left_eye, regions.right_eye,
            regions.left_eyebrow, regions.right_eyebrow,
            regions.lips,
        ]:
            skin = np.clip(skin - exclusion, 0, 1)

        if person_mask is not None:
            pm = normalize_mask(person_mask)
            from .utils import squeeze_mask
            pm = squeeze_mask(pm)
            skin *= pm
        regions.skin = skin
        regions.hair = np.clip(person_mask - regions.face_oval, 0, 1) if person_mask is not None else np.zeros((h_img, w_img), dtype=np.float32)
        regions.neck = np.zeros((h_img, w_img), dtype=np.float32)

        self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)
        return regions

    def _add_landmark_subregions(
        self,
        regions: "FaceRegions",
        landmarks: Any,
        h_img: int,
        w_img: int,
        ied: float,
        feather: int,
    ) -> None:
        """Construct internal/sub-region masks using landmarks."""
        try:
            regions.left_iris = self._iris_mask(landmarks, LEFT_IRIS, w_img, h_img, ied)
            regions.right_iris = self._iris_mask(landmarks, RIGHT_IRIS, w_img, h_img, ied)
        except (IndexError, AttributeError):
            regions.left_iris = np.zeros((h_img, w_img), dtype=np.float32)
            regions.right_iris = np.zeros((h_img, w_img), dtype=np.float32)

        regions.nose = self._mask(landmarks, NOSE, w_img, h_img, feather)
        regions.left_under_eye = self._mask(landmarks, LEFT_UNDER_EYE, w_img, h_img, feather)
        regions.right_under_eye = self._mask(landmarks, RIGHT_UNDER_EYE, w_img, h_img, feather)
        regions.forehead = self._forehead_mask(landmarks, w_img, h_img, feather)
        regions.left_cheek = self._mask(landmarks, LEFT_CHEEK, w_img, h_img, feather)
        regions.right_cheek = self._mask(landmarks, RIGHT_CHEEK, w_img, h_img, feather)

        regions.nose_bridge = self._mask(landmarks, [168, 6, 197, 195], w_img, h_img, feather)
        regions.forehead_center = self._circle_mask(landmarks, 151, ied * 0.25, w_img, h_img, feather)
        regions.cheek_highlights_l = self._circle_mask(landmarks, 117, ied * 0.22, w_img, h_img, feather)
        regions.cheek_highlights_r = self._circle_mask(landmarks, 346, ied * 0.22, w_img, h_img, feather)
        regions.nasolabial_l = self._mask(landmarks, LEFT_NASOLABIAL, w_img, h_img, feather)
        regions.nasolabial_r = self._mask(landmarks, RIGHT_NASOLABIAL, w_img, h_img, feather)
        regions.crows_feet_l = self._mask(landmarks, LEFT_CROWS_FEET, w_img, h_img, feather)
        regions.crows_feet_r = self._mask(landmarks, RIGHT_CROWS_FEET, w_img, h_img, feather)
        regions.jawline_contour = np.clip(
            self._mask(landmarks, [234, 127, 93, 132, 58, 172, 136, 150], w_img, h_img, feather) +
            self._mask(landmarks, [454, 323, 361, 288, 397, 365, 379, 378], w_img, h_img, feather),
            0, 1
        )

        # For sub-skin highlights/contours, intersect them with the skin mask to be perfectly clean.
        # The forehead band sits above the skin region (hairline), so it must be clipped to the
        # broader face_oval (which includes the upper-face label) rather than skin, else it vanishes.
        _face_oval = getattr(regions, 'face_oval', None)
        for attr in ['nose_bridge', 'forehead_center', 'cheek_highlights_l', 'cheek_highlights_r', 'jawline_contour', 'left_under_eye', 'right_under_eye', 'left_cheek', 'right_cheek', 'forehead', 'nasolabial_l', 'nasolabial_r', 'crows_feet_l', 'crows_feet_r']:
            val = getattr(regions, attr)
            if val is None:
                continue
            if attr == 'forehead':
                # Forehead band sits above the skin region; the landmark polygon is
                # already a clean shape, so keep it unclipped (wrinkle_soften excludes
                # hair/eyebrows/eyes internally).
                setattr(regions, attr, val)
            else:
                setattr(regions, attr, val * regions.skin)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask(
        self,
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        pts = get_points(landmarks, indices, w, h)
        return create_polygon_mask(pts, (h, w), feather_radius=feather)

    def _circle_mask(
        self,
        landmarks: Any,
        index: int,
        radius_px: float,
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        """Create a circular feathered mask around a single landmark."""
        lm = landmarks.landmark[index]
        cx, cy = int(lm.x * w), int(lm.y * h)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), int(radius_px), 1.0, -1)
        return feather_mask(mask, radius=feather)

    def _iris_mask(
        self,
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
        ied: float,
    ) -> np.ndarray:
        """Circle mask for iris based on iris landmarks."""
        pts = get_points(landmarks, indices, w, h)
        center = pts.mean(axis=0).astype(int)
        radius = max(int(ied * 0.07), 4)  # iris ≈ 7% of IED
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, tuple(center), radius, 1.0, -1)
        return feather_mask(mask, radius=max(radius // 3, 2))

    def _forehead_mask(
        self,
        landmarks: Any,
        w: int,
        h: int,
        feather: int,
    ) -> np.ndarray:
        """Forehead: area between face oval top and eyebrow line."""
        top_pts = get_points(landmarks, FOREHEAD_TOP, w, h)
        bot_pts = get_points(landmarks, FOREHEAD_BOTTOM, w, h)
        # Combine into a closed polygon: top_pts forward + bot_pts reversed
        poly = np.vstack([top_pts, bot_pts[::-1]])
        return create_polygon_mask(poly, (h, w), feather_radius=feather)

"""Face region parsing — generates per-region masks from MediaPipe landmarks.

Regions produced:
    skin, forehead, left_cheek, right_cheek, nose,
    left_eye, right_eye, left_eyebrow, right_eyebrow,
    left_iris, right_iris, lips, mouth_interior (for teeth),
    left_under_eye, right_under_eye, face_oval.

All masks are float32 (H, W) in [0, 1] with soft feathered edges.
"""

import cv2
import numpy as np
import os
import onnxruntime as ort
from .perf_optimizations import build_ort_providers

from .utils import get_points, create_polygon_mask, feather_mask

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
    ]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)


class FaceParser:
    """Generate precise per-region masks using BiSeNet ONNX + MediaPipe Face Mesh coordinates."""

    def __init__(self):
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

    def parse(self, landmarks, img_bgr, face_bbox, person_mask=None, ied=100.0):
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

                    bisenet_masks['face_oval'] = (
                        (full_label_map == 1) | (full_label_map == 2) | (full_label_map == 3) |
                        (full_label_map == 4) | (full_label_map == 5) | (full_label_map == 10) |
                        (full_label_map == 11) | (full_label_map == 12) | (full_label_map == 13)
                    ).astype(np.float32)

                    # Apply feathering
                    for k in ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'face_oval', 'neck', 'hair']:
                        if k in bisenet_masks:
                            r = feather // 2 if k in ['left_eye', 'right_eye', 'lips', 'left_eyebrow', 'right_eyebrow'] else feather
                            bisenet_masks[k] = feather_mask(bisenet_masks[k], radius=r)

                    # Clean skin mask after feathering
                    for excl_k in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'mouth_interior']:
                        if excl_k in bisenet_masks and bisenet_masks[excl_k] is not None:
                            bisenet_masks['skin'] = np.clip(bisenet_masks['skin'] - bisenet_masks[excl_k], 0.0, 1.0)
            except Exception:
                pass

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

        # Fallback to landmarks if BiSeNet failed or has empty skin
        if regions.skin is None or regions.skin.max() < 0.01:
            regions = self._landmark_fallback_only(landmarks, img_bgr, person_mask, ied)
        else:
            self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)

        return regions

    def parse_batch(self, crop_list, landmarks_compat_list, face_bbox_list, person_masks, ieds):
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
            except Exception:
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
                    import logging
                    logging.getLogger(__name__).warning(
                        "Batch inference failed (likely due to CoreML dynamic batch limits): %s. "
                        "Falling back to sequential inference (batch size 1).", e
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
                for k in ['skin', 'left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'lips', 'face_oval', 'neck', 'hair']:
                    if k in bisenet_masks:
                        r = feather // 2 if k in ['left_eye', 'right_eye', 'lips', 'left_eyebrow', 'right_eyebrow'] else feather
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

                # Handle fallback if skin is empty
                if regions.skin is None or regions.skin.max() < 0.01:
                    regions = self._landmark_fallback_only(
                        landmarks_compat_list[idx_face], crop_list[idx_face], person_masks[idx_face], ieds[idx_face]
                    )
                else:
                    self._add_landmark_subregions(regions, landmarks_compat_list[idx_face], h_img, w_img, ieds[idx_face], feather)

                results[idx_face] = regions
        except Exception as batch_err:
            import logging
            logging.getLogger(__name__).error(
                "Critical failure in batch parsing pipeline: %s. Falling back to sequential single-face parse.",
                batch_err
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
                    logging.getLogger(__name__).error(
                        "Sequential single-face fallback also failed for face %d: %s. Using landmark fallback.",
                        i, parse_err
                    )
                    results[i] = self._landmark_fallback_only(
                        landmarks_compat_list[i], crop_list[i], person_masks[i], ieds[i]
                    )

        return results

    def _landmark_fallback_only(self, landmarks, img_bgr, person_mask, ied):
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
            pm = person_mask.astype(np.float32)
            if pm.max() > 1.0:
                pm /= 255.0
            if pm.ndim == 3:
                pm = pm[:, :, 0]
            skin *= pm
        regions.skin = skin
        regions.hair = np.clip(person_mask - regions.face_oval, 0, 1) if person_mask is not None else np.zeros((h_img, w_img), dtype=np.float32)
        regions.neck = np.zeros((h_img, w_img), dtype=np.float32)

        self._add_landmark_subregions(regions, landmarks, h_img, w_img, ied, feather)
        return regions

    def _add_landmark_subregions(self, regions, landmarks, h_img, w_img, ied, feather):
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
        regions.jawline_contour = np.clip(
            self._mask(landmarks, [234, 127, 93, 132, 58, 172, 136, 150], w_img, h_img, feather) +
            self._mask(landmarks, [454, 323, 361, 288, 397, 365, 379, 378], w_img, h_img, feather),
            0, 1
        )

        # For sub-skin highlights/contours, intersect them with the skin mask to be perfectly clean
        for attr in ['nose_bridge', 'forehead_center', 'cheek_highlights_l', 'cheek_highlights_r', 'jawline_contour', 'left_under_eye', 'right_under_eye', 'left_cheek', 'right_cheek', 'forehead']:
            val = getattr(regions, attr)
            if val is not None:
                setattr(regions, attr, val * regions.skin)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask(self, landmarks, indices, w, h, feather):
        pts = get_points(landmarks, indices, w, h)
        return create_polygon_mask(pts, (h, w), feather_radius=feather)

    def _circle_mask(self, landmarks, index, radius_px, w, h, feather):
        """Create a circular feathered mask around a single landmark."""
        lm = landmarks.landmark[index]
        cx, cy = int(lm.x * w), int(lm.y * h)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), int(radius_px), 1.0, -1)
        return feather_mask(mask, radius=feather)

    def _iris_mask(self, landmarks, indices, w, h, ied):
        """Circle mask for iris based on iris landmarks."""
        pts = get_points(landmarks, indices, w, h)
        center = pts.mean(axis=0).astype(int)
        radius = max(int(ied * 0.07), 4)  # iris ≈ 7% of IED
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, tuple(center), radius, 1.0, -1)
        return feather_mask(mask, radius=max(radius // 3, 2))

    def _forehead_mask(self, landmarks, w, h, feather):
        """Forehead: area between face oval top and eyebrow line."""
        top_pts = get_points(landmarks, FOREHEAD_TOP, w, h)
        bot_pts = get_points(landmarks, FOREHEAD_BOTTOM, w, h)
        # Combine into a closed polygon: top_pts forward + bot_pts reversed
        poly = np.vstack([top_pts, bot_pts[::-1]])
        return create_polygon_mask(poly, (h, w), feather_radius=feather)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from pathlib import Path

_MODEL_PATH = str(
    Path(__file__).parent / "face_landmarker_v2_with_blendshapes.task"
)

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=10,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


_LANDMARK_INDICES = {
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
                 160, 161, 246],
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
                  386, 387, 388, 466],
    "left_iris": [468, 469, 470, 471],
    "right_iris": [472, 473, 474, 475],
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
                   269, 267, 0, 37, 39, 40, 185],
    "lips_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318,
                   402, 317, 14, 87, 178, 88, 95],
}


def _lm(l, idx):
    return l[idx]


def _landmarks_to_points(landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = _lm(landmarks, i)
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def _mask_from_points(shape, points, feather=15):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def _inter_eye_distance(landmarks, w, h):
    left = _lm(landmarks, 33)
    right = _lm(landmarks, 263)
    dx = (right.x - left.x) * w
    dy = (right.y - left.y) * h
    return np.sqrt(dx * dx + dy * dy)


def _face_skin_mask(shape, landmarks, w, h):
    oval = _landmarks_to_points(landmarks, _LANDMARK_INDICES["face_oval"], w, h)
    le = _landmarks_to_points(landmarks, _LANDMARK_INDICES["left_eye"], w, h)
    re = _landmarks_to_points(landmarks, _LANDMARK_INDICES["right_eye"], w, h)
    lo = _landmarks_to_points(landmarks, _LANDMARK_INDICES["lips_outer"], w, h)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [oval], 255)
    cv2.fillPoly(mask, [le], 0)
    cv2.fillPoly(mask, [re], 0)
    cv2.fillPoly(mask, [lo], 0)

    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    return mask.astype(np.float32) / 255.0


def _frequency_separate(img, radius):
    low = cv2.GaussianBlur(img, (0, 0), radius)
    high = cv2.subtract(img.astype(np.int16), low.astype(np.int16)) + 128
    high = np.clip(high, 0, 255).astype(np.uint8)
    return low, high


def _frequency_recombine(low, high):
    result = cv2.add(low.astype(np.int16), high.astype(np.int16)) - 128
    return np.clip(result, 0, 255).astype(np.uint8)


def _dodge_burn(img_bgr, landmarks, w, h, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    left_eye = _landmarks_to_points(landmarks, _LANDMARK_INDICES["left_eye"], w, h)
    right_eye = _landmarks_to_points(landmarks, _LANDMARK_INDICES["right_eye"], w, h)

    for eye_pts in [left_eye, right_eye]:
        cx = int(np.mean(eye_pts[:, 0]))
        cy = int(np.mean(eye_pts[:, 1]))
        rx = int(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])) + 10
        ry = int(np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])) + 10
        under_cy = cy + ry

        mask = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        cv2.ellipse(mask, (cx, under_cy), (rx, ry // 2), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(rx, ry) // 2)

        lab[:, :, 0] = np.clip(lab[:, :, 0] + mask * 15 * s, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - mask * 8 * s, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _enhance_eyes_mediapipe(img_bgr, landmarks, w, h, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0
    result = img_bgr.copy()

    for eye_key, iris_key in [("left_eye", "left_iris"), ("right_eye", "right_iris")]:
        eye_pts = _landmarks_to_points(landmarks, _LANDMARK_INDICES[eye_key], w, h)
        iris_pts = _landmarks_to_points(landmarks, _LANDMARK_INDICES[iris_key], w, h)

        cx = int(np.mean(iris_pts[:, 0]))
        cy = int(np.mean(iris_pts[:, 1]))
        iris_radius = max(
            int(np.max(iris_pts[:, 0]) - np.min(iris_pts[:, 0])) // 2 + 2, 5
        )
        eye_radius = max(
            int(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])) // 2 + 5, 10
        )

        iris_mask = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        cv2.circle(iris_mask, (cx, cy), iris_radius, 1, -1)
        iris_mask = cv2.GaussianBlur(iris_mask, (0, 0), iris_radius // 2)

        sclera_mask = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        cv2.circle(sclera_mask, (cx, cy), eye_radius, 1, -1)
        sclera_mask = np.clip(sclera_mask - iris_mask, 0, 1)
        sclera_mask = cv2.GaussianBlur(sclera_mask, (0, 0), 3)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                          dtype=np.float32)
        sharpened = cv2.filter2D(result, -1, kernel)

        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 - iris_mask * s) +
                sharpened[:, :, c] * iris_mask * s,
                0, 255
            ).astype(np.uint8)

        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + iris_mask * 30 * s, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        lab2 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2[:, :, 0] = np.clip(lab2[:, :, 0] + sclera_mask * 20 * s, 0, 255)
        result = cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result


def _enhance_lips(img_bgr, landmarks, w, h, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0

    outer = _landmarks_to_points(landmarks, _LANDMARK_INDICES["lips_outer"], w, h)
    inner = _landmarks_to_points(landmarks, _LANDMARK_INDICES["lips_inner"], w, h)

    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outer], 255)
    cv2.fillPoly(mask, [inner], 0)
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    mask_f = mask.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + mask_f * 40 * s, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    result = img_bgr * (1 - mask_f[..., None]) + saturated * mask_f[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def _whiten_skin_lab(img_bgr, skin_mask, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + skin_mask * 20 * s, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] - skin_mask * 8 * s, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] - skin_mask * 8 * s, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _adjust_contrast(img, contrast):
    if contrast == 0:
        return img
    f = (259 * (contrast + 255)) / (255 * (259 - contrast))
    return np.clip(f * (img.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)


def detect_faces(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_landmarker()
    result = landmarker.detect(mp_img)
    return result.face_landmarks if result and result.face_landmarks else []


def retouch(
    img_bgr,
    smooth=50,
    whiten=30,
    eye_enhance=30,
    contrast=0,
    lip_enhance=0,
    dark_circles=0,
    no_texture=False,
):
    result = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    faces = detect_faces(img_bgr)

    if faces:
        combined_skin_mask = np.zeros((h, w), dtype=np.float32)

        for face_landmarks in faces:
            skin_mask = _face_skin_mask(img_bgr.shape, face_landmarks, w, h)
            combined_skin_mask = np.maximum(combined_skin_mask, skin_mask)

            dist = _inter_eye_distance(face_landmarks, w, h)
            rad = max(int(dist / 20), 3)

            if eye_enhance > 0:
                result = _enhance_eyes_mediapipe(
                    result, face_landmarks, w, h, eye_enhance
                )

            if lip_enhance > 0:
                result = _enhance_lips(
                    result, face_landmarks, w, h, lip_enhance
                )

            if dark_circles > 0:
                result = _dodge_burn(
                    result, face_landmarks, w, h, dark_circles
                )

            if smooth > 0:
                face_skin = _face_skin_mask(img_bgr.shape, face_landmarks, w, h)
                low, high = _frequency_separate(result, rad)
                smooth_strength = smooth / 100.0

                smoothed_low = cv2.bilateralFilter(
                    low, int(5 + smooth_strength * 25),
                    int(10 + smooth_strength * 90),
                    int(10 + smooth_strength * 90)
                )

                if no_texture:
                    result = result * (1 - face_skin[..., None]) + \
                             smoothed_low * face_skin[..., None]
                else:
                    recombined = _frequency_recombine(smoothed_low, high)
                    result = result * (1 - face_skin[..., None]) + \
                             recombined * face_skin[..., None]

                result = np.clip(result, 0, 255).astype(np.uint8)

        if whiten > 0:
            result = _whiten_skin_lab(result, combined_skin_mask, whiten)

    else:
        if smooth > 0:
            low = cv2.GaussianBlur(result, (0, 0), 15)
            high = cv2.subtract(result.astype(np.int16), low.astype(np.int16)) + 128
            high = np.clip(high, 0, 255).astype(np.uint8)
            smoothed_low = cv2.bilateralFilter(low, 15, 50, 50)
            recombined = _frequency_recombine(smoothed_low, high)
            result = recombined

        if whiten > 0:
            fake_mask = np.ones((h, w), dtype=np.float32) * 0.3
            result = _whiten_skin_lab(result, fake_mask, whiten)

    if contrast != 0:
        result = _adjust_contrast(result, contrast)

    return result

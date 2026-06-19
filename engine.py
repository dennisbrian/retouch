import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from pathlib import Path
from dataclasses import dataclass

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


@dataclass
class _FaceData:
    landmarks: object
    w: int
    h: int
    skin_mask: np.ndarray
    ied: float
    rad: int
    points: dict


def _landmarks_to_points(landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = landmarks.landmark[i]
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
    left = landmarks.landmark[33]
    right = landmarks.landmark[263]
    dx = (right.x - left.x) * w
    dy = (right.y - left.y) * h
    return np.sqrt(dx * dx + dy * dy)


def _face_skin_mask(shape, points, w, h):
    oval = points["face_oval"]
    le = points["left_eye"]
    re = points["right_eye"]
    lo = points["lips_outer"]

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [oval], 255)
    cv2.fillPoly(mask, [le], 0)
    cv2.fillPoly(mask, [re], 0)
    cv2.fillPoly(mask, [lo], 0)

    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    return mask.astype(np.float32) / 255.0


def _frequency_separate(img, radius):
    low = cv2.GaussianBlur(img, (0, 0), radius).astype(np.float32)
    high = img.astype(np.float32) - low
    return low, high


def _frequency_recombine(low, high):
    return np.clip(low + high, 0, 255).astype(np.uint8)


def _dodge_burn(img_bgr, face_data, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for eye_key in ["left_eye", "right_eye"]:
        eye_pts = face_data.points[eye_key]
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


def _enhance_eyes_mediapipe(img_bgr, face_data, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0
    result = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    combined_iris = np.zeros((h, w), dtype=np.float32)
    combined_sclera = np.zeros((h, w), dtype=np.float32)

    for eye_key, iris_key in [("left_eye", "left_iris"), ("right_eye", "right_iris")]:
        eye_pts = face_data.points[eye_key]
        iris_pts = face_data.points[iris_key]

        cx = int(np.mean(iris_pts[:, 0]))
        cy = int(np.mean(iris_pts[:, 1]))
        iris_radius = max(
            int(np.max(iris_pts[:, 0]) - np.min(iris_pts[:, 0])) // 2 + 2, 5
        )
        eye_radius = max(
            int(np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])) // 2 + 5, 10
        )

        iris_mask_part = np.zeros((h, w), dtype=np.float32)
        cv2.circle(iris_mask_part, (cx, cy), iris_radius, 1, -1)
        iris_mask_part = cv2.GaussianBlur(iris_mask_part, (0, 0), iris_radius // 2)
        combined_iris = np.maximum(combined_iris, iris_mask_part)

        sclera_part = np.zeros((h, w), dtype=np.float32)
        cv2.circle(sclera_part, (cx, cy), eye_radius, 1, -1)
        sclera_part = np.clip(sclera_part - iris_mask_part, 0, 1)
        sclera_part = cv2.GaussianBlur(sclera_part, (0, 0), 3)
        combined_sclera = np.maximum(combined_sclera, sclera_part)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened = cv2.filter2D(result, -1, kernel)

    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c] * (1 - combined_iris * s) +
            sharpened[:, :, c] * combined_iris * s,
            0, 255
        ).astype(np.uint8)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + combined_iris * 30 * s, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + combined_sclera * 20 * s, 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result


def _enhance_lips(img_bgr, face_data, strength):
    if strength == 0:
        return img_bgr
    s = strength / 100.0

    outer = face_data.points["lips_outer"]
    inner = face_data.points["lips_inner"]

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
    lab[:, :, 1] = np.clip(lab[:, :, 1] - skin_mask * 2 * s, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + skin_mask * 4 * s, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _adjust_contrast(img, contrast):
    if contrast == 0:
        return img
    t = contrast / 100.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    l = np.clip(0.5 + (l - 0.5) * (1.0 + t * 0.5), 0, 1)
    lab[:, :, 0] = l * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def detect_faces(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_landmarker()
    result = landmarker.detect(mp_img)
    return result.face_landmarks if result and result.face_landmarks else []


def _build_face_data(face_landmarks, img_shape, w, h):
    points = {
        k: _landmarks_to_points(face_landmarks, v, w, h)
        for k, v in _LANDMARK_INDICES.items()
    }
    skin_mask = _face_skin_mask(img_shape, points, w, h)
    ied = _inter_eye_distance(face_landmarks, w, h)
    rad = max(int(ied / 20), 3)
    return _FaceData(
        landmarks=face_landmarks,
        w=w, h=h,
        skin_mask=skin_mask,
        ied=ied, rad=rad,
        points=points,
    )


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
        face_datas = [_build_face_data(f, img_bgr.shape, w, h) for f in faces]

        combined_skin_mask = np.zeros((h, w), dtype=np.float32)
        for fd in face_datas:
            combined_skin_mask = np.maximum(combined_skin_mask, fd.skin_mask)

        if smooth > 0:
            avg_rad = max(int(np.mean([fd.rad for fd in face_datas])), 3)
            low, high = _frequency_separate(result, avg_rad)
        else:
            low = high = None

        for fd in face_datas:
            if eye_enhance > 0:
                result = _enhance_eyes_mediapipe(result, fd, eye_enhance)

            if lip_enhance > 0:
                result = _enhance_lips(result, fd, lip_enhance)

            if dark_circles > 0:
                result = _dodge_burn(result, fd, dark_circles)

            if smooth > 0:
                smooth_strength = smooth / 100.0
                smoothed_low = cv2.bilateralFilter(
                    low, int(5 + smooth_strength * 25),
                    int(10 + smooth_strength * 90),
                    int(10 + smooth_strength * 90)
                )

                if no_texture:
                    face_blend = smoothed_low
                else:
                    face_blend = _frequency_recombine(smoothed_low, high)

                result = result * (1 - fd.skin_mask[..., None]) + \
                         face_blend * fd.skin_mask[..., None]
                result = np.clip(result, 0, 255).astype(np.uint8)

        if whiten > 0:
            result = _whiten_skin_lab(result, combined_skin_mask, whiten)

    else:
        if smooth > 0:
            low = cv2.GaussianBlur(result, (0, 0), 15).astype(np.float32)
            high = result.astype(np.float32) - low
            smoothed_low = cv2.bilateralFilter(low.astype(np.uint8), 15, 50, 50)
            recombined = np.clip(smoothed_low.astype(np.float32) + high, 0, 255).astype(np.uint8)
            result = recombined

        if whiten > 0:
            fake_mask = np.ones((h, w), dtype=np.float32) * 0.3
            result = _whiten_skin_lab(result, fake_mask, whiten)

    if contrast != 0:
        result = _adjust_contrast(result, contrast)

    return result

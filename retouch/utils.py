"""Shared utilities for the face retouching pipeline."""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def feather_mask(mask, radius=None, sigma=None):
    """Apply Gaussian feathering to a mask for soft edges.

    Args:
        mask: Float mask (H, W), values 0.0–1.0.
        radius: Feather radius in pixels.
        sigma: Gaussian sigma. Derived from radius if not given.

    Returns:
        Feathered float mask (H, W), values 0.0–1.0.
    """
    if mask is None or mask.size == 0:
        return mask

    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f /= 255.0

    if sigma is None and radius is None:
        return mask_f

    if sigma is None:
        sigma = max(radius / 3.0, 1.0)
    if radius is None:
        radius = int(sigma * 3)

    ksize = max(radius * 2 + 1, 3)
    return cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)


def blend_masked(original, processed, mask):
    """Alpha-blend *processed* onto *original* using a soft mask.

    Args:
        original: (H, W, C) uint8 image.
        processed: (H, W, C) uint8 image.
        mask: (H, W) float mask, 0.0–1.0.

    Returns:
        Blended (H, W, C) uint8 image.
    """
    if mask is None:
        return processed

    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = m[:, :, np.newaxis]

    out = original.astype(np.float32) * (1.0 - m) + processed.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def create_polygon_mask(points, img_shape, feather_radius=0):
    """Create a filled-polygon mask from an ordered set of (x, y) points.

    Args:
        points: (N, 2) int32 array of pixel coordinates.
        img_shape: (H, W) or (H, W, C).
        feather_radius: Gaussian feather radius. 0 = hard edge.

    Returns:
        Float mask (H, W), 0.0–1.0.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    mask_f = mask.astype(np.float32) / 255.0
    if feather_radius > 0:
        mask_f = feather_mask(mask_f, radius=feather_radius)
    return mask_f


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def get_points(landmarks, indices, w, h):
    """Extract pixel (x, y) coords from MediaPipe landmarks.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.
        indices: List[int] of landmark indices.
        w, h: Image width and height.

    Returns:
        (N, 2) int32 numpy array.
    """
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def inter_eye_distance(landmarks, w, h):
    """Distance between iris centers (or eye-corner midpoints as fallback).

    Returns distance in pixels.
    """
    try:
        left = landmarks.landmark[468]   # left iris center
        right = landmarks.landmark[473]  # right iris center
    except (IndexError, AttributeError):
        # Fallback: midpoint of eye corners
        li, lo = landmarks.landmark[133], landmarks.landmark[33]
        ri, ro = landmarks.landmark[362], landmarks.landmark[263]
        lx, ly = (li.x + lo.x) / 2, (li.y + lo.y) / 2
        rx, ry = (ri.x + ro.x) / 2, (ri.y + ro.y) / 2
        dx, dy = (rx - lx) * w, (ry - ly) * h
        return np.sqrt(dx * dx + dy * dy)

    dx = (right.x - left.x) * w
    dy = (right.y - left.y) * h
    return np.sqrt(dx * dx + dy * dy)


def adaptive_ksize(face_width, factor=0.1, minimum=3):
    """Odd kernel size proportional to face width."""
    size = max(int(face_width * factor), minimum)
    return size | 1  # ensure odd


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def vibrance(img_bgr, mask, strength):
    """Smart saturation: boosts under-saturated pixels more.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: (H, W) float mask 0–1.
        strength: 0.0–1.0.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    if strength <= 0:
        return img_bgr

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    factor = 1.0 + strength * (1.0 - s / 255.0)
    hsv[:, :, 1] = np.clip(s * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return blend_masked(img_bgr, result, mask)


def apply_curve(channel, curve_points):
    """Apply a piecewise-linear tone curve via LUT.

    Args:
        channel: Single-channel uint8 array.
        curve_points: list of (input, output) in 0–255.

    Returns:
        Tone-mapped channel (same shape), uint8.
    """
    xs = [p[0] for p in curve_points]
    ys = [p[1] for p in curve_points]
    lut = np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)
    return cv2.LUT(channel, lut)


def correct_exposure(
    img_bgr,
    face_bboxes=None,
    target_mean=120,
    min_threshold=95,
    max_threshold=165,
    face_target_mean=145,
    face_min_threshold=105,
    face_max_threshold=185
):
    """Normalize the image exposure using LAB space and a bounded Gamma curve.
    Prioritizes face luminance if face_bboxes are provided.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        face_bboxes: List of Tuple (x, y, w, h) bounding boxes.
        target_mean: Target global mean luminance (0-255).
        min_threshold: Lower bound for global mean luminance.
        max_threshold: Upper bound for global mean luminance.
        face_target_mean: Target face mean luminance (0-255).
        face_min_threshold: Lower bound for face mean luminance.
        face_max_threshold: Upper bound for face mean luminance.

    Returns:
        ((H, W, 3) uint8 BGR image, was_corrected bool).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    use_face = False
    current_mean = 0.0

    if face_bboxes is not None and len(face_bboxes) > 0:
        # Sort face bboxes by area to find the largest (primary) face
        try:
            sorted_boxes = sorted(face_bboxes, key=lambda b: b[2] * b[3], reverse=True)
            primary_box = sorted_boxes[0]
            xf, yf, wf, hf = primary_box
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = max(0, int(xf)), max(0, int(yf)), min(w, int(xf+wf)), min(h, int(yf+hf))
            face_crop = l_chan[y1:y2, x1:x2]
            if face_crop.size > 0:
                current_mean = np.mean(face_crop)
                use_face = True
        except Exception:
            pass

    if not use_face:
        current_mean = np.mean(l_chan)
        t_min = min_threshold
        t_max = max_threshold
        t_mean = target_mean
    else:
        t_min = face_min_threshold
        t_max = face_max_threshold
        t_mean = face_target_mean

    # Only adjust if exposure falls outside the safe range [t_min, t_max]
    if current_mean < t_min or current_mean > t_max:
        if current_mean < 1.0:
            current_mean = 1.0
        elif current_mean > 254.0:
            current_mean = 254.0

        # Calculate gamma using log ratio
        # L_target / 255.0 = (L_current / 255.0)^gamma
        # gamma = ln(L_target / 255.0) / ln(L_current / 255.0)
        log_target = np.log(t_mean / 255.0)
        log_current = np.log(current_mean / 255.0)
        gamma = log_target / log_current

        # Safety bound the gamma shift to prevent extreme artifacts
        gamma = np.clip(gamma, 0.75, 1.25)

        # Apply gamma curve
        l_norm = l_chan / 255.0
        l_new = np.power(l_norm, gamma) * 255.0
        lab[:, :, 0] = np.clip(l_new, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), True

    return img_bgr, False

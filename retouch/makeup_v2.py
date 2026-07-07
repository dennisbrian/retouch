"""Makeup engine v2 — eyeshadow, eyeliner, contour, brows, ombre lips.

Builds on the original MakeupEngine with five new cosmetic operations,
all of which:

* Work in LAB space for natural, tone-preserving color blending.
* Accept uint8 or float32 BGR [0, 255] input (dtype-aware).
* Use feathered polygon masks built from MediaPipe landmarks so the
  makeup boundaries stay soft and never reveal the polygon silhouette.
* Preserve high-frequency skin detail by blending *onto* the original
  rather than blurring the result.

All public ``apply_*`` methods are no-ops when their ``strength``/opacity
argument is zero.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .parsing import (
    FACE_OVAL,
    LEFT_EYE,
    LEFT_EYEBROW,
    LEFT_UNDER_EYE,
    LIPS_INNER,
    LIPS_OUTER,
    NOSE,
    RIGHT_EYE,
    RIGHT_EYEBROW,
    RIGHT_UNDER_EYE,
)
from .utils import (
    apply_u8_op_float,
    bgr_f32_to_lab_f32,
    create_polygon_mask,
    feather_mask,
    get_points,
    lab_f32_to_bgr_f32,
    normalize_mask,
)


# Predefined makeup color palettes (BGR tuples in 0-255).
EYESHADOW_COLORS = {
    "nude":     (180, 170, 200),
    "rose":     (170, 140, 200),
    "smoky":    (90, 80, 110),
    "brown":    (90, 110, 150),
    "plum":     (110, 70, 130),
    "gold":     (90, 170, 210),
}

EYELINER_COLORS = {
    "black": (0, 0, 0),
    "brown": (40, 50, 70),
    "navy":  (50, 50, 20),
}

BROW_COLORS = {
    "black":  (30, 30, 30),
    "brown":  (50, 70, 100),
    "auburn": (40, 60, 110),
    "blonde": (110, 130, 160),
}


# =====================================================================
# Internal helpers
# =====================================================================


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert BGR [0,255] to float32 LAB (uint8-scale: L[0,255], a/b@128)."""
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert float32 LAB (uint8-scale) back to BGR matching input dtype."""
    clipped = np.clip(lab, 0.0, 255.0)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _resolve_color(
    color: Union[str, Tuple[int, int, int]],
    palette: dict,
    default_name: str,
) -> Tuple[int, int, int]:
    """Resolve a color spec (string key or BGR tuple) against a palette."""
    if isinstance(color, str):
        return palette.get(color, palette[default_name])
    return (int(color[0]), int(color[1]), int(color[2]))


def _lab_color_shift(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    strength: float,
) -> np.ndarray:
    """Blend a solid BGR color into *img_bgr* inside *mask* using soft-light in LAB.

    The blend preserves the L (luminance) structure of the underlying skin
    while shifting the a/b chroma channels toward the target color. This is
    the same approach used by the v1 blush engine and keeps skin texture
    intact.
    """
    is_float = img_bgr.dtype == np.float32
    lab = _to_lab(img_bgr, is_float)

    # Compute the target a/b from the target color.
    color_bgr_u8 = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
    target_lab = cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_a = float(target_lab[0, 0, 1])
    target_b = float(target_lab[0, 0, 2])

    m = mask * strength  # (H, W)
    m3 = m[:, :, np.newaxis]

    # Shift a/b channels toward the target color.
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - m) + target_a * m
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - m) + target_b * m

    # Subtle luminance nudge toward the color's L for natural depth.
    target_l = float(target_lab[0, 0, 0])
    lab[:, :, 0] = lab[:, :, 0] * (1.0 - m * 0.3) + target_l * (m * 0.3)

    return _from_lab(lab, is_float)


def _soft_light_tint(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    strength: float,
) -> np.ndarray:
    """Soft-light blend of a solid color layer, gated by mask*strength.

    Used for opaque makeup like eyeliner/brows where the underlying skin
    texture should be replaced more strongly than a pure LAB chroma shift.
    """
    is_float = img_bgr.dtype == np.float32

    img_f = img_bgr.astype(np.float32)
    layer = np.full_like(img_f, color_bgr, dtype=np.float32)

    # Pegtop soft-light in [0,1].
    base = img_f / 255.0
    lay = layer / 255.0
    blended = (1.0 - 2.0 * lay) * base * base + 2.0 * lay * base
    blended = np.clip(blended * 255.0, 0.0, 255.0)

    m = mask * strength
    if m.ndim == 2:
        m = m[:, :, np.newaxis]

    out = img_f * (1.0 - m) + blended * m
    if is_float:
        return np.clip(out, 0.0, 255.0).astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _lm_xy(landmarks: Any, idx: int, w: int, h: int) -> Tuple[float, float]:
    """Return (x, y) in pixels for a single landmark index."""
    lm = landmarks.landmark[idx]
    return (float(lm.x * w), float(lm.y * h))


def _feather_polygon(
    pts: np.ndarray,
    shape: Tuple[int, int],
    feather: int,
) -> np.ndarray:
    """Create a filled polygon mask with Gaussian feathering."""
    return create_polygon_mask(pts, shape, feather_radius=feather)


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a float32 [0,1] mask by *iterations* using a small kernel."""
    if iterations <= 0:
        return mask
    k = np.ones((3, 3), np.uint8)
    m_u8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
    dilated = cv2.dilate(m_u8, k, iterations=iterations)
    return dilated.astype(np.float32) / 255.0


def _gaussian_blur_safe(mask: np.ndarray, ksize: int) -> np.ndarray:
    """GaussianBlur with guaranteed odd ksize >= 3."""
    k = max(int(ksize), 3) | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


# =====================================================================
# MakeupEngineV2
# =====================================================================


class MakeupEngineV2:
    """Professional makeup engine: eyeshadow, liner, contour, brows, ombre lips.

    All methods accept ``img_bgr`` (uint8 or float32 BGR [0,255]) and
    ``landmarks`` (MediaPipe NormalizedLandmarkList). They return an image
    of the same dtype/shape.
    """

    # ------------------------------------------------------------------
    # Eyeshadow
    # ------------------------------------------------------------------

    def apply_eyeshadow(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        color: Union[str, Tuple[int, int, int]] = "rose",
        strength: int = 40,
        style: str = "natural",
    ) -> np.ndarray:
        """Apply eyeshadow above the upper eyelid.

        Builds a soft band between the upper eye contour and the eyebrow,
        dilated upward and feathered so the gradient fades into the brow
        bone. Two styles are supported:

        * ``"natural"`` — soft LAB chroma shift toward the target color.
        * ``"smoky"``   — same chroma shift plus a darkening pass on the
          outer corner for a smoky gradient.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            landmarks: MediaPipe NormalizedLandmarkList.
            color: Palette key (see EYESHADOW_COLORS) or BGR tuple.
            strength: 0-100 intensity.
            style: "natural" or "smoky".
        """
        if strength <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.apply_eyeshadow,
                landmarks,
                color,
                strength,
                style,
            )

        h, w = img_bgr.shape[:2]
        s = strength / 100.0
        color_bgr = _resolve_color(color, EYESHADOW_COLORS, "rose")

        # Upper-lid contour: the top arc of each eye (indices 158..163 for
        # left, 385..390 for right in MediaPipe FaceMesh). We use the full
        # eye polygon and extract its upper half by combining with the
        # eyebrow polygon — the band between eye-top and brow-bottom is
        # the eyeshadow zone.
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        for eye_idx, brow_idx in (
            (LEFT_EYE, LEFT_EYEBROW),
            (RIGHT_EYE, RIGHT_EYEBROW),
        ):
            shadow_mask = np.maximum(
                shadow_mask,
                self._eyeshadow_band(landmarks, eye_idx, brow_idx, w, h),
            )

        if shadow_mask.max() < 0.01:
            return img_bgr

        # Feather heavily for a soft gradient.
        feather_k = max(int(min(h, w) * 0.04), 5) | 1
        shadow_mask = _gaussian_blur_safe(shadow_mask, feather_k)

        # LAB chroma shift toward the target color.
        result = _lab_color_shift(img_bgr, shadow_mask, color_bgr, s * 0.7)

        # Smoky style: darken the outer third for depth.
        if style == "smoky":
            result = self._smoky_pass(result, landmarks, shadow_mask, s)

        return result

    def _eyeshadow_band(
        self,
        landmarks: Any,
        eye_idx: Sequence[int],
        brow_idx: Sequence[int],
        w: int,
        h: int,
    ) -> np.ndarray:
        """Build the eyeshadow band mask for one eye (between eye-top and brow)."""
        eye_pts = get_points(landmarks, eye_idx, w, h)
        brow_pts = get_points(landmarks, brow_idx, w, h)

        # Upper eye contour: top half of the eye polygon (highest y = lowest row).
        eye_top = eye_pts[eye_pts[:, 1].argsort()[: len(eye_pts) // 2]]
        # Lower brow contour: bottom half of the brow polygon.
        brow_bot = brow_pts[brow_pts[:, 1].argsort()[len(brow_pts) // 2:]]

        # The band polygon: eye-top left→right, then brow-bottom right→left.
        band = np.vstack([eye_top, brow_bot[::-1]]).astype(np.int32)
        mask = _feather_polygon(band, (h, w), feather=max(3, int(min(h, w) * 0.01)))

        # Dilate upward slightly so the shadow extends past the polygon edge.
        mask = _dilate_mask(mask, iterations=1)
        return mask

    def _smoky_pass(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        shadow_mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Darken the outer corner region for a smoky gradient."""
        is_float = img_bgr.dtype == np.float32
        h, w = img_bgr.shape[:2]

        # Outer corners: landmark 33 (left eye outer) and 263 (right eye outer).
        outer_mask = np.zeros((h, w), dtype=np.float32)
        for idx in (33, 263):
            try:
                cx, cy = _lm_xy(landmarks, idx, w, h)
                radius = max(int(min(h, w) * 0.04), 5)
                cv2.circle(outer_mask, (int(cx), int(cy)), radius, 1.0, -1)
            except (IndexError, AttributeError):
                continue

        outer_mask = _gaussian_blur_safe(outer_mask, max(int(min(h, w) * 0.03), 5) | 1)
        darken_mask = np.clip(shadow_mask * outer_mask, 0.0, 1.0) * (strength * 0.5)

        lab = _to_lab(img_bgr, is_float)
        lab[:, :, 0] = np.clip(lab[:, :, 0] - darken_mask * 40.0, 0.0, 255.0)
        return _from_lab(lab, is_float)

    # ------------------------------------------------------------------
    # Eyeliner
    # ------------------------------------------------------------------

    def apply_eyeliner(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        color: Union[str, Tuple[int, int, int]] = "black",
        thickness: int = 2,
        style: str = "classic",
    ) -> np.ndarray:
        """Apply eyeliner along the upper lash line.

        Draws a tapered stroke along the upper eyelid contour with optional
        wing extension at the outer corner.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            landmarks: MediaPipe NormalizedLandmarkList.
            color: Palette key (see EYELINER_COLORS) or BGR tuple.
            thickness: Line thickness in pixels (scaled with face size).
            style: "classic" (plain lash line) or "wing" (outer wing flick).
        """
        if thickness <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.apply_eyeliner,
                landmarks,
                color,
                thickness,
                style,
            )

        h, w = img_bgr.shape[:2]
        color_bgr = _resolve_color(color, EYELINER_COLORS, "black")

        # Scale thickness with face size if it looks like a percentage.
        t = max(int(thickness), 1)

        liner_mask = np.zeros((h, w), dtype=np.float32)
        for eye_idx, outer_idx, inner_idx in (
            (LEFT_EYE, 33, 133),
            (RIGHT_EYE, 263, 362),
        ):
            liner_mask = np.maximum(
                liner_mask,
                self._lash_line_mask(landmarks, eye_idx, outer_idx, inner_idx, w, h, t, style),
            )

        if liner_mask.max() < 0.01:
            return img_bgr

        # Feather the liner edges very slightly for a painted, not stamped, look.
        liner_mask = _gaussian_blur_safe(liner_mask, 3)

        return _soft_light_tint(img_bgr, liner_mask, color_bgr, 0.85)

    def _lash_line_mask(
        self,
        landmarks: Any,
        eye_idx: Sequence[int],
        outer_idx: int,
        inner_idx: int,
        w: int,
        h: int,
        thickness: int,
        style: str,
    ) -> np.ndarray:
        """Build the eyeliner stroke mask for one eye."""
        pts = get_points(landmarks, eye_idx, w, h)
        # Upper lash line: the top half of the eye contour.
        upper = pts[pts[:, 1].argsort()[: len(pts) // 2]]
        upper = upper[upper[:, 0].argsort()]  # sort by x for a clean stroke

        mask = np.zeros((h, w), dtype=np.float32)
        if len(upper) < 2:
            return mask

        # Draw the lash line as an anti-aliased polyline.
        cv2.polylines(
            mask.reshape(h, w, 1).astype(np.uint8) if mask.dtype == np.float32 else mask,
            [upper.reshape(-1, 1, 2).astype(np.int32)],
            False,
            255,
            max(thickness, 1),
            cv2.LINE_AA,
        )
        # polylines needs uint8; convert back.
        mask = (mask > 0).astype(np.float32)
        # Redraw properly on uint8 then normalize.
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(
            mask_u8,
            [upper.reshape(-1, 1, 2).astype(np.int32)],
            False,
            255,
            max(thickness, 1),
            cv2.LINE_AA,
        )
        mask = mask_u8.astype(np.float32) / 255.0

        # Optional wing: extend from the outer corner outward and slightly up.
        if style == "wing":
            try:
                ox, oy = _lm_xy(landmarks, outer_idx, w, h)
                # Wing endpoint: 8% of face width outward, 4% up.
                wing_len = max(int(min(h, w) * 0.08), 4)
                wx = int(ox + wing_len if outer_idx == 33 else ox - wing_len)
                wy = int(oy - wing_len * 0.5)
                wing_pts = np.array([[int(ox), int(oy)], [wx, wy]], dtype=np.int32)
                wing_u8 = np.zeros((h, w), dtype=np.uint8)
                cv2.polylines(
                    wing_u8,
                    [wing_pts.reshape(-1, 1, 2)],
                    False,
                    255,
                    max(thickness, 1),
                    cv2.LINE_AA,
                )
                mask = np.maximum(mask, wing_u8.astype(np.float32) / 255.0)
            except (IndexError, AttributeError):
                pass

        return mask

    # ------------------------------------------------------------------
    # Contour
    # ------------------------------------------------------------------

    def apply_contour(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        strength: int = 30,
    ) -> np.ndarray:
        """Apply face contouring: cheekbones, jaw, and nose sides.

        Uses a darkening pass (cool taupe) on the hallows and a brightening
        pass on the cheekbone highlight. All shifts happen in LAB L channel
        only, so skin tone stays natural.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            landmarks: MediaPipe NormalizedLandmarkList.
            strength: 0-100 contour intensity.
        """
        if strength <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.apply_contour,
                landmarks,
                strength,
            )

        h, w = img_bgr.shape[:2]
        s = strength / 100.0

        is_float = img_bgr.dtype == np.float32
        lab = _to_lab(img_bgr, is_float)

        # --- Darken: cheekbone hollows + jaw sides + nose sides ---
        darken_mask = np.zeros((h, w), dtype=np.float32)
        darken_mask = np.maximum(
            darken_mask,
            self._cheekbone_hollow_mask(landmarks, w, h),
        )
        darken_mask = np.maximum(
            darken_mask,
            self._jaw_sides_mask(landmarks, w, h),
        )
        darken_mask = np.maximum(
            darken_mask,
            self._nose_side_mask(landmarks, w, h),
        )

        feather_k = max(int(min(h, w) * 0.03), 5) | 1
        darken_mask = _gaussian_blur_safe(darken_mask, feather_k)

        # Cool taupe shift: darken L, nudge a toward green (cooler).
        lab[:, :, 0] = np.clip(lab[:, :, 0] - darken_mask * 25.0 * s, 0.0, 255.0)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - darken_mask * 6.0 * s, 0.0, 255.0)

        # --- Brighten: cheekbone highlight + nose bridge ---
        highlight_mask = np.zeros((h, w), dtype=np.float32)
        highlight_mask = np.maximum(
            highlight_mask,
            self._cheekbone_highlight_mask(landmarks, w, h),
        )
        highlight_mask = np.maximum(
            highlight_mask,
            self._nose_bridge_highlight_mask(landmarks, w, h),
        )
        highlight_mask = _gaussian_blur_safe(highlight_mask, feather_k)

        lab[:, :, 0] = np.clip(lab[:, :, 0] + highlight_mask * 18.0 * s, 0.0, 255.0)

        return _from_lab(lab, is_float)

    def _cheekbone_hollow_mask(
        self, landmarks: Any, w: int, h: int,
    ) -> np.ndarray:
        """Hollow under the cheekbone — runs from ear toward mouth corner."""
        # Use landmarks 116/345 (under-cheek) and 50/280 (jaw near ear).
        try:
            indices = [116, 117, 118, 50, 205, 425, 344, 345, 346, 280]
            pts = get_points(landmarks, indices, w, h)
        except (IndexError, AttributeError):
            return np.zeros((h, w), dtype=np.float32)
        return _feather_polygon(pts, (h, w), feather=max(3, int(min(h, w) * 0.02)))

    def _jaw_sides_mask(self, landmarks: Any, w: int, h: int) -> np.ndarray:
        """Sides of the jaw (below cheekbone to chin line)."""
        try:
            left_pts = get_points(landmarks, [172, 136, 150, 149, 176], w, h)
            right_pts = get_points(landmarks, [397, 365, 379, 378, 400], w, h)
        except (IndexError, AttributeError):
            return np.zeros((h, w), dtype=np.float32)
        m1 = _feather_polygon(left_pts, (h, w), feather=max(3, int(min(h, w) * 0.02)))
        m2 = _feather_polygon(right_pts, (h, w), feather=max(3, int(min(h, w) * 0.02)))
        return np.maximum(m1, m2)

    def _nose_side_mask(self, landmarks: Any, w: int, h: int) -> np.ndarray:
        """Two thin stripes along the sides of the nose bridge."""
        try:
            left_pts = get_points(landmarks, [220, 115, 48, 64], w, h)
            right_pts = get_points(landmarks, [439, 344, 278, 294], w, h)
        except (IndexError, AttributeError):
            return np.zeros((h, w), dtype=np.float32)
        m1 = _feather_polygon(left_pts, (h, w), feather=max(2, int(min(h, w) * 0.01)))
        m2 = _feather_polygon(right_pts, (h, w), feather=max(2, int(min(h, w) * 0.01)))
        return np.maximum(m1, m2)

    def _cheekbone_highlight_mask(
        self, landmarks: Any, w: int, h: int,
    ) -> np.ndarray:
        """Highlight on top of the cheekbone (landmark 116/345 area, raised)."""
        try:
            cx_l, cy_l = _lm_xy(landmarks, 116, w, h)
            cx_r, cy_r = _lm_xy(landmarks, 345, w, h)
        except (IndexError, AttributeError):
            return np.zeros((h, w), dtype=np.float32)
        radius = max(int(min(h, w) * 0.04), 4)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (int(cx_l), int(cy_l - radius * 0.5)), radius, 1.0, -1)
        cv2.circle(mask, (int(cx_r), int(cy_r - radius * 0.5)), radius, 1.0, -1)
        return mask

    def _nose_bridge_highlight_mask(
        self, landmarks: Any, w: int, h: int,
    ) -> np.ndarray:
        """Thin highlight stripe down the nose bridge center."""
        try:
            top = _lm_xy(landmarks, 168, w, h)
            bot = _lm_xy(landmarks, 2, w, h)
        except (IndexError, AttributeError):
            return np.zeros((h, w), dtype=np.float32)
        cx1, cy1 = int(top[0]), int(top[1])
        cx2, cy2 = int(bot[0]), int(bot[1])
        thickness = max(int(min(h, w) * 0.015), 2)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (cx1, cy1), (cx2, cy2), 1.0, thickness, cv2.LINE_AA)
        return mask

    # ------------------------------------------------------------------
    # Brows
    # ------------------------------------------------------------------

    def apply_brows(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        color: Union[str, Tuple[int, int, int]] = "brown",
        thickness: int = 2,
    ) -> np.ndarray:
        """Tint and slightly thicken the eyebrows.

        Applies a soft-light tint over the existing eyebrow region (from the
        landmark polygon) and dilates the region by *thickness* pixels to
        simulate filling sparse brows.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            landmarks: MediaPipe NormalizedLandmarkList.
            color: Palette key (see BROW_COLORS) or BGR tuple.
            thickness: Dilation amount in pixels (also scales tint opacity).
        """
        if thickness <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.apply_brows,
                landmarks,
                color,
                thickness,
            )

        h, w = img_bgr.shape[:2]
        color_bgr = _resolve_color(color, BROW_COLORS, "brown")
        t = max(int(thickness), 1)

        brow_mask = np.zeros((h, w), dtype=np.float32)
        for brow_idx in (LEFT_EYEBROW, RIGHT_EYEBROW):
            try:
                pts = get_points(landmarks, brow_idx, w, h)
            except (IndexError, AttributeError):
                continue
            m = _feather_polygon(pts, (h, w), feather=max(2, int(min(h, w) * 0.008)))
            brow_mask = np.maximum(brow_mask, m)

        if brow_mask.max() < 0.01:
            return img_bgr

        # Dilate to fill sparse areas and add thickness.
        brow_mask = _dilate_mask(brow_mask, iterations=t)

        # Feather the dilated edges.
        brow_mask = _gaussian_blur_safe(brow_mask, max(3, t | 1))

        # Tint opacity scales gently with thickness.
        opacity = min(0.3 + t * 0.08, 0.7)
        return _soft_light_tint(img_bgr, brow_mask, color_bgr, opacity)

    # ------------------------------------------------------------------
    # Ombre lips
    # ------------------------------------------------------------------

    def apply_ombre_lips(
        self,
        img_bgr: np.ndarray,
        landmarks: Any,
        color1: Union[str, Tuple[int, int, int]] = "red",
        color2: Union[str, Tuple[int, int, int]] = "pink",
    ) -> np.ndarray:
        """Apply an ombre lip tint: color1 at the outer edges, color2 at the center.

        The gradient is built from a vertical (or radial) weight mask over
        the lip region and the two colors are blended in LAB space so the
        underlying lip luminance/texture is preserved.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            landmarks: MediaPipe NormalizedLandmarkList.
            color1: Outer lip color (palette key from EYESHADOW_COLORS or BGR tuple).
            color2: Inner/center lip color.
        """
        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr,
                self.apply_ombre_lips,
                landmarks,
                color1,
                color2,
            )

        h, w = img_bgr.shape[:2]
        outer_bgr = _resolve_color(color1, EYESHADOW_COLORS, "rose")
        inner_bgr = _resolve_color(color2, EYESHADOW_COLORS, "nude")

        # Build the lip mask from the outer lip contour.
        try:
            lip_pts = get_points(landmarks, LIPS_OUTER, w, h)
        except (IndexError, AttributeError):
            return img_bgr

        lip_mask = _feather_polygon(lip_pts, (h, w), feather=max(2, int(min(h, w) * 0.008)))
        if lip_mask.max() < 0.01:
            return img_bgr

        # Build the ombre gradient: weight from 0 (outer/top) to 1 (inner/bottom).
        # Use vertical position normalized within the lip bounding box.
        ys, xs = np.where(lip_mask > 0.1)
        if len(ys) < 4:
            return img_bgr
        y_min, y_max = float(ys.min()), float(ys.max())
        y_range = max(y_max - y_min, 1.0)
        # Gradient: 0 at top (outer), 1 at bottom (inner).
        grad = np.zeros((h, w), dtype=np.float32)
        yy = np.arange(h, dtype=np.float32)[:, np.newaxis]
        grad_y = np.clip((yy - y_min) / y_range, 0.0, 1.0)
        grad = np.broadcast_to(grad_y, (h, w)).copy()

        # Feather the gradient so the transition is smooth.
        grad = _gaussian_blur_safe(grad, max(int(min(h, w) * 0.02), 5) | 1)

        # Blend two color shifts: color1 where grad<0.5, color2 where grad>0.5.
        weight_inner = np.clip((grad - 0.3) / 0.4, 0.0, 1.0)  # smooth ramp
        weight_outer = 1.0 - weight_inner

        # Outer color shift.
        outer_shift = _lab_color_shift(img_bgr, lip_mask * weight_outer, outer_bgr, 0.5)
        # Inner color shift on top of that.
        result = _lab_color_shift(outer_shift, lip_mask * weight_inner, inner_bgr, 0.6)

        return result

"""Body reshape engine — MediaPipe Pose-aware full-body reshaping.

Enables proportional body editing (limb length, torso width, shoulder width)
using MediaPipe Pose 33 landmarks with pose-aware warping. Uses the same
(1−d²/R²)² kernel as F5's face liquify for consistent feel.

Technical notes:
- MediaPipe Pose outputs 33 landmarks in [0,1]² normalized coordinates
- Visibility score [0,1] per landmark; skip if < 0.3
- Limb definitions:
  - Left arm: shoulder[11] → elbow[13] → wrist[15]
  - Right arm: shoulder[12] → elbow[14] → wrist[16]
  - Left leg: hip[23] → knee[25] → ankle[27]
  - Right leg: hip[24] → knee[26] → ankle[28]
  - Torso: shoulders[11/12] ↔ hips[23/24]
- Displacement scaling: map slider 0-100 to ±15% of segment length
- Per-landmark visibility gating + person_mask boundary control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any, Dict

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# Displacement cap: at slider=±100, max control-point displacement is
# segment_length × 0.15 (conservative to avoid distortion).
_MAX_DISPLACE_FRAC = 0.15

# Radius cap (consistent with F5 face liquify).
_MAX_RADIUS_FRAC = 0.5

# Landmark visibility threshold: skip landmarks below this confidence.
_VISIBILITY_THRESHOLD = 0.3


@dataclass
class PoseContext:
    """Stores detected pose landmarks, visibility, and per-feature state."""

    landmarks: Optional[List[Tuple[float, float]]] = None
    """List of (x, y) in [0,1]² normalized coordinates for each landmark."""

    visibility: Optional[List[float]] = None
    """Visibility confidence [0,1] for each landmark."""

    detected: bool = False
    """Whether pose detection succeeded."""

    body_visible: bool = False
    """Whether key body landmarks (shoulders, hips, ankles) are visible."""

    feature_flags: dict[str, bool] = field(default_factory=lambda: {
        "arm_length": True,
        "leg_length": True,
        "torso_width": True,
        "shoulder_width": True,
        "hip_width": True,
    })
    """Per-feature active flags (for gating individual warps)."""


class PoseDetector:
    """Wrapper around MediaPipe Pose for robust pose landmark extraction.

    Currently a graceful no-op fallback when pose_landmarker.task model is not
    available. In production, this would load the MediaPipe Tasks API PoseLandmarker.
    """

    def __init__(self):
        """Initialize MediaPipe Pose detector.

        Note: pose_landmarker.task must be downloaded separately. If unavailable,
        pose detection is disabled and body reshape becomes a no-op.
        """
        self.detector = None
        try:
            from mediapipe.tasks import vision
            from mediapipe.tasks import BaseOptions
            import os

            # Try to load pose landmarker model
            pose_model_path = self._get_pose_model_path()
            if pose_model_path is None:
                logger.debug("Pose model not found; pose detection disabled")
                return

            base_options = BaseOptions(model_asset_path=pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            logger.debug(f"PoseLandmarker initialized from {pose_model_path}")

        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Failed to initialize MediaPipe Pose: {type(e).__name__}: {e}")

    @staticmethod
    def _get_pose_model_path() -> Optional[str]:
        """Try to find pose landmarker model in standard locations.

        Returns path if found, None otherwise.
        """
        import os

        candidates = [
            # User home directory
            os.path.expanduser("~/.mediapipe/pose_landmarker.task"),
            # System paths
            "/opt/mediapipe/pose_landmarker.task",
            # Local project
            "./models/pose_landmarker.task",
            # Parent directory
            "../models/pose_landmarker.task",
        ]

        for path in candidates:
            try:
                if os.path.exists(path):
                    return os.path.abspath(path)
            except (OSError, TypeError):
                pass

        return None

    def detect(self, img_bgr: np.ndarray) -> PoseContext:
        """Detect pose landmarks in image.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.

        Returns:
            PoseContext with detected landmarks and visibility.
        """
        ctx = PoseContext()

        if self.detector is None:
            logger.debug("Pose detector not initialized")
            return ctx

        # Convert to RGB for MediaPipe
        if img_bgr.dtype == np.uint8:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        try:
            # Convert to MediaPipe Image format
            from mediapipe import Image

            mp_img = Image(image_format=Image.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_img)

            if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
                logger.debug("No pose detected in image")
                return ctx

            # Extract landmarks and visibility from the first detected pose
            pose = results.pose_landmarks[0]
            landmarks = []
            visibility = []
            for lm in pose:
                landmarks.append((lm.x, lm.y))
                visibility.append(lm.visibility if hasattr(lm, 'visibility') else 1.0)

            ctx.landmarks = landmarks
            ctx.visibility = visibility
            ctx.detected = True

            # Check if key body landmarks are visible (shoulders, hips, ankles)
            key_indices = [11, 12, 23, 24, 27, 28]  # shoulders, hips, ankles
            ctx.body_visible = all(
                visibility[i] >= _VISIBILITY_THRESHOLD for i in key_indices
                if i < len(visibility)
            )

            logger.debug(
                f"Pose detected: {len(landmarks)} landmarks, "
                f"body_visible={ctx.body_visible}"
            )

        except Exception as e:
            logger.warning(f"Pose detection failed: {e}")

        return ctx


class BodyReshaper:
    """Full-body warp engine with per-feature sliders."""

    def __init__(self):
        """Initialize body reshaper with MediaPipe Pose detector."""
        self.detector = PoseDetector()
        self.pose_ctx: Optional[PoseContext] = None

    def reshape(
        self,
        img_bgr: np.ndarray,
        arm_length: float = 0.0,
        leg_length: float = 0.0,
        torso_width: float = 0.0,
        shoulder_width: float = 0.0,
        hip_width: float = 0.0,
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply full-body reshaping with per-feature control.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            arm_length: slider -100 to +100 (negative = shorter, positive = longer).
            leg_length: slider -100 to +100.
            torso_width: slider -100 to +100 (negative = thinner, positive = wider).
            shoulder_width: slider -100 to +100.
            hip_width: slider -100 to +100.
            person_mask: optional (H, W) float32 [0,1] mask for boundary control.

        Returns:
            (H, W, 3) reshaped image, same dtype as input.
        """
        orig_dtype = img_bgr.dtype
        h, w = img_bgr.shape[:2]

        # Skip if no sliders are active
        if not any([arm_length, leg_length, torso_width, shoulder_width, hip_width]):
            return img_bgr

        # Detect pose (or use cached)
        if self.pose_ctx is None:
            self.pose_ctx = self.detector.detect(img_bgr)

        ctx = self.pose_ctx
        if not ctx.detected or not ctx.landmarks:
            logger.debug("No pose detected, skipping body reshape")
            return img_bgr

        # Build warps
        warps = []
        if arm_length != 0 and ctx.feature_flags.get("arm_length", True):
            warps.extend(self._arm_length_warps(ctx, w, h, arm_length))
        if leg_length != 0 and ctx.feature_flags.get("leg_length", True):
            warps.extend(self._leg_length_warps(ctx, w, h, leg_length))
        if torso_width != 0 and ctx.feature_flags.get("torso_width", True):
            warps.extend(self._torso_width_warps(ctx, w, h, torso_width))
        if shoulder_width != 0 and ctx.feature_flags.get("shoulder_width", True):
            warps.extend(self._shoulder_width_warps(ctx, w, h, shoulder_width))
        if hip_width != 0 and ctx.feature_flags.get("hip_width", True):
            warps.extend(self._hip_width_warps(ctx, w, h, hip_width))

        if not warps:
            return img_bgr

        # Apply warps
        return self._apply_warps(img_bgr, warps, person_mask=person_mask)

    # ------------------------------------------------------------------
    # Per-feature warp builders
    # ------------------------------------------------------------------

    def _arm_length_warps(
        self, ctx: PoseContext, w: int, h: int, slider: float
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Extend/compress arm length by warping wrist away from/toward shoulder.

        Landmarks: shoulder[11/12] → elbow[13/14] → wrist[15/16].
        """
        if slider == 0 or not ctx.landmarks or not ctx.visibility:
            return []

        warps = []
        # Left arm: shoulder[11], elbow[13], wrist[15]
        # Right arm: shoulder[12], elbow[14], wrist[16]
        for sh_idx, el_idx, wr_idx in [(11, 13, 15), (12, 14, 16)]:
            if (
                ctx.visibility[sh_idx] < _VISIBILITY_THRESHOLD
                or ctx.visibility[wr_idx] < _VISIBILITY_THRESHOLD
            ):
                continue

            sh_x, sh_y = ctx.landmarks[sh_idx]
            wr_x, wr_y = ctx.landmarks[wr_idx]

            # Arm length in image coords
            arm_len_px = (
                ((wr_x - sh_x) ** 2 + (wr_y - sh_y) ** 2) ** 0.5 * max(w, h)
            )
            if arm_len_px < 10:
                continue

            # Displacement magnitude: slider [0-100] → ±15% of arm length
            disp_mag = (slider / 100.0) * arm_len_px * _MAX_DISPLACE_FRAC

            # Direction: from shoulder to wrist (normalized)
            if arm_len_px > 0:
                dx = (wr_x - sh_x) / (arm_len_px / max(w, h))
                dy = (wr_y - sh_y) / (arm_len_px / max(w, h))
            else:
                continue

            # Wrist control point and target
            cx = int(wr_x * w)
            cy = int(wr_y * h)
            tx = int(cx + dx * disp_mag)
            ty = int(cy + dy * disp_mag)

            # Radius proportional to arm length
            radius = max(int(arm_len_px * 0.4), 10)

            warps.append(((cx, cy), (tx, ty), radius))

        return warps

    def _leg_length_warps(
        self, ctx: PoseContext, w: int, h: int, slider: float
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Extend/compress leg length by warping ankle away from/toward hip.

        Landmarks: hip[23/24] → knee[25/26] → ankle[27/28].
        """
        if slider == 0 or not ctx.landmarks or not ctx.visibility:
            return []

        warps = []
        # Left leg: hip[23], knee[25], ankle[27]
        # Right leg: hip[24], knee[26], ankle[28]
        for hp_idx, an_idx in [(23, 27), (24, 28)]:
            if (
                ctx.visibility[hp_idx] < _VISIBILITY_THRESHOLD
                or ctx.visibility[an_idx] < _VISIBILITY_THRESHOLD
            ):
                continue

            hp_x, hp_y = ctx.landmarks[hp_idx]
            an_x, an_y = ctx.landmarks[an_idx]

            # Leg length in image coords
            leg_len_px = (
                ((an_x - hp_x) ** 2 + (an_y - hp_y) ** 2) ** 0.5 * max(w, h)
            )
            if leg_len_px < 10:
                continue

            # Displacement magnitude: slider [0-100] → ±15% of leg length
            disp_mag = (slider / 100.0) * leg_len_px * _MAX_DISPLACE_FRAC

            # Direction: from hip to ankle (normalized)
            if leg_len_px > 0:
                dx = (an_x - hp_x) / (leg_len_px / max(w, h))
                dy = (an_y - hp_y) / (leg_len_px / max(w, h))
            else:
                continue

            # Ankle control point and target
            cx = int(an_x * w)
            cy = int(an_y * h)
            tx = int(cx + dx * disp_mag)
            ty = int(cy + dy * disp_mag)

            # Radius proportional to leg length
            radius = max(int(leg_len_px * 0.4), 10)

            warps.append(((cx, cy), (tx, ty), radius))

        return warps

    def _torso_width_warps(
        self, ctx: PoseContext, w: int, h: int, slider: float
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Expand/compress torso width by warping sides outward/inward.

        Uses shoulder width as reference.
        """
        if slider == 0 or not ctx.landmarks or not ctx.visibility:
            return []

        warps = []
        # Shoulders: [11] (left), [12] (right)
        # Hips: [23] (left), [24] (right)
        sh_l_x, sh_l_y = ctx.landmarks[11]
        sh_r_x, sh_r_y = ctx.landmarks[12]

        if ctx.visibility[11] < _VISIBILITY_THRESHOLD or ctx.visibility[12] < _VISIBILITY_THRESHOLD:
            return []

        torso_width_px = abs((sh_r_x - sh_l_x) * w)
        if torso_width_px < 10:
            return []

        # Displacement magnitude
        disp_mag = (slider / 100.0) * torso_width_px * _MAX_DISPLACE_FRAC

        # Warp shoulders outward (positive) / inward (negative)
        # Left shoulder: move left (negative x direction)
        cx_l = int(sh_l_x * w)
        cy_l = int(sh_l_y * h)
        tx_l = int(cx_l - disp_mag)  # Move outward/inward
        ty_l = cy_l

        radius = max(int(torso_width_px * 0.3), 10)
        warps.append(((cx_l, cy_l), (tx_l, ty_l), radius))

        # Right shoulder: move right (positive x direction)
        cx_r = int(sh_r_x * w)
        cy_r = int(sh_r_y * h)
        tx_r = int(cx_r + disp_mag)  # Move outward/inward
        ty_r = cy_r

        warps.append(((cx_r, cy_r), (tx_r, ty_r), radius))

        return warps

    def _shoulder_width_warps(
        self, ctx: PoseContext, w: int, h: int, slider: float
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Expand/compress shoulder width specifically.

        Similar to torso_width but with separate control.
        """
        if slider == 0 or not ctx.landmarks or not ctx.visibility:
            return []

        warps = []
        # Use shoulder positions
        sh_l_x, sh_l_y = ctx.landmarks[11]
        sh_r_x, sh_r_y = ctx.landmarks[12]

        if ctx.visibility[11] < _VISIBILITY_THRESHOLD or ctx.visibility[12] < _VISIBILITY_THRESHOLD:
            return []

        shoulder_width_px = abs((sh_r_x - sh_l_x) * w)
        if shoulder_width_px < 10:
            return []

        # Displacement magnitude (same formula)
        disp_mag = (slider / 100.0) * shoulder_width_px * _MAX_DISPLACE_FRAC

        # Left shoulder
        cx_l = int(sh_l_x * w)
        cy_l = int(sh_l_y * h)
        tx_l = int(cx_l - disp_mag)
        ty_l = cy_l

        radius = max(int(shoulder_width_px * 0.3), 10)
        warps.append(((cx_l, cy_l), (tx_l, ty_l), radius))

        # Right shoulder
        cx_r = int(sh_r_x * w)
        cy_r = int(sh_r_y * h)
        tx_r = int(cx_r + disp_mag)
        ty_r = cy_r

        warps.append(((cx_r, cy_r), (tx_r, ty_r), radius))

        return warps

    def _hip_width_warps(
        self, ctx: PoseContext, w: int, h: int, slider: float
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
        """Expand/compress hip width.

        Uses hip positions [23] (left), [24] (right).
        """
        if slider == 0 or not ctx.landmarks or not ctx.visibility:
            return []

        warps = []
        # Hips: [23] (left), [24] (right)
        hip_l_x, hip_l_y = ctx.landmarks[23]
        hip_r_x, hip_r_y = ctx.landmarks[24]

        if ctx.visibility[23] < _VISIBILITY_THRESHOLD or ctx.visibility[24] < _VISIBILITY_THRESHOLD:
            return []

        hip_width_px = abs((hip_r_x - hip_l_x) * w)
        if hip_width_px < 10:
            return []

        # Displacement magnitude
        disp_mag = (slider / 100.0) * hip_width_px * _MAX_DISPLACE_FRAC

        # Left hip
        cx_l = int(hip_l_x * w)
        cy_l = int(hip_l_y * h)
        tx_l = int(cx_l - disp_mag)
        ty_l = cy_l

        radius = max(int(hip_width_px * 0.3), 10)
        warps.append(((cx_l, cy_l), (tx_l, ty_l), radius))

        # Right hip
        cx_r = int(hip_r_x * w)
        cy_r = int(hip_r_y * h)
        tx_r = int(cx_r + disp_mag)
        ty_r = cy_r

        warps.append(((cx_r, cy_r), (tx_r, ty_r), radius))

        return warps

    # ------------------------------------------------------------------
    # Warp application (using same kernel as F5 face liquify)
    # ------------------------------------------------------------------

    def _apply_warps(
        self,
        img_bgr: np.ndarray,
        warps: List[Tuple[Tuple[int, int], Tuple[int, int], int]],
        person_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Accumulate displacement maps from warps and apply remap.

        Implements (1 − d²/R²)² translation kernel.

        Args:
            img_bgr: (H, W, 3) input image.
            warps: list of (ControlPoint, TargetPoint, Radius) tuples.
            person_mask: optional (H, W) float32 [0,1] mask for boundary control.

        Returns:
            (H, W, 3) warped image, same dtype as input.
        """
        if not warps:
            return img_bgr

        h, w = img_bgr.shape[:2]
        orig_dtype = img_bgr.dtype

        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Compute max radius and displacement caps
        max_radius = self._max_radius_for_warps(warps, w, h)
        max_disp_cap = max_radius * 2.0 * _MAX_DISPLACE_FRAC if max_radius > 0 else float("inf")
        radius_hard_cap = max_radius * 2.0 * _MAX_RADIUS_FRAC if max_radius > 0 else float("inf")

        has_deformations = False
        for C, T, R in warps:
            cx, cy = C
            tx, ty = T
            ux = float(tx - cx)
            uy = float(ty - cy)

            # Enforce displacement cap
            disp_mag = (ux * ux + uy * uy) ** 0.5
            if disp_mag > max_disp_cap and disp_mag > 0:
                scale = max_disp_cap / disp_mag
                ux *= scale
                uy *= scale

            R_eff = min(int(R), int(radius_hard_cap)) if radius_hard_cap != float("inf") else int(R)
            if R_eff < 2:
                continue

            x1 = max(int(cx) - R_eff, 0)
            y1 = max(int(cy) - R_eff, 0)
            x2 = min(int(cx) + R_eff, w)
            y2 = min(int(cy) + R_eff, h)

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            local_dx = map_x[y1:y2, x1:x2] - cx
            local_dy = map_y[y1:y2, x1:x2] - cy
            dist_sq = local_dx * local_dx + local_dy * local_dy
            R_sq = float(R_eff) * float(R_eff)

            mask = dist_sq < R_sq
            if not np.any(mask):
                continue

            t = 1.0 - dist_sq / R_sq
            weight = t * t
            weight[~mask] = 0.0

            map_x[y1:y2, x1:x2] -= weight * ux
            map_y[y1:y2, x1:x2] -= weight * uy
            has_deformations = True

        if not has_deformations:
            return img_bgr

        reshaped = cv2.remap(
            img_bgr,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Apply person mask if provided
        if person_mask is not None and person_mask.max() > 0.01:
            m3 = person_mask[:, :, None]
            reshaped_f = reshaped.astype(np.float32)
            orig_f = img_bgr.astype(np.float32)
            blended = orig_f * (1.0 - m3) + reshaped_f * m3
            reshaped = np.clip(blended, 0, 255).astype(orig_dtype)

        if reshaped.dtype != orig_dtype:
            reshaped = reshaped.astype(orig_dtype)

        return reshaped

def suggest_body_reshape(pose_ctx: PoseContext) -> Dict[str, float]:
    """Suggest balanced T3 body-reshape proportions from detected pose.

    Returns 0-100 values (50 = no change) for keys
    ``arm_length``, ``leg_length``, ``torso_width``, ``shoulder_width``,
    ``hip_width``. Corrections are SMALL and conservative — they nudge the
    figure toward balanced proportions, never toward extremes.

    Args:
        pose_ctx: detected pose (33 normalized [0,1]² landmarks, optional
            visibility + per-feature ``feature_flags``).

    Returns:
        Dict with the five suggested 0-100 values. All 50.0 (neutral) when
        the pose is unusable.
    """
    keys = ("arm_length", "leg_length", "torso_width", "shoulder_width", "hip_width")
    neutral: Dict[str, float] = {k: 50.0 for k in keys}

    lm = pose_ctx.landmarks
    vis = pose_ctx.visibility
    flags = pose_ctx.feature_flags or {}
    if lm is None or len(lm) < 29:
        return neutral

    def dist(i: int, j: int) -> float:
        x1, y1 = lm[i]
        x2, y2 = lm[j]
        return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

    def mid(i: int, j: int) -> Tuple[float, float]:
        x1, y1 = lm[i]
        x2, y2 = lm[j]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def dist_pt(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return float(((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5)

    def vis_ok(i: int) -> bool:
        if vis is None or i >= len(vis):
            return True
        return float(vis[i]) >= _VISIBILITY_THRESHOLD

    result = dict(neutral)

    # --- Shoulder : hip width ratio (ideal ~1.3) ---
    if (
        vis_ok(11) and vis_ok(12) and vis_ok(23) and vis_ok(24)
        and flags.get("shoulder_width", True) and flags.get("hip_width", True)
    ):
        sh_w = dist(11, 12)
        hp_w = dist(23, 24)
        if sh_w > 1e-6 and hp_w > 1e-6:
            ratio = sh_w / hp_w
            ideal = 1.3
            dev = (ratio - ideal) / ideal  # + = shoulders wide vs hips
            delta = float(np.clip(dev * 60.0, -30.0, 30.0))
            # shoulders wide -> narrow shoulders, widen hips (opposite nudges)
            result["shoulder_width"] = 50.0 - delta
            result["hip_width"] = 50.0 + delta

    # --- Torso length / leg length (need ankles) ---
    if vis_ok(11) and vis_ok(12) and vis_ok(23) and vis_ok(24) and vis_ok(27) and vis_ok(28):
        torso = dist_pt(mid(11, 12), mid(23, 24))
        leg = dist_pt(mid(23, 24), mid(27, 28))
        if torso > 1e-6 and leg > 1e-6:
            # --- Leg : torso ratio (ideal ~1.2) ---
            if flags.get("leg_length", True):
                ratio = leg / torso
                ideal = 1.2
                dev = (ratio - ideal) / ideal  # - = legs short vs torso
                delta = float(np.clip(dev * 40.0, -30.0, 30.0))
                result["leg_length"] = 50.0 - delta

            # --- Arm : torso ratio (ideal ~1.0) ---
            if flags.get("arm_length", True):
                arm = (dist(11, 15) + dist(12, 16)) / 2.0
                if arm > 1e-6:
                    ratio = arm / torso
                    ideal = 1.0
                    dev = (ratio - ideal) / ideal  # - = arms short vs torso
                    delta = float(np.clip(dev * 40.0, -20.0, 20.0))
                    result["arm_length"] = 50.0 - delta

    # Clamp to a safe band; torso_width stays neutral (no auto heuristic).
    for k in keys:
        result[k] = float(np.clip(result[k], 20.0, 80.0))
    return result


    @staticmethod
    def _max_radius_for_warps(
        warps: List[Tuple[Tuple[int, int], Tuple[int, int], int]],
        w: int,
        h: int,
    ) -> float:
        """Compute max radius across all warps."""
        if not warps:
            return 0.0
        return max(R for _, _, R in warps)

"""Geometry manipulation layer — photographer-grade face slimming and reshaping.

Applies local translation warping (liquid warp) on specific jaw, cheek, and chin
landmarks to perform non-destructive, subtle reshaping. F5 extends this to a
full face-aware liquify slider bank (eye size, nose width, jaw width, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .parsing import (
    FACE_OVAL,
    LEFT_EYE,
    LEFT_IRIS,
    LIPS_OUTER,
    RIGHT_EYE,
    RIGHT_IRIS,
)
from .utils import create_polygon_mask, feather_mask, get_points

logger = logging.getLogger(__name__)

# Displacement cap (§2.4 of PLAN_F5_LIQUIFY.md). At slider=±50, max control-
# point displacement is face_width × 0.06; at the extended ±100 cap, it's
# face_width × 0.12. Each slider's k is mapped so the cap holds at ±100.
_MAX_DISPLACE_FRAC = 0.12

# Radius cap (§4.3 mechanism #1). Every warp radius R ≤ face_width × 0.5.
_MAX_RADIUS_FRAC = 0.5

# Number of radial translation warps used to compose a scale warp (§2.3).
_SCALE_RING_N = 8

# Yaw gate (profile-view guard). MediaPipe jaw-landmark reliability degrades
# on yawed faces, so warps keyed on 234/454/152 mis-scale lopsidedly. Warp
# strength ramps smoothly 1.0 -> 0.0 between these yaw_ratio values (same
# ratio formula as skin.py's neck gate). skin.py hard-gates at 1.3 because
# that marks neck-depth-plane onset; slimming still tolerates mild yaw, so
# the ramp starts at 1.3 and zeroes at 1.6 (strong three-quarter view).
_YAW_DAMPEN_START = 1.3
_YAW_DAMPEN_END = 1.6

# Landmark index for the nose-bridge center, used as the facial vertical mirror
# axis for bilateral symmetry (§6.3).
_NOSE_BRIDGE_CENTER_IDX = 168

# Tunable per-slider scale coefficients. Each maps slider∈[-100, +100] to a
# feature-specific k factor. Calibrated so |T−C| ≤ fw × 0.12 at the cap.
_EYE_SIZE_K = 0.18
_EYE_DISTANCE_K = 0.10
_NOSE_WIDTH_K = 0.18
_NOSE_LENGTH_K = 0.08
_JAW_WIDTH_K = 0.10
_CHIN_LENGTH_K = 0.08
_MOUTH_SIZE_K = 0.18
_SMILE_K = 0.10
_FOREHEAD_K = 0.08
_NECK_WIDTH_K = 0.10
_NECK_LENGTH_K = 0.08


Warp = Tuple[Tuple[int, int], Tuple[int, int], int]


class FaceReshaper:
    """Photographer-grade face slimming and reshaping using local translation warping."""

    def __init__(self) -> None:
        # AA6 reads this exact remap field after ``reshape``. It is diagnostic
        # state only and is reset for every invocation, including no-op calls.
        self.last_displacement_field: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _read_reshape_vals(
        ctx: Any = None,
        strength: Optional[int] = None,
    ) -> Tuple[float, dict, dict]:
        """Read slimming + reshape + side slider values from one context."""
        if ctx is not None:
            slimming_val = float(getattr(ctx, "slimming", 0.0) or 0.0)
            reshape_vals = {
                "eye_size": float(getattr(ctx, "reshape_eye_size", 0.0) or 0.0),
                "eye_distance": float(getattr(ctx, "reshape_eye_distance", 0.0) or 0.0),
                "nose_width": float(getattr(ctx, "reshape_nose_width", 0.0) or 0.0),
                "nose_length": float(getattr(ctx, "reshape_nose_length", 0.0) or 0.0),
                "jaw_width": float(getattr(ctx, "reshape_jaw_width", 0.0) or 0.0),
                "chin_length": float(getattr(ctx, "reshape_chin_length", 0.0) or 0.0),
                "mouth_size": float(getattr(ctx, "reshape_mouth_size", 0.0) or 0.0),
                "smile": float(getattr(ctx, "reshape_smile", 0.0) or 0.0),
                "forehead": float(getattr(ctx, "reshape_forehead", 0.0) or 0.0),
            }
            side_vals = {
                "jaw_width_l": float(getattr(ctx, "reshape_jaw_width_l", 0.0) or 0.0),
                "jaw_width_r": float(getattr(ctx, "reshape_jaw_width_r", 0.0) or 0.0),
                "nose_width_l": float(getattr(ctx, "reshape_nose_width_l", 0.0) or 0.0),
                "nose_width_r": float(getattr(ctx, "reshape_nose_width_r", 0.0) or 0.0),
                "eye_size_l": float(getattr(ctx, "reshape_eye_size_l", 0.0) or 0.0),
                "eye_size_r": float(getattr(ctx, "reshape_eye_size_r", 0.0) or 0.0),
                "neck_width": float(getattr(ctx, "reshape_neck_width", 0.0) or 0.0),
                "neck_length": float(getattr(ctx, "reshape_neck_length", 0.0) or 0.0),
            }
        else:
            slimming_val = float(strength or 0)
            reshape_vals = {k: 0.0 for k in (
                "eye_size", "eye_distance", "nose_width", "nose_length",
                "jaw_width", "chin_length", "mouth_size", "smile", "forehead",
            )}
            side_vals = {k: 0.0 for k in (
                "jaw_width_l", "jaw_width_r", "nose_width_l", "nose_width_r",
                "eye_size_l", "eye_size_r", "neck_width", "neck_length",
            )}
        return slimming_val, reshape_vals, side_vals

    def reshape(
        self,
        img_bgr: np.ndarray,
        faces: List[Any],
        ctx: Any = None,
        strength: Optional[int] = None,
        face_ctxs: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Perform face reshaping (slimming + F5 liquify) for all faces.

        B-lite: when ``face_ctxs`` is provided (same length as faces), each face
        reads its own reshape/slimming values; warps still accumulate into one
        global remap (safe when face supports are disjoint).

        Backward-compatible: ``strength`` int + ``ctx=None`` = slimming only.
        """
        h, w = img_bgr.shape[:2]
        self.last_displacement_field = np.zeros((h, w, 2), dtype=np.float32)
        if not faces:
            return img_bgr

        # Early exit: any face (or global ctx) non-zero
        check_ctxs: List[Any] = list(face_ctxs) if face_ctxs else [ctx]
        any_active = False
        for c in check_ctxs:
            sv, rv, sdv = self._read_reshape_vals(c, strength if c is None else None)
            if sv > 0 or any(v != 0 for v in rv.values()) or any(v != 0 for v in sdv.values()):
                any_active = True
                break
        if not any_active and ctx is not None and face_ctxs:
            sv, rv, sdv = self._read_reshape_vals(ctx, None)
            if sv > 0 or any(v != 0 for v in rv.values()) or any(v != 0 for v in sdv.values()):
                any_active = True
        if not any_active:
            return img_bgr

        warps: List[Warp] = []
        face_oval_pts_per_face: List[np.ndarray] = []
        max_face_width = 0.0

        for i, face in enumerate(faces):
            c = face_ctxs[i] if face_ctxs is not None and i < len(face_ctxs) else ctx
            slimming_val, reshape_vals, side_vals = self._read_reshape_vals(
                c, strength if c is None else None,
            )

            lm_obj = face.landmarks
            landmarks = lm_obj.landmark
            fw = self._face_width(landmarks, w)
            if fw < 10.0:
                continue

            if fw > max_face_width:
                max_face_width = fw

            # Yaw gate: dampen jaw-keyed warps on yawed faces (see
            # _YAW_DAMPEN_START comment). Frontal faces get factor 1.0 →
            # byte-identical warps.
            yaw_damp = self._yaw_dampen_factor(landmarks)
            slimming_val *= yaw_damp
            reshape_vals["jaw_width"] *= yaw_damp
            reshape_vals["chin_length"] *= yaw_damp
            side_vals["jaw_width_l"] *= yaw_damp
            side_vals["jaw_width_r"] *= yaw_damp

            warps.extend(self._slimming_warps(landmarks, fw, slimming_val, h, w))

            if side_vals["eye_size_l"] != 0 or side_vals["eye_size_r"] != 0:
                warps.extend(self._eye_size_warps(
                    lm_obj, landmarks, fw, 0.0, h, w,
                    slider_l=side_vals["eye_size_l"], slider_r=side_vals["eye_size_r"],
                ))
            else:
                warps.extend(self._eye_size_warps(lm_obj, landmarks, fw, reshape_vals["eye_size"], h, w))

            warps.extend(self._eye_distance_warps(landmarks, fw, reshape_vals["eye_distance"], h, w))

            if side_vals["nose_width_l"] != 0 or side_vals["nose_width_r"] != 0:
                warps.extend(self._nose_width_warps(
                    landmarks, fw, 0.0, h, w,
                    slider_l=side_vals["nose_width_l"], slider_r=side_vals["nose_width_r"],
                ))
            else:
                warps.extend(self._nose_width_warps(landmarks, fw, reshape_vals["nose_width"], h, w))

            warps.extend(self._nose_length_warps(landmarks, fw, reshape_vals["nose_length"], h, w))

            if side_vals["jaw_width_l"] != 0 or side_vals["jaw_width_r"] != 0:
                warps.extend(self._jaw_width_warps(
                    landmarks, fw, 0.0, h, w,
                    slider_l=side_vals["jaw_width_l"], slider_r=side_vals["jaw_width_r"],
                ))
            else:
                warps.extend(self._jaw_width_warps(landmarks, fw, reshape_vals["jaw_width"], h, w))

            warps.extend(self._chin_length_warps(landmarks, fw, reshape_vals["chin_length"], h, w))
            warps.extend(self._mouth_size_warps(lm_obj, landmarks, fw, reshape_vals["mouth_size"], h, w))
            warps.extend(self._smile_warps(landmarks, fw, reshape_vals["smile"], h, w))
            warps.extend(self._forehead_warps(lm_obj, landmarks, fw, reshape_vals["forehead"], h, w))
            warps.extend(self._neck_width_warps(landmarks, fw, side_vals["neck_width"], h, w))
            warps.extend(self._neck_length_warps(landmarks, fw, side_vals["neck_length"], h, w))

            face_oval_pts_per_face.append(
                get_points(lm_obj, FACE_OVAL, w, h)
            )

        if not warps:
            return img_bgr

        face_oval_mask = self._build_face_oval_mask(
            face_oval_pts_per_face, h, w, warps, max_face_width,
        )
        return self._apply_warps(img_bgr, warps, face_oval_mask=face_oval_mask)

    # ------------------------------------------------------------------
    # Core warp application (extracted from legacy reshape())
    # ------------------------------------------------------------------

    def _apply_warps(
        self,
        img_bgr: np.ndarray,
        warps: Sequence[Warp],
        face_oval_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Accumulate displacement maps from ``warps`` and apply a single remap.

        Implements the ``(1 − d²/R²)²`` translation kernel (§2.2 of the plan).
        Enforces displacement-cap and radius-cap clamps as defense-in-depth
        (recipe JSON with extreme values cannot bypass them).

        Args:
            img_bgr: (H, W, 3) input image.
            warps: list of (ControlPoint, TargetPoint, Radius) tuples.
            face_oval_mask: optional (H, W) float32 [0,1] mask for boundary
                bleed control (§4.3 mechanism #2). If None, no mask gate.

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

        max_radius = float(self._max_radius_for_warps(warps, w, h))
        max_disp_cap = max_radius * 2.0 * _MAX_DISPLACE_FRAC if max_radius > 0 else float("inf")
        radius_hard_cap = max_radius * 2.0 * _MAX_RADIUS_FRAC if max_radius > 0 else float("inf")

        has_deformations = False
        for C, T, R in warps:
            cx, cy = C
            tx, ty = T
            ux = float(tx - cx)
            uy = float(ty - cy)

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

        # ``map_*`` are source coordinates for each output pixel. Their
        # inverse delta is the exact visual displacement used by AA6.
        grid_x = np.arange(w, dtype=np.float32)[None, :]
        grid_y = np.arange(h, dtype=np.float32)[:, None]
        displacement = np.dstack((grid_x - map_x, grid_y - map_y))

        if face_oval_mask is not None and face_oval_mask.max() > 0.01:
            m3 = face_oval_mask[:, :, None]
            reshaped_f = reshaped.astype(np.float32)
            orig_f = img_bgr.astype(np.float32)
            blended = orig_f * (1.0 - m3) + reshaped_f * m3
            reshaped = np.clip(blended, 0, 255).astype(orig_dtype)
            displacement *= face_oval_mask[:, :, None].astype(np.float32)

        self.last_displacement_field = displacement.astype(np.float32, copy=False)

        if reshaped.dtype != orig_dtype:
            reshaped = reshaped.astype(orig_dtype)
        return reshaped

    # ------------------------------------------------------------------
    # Slimming (existing warp set, factored out — byte-identical behavior)
    # ------------------------------------------------------------------

    def _slimming_warps(
        self,
        landmarks: Any,
        fw: float,
        strength: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Jaw + cheek + chin inward compression (legacy slimming)."""
        if strength <= 0:
            return []
        s = strength / 100.0
        warps: List[Warp] = []
        horiz, vert = FaceReshaper._face_local_axes(landmarks, w, h)

        c_rjaw = (int(landmarks[234].x * w), int(landmarks[234].y * h))
        d_rjaw = horiz * (fw * 0.04 * s)
        t_rjaw = (int(c_rjaw[0] + d_rjaw[0]), int(c_rjaw[1] + d_rjaw[1]))
        warps.append((c_rjaw, t_rjaw, int(fw * 0.5)))

        c_ljaw = (int(landmarks[454].x * w), int(landmarks[454].y * h))
        d_ljaw = horiz * (fw * 0.04 * s)
        t_ljaw = (int(c_ljaw[0] - d_ljaw[0]), int(c_ljaw[1] - d_ljaw[1]))
        warps.append((c_ljaw, t_ljaw, int(fw * 0.5)))

        c_rchk = (int(landmarks[117].x * w), int(landmarks[117].y * h))
        d_rchk = horiz * (fw * 0.02 * s)
        t_rchk = (int(c_rchk[0] + d_rchk[0]), int(c_rchk[1] + d_rchk[1]))
        warps.append((c_rchk, t_rchk, int(fw * 0.4)))

        c_lchk = (int(landmarks[346].x * w), int(landmarks[346].y * h))
        d_lchk = horiz * (fw * 0.02 * s)
        t_lchk = (int(c_lchk[0] - d_lchk[0]), int(c_lchk[1] - d_lchk[1]))
        warps.append((c_lchk, t_lchk, int(fw * 0.4)))

        c_chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
        d_chin = vert * (fw * 0.015 * s)
        t_chin = (int(c_chin[0] - d_chin[0]), int(c_chin[1] - d_chin[1]))
        warps.append((c_chin, t_chin, int(fw * 0.35)))

        return warps

    # ------------------------------------------------------------------
    # F5 per-feature warp builders
    # ------------------------------------------------------------------

    def _eye_size_warps(
        self,
        lm_obj: Any,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
        slider_l: Optional[float] = None,
        slider_r: Optional[float] = None,
    ) -> List[Warp]:
        """Radial scale of both eye outlines. Positive = enlarge, negative = shrink.

        Composed from N=8 radial translation warps per eye around the iris
        center (§2.3 of the plan). When ``slider_l``/``slider_r`` are provided,
        each eye is driven by its own strength; otherwise both use ``slider``
        (byte-identical to the legacy symmetric path).
        """
        s_left = slider_l if slider_l is not None else slider
        s_right = slider_r if slider_r is not None else slider
        if s_left == 0 and s_right == 0:
            return []
        warps: List[Warp] = []
        for iris_indices, eye_indices, s_eye in (
            (LEFT_IRIS, LEFT_EYE, s_left),
            (RIGHT_IRIS, RIGHT_EYE, s_right),
        ):
            if s_eye == 0:
                continue
            k = np.clip(s_eye / 100.0, -1.0, 1.0) * _EYE_SIZE_K
            center = self._centroid(landmarks, iris_indices, w, h)
            if center is None:
                center = self._centroid(landmarks, eye_indices, w, h)
            if center is None:
                continue
            cx, cy = center
            eye_pts = get_points(lm_obj, eye_indices, w, h)
            if len(eye_pts) == 0:
                continue
            r0 = float(np.max(np.linalg.norm(eye_pts - np.array(center, dtype=np.float32), axis=1)))
            if r0 < 2.0:
                continue
            warps.extend(self._radial_scale_warps(center, r0, k, int(fw * 0.5)))
        return warps

    def _eye_distance_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Translate both eyes outward (positive) / inward (negative) along the
        inter-eye axis (133 ↔ 362)."""
        if slider == 0:
            return []
        k = np.clip(slider / 100.0, -1.0, 1.0) * _EYE_DISTANCE_K
        p_inner_l = (landmarks[133].x * w, landmarks[133].y * h)
        p_inner_r = (landmarks[362].x * w, landmarks[362].y * h)
        axis = np.array([p_inner_r[0] - p_inner_l[0], p_inner_r[1] - p_inner_l[1]], dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-3:
            return []
        dir_vec = axis / norm
        disp = fw * k
        warps: List[Warp] = []
        for iris_indices, sign in ((LEFT_IRIS, -1.0), (RIGHT_IRIS, 1.0)):
            center = self._centroid(landmarks, iris_indices, w, h)
            if center is None:
                continue
            cx, cy = center
            tx = int(cx + dir_vec[0] * disp * sign)
            ty = int(cy + dir_vec[1] * disp * sign)
            warps.append(((cx, cy), (tx, ty), int(fw * 0.45)))
        return warps

    def _nose_width_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
        slider_l: Optional[float] = None,
        slider_r: Optional[float] = None,
    ) -> List[Warp]:
        """Radial scale of the alae around the nose bridge (168). Negative = narrow.

        When ``slider_l``/``slider_r`` are provided, each ala (48=left, 278=right)
        is translated independently along its bridge→ala axis; otherwise the
        legacy symmetric radial-scale path is used (byte-identical).
        """
        if slider_l is None and slider_r is None:
            if slider == 0:
                return []
            k = np.clip(slider / 100.0, -1.0, 1.0) * _NOSE_WIDTH_K
            center = (int(landmarks[168].x * w), int(landmarks[168].y * h))
            ala_l = (landmarks[48].x * w, landmarks[48].y * h)
            ala_r = (landmarks[278].x * w, landmarks[278].y * h)
            pts = np.array([ala_l, ala_r], dtype=np.float32)
            r0 = float(np.max(np.linalg.norm(pts - np.array(center, dtype=np.float32), axis=1)))
            if r0 < 2.0:
                return []
            return self._radial_scale_warps(center, r0, k, int(fw * 0.4))

        s_left = slider_l if slider_l is not None else slider
        s_right = slider_r if slider_r is not None else slider
        cx = int(landmarks[168].x * w)
        cy = int(landmarks[168].y * h)
        R = int(fw * 0.4)
        warps: List[Warp] = []
        for ala_idx, s_ala in ((48, s_left), (278, s_right)):
            if s_ala == 0:
                continue
            k = np.clip(s_ala / 100.0, -1.0, 1.0) * _NOSE_WIDTH_K
            px = landmarks[ala_idx].x * w
            py = landmarks[ala_idx].y * h
            vx = px - cx
            vy = py - cy
            tx = int(cx + (1.0 + k) * vx)
            ty = int(cy + (1.0 + k) * vy)
            warps.append(((int(px), int(py)), (tx, ty), R))
        return warps

    def _nose_length_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Translate nose tip (4) up (positive) / down (negative)."""
        if slider == 0:
            return []
        k = np.clip(slider / 100.0, -1.0, 1.0) * _NOSE_LENGTH_K
        cx = int(landmarks[4].x * w)
        cy = int(landmarks[4].y * h)
        disp = -fw * k  # negative Y = up
        tx = cx
        ty = int(cy + disp)
        return [((cx, cy), (tx, ty), int(fw * 0.35))]

    def _jaw_width_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
        slider_l: Optional[float] = None,
        slider_r: Optional[float] = None,
    ) -> List[Warp]:
        """Inward (positive) / outward (negative) compression at jaw angles
        (234=right / 454=left). Independent of slimming.

        When ``slider_l``/``slider_r`` are provided, each jaw angle is driven by
        its own strength; otherwise both use ``slider`` (byte-identical to the
        legacy symmetric path).
        """
        s_right = slider_r if slider_r is not None else slider
        s_left = slider_l if slider_l is not None else slider
        if s_right == 0 and s_left == 0:
            return []
        R = int(fw * 0.5)
        warps: List[Warp] = []
        horiz, _vert = FaceReshaper._face_local_axes(landmarks, w, h)
        if s_right != 0:
            k_r = np.clip(s_right / 100.0, -1.0, 1.0) * _JAW_WIDTH_K
            d_rjaw = horiz * (fw * k_r)
            c_rjaw = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            t_rjaw = (int(c_rjaw[0] + d_rjaw[0]), int(c_rjaw[1] + d_rjaw[1]))
            warps.append((c_rjaw, t_rjaw, R))
        if s_left != 0:
            k_l = np.clip(s_left / 100.0, -1.0, 1.0) * _JAW_WIDTH_K
            d_ljaw = horiz * (fw * k_l)
            c_ljaw = (int(landmarks[454].x * w), int(landmarks[454].y * h))
            t_ljaw = (int(c_ljaw[0] - d_ljaw[0]), int(c_ljaw[1] - d_ljaw[1]))
            warps.append((c_ljaw, t_ljaw, R))
        return warps

    def _chin_length_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Translate chin point 152 up (positive) / down (negative)."""
        if slider == 0:
            return []
        k = np.clip(slider / 100.0, -1.0, 1.0) * _CHIN_LENGTH_K
        cx = int(landmarks[152].x * w)
        cy = int(landmarks[152].y * h)
        disp = -fw * k  # positive slider = up
        tx = cx
        ty = int(cy + disp)
        return [((cx, cy), (tx, ty), int(fw * 0.35))]

    def _mouth_size_warps(
        self,
        lm_obj: Any,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Radial scale of the outer lip contour around the mouth center
        (midpoint of 13/14)."""
        if slider == 0:
            return []
        k = np.clip(slider / 100.0, -1.0, 1.0) * _MOUTH_SIZE_K
        center = self._mouth_center(landmarks, w, h)
        if center is None:
            return []
        lip_pts = get_points(lm_obj, LIPS_OUTER, w, h)
        if len(lip_pts) == 0:
            return []
        r0 = float(np.max(np.linalg.norm(lip_pts - np.array(center, dtype=np.float32), axis=1)))
        if r0 < 2.0:
            return []
        return self._radial_scale_warps(center, r0, k, int(fw * 0.4))

    def _smile_warps(
        self,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Lift mouth corners (61/291) up + slightly out. Negative = subtle frown."""
        if slider == 0:
            return []
        k = np.clip(slider / 100.0, -1.0, 1.0) * _SMILE_K
        horiz, vert = FaceReshaper._face_local_axes(landmarks, w, h)
        up = vert * (-fw * k)  # up along the face's own vertical
        out = horiz * (fw * k * 0.5)  # outward along the face's own horizontal
        warps: List[Warp] = []
        c_l = (int(landmarks[61].x * w), int(landmarks[61].y * h))
        t_l = (int(c_l[0] - out[0] + up[0]), int(c_l[1] - out[1] + up[1]))
        warps.append((c_l, t_l, int(fw * 0.3)))
        c_r = (int(landmarks[291].x * w), int(landmarks[291].y * h))
        t_r = (int(c_r[0] + out[0] + up[0]), int(c_r[1] + out[1] + up[1]))
        warps.append((c_r, t_r, int(fw * 0.3)))
        return warps

    def _forehead_warps(
        self,
        lm_obj: Any,
        landmarks: Any,
        fw: float,
        slider: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Translate the hairline band up (positive, more forehead) / down (negative).

        Uses the FOREHEAD_TOP landmark ring centroid as the control point and
        moves it vertically.
        """
        if slider == 0:
            return []
        from .parsing import FOREHEAD_TOP
        k = np.clip(slider / 100.0, -1.0, 1.0) * _FOREHEAD_K
        center = self._centroid(landmarks, FOREHEAD_TOP, w, h)
        if center is None:
            return []
        cx, cy = center
        disp = -fw * k  # positive = up
        tx = cx
        ty = int(cy + disp)
        return [((cx, cy), (tx, ty), int(fw * 0.4))]

    def _neck_width_warps(
        self,
        landmarks: Any,
        fw: float,
        strength: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Narrow (positive) / widen (negative) the neck/lower-face band.

        No dedicated neck landmarks exist in the MediaPipe mesh, so the jaw
        angles (234/454) plus lower jaw-line points are translated horizontally
        inward with a large downward-reaching radius (R = fw × 0.5, at the
        _MAX_RADIUS_FRAC cap).
        """
        if strength == 0:
            return []
        k = np.clip(strength / 100.0, -1.0, 1.0) * _NECK_WIDTH_K
        disp = fw * k
        R = int(fw * _MAX_RADIUS_FRAC)
        warps: List[Warp] = []
        for idx in (234, 58, 172):  # subject-right jaw/neck → push inward (+x)
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            warps.append(((cx, cy), (int(cx + disp), cy), R))
        for idx in (454, 288, 397):  # subject-left jaw/neck → push inward (−x)
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            warps.append(((cx, cy), (int(cx - disp), cy), R))
        return warps

    def _neck_length_warps(
        self,
        landmarks: Any,
        fw: float,
        strength: float,
        h: int,
        w: int,
    ) -> List[Warp]:
        """Lengthen (positive) / shorten (negative) the neck by translating the
        chin (152) and jaw angles (234/454) vertically downward."""
        if strength == 0:
            return []
        k = np.clip(strength / 100.0, -1.0, 1.0) * _NECK_LENGTH_K
        disp = fw * k  # positive = down (lengthen)
        R = int(fw * _MAX_RADIUS_FRAC)
        warps: List[Warp] = []
        for idx in (152, 234, 454):
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            warps.append(((cx, cy), (cx, int(cy + disp)), R))
        return warps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _yaw_dampen_factor(landmarks: Any) -> float:
        """Smoothstep 1.0 → 0.0 warp-strength factor vs. face yaw.

        yaw_ratio = max/min of the nose-tip (6) → left/right-ear (234/454)
        x-distances — same formula as skin.py's neck gate. 1.0 for frontal
        faces (ratio ≤ 1.3), 0.0 for strong profiles (ratio ≥ 1.6).
        """
        d_left = abs(landmarks[6].x - landmarks[234].x)
        d_right = abs(landmarks[454].x - landmarks[6].x)
        yaw_ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
        if yaw_ratio <= _YAW_DAMPEN_START:
            return 1.0
        if yaw_ratio >= _YAW_DAMPEN_END:
            return 0.0
        t = (yaw_ratio - _YAW_DAMPEN_START) / (_YAW_DAMPEN_END - _YAW_DAMPEN_START)
        return float(t * t * (3.0 - 2.0 * t))

    @staticmethod
    def _face_width(landmarks: Any, w: int) -> float:
        x_234 = landmarks[234].x * w
        x_454 = landmarks[454].x * w
        return abs(x_454 - x_234)

    @staticmethod
    def _face_local_axes(
        landmarks: Any,
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unit vectors for this face's own horizontal/vertical, independent
        of head roll in the frame.

        Warps like jaw compression or a mouth-corner lift are defined
        relative to the face's own anatomy ("pull the jaw toward the face
        center", "lift the mouth corner up"), not the camera's screen axes.
        For an upright face the two coincide, so a hardcoded screen ±x/±y
        offset looks right — but on a rolled/tilted portrait (e.g. a
        close-up with the head canted ~50° off vertical) screen-horizontal
        is no longer close to the true jaw-to-jaw axis, and a warp aimed
        along it pushes tissue sideways across the face instead of toward
        its intended target, producing a lopsided swollen-cheek look.

        Horizontal: anatomical right jaw (234) -> left jaw (454), matching
        the same landmark pair every jaw/cheek warp already keys off of.
        Vertical: forehead top (10) -> chin (152), perpendicular-ish to the
        horizontal by construction of the face shape, not enforced exactly
        orthogonal (real anatomy isn't a perfect rectangle) — each caller
        picks whichever axis matches the warp's intent.
        """
        p_r = np.array([landmarks[234].x * w, landmarks[234].y * h], dtype=np.float32)
        p_l = np.array([landmarks[454].x * w, landmarks[454].y * h], dtype=np.float32)
        horiz = p_l - p_r
        horiz_norm = float(np.linalg.norm(horiz))
        horiz = horiz / horiz_norm if horiz_norm > 1e-3 else np.array([1.0, 0.0], dtype=np.float32)

        p_top = np.array([landmarks[10].x * w, landmarks[10].y * h], dtype=np.float32)
        p_chin = np.array([landmarks[152].x * w, landmarks[152].y * h], dtype=np.float32)
        vert = p_chin - p_top
        vert_norm = float(np.linalg.norm(vert))
        vert = vert / vert_norm if vert_norm > 1e-3 else np.array([0.0, 1.0], dtype=np.float32)

        return horiz, vert

    @staticmethod
    def _centroid(
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int]]:
        if len(indices) == 0:
            return None
        try:
            xs = np.array([landmarks[i].x for i in indices], dtype=np.float32) * w
            ys = np.array([landmarks[i].y for i in indices], dtype=np.float32) * h
        except (IndexError, AttributeError):
            return None
        return (int(xs.mean()), int(ys.mean()))

    @staticmethod
    def _centroid_from_indices(
        landmarks: Any,
        indices: Sequence[int],
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int]]:
        return FaceReshaper._centroid(landmarks, indices, w, h)

    @staticmethod
    def _mouth_center(
        landmarks: Any,
        w: int,
        h: int,
    ) -> Optional[Tuple[int, int]]:
        try:
            cx = int((landmarks[13].x + landmarks[14].x) * 0.5 * w)
            cy = int((landmarks[13].y + landmarks[14].y) * 0.5 * h)
        except (IndexError, AttributeError):
            return None
        return (cx, cy)

    @staticmethod
    def _radial_scale_warps(
        center: Tuple[int, int],
        r0: float,
        k: float,
        R_ring: int,
    ) -> List[Warp]:
        """Compose a radial scale warp from N translation warps (§2.3).

        ``k > 0`` enlarges (outward push), ``k < 0`` shrinks (inward pull).
        """
        cx, cy = center
        warps: List[Warp] = []
        for i in range(_SCALE_RING_N):
            theta = 2.0 * np.pi * i / _SCALE_RING_N
            dx = r0 * np.cos(theta)
            dy = r0 * np.sin(theta)
            px = int(cx + dx)
            py = int(cy + dy)
            tx = int(cx + (1.0 + k) * dx)
            ty = int(cy + (1.0 + k) * dy)
            warps.append(((px, py), (tx, ty), R_ring))
        return warps

    @staticmethod
    def _max_radius_for_warps(
        warps: Sequence[Warp],
        w: int,
        h: int,
    ) -> float:
        """Estimate the face_width from the largest warp radius present.

        Used to derive the displacement and radius caps when a face_width is
        not directly available (e.g. warps passed without ctx). Returns 0 if
        no warps.
        """
        if not warps:
            return 0.0
        max_r = max(R for _, _, R in warps)
        return float(max_r)

    @staticmethod
    def _build_face_oval_mask(
        face_oval_pts_list: List[np.ndarray],
        h: int,
        w: int,
        warps: Sequence[Warp],
        face_width: float,
    ) -> Optional[np.ndarray]:
        """Build a coarse landmark face-oval mask for boundary bleed control.

        Per §4.3 / §5.3 option (B): uses ``create_polygon_mask`` +
        ``feather_mask`` over the FACE_OVAL landmarks. Returns None if no
        faces or no warps with radius > fw × 0.4.
        """
        if not face_oval_pts_list or face_width < 10.0:
            return None
        big_radius = any(R > face_width * 0.4 for _, _, R in warps)
        if not big_radius:
            return None
        mask = np.zeros((h, w), dtype=np.float32)
        feather = max(int(face_width * 0.08), 3)
        for pts in face_oval_pts_list:
            if len(pts) < 3:
                continue
            poly = create_polygon_mask(pts, (h, w), feather_radius=feather)
            mask = np.maximum(mask, poly)
        return feather_mask(mask, radius=feather)

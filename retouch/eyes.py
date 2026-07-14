"""Eye enhancement module — whites, iris, and catchlight processing.

Three independent sub-modules:
    1. Eye whites: remove redness/yellowness via LAB correction.
    2. Iris: boost clarity, micro-contrast, saturation, and sharpness.
    3. Catchlight: detect and amplify existing highlights (never create fakes).
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .utils import (
    feather_mask,
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
)


def _to_lab(img: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        return bgr_f32_to_lab_f32(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _bgr_to_hsv_u8_conv(img: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        from .grading import _bgr_f_to_hsv_u8_conv
        return _bgr_f_to_hsv_u8_conv(img * (1.0 / 255.0))
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)


def _hsv_u8_conv_to_bgr(img: np.ndarray, is_float: bool) -> np.ndarray:
    clipped = np.clip(img, 0, 255)
    if is_float:
        from .grading import _hsv_u8_conv_to_bgr_f
        return np.clip(_hsv_u8_conv_to_bgr_f(clipped) * 255.0, 0.0, 255.0)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_HSV2BGR)


class EyeEnhancer:
    """Professional eye enhancement pipeline."""

    def enhance(
        self,
        img_bgr: np.ndarray,
        regions: Any,
        strength: int = 40,
        catchlight_strength: Optional[int] = None,
        vessel_strength: Optional[int] = None,
    ) -> np.ndarray:
        """Run the full eye enhancement pipeline.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            regions: FaceRegions from the parser.
            strength: 0–100 overall intensity.
            catchlight_strength: 0–100 catchlight-specific intensity. If None,
                falls back to ``strength`` (backward compatible).

        Returns:
            (H, W, 3) result matching input dtype.
        """
        if strength <= 0 and (vessel_strength is None or vessel_strength <= 0):
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        s = strength / 100.0
        result = img_bgr.copy()

        # Eye whites — combine both eyes
        whites_mask_l = getattr(regions, "left_sclera", None)
        if whites_mask_l is None:
            whites_mask_l = np.clip(regions.left_eye - regions.left_iris, 0, 1)
        whites_mask_r = getattr(regions, "right_sclera", None)
        if whites_mask_r is None:
            whites_mask_r = np.clip(regions.right_eye - regions.right_iris, 0, 1)
        whites_mask = np.clip(whites_mask_l + whites_mask_r, 0, 1)
        result = self._enhance_whites(result, whites_mask, s)

        # Sclera vessel removal — iris-safe, runs only inside whites_mask
        v_s = vessel_strength if vessel_strength is not None else strength
        if v_s > 0:
            result = self._remove_sclera_vessels(result, whites_mask, v_s)

        # Iris — sculpt each eye separately
        if regions.left_iris is not None and regions.left_iris.max() > 0.01:
            result = self._sculpt_iris(result, regions.left_iris, s)
        if regions.right_iris is not None and regions.right_iris.max() > 0.01:
            result = self._sculpt_iris(result, regions.right_iris, s)

        # Catchlights — detect and amplify existing highlights
        iris_mask = np.clip(regions.left_iris + regions.right_iris, 0, 1)
        cl_s = catchlight_strength if catchlight_strength is not None else strength
        result = self._enhance_catchlights(result, iris_mask, cl_s / 100.0)

        # Specular catchlight boost — small +10% on detected iris specular pixels
        if catchlight_strength and catchlight_strength > 0:
            s_spec = catchlight_strength / 100.0
            boost = 0.10 * s_spec
            for iris_m in (regions.left_iris, regions.right_iris):
                if iris_m is None or iris_m.max() < 0.01:
                    continue
                lab = _to_lab(result, is_float)
                l_chan = lab[:, :, 0]
                specular = np.clip((l_chan - 220.0) / 20.0, 0.0, 1.0) * (iris_m > 0.1).astype(np.float32)
                if specular.max() < 0.01:
                    continue
                h, w = l_chan.shape
                k = max(3, int(min(h, w) * 0.005)) | 1
                specular = cv2.GaussianBlur(specular, (k, k), 0)
                lab[:, :, 0] = np.minimum(l_chan + 255.0 * boost * specular, 255.0)
                result = _from_lab(lab, is_float)

        return result

    def _remove_sclera_vessels(
        self,
        img_bgr: np.ndarray,
        whites_mask: np.ndarray,
        strength: int,
    ) -> np.ndarray:
        """Remove thin red blood vessels from the sclera via inpaint.

        Vessels are detected as pixels redder than the local sclera median
        (relative redness in the LAB a-channel), kept thin via morphological
        opening, then inpainted. Runs only inside the iris-excluded
        ``whites_mask`` so the iris, pupil and skin are never touched.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            whites_mask: (H, W) float32 sclera mask (eye region minus iris).
            strength: 0–100 removal intensity.

        Returns:
            (H, W, 3) result matching input dtype.
        """
        if strength <= 0:
            return img_bgr
        s = strength / 100.0
        is_float = img_bgr.dtype == np.float32

        lab = _to_lab(img_bgr, is_float)
        a = lab[:, :, 1] - 128.0
        wm = whites_mask
        idx = wm > 0.1
        if idx.sum() < 10:
            return img_bgr

        med = float(np.median(a[idx]))
        sd = max(float(a[idx].std()), 1e-3)
        k = 3.0 - 2.0 * s  # higher strength -> lower threshold -> more removed
        vessel = (a - med > k * sd) & (wm > 0.25)
        vessel = vessel.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel = cv2.dilate(vessel, kernel, iterations=1)
        if vessel.sum() == 0:
            return img_bgr

        img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        inp = cv2.inpaint(
            img_u8, (vessel * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA
        )
        inp = inp.astype(np.float32)

        m = cv2.dilate(vessel, kernel, iterations=2).astype(np.float32)
        m = feather_mask(m, radius=4) * s
        out = img_bgr * (1.0 - m[:, :, None]) + inp * m[:, :, None]
        if not is_float:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _enhance_whites(
        self,
        img_bgr: np.ndarray,
        whites_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Remove redness and yellowness from eye whites.

        Works in LAB space:
            • Reduce 'a' channel (green-red) to remove redness.
            • Reduce 'b' channel (blue-yellow) to remove yellowness.
            • Slightly lift 'L' for brighter whites.
        """
        if whites_mask is None or whites_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        lab = _to_lab(img_bgr, is_float)
        
        # Only apply whitening to pixels that are relatively bright (L > 100)
        # to prevent eyelashes/eyeliner and dark shadow areas from turning gray/dusty.
        bright_sclera = (lab[:, :, 0] > 100.0).astype(np.float32)
        m = whites_mask * strength * bright_sclera

        # Reduce redness (a channel > 128 means more red)
        lab[:, :, 1] = lab[:, :, 1] - m * np.clip(lab[:, :, 1] - 128, 0, 30) * 0.4

        # Reduce yellowness (b channel > 128 means more yellow)
        lab[:, :, 2] = lab[:, :, 2] - m * np.clip(lab[:, :, 2] - 128, 0, 30) * 0.4

        # Subtle brightness lift
        lab[:, :, 0] = np.clip(lab[:, :, 0] + m * 6, 0, 255)

        return _from_lab(lab, is_float)

    def _sculpt_iris(
        self,
        img_bgr: np.ndarray,
        iris_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Perform 3D iris sculpting: darken pupil & limbal ring, brighten iris body,
        and boost micro-contrast and saturation.
        """
        if iris_mask is None or iris_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        # Find center and radius of this iris
        y_indices, x_indices = np.where(iris_mask > 0.5)
        if len(x_indices) < 5:
            return img_bgr

        cy, cx = y_indices.mean(), x_indices.mean()
        r = max(np.sqrt(len(x_indices) / np.pi), 3.0)

        # We will work in a local crop around the iris for high performance
        h, w = img_bgr.shape[:2]
        pad = int(r * 1.5)
        x1 = max(int(cx - pad), 0)
        y1 = max(int(cy - pad), 0)
        x2 = min(int(cx + pad), w)
        y2 = min(int(cy + pad), h)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return img_bgr

        crop = img_bgr[y1:y2, x1:x2].copy()
        crop_mask = iris_mask[y1:y2, x1:x2]

        # Coordinates relative to center
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r

        # 3D Sculpting masks
        # 1. Pupil mask (center dark area)
        pupil_mask = np.clip(1.0 - dist / 0.35, 0, 1) * crop_mask
        
        # 2. Iris body mask (glowing middle ring)
        # Peak at d=0.55
        body_mask = np.clip(1.0 - np.abs(dist - 0.55) / 0.25, 0, 1) * crop_mask

        # 3. Limbal ring mask (dark outer edge)
        # Peak at d=0.9
        limbal_mask = np.clip(1.0 - np.abs(dist - 0.9) / 0.2, 0, 1) * crop_mask

        # Apply edits in LAB and HSV spaces on the crop
        lab = _to_lab(crop, is_float)

        # Pupil: darken
        lab[:, :, 0] = np.clip(lab[:, :, 0] - pupil_mask * 45.0 * strength, 0, 255)

        # Iris body: brighten
        lab[:, :, 0] = np.clip(lab[:, :, 0] + body_mask * 25.0 * strength, 0, 255)

        # Limbal ring: darken
        lab[:, :, 0] = np.clip(lab[:, :, 0] - limbal_mask * 30.0 * strength, 0, 255)

        crop_edited = _from_lab(lab, is_float)

        # HSV saturation boost on the iris body
        hsv = _bgr_to_hsv_u8_conv(crop_edited, is_float)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + body_mask * hsv[:, :, 1] * 0.45 * strength, 0, 255)
        crop_edited = _hsv_u8_conv_to_bgr(hsv, is_float)

        # Sharpness via unsharp mask on the crop
        blurred = cv2.GaussianBlur(crop_edited, (3, 3), 1.0)
        crop_sharp = cv2.addWeighted(crop_edited, 1.0 + strength * 0.5, blurred, -strength * 0.5, 0)

        # Blend the edited crop back into the image
        img_bgr_out = img_bgr.copy()
        m = crop_mask[:, :, np.newaxis]
        blended = np.clip(
            crop.astype(np.float32) * (1.0 - m) + crop_sharp.astype(np.float32) * m,
            0, 255
        )
        if is_float:
            img_bgr_out[y1:y2, x1:x2] = blended
        else:
            img_bgr_out[y1:y2, x1:x2] = blended.astype(np.uint8)

        return img_bgr_out

    def _enhance_catchlights(
        self,
        img_bgr: np.ndarray,
        iris_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Detect and amplify existing catchlights. Never create fakes.

        Catchlights are small, bright specular highlights in the iris.
        We find them via thresholding in the iris region and amplify only
        existing ones.
        """
        if iris_mask is None or iris_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        if is_float:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Only look within iris
        iris_gray = gray * iris_mask

        # Detect existing bright spots (catchlights)
        # Use adaptive threshold relative to iris brightness
        iris_pixels = gray[iris_mask > 0.5]
        if len(iris_pixels) == 0:
            return img_bgr

        iris_mean = iris_pixels.mean()
        iris_std = max(iris_pixels.std(), 1.0)
        threshold = iris_mean + iris_std * 1.2

        catchlight_mask = ((iris_gray > threshold) & (iris_mask > 0.3)).astype(np.float32)

        if catchlight_mask.sum() < 2:
            return img_bgr

        # Feather the catchlight mask
        catchlight_mask = feather_mask(catchlight_mask, radius=2)

        # Amplify: brighten catchlight areas using a safe soft-clipping formula
        # to subtly enhance the catchlights without blowing them out (max 0.25 boost coefficient)
        lab = _to_lab(img_bgr, is_float)
        l_chan = lab[:, :, 0]
        
        l_boost = (255.0 - l_chan) * catchlight_mask * 0.25 * strength
        lab[:, :, 0] = np.clip(l_chan + l_boost, 0, 255)
        
        return _from_lab(lab, is_float)

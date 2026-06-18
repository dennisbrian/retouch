"""Colour grading presets — final output tone and mood.

Presets:
    natural   — warm, soft, Instagram-style.
    magazine  — cool shadows, matte, fashion editorial.
    beauty    — clean, bright, commercial beauty ad.
    cosplay   — vibrant saturated, anime-inspired colours.
"""

import cv2
import numpy as np

from .utils import apply_curve


PRESETS = {
    "natural": {
        "description": "Warm, soft, Instagram-style natural look",
        "curves": {
            # Gentle S-curve for contrast
            "L": [(0, 0), (40, 35), (128, 132), (200, 210), (255, 255)],
        },
        "warmth": 0.04,           # slight warm shift
        "saturation_boost": 0.05, # subtle boost
        "shadow_lift": 5,         # don't crush blacks
        "vignette": 0.0,
    },
    "magazine": {
        "description": "Cool shadows, matte, fashion editorial look",
        "curves": {
            # Lifted blacks, compressed highlights for matte look
            "L": [(0, 15), (30, 35), (128, 128), (220, 215), (255, 240)],
        },
        "warmth": -0.02,          # cool shift
        "saturation_boost": -0.05, # slightly desaturated
        "shadow_lift": 15,        # matte lifted shadows
        "vignette": 0.15,
        "split_tone": {
            "shadows": (110, 135),  # teal shadows (a, b in LAB)
            "highlights": (132, 115), # warm highlights
        },
    },
    "beauty": {
        "description": "Clean, bright, commercial beauty",
        "curves": {
            # Soft contrast, bright midtones
            "L": [(0, 5), (50, 55), (128, 140), (200, 210), (255, 255)],
        },
        "warmth": 0.01,
        "saturation_boost": 0.03,
        "shadow_lift": 3,
        "vignette": 0.0,
        "glow": 0.04,             # 4% soft bloom
    },
    "cosplay": {
        "description": "Vibrant saturated, anime-inspired colours",
        "curves": {
            # Strong S-curve for punchy contrast
            "L": [(0, 0), (50, 35), (128, 135), (200, 220), (255, 255)],
        },
        "warmth": -0.01,          # slightly cooler whites
        "saturation_boost": 0.18, # vivid colours
        "shadow_lift": 0,
        "vignette": 0.08,
        "clarity": 0.3,           # micro-contrast boost
        "glow": 0.06,             # 6% soft bloom
    },
    "dreamy": {
        "description": "Dreamy soft-focus look with warm highlights and pastel shadows",
        "curves": {
            # Soft contrast, slightly lifted shadows
            "L": [(0, 10), (45, 50), (128, 132), (200, 215), (255, 250)],
        },
        "warmth": 0.03,
        "saturation_boost": 0.08,
        "shadow_lift": 8,
        "vignette": 0.05,
        "glow": 0.08,             # 8% soft bloom
        "split_tone": {
            "shadows": (124, 126),  # cool/pastel shadows (ab in LAB)
            "highlights": (133, 131), # warm highlights
        },
    },
    "anime": {
        "description": "Vibrant, high-contrast, anime-inspired colors",
        "curves": {
            # Strong S-curve for punchy contrast
            "L": [(0, 0), (40, 30), (128, 135), (210, 230), (255, 255)],
        },
        "warmth": 0.01,
        "saturation_boost": 0.22,
        "shadow_lift": 0,
        "vignette": 0.06,
        "glow": 0.05,             # 5% soft bloom
        "clarity": 0.15,
    },
}


class ColorGrader:
    """Apply colour grading presets to finalise image tone/mood."""

    def grade(self, img_bgr, preset="natural", intensity=1.0):
        """Apply a colour grading preset.

        Args:
            img_bgr: (H, W, 3) uint8.
            preset: Preset name or dict of settings.
            intensity: 0.0–1.0 blend with original.

        Returns:
            (H, W, 3) uint8 graded image.
        """
        if isinstance(preset, str):
            settings = PRESETS.get(preset, PRESETS["natural"])
        else:
            settings = preset

        result = img_bgr.copy()

        # ---- Tone curve (in LAB luminance) ----
        if "curves" in settings and "L" in settings["curves"]:
            result = self._apply_luminance_curve(result, settings["curves"]["L"])

        # ---- Shadow lift ----
        if settings.get("shadow_lift", 0) > 0:
            result = self._lift_shadows(result, settings["shadow_lift"])

        # ---- Warmth shift ----
        if abs(settings.get("warmth", 0)) > 0.001:
            result = self._adjust_warmth(result, settings["warmth"])

        # ---- Saturation ----
        if abs(settings.get("saturation_boost", 0)) > 0.001:
            result = self._adjust_saturation(result, settings["saturation_boost"])

        # ---- Split toning ----
        if "split_tone" in settings:
            result = self._split_tone(result, settings["split_tone"])

        # ---- Clarity (micro-contrast) ----
        if settings.get("clarity", 0) > 0:
            result = self._add_clarity(result, settings["clarity"])

        # ---- Vignette ----
        if settings.get("vignette", 0) > 0:
            result = self._add_vignette(result, settings["vignette"])

        # ---- Smart Glow (Soft Bloom) ----
        if settings.get("glow", 0) > 0:
            result = self._add_glow(result, settings["glow"])

        # ---- Blend with original by intensity ----
        if intensity < 1.0:
            result = cv2.addWeighted(img_bgr, 1.0 - intensity, result, intensity, 0)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_glow(self, img_bgr, opacity):
        """Create a soft glow/bloom layer confined strictly to highlights (L > 180)."""
        if opacity <= 0:
            return img_bgr

        img_f = img_bgr.astype(np.float32)
        h, w = img_bgr.shape[:2]

        # Heavy Gaussian blur proportional to image dimensions
        ksize = max(int(min(h, w) * 0.05), 15) | 1
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)

        # Screen blend mode formula: 255 - (255 - A) * (255 - B) / 255
        screen = 255.0 - (255.0 - img_f) * (255.0 - blurred) / 255.0

        # Mask strictly to highlights (L > 180 in LAB space)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)

        # Soft transition: 0 at L=180, 1 at L=220
        highlight_mask = np.clip((l_chan - 180.0) / 40.0, 0, 1)[:, :, np.newaxis]

        # Blend original with screen layer based on mask and opacity
        result = img_f * (1.0 - highlight_mask * opacity) + screen * (highlight_mask * opacity)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_luminance_curve(self, img, curve_points):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = apply_curve(lab[:, :, 0], curve_points)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _lift_shadows(self, img, lift):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        # Only lift dark pixels (below midtone)
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        lab[:, :, 0] = np.clip(l + shadow_mask * lift, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _adjust_warmth(self, img, warmth):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        # b channel: positive warmth = more yellow, negative = more blue
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth * 30, 0, 255)
        # a channel: slight push for warm feel
        lab[:, :, 1] = np.clip(lab[:, :, 1] + warmth * 10, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _adjust_saturation(self, img, boost):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + boost), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _split_tone(self, img, tones):
        """Apply split toning: colour shadows and highlights independently."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]

        shadow_weight = np.clip(1.0 - l / 128.0, 0, 1)[:, :, np.newaxis]
        highlight_weight = np.clip((l - 128.0) / 128.0, 0, 1)[:, :, np.newaxis]

        shadow_ab = np.array(tones["shadows"], dtype=np.float32)
        highlight_ab = np.array(tones["highlights"], dtype=np.float32)

        ab = lab[:, :, 1:3]
        ab = ab + (shadow_ab - 128) * shadow_weight * 0.15
        ab = ab + (highlight_ab - 128) * highlight_weight * 0.15

        lab[:, :, 1:3] = np.clip(ab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _add_clarity(self, img, strength):
        """Micro-contrast via unsharp mask on luminance."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        blurred = cv2.GaussianBlur(l, (0, 0), 10)
        l_sharp = np.clip(l + (l - blurred) * strength, 0, 255)
        lab[:, :, 0] = l_sharp.astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _add_vignette(self, img, strength):
        """Radial darkening vignette."""
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - (dist / max_dist) ** 2 * strength
        vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

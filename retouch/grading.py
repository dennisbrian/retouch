"""Colour grading presets — final output tone and mood.

Presets:
    natural   — warm, soft, Instagram-style.
    magazine  — cool shadows, matte, fashion editorial.
    beauty    — clean, bright, commercial beauty ad.
    cosplay   — vibrant saturated, anime-inspired colours.
"""

import cv2
import numpy as np

from .utils import apply_curve, blend_masked


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
        "glow": 0.02,             # 2% soft bloom (reduced by 50%)
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
        "glow": 0.03,             # 3% soft bloom (reduced by 50%)
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
        "glow": 0.04,             # 4% soft bloom (reduced by 50%)
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
        "glow": 0.02,             # 2% soft bloom (reduced by 60%)
        "clarity": 0.15,
    },
    "scifi": {
        "description": "Cool cyan shadows, pink/magenta highlights, vibrant neon styling",
        "curves": {
            # High-contrast S-curve with slightly lifted blacks
            "L": [(0, 5), (45, 25), (128, 130), (210, 235), (255, 255)],
        },
        "warmth": -0.04,          # cool/cyan highlights
        "saturation_boost": 0.24, # highly saturated neon colors
        "shadow_lift": 5,
        "vignette": 0.05,
        "clarity": 0.25,
        "glow": 0.05,             # 5% bloom (reduced by 50%)
        "split_tone": {
            "shadows": (110, 122),    # deep teal/cyan shadows (a=110, b=122 in LAB)
            "highlights": (134, 112), # pink/magenta highlights (a=134, b=112 in LAB)
        },
    },
    "cyber_doll": {
        "description": "High-impact cyan/pink cosplay doll finish",
        "curves": {
            "L": [(0, 0), (35, 20), (128, 136), (205, 238), (255, 255)],
        },
        "rgb_curves": {
            "B": [(0, 8), (110, 140), (255, 255)],
            "R": [(0, 0), (128, 128), (210, 238), (255, 255)],
        },
        "warmth": -0.05,
        "saturation_boost": 0.30,
        "shadow_lift": 2,
        "vignette": 0.08,
        "clarity": 0.34,
        "glow": 0.07,             # 7% bloom (reduced by 50%)
        "glow_tint": (235, 135, 210),
        "chromatic_aberration": 1.2,
        "split_tone": {
            "shadows": (108, 119),
            "highlights": (138, 111),
        },
    },
    "film": {
        "description": "Warm film look (Kodak Portra style) with cyan shadows, lifted blacks, and grain",
        "curves": {
            "L": [(0, 10), (50, 45), (128, 130), (200, 215), (255, 245)],
        },
        "rgb_curves": {
            "R": [(0, 0), (128, 132), (255, 255)],
            "B": [(0, 5), (128, 122), (255, 250)],
        },
        "warmth": 0.03,
        "saturation_boost": 0.10,
        "shadow_lift": 10,
        "vignette": 0.05,
        "grain": 0.08,
        "lut": "kodak",
        "split_tone": {
            "shadows": (120, 124),     # subtle green/blue shadows
            "highlights": (131, 133),  # warm orange highlights
        },
    },
    "cyberpunk": {
        "description": "Neon cyberpunk: teal midtones, magenta/pink highlights, heavy bloom, and chromatic aberration",
        "curves": {
            "L": [(0, 0), (45, 30), (128, 135), (210, 230), (255, 255)],
        },
        "rgb_curves": {
            "B": [(0, 0), (128, 145), (255, 255)],
            "R": [(0, 0), (128, 120), (255, 255)],
        },
        "warmth": -0.05,
        "saturation_boost": 0.25,
        "shadow_lift": 0,
        "vignette": 0.08,
        "glow": 0.06,             # 6% bloom (reduced by 50%)
        "glow_tint": (220, 110, 180), # pink-magenta glow (BGR)
        "chromatic_aberration": 4.0,
        "split_tone": {
            "shadows": (110, 122),     # teal shadows
            "highlights": (134, 112),  # pink highlights
        },
    },
    "golden_hour": {
        "description": "Rich golden warm look with soft sun-drenched glow and halation",
        "curves": {
            "L": [(0, 0), (50, 48), (128, 138), (200, 215), (255, 255)],
        },
        "warmth": 0.06,
        "saturation_boost": 0.15,
        "shadow_lift": 5,
        "glow": 0.04,             # 4% glow (reduced by 50%)
        "glow_tint": (120, 200, 255), # golden-yellow glow (BGR)
        "halation": {
            "threshold": 210,
            "radius": 21,
            "intensity": 0.4
        },
    },
    "bw_noir": {
        "description": "High-contrast dramatic black and white",
        "curves": {
            "L": [(0, 0), (50, 30), (128, 128), (205, 225), (255, 255)],
        },
        "saturation_boost": -1.0, # pure black and white
        "vignette": 0.18,
        "clarity": 0.4,
        "grain": 0.06,
    },
    "fantasy": {
        "description": "Ethereal fantasy goddess: low contrast, cool blue/purple shadows, silver highlights, clarity reduction",
        "curves": {
            # Lifted blacks, flattened highlights for low contrast
            "L": [(0, 12), (64, 72), (128, 128), (192, 184), (255, 245)],
        },
        "warmth": -0.03,            # gentler cool shift to protect skin healthy tone
        "saturation_boost": -0.02,  # less desaturation
        "shadow_lift": 10,          # moderate shadow lift to keep face structure
        "vignette": 0.0,
        "clarity": -0.08,           # gentler soft focus (avoiding muddy/dirty face)
        "glow": 0.12,               # 12% bloom (reduced from 28% to prevent soft scaling)
        "glow_tint": (255, 235, 242), # cool white/silver glow (BGR)
        "haze": 0.05,               # 5% atmospheric haze (reduced from 12% to prevent soft scaling)
        "sparkles": 0.25,           # procedurally place sparkles on highlight peaks
        "split_tone": {
            "shadows": (124, 124),    # cleaner shadows (less cyan/green cast)
            "highlights": (128, 127), # clean highlights
        }
    },
    "pink_dream": {
        "description": "Xiaohongshu inspired pink anime cosplay style with blue-indigo shadows",
        "curves": {
            "L": [(0, 6), (64, 68), (128, 140), (192, 204), (255, 248)],
        },
        "rgb_curves": {
            "R": [(0, 2), (64, 66), (128, 138), (192, 202), (255, 252)],
            "G": [(0, 0), (64, 54), (128, 120), (192, 186), (255, 242)],
            "B": [(0, 22), (64, 72), (128, 130), (192, 192), (255, 245)],
        },
        "white_balance": {
            "R": 1.04, "G": 0.98, "B": 0.98
        },
        "calibration": {
            "red": {"hue": 8.0, "sat": 15.0},
            "green": {"hue": -30.0, "sat": -20.0},
            "blue": {"hue": 10.0, "sat": 8.0}
        },
        "hsl_adjustments": {
            "hue": {"red": 5, "orange": -10, "yellow": -30, "cyan": -10, "blue": -10, "purple": 8, "magenta": 8},
            "saturation": {"red": 12, "orange": -10, "yellow": -65, "green": -85, "cyan": -20, "blue": 18, "purple": 30, "magenta": 30},
            "luminance": {"red": 10, "orange": 22, "yellow": 0, "cyan": 0, "blue": 8, "purple": 8, "magenta": 18}
        },
        "split_tone_three_way": {
            "shadows": {"hue": 255.0, "sat": 25.0},
            "midtones": {"hue": 325.0, "sat": 12.0},
            "highlights": {"hue": 215.0, "sat": 6.0},
            "balance": 10.0
        },
        "glow": 0.025,              # 2.5% glow (reduced from 5%)
        "orton_glow": 0.01,         # 1% Orton glow (reduced from 2%)
        "vignette": 0.06,
        "grain": 0.02,
    },
    "blue_dream": {
        "description": "Dreamy soft cool cosplay style with cool cyan highlights and blue shadows",
        "curves": {
            "L": [(0, 10), (32, 28), (128, 128), (220, 215), (255, 248)],
        },
        "white_balance": {
            "R": 0.94, "G": 1.00, "B": 1.06
        },
        "calibration": {
            "red": {"hue": -5.0, "sat": -5.0},
            "green": {"hue": 10.0, "sat": 5.0},
            "blue": {"hue": -10.0, "sat": 10.0}
        },
        "split_tone_three_way": {
            "shadows": {"hue": 240.0, "sat": 18.0},
            "midtones": {"hue": 190.0, "sat": 12.0},
            "highlights": {"hue": 60.0, "sat": 8.0},
            "balance": -10.0
        },
        "glow": 0.08,               # 8% glow (reduced from 20%)
        "orton_glow": 0.04,         # 4% Orton glow (reduced from 10%)
        "haze": 0.04,               # 4% haze (reduced from 10%)
        "vignette": 0.05,
        "grain": 0.03,
    },
    "xhs_ultrasoft": {
        "description": "Xiaohongshu bright flat contrast style with clean skin highlights",
        "curves": {
            "L": [(0, 20), (45, 42), (128, 130), (210, 215), (255, 240)],
        },
        "hsl_adjustments": {
            "hue": {"orange": -4, "yellow": -15},
            "saturation": {"orange": -20, "yellow": -45},
            "luminance": {"orange": 30, "yellow": 5}
        },
        "split_tone_three_way": {
            "shadows": {"hue": 270.0, "sat": 8.0},
            "midtones": {"hue": 340.0, "sat": 8.0},
            "highlights": {"hue": 50.0, "sat": 6.0},
            "balance": 0.0
        },
        "glow": 0.06,               # 6% glow (reduced from 15%)
        "orton_glow": 0.06,         # 6% Orton glow (reduced from 15%)
        "grain": 0.02,
    },
    "meitu_clone": {
        "description": "Meitu inspired vibrant pink cosplay finish",
        "curves": {
            "L": [(0, 20), (64, 85), (128, 143), (192, 198), (255, 245)],
        },
        "rgb_curves": {
            "R": [(0, 5), (64, 70), (128, 138), (192, 200), (255, 255)],
            "G": [(0, 0), (64, 62), (128, 126), (192, 190), (255, 250)],
            "B": [(0, 10), (64, 58), (128, 118), (192, 185), (255, 240)],
        },
        "white_balance": {
            "R": 1.08, "G": 1.00, "B": 0.92
        },
        "calibration": {
            "red": {"hue": 12.0, "sat": 15.0},
            "green": {"hue": -15.0, "sat": -5.0},
            "blue": {"hue": 15.0, "sat": -10.0}
        },
        "hsl_adjustments": {
            "hue": {"red": 5, "orange": -8, "yellow": -30, "cyan": -50, "blue": -25, "purple": 10, "magenta": 10},
            "saturation": {"red": 8, "orange": -15, "yellow": -60, "green": -80, "cyan": -70, "blue": -50, "purple": 15, "magenta": 20},
            "luminance": {"red": 10, "orange": 25, "yellow": 0, "cyan": 0, "blue": 15, "purple": 15, "magenta": 25}
        },
        "split_tone_three_way": {
            "shadows": {"hue": 285.0, "sat": 8.0},
            "midtones": {"hue": 325.0, "sat": 6.0},
            "highlights": {"hue": 40.0, "sat": 6.0},
            "balance": 20.0
        },
        "glow": 0.02,              # 2% glow (reduced from 4%)
        "orton_glow": 0.01,         # 1% Orton glow (reduced from 2%)
        "vignette": 0.08,
        "grain": 0.02,
    },
}


class ColorGrader:
    """Apply colour grading presets to finalise image tone/mood."""

    def grade(self, img_bgr, preset="natural", intensity=1.0, split_tone_mask=None, glow_mask=None, haze_mask=None):
        """Apply a colour grading preset.

        Args:
            img_bgr: (H, W, 3) uint8.
            preset: Preset name or dict of settings.
            intensity: 0.0–1.0 blend with original.
            split_tone_mask: Optional mask to apply split-toning only to specific regions.
            glow_mask: Optional mask to restrict glow/bloom (smart bloom and Orton glow).

        Returns:
            (H, W, 3) uint8 graded image.
        """
        if isinstance(preset, str):
            settings = PRESETS.get(preset, PRESETS["natural"])
        else:
            settings = preset

        result = img_bgr.copy()

        # ---- White balance ----
        if "white_balance" in settings:
            result = self._adjust_white_balance(result, settings["white_balance"])

        # ---- Tone curve (in LAB luminance) ----
        if "curves" in settings and "L" in settings["curves"]:
            result = self._apply_luminance_curve(result, settings["curves"]["L"])

        # ---- RGB curves ----
        if "rgb_curves" in settings:
            result = self._apply_rgb_curves(result, settings["rgb_curves"])

        # ---- Shadow lift ----
        if settings.get("shadow_lift", 0) > 0:
            result = self._lift_shadows(result, settings["shadow_lift"])

        # ---- Primary Color Calibration ----
        if "calibration" in settings:
            result = self._apply_calibration(result, settings["calibration"])

        # ---- Warmth shift ----
        if abs(settings.get("warmth", 0)) > 0.001:
            result = self._adjust_warmth(result, settings["warmth"])

        # ---- Saturation ----
        if abs(settings.get("saturation_boost", 0)) > 0.001:
            result = self._adjust_saturation(result, settings["saturation_boost"])

        # ---- HSL selective color adjustments ----
        if "hsl_adjustments" in settings:
            result = self._apply_hsl_adjustments(result, settings["hsl_adjustments"])
        elif "hsl_hue_shift" in settings:
            result = self._hsl_hue_shift(result, settings["hsl_hue_shift"])

        # ---- Split toning ----
        if "split_tone_three_way" in settings:
            result = self._split_tone_three_way(result, settings["split_tone_three_way"], split_tone_mask)
        elif "split_tone" in settings:
            result = self._split_tone(result, settings["split_tone"], split_tone_mask)

        # ---- Clarity (micro-contrast) ----
        if settings.get("clarity", 0) != 0:
            result = self._add_clarity(result, settings["clarity"])

        # ---- Atmospheric Haze ----
        if settings.get("haze", 0) > 0:
            result = self._add_haze(result, settings["haze"], mask=haze_mask)

        # ---- Halation ----
        if "halation" in settings:
            h_conf = settings["halation"]
            result = self._add_halation(
                result,
                threshold=h_conf.get("threshold", 200),
                radius=h_conf.get("radius", 15),
                intensity=h_conf.get("intensity", 0.3)
            )

        # ---- Vignette ----
        if settings.get("vignette", 0) > 0:
            result = self._add_vignette(result, settings["vignette"])

        # ---- Smart Glow (Soft Bloom) ----
        if settings.get("glow", 0) > 0:
            result = self._add_glow(
                result,
                settings["glow"],
                tint=settings.get("glow_tint", None),
                mask=glow_mask
            )

        # ---- Orton Glow ----
        if settings.get("orton_glow", 0) > 0:
            result = self._add_orton_glow(result, settings["orton_glow"], mask=glow_mask)

        # ---- Sparkles overlay ----
        if settings.get("sparkles", 0) > 0:
            result = self._add_sparkles(result, settings["sparkles"])

        # ---- Chromatic Aberration ----
        if settings.get("chromatic_aberration", 0) > 0:
            result = self._add_chromatic_aberration(result, settings["chromatic_aberration"])

        # ---- LUT Emulation ----
        if "lut" in settings:
            result = self._add_lut_emulation(result, settings["lut"])

        # ---- Grain ----
        if settings.get("grain", 0) > 0:
            result = self._add_grain(result, settings["grain"])

        # ---- Blend with original by intensity ----
        if intensity < 1.0:
            result = cv2.addWeighted(img_bgr, 1.0 - intensity, result, intensity, 0)

        return result

    def add_impact_finish(self, img_bgr, strength):
        """Apply a final high-impact finish without changing local face geometry."""
        if strength <= 0:
            return img_bgr

        s = np.clip(strength / 100.0, 0.0, 1.0)
        original = img_bgr.copy()
        result = img_bgr.copy()

        result = self._apply_luminance_curve(
            result,
            [
                (0, 0),
                (45, int(45 - 18 * s)),
                (128, int(128 + 7 * s)),
                (210, int(210 + 20 * s)),
                (255, 255),
            ],
        )
        result = self._adjust_saturation(result, 0.10 * s)
        result = self._add_clarity(result, 0.18 * s)
        result = self._add_glow(result, 0.06 * s, tint=(235, 140, 210))

        return cv2.addWeighted(original, 1.0 - s, result, s, 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_glow(self, img_bgr, opacity, tint=None, mask=None):
        """Create a soft glow/bloom layer confined strictly to highlights (L > 210) and spatial mask."""
        if opacity <= 0:
            return img_bgr

        img_f = img_bgr.astype(np.float32)
        h, w = img_bgr.shape[:2]

        # Reduced blur radius factor for sharper core glow (Issue 2)
        ksize = max(int(min(h, w) * 0.025), 7) | 1
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)

        if tint is not None:
            tint_arr = np.array(tint, dtype=np.float32)
            if tint_arr.max() <= 1.0:
                tint_arr = tint_arr * 255.0
            tint_layer = np.ones_like(blurred) * tint_arr
            # Use additive blending to tint the glow without suppressing cool channels (Issue 11)
            blurred = cv2.addWeighted(blurred, 0.7, tint_layer, 0.3, 0)

        # Screen blend mode formula
        screen = 255.0 - (255.0 - img_f) * (255.0 - blurred) / 255.0

        # Mask strictly to highlights (L > 225 in LAB space)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)

        # Soft transition: 0 at L=225, 1 at L=245
        highlight_mask = np.clip((l_chan - 225.0) / 20.0, 0, 1)[:, :, np.newaxis]

        # Combine with spatial mask if provided
        if mask is not None:
            m_f = mask.astype(np.float32)
            if m_f.max() > 1.0:
                m_f = m_f / 255.0
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            highlight_mask = highlight_mask * m_f

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
        s = hsv[:, :, 1]
        factor = 1.0 + boost * (1.0 - s / 255.0)
        hsv[:, :, 1] = np.clip(s * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _split_tone(self, img, tones, mask=None):
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
        split_toned = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if mask is not None:
            from .utils import blend_masked
            return blend_masked(img, split_toned, mask)
        return split_toned

    def _guided_filter(self, guide, src, r, eps):
        """Fast O(N) Guided Filter implementation using boxFilter."""
        mean_I = cv2.boxFilter(guide, -1, (r, r))
        mean_p = cv2.boxFilter(src, -1, (r, r))
        mean_Ip = cv2.boxFilter(guide * src, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, -1, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))

        return mean_a * guide + mean_b

    def _add_clarity(self, img, strength):
        """Edge-preserving micro-contrast (clarity) using Guided Filter."""
        if strength == 0:
            return img

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)

        h, w = img.shape[:2]
        r = max(int(min(h, w) * 0.015), 5)
        eps = 0.02 * (255.0 ** 2)

        base = self._guided_filter(l_chan / 255.0, l_chan / 255.0, r, eps / (255.0 ** 2)) * 255.0
        detail = l_chan - base

        l_new = np.clip(base + detail * (1.0 + strength), 0, 255)
        lab[:, :, 0] = l_new.astype(np.uint8)
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

    def _apply_rgb_curves(self, img_bgr, curves_dict):
        """Apply tone curves to R, G, and B channels individually."""
        b, g, r = cv2.split(img_bgr)
        if "R" in curves_dict:
            r = apply_curve(r, curves_dict["R"])
        if "G" in curves_dict:
            g = apply_curve(g, curves_dict["G"])
        if "B" in curves_dict:
            b = apply_curve(b, curves_dict["B"])
        return cv2.merge([b, g, r])

    def _hsl_hue_shift(self, img_bgr, shifts_dict):
        """Selective HSL hue shift per color range."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        ranges = {
            "red": [(0, 10), (170, 180)],
            "orange": [(10, 25)],
            "yellow": [(25, 35)],
            "green": [(35, 80)],
            "cyan": [(80, 105)],
            "blue": [(105, 140)],
            "magenta": [(140, 170)],
        }
        for color, shift_deg in shifts_dict.items():
            if color not in ranges or shift_deg == 0:
                continue
            shift_cv = float(shift_deg) / 2.0
            mask = np.zeros_like(h, dtype=bool)
            for low, high in ranges[color]:
                mask |= (h >= low) & (h <= high)
            hsv[:, :, 0] = np.where(mask, (h + shift_cv) % 180, h)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _add_chromatic_aberration(self, img_bgr, max_disp):
        """Radial chromatic aberration simulating lens dispersion."""
        if max_disp <= 0:
            return img_bgr
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        max_r = np.sqrt(cx*cx + cy*cy)
        factor = (r / max_r) * max_disp
        map_x_r = (x + (dx / (r + 1e-5)) * factor).astype(np.float32)
        map_y_r = (y + (dy / (r + 1e-5)) * factor).astype(np.float32)
        map_x_b = (x - (dx / (r + 1e-5)) * factor).astype(np.float32)
        map_y_b = (y - (dy / (r + 1e-5)) * factor).astype(np.float32)
        b, g, r_chan = cv2.split(img_bgr)
        r_new = cv2.remap(r_chan, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        b_new = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return cv2.merge([b_new, g, r_new])

    def _add_halation(self, img_bgr, threshold=200, radius=15, intensity=0.3):
        """Highlight halation: warm glow bleeding from extreme highlights."""
        if intensity <= 0:
            return img_bgr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        halation_color = np.zeros_like(img_bgr, dtype=np.uint8)
        halation_color[:, :, 2] = thresh
        halation_color[:, :, 1] = (thresh * 0.45).astype(np.uint8)
        ksize = radius | 1
        blurred_halation = cv2.GaussianBlur(halation_color, (ksize, ksize), 0).astype(np.float32)
        img_f = img_bgr.astype(np.float32)
        screen = 255.0 - (255.0 - img_f) * (255.0 - blurred_halation * intensity) / 255.0
        return np.clip(screen, 0, 255).astype(np.uint8)

    def _add_grain(self, img_bgr, strength):
        """Generate organic, luminance-weighted film grain peaking in midtones."""
        if strength <= 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        gw, gh = max(w // 2, 64), max(h // 2, 64)
        noise = np.random.normal(0, 255.0 * strength, (gh, gw)).astype(np.float32)
        noise_scaled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noise_blurred = cv2.GaussianBlur(noise_scaled, (3, 3), 0.5)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)

        # Weight peaks at 128 (midtone) and tapers off towards 0 and 255
        weight = np.exp(-((l_chan - 128.0) ** 2) / (2.0 * 64.0 ** 2))

        weighted_noise = (noise_blurred * weight)[:, :, np.newaxis]
        img_f = img_bgr.astype(np.float32)
        return np.clip(img_f + weighted_noise, 0, 255).astype(np.uint8)

    def _adjust_white_balance(self, img_bgr, multipliers):
        """Adjust white balance using channel multipliers."""
        b_mult = multipliers.get("B", 1.0)
        g_mult = multipliers.get("G", 1.0)
        r_mult = multipliers.get("R", 1.0)

        if abs(b_mult - 1.0) < 0.001 and abs(g_mult - 1.0) < 0.001 and abs(r_mult - 1.0) < 0.001:
            return img_bgr

        img_f = img_bgr.astype(np.float32)
        img_f[:, :, 0] *= b_mult
        img_f[:, :, 1] *= g_mult
        img_f[:, :, 2] *= r_mult
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_calibration(self, img_bgr, calibration):
        """Apply primary color calibration slider adjustments."""
        if not calibration:
            return img_bgr

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]

        r_hue_shift = calibration.get("red", {}).get("hue", 0) * 0.15 / 2.0
        r_sat_shift = calibration.get("red", {}).get("sat", 0) * 0.005

        g_hue_shift = calibration.get("green", {}).get("hue", 0) * 0.15 / 2.0
        g_sat_shift = calibration.get("green", {}).get("sat", 0) * 0.005

        b_hue_shift = calibration.get("blue", {}).get("hue", 0) * 0.15 / 2.0
        b_sat_shift = calibration.get("blue", {}).get("sat", 0) * 0.005

        if r_hue_shift == 0 and r_sat_shift == 0 and g_hue_shift == 0 and g_sat_shift == 0 and b_hue_shift == 0 and b_sat_shift == 0:
            return img_bgr

        sigma = 15.0

        # Compute weights for red, green, blue primary regions
        dist_r = np.minimum(np.abs(h - 0.0), np.abs(h - 180.0))
        w_r = np.exp(-(dist_r ** 2) / (2.0 * sigma ** 2))

        dist_g = np.abs(h - 60.0)
        w_g = np.exp(-(dist_g ** 2) / (2.0 * sigma ** 2))

        dist_b = np.abs(h - 120.0)
        w_b = np.exp(-(dist_b ** 2) / (2.0 * sigma ** 2))

        h_new = (h + w_r * r_hue_shift + w_g * g_hue_shift + w_b * b_hue_shift) % 180.0
        s_new = np.clip(s * (1.0 + w_r * r_sat_shift + w_g * g_sat_shift + w_b * b_sat_shift), 0, 255)

        hsv[:, :, 0] = h_new
        hsv[:, :, 1] = s_new

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _add_orton_glow(self, img_bgr, opacity, blur_radius=35, mask=None):
        """Dreamy Orton glow finish."""
        if opacity <= 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float32) / 255.0

        # Scale blur radius proportional to image size (Issue 9)
        ksize = max(blur_radius, int(min(h, w) * 0.015)) | 1
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)

        # Soft Light Pegtop blending
        soft_light = (1.0 - 2.0 * blurred) * (img_f ** 2) + 2.0 * blurred * img_f

        result = img_f * (1.0 - opacity) + soft_light * opacity
        orton_img = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        if mask is not None:
            from .utils import blend_masked
            return blend_masked(img_bgr, orton_img, mask)
        return orton_img

    def _split_tone_three_way(self, img_bgr, tones, mask=None):
        """Apply high-end 3-way split toning (shadows, midtones, highlights) in LAB space."""
        if not tones:
            return img_bgr

        shadows = tones.get("shadows", {})
        midtones = tones.get("midtones", {})
        highlights = tones.get("highlights", {})
        balance = tones.get("balance", 0.0)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_val = lab[:, :, 0]
        l_norm = l_val / 255.0

        def hsl_to_lab_offsets(hue, sat):
            if hue is None or sat is None or sat == 0:
                return 0.0, 0.0
            theta = np.radians(hue)
            chroma = (sat / 100.0) * 25.0
            a = chroma * np.cos(theta)
            b = chroma * np.sin(theta)
            return a, b

        s_a, s_b = hsl_to_lab_offsets(shadows.get("hue"), shadows.get("sat"))
        m_a, m_b = hsl_to_lab_offsets(midtones.get("hue"), midtones.get("sat"))
        h_a, h_b = hsl_to_lab_offsets(highlights.get("hue"), highlights.get("sat"))

        if abs(s_a) < 0.01 and abs(s_b) < 0.01 and abs(m_a) < 0.01 and abs(m_b) < 0.01 and abs(h_a) < 0.01 and abs(h_b) < 0.01:
            return img_bgr

        shift = (balance / 100.0) * 0.2

        w_shadow = 1.0 - 1.0 / (1.0 + np.exp(-(l_norm - (0.3 + shift)) * 10.0))
        w_highlight = 1.0 / (1.0 + np.exp(-(l_norm - (0.7 + shift)) * 10.0))
        w_midtone = np.clip(1.0 - w_shadow - w_highlight, 0.0, 1.0)

        ab = lab[:, :, 1:3].copy()
        ab[:, :, 0] += (s_a * w_shadow + m_a * w_midtone + h_a * w_highlight)
        ab[:, :, 1] += (s_b * w_shadow + m_b * w_midtone + h_b * w_highlight)

        lab[:, :, 1:3] = np.clip(ab, 0, 255)
        split_toned = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if mask is not None:
            from .utils import blend_masked
            return blend_masked(img_bgr, split_toned, mask)
        return split_toned

    def _add_lut_emulation(self, img_bgr, lut_preset):
        """3D LUT approximation via polynomial mapping."""
        presets = {
            "kodak": {
                "R": [0.00001 * (x**2) + 0.8 * x for x in range(256)],
                "G": [0.95 * x for x in range(256)],
                "B": [-0.00002 * (x**2) + 1.1 * x for x in range(256)],
            },
            "fuji": {
                "R": [0.9 * x for x in range(256)],
                "G": [0.000015 * (x**2) + 0.85 * x for x in range(256)],
                "B": [1.05 * x for x in range(256)],
            }
        }
        if lut_preset not in presets:
            return img_bgr
        b, g, r = cv2.split(img_bgr)
        lut_data = presets[lut_preset]
        lut_b = np.clip(lut_data["B"], 0, 255).astype(np.uint8)
        lut_g = np.clip(lut_data["G"], 0, 255).astype(np.uint8)
        lut_r = np.clip(lut_data["R"], 0, 255).astype(np.uint8)
        b = cv2.LUT(b, lut_b)
        g = cv2.LUT(g, lut_g)
        r = cv2.LUT(r, lut_r)
        return cv2.merge([b, g, r])

    def grade_stack(self, img_bgr, preset_weights):
        """Chain/blend multiple presets with custom weights."""
        if not preset_weights:
            return img_bgr
        total_w = sum(preset_weights.values())
        if total_w <= 0:
            return img_bgr
        result = img_bgr.copy()
        for preset_name, weight in preset_weights.items():
            if weight <= 0:
                continue
            result = self.grade(result, preset_name, weight)
        return result

    def _apply_hsl_adjustments(self, img_bgr, adjustments):
        """Apply Hue, Saturation, and Luminance adjustments per color range."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        ranges = {
            "red": [(0, 8), (172, 180)],
            "orange": [(8, 20)],
            "yellow": [(20, 35)],
            "green": [(35, 80)],
            "cyan": [(80, 105)],
            "blue": [(105, 140)],
            "magenta": [(140, 172)],
        }
        
        hue_adj = adjustments.get("hue", {})
        sat_adj = adjustments.get("saturation", {})
        lum_adj = adjustments.get("luminance", {})
        
        for color, r_list in ranges.items():
            h_shift = hue_adj.get(color, 0)
            s_shift = sat_adj.get(color, 0)
            l_shift = lum_adj.get(color, 0)
            
            if h_shift == 0 and s_shift == 0 and l_shift == 0:
                continue
                
            mask = np.zeros_like(h, dtype=bool)
            for low, high in r_list:
                mask |= (h >= low) & (h <= high)
                
            if h_shift != 0:
                # HSV hue range is [0, 180], so degrees shift is divided by 2
                shift_cv = float(h_shift) / 2.0
                h = np.where(mask, (h + shift_cv) % 180, h)
                
            if s_shift != 0:
                # Saturation is [0, 255]
                s = np.where(mask, np.clip(s + s_shift, 0, 255), s)
                
            if l_shift != 0:
                # Value/Luminance is [0, 255]
                v = np.where(mask, np.clip(v + l_shift, 0, 255), v)
                
        hsv[:, :, 0] = h
        hsv[:, :, 1] = s
        hsv[:, :, 2] = v
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def color_transfer(self, img_bgr, ref_bgr, intensity=1.0, mask=None):
        """Match colour statistics of input to reference image (Reinhard's method).

        Transfers the colour palette of the reference image to the input by
        matching mean and standard deviation in LAB colour space.
        Only the A and B channels are modified — L (luminance) is preserved
        from the source to avoid distorting brightness/detail.

        Args:
            img_bgr: (H, W, 3) uint8 input image.
            ref_bgr: (H, W, 3) uint8 reference image.
            intensity: 0.0–1.0 blend strength.
            mask: Optional (H, W) float mask to restrict transfer region.

        Returns:
            (H, W, 3) uint8 colour-transferred image.
        """
        if ref_bgr is None:
            return img_bgr

        lab_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_l = lab_src[:, :, 0].copy()
        src_a = lab_src[:, :, 1]
        src_b = lab_src[:, :, 2]
        ref_a = lab_ref[:, :, 1]
        ref_b = lab_ref[:, :, 2]

        mean_a_src = src_a.mean()
        std_a_src = src_a.std()
        mean_b_src = src_b.mean()
        std_b_src = src_b.std()
        mean_a_ref = ref_a.mean()
        std_a_ref = ref_a.std()
        mean_b_ref = ref_b.mean()
        std_b_ref = ref_b.std()

        a_result = (src_a - mean_a_src) * (std_a_ref / (std_a_src + 1e-6)) + mean_a_ref
        b_result = (src_b - mean_b_src) * (std_b_ref / (std_b_src + 1e-6)) + mean_b_ref

        lab_result = np.stack([src_l, a_result, b_result], axis=2)
        lab_result = np.clip(lab_result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

        if intensity < 1.0:
            result = cv2.addWeighted(img_bgr, 1.0 - intensity, result, intensity, 0)

        if mask is not None:
            return blend_masked(img_bgr, result, mask)
        return result

    def color_transfer_hist(self, img_bgr, ref_bgr, intensity=1.0, mask=None):
        """Match colour via per-channel histogram matching (more precise).

        Distributes the exact histogram shape of the reference across the
        A and B channels of LAB. The L channel is preserved from the source
        to avoid distorting luminance.

        Args:
            img_bgr: (H, W, 3) uint8 input image.
            ref_bgr: (H, W, 3) uint8 reference image.
            intensity: 0.0–1.0 blend strength.
            mask: Optional (H, W) float mask to restrict transfer region.

        Returns:
            (H, W, 3) uint8 colour-transferred image.
        """
        if ref_bgr is None:
            return img_bgr

        lab_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)

        src_l = lab_src[:, :, 0].copy()

        channels = [src_l]
        for c in (1, 2):
            src_chan = lab_src[:, :, c].ravel()
            ref_chan = lab_ref[:, :, c].ravel()
            src_sorted = np.sort(src_chan)
            ref_sorted = np.sort(ref_chan)
            # Use midpoint averaging to resolve banding on smooth regions (Issue 10)
            left = np.searchsorted(src_sorted, src_chan, side="left")
            right = np.searchsorted(src_sorted, src_chan, side="right")
            indices = (left + right) // 2
            mapped = ref_sorted[np.clip(indices, 0, len(ref_sorted) - 1)]
            channels.append(mapped.reshape(lab_src.shape[:2]).astype(np.uint8))

        lab_result = np.stack(channels, axis=2)
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

        if intensity < 1.0:
            result = cv2.addWeighted(img_bgr, 1.0 - intensity, result, intensity, 0)

        if mask is not None:
            return blend_masked(img_bgr, result, mask)
        return result

    def _add_haze(self, img_bgr, strength, mask=None):
        """Create a dreamy atmospheric fog/haze over the entire frame."""
        if strength <= 0:
            return img_bgr
            
        h, w = img_bgr.shape[:2]
        img_f = img_bgr.astype(np.float32)
        
        # Blur the image heavily to get the ambient colors
        ksize = max(int(min(h, w) * 0.08), 25) | 1
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
        
        # Add a soft pastel blue-purple tone to the haze to make it look magical (silver-white/blue)
        haze_tint = np.array([245.0, 230.0, 240.0], dtype=np.float32) / 255.0 # BGR: light lavender/blue
        haze_layer = blurred * haze_tint
        
        # Screen blend mode for fog
        screen = 255.0 - (255.0 - img_f) * (255.0 - haze_layer) / 255.0
        
        if mask is not None:
            m_f = mask.astype(np.float32)
            if m_f.max() > 1.0:
                m_f = m_f / 255.0
            if m_f.ndim == 2:
                m_f = m_f[:, :, np.newaxis]
            haze_factor = m_f * strength
        else:
            haze_factor = strength

        # Blend haze based on strength and mask
        result = img_f * (1.0 - haze_factor) + screen * haze_factor
        
        # Lift shadows slightly to wash out deep blacks (Dehaze: -20 effect)
        lab = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        l = lab[:, :, 0]
        shadow_lift = 25.0 * strength
        shadow_mask = np.clip(1.0 - l / 128.0, 0, 1)
        if mask is not None:
            m_2d = m_f.squeeze(-1) if m_f.ndim == 3 else m_f
            lab[:, :, 0] = np.clip(l + shadow_mask * (shadow_lift * m_2d), 0, 255)
        else:
            lab[:, :, 0] = np.clip(l + shadow_mask * shadow_lift, 0, 255)
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _add_sparkles(self, img_bgr, opacity):
        """Add procedural glowing star particles on extreme highlight peaks."""
        if opacity <= 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Find extreme highlights (pixels > 230)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Find local maxima to avoid putting sparkles on every single highlight pixel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        local_max = cv2.dilate(gray, kernel)
        peaks = (gray == local_max) & (thresh > 0)
        
        y_indices, x_indices = np.where(peaks)
        if len(x_indices) == 0:
            return img_bgr

        # Limit the number of sparkles to prevent clutter (e.g. max 50 sparkles)
        num_sparkles = min(len(x_indices), 50)
        # Randomly select a subset
        indices = np.random.choice(len(x_indices), num_sparkles, replace=False)
        
        # Create a blank sparkle overlay
        sparkle_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        
        for idx in indices:
            cx, cy = x_indices[idx], y_indices[idx]
            
            # Determine a random size for the star (cross arm length: 6 to 12 pixels)
            size = np.random.randint(6, 13)
            
            color = (255, 255, 255)
            
            # Draw core diamond (rotated square) or circle
            cv2.circle(sparkle_overlay, (cx, cy), 1, color, -1)
            
            # Draw star arms
            cv2.line(sparkle_overlay, (cx - size, cy), (cx + size, cy), (240, 245, 255), 1)
            cv2.line(sparkle_overlay, (cx, cy - size), (cx, cy + size), (240, 245, 255), 1)
            
            # Draw diagonal soft flares
            diag_size = int(size * 0.6)
            cv2.line(sparkle_overlay, (cx - diag_size, cy - diag_size), (cx + diag_size, cy + diag_size), (220, 230, 255), 1)
            cv2.line(sparkle_overlay, (cx - diag_size, cy + diag_size), (cx + diag_size, cy - diag_size), (220, 230, 255), 1)
            
        # Blur the sparkles slightly to make them look like glowing light flares
        sparkle_overlay = cv2.GaussianBlur(sparkle_overlay, (3, 3), 0)
        
        # Screen blend mode
        img_f = img_bgr.astype(np.float32)
        sparkle_f = sparkle_overlay.astype(np.float32)
        
        screened = 255.0 - (255.0 - img_f) * (255.0 - sparkle_f * opacity) / 255.0
        return np.clip(screened, 0, 255).astype(np.uint8)


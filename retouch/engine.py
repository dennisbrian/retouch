"""RetouchEngine — main pipeline orchestrator.

Pipeline order:
    1. Face Detection (MediaPipe + optional RetinaFace)
    2. Person Segmentation (MediaPipe Selfie Segmentation)
    3. Face Region Parsing (landmarks → per-region masks)
    4. 3-Level Frequency Separation
    5. Skin Smoothing (frequency-based)
    6. Blemish Removal (inpainting)
    7. Under-Eye Dark Circle Repair
    8. Eye Enhancement (whites, iris, catchlight)
    9. Lip Enhancement (vibrance, tint)
    10. Teeth Whitening
    11. Skin Tone Equalization (CLAHE)
    12. Skin Whitening (LAB)
    13. Contrast Adjustment
    14. Colour Grading (preset)
"""

import cv2
import numpy as np

from .detection import FaceDetector
from .parsing import FaceParser
from .geometry import FaceReshaper
from .makeup import MakeupEngine
from .frequency import separate as freq_separate, combine as freq_combine
from .skin import SkinProcessor
from .blemish import BlemishRemover
from .eyes import EyeEnhancer
from .undereye import UnderEyeRepairer
from .lips import LipEnhancer
from .teeth import TeethWhitener
from .grading import ColorGrader
from .hair import HairEnhancer


from .recipes import RECIPES
from .utils import correct_exposure


class RetouchEngine:
    """Professional-grade automated face retouching engine.

    Usage::

        engine = RetouchEngine()
        result = engine.process(img_bgr, preset='cosplay')

        # Or with custom parameters:
        result = engine.process(img_bgr, smooth=60, whiten=30, eye_enhance=40)
    """

    def __init__(self, max_faces=10, min_confidence=0.5):
        self._detector = FaceDetector(
            max_faces=max_faces,
            min_confidence=min_confidence,
            refine_landmarks=True,
        )
        self._parser = FaceParser()
        self._reshaper = FaceReshaper()
        self._makeup = MakeupEngine()
        self._skin = SkinProcessor()
        self._blemish = BlemishRemover()
        self._eyes = EyeEnhancer()
        self._undereye = UnderEyeRepairer()
        self._lips = LipEnhancer()
        self._teeth = TeethWhitener()
        self._grader = ColorGrader()
        self._hair = HairEnhancer()

    def process(
        self,
        img_bgr,
        recipe=None,
        preset=None,
        # Individual controls (None defaults to recipe values)
        smooth=None,
        whiten=None,
        eye_enhance=None,
        dark_circles=None,
        blemish=None,
        lip_enhance=None,
        lip_tint=None,
        teeth_whiten=None,
        equalize=None,
        contrast=None,
        brightness=None,
        highlights=None,
        shadows=None,
        whites=None,
        blacks=None,
        color_grade=None,
        grade_intensity=None,
        texture_opacity=None,
        mid_reduction=None,
        nose_smooth=None,
        hair_enhance=None,
        dodge_burn=None,
        slimming=None,
        blush=None,
        lip_finish=None,
        specular_bloom=None,
        specular_bloom_tone=None,
        whiten_tone=None,
        auto_exposure=False,
        impact=None,
        chromatic_aberration=None,
        halation=None,
        grain=None,
        lut=None,
        color_grade_stack=None,
        color_ref=None,
        color_transfer_intensity=1.0,
        fast=False,
    ):
        """Process a single image through the Retouch Recipe System.

        Args:
            img_bgr: (H, W, 3) uint8 BGR input image.
            recipe: Recipe name (cosplay, xiaohongshu, natural, portrait, etc.)
            preset: Legacy alias for recipe.
            ...
        """
        # Fast preview: downscale then upscale after processing
        orig_h, orig_w = img_bgr.shape[:2]
        if fast:
            max_preview = 800
            scale = min(max_preview / orig_w, max_preview / orig_h, 1.0)
            if scale < 1.0:
                img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)

        # Resolve active recipe
        active_recipe = recipe or preset or "natural"
        if active_recipe not in RECIPES:
            active_recipe = "natural"
        rec = RECIPES[active_recipe]

        # Extract defaults from recipe (convert float ratios to percents)
        r_smooth = int(rec["frequency"]["smooth"] * 100)
        r_equalize = int(rec["skin"].get("equalize", 0) * 100)
        
        # Rosy or Porcelain foundation
        r_rosy = rec["skin"].get("rosy", rec["skin"].get("porcelain", 0))
        r_whiten = int(r_rosy * 100)
        
        r_eye = int(rec["eyes"].get("iris", rec["eyes"].get("whites", 0)) * 100)
        r_dark_circles = int(rec["eyes"].get("whites", 0) * 100)
        r_blemish = r_smooth
        
        r_lip = int(rec["lips"].get("gloss", 0) * 100)
        r_lip_tint = rec["lips"].get("tint", None)
        r_teeth_whiten = int(rec["eyes"].get("whites", 0) * 100)

        r_hair = int(rec["hair"].get("shine", 0) * 100)
        r_dodge_burn = int(rec["dodge_burn"].get("amount", 0) * 100)
        
        r_color_grade = rec["color_harmony"].get("preset", None)
        r_grade_intensity = rec["color_harmony"].get("amount", 0.0)
        resolved_grade_intensity = (
            grade_intensity if grade_intensity is not None
            else (1.0 if color_grade is not None else r_grade_intensity)
        )
        r_texture = rec["texture"].get("opacity", 1.0)
        r_mid = rec["frequency"].get("mid_reduction", 0.35 if r_smooth < 50 else 0.45)
        r_impact = int(rec.get("finish", {}).get("impact", 0.0) * 100)
        r_contrast = rec.get("contrast", 0)
        r_specular_bloom = rec.get("specular_bloom", 0)
        if "porcelain" in rec["skin"]:
            r_whiten_tone = "porcelain"
        else:
            r_whiten_tone = "rosy"
        r_specular_bloom_tone = rec.get("specular_bloom_tone", "rosy")

        # Reshaping and makeup defaults based on style
        if active_recipe == "fantasy_goddess":
            r_slimming = 35
            r_blush = 30
        elif active_recipe in ("scifi_cosplay", "cyber_doll", "pink_dream", "meitu_clone"):
            r_slimming = 30 if active_recipe != "scifi_cosplay" else 0
            r_blush = 35 if active_recipe in ("scifi_cosplay", "cyber_doll") else 30
        else:
            r_slimming = 30 if active_recipe in ("cosplay", "cosplay_3d", "cosplay_no_eq", "anime_cosplay", "anime", "xiaohongshu", "idol", "blue_dream", "xhs_ultrasoft") else 0
            r_blush = 25 if active_recipe in ("cosplay", "cosplay_3d", "cosplay_no_eq", "anime_cosplay", "anime", "xiaohongshu", "idol", "wedding", "blue_dream", "xhs_ultrasoft") else 0
        r_lip_finish = "matte" if active_recipe in ("wedding", "magazine") else ("velvet" if active_recipe in ("korean_beauty", "xhs_ultrasoft") else "gloss")

        # Merge overrides
        p = {
            "smooth": smooth if smooth is not None else r_smooth,
            "whiten": whiten if whiten is not None else r_whiten,
            "eye_enhance": eye_enhance if eye_enhance is not None else r_eye,
            "dark_circles": dark_circles if dark_circles is not None else r_dark_circles,
            "blemish": blemish if blemish is not None else r_blemish,
            "lip_enhance": lip_enhance if lip_enhance is not None else r_lip,
            "lip_tint": lip_tint if lip_tint is not None else r_lip_tint,
            "teeth_whiten": teeth_whiten if teeth_whiten is not None else r_teeth_whiten,
            "equalize": equalize if equalize is not None else r_equalize,
            "contrast": contrast if contrast is not None else r_contrast,
            "brightness": brightness,
            "highlights": highlights,
            "shadows": shadows,
            "whites": whites,
            "blacks": blacks,
            "color_ref": None,
            "color_transfer_intensity": 1.0,
            "color_grade": color_grade if color_grade is not None else r_color_grade,
            "grade_intensity": resolved_grade_intensity,
            "texture_opacity": texture_opacity if texture_opacity is not None else r_texture,
            "mid_reduction": mid_reduction if mid_reduction is not None else r_mid,
            "nose_smooth": nose_smooth,
            "hair_enhance": hair_enhance if hair_enhance is not None else r_hair,
            "dodge_burn": dodge_burn if dodge_burn is not None else r_dodge_burn,
            "slimming": slimming if slimming is not None else r_slimming,
            "blush": blush if blush is not None else r_blush,
            "lip_finish": lip_finish if lip_finish is not None else r_lip_finish,
            "specular_bloom": specular_bloom if specular_bloom is not None else r_specular_bloom,
            "specular_bloom_tone": specular_bloom_tone if specular_bloom_tone is not None else r_specular_bloom_tone,
            "whiten_tone": whiten_tone if whiten_tone is not None else r_whiten_tone,
            "auto_exposure": auto_exposure,
            "impact": impact if impact is not None else r_impact,
            "chromatic_aberration": chromatic_aberration,
            "halation": halation,
            "grain": grain,
            "lut": lut,
            "color_grade_stack": color_grade_stack,
            "color_ref": color_ref,
            "color_transfer_intensity": color_transfer_intensity,
        }

        # Apply Auto-Exposure Correction if enabled
        if p["auto_exposure"]:
            faces = self._detector.detect(img_bgr)
            if faces:
                bboxes = [f.bbox for f in faces]
                img_bgr, corrected = correct_exposure(img_bgr, face_bboxes=bboxes)
                if corrected:
                    faces = self._detector.detect(img_bgr)
            else:
                img_bgr, corrected = correct_exposure(img_bgr, face_bboxes=None)
                if corrected:
                    faces = self._detector.detect(img_bgr)
        else:
            faces = self._detector.detect(img_bgr)

        person_mask = self._detector.segment_person(img_bgr)

        if not faces:
            # No face detected — apply minimal global processing
            result = img_bgr.copy()
            if p["contrast"]:
                result = self._adjust_contrast(result, p["contrast"])
            if p["color_grade"]:
                result = self._grader.grade(result, p["color_grade"], p["grade_intensity"])
            if p["impact"] > 0:
                result = self._grader.add_impact_finish(result, p["impact"])
            return result

        # Apply photographer-grade face reshaping/slimming first
        if p["slimming"] > 0:
            img_bgr = self._reshaper.reshape(img_bgr, faces, p["slimming"])

        # ---- Process each face ----
        result = img_bgr.copy()

        # Accumulate sharpening mask for eyes, brows, and hair edges
        h_img, w_img = img_bgr.shape[:2]
        accumulated_sharpen_mask = np.zeros((h_img, w_img), dtype=np.float32)
        accumulated_skin_hair_mask = np.zeros((h_img, w_img), dtype=np.float32)
        accumulated_skin_mask = np.zeros((h_img, w_img), dtype=np.float32)
        accumulated_lips_mask = np.zeros((h_img, w_img), dtype=np.float32)

        for face in faces:
            face_width = face.ied * 2.5

            # ---- 2. Parse face regions using BiSeNet + Face Mesh ----
            regions = self._parser.parse(
                face.landmarks, img_bgr, face.bbox, person_mask, face.ied
            )

            # Accumulate skin + hair mask for split toning restriction
            face_skin_hair = np.zeros((h_img, w_img), dtype=np.float32)
            if regions.skin is not None:
                s_mask = regions.skin.astype(np.float32)
                if s_mask.max() > 1.0:
                    s_mask = s_mask / 255.0
                face_skin_hair = np.clip(face_skin_hair + s_mask, 0.0, 1.0)
                accumulated_skin_mask = np.clip(accumulated_skin_mask + s_mask, 0.0, 1.0)
            if regions.hair is not None:
                h_mask_raw = regions.hair.astype(np.float32)
                if h_mask_raw.max() > 1.0:
                    h_mask_raw = h_mask_raw / 255.0
                face_skin_hair = np.clip(face_skin_hair + h_mask_raw, 0.0, 1.0)
            accumulated_skin_hair_mask = np.clip(accumulated_skin_hair_mask + face_skin_hair, 0.0, 1.0)

            # Accumulate lips mask to protect them from downstream white costume highlight recovery
            if regions.lips is not None:
                l_mask = regions.lips.astype(np.float32)
                if l_mask.max() > 1.0:
                    l_mask = l_mask / 255.0
                accumulated_lips_mask = np.clip(accumulated_lips_mask + l_mask, 0.0, 1.0)

            # ---- 3. Frequency separation ----
            original_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            layers = freq_separate(result, face_width)

            # ---- 4. Frequency-based smoothing ----
            smooth_mask = regions.skin.astype(np.float32)
            for exclude_region in [regions.left_eye, regions.right_eye,
                                   regions.left_under_eye, regions.right_under_eye,
                                   regions.left_eyebrow, regions.right_eyebrow,
                                   regions.lips]:
                if exclude_region is not None:
                    smooth_mask = np.clip(smooth_mask - exclude_region.astype(np.float32), 0, 1)
            if p["nose_smooth"] is not None:
                nose_mask = regions.nose.astype(np.float32) * smooth_mask
                face_without_nose = np.clip(smooth_mask - nose_mask, 0, 1)
                result = freq_combine(
                    layers,
                    skin_mask=face_without_nose,
                    smooth_strength=p["smooth"] / 100.0,
                    mid_reduction=p["mid_reduction"],
                    texture_opacity=p["texture_opacity"],
                    face_width=face_width,
                )
                layers = freq_separate(result, face_width)
                result = freq_combine(
                    layers,
                    skin_mask=nose_mask,
                    smooth_strength=p["nose_smooth"] / 100.0,
                    mid_reduction=p["mid_reduction"],
                    texture_opacity=p["texture_opacity"],
                    face_width=face_width,
                )
            else:
                result = freq_combine(
                    layers,
                    skin_mask=smooth_mask,
                    smooth_strength=p["smooth"] / 100.0,
                    mid_reduction=p["mid_reduction"],
                    texture_opacity=p["texture_opacity"],
                    face_width=face_width,
                )

            # ---- 5. Skin Equalization ----
            if p["equalize"] > 0:
                result = self._skin.equalize(result, regions.skin, p["equalize"], ref_lab=original_lab)

            # ---- 6. Adaptive Rosy Foundation ----
            if p["whiten"] > 0:
                result = self._skin.whiten(result, regions.skin, p["whiten"], tone=p["whiten_tone"])

            # ---- 6b. Specular Pink Bloom ----
            if p["specular_bloom"] > 0:
                result = self._skin.apply_specular_bloom(result, regions.skin, p["specular_bloom"], tone=p["specular_bloom_tone"])

            # ---- 7. Blemish Removal ----
            if p["blemish"] > 0:
                result = self._blemish.remove(result, regions.skin, p["blemish"])

            # Under-eye Repair
            if p["dark_circles"] > 0:
                result = self._undereye.repair(result, regions, p["dark_circles"])

            # ---- 8. Neck Matching ----
            if p["whiten"] > 0 or p["equalize"] > 0:
                result = self._skin.harmonize_neck(
                    result, face.landmarks, person_mask, regions.skin, regions.neck, strength=max(p["whiten"], p["equalize"])
                )

            # ---- 9. Eye Enhancement ----
            if p["eye_enhance"] > 0:
                result = self._eyes.enhance(result, regions, p["eye_enhance"])

            # Teeth Whitening
            if p["teeth_whiten"] > 0:
                result = self._teeth.whiten(
                    result, regions.mouth_interior, p["teeth_whiten"]
                )

            # ---- 10. Lip Gloss & Tint ----
            if p["lip_enhance"] > 0:
                result = self._lips.enhance(
                    result, regions.lips, p["lip_enhance"], tint=p["lip_tint"], finish=p["lip_finish"]
                )

            # ---- 10b. Blush Wash ----
            if p["blush"] > 0:
                result = self._makeup.apply_blush(
                    result, face.landmarks, face_width, p["blush"],
                    regions=regions,
                    nose_blush=(active_recipe in ("scifi_cosplay", "cyber_doll", "cosplay", "anime", "fantasy_goddess", "pink_dream", "meitu_clone")),
                    under_eye_blush=(active_recipe in ("scifi_cosplay", "cyber_doll", "cosplay", "anime", "fantasy_goddess", "pink_dream", "meitu_clone"))
                )

            # ---- 11. Hair Shine ----
            if p["hair_enhance"] > 0:
                result = self._hair.enhance(
                    result, person_mask, regions.face_oval, face.bbox, p["hair_enhance"], regions.hair
                )

            # ---- 12. Micro Dodge & Burn ----
            if p["dodge_burn"] > 0:
                result = self._skin.dodge_burn(result, regions, p["dodge_burn"])

            # Accumulate face selective sharpening mask (eyes scaled by 1.5, other regions scaled by 0.8)
            face_eye_sharpen = np.zeros((h_img, w_img), dtype=np.float32)
            if regions.left_eye is not None:
                face_eye_sharpen = np.clip(face_eye_sharpen + regions.left_eye, 0.0, 1.0)
            if regions.right_eye is not None:
                face_eye_sharpen = np.clip(face_eye_sharpen + regions.right_eye, 0.0, 1.0)

            face_other_sharpen = np.zeros((h_img, w_img), dtype=np.float32)
            if regions.left_eyebrow is not None:
                face_other_sharpen = np.clip(face_other_sharpen + regions.left_eyebrow, 0.0, 1.0)
            if regions.right_eyebrow is not None:
                face_other_sharpen = np.clip(face_other_sharpen + regions.right_eyebrow, 0.0, 1.0)

            # Hair boundary edge detection via erosion subtraction
            if regions.hair is not None and regions.hair.max() > 0.01:
                eroded_hair = cv2.erode(regions.hair, np.ones((5, 5), np.uint8))
                hair_edges = np.clip(regions.hair - eroded_hair, 0.0, 1.0)
                face_other_sharpen = np.clip(face_other_sharpen + hair_edges, 0.0, 1.0)

            # Clamp mask to [0.0, 1.0] and boost amount in application instead (Issue 2)
            face_sharpen_weighted = np.clip(
                np.maximum(face_eye_sharpen * 1.0, face_other_sharpen * 0.53), 0.0, 1.0
            )
            accumulated_sharpen_mask = np.maximum(accumulated_sharpen_mask, face_sharpen_weighted)

        # ---- 13. Global contrast ----
        if p["contrast"]:
            result = self._adjust_contrast(result, p["contrast"])

        # ---- 13b. Global brightness ----
        if p.get("brightness") is not None:
            b = p["brightness"]
            if b != 0:
                gamma = 1.0 - (b / 100.0)
                lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
                result = cv2.LUT(result, lut)

        # ---- 13c. Tonal adjustment (highlights/shadows/whites/blacks) ----
        if any(p.get(k) for k in ("highlights", "shadows", "whites", "blacks")):
            result = self._adjust_tonal(
                result,
                shadows=p.get("shadows", 0),
                highlights=p.get("highlights", 0),
                whites=p.get("whites", 0),
                blacks=p.get("blacks", 0),
            )

        # ---- 14. Colour transfer (reference-based) ----
        if p.get("color_ref") is not None:
            result = self._grader.color_transfer(
                result, p["color_ref"], intensity=p.get("color_transfer_intensity", 1.0)
            )

        # ---- 15. Colour grading ----

        # Compute glow mask: allow glow on skin and background, but preserve hair, eyes, and costume details (Issue 1)
        pm_norm = person_mask.astype(np.float32)
        if pm_norm.max() > 1.0:
            pm_norm /= 255.0
        if pm_norm.ndim == 3:
            pm_norm = pm_norm.squeeze(-1)
        # Foreground sharp region = person_mask - accumulated_skin_mask
        sharp_fg = np.clip(pm_norm - accumulated_skin_mask, 0.0, 1.0)
        g_mask = 1.0 - sharp_fg

        if p["color_grade_stack"]:
            result = self._grader.grade_stack(result, p["color_grade_stack"])
        elif p["color_grade"]:
            from .grading import PRESETS
            settings = PRESETS.get(p["color_grade"], PRESETS["natural"]).copy()
            if p["chromatic_aberration"] is not None:
                settings["chromatic_aberration"] = p["chromatic_aberration"]
            if p["halation"] is not None:
                settings["halation"] = p["halation"]
            if p["grain"] is not None:
                settings["grain"] = p["grain"]
            if p["lut"] is not None:
                settings["lut"] = p["lut"]
            result = self._grader.grade(result, settings, p["grade_intensity"], split_tone_mask=s_mask, glow_mask=g_mask, haze_mask=g_mask)
        else:
            settings = {}
            if p["chromatic_aberration"] is not None:
                settings["chromatic_aberration"] = p["chromatic_aberration"]
            if p["halation"] is not None:
                settings["halation"] = p["halation"]
            if p["grain"] is not None:
                settings["grain"] = p["grain"]
            if p["lut"] is not None:
                settings["lut"] = p["lut"]
            if settings:
                result = self._grader.grade(result, settings, 1.0, split_tone_mask=s_mask, glow_mask=g_mask, haze_mask=g_mask)

        # Highlight/White Costume Pearl/Lavender lift (Issue 4)
        if p["color_grade"] in ("pink_dream", "meitu_clone"):
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            l_val = lab[:, :, 0]
            a_val = lab[:, :, 1]
            b_val = lab[:, :, 2]
            
            # White colors: high luminance (L > 170) and low saturation
            # Excluding skin, hair, and lips to preserve face, neck, wig, and lip details
            non_skin_hair_lips = 1.0 - np.clip(accumulated_skin_hair_mask + accumulated_lips_mask, 0.0, 1.0)
            white_cond = (l_val > 170) & (np.abs(a_val - 128.0) < 15) & (np.abs(b_val - 128.0) < 15)
            white_mask = white_cond.astype(np.float32) * non_skin_hair_lips
            
            if white_mask.max() > 0.01:
                # Soft feathering to avoid halos
                white_mask_blurred = cv2.GaussianBlur(white_mask, (15, 15), 0)
                s_val = p["grade_intensity"]
                
                l_new = np.clip(l_val + 10.0 * s_val * white_mask_blurred, 0, 255)
                a_new = np.clip(a_val + 1.0 * s_val * white_mask_blurred, 0, 255)
                b_new = np.clip(b_val + 3.0 * s_val * white_mask_blurred, 0, 255)
                
                lab[:, :, 0] = l_new
                lab[:, :, 1] = a_new
                lab[:, :, 2] = b_new
                result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # ---- 16. Selective final sharpening ----
        if accumulated_sharpen_mask.max() > 0.01:
            result = self._apply_selective_sharpening(
                result, accumulated_sharpen_mask, radius=0.8, amount=1.2, threshold=2
            )

        # ---- 17. Global high-impact finish ----
        if p["impact"] > 0:
            result = self._grader.add_impact_finish(result, p["impact"])

        # Upscale back if fast preview
        if fast and scale < 1.0:
            result = cv2.resize(result, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)

        return result

    @staticmethod
    def _apply_selective_sharpening(img_bgr, mask, radius=0.8, amount=0.8, threshold=2):
        """Apply selective Photoshop-style unsharp masking over a soft mask."""
        img_f = img_bgr.astype(np.float32)
        blurred = cv2.GaussianBlur(img_f, (0, 0), radius)
        high_freq = img_f - blurred

        mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask

        if threshold > 0:
            gray_high = cv2.cvtColor(np.abs(high_freq).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            threshold_mask = (gray_high >= threshold)[:, :, np.newaxis]
            sharpened_diff = threshold_mask * (high_freq * (amount * mask_3d))
        else:
            sharpened_diff = high_freq * (amount * mask_3d)

        sharpened_img = np.clip(img_f + sharpened_diff, 0, 255).astype(np.uint8)
        return sharpened_img

    def close(self):
        """Release detector resources."""
        self._detector.close()

    @staticmethod
    def _adjust_contrast(img, contrast):
        """Simple contrast adjustment."""
        if contrast == 0:
            return img
        f = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
        result = f * (img.astype(np.float32) - 128.0) + 128.0
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_tonal(img, shadows=0, highlights=0, whites=0, blacks=0):
        """Parametric tonal adjustment via LUT curve.
        All values are -100 to +100.
        """
        if not any([shadows, highlights, whites, blacks]):
            return img
        x = np.arange(256, dtype=np.float32)
        y = x.copy()
        # Blacks — lower 25%, peak at 0
        if blacks:
            w = np.clip(1.0 - x / 64.0, 0, 1)
            y = y + (blacks / 100.0) * 48 * w
        # Shadows — lower 50%, peak around 25%
        if shadows:
            w = np.clip(1.0 - np.abs(x - 25.0) / 100.0, 0, 1)
            w = w * w * (3 - 2 * w)
            y = y + (shadows / 100.0) * 48 * w
        # Highlights — upper 50%, peak around 75%
        if highlights:
            w = np.clip(1.0 - np.abs(x - 230.0) / 100.0, 0, 1)
            w = w * w * (3 - 2 * w)
            y = y + (highlights / 100.0) * 48 * w
        # Whites — upper 25%, peak at 255
        if whites:
            w = np.clip(1.0 - (255.0 - x) / 64.0, 0, 1)
            y = y + (whites / 100.0) * 48 * w
        y = np.clip(y, 0, 255).astype(np.uint8)
        return cv2.LUT(img, y)


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible API)
# ---------------------------------------------------------------------------

def retouch(
    img_bgr,
    smooth=50,
    whiten=30,
    eye_enhance=30,
    contrast=0,
    preset=None,
    **kwargs,
):
    """One-shot retouching function (backward-compatible with v1 API).

    See RetouchEngine.process() for full parameter list.
    """
    engine = RetouchEngine()
    try:
        return engine.process(
            img_bgr,
            smooth=smooth,
            whiten=whiten,
            eye_enhance=eye_enhance,
            contrast=contrast,
            preset=preset,
            **kwargs,
        )
    finally:
        engine.close()

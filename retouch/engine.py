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
        color_grade=None,
        grade_intensity=None,
        texture_opacity=None,
        mid_reduction=None,
        hair_enhance=None,
        dodge_burn=None,
        slimming=None,
        blush=None,
        lip_finish=None,
        auto_exposure=False,
    ):
        """Process a single image through the Retouch Recipe System.

        Args:
            img_bgr: (H, W, 3) uint8 BGR input image.
            recipe: Recipe name (cosplay, xiaohongshu, natural, portrait, etc.)
            preset: Legacy alias for recipe.
            ...
        """
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
        r_texture = rec["texture"].get("opacity", 1.0)
        r_mid = 0.35 if r_smooth < 50 else 0.45

        # Reshaping and makeup defaults based on style
        r_slimming = 30 if active_recipe in ("cosplay", "cosplay_3d", "cosplay_no_eq", "anime_cosplay", "anime", "xiaohongshu", "idol") else 0
        r_blush = 25 if active_recipe in ("cosplay", "cosplay_3d", "cosplay_no_eq", "anime_cosplay", "anime", "xiaohongshu", "idol", "wedding") else 0
        r_lip_finish = "matte" if active_recipe in ("wedding", "magazine") else ("velvet" if active_recipe == "korean_beauty" else "gloss")

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
            "contrast": contrast if contrast is not None else 0,
            "color_grade": color_grade if color_grade is not None else r_color_grade,
            "grade_intensity": grade_intensity if grade_intensity is not None else r_grade_intensity,
            "texture_opacity": texture_opacity if texture_opacity is not None else r_texture,
            "mid_reduction": mid_reduction if mid_reduction is not None else r_mid,
            "hair_enhance": hair_enhance if hair_enhance is not None else r_hair,
            "dodge_burn": dodge_burn if dodge_burn is not None else r_dodge_burn,
            "slimming": slimming if slimming is not None else r_slimming,
            "blush": blush if blush is not None else r_blush,
            "lip_finish": lip_finish if lip_finish is not None else r_lip_finish,
            "auto_exposure": auto_exposure,
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
            return result

        # Apply photographer-grade face reshaping/slimming first
        if p["slimming"] > 0:
            img_bgr = self._reshaper.reshape(img_bgr, faces, p["slimming"])

        # ---- Process each face ----
        result = img_bgr.copy()

        for face in faces:
            face_width = face.ied * 2.5

            # ---- 2. Parse face regions using BiSeNet + Face Mesh ----
            regions = self._parser.parse(
                face.landmarks, img_bgr, face.bbox, person_mask, face.ied
            )

            # ---- 3. Frequency separation ----
            layers = freq_separate(result, face_width)

            # ---- 4. Frequency-based smoothing ----
            result = freq_combine(
                layers,
                skin_mask=regions.skin,
                smooth_strength=p["smooth"] / 100.0,
                mid_reduction=p["mid_reduction"],
                texture_opacity=p["texture_opacity"],
                face_width=face_width,
            )

            # ---- 5. Skin Equalization ----
            if p["equalize"] > 0:
                result = self._skin.equalize(result, regions.skin, p["equalize"])

            # ---- 6. Adaptive Rosy Foundation ----
            if p["whiten"] > 0:
                result = self._skin.whiten(result, regions.skin, p["whiten"])

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
                    result, face.landmarks, face_width, p["blush"]
                )

            # ---- 11. Hair Shine ----
            if p["hair_enhance"] > 0:
                result = self._hair.enhance(
                    result, person_mask, regions.face_oval, face.bbox, p["hair_enhance"], regions.hair
                )

            # ---- 12. Micro Dodge & Burn ----
            if p["dodge_burn"] > 0:
                result = self._skin.dodge_burn(result, regions, p["dodge_burn"])

        # ---- 13. Global contrast ----
        if p["contrast"]:
            result = self._adjust_contrast(result, p["contrast"])

        # ---- 14. Colour grading ----
        if p["color_grade"]:
            result = self._grader.grade(result, p["color_grade"], p["grade_intensity"])

        return result

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

"""StyleAnalyzer & StyleApplier — automatic style cloning and subject-aware color matching."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .frequency import separate as freq_separate


@dataclass
class StyleProfile:
    """Extracted editing parameters describing a style delta (Original vs. Edited)."""

    # Global adjustments
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    saturation_delta: float = 0.0

    # Face/Skin adjustments
    skin_l_mean_delta: float = 0.0
    skin_a_mean_delta: float = 0.0
    skin_b_mean_delta: float = 0.0
    skin_smooth_strength: float = 0.0
    skin_mid_reduction: float = 0.0
    skin_texture_opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StyleProfile:
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> StyleProfile:
        return cls.from_dict(json.loads(s))

    def save(self, filepath: str) -> None:
        """Save the style profile to a JSON file."""
        import os
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> StyleProfile:
        """Load a style profile from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


class StyleAnalyzer:
    """Analyze original vs. edited image differences to extract a StyleProfile."""

    def __init__(self, engine=None):
        from .engine import RetouchEngine
        self.engine = engine or RetouchEngine()

    def extract(self, original_img: np.ndarray, edited_img: np.ndarray) -> StyleProfile:
        """Measure difference between original and edited image to build StyleProfile.

        Note:
            - Both input images must be aligned (identical crop and dimensions).
            - If no face is detected, skin statistics and texture metrics default to 0.0 (or 1.0 for opacity).
        """
        # Ensure identical sizes
        if original_img.shape[:2] != edited_img.shape[:2]:
            edited_img = cv2.resize(edited_img, (original_img.shape[1], original_img.shape[0]))

        orig_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        edit_lab = cv2.cvtColor(edited_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        orig_l, orig_a, orig_b = cv2.split(orig_lab)
        edit_l, edit_a, edit_b = cv2.split(edit_lab)

        # Global Brightness Delta (difference in LAB L channel mean)
        brightness_delta = float(np.mean(edit_l) - np.mean(orig_l))

        # Global Contrast (difference of 95th-5th percentiles mapped to slider range [-50, 50] to resist exposure bias)
        orig_p95 = np.percentile(orig_l, 95)
        orig_p5 = np.percentile(orig_l, 5)
        edit_p95 = np.percentile(edit_l, 95)
        edit_p5 = np.percentile(edit_l, 5)
        
        orig_contrast_range = orig_p95 - orig_p5
        edit_contrast_range = edit_p95 - edit_p5
        
        contrast_ratio = float(edit_contrast_range / (orig_contrast_range + 1e-5))
        contrast_delta = float(np.clip((contrast_ratio - 1.0) * 50.0, -50.0, 50.0))

        # Global Saturation Delta (mean HSV S channel difference)
        orig_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        edit_hsv = cv2.cvtColor(edited_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_delta = float(np.mean(edit_hsv[:, :, 1]) - np.mean(orig_hsv[:, :, 1]))

        # Local skin statistics
        faces = self.engine._detector.detect(original_img)

        skin_l_mean_delta = 0.0
        skin_a_mean_delta = 0.0
        skin_b_mean_delta = 0.0
        skin_smooth_strength = 0.0
        skin_mid_reduction = 0.0
        skin_texture_opacity = 1.0

        if faces:
            # Sort faces by bounding box area descending so the largest face is processed first
            faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
            h_img, w_img = original_img.shape[:2]
            combined_skin_mask = np.zeros((h_img, w_img), dtype=np.float32)
            person_mask = self.engine._detector.segment_person(original_img)

            total_area = 0.0
            weighted_face_width = 0.0

            for face in faces:
                area = face.bbox[2] * face.bbox[3]
                total_area += area
                weighted_face_width += (face.ied * 2.5) * area

                regions = self.engine._parser.parse(
                    face.landmarks, original_img, face.bbox, person_mask, face.ied
                )
                if regions.skin is not None:
                    s_mask = regions.skin.copy().astype(np.float32)
                    if s_mask.max() > 1.0:
                        s_mask /= 255.0
                    combined_skin_mask = np.maximum(combined_skin_mask, s_mask)

            if total_area > 0:
                face_width = weighted_face_width / total_area
            else:
                face_width = faces[0].ied * 2.5

            if combined_skin_mask.max() > 0.01:
                skin_indices = combined_skin_mask > 0.3

                if np.sum(skin_indices) > 10:
                    orig_skin_l = orig_l[skin_indices]
                    orig_skin_a = orig_a[skin_indices]
                    orig_skin_b = orig_b[skin_indices]

                    edit_skin_l = edit_l[skin_indices]
                    edit_skin_a = edit_a[skin_indices]
                    edit_skin_b = edit_b[skin_indices]

                    skin_l_mean_delta = float(np.mean(edit_skin_l) - np.mean(orig_skin_l))
                    skin_a_mean_delta = float(np.mean(edit_skin_a) - np.mean(orig_skin_a))
                    skin_b_mean_delta = float(np.mean(edit_skin_b) - np.mean(orig_skin_b))

                    # Analyze textures via frequency separation
                    face_width = face.ied * 2.5
                    orig_layers = freq_separate(original_img, face_width)
                    edit_layers = freq_separate(edited_img, face_width)

                    # Convert multi-channel layers to 1D luminance arrays to avoid channel shape bugs
                    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights
                    orig_mid_gray = np.dot(orig_layers.mid, weights)
                    edit_mid_gray = np.dot(edit_layers.mid, weights)
                    orig_high_gray = np.dot(orig_layers.high, weights)
                    edit_high_gray = np.dot(edit_layers.high, weights)

                    # Extract mid frequency energy inside skin mask
                    orig_mid_skin = orig_mid_gray[skin_indices]
                    edit_mid_skin = edit_mid_gray[skin_indices]
                    orig_mid_energy = np.mean(np.abs(orig_mid_skin))
                    edit_mid_energy = np.mean(np.abs(edit_mid_skin))

                    mid_ratio = float(edit_mid_energy / (orig_mid_energy + 1e-5))
                    skin_mid_reduction = float(np.clip(1.0 - mid_ratio, 0.0, 1.0))

                    # Extract high frequency energy inside skin mask
                    orig_high_skin = orig_high_gray[skin_indices]
                    edit_high_skin = edit_high_gray[skin_indices]
                    orig_high_energy = np.mean(np.abs(orig_high_skin))
                    edit_high_energy = np.mean(np.abs(edit_high_skin))

                    high_ratio = float(edit_high_energy / (orig_high_energy + 1e-5))
                    skin_texture_opacity = float(np.clip(high_ratio, 0.0, 1.0))

                    # Estimate overall skin smooth strength based on mid-frequency variance suppression
                    orig_mid_std = np.std(orig_mid_skin)
                    edit_mid_std = np.std(edit_mid_skin)
                    if orig_mid_std > 0:
                        skin_smooth_strength = float(np.clip(1.0 - edit_mid_std / orig_mid_std, 0.0, 1.0))

        return StyleProfile(
            brightness_delta=brightness_delta,
            contrast_delta=contrast_delta,
            saturation_delta=saturation_delta,
            skin_l_mean_delta=skin_l_mean_delta,
            skin_a_mean_delta=skin_a_mean_delta,
            skin_b_mean_delta=skin_b_mean_delta,
            skin_smooth_strength=skin_smooth_strength,
            skin_mid_reduction=skin_mid_reduction,
            skin_texture_opacity=skin_texture_opacity,
        )


class StyleApplier:
    """Apply style profile attributes or perform subject-aware color transfer."""

    def __init__(self, engine=None):
        from .engine import RetouchEngine
        self.engine = engine or RetouchEngine()

    def apply(self, target_img: np.ndarray, profile: StyleProfile) -> np.ndarray:
        """Process target_img using parameter values resolved from the StyleProfile."""
        # 1. Map global parameters (brightness delta L maps to slider value ≈ scale by 2.0)
        brightness_val = np.clip(profile.brightness_delta * 2.0, -100.0, 100.0)

        # 2. Map skin parameters
        smooth_val = np.clip(profile.skin_smooth_strength * 100.0, 0.0, 100.0)
        whiten_val = np.clip(profile.skin_l_mean_delta * 4.0, -100.0, 100.0)
        mid_red_val = np.clip(profile.skin_mid_reduction, 0.0, 1.0)
        tex_op_val = np.clip(profile.skin_texture_opacity, 0.0, 1.0)
        
        if profile.skin_a_mean_delta > 1.0:
            whiten_tone = "rosy"
        elif profile.skin_b_mean_delta < -1.0:
            whiten_tone = "porcelain"
        else:
            whiten_tone = "neutral"

        return self.engine.process(
            target_img,
            smooth=smooth_val,
            mid_reduction=mid_red_val,
            texture_opacity=tex_op_val,
            whiten=whiten_val,
            whiten_tone=whiten_tone,
            contrast=profile.contrast_delta,
            brightness=brightness_val,
        )


# ---------------------------------------------------------------------------
# Subject-Aware Color Transfer Helpers
# ---------------------------------------------------------------------------

def weighted_mean_std(data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the weighted mean and standard deviation of a LAB/BGR image under a soft mask."""
    h, w = data.shape[:2]
    # Downsample if image is > 1 MP to save massive memory and CPU time
    if h * w > 1024 * 1024:
        scale = 1024.0 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        data = cv2.resize(data, (nw, nh), interpolation=cv2.INTER_AREA)
        weights = cv2.resize(weights, (nw, nh), interpolation=cv2.INTER_AREA)

    w = weights[:, :, np.newaxis] if weights.ndim == 2 else weights
    sum_w = np.sum(w)
    if sum_w < 1e-3:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    mean = np.sum(data * w, axis=(0, 1)) / sum_w
    variance = np.sum(((data - mean) ** 2) * w, axis=(0, 1)) / sum_w
    std = np.sqrt(variance) + 1e-5
    return mean, std


def reinhard_transfer_masked(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """Perform Reinhard color transfer from ref_img to src_img, restricted to the masks."""
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute soft mask weighted statistics to avoid hard thresholds
    mean_src, std_src = weighted_mean_std(src_lab, src_mask)
    mean_ref, std_ref = weighted_mean_std(ref_lab, ref_mask)

    if np.sum(src_mask) < 0.1 or np.sum(ref_mask) < 0.1:
        return src_img.copy()

    trans_lab = src_lab.copy()

    # Match channel-wise: (val - mean_src) * (std_ref / std_src) + mean_ref
    # Then blend back using the soft mask
    for c in range(3):
        val = src_lab[:, :, c]
        # Guard against ratio explosion if src std is extremely low
        ratio = np.clip(std_ref[c] / std_src[c], 0.3, 3.0)
        trans_val = (val - mean_src[c]) * ratio + mean_ref[c]
        trans_lab[:, :, c] = val * (1.0 - src_mask) + trans_val * src_mask

    trans_lab = np.clip(trans_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(trans_lab, cv2.COLOR_LAB2BGR)


def subject_aware_transfer(
    engine,
    target_img: np.ndarray,
    ref_img: np.ndarray,
    target_faces=None,
    target_person=None,
) -> np.ndarray:
    """Segment and match skin, hair, and background color statistics independently from the source image.

    This avoids compounding transfer and color drift in overlapping regions.
    """
    # 1. Target masks
    target_faces = target_faces if target_faces is not None else engine._detector.detect(target_img)
    if target_person is None:
        target_person = engine._detector.segment_person(target_img)
    
    target_person_f = target_person.astype(np.float32)
    if target_person_f.max() > 1.0:
        target_person_f /= 255.0
    if target_person_f.ndim == 3:
        target_person_f = target_person_f[:, :, 0]

    target_bg_mask = 1.0 - target_person_f

    # 2. Reference masks
    ref_faces = engine._detector.detect(ref_img)
    ref_person = engine._detector.segment_person(ref_img)
    
    ref_person_f = ref_person.astype(np.float32)
    if ref_person_f.max() > 1.0:
        ref_person_f /= 255.0
    if ref_person_f.ndim == 3:
        ref_person_f = ref_person_f[:, :, 0]

    ref_bg_mask = 1.0 - ref_person_f

    # Base transfer for background
    result = reinhard_transfer_masked(target_img, ref_img, target_bg_mask, ref_bg_mask)

    # --- Skin & Hair Transfer (if faces detected in both) ---
    if target_faces and ref_faces:
        t_face = target_faces[0]
        r_face = ref_faces[0]

        t_regions = engine._parser.parse(
            t_face.landmarks, target_img, t_face.bbox, target_person_f, t_face.ied
        )
        r_regions = engine._parser.parse(
            r_face.landmarks, ref_img, r_face.bbox, ref_person_f, r_face.ied
        )

        # Skin transfer (matched from original target_img and blended back)
        if t_regions.skin is not None and r_regions.skin is not None:
            t_skin = t_regions.skin.astype(np.float32)
            if t_skin.max() > 1.0:
                t_skin /= 255.0
            r_skin = r_regions.skin.astype(np.float32)
            if r_skin.max() > 1.0:
                r_skin /= 255.0
            
            skin_trans = reinhard_transfer_masked(target_img, ref_img, t_skin, r_skin)
            t_skin_3d = t_skin[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1.0 - t_skin_3d) + skin_trans.astype(np.float32) * t_skin_3d).astype(np.uint8)

        # Hair transfer (matched from original target_img and blended back)
        if t_regions.hair is not None and r_regions.hair is not None:
            t_hair = t_regions.hair.astype(np.float32)
            if t_hair.max() > 1.0:
                t_hair /= 255.0
            r_hair = r_regions.hair.astype(np.float32)
            if r_hair.max() > 1.0:
                r_hair /= 255.0
            
            hair_trans = reinhard_transfer_masked(target_img, ref_img, t_hair, r_hair)
            t_hair_3d = t_hair[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1.0 - t_hair_3d) + hair_trans.astype(np.float32) * t_hair_3d).astype(np.uint8)

    return result

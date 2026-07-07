"""StyleAnalyzer & StyleApplier — automatic style cloning and subject-aware color matching."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .detection import FaceDetector
from .frequency import separate as freq_separate
from .parsing import FaceParser
from .style_transfer import (
    reinhard_transfer_masked,
    subject_aware_transfer,
    weighted_mean_std,
)
from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, normalize_mask


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

    def __init__(
        self,
        engine: Optional[Any] = None,
        detector: Optional[FaceDetector] = None,
        parser: Optional[FaceParser] = None,
    ) -> None:
        # Prefer to depend on a face detector + parser (the analyzer only needs
        # detection + parsing, not the full retouching pipeline). This breaks
        # the engine→style→engine cycle at module load time. ``engine`` is
        # still accepted (and stored) for backwards compatibility — it's only
        # materialized lazily on first access to keep __init__ cheap.
        self._engine = engine
        if engine is not None:
            self._detector = detector or getattr(engine, "_detector", None) or FaceDetector()
            self._parser = parser or getattr(engine, "_parser", None) or FaceParser()
        else:
            self._detector = detector or FaceDetector()
            self._parser = parser or FaceParser()

    @property
    def engine(self) -> Optional[Any]:
        """Backwards-compatible accessor for the underlying engine.

        Created lazily on first access so importing ``retouch.style`` does not
        pull in ``retouch.engine`` at module load time.
        """
        if self._engine is None:
            from .engine import RetouchEngine
            self._engine = RetouchEngine()
        return self._engine

    @engine.setter
    def engine(self, value: Any) -> None:
        self._engine = value

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

        # Global Brightness Delta (difference in LAB L channel median, more robust than mean)
        brightness_delta = float(np.percentile(edit_l, 50) - np.percentile(orig_l, 50))

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
        faces = self._detector.detect(original_img)

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
            person_mask = self._detector.segment_person(original_img)

            total_area = 0.0
            weighted_face_width = 0.0

            for face in faces:
                area = face.bbox[2] * face.bbox[3]
                total_area += area
                weighted_face_width += (face.ied * 2.5) * area

                regions = self._parser.parse(
                    face.landmarks, original_img, face.bbox, person_mask, face.ied
                )
                if regions.skin is not None:
                    s_mask = normalize_mask(regions.skin.copy())
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
                    # Crop to skin bounding box first — 5-20x faster on large portraits
                    fs_radius = face_width
                    skin_ys, skin_xs = np.where(skin_indices)
                    y0 = max(0, int(skin_ys.min()) - 50)
                    y1 = min(h_img, int(skin_ys.max()) + 50)
                    x0 = max(0, int(skin_xs.min()) - 50)
                    x1 = min(w_img, int(skin_xs.max()) + 50)

                    crop_orig = original_img[y0:y1, x0:x1]
                    crop_edit = edited_img[y0:y1, x0:x1]
                    crop_skin = skin_indices[y0:y1, x0:x1]

                    orig_layers = freq_separate(crop_orig, fs_radius)
                    edit_layers = freq_separate(crop_edit, fs_radius)

                    # Convert multi-channel layers to 1D luminance arrays to avoid channel shape bugs
                    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights
                    orig_mid_gray = np.dot(orig_layers.mid, weights)
                    edit_mid_gray = np.dot(edit_layers.mid, weights)
                    orig_high_gray = np.dot(orig_layers.high, weights)
                    edit_high_gray = np.dot(edit_layers.high, weights)

                    # Extract mid frequency energy inside skin mask
                    orig_mid_skin = orig_mid_gray[crop_skin]
                    edit_mid_skin = edit_mid_gray[crop_skin]
                    orig_mid_energy = np.mean(np.abs(orig_mid_skin))
                    edit_mid_energy = np.mean(np.abs(edit_mid_skin))

                    mid_ratio = float(edit_mid_energy / (orig_mid_energy + 1e-5))
                    skin_mid_reduction = float(np.clip(1.0 - mid_ratio, 0.0, 1.0))

                    # Extract high frequency energy inside skin mask
                    orig_high_skin = orig_high_gray[crop_skin]
                    edit_high_skin = edit_high_gray[crop_skin]
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

    def extract_look(
        self,
        reference_img: np.ndarray,
        base_img: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Reverse-engineer an editable preset from a reference image.

        Two modes:
          - Paired: base_img (original) + reference_img (edited) → extract delta
          - Unpaired: reference_img only → extract absolute look characteristics

        Args:
            reference_img: BGR uint8 image representing the target look.
            base_img: Optional BGR uint8 original image (for paired extraction).
            name: Optional preset name. If None, auto-generated from filename.
            save: If True, write preset JSON to presets/ directory.

        Returns:
            Preset dictionary with keys: description, curves, white_balance,
            split_tone_three_way, hsl_adjustments.
        """
        if base_img is not None:
            if base_img.shape[:2] != reference_img.shape[:2]:
                reference_img = cv2.resize(
                    reference_img,
                    (base_img.shape[1], base_img.shape[0]),
                )
            source_for_curve = base_img
        else:
            source_for_curve = reference_img

        preset: Dict[str, Any] = {}

        preset["curves"] = self._extract_l_curve(source_for_curve, reference_img)
        preset["white_balance"] = self._extract_white_balance(reference_img)
        preset["split_tone_three_way"] = self._extract_split_tone(reference_img)
        preset["hsl_adjustments"] = self._extract_hsl_adjustments(
            source_for_curve, reference_img
        )

        if name is None:
            name = "extracted_look"
        preset["description"] = f"extracted from {name}"

        if save:
            self._save_preset(preset, name)

        return preset

    def _extract_l_curve(
        self,
        base_img: np.ndarray,
        reference_img: np.ndarray,
    ) -> Dict[str, List[List[int]]]:
        """Extract L channel curve from percentile mapping."""
        base_lab = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        base_l = base_lab[:, :, 0].ravel()
        ref_l = ref_lab[:, :, 0].ravel()

        percentiles = [0, 5, 25, 50, 75, 95, 100]
        base_pts = np.percentile(base_l, percentiles)
        ref_pts = np.percentile(ref_l, percentiles)

        curve_points = []
        for b, r in zip(base_pts, ref_pts):
            curve_points.append([int(round(b)), int(round(r))])

        curve_points = self._ensure_monotonic(curve_points)

        return {"L": curve_points}

    def _ensure_monotonic(
        self, points: List[List[int]]
    ) -> List[List[int]]:
        """Ensure curve points are monotonically non-decreasing in Y."""
        if len(points) < 2:
            return points
        result = [points[0][:]]
        for i in range(1, len(points)):
            x, y = points[i]
            prev_y = result[-1][1]
            if y < prev_y:
                y = prev_y
            result.append([x, y])
        return result

    def _extract_white_balance(
        self, reference_img: np.ndarray
    ) -> Dict[str, float]:
        """Extract white balance from gray-world assumption."""
        img_f = reference_img.astype(np.float32)
        b_mean = np.mean(img_f[:, :, 0])
        g_mean = np.mean(img_f[:, :, 1])
        r_mean = np.mean(img_f[:, :, 2])

        gray = (b_mean + g_mean + r_mean) / 3.0

        if gray < 1.0:
            return {"R": 1.0, "G": 1.0, "B": 1.0}

        r_gain = gray / r_mean
        g_gain = gray / g_mean
        b_gain = gray / b_mean

        max_gain = max(r_gain, g_gain, b_gain)
        if max_gain > 0:
            r_gain /= max_gain
            g_gain /= max_gain
            b_gain /= max_gain

        return {
            "R": round(float(r_gain), 3),
            "G": round(float(g_gain), 3),
            "B": round(float(b_gain), 3),
        }

    def _extract_split_tone(
        self, reference_img: np.ndarray
    ) -> Dict[str, Any]:
        """Extract split tone from LAB a/b in luminance bands."""
        lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1] - 128.0
        b_chan = lab[:, :, 2] - 128.0

        shadow_mask = l_chan < 85
        midtone_mask = (l_chan >= 85) & (l_chan <= 170)
        highlight_mask = l_chan > 170

        def band_to_hue_sat(
            mask: np.ndarray, a: np.ndarray, b: np.ndarray
        ) -> Tuple[float, float]:
            if np.sum(mask) < 100:
                return 0.0, 0.0
            a_mean = np.mean(a[mask])
            b_mean = np.mean(b[mask])
            chroma = np.sqrt(a_mean ** 2 + b_mean ** 2)
            hue_rad = np.arctan2(b_mean, a_mean)
            hue_deg = np.degrees(hue_rad) % 360.0
            sat = min(chroma / 25.0 * 100.0, 100.0)
            return round(float(hue_deg), 1), round(float(sat), 1)

        s_hue, s_sat = band_to_hue_sat(shadow_mask, a_chan, b_chan)
        m_hue, m_sat = band_to_hue_sat(midtone_mask, a_chan, b_chan)
        h_hue, h_sat = band_to_hue_sat(highlight_mask, a_chan, b_chan)

        l_norm = l_chan / 255.0
        balance = float(np.mean(l_norm) - 0.5) * 100.0

        return {
            "shadows": {"hue": s_hue, "sat": s_sat},
            "midtones": {"hue": m_hue, "sat": m_sat},
            "highlights": {"hue": h_hue, "sat": h_sat},
            "balance": round(balance, 1),
        }

    def _extract_hsl_adjustments(
        self,
        base_img: np.ndarray,
        reference_img: np.ndarray,
    ) -> Dict[str, Dict[str, int]]:
        """Extract per-hue HSL adjustments from saturation differences."""
        color_centers = {
            "red": 0.0, "orange": 14.0, "yellow": 27.0, "green": 57.0,
            "cyan": 92.0, "blue": 122.0, "purple": 148.0, "magenta": 156.0,
        }
        sigma = 10.0

        base_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor(reference_img, cv2.COLOR_BGR2HSV).astype(np.float32)

        base_h = base_hsv[:, :, 0]
        base_s = base_hsv[:, :, 1]
        ref_s = ref_hsv[:, :, 1]

        sat_adjusts: Dict[str, int] = {}
        for color, center in color_centers.items():
            dist = np.abs(base_h - center)
            dist = np.minimum(dist, 180.0 - dist)
            weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

            if np.sum(weight) < 100:
                continue

            base_sat_weighted = np.sum(base_s * weight) / np.sum(weight)
            ref_sat_weighted = np.sum(ref_s * weight) / np.sum(weight)
            delta = ref_sat_weighted - base_sat_weighted

            if abs(delta) > 2.0:
                sat_adjusts[color] = int(round(delta))

        result: Dict[str, Dict[str, int]] = {}
        if sat_adjusts:
            result["saturation"] = sat_adjusts

        return result

    def _save_preset(
        self, preset: Dict[str, Any], name: str
    ) -> Path:
        """Save preset to presets/ directory."""
        presets_dir = Path(__file__).resolve().parent.parent / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)

        filename = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
        filename = filename.strip("_") or "extracted_look"
        filepath = presets_dir / f"{filename}.json"

        suffix = 2
        while filepath.exists():
            filepath = presets_dir / f"{filename}_{suffix}.json"
            suffix += 1

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)

        logger.info("Saved extracted preset to %s", filepath)
        return filepath


class StyleApplier:
    """Apply style profile attributes or perform subject-aware color transfer."""

    def __init__(
        self,
        engine: Optional[Any] = None,
        detector: Optional[FaceDetector] = None,
        parser: Optional[FaceParser] = None,
    ) -> None:
        # StyleApplier still needs the full engine for ``engine.process()``,
        # but detection/parsing is done via the lighter detector+parser when
        # given, avoiding reliance on the engine's private attributes. The
        # engine is materialized lazily on first access to keep __init__ cheap
        # and to avoid pulling ``retouch.engine`` in at module load time.
        self._engine = engine
        self._detector = detector or (getattr(engine, "_detector", None) if engine is not None else None)
        self._parser = parser or (getattr(engine, "_parser", None) if engine is not None else None)

    @property
    def engine(self) -> Optional[Any]:
        """Backwards-compatible accessor for the underlying engine.

        Created lazily on first access so importing ``retouch.style`` does not
        pull in ``retouch.engine`` at module load time.
        """
        if self._engine is None:
            from .engine import RetouchEngine
            self._engine = RetouchEngine()
        return self._engine

    @engine.setter
    def engine(self, value: Any) -> None:
        self._engine = value

    def _ensure_detector_and_parser(self) -> None:
        """Lazily create a FaceDetector/FaceParser if neither was injected and
        no engine is bound. Avoids the heavy default-detection at __init__ time
        while still letting the class be used standalone.
        """
        if self._detector is None:
            self._detector = FaceDetector()
        if self._parser is None:
            self._parser = FaceParser()

    def apply(self, target_img: np.ndarray, profile: StyleProfile) -> np.ndarray:
        """Process target_img using parameter values resolved from the StyleProfile."""
        # 1. Map global parameters (brightness delta L maps to slider value ≈ scale by 1.0)
        #    gamma model: brightness=10 → gamma=0.9 → mid-gray +9L, so factor ~1.0
        brightness_val = np.clip(profile.brightness_delta * 1.0, -100.0, 100.0)

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

        result = self.engine.process(
            target_img,
            smooth=smooth_val,
            mid_reduction=mid_red_val,
            texture_opacity=tex_op_val,
            whiten=whiten_val,
            whiten_tone=whiten_tone,
            contrast=profile.contrast_delta,
            brightness=brightness_val,
            saturation=profile.saturation_delta,
        )

        # 3. Direct LAB skin tone shift (preserves fine-grained A/B deltas)
        if abs(profile.skin_a_mean_delta) > 0.5 or abs(profile.skin_b_mean_delta) > 0.5:
            self._ensure_detector_and_parser()
            faces = self._detector.detect(result)
            if faces:
                faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
                person = self._detector.segment_person(result)
                person_f = normalize_mask(person)
                if person_f.ndim == 3:
                    person_f = person_f[:, :, 0]
                regions = self._parser.parse(
                    faces[0].landmarks, result, faces[0].bbox, person_f, faces[0].ied
                )
                if regions.skin is not None:
                    s_mask = normalize_mask(regions.skin)
                    is_float = result.dtype == np.float32
                    if is_float:
                        lab = bgr_f32_to_lab_f32(result)
                        lab[:, :, 1] += profile.skin_a_mean_delta * s_mask
                        lab[:, :, 2] += profile.skin_b_mean_delta * s_mask
                        result = lab_f32_to_bgr_f32(lab)
                    else:
                        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
                        lab[:, :, 1] += profile.skin_a_mean_delta * s_mask
                        lab[:, :, 2] += profile.skin_b_mean_delta * s_mask
                        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result


__all__ = [
    "StyleProfile",
    "StyleAnalyzer",
    "StyleApplier",
    "subject_aware_transfer",
    "reinhard_transfer_masked",
    "weighted_mean_std",
]

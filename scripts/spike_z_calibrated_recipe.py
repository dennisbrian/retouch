#!/usr/bin/env python3
"""Z1/Z2 prototype — measurement-calibrated recipe strengths, end to end.

Evidence spike for RESEARCH_PERCEPTUAL_CALIBRATION_Z_2026_07_17.md (Z1
facial contrast, Z2 skin homogeneity), honoring the scope corrections in
TODO_PERCEPTUAL_CALIBRATION_Z_2026_07_17.md:

- Every measurement is WITHIN-FACE relative (feature vs this face's own
  surrounding skin; blotch dispersion vs this face's own median chroma).
  No demographic inference, no absolute "correct" appearance targets.
- Suggestions are bounded deltas on top of `calibrated_natural_v1`'s
  conservative base, clamped by per-profile caps. A face already at or
  above its own restore floor receives zero adjustment.
- Verification, not trust: the spike re-measures both renders and enforces
  the Z2 pore-energy floor (render keeps >= 85% of source pore energy).

This is a spike, not the production module: the Z TODO's Stage Z0/Z1/Z2
(`retouch/perceptual_metrics.py`) owns the reviewed implementation; this
script exists so that work starts from measured behavior on real assets
(same pattern as spike_harmony_metrics.py -> harmony.py).

Run: python3 scripts/spike_z_calibrated_recipe.py [image ...]
Writes test_output/z_calibrated_recipe/<name>_panel.jpg + a metrics report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULTS = [
    ROOT / "test_output" / "DSCF4550.jpg",
    ROOT / "test_output" / "DSCF7011.jpg",
    ROOT / "test_output" / "DSCF7142.jpg",
]
RECIPE_JSON = ROOT / "recipes" / "calibrated_natural_v1.json"
RECIPE_NAME = "calibrated_natural_v1"

_MIN_SUPPORT_PX = 200
_EPS = 1e-6

# ---------------------------------------------------------------------------
# Z1 — facial feature contrast (within-face relative)
# ---------------------------------------------------------------------------

_FEATURES = {
    "eyes": ("eyes", "left_eye", "right_eye"),
    "lips": ("lips",),
    "eyebrows": ("eyebrows", "left_eyebrow", "right_eyebrow"),
}


def _get_mask(regions, name: str) -> Optional[np.ndarray]:
    m = regions.get(name) if isinstance(regions, Mapping) else getattr(regions, name, None)
    if m is None:
        return None
    m = m[..., 0] if m.ndim == 3 else m
    return m.astype(np.float32)


def _feature_mask(regions, feature: str) -> Optional[np.ndarray]:
    """Resolve synthetic combined masks and production FaceRegions aliases."""
    combined = None
    for name in _FEATURES[feature]:
        mask = _get_mask(regions, name)
        if mask is not None:
            combined = mask if combined is None else np.maximum(combined, mask)
    return combined


def _lab_f32(img_bgr: np.ndarray) -> np.ndarray:
    u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2LAB).astype(np.float32)


def feature_contrast(img_bgr: np.ndarray, regions, face_width: float) -> Dict[str, Dict[str, float]]:
    """Per-feature L*/a*/b* contrast vs this face's own surrounding skin.

    L contrast is Michelson-style ((L_ring - L_feat) / (L_ring + L_feat)),
    positive when the feature is darker than its surround — approximately
    stable under multiplicative luminance change, i.e. tone-relative.
    a/b are signed deltas (feature minus surround) in CIELAB units.
    """
    lab = _lab_f32(img_bgr)
    skin = _get_mask(regions, "skin")
    if skin is None:
        return {}
    feature_masks = {name: _feature_mask(regions, name) for name in _FEATURES}
    all_feats = [mask for mask in feature_masks.values() if mask is not None]
    others = np.zeros(skin.shape, np.uint8)
    for f in all_feats:
        others |= (f > 0.3).astype(np.uint8)
    r_out = max(6, int(round(face_width * 0.05)))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
    k_in = np.ones((5, 5), np.uint8)

    out: Dict[str, Dict[str, float]] = {}
    for name, feat_m in feature_masks.items():
        if feat_m is None:
            continue
        feat = feat_m > 0.3
        ring = (cv2.dilate(feat.astype(np.uint8), k_out) > 0) \
            & (cv2.dilate(feat.astype(np.uint8), k_in) == 0) \
            & (skin > 0.3) & (others == 0)
        n_f, n_r = int(feat.sum()), int(ring.sum())
        if n_f < _MIN_SUPPORT_PX or n_r < _MIN_SUPPORT_PX:
            out[name] = {"confidence": 0.0}
            continue
        lf, af, bf = (float(np.median(lab[..., c][feat])) for c in range(3))
        lr, ar, br = (float(np.median(lab[..., c][ring])) for c in range(3))
        out[name] = {
            "l_contrast": (lr - lf) / max(lr + lf, _EPS),
            "a_delta": af - ar,
            "b_delta": bf - br,
            "confidence": 1.0,
            "feat_px": n_f,
            "ring_px": n_r,
        }
    return out


# ---------------------------------------------------------------------------
# Z2 — skin homogeneity state (within-face relative)
# ---------------------------------------------------------------------------

def homogeneity_state(img_bgr: np.ndarray, skin_mask: np.ndarray, face_width: float) -> Dict[str, float]:
    """Blotch-band chroma dispersion + pore-band energy, face-scale-relative."""
    lab = _lab_f32(img_bgr)
    skin = skin_mask[..., 0] if skin_mask.ndim == 3 else skin_mask
    sel = skin > 0.5
    if int(sel.sum()) < 4 * _MIN_SUPPORT_PX:
        return {"available": 0.0}

    def _mad_std(v: np.ndarray) -> float:
        med = np.median(v)
        return float(1.4826 * np.median(np.abs(v - med)))

    s1 = max(1.5, face_width / 60.0)
    s2 = max(2.5 * s1, face_width / 15.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    a_band = cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)
    b_band = cv2.GaussianBlur(b, (0, 0), s1) - cv2.GaussianBlur(b, (0, 0), s2)
    sigma_c = float(np.hypot(_mad_std(a_band[sel]), _mad_std(b_band[sel])))
    median_c = float(np.median(np.hypot(a[sel], b[sel])))

    s_pore = max(1.2, face_width / 300.0)
    l_high = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), s_pore)
    return {
        "available": 1.0,
        "sigma_c_blotch": sigma_c,
        "median_chroma": median_c,
        # Dimensionless: this face's blotch dispersion vs its own chroma level.
        "rel_blotch": sigma_c / max(median_c, 1.0),
        "pore_energy": _mad_std(l_high[sel]),
    }


# ---------------------------------------------------------------------------
# Target-seeking suggestions (bounded deltas, intent profiles)
# ---------------------------------------------------------------------------

# Caps are engine-scale (0-100) ceilings for the *added* strength; the
# restore floor is a fraction of THIS face's strongest feature contrast.
# rel_blotch ramp endpoints are prototype seeds, printed by every run so the
# corpus pass can recalibrate them; they are within-face ratios, not
# absolute skin values.
PROFILES = {
    "natural": dict(eye_cap=20.0, lip_cap=16.0, even_cap=25.0,
                    restore_floor=0.60, blotch_lo=0.08, blotch_hi=0.20),
    "soft_portrait": dict(eye_cap=12.0, lip_cap=10.0, even_cap=35.0,
                          restore_floor=0.50, blotch_lo=0.06, blotch_hi=0.16),
    "editorial": dict(eye_cap=28.0, lip_cap=22.0, even_cap=20.0,
                      restore_floor=0.70, blotch_lo=0.10, blotch_hi=0.24),
}


def suggest_strengths(contrast: Dict[str, Dict[str, float]],
                      homog: Dict[str, float],
                      profile: str = "natural") -> Dict[str, float]:
    """Bounded, within-face-relative parameter deltas for the base recipe.

    A feature already at/above ``restore_floor`` x (this face's strongest
    feature contrast) gets nothing. Deficient features get a strength
    proportional to their deficit, clamped by the profile cap.
    """
    p = PROFILES[profile]
    conf = {n: c for n, c in contrast.items() if c.get("confidence", 0.0) >= 0.5}
    out: Dict[str, float] = {}
    if conf:
        ref = max(c["l_contrast"] for c in conf.values())
        floor = p["restore_floor"] * max(ref, _EPS)
        if ref > 0:
            for name, cap_key, param in (("eyes", "eye_cap", "eye_enhance"),
                                          ("lips", "lip_cap", "lip_enhance")):
                c = conf.get(name)
                if c is None:
                    continue
                deficit = np.clip((floor - c["l_contrast"]) / max(floor, _EPS), 0.0, 1.0)
                val = round(float(p[cap_key] * deficit), 1)
                if val >= 1.0:
                    out[param] = val
    if homog.get("available"):
        t = np.clip((homog["rel_blotch"] - p["blotch_lo"]) /
                    max(p["blotch_hi"] - p["blotch_lo"], _EPS), 0.0, 1.0)
        val = round(float(p["even_cap"] * t), 1)
        if val >= 1.0:
            out["skin_chroma_even"] = val
    return out


def pore_floor_ok(src_state: Dict[str, float], render_state: Dict[str, float],
                  floor: float = 0.85) -> bool:
    """Z2 safety guard: the render must keep >= floor of source pore energy."""
    if not (src_state.get("available") and render_state.get("available")):
        return True
    return render_state["pore_energy"] >= floor * src_state["pore_energy"]


# ---------------------------------------------------------------------------
# Demo through the real engine
# ---------------------------------------------------------------------------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = np.ascontiguousarray(img)
    cv2.putText(out, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(out, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    from retouch.detection import FaceDetector
    from retouch.engine import RetouchEngine
    from retouch.parsing import FaceParser
    from retouch.recipes import RECIPES
    from retouch.recipe_loader import import_recipe

    if RECIPE_NAME not in RECIPES:
        import_recipe(RECIPE_JSON, save=False)

    paths = [Path(p) for p in sys.argv[1:]] or DEFAULTS
    out_dir = ROOT / "test_output" / "z_calibrated_recipe"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = RetouchEngine()
    det, parser = FaceDetector(), FaceParser()

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name}: unreadable, skipped")
            continue
        scale = 1600.0 / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = det.detect(img)
        if not faces:
            print(f"{path.name}: no face, skipped")
            continue
        f = faces[0]
        regions = parser.parse(f.landmarks, img, f.bbox, ied=f.ied)
        face_width = float(f.bbox[2])

        src_contrast = feature_contrast(img, regions, face_width)
        src_homog = homogeneity_state(img, regions.skin, face_width)
        suggested = suggest_strengths(src_contrast, src_homog, "natural")

        base = engine.process(img.copy(), recipe=RECIPE_NAME)
        calib = engine.process(img.copy(), recipe=RECIPE_NAME,
                               face_contexts=base.face_contexts, **suggested)

        rows = [f"{path.stem}: suggested={suggested or '{} (face within its own bands)'}"]
        for label, render in (("base", np.asarray(base)), ("calibrated", np.asarray(calib))):
            rc = feature_contrast(render, regions, face_width)
            rh = homogeneity_state(render, regions.skin, face_width)
            guard = "PASS" if pore_floor_ok(src_homog, rh) else "FAIL"
            eyes = rc.get("eyes", {}).get("l_contrast", float("nan"))
            lips = rc.get("lips", {}).get("l_contrast", float("nan"))
            rows.append(
                f"  {label:<10} eyesC={eyes:.3f} lipsC={lips:.3f} "
                f"relBlotch={rh.get('rel_blotch', float('nan')):.3f} "
                f"pore={rh.get('pore_energy', float('nan')):.2f} poreFloor={guard}")
        src_eyes = src_contrast.get("eyes", {}).get("l_contrast", float("nan"))
        src_lips = src_contrast.get("lips", {}).get("l_contrast", float("nan"))
        rows.insert(1, f"  {'source':<10} eyesC={src_eyes:.3f} lipsC={src_lips:.3f} "
                       f"relBlotch={src_homog.get('rel_blotch', float('nan')):.3f} "
                       f"pore={src_homog.get('pore_energy', float('nan')):.2f}")
        print("\n".join(rows))

        panel = np.hstack([_label(img.copy(), "source"),
                           _label(np.asarray(base).copy(), "base recipe"),
                           _label(np.asarray(calib).copy(),
                                  f"calibrated {suggested if suggested else '(no delta)'}")])
        s = 2400.0 / panel.shape[1]
        panel = cv2.resize(panel, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{path.stem}_panel.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\npanels: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared metrics for the golden hard-case QA harness.

Used by tests/test_golden_qa.py (CI) and scripts/qa/golden_qa.py (the full
recipe sweep and reference approval). Cases live in tests/golden/ (see its
README.md); every check here is relative to the photo itself, so the same
thresholds hold on darker and lighter skin.

Checks
------
- face count: faces the engine retouches == people in the photo.
- skin leak: with a recipe's skin ops on vs off (everything else equal),
  the pixels that change by more than ``CHANGE_LEVELS`` outside the skin
  mask (grown by ``SKIN_GROW_FW`` face widths for feathering), as a share
  of the skin mask's own area.
- object change: the share of a planted non-skin object (clip, prop,
  frames, poster) inside the face box grown 1.5x that changes at all.
- SSIM against an approved reference output (tests/golden/references/).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GOLDEN = Path(__file__).parent / "golden"
REFERENCES = GOLDEN / "references"

#: A pixel "changed" when any channel moved by more than this many levels.
CHANGE_LEVELS = 3.0
#: Skin mask is grown by this share of the face width before counting a
#: change as outside it (ops feather their edges by up to ~3% face width).
SKIN_GROW_FW = 0.04
#: Max changed-outside-skin pixels, as a share of the skin mask's area.
SKIN_LEAK_MAX = 0.02
#: Max share of the hair mask (away from the grown skin) that may change.
HAIR_CHANGE_MAX = 0.02
#: Ops that edit the face's hair on purpose: Shadow Lift's scope is skin +
#: hair (see perf_optimizations, "Shadow lift"), and Face Polish raises the
#: Shadow Lift floor. Their hair changes are allowed.
EDITS_HAIR = frozenset({"shadow_lift", "face_polish"})
#: Ops that by design carry their edit onto same-person skin outside the
#: face skin mask (forehead under the hairline, ears, body): their skin-leak
#: number is reported but not enforced; hair and object checks still apply.
SPREADS_TO_BODY_SKIN = frozenset({"skin_warmth"})
#: Max share of a planted object (inside the grown face box) that may change.
OBJECT_CHANGE_MAX = 0.02
#: Min SSIM (luma, Gaussian window) of an output against its approved reference.
SSIM_MIN = 0.97

#: Recipes with approved reference outputs (one per family the shoots use).
REFERENCE_RECIPES = ("natural", "cosplay_clear_v1", "studio_porcelain_clear_v1")

#: Face ops that should only change skin. Eye, lip, brow, hair, reshape and
#: makeup-colour ops legitimately change other face parts and are left out.
SKIN_OPS: Tuple[str, ...] = (
    "smooth", "whiten", "equalize", "blemish", "nose_smooth", "micro_restore",
    "micro_dodge_burn", "mid_reduction", "blotch_reduction", "pore_synthesis",
    "specular_bloom", "specular_finish_strength", "albedo_even",
    "makeup_coverage_even", "makeup_cake_reduce", "hemoglobin_smooth",
    "vein_attenuate", "dodge_burn", "face_exposure", "face_polish",
    "skin_flatten", "skin_quantize", "skin_unify", "skin_hue_unify",
    "skin_chroma_even", "redness_even", "skin_glow", "shine_removal",
    "wrinkle_soften", "wrinkle_soften_forehead", "wrinkle_soften_nasolabial",
    "texture_transplant", "shadow_lift", "face_shadow_fill", "skin_sss",
    "freckle_removal", "spot_heal", "powder_finish", "highlight_repair",
    "nose_shape", "nose_highlight", "skin_warmth",
)


@dataclass
class Case:
    name: str
    persons: int
    kind: str
    source: str
    why: str
    objects_mask: Optional[str] = None
    known_gap: Optional[str] = None
    object_gap: Optional[str] = None
    image: np.ndarray = field(default=None, repr=False)

    @property
    def objects(self) -> Optional[np.ndarray]:
        if not self.objects_mask:
            return None
        m = cv2.imread(str(GOLDEN / self.objects_mask), cv2.IMREAD_GRAYSCALE)
        # Drop the 1 px anti-aliased rim: it is part skin by construction.
        return cv2.erode((m > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


@lru_cache(maxsize=1)
def load_cases() -> Dict[str, Case]:
    spec = json.loads((GOLDEN / "cases.json").read_text())
    out = {}
    for name, entry in spec.items():
        img = cv2.imread(str(GOLDEN / "inputs" / f"{name}.jpg"), cv2.IMREAD_COLOR)
        out[name] = Case(name=name, image=img, **entry)
    return out


def _param_specs():
    from retouch.params import PROCESSING_PARAMS

    return {p.name: p for p in PROCESSING_PARAMS}


def skin_op_names() -> List[str]:
    """SKIN_OPS that exist as engine parameters on this checkout."""
    specs = _param_specs()
    return [n for n in SKIN_OPS if n in specs]


def probe_value(name: str):
    """A strong but in-range setting for one op (80% of its max)."""
    spec = _param_specs()[name]
    hi = spec.max_val if spec.max_val is not None else (1.0 if spec.cli_type is float else 100)
    val = 0.8 * hi
    return int(round(val)) if spec.cli_type is int else float(val)


def skin_ops_off() -> Dict[str, object]:
    """Process kwargs that switch every skin op off."""
    specs = _param_specs()
    return {n: (0 if specs[n].cli_type is int else 0.0) for n in skin_op_names()}


#: Random film grain is drawn from the global RNG after the face ops, so any
#: op that draws a random number reshuffles the grain over the whole frame.
#: Leak and object comparisons switch grain off on both sides.
NO_GRAIN = {"grain": 0.0, "grain_strength": 0.0, "micro_grain": 0}


#: Body-skin ops follow the face on purpose (Match Face copies its colour;
#: Body Shadow Lift's local average reaches across the face), so a face edit
#: moves body pixels through them. Recipe-level leak renders hold them off
#: on both sides so the number measures the face ops alone.
BODY_OPS_OFF = {
    "body_smooth": 0, "body_equalize": 0, "body_skin_even": 0, "body_whiten": 0,
    "body_match_face": 0, "body_relight": 0, "body_dodge_burn": 0, "body_shadow_lift": 0,
}


def changed(a: np.ndarray, b: np.ndarray, levels: float = CHANGE_LEVELS) -> np.ndarray:
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    return d.max(axis=2) > levels


def face_width(result) -> float:
    ctxs = getattr(result, "face_contexts", None) or []
    widths = [float(c.face_data.bbox[2]) for c in ctxs]
    return max(widths) if widths else 0.0


def region_mask(result, shape, name: str) -> np.ndarray:
    """One FaceRegions mask (all faces) pasted into frame coordinates.

    Regions are stored at the face ROI crop size; the ROI is rebuilt exactly
    as RetouchEngine._stage_per_face crops it.
    """
    from retouch.engine import RetouchEngine

    out = np.zeros(shape, np.float32)
    for c in getattr(result, "face_contexts", None) or []:
        m = getattr(c.regions, name, None)
        if m is None:
            continue
        x, y, w, h = (int(v) for v in c.face_data.bbox)
        pt, pb, pl, pr = RetouchEngine._compute_face_roi_padding(w, h)
        y1, y2 = max(0, y - pt), min(shape[0], y + h + pb)
        x1, x2 = max(0, x - pl), min(shape[1], x + w + pr)
        m = np.asarray(m, np.float32)
        if m.max() > 1.5:
            m = m / 255.0
        if m.shape != (y2 - y1, x2 - x1):
            m = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], m)
    return out


def allowed_skin(result, shape, include_hair: bool = False) -> Optional[np.ndarray]:
    """Grown skin mask: the engine's face skin plus the neck below it (and
    the face's hair mask, for ops that edit hair on purpose)."""
    m = getattr(result, "skin_mask", None)
    if m is None:
        return None
    m = np.asarray(m, np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.maximum(m, region_mask(result, shape, "neck"))
    if include_hair:
        m = np.maximum(m, region_mask(result, shape, "hair"))
    grow = max(1, int(round(SKIN_GROW_FW * max(face_width(result), 1.0))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate((m > 0.05).astype(np.uint8), k) > 0


def skin_leak(on, off, ref_result, include_hair: bool = False) -> Tuple[float, int]:
    """(leak share of skin area, leaked pixel count) between two renders."""
    ch = changed(on, off)
    allowed = allowed_skin(ref_result, ch.shape, include_hair)
    if allowed is None or not allowed.any():
        return (0.0 if not ch.any() else 1.0), int(ch.sum())
    leaked = ch & ~allowed
    return float(leaked.sum()) / float(allowed.sum()), int(leaked.sum())


def hair_change(on, off, ref_result) -> Optional[float]:
    """Share of the hair mask (away from grown skin) that changed."""
    shape = np.asarray(on).shape[:2]
    hair = region_mask(ref_result, shape, "hair") > 0.5
    allowed = allowed_skin(ref_result, shape)
    if allowed is not None:
        hair &= ~allowed
    if hair.sum() < 50:
        return None
    return float((changed(on, off) & hair).sum()) / float(hair.sum())


def edits_hair(result) -> bool:
    """True when the render's settings include an op that edits hair on purpose."""
    params = getattr(result, "params", None)
    return any(float(getattr(params, n, 0) or 0) > 0 for n in EDITS_HAIR)


def face_zone(result, shape, grow: float = 1.5) -> np.ndarray:
    zone = np.zeros(shape, bool)
    for c in getattr(result, "face_contexts", None) or []:
        x, y, w, h = (float(v) for v in c.face_data.bbox)
        cx, cy = x + w / 2, y + h / 2
        x1, x2 = int(max(0, cx - grow * w / 2)), int(min(shape[1], cx + grow * w / 2))
        y1, y2 = int(max(0, cy - grow * h / 2)), int(min(shape[0], cy + grow * h / 2))
        zone[y1:y2, x1:x2] = True
    return zone


def object_change(on, off, objects: np.ndarray, ref_result) -> Optional[float]:
    """Share of the planted object inside the grown face box that changed."""
    zone = objects & face_zone(ref_result, objects.shape)
    if zone.sum() < 20:
        return None
    return float((changed(on, off) & zone).sum()) / float(zone.sum())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luma with an 11x11 sigma-1.5 Gaussian window."""
    ga = cv2.cvtColor(np.asarray(a, np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(np.asarray(b, np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    ma, mb = blur(ga), blur(gb)
    va, vb = blur(ga * ga) - ma * ma, blur(gb * gb) - mb * mb
    cov = blur(ga * gb) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


def reference_path(case: str, recipe: str) -> Path:
    return REFERENCES / f"{case}__{recipe}.jpg"


def to_u8(img) -> np.ndarray:
    return np.clip(np.asarray(img, np.float32) + 0.5, 0, 255).astype(np.uint8) \
        if np.asarray(img).dtype != np.uint8 else np.asarray(img)

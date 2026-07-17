"""Z1/Z2 prototype tests — calibrated_natural_v1 recipe + spike metrics.

The metric implementations live in scripts/spike_z_calibrated_recipe.py
(spike-first pattern, like spike_harmony_metrics.py before harmony.py);
these tests pin the behaviors the production perceptual_metrics module
must reproduce when the Z0-Z2 stage promotes them.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "spike_z_calibrated_recipe", ROOT / "scripts" / "spike_z_calibrated_recipe.py")
spike = importlib.util.module_from_spec(_spec)
sys.modules["spike_z_calibrated_recipe"] = spike
_spec.loader.exec_module(spike)

RECIPE_JSON = ROOT / "recipes" / "calibrated_natural_v1.json"


# --------------------------------------------------------------------------- #
# Recipe JSON
# --------------------------------------------------------------------------- #

def test_recipe_json_validates_imports_and_resolves():
    from retouch.recipes import RECIPES
    from retouch.recipe_loader import import_recipe
    from retouch.engine import resolve_recipe

    name = "calibrated_natural_v1"
    already = name in RECIPES
    if not already:
        assert import_recipe(RECIPE_JSON, save=False) == name
    try:
        resolved = resolve_recipe(name)
        # Recipe JSON uses normalized values; the resolver exposes the engine
        # strength scale used by FrequencyProcessor.
        assert resolved["frequency"]["smooth"] == pytest.approx(25.0)
        assert resolved["eyes"]["whites"] == pytest.approx(0.10)
        assert resolved["skin"]["chroma_even"] == pytest.approx(0.15)
    finally:
        if not already:
            del RECIPES[name]


# --------------------------------------------------------------------------- #
# Synthetic face fixture
# --------------------------------------------------------------------------- #

def _synthetic_face(tone=(140, 170, 200), lip_darken=60, eye_darken=90,
                    lip_redden=25, noise=4.0, seed=5):
    """Skin canvas with two eye patches and a lip patch; masks as a dict."""
    rng = np.random.RandomState(seed)
    img = np.full((240, 240, 3), tone, dtype=np.float32)
    img += rng.normal(0, noise, img.shape).astype(np.float32)

    masks = {n: np.zeros((240, 240), np.float32) for n in
             ("skin", "eyes", "lips", "eyebrows")}
    masks["skin"][:, :] = 1.0
    masks["eyes"][70:95, 50:100] = 1.0
    masks["eyes"][70:95, 140:190] = 1.0
    masks["lips"][160:185, 90:150] = 1.0
    for name, mask in (("eyes", eye_darken), ("lips", lip_darken)):
        sel = masks[name] > 0.5
        img[sel] -= mask
    img[masks["lips"] > 0.5, 2] += lip_redden  # lips redder than skin
    masks["skin"] -= np.maximum(masks["eyes"], masks["lips"])
    return np.clip(img, 0, 255).astype(np.uint8), masks


# --------------------------------------------------------------------------- #
# Z1 metric behavior
# --------------------------------------------------------------------------- #

def test_feature_contrast_is_monotone_in_feature_darkness():
    img_soft, masks = _synthetic_face(lip_darken=25)
    img_hard, _ = _synthetic_face(lip_darken=80)
    soft = spike.feature_contrast(img_soft, masks, face_width=160)
    hard = spike.feature_contrast(img_hard, masks, face_width=160)
    assert hard["lips"]["l_contrast"] > soft["lips"]["l_contrast"] > 0
    assert soft["lips"]["confidence"] == 1.0


@pytest.mark.parametrize("scale", (1.0, 0.75, 0.55, 0.4))
def test_feature_contrast_is_stable_under_multiplicative_tone_change(scale):
    """Michelson-style contrast must not collapse on darker skin tones."""
    img, masks = _synthetic_face()
    darker = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    ref = spike.feature_contrast(img, masks, face_width=160)["eyes"]["l_contrast"]
    got = spike.feature_contrast(darker, masks, face_width=160)["eyes"]["l_contrast"]
    assert got > 0
    assert abs(got - ref) / ref < 0.4, (scale, ref, got)


def test_feature_contrast_low_support_reports_zero_confidence():
    img, masks = _synthetic_face()
    masks["eyebrows"][10:12, 10:14] = 1.0  # 8 px, below support floor
    out = spike.feature_contrast(img, masks, face_width=160)
    assert out["eyebrows"]["confidence"] == 0.0


# --------------------------------------------------------------------------- #
# Z2 metric behavior
# --------------------------------------------------------------------------- #

def _add_chroma_blotch(img, seed=9, amp=10.0):
    rng = np.random.RandomState(seed)
    field = rng.normal(0, 1, img.shape[:2]).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), 8.0)
    field *= amp / max(field.std(), 1e-6)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + field, 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_homogeneity_separates_blotchy_from_clean_and_pore_leg_is_independent():
    img, masks = _synthetic_face()
    blotchy = _add_chroma_blotch(img)
    clean = spike.homogeneity_state(img, masks["skin"], face_width=160)
    mottled = spike.homogeneity_state(blotchy, masks["skin"], face_width=160)
    assert mottled["rel_blotch"] > 1.5 * clean["rel_blotch"]
    # Chroma mottle must not read as pore-texture change (TODO Z2 QA gate).
    assert abs(mottled["pore_energy"] - clean["pore_energy"]) < 0.35 * clean["pore_energy"]


def test_pore_floor_guard_flags_blurred_render():
    img, masks = _synthetic_face(noise=8.0)
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    src = spike.homogeneity_state(img, masks["skin"], face_width=160)
    ok = spike.homogeneity_state(img.copy(), masks["skin"], face_width=160)
    bad = spike.homogeneity_state(blurred, masks["skin"], face_width=160)
    assert spike.pore_floor_ok(src, ok) is True
    assert spike.pore_floor_ok(src, bad) is False


# --------------------------------------------------------------------------- #
# Suggestions: bounded, within-face, zero when already at floor
# --------------------------------------------------------------------------- #

def test_suggestions_zero_for_high_contrast_face_and_bounded_for_washed_out():
    strong_img, masks = _synthetic_face(lip_darken=70, eye_darken=95)
    weak_img, _ = _synthetic_face(lip_darken=8, eye_darken=95)

    strong = spike.suggest_strengths(
        spike.feature_contrast(strong_img, masks, 160),
        spike.homogeneity_state(strong_img, masks["skin"], 160))
    weak = spike.suggest_strengths(
        spike.feature_contrast(weak_img, masks, 160),
        spike.homogeneity_state(weak_img, masks["skin"], 160))

    assert "lip_enhance" not in strong and "eye_enhance" not in strong
    assert weak.get("lip_enhance", 0) > 0
    caps = spike.PROFILES["natural"]
    for key, cap in (("eye_enhance", caps["eye_cap"]),
                     ("lip_enhance", caps["lip_cap"]),
                     ("skin_chroma_even", caps["even_cap"])):
        assert 0 <= weak.get(key, 0) <= cap


def test_suggestions_trigger_evening_only_on_blotchy_skin():
    img, masks = _synthetic_face()
    blotchy = _add_chroma_blotch(img, amp=14.0)
    clean_s = spike.suggest_strengths(
        spike.feature_contrast(img, masks, 160),
        spike.homogeneity_state(img, masks["skin"], 160))
    blotchy_s = spike.suggest_strengths(
        spike.feature_contrast(blotchy, masks, 160),
        spike.homogeneity_state(blotchy, masks["skin"], 160))
    assert blotchy_s.get("skin_chroma_even", 0) > clean_s.get("skin_chroma_even", 0)

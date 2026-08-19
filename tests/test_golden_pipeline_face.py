"""Golden-output harness for the FACE path (per-face skin/eye/lip/hair ops).

test_golden_pipeline.py's synthetic image has no detectable face, so every
run there falls through to RetouchEngine._no_face_fallback and per-face
stages (skin tone unification, eye/lip/hair enhancement, BiSeNet-adjacent
region ops) never execute — that harness's "natural" snapshot is byte-
identical to its raw input. This harness supplies a real FaceContext (via
tests/golden_face_fixture.py — frozen anatomical landmarks, ONNX-free
landmark-only region masks) through RetouchEngine.process(face_contexts=...)
so the actual per-face pipeline runs and its output hash is what gets
locked in.

Uses its own snapshot file (golden_pipeline_face_snapshots.json) — separate
from golden_pipeline_snapshots.json, which is the P3 stage-registry byte-
identity guarantee and must not be touched by face-path changes.

Usage:
    python3 -m pytest tests/test_golden_pipeline_face.py -v
    python3 -m pytest tests/test_golden_pipeline_face.py --update-snapshot
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from tests.golden_face_fixture import make_face_context, make_synthetic_face_image


SNAPSHOT_PATH = Path(__file__).parent / "golden_pipeline_face_snapshots.json"

# Same 3-recipe sample as test_golden_pipeline.py, for the same reason
# (grain uses seed=None; full recipe coverage lives in test_recipes.py).
_SAMPLE_RECIPES = ["natural", "porcelain_unified_v1", "outdoor_harsh_sun_v1"]


def _hash_result(result: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(result).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine
    return RetouchEngine()


@pytest.fixture(scope="module")
def synthetic_face_img() -> np.ndarray:
    return make_synthetic_face_image()


@pytest.fixture(scope="module")
def face_context(synthetic_face_img):
    h, w = synthetic_face_img.shape[:2]
    return make_face_context(w, h, synthetic_face_img)


def _load_snapshots() -> Dict[str, str]:
    if SNAPSHOT_PATH.exists():
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    return {}


def _save_snapshots(snapshots: Dict[str, str]) -> None:
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshots, f, indent=2, sort_keys=True)


@pytest.mark.slow
@pytest.mark.parametrize("recipe_name", _SAMPLE_RECIPES)
def test_golden_face_output_stable(engine, synthetic_face_img, face_context, recipe_name, request):
    """Every sampled recipe's face-path output hash must match the golden
    snapshot. A failure here means a per-face stage's output changed —
    investigate before updating with --update-snapshot.
    """
    snapshots = _load_snapshots()

    result = engine.process(synthetic_face_img, recipe=recipe_name, face_contexts=[face_context])
    actual_hash = _hash_result(result)

    if request.config.getoption("--update-snapshot"):
        snapshots[recipe_name] = actual_hash
        _save_snapshots(snapshots)
        pytest.skip(f"Snapshot updated for {recipe_name}: {actual_hash}")

    if recipe_name not in snapshots:
        snapshots[recipe_name] = actual_hash
        _save_snapshots(snapshots)
        pytest.skip(f"New snapshot created for {recipe_name}: {actual_hash}")

    expected_hash = snapshots[recipe_name]
    assert actual_hash == expected_hash, (
        f"Golden face-path output changed for recipe '{recipe_name}': "
        f"expected {expected_hash}, got {actual_hash}. "
        f"If this is intentional, re-run with --update-snapshot."
    )


@pytest.mark.parametrize("recipe_name", _SAMPLE_RECIPES)
def test_golden_face_output_not_identity(engine, synthetic_face_img, face_context, recipe_name):
    """Guards against the exact bug this harness exists to catch: a
    recipe silently reducing to a no-op because the face path was bypassed
    (e.g. detection finding zero faces, as in test_golden_pipeline.py)."""
    result = engine.process(synthetic_face_img, recipe=recipe_name, face_contexts=[face_context])
    assert not np.array_equal(np.asarray(result), synthetic_face_img), (
        f"Recipe '{recipe_name}' produced output identical to input — "
        f"the face path may not be executing (bypassed detection/parsing?)."
    )


def test_face_context_regions_nondegenerate(face_context):
    """Sanity check on the fixture itself: key region masks must be
    non-empty and non-degenerate, or downstream per-face ops silently idle
    (e.g. unify_hue_line returns input unchanged when skin_mask is None)."""
    regions = face_context.regions
    for attr in ("face_oval", "skin", "left_eye", "right_eye", "lips"):
        mask = getattr(regions, attr)
        assert mask is not None, f"regions.{attr} is None"
        assert mask.max() > 0.5, f"regions.{attr} is degenerate (max={mask.max()})"
        assert mask.sum() > 10, f"regions.{attr} is near-empty (sum={mask.sum()})"


def test_synthetic_face_image_deterministic():
    """The synthetic face image must be byte-identical across runs."""
    img1 = make_synthetic_face_image()
    img2 = make_synthetic_face_image()
    assert np.array_equal(img1, img2), "Synthetic face image is not deterministic"

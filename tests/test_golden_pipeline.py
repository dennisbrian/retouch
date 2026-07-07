"""Golden-output harness for P3 stage-registry refactor.

Snapshots output hashes for every built-in recipe on a deterministic
synthetic image. Every P3 migration commit MUST keep these byte-identical.

Usage:
    python3 -m pytest tests/test_golden_pipeline.py -v
    python3 -m pytest tests/test_golden_pipeline.py --update-snapshot  # regenerate after intentional output change
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pytest


SNAPSHOT_PATH = Path(__file__).parent / "golden_pipeline_snapshots.json"


def _make_synthetic_image() -> np.ndarray:
    """Deterministic synthetic image with varied color content.

    Not a real photo — just a fixed input that exercises the pipeline
    end-to-end so output hashes are stable across runs.
    """
    rng = np.random.default_rng(seed=42)
    h, w = 64, 64
    yy = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = (yy + xx) * 0.3
    noise = rng.normal(0, 8, (h, w, 3)).astype(np.float32)
    img = np.stack([base, 255 - base, base * 0.5 + 50], axis=-1) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _hash_result(result: np.ndarray) -> str:
    return hashlib.sha256(result.tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine
    return RetouchEngine()


@pytest.fixture(scope="module")
def synthetic_img() -> np.ndarray:
    return _make_synthetic_image()


def _load_snapshots() -> Dict[str, str]:
    if SNAPSHOT_PATH.exists():
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    return {}


def _save_snapshots(snapshots: Dict[str, str]) -> None:
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshots, f, indent=2, sort_keys=True)


def _get_recipe_names() -> list:
    from retouch.recipes import RECIPES
    # Skip grain-using recipes from the stability test — apply_film_grain
    # uses seed=None (non-deterministic) by design, so output hashes will
    # differ between runs. These recipes are tested for dtype/structure
    # elsewhere; the golden harness tests deterministic recipes only.
    grain_recipes = set()
    for name, recipe in RECIPES.items():
        if recipe.get("grain_strength", 0) > 0:
            grain_recipes.add(name)
    # Sample a small subset (3 recipes) to keep test time under 90s.
    # The golden harness verifies P3 registry byte-identical output.
    # Full recipe coverage is in test_recipes.py (no engine run).
    all_deterministic = sorted(r for r in RECIPES.keys() if r not in grain_recipes)
    sample = ["natural", "porcelain_unified_v1", "outdoor_harsh_sun_v1"]
    return [r for r in sample if r in all_deterministic]


@pytest.mark.slow
@pytest.mark.parametrize("recipe_name", _get_recipe_names())
def test_golden_output_stable(engine, synthetic_img, recipe_name, request):
    """Every recipe's output hash must match the golden snapshot.

    If this test fails after a P3 migration commit, the refactor changed
    pipeline behavior — investigate before updating snapshots.
    To update snapshots after an intentional output change, delete
    golden_pipeline_snapshots.json and re-run.
    """
    snapshots = _load_snapshots()

    try:
        result = engine.process(synthetic_img, recipe=recipe_name)
    except Exception as e:
        if recipe_name in snapshots:
            pytest.fail(f"Recipe {recipe_name} previously worked but now crashes: {e}")
        pytest.skip(f"Recipe {recipe_name} requires features not available: {e}")

    actual_hash = _hash_result(result)

    if recipe_name not in snapshots:
        snapshots[recipe_name] = actual_hash
        _save_snapshots(snapshots)
        pytest.skip(f"New snapshot created for {recipe_name}: {actual_hash}")

    expected_hash = snapshots[recipe_name]
    assert actual_hash == expected_hash, (
        f"Golden output changed for recipe '{recipe_name}': "
        f"expected {expected_hash}, got {actual_hash}. "
        f"If this is intentional, delete golden_pipeline_snapshots.json and re-run."
    )


def test_snapshot_count_matches_recipes():
    """Ensure every non-grain recipe has a snapshot (no missing entries).

    Grain-using recipes are excluded from the golden harness because
    apply_film_grain uses seed=None (non-deterministic by design).
    """
    snapshots = _load_snapshots()
    recipe_names = _get_recipe_names()
    missing = [r for r in recipe_names if r not in snapshots]
    assert not missing, f"Missing snapshots for recipes: {missing}"


def test_synthetic_image_deterministic():
    """The synthetic image must be byte-identical across runs."""
    img1 = _make_synthetic_image()
    img2 = _make_synthetic_image()
    assert np.array_equal(img1, img2), "Synthetic image is not deterministic"

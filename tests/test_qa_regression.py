"""Regression tests: run all built-in recipes, assert no CRITICAL QA warnings."""
from __future__ import annotations

import numpy as np
import pytest

from retouch.engine import RetouchEngine
from retouch.recipes import RECIPES
from retouch.qa_detectors import QAWarning


@pytest.fixture(scope="module")
def engine():
    eng = RetouchEngine()
    yield eng
    eng.close()


def _make_test_image(h=256, w=256):
    np.random.seed(42)
    img = np.random.randint(30, 220, (h, w, 3), dtype=np.uint8)
    return img


RECIPE_NAMES = [name for name in RECIPES if name != "__builtins__"]


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_recipe_produces_no_critical_qa(engine, recipe_name):
    img = _make_test_image()
    result = engine.process(img, recipe=recipe_name)
    qa = getattr(result, "qa", [])
    critical = [w for w in qa if w.score > 0.8]
    assert len(critical) == 0, (
        f"Recipe '{recipe_name}' produced {len(critical)} critical QA warnings: "
        + "; ".join(f"{w.detector}={w.score:.2f}" for w in critical)
    )


def test_all_recipes_run_without_crash(engine):
    for name in RECIPE_NAMES:
        img = _make_test_image()
        result = engine.process(img, recipe=name)
        assert result is not None
        assert result.shape == (256, 256, 3)

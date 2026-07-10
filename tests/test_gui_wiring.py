"""GUI wiring tests for the previously "unwired" features.

Covers:
  * F6 Look Extractor -> hidden ``look_params`` State (with ``_``-key stripping)
  * T4 Recipe Cookbook search -> dropdown population
  * LUT hot-reload caller
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

import gui
from retouch.recipe_cookbook import search_recipes


def test_look_extractor_strips_underscore_keys(tmp_path):
    """on_extract_look returns engine_params with ``_``-prefixed keys removed."""
    assert hasattr(gui, "on_extract_look")

    ref = np.zeros((64, 64, 3), dtype=np.uint8)
    ref[:, :, :] = (120, 90, 200)
    p = tmp_path / "ref.png"
    cv2.imwrite(str(p), ref)

    with mock.patch.object(gui.LookExtractor, "extract") as fake_extract:
        fake_extract.return_value = {
            "engine_params": {
                "shadows": 5.0,
                "saturation": 12.0,
                "_split_tone_three_way": {"shadows": {"hue": 0, "sat": 0}},
                "_internal": "drop-me",
            },
            "preset": {},
            "mode": "unpaired",
        }
        clean, status = gui.on_extract_look(str(p), None)

    assert isinstance(clean, dict)
    assert "_split_tone_three_way" not in clean
    assert "_internal" not in clean
    assert clean.get("shadows") == 5.0
    assert "extracted" in status.lower()


def test_look_extractor_real_image(tmp_path):
    """End-to-end-ish: extract from a real written image, no crash, no ``_`` keys."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    p = tmp_path / "ref.png"
    cv2.imwrite(str(p), img)

    clean, status = gui.on_extract_look(str(p), None)
    assert isinstance(clean, dict)
    assert all(not str(k).startswith("_") for k in clean)
    assert "failed" not in status.lower()


def test_cookbook_search_returns_results():
    assert hasattr(gui, "on_search_recipes")
    assert hasattr(gui, "on_select_cookbook")

    results = search_recipes("portrait")
    assert len(results) > 0

    update, msg = gui.on_search_recipes("portrait")
    assert update is not None
    assert len(update.get("choices", [])) > 0


def test_lut_reload_callable():
    assert hasattr(gui, "on_reload_luts")
    status = gui.on_reload_luts()
    assert isinstance(status, str)

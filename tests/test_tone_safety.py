"""P0 tone-safety contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from retouch.color_science import ToneObservation, measure_tone_observation
from retouch.engine import RetouchEngine


def test_tone_observation_is_continuous_and_has_no_demographic_label():
    image = np.full((48, 48, 3), (132, 112, 96), dtype=np.uint8)

    observation = measure_tone_observation(image)

    assert isinstance(observation, ToneObservation)
    assert 0.0 <= observation.confidence <= 1.0
    assert 0.0 <= observation.uncertainty <= 1.0
    assert observation.valid_fraction == pytest.approx(1.0)
    payload = observation.to_dict()
    assert "type_index" not in payload
    assert "label" not in payload
    assert payload["L_mean"] != payload["ita_mean"]


def test_tone_observation_reports_highlights_as_uncertainty():
    image = np.full((48, 48, 3), (132, 112, 96), dtype=np.uint8)
    image[:24, :24] = 255

    observation = measure_tone_observation(image)

    assert observation.highlight_fraction > 0.20
    assert observation.uncertainty > 0.0
    assert observation.valid_fraction < 1.0


def test_tone_observation_accepts_float_image_and_mask():
    image = np.full((32, 32, 3), (0.52, 0.44, 0.38), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    observation = measure_tone_observation(image, mask)

    assert observation.valid_fraction == pytest.approx(1.0)
    assert observation.confidence > 0.9


def test_tone_observation_rejects_mismatched_mask():
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask must match"):
        measure_tone_observation(image, np.ones((8, 8), dtype=np.float32))


def test_face_suggestion_is_neutral_and_exposes_only_diagnostic_tone_data():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    face = SimpleNamespace(bbox=(20, 20, 60, 60))
    engine = RetouchEngine.__new__(RetouchEngine)

    result = engine.suggest_face_params(image, faces_data=[face])

    assert result[0]["recipe"] == "natural"
    assert result[0]["selection_reason"] == "neutral default; demographic inference disabled"
    assert "tone_observation" in result[0]
    assert "label" not in result[0]["tone_observation"]

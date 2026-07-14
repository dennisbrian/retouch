"""Real-model regression/plumbing test for retouch/parsing.py (BiSeNet ONNX).

Guards against a SILENT class-map / model-export break: the 19-class layout is
bespoke and unverified (docs/audit). If the export renames a class or the input
name changes, these tests fail instead of producing corrupt masks quietly.

Skipped automatically when the ONNX model is absent (e.g. CI without models).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from retouch.detection import _Landmark, _LandmarkCompat
from retouch.parsing import FaceParser, FaceRegions

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "retouch", "..", "models", "resnet18.onnx"
)
_HAS_MODEL = os.path.exists(_MODEL_PATH)

skip_no_model = pytest.mark.skipif(
    not _HAS_MODEL, reason="BiSeNet ONNX model (models/resnet18.onnx) not present"
)


def _dummy_landmarks(num: int = 478) -> _LandmarkCompat:
    rng = np.random.default_rng(0)
    pts = [_Landmark(0.5, 0.5, 0.0) for _ in range(num)]
    for i in range(num):
        pts[i] = _Landmark(
            float(rng.uniform(0.1, 0.9)),
            float(rng.uniform(0.1, 0.9)),
            0.0,
        )
    return _LandmarkCompat(pts)


def _face_like_img(h: int = 400, w: int = 400) -> np.ndarray:
    """Crude synthetic face so BiSeNet has something to parse."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)  # skin-ish gray
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.5, w * 0.5
    face = ((xx - cx) / (w * 0.35)) ** 2 + ((yy - cy) / (h * 0.4)) ** 2 <= 1.0
    img[face] = (200, 175, 160)  # skin tone
    # hair on top
    img[(yy < h * 0.32) & face] = (60, 40, 35)
    # eyes
    for ex in (w * 0.38, w * 0.62):
        ey, ex = int(h * 0.45), int(ex)
        img[ey - 6 : ey + 6, ex - 10 : ex + 10] = (40, 30, 25)
    # lips
    img[int(h * 0.62) - 6 : int(h * 0.62) + 6, int(w * 0.5) - 18 : int(w * 0.5) + 18] = (160, 40, 50)
    return img


@skip_no_model
def test_model_output_has_19_classes() -> None:
    parser = FaceParser()
    assert parser._sess is not None, "expected a loaded ONNX session"
    x = np.random.rand(1, 3, 512, 512).astype(np.float32)
    outs = parser._sess.run(None, {"input": x})
    # The main head is outs[0]; it must carry 19 semantic classes.
    assert outs[0].shape[-3] == 19


@skip_no_model
@pytest.mark.parametrize("attr", ["skin", "lips", "hair", "left_eye", "right_eye", "face_oval"])
def test_parse_populates_core_regions(attr: str) -> None:
    parser = FaceParser()
    img = _face_like_img()
    bbox = (40, 40, 320, 360)
    regions = parser.parse(_dummy_landmarks(), img, bbox)
    assert isinstance(regions, FaceRegions)
    val = getattr(regions, attr)
    # Region must exist (attribute present) and, if produced, be float32 [0,1].
    assert val is not None, f"region '{attr}' was not produced"
    assert val.dtype == np.float32
    assert val.min() >= 0.0 and val.max() <= 1.0

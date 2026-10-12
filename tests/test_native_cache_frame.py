"""Cached FaceContexts on the F8.2 native path must be converted from their
declared pixel frame exactly once.

Regression: ``_process_native_faces`` returned contexts whose boxes were
already native, then treated them as proxy boxes on the next (GUI cache-hit)
render and scaled them by ``1/proxy_scale`` again. On a real 6240x4160
portrait the warm-cache face box moved from x=2611 to x=7955 (off-image) and
the face edits changed. See
docs/plans/RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23.md §3.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

W, H = 4096, 3072  # > PROXY_MAX_DIM → native path, proxy_scale = 0.5


class _StopAtReshape(Exception):
    pass


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    eng = RetouchEngine()
    yield eng
    eng.close()


def _faces_reaching_reshape(engine, monkeypatch, img, contexts):
    captured = {}

    def _spy(img_bgr, faces, ctx):
        captured["faces"] = list(faces)
        raise _StopAtReshape

    monkeypatch.setattr(engine, "_stage_reshape", _spy)
    monkeypatch.setattr(engine._detector, "segment_person", lambda _img: None)
    with pytest.raises(_StopAtReshape):
        engine.process(
            img, recipe="natural", auto_exposure=False, quality="full",
            face_contexts=contexts,
        )
    return captured["faces"]


@pytest.fixture(scope="module")
def image():
    return make_synthetic_face_image(W, H)


def test_native_frame_context_is_not_rescaled(engine, monkeypatch, image):
    fc = dataclasses.replace(make_face_context(W, H, image), frame_size=(W, H))
    faces = _faces_reaching_reshape(engine, monkeypatch, image, [fc])
    assert faces[0].bbox == fc.face_data.bbox
    assert faces[0].ied == pytest.approx(fc.face_data.ied)


def test_proxy_frame_context_is_scaled_once(engine, monkeypatch, image):
    native = make_face_context(W, H, image)
    x, y, w, h = native.face_data.bbox
    proxy_face = dataclasses.replace(
        native.face_data, bbox=(x // 2, y // 2, w // 2, h // 2),
        ied=native.face_data.ied / 2,
    )
    fc = dataclasses.replace(native, face_data=proxy_face, frame_size=(W // 2, H // 2))
    faces = _faces_reaching_reshape(engine, monkeypatch, image, [fc])
    assert faces[0].bbox == (x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2)


def test_legacy_context_without_frame_keeps_proxy_assumption(engine, monkeypatch, image):
    native = make_face_context(W, H, image)
    x, y, w, h = native.face_data.bbox
    proxy_face = dataclasses.replace(native.face_data, bbox=(x // 2, y // 2, w // 2, h // 2))
    fc = dataclasses.replace(native, face_data=proxy_face)  # frame_size=None
    faces = _faces_reaching_reshape(engine, monkeypatch, image, [fc])
    assert faces[0].bbox == (x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2)

"""Tests for ``RetouchEngine.process(progress_cb=...)`` live progress events.

Uses the frozen real-landmark face fixture (tests/golden_face_fixture.py) via
``face_contexts=`` so the real per-face path runs without any detector model.
Detection is mocked for the no-face / proxy cases.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest

from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

Event = Tuple[str, Dict[str, Any]]


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    with RetouchEngine() as eng:
        yield eng


@pytest.fixture(scope="module")
def face_img() -> np.ndarray:
    return make_synthetic_face_image()


@pytest.fixture(scope="module")
def face_ctx(face_img):
    h, w = face_img.shape[:2]
    return make_face_context(w, h, face_img)


@pytest.fixture
def no_detect(engine, monkeypatch):
    """Force zero detected faces (and no person mask) without any model."""
    monkeypatch.setattr(engine._detector, "detect", lambda img: [])
    monkeypatch.setattr(engine._detector, "segment_person", lambda img: None)


def _recorder() -> Tuple[List[Event], Any]:
    events: List[Event] = []

    def cb(kind: str, payload: dict) -> None:
        events.append((kind, dict(payload)))

    return events, cb


def _stages(events: List[Event]) -> List[str]:
    return [p["stage"] for k, p in events if k == "stage"]


def _face_indices(events: List[Event]) -> List[Tuple[int, int]]:
    return [(p["index"], p["total"]) for k, p in events if k == "face"]


def _assert_face_path_order(engine, events: List[Event], n_faces: int) -> None:
    stages = _stages(events)
    assert stages[:3] == ["detection", "reshape", "per_face"], stages

    # Registry stages are reported in registry order (only those that ran).
    registry_order = engine._global_registry.names()
    reg_seen = [s for s in stages if s in registry_order]
    assert reg_seen, "no global-registry stage events"
    assert reg_seen == sorted(reg_seen, key=registry_order.index)
    assert stages.index("qa") > stages.index(reg_seen[-1])

    # Face events: after per_face starts, before the first global stage, 1..n.
    kinds = [k for k, _ in events]
    per_face_pos = events.index(("stage", {"stage": "per_face"}))
    first_global_pos = events.index(("stage", {"stage": reg_seen[0]}))
    face_pos = [i for i, k in enumerate(kinds) if k == "face"]
    assert face_pos, "no face events"
    assert all(per_face_pos < i < first_global_pos for i in face_pos)
    assert _face_indices(events) == [(i, n_faces) for i in range(1, n_faces + 1)]


# ---------------------------------------------------------------------------
# (1) event order on the real single-face path
# ---------------------------------------------------------------------------

def test_stage_and_face_events_in_pipeline_order(engine, face_img, face_ctx):
    events, cb = _recorder()
    engine.process(face_img, recipe="natural", face_contexts=[face_ctx], progress_cb=cb)
    _assert_face_path_order(engine, events, n_faces=1)
    assert all(k in ("stage", "face") for k, _ in events)


# ---------------------------------------------------------------------------
# (2) byte-identical output with and without a callback
# ---------------------------------------------------------------------------

def test_output_byte_identical_with_and_without_callback(engine, face_img, face_ctx):
    base_a = np.asarray(engine.process(face_img, recipe="natural", face_contexts=[face_ctx]))
    base_b = np.asarray(engine.process(face_img, recipe="natural", face_contexts=[face_ctx]))
    if not np.array_equal(base_a, base_b):  # pragma: no cover - guard only
        pytest.skip("'natural' is not deterministic run-to-run on this platform")
    _, cb = _recorder()
    with_cb = engine.process(
        face_img, recipe="natural", face_contexts=[face_ctx], progress_cb=cb,
    )
    assert np.array_equal(np.asarray(with_cb), base_a)


def test_progress_cb_is_not_a_processing_param(engine, face_img, face_ctx):
    _, cb = _recorder()
    result = engine.process(
        face_img, recipe="natural", face_contexts=[face_ctx], progress_cb=cb,
    )
    assert not hasattr(result.params, "progress_cb")
    assert "progress_cb" not in vars(result.params)
    assert not any(
        "progress" in str(k) for k in vars(result.params)
    ), "no progress state may leak into ProcessingContext"
    from retouch.params import PROCESSING_PARAMS

    assert "progress_cb" not in {p.name for p in PROCESSING_PARAMS}


# ---------------------------------------------------------------------------
# (3) a raising / bogus callback never breaks processing
# ---------------------------------------------------------------------------

def test_raising_callback_does_not_break_processing(engine, face_img, face_ctx, caplog):
    calls = []

    def bad_cb(kind, payload):
        calls.append(kind)
        raise RuntimeError("boom")

    base = np.asarray(engine.process(face_img, recipe="natural", face_contexts=[face_ctx]))
    with caplog.at_level(logging.WARNING, logger="retouch.engine"):
        out = engine.process(
            face_img, recipe="natural", face_contexts=[face_ctx], progress_cb=bad_cb,
        )
    assert np.array_equal(np.asarray(out), base)
    assert "face" in calls and "stage" in calls  # kept being called after raising
    assert any("progress_cb raised" in r.getMessage() for r in caplog.records)


def test_non_callable_progress_cb_is_ignored(engine, face_img, face_ctx):
    out = engine.process(
        face_img, recipe="natural", face_contexts=[face_ctx], progress_cb="not-callable",
    )
    assert np.asarray(out).shape == face_img.shape


def test_callback_scope_resets_even_when_process_raises(engine):
    from retouch.engine import _PROGRESS_CB

    events, cb = _recorder()
    with pytest.raises(ValueError):
        engine.process(np.zeros((0, 0, 3), np.uint8), progress_cb=cb)
    assert _PROGRESS_CB.get() is None
    assert events == []


# ---------------------------------------------------------------------------
# (4) no-face path still reports stages
# ---------------------------------------------------------------------------

def test_no_face_path_emits_stage_events(engine, face_img, no_detect):
    events, cb = _recorder()
    engine.process(face_img, recipe="natural", progress_cb=cb)
    assert _stages(events) == ["detection", "no_face_fallback"]
    assert _face_indices(events) == []


# ---------------------------------------------------------------------------
# Multi-face dispatch paths: pool (with fallback re-run) and ThreadPool
# ---------------------------------------------------------------------------

def test_threadpool_fallback_emits_face_events(engine, face_img, face_ctx, monkeypatch):
    def pool_down(payloads, **kwargs):
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(engine._face_pool, "process_faces", pool_down)
    h, w = face_img.shape[:2]
    ctx2 = make_face_context(w, h, face_img, index=1)
    events, cb = _recorder()
    engine.process(face_img, recipe="natural", face_contexts=[face_ctx, ctx2], progress_cb=cb)
    _assert_face_path_order(engine, events, n_faces=2)


def test_pool_path_counts_each_face_once(engine, face_img, face_ctx, monkeypatch):
    """Pool reports both faces done, then returns None entries so the engine
    re-runs them in-process: each face must still be counted exactly once."""
    seen_kwargs = {}

    def fake_pool(payloads, on_face_done=None):
        seen_kwargs["on_face_done"] = on_face_done
        for i in range(len(payloads)):
            on_face_done(i)
        return [None] * len(payloads)

    monkeypatch.setattr(engine._face_pool, "process_faces", fake_pool)
    h, w = face_img.shape[:2]
    ctx2 = make_face_context(w, h, face_img, index=1)
    events, cb = _recorder()
    engine.process(face_img, recipe="natural", face_contexts=[face_ctx, ctx2], progress_cb=cb)
    assert callable(seen_kwargs["on_face_done"])
    _assert_face_path_order(engine, events, n_faces=2)


def test_pool_not_given_hook_without_callback(engine, face_img, face_ctx, monkeypatch):
    """No callback → pool is called exactly as before (no extra kwarg)."""
    calls = []

    def strict_pool(payloads):
        calls.append(len(payloads))
        return None  # → ThreadPool fallback

    monkeypatch.setattr(engine._face_pool, "process_faces", strict_pool)
    h, w = face_img.shape[:2]
    ctx2 = make_face_context(w, h, face_img, index=1)
    engine.process(face_img, recipe="natural", face_contexts=[face_ctx, ctx2])
    assert calls == [2]


def test_real_face_pool_emits_live_face_events(face_img, face_ctx):
    """Real FaceProcessorPool worker processes; callback stays in the parent."""
    from retouch.perf_optimizations import FaceProcessorPool

    pool = FaceProcessorPool(max_workers=2)
    done: List[int] = []
    h, w = face_img.shape[:2]
    try:
        from retouch.engine import RetouchEngine

        with RetouchEngine() as eng:
            eng._face_pool = pool
            ctx2 = make_face_context(w, h, face_img, index=1)
            events, cb = _recorder()
            orig = pool.process_faces

            def spy(payloads, on_face_done=None):
                def hook(i):
                    done.append(i)
                    on_face_done(i)
                return orig(payloads, on_face_done=hook)

            pool.process_faces = spy
            eng.process(face_img, recipe="natural", face_contexts=[face_ctx, ctx2], progress_cb=cb)
    finally:
        pool.shutdown()
    assert sorted(done) == [0, 1]
    _assert_face_path_order(eng, events, n_faces=2)


# ---------------------------------------------------------------------------
# Proxy paths (> PROXY_MAX_DIM)
# ---------------------------------------------------------------------------

def _big_image() -> np.ndarray:
    # 5:6 like the fixture; long side 2400 > PROXY_MAX_DIM (2048).
    return make_synthetic_face_image(w=2000, h=2400)


def test_proxy_full_no_face_emits_stages(engine, no_detect):
    events, cb = _recorder()
    engine.process(_big_image(), recipe="natural", quality="full", progress_cb=cb)
    assert _stages(events) == ["detection", "no_face_fallback"]


def test_proxy_draft_no_face_emits_stages(engine, no_detect):
    events, cb = _recorder()
    engine.process(_big_image(), recipe="natural", quality="draft", progress_cb=cb)
    assert _stages(events) == ["detection", "proxy_upscale", "no_face_fallback"]


def test_proxy_draft_with_face_emits_stages_and_faces(engine):
    from retouch.engine import PROXY_MAX_DIM

    big = _big_image()
    h, w = big.shape[:2]
    s = PROXY_MAX_DIM / float(max(h, w))
    pw, ph = int(w * s), int(h * s)
    proxy = cv2.resize(big, (pw, ph), interpolation=cv2.INTER_AREA)
    ctx_proxy = make_face_context(pw, ph, proxy)

    events, cb = _recorder()
    out = engine.process(
        big, recipe="natural", quality="draft", face_contexts=[ctx_proxy], progress_cb=cb,
    )
    assert np.asarray(out).shape == big.shape
    stages = _stages(events)
    assert stages[:4] == ["detection", "reshape", "per_face", "proxy_upscale"], stages
    assert _face_indices(events) == [(1, 1)]
    assert stages[-1] == "qa"

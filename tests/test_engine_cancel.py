"""Tests for cooperative cancellation in ``RetouchEngine.process``.

Mirrors retouch/video/export.py's semantics: a cancel raises
``RetouchCancelled`` (a requested stop, not a failure), the exception
propagates to the caller, and no output is produced — the engine returns an
in-memory result, so cancelling before the return means the caller never has
a half-written file to publish. Uses the frozen real-landmark fixture like
tests/test_engine_progress.py so the real per-face path runs model-free.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from tests.golden_face_fixture import make_face_context, make_synthetic_face_image


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


def _run(engine, face_img, face_ctx, **cancel_kwargs):
    return engine.process(
        face_img, recipe="natural", face_contexts=[face_ctx], **cancel_kwargs
    )


# ---------------------------------------------------------------------------
# cancel via callable (video-module API shape: cancelled=lambda: ...)
# ---------------------------------------------------------------------------

def test_precancelled_callable_raises_before_work(engine, face_img, face_ctx):
    from retouch.engine import RetouchCancelled

    with pytest.raises(RetouchCancelled):
        _run(engine, face_img, face_ctx, cancelled=lambda: True)


def test_cancel_during_pipeline_raises(engine, face_img, face_ctx):
    """Cancel set from inside the progress callback (i.e. mid-pipeline) must
    raise out of process() at the next stage boundary — NOT be swallowed by
    the swallow-by-design progress handler."""
    from retouch.engine import RetouchCancelled

    event = threading.Event()
    seen: list = []

    def cb(kind, payload):
        seen.append(kind)
        event.set()  # "user clicked cancel" once processing starts

    with pytest.raises(RetouchCancelled):
        _run(engine, face_img, face_ctx, cancelled=event.is_set, progress_cb=cb)
    assert seen, "cancel fired before any progress event; not a mid-pipeline test"


# ---------------------------------------------------------------------------
# cancel_event and cancel_file options
# ---------------------------------------------------------------------------

def test_precancelled_event_raises(engine, face_img, face_ctx):
    from retouch.engine import RetouchCancelled

    event = threading.Event()
    event.set()
    with pytest.raises(RetouchCancelled):
        _run(engine, face_img, face_ctx, cancel_event=event)


def test_precancelled_file_raises(engine, face_img, face_ctx, tmp_path: Path):
    from retouch.engine import RetouchCancelled

    cancel_file = tmp_path / "stop"
    cancel_file.touch()
    with pytest.raises(RetouchCancelled):
        _run(engine, face_img, face_ctx, cancel_file=cancel_file)


def test_missing_cancel_file_does_not_cancel(engine, face_img, face_ctx, tmp_path: Path):
    out = _run(
        engine, face_img, face_ctx, cancel_file=tmp_path / "never-created"
    )
    assert np.asarray(out).shape == face_img.shape


# ---------------------------------------------------------------------------
# no partial output + scope hygiene + happy path
# ---------------------------------------------------------------------------

def test_cancelled_run_writes_no_output(engine, face_img, face_ctx, tmp_path: Path):
    """The engine's contract: cancel before the result is returned and the
    caller has nothing to write. A cancelled run must not reach the return —
    so a caller-side write guarded on success never executes."""
    from retouch.engine import RetouchCancelled

    out_path = tmp_path / "out.png"
    try:
        _run(engine, face_img, face_ctx, cancelled=lambda: True)
    except RetouchCancelled:
        pass  # caller skips the write; no partial file exists
    else:  # pragma: no cover - cancel must raise
        pytest.fail("cancelled run returned a result")
    assert not out_path.exists()  # nothing was published


def test_cancel_scope_resets_after_raise(engine, face_img, face_ctx):
    from retouch.engine import _CANCEL_CHECK, RetouchCancelled

    with pytest.raises(RetouchCancelled):
        _run(engine, face_img, face_ctx, cancelled=lambda: True)
    assert _CANCEL_CHECK.get() is None
    # next run on the same engine is unaffected
    out = _run(engine, face_img, face_ctx)
    assert np.asarray(out).shape == face_img.shape


def test_happy_path_unaffected_by_cancel_options(engine, face_img, face_ctx):
    base = np.asarray(_run(engine, face_img, face_ctx))
    if not np.array_equal(
        base, np.asarray(_run(engine, face_img, face_ctx))
    ):  # pragma: no cover - guard only
        pytest.skip("'natural' is not deterministic run-to-run on this platform")
    out = _run(
        engine, face_img, face_ctx,
        cancelled=lambda: False,
        cancel_event=threading.Event(),  # unset
    )
    assert np.array_equal(np.asarray(out), base)


def test_raising_callback_still_does_not_cancel(engine, face_img, face_ctx):
    """Swallow-by-design is preserved: only the cancel check escalates, a
    raising progress callback still never breaks processing."""
    from retouch.engine import RetouchCancelled

    def bad_cb(kind, payload):
        raise RuntimeError("boom")

    out = _run(engine, face_img, face_ctx, progress_cb=bad_cb)
    assert np.asarray(out).shape == face_img.shape

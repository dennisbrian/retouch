"""Parity between inline per-face processors and FaceProcessorPool worker processors.

Known gap (documented, not fixable from engine.py):
    retouch/perf_optimizations.py:_get_worker_processors (lines ~1595-1621)
    builds the worker-side processor dict WITHOUT a 'frequency' entry, while
    the inline path (engine.py:~4133-4145) passes self._frequency. The
    worker dict is assembled entirely inside that perf_optimizations function
    and _process_single_face_worker calls it with no arguments, so there is
    no engine-side hook to inject the engine's FrequencySeparator without
    editing perf_optimizations.py (excluded from this change wave).

    Behavioral impact is nominal: _process_face_core line ~320 has a fallback
    ``processors.get('frequency') or FrequencySeparator()`` and
    FrequencySeparator is stateless, so the worker creates an equivalent
    transient instance per call. The xfail below documents the key-set gap
    so it flips to XPASS (strict) the moment the worker dict is fixed.
"""

import pickle

import numpy as np
import pytest

from retouch.engine import ProcessingContext, RetouchEngine, _worker_payload_ctx
from retouch.perf_optimizations import _get_worker_processors

INLINE_PROCESSOR_KEYS = {
    'skin', 'relighter', 'blemish', 'undereye', 'eyes', 'teeth', 'lips',
    'makeup', 'makeup_v2', 'hair', 'frequency',
}


class TestProcessorKeyParity:
    def test_inline_key_set_matches_source_of_truth(self):
        """Guard the expected inline key set against engine drift."""
        eng = RetouchEngine.__new__(RetouchEngine)  # no model loading
        from retouch.blemish import BlemishRemover
        from retouch.eyes import EyeEnhancer
        from retouch.hair import HairEnhancer
        from retouch.lips import LipEnhancer
        from retouch.makeup import MakeupEngine
        from retouch.makeup_v2 import MakeupEngineV2
        from retouch.relight import Relighter
        from retouch.skin import SkinProcessor
        from retouch.teeth import TeethWhitener
        from retouch.undereye import UnderEyeRepairer
        from retouch.frequency import FrequencySeparator
        eng._skin = SkinProcessor()
        eng._relighter = Relighter()
        eng._blemish = BlemishRemover()
        eng._undereye = UnderEyeRepairer()
        eng._eyes = EyeEnhancer()
        eng._teeth = TeethWhitener()
        eng._lips = LipEnhancer()
        eng._makeup = MakeupEngine()
        eng._makeup_v2 = MakeupEngineV2()
        eng._hair = HairEnhancer()
        eng._frequency = FrequencySeparator()
        inline = {
            'skin': eng._skin,
            'relighter': eng._relighter,
            'blemish': eng._blemish,
            'undereye': eng._undereye,
            'eyes': eng._eyes,
            'teeth': eng._teeth,
            'lips': eng._lips,
            'makeup': eng._makeup,
            'makeup_v2': eng._makeup_v2,
            'hair': eng._hair,
            'frequency': eng._frequency,
        }
        assert set(inline) == INLINE_PROCESSOR_KEYS

    def test_worker_processors_exist_for_common_keys(self):
        """Keys both paths must share are present in the worker dict."""
        worker = _get_worker_processors()
        assert INLINE_PROCESSOR_KEYS - {"frequency"} <= set(worker)

    def test_worker_dict_parity_with_inline(self):
        """Worker dict must expose the same key set as the inline dict.

        Fixed: perf_optimizations._get_worker_processors now constructs a
        shared stateless FrequencySeparator for the 'frequency' key, matching
        the inline dict in engine.py.
        """
        worker = _get_worker_processors()
        assert set(worker) == INLINE_PROCESSOR_KEYS


class TestWorkerCtxRoundTrip:
    def test_ndarray_fields_survive_payload_ctx(self):
        """ndarray ctx fields are picklable and must reach the worker."""
        ctx = ProcessingContext()
        ctx.freckle_preserve_mask = (np.arange(12, dtype=np.uint8).reshape(3, 4))
        ctx._protect_roi = np.full((3, 4), 0.5, dtype=np.float32)
        ctx.color_ref = np.zeros((2, 2, 3), dtype=np.float32)
        ctx.hi_ref = np.full((2, 2, 3), 1.5, dtype=np.float32)
        ctx.cross_region_skin_mask = np.ones((3, 3), dtype=np.uint8)
        ctx.cross_region_protect_mask = np.zeros((3, 3), dtype=np.uint8)
        ctx._effect_previews = {"hair_remove_flyaways": {"bytes": b"x" * 100}}

        payload = _worker_payload_ctx(ctx)
        assert payload is not ctx
        # Original untouched (worker payload is a private copy).
        assert "_effect_previews" in ctx.__dict__

        # _effect_previews dropped: bulky preview buffers, worker rebuilds
        # its own preview namespace.
        assert "_effect_previews" not in payload.__dict__

        # ndarray fields kept, with values intact. The user protection mask
        # travels as the face's ROI crop; the full-frame source arrays stay
        # in the parent (the core reads only _protect_roi).
        np.testing.assert_array_equal(payload._protect_roi, ctx._protect_roi)
        assert payload.freckle_preserve_mask is None
        assert ctx.freckle_preserve_mask is not None
        np.testing.assert_array_equal(payload.color_ref, ctx.color_ref)
        np.testing.assert_array_equal(payload.hi_ref, ctx.hi_ref)
        np.testing.assert_array_equal(payload.cross_region_skin_mask, ctx.cross_region_skin_mask)
        np.testing.assert_array_equal(payload.cross_region_protect_mask, ctx.cross_region_protect_mask)

    def test_payload_ctx_pickles_through_spawn_protocol(self):
        """The pool uses spawn+pickle; the payload ctx must survive it."""
        ctx = ProcessingContext()
        ctx._protect_roi = np.arange(6, dtype=np.float32).reshape(2, 3) / 6.0
        ctx.smooth = 42.0
        ctx.active_recipe = "cosplay"

        payload = _worker_payload_ctx(ctx)
        rt = pickle.loads(pickle.dumps(payload))

        assert rt.smooth == 42.0
        assert rt.active_recipe == "cosplay"
        np.testing.assert_array_equal(rt._protect_roi, ctx._protect_roi)
        # Scalar/str/dict field types survive the round trip.
        assert isinstance(rt.smooth, float)
        assert isinstance(rt.active_recipe, str)
        assert isinstance(rt._safe_auto_decisions, list)

    def test_identity_fields_preserved(self):
        """Common scalar field types survive _worker_payload_ctx untouched."""
        ctx = ProcessingContext()
        ctx.smooth = 0.35
        ctx.eye_gate = False
        ctx.skin_locus = {"a": 1.0}
        ctx.face_params = {0: {"smooth": 0.1}}

        payload = _worker_payload_ctx(ctx)
        assert payload.smooth == 0.35
        assert payload.eye_gate is False
        assert payload.skin_locus == {"a": 1.0}
        assert payload.face_params == {0: {"smooth": 0.1}}

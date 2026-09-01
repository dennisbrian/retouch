"""Float32 masks that overshoot 1.0 by an epsilon (1.0000002 after feathering /
resampling; observed on 20/83 DSCF corpus images, 2026-09-02) must NOT be
mistaken for 0-255 masks. With the old `max() > 1.0` test they were divided by
255, the composite alpha collapsed to 1/255 and every skin-region op on that
face was silently discarded (lips/sharpen survived, so it looked like
"some ops no-op on some images")."""
from __future__ import annotations

import numpy as np
import pytest

EPS_OVER_ONE = np.float32(1.0) + np.float32(2.4e-7)  # 1.0000002


@pytest.mark.parametrize("fn_path", [
    "retouch.perf_optimizations._norm_mask",
    "retouch.utils.normalize_mask",
    "retouch.makeup_unmix._prep_mask",
])
def test_epsilon_over_one_is_not_treated_as_0_255(fn_path):
    mod, _, name = fn_path.rpartition(".")
    fn = getattr(__import__(mod, fromlist=[name]), name, None)
    if fn is None:
        pytest.skip(f"{fn_path} not present")
    m = np.full((8, 8), EPS_OVER_ONE, dtype=np.float32)
    out = fn(m, 8, 8) if name == "_prep_mask" else fn(m)
    assert out.max() >= 0.999, f"{fn_path} collapsed an epsilon-over-1 mask to {out.max()}"
    assert out.max() <= 1.0


@pytest.mark.parametrize("fn_path", [
    "retouch.perf_optimizations._norm_mask",
    "retouch.utils.normalize_mask",
])
def test_uint8_masks_still_normalised(fn_path):
    mod, _, name = fn_path.rpartition(".")
    fn = getattr(__import__(mod, fromlist=[name]), name)
    out = fn(np.full((8, 8), 255, dtype=np.uint8))
    assert abs(out.max() - 1.0) < 1e-6
    out = fn(np.full((8, 8), 128, dtype=np.uint8))
    assert abs(out.max() - 128 / 255.0) < 1e-6


def test_feather_mask_never_overshoots_one():
    from retouch.utils import feather_mask
    m = np.zeros((64, 64), np.float32); m[16:48, 16:48] = 1.0
    out = feather_mask(m, radius=6)
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_face_core_keeps_full_skin_alpha_for_epsilon_mask():
    """End-to-end at the face-core boundary: a skin mask peaking at 1.0000002
    must come back as a ~1.0 composite mask, not 1/255."""
    from retouch.perf_optimizations import _process_face_core
    from tests.test_engine import _build_synthetic_regions, _build_synthetic_face
    from tests.test_undereye_alias import _processors, _ctx

    roi = 128
    regions = _build_synthetic_regions(roi, roi, skin_value=1.0)
    regions.skin = (regions.skin * EPS_OVER_ONE).astype(np.float32)
    assert regions.skin.max() > 1.0
    canvas = np.full((roi, roi, 3), 150, np.uint8)
    fr = _process_face_core(canvas, regions, _build_synthetic_face(ied=20.0, size=80), _ctx(0, 0),
                            0, 0, roi, roi, np.ones((roi, roi), np.float32), _processors())
    assert fr.skin_mask.max() >= 0.999
    assert fr.skin_hair_mask.max() >= 0.999

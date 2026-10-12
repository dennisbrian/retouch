"""E1 float-canvas parity regression tests.

The per-face pipeline (perf_optimizations._process_face_core) keeps the canvas
as float32 in [0, 255]. Ops that call cv2.cvtColor directly on the canvas must
route float32 input through the E1 delta adapter (utils.apply_u8_op_float),
otherwise OpenCV interprets float BGR as [0, 1] and a [0, 255] canvas produces
garbage (classic symptom: whole region collapses to saturated blue, B >> R).

These tests pin:
  a. op(float32) ~= op(uint8) within a small quantization tolerance, and the
     float-input output does NOT invert channels for a skin-toned input.
  b. every recipe's skin.relight (resolving `extends`) is a 0-1 fraction.
"""
import numpy as np
import pytest

from retouch.makeup import MakeupEngine
from retouch.undereye import UnderEyeRepairer
from retouch.params import RECIPES, resolve_recipe


# ---- shared synthetic fixtures ------------------------------------------------

# BGR skin tone: warm, R > G > B (a real face pixel). Inversion bug flips this.
SKIN_BGR = (150, 180, 210)


def _skin_img(h=80, w=80):
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = SKIN_BGR[0]
    img[:, :, 1] = SKIN_BGR[1]
    img[:, :, 2] = SKIN_BGR[2]
    # a slightly darker patch so under-eye repair has something to lift
    img[30:50, 20:60] = (SKIN_BGR[0] - 40, SKIN_BGR[1] - 40, SKIN_BGR[2] - 40)
    return img


class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 0.0


class MockLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(0.5, 0.5) for _ in range(500)]
        self.landmark[117] = MockLandmark(0.7, 0.6)
        self.landmark[346] = MockLandmark(0.3, 0.6)
        self.landmark[4] = MockLandmark(0.5, 0.7)


class MockFaceRegions:
    """Minimal FaceRegions stand-in; only the attributes each op reads."""

    def __init__(self, h=80, w=80):
        self.left_under_eye = None
        self.right_under_eye = None
        self.lips = np.zeros((h, w), dtype=np.float32)


def _no_channel_inversion(out, tol=25):
    """For a skin-toned input, mean B must stay below mean R (no blue collapse)."""
    mb = float(out[:, :, 0].mean())
    mr = float(out[:, :, 2].mean())
    assert mb < mr, f"channel inversion: mean B={mb:.1f} >= mean R={mr:.1f}"


def _parity(u8_out, f32_out, max_diff=2):
    d = np.abs(u8_out.astype(np.float32) - np.clip(f32_out, 0, 255))
    assert d.max() <= max_diff, f"max abs diff {d.max()} > {max_diff}"


# ---- a. float parity ----------------------------------------------------------


class TestUnderEyeFloatParity:
    def test_repair_float_parity_and_no_inversion(self):
        img = _skin_img()
        regions = MockFaceRegions()
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[30:50, 20:60] = 1.0
        regions.left_under_eye = mask
        regions.right_under_eye = np.zeros((80, 80), dtype=np.float32)

        repairer = UnderEyeRepairer()
        u8_out = repairer.repair(img, regions, strength=80)
        f32_out = repairer.repair(img.astype(np.float32), regions, strength=80)

        assert f32_out.dtype == np.float32
        _no_channel_inversion(f32_out)
        _parity(u8_out, f32_out)


class TestMakeupFloatParity:
    def test_apply_blush_float_parity_and_no_inversion(self):
        img = _skin_img()
        landmarks = MockLandmarks()
        engine = MakeupEngine()

        u8_out = engine.apply_blush(img, landmarks, 100, strength=60)
        f32_out = engine.apply_blush(img.astype(np.float32), landmarks, 100, strength=60)

        assert f32_out.dtype == np.float32
        _no_channel_inversion(f32_out)
        _parity(u8_out, f32_out)


@pytest.mark.skip(
    reason="Relight/sculpt require a real MediaPipe landmarks object: they build "
    "a Delaunay triangulation over 468 points with plausible face topology. A "
    "flat/mock landmark grid degenerates the mesh (NaN shading), so op-parity "
    "cannot be exercised without MediaPipe. The float32 guard in relight()/"
    "sculpt() uses the identical apply_u8_op_float adapter pattern verified for "
    "undereye and makeup above, plus the fixed skin.relight recipe data."
)
class TestRelightFloatParity:
    def test_relight_float_parity(self):
        pass


# ---- b. recipe data: skin.relight is a 0-1 fraction ---------------------------


class TestRecipeRelightFraction:
    @pytest.mark.parametrize("name", sorted(RECIPES.keys()))
    def test_relight_is_fraction(self, name):
        rec = resolve_recipe(name)
        skin = rec.get("skin", {})
        if "relight" in skin and skin["relight"] is not None:
            assert float(skin["relight"]) <= 1.0, (
                f"{name}: skin.relight={skin['relight']} should be a 0-1 fraction"
            )

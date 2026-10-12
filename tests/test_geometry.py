"""Tests for retouch/geometry.py — FaceReshaper."""
import numpy as np
import pytest
from retouch.geometry import FaceReshaper
from retouch.detection import FaceData, _Landmark, _LandmarkCompat


def _make_mock_landmark(x, y):
    return _Landmark(x, y, z=0.0)


def _make_face_data(cx, cy, face_w, face_h):
    lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(500)]
    lm_list[234] = _make_mock_landmark((cx - face_w // 2) / 400.0, cy / 400.0)
    lm_list[454] = _make_mock_landmark((cx + face_w // 2) / 400.0, cy / 400.0)
    lm_list[117] = _make_mock_landmark(0.6, 0.55)
    lm_list[346] = _make_mock_landmark(0.4, 0.55)
    lm_list[152] = _make_mock_landmark(0.5, 0.7)
    compat = _LandmarkCompat(lm_list)
    return FaceData(landmarks=compat, bbox=(cx - face_w // 2, cy - face_h // 2, face_w, face_h), ied=50.0)


@pytest.fixture
def reshaper():
    return FaceReshaper()


@pytest.fixture
def img():
    return np.full((400, 400, 3), 128, dtype=np.uint8)


class TestReshape:
    def test_zero_strength(self, reshaper, img):
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(img, [face], strength=0)
        assert np.all(result == img)

    def test_no_faces(self, reshaper, img):
        result = reshaper.reshape(img, [], strength=50)
        assert np.all(result == img)

    def test_output_shape(self, reshaper, img):
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(img, [face], strength=50)
        assert result.shape == (400, 400, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, reshaper):
        grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        grad_img = np.stack([grad] * 3, axis=-1)
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(grad_img, [face], strength=80)
        assert not np.allclose(result, grad_img)

    def test_multiple_faces(self, reshaper, img):
        face1 = _make_face_data(150, 150, 60, 75)
        face2 = _make_face_data(300, 300, 60, 75)
        result = reshaper.reshape(img, [face1, face2], strength=50)
        assert result.shape == (400, 400, 3)

    def test_tiny_face_skipped(self, reshaper, img):
        face = _make_face_data(200, 200, 8, 10)
        result = reshaper.reshape(img, [face], strength=50)
        assert np.all(result == img)

    def test_face_at_edge(self, reshaper, img):
        face = _make_face_data(20, 20, 60, 75)
        result = reshaper.reshape(img, [face], strength=50)
        assert result.shape == (400, 400, 3)


# ---------------------------------------------------------------------------
# Backlog #6 — Reshape completeness (L/R side variants + neck)
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _nose_compat():
    lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(500)]
    lm_list[234] = _make_mock_landmark(0.3, 0.5)
    lm_list[454] = _make_mock_landmark(0.7, 0.5)
    lm_list[168] = _make_mock_landmark(0.5, 0.45)
    lm_list[48] = _make_mock_landmark(0.45, 0.5)
    lm_list[278] = _make_mock_landmark(0.55, 0.5)
    return _LandmarkCompat(lm_list)


def _nose_landmarks():
    return _nose_compat().landmark


class TestReshapeParamWiring:
    def test_context_carries_new_fields(self):
        from retouch.engine import build_context
        rec = {"reshape": {
            "jaw_width_l": 40, "jaw_width_r": 10,
            "nose_width_l": 20, "nose_width_r": 5,
            "eye_size_l": 15, "eye_size_r": 8,
            "neck_width": 25, "neck_length": 12,
        }}
        ctx = build_context("natural", rec, {})
        assert ctx.reshape_jaw_width_l == 40
        assert ctx.reshape_jaw_width_r == 10
        assert ctx.reshape_nose_width_l == 20
        assert ctx.reshape_nose_width_r == 5
        assert ctx.reshape_eye_size_l == 15
        assert ctx.reshape_eye_size_r == 8
        assert ctx.reshape_neck_width == 25
        assert ctx.reshape_neck_length == 12

    def test_defaults_zero(self):
        from retouch.engine import build_context
        ctx = build_context("natural", {}, {})
        for attr in (
            "reshape_jaw_width_l", "reshape_jaw_width_r",
            "reshape_nose_width_l", "reshape_nose_width_r",
            "reshape_eye_size_l", "reshape_eye_size_r",
            "reshape_neck_width", "reshape_neck_length",
        ):
            assert getattr(ctx, attr) == 0


class TestSideWarps:
    def test_jaw_global_two_sides(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 2

    def test_jaw_left_only(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0)
        assert len(warps) == 1

    def test_jaw_both_sides_independent(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=10.0)
        assert len(warps) == 2

    def test_jaw_global_byte_identical(self, reshaper):
        lm = _nose_landmarks()
        legacy = reshaper._jaw_width_warps(lm, 160.0, 30.0, 400, 400)
        via_side = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=30.0)
        assert legacy == via_side

    def test_nose_global_radial(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._nose_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 8  # _SCALE_RING_N

    def test_nose_left_only(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._nose_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0)
        assert len(warps) == 1

    def test_eye_side_no_crash(self, reshaper):
        compat = _nose_compat()
        warps = reshaper._eye_size_warps(
            compat, compat.landmark, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0
        )
        assert isinstance(warps, list)


class TestNeckWarps:
    def test_neck_width_zero(self, reshaper):
        lm = _nose_landmarks()
        assert reshaper._neck_width_warps(lm, 160.0, 0.0, 400, 400) == []

    def test_neck_width_active(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._neck_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 6

    def test_neck_length_zero(self, reshaper):
        lm = _nose_landmarks()
        assert reshaper._neck_length_warps(lm, 160.0, 0.0, 400, 400) == []

    def test_neck_length_active(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._neck_length_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 3


class TestBackwardCompat:
    def _grad(self):
        grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        return np.stack([grad] * 3, axis=-1)

    def test_side_zero_matches_global(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx_global = SimpleNamespace(reshape_jaw_width=30.0)
        ctx_side_zero = SimpleNamespace(
            reshape_jaw_width=30.0,
            reshape_jaw_width_l=0.0, reshape_jaw_width_r=0.0,
            reshape_nose_width_l=0.0, reshape_nose_width_r=0.0,
            reshape_eye_size_l=0.0, reshape_eye_size_r=0.0,
            reshape_neck_width=0.0, reshape_neck_length=0.0,
        )
        out_g = reshaper.reshape(grad.copy(), [face], ctx_global)
        out_s = reshaper.reshape(grad.copy(), [face], ctx_side_zero)
        assert np.array_equal(out_g, out_s)

    def test_side_value_changes_output(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx = SimpleNamespace(reshape_jaw_width_l=40.0)
        out = reshaper.reshape(grad.copy(), [face], ctx)
        assert not np.array_equal(out, grad)

    def test_neck_value_changes_output(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx = SimpleNamespace(reshape_neck_width=40.0)
        out = reshaper.reshape(grad.copy(), [face], ctx)
        assert not np.array_equal(out, grad)


# ---------------------------------------------------------------------------
# Rolled-face warp direction (AE-series fix — punched-cheek regression on a
# ~52 deg rolled cosplay close-up, see RESEARCH_FRONTIER_AE_2026_07_20.md).
# Jaw/cheek/chin/smile warps must pull along the face's OWN axes (jaw-to-jaw,
# forehead-to-chin), not raw screen +-x/+-y, or a rolled head gets pushed
# sideways across the face instead of compressed toward its center.
# ---------------------------------------------------------------------------

def _rolled_landmarks(angle_deg, cx=0.5, cy=0.5, r=0.2):
    """478-landmark fixture with 234/454/152/10 placed on a face rotated
    ``angle_deg`` from upright (0 deg = landmark 234/454 level, matching the
    existing fixtures' implicit assumption)."""
    theta = np.radians(angle_deg)
    lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(478)]
    # 234=anatomical right jaw, 454=anatomical left jaw: opposite ends of the
    # horizontal axis, then rotated by theta around the face center.
    lm_list[234] = _make_mock_landmark(cx - r * np.cos(theta), cy - r * np.sin(theta))
    lm_list[454] = _make_mock_landmark(cx + r * np.cos(theta), cy + r * np.sin(theta))
    # 10=forehead top, 152=chin: opposite ends of the vertical axis, rotated
    # by the same theta so it stays perpendicular-ish to the horizontal.
    lm_list[10] = _make_mock_landmark(cx + r * np.sin(theta), cy - r * np.cos(theta))
    lm_list[152] = _make_mock_landmark(cx - r * np.sin(theta), cy + r * np.cos(theta))
    lm_list[117] = _make_mock_landmark(cx - 0.6 * r * np.cos(theta), cy - 0.6 * r * np.sin(theta))
    lm_list[346] = _make_mock_landmark(cx + 0.6 * r * np.cos(theta), cy + 0.6 * r * np.sin(theta))
    lm_list[61] = _make_mock_landmark(cx - 0.3 * r * np.cos(theta), cy - 0.3 * r * np.sin(theta) + 0.05)
    lm_list[291] = _make_mock_landmark(cx + 0.3 * r * np.cos(theta), cy + 0.3 * r * np.sin(theta) + 0.05)
    return _LandmarkCompat(lm_list).landmark


class TestRolledFaceWarpDirection:
    def test_face_local_axes_matches_screen_axes_at_zero_roll(self, reshaper):
        lm = _rolled_landmarks(0.0)
        horiz, vert = reshaper._face_local_axes(lm, 400, 400)
        assert horiz == pytest.approx([1.0, 0.0], abs=1e-4)
        assert vert == pytest.approx([0.0, 1.0], abs=1e-4)

    def test_face_local_axes_rotates_with_roll(self, reshaper):
        lm = _rolled_landmarks(52.0)
        horiz, vert = reshaper._face_local_axes(lm, 400, 400)
        expected_horiz = [np.cos(np.radians(52.0)), np.sin(np.radians(52.0))]
        assert horiz == pytest.approx(expected_horiz, abs=1e-4)
        # horiz and vert stay perpendicular under this fixture's construction
        assert float(np.dot(horiz, vert)) == pytest.approx(0.0, abs=1e-4)

    def test_slimming_jaw_pull_follows_roll_not_screen_axis(self, reshaper):
        """The core regression: at 52 deg roll, the jaw warp's displacement
        vector must point along the true (rotated) jaw axis, not along
        screen-horizontal. Pre-fix, this displacement was fixed at
        (+something, 0) regardless of roll — i.e. ~52 deg off the true axis,
        which is what produced the visible lopsided cheek bulge."""
        # fw/coords scaled up (not just a real photo's 100-1000px range, but
        # large enough that int()-pixel truncation on the warp endpoints is
        # negligible relative to the displacement magnitude being measured —
        # at fw=100 the raw displacement is ~2px and truncation alone adds
        # several degrees of noise unrelated to warp correctness).
        lm = _rolled_landmarks(52.0, r=0.2)
        warps = reshaper._slimming_warps(lm, fw=1000.0, strength=50.0, h=2000, w=2000)
        c_rjaw, t_rjaw, _r = warps[0]
        disp = np.array([t_rjaw[0] - c_rjaw[0], t_rjaw[1] - c_rjaw[1]], dtype=np.float64)
        disp_norm = disp / np.linalg.norm(disp)
        expected_dir = np.array([np.cos(np.radians(52.0)), np.sin(np.radians(52.0))])
        angle_off = np.degrees(np.arccos(np.clip(abs(np.dot(disp_norm, expected_dir)), -1, 1)))
        assert angle_off < 1.0, (
            f"jaw warp displacement is {angle_off:.1f} deg off the true rolled "
            "jaw axis (regression: hardcoded screen +-x pull instead of the "
            "face's own rotated axis)"
        )

    def test_slimming_upright_byte_identical_to_legacy_axis(self, reshaper):
        """At zero roll (the common case), the new axis-aware warp must
        reduce to the exact legacy screen +-x/-y displacement."""
        lm = _rolled_landmarks(0.0)
        warps = reshaper._slimming_warps(lm, fw=100.0, strength=50.0, h=400, w=400)
        c_rjaw, t_rjaw, _r = warps[0]
        assert t_rjaw[1] == c_rjaw[1]  # no y-drift at zero roll
        assert t_rjaw[0] > c_rjaw[0]  # pulls inward (+x) same as legacy

    def test_jaw_width_pull_follows_roll(self, reshaper):
        lm = _rolled_landmarks(52.0, r=0.2)
        warps = reshaper._jaw_width_warps(lm, fw=1000.0, slider=30.0, h=2000, w=2000)
        c_rjaw, t_rjaw, _r = warps[0]
        disp = np.array([t_rjaw[0] - c_rjaw[0], t_rjaw[1] - c_rjaw[1]], dtype=np.float64)
        assert np.linalg.norm(disp) > 1e-6
        disp_norm = disp / np.linalg.norm(disp)
        expected_dir = np.array([np.cos(np.radians(52.0)), np.sin(np.radians(52.0))])
        angle_off = np.degrees(np.arccos(np.clip(abs(np.dot(disp_norm, expected_dir)), -1, 1)))
        assert angle_off < 1.0

    def test_smile_lift_follows_roll(self, reshaper):
        """_smile_warps combines an "up" and an "out" component (by design —
        see its docstring), so the two mouth corners' displacements differ
        only in the sign of the "out" term. Cancel it out by summing both
        corners' displacements: the "out" components are equal and opposite
        (they cancel), leaving 2x the shared "up" component, which is what
        must track the rolled vertical axis rather than screen -y."""
        lm = _rolled_landmarks(52.0, r=0.2)
        warps = reshaper._smile_warps(lm, fw=1000.0, slider=30.0, h=2000, w=2000)
        c_l, t_l, _r = warps[0]
        c_r, t_r, _r2 = warps[1]
        disp_l = np.array([t_l[0] - c_l[0], t_l[1] - c_l[1]], dtype=np.float64)
        disp_r = np.array([t_r[0] - c_r[0], t_r[1] - c_r[1]], dtype=np.float64)
        up_component = (disp_l + disp_r) / 2.0
        assert np.linalg.norm(up_component) > 1e-6
        expected_vert = np.array([np.sin(np.radians(52.0)), -np.cos(np.radians(52.0))])
        screen_vert = np.array([0.0, -1.0])
        up_norm = up_component / np.linalg.norm(up_component)
        angle_to_rolled = np.degrees(np.arccos(np.clip(abs(np.dot(up_norm, expected_vert)), -1, 1)))
        angle_to_screen = np.degrees(np.arccos(np.clip(abs(np.dot(up_norm, screen_vert)), -1, 1)))
        assert angle_to_rolled < 5.0, (
            f"smile lift is {angle_to_rolled:.1f} deg off the true rolled "
            f"vertical (vs {angle_to_screen:.1f} deg off screen -y) — "
            "regression: hardcoded screen -y lift instead of the face's own "
            "rotated vertical axis"
        )

    def test_reshape_pipeline_no_crash_on_rolled_face(self, reshaper):
        """End-to-end sanity: a fully rolled face still reshapes without
        error and actually changes the image (no silent no-op)."""
        grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        grad_img = np.stack([grad] * 3, axis=-1)
        lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(478)]
        theta = np.radians(52.0)
        cx, cy, r = 0.5, 0.5, 0.2
        lm_list[234] = _make_mock_landmark(cx - r * np.cos(theta), cy - r * np.sin(theta))
        lm_list[454] = _make_mock_landmark(cx + r * np.cos(theta), cy + r * np.sin(theta))
        lm_list[10] = _make_mock_landmark(cx + r * np.sin(theta), cy - r * np.cos(theta))
        lm_list[152] = _make_mock_landmark(cx - r * np.sin(theta), cy + r * np.cos(theta))
        lm_list[117] = _make_mock_landmark(cx - 0.6 * r * np.cos(theta), cy - 0.6 * r * np.sin(theta))
        lm_list[346] = _make_mock_landmark(cx + 0.6 * r * np.cos(theta), cy + 0.6 * r * np.sin(theta))
        compat = _LandmarkCompat(lm_list)
        face = FaceData(landmarks=compat, bbox=(80, 80, 240, 240), ied=50.0)
        out = reshaper.reshape(grad_img.copy(), [face], strength=80)
        assert out.shape == (400, 400, 3)
        assert not np.array_equal(out, grad_img)

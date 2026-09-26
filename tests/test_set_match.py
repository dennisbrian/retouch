"""Tests for retouch/set_match.py (match a set to one hero frame)."""

from types import SimpleNamespace

import numpy as np
import pytest

from retouch import set_match as sm
from retouch.params import get_param
from retouch.set_match import (
    IDENTITY,
    FrameStats,
    apply_gains,
    coerce_hero,
    compute_gains,
    match_to_hero,
    measure_frame,
)


def _encode(lin):
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def _decode(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _scene(seed=0, size=(240, 320)):
    """Smooth, mid-toned uint8 BGR scene with some colour variety."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = np.stack([
        0.35 + 0.25 * np.sin(xx / 37.0),
        0.40 + 0.20 * np.cos(yy / 29.0),
        0.45 + 0.20 * np.sin((xx + yy) / 51.0),
    ], axis=2)
    base += rng.normal(0, 0.01, base.shape)
    return np.clip(base * 255, 0, 255).astype(np.uint8)


def _shift(img_u8, gains_bgr):
    lin = _decode(img_u8.astype(np.float64) / 255.0) * np.asarray(gains_bgr)
    return np.clip(np.round(_encode(np.clip(lin, 0, 1)) * 255), 0, 255).astype(np.uint8)


class TestGains:
    def test_self_match_is_identity(self):
        stats = measure_frame(_scene())
        g = compute_gains(stats, stats)
        assert g.is_identity
        assert g.exposure_ev == pytest.approx(0.0, abs=1e-9)

    def test_zero_strength_is_identity(self):
        a, b = measure_frame(_scene(0)), measure_frame(_shift(_scene(0), (0.8, 1.0, 1.2)))
        assert compute_gains(a, b, 0.0) is IDENTITY

    def test_skin_basis_recovers_known_shift(self):
        hero = FrameStats((0.2, 0.2, 0.2), (0.30, 0.35, 0.45), 1000)
        shift = np.array([0.9, 1.0, 1.1]) * 0.7
        frame = FrameStats((0.2, 0.2, 0.2), tuple(np.array(hero.skin_bgr) * shift), 1000)
        g = compute_gains(hero, frame)
        assert g.basis == "skin"
        np.testing.assert_allclose(np.array(g.gains_bgr) * shift, 1.0, rtol=1e-6)

    def test_scene_fallback_when_either_side_has_no_face(self):
        hero = FrameStats((0.2, 0.2, 0.2), (0.3, 0.35, 0.45), 1000)
        frame = FrameStats((0.1, 0.1, 0.1))
        g = compute_gains(hero, frame)
        assert g.basis == "scene"
        assert g.exposure_ev == pytest.approx(1.0)  # exactly 1 EV, at the scene clamp

    def test_exposure_and_wb_are_clamped(self):
        hero = FrameStats((0.2,) * 3, (0.40, 0.40, 0.40), 1000)
        frame = FrameStats((0.2,) * 3, (0.01, 0.05, 0.02), 1000)
        g = compute_gains(hero, frame)
        assert abs(g.exposure_ev) <= sm.SKIN_MAX_EV + 1e-9
        lo, hi = sm.SKIN_WB_RANGE
        # After luminance renormalisation the gains may leave the raw clamp
        # slightly, but never by more than the renormalisation factor.
        assert max(g.wb_bgr) / min(g.wb_bgr) <= hi / lo + 1e-6

    def test_gains_are_skin_tone_invariant(self):
        """Same relative shift -> same correction, whatever the skin's own level.

        The anchor is the same person's skin in both frames, so a darker or
        lighter complexion (both frames scaled alike) must not change the
        correction. No absolute gate may creep in.
        """
        shift = np.array([1.1, 1.0, 0.85]) * 1.3
        out = []
        for level in (0.03, 0.12, 0.45):
            skin = np.array([0.8, 0.9, 1.0]) * level
            hero = FrameStats((0.2,) * 3, tuple(skin), 1000)
            frame = FrameStats((0.2,) * 3, tuple(skin * shift), 1000)
            out.append(compute_gains(hero, frame).gains_bgr)
        np.testing.assert_allclose(out[0], out[1], rtol=1e-9)
        np.testing.assert_allclose(out[0], out[2], rtol=1e-9)

    def test_strength_interpolates_geometrically(self):
        hero = FrameStats((0.2,) * 3, (0.3, 0.3, 0.3), 1000)
        frame = FrameStats((0.2,) * 3, (0.15, 0.15, 0.15), 1000)
        full = compute_gains(hero, frame, 1.0)
        half = compute_gains(hero, frame, 0.5)
        assert full.exposure_ev == pytest.approx(1.0)
        assert half.exposure_ev == pytest.approx(0.5)


class TestApply:
    def test_identity_returns_input_unchanged(self):
        img = _scene()
        assert apply_gains(img, IDENTITY) is img

    def test_whole_frame_match_pulls_shifted_frame_back(self):
        hero_img = _scene(1)
        # Within the whole-frame clamps: a mild cast plus -0.3 EV.
        frame = _shift(hero_img, np.array([0.94, 1.0, 1.06]) * 0.8)
        out, g = match_to_hero(frame, measure_frame(hero_img))
        assert g.basis == "scene"
        err_before = np.abs(frame.astype(int) - hero_img.astype(int)).mean()
        err_after = np.abs(out.astype(int) - hero_img.astype(int)).mean()
        assert err_after < 0.25 * err_before

    def test_float_input_keeps_dtype_and_range(self):
        img = _scene(2).astype(np.float32)  # engine float input is [0, 255]
        gains = compute_gains(FrameStats((0.3,) * 3), FrameStats((0.2,) * 3))
        out = apply_gains(img, gains)
        assert out.dtype == np.float32
        assert out.max() <= 255.0 + 1e-3 and out.mean() > img.mean()

    def test_blown_pixels_are_untouched(self):
        img = _scene(3)
        img[:20, :20] = 255
        gains = compute_gains(FrameStats((0.2,) * 3), FrameStats((0.18, 0.2, 0.22)))
        assert not gains.is_identity
        out = apply_gains(img, gains)
        assert np.array_equal(out[:20, :20], img[:20, :20])
        assert not np.array_equal(out[40:, 40:], img[40:, 40:])

    def test_brightening_never_clips_midtones_hard(self):
        img = np.full((10, 10, 3), 200, np.uint8)
        gains = compute_gains(FrameStats((0.4,) * 3), FrameStats((0.2,) * 3))  # +1 EV
        out = apply_gains(img, gains)
        assert out.max() < 255  # the shoulder rolls off instead of clipping


class TestMeasure:
    def test_skin_anchor_uses_face_landmarks(self):
        # A face-oval landmark set roughly filling the middle of the frame.
        img = np.full((200, 200, 3), 60, np.uint8)
        img[40:160, 50:150] = (120, 140, 190)  # "skin" block, BGR

        class _LM:
            def __init__(self):
                self.landmark = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
                from retouch.parsing import FACE_OVAL
                n = len(FACE_OVAL)
                for i, idx in enumerate(FACE_OVAL):
                    t = 2 * np.pi * i / n
                    self.landmark[idx] = SimpleNamespace(
                        x=0.5 + 0.22 * np.sin(t), y=0.5 - 0.27 * np.cos(t))

        face = SimpleNamespace(landmarks=_LM(), bbox=(56, 46, 88, 108))
        stats = measure_frame(img, faces=[face])
        assert stats.skin_bgr is not None and stats.skin_pixels >= sm.MIN_SKIN_PIXELS
        lin = _decode(np.array([120, 140, 190]) / 255.0)
        np.testing.assert_allclose(stats.skin_bgr, lin, rtol=0.02)

    def test_no_detector_gives_scene_only(self):
        stats = measure_frame(_scene())
        assert stats.skin_bgr is None

    def test_detector_failure_falls_back(self):
        class Boom:
            def detect(self, _img):
                raise RuntimeError("no runtime")

        assert measure_frame(_scene(), detector=Boom()).skin_bgr is None


class TestCoerceHero:
    def test_round_trips_dict_form(self):
        stats = FrameStats((0.1, 0.2, 0.3), (0.3, 0.35, 0.4), 999)
        assert coerce_hero(stats.to_dict()) == stats
        assert coerce_hero(FrameStats((0.1, 0.2, 0.3)).to_dict()).skin_bgr is None

    def test_accepts_image(self):
        assert isinstance(coerce_hero(_scene()), FrameStats)

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            coerce_hero("hero.jpg")


def test_param_is_registered_and_off_by_default():
    spec = get_param("set_match")
    assert spec.default == 0 and spec.min_val == 0 and spec.max_val == 100
    assert spec.recipe_key is None


class _Args(SimpleNamespace):
    """argparse stand-in: every flag not given reads as None."""

    def __getattr__(self, _name):
        return None


def test_cli_match_hero_defaults_strength_to_full():
    import cli

    params = cli.build_params(_Args(match_hero="hero.jpg"))
    assert params["set_match_hero_path"] == "hero.jpg"
    assert params["set_match"] == 100
    assert cli.build_params(_Args(match_hero="hero.jpg", set_match=40))["set_match"] == 40
    assert "set_match_hero_path" not in cli.build_params(_Args())


def test_cli_help_lists_match_hero():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "cli.py"), "--help"],
                         capture_output=True, text=True, timeout=120, cwd=root)
    assert "--match-hero" in out.stdout and "--set-match" in out.stdout


def test_session_save_keeps_measured_hero_as_numbers(tmp_path):
    import json

    import cli

    hero = FrameStats((0.1, 0.2, 0.3), (0.3, 0.35, 0.4), 999)
    target = cli._save_session(
        {"recipe": "natural", "set_match": 100, "set_match_hero": hero},
        tmp_path / "a.jpg", tmp_path / "a_out.jpg", None,
    )
    saved = json.loads(open(target).read())
    params = saved.get("params", saved)
    assert coerce_hero(params["set_match_hero"]) == hero


def test_global_only_path_applies_scene_match():
    import cli

    hero_img = _scene(4)
    frame = _shift(hero_img, (1.2, 1.0, 0.85))
    params = {"recipe": "natural", "set_match": 100, "set_match_hero": measure_frame(hero_img)}
    matched = cli._apply_global_finish(frame, dict(params))
    unmatched = cli._apply_global_finish(frame, {"recipe": "natural"})
    ref = cli._apply_global_finish(hero_img, {"recipe": "natural"})
    assert (np.abs(matched.astype(int) - ref.astype(int)).mean()
            < np.abs(unmatched.astype(int) - ref.astype(int)).mean())


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    eng = RetouchEngine()
    yield eng
    eng.close()


class TestEngineHook:
    def test_hero_shift_reported_and_applied(self, engine):
        hero_img = _scene(5)
        frame = _shift(hero_img, np.array([0.94, 1.0, 1.06]) * 0.8)
        hero = measure_frame(hero_img)
        res = engine.process(frame, recipe="natural", set_match=100, set_match_hero=hero.to_dict())
        diag = res.runtime_diagnostics["set_match"]
        assert diag["basis"] == "scene" and diag["exposure_ev"] > 0.2
        plain = engine.process(frame, recipe="natural")
        assert "set_match" not in plain.runtime_diagnostics
        ref = engine.process(hero_img, recipe="natural")
        err = lambda a: np.abs(np.asarray(a, dtype=float) - np.asarray(ref, dtype=float)).mean()
        assert err(res) < err(plain)

    def test_off_without_hero_or_strength(self, engine):
        img = _scene(6)
        hero = measure_frame(_scene(7)).to_dict()
        for kwargs in ({"set_match": 100}, {"set_match": 0, "set_match_hero": hero}):
            res = engine.process(img, recipe="natural", **kwargs)
            assert "set_match" not in res.runtime_diagnostics

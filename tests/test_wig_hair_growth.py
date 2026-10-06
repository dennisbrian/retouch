"""Wig growth for the BiSeNet-less hair mask (``parsing._grow_wig_hair``).

Synthetic class probabilities stand in for the multiclass segmenter's real
failure on cosplay wigs: a patch of fringe is called hair, the rest of the
wig is called an accessory ("others"). No model is downloaded.
"""
import numpy as np
import pytest

from retouch.parsing import (
    _MC_CLOTHES,
    _MC_FACE_SKIN,
    _MC_HAIR,
    _MC_NUM_CLASSES,
    _MC_OTHERS,
    FaceParser,
    _grow_wig_hair,
)

H, W = 300, 300
WHITE, BLACK, RED, SKIN = (205, 205, 210), (25, 25, 25), (30, 30, 200), (150, 170, 200)


def _scene(wig=WHITE, costume=RED, hat=BLACK, costume_xs=slice(40, 260)):
    """Face centre, wig round the head and down both sides, hat on top,
    costume below, and a prop held away from the head in the wig's colour."""
    img = np.zeros((H, W, 3), np.uint8)
    p = np.zeros((H, W, _MC_NUM_CLASSES), np.float32)
    p[..., 0] = 1.0

    def put(ys, xs, cls, col):
        img[ys, xs] = col
        p[ys, xs, :] = 0.0
        p[ys, xs, cls] = 1.0

    put(slice(40, 230), slice(80, 220), _MC_OTHERS, wig)      # wig block, mislabelled
    put(slice(60, 90), slice(110, 190), _MC_HAIR, wig)        # fringe the model got right
    put(slice(10, 40), slice(80, 220), _MC_OTHERS, hat)       # hat touching the wig
    put(slice(90, 170), slice(110, 190), _MC_FACE_SKIN, SKIN)  # face
    put(slice(230, 300), costume_xs, _MC_CLOTHES, costume)
    put(slice(250, 290), slice(0, 30), _MC_OTHERS, wig)       # prop, far from head
    face = np.zeros((H, W), bool)
    face[90:170, 110:190] = True
    return img, p, face


def test_mislabelled_wig_grows_back():
    img, p, face = _scene()
    hair = _grow_wig_hair(img, p, face, 80)
    assert hair[200, 100] > 0.9      # wig side below the fringe
    assert hair[50, 90] > 0.9        # wig crown
    assert hair[25, 150] < 0.05      # hat stays out
    assert hair[260, 150] < 0.05     # costume stays out
    assert hair[270, 15] < 0.05      # same-coloured prop, not connected
    assert hair[130, 150] < 0.05     # face never becomes hair


def test_never_below_segmenter_hair():
    img, p, face = _scene()
    hair = _grow_wig_hair(img, p, face, 80)
    assert np.all(hair >= p[:, :, _MC_HAIR] - 1e-6)


def test_hair_matching_the_costume_does_not_grow():
    # Dark hair over a dark outfit: colour cannot separate them, so keep
    # the segmenter's hair rather than swallow the outfit.
    # (A narrow outfit keeps the would-be growth under the wall guard, so
    # this checks the colour test itself.)
    img, p, face = _scene(wig=BLACK, costume=BLACK, hat=BLACK, costume_xs=slice(120, 180))
    hair = _grow_wig_hair(img, p, face, 80)
    np.testing.assert_array_equal(hair, p[:, :, _MC_HAIR])


def test_too_few_seeds_is_a_no_op():
    img, p, face = _scene()
    p[60:90, 110:190, _MC_HAIR] = 0.0
    p[60:90, 110:190, _MC_OTHERS] = 1.0
    hair = _grow_wig_hair(img, p, face, 80)
    assert hair.max() == 0.0


def test_growth_the_size_of_a_wall_is_rejected():
    img, p, face = _scene()
    tiny_face = np.zeros_like(face)
    tiny_face[125:130, 145:150] = True   # 25 px face: wig is far over 5x
    hair = _grow_wig_hair(img, p, tiny_face, 80)
    np.testing.assert_array_equal(hair, p[:, :, _MC_HAIR])


def test_float_image_matches_uint8():
    img, p, face = _scene()
    a = _grow_wig_hair(img, p, face, 80)
    b = _grow_wig_hair(img.astype(np.float32) / 255.0, p, face, 80)
    np.testing.assert_allclose(a, b)


class TestFullImageFallback:
    @pytest.fixture
    def parser(self, monkeypatch):
        monkeypatch.setenv("RETOUCH_CLASS_SEGMENTER", "0")
        return FaceParser()

    def test_none_without_any_model(self, parser):
        parser._sess = None
        assert parser.parse_hair_full_image(np.full((40, 40, 3), 128, np.uint8)) is None

    def test_uses_class_segmenter_with_wig_growth(self, parser, monkeypatch):
        img, p, _face = _scene()
        parser._sess = None
        monkeypatch.setattr(parser, "_segment_classes", lambda _img: p)
        hair = parser.parse_hair_full_image(img)
        assert hair.shape == (H, W) and hair.dtype == np.float32
        assert hair[200, 100] > 0.9 and hair[25, 150] < 0.05


# --- Seedless wig mask (_seedless_wig_hair) -------------------------------
# The segmenter's real failure on white, pastel and bright wigs: not one wig
# pixel is called hair; the whole wig comes back as "clothes".

LAVENDER = (225, 180, 200)   # BGR


def _strands(img, ys, xs, col, rng, amp=14.0):
    """Paint wig fibre: vertical strands (fine stripes with random phase)."""
    hh, ww = ys.stop - ys.start, xs.stop - xs.start
    x = np.arange(ww)[None, :] + rng.normal(0, 0.3, (hh, 1))
    phase = rng.uniform(0, 2 * np.pi, (1, ww)).repeat(hh, 0) * 0.15
    shade = amp * np.sin(2 * np.pi * x / 4.0 + phase)
    img[ys, xs] = np.clip(np.array(col, np.float32)[None, None] + shade[..., None], 0, 255)


def _seedless_scene(wig=LAVENDER, costume=RED, gain=1.0):
    """Wig fully labelled clothes, around a face, strands painted in."""
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.uint8)
    p = np.zeros((H, W, _MC_NUM_CLASSES), np.float32)
    p[..., 0] = 1.0

    def lab(ys, xs, cls):
        p[ys, xs, :] = 0.0
        p[ys, xs, cls] = 1.0

    _strands(img, slice(40, 230), slice(80, 220), wig, rng)
    lab(slice(40, 230), slice(80, 220), _MC_CLOTHES)
    img[90:170, 110:190] = SKIN
    lab(slice(90, 170), slice(110, 190), _MC_FACE_SKIN)
    img[230:300, 40:260] = costume
    lab(slice(230, 300), slice(40, 260), _MC_CLOTHES)
    face = np.zeros((H, W), bool)
    face[90:170, 110:190] = True
    if gain != 1.0:
        lin = (img / 255.0) ** 2.2 * gain
        img = (255 * np.clip(lin, 0, 1) ** (1 / 2.2)).astype(np.uint8)
    return img, p, face


class TestSeedlessWig:
    def test_wig_with_no_hair_pixels_is_found(self):
        img, p, face = _seedless_scene()
        assert p[:, :, _MC_HAIR].max() == 0.0
        hair = _grow_wig_hair(img, p, face, 80)
        assert (hair[45:225, 85:105] > 0.5).mean() > 0.9    # left side
        assert (hair[45:85, 85:215] > 0.5).mean() > 0.9     # crown
        assert hair[180:225, 195:215].mean() > 0.9          # right side, below the face
        assert hair[100:160, 120:180].max() < 0.05          # face never
        assert hair[240:300, 40:260].max() < 0.05           # costume never

    @pytest.mark.parametrize("wig", [(235, 235, 238), (230, 200, 160), (170, 230, 190), (60, 40, 200)])
    def test_white_and_coloured_wigs(self, wig):
        img, p, face = _seedless_scene(wig=wig)
        hair = _grow_wig_hair(img, p, face, 80)
        assert (hair[45:225, 85:105] > 0.5).mean() > 0.9

    @pytest.mark.parametrize("gain", [0.45, 1.3])
    def test_tone_invariant(self, gain):
        img, p, face = _seedless_scene(gain=gain)
        hair = _grow_wig_hair(img, p, face, 80)
        assert (hair[45:225, 85:105] > 0.5).mean() > 0.9
        assert hair[100:160, 120:180].max() < 0.05

    def test_washed_out_highlight_stays_in(self):
        img, p, face = _seedless_scene()
        img[120:140, 85:105] = (238, 218, 228)    # lavender washing out to white
        hair = _grow_wig_hair(img, p, face, 80)
        assert hair[125:135, 88:102].mean() > 0.9

    def test_untextured_cluster_is_not_a_wig(self):
        # A smooth wig-coloured hood: colour alone is not enough.
        img, p, face = _seedless_scene()
        img[40:230, 80:220] = LAVENDER
        img[90:170, 110:190] = SKIN
        hair = _grow_wig_hair(img, p, face, 80)
        assert hair.max() == 0.0

    def test_most_strand_like_cluster_is_the_wig(self):
        # A grey knit headpiece over the crown, down to the brow, holds more
        # of the head zone than the wig and has some texture of its own; the
        # wig's cleaner strands still win.
        img, p, face = _seedless_scene()
        rng = np.random.default_rng(1)
        _strands(img, slice(30, 90), slice(80, 220), (60, 60, 60), rng)
        img[30:90, 80:220] = np.clip(
            img[30:90, 80:220].astype(np.float32) + rng.normal(0, 10, (60, 140, 1)), 0, 255
        )
        p[30:90, 80:220, :] = 0.0
        p[30:90, 80:220, _MC_OTHERS] = 1.0
        hair = _grow_wig_hair(img, p, face, 80)
        assert hair[35:85, 85:215].mean() < 0.05
        assert (hair[95:225, 85:105] > 0.5).mean() > 0.9

    def test_unchanged_when_the_segmenter_covers_the_head(self):
        # The original scene: the segmenter calls the fringe hair, which
        # covers enough of the head ring that only the seeded growth runs.
        img, p, face = _scene()
        p[40:230, 80:110, :] = 0.0
        p[40:230, 80:110, _MC_HAIR] = 1.0
        p[40:230, 190:220, :] = 0.0
        p[40:230, 190:220, _MC_HAIR] = 1.0
        from retouch.parsing import _grow_wig_from_seeds
        np.testing.assert_array_equal(
            _grow_wig_hair(img, p, face, 80), _grow_wig_from_seeds(img, p, face, 80)
        )

    def test_never_below_segmenter_hair(self):
        img, p, face = _seedless_scene()
        p[60:70, 120:130, :] = 0.0
        p[60:70, 120:130, _MC_HAIR] = 1.0
        hair = _grow_wig_hair(img, p, face, 80)
        assert np.all(hair >= p[:, :, _MC_HAIR] - 1e-6)

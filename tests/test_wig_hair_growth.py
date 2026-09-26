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

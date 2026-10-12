"""Face Shadow Fill: lifts deep face shadows in stops, keeps modelling and texture.

Darker and lighter skin are simulated by scaling linear exposure (and, for
the real-photo numbers in the commit, recolouring painted skin); there is no
real darker-skin photo in these tests.
"""
import cv2
import numpy as np
import pytest

from retouch import face_shadow_fill as F
from retouch.params import PROCESSING_PARAMS

IED = 60.0
H, W = 240, 200


def _lin(img):
    return F._srgb_to_linear(np.clip(img.astype(np.float32) / 255.0, 0, 1))


def _stops(img):
    lin = _lin(img)
    y = 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]
    return np.log2(y + 1e-7)


def _face(gain=1.0, seed=0):
    """Skin-coloured ellipse: a fringe shadow band (-1.5 stops) across the
    top, a gentle 0.25-stop side-to-side modelling gradient, fine texture,
    a dark nostril-sized dot and a pink makeup patch in the lit area."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    skin = (((xx - 100) / 85) ** 2 + ((yy - 120) / 110) ** 2 <= 1).astype(np.float32)
    base = np.array([0.32, 0.42, 0.62], np.float32)  # BGR linear, light skin
    illum = 2.0 ** (-0.25 * xx / W)  # modelling, below the knee
    shadow = (yy > 30) & (yy < 70)
    illum = np.where(shadow, illum * 2.0 ** -1.5, illum)
    texture = 2.0 ** (rng.normal(0, 0.05, (H, W)))
    lin = base * (illum * texture)[..., None] * gain
    nostril = (xx - 100) ** 2 + (yy - 170) ** 2 <= 4 ** 2
    lin[nostril] *= 0.15
    pink = (np.abs(xx - 60) < 14) & (np.abs(yy - 130) < 10)
    lin[pink] = lin[pink] * np.array([0.55, 0.35, 0.8], np.float32)
    img = (F._linear_to_srgb(lin) * 255).astype(np.float32)
    regions = {'skin': skin, 'shadow': shadow & (skin > 0) & (np.abs(xx - 100) < 40),
               'lit': (~shadow) & (skin > 0) & (yy > 90) & (yy < 150) & (np.abs(xx - 130) < 20),
               'nostril': nostril, 'pink': pink}
    return img, regions


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == 'face_shadow_fill')
    assert spec.default == 0
    assert spec.cli_flag == 'face-shadow-fill'
    assert spec.recipe_key == 'skin.face_shadow_fill'


def test_strength_zero_and_missing_mask_are_no_ops():
    img, r = _face()
    assert F.face_shadow_fill(img, 0, IED, r['skin']) is img
    assert F.face_shadow_fill(img, 100, IED, None) is img
    assert F.face_shadow_fill(img, 100, IED, np.zeros((H, W), np.float32)) is img


@pytest.mark.parametrize('strength, lo, hi', [(50, 0.25, 0.5), (100, 0.6, 0.95)])
def test_lifts_the_cast_shadow_by_its_depth_beyond_the_knee(strength, lo, hi):
    img, r = _face()
    out = F.face_shadow_fill(img, strength, IED, r['skin'])
    lift = _stops(out) - _stops(img)
    assert lo < np.median(lift[r['shadow']]) < hi


def test_keeps_modelling_texture_and_everything_outside_the_skin():
    img, r = _face()
    out = F.face_shadow_fill(img, 100, IED, r['skin'])
    lift = _stops(out) - _stops(img)
    # The lit side's 0.25-stop modelling is below the knee: untouched.
    assert np.abs(lift[r['lit']]).max() < 0.02
    # Texture inside the shadow keeps its spread (one smooth gain).
    band = r['shadow'].copy()
    band[:40] = band[60:] = False  # away from the shadow's own edges

    def fine(x):
        return x - cv2.GaussianBlur(x, (0, 0), 3)

    before, after = fine(_stops(img))[band], fine(_stops(out))[band]
    assert np.std(after) == pytest.approx(np.std(before), rel=0.1)
    np.testing.assert_array_equal(out[r['skin'] == 0], img[r['skin'] == 0])


def test_makeup_and_small_dark_features_are_not_read_as_shadow():
    img, r = _face()
    out = F.face_shadow_fill(img, 100, IED, r['skin'])
    lift = _stops(out) - _stops(img)
    assert np.median(lift[r['pink']]) < 0.05
    assert np.median(lift[r['nostril']]) < 0.05
    # Hue of the pink patch is kept even where something changed.
    lin_in, lin_out = _lin(img)[r['pink']], _lin(out)[r['pink']]
    ratio_in = lin_in / lin_in.sum(axis=1, keepdims=True)
    ratio_out = lin_out / lin_out.sum(axis=1, keepdims=True)
    assert np.abs(ratio_out - ratio_in).max() < 0.01


def test_excluded_features_never_change():
    img, r = _face()
    exclude = np.zeros((H, W), np.float32)
    exclude[30:70, 80:120] = 1.0  # e.g. an eye inside the shadow band
    out = F.face_shadow_fill(img, 100, IED, r['skin'], exclude_mask=exclude)
    np.testing.assert_array_equal(out[exclude > 0], img[exclude > 0])
    assert np.abs(out - img).max() > 1  # the rest of the band still lifts


@pytest.mark.parametrize('gain', [0.45, 1.6])
def test_same_lift_in_stops_for_darker_and_lighter_exposure(gain):
    """Simulated: the same face at 0.45x / 1.6x linear exposure."""
    ref, r = _face()
    shot, _ = _face(gain=gain)
    lift_ref = _stops(F.face_shadow_fill(ref, 100, IED, r['skin'])) - _stops(ref)
    lift = _stops(F.face_shadow_fill(shot, 100, IED, r['skin'])) - _stops(shot)
    assert np.median(lift[r['shadow']]) == pytest.approx(
        np.median(lift_ref[r['shadow']]), abs=0.03)


def test_uint8_input_keeps_dtype():
    img, r = _face()
    out = F.face_shadow_fill(np.rint(img).astype(np.uint8), 100, IED, r['skin'] * 255)
    assert out.dtype == np.uint8
    assert (out.astype(int) - np.rint(img))[r['shadow']].mean() > 5


class TestEngineWiring:
    @pytest.fixture(scope='class')
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import (
            make_face_context,
            make_synthetic_face_image,
        )

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        # A cast shadow over the upper face, as a fringe or brim would throw.
        lm = fc.face_data.landmarks.landmark
        brow_y = int(min(lm[105].y, lm[334].y) * h)
        shaded = img.astype(np.float32)
        band = slice(max(0, brow_y - int(0.4 * fc.face_data.ied)), brow_y + int(0.25 * fc.face_data.ied))
        shaded[band] = F._linear_to_srgb(_lin(shaded[band]) * 0.3) * 255
        shaded = np.rint(shaded).astype(np.uint8)
        eng = RetouchEngine()
        off = np.asarray(eng.process(shaded, recipe='natural', face_contexts=[fc]))
        again = np.asarray(eng.process(shaded, recipe='natural', face_contexts=[fc],
                                       face_shadow_fill=0))
        on = np.asarray(eng.process(shaded, recipe='natural', face_contexts=[fc],
                                    face_shadow_fill=100))
        return off, again, on, band

    def test_default_off_changes_nothing(self, run):
        off, again, _, _ = run
        assert np.array_equal(off, again)

    def test_on_lifts_the_shadowed_band(self, run):
        off, _, on, band = run
        diff = on.astype(int) - off.astype(int)
        assert diff[band].mean() > 1, 'face_shadow_fill=100 had no effect: op not wired'
        outside = np.ones(diff.shape[0], bool)
        outside[band] = False
        assert np.abs(diff[outside]).mean() < diff[band].mean() / 3

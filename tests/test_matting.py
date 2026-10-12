import numpy as np

from retouch.matting import build_auto_trimap, refine_alpha_matte


def test_auto_trimap_has_known_and_unknown_regions():
    mask = np.zeros((96, 96), np.float32)
    mask[18:78, 20:76] = 1.0
    trimap = build_auto_trimap(mask, band_radius=3)
    assert np.any(trimap == 0)
    assert np.any(trimap == 128)
    assert np.any(trimap == 255)


def _wisp_fixture():
    """Foreground block plus a hair strand extending past its silhouette."""
    fg = np.zeros((200, 200), np.float32)
    fg[60:140, 60:140] = 1.0
    hair = np.zeros((200, 200), np.float32)
    hair[95:105, 140:180] = 1.0  # strand 40px past the foreground edge
    return fg, hair


def test_hair_mask_marks_wisps_past_the_silhouette_unknown():
    """Regression: the hair branch used to be vacuous (Z3, 2026-07-30).

    ``trimap[hair_band & ~core & ~background] = 128`` assigned 128 to pixels
    that were already 128 by construction, so wisps outside the silhouette
    stayed hard background (0) and the solver erased them.  Hair evidence must
    carve into the background set instead.
    """
    fg, hair = _wisp_fixture()
    strand = np.zeros((200, 200), bool)
    strand[95:105, 145:180] = True

    without = build_auto_trimap(fg)
    with_hair = build_auto_trimap(fg, hair_mask=hair)

    # Before the fix these were byte-identical (0-pixel delta).
    assert not np.array_equal(without, with_hair)
    assert np.all(without[strand] == 0), "strand should start as hard background"
    assert np.all(with_hair[strand] == 128), "hair evidence must admit the strand"


def test_hair_mask_does_not_disturb_known_foreground_or_the_plain_path():
    fg, hair = _wisp_fixture()
    core = np.zeros((200, 200), bool)
    core[80:120, 80:120] = True

    with_hair = build_auto_trimap(fg, hair_mask=hair)
    assert np.all(with_hair[core] == 255), "known foreground must stay exact"

    # Supplying no hair mask must be byte-identical to the pre-fix behaviour.
    assert np.array_equal(build_auto_trimap(fg), build_auto_trimap(fg, hair_mask=None))


def test_refine_matte_locks_known_regions_and_stays_bounded():
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    image[:, 48:] = (180, 80, 40)
    mask = np.zeros((96, 96), np.float32)
    mask[18:78, 20:76] = 1.0
    result = refine_alpha_matte(image, mask, band_radius=3)
    assert result.alpha.dtype == np.float32
    assert np.isfinite(result.alpha).all()
    assert result.alpha.min() >= 0.0 and result.alpha.max() <= 1.0
    assert np.all(result.alpha[result.trimap == 0] == 0.0)
    assert np.all(result.alpha[result.trimap == 255] == 1.0)


def test_unusable_trimap_returns_input_mask_as_fallback():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.ones((32, 32), np.float32)
    result = refine_alpha_matte(image, mask)
    assert result.used_fallback
    assert np.array_equal(result.alpha, mask)

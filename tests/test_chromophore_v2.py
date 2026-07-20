"""Synthetic regression tests for the constrained chromophore v2 core."""

import numpy as np

from retouch.chromophore_v2 import (
    HEMOGLOBIN_PRIOR_RGB,
    MELANIN_PRIOR_RGB,
    decompose_chromophores_v2,
    delta_e_76,
    recompose_chromophores_v2,
    reduce_hemoglobin_variance,
    shift_hemoglobin,
)


def _linear_to_srgb(linear):
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _srgb_to_linear(srgb):
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )


def _prior_axis(axis):
    neutral = np.ones(3, dtype=np.float32) / np.sqrt(3.0)
    axis = axis - neutral * np.dot(axis, neutral)
    return axis / np.linalg.norm(axis)


def _synthetic_skin(h=64, w=80, illuminant=None):
    """Render known factor maps in the exact v2 optical-density model."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mel = 0.10 * np.sin(xx / 9.0) + 0.03 * np.cos(yy / 5.0)
    hb = 0.08 * np.cos(xx / 6.0) + 0.025 * np.sin(yy / 4.0)
    shade = 1.05 + 0.25 * (yy / max(h - 1, 1))
    neutral = np.ones(3, dtype=np.float32) / np.sqrt(3.0)
    od = (
        shade[..., None] * neutral
        + mel[..., None] * _prior_axis(MELANIN_PRIOR_RGB)
        + hb[..., None] * _prior_axis(HEMOGLOBIN_PRIOR_RGB)
    )
    if illuminant is not None:
        # A neutral scene-light factor changes only the neutral OD direction.
        od = od - np.log(float(illuminant))
    rgb_linear = np.exp(-od)
    return (_linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))[..., ::-1] * 255.0).astype(np.float32)


def test_recompose_is_finite_gamut_safe_and_deterministic():
    img = _synthetic_skin()
    first = decompose_chromophores_v2(img)
    second = decompose_chromophores_v2(img)
    out = recompose_chromophores_v2(first)
    assert np.array_equal(first.axes_rgb, second.axes_rgb)
    assert np.array_equal(first.melanin, second.melanin)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 255.0
    assert np.max(np.abs(out - img)) < 2e-3


def test_neutral_shading_does_not_change_chromophore_coordinates():
    reference = _synthetic_skin()
    shaded = _synthetic_skin(illuminant=0.72)
    a = decompose_chromophores_v2(reference)
    b = decompose_chromophores_v2(shaded)
    assert np.max(np.abs(a.melanin - b.melanin)) < 2e-4
    assert np.max(np.abs(a.hemoglobin - b.hemoglobin)) < 2e-4


def test_hemoglobin_edit_preserves_melanin_in_fixed_coordinate_system():
    img = _synthetic_skin()
    before = decompose_chromophores_v2(img)
    out = reduce_hemoglobin_variance(img, 0.75, decomposition=before)
    after = decompose_chromophores_v2(out, axes_rgb=before.axes_rgb)
    assert np.max(np.abs(after.melanin - before.melanin)) < 2e-4
    # AA5's perceptual ΔE cap may limit a full-strength slider move; the
    # invariant is a monotone variance reduction, not an uncapped percentage.
    assert after.hemoglobin.std() < before.hemoglobin.std()


def test_hemoglobin_shift_preserves_melanin_and_moves_relative_flush():
    img = _synthetic_skin()
    before = decompose_chromophores_v2(img)
    out = shift_hemoglobin(img, -0.6, decomposition=before)
    after = decompose_chromophores_v2(out, axes_rgb=before.axes_rgb)
    assert np.max(np.abs(after.melanin - before.melanin)) < 2e-4
    assert float(after.hemoglobin.mean()) < float(before.hemoglobin.mean())


def test_hemoglobin_edits_stay_inside_the_aa5_delta_e_budget():
    img = _synthetic_skin()
    mask = np.ones(img.shape[:2], dtype=np.float32)
    shifted = shift_hemoglobin(img, 1.0, skin_mask=mask)
    evened = reduce_hemoglobin_variance(img, 1.0, skin_mask=mask)

    assert float(delta_e_76(img, shifted, skin_mask=mask).max()) <= 2.401
    assert float(delta_e_76(img, evened, skin_mask=mask).max()) <= 2.401


def test_global_colour_cast_does_not_change_relative_hb_variance_or_axis_choice():
    img = _synthetic_skin()
    # A per-channel gain is a global OD offset. It should not perturb the
    # centred constrained-ICA axis estimate or the relative redness variance.
    rgb_linear = _srgb_to_linear(img[..., ::-1] / 255.0)
    cast_rgb = _linear_to_srgb(
        np.clip(rgb_linear * np.array([1.08, 0.90, 0.95], dtype=np.float32), 0.0, 1.0)
    )
    cast = (cast_rgb[..., ::-1] * 255.0).astype(np.float32)
    a = decompose_chromophores_v2(img)
    b = decompose_chromophores_v2(cast)
    assert np.max(np.abs(a.axes_rgb - b.axes_rgb)) < 2e-6
    assert abs(float(a.hemoglobin.std() - b.hemoglobin.std())) < 2e-4


def test_mask_and_zero_strength_are_exact_noops():
    img = _synthetic_skin()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    zero = reduce_hemoglobin_variance(img, 0.0)
    shift_zero = shift_hemoglobin(img, 0.0)
    empty = reduce_hemoglobin_variance(img, 0.7, skin_mask=mask)
    assert zero is img
    assert shift_zero is img
    assert empty is img


def test_small_uniform_image_reports_fallback_without_failure():
    img = np.full((8, 8, 3), 150, dtype=np.uint8)
    dec = decompose_chromophores_v2(img)
    assert dec.used_fallback
    assert dec.confidence == 0.0
    assert np.isfinite(recompose_chromophores_v2(dec)).all()

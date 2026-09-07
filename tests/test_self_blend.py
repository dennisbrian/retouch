import numpy as np
import pytest

from scripts.qa.p5_self_blend_experiment import blend as independent_blend
from retouch.self_blend import (
    ANALYTICAL_SELF_BLEND_MODES,
    apply_self_blend,
    self_blend_transfer,
)


def test_all_modes_are_finite_and_bounded():
    x = np.linspace(0, 1, 65537)
    for mode in ANALYTICAL_SELF_BLEND_MODES:
        y = self_blend_transfer(x, mode)
        assert np.isfinite(y).all()
        assert y.min() >= 0 and y.max() <= 1


def test_overlay_and_hard_light_self_transfer_are_identical():
    x = np.linspace(0, 1, 10001)
    np.testing.assert_array_equal(
        self_blend_transfer(x, "overlay"), self_blend_transfer(x, "hard_light")
    )


def test_self_transfer_matches_independent_two_input_blend_on_diagonal():
    x = np.linspace(0, 1, 257)
    for mode in ANALYTICAL_SELF_BLEND_MODES:
        expected = independent_blend(x, x, mode)
        np.testing.assert_allclose(
            self_blend_transfer(x, mode), expected, rtol=0, atol=2e-15,
        )


def test_overlay_and_hard_light_are_not_general_two_input_aliases():
    backdrop = np.array([0.25])
    source = np.array([0.75])
    overlay = independent_blend(backdrop, source, "overlay")
    hard_light = independent_blend(backdrop, source, "hard_light")
    assert overlay[0] == pytest.approx(0.375)
    assert hard_light[0] == pytest.approx(0.625)
    assert not np.array_equal(overlay, hard_light)


def test_vivid_light_and_linear_light_differ_inside_clipping_regions():
    x = np.array([0.4, 0.6])
    assert not np.allclose(
        self_blend_transfer(x, "vivid_light"), self_blend_transfer(x, "linear_light")
    )


def test_boundaries_and_endpoints():
    x = np.array([0, 1 / 3, 0.5, 2 / 3, 1], dtype=np.float64)
    for mode in ANALYTICAL_SELF_BLEND_MODES:
        y = self_blend_transfer(x, mode)
        assert y[0] == pytest.approx(0)
        assert y[-1] == pytest.approx(0 if mode == "exclusion" else 1)
    np.testing.assert_allclose(self_blend_transfer(x, "linear_light"), [0, 0, .5, 1, 1])
    np.testing.assert_allclose(self_blend_transfer(x, "vivid_light"), [0, 0, .5, 1, 1])


def test_screen_192_is_239_not_historical_247():
    output = self_blend_transfer(192 / 255, "screen") * 255
    assert output == pytest.approx(239.43529411764706)
    assert int(np.floor(output + .5)) == 239


def test_uint8_quantization_and_amount_identity():
    image = np.array([[[0, 64, 128], [192, 255, 85]]], dtype=np.uint8)
    np.testing.assert_array_equal(apply_self_blend(image, "multiply", 0), image)
    output = apply_self_blend(image, "screen", 1)
    np.testing.assert_array_equal(output[0, 0], [0, 112, 192])
    assert output.dtype == np.uint8

    image16 = np.array([[[0, 16384, 32768], [49152, 65535, 21845]]], dtype=np.uint16)
    output16 = apply_self_blend(image16, "multiply", 1.0)
    assert output16.dtype == np.uint16
    np.testing.assert_array_equal(output16[0, 0], [0, 4096, 16384])


@pytest.mark.parametrize("amount", [0.25, 0.5, 0.75, 1.0])
def test_amount_is_linear_opacity_in_selected_encoded_domain(amount):
    image = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float64)
    expected = (1.0 - amount) * image + amount * self_blend_transfer(image, "screen")
    np.testing.assert_allclose(
        apply_self_blend(image, "screen", amount), expected, rtol=0, atol=1e-14,
    )


def test_float_preserves_dtype_and_domain_is_explicit():
    image = np.array([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    np.testing.assert_array_equal(
        apply_self_blend(image, "multiply", 0.0, domain="linear"), image
    )
    encoded = apply_self_blend(image, "multiply", .5)
    linear = apply_self_blend(image, "multiply", .5, domain="linear")
    assert encoded.dtype == linear.dtype == np.float32
    assert not np.allclose(encoded, linear)


def test_invalid_inputs_and_singular_endpoints_are_safe():
    for mode in ("color_dodge", "color_burn", "vivid_light"):
        y = self_blend_transfer(np.array([0., np.nextafter(0., 1.), 1., np.nextafter(1., 0.)]), mode)
        assert np.isfinite(y).all()
    with pytest.raises(ValueError):
        self_blend_transfer(np.array([np.nan]), "screen")
    with pytest.raises(ValueError):
        apply_self_blend(np.zeros((2, 2, 3), dtype=np.uint8), "screen", 1.1)
    with pytest.raises(ValueError):
        apply_self_blend(np.zeros((2, 2, 3), dtype=np.uint8), "screen", [0.5])

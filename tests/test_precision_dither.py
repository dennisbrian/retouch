"""Regression tests for the opt-in final-delivery uint8 dither path."""

import numpy as np

from retouch.precision import (
    blue_noise_lsb_noise,
    to_uint8,
    to_uint8_dithered,
)


def test_dithered_uint8_input_is_byte_identical() -> None:
    rng = np.random.default_rng(41)
    image = rng.integers(0, 256, (19, 31, 3), dtype=np.uint8)

    result = to_uint8_dithered(image)

    assert np.array_equal(result, image)
    assert result is not image


def test_dither_tile_is_deterministic_and_tiled_by_origin() -> None:
    first = blue_noise_lsb_noise((64, 64))
    second = blue_noise_lsb_noise((64, 64))
    repeated = blue_noise_lsb_noise((128, 128))
    shifted = blue_noise_lsb_noise((64, 64), origin=(64, 64))

    assert np.array_equal(first, second)
    assert np.array_equal(first, shifted)
    assert np.array_equal(first, repeated[:64, :64])
    assert np.array_equal(first, repeated[64:, 64:])


def test_dither_is_zero_mean_and_strictly_sub_lsb() -> None:
    noise = blue_noise_lsb_noise((64, 64))

    assert np.isclose(float(noise.mean()), 0.0, atol=1e-7)
    assert float(noise.min()) > -0.5
    assert float(noise.max()) < 0.5


def test_half_density_tile_suppresses_low_frequency_energy() -> None:
    """Void-and-cluster half tones must not regress into ordered/white noise."""
    pattern = (blue_noise_lsb_noise((64, 64)) < 0.0).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(pattern - pattern.mean()))) ** 2
    y, x = np.indices(pattern.shape)
    radius = np.hypot(y - 32, x - 32)
    low = spectrum[(radius > 0) & (radius <= 6)].mean()
    high = spectrum[radius >= 14].mean()

    # The generated 50% threshold tile measures about 0.009.  A generous
    # ceiling protects the blue-noise spectral notch without overfitting.
    assert float(low / high) < 0.05


def test_dithered_conversion_clips_to_uint8_gamut() -> None:
    image = np.array(
        [[[-0.1, 0.0, 0.1], [0.5, 1.0, 1.1]]], dtype=np.float32
    )

    result = to_uint8_dithered(image)

    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_dither_decorrelates_smooth_ramp_quantization() -> None:
    """A flat vertical ramp should not quantize to identical rows everywhere."""
    # Span about sixteen display codes across a deliberately long smooth ramp.
    ramp = np.linspace(100.1 / 255.0, 115.9 / 255.0, 512, dtype=np.float32)
    image = np.broadcast_to(ramp[np.newaxis, :, np.newaxis], (64, 512, 1))

    undithered = to_uint8(image)[..., 0]
    dithered = to_uint8_dithered(image)[..., 0]
    delta = dithered.astype(np.int16) - undithered.astype(np.int16)

    # Normal conversion makes every row identical, creating coherent bands.
    assert np.unique(undithered, axis=0).shape[0] == 1
    # The threshold tile breaks that coherence without a multi-code jump.
    assert np.unique(dithered, axis=0).shape[0] > 16
    assert np.count_nonzero(delta) > 0
    assert int(np.abs(delta).max()) <= 1

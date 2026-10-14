"""Grain and Add Noise adjustment kinds: records, kernels, band invariance."""
import numpy as np
import pytest
from pathlib import Path

from retouch.editor import (Document, DocumentHistory, Layer, UnsupportedFeature,
                            composite, load_project)
from retouch.editor.adjustments import adjustment, grade, validate_adjustment
from retouch.editor.selections import selection


def solid(color, width=4, height=4, alpha=255):
    a = np.empty((height, width, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a


# --- Grain and Add Noise -------------------------------------------------------
#
# Native record shapes and the C kernel ports (Rendering/AdjustPixels.c and
# NoisePixels.c at 11d8d7a). Grain: amount 0-100, size 0.5-20 (document
# pixels), roughness 0-100, seed uint32. Add Noise: amount 0.1-400,
# gaussian, monochromatic, seed uint32. Both patterns stay fixed in the
# document however the canvas splits its drawing.

@pytest.mark.parametrize('kind', ['Grain', 'Add Noise'])
def test_grain_noise_identity_at_zero_strength(kind):
    # Grain amount=0 is the kernel's own early-out; Add Noise has no zero
    # (minimum 0.1), so identity is only claimed for Grain.
    if kind == 'Add Noise':
        pytest.skip('Add Noise amount floor is 0.1, never identity')
    gray = np.full((1, 1, 3), .5, np.float32)
    result = grade(gray, adjustment('Grain', amount=0))
    np.testing.assert_array_equal(result, gray)


def test_grain_seed_changes_pattern_and_amount_scales_it():
    base = np.full((8, 8, 3), .5, np.float32)
    one = grade(base, adjustment('Grain', amount=30, seed=1))
    two = grade(base, adjustment('Grain', amount=30, seed=2))
    assert not np.array_equal(one, two)
    stronger = grade(base, adjustment('Grain', amount=90, seed=1))
    assert abs(stronger - .5).max() > abs(one - .5).max()


@pytest.mark.parametrize('gaussian, monochromatic', [(False, False), (False, True), (True, False), (True, True)])
def test_add_noise_variants_change_colour_and_keep_range(gaussian, monochromatic):
    gray = np.full((8, 8, 3), .5, np.float32)
    result = grade(gray, adjustment('Add Noise', amount=50, gaussian=gaussian,
                                    monochromatic=monochromatic, seed=7))
    assert not np.array_equal(result, gray)
    assert result.min() >= 0 and result.max() <= 1
    if monochromatic:
        # Monochromatic shifts brightness only: every channel moves together.
        np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-6)
        np.testing.assert_allclose(result[..., 1], result[..., 2], atol=1e-6)


@pytest.mark.parametrize('kind, kwargs, bad', [
    ('Grain', dict(amount=101), ValueError),
    ('Grain', dict(size=.4), ValueError),
    ('Grain', dict(size=20.5), ValueError),
    ('Grain', dict(roughness=101), ValueError),
    ('Grain', dict(seed=-1), ValueError),
    ('Grain', dict(seed=2**32), ValueError),
    ('Add Noise', dict(amount=.05), ValueError),
    ('Add Noise', dict(amount=401), ValueError),
    ('Add Noise', dict(seed=2**32), ValueError),
    ('Add Noise', dict(gaussian=1), ValueError),
])
def test_grain_noise_validation_rejects_out_of_range(kind, kwargs, bad):
    with pytest.raises(bad):
        adjustment(kind, **kwargs)


def test_grain_noise_settings_do_not_ride_other_kinds():
    record = adjustment('Grain', amount=50, seed=3)
    record['noiseAmount'] = 10
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)
    record = adjustment('Add Noise', amount=50)
    record['grainSettings'] = dict(amount=1, size=1, roughness=0, seed=0)
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)


@pytest.mark.parametrize('kind', ['Grain', 'Add Noise'])
def test_grain_noise_kind_now_supported_but_others_still_refused(kind):
    from retouch.editor.adjustments import KINDS
    assert kind in KINDS
    record = dict(kind='Gaussian Blur', hue=0, saturation=0, lightness=0, colorize=False,
                  range='master', invertRange=False,
                  levels=dict(channel='RGB', ranges=[dict(black=0, white=255, gamma=1,
                                                          outputBlack=0, outputWhite=255)] * 4),
                  curves=dict(channel='RGB',
                              channels=[[dict(x=0, y=0), dict(x=255, y=255)]] * 4))
    with pytest.raises(UnsupportedFeature, match='Supported adjustments'):
        validate_adjustment(record)


@pytest.mark.parametrize('kind, kwargs', [
    ('Grain', dict(amount=40, size=3, roughness=10, seed=9)),
    ('Add Noise', dict(amount=25, gaussian=True, monochromatic=True, seed=123456)),
])
def test_grain_noise_adjustment_layer_round_trips(kind, kwargs, tmp_path):
    base = Layer('Base', solid((80, 100, 120), alpha=128))
    settings = adjustment(kind, **kwargs)
    adj = Layer(kind, size=(4, 4), adjustment=settings,
                mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4), opacity=.5)
    h = DocumentHistory(Document(4, 4, [base]))
    before = composite(h.document)
    h.add_layer(adj)
    image = composite(h.document)
    np.testing.assert_array_equal(image[..., 3], before[..., 3])
    np.testing.assert_array_equal(image[:, 2:], before[:, 2:])
    assert not np.array_equal(image[:, :2], before[:, :2])
    h.save(tmp_path / (kind.replace(' ', '_') + '.comp'))
    again = load_project(tmp_path / (kind.replace(' ', '_') + '.comp'))
    assert again.layers[-1].adjustment == settings
    np.testing.assert_array_equal(composite(again), image)
    h.undo()
    np.testing.assert_array_equal(composite(h.document), before)


@pytest.mark.parametrize('kind, kwargs', [
    ('Grain', dict(amount=40, size=2.0, roughness=50, seed=12345)),
    ('Add Noise', dict(amount=40, gaussian=True, monochromatic=False, seed=777)),
])
def test_grain_noise_band_invariance(kind, kwargs):
    # The pattern is placed in document space, so slicing the canvas into
    # bands must not move it: full render == band render bit for bit.
    rng = np.random.default_rng(4)
    base = np.zeros((53, 16, 4), np.uint8)
    for c in range(3):
        base[..., c] = rng.integers(0, 256, (53, 16)).astype(np.uint8)
    base[..., 3] = (128 + rng.integers(0, 128, (53, 16))).astype(np.uint8)
    doc = Document(16, 53, [Layer('Base', base),
                            Layer(kind, size=(16, 53), adjustment=adjustment(kind, **kwargs))])
    full = composite(doc)
    for band_rows in (16, 7, 1):
        np.testing.assert_array_equal(full, composite(doc, band_rows=band_rows))


# --- Grain / Add Noise oracle values (compiled C kernels at 11d8d7a) ------------
#
# Expected bytes come from Rendering/AdjustPixels.c adjust_grain and
# NoisePixels.c noise_add_at compiled and run on the same input buffer; the
# port must match them exactly (the band path re-enters the C kernels' byte
# domain through the same 255/alpha premultiply chain).

def _grain_noise_buffer():
    # The 24x24/16x16 buffers the C harness generated with its own
    # xorshift (bufseed 7): reproduced deterministically here.
    U32 = np.uint32
    s = U32(7)
    def rng():
        nonlocal s
        s = U32((s ^ (U32(s << U32(13)))) & 0xFFFFFFFF)
        s = U32((s ^ (s >> U32(17))) & 0xFFFFFFFF)
        s = U32((s ^ (U32(s << U32(5)))) & 0xFFFFFFFF)
        return s
    buf = np.zeros((24, 24, 4), np.uint8)
    for y in range(24):
        for x in range(24):
            buf[y, x, 0] = rng() & 0xFF
            buf[y, x, 1] = rng() & 0xFF
            buf[y, x, 2] = rng() & 0xFF
            buf[y, x, 3] = 128 + (int(rng()) % 128)
    return buf


def _grain_oracle_bytes():
    """First rows of adjust_grain(seed=12345, amount=40, size=2, rough=50)."""
    try:
        text = (Path(__file__).parent / 'fixtures' / 'grain_noise_reference.txt').read_text()
        return {name: np.frombuffer(bytes.fromhex(hexs), np.uint8)
                .reshape(tuple(int(d) for d in shape.split(',')))
                for name, shape, hexs in (line.split() for line in text.splitlines())}
    except FileNotFoundError:
        return {}


@pytest.mark.skipif(not (Path(__file__).parent / 'fixtures' / 'grain_noise_reference.txt').exists(),
                    reason='kernel fixture not generated yet')
def test_grain_noise_match_native_kernel_fixture():
    reference = _grain_oracle_bytes()
    buf = _grain_noise_buffer()
    rgb = buf[..., :3].astype(np.float32) / np.float32(255.0)
    a = buf[..., 3].astype(np.float32) / np.float32(255.0)
    from retouch.editor.adjustments import _grade_grain, _grade_noise
    grain = _grade_grain(rgb, a, dict(amount=40, size=2.0, roughness=50, seed=12345),
                         0.0, 0.0, 1.0)
    np.testing.assert_array_equal(np.rint(grain * 255).astype(np.uint8),
                                  reference['grain'][..., :3])
    noise = _grade_noise(rgb, a, dict(noiseAmount=25.0, noiseGaussian=False,
                                      noiseMonochromatic=False, noiseSeed=7), 0, 0)
    np.testing.assert_array_equal(np.rint(noise * 255).astype(np.uint8),
                                  reference['noise'][..., :3])

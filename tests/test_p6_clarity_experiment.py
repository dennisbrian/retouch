"""Offline P6 harness checks; no RetouchEngine construction or model loading."""
import numpy as np
import pytest

from scripts.qa import p6_clarity_experiment as p


@pytest.mark.parametrize('value', p.VALUES)
def test_actual_float_production_parity(value):
    image = np.random.default_rng(7).uniform(.05, .95, (123, 151, 3)).astype(np.float32)
    actual = p.GRADER._F_add_clarity(image, value/100)
    assert np.array_equal(p.apply(p.prepare(image), 'production_current', value), actual)


@pytest.mark.parametrize('arm', p.ARMS)
def test_zero_and_finite(arm):
    image = np.random.default_rng(2).uniform(0, 1, (96, 104, 3)).astype(np.float32)
    b = p.prepare(image)
    if arm != 'conversion_only':
        assert np.array_equal(p.apply(b, arm, 0), image)
    for value in (4, 10):
        out = p.apply(b, arm, value)
        assert out.dtype == np.float32 and np.isfinite(out).all()
        assert 0 <= out.min() <= out.max() <= 1


def test_flat_does_not_create_detail():
    image = np.full((100, 100, 3), .5, np.float32)
    b = p.prepare(image)
    assert abs(b['detail']).max() < 1e-6
    for arm in p.ARMS[3:]:
        assert abs(p.apply(b, arm, 10)-image).max() < 1e-6


def test_deterministic_fixture_coverage():
    a, b = p.synthetic(11, 128), p.synthetic(11, 128)
    assert len(a) == 22 and len({v[0] for v in a}) == 22
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert {'noise', 'compression', 'quantization', 'smooth', 'structure', 'edge'} == {x[3] for x in a}
    assert all(np.isfinite(x[1]).all() and x[1].shape == (128, 128, 3) for x in a)


def test_noise_amplification_and_gate_tradeoff_measurements():
    _, image, clean, kind = next(x for x in p.synthetic(11, 128) if x[0] == 'gaussian_2DN')
    rows = p.evaluate(image, clean, kind, {}, values=(4,), arms=('disabled', 'matched_current', 'variance_snr'))
    ratio = rows[1]['paired_noise_rms_DN']/rows[0]['paired_noise_rms_DN']
    assert 1.02 < ratio < 1.06
    assert rows[2]['paired_noise_rms_DN'] < rows[1]['paired_noise_rms_DN']
    assert rows[1]['haar_MAD_DN'] > rows[0]['haar_MAD_DN']


def test_fixed_sensitivity_interior_and_halos():
    _, image, clean, kind = next(x for x in p.synthetic(11) if x[0] == 'hard_edge')
    rows = p.evaluate(image, clean, kind, {'window': 58}, values=(10,),
                      arms=('production_current',), metric_pad=116)
    assert all(np.isfinite(v) for v in rows[0].values() if isinstance(v, (float, int)))
    assert rows[0]['overshoot_DN'] > 0  # Midrange edge: clipping cannot hide halos.


def test_padded_context_matches_full_operator_at_fixed_window():
    image = np.random.default_rng(7).uniform(.1, .9, (256, 300, 3)).astype(np.float32)
    full = p.apply(p.prepare(image, window=15), 'production_current', 4)
    crop = p.apply(p.prepare(image[40:220, 40:260], window=15), 'production_current', 4)
    np.testing.assert_allclose(crop[40:140, 40:180], full[80:180, 80:220], atol=1e-6)


def test_plane_detrending_and_block_phase():
    yy, xx = np.mgrid[:64, :64]
    plane = 12+.2*xx+.7*yy
    assert abs(p.plane_residual(plane)).max() < 1e-12
    blocked = p.rgb_gray((.4+(xx//8)*.005).astype(np.float32))
    aligned = p.measures(blocked)['block_excess_DN2']
    shifted = p.measures(blocked[:, 4:60], origin=(4, 0))['block_excess_DN2']
    assert abs(aligned-(.005*255)**2/2) < 1e-4 and abs(aligned-shifted) < 1e-4


def test_zero_signal_gate_and_quantized_identity():
    zero = np.zeros((48, 48), np.float32)
    assert np.array_equal(p.gate_variance(zero, .01), zero)
    image = p.rgb_gray(zero+.5)
    result = p.quantized_change(image, image)
    assert result['changed_u8_pixel_fraction'] == result['quantized_change_rms_DN'] == 0


def test_reject_overwrite_and_small_interior(tmp_path):
    with pytest.raises(ValueError, match='overwrite'):
        p.run(tmp_path)
    image = np.full((32, 32, 3), .5, np.float32)
    with pytest.raises(ValueError, match='small'):
        p.evaluate(image, image, 'smooth', {}, metric_pad=16)


def test_matched_noise_counterfactual_identity(monkeypatch):
    synthetic = p.synthetic
    monkeypatch.setattr(p, 'synthetic', lambda seed: synthetic(seed, 128))
    monkeypatch.setattr(p, 'SEEDS', (11,))
    monkeypatch.setattr(p, 'VALUES', (0, 4))
    monkeypatch.setattr(p, 'ARMS', ('disabled', 'matched_current', 'variance_snr'))
    rows = p.mixed_detail_probe()
    assert len(rows) == 24
    for row in rows:
        assert np.isfinite(row['structure_in_noise_projection_gain'])
        if row['arm'] == 'disabled' or row['recipe_clarity'] == 0:
            assert abs(row['structure_in_noise_projection_gain']-1) < 1e-5

"""Isolated P5 research tests. Synthetic exports here are NEVER Photoshop evidence."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import ImageCms
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p5", ROOT / "scripts/qa/p5_self_blend_experiment.py")
p5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p5)


def scalar_definition(b, s, mode):
    """Literal scalar reference, independently transcribed from cited definitions.

    Eight modes: W3C Compositing-1 section 10.1. LL/VL: bounded canonical
    definitions in the P5 report, not a normative Adobe implementation.
    """
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return 1 - (1-b)*(1-s)
    if mode == "overlay":
        return 2*b*s if b <= .5 else 1-2*(1-b)*(1-s)
    if mode == "hard_light":
        return 2*b*s if s <= .5 else 1-2*(1-b)*(1-s)
    if mode == "soft_light":
        if s <= .5:
            return b-(1-2*s)*b*(1-b)
        d = 16*b**3-12*b**2+4*b if b <= .25 else math.sqrt(b)
        return b+(2*s-1)*(d-b)
    if mode == "color_dodge":
        if b == 0:
            return 0
        if s == 1:
            return 1
        return min(1, b/(1-s))
    if mode == "color_burn":
        if b == 1:
            return 1
        if s == 0:
            return 0
        return 1-min(1, (1-b)/s)
    if mode == "exclusion":
        return b+s-2*b*s
    if mode == "linear_light":
        return min(1, max(0, b+2*s-1))
    if mode == "vivid_light":
        return scalar_definition(b, 2*s, "color_burn") if s <= .5 else scalar_definition(b, 2*s-1, "color_dodge")
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", p5.MODES)
def test_two_input_definition_all_8bit_pairs(mode):
    values = np.arange(256)/255
    b, s = np.meshgrid(values, values, indexing="ij")
    expected = np.array([scalar_definition(float(bb), float(ss), mode)
                         for bb, ss in zip(b.ravel(), s.ravel())]).reshape(b.shape)
    np.testing.assert_allclose(p5.blend(b, s, mode), expected, atol=8e-16, rtol=0)


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 1e-12), (np.float32, 5e-7)])
@pytest.mark.parametrize("mode", p5.MODES)
def test_dense_self_reduction_and_nextafter(mode, dtype, tolerance):
    x = np.unique(np.r_[np.linspace(0, 1, 65536, dtype=dtype), p5.boundary_probes(dtype)])
    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        actual = p5.self_transfer(x, mode)
        np.testing.assert_allclose(actual, p5.blend(x, x, mode), rtol=0, atol=tolerance)
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    for boundary in (0, .25, 1/3, .5, 2/3, 1):
        assert dtype(boundary) in x


def test_equivalences_and_non_equivalences():
    x = np.linspace(0, 1, 10001)
    np.testing.assert_array_equal(p5.self_transfer(x, "overlay"), p5.self_transfer(x, "hard_light"))
    assert p5.blend(.25, .75, "overlay") == .375
    assert p5.blend(.25, .75, "hard_light") == .625
    for x in (.4, .6):
        assert not np.isclose(p5.self_transfer(x, "vivid_light"), p5.self_transfer(x, "linear_light"))
    for x in (0, 1/3, .5, 2/3, 1):
        assert p5.self_transfer(x, "vivid_light") == pytest.approx(p5.self_transfer(x, "linear_light"))


@pytest.mark.parametrize("boundary", [1/3, .5, 2/3])
def test_vivid_boundary_neighbors(boundary):
    x = np.array([np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1)])
    y = p5.self_transfer(x, "vivid_light")
    np.testing.assert_allclose(y, p5.blend(x, x, "vivid_light"), atol=5e-16, rtol=0)
    assert np.all(np.diff(y) >= 0)
    assert np.all(p5.self_transfer(x[x <= 1/3], "vivid_light") == 0) if np.any(x <= 1/3) else True
    assert np.all(p5.self_transfer(x[x >= 2/3], "vivid_light") == 1) if np.any(x >= 2/3) else True


def test_singular_precedence_and_soft_light_polynomial():
    assert p5.blend(0, 1, "color_dodge") == 0
    assert p5.blend(1, 0, "color_burn") == 1
    probes = p5.boundary_probes()
    b, s = np.meshgrid(probes, probes)
    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        for mode in ("color_dodge", "color_burn", "vivid_light"):
            assert np.isfinite(p5.blend(b, s, mode)).all()
    for b in (.01, .125, .25, np.nextafter(.25, 1)):
        assert p5.blend(b, .8, "soft_light") == pytest.approx(scalar_definition(b, .8, "soft_light"), abs=1e-15)
    assert not np.isclose(p5.blend(.01, .8, "soft_light"), .01+.6*(math.sqrt(.01)-.01))


@pytest.mark.parametrize("bits", [8, 16])
def test_integer_boundary_and_midpoint_records(bits):
    maximum = 2**bits-1
    codes = sorted({0, maximum} | {int(maximum*b)+d for b in (.25, 1/3, .5, 2/3) for d in (-1, 0, 1, 2)})
    for code in codes:
        for mode in p5.MODES:
            assert float(p5.self_transfer(code/maximum, mode))*maximum == pytest.approx(
                scalar_definition(code/maximum, code/maximum, mode)*maximum, abs=1e-10)
    midpoint = (maximum+1)//2
    assert midpoint/maximum > .5
    assert int(p5.quantize(p5.self_transfer(midpoint/maximum, "linear_light"), bits)) == midpoint+1


def test_historical_points_are_not_diagram_fits():
    assert float(p5.self_transfer(192/255, "screen"))*255 == pytest.approx(239.43529411764706)
    checks = [("multiply", 64, 16), ("overlay", 128, 128), ("soft_light", 51, 27),
              ("color_dodge", 85, 128), ("color_burn", 128, 2), ("exclusion", 128, 127),
              ("linear_light", 128, 129), ("vivid_light", 128, 129)]
    for mode, code, expected in checks:
        assert int(p5.quantize(p5.self_transfer(code/255, mode), 8)) == expected


@pytest.mark.parametrize("model", p5.MODELS)
@pytest.mark.parametrize("mode", p5.MODES)
def test_opacity_domains_and_simple_mask(mode, model):
    x = np.linspace(0, 1, 999).reshape(3, 111, 3)
    original = x.copy()
    np.testing.assert_array_equal(p5.predict(x, mode, 0, model), x)
    np.testing.assert_array_equal(p5.predict(x, mode, .75, model, mask=np.zeros(x.shape[:-1])), x)
    b = p5.decode(x) if model in ("linear", "linear_blend_encoded_opacity") else x
    operator = p5.blend(b, b, mode)
    encoded_operator = p5.encode(operator) if model in ("linear", "linear_blend_encoded_opacity") else operator
    np.testing.assert_allclose(p5.predict(x, mode, 1, model), encoded_operator, rtol=0, atol=1e-14)
    for p in p5.OPACITIES:
        if model in ("linear", "encoded_blend_linear_opacity"):
            expected = p5.encode((1-p)*p5.decode(x)+p*p5.decode(encoded_operator))
        else:
            expected = (1-p)*x+p*encoded_operator
        np.testing.assert_allclose(p5.predict(x, mode, p, model), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(p5.predict(x, mode, 1, model, mask=np.full(x.shape[:-1], .25)), p5.predict(x, mode, .25, model))
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("mode", p5.MODES)
def test_lut_off_grid_error_and_identity_nodes(mode):
    x = np.linspace(0, 1, 200003)
    coarse = np.max(np.abs(p5.lookup(x, mode, 4096)-p5.self_transfer(x, mode)))
    fine = np.max(np.abs(p5.lookup(x, mode, 65536)-p5.self_transfer(x, mode)))
    assert coarse*255 < .07
    assert fine*65535 < 1.01
    assert fine <= coarse+1e-15
    nodes = np.linspace(0, 1, 4096)
    np.testing.assert_array_equal(p5.lookup(nodes, mode, 4096), p5.self_transfer(nodes, mode))


def test_exclusion_may_descend_without_spline_constraints():
    x = np.linspace(0, 1, 1000)
    assert np.all(np.diff(p5.predict(x, "exclusion", .25)) > 0)
    for p in (.5, .75, 1):
        assert np.any(np.diff(p5.predict(x, "exclusion", p)) < 0)


def synthetic_set(root, full=False, profile=True):
    root.mkdir()
    # Pixel oracle fixture only: NEVER presented as an actual Photoshop capture.
    data = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (1, 1, 3))
    if full:
        ramp = np.arange(256, dtype=np.uint8)
        pure = np.zeros((3, 256, 3), dtype=np.uint8)
        for channel in range(3):
            pure[channel, :, channel] = ramp
        data = np.concatenate((data, pure))
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes() if profile else None
    p5.save_rgb(root/"baseline.png", data, icc)
    p5.save_rgb(root/"normal_100.png", data, icc)
    pairs = [(m, p) for m in p5.MODES for p in p5.OPACITIES] if full else [("multiply", .5)]
    for mode, opacity in pairs:
        prediction = p5.quantize(p5.predict(data.astype(float)/255, mode, opacity), 8)
        p5.save_rgb(root/f"{mode}_{int(opacity*100):03d}.png", prediction, icc)
    return data


def test_discovery_and_unattested_comparison(tmp_path):
    root = tmp_path/"external"
    synthetic_set(root)
    _, capture, metadata, missing = p5.discover_goldens(root)
    assert metadata is None and len(capture["cases"]) == 1 and len(missing) == 39
    p5.compare(root, tmp_path/"comparison")
    report = json.loads((tmp_path/"comparison/photoshop_comparison.json").read_text())
    assert not report["photoshop_verified"]
    assert report["evidence_class"] == "EXTERNAL_IMAGE_COMPARISON_PHOTOSHOP_UNVERIFIED"
    assert len(report["results"]) == 8
    assert report["results"][0]["max_error_DN"] == 0
    with pytest.raises(ValueError, match="Incomplete"):
        p5.compare(root, tmp_path/"strict", require_complete=True)


def test_full_filename_set_and_provenance_downgrade(tmp_path):
    root = tmp_path/"mock_capture"
    synthetic_set(root, full=True)
    # Deliberately false provenance fixture exercises the gate; it is NOT evidence.
    metadata = dict(producer="external_photoshop_exports", capture_status="complete", photoshop_version="MOCK TEST ONLY",
                    document_profile="sRGB IEC61966-2.1", capture_notes="Synthetic gate test NOT Photoshop",
                    operator="unit test", settings_confirmed=True, blend_gamma_setting={"enabled": False},
                    settings_evidence="settings.txt", fill_opacity=100, opaque_equal_layers=True,
                    blend_if="disabled", layer_effects=False, masks="none", bits=8)
    (root/"settings.txt").write_text("Synthetic test attestation, not a screenshot")
    (root/"external_golden_metadata.json").write_text(json.dumps(metadata))
    _, capture, _, missing = p5.discover_goldens(root)
    assert not missing
    assert not p5.assess_capture(capture, root, missing)[2]
    p5.compare(root, tmp_path/"gate_only")
    report = json.loads((tmp_path/"gate_only/photoshop_comparison.json").read_text())
    assert len(report["results"]) == 320 and report["photoshop_verified"]
    metadata["blend_gamma_setting"] = {"enabled": "UNKNOWN", "gamma": "UNAVAILABLE"}
    (root/"external_golden_metadata.json").write_text(json.dumps(metadata))
    p5.compare(root, tmp_path/"downgrade")
    report = json.loads((tmp_path/"downgrade/photoshop_comparison.json").read_text())
    assert not report["photoshop_verified"] and report["results"][0]["max_error_DN"] == 0


@pytest.mark.parametrize("problem", ["missing_baseline", "empty", "duplicate", "bad_name", "shape", "cases", "escape"])
def test_malformed_sets(tmp_path, problem):
    root = tmp_path/"bad"
    data = synthetic_set(root)
    if problem == "missing_baseline":
        (root/"baseline.png").unlink()
    elif problem == "empty":
        (root/"multiply_050.png").unlink()
    elif problem == "duplicate":
        shutil.copyfile(root/"multiply_050.png", root/"multiply_050.tif")
    elif problem == "bad_name":
        (root/"multiply_050.png").rename(root/"multiply_50.png")
    elif problem == "shape":
        p5.save_rgb(root/"multiply_050.png", data[:, :10])
    elif problem == "cases":
        (root/"external_golden_metadata.json").write_text('{"cases": [{"mode": "multiply"}]}')
    else:
        (root/"external_golden_metadata.json").write_text('{"baseline": "../escape.png"}')
    with pytest.raises(ValueError):
        p5.compare(root, tmp_path/"no_output")
    assert not (tmp_path/"no_output").exists()


def test_profile_and_normal_control_downgrade(tmp_path):
    root = tmp_path/"bad_profile"
    data = synthetic_set(root)
    p5.save_rgb(root/"multiply_050.png", data)  # untagged and wrong output; still scoreable
    p5.save_rgb(root/"normal_100.png", np.zeros_like(data))
    _, capture, _, missing = p5.discover_goldens(root)
    issues = p5.assess_capture(capture, root, missing)[2]
    assert any("ICC mismatch" in i for i in issues)
    assert any("Normal duplicate" in i for i in issues)


def test_invalid_metadata_is_numeric_comparison_only(tmp_path):
    root = tmp_path/"bad_metadata"
    synthetic_set(root)
    (root/"capture_metadata.json").write_text('{"producer": invalid json')
    p5.compare(root, tmp_path/"still_comparable")
    report = json.loads((tmp_path/"still_comparable/photoshop_comparison.json").read_text())
    assert not report["photoshop_verified"]
    assert report["capture_metadata"]["metadata_read_error"]
    assert report["results"][0]["max_error_DN"] == 0


def test_ramp_coverage_required_for_claim(tmp_path):
    root = tmp_path/"no_color_ramps"
    synthetic_set(root)
    _, capture, _, missing = p5.discover_goldens(root)
    assert sum("pure" in issue for issue in p5.assess_capture(capture, root, missing)[2]) == 3


@pytest.mark.parametrize("bad", [[-0.01], [1.01], [float('nan')], [float('inf')], []])
def test_invalid_input_rejected_not_clamped(bad):
    with pytest.raises(ValueError):
        p5.self_transfer(bad, "color_dodge")


def test_srgb_and_power_gamma_transfer():
    # Piecewise sRGB transfer from CSS Color 4, not a single gamma approximation.
    x = np.array([0, .04045, .1, .5, 1])
    expected = np.array([v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in x])
    np.testing.assert_allclose(p5.decode(x), expected, atol=1e-16, rtol=0)
    np.testing.assert_allclose(p5.encode(p5.decode(x)), x, atol=3e-8, rtol=0)
    np.testing.assert_allclose(p5.encode(p5.decode(x, 'gamma', 2.2), 'gamma', 2.2), x, atol=1e-15)
    assert not np.allclose(p5.decode(x), p5.decode(x, 'gamma', 2.2))


def test_16bit_external_comparison_and_optional_error_images(tmp_path):
    root = tmp_path/'external16'
    root.mkdir()
    data = np.repeat(np.array([0, 21844, 21845, 21846, 32767, 32768, 43689, 43690, 43691, 65535], dtype=np.uint16)[None, :, None], 3, axis=2)
    p5.save_rgb(root/'baseline.png', data)
    prediction = p5.quantize(p5.predict(data.astype(float)/65535, 'vivid_light', .75), 16)
    p5.save_rgb(root/'vivid_light_075.png', prediction)
    p5.compare(root, tmp_path/'scored16', save_error_images=True)
    report = json.loads((tmp_path/'scored16/photoshop_comparison.json').read_text())
    assert report['bits'] == 16 and not report['photoshop_verified']
    assert report['results'][0]['max_error_DN'] == 0
    assert len(list((tmp_path/'scored16').glob('*_error_DN.npy'))) == 8


def test_missing_declared_control_downgrades_and_source_change_detected(tmp_path):
    root = tmp_path/'source_change'
    data = synthetic_set(root)
    p5.save_rgb(root/'source.png', np.zeros_like(data))
    metadata = dict(normal_control='missing.tif', source_fixture='source.png')
    (root/'external_golden_metadata.json').write_text(json.dumps(metadata))
    p5.compare(root, tmp_path/'scored')
    report = json.loads((tmp_path/'scored/photoshop_comparison.json').read_text())
    assert not report['photoshop_verified']
    assert any('Source fixture changed' in issue for issue in report['provenance_issues'])
    assert any('Normal duplicate control missing' in issue for issue in report['provenance_issues'])


def test_mismatch_locations_bounded_and_signed():
    baseline = np.full((4, 10, 3), 123, dtype=np.uint8)
    actual = np.full_like(baseline, 10)
    predicted = np.full_like(baseline, 8)
    predicted[0, 0, 0] = 13
    predicted[0, 1, 2] = 13
    metrics = p5.errors(predicted, actual, baseline, limit=10)
    assert metrics["max_error_DN"] == 3
    assert metrics["pixels_over_1_DN"] == 40 and metrics["channel_samples_over_1_DN"] == 120
    assert len(metrics["first_mismatches"]) == 10
    assert metrics["max_error_location_count"] == 2
    assert metrics["max_error_locations"][1] == dict(x=1, y=0, channel="B", input_DN=123., predicted_DN=13., actual_DN=10., signed_error_DN=3.)
    assert metrics["first_mismatches"][1]["signed_error_DN"] == -2
    assert not p5.errors(actual, actual)["max_error_locations"]
    predicted[:] = 11
    assert not p5.errors(predicted, actual)["pixels_over_1_DN"]


@pytest.mark.parametrize("bits", [8, 16])
def test_fixture_roundtrip_and_deterministic_pixels(tmp_path, bits):
    data = p5.atlas(bits)
    np.testing.assert_array_equal(data, p5.atlas(bits))
    assert np.unique(data[:data.shape[0]//10, :, 0]).size == 2**bits
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    file = tmp_path/f"fixture_{bits}.png"
    p5.save_rgb(file, data, profile)
    recovered, icc = p5.load_rgb(file)
    np.testing.assert_array_equal(recovered, data)
    assert icc is not None


def test_capture_script_mock_control_flow():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for mock JSX control-flow test; not a Photoshop runtime test")
    result = subprocess.run([node, str(ROOT/"tests/fixtures/p5_capture_mock.cjs"), str(ROOT/"scripts/qa/p5_photoshop_capture.jsx")],
                            capture_output=True, text=True, check=True)
    assert "42 lossless exports; 40 cases; UNKNOWN gamma; cleanup OK" in result.stdout

"""Offline FA-02 contracts. No inference, detector, production wiring or photos."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/qa"))
import fa02_texture_representation_experiment as exp
from fa02_texture_controls import create_controls


@pytest.fixture
def cfg():
    return copy.deepcopy(exp.DEFAULT_CONFIG)


@pytest.fixture
def geo(cfg):
    case = {"inter_eye_distance_px": 200.0, "face_width_px": 500.0}
    return {**exp.geometry(case, cfg), "face_width_px": 500.0}


@pytest.fixture
def canvases():
    rng = np.random.default_rng(713)
    x = rng.uniform(100, 180, (160, 192, 3)).astype(np.float32)
    s = exp.gaussian(x, 2)
    return x, s


@pytest.mark.parametrize("arm", exp.ARMS)
def test_identity_and_zero_gain(arm, cfg, geo, canvases):
    x, s = canvases
    mask = np.ones(x.shape[:2], np.float32)
    same = exp.restore(x, x, arm, geo, cfg, mask)
    assert np.array_equal(same["output"], x)
    cfg["gain"] = 0
    assert np.array_equal(exp.restore(x, s, arm, geo, cfg, mask)["output"], s)


@pytest.mark.parametrize("arm", exp.ARMS)
def test_inputs_immutable_signed_and_no_output_outside_support(arm, cfg, geo, canvases):
    x, s = canvases
    x0, s0 = x.copy(), s.copy()
    mask = np.zeros(x.shape[:2], np.float32)
    mask[45:110, 50:140] = 0.4
    out = exp.restore(x, s, arm, geo, cfg, mask)
    assert out["output"].dtype == np.float32
    assert np.array_equal(x, x0) and np.array_equal(s, s0)
    assert np.all(out["delta"][mask == 0] == 0)
    if arm != exp.ARMS[0]:
        assert out["detail"].min() < 0 < out["detail"].max()


@pytest.mark.parametrize("arm", exp.ARMS[2:])
def test_constant_and_ramp_rejection(arm, cfg, geo):
    yy, xx = np.indices((160, 192), dtype=np.float32)
    s = np.full((160, 192, 3), 150, np.float32)
    for residual in (np.full_like(s, 10), np.repeat((xx*0.03+yy*0.01)[..., None], 3, axis=2)):
        detail, _ = exp.extract_bands(s+residual, s, arm, geo, cfg)
        assert np.max(np.abs(detail[40:-40, 40:-40])) < 5e-5


def test_raw_matches_current_production_algebra(cfg, geo, canvases, monkeypatch):
    # Supply exactly the same external mask to the unchanged implementation.
    # This does not exercise its dimensional-mask selection or call any detector.
    from retouch.skin import SkinProcessor
    x, s = canvases
    mask = np.full(x.shape[:2], 0.6, np.float32)
    monkeypatch.setattr(SkinProcessor, "_build_dimensional_mask", staticmethod(lambda *a, **kw: mask))
    actual = SkinProcessor().restore_micro_texture(s, x, object(), strength=20, smooth_strength=0.5)
    expected = exp.restore(x, s, exp.ARMS[1], geo, cfg, mask)["output"]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)


def test_frequency_is_actual_existing_band_difference(cfg, geo, canvases):
    from retouch.frequency import FrequencySeparator
    x, s = canvases
    left, right = [FrequencySeparator().separate(v, 500) for v in (x, s)]
    detail, _ = exp.extract_bands(x, s, exp.ARMS[5], geo, cfg)
    expected = cfg["frequency_mid_gain"]*(left.mid-right.mid)+cfg["frequency_high_gain"]*(left.high-right.high)
    np.testing.assert_allclose(detail, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("arm", (exp.ARMS[1], exp.ARMS[2], exp.ARMS[3], exp.ARMS[5]))
def test_linear_source_difference_equivalence(arm, cfg, geo, canvases):
    x, s = canvases
    zero = np.zeros_like(s)
    direct, _ = exp.extract_bands(x, s, arm, geo, cfg)
    tx, _ = exp.extract_bands(x, zero, arm, geo, cfg)
    ts, _ = exp.extract_bands(s, zero, arm, geo, cfg)
    np.testing.assert_allclose(direct, tx-ts, atol=1.5e-4, rtol=0)


def test_one_hot_multiscale_and_telescope(cfg, geo, canvases):
    x, s = canvases
    cfg["multiscale_weights"] = [1, 0, 0, 0]
    one, _ = exp.extract_bands(x, s, exp.ARMS[3], geo, cfg)
    dog, _ = exp.extract_bands(x, s, exp.ARMS[2], geo, cfg)
    np.testing.assert_array_equal(one, dog)
    cfg["multiscale_weights"] = [1, 1, 1, 1]
    multi, _ = exp.extract_bands(x, s, exp.ARMS[3], geo, cfg)
    expected = exp.gaussian(x-s, geo["sigmas_px"][0])-exp.gaussian(x-s, geo["sigmas_px"][-1])
    np.testing.assert_allclose(multi, expected, atol=1e-5, rtol=0)


def test_orientation_off_parity_bounded_and_isotropic_path(cfg, geo, canvases):
    x, s = canvases
    cfg["orientation_attenuation"] = 0
    a, _ = exp.extract_bands(x, s, exp.ARMS[3], geo, cfg)
    b, _ = exp.extract_bands(x, s, exp.ARMS[4], geo, cfg)
    np.testing.assert_array_equal(a, b)
    cfg["orientation_attenuation"] = 0.5
    _, bands = exp.extract_bands(x, s, exp.ARMS[4], geo, cfg)
    for name, value in bands.items():
        if name.startswith("orientation_gate"):
            assert value.min() >= 0.5 and value.max() <= 1
    flat = np.full_like(x, 150)
    _, bands = exp.extract_bands(flat, s, exp.ARMS[4], geo, cfg)
    assert np.all(bands["orientation_gate_0"] == 1)


@pytest.mark.parametrize("arm", exp.ARMS)
def test_common_guard_stops_corrected_impulse_neighbor_leakage(arm, cfg, geo):
    s = np.full((192, 192, 3), 150, np.float32)
    x = s.copy()
    x[96, 96] -= 40
    masks = {key: np.zeros(s.shape[:2], np.float32) for key in exp.MASKS}
    masks["allow"][:] = 1
    masks["corrected"][96, 96] = 1
    eligibility = exp.map_external_supports(masks, geo, cfg)
    result = exp.restore(x, s, arm, geo, cfg, eligibility)
    np.testing.assert_array_equal(result["output"], s)
    assert eligibility[96, 96+geo["common_radius_px"]] == 0
    assert eligibility[96, 96+geo["common_radius_px"]+1] > 0
    if arm not in (exp.ARMS[0], exp.ARMS[1]):
        assert np.any(result["detail"][94:99, 94:99] != 0)


@pytest.mark.parametrize("arm", exp.ARMS)
def test_shift_and_tile_context_equivariance(arm, cfg, geo, canvases):
    x, s = canvases
    a, _ = exp.extract_bands(x, s, arm, geo, cfg)
    b, _ = exp.extract_bands(np.roll(x, 1, axis=1), np.roll(s, 1, axis=1), arm, geo, cfg)
    np.testing.assert_allclose(a[40:-40, 40:-40], np.roll(b, -1, axis=1)[40:-40, 40:-40], atol=1e-5, rtol=0)
    tile, _ = exp.extract_bands(x[8:-8, 8:-8], s[8:-8, 8:-8], arm, geo, cfg)
    np.testing.assert_allclose(a[48:-48, 48:-48], tile[40:-40, 40:-40], atol=1e-5, rtol=0)


def test_discrete_impulse_zero_dc_and_finite_footprint(cfg, geo):
    zero = np.zeros((192, 192, 3), np.float32)
    impulse = zero.copy()
    impulse[96, 96] = 1
    for arm in exp.ARMS[2:4]+(exp.ARMS[5],):
        detail, _ = exp.extract_bands(impulse, zero, arm, geo, cfg)
        assert abs(float(detail[..., 0].sum())) < 1e-6
        assert np.all(detail[:60] == 0)
        assert np.all(detail[132:] == 0)
        assert np.max(np.abs(np.fft.fft2(detail[..., 0]))) > 0


def test_small_scales_are_logged_not_silently_upsampled(cfg):
    geo = exp.geometry({"inter_eye_distance_px": 20, "face_width_px": 50}, cfg)
    assert geo["sigmas_px"][0] == 0.6
    assert geo["collapsed_bands"] == [0, 1, 2]


def test_clipping_and_undefined_measurements(cfg, geo):
    s = np.full((160, 192, 3), 254, np.float32)
    result = exp.restore(np.zeros_like(s), s, exp.ARMS[2], geo, cfg, np.ones(s.shape[:2], np.float32))
    assert not result["clipped_pixels"].any()
    assert exp.projection(s, np.zeros_like(s), np.ones(s.shape[:2], bool)) is None
    assert exp.rms(np.array([])) is None


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path/"controls"
    create_controls(path)
    manifest = json.loads((path/"manifest.json").read_text())
    return path, manifest


def test_controls_validate_and_no_real_or_heldout_labels(corpus):
    path, manifest = corpus
    exp.validate_manifest(manifest, path)
    assert len(manifest["cases"]) == 11
    assert all(c["kind"] == "analytic_control" and c["split"] == "validation" for c in manifest["cases"])


def test_missing_supports_checksums_and_pair_mismatch_rejected(corpus):
    path, manifest = corpus
    manifest["cases"][0]["arrays_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        exp.validate_manifest(manifest, path)
    manifest["cases"][0]["arrays_sha256"] = exp.digest(path/manifest["cases"][0]["arrays"])
    manifest["cases"][3]["face_width_px"] = 400
    with pytest.raises(ValueError, match="Paired controls"):
        exp.validate_manifest(manifest, path)


def test_portrait_cannot_bypass_external_review(corpus):
    path, manifest = corpus
    manifest["purpose"] = "development_pilot"
    c = manifest["cases"][0]
    c.update(kind="portrait", split="dev")
    with pytest.raises(ValueError, match="subject/session"):
        exp.validate_manifest(manifest, path)
    c["person_ids"] = ["test_subject"]
    with pytest.raises(ValueError, match="missing owner_mapping_reference"):
        exp.validate_manifest(manifest, path)


def test_fake_holdout_and_tampered_config_rejected(corpus, tmp_path):
    path, manifest = corpus
    manifest["purpose"] = "locked_comparison"
    with pytest.raises(ValueError, match="validation only"):
        exp.validate_manifest(manifest, path)
    lock = tmp_path/"lock.json"
    exp.freeze_config(None, lock, "a_priori_validation", [])
    with pytest.raises(ValueError, match="overwrite"):
        exp.freeze_config(None, lock, "a_priori_validation", [])
    obj = json.loads(lock.read_text())
    obj["config"]["gain"] = 0.2
    exp.write_json(lock, obj)
    with pytest.raises(ValueError, match="hash mismatch"):
        exp.run(path/"manifest.json", lock, tmp_path/"result", 1)


def test_worker_and_complete_artifacts(tmp_path, corpus):
    path, manifest = corpus
    # A small full CLI-equivalent six-arm smoke run; rest of corpus is run offline.
    manifest["cases"] = [manifest["cases"][-1]]
    exp.write_json(path/"smoke.json", manifest)
    lock = tmp_path/"lock.json"
    exp.freeze_config(None, lock, "a_priori_validation", [])
    out = tmp_path/"out"
    exp.run(path/"smoke.json", lock, out, 1)
    assert json.loads((out/"COMPLETE.json").read_text())["arms"] == 6
    rows = json.loads((out/"summary.json").read_text())["results"]
    assert len(rows) == 6
    for row in rows:
        assert row["measurement"]["peak_rss_mib"] > 0
        assert row["measurement"]["warm_median_ms"] > 0
        assert row["forbidden_max_abs_delta"] == 0
        folder = out/"small_sampling"/row["arm"]
        image = cv2.imread(str(folder/"output_native.png"))
        assert image.shape == (128, 128, 3)
        arrays = exp.load_arrays(folder/"signed_arrays.npz")
        assert arrays["output"].dtype == np.float32
    with pytest.raises(ValueError, match="already exists"):
        exp.run(path/"smoke.json", lock, out, 1)
    timing_out = tmp_path/"timings"
    exp.profile_run(path/"smoke.json", lock, timing_out, 1)
    records = json.loads((timing_out/"TIMINGS.json").read_text())["measurements"]
    assert len(records) == 6
    assert all(r["measurement"]["peak_rss_mib"] > 0 for r in records)


def test_subject_and_session_splits_are_enforced(corpus):
    path, manifest = corpus
    manifest["purpose"] = "development_pilot"
    manifest["cases"] = copy.deepcopy(manifest["cases"][:2])
    for i, case in enumerate(manifest["cases"]):
        case.update(kind="portrait", split="dev" if i == 0 else "calibration",
                    person_ids=["same_subject"], session_id=f"session_{i}",
                    owner_mapping_reference="test_assertion", support_acceptance_reference="test_assertion",
                    native_source_reference=case["arrays"], source_sha256=case["arrays_sha256"],
                    color_profile="encoded BGR test", resampled=False, support_status="externally_accepted")
    with pytest.raises(ValueError, match="crosses splits"):
        exp.validate_manifest(manifest, path)
    manifest["cases"][1]["person_ids"] = ["different_subject"]
    manifest["cases"][1]["session_id"] = "session_0"
    with pytest.raises(ValueError, match="crosses splits"):
        exp.validate_manifest(manifest, path)


def test_no_detection_or_pipeline_imports():
    text = (ROOT/"scripts/qa/fa02_texture_representation_experiment.py").read_text()
    for forbidden in ("detect_marks(", "retouch.engine", "retouch.freckle", "retouch.marks", "retouch.perf_optimizations"):
        assert forbidden not in text

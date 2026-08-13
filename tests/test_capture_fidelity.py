"""P1/P1.1 capture-fidelity contracts."""

from pathlib import Path

import numpy as np
from PIL import Image

from retouch.capture_fidelity import (
    CaptureMetadata,
    SensorCalibrationProfile,
    apply_sensor_calibration,
    apply_optical_corrections,
    lens_correction_status,
    read_capture_metadata,
    select_sensor_profile,
)


def test_capture_metadata_reads_camera_lens_and_exposure(tmp_path: Path):
    path = tmp_path / "capture.jpg"
    image = Image.new("RGB", (8, 8), (100, 110, 120))
    exif = image.getexif()
    exif[271] = "Acme"
    exif[272] = "Portrait 1"
    exif[42036] = "Prime 50mm"
    exif[37386] = (50, 1)
    exif[33437] = (14, 10)
    exif[34855] = 800
    exif[33434] = (1, 125)
    image.save(path, exif=exif.tobytes())

    metadata = read_capture_metadata(path)

    assert metadata.camera_model == "Acme Portrait 1"
    assert metadata.lens_model == "Prime 50mm"
    assert metadata.focal_length_mm == 50.0
    assert metadata.aperture == 1.4
    assert metadata.iso == 800.0
    assert metadata.shutter_seconds == 1 / 125


def test_sensor_profile_selection_matches_camera_and_nearest_settings():
    metadata = CaptureMetadata(
        make="Acme", model="Portrait 1", iso=800, shutter_seconds=1 / 125,
    )
    profiles = [
        SensorCalibrationProfile(camera_model="Other", iso=800),
        SensorCalibrationProfile(camera_model="Acme Portrait 1", iso=100),
        SensorCalibrationProfile(camera_model="Acme Portrait 1", iso=800),
    ]

    selected = select_sensor_profile(metadata, profiles)

    assert selected is profiles[2]


def test_sensor_calibration_applies_dark_flat_and_hot_pixel_correction():
    raw = np.full((5, 5), 100.0, dtype=np.float32)
    raw[2, 2] = 1000.0
    dark = np.full((5, 5), 10.0, dtype=np.float32)
    flat = np.full((5, 5), 2.0, dtype=np.float32)
    flat[0, 0] = 1.0

    corrected = apply_sensor_calibration(
        raw, dark_frame=dark, flat_field=flat, hot_pixels=[(2, 2)],
    )

    assert corrected.dtype == np.float32
    assert corrected[2, 2] < 200.0
    assert corrected[1, 1] == 90.0
    assert np.all(raw[2, 2] == 1000.0)


def test_lens_correction_reports_optional_dependency_truthfully():
    status = lens_correction_status()

    assert set(status) == {"available", "backend", "reason"}
    if not status["available"]:
        assert status["backend"] is None
        assert "unavailable" in status["reason"]


def test_optical_correction_is_fail_closed_without_verified_lens_match():
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    corrected, status = apply_optical_corrections(
        image,
        CaptureMetadata(make="Unknown", model="Camera", lens_model="Unknown"),
    )

    assert np.array_equal(corrected, image)
    assert status["applied"] == ()
    assert status["requested"] == ("distortion", "tca", "vignetting")

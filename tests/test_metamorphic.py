import numpy as np

from retouch.metamorphic import generate_variants, run_lab


def _image():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :] = (80, 110, 140)
    image[10:30, 15:45] = (120, 130, 150)
    return image


def test_variants_have_inverse_contracts():
    image = _image()
    variants = generate_variants(image)
    assert {variant.name for variant in variants} == {
        "exif_rotation", "resize_proxy", "jpeg_quality", "exposure_plus",
        "white_balance_shift", "bit_depth_roundtrip",
    }
    for variant in variants:
        restored = variant.restore(variant.image_bgr)
        assert restored.shape[:2] == image.shape[:2]
        assert restored.dtype == np.uint8


def test_lab_passes_identity_processor_and_decision_stability():
    image = _image()
    observations = run_lab(
        image,
        lambda value: value,
        decision_provider=lambda value: {"action": "apply"},
        mean_threshold=8.0,
        p95_threshold=24.0,
    )
    assert len(observations) == 6
    assert all(observation.passed for observation in observations)
    assert all(observation.decision_equal is True for observation in observations)


def test_lab_flags_processor_drift_and_decision_change():
    image = _image()

    def drift(value):
        return np.clip(value.astype(np.int16) + 80, 0, 255).astype(np.uint8)

    observations = run_lab(
        image,
        drift,
        decision_provider=lambda value: {"action": "apply" if value.mean() < 100 else "skip"},
        mean_threshold=1.0,
        p95_threshold=2.0,
    )
    assert any(not observation.passed for observation in observations)

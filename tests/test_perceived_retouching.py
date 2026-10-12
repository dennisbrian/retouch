"""AA6 deterministic Kee–Farid perceived-retouching vector tests."""

import cv2
import numpy as np

from retouch.qa_detectors import perceived_retouching_vector, run_all


def _textured_image(size: int = 160) -> np.ndarray:
    rng = np.random.default_rng(20260720)
    base = np.tile(np.linspace(70, 180, size, dtype=np.float32), (size, 1))
    noise = rng.normal(0, 12, (size, size)).astype(np.float32)
    luma = np.clip(base + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)


def test_aa6_identity_is_zero_geometry_and_neutral_photometry() -> None:
    reference = _textured_image()
    face = np.ones(reference.shape[:2], dtype=np.float32)

    result = perceived_retouching_vector(reference, reference.copy(), face)

    assert result["face_warp_gradient_mean"] == 0.0
    assert result["body_warp_gradient_mean"] == 0.0
    assert result["face_ssim_cs_mean"] > 0.999
    assert abs(result["face_frequency_d_mean"]) < 1e-5


def test_aa6_geometry_is_monotone_and_aggregated_into_qa() -> None:
    reference = _textured_image()
    face = np.ones(reference.shape[:2], dtype=np.float32)
    small = np.zeros((*reference.shape[:2], 2), dtype=np.float32)
    large = small.copy()
    small[:, :, 0] = 2.0
    large[:, :, 0] = 8.0

    small_score = perceived_retouching_vector(reference, reference, face, small)
    large_score = perceived_retouching_vector(reference, reference, face, large)
    aggregated = run_all(reference, face_skin_mask=face, reference_img_bgr=reference, warp_field=large)

    assert large_score["face_warp_gradient_mean"] > small_score["face_warp_gradient_mean"] > 0.0
    assert aggregated["perceived_retouching"]["face_warp_gradient_mean"] == large_score["face_warp_gradient_mean"]


def test_aa6_photometry_increases_for_smoothing() -> None:
    reference = _textured_image()
    processed = cv2.GaussianBlur(reference, (9, 9), 2)
    face = np.ones(reference.shape[:2], dtype=np.float32)

    result = perceived_retouching_vector(processed, reference, face)

    assert result["face_ssim_cs_mean"] < 0.99
    assert result["face_frequency_d_mean"] > 0.0


def test_aa6_geometry_is_stable_across_proxy_resolution() -> None:
    reference = _textured_image(160)
    face = np.ones(reference.shape[:2], dtype=np.float32)
    warp = np.zeros((*reference.shape[:2], 2), dtype=np.float32)
    warp[:, :, 0] = 6.0
    full = perceived_retouching_vector(reference, reference, face, warp)

    proxy = cv2.resize(reference, (80, 80), interpolation=cv2.INTER_AREA)
    proxy_face = np.ones(proxy.shape[:2], dtype=np.float32)
    proxy_warp = np.zeros((*proxy.shape[:2], 2), dtype=np.float32)
    proxy_warp[:, :, 0] = 3.0
    reduced = perceived_retouching_vector(proxy, proxy, proxy_face, proxy_warp)

    assert np.isclose(
        full["face_warp_gradient_mean"], reduced["face_warp_gradient_mean"], rtol=0.05
    )

"""Shared fixtures for all retouch tests."""

import os
from pathlib import Path

import numpy as np
import cv2
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshot",
        action="store_true",
        default=False,
        help="Golden tests overwrite stored hashes instead of asserting",
    )


# ---------------------------------------------------------------------------
# Eye-visibility gate mutation harness
# ---------------------------------------------------------------------------
# Run normally:            .venv/bin/python -m pytest tests/test_eye_visibility.py -q
# Run with gate disabled:  GATE_MUTATE=1 .venv/bin/python -m pytest tests/test_eye_visibility.py -q
# Under GATE_MUTATE the gate is replaced by identity in every namespace that
# binds the symbol, so a test that still passes proves it never depended on
# the gate (the non-vacuity property the review §B3 demanded).
@pytest.fixture(autouse=True)
def _maybe_mutate_gate(monkeypatch):
    if os.environ.get("GATE_MUTATE") != "1":
        return
    import retouch.eye_enhancement
    import retouch.eye_visibility
    import retouch.eyes
    import retouch.perf_optimizations

    def _identity(regions, landmarks=None, img_bgr=None, width=None, height=None):
        return regions

    for module in (
        retouch.eye_visibility,
        retouch.eyes,
        retouch.eye_enhancement,
        retouch.perf_optimizations,
    ):
        monkeypatch.setattr(module, "gate_occluded_eye_regions", _identity)


def _check_cv2():
    """Skip if cv2 not available."""
    if cv2 is None:
        pytest.skip("OpenCV not installed")


# ---------------------------------------------------------------------------
# Basic image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_image():
    """Return a small blank (gray) uint8 BGR image, 200x150."""
    _check_cv2()
    return np.full((150, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """Return a colour gradient image, 100x100."""
    _check_cv2()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        img[y, :, 0] = y * 2.55
        img[y, :, 1] = (100 - y) * 2.55
        img[y, :, 2] = 128
    return img


@pytest.fixture
def checkerboard():
    """Return a 100x100 checkerboard pattern for high-frequency testing."""
    _check_cv2()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 10):
        for x in range(0, 100, 10):
            val = 255 if (x // 10 + y // 10) % 2 == 0 else 0
            img[y:y+10, x:x+10] = val
    return img


# ---------------------------------------------------------------------------
# Mask fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skin_mask():
    """Return a simple circular skin mask (float32, 0-1)."""
    _check_cv2()
    mask = np.zeros((150, 200), dtype=np.float32)
    cx, cy, r = 100, 75, 50
    y, x = np.ogrid[:150, :200]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 1.0
    return mask


@pytest.fixture
def half_mask():
    """Return a float mask where the left half is 1.0, right half is 0.0."""
    _check_cv2()
    mask = np.zeros((64, 128), dtype=np.float32)
    mask[:, :64] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Synthetic face image  (for pipeline integration tests)
# ---------------------------------------------------------------------------


def _draw_face(img, cx, cy, scale=1.0):
    """Draw a cartoon face on *img* centered at (cx, cy) with given scale."""
    s = scale
    # Skin-coloured oval head
    face_color = (200, 180, 160)  # BGR
    cv2.ellipse(img, (cx, cy), (int(60 * s), int(80 * s)), 0, 0, 360, face_color, -1)

    # Eyes — white sclera + dark pupil + iris
    for ex in (cx - 25 * s, cx + 25 * s):
        ey = cy - 15 * s
        # Sclera
        cv2.ellipse(img, (int(ex), int(ey)), (int(12 * s), int(8 * s)), 0, 0, 360, (230, 230, 230), -1)
        # Iris
        cv2.circle(img, (int(ex), int(ey)), int(6 * s), (80, 120, 80), -1)
        # Pupil
        cv2.circle(img, (int(ex), int(ey)), int(3 * s), (30, 30, 30), -1)
        # Catchlight
        cv2.circle(img, (int(ex + 2 * s), int(ey - 2 * s)), int(1.5 * s), (255, 255, 255), -1)

    # Eyebrows
    for bx in (cx - 25 * s, cx + 25 * s):
        by = cy - 30 * s
        cv2.ellipse(img, (int(bx), int(by)), (int(14 * s), int(3 * s)), 0, 0, 360, (50, 50, 50), -1)

    # Nose
    cv2.ellipse(img, (int(cx), int(cy + 10 * s)), (int(8 * s), int(12 * s)), 0, 0, 360, (180, 160, 140), -1)
    # Nostrils
    for nx in (cx - 4 * s, cx + 4 * s):
        cv2.circle(img, (int(nx), int(cy + 18 * s)), int(2 * s), (120, 100, 80), -1)

    # Mouth
    cv2.ellipse(img, (int(cx), int(cy + 40 * s)), (int(20 * s), int(10 * s)), 0, 0, 180, (60, 60, 160), -1)
    # Lip line
    cv2.ellipse(img, (int(cx), int(cy + 40 * s)), (int(20 * s), int(8 * s)), 0, 0, -180, (40, 40, 140), -1)

    # Hair — simple top patch
    hair_color = (40, 40, 40)
    cv2.ellipse(img, (int(cx), int(cy - 60 * s)), (int(65 * s), int(30 * s)), 0, 0, 360, hair_color, -1)


@pytest.fixture(scope="session")
def synthetic_face():
    """Return a 400x400 cartoon face image for pipeline integration tests.

    MediaPipe may or may not detect this as a face, so tests using it
    should handle both cases gracefully.
    """
    _check_cv2()
    img = np.full((400, 400, 3), 180, dtype=np.uint8)  # light gray background
    _draw_face(img, 200, 200, scale=1.5)
    return img


@pytest.fixture(scope="module")
def engine():
    """Shared RetouchEngine with guaranteed teardown."""
    from retouch.engine import RetouchEngine

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    landmarker_model = os.path.join(models_dir, "face_landmarker.task")
    if not os.path.exists(landmarker_model):
        pytest.skip("Face landmarker model not available")
    with RetouchEngine() as eng:
        yield eng


@pytest.fixture(scope="session")
def natural_image_path():
    """Path to a real human face image for integration tests.

    Returns None if no suitable image is found (tests should skip).
    Prefers the repository's known clear single-portrait reference, then
    falls back to the first JPEG from ``test_output/``.

    Session-scoped (pure filesystem lookup) so ``natural_image`` can
    depend on it without re-reading the image per test.
    """
    reference = Path("test_output/DSCF8007.jpg")
    if reference.exists():
        return str(reference)

    candidates = [
        p
        for p in sorted(Path("test_output").glob("*.jpg"))
        if not p.name.endswith("_compare.jpg")
    ]
    if candidates:
        return str(candidates[0])
    return None


@pytest.fixture(scope="session")
def natural_image(natural_image_path):
    """Real face image downscaled to ≤2048px on the long side.

    Full-res 24MP input into quality="full" takes minutes; tests use
    this fixture to stay well under the pytest timeout. Skips when no
    readable real image is available.
    """
    _check_cv2()
    if natural_image_path is None:
        pytest.skip("No real face image found (test_output/ is empty)")
    img = cv2.imread(natural_image_path)
    if img is None:
        pytest.skip(f"Could not read image: {natural_image_path}")
    from retouch.io import resize_for_processing

    img, _scale = resize_for_processing(img, 2048)
    return img

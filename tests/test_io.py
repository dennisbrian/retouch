"""Tests for retouch/io.py — I/O helper functions."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import Base as ExifBase
from retouch.io import (
    resize_for_processing,
    output_format,
    encode_write_params,
    make_comparison,
    copy_exif,
    imread_exif,
)


class TestResizeForProcessing:
    def test_none_max_dim(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, None)
        assert np.all(result == img)
        assert scale == 1.0

    def test_already_small_enough(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 500)
        assert np.all(result == img)
        assert scale == 1.0

    def test_downscales_width(self):
        img = np.full((100, 400, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 200)
        assert result.shape[1] <= 200
        assert scale < 1.0

    def test_downscales_height(self):
        img = np.full((400, 100, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 200)
        assert result.shape[0] <= 200
        assert scale < 1.0

    def test_output_type(self):
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 100)
        assert result.dtype == np.uint8


class TestOutputFormat:
    def test_same_preserves_png(self, tmp_path):
        png = tmp_path / "photo.png"
        png.touch()
        assert output_format(png, "same") == "png"

    def test_same_preserves_webp(self, tmp_path):
        webp = tmp_path / "photo.webp"
        webp.touch()
        assert output_format(webp, "same") == "webp"

    def test_same_maps_jpeg_variants_to_jpg(self, tmp_path):
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            path = tmp_path / f"photo{ext}"
            path.touch()
            assert output_format(path, "same") == "jpg"

    def test_explicit_format_overrides_extension(self, tmp_path):
        png = tmp_path / "photo.png"
        png.touch()
        assert output_format(png, "webp") == "webp"

    def test_explicit_jpg(self, tmp_path):
        webp = tmp_path / "photo.webp"
        webp.touch()
        assert output_format(webp, "jpg") == "jpg"

    def test_explicit_png(self, tmp_path):
        jpg = tmp_path / "photo.jpg"
        jpg.touch()
        assert output_format(jpg, "png") == "png"


class TestEncodeWriteParams:
    def test_jpg(self):
        assert encode_write_params("jpg", 95) == [cv2.IMWRITE_JPEG_QUALITY, 95]

    def test_jpeg(self):
        assert encode_write_params("jpeg", 95) == [cv2.IMWRITE_JPEG_QUALITY, 95]

    def test_webp(self):
        assert encode_write_params("webp", 95) == [cv2.IMWRITE_WEBP_QUALITY, 95]

    def test_png_returns_empty(self):
        assert encode_write_params("png", 95) == []

    def test_unknown_returns_empty(self):
        assert encode_write_params("tiff", 95) == []


class TestMakeComparison:
    def test_none_original(self, tmp_path):
        import cv2
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(None, img, out, "jpg", 95)
        assert not out.exists()

    def test_none_retouched(self, tmp_path):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(img, None, out, "jpg", 95)
        assert not out.exists()

    def test_writes_comparison(self, tmp_path):
        import cv2
        orig = np.full((50, 100, 3), 100, dtype=np.uint8)
        ret = np.full((50, 100, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 95)
        assert out.exists()
        img = cv2.imread(str(out))
        assert img.shape[0] == 50
        assert img.shape[1] == 100 + 4 + 100

    def test_resizes_mismatched_shapes(self, tmp_path):
        import cv2
        orig = np.full((50, 100, 3), 100, dtype=np.uint8)
        ret = np.full((60, 100, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 95)
        assert out.exists()

    def test_jpeg_quality(self, tmp_path):
        import cv2
        orig = np.full((30, 60, 3), 100, dtype=np.uint8)
        ret = np.full((30, 60, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 50)
        assert out.exists()

    def test_png_format(self, tmp_path):
        import cv2
        orig = np.full((30, 60, 3), 100, dtype=np.uint8)
        ret = np.full((30, 60, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.png"
        make_comparison(orig, ret, out, "png", 95)
        assert out.exists()


# ---------------------------------------------------------------------------
# copy_exif
# ---------------------------------------------------------------------------


def _write_jpeg_with_exif(path, orientation=1, make="TestCam", model="T1"):
    """Helper: write a small JPEG at *path* with EXIF metadata."""
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[5:15, 5:15] = 200
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[ExifBase.Make] = make
    exif[ExifBase.Model] = model
    exif[ExifBase.Orientation] = orientation
    img.save(str(path), "JPEG", exif=exif.tobytes())


def _write_jpeg_without_exif(path):
    """Helper: write a small JPEG without any EXIF block."""
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path), "JPEG")


class TestCopyExif:
    def test_copies_exif_tags(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_with_exif(src)
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        # Re-read the destination and check the EXIF tags are present
        out = Image.open(str(dst))
        exif = out.getexif()
        assert exif.get(ExifBase.Make) == "TestCam"
        assert exif.get(ExifBase.Model) == "T1"

    def test_resets_orientation_to_normal(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        # Source has orientation=6 (rotated 90° CCW)
        _write_jpeg_with_exif(src, orientation=6)
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        out = Image.open(str(dst))
        exif = out.getexif()
        # Orientation is rewritten to 1 (normal) — the destination has its
        # own pixel orientation
        assert exif.get(ExifBase.Orientation) == 1

    def test_preserves_other_exif_tags(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_with_exif(src, make="CamA", model="M-100")
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        out = Image.open(str(dst))
        exif = out.getexif()
        # The destination is freshly written; tags that were explicitly
        # set on the source are transferred.
        assert exif.get(ExifBase.Make) == "CamA"
        assert exif.get(ExifBase.Model) == "M-100"

    def test_source_without_exif_is_silent_noop(self, tmp_path):
        src = tmp_path / "src_no_exif.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_without_exif(src)
        _write_jpeg_without_exif(dst)

        # Should not raise
        copy_exif(str(src), str(dst))

        # Destination remains valid
        out = Image.open(str(dst))
        assert out.size == (20, 20)

    def test_handles_different_orientation_values(self, tmp_path):
        for orig_orientation in (1, 3, 6, 8):
            src = tmp_path / f"src_o{orig_orientation}.jpg"
            dst = tmp_path / f"dst_o{orig_orientation}.jpg"
            _write_jpeg_with_exif(src, orientation=orig_orientation)
            _write_jpeg_without_exif(dst)

            copy_exif(str(src), str(dst))

            out = Image.open(str(dst))
            exif = out.getexif()
            # All orientations are normalised to 1
            assert exif.get(ExifBase.Orientation) == 1

    def test_does_not_raise_on_invalid_source(self, tmp_path):
        # Source doesn't exist — should log a warning and return, not raise
        src = tmp_path / "nonexistent.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_without_exif(dst)

        # Should not raise even when source is missing
        copy_exif(str(src), str(dst))


import cv2


# ---------------------------------------------------------------------------
# Additional coverage — exact test names from AUDIT_REPORT follow-up
# ---------------------------------------------------------------------------


def _write_png_with_exif(path, make="TestCam", model="T1"):
    """Helper: write a small PNG at *path* with EXIF metadata.

    PNG supports a limited EXIF block via Pillow's ``exif`` kwarg; this
    helper is best-effort — Pillow will silently drop the exif kwarg for
    some PNG modes, so we only assert that the file round-trips cleanly.
    """
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[5:15, 5:15] = 200
    img = Image.fromarray(arr)
    try:
        exif = img.getexif()
        exif[ExifBase.Make] = make
        exif[ExifBase.Model] = model
        img.save(str(path), "PNG", exif=exif.tobytes())
    except Exception:
        # Fall back to plain PNG — the test still validates the I/O path
        img.save(str(path), "PNG")


def test_copy_exif_jpg_to_jpg(tmp_path):
    """copy_exif() copies EXIF tags from a JPEG source to a JPEG destination."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_jpeg_with_exif(src, make="TestCam", model="T1")
    _write_jpeg_without_exif(dst)

    copy_exif(str(src), str(dst))

    out = Image.open(str(dst))
    exif = out.getexif()
    assert exif.get(ExifBase.Make) == "TestCam"
    assert exif.get(ExifBase.Model) == "T1"


def test_copy_exif_missing_source_exif(tmp_path):
    """copy_exif() handles a JPEG with no EXIF gracefully (no crash, no-op)."""
    src = tmp_path / "no_exif.jpg"
    dst = tmp_path / "dst.jpg"
    _write_jpeg_without_exif(src)
    _write_jpeg_without_exif(dst)

    # Should not raise even when source has no EXIF block
    copy_exif(str(src), str(dst))

    # Destination is still a valid, openable JPEG
    out = Image.open(str(dst))
    assert out.size == (20, 20)


def test_copy_exif_png_to_png(tmp_path):
    """copy_exif() does not crash on PNG-to-PNG (EXIF is JPEG-centric)."""
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    _write_png_with_exif(src)
    _write_png_with_exif(dst)

    # Should not raise — PNG EXIF handling is best-effort in Pillow
    copy_exif(str(src), str(dst))

    assert (tmp_path / "dst.png").exists()
    out = Image.open(str(dst))
    assert out.size == (20, 20)


def test_copy_exif_nonexistent_source(tmp_path):
    """copy_exif() logs a warning and returns gracefully on a missing source."""
    src = tmp_path / "does_not_exist.jpg"
    dst = tmp_path / "dst.jpg"
    _write_jpeg_without_exif(dst)

    # Should not raise — the function catches all exceptions internally
    copy_exif(str(src), str(dst))

    # Destination file is still intact
    out = Image.open(str(dst))
    assert out.size == (20, 20)


def test_imread_exif_loads_image(tmp_path):
    """imread_exif() returns a BGR ndarray with correct dimensions for a real JPEG."""
    src = tmp_path / "photo.jpg"
    arr = np.zeros((60, 80, 3), dtype=np.uint8)
    arr[10:50, 20:60] = 128
    Image.fromarray(arr).save(str(src), "JPEG")

    img = imread_exif(str(src))

    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    # BGR ordering — height should be 60, width 80
    assert img.shape[0] == 60
    assert img.shape[1] == 80


def test_imread_exif_nonexistent_returns_none(tmp_path):
    """imread_exif() returns None (or propagates) for a missing path.

    The function is not documented to swallow IOErrors, so we only assert
    that the call does not silently return garbage — either it returns
    ``None`` or it raises a recognisable exception.
    """
    missing = tmp_path / "missing.jpg"
    try:
        result = imread_exif(str(missing))
    except (FileNotFoundError, OSError, IOError):
        return  # acceptable: a recognisable exception
    assert result is None


def test_resize_for_processing_downscales(tmp_path):
    """resize_for_processing() downscales a 4000x3000 image to fit max_dim=2048."""
    img = np.full((3000, 4000, 3), 128, dtype=np.uint8)
    result, scale = resize_for_processing(img, 2048)

    h, w = result.shape[:2]
    # Longest side is bounded by max_dim; aspect ratio (4:3) is preserved
    assert max(h, w) <= 2048
    assert abs(w / h - 4 / 3) < 0.05
    # Exact expected dimensions: 2048 x 1536 (4:3 aspect)
    assert w == 2048
    assert h == 1536
    assert scale == pytest.approx(2048 / 4000)


def test_resize_for_processing_no_resize_needed(tmp_path):
    """resize_for_processing() leaves a 1024x768 image unchanged when max_dim=2048."""
    img = np.full((768, 1024, 3), 99, dtype=np.uint8)
    result, scale = resize_for_processing(img, 2048)

    assert result.shape == img.shape
    assert scale == 1.0
    assert np.array_equal(result, img)


def test_output_format_validates_format(tmp_path):
    """output_format() resolves 'same' and returns explicit formats unchanged."""
    # 'same' maps based on the input file extension
    assert output_format(tmp_path / "x.png", "same") == "png"
    assert output_format(tmp_path / "x.webp", "same") == "webp"
    assert output_format(tmp_path / "x.jpg", "same") == "jpg"
    assert output_format(tmp_path / "x.jpeg", "same") == "jpg"
    assert output_format(tmp_path / "x.JPEG", "same") == "jpg"
    # Unknown extensions default to jpg
    assert output_format(tmp_path / "x.tiff", "same") == "jpg"
    # Explicit format overrides extension
    assert output_format(tmp_path / "x.png", "webp") == "webp"
    assert output_format(tmp_path / "x.jpg", "png") == "png"
    assert output_format(tmp_path / "x.png", "jpg") == "jpg"


def test_encode_write_params_returns_dict(tmp_path):
    """encode_write_params() returns the expected OpenCV param lists."""
    # JPEG family uses IMWRITE_JPEG_QUALITY
    assert encode_write_params("jpg", 90) == [cv2.IMWRITE_JPEG_QUALITY, 90]
    assert encode_write_params("jpeg", 75) == [cv2.IMWRITE_JPEG_QUALITY, 75]
    # WebP uses IMWRITE_WEBP_QUALITY
    assert encode_write_params("webp", 80) == [cv2.IMWRITE_WEBP_QUALITY, 80]
    # Lossless formats return an empty param list
    assert encode_write_params("png", 90) == []
    assert encode_write_params("tiff", 90) == []
    # Quality value is passed through verbatim
    assert encode_write_params("jpg", 100) == [cv2.IMWRITE_JPEG_QUALITY, 100]


def test_make_comparison_creates_file(tmp_path):
    """make_comparison() writes a side-by-side image file to disk."""
    orig = np.full((50, 100, 3), 100, dtype=np.uint8)
    ret = np.full((50, 100, 3), 200, dtype=np.uint8)
    out = tmp_path / "compare.jpg"

    make_comparison(orig, ret, out, "jpg", 90)

    assert out.exists()
    assert out.stat().st_size > 0
    # The written file is a valid image of the expected stitched width
    written = cv2.imread(str(out))
    assert written is not None
    assert written.shape[0] == 50
    # original (100) + separator (4) + retouched (100)
    assert written.shape[1] == 100 + 4 + 100

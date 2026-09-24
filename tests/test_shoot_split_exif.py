"""Tests for retouch.shoot_split EXIF reading and parsing.

Covers read_capture_info(), _parse_exif_datetime(), and SHOT_EXTENSIONS.
Includes stdlib-only TIFF/EXIF blob builders for JPEG, TIFF, RAF, CR3, PNG, WebP.
"""

from __future__ import annotations

import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Tuple


from retouch.shoot_split import (
    SHOT_EXTENSIONS,
    _parse_exif_datetime,
    read_capture_info,
)


# ============================================================================
# TIFF/EXIF Blob Builder (stdlib-only)
# ============================================================================


def _build_tiff_ifd(
    tags: dict[int, Tuple[int, object]],
    endian: str = "<",
    base_offset: int = 0,
    follow_with_data: bool = True,
) -> Tuple[bytes, int]:
    """Build an IFD as little-endian (II) or big-endian (MM).

    Args:
        tags: {tag_number: (type, value)} where:
              type 2 = ASCII string, 4 = LONG (uint32), 3 = SHORT (uint16)
        endian: "<" for little-endian II, ">" for big-endian MM
        base_offset: Where this IFD starts in the TIFF file
        follow_with_data: If True, append data after IFD; else return data separately

    Returns: (ifd_bytes, next_data_offset)
    """
    entries = []
    data_parts = []
    # Data starts after IFD header (2), entries (12*count), and next-IFD offset (4)
    data_offset = base_offset + 2 + len(tags) * 12 + 4

    for tag, (typ, value) in sorted(tags.items()):
        if typ == 2:  # ASCII
            text = value.encode("ascii") + b"\x00"
            count = len(text)
            if len(text) <= 4:
                val_field = text.ljust(4, b"\x00")
            else:
                current_data_offset = data_offset + sum(len(p) for p in data_parts)
                val_field = struct.pack(endian + "I", current_data_offset)
                data_parts.append(text)
        elif typ == 4:  # LONG
            val_field = struct.pack(endian + "I", value)
            count = 1
        elif typ == 3:  # SHORT
            val_field = struct.pack(endian + "H", value) + b"\x00\x00"
            count = 1
        else:
            continue

        entries.append(struct.pack(
            endian + "HHI4s",
            tag, typ, count, val_field
        ))

    ifd = struct.pack(endian + "H", len(entries))
    ifd += b"".join(entries)
    ifd += struct.pack(endian + "I", 0)  # next IFD offset (0 = none)

    if follow_with_data:
        ifd += b"".join(data_parts)
        next_offset = data_offset + sum(len(p) for p in data_parts)
    else:
        next_offset = data_offset

    return ifd, next_offset


def _build_tiff_blob(
    ifd0_tags: dict[int, Tuple[int, object]],
    exif_tags: dict[int, Tuple[int, object]] | None = None,
    endian: str = "<",
) -> bytes:
    """Build a minimal TIFF blob with optional ExifIFD.

    Args:
        ifd0_tags: Tags for IFD0 (e.g., Model, DateTime)
        exif_tags: Tags for ExifIFD (e.g., DateTimeOriginal, SubSecTimeOriginal)
        endian: "<" for II (little-endian), ">" for MM (big-endian)

    Returns: Complete TIFF file blob.
    """
    magic = 42
    header = b"II" if endian == "<" else b"MM"
    header += struct.pack(endian + "HI", magic, 8)  # version + IFD0 offset at offset 8

    if not exif_tags:
        # Simple case: just IFD0
        ifd0, _ = _build_tiff_ifd(ifd0_tags, endian, base_offset=8, follow_with_data=True)
        return header + ifd0

    # Complex case with ExifIFD:
    # Calculate offsets carefully to avoid overlaps
    # IFD0 with one more tag for ExifIFD pointer
    num_ifd0_entries = len(ifd0_tags) + 1  # +1 for ExifIFD pointer
    ifd0_header_size = 2 + num_ifd0_entries * 12 + 4
    ifd0_data_start = 8 + ifd0_header_size

    # Build IFD0 entries with ExifIFD pointer added
    ifd0_tags_with_pointer = {**ifd0_tags, 34665: (4, 0)}  # placeholder offset

    # Calculate data offsets for IFD0 tags
    entries = []
    data_parts = []
    data_offset = ifd0_data_start

    for tag, (typ, value) in sorted(ifd0_tags_with_pointer.items()):
        if typ == 2:  # ASCII
            text = value.encode("ascii") + b"\x00"
            count = len(text)
            if len(text) <= 4:
                val_field = text.ljust(4, b"\x00")
            else:
                current_data_offset = data_offset + sum(len(p) for p in data_parts)
                val_field = struct.pack(endian + "I", current_data_offset)
                data_parts.append(text)
        elif typ == 4:  # LONG
            val_field = struct.pack(endian + "I", value)
            count = 1
        else:
            continue

        entries.append(struct.pack(
            endian + "HHI4s",
            tag, typ, count, val_field
        ))

    ifd0_data = b"".join(data_parts)
    exif_offset = ifd0_data_start + len(ifd0_data)

    # Now rebuild with correct ExifIFD offset
    ifd0_tags_with_pointer[34665] = (4, exif_offset)
    ifd0, _ = _build_tiff_ifd(ifd0_tags_with_pointer, endian, base_offset=8, follow_with_data=True)
    exif_blob, _ = _build_tiff_ifd(exif_tags, endian, base_offset=exif_offset, follow_with_data=True)

    return header + ifd0 + exif_blob


# ============================================================================
# Container Wrappers
# ============================================================================


def _build_jpeg_with_exif(tiff_blob: bytes) -> bytes:
    """Wrap TIFF blob as JPEG with APP1 EXIF segment."""
    # APP0 JFIF segment (minimal)
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00" + b"\x00\x01\x00\x01\x00\x00"

    # APP1 EXIF segment: FFE1 + length + "Exif\0\0" + TIFF
    exif_payload = b"Exif\x00\x00" + tiff_blob
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif_payload) + 2) + exif_payload

    # Minimal image data (SOS + EOI)
    sos = b"\xff\xda\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    eoi = b"\xff\xd9"

    return b"\xff\xd8" + app0 + app1 + sos + eoi


def _build_tiff_file(tiff_blob: bytes) -> bytes:
    """Return TIFF blob as-is (no wrapper)."""
    return tiff_blob


def _build_raf(tiff_blob: bytes) -> bytes:
    """Build Fuji RAF with embedded JPEG at offset 92 (after the offset/length at 84)."""
    jpeg = _build_jpeg_with_exif(tiff_blob)
    magic = b"FUJIFILMCCD-RAW " + b"\x00" * 68  # 84 bytes total
    jpeg_offset = 92  # After the 8-byte offset/length at bytes 84-91
    offset_len = struct.pack(">II", jpeg_offset, len(jpeg))
    return magic[:84] + offset_len + jpeg


def _build_cr3(ifd0_tags: dict, exif_tags: dict, endian: str = "<") -> bytes:
    """Build Canon CR3 with CMT1 (IFD0) and CMT2 (Exif) boxes."""
    # Build TIFF blobs for each box
    ifd0_tiff = _build_tiff_blob(ifd0_tags, None, endian)
    exif_tiff = _build_tiff_blob(exif_tags, None, endian)

    def _make_box(typ: bytes, data: bytes) -> bytes:
        size = len(data) + 8
        return struct.pack(">I", size) + typ + data

    cmt1 = _make_box(b"CMT1", ifd0_tiff)
    cmt2 = _make_box(b"CMT2", exif_tiff)

    # ftyp box
    ftyp = struct.pack(">I", 20) + b"ftyp" + b"crx " + struct.pack(">I", 0) + b"crx "

    # MDAT box (minimal, containing our boxes)
    mdat_data = cmt1 + cmt2
    mdat = struct.pack(">I", len(mdat_data) + 8) + b"mdat" + mdat_data

    return ftyp + mdat


def _build_png_with_exif(tiff_blob: bytes) -> bytes:
    """Build PNG with eXIf chunk."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk (minimal: 1x1 rgba)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xffffffff
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    # eXIf chunk
    exif_crc = zlib.crc32(b"eXIf" + tiff_blob) & 0xffffffff
    exif = struct.pack(">I", len(tiff_blob)) + b"eXIf" + tiff_blob + struct.pack(">I", exif_crc)

    # IDAT chunk (minimal: one transparent pixel)
    idat_data = zlib.compress(b"\x00\x00\x00\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xffffffff
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND") & 0xffffffff
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    return sig + ihdr + exif + idat + iend


def _build_webp_with_exif(tiff_blob: bytes) -> bytes:
    """Build WebP with EXIF chunk."""
    exif_payload = b"Exif\x00\x00" + tiff_blob
    # EXIF chunk: "EXIF" + length + payload (length is little-endian in WebP)
    exif_size = len(exif_payload) + (len(exif_payload) & 1)
    exif_chunk = b"EXIF" + struct.pack("<I", exif_size) + exif_payload

    if exif_size & 1:
        exif_chunk += b"\x00"

    # VP8X chunk (feature flags + width/height)
    vp8x_payload = b"\x10\x00\x00\x00" + b"\x00" * 4  # flags + dimensions
    vp8x_chunk = b"VP8X" + struct.pack("<I", len(vp8x_payload)) + vp8x_payload

    # RIFF container
    chunks = vp8x_chunk + exif_chunk
    riff_size = len(chunks) + 4
    return b"RIFF" + struct.pack("<I", riff_size) + b"WEBP" + chunks


# ============================================================================
# Tests
# ============================================================================


class TestParseExifDatetime:
    """Test _parse_exif_datetime parsing and subsecond handling."""

    def test_valid_datetime(self):
        """Parse valid EXIF datetime string."""
        result = _parse_exif_datetime("2026:09:20 14:32:05")
        assert result == datetime(2026, 9, 20, 14, 32, 5)

    def test_datetime_with_subsec_81(self):
        """Subseconds '81' -> 810000 microseconds."""
        result = _parse_exif_datetime("2026:09:20 14:32:05", "81")
        assert result == datetime(2026, 9, 20, 14, 32, 5, 810000)

    def test_datetime_with_subsec_5(self):
        """Subseconds '5' -> 500000 microseconds (padded)."""
        result = _parse_exif_datetime("2026:09:20 14:32:05", "5")
        assert result == datetime(2026, 9, 20, 14, 32, 5, 500000)

    def test_datetime_with_subsec_single_digit(self):
        """Subseconds '123456' handled correctly."""
        result = _parse_exif_datetime("2026:09:20 14:32:05", "123456")
        assert result == datetime(2026, 9, 20, 14, 32, 5, 123456)

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace and nulls are stripped."""
        result = _parse_exif_datetime("  2026:09:20 14:32:05\x00  ")
        assert result == datetime(2026, 9, 20, 14, 32, 5)

    def test_none_input(self):
        """None input returns None."""
        result = _parse_exif_datetime(None)
        assert result is None

    def test_empty_string(self):
        """Empty string returns None."""
        result = _parse_exif_datetime("")
        assert result is None

    def test_blank_with_spaces_and_colons(self):
        """Blank/spaces-only datetime returns None."""
        result = _parse_exif_datetime("    :  :     :  :  ")
        assert result is None

    def test_zeroed_clock(self):
        """All-zeros datetime returns None (camera clock not set)."""
        result = _parse_exif_datetime("0000:00:00 00:00:00")
        assert result is None

    def test_invalid_format(self):
        """Malformed datetime returns None."""
        result = _parse_exif_datetime("not a date")
        assert result is None

    def test_subsec_with_non_digits(self):
        """Non-digit characters in subsec are stripped."""
        result = _parse_exif_datetime("2026:09:20 14:32:05", "81abc")
        assert result == datetime(2026, 9, 20, 14, 32, 5, 810000)


class TestReadCaptureInfo:
    """Test read_capture_info with various formats."""

    def test_jpeg_with_datetime_original(self, tmp_path: Path):
        """Read JPEG with DateTimeOriginal in ExifIFD."""
        exif_tags = {36867: (2, "2026:09:20 14:32:05")}
        ifd0_tags = {272: (2, "Test Camera")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        jpeg = _build_jpeg_with_exif(tiff)

        path = tmp_path / "test.jpg"
        path.write_bytes(jpeg)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 20, 14, 32, 5)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Test Camera"

    def test_jpeg_fallback_to_datetime_digitized(self, tmp_path: Path):
        """JPEG falls back to DateTimeDigitized if DateTimeOriginal missing."""
        exif_tags = {36868: (2, "2026:09:19 10:20:30")}
        ifd0_tags = {272: (2, "Canon EOS")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        jpeg = _build_jpeg_with_exif(tiff)

        path = tmp_path / "test.jpg"
        path.write_bytes(jpeg)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 19, 10, 20, 30)
        assert info.source == "DateTimeDigitized"
        assert info.camera == "Canon EOS"

    def test_jpeg_fallback_to_datetime(self, tmp_path: Path):
        """JPEG falls back to IFD0 DateTime if Exif tags absent."""
        ifd0_tags = {306: (2, "2026:09:18 08:15:45"), 272: (2, "Nikon D850")}
        tiff = _build_tiff_blob(ifd0_tags, None)
        jpeg = _build_jpeg_with_exif(tiff)

        path = tmp_path / "test.jpg"
        path.write_bytes(jpeg)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 18, 8, 15, 45)
        assert info.source == "DateTime"
        assert info.camera == "Nikon D850"

    def test_tiff_file_direct(self, tmp_path: Path):
        """Read TIFF file directly (.tif, .dng, etc.)."""
        exif_tags = {36867: (2, "2026:09:17 16:45:00")}
        ifd0_tags = {272: (2, "Fuji X-T5")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)

        path = tmp_path / "test.tif"
        path.write_bytes(tiff)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 17, 16, 45, 0)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Fuji X-T5"

    def test_raf_file(self, tmp_path: Path):
        """Read Fuji RAF file (TIFF extracted from embedded JPEG)."""
        exif_tags = {36867: (2, "2026:09:16 12:30:15")}
        ifd0_tags = {272: (2, "Fuji X-Pro3")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        raf = _build_raf(tiff)

        path = tmp_path / "test.raf"
        path.write_bytes(raf)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 16, 12, 30, 15)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Fuji X-Pro3"

    def test_cr3_file(self, tmp_path: Path):
        """Read Canon CR3 (TIFF from CMT2 Exif box takes priority)."""
        ifd0_tags = {272: (2, "Canon EOS R5")}
        exif_tags = {36867: (2, "2026:09:15 09:00:45")}
        cr3 = _build_cr3(ifd0_tags, exif_tags)

        path = tmp_path / "test.cr3"
        path.write_bytes(cr3)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 15, 9, 0, 45)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Canon EOS R5"

    def test_png_with_exif(self, tmp_path: Path):
        """Read PNG with eXIf chunk."""
        exif_tags = {36867: (2, "2026:09:14 15:22:10")}
        ifd0_tags = {272: (2, "Smartphone")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        png = _build_png_with_exif(tiff)

        path = tmp_path / "test.png"
        path.write_bytes(png)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 14, 15, 22, 10)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Smartphone"

    def test_webp_with_exif(self, tmp_path: Path):
        """Read WebP with EXIF chunk."""
        exif_tags = {36867: (2, "2026:09:13 11:11:00")}
        ifd0_tags = {272: (2, "Google Pixel")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        webp = _build_webp_with_exif(tiff)

        path = tmp_path / "test.webp"
        path.write_bytes(webp)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 13, 11, 11, 0)
        assert info.source == "DateTimeOriginal"
        assert info.camera == "Google Pixel"

    def test_subsec_time_original_in_exif(self, tmp_path: Path):
        """SubSecTimeOriginal subseconds are included."""
        exif_tags = {36867: (2, "2026:09:12 07:33:22"), 37521: (2, "75")}
        ifd0_tags = {272: (2, "Sony A7R")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        jpeg = _build_jpeg_with_exif(tiff)

        path = tmp_path / "test.jpg"
        path.write_bytes(jpeg)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 12, 7, 33, 22, 750000)
        assert info.source == "DateTimeOriginal"

    def test_big_endian_tiff(self, tmp_path: Path):
        """Read big-endian (MM) TIFF blob."""
        exif_tags = {36867: (2, "2026:09:11 13:44:55")}
        ifd0_tags = {272: (2, "Big Endian Cam")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags, endian=">")

        path = tmp_path / "test.tif"
        path.write_bytes(tiff)
        info = read_capture_info(path)

        assert info.time == datetime(2026, 9, 11, 13, 44, 55)
        assert info.camera == "Big Endian Cam"

    def test_missing_file(self, tmp_path: Path):
        """Missing file returns time=None."""
        path = tmp_path / "nonexistent.jpg"
        info = read_capture_info(path)
        assert info.time is None

    def test_empty_file(self, tmp_path: Path):
        """Empty file returns time=None."""
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        info = read_capture_info(path)
        assert info.time is None

    def test_random_bytes(self, tmp_path: Path):
        """Random bytes (not a valid format) return time=None."""
        path = tmp_path / "garbage.jpg"
        path.write_bytes(b"this is not a valid image file")
        info = read_capture_info(path)
        assert info.time is None

    def test_truncated_jpeg(self, tmp_path: Path):
        """Truncated JPEG (cut mid-APP1) returns time=None."""
        exif_tags = {36867: (2, "2026:09:10 10:10:10")}
        ifd0_tags = {272: (2, "Truncated")}
        tiff = _build_tiff_blob(ifd0_tags, exif_tags)
        jpeg = _build_jpeg_with_exif(tiff)
        # Cut off partway through APP1
        truncated = jpeg[:50]

        path = tmp_path / "truncated.jpg"
        path.write_bytes(truncated)
        info = read_capture_info(path)
        assert info.time is None

    def test_tiff_offset_past_eof(self, tmp_path: Path):
        """TIFF with IFD offset pointing past EOF returns time=None."""
        # Build a minimal TIFF with bad IFD offset
        tiff = b"II" + struct.pack("<HI", 42, 9999)  # IFD offset way past EOF
        tiff += b"\x00" * 100  # pad to make it look real

        path = tmp_path / "bad_offset.tif"
        path.write_bytes(tiff)
        info = read_capture_info(path)
        assert info.time is None

    def test_tiff_ifd_too_many_entries(self, tmp_path: Path):
        """TIFF claiming 60000 entries returns time=None (bounds check)."""
        # Build TIFF with IFD claiming way too many entries
        tiff = b"II" + struct.pack("<HI", 42, 8)
        tiff += struct.pack("<H", 60000)  # claim 60000 entries (over _MAX_IFD_ENTRIES)
        tiff += b"\x00" * 100

        path = tmp_path / "too_many_ifd.tif"
        path.write_bytes(tiff)
        info = read_capture_info(path)
        assert info.time is None

    def test_no_datetime_tags(self, tmp_path: Path):
        """File with no datetime tags returns time=None."""
        ifd0_tags = {272: (2, "Camera")}  # model only, no datetime
        tiff = _build_tiff_blob(ifd0_tags, None)

        path = tmp_path / "no_datetime.tif"
        path.write_bytes(tiff)
        info = read_capture_info(path)
        assert info.time is None
        assert info.camera == "Camera"


class TestShotExtensions:
    """Test SHOT_EXTENSIONS constant."""

    def test_extensions_match_retouch_io(self):
        """SHOT_EXTENSIONS matches retouch.io.IMAGE_EXTENSIONS."""
        from retouch.io import IMAGE_EXTENSIONS

        assert SHOT_EXTENSIONS == IMAGE_EXTENSIONS, (
            f"SHOT_EXTENSIONS {sorted(SHOT_EXTENSIONS)} != "
            f"IMAGE_EXTENSIONS {sorted(IMAGE_EXTENSIONS)}"
        )


def test_reads_exif_ifd_written_by_pillow(tmp_path: Path):
    """Independent writer: Pillow nests a dict under 0x8769 as a real Exif IFD."""
    from PIL import Image

    exif = Image.Exif()
    exif[272] = "X-T5"
    exif[0x8769] = {36867: "2026:09:20 14:32:05", 37521: "42"}
    path = tmp_path / "pillow_exif.jpg"
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=exif)

    info = read_capture_info(path)

    assert info.time == datetime(2026, 9, 20, 14, 32, 5, 420000)
    assert info.source == "DateTimeOriginal"
    assert info.camera == "X-T5"

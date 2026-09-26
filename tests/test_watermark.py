"""Tests for retouch.watermark (credit / logo overlay on output copies)."""

import argparse

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch import watermark as wm


def _diff_box(a, b):
    """Bounding box (x0, y0, x1, y1) of pixels that differ between a and b."""
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    if d.ndim == 3:
        d = d.max(axis=2)
    ys, xs = np.nonzero(d)
    assert ys.size, "nothing changed"
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class TestSpec:
    def test_empty_spec_is_disabled_and_a_no_op(self):
        img = np.full((200, 300, 3), 90, np.uint8)
        spec = wm.WatermarkSpec(text="   ")
        assert not spec.enabled
        out, spot = wm.apply_watermark(img, spec)
        assert spot is None
        assert out is img

    @pytest.mark.parametrize("kwargs", [
        {"position": "middle"}, {"opacity": 1.5}, {"size": 0.5},
        {"color": "red"}, {"logo": "/no/such/logo.png"}, {"font": "/no/such.ttf"},
    ])
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            wm.WatermarkSpec(text="x", **kwargs)

    def test_year_placeholder(self):
        import datetime
        spec = wm.WatermarkSpec(text="© Me {year}")
        assert spec.resolved_text() == f"© Me {datetime.date.today().year}"


class TestPlacement:
    def test_named_corners(self):
        W, H, mw, mh, m = 1000, 800, 200, 50, 20
        assert wm.choose_position(W, H, mw, mh, m, position="bottom-right")[1:] == (780, 730)
        assert wm.choose_position(W, H, mw, mh, m, position="top-left")[1:] == (20, 20)
        assert wm.choose_position(W, H, mw, mh, m, position="bottom-center")[1:] == (400, 730)

    def test_auto_prefers_bottom_right_without_faces(self):
        assert wm.choose_position(1000, 800, 200, 50, 20)[0] == "bottom-right"

    def test_auto_moves_off_a_face_in_the_corner(self):
        face = (760, 600, 150, 150)  # bottom-right
        spot = wm.choose_position(1000, 800, 200, 50, 20, [face])[0]
        assert spot == "bottom-left"

    def test_auto_counts_headroom_around_the_face(self):
        # Box ends above the corner mark, but the padded box (wig/chin) reaches it.
        face = (760, 560, 150, 120)
        spot = wm.choose_position(1000, 800, 200, 50, 20, [face])[0]
        assert spot != "bottom-right"

    def test_auto_takes_least_overlap_when_every_spot_is_covered(self):
        big = (0, 0, 1000, 800)
        small_bottom_left = (0, 700, 150, 100)
        spot = wm.choose_position(1000, 800, 200, 50, 20, [big])[0]
        assert spot in wm.AUTO_ORDER
        spot2 = wm.choose_position(
            1000, 800, 200, 50, 20, [(700, 0, 300, 800), small_bottom_left])[0]
        assert spot2 in ("top-left", "bottom-center")

    def test_stamp_lands_clear_of_the_face(self):
        img = np.full((600, 800, 3), 60, np.uint8)
        face = (600, 420, 150, 150)
        out, spot = wm.apply_watermark(img, wm.WatermarkSpec(text="© Studio"), [face])
        x0, y0, x1, y1 = _diff_box(img, out)
        assert spot == "bottom-left"
        assert x1 < 600  # nothing touched under the face


class TestRendering:
    def test_white_on_dark_black_on_light(self):
        dark = np.full((400, 600, 3), 30, np.uint8)
        light = np.full((400, 600, 3), 225, np.uint8)
        spec = wm.WatermarkSpec(text="© Studio", opacity=1.0)
        out_d, _ = wm.apply_watermark(dark, spec)
        out_l, _ = wm.apply_watermark(light, spec)
        assert out_d.max() > 200
        assert out_l.min() < 60

    def test_forced_colour(self):
        light = np.full((400, 600, 3), 225, np.uint8)
        out, _ = wm.apply_watermark(light, wm.WatermarkSpec(text="Hi", color="white", opacity=1.0))
        assert out.max() == 255

    def test_opacity_scales_the_change(self):
        img = np.full((400, 600, 3), 40, np.uint8)
        lo, _ = wm.apply_watermark(img, wm.WatermarkSpec(text="Hi", opacity=0.3))
        hi, _ = wm.apply_watermark(img, wm.WatermarkSpec(text="Hi", opacity=0.9))
        assert hi.max() > lo.max()

    @pytest.mark.parametrize("dtype,value", [
        (np.uint8, 50), (np.uint16, 50 * 257), (np.float32, 50.0), (np.float32, 50 / 255.0),
    ])
    def test_keeps_dtype_and_range(self, dtype, value):
        img = np.full((300, 400, 3), value, dtype)
        orig = img.copy()
        out, spot = wm.apply_watermark(img, wm.WatermarkSpec(text="© A"))
        assert spot == "bottom-right"
        assert out.dtype == img.dtype
        scale = wm._value_scale(img)
        assert out.max() <= scale + 1e-6
        assert out.max() > img.max()  # white text on a dark frame
        assert np.array_equal(img, orig)  # input untouched

    def test_grayscale_image(self):
        img = np.full((300, 400), 40, np.uint8)
        out, _ = wm.apply_watermark(img, wm.WatermarkSpec(text="© A"))
        assert out.shape == img.shape and out.max() > 40

    def test_mark_scales_with_short_side(self):
        spec = wm.WatermarkSpec(text="© Studio")
        small = wm.build_mark(spec, 1080)
        big = wm.build_mark(spec, 4000)
        ratio = big.size[1] / small.size[1]
        assert 3.0 < ratio < 4.4

    def test_only_the_mark_area_changes(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (500, 700, 3), dtype=np.uint8)
        out, _ = wm.apply_watermark(img, wm.WatermarkSpec(text="© Studio"))
        x0, y0, x1, y1 = _diff_box(img, out)
        assert x0 > 700 * 0.4 and y0 > 500 * 0.7

    def test_too_small_image_is_left_alone(self):
        img = np.full((20, 30, 3), 40, np.uint8)
        out, spot = wm.apply_watermark(img, wm.WatermarkSpec(text="A long studio credit line"))
        assert spot is None and np.array_equal(out, img)

    def test_logo_keeps_its_colour(self, tmp_path):
        logo = np.zeros((40, 40, 4), np.uint8)
        logo[..., 2] = 255  # red in BGRA
        logo[..., 3] = 255
        path = tmp_path / "logo.png"
        cv2.imwrite(str(path), logo)
        img = np.full((400, 600, 3), 30, np.uint8)
        out, _ = wm.apply_watermark(img, wm.WatermarkSpec(logo=path, opacity=1.0))
        x0, y0, x1, y1 = _diff_box(img, out)
        patch = out[y0:y1, x0:x1].reshape(-1, 3)
        reddest = patch[np.argmax(patch[:, 2].astype(int) - patch[:, 0])]
        assert reddest[2] > 200 and reddest[0] < 60


class TestFont:
    def test_builtin_font_covers_credit_symbols(self):
        font = wm.load_font(40)
        assert wm.missing_glyphs("© @studio 2026 · Photo", font) == []

    def test_reports_characters_the_builtin_font_lacks(self, caplog):
        font = wm.load_font(40)
        assert "é" in wm.missing_glyphs("Chloé", font)
        with caplog.at_level("WARNING", logger="retouch.watermark"):
            wm.build_mark(wm.WatermarkSpec(text="Chloé"), 1000)
        assert "no glyph" in caplog.text


class TestExif:
    def test_copyright_sign_becomes_ascii(self):
        raw = wm.credit_exif("© Alex Studio")
        exif = Image.Exif()
        exif.load(raw)
        assert exif[0x8298] == "(C) Alex Studio"

    def test_non_ascii_credit_gets_no_field(self):
        assert wm.credit_exif("写真 Alex") is None
        assert wm.credit_exif("") is None


class TestFiles:
    def _photo(self, path, value=60):
        cv2.imwrite(str(path), np.full((400, 600, 3), value, np.uint8))
        return path

    def test_writes_a_copy_and_leaves_the_master(self, tmp_path):
        src = self._photo(tmp_path / "a.png")
        before = src.read_bytes()
        spec = wm.WatermarkSpec(text="© Alex", position="bottom-left")
        res = wm.export_watermarked(src, tmp_path / "watermarked", spec)
        assert res.status == "done" and res.spot == "bottom-left"
        assert res.path == tmp_path / "watermarked" / "a.jpg"
        assert src.read_bytes() == before
        with Image.open(res.path) as im:
            assert im.getexif().get(0x8298) == "(C) Alex"
            assert im.size == (600, 400)

    def test_skips_existing_unless_forced(self, tmp_path):
        src = self._photo(tmp_path / "a.png")
        spec = wm.WatermarkSpec(text="x", position="top-left")
        out_dir = tmp_path / "wm"
        assert wm.export_watermarked(src, out_dir, spec).status == "done"
        assert wm.export_watermarked(src, out_dir, spec).status == "skipped"
        assert wm.export_watermarked(src, out_dir, spec, force=True).status == "done"

    def test_unreadable_file_fails_without_raising(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        summary = wm.export_folder([bad], tmp_path / "wm",
                                   wm.WatermarkSpec(text="x", position="top-left"))
        assert summary["failed"] == 1

    def test_cli_main(self, tmp_path):
        self._photo(tmp_path / "a.png")
        self._photo(tmp_path / "b.jpg")
        rc = wm.main([str(tmp_path), "--text", "© Me", "--position", "top-right",
                      "--opacity", "80"])
        assert rc == 0
        assert sorted(p.name for p in (tmp_path / "watermarked").iterdir()) == ["a.jpg", "b.jpg"]

    def test_cli_needs_text_or_logo(self, tmp_path):
        with pytest.raises(SystemExit):
            wm.main([str(tmp_path)])

    def test_spec_from_prefixed_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--watermark")
        wm.add_cli_args(parser, prefix="watermark-")
        args = parser.parse_args(["--watermark", "x", "--watermark-opacity", "50",
                                  "--watermark-position", "top-left", "--watermark-size", "5"])
        spec = wm.spec_from_args(args.watermark, args, prefix="watermark-")
        assert (spec.opacity, spec.position, spec.size) == (0.5, "top-left", 0.05)
        args.watermark_opacity = 150
        with pytest.raises(ValueError):
            wm.spec_from_args("x", args, prefix="watermark-")


class TestSocialAndReviewIntegration:
    def test_watermarked_folder_is_not_a_crop_source(self, tmp_path):
        from retouch.social_crops import find_crop_sources

        cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
        (tmp_path / "watermarked").mkdir()
        cv2.imwrite(str(tmp_path / "watermarked" / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
        assert [p.name for p in find_crop_sources(tmp_path, recursive=True)] == ["a.jpg"]

    def test_review_page_skips_watermarked(self):
        from retouch.review_page import _BACKFILL_SKIP_DIRS

        assert "watermarked" in _BACKFILL_SKIP_DIRS

    def test_crop_watermark_maps_faces_into_the_crop(self):
        from retouch.social_crops import CropPlan, _watermark_crop

        crop = np.full((500, 400, 3), 50, np.uint8)
        # Face at the crop's bottom right, given in full-image pixels (crop at x=1000).
        plan = CropPlan(x=1000, y=0, w=800, h=1000, reason="face", n_faces=1)
        face_full = (1000 + 560, 740, 240, 240)
        out = _watermark_crop(crop, plan, [face_full], wm.WatermarkSpec(text="© A"))
        x0, y0, x1, y1 = _diff_box(crop, out)
        assert x1 <= 400 * 0.6  # moved to the left, away from the face


class TestGuiHandlers:
    def test_stamp_needs_text(self):
        gui = pytest.importorskip("gui")
        path, msg = gui.on_stamp_watermark(None, None, "", "auto", 70)
        assert path is None and "credit" in msg

    def test_stamp_needs_a_photo(self):
        gui = pytest.importorskip("gui")
        path, msg = gui.on_stamp_watermark(None, None, "© A", "top-left", 70)
        assert path is None and "Process a photo" in msg

    def test_stamps_the_preview(self):
        gui = pytest.importorskip("gui")
        rgb = np.full((300, 400, 3), 40, np.uint8)
        path, msg = gui.on_stamp_watermark(None, rgb, "© A", "top-left", 70)
        out = cv2.imread(path)
        assert out.shape == (300, 400, 3)
        assert out[:60, :200].max() > 150 and out[200:, 250:].max() < 60

    def test_stamps_the_exported_file(self, tmp_path):
        gui = pytest.importorskip("gui")
        src = tmp_path / "export.png"
        cv2.imwrite(str(src), np.full((300, 400, 3), 40, np.uint8))
        path, msg = gui.on_stamp_watermark(str(src), None, "© A", "bottom-left", 70)
        assert path.endswith("export.jpg") and "exported" in msg

    def test_batch_spec(self):
        from gui_batch import _batch_watermark_spec

        assert _batch_watermark_spec("  ", "auto") is None
        assert _batch_watermark_spec("© A", "top-left").position == "top-left"


class TestBatchCli:
    def test_flags_parse_and_validate(self, monkeypatch, tmp_path):
        import cli

        monkeypatch.setattr("sys.argv", ["cli.py", str(tmp_path), "-o", str(tmp_path / "o"),
                                         "--watermark", "", "--dry-run"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2

"""Unit tests for retouch.social_crops (platform crop-export helper).

Written against the contract for a module the lead is authoring in parallel;
see the task brief for the full spec. Uses only synthetic numpy/cv2 images
and fake detector objects -- no MediaPipe, no real photos.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.social_crops import (
    FORMATS,
    DEFAULT_FORMATS,
    CropResult,
    parse_formats,
    select_subject_faces,
    plan_crop,
    render_crop,
    detect_faces,
    export_social_crops,
    find_crop_sources,
    export_folder,
    main,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_image(w, h, color=(120, 140, 160)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def write_image(path: Path, w, h, color=(120, 140, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), make_image(w, h, color))
    return path


class FakeDetector:
    """Stand-in for a MediaPipe-backed detector: records the image it was
    given and returns fixed boxes (in that same image's pixel space)."""

    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.seen_shape = None
        self.calls = 0

    def detect(self, img_small):
        self.calls += 1
        self.seen_shape = img_small.shape[:2]  # (h, w)
        return [SimpleNamespace(bbox=b) for b in self.boxes]


IMAGE_SIZES = [
    ("landscape", 6000, 4000),
    ("portrait", 4000, 6000),
    ("square", 3000, 3000),
]

FORMAT_KEYS = ["4:5", "9:16", "1:1", "3:4"]


def expected_max_crop(img_w, img_h, ratio):
    """Ground truth for 'largest rectangle of this aspect that fits'."""
    rw, rh = ratio
    h_for_full_width = img_w * rh / rw
    if h_for_full_width <= img_h + 1e-6:
        return img_w, round(h_for_full_width)
    w_for_full_height = img_h * rw / rh
    return round(w_for_full_height), img_h


def fmts(*keys):
    """export_social_crops/export_folder take SocialFormat objects (as
    parse_formats returns), not bare key strings -- this mirrors that."""
    return [FORMATS[k] for k in keys]


# ---------------------------------------------------------------------------
# SocialFormat / FORMATS / DEFAULT_FORMATS
# ---------------------------------------------------------------------------

class TestFormats:
    def test_formats_has_four_entries(self):
        assert set(FORMATS.keys()) == {"4:5", "9:16", "1:1", "3:4"}

    @pytest.mark.parametrize(
        "key,width,height",
        [
            ("4:5", 1080, 1350),
            ("9:16", 1080, 1920),
            ("1:1", 1080, 1080),
            ("3:4", 1080, 1440),
        ],
    )
    def test_format_dimensions(self, key, width, height):
        fmt = FORMATS[key]
        assert fmt.width == width
        assert fmt.height == height
        assert fmt.key == key

    def test_format_is_frozen(self):
        fmt = FORMATS["4:5"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            fmt.width = 999

    def test_format_slug_label_and_ratio_for_every_entry(self):
        # One test, all four formats -- avoids four near-duplicate node names
        # for what is really a single "shape of SocialFormat" check.
        assert FORMATS["4:5"].slug == "4x5"
        for fmt in FORMATS.values():
            assert isinstance(fmt.label, str) and fmt.label
            rw, rh = fmt.ratio
            assert abs(fmt.width / fmt.height - rw / rh) < 1e-6

    def test_default_formats(self):
        assert DEFAULT_FORMATS == ("4:5", "9:16", "1:1")
        assert "3:4" not in DEFAULT_FORMATS


# ---------------------------------------------------------------------------
# parse_formats
# ---------------------------------------------------------------------------

def _keys(parsed):
    """parse_formats returns SocialFormat objects, not bare key strings --
    reduce to keys for easy comparison against the spec's key spelling."""
    return tuple(f.key for f in parsed)


class TestParseFormats:
    def test_accepted_spec_forms(self):
        # Every valid spelling/shape parse_formats must accept, checked in
        # one place rather than one node per spelling.
        assert _keys(parse_formats("4:5")) == ("4:5",)
        assert _keys(parse_formats("4x5")) == ("4:5",)
        assert _keys(parse_formats("4:5,9:16")) == ("4:5", "9:16")
        assert _keys(parse_formats("4x5,9:16")) == ("4:5", "9:16")
        assert _keys(parse_formats(" 4:5 , 9:16 ")) == ("4:5", "9:16")
        assert _keys(parse_formats("4:5,9:16,4:5")) == ("4:5", "9:16")  # dedup, order kept
        assert _keys(parse_formats(["4:5", "1:1"])) == ("4:5", "1:1")
        assert _keys(parse_formats("all")) == tuple(FORMATS.keys())
        # elements are the actual SocialFormat objects from FORMATS
        assert parse_formats("4:5") == [FORMATS["4:5"]]

    @pytest.mark.parametrize("bad_spec", ["16:9", "", [], "bogus", "4:5,bogus"])
    def test_invalid_spec_raises(self, bad_spec):
        with pytest.raises(ValueError):
            parse_formats(bad_spec)


# ---------------------------------------------------------------------------
# select_subject_faces
# ---------------------------------------------------------------------------

class TestSelectSubjectFaces:
    def test_empty_in_empty_out(self):
        assert select_subject_faces([]) == []

    def test_single_face_kept(self):
        boxes = [(10, 10, 100, 100)]
        assert select_subject_faces(boxes) == boxes

    def test_small_background_face_dropped(self):
        main_box = (100, 100, 200, 200)  # area 40000
        bg_box = (0, 0, 50, 50)  # area 2500, ratio 0.0625 < 0.35
        result = select_subject_faces([main_box, bg_box])
        assert result == [main_box]

    def test_similar_size_faces_both_kept(self):
        a = (100, 100, 200, 200)
        b = (400, 100, 190, 190)  # area ratio ~0.9025, well above 0.35
        result = select_subject_faces([a, b])
        assert a in result and b in result
        assert len(result) == 2

    def test_custom_threshold(self):
        main_box = (0, 0, 200, 200)  # area 40000
        mid_box = (300, 0, 100, 100)  # area 10000, ratio 0.25
        assert select_subject_faces([main_box, mid_box], min_rel_area=0.35) == [main_box]
        result = select_subject_faces([main_box, mid_box], min_rel_area=0.2)
        assert main_box in result and mid_box in result

    def test_boundary_at_threshold_kept(self):
        main_box = (0, 0, 100, 100)  # area 10000
        # area exactly 0.35 * 10000 = 3500 -> side ~59.16, use 60x60=3600 (>=)
        boxes = [main_box, (0, 0, 60, 60)]
        result = select_subject_faces(boxes, min_rel_area=0.35)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# plan_crop: geometry invariants (no faces)
# ---------------------------------------------------------------------------

GEOMETRY_CASES = [
    (name, img_w, img_h, fmt_key)
    for (name, img_w, img_h) in IMAGE_SIZES
    for fmt_key in FORMAT_KEYS
]


class TestPlanCropGeometry:
    @pytest.mark.parametrize("name,img_w,img_h,fmt_key", GEOMETRY_CASES)
    def test_crop_geometry_invariants(self, name, img_w, img_h, fmt_key):
        """Bounds, max-fit size and aspect ratio all in one pass per case,
        to cover every (size, format) combination without tripling the
        parametrize count."""
        fmt = FORMATS[fmt_key]
        plan = plan_crop(img_w, img_h, [], fmt)

        # inside the image
        assert plan.x >= 0 and plan.y >= 0
        assert plan.x + plan.w <= img_w
        assert plan.y + plan.h <= img_h

        # largest rectangle of this aspect that fits
        exp_w, exp_h = expected_max_crop(img_w, img_h, fmt.ratio)
        assert abs(plan.w - exp_w) <= 1
        assert abs(plan.h - exp_h) <= 1
        assert plan.w == img_w or plan.h == img_h

        # aspect ratio matches within ~1px rounding
        rw, rh = fmt.ratio
        assert abs(plan.w * rh - plan.h * rw) <= max(rw, rh)

    @pytest.mark.parametrize("name,img_w,img_h", IMAGE_SIZES)
    def test_no_face_is_horizontally_centered(self, name, img_w, img_h):
        fmt = FORMATS["1:1"]
        plan = plan_crop(img_w, img_h, [], fmt)
        assert plan.reason == "no_face"
        assert plan.n_faces == 0
        crop_center_x = plan.x + plan.w / 2
        assert abs(crop_center_x - img_w / 2) <= 1.0


# ---------------------------------------------------------------------------
# plan_crop: single-face behavior
# ---------------------------------------------------------------------------

class TestPlanCropSingleFace:
    def test_crop_contains_face_when_it_fits(self):
        img_w, img_h = 4000, 6000
        fmt = FORMATS["4:5"]
        # face well inside the image, small relative to the crop
        face = (1800, 2500, 300, 300)
        plan = plan_crop(img_w, img_h, [face], fmt)
        fx, fy, fw, fh = face
        assert plan.n_faces == 1
        assert plan.reason == "face"
        assert plan.x <= fx
        assert plan.y <= fy
        assert plan.x + plan.w >= fx + fw
        assert plan.y + plan.h >= fy + fh

    def test_face_center_x_matches_crop_center_when_unconstrained(self):
        img_w, img_h = 6000, 4000
        fmt = FORMATS["1:1"]  # crop w < img_w so there's room to center
        face = (3000, 1800, 200, 200)
        plan = plan_crop(img_w, img_h, [face], fmt)
        face_center_x = 3000 + 100
        crop_center_x = plan.x + plan.w / 2
        # face is far from either edge -> should be centered exactly (bar rounding)
        assert abs(crop_center_x - face_center_x) <= 1.0

    def test_face_center_x_clamped_near_left_edge(self):
        img_w, img_h = 6000, 4000
        fmt = FORMATS["1:1"]
        face = (10, 1800, 200, 200)  # near left edge
        plan = plan_crop(img_w, img_h, [face], fmt)
        # crop can't extend past x=0, so it should hug the left edge
        assert plan.x == 0

    def test_headroom_added_above_face_when_room_exists(self):
        img_w, img_h = 2000, 6000
        fmt = FORMATS["9:16"]  # tall crop, plenty of vertical room
        face = (900, 2800, 200, 200)
        plan = plan_crop(img_w, img_h, [face], fmt)
        fx, fy, fw, fh = face
        assert plan.y <= fy - 0.5 * fh + 1.0

    def test_crop_never_cuts_through_face_near_top(self):
        img_w, img_h = 2000, 6000
        fmt = FORMATS["9:16"]
        face = (900, 5, 200, 200)  # almost at the top, no room for full headroom
        plan = plan_crop(img_w, img_h, [face], fmt)
        fx, fy, fw, fh = face
        # crop top must not be below the face's top (that would cut the face)
        assert plan.y <= fy
        if fh <= plan.h:
            assert plan.y + plan.h >= fy + fh

    def test_small_background_face_ignored(self):
        img_w, img_h = 4000, 6000
        fmt = FORMATS["4:5"]
        main_face = (1800, 2500, 400, 400)
        bg_face = (100, 100, 60, 60)  # area ratio well under 0.35
        plan = plan_crop(img_w, img_h, [main_face, bg_face], fmt)
        assert plan.n_faces == 1
        assert plan.reason == "face"


# ---------------------------------------------------------------------------
# plan_crop: multi-face behavior
# ---------------------------------------------------------------------------

class TestPlanCropMultiFace:
    def test_two_similar_faces_are_group(self):
        img_w, img_h = 6000, 4000
        fmt = FORMATS["1:1"]
        a = (2200, 1800, 300, 300)
        b = (2900, 1800, 290, 290)
        plan = plan_crop(img_w, img_h, [a, b], fmt)
        assert plan.reason == "group"
        assert plan.n_faces == 2

    def test_group_both_faces_contained_when_they_fit(self):
        img_w, img_h = 6000, 4000
        fmt = FORMATS["1:1"]
        a = (2200, 1800, 300, 300)
        b = (2900, 1800, 290, 290)
        plan = plan_crop(img_w, img_h, [a, b], fmt)
        for (fx, fy, fw, fh) in (a, b):
            assert plan.x <= fx
            assert plan.y <= fy
            assert plan.x + plan.w >= fx + fw
            assert plan.y + plan.h >= fy + fh

    def test_faces_wider_than_crop_reason(self):
        img_w, img_h = 6000, 2000
        fmt = FORMATS["9:16"]  # narrow crop
        # two faces spread far apart, symmetric about the image's horizontal center
        a = (500, 800, 200, 200)
        b = (5300, 800, 200, 200)
        plan = plan_crop(img_w, img_h, [a, b], fmt)
        assert plan.reason == "faces_wider_than_crop"
        assert plan.n_faces == 2

    def test_faces_wider_than_crop_centered_on_group(self):
        img_w, img_h = 6000, 2000
        fmt = FORMATS["9:16"]
        a = (500, 800, 200, 200)
        b = (5300, 800, 200, 200)
        plan = plan_crop(img_w, img_h, [a, b], fmt)
        group_center_x = ((500 + 700) + (5300 + 5500)) / 2 / 2  # mean of the two centers
        crop_center_x = plan.x + plan.w / 2
        # symmetric placement -> group center is the image's horizontal midpoint
        assert abs(crop_center_x - img_w / 2) <= max(5.0, 0.02 * img_w)
        assert abs(group_center_x - img_w / 2) < 1e-6  # sanity on the fixture itself


# ---------------------------------------------------------------------------
# render_crop
# ---------------------------------------------------------------------------

class TestRenderCrop:
    def test_platform_size_matches_format_dims(self):
        img = make_image(4000, 6000)
        fmt = FORMATS["4:5"]
        plan = plan_crop(4000, 6000, [], fmt)
        out = render_crop(img, plan, fmt, size="platform")
        assert out.shape == (fmt.height, fmt.width, 3)
        assert out.dtype == np.uint8

    def test_full_size_matches_crop_dims(self):
        img = make_image(4000, 6000)
        fmt = FORMATS["4:5"]
        plan = plan_crop(4000, 6000, [], fmt)
        out = render_crop(img, plan, fmt, size="full")
        assert out.shape == (plan.h, plan.w, 3)

    def test_platform_never_upscales_small_crop(self):
        # tiny image whose max-fitting crop is far smaller than the platform target
        img = make_image(200, 250)
        fmt = FORMATS["4:5"]  # target 1080x1350
        plan = plan_crop(200, 250, [], fmt)
        assert plan.w <= 200 and plan.h <= 250
        out = render_crop(img, plan, fmt, size="platform")
        assert out.shape == (plan.h, plan.w, 3)
        assert out.shape != (fmt.height, fmt.width, 3)

    def test_sharpen_zero_matches_plain_inter_area_resize_platform(self):
        img = make_image(4000, 6000, color=(50, 100, 200))
        # give it some texture so resize isn't a flat no-op
        rng = np.random.default_rng(0)
        img = img.astype(np.int16)
        img += rng.integers(-20, 20, size=img.shape, dtype=np.int16)
        img = np.clip(img, 0, 255).astype(np.uint8)

        fmt = FORMATS["1:1"]
        plan = plan_crop(4000, 6000, [], fmt)
        cropped = img[plan.y:plan.y + plan.h, plan.x:plan.x + plan.w]
        expected = cv2.resize(cropped, (fmt.width, fmt.height), interpolation=cv2.INTER_AREA)

        out = render_crop(img, plan, fmt, size="platform", sharpen=0)
        assert np.array_equal(out, expected)

    def test_sharpen_zero_full_size_matches_raw_crop(self):
        img = make_image(4000, 6000, color=(50, 100, 200))
        fmt = FORMATS["1:1"]
        plan = plan_crop(4000, 6000, [], fmt)
        cropped = img[plan.y:plan.y + plan.h, plan.x:plan.x + plan.w]
        out = render_crop(img, plan, fmt, size="full", sharpen=0)
        assert np.array_equal(out, cropped)

    def test_output_is_uint8_bgr_shape(self):
        img = make_image(3000, 3000)
        fmt = FORMATS["1:1"]
        plan = plan_crop(3000, 3000, [], fmt)
        out = render_crop(img, plan, fmt)
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# detect_faces
# ---------------------------------------------------------------------------

class TestDetectFaces:
    def test_no_faces_returns_empty(self):
        img = make_image(800, 600)
        det = FakeDetector([])
        assert detect_faces(img, detector=det) == []
        assert det.calls == 1

    def test_boxes_scaled_back_to_full_image(self):
        img = make_image(4000, 3000)
        # Give the detector a box in whatever small-image space it's called with,
        # and verify the returned box maps back to (approximately) the same
        # fractional location in the full image.
        rel_box = (0.25, 0.30, 0.10, 0.15)  # x,y,w,h as fractions

        class RelDetector:
            def __init__(self):
                self.seen_shape = None

            def detect(self, img_small):
                self.seen_shape = img_small.shape[:2]
                sh, sw = self.seen_shape
                x = int(rel_box[0] * sw)
                y = int(rel_box[1] * sh)
                w = int(rel_box[2] * sw)
                h = int(rel_box[3] * sh)
                return [SimpleNamespace(bbox=(x, y, w, h))]

        rd = RelDetector()
        boxes = detect_faces(img, detector=rd, max_side=1280)
        assert len(boxes) == 1
        x, y, w, h = boxes[0]
        img_w, img_h = 4000, 3000
        assert abs(x / img_w - rel_box[0]) < 0.02
        assert abs(y / img_h - rel_box[1]) < 0.02
        assert abs(w / img_w - rel_box[2]) < 0.02
        assert abs(h / img_h - rel_box[3]) < 0.02

    def test_max_side_bounds_small_image_dimension(self):
        img = make_image(4000, 3000)
        det = FakeDetector([(0, 0, 10, 10)])
        detect_faces(img, detector=det, max_side=1280)
        sh, sw = det.seen_shape
        assert max(sh, sw) <= 1280

    def test_small_image_not_upscaled(self):
        img = make_image(600, 400)
        det = FakeDetector([(10, 10, 20, 20)])
        detect_faces(img, detector=det, max_side=1280)
        sh, sw = det.seen_shape
        assert sw <= 600 and sh <= 400

    def test_multiple_boxes_all_returned(self):
        img = make_image(800, 600)
        det = FakeDetector([(10, 10, 20, 20), (100, 100, 30, 30)])
        boxes = detect_faces(img, detector=det)
        assert len(boxes) == 2


# ---------------------------------------------------------------------------
# export_social_crops
# ---------------------------------------------------------------------------

class TestExportSocialCrops:
    def test_writes_one_file_per_format(self, tmp_path):
        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([(300, 200, 100, 100)])
        results = export_social_crops(
            img_path, social_dir, fmts("4:5", "1:1"), detector=det,
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, CropResult)
            assert r.status == "done"
            assert r.plan is not None
            slug = FORMATS[r.fmt_key].slug
            expected_path = social_dir / slug / f"photo_{slug}.jpg"
            assert expected_path.exists()

    def test_output_has_no_exif(self, tmp_path):
        from PIL import Image

        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        export_social_crops(img_path, social_dir, fmts("1:1"), detector=det)
        slug = FORMATS["1:1"].slug
        out_path = social_dir / slug / f"photo_{slug}.jpg"
        with Image.open(out_path) as im:
            exif = im.getexif()
            assert len(exif) == 0

    def test_existing_file_skipped_without_force(self, tmp_path):
        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        export_social_crops(img_path, social_dir, fmts("1:1"), detector=det)

        slug = FORMATS["1:1"].slug
        out_path = social_dir / slug / f"photo_{slug}.jpg"
        mtime_before = out_path.stat().st_mtime_ns

        results = export_social_crops(img_path, social_dir, fmts("1:1"), detector=det)
        assert results[0].status == "skipped"
        assert out_path.stat().st_mtime_ns == mtime_before

    def test_force_overwrites_existing_file(self, tmp_path):
        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        export_social_crops(img_path, social_dir, fmts("1:1"), detector=det)
        results = export_social_crops(
            img_path, social_dir, fmts("1:1"), detector=det, force=True,
        )
        assert results[0].status == "done"

    def test_result_fmt_keys_match_requested(self, tmp_path):
        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        results = export_social_crops(
            img_path, social_dir, fmts("4:5", "9:16", "1:1"), detector=det,
        )
        assert {r.fmt_key for r in results} == {"4:5", "9:16", "1:1"}

    def test_result_path_is_pathlib_path(self, tmp_path):
        img_path = write_image(tmp_path / "photo.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        results = export_social_crops(img_path, social_dir, fmts("1:1"), detector=det)
        assert isinstance(results[0].path, Path)


# ---------------------------------------------------------------------------
# find_crop_sources
# ---------------------------------------------------------------------------

class TestFindCropSources:
    def test_finds_images_in_folder(self, tmp_path):
        write_image(tmp_path / "a.jpg", 100, 100)
        write_image(tmp_path / "b.png", 100, 100)
        (tmp_path / "notes.txt").write_text("hello")
        found = find_crop_sources(tmp_path)
        names = {p.name for p in found}
        assert names == {"a.jpg", "b.png"}

    def test_skips_compare_suffixed_files(self, tmp_path):
        write_image(tmp_path / "a.jpg", 100, 100)
        write_image(tmp_path / "a_compare.jpg", 100, 100)
        found = find_crop_sources(tmp_path)
        names = {p.name for p in found}
        assert "a_compare.jpg" not in names
        assert "a.jpg" in names

    def test_skips_social_subfolder(self, tmp_path):
        write_image(tmp_path / "a.jpg", 100, 100)
        write_image(tmp_path / "social" / "4x5" / "a_4x5.jpg", 100, 100)
        found = find_crop_sources(tmp_path, recursive=True)
        assert all("social" not in p.parts for p in found)
        assert any(p.name == "a.jpg" for p in found)

    def test_non_recursive_ignores_subfolders(self, tmp_path):
        write_image(tmp_path / "top.jpg", 100, 100)
        write_image(tmp_path / "sub" / "nested.jpg", 100, 100)
        found = find_crop_sources(tmp_path, recursive=False)
        names = {p.name for p in found}
        assert "top.jpg" in names
        assert "nested.jpg" not in names

    def test_recursive_finds_nested_files(self, tmp_path):
        write_image(tmp_path / "top.jpg", 100, 100)
        write_image(tmp_path / "sub" / "nested.jpg", 100, 100)
        found = find_crop_sources(tmp_path, recursive=True)
        names = {p.name for p in found}
        assert "top.jpg" in names
        assert "nested.jpg" in names

    def test_single_file_input_returns_that_file(self, tmp_path):
        img_path = write_image(tmp_path / "solo.jpg", 100, 100)
        found = find_crop_sources(img_path)
        assert found == [img_path]

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert find_crop_sources(tmp_path) == []


# ---------------------------------------------------------------------------
# export_folder
# ---------------------------------------------------------------------------

class TestExportFolder:
    def test_counts_and_results_consistent(self, tmp_path):
        p1 = write_image(tmp_path / "a.jpg", 800, 600)
        p2 = write_image(tmp_path / "b.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        summary = export_folder(
            [p1, p2], social_dir, fmts("4:5", "1:1"), detector=det,
        )
        assert summary["done"] == 4
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        assert len(summary["results"]) == 4

    def test_skips_are_counted_on_second_run(self, tmp_path):
        p1 = write_image(tmp_path / "a.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        export_folder([p1], social_dir, fmts("1:1"), detector=det)
        summary = export_folder([p1], social_dir, fmts("1:1"), detector=det)
        assert summary["skipped"] == 1
        assert summary["done"] == 0

    def test_force_redoes_all(self, tmp_path):
        p1 = write_image(tmp_path / "a.jpg", 800, 600)
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        export_folder([p1], social_dir, fmts("1:1"), detector=det)
        summary = export_folder([p1], social_dir, fmts("1:1"), detector=det, force=True)
        assert summary["done"] == 1
        assert summary["skipped"] == 0

    def test_unreadable_file_counts_as_failed(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not actually an image")
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        summary = export_folder([bad], social_dir, fmts("1:1"), detector=det)
        assert summary["failed"] >= 1
        failed_results = [r for r in summary["results"] if r.status == "failed"]
        assert failed_results
        assert failed_results[0].error

    def test_empty_paths_gives_zero_counts(self, tmp_path):
        social_dir = tmp_path / "social"
        det = FakeDetector([])
        summary = export_folder([], social_dir, fmts("1:1"), detector=det)
        assert summary["done"] == 0
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        assert summary["results"] == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def _patch_detector(self, monkeypatch, boxes=()):
        import retouch.social_crops as sc

        def fake_make_detector():
            return FakeDetector(list(boxes))

        monkeypatch.setattr(sc, "_make_detector", fake_make_detector)

    def test_default_output_for_folder_is_social_subdir(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "a.jpg", 800, 600)
        rc = main([str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "social").exists()

    def test_default_output_for_file_is_parent_social(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        img_path = write_image(tmp_path / "a.jpg", 800, 600)
        rc = main([str(img_path)])
        assert rc == 0
        assert (tmp_path / "social").exists()

    def test_custom_output_dir(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "a.jpg", 800, 600)
        out_dir = tmp_path / "custom_out"
        rc = main([str(tmp_path), "-o", str(out_dir)])
        assert rc == 0
        assert out_dir.exists()
        assert not (tmp_path / "social").exists()

    def test_formats_flag_limits_output(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "a.jpg", 800, 600)
        rc = main([str(tmp_path), "--formats", "1:1"])
        assert rc == 0
        social_dir = tmp_path / "social"
        produced_slugs = {p.name for p in social_dir.iterdir() if p.is_dir()}
        assert produced_slugs == {"1x1"}

    def test_missing_input_returns_one(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        missing = tmp_path / "does_not_exist"
        rc = main([str(missing)])
        assert rc == 1

    def test_empty_folder_returns_one(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        rc = main([str(tmp_path)])
        assert rc == 1

    def test_force_flag_reexports(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "a.jpg", 800, 600)
        assert main([str(tmp_path)]) == 0
        # Without force, second run should still succeed (all skipped, not an error).
        assert main([str(tmp_path)]) == 0
        assert main([str(tmp_path), "--force"]) == 0

    def test_recursive_flag_finds_nested(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "sub" / "nested.jpg", 800, 600)
        rc = main([str(tmp_path), "-r"])
        assert rc == 0

    def test_size_flag_full(self, tmp_path, monkeypatch):
        self._patch_detector(monkeypatch)
        write_image(tmp_path / "a.jpg", 800, 600)
        rc = main([str(tmp_path), "--size", "full", "--formats", "1:1"])
        assert rc == 0


# ---------------------------------------------------------------------------
# find_subject_top / subject_top headroom (person mask above the face)
# ---------------------------------------------------------------------------

class TestSubjectTop:
    def test_finds_mask_top_above_face(self):
        from retouch.social_crops import find_subject_top
        mask = np.zeros((600, 400), np.float32)
        mask[80:, 150:250] = 1.0  # headpiece reaching up to row 80
        assert find_subject_top(mask, [(170, 200, 60, 70)]) == 80

    def test_ignores_mask_outside_face_band_and_far_above(self):
        from retouch.social_crops import find_subject_top
        mask = np.zeros((600, 400), np.float32)
        mask[0:50, 0:40] = 1.0     # a neighbour far to the side
        mask[5:15, 180:220] = 1.0  # more than 3 face heights above
        mask[150:, 180:220] = 1.0
        assert find_subject_top(mask, [(170, 300, 60, 50)]) == 150

    def test_none_without_mask_or_faces(self):
        from retouch.social_crops import find_subject_top
        assert find_subject_top(None, [(0, 0, 10, 10)]) is None
        assert find_subject_top(np.ones((50, 50), np.float32), []) is None

    def test_plan_keeps_headpiece_when_chin_still_fits(self):
        from retouch.social_crops import plan_crop
        fmt = FORMATS["1:1"]
        face = (1800, 1500, 400, 450)
        plain = plan_crop(4000, 6000, [face], fmt)
        tall = plan_crop(4000, 6000, [face], fmt, subject_top=200)
        assert tall.y < plain.y
        assert tall.y <= 200
        assert tall.y + tall.h >= face[1] + face[3]

    def test_plan_drops_headpiece_when_it_would_cut_the_chin(self):
        from retouch.social_crops import plan_crop
        fmt = FORMATS["1:1"]
        face = (900, 1400, 400, 450)
        # 1:1 of a 2000x3000 image is 2000 px tall; a headpiece at row 0 plus
        # the chin at 1850+ does not fit, so the face-only headroom wins.
        plan = plan_crop(2000, 3000, [face], fmt, subject_top=0)
        assert plan.y + plan.h >= face[1] + face[3]

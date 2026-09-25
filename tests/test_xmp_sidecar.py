"""Tests for retouch.xmp_sidecar: XMP ratings, labels and keywords."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch import xmp_sidecar as xs
from retouch.review_page import ReviewRecord, record_key, write_review_record

LIGHTROOM_SIDECAR = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    crs:Exposure2012="+0.35"
    crs:Temperature="5600">
   <xmp:Rating>4</xmp:Rating>
   <dc:subject>
    <rdf:Bag>
     <rdf:li>Anime Expo</rdf:li>
    </rdf:Bag>
   </dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def _jpeg(path: Path, value: int = 128) -> Path:
    img = np.full((32, 48, 3), value, np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# Packets and sidecars
# ---------------------------------------------------------------------------

class TestFields:
    def test_rating_range(self):
        with pytest.raises(ValueError):
            xs.XmpFields(rating=6)
        with pytest.raises(ValueError):
            xs.XmpFields(rating=-2)
        assert xs.XmpFields(rating=-1).rating == -1

    def test_keywords_deduplicated(self):
        assert xs.XmpFields(keywords=["a", " a ", "", "b"]).keywords == ["a", "b"]

    def test_keyword_slug(self):
        assert xs.keyword("qa", "Plastic Skin") == "retouch-qa-plastic-skin"
        assert xs.keyword("recipe", "cosplay_clear_v1") == "retouch-recipe-cosplay-clear-v1"


class TestSidecar:
    def test_new_sidecar_round_trip(self, tmp_path):
        src = tmp_path / "DSCF0001.RAF"
        src.write_bytes(b"raw")
        written = xs.write_sidecar(src, xs.XmpFields(rating=3, label="Green", keywords=["retouch-pick"]))
        assert written == tmp_path / "DSCF0001.xmp"
        fields = xs.read_fields(written.read_text())
        assert fields["rating"] == 3
        assert fields["label"] == "Green"
        assert fields["keywords"] == ["retouch-pick"]
        assert set(fields["managed"]) == {"Rating", "Label", "Keywords"}
        assert src.read_bytes() == b"raw"  # the source is never touched

    def test_empty_fields_write_nothing(self, tmp_path):
        src = tmp_path / "a.RAF"
        src.write_bytes(b"raw")
        assert xs.write_sidecar(src, xs.XmpFields()) is None
        assert not (tmp_path / "a.xmp").exists()

    def test_merge_keeps_other_editors_data_and_human_rating(self, tmp_path):
        src = tmp_path / "DSCF0002.RAF"
        src.write_bytes(b"raw")
        side = tmp_path / "DSCF0002.xmp"
        side.write_text(LIGHTROOM_SIDECAR)
        xs.write_sidecar(src, xs.XmpFields(rating=-1, label="Red", keywords=["retouch-reject"]))
        text = side.read_text()
        fields = xs.read_fields(text)
        assert fields["rating"] == 4, "a rating set in Lightroom must not be overwritten"
        assert fields["label"] == "Red", "an empty label may be filled"
        assert fields["keywords"] == ["Anime Expo", "retouch-reject"]
        assert "crs:Exposure2012=\"+0.35\"" in text
        assert "crs:Temperature=\"5600\"" in text
        # one-time backup of the foreign sidecar
        backup = tmp_path / "DSCF0002.xmp.orig"
        assert backup.read_text() == LIGHTROOM_SIDECAR

    def test_zero_rating_counts_as_unrated(self, tmp_path):
        src = tmp_path / "z.RAF"
        src.write_bytes(b"raw")
        (tmp_path / "z.xmp").write_text(LIGHTROOM_SIDECAR.replace("<xmp:Rating>4<", "<xmp:Rating>0<"))
        xs.write_sidecar(src, xs.XmpFields(rating=-1, label="Red"))
        assert xs.read_fields((tmp_path / "z.xmp").read_text())["rating"] == -1

    def test_backup_made_only_once(self, tmp_path):
        src = tmp_path / "b.RAF"
        src.write_bytes(b"raw")
        (tmp_path / "b.xmp").write_text(LIGHTROOM_SIDECAR)
        xs.write_sidecar(src, xs.XmpFields(label="Green", keywords=["retouch-pick"]))
        xs.write_sidecar(src, xs.XmpFields(label="Red", keywords=["retouch-reject"]))
        assert (tmp_path / "b.xmp.orig").read_text() == LIGHTROOM_SIDECAR

    def test_own_values_update_and_old_keywords_drop(self, tmp_path):
        src = tmp_path / "c.RAF"
        src.write_bytes(b"raw")
        xs.write_sidecar(src, xs.XmpFields(label="Green", keywords=["retouch-pick", "retouch-qa-banding"]))
        xs.write_sidecar(src, xs.XmpFields(rating=-1, label="Red", keywords=["retouch-reject"]))
        fields = xs.read_fields((tmp_path / "c.xmp").read_text())
        assert fields["rating"] == -1
        assert fields["label"] == "Red"
        assert fields["keywords"] == ["retouch-reject"]
        # clearing: retouch owns the label, so None removes it
        xs.write_sidecar(src, xs.XmpFields())
        fields = xs.read_fields((tmp_path / "c.xmp").read_text())
        assert fields["label"] is None and fields["rating"] is None and fields["keywords"] == []

    def test_label_changed_in_other_editor_is_kept(self, tmp_path):
        src = tmp_path / "d.RAF"
        src.write_bytes(b"raw")
        xs.write_sidecar(src, xs.XmpFields(label="Green"))
        side = tmp_path / "d.xmp"
        # Lightroom rewrites the file without our marker after the user relabels.
        # Another editor relabels but keeps our namespace (Lightroom and
        # exiftool preserve unknown XMP properties).
        text = side.read_text()
        assert 'retouch:Managed="Label=Green"' in text
        side.write_text(text.replace('xmp:Label="Green"', 'xmp:Label="Blue"'))
        xs.write_sidecar(src, xs.XmpFields(label="Red"))
        assert xs.read_fields(side.read_text())["label"] == "Blue"
        xs.write_sidecar(src, xs.XmpFields(label="Red"))
        assert xs.read_fields(side.read_text())["label"] == "Blue"

    def test_unchanged_returns_none(self, tmp_path):
        src = tmp_path / "e.RAF"
        src.write_bytes(b"raw")
        f = xs.XmpFields(label="Green", keywords=["retouch-pick"])
        assert xs.write_sidecar(src, f) is not None
        assert xs.write_sidecar(src, f) is None

    def test_corrupt_sidecar_left_untouched(self, tmp_path):
        src = tmp_path / "f.RAF"
        src.write_bytes(b"raw")
        side = tmp_path / "f.xmp"
        side.write_text("<x:xmpmeta not xml")
        with pytest.raises(xs.XmpError):
            xs.write_sidecar(src, xs.XmpFields(label="Red"))
        assert side.read_text() == "<x:xmpmeta not xml"

    def test_bare_rdf_packet_accepted(self):
        bare = ('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="2"/>'
                '</rdf:RDF>')
        text, changed = xs.build_packet(xs.XmpFields(label="Green"), bare)
        assert changed
        fields = xs.read_fields(text)
        assert fields["rating"] == 2 and fields["label"] == "Green"


class TestJpegEmbed:
    def test_embed_and_update_single_segment(self, tmp_path):
        out = _jpeg(tmp_path / "out.jpg")
        before = cv2.imread(str(out))
        assert xs.embed_in_jpeg(out, xs.XmpFields(label="Yellow", keywords=["retouch-qa-banding"]))
        assert xs.embed_in_jpeg(out, xs.XmpFields(label="Green", keywords=["retouch-pick"]))
        data = out.read_bytes()
        assert data.count(b"http://ns.adobe.com/xap/1.0/\x00") == 1
        fields = xs.read_jpeg_fields(out)
        assert fields["label"] == "Green"
        assert fields["keywords"] == ["retouch-pick"]
        after = cv2.imread(str(out))
        assert np.array_equal(before, after), "pixels must not change"

    def test_embed_keeps_exif_first(self, tmp_path):
        from PIL import Image

        out = tmp_path / "exif.jpg"
        img = Image.new("RGB", (16, 16), (90, 80, 70))
        exif = Image.Exif()
        exif[0x010F] = "FUJIFILM"
        img.save(out, exif=exif.tobytes())
        xs.embed_in_jpeg(out, xs.XmpFields(label="Red"))
        with Image.open(out) as reopened:
            assert reopened.getexif().get(0x010F) == "FUJIFILM"
            reopened.load()
        assert xs.read_jpeg_fields(out)["label"] == "Red"

    def test_no_embed_when_empty(self, tmp_path):
        out = _jpeg(tmp_path / "plain.jpg")
        data = out.read_bytes()
        assert not xs.embed_in_jpeg(out, xs.XmpFields())
        assert out.read_bytes() == data
        assert xs.read_jpeg_fields(out) is None

    def test_not_a_jpeg(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"PNG....")
        with pytest.raises(xs.XmpError):
            xs.embed_in_jpeg(bad, xs.XmpFields(label="Red"))

    def test_output_metadata_png_gets_sidecar(self, tmp_path):
        out = tmp_path / "out.png"
        assert cv2.imwrite(str(out), np.zeros((8, 8, 3), np.uint8))
        assert xs.write_output_metadata(out, xs.XmpFields(label="Red")) == tmp_path / "out.xmp"


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

class TestReviewMapping:
    def test_clean_record_has_no_label_or_stars(self):
        rec = ReviewRecord(source="/x/a.jpg", status="done", recipe="cosplay_clear_v1", qa=[
            {"detector": "banding", "flagged": False}])
        f = xs.fields_for_review(rec)
        assert f.rating is None and f.label is None
        assert f.keywords == ["retouch-recipe-cosplay-clear-v1"]

    def test_flagged_record_is_yellow(self):
        rec = ReviewRecord(source="/x/a.jpg", status="qa_fail", qa=[
            {"detector": "plastic_skin", "flagged": True}])
        f = xs.fields_for_review(rec)
        assert f.label == "Yellow"
        assert "retouch-qa-plastic-skin" in f.keywords and "retouch-qa-fail" in f.keywords

    def test_decisions_override_label(self):
        rec = ReviewRecord(source="/x/a.jpg", status="done", qa=[{"detector": "seam", "flagged": True}])
        pick = xs.fields_for_review(rec, "pick")
        assert pick.label == "Green" and pick.rating is None and "retouch-pick" in pick.keywords
        reject = xs.fields_for_review(rec, "reject")
        assert reject.label == "Red" and reject.rating == -1 and "retouch-reject" in reject.keywords
        assert "retouch-qa-seam" in reject.keywords


def _asset(**kw):
    base = dict(decision="hold", decision_origin="automatic", rating=None, labels=[],
                faces=[], burst_ids=[], culling_evidence={}, override_history=[])
    base.update(kw)
    return SimpleNamespace(**base)


class TestShootMapping:
    def test_unreviewed_asset_gets_no_label(self):
        # A re-scan flips decision_origin to "human" without any saved review.
        f = xs.fields_for_shoot_asset(_asset(decision_origin="human"))
        assert f.label is None and f.rating is None and f.keywords == []

    def test_human_select_with_stars(self):
        f = xs.fields_for_shoot_asset(_asset(decision="select", decision_origin="human", rating=4,
                                             labels=["Frieren"], override_history=[object()]))
        assert f.rating == 4 and f.label == "Green"
        assert f.keywords == ["retouch-pick", "Frieren"]

    def test_human_reject_without_rating_is_minus_one(self):
        f = xs.fields_for_shoot_asset(_asset(decision="reject", decision_origin="human",
                                             override_history=[object()]))
        assert f.rating == -1 and f.label == "Red"

    def test_automatic_evidence_is_keywords_only(self):
        f = xs.fields_for_shoot_asset(_asset(
            burst_ids=["b1"],
            culling_evidence={"candidate": {"rank": 1, "evidence": {"flags": ["eyes_closed"]}}},
        ))
        assert f.label is None and f.rating is None
        assert f.keywords == ["retouch-burst", "retouch-burst-best", "retouch-closed-eyes"]

    def test_face_evidence_closed_eyes(self):
        f = xs.fields_for_shoot_asset(_asset(faces=[SimpleNamespace(eyes_open="no")]))
        assert f.keywords == ["retouch-closed-eyes"]


# ---------------------------------------------------------------------------
# Bulk writers and CLIs
# ---------------------------------------------------------------------------

def _batch(tmp_path):
    src_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()
    raw = src_dir / "DSCF0001.RAF"
    raw.write_bytes(b"raw bytes")
    jpg_src = _jpeg(src_dir / "DSCF0002.JPG", 100)
    out1 = _jpeg(out_dir / "DSCF0001_retouched.jpg", 110)
    out2 = _jpeg(out_dir / "DSCF0002_retouched.jpg", 120)
    write_review_record(out_dir, ReviewRecord(source=str(raw), output=str(out1), status="done",
                                              recipe="cosplay_clear_v1"))
    write_review_record(out_dir, ReviewRecord(source=str(jpg_src), output=str(out2), status="qa_fail",
                                              qa=[{"detector": "banding", "flagged": True}]))
    return raw, jpg_src, out1, out2, out_dir


class TestBulk:
    def test_write_review_xmp_with_decisions(self, tmp_path):
        raw, jpg_src, out1, out2, root = _batch(tmp_path)
        jpg_bytes = jpg_src.read_bytes()
        counts = xs.write_review_xmp(root, {record_key(raw): "pick", record_key(jpg_src): "reject"})
        assert counts["errors"] == 0 and counts["missing"] == 0
        assert counts["sidecars"] == 2 and counts["embedded"] == 2
        assert xs.read_fields((raw.parent / "DSCF0001.xmp").read_text())["label"] == "Green"
        side2 = xs.read_fields((jpg_src.parent / "DSCF0002.xmp").read_text())
        assert side2["label"] == "Red" and side2["rating"] == -1
        assert "retouch-qa-banding" in side2["keywords"]
        assert jpg_src.read_bytes() == jpg_bytes, "JPEG sources are never modified"
        assert xs.read_jpeg_fields(out1)["label"] == "Green"
        assert xs.read_jpeg_fields(out2)["rating"] == -1
        again = xs.write_review_xmp(root, {record_key(raw): "pick", record_key(jpg_src): "reject"})
        assert again["unchanged"] == 4

    def test_review_apply_xmp_tags_copied_picks(self, tmp_path):
        from retouch import review_page

        raw, _jpg_src, _out1, _out2, root = _batch(tmp_path)
        decisions = tmp_path / "decisions.json"
        decisions.write_text(json.dumps({"decisions": {record_key(raw): "pick"}}))
        assert review_page.main(["apply", str(root), str(decisions), "--xmp"]) == 0
        copied = root / "picks" / "DSCF0001_retouched.jpg"
        assert xs.read_jpeg_fields(copied)["label"] == "Green"

    def test_cli_review_and_shoot(self, tmp_path, capsys):
        from retouch.shoot_review import ReviewAsset, ShootReviewManifest

        _raw, _jpg_src, _out1, _out2, root = _batch(tmp_path)
        assert xs.main(["review", str(root), "--no-outputs"]) == 0
        assert "2 sidecar(s) written" in capsys.readouterr().out

        shoot = tmp_path / "shoot"
        shoot.mkdir()
        (shoot / "IMG_1.CR3").write_bytes(b"raw")
        manifest = ShootReviewManifest(shoot)
        manifest.assets["a1"] = ReviewAsset(asset_instance_id="a1", relative_path="IMG_1.CR3",
                                            source_sha256="x", content_id="x", width=1, height=1)
        manifest.assets["a2"] = ReviewAsset(asset_instance_id="a2", relative_path="gone.CR3",
                                            source_sha256="y", content_id="y", width=1, height=1)
        manifest.set_human_review("a1", "select", rating=5)
        path = manifest.save()
        assert xs.main(["shoot", str(path)]) == 0
        out = capsys.readouterr().out
        assert "1 sidecar(s) written" in out and "1 file(s) missing" in out
        fields = xs.read_fields((shoot / "IMG_1.xmp").read_text())
        assert fields["rating"] == 5 and fields["label"] == "Green"

    def test_gui_shoot_handler(self, tmp_path):
        from gui_shoot import on_export_shoot_review_xmp
        from retouch.shoot_review import ReviewAsset, ShootReviewManifest

        (tmp_path / "IMG_2.NEF").write_bytes(b"raw")
        manifest = ShootReviewManifest(tmp_path)
        manifest.assets["a"] = ReviewAsset(asset_instance_id="a", relative_path="IMG_2.NEF",
                                           source_sha256="z", content_id="z", width=1, height=1)
        manifest.set_human_review("a", "reject")
        path = manifest.save()
        msg = on_export_shoot_review_xmp(str(path))
        assert "1 sidecar(s) written" in msg
        assert xs.read_fields((tmp_path / "IMG_2.xmp").read_text())["label"] == "Red"
        assert "XMP export failed" in on_export_shoot_review_xmp("")


def test_cli_xmp_requires_review(monkeypatch, capsys):
    import cli

    monkeypatch.setattr("sys.argv", ["cli.py", "x.jpg", "--xmp", "--no-review"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--xmp needs the review records" in capsys.readouterr().err


def test_gui_batch_xmp_helper(tmp_path):
    import gui_batch

    _raw, _jpg_src, out1, _out2, root = _batch(tmp_path)
    line = gui_batch._write_batch_xmp(root)
    assert line.startswith("XMP: 2 sidecar(s) written, 2 JPEG output(s) tagged")
    assert xs.read_jpeg_fields(out1)["keywords"] == ["retouch-recipe-cosplay-clear-v1"]
    assert gui_batch._write_batch_xmp(None).startswith("XMP: skipped")

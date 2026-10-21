"""Tests for retouch.cover_hook (cover score, slide suggestion, Claude look plumbing).

Synthetic images and fake detectors only: no MediaPipe, no network, no claude CLI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch import cover_hook
from retouch.cover_hook import (
    CoverScore,
    LookCache,
    LookError,
    apply_look,
    claude_look,
    crop_keep,
    finalize,
    framing_from_faces,
    list_photos,
    look_backend,
    main,
    measure_photo,
    score_shoot,
    suggest,
    update_manifest,
    write_scores,
)
from retouch.image_analyzer import colour_pop_score
from retouch.social_crops import FORMATS, plan_crop

GREY = (128, 128, 128)
RED = (30, 30, 220)      # BGR
GREEN = (40, 200, 40)
SKIN = (140, 165, 215)   # light skin, BGR


def solid(w, h, colour=GREY):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = colour
    return img


def textured(w, h, colour=GREY, seed=0):
    """A flat colour with gentle structure, so each seed is a different 'pose'."""
    rng = np.random.default_rng(seed)
    img = solid(w, h, colour).astype(np.int16)
    for _ in range(12):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        img[y:y + int(rng.integers(20, 120)), x:x + int(rng.integers(20, 120))] += int(rng.integers(-60, 60))
    return np.clip(img, 0, 255).astype(np.uint8)


def write(path: Path, img) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)
    return path


class FakeDetector:
    """Returns fixed face boxes (x, y, w, h) in the pixel space of the image it is given."""

    available = True

    def __init__(self, boxes=(), by_shape=None):
        self.boxes = list(boxes)
        self.by_shape = by_shape or {}
        self.closed = False

    def detect(self, img):
        boxes = self.by_shape.get(img.shape[:2], self.boxes)
        return [SimpleNamespace(bbox=b, landmarks=None) for b in boxes]

    def close(self):
        self.closed = True


def good_look(**over):
    look = {"kind": "photo", "framing": "full-body", "people": 1, "costume_readable": True,
            "setting": "outdoor", "note": "red wig"}
    look.update(over)
    return look


# ---------------------------------------------------------------------------
# colour pop (image_analyzer)
# ---------------------------------------------------------------------------

class TestColourPop:
    def test_grey_has_no_pop(self):
        assert colour_pop_score(solid(200, 300))["score"] == 0.0

    def test_one_vivid_colour_scores_full(self):
        out = colour_pop_score(solid(200, 300, RED))
        assert out["score"] == pytest.approx(1.0)
        assert out["hue_concentration"] == pytest.approx(1.0)

    def test_two_vivid_colours_score_lower_than_one(self):
        img = solid(200, 300, RED)
        img[:, 100:] = GREEN
        assert colour_pop_score(img)["score"] < colour_pop_score(solid(200, 300, RED))["score"]

    def test_skin_colour_is_not_vivid(self):
        assert colour_pop_score(solid(200, 300, SKIN))["vivid_share"] == 0.0

    def test_same_score_for_darker_exposure_of_skin(self):
        darker = (np.array(SKIN) * 0.45).astype(np.uint8)
        assert colour_pop_score(solid(200, 300, tuple(int(v) for v in darker)))["vivid_share"] == 0.0

    def test_mask_limits_the_measured_area(self):
        img = solid(200, 300)
        img[:, :50] = RED
        mask = np.zeros((300, 200), np.float32)
        mask[:, 100:] = 1.0
        assert colour_pop_score(img)["score"] > 0.5
        assert colour_pop_score(img, mask)["score"] == 0.0

    def test_empty_mask_measures_the_frame(self):
        assert colour_pop_score(solid(200, 300, RED), np.zeros((300, 200), np.float32))["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# offline measures
# ---------------------------------------------------------------------------

class TestFraming:
    @pytest.mark.parametrize("face_h,expected", [
        (30, "wide"), (100, "full-body"), (160, "three-quarter"), (280, "half-body"), (400, "close-up"),
    ])
    def test_single_face(self, face_h, expected):
        assert framing_from_faces([(0, 0, 80, face_h)], 1000)[0] == expected

    def test_two_faces_are_a_pair(self):
        assert framing_from_faces([(0, 0, 80, 300), (200, 0, 80, 300)], 1000)[0] == "pair"

    def test_no_face_is_unknown(self):
        assert framing_from_faces([], 1000) == ("unknown", None)


class TestCropKeep:
    def test_centred_head_is_kept(self):
        faces = [(450, 200, 100, 120)]
        plan = plan_crop(1000, 1000, faces, FORMATS["4:5"])
        head, person = crop_keep(plan, faces, None, 1000, 1000)
        assert head == pytest.approx(1.0) and person is None

    def test_faces_wider_than_crop_lose_head_area(self):
        faces = [(20, 300, 120, 140), (1460, 300, 120, 140)]
        plan = plan_crop(1600, 900, faces, FORMATS["4:5"])
        assert plan.reason == "faces_wider_than_crop"
        head, _ = crop_keep(plan, faces, None, 1600, 900)
        assert head < 0.2

    def test_person_mask_share(self):
        mask = np.zeros((900, 1600), np.float32)
        mask[100:800, 50:250] = 1.0          # a person at the left edge of a landscape frame
        plan = plan_crop(1600, 900, [], FORMATS["4:5"])
        head, person = crop_keep(plan, [], mask, 1600, 900)
        assert head is None and person == pytest.approx(0.0)


class TestMeasurePhoto:
    def test_centred_single_subject(self):
        result = measure_photo("x.jpg", FakeDetector([(450, 150, 100, 110)]), image=solid(1000, 1250, RED))
        assert result.error == ""
        assert result.checks["crop"] == pytest.approx(1.0)
        assert result.checks["colour"] == pytest.approx(1.0)
        assert result.checks["photo"] is None
        assert result.framing == "full-body" and result.framing_source == "proxy"
        assert result.eyes_open == "uncertain"

    def test_no_face_leaves_crop_and_costume_unmeasured(self):
        result = measure_photo("x.jpg", FakeDetector([]), image=solid(1000, 1250))
        assert result.checks["crop"] is None and result.checks["costume"] is None
        assert result.eyes_open == "no_face"
        assert any("crop not checked" in n for n in result.notes)

    def test_no_detector_still_scores_colour_and_light(self):
        result = finalize(measure_photo("x.jpg", None, image=solid(1000, 1250, RED)), None)
        assert result.checks["colour"] is not None and result.checks["light"] is not None
        assert result.checked_weight == cover_hook.WEIGHTS["colour"] + cover_hook.WEIGHTS["light"]
        assert result.score > 0

    def test_unreadable_file_is_recorded_not_raised(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        result = finalize(measure_photo(bad, FakeDetector([])), None)
        assert result.error and result.score == 0.0

    def test_split_pair_in_landscape_scores_low_on_crop(self):
        faces = [(20, 300, 120, 140), (1460, 300, 120, 140)]
        result = measure_photo("x.jpg", FakeDetector(faces), image=solid(1600, 900))
        assert result.crop_reason["4:5"] == "faces_wider_than_crop"
        assert result.checks["crop"] <= cover_hook.CROP_SPLIT_FACES_CAP


class TestEyeGateWiring:
    """The gate reads retouch.face_quality's own vocabulary ("yes" / "no" / "uncertain")."""

    def evidence(self, left, right):
        from retouch.face_quality import EYES_CLOSED_BELOW, classify_eyes_open
        from retouch.shoot_review import FaceQualityEvidence

        state, _ = classify_eyes_open(left, right)
        return FaceQualityEvidence(face_id="f", eyes_open=state, face_sharpness=0.3, measurement_details={
            "left_eye_aperture": left, "right_eye_aperture": right, "eyes_closed_below": EYES_CLOSED_BELOW})

    @pytest.mark.parametrize("left,right,expected,gate", [
        (0.40, 0.45, "open", 1.0),
        (0.05, 0.08, "closed", cover_hook.FACE_GATE_CLOSED),
        (0.05, 0.45, "wink", 1.0),
        (0.17, 0.18, "uncertain", 1.0),
    ])
    def test_states_and_gate(self, monkeypatch, left, right, expected, gate):
        import retouch.face_quality as face_quality

        monkeypatch.setattr(face_quality.FaceQualityAnalyzer, "analyze",
                            lambda self_, img, faces: [self.evidence(left, right) for _ in faces])
        result = measure_photo("x.jpg", FakeDetector([(450, 150, 100, 110)]), image=solid(1000, 1250))
        assert result.eyes_open == expected
        assert result.face_sharpness == 0.3
        assert finalize(result, None).face_gate == gate

    def test_missing_apertures_with_no_counts_as_closed(self):
        from retouch.shoot_review import FaceQualityEvidence

        assert cover_hook._eye_state(FaceQualityEvidence(face_id="f", eyes_open="no")) == "closed"


class TestFinalize:
    def base(self, **over):
        r = CoverScore(path="a.jpg", checks={"photo": None, "crop": 1.0, "costume": 1.0, "colour": 1.0, "light": 1.0},
                       eyes_open="open")
        for k, v in over.items():
            setattr(r, k, v)
        return r

    def test_unmeasured_check_is_left_out_not_failed(self):
        r = finalize(self.base(), None)
        assert r.score == 100.0
        assert r.checked_weight == 100 - cover_hook.WEIGHTS["photo"]

    def test_poster_costs_its_weight(self):
        r = self.base()
        r.checks["photo"] = 0.0
        assert finalize(r, None).score == pytest.approx(100.0 - cover_hook.WEIGHTS["photo"])

    def test_closed_eyes_demote(self):
        assert finalize(self.base(eyes_open="closed"), None).score == pytest.approx(100 * cover_hook.FACE_GATE_CLOSED)

    def test_soft_face_is_relative_to_the_shoot(self):
        assert finalize(self.base(face_sharpness=0.2), 0.5).face_gate == cover_hook.FACE_GATE_SOFT
        assert finalize(self.base(face_sharpness=0.4), 0.5).face_gate == 1.0
        assert finalize(self.base(face_sharpness=0.2), None).face_gate == 1.0

    def test_weights_sum_to_100(self):
        assert sum(cover_hook.WEIGHTS.values()) == 100
        assert set(cover_hook.WEIGHTS) | {"face"} == set(cover_hook.WEIGHT_REASONS)


# ---------------------------------------------------------------------------
# the Claude look
# ---------------------------------------------------------------------------

class TestLook:
    def test_apply_look_fills_photo_and_costume(self):
        r = CoverScore(path="a.jpg", checks={"light": 0.6}, measured_light=0.6)
        apply_look(r, good_look(framing="half-body", setting="outdoor"))
        assert r.checks["photo"] == 1.0
        assert r.checks["costume"] == cover_hook.FRAMING_SCORES["half-body"]
        assert r.checks["light"] == pytest.approx(0.8)
        assert r.framing_source == "vision"

    def test_two_people_are_a_pair_and_hidden_costume_costs(self):
        r = CoverScore(path="a.jpg")
        apply_look(r, good_look(framing="half-body", people=2, costume_readable=False))
        assert r.framing == "pair"
        assert r.checks["costume"] == pytest.approx(cover_hook.COSTUME_HIDDEN_FACTOR)

    def test_poster_scores_zero_on_photo(self):
        r = CoverScore(path="a.jpg")
        apply_look(r, good_look(kind="poster"))
        assert r.checks["photo"] == 0.0

    def test_cached_answer_is_not_asked_again(self, tmp_path):
        photo = write(tmp_path / "a.jpg", solid(300, 400, RED))
        calls = []

        def ask(small):
            calls.append(small)
            assert small.exists() and small.parent == tmp_path / "cache"
            return good_look()

        assert claude_look(photo, LookCache(tmp_path / "cache"), ask=ask)["kind"] == "photo"
        assert claude_look(photo, LookCache(tmp_path / "cache"), ask=ask)["kind"] == "photo"
        assert len(calls) == 1
        claude_look(photo, LookCache(tmp_path / "cache"), ask=ask, refresh=True)
        assert len(calls) == 2

    def test_bad_answer_raises_and_is_not_cached(self, tmp_path):
        photo = write(tmp_path / "a.jpg", solid(300, 400))
        cache = LookCache(tmp_path / "cache")
        with pytest.raises(LookError):
            claude_look(photo, cache, ask=lambda small: {"kind": "painting"})
        assert cache.looks == {}

    def test_look_copy_is_downscaled(self, tmp_path):
        photo = write(tmp_path / "a.jpg", solid(3000, 2000))
        seen = {}

        def ask(small):
            seen["shape"] = cv2.imread(str(small)).shape
            return good_look()

        claude_look(photo, LookCache(tmp_path / "cache"), ask=ask)
        assert max(seen["shape"][:2]) == cover_hook.LOOK_MAX_SIDE

    def test_backend_is_cli_without_key_or_model(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("RETOUCH_COVER_LOOK_MODEL", raising=False)
        assert look_backend() == "claude-code"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert look_backend() == "claude-code"   # a key alone is not enough: no model named

    def test_cli_call_strips_credentials_and_reads_structured_output(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"], seen["env"] = cmd, kwargs["env"]
            return SimpleNamespace(stdout=json.dumps({"structured_output": good_look()}), stderr="", returncode=0)

        monkeypatch.setattr(cover_hook.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(cover_hook.subprocess, "run", fake_run)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        monkeypatch.setenv("CLAUDECODE", "1")
        look = cover_hook._look_via_cli(tmp_path / "x.jpg")
        assert look["kind"] == "photo"
        assert "ANTHROPIC_API_KEY" not in seen["env"] and "CLAUDECODE" not in seen["env"]
        assert seen["cmd"][:2] == ["/usr/bin/claude", "-p"]
        assert "--no-session-persistence" in seen["cmd"] and "--json-schema" in seen["cmd"]

    def test_cli_falls_back_to_json_in_result_text(self, monkeypatch, tmp_path):
        text = "Here you go: " + json.dumps(good_look(kind="collage"))
        monkeypatch.setattr(cover_hook.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(cover_hook.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
            stdout=json.dumps({"result": text}), stderr="", returncode=0))
        assert cover_hook._look_via_cli(tmp_path / "x.jpg")["kind"] == "collage"

    @pytest.mark.parametrize("stdout", ["not json", json.dumps({"is_error": True, "result": "limit"})])
    def test_cli_errors_raise_look_error(self, monkeypatch, tmp_path, stdout):
        monkeypatch.setattr(cover_hook.shutil, "which", lambda name: "/usr/bin/claude")
        monkeypatch.setattr(cover_hook.subprocess, "run", lambda cmd, **kw: SimpleNamespace(
            stdout=stdout, stderr="boom", returncode=1))
        with pytest.raises(LookError):
            cover_hook._look_via_cli(tmp_path / "x.jpg")

    def test_cli_missing_or_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cover_hook.shutil, "which", lambda name: None)
        with pytest.raises(LookError, match="not installed"):
            cover_hook._look_via_cli(tmp_path / "x.jpg")
        monkeypatch.setattr(cover_hook.shutil, "which", lambda name: "/usr/bin/claude")

        def slow(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 1)

        monkeypatch.setattr(cover_hook.subprocess, "run", slow)
        with pytest.raises(LookError, match="took more than"):
            cover_hook._look_via_cli(tmp_path / "x.jpg")


# ---------------------------------------------------------------------------
# scoring a shoot, suggestions
# ---------------------------------------------------------------------------

@pytest.fixture
def shoot(tmp_path):
    """Three portrait frames: vivid, grey, and a second grey one."""
    folder = tmp_path / "shoot"
    write(folder / "a_red.jpg", textured(800, 1000, RED, seed=1))
    write(folder / "b_grey.jpg", textured(800, 1000, GREY, seed=2))
    write(folder / "c_grey.jpg", textured(800, 1000, GREY, seed=3))
    return folder


class TestScoreShoot:
    def test_offline_ranking_prefers_the_vivid_frame_and_skips_vision_checks(self, shoot):
        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]))
        assert Path(ranked[0].path).name == "a_red.jpg"
        assert all(r.checks["photo"] is None and r.look is None for r in ranked)
        assert all(r.kind == "not checked" for r in ranked)

    def test_vision_looks_only_at_the_top_candidates(self, shoot):
        asked = []

        def ask(small):
            asked.append(small)
            return good_look()

        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]), vision=True,
                             vision_top=1, cache_dir=shoot / "cover-hook" / ".cache", ask=ask)
        assert len(asked) == 1
        assert [r.look is not None for r in ranked].count(True) == 1
        assert Path(next(r for r in ranked if r.look).path).name == "a_red.jpg"

    def test_a_poster_drops_below_a_photo(self, shoot):
        def ask(small):
            red = cv2.imread(str(small))[..., 2].mean() > 150
            return good_look(kind="poster" if red else "photo")

        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]), vision=True,
                             vision_top=3, cache_dir=shoot / "cover-hook" / ".cache", ask=ask)
        assert Path(ranked[-1].path).name == "a_red.jpg"
        assert suggest(ranked, same_shot=lambda a, b: False).cover != ranked[-1].path

    def test_failed_look_keeps_the_offline_score(self, shoot):
        def ask(small):
            raise LookError("offline")

        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]), vision=True,
                             cache_dir=shoot / "cover-hook" / ".cache", ask=ask)
        assert all(r.look is None and r.score > 0 for r in ranked)
        assert any("Claude look failed" in n for n in ranked[0].notes)

    def test_vision_needs_a_cache_dir(self, shoot):
        with pytest.raises(ValueError):
            score_shoot(list_photos(shoot), vision=True)


def scored(name, score, framing="full-body", **over):
    r = CoverScore(path=name, score=score, framing=framing)
    for k, v in over.items():
        setattr(r, k, v)
    return r


class TestSuggest:
    def test_tight_shot_follows_a_wide_cover(self):
        ranked = [scored("a", 90), scored("b", 80), scored("c", 70, "half-body"), scored("d", 60)]
        out = suggest(ranked, same_shot=lambda a, b: False)
        assert out.cover == "a" and out.slides == ["c", "b"]
        assert "tight shot" in out.reasons["c"]

    def test_wide_shot_follows_a_tight_cover(self):
        ranked = [scored("a", 90, "close-up"), scored("b", 80, "half-body"), scored("c", 70, "pair")]
        assert suggest(ranked, same_shot=lambda a, b: False).slides == ["c", "b"]

    def test_near_duplicates_of_chosen_photos_are_skipped(self):
        ranked = [scored("a", 90), scored("a2", 85), scored("b", 70), scored("b2", 65), scored("c", 50)]
        out = suggest(ranked, same_shot=lambda x, y: x[0] == y[0])
        assert out.cover == "a" and out.slides == ["b", "c"]

    def test_posters_errors_and_zero_scores_are_never_suggested(self):
        ranked = [scored("p", 95, kind="poster"), scored("e", 90, error="unreadable"), scored("z", 0.0), scored("a", 40)]
        out = suggest(ranked, same_shot=lambda a, b: False)
        assert out.cover == "a" and out.slides == []

    def test_looked_at_photos_are_preferred_when_any_exist(self):
        ranked = [scored("offline", 95), scored("looked", 80, look=good_look())]
        assert suggest(ranked, same_shot=lambda a, b: False).cover == "looked"

    def test_empty(self):
        out = suggest([])
        assert out.cover is None and out.slides == []

    def test_real_duplicate_check_drops_a_copy(self, tmp_path):
        a = write(tmp_path / "a.jpg", textured(600, 800, RED, seed=5))
        a2 = write(tmp_path / "a2.jpg", textured(600, 800, RED, seed=5))
        b = write(tmp_path / "b.jpg", textured(600, 800, GREEN, seed=9))
        out = suggest([scored(str(a), 90), scored(str(a2), 85), scored(str(b), 60)])
        assert out.cover == str(a) and out.slides == [str(b)]


# ---------------------------------------------------------------------------
# output, manifest, CLI
# ---------------------------------------------------------------------------

class TestOutput:
    def test_list_photos_skips_own_and_derived_folders(self, shoot):
        write(shoot / "cover-hook" / "4x5" / "a_red_4x5.jpg", solid(100, 125))
        write(shoot / "social" / "4x5" / "x.jpg", solid(100, 125))
        write(shoot / "watermarked" / "y.jpg", solid(100, 125))
        assert [p.name for p in list_photos(shoot, recursive=True)] == ["a_red.jpg", "b_grey.jpg", "c_grey.jpg"]

    def test_write_scores_json_and_csv(self, shoot):
        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]))
        target = write_scores(ranked, suggest(ranked, same_shot=lambda a, b: False), shoot / "cover-hook", shoot,
                              vision=False)
        data = json.loads(target.read_text())
        assert data["schema"] == "retouch.cover_hook" and data["vision"] is False
        assert [p["rank"] for p in data["photos"]] == [1, 2, 3]
        assert data["photos"][0]["path"] == "a_red.jpg" and data["suggestion"]["cover"] == "a_red.jpg"
        rows = (shoot / "cover-hook" / cover_hook.SCORES_CSV).read_text().splitlines()
        assert rows[0].startswith("rank,file,score,photo,crop") and len(rows) == 4


class TestManifest:
    def make(self, shoot, photo: Path, sha=None):
        from retouch.shoot_review import ReviewAsset, ShootReviewManifest

        manifest = ShootReviewManifest(shoot)
        sha = sha or cover_hook._sha256(photo)
        asset = ReviewAsset(asset_instance_id="asset-1", relative_path=photo.name, source_sha256=sha,
                            content_id=sha, width=800, height=1000, decision="select", decision_origin="human")
        manifest.assets[asset.asset_id] = asset
        return manifest.save(shoot / "manifest.json")

    def test_only_matching_assets_get_scores_and_decisions_stay(self, shoot):
        from retouch.shoot_review import ShootReviewManifest

        path = self.make(shoot, shoot / "a_red.jpg")
        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]))
        assert update_manifest(path, ranked) == 1
        asset = ShootReviewManifest.load(path).assets["asset-1"]
        assert asset.culling_evidence["cover_hook"]["rank"] == 1
        assert asset.culling_evidence["cover_hook"]["score"] == ranked[0].score
        assert asset.decision == "select" and asset.decision_origin == "human"

    def test_changed_content_is_left_alone(self, shoot):
        from retouch.shoot_review import ShootReviewManifest

        path = self.make(shoot, shoot / "a_red.jpg", sha="0" * 64)
        before = path.read_text()
        ranked = score_shoot(list_photos(shoot), detector=FakeDetector([(350, 150, 100, 100)]))
        assert update_manifest(path, ranked) == 0
        assert path.read_text() == before
        assert "cover_hook" not in ShootReviewManifest.load(path).assets["asset-1"].culling_evidence

    def test_missing_manifest_is_not_created(self, shoot):
        target = shoot / "none.json"
        assert update_manifest(target, []) == 0
        assert not target.exists()


class TestMain:
    @pytest.fixture(autouse=True)
    def fake_detector(self, monkeypatch):
        import retouch.detection as detection

        self.detector = FakeDetector(by_shape={(1000, 800): [(350, 150, 100, 100)]})
        monkeypatch.setattr(detection, "FaceDetector", lambda **kw: self.detector)

    def test_offline_run_writes_scores_and_crop_and_is_repeatable(self, shoot, capsys):
        assert main([str(shoot)]) == 0
        out = capsys.readouterr().out
        assert "Scoring 3 photos" in out and "add --vision" in out and "cover:" in out
        crop = shoot / "cover-hook" / "4x5" / "a_red_4x5.jpg"
        assert crop.exists()
        h, w = cv2.imread(str(crop)).shape[:2]
        assert w / h == pytest.approx(0.8, abs=0.01)
        tall = shoot / "cover-hook" / "3x4" / "a_red_3x4.jpg"
        h, w = cv2.imread(str(tall)).shape[:2]
        assert w / h == pytest.approx(0.75, abs=0.01)
        assert self.detector.closed
        assert main([str(shoot), "-r"]) == 0      # the second run must not score its own crop
        data = json.loads((shoot / "cover-hook" / cover_hook.SCORES_JSON).read_text())
        assert len(data["photos"]) == 3

    def test_no_crop_and_custom_output(self, shoot, tmp_path):
        out = tmp_path / "elsewhere"
        assert main([str(shoot), "-o", str(out), "--no-crop"]) == 0
        assert (out / cover_hook.SCORES_JSON).exists() and not (out / "4x5").exists()
        assert not (shoot / "cover-hook").exists()

    def test_single_files_and_manifest_flag(self, shoot, capsys):
        assert main([str(shoot / "a_red.jpg"), str(shoot / "b_grey.jpg"), "--no-crop",
                     "--manifest", str(shoot / "missing.json")]) == 0
        assert "Manifest: 0 assets updated" in capsys.readouterr().out

    def test_missing_input_and_empty_folder(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope")]) == 1
        (tmp_path / "empty").mkdir()
        assert main([str(tmp_path / "empty")]) == 1
        assert "no photos found" in capsys.readouterr().err

    def test_vision_flag_uses_the_cache(self, shoot, monkeypatch, capsys):
        calls = []

        def fake_cli(small):
            calls.append(small)
            return good_look()

        monkeypatch.setattr(cover_hook, "_look_via_cli", fake_cli)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert main([str(shoot), "--vision", "--vision-top", "2", "--no-crop"]) == 0
        assert len(calls) == 2
        assert main([str(shoot), "--vision", "--vision-top", "2", "--no-crop"]) == 0
        assert len(calls) == 2
        assert "add --vision" not in capsys.readouterr().out

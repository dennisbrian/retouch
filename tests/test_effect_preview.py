"""Stage evidence is bounded, opt-in, and never reused across source/revisions."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.effect_preview import capture, initialize, save_previews, skipped
from retouch.engine import ProcessingContext, ProcessingResult, RetouchEngine


def test_opt_in_and_disabled_effect_status():
    ctx = ProcessingContext()
    initialize(ctx, False)
    img = np.zeros((20, 30, 3), np.float32)
    capture(ctx, "body_skin_even", img, img.copy(), "Skipped")
    assert ctx._effect_previews == {}
    initialize(ctx, True)
    skipped(ctx, "body_skin_even", "No person")
    assert ctx._effect_previews["body_skin_even"]["status"].startswith("Off")


def test_changed_pixels_and_stage_images_are_bounded_and_do_not_mutate_render(tmp_path):
    ctx = ProcessingContext(body_skin_even=50)
    initialize(ctx, True)
    before = np.full((900, 1200, 3), 0.5, np.float32)
    after = before.copy()
    after[300:600, 400:800] += 0.1
    original = after.copy()
    capture(ctx, "body_skin_even", before, after, "Skipped")
    item = ctx._effect_previews["body_skin_even"]
    assert item["after"].shape == (600, 800, 3)
    assert "11.11%" in item["status"]
    np.testing.assert_array_equal(item["overlay"][0], item["after"][0])
    assert np.any(item["overlay"][300, 350] != item["after"][300, 350])
    np.testing.assert_array_equal(after, original)
    saved = save_previews(ctx._effect_previews, tmp_path)
    assert cv2.imread(saved["body_skin_even"]["overlay"]).shape == (600, 800, 3)
    assert all(not isinstance(v, np.ndarray) for record in saved.values() for v in record.values())
    result = ProcessingResult(after, params=ctx)
    assert result.effect_previews["body_skin_even"]["status"] == item["status"]


def test_no_op_has_status_and_no_images():
    ctx = ProcessingContext(neck_tone_match=50)
    initialize(ctx, True)
    image = np.full((32, 32, 3), 0.5, np.float32)
    capture(ctx, "neck_tone_match", image, image, "No suitable neck skin")
    assert ctx._effect_previews["neck_tone_match"] == {"status": "No suitable neck skin"}
    capture(ctx, "neck_tone_match", image, image.copy(), "Skipped")
    assert "No visible correction" in ctx._effect_previews["neck_tone_match"]["status"]


def test_body_stage_captures_only_evening_not_other_body_ops():
    from tests.test_body_skin_even import H, W, _bgr, _blob, _plant_red, _skin
    before = _bgr(_plant_red(_skin(), _blob(), scale=0.5))
    ctx = ProcessingContext(body_skin_even=100, body_whiten=25)
    initialize(ctx, True)
    def segment(image):
        p = np.zeros(image.shape[:2] + (6,), np.float32)
        p[..., 2] = 1
        return p
    engine = RetouchEngine.__new__(RetouchEngine)
    engine._parser = SimpleNamespace(parse_hair_full_image=lambda image: None, _segment_classes=segment)
    result = engine._stage_body_skin(before, ctx, np.ones((H, W), np.float32), None, None, None, [], H, W)
    item = ctx._effect_previews["body_skin_even"]
    assert "Changed" in item["status"]
    from retouch.body_skin_even import even_body_skin
    isolated = even_body_skin(before, np.ones((H, W), np.float32), 1.0)
    np.testing.assert_array_equal(item["after"], np.clip(isolated * 255 + 0.5, 0, 255).astype(np.uint8))
    assert not np.array_equal(item["after"], np.clip(result * 255 + 0.5, 0, 255).astype(np.uint8))


def test_neck_stage_records_real_plan_output():
    from tests.test_neck_tone_match import _scene, _landmarks, _face_mask
    img, person = _scene()
    skin = _face_mask()
    face = _landmarks()
    edited = img.copy()
    edited[skin > 0.5] = np.clip(edited[skin > 0.5] * 1.15, 0, 1)
    ctx = ProcessingContext(neck_tone_match=100)
    ctx._neck_ref = img
    initialize(ctx, True)
    engine = RetouchEngine.__new__(RetouchEngine)
    engine._parser = SimpleNamespace(parse_hair_full_image=lambda image: None)
    result = engine._stage_neck_tone_match(edited, ctx, person, skin, None, [face])
    assert "Changed" in ctx._effect_previews["neck_tone_match"]["status"]
    assert np.max(abs(result - edited)) > 0.01


def test_neck_stage_explains_painted_face_skip():
    from tests.test_neck_tone_match import _scene, _landmarks, _face_mask
    img, person = _scene(face_lab=(85, 0, 0))
    ctx = ProcessingContext(neck_tone_match=100)
    initialize(ctx, True)
    engine = RetouchEngine.__new__(RetouchEngine)
    engine._parser = SimpleNamespace(parse_hair_full_image=lambda image: None)
    result = engine._stage_neck_tone_match(img, ctx, person, _face_mask(), None, [_landmarks()])
    assert result is img
    item = ctx._effect_previews["neck_tone_match"]
    assert "Painted faces" in item["status"] and "overlay" not in item


@pytest.mark.parametrize("mismatch", ["revision", "source", "failed", "expired", "modified"])
def test_gui_rejects_stale_or_missing_artifacts(tmp_path, mismatch):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache, source_identity
    source = tmp_path / "photo.png"
    source.write_bytes(b"original")
    artifact = tmp_path / "overlay.png"
    cv2.imwrite(str(artifact), np.zeros((20, 30, 3), np.uint8))
    evidence = {
        "source_path": str(source), "render_revision": 3,
        "effect_source_identity": source_identity(source).to_dict(),
        "render_attempt": {"status": "completed"},
        "effect_previews": {"body_skin_even": {"overlay": str(artifact), "status": "Changed"}},
    }
    revision, paths = 3, [str(source)]
    if mismatch == "revision": revision = 4
    if mismatch == "source": paths = [str(tmp_path / "other.png")]
    if mismatch == "failed": evidence["render_attempt"]["status"] = "failed"
    if mismatch == "expired": artifact.unlink()
    if mismatch == "modified": source.write_bytes(b"modified")
    cache = GuiPreviewCache()
    cache.set_latest_render(evidence)
    image, status = gui.show_effect_preview("body_skin_even", "overlay", paths, revision, cache)
    assert image is None and "Changed" != status


def test_gui_returns_rgb_and_skip_status(tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache
    path = tmp_path / "after.png"
    cv2.imwrite(str(path), np.full((20, 30, 3), (20, 40, 80), np.uint8))
    cache = GuiPreviewCache()
    cache.set_latest_render({
        "source_path": str(path), "render_revision": 3,
        "render_attempt": {"status": "completed"},
        "effect_previews": {"body_skin_even": {"after": str(path), "status": "Changed"},
                            "neck_tone_match": {"status": "No suitable neck skin"}},
    })
    rgb, status = gui.show_effect_preview("body_skin_even", "after", [str(path)], 3, cache)
    assert tuple(rgb[0, 0]) == (80, 40, 20) and status == "Changed"
    assert gui.show_effect_preview("neck_tone_match", "overlay", [str(path)], 3, cache) == (None, "No suitable neck skin")

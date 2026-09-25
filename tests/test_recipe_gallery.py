"""Tests for retouch.recipe_gallery (recipe previews on the user's photo)."""

import numpy as np
import pytest

from retouch import recipe_gallery as rg
from retouch.recipes import CURATED_RECIPE_NAMES, RECOMMENDED_RECIPE_NAMES


class _FakeResult(np.ndarray):
    pass


class _FakeEngine:
    """Records calls; returns the input brightened, with fake face contexts."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def process(self, img, recipe=None, face_contexts=None):
        self.calls.append((recipe, face_contexts))
        if recipe in self.fail:
            raise RuntimeError("boom")
        out = np.clip(img.astype(np.int16) + 10, 0, 255).astype(np.uint8).view(_FakeResult)
        out.face_contexts = ["ctx"]
        return out


def _photo(h=1000, w=600):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestGroups:
    def test_recommended_is_default(self):
        assert rg.gallery_recipe_names() == list(RECOMMENDED_RECIPE_NAMES)

    def test_all_is_curated_list(self):
        assert rg.gallery_recipe_names(rg.GROUP_ALL) == list(CURATED_RECIPE_NAMES)

    @pytest.mark.parametrize("group", rg.gallery_groups())
    def test_every_group_is_curated_and_nonempty(self, group):
        names = rg.gallery_recipe_names(group)
        assert names
        assert set(names) <= set(CURATED_RECIPE_NAMES)

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            rg.gallery_recipe_names("nope")


class TestPrepareSource:
    def test_long_edge_capped(self):
        out = rg.prepare_gallery_source(_photo(1000, 600), max_dim=512)
        assert max(out.shape[:2]) == 512
        assert out.dtype == np.uint8

    def test_small_image_not_upscaled(self):
        out = rg.prepare_gallery_source(_photo(200, 100), max_dim=512)
        assert out.shape[:2] == (200, 100)

    def test_uint16_scaled_to_8bit(self):
        img = np.full((10, 10, 3), 65535, dtype=np.uint16)
        assert rg.prepare_gallery_source(img).max() == 255

    def test_rejects_grayscale(self):
        with pytest.raises(ValueError):
            rg.prepare_gallery_source(np.zeros((10, 10), dtype=np.uint8))


class TestRenderer:
    def test_renders_each_recipe_and_reuses_face_contexts(self):
        engine = _FakeEngine()
        original, tiles = rg.RecipeGalleryRenderer(max_dim=128).render(
            _photo(), ["a", "b", "c"], engine)
        assert [t.recipe for t in tiles] == ["a", "b", "c"]
        assert engine.calls[0][1] is None
        assert all(ctx == ["ctx"] for _, ctx in engine.calls[1:])
        assert original.shape == tiles[0].image_rgb.shape
        assert max(original.shape[:2]) == 128

    def test_cache_skips_engine_on_second_run(self):
        engine = _FakeEngine()
        renderer = rg.RecipeGalleryRenderer(max_dim=64)
        renderer.render(_photo(), ["a", "b"], engine)
        renderer.render(_photo(), ["a", "b"], engine)
        assert len(engine.calls) == 2

    def test_cache_is_bounded(self):
        renderer = rg.RecipeGalleryRenderer(max_dim=32, cache_size=2)
        renderer.render(_photo(), ["a", "b", "c"], _FakeEngine())
        key = renderer.source_key(rg.prepare_gallery_source(_photo(), 32))
        assert renderer.cached(key, "a") is None
        assert renderer.cached(key, "c") is not None

    def test_failed_recipe_gets_placeholder_and_others_render(self):
        original, tiles = rg.RecipeGalleryRenderer(max_dim=64).render(
            _photo(), ["a", "bad", "c"], _FakeEngine(fail={"bad"}))
        assert [t.ok for t in tiles] == [True, False, True]
        assert tiles[1].image_rgb.shape == original.shape

    def test_progress_reports_every_recipe_then_done(self):
        seen = []
        rg.RecipeGalleryRenderer(max_dim=32).render(
            _photo(), ["a", "b"], _FakeEngine(), progress=lambda d, t, n: seen.append((d, t, n)))
        assert seen == [(0, 2, "a"), (1, 2, "b"), (2, 2, "")]


class TestComposition:
    def test_before_after_width(self):
        a = np.zeros((50, 40, 3), np.uint8)
        out = rg.before_after(a, a, separator=4)
        assert out.shape == (50, 84, 3)

    def test_contact_sheet_grid(self):
        tiles = [(np.zeros((100, 60, 3), np.uint8), f"r{i}") for i in range(7)]
        sheet = rg.contact_sheet(tiles, columns=3, tile_height=100, gap=6)
        assert sheet.shape[0] == 3 * (100 + 26 + 6) + 6
        assert sheet.shape[1] == 3 * (60 + 6) + 6


def test_cli_writes_sheet(tmp_path, monkeypatch):
    import cv2
    import retouch.engine as engine_mod

    photo = tmp_path / "p.jpg"
    cv2.imwrite(str(photo), _photo(200, 120))
    monkeypatch.setattr(engine_mod, "RetouchEngine", _FakeEngine)
    out = tmp_path / "sheet.jpg"
    assert rg.main([str(photo), "-o", str(out), "--recipes", "natural,portrait"]) == 0
    assert cv2.imread(str(out)) is not None

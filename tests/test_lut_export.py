"""Tests for retouch.lut_export (recipe -> .cube 3D LUT)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from retouch.lut import CubeLUT, load_cube
from retouch.lut_export import (
    _lut_context,
    _render_colour_stages,
    changes_colour,
    export_all_recipe_luts,
    export_recipe_lut,
    recipe_lut,
    skipped_steps,
    write_cube,
)

REPO = Path(__file__).resolve().parent.parent

# Strong looks whose recipes also set clarity, bloom, glow, vignette and grain,
# so a spatial op leaking into the lattice render would show up here.
LOOK_RECIPES = ["cosplay_feed_pop_v1", "game_character_v1", "apex_cinema_v1"]


def _photo_like(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([x / w, y / h, (x + y) / (w + h)], axis=-1)
    noise = rng.uniform(-0.15, 0.15, size=(h, w, 3)).astype(np.float32)
    return np.clip((base + noise) * 255.0, 0, 255).astype(np.uint8)


class TestCubeFile:
    def test_round_trip_is_lossless(self, tmp_path):
        rng = np.random.default_rng(1)
        arr = rng.uniform(0, 1, size=(5, 5, 5, 3)).astype(np.float32)
        path = write_cube(CubeLUT(arr), tmp_path / "x.cube", title="x")
        back = load_cube(path).array
        np.testing.assert_allclose(back, arr, atol=1e-6)

    def test_red_varies_fastest(self, tmp_path):
        path = write_cube(CubeLUT(3), tmp_path / "id.cube")
        lines = path.read_text().splitlines()
        assert "LUT_3D_SIZE 3" in lines
        data = [l for l in lines if l and l[0].isdigit()]
        assert len(data) == 27
        assert data[0] == "0.000000 0.000000 0.000000"
        assert data[1] == "0.500000 0.000000 0.000000"  # R steps first
        assert data[3] == "0.000000 0.500000 0.000000"  # then G
        assert data[9] == "0.000000 0.000000 0.500000"  # then B

    def test_title_quotes_are_neutralised(self, tmp_path):
        path = write_cube(CubeLUT(2), tmp_path / "t.cube", title='a"b')
        assert path.read_text().splitlines()[0] == "TITLE \"a'b\""


class TestRecipeLut:
    def test_retouch_only_recipe_is_identity(self):
        assert not changes_colour(recipe_lut("natural", size=9))

    @pytest.mark.parametrize("recipe", LOOK_RECIPES)
    def test_look_recipe_changes_colour(self, recipe):
        assert changes_colour(recipe_lut(recipe, size=9))

    @pytest.mark.parametrize("recipe", LOOK_RECIPES)
    def test_lut_matches_engine_colour_stages(self, recipe):
        img = _photo_like()
        direct = _render_colour_stages(img.astype(np.float32) / 255.0, _lut_context(recipe))
        direct = np.clip(direct * 255.0 + 0.5, 0, 255).astype(np.int32)
        via_lut = recipe_lut(recipe).apply(img).astype(np.int32)
        diff = np.abs(direct - via_lut)
        assert diff.mean() < 1.5
        assert np.percentile(diff, 99) <= 8

    @pytest.mark.parametrize("recipe", LOOK_RECIPES)
    def test_colour_stages_are_per_pixel(self, recipe):
        """Shuffling pixels must not change any pixel's output.

        Guards the lattice render: a clarity/glow/vignette/grain step that
        slipped past the neutralised grader would make a pixel depend on
        its neighbours (or position) and break the LUT.
        """
        img = _photo_like(48, 64).astype(np.float32) / 255.0
        ctx = _lut_context(recipe)
        flat = img.reshape(-1, 1, 3)
        perm = np.random.default_rng(2).permutation(flat.shape[0])
        straight = _render_colour_stages(flat.reshape(48, 64, 3), ctx).reshape(-1, 3)
        shuffled = _render_colour_stages(flat[perm].reshape(48, 64, 3), ctx).reshape(-1, 3)
        np.testing.assert_allclose(shuffled, straight[perm], atol=1.01 / 255.0)

    def test_rejects_bad_size(self):
        with pytest.raises(ValueError):
            recipe_lut("natural", size=1)


class TestSkippedSteps:
    def test_reports_spatial_steps(self):
        steps = skipped_steps("game_character_v1")
        for label in ("glow", "vignette", "film grain", "sharpening"):
            assert label in steps

    def test_natural_skips_nothing_spatial(self):
        assert "vignette" not in skipped_steps("natural")


class TestExport:
    def test_export_to_folder_names_file_after_recipe(self, tmp_path):
        path, skipped = export_recipe_lut("cosplay_feed_pop_v1", tmp_path, size=9)
        assert path == tmp_path / "cosplay_feed_pop_v1.cube"
        assert load_cube(path).size == 9
        assert "sharpening" in skipped

    def test_export_to_explicit_file(self, tmp_path):
        target = tmp_path / "sub" / "look.cube"
        path, _ = export_recipe_lut("game_character_v1", target, size=5)
        assert path == target and target.exists()

    def test_unknown_recipe_raises(self, tmp_path):
        with pytest.raises(KeyError):
            export_recipe_lut("no_such_recipe", tmp_path)
        assert not list(tmp_path.iterdir())

    def test_export_all_skips_identity_recipes(self, tmp_path):
        paths = export_all_recipe_luts(
            tmp_path, size=5, recipes=["natural", "cosplay_feed_pop_v1"]
        )
        assert [p.name for p in paths] == ["cosplay_feed_pop_v1.cube"]


class TestCommandLine:
    def test_module_cli_writes_cube(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-m", "retouch.lut_export", "cosplay_feed_pop_v1",
             "-o", str(tmp_path), "--size", "5"],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 0, out.stderr
        assert (tmp_path / "cosplay_feed_pop_v1.cube").exists()
        assert "Not in the LUT" in out.stdout

    def test_module_cli_unknown_recipe_exits_2(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-m", "retouch.lut_export", "nope", "-o", str(tmp_path)],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 2
        assert "unknown recipe" in out.stderr

    def test_batch_cli_export_lut_flag(self, tmp_path):
        target = tmp_path / "neon.cube"
        out = subprocess.run(
            [sys.executable, "cli.py", "--recipe", "cosplay_neon_night_v1",
             "--export-lut", str(target), "--lut-size", "5"],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 0, out.stderr
        assert load_cube(target).size == 5

    def test_batch_cli_export_lut_needs_recipe(self):
        out = subprocess.run(
            [sys.executable, "cli.py", "--export-lut"],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 2
        assert "--export-lut needs --recipe" in out.stderr


class TestGuiHandler:
    def test_saves_cube_for_look_recipe(self):
        gui = pytest.importorskip("gui")
        path, status = gui.on_export_recipe_lut("cosplay_feed_pop_v1")
        assert path and Path(path).name == "cosplay_feed_pop_v1.cube"
        assert load_cube(path).size == 33
        assert "Not in the LUT" in status

    def test_retouch_only_recipe_gives_no_file(self):
        gui = pytest.importorskip("gui")
        path, status = gui.on_export_recipe_lut("natural")
        assert path is None and "no colour look" in status

    def test_no_recipe_selected(self):
        gui = pytest.importorskip("gui")
        assert gui.on_export_recipe_lut(None)[0] is None

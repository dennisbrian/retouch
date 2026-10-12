# Test Report — 2026-07-04

## Summary
- **Passed:** 2234
- **Failed:** 5
- **Skipped:** 5
- **Duration:** 11m 16s
- **Warnings:** 38 (library deprecations only)

## Failures

### 1. `test_gui.py::TestIntegrationConstantCrossRef::test_ext_map_keys_match_radio_choices`
- `EXT_MAP` has extra key `'PNG-16'` not expected by test

### 2. `test_integration_pipeline.py::TestPipelineOrder::test_stages_execute_in_documented_order`
- Stage order changed: `_stage_subject_separation` now runs before `_stage_global`

### 3. `test_recipe_integration.py::TestRecipeOverride::test_override_top_level_scalar`
- `anime_cinematic_v1` contrast = 7.0, test expects 12.0

### 4. `test_recipe_integration.py::TestBuildContextFromRecipe::test_anime_cinematic_contrast_override`
- Same contrast mismatch (7.0 vs 12.0)

### 5. `test_recipe_validation.py::test_no_dead_recipe_keys`
- Dead keys in: `action` (relight), `anime_cinematic_action` (relight), `anime_crystal_void` (background_blur, background_desaturation, light_wrap, blue_shadow_grade, cyan_midtone_grade, subject_sharpen, matte_black), `body_match_v1` (body_skin), `outdoor_backlit_v1` (relight_azimuth, relight_elevation), `outdoor_golden_hour_v1` (relight_azimuth, relight_elevation)

## Skipped
- `test_io_icc.py::test_srgb_to_lab_changes_values_meaningfully` — missing optional dep
- 4 others (expected skips)

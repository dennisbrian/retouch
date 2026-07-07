# Session Summary — 2026-07-06

> Compact progress log. Source of truth for plan docs remains `MASTER_PLAN.md`.
> Model: rootsys/fiq/glm-5.2

## Goal
- Continue developing the Retouch Engine — implement F8.2 (native-res face crops), advance F1+E2 (float32 global pipeline), wire GUI, add tests, and close out Phase 1.

## Constraints & Preferences
- Do NOT run full pytest — only targeted tests touched by edits
- Parallel agents must touch different files to avoid merge conflicts
- Update MASTER_PLAN.md after each work item, including model name (rootsys/fiq/glm-5.2)
- Audit MASTER_PLAN back to owner after updates
- Check GUI side for wiring needs after engine changes
- Follow AGENTS.md rules: float32 internal, explicit cvtColor at boundaries, type hints, no `except: pass`

## Progress

### Done

- **F8.2 native-res face crops** — `_process_native_faces()` in `engine.py`: detection+segmentation at proxy, face bboxes+ied scale to native (landmarks already normalized [0,1]), reshape+per-face+composite at native. `quality="draft"` ctx flag keeps legacy F8.1 path. `local_clarity` hardcoded `radius=20` fixed to scale with `face_width` in `perf_optimizations.py`. `face_contexts` cache invalidated at proxy→native boundary. Tests: 112 engine/F8 + 2 proxy-pipeline pass.

- **F1+E2 (partial) — Wave 1 (3 parallel agents):**
  - Agent A (`grading.py`): all 12 fake-float `_F_*` methods made genuinely float-native, 4 float colorspace helpers added (`_bgr_f_to_lab_u8_conv`/`_lab_u8_conv_to_bgr_f`/`_bgr_f_to_hsv_u8_conv`/`_hsv_u8_conv_to_bgr_f`). 189 grading tests pass.
  - Agent B (`utils.py`): `vibrance()` now accepts float32 [0,255] with dtype-aware branches, uint8 path preserved byte-identical. 8 vibrance tests pass.
  - Agent C (`io.py`): `write_image_16bit()` + `read_image_16bit()` added (float32↔uint16 via ×257, PNG/TIFF/RAW support, path-traversal guard). 34 io tests pass.

- **F1+E2 — Wave 2 (2 parallel agents):**
  - Agent D (`engine.py` `_stage_grade`): 3/21 `_to_uint8_if_float` sites migrated to float-native (`_split_tone_three_way`, `apply_global_bloom`, `_add_vignette`), 17 remain inherently-uint8 (annotated with reasons: cv2.LUT, cvtColor uint8-convention LAB, color_space.bgr_to_lch). Helpers kept.
  - Agent E (tests): `test_float_pipeline.py` (52 tests), `test_io_16bit.py` (22 tests, 3 xfailed path-traversal → fixed). 72 new tests pass.

- **F1+E2 — Wave 3 (Fable direct):**
  - `_F_adjust_vibrance` in `engine.py` updated to call float-native `vibrance()` directly (no uint8 round-trip).
  - `_F_apply_uniform_saturation` made genuinely float-native.
  - `_stage_subject_separation` made float-native via `bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32`.
  - `_no_face_fallback` now converts to float32 at top, float-native through subject_separation/global/grade.
  - Path-traversal guard in `_resolve_safe_path` fixed (defective `except ValueError: continue` bug — restructured to raise outside try/except).
  - GUI wired: `quality_tier` radio (Full/Draft) + PNG-16 export format + `write_image_16bit` call path. `PROCESS_INPUT_KEYS` 112→113. test_gui alignment 3 pass.

- **P2 perf guards** — `_large_sigma_blur(img, ksize, max_compute_dim=1400)` helper in `grading.py`. Sub-1400px returns exact `cv2.GaussianBlur` (byte-identical). 6 sites swapped (lines ~446, 890, 1392, 1408, 1441, 1630). `benchmark.py`: `assert_grading_peak_rss(4K/6K, max_peak_mb=4096)` + `--peak-rss` CLI flag. 188 grading tests pass.

- **MASTER_PLAN.md** updated: row 3 (F8.2 DONE), row 3b (P2 DONE), row 4 (F1+E2 partial with full agent wave details), `▶ RESUME HERE` note updated to reflect Phase 1 closed.

### In Progress
- (none — all spawned work completed)

### Blocked
- **17 inherently-uint8 `_to_uint8_if_float` sites** in `engine.py` `_stage_grade` — need `color_space.py` `bgr_to_lch`/`bgr_to_lab` float-native variants (broader blast radius, deferred)
- **A2 competitive tuning** — blocked on A1 corpus (owner action: run Evoto/R4me/PixCake comparison set)

## Key Decisions
- F8.2 runs BiSeNet on native-res crops (sharper region boundaries) instead of the plan's "upscale per-face" suggestion — same 512×512 ONNX inference cost, better mask fidelity
- `face_contexts` cache invalidated when `proxy_scale < 1.0` in F8.2 path (proxy-res regions can't be reused at native)
- `_F_adjust_vibrance` kept as uint8 round-trip initially, then upgraded to float-native after Agent B fixed `vibrance()` in utils.py
- 17 `_to_uint8_if_float` sites annotated as inherently-uint8 rather than force-migrated (need color_space.py changes first)
- Path-traversal guard restructured: `try/except ValueError` loop moved so `raise` is outside the `except` block

## Next Steps
- Owner runs A1 corpus (Evoto/R4me/PixCake) — unblocks A2 and Phase 2 quality thesis
- Visual QA on F8.2 (real >2048px photo crop comparison)
- `color_space.py` float-native `bgr_to_lch`/`bgr_to_lab` variants (unblocks remaining 17 inherently-uint8 sites)
- Phase 3: P3 (stage-registry), F2 (sessions/undo), F3 (local masks)

## Critical Context
- `PROXY_MAX_DIM = 2048` (`engine.py:128`) — images >2048px trigger proxy path
- MediaPipe landmarks are normalized [0,1] — resolution-independent, no scaling needed proxy→native
- BiSeNet internally resizes to 512×512 for ONNX inference, then resizes prediction back to crop size — native crops give sharper mask boundaries at no extra inference cost
- `guided_filter` `max_dim=1200` is a compute-downsample (coefficients at 1200, applied at full res), not a fidelity cap
- `test_detects_face` failure is pre-existing (no face detected in real test image — unrelated to any changes)
- All work is uncommitted (working tree clean at session start, now has changes across 12 files)
- Model used: rootsys/fiq/glm-5.2

## Relevant Files
- `retouch/engine.py` — F8.2 `_process_native_faces()`, F1/E2 float helpers, `_stage_grade` wiring, `quality` field, `process(quality=)` kwarg
- `retouch/grading.py` — 12 `_F_*` float-native methods, `_large_sigma_blur` P2 guard, 4 float colorspace helpers
- `retouch/utils.py` — `vibrance()` float-native (dtype-aware branches)
- `retouch/io.py` — `write_image_16bit()`/`read_image_16bit()`, `_resolve_safe_path` guard fix
- `retouch/perf_optimizations.py` — `local_clarity` radius scaling, E1 `_tr()` bisect scaffolding (temporary, still present)
- `gui.py` — `quality_tier` radio, PNG-16 export, `PROCESS_INPUT_KEYS` (113), `process_image` wiring
- `scripts/bench/benchmark.py` — `assert_grading_peak_rss` + `--peak-rss` CLI
- `tests/test_float_pipeline.py` — 52 tests for grading `_F_*` + vibrance + engine float helpers
- `tests/test_io_16bit.py` — 22 tests for 16-bit round-trip + path traversal
- `tests/test_integration.py` — updated proxy-pipeline tests (F8.2 full vs draft)
- `tests/test_gui.py` — `PROCESS_INPUT_KEYS` count assertion 112→113
- `MASTER_PLAN.md` — row 3/3b/4 updated, `▶ RESUME HERE` note updated
- `PLAN_F8_EXECUTION.md` — F8.2 row with DONE receipt

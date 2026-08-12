# Progress (as of 2026-07-10)

> Snapshot of `MASTER_PLAN.md` status. Source of truth for stage receipts remains
> `MASTER_PLAN.md`; this file is a condensed progress view.

## Headline

- **~95% of plan scope complete.**
- All **6 phases ✅ COMPLETE** + P4.a ✅. 37 numbered stage rows, all `✅ DONE`.
- **11/11 flagship recipes**, **85 recipes** total.
- Core engine is **feature-complete and unit-tested**.
- **Post-plan (2026-07-10) — all COMMITTED**: `af54d2d` RAF import faithful neutral dev +
  highlight recovery + `face_exposure` param; `ef31d55` sclera vessel removal + auto backdrop
  cleanup; `742825f` fabric wrinkles; `25c445c` per-region wrinkles; `1b89d1e` reshape
  completeness (L/R + neck); `3c3f78d` auto body reshape; `ea7697e` forehead-mask fix;
  `40721d6` recipe-integrity dead-key fixes; `0610660` wiring fix — T3 `body_reshape_*` +
  A3 `cosplay_*` + `face_exposure` now recipe-reachable (`body_reshape_demo_v1`,
  `cosplay_wiring_demo_v1`, `portrait` `skin.face_exposure`); GUI sliders deferred.
  Visual-QA renders generated under `test_output/visual_qa*/`; review/sign-off still pending.

## Phase status

| Phase | Status |
|---|---|
| Phase 1 — Quality Floor | ✅ |
| Phase 2 — Skin Supremacy | ✅ (except A2, blocked on A1) |
| Phase 3 — Architecture & Workflow | ✅ |
| Phase 4 — Manual Tools | ✅ |
| Phase 5 — Intelligence | ✅ |
| Phase 6 — Creative Expansion & Moat | ✅ |
| P4.a (model-fetch infra) | ✅ |
| Phase 7 — Ship | 🔄 mostly done (2026-08-12) |

## Leftover (~5%)

### 0. Wiring debt + distribution gaps (2026-07-10 snapshot)
- **Unwired islands — mostly resolved (2nd pass, UNCOMMITTED)**: `plugin_api.py` (T4) now
  discovered+initialised in `RetouchEngine.__init__` (guarded, never fatal); `recipe_cookbook.py`
  (T4) + `look_extractor.py` (F6) now reachable via new `cli.py` flags (`--list-recipes`,
  `--search-recipes`, `--extract-look`/`--look-base`). `raw_develop.py` (T5) still unwired (gamma
  bug, out of scope).
- **GUI sliders for the 9 audit params** (`face_exposure`, `cosplay_wig_lace_blend`/
  `cosplay_stockings_smooth`/`cosplay_consistency_strength`, `body_reshape_arm/leg/torso/
  shoulder/hip_width` + `auto_body_reshape`) added as visible sliders. Fixing the GUI also
  required restoring `_process_inputs` ↔ `param_names()` alignment — 26 params were missing
  from `_process_inputs` (incl. the 9 audit params + 17 others: `wrinkle_soften_*`,
  `reshape_*_l/r`, `eye_sclera_vessel_remove`, `backdrop_cleanup`, `fabric_wrinkle_smooth`,
  `neural_*`). All 26 now present (9 visible, 17 hidden `gr.State`), so transport keys
  (`show_compare`/`fast`/export/debug) map correctly again. `test_params_gui_wiring` passes.
- **2 still GUI-invisible params**: `color_transfer_intensity`, `freckle_preserve_mask`
  (excluded from `PROCESS_INPUT_KEYS` by design).
- **Remaining auto-gap backlog item**: #5 auto stray-hair (A4-gated — needs a hair-strand
  segmentation model / evidence gate).
- **`models/manifest.json` URLs are placeholders** (`github.com/owner/retouch-models`).
- **No packaging metadata**: no `pyproject.toml` / `setup.py`; requirements unpinned.

### 1. Phase 7 — Ship (~80% done 2026-08-12)
- ✅ Update-check (`retouch/update_check.py` + GUI launch toast), diagnostics
  (`retouch/diagnostics.py`, rotating log + GUI 🩺 accordion), signing/notarization
  hooks in `scripts/build/build_app.sh`, `BUILD.md`. 17 new tests.
- 🔲 Remaining (owner/hardware-gated): run signing with real Developer ID
  credentials; Windows build validation on real hardware (browser-mode fallback
  documented in BUILD.md).
- Packaging/shipping work, not algorithm.

### 2. A1 / A2 — owner-gated tuning (blocked)
- **A1**: owner must supply Evoto / R4me / PixCake reference corpus (dev cannot do).
- **A2**: tune S2/S3/C1/C2 + freckle protection + E3 response-curve remaps to beat
  Retouch4me — blocked on A1.

### 3. Visual-QA verification passes PENDING (no new code — real-image runs only)
- F5 liquify (`geometry.py`, Visual-Critical: No-Edge-Tearing / Natural-Output / No-Halo / Background-line gates).
- F10 smart (`--smart` 20-photo mixed-folder acceptance per PLAN_TIERQ §4).
- 2026-07-09 `clear`/showcase family (`frequency`/`skin`/`freckle`/`under-eye`/`eye` modules).
- `masterwork_v1` — commit cites QA on `DSCF6102`, but that file is **not present** in
  `test_output/` (only DSCF4xxx / DSCF7xxx / DSCF8007 exist). Verify or fix the note.

### 4. Known bug, still-unfixed (documented, decision pending)
- `anime_cinematic_v1`: `relight_azimuth` / `relight_elevation` nested under `"skin"`
  instead of top level → silently ignored (resolves to defaults 0°/30° since before
  this session). Fix requires updating the recipe **and** `test_recipe_validation.py`'s
  `_VALID_KEYS` (which builds from `spec.recipe_key`, not `spec.engine_recipe_key`).

## Recipe generation artifacts
- `test_output/masterwork_v1/DSCF8007.jpg` + `_compare.jpg` + `.session.json` generated
  (denoise-first pipeline; ~120s CPU-pinned denoise on 6K).
- Full 85-recipe sweep on `DSCF8007.jpg` was **aborted** by the user; re-runnable via
  `scripts/recipes/recipe_sweep.py` or a per-recipe CLI loop. Output subdirs
  `test_output/recipes/<recipe>/` are partially populated.

### 5. Post-plan RAF-import + `face_exposure` work (2026-07-10 — COMMITTED in `af54d2d`)
- **RAF import (`retouch/io.py`)**: `bright=1.5→1.0` + `output_color=sRGB` on both raw
  paths (faithful neutral dev, no over-light); added `highlight_mode=ReconstructDefault`
  (recover blown highlights, no magenta clip). 92 tests pass. **Research 2026-07-10 on
  `_DSF1853.RAF`: ~1.01M clipped near-white px under `Clip` → 307 after reconstruct (mean
  85.5→54.3) — HIGH-impact, not cosmetic; must visually QA.** `demosaic_algorithm` and
  `fbdd_noise_reduction` are both no-ops for X-Trans (skip). Next win: **16-bit live
  ingestion** (8-bit gives 108 vs 4182 unique levels in a smooth patch → banding risk).
- **`face_exposure` param**: new masked L-lift skin-brightness knob, 0–100, independent of
  `relight` (caps at 1.0). Files: `params.py`/`engine.py`/`skin.py`/`perf_optimizations.py`
  + `tests/test_skin.py::TestFaceExposureLift` (3/3). 10-level RAF sample rendered
  (`_DSF1853_fe1..fe10.jpg`). Fixed `cli_type="float"`→`float` argparse crash.
  **[VISUAL QA PENDING]** (Visual-Critical `skin.py`); GUI/CLI slider not wired.
- Post-plan feature work is **committed and pushed** through `0610660`; current uncommitted tree is the in-flight 2nd-pass wiring + RAW-ingest work.
- `MASTER_PLAN.md` updated with a "Post-plan enhancements (2026-07-10)" section.
- **RAW research (2026-07-10)**: new `PLAN_RAW_PROCESSING.md` — full-data RAF ingestion
  audit + rawpy parameter research + wiring plan. Key findings: T5's `raw_develop.py` is an
  unwired island with a **gamma bug** (claims linear, returns BT.709-encoded — rawpy default
  gamma `(2.222, 4.5)`; reported not fixed); live path still 8-bit; recommended next slice =
  wire `read_image_16bit` into gui/cli/batch + float32 `process()` entry (~2–4 d).
- **RAW research 2nd pass (2026-07-10, measured on `_DSF1853.RAF`)**: 16-bit decode is
  **free** (9.09 s vs 9.22 s for 8-bit — cost is X-Trans Markesteijn demosaic either way);
  `half_size=True` is **54× faster** (0.17 s, 3123px — GUI fast path); `extract_thumb()`
  yields the embedded **full-size camera JPEG** (3.7 MB, film sim applied) for instant GUI
  first-paint / camera-rendition reference; RAF makernotes carry `FilmMode` (Velvia on ref)
  + lens info via exiftool → F10 could auto-select the matching Fuji recipe; rawpy exposes
  as-shot WB directly (`camera_whitebalance`). GUI needs a **RAF decode cache** (9 s/decode
  is unusable in the slider loop). lensfunpy (distortion/CA/vignetting) deferred — needs
  linear input, pairs with the raw_develop gamma fix. Full detail: `PLAN_RAW_PROCESSING.md` §5.
- **Whole-flow finding (3rd pass)**: **PNG-16 export writes 8-bit data** — `process()` always
  quantizes to uint8 at exit (`engine.py:1973`), then gui.py:552 feeds that into
  `write_image_16bit`. Middle of pipeline is float (E1/F1) but BOTH ends quantize; Step 1
  must add a `return_float` option to `process()` alongside 16-bit ingestion or neither end
  buys anything (`PLAN_RAW_PROCESSING.md` §5.7).

- **Auto-gap backlog (owner request 2026-07-10)**: owner wants "everything auto" — manual-only
  coverage re-scored as missing. 7-item backlog added to `MASTER_PLAN.md` Post-plan section:
  auto backdrop cleanup, sclera vessel removal, auto fabric wrinkles (**BiSeNet cloth label 16
  already available, unused**), per-region wrinkle sliders, auto stray hair (only true A4-class
  item), neck/L-R reshape, one-click auto body reshape. ~7–9 wk total; items 1–4/6–7 classical.

- **Wiring-debt audit (2026-07-10 discovery pass)**: 4 unwired islands all marked ✅ with green
  tests — `raw_develop.py` (T5), `plugin_api.py` (T4, discover never called), `recipe_cookbook.py`
  (T4 UI, no GUI), `look_extractor.py` (F6, unreachable) — plus LUT hot-reload (known). 11
  GUI-invisible params; **T3 `body_reshape_*` and A3 `cosplay_*` appear in NO recipe either →
  dark in both user paths.** Root cause: "✅ DONE" = module+tests, not user-reachable. ~4–6 d
  to wire everything. Detail: MASTER_PLAN.md Post-plan "Wiring-debt audit" row.

- **Recipe integrity audit (2026-07-10 final discovery pass)**: 4 dead-key classes across
  21 recipe-instances, verified via `recipe_to_params()` — **`film.preset` (6 film-branded
  recipes get NO film look; preset names don't exist anywhere)**, `skin.exposure_lock`
  (9 recipes, param doesn't exist), `frequency.nose_smooth` (5 recipes; ParamSpec has no
  recipe_key), `skin.micro_dodge_burn` (1; correct key `skin.micro_db`). **`masterwork_v1`
  flagship carries two dead keys.** Root cause: `test_no_dead_recipe_keys` validates
  top-level keys only, never recurses into nested roots. ~0.5–1 d to fix (+ film.preset
  design decision). Detail: MASTER_PLAN.md Post-plan "Recipe integrity audit" row.

#### Research log (2026-07-10, RAF import)
- **Dead parallel RAW path**: `retouch/raw_develop.py` (`RawDeveloper`) decodes to 16-bit
  **linear** RGB `[0,1]` but is NOT wired into engine/cli/gui (only `tests/test_raw_develop.py`
  references it). It also lacks `highlight_mode` (defaults to clip). Not a drop-in for 16-bit
  live ingestion (wrong color convention); `read_image_16bit` (sRGB float32) is the right base.
- **16-bit live-ingestion feasibility**: engine core is float-native — input converted at
  `engine.py:1454` (`native_img_bgr.astype(np.float32)`), so it accepts float32 `[0,255]`.
  **Blocker = detection boundary** (`engine.py:1421` → `detection.py` builds
  `mp.Image(SRGB, data=img_rgb)`; MediaPipe requires **uint8**). Feeding float32 breaks face
  detection. Fix options: (a) detection converts float→uint8 defensively, or (b) engine keeps a
  uint8 copy for detection + float for processing. gui/cli/`style_library` callers also assume
  uint8 (`resize_for_processing` docstring, `imwrite`) — audit before switching `imread_exif`.
- **White balance**: `use_camera_wb=True` applies the in-camera multipliers
  (`camera_whitebalance=[641,302,529,0]`) → warmer rendition (mean RGB [55,47,42]) vs
  `use_auto_wb` (neutral [46,44,43]). Faithful to camera; correct choice.
- **Tone curve / film-sim gap (open)**: rawpy applies sRGB gamma only — no Fuji film-sim
  S-curve (proprietary, not in RAF). Import is a *neutral* dev, not SOOC-faithful in tone.
  Cannot quantify without the camera JPEG; flagged as future research, not actionable now.
- **No-ops confirmed for X-Trans**: `demosaic_algorithm` (AHD==DHT; AMAZE/LMMSE need GPL packs)
  and `fbdd_noise_reduction` (Off/Light/Full identical) have zero effect — do not add.

### 6. Sclera vessel removal (2026-07-10, committed)
- **`eye_sclera_vessel_remove`** (0–100): new `eyes.py` `EyeEnhancer._remove_sclera_vessels`
  — detects red vessels via relative LAB-a redness inside the iris-excluded sclera mask,
  inpaints them (Telea). Iris/pupil/skin untouched. Wired `params.py`→`engine.py`→
  `perf_optimizations.py`; gate `eye_enhance>0 OR vessel>0` so vessel-only mode works.
  `tests/test_eyes.py::TestScleraVesselRemoval` 5/5 pass. End-to-end `process()` verified on
  `DSCF8007.jpg`; compare `test_output/eyes_vessel/compare_off_vs_on.jpg`.
  **[VISUAL QA PENDING]** — Visual-Critical (`eyes.py`).

### 7. Auto backdrop cleanup (2026-07-10, committed)
- **`backdrop_cleanup`** (0–100): new `retouch/backdrop.py::clean_backdrop` — detects dust/folds/dirt
  as high-frequency luminance outliers vs `GaussianBlur` (std-dev threshold; detail band preserved,
  only outliers removed, Telea inpaint). Subject edge protected by **eroding** `~person_mask` ~8px
  then feathering before inpaint (fixes a subject-edge-bleed trap); composite is **hard** over the
  eroded region. `ParamSpec backdrop_cleanup` (`cli_flag="backdrop-cleanup"`,
  `recipe_key="background.backdrop_cleanup"`, `conversion="recipe_pct"`) wired `params.py`→`engine.py`
  (`_run_global_phases` before `to_uint8`, both registry + hardcoded paths)→`ProcessingContext`.
  `tests/test_backdrop.py` 6/6 pass. **[VISUAL QA PENDING]** — Visual-Critical-adjacent.

### 8. Auto fabric/clothing wrinkle smoothing (2026-07-10, committed)
- **`fabric_wrinkle_smooth`** (0–100, `cli_type=float`): new `retouch/fabric.py::smooth_fabric_wrinkles`
  — detects mid-frequency fold ridges via difference-of-Gaussians on the LAB L channel (dark folds =
  negative DoG), lightens them capped at 60% depth reduction. Fabric weave (high-freq) and gentle
  low-freq shading preserved; **no** GaussianBlur-on-cloth smoothing (mirrors `skin.wrinkle_soften`).
  Cloth mask = `person_mask − acc_skin_hair` (skin/hair/neck excluded) derived in `engine._run_global_phases`;
  `FaceRegions.cloth` also exposed from BiSeNet label 16 (feathered) in `parsing.py` `parse()`/`parse_batch()`.
  `ParamSpec fabric_wrinkle_smooth` (`cli_flag="fabric-wrinkle-smooth"`, `recipe_key="fabric.wrinkle_smooth"`,
  `conversion="recipe_pct"`) wired `params.py`→`engine.py` (`_run_global_phases` right after backdrop, before
  `to_uint8`, both paths)→`ProcessingContext` + `process()` kwarg + overrides entry + `retouch()` wrapper.
  `tests/test_fabric.py` 7/7 pass. **[VISUAL QA PENDING]** — Visual-Critical-adjacent.

### 9. Per-region wrinkle sliders (2026-07-10, committed)
- **`wrinkle_soften_forehead` / `wrinkle_soften_nasolabial` / `wrinkle_soften_neck`** (0–100, `cli_type=float`):
  `retouch/skin.py::wrinkle_soften` split into `_wrinkle_soften_masked` (scoped to one zone-mask) run
  per-region via `region_strengths={"forehead","nasolabial","neck"}`. Region path uses attrs
  `("forehead",)`, `("nasolabial_l","nasolabial_r")`, `("neck",)`; eye/hair/eyebrow exclusion kept per-region.
  Any region strength > 0 takes the per-region path (global `strength` ignored for those regions); all-zero/`None`
  falls back to the global union path → `skin.wrinkle_soften` recipes stay backward compatible. Three `ParamSpec`s
  (`cli_flag="wrinkle-soften-*"`, `recipe_key="skin.wrinkle_soften_*"`, `conversion="recipe_pct"`) wired
  `params.py`→`engine.py` (`ProcessingContext` fields + `process()` kwargs + overrides entries + `retouch()`
  wrapper)→`perf_optimizations.py` builds `region_strengths` from `ctx`. `tests/test_skin.py::TestPerRegionWrinkle`
  pass. **[VISUAL QA PENDING]** — Visual-Critical (`skin.py`).

### 10. Reshape completeness — L/R variants + neck (auto-gap #6, 2026-07-10, committed)
- **8 new reshape params** (all `conversion="gui_direct"`, `recipe_key="reshape.*"`, −50..50, mirror existing
  reshape specs): `reshape_jaw_width_l/r`, `reshape_nose_width_l/r`, `reshape_eye_size_l/r`,
  `reshape_neck_width`, `reshape_neck_length`. Wired `params.py`→`engine.py` (`ProcessingContext` fields +
  `process()` kwargs + overrides entries + `_any_reshape_active` gate; `retouch()` wrapper forwards via `**kwargs`).
- **L/R got a split:** jaw_width (234=R/454=L), nose_width (48=L/278=R alae), eye_size (LEFT/RIGHT iris).
  `_jaw_width_warps`/`_nose_width_warps`/`_eye_size_warps` accept optional `slider_l`/`slider_r`.
- **Precedence:** in `reshape()`, if either side variant of a feature ≠ 0 → per-side path (each side driven by
  its own strength, untouched side skipped); if BOTH side variants = 0 → global symmetric `reshape.<key>` path,
  **byte-identical to legacy** (test `test_jaw_global_byte_identical`, `test_side_zero_matches_global`).
- **Neck:** `_neck_width_warps` (jaw angles 234/454 + jaw-line 58/172/288/397, horizontal inward, R=fw×0.5 cap),
  `_neck_length_warps` (chin 152 + jaw angles, vertical). No MediaPipe neck landmarks → jaw/chin band is the proxy.
- **Left symmetric (no L/R split), by design:** eye_distance, nose_length, chin_length, mouth_size, smile,
  forehead — centered/vertical or naturally-paired controls where independent L/R has no clear photographic meaning.
- `tests/test_geometry.py` +16 pass (155 geometry+params total); regression backdrop/fabric/eyes/skin 130 pass.
  **[VISUAL QA PENDING]** — Visual-Critical (`geometry.py`).

### 11. One-click auto body reshape (auto-gap #7, 2026-07-10, committed)
- **`suggest_body_reshape(pose_ctx)`** (new, `retouch/body_reshape.py`): returns 0–100 suggestions
  (50 = neutral) for `arm_length`/`leg_length`/`torso_width`/`shoulder_width`/`hip_width`. Gentle,
  capped corrections toward balanced proportions (shoulder:hip ~1.3, leg:torso ~1.2, arm:torso ~1.0).
  Guards: no landmarks / low-visibility / disabled `feature_flags` → all 50.0.
- **`auto_body_reshape`** (0–100, `cli_type=float`): `ParamSpec` (`cli_flag="auto-body-reshape"`,
  `recipe_key="body_reshape.auto"`, `conversion="gui_direct"`) wired `params.py`→`engine.py`
  (`ProcessingContext.auto_body_reshape` default 0 + `process()` kwarg + overrides entry)→
  `_stage_body_reshape`. When `> 0`, the stage detects pose on the uint8 frame, blends
  `centered = (suggested − 50)·(auto/100) + (manual − 50)`, and proceeds (no early-return even if
  manual sliders are neutral); falls back to manual-only early-return only if pose detection finds
  nothing. `retouch()` forwards via `**kwargs`.
- **Manual + auto combine additively** in centered (−50..+50) space: auto supplies a proportion-based
  offset scaled by strength; manual slider offsets add on top, so a user can fine-tune the auto result.
- `tests/test_body_reshape.py` 8 pass (7 heuristics + 1 ParamSpec; **no MediaPipe model load**);
  full suite green: `test_body_reshape`+`test_params` 140, regression `test_geometry/backdrop/fabric/eyes/skin` 153.
  **[VISUAL QA PENDING]** — real-photo pass deferred until a pose model is available.
- **Auto-gap backlog status**: #1 DONE, #2 DONE, #3 DONE, #4 DONE, #6 DONE, #7 DONE (all committed & pushed, `af54d2d`..`0610660`);
  #5 parked (A4-gated — needs a hair-strand segmentation model / evidence gate).

## Last updated
2026-07-10 (end of day) — four discovery passes documented in MASTER_PLAN.md Post-plan section:
RAW processing (`PLAN_RAW_PROCESSING.md` + T5 audit), auto-gap backlog (owner request),
wiring-debt audit (4 unwired islands + 11 GUI-invisible params), recipe integrity audit
(4 dead-key classes, 21 instances, guard-test blind spot). Earlier same day: Post-plan section
+ §5 for RAF-import / `face_exposure` work (committed `af54d2d`). Then §6 sclera vessel removal +
  §7 backdrop cleanup, §8 fabric wrinkle smoothing (both committed). Prior: `0822b9c` RESUME.

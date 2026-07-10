# Progress (as of 2026-07-10)

> Snapshot of `MASTER_PLAN.md` status. Source of truth for stage receipts remains
> `MASTER_PLAN.md`; this file is a condensed progress view.

## Headline

- **~95% of plan scope complete.**
- All **6 phases ✅ COMPLETE** + P4.a ✅. 37 numbered stage rows, all `✅ DONE`.
- **11/11 flagship recipes**, **85 recipes** total.
- Core engine is **feature-complete and unit-tested**.
- **Post-plan (2026-07-10, UNCOMMITTED)**: RAF import faithful neutral dev + highlight
  recovery, and a new `face_exposure` skin-brightness param. Both implemented; visual QA
  pending; not yet committed (3 commits already ahead of origin are recipe/doc only).

## Phase status

| Phase | Status |
|---|---|
| Phase 1 (foundations) | ✅ |
| Phase 2 (manual tools) | ✅ (except A2, blocked on A1) |
| Phase 3 (creative ecosystem) | ✅ |
| Phase 4 (fidelity: C3/F4/F4.b/F5/F6/F7/H1/H2) | ✅ |
| Phase 5 (harmonize/F9/F10/A5) | ✅ |
| Phase 6 (T1/T2/T3/T4/T5/A3/A4) | ✅ |
| P4.a (model-fetch infra) | ✅ |
| Phase 7 (P4 distribution hardening) | 🔴 not started |

## Leftover (~5%)

### 1. Phase 7 — P4 distribution hardening (0% started)
- Code signing, update-check, diagnostics, Windows build.
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

### 5. Post-plan RAF-import + `face_exposure` work (2026-07-10 — UNCOMMITTED)
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
- All 6 modified files are **uncommitted** (3 commits already ahead of origin: recipe/doc only).
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

## Last updated
2026-07-10 — RAW-processing research doc (`PLAN_RAW_PROCESSING.md`) + T5 status-audit note in
`MASTER_PLAN.md`. Earlier same day: Post-plan section + §5 for RAF-import / `face_exposure`
work (uncommitted). Prior: `0822b9c` RESUME + #11/showcase-family.

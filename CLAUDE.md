# Claude.md — Pro Max Face Retouch Engine

**Project:** Professional automated face retouching pipeline  
**Repository:** https://github.com/dennisbrian/retouch  
**Status:** Mature (v2.0.0 — Fuji-quality color recipe system)  
**Last Updated:** 2026-09-28

---

## Git Conventions
- NEVER add 'Co-Authored-By: Claude' or 'Generated with Claude Code' trailers to commit messages.
- Before switching branches (e.g. to hotfix/production), run `git status`. Stash local config (config/db.php, dev-login changes) with a descriptive message, and never commit it.
- When porting a change to production, branch or worktree off `origin/production` and cherry-pick only the requested commits. Never merge whole feature branches unless explicitly asked.
- After any fix, confirm `git status` is clean BEFORE testing on staging. Uncommitted fixes do not deploy.

## Audit & Review Scope
- When asked to audit or review a branch, scope it to files changed on THAT branch only: `git diff --name-only origin/production...HEAD` (or the named base). Confirm the file list with me before reviewing. Do not audit unrelated modules or the full master..HEAD diff.
- Before claiming something has 'no tests' or 'no coverage', check untracked files too (`git status --porcelain`, `ls tests/`).
- Record hypotheses in memory/docs only AFTER they are verified by a discriminating experiment.

---

## Quick Context

This is a **production-grade image processing engine** that applies professional retouching to portraits via a modular 7-stage pipeline. It combines:
- MediaPipe face detection + 478-point landmarks
- BiSeNet ONNX semantic segmentation (skin, lips, eyes, hair, etc.)
- Frequency-separation skin smoothing + component enhancement
- Color grading + Fuji film simulation presets
- Virtual studio relighting + advanced lens effects

**~98k LOC, 140 modules in retouch/, 6,518 tests — see [docs/REPO_STATS.md](docs/REPO_STATS.md) (generated; run `scripts/dev/governance --write-stats` to refresh).**

---

## For Claude Code Users

### Do NOT Spawn Agents for File Reads
- **NO:** `/code-review ultra`, `/verify`, Explore agent for simple file exploration
- **YES:** Use native Haiku tools (Read, Bash grep, find)
- **Why:** Budget-conscious — each agent spawn costs tokens. Stay cheap.

### Model Strategy
Task difficulty decides the model, not a flat cheapest-first cascade:
- **Mechanical/routine** (wiring a param, doc hygiene, test-file writing to a tight spec): Haiku.
- **Bug fixes**: Opus.
- **Hard/judgment-heavy** (multi-site refactors, calibration/research work, new feature design with open judgment calls): Fable, inline — not delegated to a subagent.
- **Thinking mode:** OFF by default, prompted only when needed for complex reasoning

### Key Commands
```bash
# Run full test suite
.venv/bin/python -m pytest tests/ -q

# Test syntax
for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done

# Run engine directly
python3 -c "from retouch import RetouchEngine; engine = RetouchEngine(); result = engine.process(cv2.imread('test.jpg'))"

# GUI
python3 gui.py  # Opens http://127.0.0.1:7860

# CLI
python3 cli.py /path/to/photos -o /out --recipe cosplay --workers 4

# Benchmarks
python3 scripts/bench/benchmark.py
```

---

## Architecture Decisions

### Single-Source-of-Truth: `retouch/params.py`
Every tunable parameter is registered in `PROCESSING_PARAMS` (a list of `ParamSpec` objects) — GUI, CLI, engine defaults, and recipes all auto-generate from it.

**When adding a new parameter:**
1. Add one `ParamSpec` entry in `params.py` — CLI flag + engine defaults auto-wire.
2. GUI does NOT auto-wire: add a matching entry to `_process_input_components` (name → Gradio component, or a `gr.State(...)` placeholder) in `gui.py`, keyed by name. Order is derived, not hand-placed — forgetting the entry raises an `AssertionError` at import time (see `tests/test_gui.py::TestProcessInputKeys`), not a silent slider mismatch.
3. Same pattern applies to `_recipe_output_components` / `RECIPE_OUTPUT_KEYS` for recipe/reset-handler outputs (see `tests/test_gui.py`, `TestRecipeOutputKeys`-style tests). The 15 `reset_*` handlers are still hand-ordered pairs (latent, low-risk), but `TestResetFunctions::test_reset_function_key_order_matches_click_output_order` in `tests/test_gui.py` statically compares each handler's return-tuple key order against its `.click(outputs=[...])` order and fails on drift — see the 2026-09-09 entry below.

### Modular Pipeline (7 Stages)
```
Stage 0: Detection & Segmentation (faces, landmarks, person mask)
Stage 1: Face Reshaping (liquid warping for slimming/chin lift)
Stage 2: Per-Face Processing (frequency sep → component enhancement)
Stage 3: Global Tonal Adjustments (contrast, brightness, curves)
Stage 4: Subject-Background Separation (optional)
Stage 5: Color Grading (presets, LUTs, split-toning, Fuji foundation)
Stage 6: Sharpening & Impact Finish (selective sharpening, global glow)
```

Each stage is a private method (`_stage_*`) that can be tested or bypassed independently.

### FaceContext Caching
Supply cached `face_contexts` to `process()` to skip detection + parsing. Critical for GUI slider interaction — avoid redundant inference.

### Proxy Resolution for High-Res Images
Above `PROXY_MAX_DIM` (2048px) the pipeline branches on `quality`
(`engine.py::_process_with_proxy`). **The two paths have very different cost —
the proxy does NOT bound runtime in the default mode.**

- **`quality="full"` (F8.2, DEFAULT):** only detection + segmentation run at the
  2048px proxy. Reshaping, per-face work (BiSeNet parsing, frequency separation,
  skin ops, composite) and the global stages all run at **native** resolution, so
  face texture is never resampled. Cost scales with native pixels.
- **`quality="draft"` (F8.1 legacy):** downscale → run stages 0–2 at proxy →
  upscale + composite onto native, with F8.0 detail reinjection. For fast batch
  contact sheets.

Measured on a 6240×4160 (24 MP) input, `natural`, M3 Pro / macOS 25.5
(2026-07-22), one recipe:

| Path | Runtime | Peak RAM |
|---|---|---|
| `--max-dim 2048` (pre-shrunk before the engine) | 5.3 s | low |
| full-res, `quality="draft"` | 23.5 s | 8.1 GB |
| full-res, `quality="full"` (default) | **27–325 s observed on the current native corpus** | **5.64 GB observed peak** |

The older "7.5 GB → 1.84 GB, 15.3s → 3.09s" figure describes the **draft**
round-trip only; it does not apply to the default full path. The older
">25 min, never observed completing" figure is stale: current 26 MP corpus
runs complete in roughly 27–325 s depending on recipe and face workload. The
5.64 GB value is one measured peak, not a hard memory guarantee; prefer
`--max-dim 2048` or `quality="draft"` for multi-recipe sweeps.

---

## Testing & Quality

### Test Coverage
- **Test count:** see [docs/REPO_STATS.md](docs/REPO_STATS.md) (generated) — full-suite pass/fail baseline not re-run at this count
- **89% coverage** on core modules (per `pytest --cov`; last measured 2026-07-01)
- **Unit + integration tests** for every public API
- **Deep algorithmic verification** (monotonicity checks, round-trip stability, etc.)

### Audit Status
- **Security:** No hardcoded secrets, API keys, or absolute paths
- **Performance:** Benchmarked at 702ms per-face (400×400, detection mocked, 2026-06-23 benchmark), 4.7ms for no-face global-only (400×400, legacy median, 2026-08-18)
- **Process:** Weekly syntax checks, monthly algorithmic re-audit (`docs/review/AUDIT_REPORT.md` — last cycle 2026-07-06; report's own §0 flags its Cycle 3 verdict as superseded by `docs/review/TEST_REPORT_2026-07-04.md`, so read past the executive summary before citing it as current status)

### Known Limitations
- Detection on real corpora (2026-08-19 studies, `docs/plans/RESEARCH_DETECTION_RECALL_2026_08_19.md` + `RESEARCH_POSTERFP_VETO_2026_08_19.md`): subject-face recall 100% on the 83-image DSCF corpus (dual-scale person-gated augment, `8811320`). Poster-FP precision: a validated joint veto (person-coverage<0.5 OR [coverage<0.9 AND texture<80]) kills 14/18 anime-poster FPs with 0/82 real-face collateral — implementation awaits owner sign-off + group-shot re-validation; 4 on-person poster FPs are unvetoable by any measured feature. Texture-only vetoes are UNSAFE (real faces degrade to Lap-var 1–170 under blur/makeup). RetinaFace: `pip` resolution structurally conflicts with the pinned runtime (forces protobuf 6/TF 2.20/numpy 2/cv2 5) — MediaPipe-only is the supported detection path; the `cc61e67` F1/F2 fixes are documented-not-live. Background/crowd-face recall 27–35% (out of scope for portrait retouch).

### Verification & Honesty
- Never simulate or describe pipeline/engine output in prose — actually invoke `RetouchEngine`/CLI/GUI and show genuine results. If you can't run it (no test image, no GPU, etc.), say so explicitly instead of narrating a plausible-looking result.
- After any code fix, run the relevant regression tests (or full suite, per repo convention — no full pytest unprompted) and verify against a real rendered image before claiming the fix works. Delta metrics alone can hide face-region damage — view the actual output.

---

## Outstanding Fixes

> Resolved items are one-liners; full root-cause/verification detail lives in
> the commit message (`git show <hash>` or `git log --grep "<keyword>"`), not
> here — this file is sent on every request, so keep it an index, not an essay.

- ✅ 2026-07-03/04 batch (B&W mixer literals, E1 float32 canvas, equalize pale-face + s² non-linearity, sharpen inert, white-balance literals, `_build_dimensional_mask` shape fix) — `260fe02`, `e0b294c`, `1640eeb`, `6331994`
- ✅ 2026-07-13 `skin.locus` recipe override wiring; test hang was a MediaPipe teardown deadlock (bare `RetouchEngine()`, not init abort) — `36cb93a`
- ✅ 2026-07-13 `_process_inputs` argument-order footgun — name-keyed dict + import-time drift guard, see [Architecture Decisions](#single-source-of-truth-retouchparamspy) — `f0b656b`, `da1f8cb`
- ✅ 2026-07-13 `_recipe_outputs` mirror-image footgun (same fix pattern; closed a live 107-vs-111 value drift) — see [Architecture Decisions](#single-source-of-truth-retouchparamspy)
- ✅ 2026-07-14 `tests/test_cosplay_moat.py` bare-`RetouchEngine()` → module `engine` fixture (same teardown pattern as skin_locus)
- ✅ 2026-07-14 `test_ext_map_keys_match_radio_choices` — expect `PNG-16` (map already had it; test was stale)
- ✅ 2026-07-21 LUT hot-reload wired: GUI watcher (gui.py) + CLI `--reload-luts` flag — `7c8d607`
- ✅ 2026-08-11 redundant `ci.yml` workflow removed — `7b97df9`
- ✅ 2026-08-19 golden-v2 face-path harness: `test_golden_pipeline.py`'s synthetic image has no detectable face, so the per-face pipeline was untested. `tests/test_golden_pipeline_face.py` + `tests/golden_face_fixture.py` inject a real `FaceContext` via `face_contexts=`; mutation-verified.
- ✅ 2026-08-26 P0/P1 eye handedness fix: BiSeNet subject-anatomical vs parsing.py's camera-viewer convention were opposite, so sclera ops were a no-op and painted the other eye's iris. Swap applied in `_masks_from_label_map`; golden face hashes updated. Detail: `docs/review/REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md` §P0/P1.
- ✅ 2026-08-26 per-eye `_enhance_catchlights` (B6 latent bug): pooled-iris mask coupled the eyes; now called once per iris.
- ✅ 2026-08-26 eye-visibility gate shipped (EAR-primary + tone-adaptive contrast secondary): `retouch/eye_visibility.py`, ParamSpec `eye_gate` (default on), fails open when BiSeNet eye mask < 100 px. Mutation-tested: `tests/test_eye_visibility.py`. **Initial thresholds superseded by the 2026-08-31 entry below.**
- ✅ 2026-08-31 eye-gate threshold recalibration: `docs/plans/RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md` found `_MIN_EAR=0.12` missed 84% of closed eyes (mislabeled anchor). Shipped `_MIN_EAR=0.285` / `_MIN_CONTRAST=0.55` — 0/38 occluded eyes missed, 5.3% visible false-gated on the 83-image corpus. Still provisional: no hair/wig/sunglasses cases, no Fitzpatrick IV-VI stratum yet.
- ✅ 2026-09-02 **composite-mask epsilon bug (P0)**: every mask normaliser used `max() > 1.0` to detect 0–255 masks; float32 masks overshoot 1.0 by one ulp after feathering on ~24% of real portraits, silently discarding skin/hair/neck ops. Threshold `> 1.5` + clip at 21 sites — `f69ab1e`, `tests/test_mask_norm_epsilon.py`. **Any past "op X does nothing on image Y" conclusion may have been this.** Detail: `docs/plans/RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md` §5.
- ✅ 2026-09-02 yaw-gate recalibration: nose-bridge/temple ratio bands moved to shared `utils.YAW_GATE_START/END` = 2.5→4.0 (old bands zeroed slimming on 58/83 corpus faces) — `9c26493`. **Its shared ramp helper shipped inverted — see the yaw-ramp entry below.** Study: `docs/plans/RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md`.
- ✅ 2026-09-02 audit-finding closeouts: undereye `dark_circles` + `undereye_darken_removal` double-apply (27 recipes) aliased at dispatch, max not sum — `e73d3ba`; lips/teeth whole-image LAB roundtrips contained via `utils.restore_outside_support` — `46be031`. Eye-v0 uint8 ROI roundtrip resolved below; dark-circle op resolved below (v2).
- ✅ 2026-09-02 **post-epsilon re-audit** (83 faces, `docs/plans/RESEARCH_POST_EPSILON_FACEOP_REAUDIT_2026_09_02.md`): post-fix skin delta matches controls — default strengths are NOT over-strong, no re-tune needed. Remaining near-zero skin ops are multi-face images whose largest detection is a hand/background person — corpus hygiene, not op defects.
- ✅ 2026-09-02 **yaw ramp inverted (P1)**: `utils.yaw_gate_factor` returned the raw smoothstep instead of 1−smoothstep, so relight/sculpt/slimming got ~0 strength just past `YAW_GATE_START`. Fixed + monotonicity test (`tests/test_utils.py::TestYawGateFactor`).
- ✅ 2026-09-02 **neck "depth gate" was a guaranteed no-op on frontal faces**: `skin.harmonize_neck`'s eye-corner/nose-bridge plane check lost 100% of the neck mask on 15/15 frontal corpus faces. Gate removed; chroma gate now applied to the BiSeNet neck path too, LAB roundtrip contained via `restore_outside_support`.
- ✅ 2026-09-02 **dark-circle op was inert on every face (v1 → v2)**: v1's detector lit the lash line, not skin shadow, on real faces (55/128 recipes set the key; none did anything). v2 (`retouch/undereye.py`) is tone-invariant (relative to a cheek-ring median), ROI-confined, feathered eye-contour exclusion. **Limitation:** aegyo-sal contour makeup reads as shadow at strength ≥0.45; no darker-skin sample in corpus. Study: `docs/plans/RESEARCH_DARK_CIRCLE_OP_2026_09_02.md`.
- ⚠️ 2026-09-02 golden face hashes are interpreter-specific: bare `python3` (cv2 4.13) ≠ `.venv/bin/python` (cv2 4.11). Snapshots are pinned to `.venv`; run `RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python -m pytest …` — `efb17af`.
- ✅ 2026-09-09 static drift guard added for the 15 `reset_*` handlers in gui.py: `tests/test_gui.py::TestResetFunctions::test_reset_function_key_order_matches_click_output_order` statically compares each handler's return-tuple key order against its `.click(outputs=[...])` order and fails on drift.
- ✅ 2026-09-14 **eye-v0 uint8 ROI roundtrip fixed**: `perf_optimizations.py`'s eye-v0 dispatch wrapped the whole canvas in `float→uint8→float`, dithering non-eye pixels (wig, hand, background) even though the edit is masked to sclera/iris. Cast removed — canvas now passed through in native float32.
- ✅ 2026-09-14 **`cli.py` disk-space preflight added**: `_check_disk_space` estimates output size as `input_bytes × 2.0` and exits if projected free space would drop below 5 GiB; `--skip-disk-check` bypasses it; walks up to the nearest existing ancestor dir for a not-yet-created `--output` path. Regression test: `tests/test_cli_helpers.py::TestCheckDiskSpace::test_checks_volume_of_nearest_existing_ancestor`.
- ✅ 2026-09-14/15 **`cli.py --workers>1` multi-face batch hang, actually fixed this time (`cef4eff`)**: `c8ddc6f`'s atexit hook never fires on a multi-face image because `RetouchEngine`'s inner `FaceProcessorPool` spawns non-daemon grandchildren that block interpreter shutdown, so the outer `ProcessPoolExecutor.shutdown(wait=True)` hung forever. Fixed by shutting the inner pool down inline, synchronously, in `_process_single`'s `finally`; atexit remains a single-face/abrupt-exit backstop only. Verified clean exit + zero orphaned processes on real single-face and multi-face batches.
- ✅ 2026-09-25 **QA flag recalibration**: banding / plastic_skin / seam / asymmetry flagged 100% of real renders (absolute measures over the person mask); now differential vs the input photo (face-skin mask for texture checks), color_drift tail gate p99>10° → p95>20°. 35 renders: 0 false flags; planted defects still flag. `docs/plans/RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md`, `tests/test_qa_flag_calibration.py`.
- ✅ 2026-09-25 **person gate dropped real faces under near-white wigs (P1)**: the full-frame selfie mask lost the head against a blown window (DSCF3773 coverage 0.000; 9/47 pilot faces dropped). Gate rejects now get a second opinion: multiclass face-skin coverage on a 1.5x face crop >= 0.45 keeps the face (56 real faces min 0.478, 931 background boxes max 0.302); model unavailable -> old behaviour. Tests: `tests/test_detection_person_gate.py::TestFaceSkinRescue`. Bug write-up: PR #17.

- ✅ 2026-09-25 **camera C2PA passthrough made retouched JPEGs verify as Invalid** (`assertion.dataHash.mismatch`): `io.py` no longer copies a source manifest; opt-in `--edit-report` (JSON per photo) and `--sign-cert/--sign-key` (new manifest, user's own cert, camera manifest kept as parent) in `retouch/edit_report.py`, `retouch/content_credentials.py`. Guide: `docs/guides/CONTENT_CREDENTIALS.md`.
- ✅ 2026-09-25 **glasses / goggle / visor glare removal** (opt-in `lens_glare` 0-100, `--lens-glare`, GUI Eyes & Lips): `retouch/lens_glare.py` subtracts an additive low-frequency veil measured above a closing+opening local reference (tone-invariant margin), runs before reshape. Planted glare on real faces: 40-70% lifted; glare wholly inside the eye opening is kept. Tests: `tests/test_lens_glare.py`.
- ✅ 2026-09-25 **watermark / credit overlay** (opt-in `--watermark TEXT`, `./run watermark`, GUI accordion + Batch tab box): `retouch/watermark.py` stamps copies into `<output>/watermarked/` (masters untouched) and social crops; `auto` placement skips corners that touch a padded face box. Built-in font = Pillow's embedded Aileron (CC0; no accents/CJK). Tests: `tests/test_watermark.py`.
- ✅ 2026-09-26 **shoot-wide duplicate detection** (`./run dupes FOLDER`, `retouch/duplicates.py`, also rows in the GUI Shoot scan): groups same pose + framing at any time gap (5-scale x 5-tilt NCC alignment, then a 12x12 gradient-cell pose check), complete linkage, keeper via `rank_burst_candidates`; report only unless `--move-extras`. 0 cross-clip groups on 1,944 real video frames, 0 on 94 public portraits; close-up expression changes can split true repeats. Tests: `tests/test_duplicates.py`. Auto strap/zip/tag healing (#14) was tried first and dropped: the 256 px class segmenter can't see real-width straps.
- ✅ 2026-09-26 **prosthetic edge blending** (opt-in `prosthetic_blend` 0-100, `--prosthetic-blend`, GUI Cosplay & Body): `retouch/prosthetic_blend.py` finds long seams around ears/forehead on the pre-face-edit frame (chromaticity or lightness edge + colour offset across the line, tone-relative) and ramps them out on the retouched image. Simulated appliances only: finds 55-80% of a forehead piece's edge; short seams missed. 0 seams on 7 real photos (13 faces). Tests: `tests/test_prosthetic_blend.py`.
- ✅ 2026-09-25 **flash red-eye fix** (opt-in `red_eye` 0-100, `--red-eye`, GUI Eyes & Lips): `retouch/red_eye.py` finds the red pupil from the iris landmarks with a hue-gated red ratio measured against the face's own skin (tone-invariant), requires a red pupil *core* (red contact lenses keep a dark core) and a non-red surround, then neutralises it to a dark grey, catchlight kept. Public red-eye photos: 39/46 red eyes fixed, 0 changes on 11 control photos (incl. darker skin, Alex's wig uploads). Tests: `tests/test_red_eye.py`.
- ✅ 2026-09-26 **wig hair masks without BiSeNet**: the multiclass segmenter labels most of a pale/coloured wig "others" (Alex's white wig: 11.6k px of hair mask, bangs only). `parsing._grow_wig_hair` grows hair into connected accessory/clothes pixels whose Lab colour is common in the confident hair and rare in the costume (157k px; hat, collar, prop stay out; darker-skin natural hair unchanged). `parse_hair_full_image` now falls back to it, so long wigs over the chest leave body skin. Also: fallback skin was zeroed when the person mask lost the head under a pale wig; face-skin class now vouches. Same-coloured hair and costume: no growth. Tests: `tests/test_wig_hair_growth.py`.
- ✅ 2026-09-27 **automatic spot healing** (opt-in `spot_heal` 0-100, `--spot-heal`, GUI Skin Smoothing): `retouch/spot_heal_auto.py` finds dark/red blobs against a skin-filled local median, divided by the face's own texture spread at 3 spot scales (tone-invariant) plus a 2.5 delta-E floor; keeps moles (non-red, wide or >16% darker), bright dots, lines, shading (ring test) and anything within 3% face width of eyes/brows/lips/hair. Heals each spot as a plane fitted to its ring + donor-patch texture; skips spots on a step edge. Runs first in the per-face core. Planted spots on Alex's 4 wig/bunny faces, real and simulated darker skin: pimples fixed 31/60, moles kept 22/22, 0.09% stray skin pixels (legacy `blemish` at 60: 0/60, 1/22, 1.77%); faint brown marks 2/24. Tests: `tests/test_spot_heal_auto.py`.
- ✅ 2026-09-26 **skin mask v2 on the no-BiSeNet path** (what every fresh install uses): `parsing._refine_fallback_with_classes` ramps the multiclass segmenter's soft confidences to 0/1 (skin used to cap at 0.87-0.95, so every skin op ran at ~90%), guided-filters the class factor to the photo so the 256 px blocks no longer step the mask, and the 40% loss guard now ignores hair above the eye line (full bangs used to throw the whole refinement away and get smoothed as skin, DSCF3503). Skin ops are ~25-35% stronger on confident skin as a result. Darker skin tested on simulated tone only. Tests: `tests/test_parsing_class_segmenter.py::TestSkinMaskV2`.
- ✅ 2026-09-26 **face polish + two inert/harmful skin ops fixed**: `skin.shine_removal` never fired on real faces (gate wanted chroma < 40% of skin; real highlights keep 75-90%, and a global-median L gate caught the lit side) — v2 measures against a two-pass local skin baseline. `apply_specular_finish` built its baseline from the whole crop (hair/background), so on real faces lit skin read as specular and non-default finishes darkened (matte/powder) or blew out (dewy/glass) the face — now uses the skin mask. Changes output of the 33 recipes that set shine removal and 19 that set a finish. New `face_polish` 0-100 (`--face-polish`, GUI slider, `skin.face_polish`) raises shine_removal/shadow_lift/face_exposure floors, tuned on 13 owner-supplied reference edits (measured only). Tests: `tests/test_face_polish.py`.
- ✅ 2026-09-27 **Keep Nose Shape** (opt-in `nose_shape` 0-100, `--nose-shape`, GUI Skin Tone, `skin.nose_shape`): the flat, pale "painted" nose on white face paint came from skin smoothing (kept 77-85% of the nose's broad shading), not whitening (+1 L). `retouch/nose_shape.py` adds back blur(pre-smooth) - blur(smoothed) inside the grown landmark nose mask right after smoothing, so texture stays smoothed and later tone ops still apply. Alex's 2 bunny photos + simulated darker skin: 96-100% of shading kept at 100. Tests: `tests/test_nose_shape.py`.
- ✅ 2026-09-28 **Powder Finish** (opt-in `powder_finish` 0-100, `--powder-finish`, GUI next to Face Polish, `skin.powder_finish`): the old `specular_finish="powder"`/`"matte"` subtract grey sized by whole-pixel intensity, so on white paint/pale skin every lit cheek went grey (DSCF3518 face L 159->134 at 1.0; recipes use 0.12-0.24, left unchanged). `retouch/powder_finish.py` pulls L excess over a masked local skin baseline (0.2 IED, gate in units of the face's own fine-texture sigma) back toward it with a/b toward local skin colour, gives back the broad part of the change (overall brightness kept), trims bright fine sparkle. No chroma gate, so it works on face paint where shine_removal doesn't. Bunny photos at 50 (on cosplay_clear_v1 + face_polish 50): shine p99 22-28 -> 17-21, reference edits' median 24.5; simulated darker skin gets the same relative reduction. Tests: `tests/test_powder_finish.py`.
- ✅ 2026-09-27 **Nose Bridge Highlight** (opt-in `nose_highlight` 0-100, `--nose-highlight`, GUI Skin Tone, `skin.nose_highlight`): `dodge_burn`'s bridge highlight never rendered (`regions.nose_bridge` is 4 near-collinear midline landmarks: ~50-80 px at 0.2 peak; dodge_burn 100 moved the bridge 0 L) and `sculpt` darkened it. `retouch/nose_highlight.py` draws a Gaussian ridge 168->4 (sigma 0.045 IED, faded at both ends, skin-masked) and applies a linear-light gain with a shoulder after dodge_burn. Bridge-minus-sides at 100: +15/+26 L on Alex's 2 bunny photos (natural: -5/+6; 13 reference edits median +16), +13/+15 on simulated darker skin. `regions.nose_bridge` itself is unchanged (fixing it would change dodge_burn recipes). Tests: `tests/test_nose_highlight.py`.
- ✅ 2026-09-25 **XMP sidecars for picks, ratings and flags** (opt-in `--xmp` on batch and `review apply`, `./run xmp`, GUI Batch checkbox + Shoot tab button): `retouch/xmp_sidecar.py` writes Green/Red/Yellow labels, reject rating -1, stars and `retouch-*` keywords to `<stem>.xmp` beside sources and embedded in JPEG outputs. Merges existing sidecars; never overwrites a rating/label set elsewhere (`retouch:Managed` records what it wrote). Tests: `tests/test_xmp_sidecar.py`.
- ✅ 2026-09-27 **closed-eye flag checked on glasses, darker skin, bursts**: stills and glasses clips misfire-free (Yale glasses 16/16 open, sleepy 14/15 closed; simulated darker skin agrees 93.5-100% per clip; opaque sunglasses read open, never closed). Two burst misfires fixed in `shoot_intelligence.rank_burst_candidates`: relative blink only below 0.25 (`BLINK_RELATIVE_CEILING`; a wide-eyed frame made open eyes a blink), and faces tracked by bbox centre so two people swapping the bigger face are never compared. `docs/plans/RESEARCH_CLOSED_EYE_CHECK_2026_09_27.md`, tests in `tests/test_shoot_intelligence.py`.
- ✅ 2026-09-27 **over-smoothing check (patch-scale QA)**: whole-face averages hid one waxy cheek (old checks 0/18 planted) and a cast on part of a face. `plastic_skin` and `color_drift` now also scan cheek-sized windows per face (texture: σ 5% of face, faces >= 200 px, nose skipped, < 0.60; colour: a/b shift beyond the face median, > 6 ΔE, above the strongest recipe blush) and name the spot in the review message. Planted σ1.5 cheek blur 12/12; 0 new colour flags on 127 renders; heavy anime/porcelain looks can flag Waxy skin. `docs/plans/RESEARCH_OVER_SMOOTHING_CHECK_2026_09_27.md`, `tests/test_over_smoothing_check.py`.
- ✅ 2026-09-28 **recipe split tone and fade toe only reached skin (P1)**: `_stage_grade` passed the skin mask as the *apply* mask of the recipe three-way split tone, the preset split tone and `fade_toe`, so tints never reached hair/costume/background and fade lifted almost no blacks. Now whole-frame; skin gets what it got before. Changes 59 of 146 recipes outside the skin (Alex approved). cosplay_clear_v1 + `--fade-toe 40`: darkest 1% L 0-0.5 -> 3.2-3.5 (13 reference edits: 4.0). Recipe LUT export now includes them. Tests: `tests/test_grade_mask_scope.py`.
- ✅ 2026-09-28 **Skin Warmth** (opt-in `skin_warmth` 0-100, `--skin-warmth`, GUI next to Powder Finish, `skin.warmth`): `retouch/skin_warmth.py` runs after the grade, per face: hue pulled into 30-42 deg CIELab, chroma floor 0.21 x L* (so darker skin gets no chroma push), one a*/b* offset per face added to its skin and to same-person pixels matching its colour (neck/body follow, blush kept). Off for face paint (hue outside -5..75 deg or C/L < 0.05-0.08). cosplay_clear_v1 on Alex's 2 natural-skin photos: hue 5/12 -> 17/20 at 50, 25/27 at 100 (references 32); simulated darker (same hue) 6 -> 15/14 -> 25/22; white-paint bunny faces and already-warm skin unchanged. A painted face leaves its body skin alone too. Tests: `tests/test_skin_warmth.py`.
- ✅ 2026-09-28 **Blown Highlight Repair** (opt-in `highlight_repair` 0-100, `--highlight-repair`, GUI next to Powder Finish, `skin.highlight_repair`): `retouch/highlight_repair.py` runs first in the per-face core: face skin whose brightest channel is at the 8-bit ceiling (the only absolute level: it is the encoding's, not the skin's) is rolled off below white by a monotone curve against the headroom, capped at 3x the skin's own diffuse luminance (tone-relative); the core takes colour from the unclipped skin around it blended toward the local skin baseline, plus donor-patch texture. Skipped: spots not ringed by skin (props), features + margin, painted faces (skin C/L* < 0.06 or hue outside -5..80 deg; the bunny photos read -26/-35 deg), faces > 35% blown. All results simulated (Alex's photos have no blown skin): natural skin (e8f0a230 + flash lobes) C* of blown pixels 6 -> 10 at 60; simulated darker skin brought to its own tone but large lobes stay a paler disc; painted bunny bit-identical. Tests: `tests/test_highlight_repair.py`.
- ✅ 2026-10-03 **Body Skin Evening** (opt-in `body_skin_even` 0-100, `--body-skin-even`, GUI Body Skin, `body_skin.even`): the hidden `body_equalize` (16 recipes) runs CLAHE on L*, so at 100 it made body blotches ~10% and fine texture ~25% *stronger* on Alex's bunny photos; left as is (changing it changes those recipes). `retouch/body_skin_even.py` pulls mid-scale chromaticity (a*/L*, b*/L*) blotches (0.04-0.6 face width) toward the skin around them, L* only where colour also deviates; texture, moles, shading and gloss (lighter + paler = sheen, kept) untouched; white paint off. Mask: multiclass body-skin class, guided-filter snapped, grown into connected same-chromaticity skin. Does not trigger body blemish removal. Planted redness on the 2 photos: 27-31%/54-62% of its colour gone at 50/100, moles kept 96-100%, fine texture unchanged; simulated darker skin 41% at 100 except where the segmenter drops the limb (DSCF3503 thigh, 22%). Op ~5 s at 26 MP in the cloud sandbox. Tests: `tests/test_body_skin_even.py`.
- ✅ 2026-09-30 **Wig Shine (hair deglare v2)**: `hair_deglare` (`--hair-deglare`, `hair.deglare`) was a hidden GUI state and inert on real wigs (max 0.5 L on the blonde bunny wig, 2.2 L on the white wig at 60: its low-chroma gate misses gloss that keeps fibre colour, and its whole-wig median made the lit side look like glare). `retouch/wig_shine.py` measures the strand-averaged excess over a local two-pass wig baseline (gate unit = that excess's own spread, floor 2x fine strand spread) and removes it as neutral linear light, so fibre colour returns; skips pixels far less saturated than the fibre around them (white bows), a silhouette against a brighter backdrop, low coherence and brows. Now the "Wig Shine" slider (Structure & Effects). auto_clean_v1's `hair.deglare` 30 -> 0 so that recipe looks as before. Alex's photos have little real gloss: most results are simulated (gloss band on the bunny wig: lift 19.9 -> 13.6 L at 100, fibre chroma 7.5 -> 9.0, strand texture kept; simulated dark wig same share). Per-face ROI only: far ends of long wigs untouched. Tests: `tests/test_wig_shine.py::TestWigShineMatte`.
- ✅ 2026-09-30 **Bloodshot Eye Whites (sclera vessel removal v2)**: `eye_sclera_vessel_remove` (`--eye-sclera-vessel-remove`, `eyes.sclera_vessel_remove`) was a hidden GUI state that did nothing at 20 on the bunny photos and at 60 painted up to 27 dE patches over the red contacts' rim and the waterline (its landmark sclera mask holds both); planted redness 20-25% removed. `retouch/bloodshot_eyes.py`: per eye, support = sclera mask off its rim, outside the *visible* iris (brightest angular sector of rings 1.2-2.0x the landmark iris, so a wider circle lens stays out), L / own-white ratio 0.62-0.80; a* pulled to a natural white (own p20 or a cap that follows the light via b*: the bunny whites read a* +5..+9 at b* -8 and stay), vessels (1-3 noise units over a masked local baseline) also get b*/darkening back, uniform lift <= 3.5% capped at own p99. Eye-visibility gate now applies to it. Now the "Bloodshot Eye Whites" slider (Eyes & Lips); apex_editorial_v1's 0.20 -> 0 so it looks as before. Planted vessels + pink on the 2 bunny photos, full-res render through the engine: 50-58% of the planted redness gone at 100 (26-29% at 50; the op alone 73-78%), vessels on the lid rim stay; unplanted eyes change only inside the sclera mask (mean a* -1.8, L +2), contacts/lashes/catchlights kept; darker exposure (linear gain 0.45, simulated) keeps 84-91% of the effect. No real bloodshot photo tested; red stage light gets calmed too. Tests: `tests/test_bloodshot_eyes.py`.
- ✅ 2026-10-01 **Neck Tone Match** (opt-in `neck_tone_match` 0-100, `--neck-tone-match`, GUI next to Skin Warmth, `skin.neck_tone_match`): main had a hidden auto pass (`skin.harmonize_neck`, fires whenever whiten/equalize/hue_unify/chroma_even/redness_even > 0, inside the face crop: 2-6 levels on the bunny photos, and with strong whitening it paints a hard-edged bright rectangle over wig and chest — left as is, changing it changes recipes). `retouch/neck_tone_match.py` runs in the global phase after the body-skin stages: same-person neck/chest skin keyed on its own chromaticity (a*/L*, b*/L*, mode near the face colour; hair via the full-frame hair mask) gets the face edits' own brightness change as one linear-light gain (chin shadow kept) and their a*/b* change, plus 60% of a pre-existing foundation colour gap; painted faces and body-painted necks skipped. Slider on replaces the auto pass. Alex's photos are white face paint (left alone); natural faces simulated by recolouring the paint: whiten 100 + Face Polish 100 gap L +14..+20 -> +1..+8 at 100, simulated darker (x0.45) and lighter (x1.5) alike. Pale fabric next to skin can pick up some change. Tests: `tests/test_neck_tone_match.py`.
- ✅ 2026-10-01 **Costume Clarity** (opt-in `costume_clarity` 0-100, `--costume-clarity`, GUI Structure & Effects, `fabric.costume_clarity`): `retouch/costume_clarity.py`, global stage after tone, before grading. Costume = person minus skin (segmenter skin classes only where the colour could be this person's skin, so a black glove labelled arm still counts), hair/wig masks, face masks + ellipses, painted skin; skin-coloured pixels inside it (skin through fishnet) kept. L*-only fine + guided-filter clarity bands, soft-limited at their own spread, cored below the frame's own noise. Bunny photos: gloves, ears, corset, fishnet, bows sharpen; face, wig, chest and thigh skin unchanged. Darker/lighter skin simulated only. Tests: `tests/test_costume_clarity.py`.
- ✅ 2026-10-02 **Wig Lace Blend rebuilt (hairline blend)**: `cosplay_wig_lace_blend` (`--cosplay-wig-lace-blend`, GUI Cosplay Moat, `cosplay.wig_lace_blend`) called `WigLaceBlender`, a guided filter with eps 1.0 on 0-255 data: at most 1 level of change on real wig photos. `retouch/wig_hairline.py`: per face, the hair-mask edge (re-labelled by local colour, so a coarse mask snaps to the real edge) counts as hairline only inside a forehead zone above the brows, with the hair on the far side from the face centre, face-connected skin past the band and a real colour step; a lace/glue band just outside is toned to the forehead, then the step is feathered (mid frequencies only, texture kept). Full bangs, side locks and wig edges against walls: untouched (Alex's 4 wig photos bit-identical at 100). None of the 17 cosplay photos at hand shows a hairline, so results are a public-domain portrait with a planted hard front + lace and synthetic scenes (simulated darker/lighter: lace 50-62% gone at 100, edge 10-90% width 4.5 -> 11 px at 190 px face width). The segmenter misses some bright coloured wigs (a planted lavender wig got no hair mask), and then it does nothing. 2 recipes' 0.30/0.42 -> 0. Tests: `tests/test_wig_hairline.py`.
- ✅ 2026-10-04 **Defringe (purple fringing v2)**: `purple_fringing` (`--purple-fringing`, `finish.purple_fringing`) was a hidden GUI state whose `utils.remove_purple_fringing` only took magenta pixels with a fixed 3x3 gradient > 35, so on Alex's white-wig/window photos it moved fringe chroma 21.3 -> 20.0. `retouch/defringe.py` (run in `_stage_finish`, float path, untouched pixels bit-identical): band = local max within 0.22% of the long side reaches the photo's own p99.5 L* and spans >= 25-40% of it; hues green..blue..purple (red/orange/yellow/skin never pass); colour replaced by the object's own colour beside the band (support-weighted blur per lightness bin, interpolated by L*, band pixels deep from the bright side count as support). Now the "Defringe" slider (Structure & Effects); no recipe set it. At 100: fringe chroma 21-24 -> 8-9 on the wig photos, bunny softbox bokeh 22 -> 7; real light-blue/navy edges kept; 0 skin-hue pixels changed; simulated 0.45 exposure same share removed. Gap: purple/blue/green straps or strands thinner than ~2 bands (about 28 px at 26 MP) against a window lose their colour. Tests: `tests/test_defringe.py`.
- ✅ 2026-10-05 **Stray Hair Cleanup (flyaway removal v2)**: `hair_remove_flyaways` (`--hair-remove-flyaways`, `hair.remove_flyaways`; `flyaway_cleanup` is a max-combined alias) was a hidden GUI state that removed no real strands on the bunny photos and at 35-100 changed catchlights, pupils and costume trim (it searched the wig itself, normalised against the band's strongest line, and Telea-inpainted at 2% face width). `retouch/stray_hair.py` searches face + neck skin only: Hessian ridge z on L and on the projection toward the hair colour (units = the skin's own spread), integrated along the local direction (matched line filter, 0.05 fw), step-edge ratio gate; components must be long (>= 0.045 fw), thin, fairly straight, have skin colour on both sides (wig outline kept) and dark ones must lean toward the hair colour (creases kept); eyes/brows (0.09 fw), nose (0.07), lips, under-eye, smile lines and the face outline next to non-hair are excluded. Heal = masked fill + grain sampled along the normal. Runs after spot_heal. Now the "Stray Hair Cleanup" slider (Skin Smoothing & Texture); the 4 recipes that set it 25-35 now set 0. Planted wig-colour strands through the full-res engine at 60: DSCF3503 80% removed (12/12 over half), DSCF3518 43% (7/12; strands near the face outline/collar stay); simulated darker exposure (x0.45 linear, op level) removes the same share. Real cheek strands on both photos are faded. Gaps: shoulders/arms/chest not searched, near-skin-colour strands partly found, a long chin crease can soften. Tests: `tests/test_stray_hair.py`.
- ✅ 2026-10-06 **bright wig mask (no BiSeNet)**: the multiclass segmenter calls white, pastel and bright wigs "clothes" with no hair pixel to grow from (planted lavender/white/pink/blue/mint wigs on the bunny photos: 0% found), so Wig Shine, hairline blend and the hair exclusions skipped them. `parsing._seedless_wig_hair` runs only when segmenter hair covers < 30% of the ring round the upper face: dominant chromaticity (a*/L*, b*/L*) of the non-skin pixels in that ring + bangs, kept only if its strands are coherent (fine structure tensor; rejects hats/headpieces/gowns), grown over connected same-colour hair/clothes/accessory pixels plus washed-out highlights. Planted wigs: 0-41% -> 93-98% recall (red 84%, 76%), precision 0.93-1.00, same at 0.45x exposure; bunny photos and 10 real wig reference photos bit-identical, 3 real missed white/coloured wigs gain a mask. Gaps: a touching white collar joins a white wig; clipped (blown) pastel wigs lose their blown parts. Tests: `tests/test_wig_hair_growth.py::TestSeedlessWig`.
- ✅ 2026-10-07 **Lint & Dust Cleanup (backdrop cleanup v2)**: hidden `backdrop_cleanup` (`--backdrop-cleanup`, `background.backdrop_cleanup`) Telea-inpainted every high-pass outlier outside the selfie person mask: on the bunny photos at 35 it smeared prop edges the mask missed (ears, wristband, glove, fishnet), never reached the costume, and looped over components in Python (minutes at 26 MP). `retouch/lint_dust.py` (global stage `LintDustStage` before tone): region = frame minus colour-gated skin classes, hair, paint, face pipeline masks, face ellipses; top-hats at 2.25% face width, contrast in units of the local top-hat RMS (floor = frame noise), surround range, 2x-element isolation, in-focus slope, texture count, gloss count + nearby sheen (light specks on vinyl/latex = glints), busy-surround and ring-evenness tests; light specks only on the subject (backdrop: lights), soft dark sensor-dust pass on the lighter half of the backdrop. Heal = per-channel open/close surface + donor grain, 1.5 px feather; other pixels bit-identical. Now the "Lint & Dust Cleanup" slider (Structure & Effects); auto_clean_v1's 35 -> 0. Planted lint (light on dark subject, dark on light areas), op on full-res: 6/24 + 10/26 more than half gone at 50 and 100, same at 0.45x exposure; unplanted bunny photos: 0 glints/studs/edges touched, a red hot speck and 2-3 soft dust spots on the light box healed. Gap: light lint on glossy vinyl (most of these costumes) is left; ~30-45 s at 26 MP in the cloud sandbox. Tests: `tests/test_lint_dust.py`.
---

## Code Style & Conventions

### Imports & Structure
- No circular imports (broken via `style_transfer.py` re-export leaf module)
- All public functions/classes documented with docstrings
- Private methods prefixed with `_`
- No `except: pass` in render path (all logged via `_logger` or `log_crash()`)

### Naming
- `ProcessingContext` = typed parameter bag (replaces raw dicts)
- `ProcessingResult` = ndarray subclass carrying image + metadata
- `*Processor`, `*Enhancer`, `*Analyzer` = stage/component classes
- `_stage_*` = pipeline stage methods
- `_process_*`, `_apply_*` = internal helpers

### Testing
- Test files mirror source structure: `tests/test_<module>.py`
- Use `pytest.mark.skipif(platform_condition, reason="...")` for platform-gated tests
- Parameterize tests with `@pytest.mark.parametrize`
- Mock external models with fixtures (don't download at runtime)

### Performance
- Avoid `**kwargs` in hot loops (dataclass fields instead)
- Pre-allocate large arrays
- Vectorize with NumPy (no Python loops over pixels)
- Cache pre-computed LUTs/matrices at class level
- Use `cv2.LUT` for fast 1D LUT application
- Multi-face dispatch: `RetouchEngine`'s persistent `FaceProcessorPool` (a `ProcessPoolExecutor`, max 4 workers, `perf_optimizations.py`) is the primary path; `ThreadPoolExecutor` is only the fallback if that pool raises. `cli.py --workers>1` nests this pool inside its own outer `ProcessPoolExecutor` — see the batch-hang entry under Outstanding Fixes above before changing either pool's shutdown/daemon behavior.

### Editing Conventions
- When editing multiple files/locations that share identical comment or value patterns (e.g., repeated recipe literals across presets), anchor `old_string` on unique surrounding context (preceding key, function name, etc.) rather than the shared snippet alone — an ambiguous match fails instead of silently editing the wrong occurrence.

### Tone-Invariance & Fairness
- **No Absolute Intensity Thresholds:** Do not use hardcoded absolute intensity, luminance, or reflectance threshold checks (e.g., `I > 170.0`, `L > 0.85`) on signals that scale with the subject's skin tone. These absolute gates systematically degrade or fail on darker skin tones (Fitzpatrick V-VI).
- **Tone-Adaptive Alternatives:** Use margin-above-baseline measures relative to each face's own diffuse baseline (e.g., computed via median over the skin mask or crop). See `retouch/specular.py::extract_specular` for a reference implementation.

---

## Batch Processing
- Batch jobs are long (hundreds of 26MP photos). Report progress counts periodically (e.g. 120/327) so I know it isn't stuck.
- Known issues: false-completion and single-file hangs. Always verify output count == input count and detect/skip hung files with a per-file timeout.
- Output layout: final/ and compare/ subfolders, then sync to Google Drive.
- Loosely named folders (e.g. 'cosmic day3') should be resolved with `find` and confirmed before running.
- Another Claude session may be editing the same files. Check `git status` before committing and don't commit files you didn't change.

---

## Git Workflow

### Commit Message Format
```
<type>(<scope>): <subject>

<body (optional)>

<footer (optional)>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`  
**Scopes:** `engine`, `grading`, `parsing`, `gui`, `cli`, `tests`, etc.

**Example:**
```
fix(engine): replace white_balance hardcoded literals with _DEFAULTS

Use _DEFAULTS["white_balance_kelvin"] instead of hardcoded 6500
to maintain single-source-of-truth with params.py registry.

Fixes #142
```

### Branch Protection
- `main` is linear (no merge commits)
- All commits must pass syntax + test suite
- Tag releases as `v<major>.<minor>.<patch>-<codename>`

### Commit Message Delivery
When committing, avoid heredocs with nested backticks, unescaped quotes, or apostrophes — they break shell parsing mid-commit. Write the message to a temp file and use `git commit -F <file>` instead.

---

## Debugging, Docs Map & Contributing

- **Crash logs & common issues (detection failures, slow/OOM, GUI/CLI errors):** `docs/guides/TROUBLESHOOTING.md` — do not duplicate here.
- **Full documentation map** (README, ARCHITECTURE, API, recipe/GUI/batch guides): `docs/INDEX.md`
- **New contributor checklist** (read order, test suite, pre-push checks): `docs/CONTRIBUTING.md`

---

## Contact & Support

- **Issues:** https://github.com/dennisbrian/retouch/issues
- **Discussions:** https://github.com/dennisbrian/retouch/discussions
- **Security:** Contact maintainers privately (no public disclosures)

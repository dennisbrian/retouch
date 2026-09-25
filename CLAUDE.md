# Claude.md — Pro Max Face Retouch Engine

**Project:** Professional automated face retouching pipeline  
**Repository:** https://github.com/dennisbrian/retouch  
**Status:** Mature (v2.0.0 — Fuji-quality color recipe system)  
**Last Updated:** 2026-09-24

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

**~71k LOC (retouch/*.py + gui.py/gui_advanced.py/gui_shoot.py/gui_batch.py + cli.py), 119 modules in retouch/, 4,555 tests collected (pytest --collect-only, 2026-08-26).**

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
python3 -m pytest tests/ -q

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
- **4,555 tests collected** (`pytest --collect-only`, 2026-08-26) — full-suite pass/fail baseline not re-run at this count
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

- ✅ 2026-09-25 **glasses / goggle / visor glare removal** (opt-in `lens_glare` 0-100, `--lens-glare`, GUI Eyes & Lips): `retouch/lens_glare.py` subtracts an additive low-frequency veil measured above a closing+opening local reference (tone-invariant margin), runs before reshape. Planted glare on real faces: 40-70% lifted; glare wholly inside the eye opening is kept. Tests: `tests/test_lens_glare.py`.
- ✅ 2026-09-25 **XMP sidecars for picks, ratings and flags** (opt-in `--xmp` on batch and `review apply`, `./run xmp`, GUI Batch checkbox + Shoot tab button): `retouch/xmp_sidecar.py` writes Green/Red/Yellow labels, reject rating -1, stars and `retouch-*` keywords to `<stem>.xmp` beside sources and embedded in JPEG outputs. Merges existing sidecars; never overwrites a rating/label set elsewhere (`retouch:Managed` records what it wrote). Tests: `tests/test_xmp_sidecar.py`.
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

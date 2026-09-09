# Claude.md — Pro Max Face Retouch Engine

**Project:** Professional automated face retouching pipeline  
**Repository:** https://github.com/dennisbrian/retouch  
**Status:** Mature (v2.0.0 — Fuji-quality color recipe system)  
**Last Updated:** 2026-08-26

---

## Quick Context

This is a **production-grade image processing engine** that applies professional retouching to portraits via a modular 7-stage pipeline. It combines:
- MediaPipe face detection + 478-point landmarks
- BiSeNet ONNX semantic segmentation (skin, lips, eyes, hair, etc.)
- Frequency-separation skin smoothing + component enhancement
- Color grading + Fuji film simulation presets
- Virtual studio relighting + advanced lens effects

**~71k LOC (retouch/*.py + gui.py/gui_advanced.py/gui_shoot.py/gui_batch.py + cli.py), 112 modules in retouch/, 4,535 tests collected (pytest --collect-only, 2026-08-19).**

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

- ✅ 2026-07-03/04 batch (B&W mixer literals, E1 float32 canvas, equalize pale-face + s² non-linearity, sharpen inert, white-balance literals, `_build_dimensional_mask` shape fix) — all shipped; `git log --grep` for detail (`260fe02`, `e0b294c`, `1640eeb`, `6331994`)
- ✅ 2026-07-13 `skin.locus` recipe override wiring; test hang was a MediaPipe teardown deadlock (bare `RetouchEngine()`, not init abort) — `36cb93a`
- ✅ 2026-07-13 `_process_inputs` argument-order footgun — name-keyed dict + import-time drift guard, see [Architecture Decisions](#single-source-of-truth-retouchparamspy) — `f0b656b`, `da1f8cb`
- ✅ 2026-07-13 `_recipe_outputs` mirror-image footgun (same fix pattern; closed a live 107-vs-111 value drift) — see [Architecture Decisions](#single-source-of-truth-retouchparamspy)
- ✅ 2026-07-21 LUT hot-reload wired: GUI watcher (gui.py) + CLI `--reload-luts` flag — `7c8d607`
- ✅ 2026-09-09 static drift guard added for the 15 `reset_*` handlers in gui.py: audited all 13 dict-keyed handlers (`reset_color_transfer`/`reset_debug` take no dict) — no live order drift found today. `tests/test_gui.py::TestResetFunctions::test_reset_function_key_order_matches_click_output_order` now statically compares each handler's `d["key"]` return order against its `.click(outputs=[...])` order via AST/regex (mirrors the `TestProcessInputKeys` pattern) and fails loudly on future drift. Handlers remain hand-ordered pairs — not converted to name-keyed dicts, since the static check already closes the silent-failure mode.
- ✅ 2026-08-11 redundant `ci.yml` workflow removed — `7b97df9`
- ✅ 2026-07-14 `tests/test_cosplay_moat.py` bare-`RetouchEngine()` → module `engine` fixture (same teardown pattern as skin_locus)
- ✅ 2026-07-14 `test_ext_map_keys_match_radio_choices` — expect `PNG-16` (map already had it; test was stale)
- ✅ 2026-08-19 golden-v2 face-path harness: `test_golden_pipeline.py`'s synthetic image has no detectable face (0 faces → `_no_face_fallback`), so its `natural` snapshot was byte-identical to raw input — the entire per-face pipeline was untested. `tests/test_golden_pipeline_face.py` + `tests/golden_face_fixture.py` inject a real `FaceContext` (frozen real anatomical landmarks, ONNX-free `_landmark_fallback_only` regions) via `face_contexts=`, exercising skin/eye/lip ops for real. Mutation-verified: perturbing `SKIN_LOCI`/`equalize` strength moves the recipe hashes.
- ✅ 2026-08-26 P0/P1 eye handedness fix: BiSeNet (CelebAMask-HQ subject-anatomical labels 4/5 + 2/3) vs parsing.py's camera-viewer convention were opposite, so `left_sclera = clip(left_eye - left_iris)` subtracted two disjoint eyes — a total no-op, and sclera brightening painted the *other* eye's iris. Swap applied in `_masks_from_label_map` (single translation point; both `parse()` and `parse_batch()`); verified empirically on real BiSeNet output (DSCF4463/4503/4550: `array_equal(left_sclera, left_eye)` → False, same-side iris-inside-sclera ~0.75-0.92). Golden face hashes intentionally changed (per-eye catchlight fix below) — snapshots updated. Full detail: `docs/review/REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md` §P0/P1.
- ✅ 2026-08-26 per-eye `_enhance_catchlights` (B6 latent bug): pooling both irises into one mask coupled the eyes — gating one side shifted the pooled centroid, resurfacing a synthetic catchlight on the *visible* eye the pre-change code never drew (with two eyes present the pooled centroid lands *between* the irises and `*iris_mask` annihilates it — the synthetic fallback was effectively dead code for two-eye faces). Now called once per iris.
- ✅ 2026-08-26 eye-visibility gate shipped (EAR-primary + tone-adaptive contrast secondary): `retouch/eye_visibility.py` gates per-eye enhancement when the eyelid is closed or the iris-vs-sclera contrast collapses (only when BiSeNet eye mask ≥ 100 px — class-5 collapse on ~65% of real portraits is model fragility, NOT occlusion, and fails open). Gate hoisted into `_process_face_core` so the weight-1.0 sharpen mask sees gated regions too (B5). Memoized on `FaceRegions._eye_gate_cache` (§4 perf). ParamSpec `eye_gate` (default on) + GUI checkbox + log line on every gating decision. Mutation-tested (GATE_MUTATE=1 → 2 integration tests fail): `tests/test_eye_visibility.py` + harness in `tests/conftest.py`. **Initial thresholds (EAR < 0.12 / contrast < 0.45) were later found miscalibrated — see 2026-08-31 entry below for the corrected values; do not use this entry's original numbers.**
- ✅ 2026-08-31 eye-gate threshold recalibration: the 2026-08-26 gate's shipped thresholds failed their own acceptance criterion — `docs/plans/RESEARCH_EYE_OCCLUSION_RESULTS_2026_08_31.md` (83-image DSCF corpus, 166 eyes, both BiSeNet and landmark-fallback arms) found `_MIN_EAR=0.12` missed 16/19 (84%) genuinely closed eyes because it was calibrated against a mislabeled anchor (DSCF4454-R recorded as "plainly open" at EAR 0.136 is actually closed at 300px). True measured floor: occluded EAR range 0.101–0.283, visible floor 0.194. Shipped `_MIN_EAR=0.285` / `_MIN_CONTRAST=0.55` (exhaustive grid search over both arms, re-verified directly against the study's raw `sweep_full.json`/`labels.json` rather than trusting the doc's rounded `0.28` recommendation, which sat *below* the measured occluded ceiling and would have missed DSCF4454-L) — 0/38 occluded eyes missed, 13/244 visible eyes false-gated (5.3%) on the corpus; also verified the golden-face fixture's frozen open-eye landmarks (EAR 0.288) clear the new threshold so `test_golden_pipeline_face.py` still exercises eye ops. Verified on real renders (DSCF4576, DSCF6961, DSCF4612, DSCF4454) with the gate forced on/off. Still provisional: hard-case sourcing (hair/wig/sunglasses — corpus has zero), Fitzpatrick IV-VI stratum, and a second labeler pass remain open per that doc's §6.
- ✅ 2026-09-02 **composite-mask epsilon bug (P0)**: every mask normaliser used `max() > 1.0` to detect 0–255 masks; float32 region masks overshoot 1.0 by one ulp (1.0000002) after feathering on ~24% of real portraits (20/83 DSCF), so skin/hair/neck masks were divided by 255, composite alpha over skin collapsed to 1/255 and smoothing/equalize/blemish/under-eye/relight/sculpt were silently discarded (lips + sharpen survived, which hid it). Threshold `> 1.5` + clip at 21 sites — `f69ab1e`, `tests/test_mask_norm_epsilon.py`. **Any past "op X does nothing on image Y" conclusion may have been this.** Detail: `docs/plans/RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md` §5.
- ✅ 2026-09-02 yaw-gate recalibration: nose-bridge/temple ratio bands 1.3→1.6 (slimming/jaw/chin) and 1.5→1.7 (relight, sculpt) mapped to ~2–10° of head turn and zeroed slimming on 58/83 corpus faces. Ratio kept (Spearman 0.92 vs depth-yaw), band → shared `utils.YAW_GATE_START/END` = 2.5→4.0, validated on renders — `9c26493`. **Its shared ramp helper shipped inverted — see the yaw-ramp entry below.** Neck gate resolved below (removed, not recalibrated). Study: `docs/plans/RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md`.
- ✅ 2026-09-02 audit-finding closeouts: undereye `dark_circles` + `undereye_darken_removal` double-apply (27 recipes) aliased at dispatch, max not sum — `e73d3ba`; lips/teeth whole-image LAB roundtrips contained via `utils.restore_outside_support` — `46be031`. Eye-v0 uint8 ROI roundtrip still open; dark-circle op resolved below (v2).
- ✅ 2026-09-02 **post-epsilon re-audit** (83 faces, `docs/plans/RESEARCH_POST_EPSILON_FACEOP_REAUDIT_2026_09_02.md`, `scripts/qa/post_epsilon_faceop_sweep.py`): 19/83 faces were collapsed pre-fix; post-fix their `natural` skin delta matches the 64 controls (median 2.61 vs 2.50 levels) — default strengths are NOT over-strong, no re-tune needed. The 08-31 audit's "DSCF8007 skin column nullified by plastic-skin backoff" attribution was wrong: 8007 was an epsilon victim (post-fix smooth 1.45 vs 0.00). Remaining near-zero skin ops are all multi-face images whose largest detection is a hand/background person (4589/4592/4596/4597/4599/4600/4618) — corpus hygiene, not op defects.
- ✅ 2026-09-02 **yaw ramp inverted (P1)**: `utils.yaw_gate_factor` returned the raw smoothstep (0→1) instead of 1−smoothstep, so relight/sculpt/slimming got ~0 strength just past `YAW_GATE_START` and ~full just under `YAW_GATE_END` (corpus: relight 0.00 at ratio 2.51, 1.51 at 3.99). Pre-`9c26493` geometry had the same inversion in its private 1.3→1.6 band; relight/sculpt inherited it when they moved to the shared helper. Fixed + monotonicity test (`tests/test_utils.py::TestYawGateFactor`); midpoint-only tests could not see it.
- ✅ 2026-09-02 **neck "depth gate" was a guaranteed no-op on frontal faces**: `skin.harmonize_neck` kept pixels within 0.15·face_w of an eye-corner/nose-bridge plane measured in XY only; with z scaled by face width the plane tilts into the image and the metric is ~0.45× vertical distance below the eyes — 15/15 frontal corpus faces (ratio ≤ 1.3) lost 100% of the neck mask, turned faces skipped the gate. Gate removed; chroma gate (`max(10, 1.25·(σa+σb))`, provisional, 4 pale anchors) now applied to the BiSeNet neck path too (was painting hard patches on a collar/hand: DSCF4560/4463); LAB roundtrip contained via `restore_outside_support`. The old unit test asserted the no-op on a uniform grey image (vacuous) — replaced.
- ✅ 2026-09-02 **dark-circle op was inert on every face (v1 → v2)**: the detector kept the darkest ≥100 px component of the landmark under-eye polygon ≥15 L below its surround — on real faces that is the lower lash line (the polygon's top edge *is* the lid contour), so 80% of the lift landed on lashes and 0.03 L on skin (146 corpus eyes, strength 100); the lift was also double-capped (≤9 L). 55/128 recipes set the key; none did anything. v2 (`retouch/undereye.py`): support extended 0.28·IED into the tear trough with the eye contour + lash margin excluded (feathered edge), masked low-pass darkness *relative* to a cheek-ring median (4→12% smoothstep, tone-invariant), low-frequency lift + 75% a/b pull, ROI-confined + `restore_outside_support`; dispatch passes IED / landmark eye hull / skin. Corpus: skin lift 9.3 L median at strength 1, 0 px in eye, 0 px outside support. Golden snapshots unchanged (`natural` sets no under-eye key). **Limitation:** under-eye contour makeup (aegyo-sal) reads as shadow at strength ≥0.45 — owner call on the 3 recipes that set it that high; no darker-skin sample in corpus. Study: `docs/plans/RESEARCH_DARK_CIRCLE_OP_2026_09_02.md`.
- ⚠️ 2026-09-02 golden face hashes are interpreter-specific: bare `python3` (cv2 4.13) ≠ `.venv/bin/python` (cv2 4.11). Snapshots are pinned to `.venv`; run `RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python -m pytest …` — `efb17af`.

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
- ThreadPoolExecutor (max 4 workers) for multi-face, ProcessPoolExecutor for heavy CPU workloads

### Editing Conventions
- When editing multiple files/locations that share identical comment or value patterns (e.g., repeated recipe literals across presets), anchor `old_string` on unique surrounding context (preceding key, function name, etc.) rather than the shared snippet alone — an ambiguous match fails instead of silently editing the wrong occurrence.

### Tone-Invariance & Fairness
- **No Absolute Intensity Thresholds:** Do not use hardcoded absolute intensity, luminance, or reflectance threshold checks (e.g., `I > 170.0`, `L > 0.85`) on signals that scale with the subject's skin tone. These absolute gates systematically degrade or fail on darker skin tones (Fitzpatrick V-VI).
- **Tone-Adaptive Alternatives:** Use margin-above-baseline measures relative to each face's own diffuse baseline (e.g., computed via median over the skin mask or crop). See `retouch/specular.py::extract_specular` for a reference implementation.

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

# Codebase Audit Consolidated Plan — 2026-07-14

## Context

A 25-agent tsunami research sweep was run across the engine (9 original explorations + 16 deep
audits) to map implementation reality vs. the documented architecture. This plan is the **single
consolidated ledger** of every finding, grouped by theme, with a prioritized fix list at the end.

Two cross-cutting facts frame everything below:

1. **Every published benchmark is synthetic.** `scripts/bench/benchmark.py` mocks the engine
   boundary and uses fake images (`benchmark.py:19-21`). The "702ms/face", "7.5GB→1.84GB",
   "15.3s→3.09s", "5×" figures **exclude real MediaPipe/BiSeNet/NAFNet inference cost** entirely.
   Treat them as Python-overhead lower bounds, not model-path measurements.
2. **The default `process()` path silently skips 4 stages.** `_use_global_registry` is never
   assigned anywhere in the codebase (`grep "_use_global_registry\s*="` → 0 matches), so it always
   defaults `True` and runs the 6-stage registry. The 10-call hardcoded `else` branch
   (`engine.py:2071-2149`) — which contains `_stage_background`, `_stage_cosplay_moat`,
   `_stage_local_adjustments`, `_stage_body_reshape` — is **dead code**. Features reachable only
   from that branch never execute in production.

Severity legend: **HARD** = correctness/behavior bug; **FID** = fidelity gap vs reference math;
**ARCH** = architecture/lifecycle; **HON** = honest-naming/dead-code/misleading-default;
**PERF** = performance; **TEST** = coverage gap.

---

## 1. Silent dead code & misleading defaults (HON)

| # | Finding | Evidence | Fix leverage |
|---|---------|----------|--------------|
| HON-1 | `neural` recipe default ships at 30 but segmenter is permanently `enabled=False` → zero effect, no warning | `recipes.py:3054`, `neural_boosters.py:77,130`, `engine.py:4099/4108` | **CHEAP/START** |
| HON-2 | 4 stages unreachable in production (`_stage_background`, `_stage_cosplay_moat`, `_stage_local_adjustments`, `_stage_body_reshape`) | `engine.py:2071-2149`, `_use_global_registry` unset | HIGH/needs-design |
| HON-3 | `_stage_local_adjustments` + `local_adjustments` param exist but GUI/CLI never populate them → dead at user surface | `engine.py:3855`, `params.py` (no ParamSpec), `gui.py` (no widget) | HIGH/needs-design |
| HON-4 | `spot_heal.SpotHealer`/`LamaHealer` never wired into engine `heals` hook; only old `heal.heal_region` runs | `engine.py:1398-1409`, `spot_heal.py:862` only caller | MED |
| HON-5 | Fuji sims are hand-authored parametric dicts, NOT calibrations; `luts/fuji.cube` is a demo tint never used by any sim; `presets/provia.json` ships an inert `calibration` block | `recipes.py:1414+`, `luts/ACQUISITION.md`, `scripts/recipes/generate_demo_luts.py:251` | MED/HON |
| HON-6 | `style_ref`/"subject_aware_transfer" and "neural boosters" names overpromise — both are classical Reinhard color-match / empty stubs, no neural net exists | `style_transfer.py:88`, `neural_boosters.py` | LOW/doc |

## 2. Correctness bugs (HARD)

| # | Finding | Evidence | Fix leverage |
|---|---------|----------|--------------|
| HARD-1 | Local adjustments double-mask: ops multiply by mask internally AND `apply_local_adjustment` composites again → effective coverage `m²` not `m`; soft brushes fall off too steeply; binary-only tests hide it | `regions.py:445,534,521` vs `regions.py:634-635` | MED/needs-test |
| HARD-2 | `color_transfer` (A) has **unbounded** std-ratio (channel crush on flat source) while `style_ref` (B) clips `[0.3,3.0]` — asymmetric safety | `grading.py:1514-1515` vs `style_transfer.py:77` | **CHEAP/START** |
| HARD-3 | `style_ref` transfers only the **largest** face; other faces keep original skin tone in group shots | `style_transfer.py:126-129` | MED |
| HARD-4 | `style_ref` skips clothing/eyes/lips/teeth → uncanny leftover; no face-preservation guard | `style_transfer.py:139-152` | MED |
| HARD-5 | LaMa tiling `weight` is dead code → hard tile seams at 1024px boundaries | `spot_heal.py:311,329,328` | MED |
| HARD-6 | spot_heal binarizes mask at `>=32` twice → faint strokes silently vanish; no soft-coverage inpaint | `spot_heal.py:480,483` | MED |
| HARD-7 | `color_transfer` + `style_ref` can both fire in one `process()` with contradictory channel policy (A: L untouched; B: L matched) | `engine.py:3594` then `3601` | LOW/doc |

## 3. Fidelity gaps vs reference math (FID)

| # | Feature | Gap | Evidence | Leverage |
|---|---------|-----|----------|----------|
| FID-1 | White balance (R7) | Kelvin→hue uses hardcoded `0.3`/`100.0` fudge; operates in gamma-encoded LAB, not linear RGB gain | `grading.py:1432`, `color_space.py:30-42` | HIGH/research |
| FID-2 | Fuji sims | No per-film color matrix / curve shape — only strength-scaled sigmoid + scalar grade | `engine.py:2438`, `tonal._sigmoid` | HIGH/research |
| FID-3 | BiSeNet | Custom unverified 19-class map; hard argmax discards probabilities; nearest-neighbor upscale for large faces | `parsing.py:211-215,222-237` | HIGH/test |
| FID-4 | Frequency sep (R9) | Band cutoffs implicit (sigma=0 auto-σ), coupled to face_width only; "low" retains mid-freq | `frequency.py:582-586` | LOW/acceptable |
| FID-5 | Skin chromophore | Makeup-vs-skin unmixing not invertible; RGB rank-deficient for 3 chromophores | `skin_chromophore.py`, `PLAN_P4_MAKEUP_UNMIX.md` | HIGH/research |
| FID-6 | Eye/teeth/lips | No shared light-direction model (E-COMMON-1 unmet); flat 2D ops on round features | `PLAN_FEATURE_FRONTIER.md` | HIGH/research |
| FID-7 | Body reshape | 2D warp on 3D pose → folds/self-intersections at extreme sliders; no diffeomorphic constraint | `geometry.py:33`, `body_reshape.py` | HIGH/research |

## 4. Architecture / lifecycle (ARCH)

| # | Finding | Evidence | Leverage |
|---|---------|----------|----------|
| ARCH-1 | MediaPipe force-pinned to CPU (`_resolve_delegate` returns `CPU` unconditionally) → GPU detection dead code; CoreML idles on Apple Silicon | `detection.py:304-306`, `engine.py` GPU try/except unreachable | **CHEAP/START** (perf) |
| ARCH-2 | No per-model provider denylist — `build_ort_providers()` feeds CoreML to BiSeNet/ESRGAN though CoreML is known-broken for NAFNet | `perf_optimizations.py:1113`, `enhance.py:155-160` | MED |
| ARCH-3 | `RetouchEngine` non-picklable → all model inference parent-bound; multi-face parallelism only helps cheap NumPy retouch | `detection.py`/`parsing.py` instance attrs | HIGH/research |
| ARCH-4 | `FaceParser._sess` (BiSeNet) loaded eagerly at `RetouchEngine()` even for no-face path; no `close()`/release | `engine.py:795-817`, `parsing.py:148` | MED |
| ARCH-5 | `retinaface_mv1.onnx` dead in `models/` (pip `retinaface` used instead) | `detection.py:169`, `ARCHITECTURE.md:409` | LOW |
| ARCH-6 | Stage registry covers only global phases 3–6; stages 0–2 + 4 unwrapped sub-stages not migrated (deferred, not blocked) | `stage_wrappers.py:153-158`, `stages.py:7-11` | MED |

## 5. Performance (PERF)

| # | Finding | Evidence | Leverage |
|---|---------|----------|----------|
| PERF-1 | Per-face IPC pickles 20 full-res `FaceRegions` masks/face (~335MB@2048px) instead of ROI-cropped | `engine.py:2721-2734` | MED |
| PERF-2 | BiSeNet post-processing allocates native `full_label_map` + 12 masks per face (full-res) | `parsing.py:218-247` | MED |
| PERF-3 | ~30 full-ROI float32 passes/face with redundant uint8↔float32 round-trips inside `_process_face_core` | `perf_optimizations.py:669,705,755,768,907` | MED |
| PERF-4 | LUT 3D trilinear materializes ~16×H×W×3 transient (4K → ~1.3GB) | `lut.py:132-185` | MED |
| PERF-5 | FaceContext cache discarded at proxy boundary for default high-res path → re-runs detection+BiSeNet every call | `engine.py:1791-1798` | MED |
| PERF-6 | `style_ref` re-runs ref detect+segment+parse per target image in a batch (no memoization) | `style_transfer.py:111-136` | **CHEAP/START** |

## 6. Test coverage gaps (TEST)

| # | Gap | Evidence | Leverage |
|---|-----|----------|----------|
| TEST-1 | BiSeNet real-model path 0% covered; only `max()<0.01` gate; class map unguarded | `test_parsing_fallback.py` (MagicMock), `SESSION_PROGRESS_2026-06-23.md:128` | **CHEAP/START** (guard test) |
| TEST-2 | Local adjustments: no fractional-mask test → HARD-1 invisible | `test_local_adjustments.py:44-52` | **CHEAP/START** |
| TEST-3 | Color transfer: only identity/smoke; no known-offset recovery; no std-ratio clamp assertion | `test_grading.py:147-154`, `test_style_analyzer.py:100` | **CHEAP/START** |
| TEST-4 | Style transfer: no multi-face, no clothing-leftover, no seam test | `test_style_analyzer.py:240` | MED |
| TEST-5 | No real-model benchmark harness (all mocked) | `benchmark.py:19-21`, `conftest.py:142-148` | MED |
| TEST-6 | LaMa real-model path unverified (model absent → only fallback) | `test_lama_heal.py` | LOW |

---

## Prioritized Fix Ledger

### TIER A — Cheap, high-leverage, start now (zero/low regression risk)
- **A1. Surface silent-dead neural boosters.** In `_stage_neural_boosters`, emit
  `logger.warning` when `strength>0` but `segmenter.enabled is False` (once per session, not
  per-image). Stops shipping a default that does nothing silently. (HON-1)
- **A2. Symmetric std-ratio clamp in `color_transfer`.** Apply `np.clip(ratio, 0.3, 3.0)` +
  low-variance skip to `grading.py:1514-1515` to match `style_transfer.py:77`. Prevents channel
  crush. (HARD-2)
- **A3. Enable MediaPipe GPU delegate when available.** Make `_resolve_delegate` return GPU when
  constructible; keep CPU fallback live. Resolves dead GPU path on Apple Silicon. (ARCH-1)
- **A4. Memoize `style_ref` inference.** Cache `ref_faces/ref_person/r_regions` by `id(ref_img)`
  inside `subject_aware_transfer`. Eliminates O(N) redundant inference per batch. (PERF-6)
- **A5. BiSeNet real-model regression test.** Add a test that loads `models/resnet18.onnx`, runs
  on a bundled annotated face, asserts `logits.shape[-3]==19` and class-1 (skin) IoU vs committed
  ground truth. Guarded by model presence. (TEST-1)
- **A6. Fractional-mask regression test for local adjustments.** Add `m=0.5` gradient mask test
  exposing HARD-1 (double-mask squaring). (TEST-2)
- **A7. Color-transfer recovery + clamp test.** Known a/b offset → assert recovery within clamp
  bounds; assert skip-on-flat-source. (TEST-3)

### TIER B — Medium effort, needs design/test, high value
- **B1. Fix double-mask (HARD-1):** pick ONE masking site in `regions.py` (strip internal `m`
  multiply OR drop composite at `:634-635`); ship behind A6.
- **B2. Resolve dead-stage branch (HON-2):** either migrate the 4 stages into `stage_wrappers`
  or delete the dead `else` branch + guardrail. Decouples misleading docs from reality.
- **B3. Wire local-adjustments + spot-heal brush UI (HON-3/HON-4):** add `gr.Image(tool="sketch")`
  + `gr.State`, register under `_process_input_components` (satisfies import-time drift guard),
  build dict, inject into `engine_kwargs`. Reuse `spot_heal` float32 mask contract.
- **B4. LaMa tile-seam fix (HARD-5):** implement the promised feathered `weight` blend or delete.
- **B5. Per-model provider denylist (ARCH-2).**
- **B6. ROI-crop `FaceRegions` before IPC pickling (PERF-1).**
- **B7. `style_ref` multi-face + region coverage (HARD-3/HARD-4).**

### TIER C — Research-only, needs data/calibration (not code-now)
- **C1. White-balance Planckian locus in linear RGB (FID-1).**
- **C2. Fuji per-film color matrix + curve-shape presets; de-duplicate `RECIPES` vs `presets/*.json`
  (FID-2 / HON-5).**
- **C3. Keep soft BiSeNet logits / guided-feather masks (FID-3).**
- **C4. Skin chromophore invertibility / makeup unmix (FID-5).**
- **C5. Optical-anatomical eye/teeth/lips models + shared light dir (FID-6).**
- **C6. Diffeomorphic body reshape (FID-7).**
- **C7. Real calibration harness for Fuji + benchmark against live models (TEST-5).**

---

## Execution order (this session)

1. Write this plan (done).
2. **TIER A — DONE.** A1 (neural booster warning), A2 (clamp symmetry), A3 (GPU delegate, made
   opt-in via `RETUCH_GPU=1` — default-on risks an uncatchable MediaPipe GPU-init hang on some
   platforms), A4 (`style_ref` inference memoization), A5 (BiSeNet real-model test),
   A6 (soft-mask linearity test + B1 double-mask fix), A7 (color-transfer recovery/clamp tests).
   All 75 targeted tests pass; 4 skip (ops with no measurable flat-image effect).

### Notes / deviations
- **A3 deviation:** the plan said "enable GPU delegate when available" (default-on). Verification
  showed MediaPipe's GPU delegate constructor can **hang** (not raise) on some platforms, so the
  `__init__` try/except can't catch it. Implemented as opt-in via `RETUCH_GPU=1` instead. The GPU
  path is now reachable when explicitly requested, instead of dead code.
- **B1 pulled into this session:** the double-mask fix (HARD-1) was required for A6 to pass, so it
  shipped alongside A6. All local-adjustment ops were confirmed to mask internally, so dropping the
  redundant composite is behavior-correct (binary masks unchanged; soft brushes now linear).

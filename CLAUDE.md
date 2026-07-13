# Claude.md — Pro Max Face Retouch Engine

**Project:** Professional automated face retouching pipeline  
**Repository:** https://github.com/dennisbrian/retouch  
**Status:** Mature (v2.0.0 — Fuji-quality color recipe system)  
**Last Updated:** 2026-07-13

---

## Quick Context

This is a **production-grade image processing engine** that applies professional retouching to portraits via a modular 7-stage pipeline. It combines:
- MediaPipe face detection + 478-point landmarks
- BiSeNet ONNX semantic segmentation (skin, lips, eyes, hair, etc.)
- Frequency-separation skin smoothing + component enhancement
- Color grading + Fuji film simulation presets
- Virtual studio relighting + advanced lens effects

**~42.5k LOC (retouch/*.py + gui.py + cli.py), 70 modules in retouch/, 3,630 tests collected (pytest --collect-only, 2026-07-13).**

---

## For Claude Code Users

### Do NOT Spawn Agents for File Reads
- **NO:** `/code-review ultra`, `/verify`, Explore agent for simple file exploration
- **YES:** Use native Haiku tools (Read, Bash grep, find)
- **Why:** Budget-conscious — each agent spawn costs tokens. Stay cheap.

### Model Strategy
- **Default:** Haiku (cheapest, capable for most tasks)
- **Fallback:** Fable → Sonnet 5 → Opus (only if Haiku can't handle it)
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
Every tunable parameter is registered in `PROCESSING_PARAMS` (a list of `ParamSpec` objects). This eliminates the "7-way sync" problem — GUI scales, CLI args, engine defaults, and recipes all auto-generate from this one list.

**When adding a new parameter:**
1. Add one `ParamSpec` entry to `PROCESSING_PARAMS` in `params.py`
2. The **CLI flag and engine defaults auto-wire** from the spec (argparse and
   `ProcessingContext` derive from the registry).
3. The **GUI slider does NOT fully auto-wire.** The new name is inserted into
   `gui.PROCESS_INPUT_KEYS` automatically (it comes from `param_names()`), but
   the Gradio component that supplies its value must be added by hand to the
   `_process_input_components` dict in `gui.py` — either a real slider/dropdown
   or, for not-yet-visible params, a `gr.State(...)` placeholder named
   `_<name>_state`. Add it under the correct name key; **order is handled for
   you** because the positional list is built as
   `[components[k] for k in PROCESS_INPUT_KEYS]`.
 4. The `_process_inputs` de-footgun refactor **landed 2026-07-13**: `_process_inputs`
    is now derived as `[_process_input_components[k] for k in PROCESS_INPUT_KEYS]`
    from a name→component dict, with an import-time assertion that the dict's
    keys are exactly `PROCESS_INPUT_KEYS`. Forgetting the GUI entry now makes
    importing `gui` **fail loudly** with an `AssertionError` naming the missing
    or extra key, instead of silently shifting every later slider's value into
    the wrong parameter. (Per-position parity tests live in
    `tests/test_gui.py::TestProcessInputKeys`.)

> Note: a second hand-ordered list, `_recipe_outputs` in `gui.py` (recipe/reset
> handlers), is *not* covered by this guard yet — keep it in sync manually when
> adding **visible** sliders.

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
If input > 2048px:
1. Downscale to 2048px
2. Run full pipeline at proxy
3. Upscale result + masks back to original

Cuts memory from 7.5 GB → 1.84 GB, runtime from 15.3s → 3.09s.

---

## Testing & Quality

### Test Coverage
- **3,630 tests collected** (`pytest --collect-only`, 2026-07-13) — full-suite pass/fail baseline not re-run at this count
- **89% coverage** on core modules (per `pytest --cov`; last measured 2026-07-01)
- **Unit + integration tests** for every public API
- **Deep algorithmic verification** (monotonicity checks, round-trip stability, etc.)

### Audit Status
- **Security:** No hardcoded secrets, API keys, or absolute paths
- **Performance:** Benchmarked at 702ms per-face (400×400), 6.9ms for no-face global-only
- **Process:** Weekly syntax checks, monthly algorithmic re-audit (`docs/review/AUDIT_REPORT.md`)

### Known Limitations
- Remaining ~4% undetected faces: extreme profiles, heavy occlusion, tiny faces in distance shots
- LUT hot-reload (daemon thread watcher) exists but not wired into GUI/CLI yet

---

## Outstanding Fixes

### ✅ RESOLVED 2026-07-03: Registry-Bypass Bug — B&W channel mixer _DEFAULTS (was engine.py:1212-1216, 1802-1806)
**Issue:** `bw_active` checked `ctx.bw_channel_mixer_r != 30 / != 59 / != 11` using hardcoded literals instead of `_DEFAULTS["bw_channel_mixer_*"]` — same bug class as the white-balance instance fixed earlier (WB now correctly uses `_DEFAULTS` at engine.py:1144 and :1668).

**Fix:** Both sites now use `_DEFAULTS["bw_channel_mixer_r"]` / `_g` / `_b`.

**Verified:** Comparisons source from `_DEFAULTS`, maintaining single-source-of-truth with `params.py` registry.

### ✅ RESOLVED 2026-07-04: E1 float32 canvas breaks undereye/makeup/relight (solid-blue face)
**Issue:** The E1 float32 per-face chain missed `undereye.py`, `makeup.py`, and `relight.py` — they called `cv2.cvtColor` directly on the float32 [0,255] canvas (OpenCV expects float BGR in [0,1]), so any recipe with `dark_circles>0` or `blush>0` painted the whole face solid blue (~20 of 50 recipes on real photos; reported as a "multi-face" bug but reproduces on any face).

**Fix:** `apply_u8_op_float` E1 adapters added to `UnderEyeRepairer.repair`, `MakeupEngine.apply_blush`, `Relighter.relight`/`sculpt`. Also fixed `anime_cinematic_v1` recipe data (`skin.relight` 35.0 → 0.35; convention is 0–1 fraction) and clamped the engine's relight conversion to ≤100.

**Verified:** action/wedding/idol/xhs_soft_glow render normal skin on the DSCF6102 reference photo. Regression tests: `tests/test_e1_float_parity.py` (float/uint8 parity + recipe `skin.relight` range check).

### ✅ RESOLVED 2026-07-03: Equalize grays pale/high-key faces (skin.py `equalize`)
**Issue:** Reported as "recipes look worse with 2–3 people" but actually image-type-dependent: CLAHE in `equalize` darkened very pale, bright, low-contrast skin (cosplay white makeup, L≈200) by ~10 L points and amplified skin L variance (gray mottling). Multi-face path verified bit-identical to single-face; face ROIs verified non-contaminating.

**Fix:** uint8 branch of `equalize` now (1) re-centers mean skin L after the CLAHE/protection blend, and (2) caps skin L std at its original value (asymmetric — genuine tone-evening untouched). Float path inherits via `apply_u8_op_float`.

**Verified:** on reference photo `~/Desktop/event/DSCF6102.jpg` (2 pale faces), natural recipe: skin mean L 185→194.6 (source 199.4), std 26.9→21.8 (source 23.6). Regression tests: `tests/test_skin.py::TestEqualize` bright-pale luminance + variance tests (strengths 20/60).

### ✅ RESOLVED 2026-07-03: Sharpen inert zone 1-60
Sharpen mapping now activates correctly across the full 1-100 slider range.

### ✅ RESOLVED 2026-07-03: Equalize s² non-linearity
Equalize curve now applies the correct s² transfer function.

### ✅ RESOLVED: White-balance literals (was engine.py:1098, 1570)
Both sites now use `_DEFAULTS["white_balance_kelvin"]` / `["white_balance_tint"]` — verified 2026-07-03.

### ✅ RESOLVED: `_build_dimensional_mask` WIP (skin.py, perf_optimizations.py)
Shape-aware fix is merged (`skin.py:31-61` takes a `shape` param); working tree clean as of 2026-07-03.

### ✅ RESOLVED 2026-07-03: `skin.locus` recipe override wiring
Nested `skin.locus` recipe data now passes through `recipe_loader.py`, `build_context()`, `ProcessingContext.skin_locus`, and `_process_face_core()` into `unify_hue_line(locus=...)`.

**Verified:** focused non-MediaPipe tests for context, hue-shift behavior, and GUI/params wiring pass (22 tests). Full `tests/test_skin_locus_override.py` still aborts locally inside MediaPipe engine initialization, so keep real-photo visual QA as the final confidence gate.

### MINOR: LUT Hot-Reload Unwired
**Status:** `watch_luts_dir()` daemon exists in `lut.py` but not integrated into GUI/CLI.

**Action:** Can defer — not on the critical path.

### ✅ RESOLVED 2026-07-13: `_process_inputs` argument-order footgun (gui.py)

**Issue:** `gui.PROCESS_INPUT_KEYS` (names, derived from `param_names()`) and the
`_process_inputs` component list (the Gradio `inputs=` for `process_image`) are
bound by POSITION only — Gradio calls `process_image(*values)` with no
name-based binding. `_process_inputs` was a hand-ordered 213-element
list, so adding a `ParamSpec` to `params.py` (which auto-inserts a name into
`PROCESS_INPUT_KEYS`) required manually inserting the matching component at the
exact same index. Miss the position and every later argument shifts by one — a
brightness slider silently read as contrast, no crash, corrupted output. This
recurred at least twice historically.

**Structure (AST-verified 2026-07-13):** 213 slots = 127 identity-named vars +
85 `gr.State` placeholders named `_<paramname>_state` + 1 special case
(`img_paths → img_input`). All 213 component vars are unique; no constants or
inline expressions in the list. The `gr.State` placeholders are live `outputs=`
targets (session-load / undo / redo / snapshot), so every slot must remain a
real component in identical order.

**Fix:** replaced the list literal with a name→component dict `_process_input_components`
keyed by `PROCESS_INPUT_KEYS`, derive the positional list via
`[_process_input_components[k] for k in PROCESS_INPUT_KEYS]`, and added an
import-time set-equality assertion so a forgotten/renamed entry raises an
`AssertionError` at import (loud) instead of shifting arguments (silent).
Byte-identical runtime (same components, same order, same list type; all seven
downstream `inputs=`/`outputs=` consumers untouched). Per-position parity tests
live in `tests/test_gui.py::TestProcessInputKeys`.

**Verified:** derived list is byte-identical to the former hand-ordered 213-slot
list (set + order), and the guard raises on both a missing-key and an orphan-key
simulation; `TestProcessInputKeys` green (13 tests).

**Not covered:** the separate hand-ordered `_recipe_outputs` list (gui.py) still
has the same class of risk and is deferred to a follow-up.

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

---

## Debugging & Crash Dumps

### Enable Crash Logging
```python
from retouch.utils import log_crash
# Automatically called on exceptions in gui.py render loop
# Writes to ~/.cache/retouch/crash.log
```

### Common Issues

**Face detection failing on a specific image:**
1. Check `retouch/detection.py` — RetinaFace → MediaPipe fallback should catch it
2. If both fail, image may have extreme profile or occlusion (see "Known Limitations")
3. Try `--global-only` flag in CLI (skip detection, apply only color grading)

**Slow on high-res:**
1. GUI defaults to `fast=True` (800px internal processing) — acceptable for interactive tuning
2. For export, set `fast=False` for full resolution (one-time cost, 2048px proxy applies automatically)
3. Use `--workers 8` in CLI for batch processing of 100+ images

**Out of memory:**
1. Proxy resolution activates automatically for images > 2048px — no action needed
2. If still OOM on 4K+, use `fast=True` or batch with `--workers 1` (serial processing)

---

## Documentation Map

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Quick start, install, examples | End users |
| `docs/architecture/ARCHITECTURE.md` | Deep-dive pipeline, modules, design | Engineers |
| `docs/review/AUDIT_REPORT.md` | Security, test coverage, findings | Code reviewers, maintainers |
| `docs/architecture/API.md` | Python API reference | API consumers |
| `docs/guides/RECIPE_GUIDE.md` | Creating custom presets | Content creators |
| `docs/guides/GUI.md` | Gradio UI layout & components | Frontend work |
| `docs/guides/BATCH_GUIDE.md` | Batch processing via CLI | Power users |
| `docs/FUJI_SIMS_GUIDE.md` | Fuji film simulation recipes | Portrait photographers |
| `docs/INDEX.md` | Full documentation map | All |
| `.github/workflows/` | CI/CD (test + benchmarks) | DevOps, reviewers |

---

## For New Contributors

1. **Read `docs/architecture/ARCHITECTURE.md` first** — understand the 7 stages and module responsibilities
2. **Run the test suite** — `pytest tests/ -q` (should be green)
3. **Follow the Code Style section** — naming, structure, performance conventions
4. **Before pushing:**
   - Run syntax check: `for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done`
   - Run test suite: `pytest tests/ -q`
   - Add tests for new features
5. **Submit a PR** with a descriptive message (see Git Workflow above)

---

## Contact & Support

- **Issues:** https://github.com/dennisbrian/retouch/issues
- **Discussions:** https://github.com/dennisbrian/retouch/discussions
- **Security:** Contact maintainers privately (no public disclosures)

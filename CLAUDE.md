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

**~42.5k LOC (retouch/*.py + gui.py + cli.py), 70 modules in retouch/, 3,736 tests collected (pytest --collect-only, 2026-07-15).**

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
Every tunable parameter is registered in `PROCESSING_PARAMS` (a list of `ParamSpec` objects) — GUI, CLI, engine defaults, and recipes all auto-generate from it.

**When adding a new parameter:**
1. Add one `ParamSpec` entry in `params.py` — CLI flag + engine defaults auto-wire.
2. GUI does NOT auto-wire: add a matching entry to `_process_input_components` (name → Gradio component, or a `gr.State(...)` placeholder) in `gui.py`, keyed by name. Order is derived, not hand-placed — forgetting the entry raises an `AssertionError` at import time (see `tests/test_gui.py::TestProcessInputKeys`), not a silent slider mismatch.
3. Same pattern applies to `_recipe_output_components` / `RECIPE_OUTPUT_KEYS` for recipe/reset-handler outputs (see `tests/test_gui.py`, `TestRecipeOutputKeys`-style tests). The 10 `reset_*` handlers are still hand-ordered pairs (latent, low-risk, not covered by the guard).

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
- **3,736 tests collected** (`pytest --collect-only`, 2026-07-15) — full-suite pass/fail baseline not re-run at this count
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

> Resolved items are one-liners; full root-cause/verification detail lives in
> the commit message (`git show <hash>` or `git log --grep "<keyword>"`), not
> here — this file is sent on every request, so keep it an index, not an essay.

- ✅ 2026-07-03/04 batch (B&W mixer literals, E1 float32 canvas, equalize pale-face + s² non-linearity, sharpen inert, white-balance literals, `_build_dimensional_mask` shape fix) — all shipped; `git log --grep` for detail (`260fe02`, `e0b294c`, `1640eeb`, `6331994`)
- ✅ 2026-07-13 `skin.locus` recipe override wiring; test hang was a MediaPipe teardown deadlock (bare `RetouchEngine()`, not init abort) — `36cb93a`
- ✅ 2026-07-13 `_process_inputs` argument-order footgun — name-keyed dict + import-time drift guard, see [Architecture Decisions](#single-source-of-truth-retouchparamspy) — `f0b656b`, `da1f8cb`
- ✅ 2026-07-13 `_recipe_outputs` mirror-image footgun (same fix pattern; closed a live 107-vs-111 value drift) — see [Architecture Decisions](#single-source-of-truth-retouchparamspy)
- MINOR, deferred: LUT hot-reload daemon (`lut.py::watch_luts_dir`) exists but not wired into GUI/CLI
- MINOR, deferred: 10 `reset_*` handlers in gui.py remain hand-ordered positional pairs (latent, low-risk)
- ✅ 2026-07-14 `tests/test_cosplay_moat.py` bare-`RetouchEngine()` → module `engine` fixture (same teardown pattern as skin_locus)
- ✅ 2026-07-14 `test_ext_map_keys_match_radio_choices` — expect `PNG-16` (map already had it; test was stale)

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

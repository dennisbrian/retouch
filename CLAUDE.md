# Claude.md — Pro Max Face Retouch Engine

**Project:** Professional automated face retouching pipeline  
**Repository:** https://github.com/USERNAME/REPO  
**Status:** Mature (v2.1.0 — Fuji-quality color recipe system)  
**Last Updated:** 2026-07-02

---

## Quick Context

This is a **production-grade image processing engine** that applies professional retouching to portraits via a modular 7-stage pipeline. It combines:
- MediaPipe face detection + 478-point landmarks
- BiSeNet ONNX semantic segmentation (skin, lips, eyes, hair, etc.)
- Frequency-separation skin smoothing + component enhancement
- Color grading + Fuji film simulation presets
- Virtual studio relighting + advanced lens effects

**16.5k LOC, 40 core modules, 1,626 passing tests.**

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
python3 scripts/benchmark.py
```

---

## Architecture Decisions

### Single-Source-of-Truth: `retouch/params.py`
Every tunable parameter is registered in `PROCESSING_PARAMS` (a list of `ParamSpec` objects). This eliminates the "7-way sync" problem — GUI scales, CLI args, engine defaults, and recipes all auto-generate from this one list.

**When adding a new parameter:**
1. Add one `ParamSpec` entry to `PROCESSING_PARAMS` in `params.py`
2. CLI flag, GUI slider, and engine defaults auto-wire
3. No other file edits needed

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
- **1,626 tests passing, 1 skipped** (model gate)
- **89% coverage** on core modules (per `pytest --cov`)
- **Unit + integration tests** for every public API
- **Deep algorithmic verification** (monotonicity checks, round-trip stability, etc.)

### Audit Status
- **Security:** No hardcoded secrets, API keys, or absolute paths
- **Performance:** Benchmarked at 702ms per-face (400×400), 6.9ms for no-face global-only
- **Process:** Weekly syntax checks, monthly algorithmic re-audit (AUDIT_REPORT.md)

### Known Limitations
- Remaining ~4% undetected faces: extreme profiles, heavy occlusion, tiny faces in distance shots
- LUT hot-reload (daemon thread watcher) exists but not wired into GUI/CLI yet
- B&W channel-mixer literals (30/59/11) in `engine.py` should use `_DEFAULTS` (see "Outstanding Fixes" below)
- `.3dl` LUT loader has no upper-bound on file size (theoretical DoS, not reachable from GUI)

---

## Outstanding Fixes

### MINOR: Registry-Bypass Bug — B&W channel mixer (engine.py:1212-1216, 1802-1806)
**Issue:** `bw_active` checks `ctx.bw_channel_mixer_r != 30 / != 59 / != 11` using hardcoded literals instead of `_DEFAULTS["bw_channel_mixer_*"]` — same bug class as the white-balance instance fixed earlier (WB now correctly uses `_DEFAULTS` at engine.py:1144 and :1668).

**Fix:** Replace both sites with comparisons against `_DEFAULTS["bw_channel_mixer_r"]` / `_g` / `_b`.

**Impact:** Currently matching (no behavioral effect), but violates single-source-of-truth — will silently desync if `params.py` defaults change. (Found in the 2026-07-03 audit; full findings in `PLAN_TIERE_ENGINE_FIDELITY.md`.)

### ✅ RESOLVED: White-balance literals (was engine.py:1098, 1570)
Both sites now use `_DEFAULTS["white_balance_kelvin"]` / `["white_balance_tint"]` — verified 2026-07-03.

### ✅ RESOLVED: `_build_dimensional_mask` WIP (skin.py, perf_optimizations.py)
Shape-aware fix is merged (`skin.py:31-61` takes a `shape` param); working tree clean as of 2026-07-03.

### MINOR: LUT Hot-Reload Unwired
**Status:** `watch_luts_dir()` daemon exists in `lut.py` but not integrated into GUI/CLI.

**Action:** Can defer — not on the critical path.

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
# Writes timestamped file to current dir: crash_YYYYMMDD_HHMMSS.log
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
| `ARCHITECTURE.md` | Deep-dive pipeline, modules, design | Engineers |
| `AUDIT_REPORT.md` | Security, test coverage, findings | Code reviewers, maintainers |
| `API.md` | Python API reference | API consumers |
| `RECIPE_GUIDE.md` | Creating custom presets | Content creators |
| `GUI.md` | Gradio UI layout & components | Frontend work |
| `BATCH_GUIDE.md` | Batch processing via CLI | Power users |
| `FUJI_SIMS_GUIDE.md` | Fuji film simulation recipes | Portrait photographers |
| `.github/workflows/` | CI/CD (test + benchmarks) | DevOps, reviewers |

---

## For New Contributors

1. **Read ARCHITECTURE.md first** — understand the 7 stages and module responsibilities
2. **Run the test suite** — `pytest tests/ -q` (should be green)
3. **Follow the Code Style section** — naming, structure, performance conventions
4. **Before pushing:**
   - Run syntax check: `for f in retouch/*.py gui.py cli.py; do python3 -m py_compile "$f"; done`
   - Run test suite: `pytest tests/ -q`
   - Add tests for new features
5. **Submit a PR** with a descriptive message (see Git Workflow above)

---

## Contact & Support

- **Issues:** https://github.com/USERNAME/REPO/issues
- **Discussions:** https://github.com/USERNAME/REPO/discussions
- **Security:** Contact maintainers privately (no public disclosures)

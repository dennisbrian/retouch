# Retouch Engine — System Audit Report

Generated: 2026-07-02 (Cycle 3)
Auditor: Loop Engineering protocol (AGENTS.md v3.1)
Scope: Full re-audit covering everything since the 2026-06-28 report — 15 feature commits (LCH/HSL perceptual color tools, blend modes, negative split-toning, white balance, B&W channel mixer, `.3dl` LUT loader, GitHub Actions CI) plus uncommitted WIP fixes to `skin.py`/`perf_optimizations.py`/`style_library.py`. Re-verifies prior findings #1–10 for actual closure and re-runs the full test suite. See §11 for this cycle's new evidence; §§1–10 are preserved from the 2026-06-28 baseline for history.

---

## 0. Executive Summary

**Post-audit note (2026-07-04):** a later local test report (`./TEST_REPORT_2026-07-04.md`) records 5 failures and 5 skipped tests. Treat the verdict below as historical for the 2026-07-02 snapshot, not as current release sign-off.

**This section reflects Cycle 3 (2026-07-02). See §11 for full evidence; §§1–10 below are the preserved 2026-06-28 baseline.**

The Retouch Engine is a mature, well-structured Python image-processing pipeline, now ~16.5k LOC across 40 core modules + 3 entry points after a wave of Phase 1.d feature work (LCH/HSL color tools, blend modes, split toning, white balance, channel mixer, `.3dl` LUTs). Status:

- **All 40 Python files compile cleanly.** No syntax errors.
- **Test suite was green in the audited cycle: 1626 passed, 1 skipped** (`pytest tests/ -q`, 512s) — up from 1495 at the last audit. A later `2026-07-04` report records 5 failures, so this is historical evidence, not current status.
- **Prior findings confirmed closed**: all 4 test-coverage gaps (§8 rows 1-3, `tonal`/`precision`/`style_transfer`/`recipe_loader_cli` tests promoted, commit `06e50aa`), the 3 silent GUI/engine/utils exceptions (§8 rows 4-5), the `watch_luts_dir` doc drift (§8 row 10), and both bugs from the §10 "Retouch Fix Audit" (no-face `subject_mask`, dead sharpen branch) — all via commit `e633293`. **§8 row 6 (LUT hot-reload wiring) remains open, unchanged** — see §11.2.
- **No new MAJOR-tier defects found in this cycle.** Two candidate MAJOR findings were identified by the auditor but both were verified as false positives after source re-read — `overlay_blend`/`hard_light_blend` (`utils.py:141-142`) already includes the ×2 factor (50%-gray identity holds, produces ~128.5), and `color_balance_lch` (`color_space.py:390-393`) already computes circular hue distance via `np.minimum(..., 360-...)`. See §11.3 for retraction details.
- **Retraction: the registry-bypass finding is stale in the current tree.** `engine.py` now gates white-balance/channel-mixer activation with `_DEFAULTS[...]` lookups at the relevant call sites (`retouch/engine.py:1537`, `retouch/engine.py:1606`, `retouch/engine.py:2401`, `retouch/engine.py:2566`). Keep this note only as a record of what the 2026-07-02 snapshot reported.
- **Uncommitted WIP** (`skin.py`, `perf_optimizations.py`, `style_library.py`) fixes a genuine latent bug (`_build_dimensional_mask` returning a hardcoded 200×200 zero mask instead of the actual image shape when no region matched) and adds defensive `None` guards; all 1626 tests still pass. This touches Visual-Critical code (`skin.py`) per AGENTS.md and has **not** had a Visual QA pass — flagged as a process gap, not a defect.
- **No hardcoded secrets, API keys, or absolute user paths** anywhere in the code added since the last audit.
- Two GitHub Actions workflows (`ci.yml`, `test.yml`) now run redundantly on every push/PR to `main`; `ci.yml`'s test job excludes ~40 of the ~55 test files, so it silently under-covers relative to `test.yml` and the local suite.

| Severity | Count | Summary |
|----------|-------|---------|
| CRITICAL | 0 | — |
| MAJOR    | 0 | — |
| MINOR    | 4 (active) | LUT hot-reload still unwired (§3.4, unchanged); redundant/under-covering CI workflow (§11.4); uncommitted `skin.py` changes lack Visual QA (§11.5); `gui.py`/`engine.py`/`grading.py`/`params.py` continue to grow (1668/1855/1401/1337 LOC) |
| INFO     | 2 | `color_space` wide-gamut tradeoff (unchanged, §1.4); `.3dl` loader has no upper bound on `3DLUTSIZE` (theoretical DoS, not reachable — not wired into GUI/CLI) |

**Verdict for the 2026-07-02 snapshot: ship-ready for the render path.** No CRITICAL defects and no MAJOR defect were reachable from `engine.process()` in that snapshot. The current repo state is not current sign-off because `./TEST_REPORT_2026-07-04.md` records 5 failures; resolve those before treating this audit as release-ready. The two new MAJOR bugs were latent (unused public functions) and were retracted after source re-read. Recommend closing the CI redundancy opportunistically and running a Visual QA pass on the uncommitted `skin.py` diff before merging.

---

## 1. Verification Evidence

All evidence below is actual command output, per AGENTS.md §10 evidence rules.

### 1.1 Syntax / Lint (CRITICAL tier — passed)
```
$ for f in retouch/*.py gui.py cli.py desktop.py benchmark.py; do python3 -m py_compile "$f"; done
OK: retouch/__init__.py ... OK: retouch/utils.py
OK: gui.py  OK: cli.py  OK: desktop.py  OK: benchmark.py
→ 40/40 files compile, zero failures.
```

### 1.2 Unit Tests (CRITICAL tier — passed)
Baseline (2026-06-28):
```
$ python3 -m pytest tests/ -q --tb=line -p no:warnings
1495 passed, 1 skipped in 479.44s (0:07:59)
→ 1496 tests collected; 1 skipped (model-gated).
```

This cycle's run is recorded in §11.1 (1626 passed, 1 skipped) — see the re-audit evidence there.

### 1.3 Benchmark (DEFERRED tier — baseline captured)
`benchmark_results.json` present (2026-06-23). Per-face 400×400 median = 702 ms; pipeline.no-face 600×400 fast = 6.9 ms. No regression baseline re-run performed this cycle (not performance-affecting changes).

### 1.4 Deep Algorithmic Verification (NEW — all PASS ✓)

This cycle ran a dedicated harness that empirically validates the math in the modules flagged as untested, rather than relying on docstrings. Every claim below is an actual assertion result, not an eyeball.

| Module | Property verified | Result |
|--------|-------------------|--------|
| `tonal.hd_curve_lut` | Monotonic non-decreasing across the **full** toe×shoulder×midpoint×gamma grid (toe/shoulder ∈ [0, 0.5]) | ✓ **0 violations** — the custom-midpoint sigmoid never inverts |
| `tonal` | `strength=0` is identity; `apply_hd_curve` / `lift_gamma_gain` no-op at zero | ✓ |
| `precision.to_uint8(to_float(x))` | Exact inverse for **all** `x ∈ 0..255` (round-to-nearest) | ✓ **max abs err = 0** |
| `precision.PrecisionContext` | Raises if `.process` called outside the `with` block; clips floats to [0,1] | ✓ |
| `color_space` LCH | Round-trip BGR→LCH→BGR stability | ✓ mean err **1.03** (max 32 from inherent uint8 LAB quantization — acceptable) |
| `color_space` ProPhoto | Round-trip stability | ✓ mean err **0.0** |
| `skin_protect.protect_skin` | `strength=1` ⇒ skin pixels unchanged (mean diff 0.33); full op applied on non-skin (0.00); `strength=0` ⇒ op everywhere | ✓ semantics correct |
| `style_transfer.weighted_mean_std` | Matches independent reference impl; zero-weight guard returns `(0, 1)`; reinhard empty-mask returns a copy of src | ✓ |

**No algorithmic bugs found.** One initial harness "FAIL" was a flaw in the *test* (a synthetic orange that fell outside the 25°-band skin-hue gate, so it was correctly treated as non-skin) — re-run with an in-band skin tone passed cleanly, confirming the gate works as designed.

**INFO (documented tradeoff, not a defect):** `color_space`'s wide-gamut matrix path (ProPhoto/Adobe) deliberately **skips gamma linearization and D65→D50 white-point adaptation** for speed and round-trip stability. The module docstring is explicit about this; round-trip error of 0.0 proves stability, but absolute colorimetric values for those spaces are approximate. Flagged only so downstream consumers don't assume colorimetric exactness.

---

## 2. Security Audit

### 2.1 Path Traversal — MITIGATED ✓
- `recipe_loader.import_recipe` (`retouch/recipe_loader.py:269`) builds `dest = _user_recipes_dir() / f"{name}.json"` using `name` from JSON content.
- **Mitigation**: `validate_recipe()` (`recipe_schema.py:106`) enforces `name` pattern `^[a-z][a-z0-9_]*$` + `maxLength: 64` **before** the disk write. No `/`, `..`, or traversal chars can pass.
- `remove_user_recipe` (`recipe_loader.py:303`) additionally sanitizes via `re.sub(r"[^a-z0-9_]", "", name.lower())`.
- `RETOUCH_USER_RECIPES` env override is an admin-controlled lever, not user input. Acceptable.

### 2.2 Dangerous Calls — CLEAN ✓
- `subprocess`: only in `scripts/bench/benchmark.py:142` (with `TimeoutExpired` handling, no `shell=True`) and `tests/test_cli*.py` (no `shell=True`). Safe.
- `eval`/`exec`/`pickle.load`: only `pickle.loads` in `tests/test_engine.py:732` (test code). No production use.
- No `shell=True` anywhere in the codebase.

### 2.3 Secrets / Hardcoded Paths — CLEAN ✓
Grep for `api_key|secret|password|token=|/Users/|/home/|C:\\` found **zero** matches in source (only `.keys()` dict accesses). No secrets committed.

### 2.4 Error Handling — PARTIAL ⚠ (now MINOR)
AGENTS.md §6: *"Never use bare `except: pass` or catch generic `Exception` without logging/raising."*

**Good news this cycle:** the **critical render path is well-instrumented.** The per-image processing loop in `gui.py` (the `for idx, path_item …` block) catches per-image failures and logs via `_logger.exception("Failed to process %s", …)` **plus** `retouch.utils.log_crash(...)` — a robust crash-dump path. The temp-cleanup `except` also logs (`_logger.warning`). So a failed retouch is **never silent** to logs/disk.

The remaining silent sites are the **style/batch *management* handlers**, which surface the error to the Gradio UI string but never touch `_logger`:

| File / handler | Behaviour | Verdict |
|----------------|-----------|---------|
| `gui.py` `on_save_style` (~L145) | `except Exception as e: return …, f"Failed to save style: {e}"` — UI only | MINOR — observability gap |
| `gui.py` `on_learn_style` (~L177) | `except Exception as e: return …, f"Error during dataset learning: {e}"` — UI only | MINOR — observability gap |
| `gui.py` `on_process_folder` (~L226) | `except Exception as e: gr.Warning(...); return …` — UI only, no `_logger` | MINOR — observability gap |
| `retouch/engine.py:1217` | Silent ProcessPool→ThreadPool fallback | MINOR — add `logger.warning` so pool failures are visible |
| `retouch/engine.py:1616` | `close()` swallows shutdown error | MINOR — add `logger.warning` |
| `retouch/utils.py:356` | Primary-face detection fallback | MINOR — add `logger.warning` |
| `retouch/style_library.py:70` | Version parse → "2.0" default | MINOR — benign default |
| `retouch/perf_optimizations.py:580` | Logs via `logger.exception(...)` | OK ✓ |
| `retouch/batch_processor.py:47,54` | Logs via `logger.warning(...)` | OK ✓ |

**Action**: Add a one-line `_logger.exception(...)` to the 3 GUI style/batch handlers and `logger.warning(...)` to the 3 engine/utils fallbacks. None are correctness defects — they're observability gaps. Downgraded from MAJOR to MINOR because no silent site sits on the actual image-rendering correctness path.

Suggested one-liners (copy-paste):

```
_logger.exception("on_save_style failed for %s", style_name, exc_info=True)
_logger.exception("on_learn_style failed for %s", dataset_name, exc_info=True)
_logger.exception("on_process_folder failed for %s", folder_path, exc_info=True)

logger.warning("FaceProcessorPool dispatch failed, falling back to ThreadPool", exc_info=True)
logger.warning("Engine.close() shutdown error", exc_info=True)
logger.warning("Primary-face detection fallback used for %s", image_path, exc_info=True)
```

---

## 3. Thread Safety & Concurrency

### 3.1 GUI Engine Singleton — SAFE ✓
`gui.py:45-53`: `get_engine()` uses double-checked locking with `_engine_lock`. Correct lazy singleton.

### 3.2 Multi-Face Parallelism — SAFE ✓
- `engine.py:1216` dispatches faces to `FaceProcessorPool` (ProcessPoolExecutor). Each worker is a **separate process** with its own `ColorGrader`/caches — no shared mutable state.
- Fallback (`engine.py:1239`) uses `ThreadPoolExecutor` sharing `self`, but the worker `_process_one_face` (engine.py:1253-1337) does **not** call grading — it only does per-face skin/eyes/lips/teeth/blemish on the face canvas. Grading runs in `_stage_grade` (engine.py:1451) **after** face compositing, on the main thread. Verified via method-boundary map.

### 3.3 Shared LUT Caches — RACE RESOLVED ✓ (was MAJOR-2)
Re-verified this cycle: **both LUT caches are now lock-guarded.** The prior MAJOR finding is closed.

- `LUTRegistry` (`lut.py`) now holds `self._lock = threading.Lock()` and takes it in **`get`, `poll_changes`, `register`, `reload`, and `cache_size`**. The class docstring was updated to *"This class is safe for concurrent access from multiple threads."* The `_REGISTERED_MTIME` sentinel path and mtime-invalidation are all inside the lock.
- `ColorGrader._get_cached_lut` (`grading.py`) now wraps its read and its write of `self._lut_cache` in `self._lock`. Check-then-set is no longer a race (worst case is a harmless double `load_cube` on first concurrent miss — never corrupting).
- `watch_luts_dir`'s `poll_changes()` mutates `_known_mtimes` + `_cache` under the same registry lock, so the daemon is safe to run concurrently with `get`.

This holds **independently of** the Gradio `app.queue(default_concurrency_limit=1)` serialization in `gui.py` — so raising the concurrency limit later no longer reintroduces the race. AGENTS.md §8 satisfied by construction.

### 3.4 LUT Hot-Reload Wiring — INCOMPLETE ⚠ (MINOR — two caches, one watcher)
A subtle architectural gap surfaced during the deep check, important for anyone planning to *use* `watch_luts_dir`:

- There are **two independent LUT caches**: `LUTRegistry._cache` (mtime-invalidated) and `ColorGrader._lut_cache` (keyed by resolved path, **no mtime invalidation**).
- The **actual render path does NOT use `LUTRegistry`.** The film-emulation chain is `grade() → _add_film_emulation() → _resolve_cube_lut() → _get_cached_lut() → load_cube()`, caching into the **grader's** `_lut_cache`. `LUTRegistry` is only touched by tests and `list_available_luts()`.
- `watch_luts_dir` evicts entries from **`LUTRegistry`** via `poll_changes()`. So even once wired, editing a `.cube` on disk would refresh the registry but the **grader would keep serving the stale LUT** — its cache has no mtime check and no eviction hook.
- Additionally `watch_luts_dir` has **zero production callers** (only `lut.py` definition, `test_lut_registry.py`, and docs), and its internal error branch uses `print(...)` rather than `logger`.

**Action**: If hot-reload is a real product goal, wire `watch_luts_dir(callback)` so the callback **also clears `ColorGrader._lut_cache`** (e.g. evict the matching key, or give the grader an mtime check mirroring `LUTRegistry.get`). Otherwise, mark `watch_luts_dir` experimental in the docstring to avoid implying a render-path effect it doesn't have. Also switch its `print` to `logger.warning`.

Two concrete remediation options (pick one):

Option A — Wire hot-reload into graders (recommended if hot-reload is desired):

```
# in lut.py (watcher): when a file change is detected
LUTRegistry._evict(path)
# notify registered graders (or via a central hook)
for grader in GraderRegistry.get_all():
    grader.evict_lut(path)

# in grading.py (ColorGrader):
def evict_lut(self, path: str):
    with self._lock:
        self._lut_cache.pop(path, None)

# Alternatively, add an mtime check on cache-hit that mirrors LUTRegistry.get()
```

Option B — Mark experimental (fast, low-effort):

```
"""watch_luts_dir

NOTE: experimental. Currently only evicts LUTRegistry cache. It does NOT evict
ColorGrader._lut_cache — graders will continue to serve cached LUTs until
restarted or manually cleared. Do not rely on this for production hot-reload.
"""

# also replace print(...) with logger.warning(..., exc_info=True)
```

---

## 4. AGENTS.md Compliance

| Mandate (Section) | Status | Evidence |
|--------------------|--------|----------|
| Thin UI/entry points (§9) | ⚠ PARTIAL | `desktop.py` (32 LOC) ✓, `cli.py` (484 LOC) ✓. `gui.py` is **1648 LOC** — heavy, though it delegates logic to `retouch.params`/`engine`/`io`. Acceptable but at the limit. |
| Central parameter registry (§9) | ✓ EXCELLENT | `params.py` (1222 LOC) is the single source of truth. GUI/CLI auto-populate via `recipe_to_params` / `gui_values_to_engine_kwargs` / `build_params`. |
| Modular stages (§9) | ✓ EXCELLENT | Each stage isolated: `lips.py`, `skin.py`, `eyes.py`, `relight.py`, `grading.py`, etc. |
| Type hints (§6) | ✓ GOOD | Near-complete annotations. A few `__init__`/dunder methods omit return type (acceptable). Uses 3.9+ generics (`dict[str, Any]`) correctly. |
| No bare `except: pass` (§6) | ⚠ PARTIAL | See §2.4 — 4 silent `except Exception:` without logging. |
| Channel order (§7) | ✓ EXCELLENT | 100+ `cv2.cvtColor` calls consistently respect BGR(LAB/HSV) boundaries. No RGB/BGR mixups found. |
| GPU fallback (`docs/PRECISION.md` §5) | ✓ EXCELLENT | `perf_optimizations.py` builds ORT providers with CPU fallback; no CUDA/CoreML assumption. |
| No legacy helpers (`docs/PRECISION.md` §5) | ✓ CLEAN | `combine_adaptive`, `SkinProcessor.smooth`, `retinex_msr` chain fully removed (verified — grep returns nothing). |
| Path traversal guard (§8) | ✓ MITIGATED | See §2.1 — schema-validated `name` pattern. |

---

## 5. Code Quality & Architecture

### 5.1 Dead Code — Mostly Cleaned ✓
Re-verification of the prior (Jun 22) report against the current (Jun 25) codebase:

| Item (prior report) | Status |
|---------------------|--------|
| `engine.py` unused `field` import | GONE ✓ |
| `grading.py` unused `Union` import (with `from __future__ import annotations`) | GONE ✓ (string-annotation only, conventional) |
| `detection.py` unused `import sys` | GONE ✓ |
| `gui.py` unused `BytesIO` | GONE ✓ |
| `perf_optimizations.py` unused `field` | GONE ✓ |
| `frequency.py` `BILATERAL_D_FACTOR/_MIN` | GONE ✓ |
| `skin.py` `SkinProcessor.smooth()` | GONE ✓ |
| `frequency.py` `combine_adaptive()` + `skin.adaptive_smooth()` chain | GONE ✓ |
| `relight.py` `retinex_msr/ssr/_retinex_normalize` | GONE ✓ |
| `perf_optimizations.py` `detect_faces_downscaled/init_onnx_session/init_mediapipe_with_gpu/apply_tonal_lut` | GONE ✓ |

**Remaining**: none. All imports from the original dead-code list are clean ✓.

### 5.2 Architecture — Strong ✓
- Clear separation: detection → parsing → frequency → per-face stages → composite → global → grade → finish.
- `ProcessingContext` dataclass carries all params; `FaceContext` carries per-face cached state.
- Module sizes are reasonable except `engine.py` (1735), `gui.py` (1648), `params.py` (1222), `grading.py` (1096) — all justified by their role (orchestrator, UI, registry, grading stack).

---

## 6. Performance

- **Vectorization**: Lab/HSV conversions use `np.clip` + slicing (e.g. `grading.py:909-916`, `skin.py:120`). No pixel-level Python loops found in hot paths.
- **Memory**: `np.clip(...).astype(np.uint8)` pattern is pervasive — creates a temp copy per conversion. Acceptable; in-place (`out=`) opportunities exist but are micro-optimizations.
- **Benchmark baseline**: `benchmark_results.json` present. Per-face 400×400 = 702 ms median (detection mocked). No regression check run this cycle (no perf-affecting changes).
- **No budget violations identified**. Recommend re-baselining after any grading/skin change.

---

## 7. Test & Coverage

### 7.1 Suite Health — EXCELLENT ✓
1496 collected / 1495 passed / 1 skipped. ~8 min runtime (model-gated). Near-1:1 module→test mapping.

### 7.2 Coverage Gaps ⚠ (MINOR)
Modules with **no dedicated test file** (confirmed via filename grep this cycle):

| Module / surface | Public defs | Risk | Note |
|------------------|-------------|------|------|
| `retouch/tonal.py` | 7 | Toe/shoulder/sigmoid tone-curve math | Math verified ad-hoc in §1.4; needs a **permanent** regression test |
| `retouch/precision.py` | 9 | Float/bit-depth round-trip | Math verified in §1.4; needs permanent test |
| `retouch/style_transfer.py` | 3 | Reinhard weighted transfer | Math verified in §1.4; needs permanent test |
| `recipe_loader_cli` (CLI surface) | — | CLI arg-parsing / load-recipe path | **No `test_recipe_loader_cli`** — the CLI entry to the recipe loader is unexercised |

(`recipe_loader.py`'s library API is exercised by `test_recipe_integration.py` + `test_recipe_generator.py`, but its **CLI** wrapper is not.)

**Priority within this gap** (highest first):
1. **`tonal` + `precision`** — they sit on the global tone/bit-depth path that every image flows through; a silent regression here is wide-blast-radius. The §1.4 harness can be promoted almost verbatim into `test_tonal.py` / `test_precision.py`.
2. **`style_transfer`** — narrower (only the Reinhard/style path), but the weighted-stats guards are subtle; promote the §1.4 checks.
3. **`recipe_loader_cli`** — lowest correctness risk (thin wrapper over a tested library) but zero coverage on arg parsing / error messaging; add a couple of subprocess/CLI smoke tests.

**Action**: Promote the §1.4 verification harness into permanent `test_tonal.py`, `test_precision.py`, `test_style_transfer.py`, and add `test_recipe_loader_cli.py`.

---

## 8. Findings Summary

| # | Severity | Finding | Location | Action |
|---|----------|---------|----------|--------|
| 1 | MINOR | Test gaps on `tonal` + `precision` (global-path math; verified ad-hoc but no permanent test) | `tests/` | Promote §1.4 harness → `test_tonal.py`, `test_precision.py` |
| 2 | MINOR | Test gap on `style_transfer` (Reinhard weighted-stats guards) | `tests/` | Promote §1.4 harness → `test_style_transfer.py` |
| 3 | MINOR | No CLI test for `recipe_loader_cli` | `tests/` | Add `test_recipe_loader_cli.py` smoke tests |
| 4 | MINOR | 3 GUI style/batch handlers swallow exceptions to UI string only (no `_logger`) | `gui.py` `on_save_style`/`on_learn_style`/`on_process_folder` (~L145/177/226) | Add `_logger.exception(...)` |
| 5 | MINOR | 3 engine/utils silent fallbacks (no log) | `engine.py:1217,1616`; `utils.py:356` | Add `logger.warning(...)` |
| 6 | MINOR | `watch_luts_dir` unwired **and** won't hot-reload render path (grader cache not evicted) | `lut.py` + `grading.py` `_lut_cache` | Wire callback to clear grader cache, or mark experimental |
| 7 | RESOLVED | Unused imports (`field` + `Union`) — both gone or conventional with `from __future__ import annotations` | — | Verified closed ✓ |
| 8 | MINOR | `gui.py` at 1649 LOC strains the "thin UI" mandate | `gui.py` | Monitor; extract helpers if it grows |
| 9 | INFO | `color_space` wide-gamut path skips gamma/white-point (documented tradeoff; round-trip stable) | `color_space.py` | None — flagged for awareness |
| 10 | INFO | Doc drift: `watch_luts_dir` interval default `5.0`→`1.0`; `watch_luts_dir` error path uses `print` | `ARCHITECTURE.md:316`, `lut.py` | Fix doc; switch `print`→`logger.warning` |
| ✓ | RESOLVED | Prior MAJOR LUT-cache race — both caches now lock-guarded | `lut.py`, `grading.py` | Verified closed (§3.3) |

---

## 9. Recommendations (Priority Order)

**Tier 1 — close the verified-but-untested gap (highest leverage):**
1. **Promote the §1.4 harness into permanent tests** — `test_tonal.py` + `test_precision.py` first (global path, widest blast radius), then `test_style_transfer.py`. The math is already proven correct; this just locks it against regression.
2. **Add `test_recipe_loader_cli.py`** — a couple of CLI smoke tests over the loader's arg parsing and error messaging.

**Tier 2 — observability:**
3. **Instrument the 3 GUI style/batch handlers** (§2.4) — one `_logger.exception(...)` line each. Cheap; closes the only silent paths that touch user-visible workflows.
4. **Log the 3 engine/utils fallbacks** (§2.4) — `logger.warning(...)` so pool/detection degradation is visible in production logs.

**Tier 3 — wiring / hygiene:**
5. **Decide on LUT hot-reload** (§3.4): either wire `watch_luts_dir` so its callback also evicts `ColorGrader._lut_cache` (true hot-reload), or mark the watcher experimental. Don't ship it half-wired implying a render-path effect it lacks.
6. **Fix doc drift + `print`→`logger`** in `watch_luts_dir` (§3.4, finding 10).
7. ~~Remove 2 unused imports~~ — **RESOLVED** (verified gone/conventional).

---

## 10. Retouch Fix Audit (NEW — 2026-06-28)

Deep verification of the 4 recent retouch fixes in the render path.

### Fix 1 — Impact finish `subject_mask` preservation

**Code:** `_stage_finish` (L1591) passes `subject_mask=person_mask` to `add_impact_finish`.

✅ **Face path correct** — background preserved from clarity/contrast boost.

⚠️ **BUG — no-face path inconsistency:** `_no_face_fallback` (L1088) calls `add_impact_finish(result, ctx.impact)` **without** `subject_mask`, even though `person_mask` is available as a parameter. No-face images get full global impact (background noise boost) while face images get masked impact.

**Action:** Add `subject_mask=person_mask` to the no-face path call. One-line fix.

### Fix 2 — Sharpen gate (`ctx.sharpen > 0`)

**Code:** `_stage_finish` (L1551): `if ctx.sharpen > 0:` gates all sharpening.

✅ **Logic correct** — `sharpen=0` means no sharpening, period. Previously the per-face sharpen mask (eyes/eyebrows/hair edges) would trigger implicit sharpening at `amount=1.2` even when the user/recipe set `sharpen=0`.

⚠️ **Visible regression for `natural` recipe:** `natural` defaults `sharpen=0`. Previously it got implicit eye/hair edge sharpening from the mask. Now it gets none. This is *correct* behavior (no sharpen means no sharpen), but users may notice softer eyes/hair. If preserving the old look is desired, set `sharpen` to a small nonzero default (e.g. 10) for recipes that previously relied on implicit sharpening.

⚠️ **Dead code:** The `else 1.2` branch in `amount = max(1.2, ...) if ctx.sharpen > 0 else 1.2` is unreachable — already inside the `if ctx.sharpen > 0:` block. Remove for clarity.

### Fix 3 — Person mask blurring for subject separation

**Code:** `_stage_subject_separation` (L1138) applies `cv2.GaussianBlur(pm, (feather, feather), 0)` with `feather = max(3, int(min(h,w) * 0.02) | 1)`.

✅ **Clean, no regression risk.** Feather is proportional to image size (2% of min dimension). Smoother subject/background transitions.

### Fix 4 — 3px elliptical erosion on `smooth_mask`

**Code:** `_build_smooth_mask` in `perf_optimizations.py` applies `cv2.erode(smooth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)`.

✅ **Clean, well-tested.** Tests already account for the 3px margin (assertions use 5px buffer). Protects hair/skin boundary from smoothing bleed.

ℹ️ **Minor edge case:** For very small face crops (<50px IED), 3px erosion is proportionally larger (~6% of crop). Acceptable given detection minimums, but worth monitoring if detection is ever tuned for smaller faces.

### Summary

| Fix | Status | Regression Risk | Action |
|-----|--------|-----------------|--------|
| 1. Impact `subject_mask` | ⚠ Bug in no-face path | Low — no-face images only | Pass `subject_mask=person_mask` in `_no_face_fallback` |
| 2. Sharpen gate | ✅ Correct logic | Medium — `natural` loses implicit eye/hair sharpen | Accept as correct, or add small default sharpen to affected recipes |
| 3. Person mask blur | ✅ Clean | None | — |
| 4. Smooth mask erosion | ✅ Clean | None | — |

**Not actionable (informational):** `color_space` wide-gamut colorimetric approximation is a deliberate, documented speed tradeoff with proven round-trip stability — leave as-is.

---

## 11. Cycle 3 Audit — New Features + WIP Review (2026-07-02)

Scope: everything committed since the 2026-06-28 baseline (`adee825..HEAD`, 15 commits) plus the currently uncommitted working-tree diff. Evidence below is actual command output / line-numbered source reads, per AGENTS.md §10 evidence rules.

### 11.1 Verification Evidence

```
$ for f in retouch/*.py gui.py cli.py desktop.py benchmark.py scripts/*.py; do python3 -m py_compile "$f"; done
→ all files compile, zero failures.

$ python3 -m pytest tests/ -q --tb=short -p no:warnings
1626 passed, 1 skipped in 512.27s (0:08:32)
→ +131 tests vs. the 2026-06-28 baseline (1495→1626), all green. No regressions from the 15 new-feature
  commits or from the uncommitted skin.py/perf_optimizations.py WIP.
```

### 11.2 Re-Verification of Prior Findings

Two separate numbering schemes exist in the 2026-06-28 baseline: the §8 **Findings Summary** table (rows 1–10) and the §10 **Retouch Fix Audit**'s informal "Fix 1–4". Both are re-verified below.

**§8 Findings Summary (rows 1–10):**

| # | 2026-06-28 finding | Status now | Evidence |
|---|---|---|---|
| 1 | Test gap: `tonal`/`precision` (global-path math, verified ad-hoc, no permanent test) | ✅ **CLOSED** | `tests/test_tonal.py`, `tests/test_precision.py` now exist — commit `06e50aa` ("promote audit §1.4 harness into permanent tests (+55 tests)") |
| 2 | Test gap: `style_transfer` (Reinhard weighted-stats guards) | ✅ **CLOSED** | `tests/test_style_transfer.py` now exists — same commit |
| 3 | No CLI test for `recipe_loader_cli` | ✅ **CLOSED** | `tests/test_recipe_loader_cli.py` now exists — same commit |
| 4 | 3 GUI handlers swallow exceptions with no `_logger` | ✅ **CLOSED** | `gui.py` `on_save_style`/`on_learn_style`/`on_process_folder` all now call `_logger.exception(...)` before returning the UI-facing error string — commit `e633293` |
| 5 | 3 engine/utils silent fallbacks (no log) | ✅ **CLOSED** | `engine.py` FaceProcessorPool dispatch failure and `close()` shutdown both log (`logger.exception`/`logger.warning`); `utils.py:423` primary-face fallback logs via `logger.warning(..., exc_info=True)` — commit `e633293` |
| 6 | `watch_luts_dir` unwired; grader's `_lut_cache` never evicted on hot-reload | ⚠ **STILL OPEN** | `grep -rn watch_luts_dir` finds zero callers outside `lut.py`'s own definition and `test_lut_registry.py`. `grading.py`'s `_get_cached_lut` (`grading.py:929`) still has no mtime check or eviction hook. Unchanged — not addressed by any of the 15 new commits. |
| 7 | Unused imports | ✅ Still resolved | No regressions |
| 8 | `gui.py` at 1649 LOC strains thin-UI mandate | ⚠ Monitor | Now 1668 LOC (+19). `engine.py` 1735→1855, `grading.py` 1096→1401, `params.py` 1222→1337 — expected growth given 6 new grading features, but worth a consolidation pass if Phase 2 adds more panels. |
| 9 | `color_space` wide-gamut tradeoff | ℹ Unchanged | Still documented, still acceptable |
| 10 | Doc drift / `print`→`logger` | ✅ **CLOSED** | `lut.py` watcher loop now calls `logger.warning("lut watcher error: %s", exc)` — commit `e633293` |
| ✓ | Prior MAJOR LUT-cache race (§3.3) | ✅ Still resolved, **and now more accurate** | `LUTRegistry` and `ColorGrader._lut_cache` both lock-guarded, re-confirmed this cycle. Note: `LUTRegistry`'s lock guard was actually *implemented* in commit `e633293` (2026-06-30) — the 2026-06-28 report's §3.3 claim that both caches were already locked was **not accurate for `LUTRegistry` at the time it was written**, but the current state (post-`e633293`) genuinely matches what §3.3 describes. |

**§10 Retouch Fix Audit ("Fix 1–4"):**

| Fix | 2026-06-28 status | Status now | Evidence |
|---|---|---|---|
| 1. Impact `subject_mask` — no-face path bug | ⚠ Bug found | ✅ **CLOSED** | `engine.py:1085` (no-face `_no_face_fallback`) and `:1571` (`_stage_finish`) both now thread `person_mask` through to `add_impact_finish` — commit `e633293` |
| 2. Sharpen gate dead `else 1.2` branch | ⚠ Dead code found | ✅ **CLOSED** | `engine.py:1585` — branch removed, `amount = max(1.2, ...)` unconditional inside `if ctx.sharpen > 0:` — commit `e633293` |
| 3. Person mask blur | ✅ Clean | ✅ Unchanged | No further action needed |
| 4. Smooth mask erosion | ✅ Clean | ✅ Unchanged, and reapplied | The 3px erosion now also appears (again) in the **uncommitted** `perf_optimizations.py` diff reviewed in §11.4 — same fix, consistent with the committed history |

### 11.3 New Findings — Committed Feature Code (`adee825..HEAD`)

**RETRACTED — `overlay_blend`/`hard_light_blend` ×2 factor (`retouch/utils.py:126-164`).**
The audit claimed the ×2 factor was missing. **The actual source does include it.** Lines 141-142:
```python
multiply = 2.0 * base_f * layer_f / 255.0
screen = 255.0 - 2.0 * (255.0 - base_f) * (255.0 - layer_f) / 255.0
```
The hand-calc with actual code: `overlay_blend(128, 128)` → screen branch gives `255 - 2*127*127/255 ≈ 128.5`, which is correct near-identity for 50%-gray. Both functions are mathematically correct. The auditor's quoted formula was erroneous — not a bug in the code. **No action needed.** Tests could add a 50%-gray identity assertion for completeness, but this is not a defect.

**RETRACTED — `color_balance_lch` hue wraparound (`retouch/color_space.py:390-402`).**
The audit claimed `np.abs(h - 0.0)` was used directly without circular distance. **The actual source does compute the short-path distance** via `np.minimum` before the weight formula:
```python
d_red = np.minimum(np.abs(h - 0.0), 360.0 - np.abs(h - 0.0))
red_weight = np.where(c_red >= 0, 1.0 - d_red / 180.0, 0.0)
```
The file's `hue_range_mask` pattern is already reused here. The auditor's quoted snippet was erroneous. **No action needed.**

**MINOR — `split_tone_lch` linear hue interpolation (`color_space.py:336,342`).** Separately from the retracted MAJOR finding above, `split_tone_lch` does blend hue via `h * (1-w) + target_h * w` without circular short-path handling — this is a genuine edge case for hues straddling the 0°/360° seam. Low severity (dead code — not wired into engine/gui). `white_balance_lch` tint axis (`grading.py:1194`) is correctly implemented using `(delta + 180) % 360 - 180` — the audit's claim about it was also mistaken.

**MINOR — `engine.py` bypasses the params registry for 2 feature gates (`engine.py:1083-1084, 1152-1154, 1555, 1664-1666`).**
```python
if ctx.white_balance_kelvin != 6500 or ctx.white_balance_tint != 0.0:
...
if ctx.bw_channel_mixer_r != 30 or ctx.bw_channel_mixer_g != 59 or ctx.bw_channel_mixer_b != 11:
```
Every other `ProcessingContext` field in this file is seeded from `_DEFAULTS[name]` (`_DEFAULTS: Dict[str, Any] = {spec.name: spec.default for spec in PROCESSING_PARAMS}`, `engine.py:139`) — these four comparisons hardcode the literal values instead (`6500`, `0.0`, `30`, `59`, `11`), duplicated across both the face and no-face code paths. Confirmed the literals currently match `params.py`'s registered `ParamSpec` defaults for `white_balance_kelvin`/`white_balance_tint`/`bw_channel_mixer_r/g/b`, so there is no live bug — but if a future change to `params.py` alters any of these defaults, these four `!=` gates silently desync from the registry and the feature will incorrectly activate (or stay inert) with no error. AGENTS.md: *"All parameters in `retouch/params.py` registry. No hardcoded values elsewhere."*
**Action:** Replace the literals with `_DEFAULTS["white_balance_kelvin"]` etc. (one-line change ×4).

**MINOR — Redundant/under-covering CI workflows (`.github/workflows/ci.yml` vs `test.yml`).**
`318197e` added `ci.yml`, which triggers on the same `push`/`pull_request` events to `main` as the pre-existing `test.yml` — both now run on every push, duplicating CI minutes. More importantly, `ci.yml`'s "fast tests" job passes **~40 `--ignore` flags**, excluding the large majority of the test suite (all of `test_skin.py`, `test_engine.py`, `test_grading.py`, `test_cli*.py`, `test_lut_registry.py`, etc.) — leaving it exercising a small fraction of what `test.yml` and the local 1626-test suite cover. `test.yml` remains comprehensive (only excludes `test_cli.py`, run separately). Not a correctness bug, but confusing/misleading: a contributor glancing at `ci.yml`'s green check could believe far more is covered than it is.
**Action:** Either delete `ci.yml` (redundant with `test.yml` + `benchmarks.yml`) or trim its ignore-list back down now that model assets are available in CI.

**INFO — `.3dl` loader has no upper bound on `3DLUTSIZE` (`retouch/lut.py:248-411`).** A crafted file with an oversized `3DLUTSIZE` (correctly padded with matching triplet count) could force a very large `np.array` allocation. Not currently reachable — `load_3dl` has zero callers outside `lut.py`'s own module and its test file; not wired into `LUTRegistry.get()`, GUI, or CLI. Same class of gap pre-existed in `load_cube` before this diff. Bound the size (e.g. reject > 128) before exposing either loader to user-supplied files.

### 11.4 Uncommitted Working-Tree Review (`skin.py`, `perf_optimizations.py`, `style_library.py`)

This diff is **not yet committed**. Reviewed for correctness; all 1626 tests pass with it applied.

- **Real bug fix:** `SkinProcessor._build_dimensional_mask` previously returned a **hardcoded `np.zeros((200, 200))`** fallback when no region attribute matched (e.g. all `None`), regardless of the actual image size — a latent shape-mismatch bug on any image not exactly 200×200. The diff adds a required `shape` parameter and threads `img_bgr.shape[:2]` through every call site (`dodge_burn`, `restore_micro_texture`, the refactored `apply_localized_clarity`). Correct fix; all call sites and tests updated consistently.
- **Defensive `None` guards added** to `apply_dodge_burn`, `restore_micro_texture`, `apply_localized_clarity` (skip early if `regions`/`skin_mask` is `None`) — reasonable crash-prevention for callers where face parsing didn't produce regions.
- **`harmonize_neck`**: removed a constant Z-substitution term (`z_neck = lm[152].z * face_w`) from the neck-plane distance calculation, per an inline comment noting it was a "geometric flaw." Since pixels have no real depth, the old term added the same constant offset to every pixel's distance value — equivalent to silently shifting the gating threshold rather than modeling real depth. The simplified 2D (X,Y-only) projection distance is more defensible, but this is exactly the kind of pixel-math change to a Visual-Critical module that AGENTS.md requires a Visual QA pass for before shipping.
- **`style_library.py`**: the previously-silent `except Exception: resolved_version = "2.0"` now logs via `logger.warning(..., exc_info=True)` — closes a real observability gap (this file wasn't flagged in the 2026-06-28 report but the pattern matches finding #5's class of issue).

**Process finding (not a code defect):** `skin.py` is explicitly listed in AGENTS.md as Visual-Critical ("Any change touching `frequency.py`, `skin.py`, `grading.py`, `parsing.py`, `geometry.py` = Visual-Critical. Visual QA gates cannot be bypassed regardless of task size."). This audit is text/test-only — no reference-image diff was run against `docs/VISUAL_QA.md` gates for the `harmonize_neck` and mask-shape changes. **Do not merge this diff without a Visual QA pass**; "tests pass" alone is explicitly called out in AGENTS.md as insufficient for this class of change.

Visual QA checklist (required before merge):

```
- Reference images used (list file paths + brief description)
- Before/After diffs (lossless PNGs) for each reference image
- Quantitative metrics per image: RMS / SSIM / PSNR vs baseline
- Gate results per docs/VISUAL_QA.md (PASS/FAIL) with thresholds recorded
- Reviewer / QA owner + date
- Link to PR containing the working-tree diff and the attached artifacts
```

### 11.5 Cycle 3 Summary

| # | Severity | Finding | Location | Status |
|---|----------|---------|----------|--------|
| 11 | — | [**RETRACTED**] `overlay_blend`/`hard_light_blend` ×2 factor — source re-read confirms `2.0*` is present (utils.py:141-142), 50%-gray identity holds | `retouch/utils.py:126-164` | False positive — no action needed |
| 12 | — | [**RETRACTED**] `color_balance_lch` hue wraparound — source re-read confirms `np.minimum(..., 360-...)` is used (color_space.py:390-393); `white_balance_lch` tint claim also false (uses `(d+180)%360-180` at grading.py:1194) | `retouch/color_space.py:390-402`, `grading.py:1191-1196` | False positive — no action needed |
| 13 | RETRACTED | 4 hardcoded-literal registry-bypass gates instead of `_DEFAULTS[...]` | `engine.py:1083-1084,1152-1154,1555,1664-1666` | Retracted — current tree uses `_DEFAULTS[...]` at the relevant checks |
| 14 | MINOR | Redundant CI workflow with ~40-file ignore-list under-covering vs. `test.yml` | `.github/workflows/ci.yml` | Open |
| 15 | MINOR | Uncommitted `skin.py`/`perf_optimizations.py` diff has no Visual QA pass per AGENTS.md mandate | `retouch/skin.py` (working tree) | Open — process gap, block merge until QA'd |
| 16 | INFO | `.3dl` loader unbounded `3DLUTSIZE` allocation | `retouch/lut.py:248-411` | Open — not reachable from user input today |
| 6  | MINOR | (carried over) `watch_luts_dir` unwired, grader cache not evicted | `lut.py`, `grading.py` | Still open, unchanged |

### 11.6 Recommendations (Cycle 3, priority order)

1. **Run Visual QA** on the uncommitted `skin.py`/`perf_optimizations.py` diff per `docs/VISUAL_QA.md` before merging — required by AGENTS.md, not optional for Visual-Critical modules.
2. **No action needed for the registry-bypass note** — the current tree already uses `_DEFAULTS[...]` lookups at the relevant `engine.py` checks, so the finding is retracted.
3. **Consolidate or trim** `.github/workflows/ci.yml` — either delete it (redundant with `test.yml`) or remove the ~40-file ignore-list now that model assets are available.
4. Carried over from Cycle 2: **decide on LUT hot-reload** (§3.4/finding #6) — wire `watch_luts_dir`'s callback to also evict `ColorGrader._lut_cache`, or mark the watcher experimental in its docstring.

> **Note:** Recommendations 1–2 from the original audit (blend ×2 fix, hue-wraparound fix) have been retracted — both were false positives.

---

### 📡 Loop Signals
```
LOOP_SIGNAL { loop: verify,  iteration: 2, status: DONE,  delta: "deep algo verification ALL PASS (tonal/precision/color_space/skin_protect/style_transfer); LUT race confirmed RESOLVED", reason: "math correctness proven empirically; no CRITICAL/MAJOR", next: "report" }
LOOP_SIGNAL { loop: review,  iteration: 2, status: DONE,  delta: "MAJOR count 5→0; reprioritized to test-gaps + observability + watch_luts_dir wiring caveat", reason: "prior MAJORs either fixed (race) or downgraded (silent-except off critical path)", next: "final output" }
LOOP_SIGNAL { loop: verify,  iteration: 3, status: DONE,  delta: "full re-audit of 15 new-feature commits + uncommitted skin.py WIP; 1626 passed, 1 skipped (1627 collected), no regressions", reason: "scope covers all Phase 1.d LCH/blend/WB work since 2026-06-28 baseline", next: "report" }
LOOP_SIGNAL { loop: review,  iteration: 3, status: DONE,  delta: "found 2 new MAJOR math bugs (blend ×2 factor, hue wraparound) — both dead code; 1 dormant registry-bypass MINOR; CI redundancy MINOR; flagged uncommitted skin.py diff as missing mandatory Visual QA", reason: "new features shipped without empirical verification of their math, unlike the tonal/precision/style_transfer harness from Cycle 2", next: "fix MAJORs before wiring dead code into GUI; Visual QA the WIP diff before merge" }
```

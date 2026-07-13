# Fix Plan: `test_skin_locus_override.py` Teardown Deadlock (MediaPipe `FaceLandmarker` GC Finalizer Hang)

**Status:** ✅ DONE 2026-07-13 — implemented and verified. `tests/test_skin_locus_override.py` runs to completion (14 passed in ~11s), no teardown deadlock. Engine-backed tests routed through the module-scoped `engine` fixture; 2 tests added (`test_process_recipe_locus_end_to_end`, `TestTeardown.test_module_runs_to_completion`). Sibling `test_integration.py` baseline (3 passed) unaffected. CLAUDE.md mislabel (init abort → teardown hang) corrected.

## Summary

**Target: the `skin.locus` candidate** — specifically, the fact that `tests/test_skin_locus_override.py` cannot run to completion. This was chosen over the other Outstanding Fixes candidate ("LUT Hot-Reload Unwired") after investigation, for reasons below.

Two corrections to the premise in `CLAUDE.md`, both established empirically by the planning agent:

1. **The LUT hot-reload candidate is effectively a non-issue / wrong framing.** GUI wiring for LUT reload *already exists* (`gui.py:1957` button `reload_luts_btn` → `gui.py:778` `on_reload_luts()` → `get_registry().reload()`, wired at `gui.py:2171`). The product already supports LUT reload via an explicit "Reload LUTs" button. The only thing "unwired" is the specific `watch_luts_dir()` *filesystem-watcher daemon*, which is a deliberately-deferred **feature enhancement**, not a bug. It has no failing behavior and no failing test. Building it out is net-new feature work (a `watchdog`-style background thread integrated into the Gradio app lifecycle), not a bug fix, and it is genuinely off the critical path.

2. **The `skin.locus` problem is NOT a MediaPipe *initialization* abort.** CLAUDE.md says the test "aborts locally inside MediaPipe engine initialization." That is wrong. The tests **pass**, then pytest **hangs forever during teardown** because an unclosed MediaPipe `FaceLandmarker` is finalized by the garbage collector inside pytest's `unraisableexception` plugin, and its `__del__ → close()` deadlocks waiting on a `concurrent.futures` result from a serial-dispatcher thread that is shutting down. This is a real, reproducible defect that makes an entire test file unrunnable and taints the "resolved" claim — exactly the kind of scoped bug-fix-plus-tests target requested, and small (one test file, ~5 test methods refactored, budget well under 200 tests).

The wiring feature itself (`recipe_loader → build_context → ProcessingContext.skin_locus → _process_face_core → unify_hue_line(locus=...)`) is genuinely correct; the 6 non-process tests pass and a single `engine.process()` with a recipe locus runs clean. The defect is purely **test lifecycle hygiene**, and the fix makes the end-to-end integration test actually executable so the "resolved" claim becomes verifiable on the real user path.

## Root Cause

`tests/test_skin_locus_override.py` constructs `RetouchEngine()` directly inside test bodies (lines 85, 112, 131, 216, 231) and **never calls `engine.close()`** nor uses the engine as a context manager. `RetouchEngine.close()` (`retouch/engine.py:4068`) is what releases the MediaPipe `FaceLandmarker` via `self._detector.close()` → `FaceDetector.close()` (`retouch/detection.py:287-292`) → `self._landmarker.close()`.

Because the engine is never closed, the `FaceLandmarker` survives as garbage until the interpreter's GC finalizes it. Under pytest, the `_pytest.unraisableexception` plugin calls `gc_collect_harder()` at session unconfigure, which forces the finalizer to run at teardown. `FaceLandmarker.__del__` → `close()` → `serial_dispatcher.shutdown_aware_handler` → `mediapipe_c_utils.dispatcher_wrapper` → `concurrent.futures._base.result()` → `threading.wait()` blocks forever. Captured traceback (faulthandler, 40s):

```
Timeout (0:00:40)!
Thread 0x00000001f5fa5e80 (most recent call first):
  File ".../threading.py", line 312 in wait
  File ".../concurrent/futures/_base.py", line 440 in result
  File ".../mediapipe/tasks/python/core/mediapipe_c_utils.py", line 142 in dispatcher_wrapper
  File ".../mediapipe/tasks/python/core/serial_dispatcher.py", line 74 in shutdown_aware_handler
  File ".../mediapipe/tasks/python/vision/face_landmarker.py", line 3328 in close
  File ".../mediapipe/tasks/python/vision/face_landmarker.py", line 3353 in __del__
  File ".../_pytest/unraisableexception.py", line 33 in gc_collect_harder
  File ".../_pytest/unraisableexception.py", line 94 in cleanup
  ...
  File ".../_pytest/config/__init__.py", line 1131 in _ensure_unconfigure
```

The `.` before the timeout confirms the test **passed** — the hang is teardown-only.

## Investigation Findings

- **Reproduction:** `python3 -m pytest tests/test_skin_locus_override.py -q` never terminates (killed at 2 min).
- **Isolation:** The 6 tests in `TestBuildContextSkinLocus` + `TestSkinHueShiftTowardCustomLocus` (no `engine.process()`) pass in 5.58s. The hang is exclusive to the classes that construct `RetouchEngine()` and run `process()`.
- **Not init, not cumulative:** A standalone script constructing and processing through 5 fresh `RetouchEngine()` instances completes in ~1.2s each with no slowdown or hang. A single pytest process test prints `.` (pass) then hangs at teardown. So `process()` and engine init are healthy; only GC finalization of an unclosed landmarker deadlocks.
- **Fix proven empirically:** Wrapping the same workload in `with RetouchEngine() as engine:` and then forcing `gc.collect(); gc.collect()` (mimicking `gc_collect_harder`) does **not** hang. Independently, `tests/test_integration.py` — which already uses `with RetouchEngine() as engine:` — runs its process tests and exits cleanly under pytest (`3 passed ... in 8.71s`, RC OK, no teardown hang).
- **Established convention:** `tests/conftest.py:137` already defines a module-scoped `engine` fixture: `with RetouchEngine() as eng: yield eng` ("Shared RetouchEngine with guaranteed teardown"). `test_skin_locus_override.py` simply doesn't use it. This is the canonical pattern the fix should adopt.
- **Latent sibling risk (not in scope but worth flagging):** `tests/test_cosplay_moat.py` (lines 447, 468, 489, 510, 533) also constructs bare `RetouchEngine()`; it likely shares the same latent teardown hang and should be migrated in a follow-up.
- **Environment note:** `pytest-timeout` is NOT installed, so the fix must not rely on `@pytest.mark.timeout`.
- **Wiring correctness (data flow) is sound:**
  - `retouch/recipe_loader.py:94-96` (flatten→nested `skin.locus`) and `:181-182` (nested→flat `skin_locus`).
  - `retouch/engine.py:691-719` `build_context()` extracts `recipe_skin_locus = rec.get("skin", {}).get("locus")`, applies caller-override precedence, and sets `ProcessingContext.skin_locus` (dataclass field at `engine.py:214`).
  - `retouch/perf_optimizations.py:500-503` `_process_face_core()` passes `locus=ctx.skin_locus` into `skin.unify_hue_line(...)`.
  - `retouch/skin.py:1227-1277` `unify_hue_line()` auto-detects the locus only when `locus is None`.

## Implementation Steps

All changes are confined to `tests/test_skin_locus_override.py`. No production source changes are required (the `close()`/`__exit__` lifecycle already exists and works).

### Step 1 — Use the shared `engine` fixture for engine-backed tests

The module-scoped `engine` fixture in `tests/conftest.py` already provides a guaranteed-teardown engine. Refactor every test that currently constructs `RetouchEngine()` in its body to accept the `engine` fixture parameter instead, and delete the in-body `engine = RetouchEngine()` line.

Affected methods and current construction lines:
- `TestRetouchEngineProcessSkinLocus.test_process_with_recipe_locus` — line 85 (`engine = RetouchEngine()`). **Note:** this test never actually calls `engine.process()` — it only uses `build_context`. Simplest correct fix: **delete line 85 entirely** (the local `engine` is unused). Do *not* add the fixture here; it needs no engine.
- `TestRetouchEngineProcessSkinLocus.test_process_with_caller_locus_override` — line 112. Change signature to `def test_process_with_caller_locus_override(self, engine):` and delete line 112.
- `TestRetouchEngineProcessSkinLocus.test_process_no_locus_auto_detect` — line 131. Change signature to `def test_process_no_locus_auto_detect(self, engine):` and delete line 131.
- `TestNoRegression.test_natural_recipe_still_works` — line 216. Change signature to `def test_natural_recipe_still_works(self, engine):` and delete line 216.
- `TestNoRegression.test_skin_hue_unify_without_custom_locus` — line 231. Change signature to `def test_skin_hue_unify_without_custom_locus(self, engine):` and delete line 231.

Because the `engine` fixture is `scope="module"`, all tests in the file share one engine instance that is closed exactly once at module teardown via `__exit__` — deterministic, before any GC-forced finalization.

### Step 2 — Leave the non-engine test classes untouched

`TestBuildContextSkinLocus` and `TestSkinHueShiftTowardCustomLocus` construct no engine (they use `build_context` / `SkinProcessor` directly). Do not modify them.

### Step 3 — Preserve the existing model-availability skip

The shared `engine` fixture already `pytest.skip(...)`s when `models/face_landmarker.task` is missing. By adopting the fixture, the process tests inherit that skip automatically (a strict improvement — today they'd error/hang on a machine without the model). No extra guard needed.

### Step 4 (optional hardening, recommended) — add a defensive session-teardown GC guard in conftest

To protect against *any* future test that forgets to close an engine, optionally add an autouse, session-scoped fixture in `tests/conftest.py` that runs `gc.collect()` **before** pytest's own unraisable-exception collector, or that closes tracked engines. This is optional and lower priority; the primary fix (Step 1) fully resolves the reported defect. If added, keep it minimal and document why (MediaPipe finalizer deadlock). Do not make the suite depend on `pytest-timeout` (not installed).

## Tests to Add

The goal is not net-new coverage of the wiring (already covered) but a **regression guard that the file runs to completion without a teardown deadlock**, plus making the previously-unrunnable end-to-end path actually assert on real behavior.

Add to `tests/test_skin_locus_override.py`:

1. **`TestRetouchEngineProcessSkinLocus.test_process_recipe_locus_end_to_end(self, engine)`**
   - Purpose: exercise the *real* user path the 22 narrow tests bypassed — a recipe carrying `skin.locus` driven through `engine.process()`, not just `build_context`.
   - Contents: build `resolve_recipe("natural")`, inject `{"skin": {"locus": {"h_target": 30.0, "C_target": 0.05}}}`, call `engine.process(img, recipe=<that recipe or "natural" + skin_locus kwarg>, skin_hue_unify=60.0)` on the synthetic warm-skin image.
   - Asserts: `result is not None`; `result.shape == img.shape`; `result.params.skin_locus["h_target"] == 30.0` (confirms the locus survived the full pipeline into result params, closing the gap where the old recipe-locus test only checked `build_context`).

2. **`TestTeardown.test_module_runs_to_completion(self, engine)`** (a tiny sentinel)
   - Purpose: a trivial engine-backed test placed so the module always has at least one fixture-scoped engine test; combined with running the whole file under a wall-clock check in CI, it guards against reintroducing bare `RetouchEngine()` teardown hangs.
   - Contents/assert: `assert engine is not None` and one `engine.process(small_img, recipe="natural")` returns a valid array. Keep it minimal.

No changes needed to the 6 already-passing non-engine tests. Total tests touched/added: ~7 methods edited + 2 added — well within the 50–200 budget.

## Verification

Run these commands from the repo root (`/Applications/htdocs/retouch`):

1. **Primary — the file now terminates and passes:**
   ```bash
   python3 -m pytest tests/test_skin_locus_override.py -q -p no:cacheprovider
   ```
   Success = process completes in seconds (not a 2-minute kill), reports all tests `passed` (or `skipped` if the model file is absent), and the shell **returns to prompt** (no teardown hang).

2. **Belt-and-suspenders — assert no teardown hang with faulthandler:**
   ```bash
   python3 -c "import faulthandler; faulthandler.dump_traceback_later(90, exit=True); import pytest, sys; sys.exit(pytest.main(['tests/test_skin_locus_override.py','-q','-p','no:cacheprovider']))"
   ```
   Success = prints the pass summary and exits **without** any `Timeout (...)!` / `FaceLandmarker.close` traceback.

3. **No regression in the sibling engine test file:**
   ```bash
   python3 -m pytest tests/test_integration.py -q -p no:cacheprovider -k "process or engine"
   ```
   Success = `3 passed` (matching the current baseline), clean exit.

4. **Confirm the wiring assertion actually fires** (guards against a silently-skipped model): if `models/face_landmarker.task` exists, `test_process_recipe_locus_end_to_end` must report `passed`, not `skipped`. Check with:
   ```bash
   ls models/face_landmarker.task && python3 -m pytest tests/test_skin_locus_override.py -q -p no:cacheprovider -rs
   ```

## Risks

- **Model-file dependence:** On machines lacking `models/face_landmarker.task`, all engine-backed tests will `skip` (via the fixture's existing guard). That is correct behavior, but reviewers must not mistake "all skipped" for "verified." Verification step 4 explicitly checks the tests run (not skip) when the model is present.
- **Module-scoped fixture sharing:** The `engine` fixture is `scope="module"`, so all engine tests in the file share one instance. If any future test mutates engine state, it could bleed across tests. The current tests are read-only w.r.t. engine state (they pass images and read results), so this is safe today; note it for future additions.
- **`test_hue_shift_direction` is heuristic and flaky-prone:** its assertion (line 207) already includes a `< 5.0` tolerance fudge. Not in scope to fix, but do not tighten it as part of this change; leaving it as-is avoids introducing a flaky failure.
- **Sibling files still latent:** `tests/test_cosplay_moat.py` retains bare `RetouchEngine()` constructions and likely has the same teardown hang if run in isolation. This plan does **not** fix it. Recommend a fast-follow ticket to migrate it (and any of the 13 files grepped) to the shared `engine` fixture. Do not silently expand this fix into those files without separate review.
- **Do not "fix" this by adding `pytest-timeout` marks:** the plugin is not installed; relying on it would make the suite non-portable and would only *mask* (kill) the hang rather than resolve it.
- **CLAUDE.md accuracy:** the "✅ RESOLVED 2026-07-03: `skin.locus` recipe override wiring" note is misleading (says init abort; it's a teardown hang). Updating that prose is a `docs` concern outside this code fix, but flag it so the record matches reality once the test actually runs.

## Suggested Commit Message

```
test(skin-locus): fix MediaPipe teardown deadlock in skin_locus override tests

test_skin_locus_override.py constructed RetouchEngine() directly in test
bodies and never closed it, leaving the MediaPipe FaceLandmarker to be
finalized by GC. Under pytest's unraisableexception gc_collect_harder(),
FaceLandmarker.__del__ -> close() blocks forever on a serial-dispatcher
future, hanging the whole file at teardown (previously mislabeled as a
MediaPipe init abort).

Route the engine-backed tests through the existing module-scoped `engine`
fixture (conftest.py) that guarantees `with RetouchEngine()` teardown, drop
the unused engine construction in the build_context-only test, and add an
end-to-end assertion that a recipe-supplied skin.locus survives
engine.process() into result.params. The file now runs to completion.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
```

*(Type `test` because all changes are confined to test files; if the optional conftest GC-guard in Step 4 is included, keep it `test` as it still only touches test infra.)*

## Critical Files for Implementation

- `/Applications/htdocs/retouch/tests/test_skin_locus_override.py` — the only file to edit — refactor engine-backed tests onto the shared fixture, add end-to-end assertion
- `/Applications/htdocs/retouch/tests/conftest.py` — source of the module-scoped `engine` fixture at line 137; optional session GC guard
- `/Applications/htdocs/retouch/retouch/engine.py` — `RetouchEngine.close`/`__enter__`/`__exit__` at 4068-4080; `build_context` skin_locus wiring at 691-719 — reference only, no change
- `/Applications/htdocs/retouch/retouch/detection.py` — `FaceDetector.close` at 287-292 that releases the MediaPipe landmarker — reference only
- `/Applications/htdocs/retouch/retouch/perf_optimizations.py` — `_process_face_core` passing `locus=ctx.skin_locus` at 500-503 — reference for the end-to-end assertion, no change

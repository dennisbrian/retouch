# Tier P Plan — Performance, Architecture & Distribution (Stages P1–P4)

## Context

Fourth planning round. Previous plans cover looks (`PLAN_MOONLIGHT_PORCELAIN.md`), quality foundation + workflow (`PLAN_TIER1_FOUNDATION.md` F1–F3), manual tools (`PLAN_TIER2_MANUAL_TOOLS.md` F4–F7), and fidelity + intelligence (`PLAN_TIERQ_FIDELITY_INTELLIGENCE.md` F8–F11). This round plans the three system-level dimensions none of those cover: **speed, code architecture, and shipping to users**.

**Benchmark reality (from `benchmark_results.json`, 2026-06-23):** at 400×400, per-face = **702ms of a 701ms total — 99% of all processing time**. Grading is 3.5ms, everything else sub-2ms. There is exactly one bottleneck in this system: the per-face smooth path, and inside it the `cv2.bilateralFilter` at `retouch/frequency.py:224` (with `d=-1`, kernel size grows with `sigma_space`, which grows with smooth strength).

**⚠️ Critical sequencing finding:** Tier Q's F8 moves per-face processing to **native-resolution face crops**. A 1500px face crop has ~14× the pixels of the 400px benchmark — the bilateral's cost scales with pixels × kernel area, so F8 without P1 turns 0.7s/face into **many seconds per face**. **P1 must land before or with F8.** (Recommended order updated at the bottom.)

Existing assets this plan leverages: hand-rolled self-guided filter already written and tested in `SkinProcessor.flatten` (`retouch/skin.py:556-583`); `FaceProcessorPool` multi-face parallelism + Numba JIT with graceful fallback (`retouch/perf_optimizations.py`); CI already runs (`.github/workflows/ci.yml`, `test.yml`, `benchmarks.yml`); PyInstaller + PyWebView desktop app already builds (`build_app.sh`, `desktop.py`).

---

## Stage P1 — Kill the bilateral bottleneck (~3–5 days, highest ROI in the codebase)

**Approach:** replace `cv2.bilateralFilter` in `FrequencySeparator.combine` with a guided filter — same edge-preserving character, but box-filter-based: **O(N) regardless of smoothing radius**, typically 4–8× faster, and it scales flat with F8's bigger crops.

### Steps
1. Extract the self-guided filter math from `SkinProcessor.flatten` (`skin.py:556-583`) into a shared `guided_filter(guide, src, radius, eps)` in `retouch/utils.py` (or `perf_optimizations.py`); refactor `flatten` to call it (behavior-identical — its tests must pass unchanged).
2. In `frequency.py:combine`: swap bilateral for guided filter on `low + mid_original`. Map parameters: `sigma_space` → radius (`r ≈ sigma_space`), `sigma_color` → eps (`eps ≈ sigma_color²`) — tune on the corpus. Keep the existing gaussian hybrid blend and crop-to-mask logic untouched.
3. Keep bilateral available as `smooth_engine="bilateral"` (ctx field, not a GUI slider) for A/B regression only — remove after one release.
4. Numba is already wired with fallback — if the pure-numpy guided filter needs it for the box-filter chain, `@numba.jit` is free to use; likely unnecessary (cv2.blur is SIMD).

### Verification / QA
- Quality gate: SSIM ≥ 0.98 vs bilateral output across the corpus at smooth = 30/60/90; visual A/B on skin gradients (banding) and jaw/hair edges (the failure mode where guided beats bilateral is gradient reversal — check for halos anyway).
- Perf gate in `benchmarks.yml`: per-face 400×400 median ≤ 250ms (from 702ms); add a 1500×1500 crop benchmark now (the F8 case) — target ≤ 1.5s.
- Full pytest; `test_skin.py` flatten tests unchanged.

**Files:** `retouch/frequency.py`, `retouch/skin.py`, `retouch/utils.py` (or `perf_optimizations.py`), `benchmark.py`, tests.

---

## Stage P2 — Full-res performance guards (companion to F8, ~1 week, do together)

F8 makes global stages run at native res (6000×4000 float32 = **288MB per buffer copy**). Guards:

1. **Downsample-compute / full-res-apply** for every large-radius blur in Phase 3 (bloom, glow, orton, haze, vignette): compute the low-frequency layer at ≤1200px, upsample, composite full-res. Mathematically safe for blurs (unlike downsampling the result — the F8 lesson). The `flatten` >1200px guard (`skin.py:562-568`) is the in-repo pattern.
2. **Buffer discipline:** audit Phase 3 for unnecessary `.copy()` / temporaries; prefer in-place (`out=`) numpy ops in the float pipeline (coordinates with F1); peak-RSS assertion in benchmarks (≤ 3× image buffer for the global phase).
3. **Multi-face at native res:** verify `FaceProcessorPool` handles native-res crop payloads (pickling 1500px crops across processes — measure; if IPC dominates, switch pool to threads since cv2 releases the GIL).
4. **Stage time budgets in CI:** `benchmarks.yml` already runs — add per-stage budget assertions (soft-fail warnings) so regressions surface in PRs, not user reports. Budgets from a fresh baseline run at 2K/4K/6K.

**Files:** `retouch/engine.py`, `retouch/grading.py` (blur guards), `retouch/perf_optimizations.py`, `benchmark.py`, `.github/workflows/benchmarks.yml`, tests.

---

## Stage P3 — Stage-registry architecture (~1.5 weeks, incremental, zero behavior change)

`engine.py` is 2041 lines with a hardcoded stage order. Every upcoming feature wants a hook point: F2 per-stage opacity/bypass, F11 QA probes, F4/F7 pre-pipeline steps, Tier 3 plugin API. Formalize once:

1. **`retouch/stages.py` (new):** `PipelineState` dataclass (img float32, ctx, masks bundle, timings, qa) + `Stage` protocol: `{name: str, phase: "pre"|"face"|"global"|"grade"|"finish", enabled(ctx) -> bool, run(state) -> state}`.
2. **Migrate mechanically, phase by phase**, starting with Phase 3 (the existing `_stage_*` methods are already stage-shaped — wrapping them is renaming, not rewriting). Per-face block last (it's the `_process_one_face` composite; can stay one mega-stage initially).
3. **Registry = ordered list** built in `RetouchEngine.__init__`; `process()` becomes a fold over stages. Timing collection moves into the runner (deletes ~40 lines of boilerplate).
4. **Golden-output harness first, migrate second:** before touching anything, snapshot output hashes for every built-in recipe × corpus image; every migration commit must be **byte-identical**. This is the whole safety story — cheap to build, non-negotiable.
5. Payoffs wired immediately: per-stage `opacity`/`bypass` dict on ctx (F2's stage mixer becomes trivial), and a `hooks: list[callable]` slot (F11 QA probes, future plugins).

**Files:** `retouch/stages.py` (new), `retouch/engine.py` (shrinks substantially), `tests/test_golden_pipeline.py` (new), existing tests untouched.
**Risk control:** pure refactor, small commits, golden harness gates each one. Pause-able at any phase boundary — partial migration is still a win.

---

## Stage P4 — Distribution hardening (~1 week, anytime before sharing builds)

The app already builds (`build_app.sh` → PyInstaller .app with PyWebView shell). Gaps between "builds on your Mac" and "shippable":

1. **Model acquisition UX:** `models/manifest.json` (name, url, sha256, size, license) + `retouch/model_fetch.py`: download-on-first-use with progress in GUI + checksum verify. Required before F4.b/F7 (LaMa ~200MB, SR models — can't bundle). `luts/ACQUISITION.md` is the documentation precedent.
2. **macOS signing + notarization** in `build_app.sh` (codesign + notarytool; needs your Developer ID) — without it, users get Gatekeeper-blocked.
3. **Version + update check:** embed version string (single source in `retouch/__init__.py`); on launch, non-blocking check against GitHub releases; "update available" toast. No auto-download.
4. **Diagnostics:** rotating log file in `~/Library/Logs/ProMaxRetouch/`; "Copy diagnostics" button in GUI (versions, providers from `build_ort_providers()`, last error) — makes remote debugging of user reports possible.
5. **Windows build validation:** run the PyInstaller path on Windows (DirectML provider already handled in `build_ort_providers`); document in `BUILD.md`. Timebox — if PyWebView misbehaves, ship "open in browser" mode there.

**Files:** `models/manifest.json` (new), `retouch/model_fetch.py` (new), `build_app.sh`, `desktop.py`, `gui.py`, `retouch/__init__.py`, `BUILD.md` (new), tests for manifest/checksum logic.

---

## Master sequencing

> ⚠️ **Superseded (2026-07-02, later the same day):** authority moved to **`MASTER_PLAN.md`**, which restructures all 32 stages (F/P/T/S/A tiers) into priority phases and resolves the Tier S / Tier A placements. The sequence below predates the skin tiers and is kept for historical context.

### (historical) All tiers as of Tier 3 planning

```
P1 (bilateral→guided) ──► F8 (full-res) + P2 ──► F1 (float32/16-bit) ──► F11 (self-QA) ──► P3 (registry)
     3-5 days                 ~3 wks                  ~1.5 wks              ~1.5 wks          ~1.5 wks
──► F2/F3 (sessions, brushes) ──► F4–F7 (heal, liquify, look-extract, denoise) ──► F9/F10 (adaptive)
──► T1–T5 (background replace, plugins+cookbook, makeup v2, RAW develop, body reshape) ──► P4 (ship it)
```

Why this order: P1 unblocks F8 (see critical finding). F8+P2+F1 set the quality floor. F11 locks it in with detectors. P3 right before F2 because F2's stage-mixer is trivial on the registry. Tier 3 (`PLAN_TIER3_CREATIVE_ECOSYSTEM.md`) rides on everything before it: T1 needs the mask/QA guards, T4 needs P3's registry, T5 needs F1's float pipeline, T3 needs the model-fetch piece of P4. P4 floats — schedule when you first want to hand a build to someone else; its model-fetch piece must precede F4.b/F7/T3.

## Verification (end-to-end)
1. P1: benchmark gates (≤250ms/face @400px) + SSIM ≥ 0.98 corpus check + full pytest.
2. P2: peak-RSS + per-stage budget assertions green at 2K/4K/6K in CI.
3. P3: golden-output harness byte-identical across every migration commit; final `engine.py` line count and stage list documented in `ARCHITECTURE.md`.
4. P4: fresh-machine install test (no dev tools): download app → first launch fetches models with progress → process a photo. Gatekeeper passes.

## Open items carried forward
- `anime_crystal_void` 7 dead keys — fix vehicle now designated: Tier 3 Stage T1 (`PLAN_TIER3_CREATIVE_ECOSYSTEM.md`) wires them as real background-stage params.
- ~~Recipe cookbook / plugin API / background-replace unplanned~~ — planned 2026-07-02 as Tier 3 Stages T1–T5 in `PLAN_TIER3_CREATIVE_ECOSYSTEM.md`.

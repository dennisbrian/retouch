# F8 Execution Plan — Full-Resolution Fidelity, staged slices

**Date:** 2026-07-03
**Parent:** `PLAN_TIERQ_FIDELITY_INTELLIGENCE.md` §F8 (finding + approach), `MASTER_PLAN.md` Phase 1 (F8+P2).
**Prerequisite:** P1 ✅ (guided filter, committed 10bab48 — bilateral cost bomb defused).
**Code ground truth (verified 2026-07-03):** `PROXY_MAX_DIM = 2048` (`engine.py:125`); `_process_with_proxy` (`engine.py:929-956`) downscales INTER_AREA → runs `_run_core_pipeline` (stages 0–6) entirely at proxy → `_upscale_core_result` (`engine.py:958-980`) INTER_LINEAR-upscales the finished image + 5 masks. Per-face work happens inside the core pipeline via `_process_face_core` (`perf_optimizations.py:137`).

**Strategy: three slices, each independently shippable, ordered by value/risk.** Slice F8.0 is a ~40-line quick win recovering most visible texture loss; F8.1 and F8.2 are the real architecture. This avoids a 2-week big-bang.

---

## Slice F8.0 — Detail reinjection (quick win, ~1 day Haiku work) 🟢 ship first
After `_upscale_core_result`, non-skin regions (fabric, jewelry, hair, background) have lost their native high band for no reason — nothing was done to them that required resampling.
1. In `_process_with_proxy`, keep a reference to the **original native `img_bgr`** before downscaling.
2. After upscaling: `high_band = original − gauss(original, σ≈2·(1/proxy_scale))`; reinjection weight `w = 1 − acc_skin_upscaled × (smooth_strength/100)` (skin keeps its retouched smoothness; everything else regains native texture). `result += high_band × w × k` with k≈0.85, clipped.
3. Guard: only when `proxy_scale < 1.0`; `fast=True` preview path unchanged; opt-out ctx flag `detail_reinject=1` default on.
4. Tests: 3000px synthetic with fine checker texture outside a fake skin mask → ≥90% high-band energy retained (vs ~40% today); skin region unchanged vs current output (±2 levels); no-proxy images byte-identical.
5. Risk: halos at skin boundary → weight uses the *feathered* acc_skin, and k stays <1.
**Files:** `retouch/engine.py`, tests. **This alone fixes most of the "prints look soft" problem while F8.1/2 are built.**

## Slice F8.1 — Native-res global phases (~1 week)
Phase 3+ (tonal, grading, sharpen/finish) runs on the native image instead of the proxy:
1. Restructure `_run_core_pipeline` into `_run_detection_and_faces(proxy)` → returns composited proxy result + masks, and `_run_global_phases(img, ctx, masks)` — then in `_process_with_proxy`: detection+faces at proxy, **paste upscaled face-region result onto native original** (face-bbox-limited composite: only face ROIs are proxy-res, since only they were retouched), masks upscaled, then global phases at native res.
2. P2 guards go in with this slice (they're why it's safe): downsample-compute/full-res-apply for every large-σ blur in grading/bloom/glow (grep for GaussianBlur with radius ∝ image size); peak-RSS assertion in benchmark at 4K/6K.
3. Tests: golden-consistency at ≤2048px (byte-identical — no proxy, no change); 4K image: background texture native-sharp pre-grade; per-stage time budget vs baseline ≤2.5×.
**Files:** `retouch/engine.py`, `retouch/grading.py`, `benchmark.py`, tests.

## Slice F8.2 — Native-res face crops (~1 week, the F8 finale)
Faces stop being proxy-res islands:
1. Detection/parsing stay at proxy; face bboxes/landmarks scale back to native (MediaPipe landmarks are normalized — remap is multiplication); BiSeNet region masks upscale per-face (they're smooth).
2. `_process_one_face`/`_process_face_core` receive native-res crops. Audit per-face ops for hardcoded pixel constants (kernels already scale via `estimate_face_width`/ied; the `flatten` max_dim guard is the pattern; relight already coarse-scale ✅ post-v2).
3. Multi-face perf: verify `FaceProcessorPool` with 1500px crops (IPC cost — switch to threads if pickling dominates; cv2 releases the GIL).
4. Benchmarks: 400px face ≤250ms (P1 gate, holds), 1500px face crop target ≤1.5s; 6K end-to-end ≤2.5× current.
5. `quality="draft"` ctx flag keeps the old all-proxy path for batch contact sheets.
**Files:** `retouch/engine.py`, `retouch/perf_optimizations.py`, `benchmark.py`, tests.

## Order vs Phase 2 queue
Owner is currently driving Phase 2 (skin/color quality) — fine: **F8.0 can slot into any gap (1 day)**; F8.1/F8.2 should wait until the Phase 2 queue (`PLAN_PHASE2_EXECUTION.md` Q1–Q4) lands, then run as the next big block. Rationale: Q1–Q4 are small and owner-visible daily; F8.1/2 touch engine plumbing and deserve an uninterrupted window + the golden-output harness idea from P3 (snapshot hashes before restructuring `_run_core_pipeline`).

## Verification (end-to-end, from parent plan)
6000px photo, 100% crop of fabric/jewelry/hair: native detail visibly preserved; high-band energy assertion ≥95% non-skin (F8.2); mask remap roundtrip ±1px; full pytest + benchmark deltas recorded at 2K/4K/6K.

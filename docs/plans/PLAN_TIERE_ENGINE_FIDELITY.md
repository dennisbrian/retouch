# Tier E Plan — Engine & Skin Fidelity Audit (Stages E1–E4)

**Date:** 2026-07-03
**Type:** Code-audit research + stage definitions (no code). Companion to `PLAN_TIERH_HAIR.md` (same session's planning frontier).
**Question:** Tier H covered the hair gap; what does a deep read of the *existing* core (`engine.py`, `skin.py`, `perf_optimizations.py`, `frequency.py`) reveal that no current plan covers?
**Method:** full read of all four modules (verified 2026-07-03, HEAD = `ea6f835`); every claim below carries a file:line anchor.

---

## Part 1 — Audit findings

### 1.1 🔴 The quantization ledger (flagship finding — the per-face chain is 8-bit Swiss cheese)

**Per-face chain (Stage 2):** every skin op takes a uint8 BGR canvas, converts to its working space (LAB float / LCh / OKLab), operates, and re-quantizes to uint8 via `blend_masked` (`utils.py:200-223` — always returns uint8). With a `porcelain_unified_v1`-class recipe a skin pixel passes through **~7–10 uint8 quantizations before Phase 3 even starts**: `frequency.combine` (uint8 out, `frequency.py:263-266`) → `equalize` (LAB, `skin.py:133`) → `unify_hue_line` (OKLab, `skin.py:696`) → `whiten` (LAB, `skin.py:63`) → `harmonize_neck` (LAB, `skin.py:239`) → plus flatten/quantize/bloom/dodge_burn/local_clarity when active (`perf_optimizations.py:280-396`). Each round-trip adds ±0.5-level truncation noise and one full-crop `cvtColor` + two `astype` passes. On the smooth porcelain gradients this project optimizes for (牛奶皮), accumulated 8-bit steps are exactly the banding class F11 is being built to detect — we manufacture upstream what we plan to detect downstream.

**Phase 3 "float32" stage is largely nominal.** `_run_core_pipeline` converts to float at `engine.py:1064` per F1, but `_stage_grade` then punches back down to uint8 at **16 sites** via `_to_uint8_if_float` (`engine.py:1655-1815`): style transfer, color_ref transfer, white balance, master HSL, tonal curve, grade_stack, 3-way split tone, highlight rolloff, skin_glow diffusion, bloom, glow, white-costume lift, vignette, film grain, negative split tone, B&W mixer. Only contrast/brightness/tonal/sharpen and `grade()` itself are float-native. Worse, three "float" helpers are fake: `_F_adjust_vibrance` (`engine.py:1904-1914`) and `_F_apply_uniform_saturation` (`engine.py:1927-1936`) internally round-trip through uint8, and `_stage_subject_separation` does uint8-LAB even on float input (`engine.py:1548-1561`). And the **no-face path gets zero float benefit at all**: `_no_face_fallback` is called before `to_float` (`engine.py:1024` vs `engine.py:1064`), so `--global-only` batch runs are 100 % uint8-chained.

**Consequence:** MASTER_PLAN's Phase-1 milestone claim ("float32 global pipeline") is currently ~30 % real. The 16-site list above *is* the F1 work inventory — F1's estimate should be checked against it.

### 1.2 🔴 Registry-bypass bug class has a live successor (report, not fixed — standing rule)

- The WB-literal bug in CLAUDE.md's "Outstanding Fixes" **is already fixed** — both sites use `_DEFAULTS` (`engine.py:1144`, `engine.py:1668`). CLAUDE.md is stale.
- **The same class lives on in the B&W channel mixer:** `ctx.bw_channel_mixer_r != 30 / != 59 / != 11` hardcoded at `engine.py:1212-1216` *and* `engine.py:1802-1806`, while `params.py:951/961/971` define defaults 30/59/11. Identical latent desync: currently matching, silently breaks if params.py changes. Fix is the same one-liner pattern as the WB fix, in two places.

### 1.3 🟠 Response-curve defects (sliders that lie)

- **Sharpen slider is inert from 1–60:** `amount = max(1.2, ctx.sharpen / 100.0 * 2.0)` (`engine.py:1839`) — any value 1–60 yields amount 1.2 (and 0→1 is a cliff from off to strong). The slider only responds above 60.
- **Equalize responds quadratically:** the a/b pull is scaled by `s` internally (`skin.py:167`) and the final blend is scaled by `s` again (`skin.py:184`) — effective chroma evening ≈ 0.20·s². Mid-slider is much weaker than it reads. Matters for A2 tuning sweeps and for interpreting Q1's σ_C metric vs slider position.

### 1.4 🟠 C1-era interaction traps (directly relevant to tomorrow's Q4)

- **`harmonize_neck` gate omits the C1 params:** it only runs when `ctx.whiten != 0 or ctx.equalize > 0` (`perf_optimizations.py:344`), with strength `max(abs(whiten), equalize)`. `skin_hue_unify`/`skin_chroma_even` don't trigger or scale it. Today `porcelain_unified_v1` survives via `rosy: 0.15` (`recipes.py:125`), but Q4's plan (equalize→0, whiten→hue-stable) walks straight toward a recipe where the face is C1-unified and the neck silently isn't. Gate should include the C1 params when C1 becomes the primary path.
- **`unify_hue_line` eligibility gates are hard binary steps:** `(C <= 0.18)`, `(L >= 0.25)`, `(L <= 0.95)` (`skin.py:750-753`). On smooth skin gradients these thresholds are iso-contours — one side gets the full hue rotation, the other none → potential visible seams at 100 % zoom as C1 strengths rise. Every comparable mask in skin.py uses soft ramps (`_get_highlight_protection`, shadow_protection). Fix is ~4 lines of smoothstep per gate.

### 1.5 🟡 Smaller findings

- **Smooth mask is eroded 3 px twice:** `_build_smooth_mask` erodes on every call (`perf_optimizations.py:66-67`) and is called twice per face (`perf_optimizations.py:201`, `:216`). At proxy-res small faces (ied ≈ 40 px) the doubled shrink leaves a visible unsmoothed rim at the skin boundary. Intent unclear — one erosion may be intended.
- **Dead Numba kernels, live warm-up cost:** `_apply_tonal_lut` and `_blend_highpass` (`perf_optimizations.py:675-741`) have **zero production callers** (grep-verified); only `warmup_jit_kernels` exercises them, and it runs at every `RetouchEngine()` init (`engine.py:567`). Pure startup cost for unused code — remove or wire.
- **FaceProcessorPool payload bloat:** the full `ctx` is pickled per face (`engine.py:1337-1351`), including `color_ref` when set — a full-size reference image × N faces over IPC. Also `max_workers = os.cpu_count()` (`perf_optimizations.py:540`) vs the documented 4-worker policy.
- **Face and no-face paths render the same recipe in different op order:** no-face applies tonal_curve *before* WB and impact *before* grain/negative-split/B&W (`engine.py:1140-1223`); the face path applies tonal_curve after HSL and impact dead-last after sharpen (`engine.py:1700-1849`). Duplicated ~80-line chain, already diverged — this is the concrete evidence P3 (stage registry) has been waiting for.
- **`blemish.remove` runs *after* smoothing** (`perf_optimizations.py:336`) — pro workflow (and Retouch4me Heal) heals first so detection sees unsmoothed anomalies and inpainting isn't fed pre-blurred context. Not asserted as wrong — flag for an A1-corpus A/B (order swap is trivial to test once the harness exists).

### 1.6 Stale documentation found (fixed in this session's doc updates)

- CLAUDE.md "Outstanding Fixes": WB literals → fixed; `_build_dimensional_mask` WIP → merged (shape param present at `skin.py:31-61`, working tree clean).
- `PLAN_PHASE2_EXECUTION.md` cites **`PLAN_SKINPY_UPGRADE.md` which does not exist in the repo** (grep-verified) as the Q2/Q3 blueprint with "exact anchors". Anchors re-derived and verified in this audit: S2 call site = before `if ctx.equalize > 0:` at `perf_optimizations.py:298`; `face_width` in scope from `perf_optimizations.py:170`; `_get_highlight_protection` now at `skin.py:777` (not 709 — S1 shifted it); OKLCh imports Q4 needs already present at `skin.py:14-22`.

## Part 2 — Stages

### Stage E1 — Float-resident per-face core (~1 week) 🔴 the fidelity unlock
Convert the ROI canvas to float32 LAB **once** at the top of `_process_face_core`, keep every skin op float-LAB in/out (ops that need OKLab/LCh convert from float LAB, not from uint8 BGR), single uint8 conversion at the end. Kills ~7–10 quantizations *and* ~10 full-crop `cvtColor`+`astype` passes per face (perf win rides free). Migration: adapter pattern — ops accept either dtype, migrate one op per commit, golden-image gated (same discipline as P3). Natural window: alongside F8.1/F8.2, which restructure the same code.

### Stage E2 — F1 hole-closing inventory (fold into F1, not a new stage)
The 16 `_to_uint8_if_float` sites + 3 fake-float helpers + the uint8 no-face path (§1.1) become F1's checklist. Each site migrates to a float-native implementation or gets an explicit "inherently uint8 (cv2 LUT path)" annotation. Acceptance: a synthetic smooth-gradient image through a max-stack recipe shows zero new banding steps vs single-op baseline (F11's banding detector, once built, automates this).

### Stage E3 — Response-curve & gate repairs (~2–3 days, can ride the Phase 2 queue)
1. Sharpen mapping: replace `max(1.2, …)` with a monotone map over the full 0–100 range (`engine.py:1839`).
2. Equalize linearization: decide intended response (document s² or make it s), then **re-tune affected recipes** — every recipe with `equalize` shifts if the curve changes; needs Q1's metrics first.
3. `unify_hue_line` smoothstep gates (§1.4) — natural Q4 rider.
4. `harmonize_neck` gate extended to C1 params (§1.4) — should land with or before Q4's recipe change.
5. B&W mixer literals → `_DEFAULTS` (§1.2) — the same commit pattern as the WB fix.

### Stage E4 — Pool & dead-code hygiene (~1–2 days, anytime)
Remove or wire the dead Numba kernels (+ drop warm-up if removed); slim FaceProcessorPool payloads (strip `color_ref`/`face_contexts` from the pickled ctx); align `max_workers` with policy; decide single vs double erosion in `_build_smooth_mask` (visual A/B on a small-face crop).

## Sequencing & placement

```
E3 + E4 (small, ride Phase 2 gaps) ──► E2 (≡ F1, Phase 1 slot) ──► E1 (with F8.1/F8.2 window)
```
Nothing here blocks the Phase 2 queue; §1.4 items are the only ones with a *deadline flavor* (they interact with Q4). **Adopted into MASTER_PLAN.md 2026-07-03:** E3.3/E3.4 land inside the Q4 slice; E3.5 + E4 are Phase 2 gap fillers (row 9b); E3.1/E3.2 attach to A2 (row 10); E1 joins the F8 block (Phase 1 row 3); E2 is F1's work inventory (row 4).

## Verification
- E1/E2: synthetic 16-bit-origin gradient corpus — banding step count and high-band energy before/after; golden-image byte-identity for migrated-but-unchanged ops; benchmark.py per-face time (expect improvement).
- E3: slider monotonicity tests (output delta strictly increasing in slider value — extends the existing monotonicity test pattern); recipe re-tune signed off on owner's corpus.
- E4: engine init time before/after; multi-face IPC payload size logged; full pytest throughout.

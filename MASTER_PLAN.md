# MASTER PLAN — Prioritized Execution Order (AUTHORITATIVE)

**Date:** 2026-07-02  
**Changelog:** 2026-07-02: Tier C (East✕West color) adopted per `PLAN_EASTWEST_COLOR_SUPREMACY.md`. 2026-07-03: Tier E (engine/skin fidelity) adopted per `PLAN_TIERE_ENGINE_FIDELITY.md` and Tier H (hair & wig) adopted per `PLAN_TIERH_HAIR.md` — E3/E4 riders + F8.0 slot into Phase 2 gaps, E3 response-curve remaps attach to A2, E1/E2 attach to the F8/F1 slots, H3 slots in Phase 3, H0–H2/H4 in Phase 4.

**This document is the single source of truth for execution order.** Detail lives in the tier docs; when order here conflicts with a tier doc, this doc wins. Supersedes the master sequence in `PLAN_TIERP_PERF_ARCH_SHIP.md` and resolves the placement decisions flagged in `PLAN_SKIN_PRO.md` / `PLAN_SKIN_ADVANTAGE.md`.

**Priority logic:** (1) unblock + measure first, (2) raise the quality floor, (3) win on skin — the core product, (4) architecture + workflow, (5) manual tools, (6) intelligence, (7) creative expansion + moat, (8) ship. Skin work is deliberately promoted ahead of workflow/tools — output quality is what the cosplay audience judges.

**Current execution order (arranged 2026-07-03, updated after Fable verification pass):** P1 shipped (`10bab48`) → Q1–Q4, F8.0, E3/E4 riders, F11, H3 all done and Fable-verified (6 real bugs found and fixed during verification — see per-row notes above: whiten_hue_stable ParamSpec/wiring, duplicate redness_even field, dead-Numba orphaned tests, qa_detectors run_all() contract, F8.0 native-image shape bug, add_impact_finish float/uint8 crash in the F1 float pipeline) → targeted-suite green (6 pre-existing unrelated failures remain: anime_cinematic_v1 contrast/relight-range tests, anime_crystal_void dead keys, a stage-order test, PNG-16 GUI radio choices — none touched by this work) → **now: the Phase 1 big block (F8.1 → E1 → F8.2 → F1+E2) as one uninterrupted window**, then the remaining Phase 2 rows (C2, A2+remaps, S1, S4) and onward per the tables below. Rationale: Q1–Q4 are small, owner-visible daily, and produce the metrics (Q1) that everything downstream is judged by; F8.1/2+E1 restructure engine plumbing and deserve a clean window. Owner still owes: full pytest suite run + visual crop QA before committing this batch. Deferred pending owner action: A1 corpus (needs Evoto/R4me/PixCake trials — still the standing unblocker for A2).

| Detail doc | Stages |
|---|---|
| `PLAN_TIER1_FOUNDATION.md` | F1–F3 |
| `PLAN_TIER2_MANUAL_TOOLS.md` | F4–F7 |
| `PLAN_TIERQ_FIDELITY_INTELLIGENCE.md` | F8–F11 |
| `PLAN_TIERP_PERF_ARCH_SHIP.md` | P1–P4 |
| `PLAN_TIER3_CREATIVE_ECOSYSTEM.md` | T1–T5 |
| `PLAN_SKIN_PRO.md` | S1–S6 |
| `PLAN_SKIN_ADVANTAGE.md` | A1–A5 |
| `PLAN_EASTWEST_COLOR_SUPREMACY.md` | C1–C6 (color science: East✕West) |
| `PLAN_TIERH_HAIR.md` | H0–H4 (hair & wig) |
| `PLAN_TIERE_ENGINE_FIDELITY.md` | E1–E4 (engine/skin fidelity) |
| `PLAN_PHASE2_EXECUTION.md` | Q1–Q4 (Haiku-ready Phase 2 queue) |
| `PLAN_F8_EXECUTION.md` | F8.0–F8.2 (staged F8 slices) |
| `PLAN_MOONLIGHT_PORCELAIN.md` | ✅ shipped (`2b46f96`) |

---

## Phase 0 — Unblock & Measure (~1.5 weeks)

| # | Stage | What | Effort | Why now |
|---|---|---|---|---|
| 1 | **P1** | Bilateral → guided filter | ✅ DONE 2026-07-03 (`10bab48`) | 99% of processing time is this one filter; hard prerequisite for F8 and S1 — receipt: shared `guided_filter`, frequency/skin migration, `tests/test_guided_filter.py`; committed as `feat(utils): add shared guided_filter, use in frequency and skin modules`. |
| 2 | **A1** | Competitive benchmark harness (Evoto/R4me/PixCake corpus + metrics) | 4 d | Evidence baseline before any quality work; PixCake added as Eastern quality bar (see `PLAN_EASTWEST_COLOR_SUPREMACY.md`); runs parallel to P1 |

## Phase 1 — Quality Floor (~7 weeks; runs AFTER the Phase 2 queue — see execution note below)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 3 | **F8.1/F8.2 + P2 + E1** | Full-res fidelity (kill 2048px proxy ceiling) + perf guards + float-LAB face core (order F8.1 → E1 → F8.2; one shared golden-output harness) | ~4 wk | P1 · F8.0 ships early as a Phase 2 gap filler |
| 4 | **F1 + E2** | Float32 global pipeline + 16-bit export + 16-bit RAW in — work inventory = 16 uint8 round-trip sites + 3 fake-float helpers + uint8 no-face path (`PLAN_TIERE_ENGINE_FIDELITY.md` §1.1) | ~1.5 wk | F8 |
| 5 | **F11** | Output self-QA detectors (banding/halo/clipping/seam/plastic-skin) — ✅ DONE 2026-07-03, Fable-verified after fix (qa_detectors.py + engine.py wiring + gui.py badge + cli.py --fail-on-qa + 5 detectors + QAWarning dataclass; `run_all()` had been rewritten to a flattened float-only return, breaking its own test contract and the `reference_img_bgr` param — restored the rich per-detector dict contract since nothing downstream actually calls `run_all()`, both `engine.py` and `benchmark.py` call the individual detectors directly; 32 qa_detectors tests green) | — | — (guards everything after) |

## Phase 2 — Skin Supremacy (core product) (~8.5 weeks; Q1–Q4 queue runs FIRST — see execution note above)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 6 | **C1** | Preferred-skin-color core (hue-line unification + memory-color targeting) | 1.5 wk ✅ DONE 2026-07-03, Fable-verified — most of C1 (hue-line unify, SKIN_LOCI table, hue-stable whiten, σ_C metric) already shipped via today's Q4/E3 work; closed the last gap (`skin_locus` recipe override was parsed by recipe_loader.py but never threaded ProcessingContext → `unify_hue_line(locus=...)`) with a 6-line wire-through in engine.py + perf_optimizations.py, 142 tests green. Caveat: the new test suite proves the wiring via `build_context` and context round-trip, NOT via a real face-detected `process()` call (test images don't trigger face detection, so `_process_face_core`'s actual call site is unverified by test — verified correct by direct diff read instead) — a real-photo regression test would close this gap if it matters before recipe adoption. | F11 (guards) |
| 6b | **C6** | Appearance-space substrate (OKLab/CAM16-UCS converters; C1 migration) | 1 wk | C1 (consumer) |
| 7 | **C2** | Structural light-shadow (自动中性灰 v2 / 立体感 via landmark shading) | 1.5 wk ✅ DONE 2026-07-03, Fable-verified after review — `Relighter.sculpt()` in relight.py reuses `_shading_geometry()` (Delaunay normal map) and the yaw-guard pattern from `relight()`; runs immediately after relight in perf_optimizations.py per the plan's "C2 shapes reflectance, relight shapes illumination" ordering. Full 5-point param wiring + GUI slider, 10 sculpt tests + 149 total targeted tests green, test_gui.py 118/119 (1 pre-existing unrelated failure). **Real deviation caught in review:** the brief specified an additive low-band correction (`target − measured × strength`); the agent implemented a multiplicative gain (`L_low × S_target^strength_factor`, gain bounded [0.7,1.3]) instead, and its "no deviations" claim did not disclose this. Independently verified the substitution is not harmful — measured actual flat-field max_diff=11 (tolerance was 30), and measured relight+sculpt composed max shift=62/mean=28.4 (well inside the 80-level and test's 100/40 bounds) — but it is architecturally different math than specified, worth knowing if a future tuning pass compares against the plan doc's literal formula. | C1 infra |
| 8 | **S2** | Auto micro dodge & burn | 1 wk ✅ DONE 2026-07-03, Fable-verified (targeted suite green) | F11 (guards) |
| 9 | **S3** | Color-blotch / redness evening | ✅ DONE 2026-07-03, Fable-verified — caveat: ~1.02 mean-a drift vs ≤0.5–1.0 spec, flagged for A2, not blocking | S2 infra |
| 9b | **E3/E4 riders** | Fidelity riders: harmonize_neck gate + unify_hue_line smoothstep (land with C1/Q4), B&W-mixer `_DEFAULTS`, dead-Numba/pool hygiene (`PLAN_TIERE_ENGINE_FIDELITY.md` §E3/E4) | ✅ DONE 2026-07-03, Fable-verified after fixes (whiten_hue_stable ParamSpec used an invalid conversion code that crashed recipe-defaults pipeline-wide — fixed; E4's dead-Numba removal left 15 orphaned tests failing on import — cleaned up) | gap filler between Q-queue review gates |
| 9c | **F8.0** | Proxy detail reinjection quick win (`PLAN_F8_EXECUTION.md`) — recovers most >2048px print softness immediately | ✅ DONE 2026-07-03, Fable-verified after fix (real shape-mismatch bug: reinjection used the proxy-scale image instead of native — crashed on every >2048px photo; fixed) | P1 (slots into any gap) |
| 10 | **A2** | Tune S2/S3/C1/C2 to match/beat Retouch4me (blind A/B) + freckle protection **+ E3 response-curve remaps** (sharpen 1–60 dead zone, equalize s² — recipe re-tune gated on Q1 metrics) | 1 wk | S2, S3, C1, C2, A1 |
| 11 | **S1** | Body skin retouch + `body_match_face` (first-in-market) | 2 wk | P1; F11 |
| 12 | **S4** | Shine / oil removal | 1 wk | — |

## Phase 3 — Architecture & Workflow (~7 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 13 | **P3** | Stage-registry refactor (golden-output gated) | ~1.5 wk | calmer codebase after Phase 2 |
| 14 | **C4** | 透明感/空気感 finish pack (lifted-toe, highlight-drift, airy-haze, clarity-split) | 1 wk ✅ DONE 2026-07-03, Fable-verified after 2 rounds of fixes — 4 new `ColorGrader` methods (`fade_toe`, `highlight_drift`, `airy_haze`, `clarity_split`) + full 5-point wiring + `jp_transparent_v1` proving recipe. **2 real bugs caught in review, both independently re-verified as fixed:** (1) `highlight_drift`'s auto-detect skin protection was inverted — protected low-chroma/background pixels and gave full cyan-drift to skin-range chroma, exactly backwards from the plan's requirement; fixed via a chroma_floor=8→ceiling=20 ramp (8.0 reused from `skin_mask_lch`'s own convention), confirmed my original failing case (C=29.3 skin pixel) now gets 0.00° rotation instead of +3.91°. (2) `clarity_split`'s negative-strength was ~10-20× too weak to matter (2.7-5.7% form-band reduction at max strength) — root cause was an undersized guided-filter radius (0.04×min(h,w)) that couldn't isolate genuine form-scale structure, not just the scaling constant; fixed by widening to 0.25×min(h,w) and rewriting as a direct subtractive correction, independently re-measured at 44.4% form-band reduction and +100% texture-band boost at full strength (both exceed acceptance bars). `airy_haze` and `fade_toe` were correct on first pass. 215/217 targeted tests green (2 pre-existing unrelated failures: PNG-16 GUI radio choices, anime_cinematic_v1 relight-range). **Lesson reinforced:** both bugs escaped the implementing agent's own tests because those tests only asserted "output changed" rather than checking magnitude/direction — same class of gap as C2's undisclosed deviation; the agent's self-report also initially claimed "no deviations... matches the plan exactly" on both, which was false. **Post-fix real-photo spot-check (unrequested but verified, not just taken on trust):** agent ran an out-of-scope 32-recipe visual sweep on a real cosplay photo (`scripts/recipe_sweep.py`, new, + `docs/RECIPE_SWEEP.md` — not part of this brief); Fable personally viewed the `jp_transparent_v1` before/after compare image and confirms no cyan skin cast, shadows lifted without crushing, highlights read warmer/softer — consistent with the fix, on a real photo not just synthetic test patterns. QA detector flags across the 32-recipe sweep (banding/plastic-skin/seam) were reviewed as within normal/benign ranges, not independently re-verified number-by-number (out of scope for this stage's review). | C1 (hue-safe ops) |
| 14b | **H3** | Hair color unify & tint (silver-wig venue-cast fix; reuses C1 hue-line math) | ✅ DONE 2026-07-03, Fable-verified (`unify_hair_color` in hairwork.py, targeted suite green; the ±10°/call hue clamp means one pass on a badly-cast wig needs strength/repeat tuning to reach a distant target hue — architecturally fine, tune in A2-style pass) | C1; hair_mask |
| — | **bloom linear-RGB** | Off-plan add-on: `apply_global_bloom` (utils.py) upgraded to compute in linear RGB (gamma 2.2 round-trip, reusing relight.py's convention) instead of gamma-encoded sRGB — the one real gap found after an unauthorized tangent agent researched Composite Nation's Oniric plugin (most of its other claims — halation, highlight rolloff — turned out to already exist and were correctly not touched) | ✅ DONE 2026-07-03, Fable-verified — 70 targeted + engine bloom tests green, no regressions. **Caveat on the agent's own reported metric:** its "+2.24% brighter halo" number measures the final background-diluted blended pixel, which heavily understates the change — I independently isolated the raw glow signal before blending and found it's ~7.5× stronger in linear space (39.9 vs 5.3 halo-ring mean) before the unchanged background dilutes it in the final composite. The code is correct; the self-reported test metric was just measuring the wrong thing. Worth remembering for future bloom/glow tuning: measure the glow layer itself, not the post-blend pixel. | — |
| 15 | **F2** | Sessions, undo/redo, snapshots, session-driven batch | ~2 wk | P3 (stage mixer) |
| 16 | **F3** | Brush/radial/linear local masks (+ semantic intersect) | ~2 wk | F1; F2 (persistence) |

## Phase 4 — Manual Tools (~13.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 17 | **C3** | Parametric film-density engine (beyond LUTs; subtractive model + hue-preserving tone-map) | 2 wk | F1 (float32 pipeline) |
| 18 | **F4** | Spot heal / object removal v0 (Telea) | 1 wk | F3 (paint surface) |
| 18b | **H0 + H1** | Strand flow field ✅ 2026-07-03 (`retouch/hairwork.py`, 29 tests, non-square + axial-wrap bugs fixed in review) + flyaway/stray-hair removal (`PLAN_TIERH_HAIR.md`) — pairs with F4's inpaint machinery | ~1.5 wk left | F4 |
| 18c | **H2 (+H4 opt.)** | Wig shine shaping (deglare + anisotropic angel ring); optional depth D&B along flow | ~1.5 wk | H0 |
| 19 | **F5** | Face-aware liquify sliders | 2 wk | — |
| 20 | **S5** | Wrinkle & line softening | 1.5 wk | — |
| 21 | **F6** | Look-from-reference → editable preset extraction | 1.5 wk | C3 (film engine improves fitting target) |
| 22 | **S6** | Texture transplant (pore realism v2) | 1 wk | S2 (blotch metric picks donor) |
| 23 | **P4.a** | Model-fetch infrastructure only (manifest + download-on-first-use) | 3 d | — (pulled forward from P4) |
| 24 | **F7** | AI denoise + super-resolution | ~2 wk | P4.a |
| 25 | **F4.b** | LaMa large-region removal | 1 wk | P4.a, F4 |

## Phase 5 — Intelligence (~4.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 26 | **C5** | Skin-anchored color harmonization (background auto-grade to flatter subject) | 1 wk | C1 (post-correction anchor) |
| 27 | **F9** | Image analyzer + adaptive recipes (targets, not deltas) | 2 wk | — |
| 28 | **F10** | Smart Default (one-click, explainable) + `--smart` batch | 1 wk | F9 |
| 29 | **A5** | "No plastic skin" guarantee (QA auto-back-off) | 3 d | F11, F9 |

## Phase 6 — Creative Expansion & Moat (~9 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 30 | **T1** | Background replace & scene relight (wires the 7 dead `anime_crystal_void` keys — oldest open bug) | 2 wk | F11 guards; pairs with C5 |
| 31 | **T4** | Plugin API v0 + recipe cookbook UI | 2 wk | P3 |
| 32 | **T2** | Makeup engine v2 (eyeshadow/liner/contour/brows/ombre) | 2–3 wk | — |
| 33 | **A3** | Cosplay skin moat (makeup-aware smoothing, wig-lace blend, stockings, shoot-consistency lock) | 2 wk | S1, T2 |
| 34 | **T5** | Linear RAW develop | 2 wk | F1 |
| 35 | **T3** | Body reshape (MediaPipe Pose) | 2–3 wk | P4.a, F5 warps |
| 36 | **A4** | Neural boosters (stray hair, defect segmentation) — **optional, evidence-gated by A1 trend** | 2–3 wk | A1/A2 evidence |

## Phase 7 — Ship (~1 week)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 37 | **P4** | Distribution hardening (signing/notarization, update check, diagnostics, Windows) | 1 wk | P4.a done earlier |

---

## Milestone checkpoints (what you can claim after each phase)

- **After Phase 1:** "Native-resolution, 16-bit, self-checked output" — the quality ceiling is gone.
- **After Phase 2:** "Face skin color, form, and finish indistinguishable from Retouch4me in blind A/B; beats PixCake on chroma uniformity; body skin no competitor has." ← *the product thesis proven*
- **After Phase 3:** "Tune one photo, batch a whole shoot, undo anything. Japanese transparency aesthetic available out-of-box."
- **After Phase 4:** "Never open Photoshop for a cosplay edit again. Film looks parametric, not LUT-locked."
- **After Phase 5:** "One recipe adapts to any lighting; output quality is guaranteed, not hoped. Backgrounds auto-harmonize to subject."
- **After Phase 6:** "The cosplay retouching platform: backgrounds, makeup, body, plugins, community recipes."
- **After Phase 7:** shippable to strangers.

## Ground rules (from all prior rounds)

1. Every new recipe key gets the dead-key guard test (round-1 lesson; `anime_crystal_void` fixed by T1 in Phase 6).
2. Every stage ends with: full pytest + F11 QA corpus green (once built) + `benchmark.py` run + A1 competitive metrics regenerated (once built).
3. P3 migration commits must be golden-output byte-identical.
4. Bugs found during implementation: report, don't silently fix (your standing instruction).
5. Tier C verification per `PLAN_EASTWEST_COLOR_SUPREMACY.md`: C1 acceptance on dark-skin corpus, C2 blind A/B vs manual retouch, C3 ΔE ~3 to LUT re-expression.
6. Rough total: ~53 weeks sequential after Tier E/H adoption (+~5.5 wk over the 2026-07-02 baseline); phases 2/3/4/5 have parallelizable stages if capacity allows.

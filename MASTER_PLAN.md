# MASTER PLAN — Prioritized Execution Order (AUTHORITATIVE)

**Date:** 2026-07-02  
**Changelog:** 2026-07-02: Tier C (East✕West color) adopted per `PLAN_EASTWEST_COLOR_SUPREMACY.md`.

**This document is the single source of truth for execution order.** Detail lives in the tier docs; when order here conflicts with a tier doc, this doc wins. Supersedes the master sequence in `PLAN_TIERP_PERF_ARCH_SHIP.md` and resolves the placement decisions flagged in `PLAN_SKIN_PRO.md` / `PLAN_SKIN_ADVANTAGE.md`.

**Priority logic:** (1) unblock + measure first, (2) raise the quality floor, (3) win on skin — the core product, (4) architecture + workflow, (5) manual tools, (6) intelligence, (7) creative expansion + moat, (8) ship. Skin work is deliberately promoted ahead of workflow/tools — output quality is what the cosplay audience judges.

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
| `PLAN_MOONLIGHT_PORCELAIN.md` | ✅ shipped (`2b46f96`) |

---

## Phase 0 — Unblock & Measure (~1.5 weeks)

| # | Stage | What | Effort | Why now |
|---|---|---|---|---|
| 1 | **P1** | Bilateral → guided filter | 3–5 d | 99% of processing time is this one filter; hard prerequisite for F8 and S1 — **implemented in working tree 2026-07-02 · pending test-suite approval + visual QA (see P1_IMPLEMENTATION_NOTES.md)** |
| 2 | **A1** | Competitive benchmark harness (Evoto/R4me/PixCake corpus + metrics) | 4 d | Evidence baseline before any quality work; PixCake added as Eastern quality bar (see `PLAN_EASTWEST_COLOR_SUPREMACY.md`); runs parallel to P1 |

## Phase 1 — Quality Floor (~5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 3 | **F8 + P2** | Full-res fidelity (kill 2048px proxy ceiling) + perf guards | ~3 wk | P1 |
| 4 | **F1** | Float32 global pipeline + 16-bit export + 16-bit RAW in | ~1.5 wk | F8 |
| 5 | **F11** | Output self-QA detectors (banding/halo/clipping/seam/plastic-skin) | ~1.5 wk | — (measures F8/F1 wins; guards everything after) |

## Phase 2 — Skin Supremacy (core product) (~7.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 6 | **C1** | Preferred-skin-color core (hue-line unification + memory-color targeting) | 1.5 wk | F11 (guards) |
| 6b | **C6** | Appearance-space substrate (OKLab/CAM16-UCS converters; C1 migration) | 1 wk | C1 (consumer) |
| 7 | **C2** | Structural light-shadow (自动中性灰 v2 / 立体感 via landmark shading) | 1.5 wk | C1 infra |
| 8 | **S2** | Auto micro dodge & burn | 1 wk | F11 (guards) |
| 9 | **S3** | Color-blotch / redness evening | 3 d | S2 infra |
| 10 | **A2** | Tune S2/S3/C1/C2 to match/beat Retouch4me (blind A/B) + freckle protection | 1 wk | S2, S3, C1, C2, A1 |
| 11 | **S1** | Body skin retouch + `body_match_face` (first-in-market) | 2 wk | P1; F11 |
| 12 | **S4** | Shine / oil removal | 1 wk | — |

## Phase 3 — Architecture & Workflow (~6.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 13 | **P3** | Stage-registry refactor (golden-output gated) | ~1.5 wk | calmer codebase after Phase 2 |
| 14 | **C4** | 透明感/空気感 finish pack (lifted-toe, highlight-drift, airy-haze, clarity-split) | 1 wk | C1 (hue-safe ops) |
| 15 | **F2** | Sessions, undo/redo, snapshots, session-driven batch | ~2 wk | P3 (stage mixer) |
| 16 | **F3** | Brush/radial/linear local masks (+ semantic intersect) | ~2 wk | F1; F2 (persistence) |

## Phase 4 — Manual Tools (~10 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 17 | **C3** | Parametric film-density engine (beyond LUTs; subtractive model + hue-preserving tone-map) | 2 wk | F1 (float32 pipeline) |
| 18 | **F4** | Spot heal / object removal v0 (Telea) | 1 wk | F3 (paint surface) |
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
6. Rough total: ~47 weeks sequential; phases 2/3/4/5 have parallelizable stages if capacity allows.

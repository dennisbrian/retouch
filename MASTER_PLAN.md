# MASTER PLAN — Prioritized Execution Order (AUTHORITATIVE)

**Date:** 2026-07-02
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
| `PLAN_MOONLIGHT_PORCELAIN.md` | ✅ shipped (`2b46f96`) |

---

## Phase 0 — Unblock & Measure (~1.5 weeks)

| # | Stage | What | Effort | Why now |
|---|---|---|---|---|
| 1 | **P1** | Bilateral → guided filter | 3–5 d | 99% of processing time is this one filter; hard prerequisite for F8 and S1 |
| 2 | **A1** | Competitive benchmark harness (Evoto/R4me corpus + metrics) | 4 d | Evidence baseline before any quality work; runs parallel to P1 |

## Phase 1 — Quality Floor (~5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 3 | **F8 + P2** | Full-res fidelity (kill 2048px proxy ceiling) + perf guards | ~3 wk | P1 |
| 4 | **F1** | Float32 global pipeline + 16-bit export + 16-bit RAW in | ~1.5 wk | F8 |
| 5 | **F11** | Output self-QA detectors (banding/halo/clipping/seam/plastic-skin) | ~1.5 wk | — (measures F8/F1 wins; guards everything after) |

## Phase 2 — Skin Supremacy (core product) (~5.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 6 | **S2** | Auto micro dodge & burn | 1 wk | F11 (guards) |
| 7 | **S3** | Color-blotch / redness evening | 3 d | S2 infra |
| 8 | **A2** | Tune S2/S3 to match/beat Retouch4me (blind A/B) + freckle protection | 1 wk | S2, S3, A1 |
| 9 | **S1** | Body skin retouch + `body_match_face` (first-in-market) | 2 wk | P1; F11 |
| 10 | **S4** | Shine / oil removal | 1 wk | — |

## Phase 3 — Architecture & Workflow (~5.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 11 | **P3** | Stage-registry refactor (golden-output gated) | ~1.5 wk | calmer codebase after Phase 2 |
| 12 | **F2** | Sessions, undo/redo, snapshots, session-driven batch | ~2 wk | P3 (stage mixer) |
| 13 | **F3** | Brush/radial/linear local masks (+ semantic intersect) | ~2 wk | F1; F2 (persistence) |

## Phase 4 — Manual Tools (~8 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 14 | **F4** | Spot heal / object removal v0 (Telea) | 1 wk | F3 (paint surface) |
| 15 | **F5** | Face-aware liquify sliders | 2 wk | — |
| 16 | **S5** | Wrinkle & line softening | 1.5 wk | — |
| 17 | **F6** | Look-from-reference → editable preset extraction | 1.5 wk | — |
| 18 | **S6** | Texture transplant (pore realism v2) | 1 wk | S2 (blotch metric picks donor) |
| 19 | **P4.a** | Model-fetch infrastructure only (manifest + download-on-first-use) | 3 d | — (pulled forward from P4) |
| 20 | **F7** | AI denoise + super-resolution | ~2 wk | P4.a |
| 21 | **F4.b** | LaMa large-region removal | 1 wk | P4.a, F4 |

## Phase 5 — Intelligence (~3.5 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 22 | **F9** | Image analyzer + adaptive recipes (targets, not deltas) | 2 wk | — |
| 23 | **F10** | Smart Default (one-click, explainable) + `--smart` batch | 1 wk | F9 |
| 24 | **A5** | "No plastic skin" guarantee (QA auto-back-off) | 3 d | F11, F9 |

## Phase 6 — Creative Expansion & Moat (~9 weeks)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 25 | **T1** | Background replace & scene relight (wires the 7 dead `anime_crystal_void` keys — oldest open bug) | 2 wk | F11 guards |
| 26 | **T4** | Plugin API v0 + recipe cookbook UI | 2 wk | P3 |
| 27 | **T2** | Makeup engine v2 (eyeshadow/liner/contour/brows/ombre) | 2–3 wk | — |
| 28 | **A3** | Cosplay skin moat (makeup-aware smoothing, wig-lace blend, stockings, shoot-consistency lock) | 2 wk | S1, T2 |
| 29 | **T5** | Linear RAW develop | 2 wk | F1 |
| 30 | **T3** | Body reshape (MediaPipe Pose) | 2–3 wk | P4.a, F5 warps |
| 31 | **A4** | Neural boosters (stray hair, defect segmentation) — **optional, evidence-gated by A1 trend** | 2–3 wk | A1/A2 evidence |

## Phase 7 — Ship (~1 week)

| # | Stage | What | Effort | Depends on |
|---|---|---|---|---|
| 32 | **P4** | Distribution hardening (signing/notarization, update check, diagnostics, Windows) | 1 wk | P4.a done earlier |

---

## Milestone checkpoints (what you can claim after each phase)

- **After Phase 1:** "Native-resolution, 16-bit, self-checked output" — the quality ceiling is gone.
- **After Phase 2:** "Face skin indistinguishable from Retouch4me in blind A/B; body skin no competitor has." ← *the product thesis proven*
- **After Phase 3:** "Tune one photo, batch a whole shoot, undo anything."
- **After Phase 4:** "Never open Photoshop for a cosplay edit again."
- **After Phase 5:** "One recipe adapts to any lighting; output quality is guaranteed, not hoped."
- **After Phase 6:** "The cosplay retouching platform: backgrounds, makeup, body, plugins, community recipes."
- **After Phase 7:** shippable to strangers.

## Ground rules (from all prior rounds)

1. Every new recipe key gets the dead-key guard test (round-1 lesson; `anime_crystal_void` fixed by T1 in Phase 6).
2. Every stage ends with: full pytest + F11 QA corpus green (once built) + `benchmark.py` run + A1 competitive metrics regenerated (once built).
3. P3 migration commits must be golden-output byte-identical.
4. Bugs found during implementation: report, don't silently fix (your standing instruction).
5. Rough total: ~39 weeks sequential; phases 2/3 and 4/5 have parallelizable stages if capacity allows.

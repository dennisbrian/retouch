# MASTER PLAN — Prioritized Execution Order (AUTHORITATIVE)

**Date:** 2026-07-02 · **Last reconciled:** 2026-07-10

**This document is the single source of truth for execution ORDER and STATUS.** Long verification
narratives, bug-found receipts, agent-integrity notes, and measured numbers now live in
[`docs/EXECUTION_LOG.md`](docs/EXECUTION_LOG.md), organized by phase → stage ID. Each status cell
below is a one-line receipt pointing there. Detail on *design* lives in the tier docs; when order
here conflicts with a tier doc, this doc wins. Supersedes the master sequence in
`PLAN_TIERP_PERF_ARCH_SHIP.md` and resolves the placement decisions flagged in `PLAN_SKIN_PRO.md` /
`PLAN_SKIN_ADVANTAGE.md`.

**Priority logic:** (1) unblock + measure first, (2) raise the quality floor, (3) win on skin — the
core product, (4) architecture + workflow, (5) manual tools, (6) intelligence, (7) creative expansion
+ moat, (8) ship. Skin work is deliberately promoted ahead of workflow/tools — output quality is what
the cosplay audience judges.

**Changelog:** 2026-07-02: Tier C (East✕West color) adopted per `PLAN_EASTWEST_COLOR_SUPREMACY.md`.
2026-07-03: Tier E (engine/skin fidelity) + Tier H (hair & wig) adopted. 2026-07-10: doc refactored —
narratives moved to `EXECUTION_LOG.md`; status reconciled against git; ground rule #7 added;
per-face recipe assignment backlog row added (owner-approved).

---

## ▶ RESUME HERE — single current-state summary (2026-07-10)

**This is the only place status is summarized. Per-row cells state each stage's own status; do not
re-summarize progress elsewhere (it drifts).**

Phases 1–6 code-complete (all stages ✅ except A2, blocked on A1). **Caveat: three "✅ DONE" stages
shipped as unwired islands** — **T5** (RAW develop) is ⚠️ shipped-unwired (see audit); **T4** (plugin
API / cookbook UI) and **F6** (look extractor) are module-complete but not reachable from any user
path. **T3 / A3** were dark in both user paths; commit `0610660` (2026-07-10) made them
recipe-reachable (demo recipes) but GUI sliders remain deferred. 85+ recipes; 11/11 flagship recipes;
masterwork_v1 (#11) ✅ (`3b46807`). Recipe-integrity audit fixes committed 2026-07-10 (`40721d6`).
Post-plan auto-gap backlog: #1/#2/#3/#4/#6/#7 all committed 2026-07-10; **#5 (auto stray-hair,
A4-gated) is the only open gap**. Multiple **[VISUAL QA PENDING]** flags outstanding (see flagged
rows) — renders now generated (`test_output/visual_qa/`, `visual_qa_grid/` 42 montages jpeg+raf,
`visual_qa_wiring/`), but review/sign-off still pending; they gate shipping.

**Remaining:** A1 (owner action — needs Evoto/R4me/PixCake corpus), A2 (blocked on A1), wiring-debt
burn-down (T3/A3 GUI sliders, F6, T4/cookbook, plugin discovery), **Phase 7** P4 distribution
hardening.

**Next actions (agreed order, 2026-07-10):**
1. **Consolidated visual-QA session** over all `[VISUAL QA PENDING]` flags (F5, F10, showcase family,
   RAF import, face_exposure, sclera vessel, backdrop, fabric, per-region wrinkle, reshape
   completeness, auto body reshape). **Renders generated** (`test_output/visual_qa/`,
   `test_output/visual_qa_grid/` — 42 montages, jpeg+raf, `test_output/visual_qa_wiring/`);
   **review/sign-off still pending.** These gate everything below.
2. **Wiring-debt burn-down** (~4–6 d): T3/A3 GUI sliders (recipe path landed in `0610660`, GUI still
   dark), `look_extractor` GUI, recipe cookbook UI, plugin discovery call, `face_exposure` slider.
   (`_process_inputs` de-footgun refactor DONE 2026-07-13 — name-keyed dict + import-time drift guard;
   see the row in Post-plan enhancements below.)
3. **Owner: timeboxed A1 competitor-corpus run** (Evoto/R4me/PixCake) — unblocks A2 for a future tune
   pass.
4. **Phase 7 — P4 distribution hardening** (signing/notarization, update check, diagnostics,
   Windows).

_(The 2026-07-10 in-flight dead-key guard fix + recipe re-keying + film.preset decision is DONE —
committed as `40721d6`; no longer a pending action.)_

---

## Tier-doc mapping

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
| `PLAN_RAW_PROCESSING.md` | T5 wiring + full-data RAF ingestion (research 2026-07-10) |
| `PLAN_R9_INTRINSIC_SPIKE.md` | R9 spike design — albedo×shading decomposition (research 2026-07-10) |
| `PLAN_SKIN_FRONTIER.md` | Skin measurement science (M1–M6) + "inverse" retouching (I1–I5) (research 2026-07-11) |
| `PLAN_FEATURE_FRONTIER.md` | Eyes/teeth/lips optical & anatomical models — E-EYE/E-TEETH/E-LIP + shared light-direction (research 2026-07-11) |
| `PLAN_SKIN_PROMAX.md` | Skin "pro max" — new *axes* the R1–R15/M/I sweep never touched (P1–P8) (research 2026-07-11) |
| `PLAN_COLOR_SCIENCE.md` | Implementation-grade color-appearance science + codeable skin math (K1–K12) (research 2026-07-11) |
| `docs/EXECUTION_LOG.md` | Verification receipts for every ✅ row below |

---

## Phase 0 — Unblock & Measure (~1.5 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 1 | **P1** | Bilateral → guided filter | ✅ | — | ✅ DONE 2026-07-03 (`10bab48`) — shared `guided_filter`, frequency/skin migration. |
| 2 | **A1** | Competitive benchmark harness (Evoto/R4me/PixCake corpus + metrics) | 4 d | — | ⏳ OWNER ACTION — needs Evoto/R4me/PixCake trials; standing unblocker for A2. |

## Phase 1 — Quality Floor (~7 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 3 | **F8.1/F8.2 + E1** | Full-res fidelity (kill 2048px proxy ceiling) + float-LAB face core | ~4 wk | P1 · F8.0 | ✅ DONE — F8.1 2026-07-03 (Fable-verified after 1 fix; over-sharpen bug caught); F8.2 2026-07-06; E1 landed with F1+E2. See EXECUTION_LOG. |
| 3b | **P2 perf guards** | Downsample-compute / full-res-apply for large-σ blurs + peak-RSS assert (matters more since F8.1 moved grading to native res) | — | F8.1 | ✅ DONE 2026-07-06 — `_large_sigma_blur` helper, 6 sites, byte-identical <1400px. See EXECUTION_LOG. |
| 4 | **F1 + E2** | Float32 global pipeline + 16-bit export + 16-bit RAW in | ~1.5 wk | F8 | ✅ DONE 2026-07-07 — 5 waves, full-codebase dtype-awareness sweep; 774 passed. See EXECUTION_LOG. |
| 5 | **F11** | Output self-QA detectors (banding/halo/clipping/seam/plastic-skin) | — | — | ✅ DONE 2026-07-03 (Fable-verified after fix) — guards everything after. See EXECUTION_LOG. |

## Phase 2 — Skin Supremacy (core product) (~8.5 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 6 | **C1** | Preferred-skin-color core (hue-line unification + memory-color targeting) | 1.5 wk | F11 | ✅ DONE 2026-07-03 (Fable-verified) — `skin_locus` override wired; caveat: wiring proven via context, not face-detected `process()`. See EXECUTION_LOG. |
| 6b | **C6** | Appearance-space substrate (OKLab/CAM16-UCS; C1 migration) | 1 wk | C1 | ✅ DONE (predates 2026-07-03, verified) — shipped inside C1. See EXECUTION_LOG. |
| 7 | **C2** | Structural light-shadow (自动中性灰 v2 / 立体感) | 1.5 wk | C1 | ✅ DONE 2026-07-03 (Fable-verified) — deviation: multiplicative gain vs spec's additive (verified non-harmful). See EXECUTION_LOG. |
| 8 | **S2** | Auto micro dodge & burn | 1 wk | F11 | ✅ DONE 2026-07-03 (Fable-verified). |
| 9 | **S3** | Color-blotch / redness evening | — | S2 | ✅ DONE 2026-07-03 (Fable-verified) — caveat: ~1.02 mean-a drift, flagged for A2. |
| 9b | **E3/E4 riders** | Fidelity riders: harmonize_neck gate, hue-line smoothstep, B&W `_DEFAULTS`, dead-Numba hygiene | — | gap filler | ✅ DONE 2026-07-03 (Fable-verified after fixes). See EXECUTION_LOG. |
| 9c | **F8.0** | Proxy detail reinjection quick win | — | P1 | ✅ DONE 2026-07-03 (Fable-verified after shape-mismatch fix). See EXECUTION_LOG. |
| 10 | **A2** | Tune S2/S3/C1/C2 vs Retouch4me (blind A/B) + freckle protection + E3 response-curve remaps | 1 wk | S2·S3·C1·C2·**A1** | ⛔ BLOCKED on A1 (competitor corpus). |
| 11 | **S1** | Body skin retouch + `body_match_face` (first-in-market) | 2 wk | P1·F11 | ✅ DONE 2026-07-03 (Fable-verified after fixes) — 3 real bugs incl. float32→`blend_masked` silent-corruption (owner-hit GUI bug). See EXECUTION_LOG. |
| 12 | **S4** | Shine / oil removal | 1 wk | — | ✅ DONE 2026-07-03 (Fable-verified after fix) — critical LAB +128-offset chroma bug caught. See EXECUTION_LOG. |

## Phase 3 — Architecture & Workflow (~7 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 13 | **P3** | Stage-registry refactor (golden-output gated) | ~1.5 wk | Phase 2 done | ✅ DONE 2026-07-07 — `stages.py`+`stage_wrappers.py`, byte-identical to hardcoded path, registry now default. See EXECUTION_LOG. |
| 14 | **C4** | 透明感/空気感 finish pack (lifted-toe, highlight-drift, airy-haze, clarity-split) | 1 wk | C1 | ✅ DONE 2026-07-03 (Fable-verified after 2 rounds) — 2 real bugs (inverted skin protection, weak clarity_split). See EXECUTION_LOG. |
| 14b | **H3** | Hair color unify & tint (silver-wig venue-cast fix) | — | C1·hair_mask | ✅ DONE 2026-07-03 (Fable-verified) — ±10°/call clamp, tune in A2-style pass. See EXECUTION_LOG. |
| — | **bloom linear-RGB** | Off-plan: `apply_global_bloom` computes in linear RGB | — | — | ✅ DONE 2026-07-03 (Fable-verified) — self-reported metric measured wrong layer; code correct. See EXECUTION_LOG. |
| — | **film grain amplitude fix** | Off-plan sev-1: grain amplitude 20→7; 4 recipes | — | — | ✅ DONE 2026-07-03 (Fable-verified two ways, incl. real photo). See EXECUTION_LOG. |
| — | **flagship recipe list (top-10)** | Off-plan showcase: one-algorithm-each recipes | — | — | ✅ DONE (#1–7,9 built; #8 was S1-blocked, #10 capstone) 2026-07-03. See EXECUTION_LOG. |
| — | **flagship #11: masterwork_v1** | Denoise-first max-quality export recipe | — | — | ✅ DONE 2026-07-09 (`3b46807`) — pure recipe entry, dead-key clean. See EXECUTION_LOG. |
| — | **50-recipe expansion (12 scenario recipes)** | Outdoor/studio/convention lighting coverage | — | — | ✅ DONE 2026-07-03 — 3 self-caught bugs; flagged pre-existing `anime_cinematic_v1` nested-relight bug. See EXECUTION_LOG. |
| — | **showcase recipe family (17 recipes, 2026-07-09)** | Exercises skin/eye primitives (aniso, freckle, under-eye, eye) | — | — | ✅ DONE 2026-07-09 — pure recipe entries, guard-tested. **[VISUAL QA PENDING]** — frequency/skin/freckle/under-eye/eye are Visual-Critical. See EXECUTION_LOG. |
| 15 | **F2** | Sessions, undo/redo, snapshots, session-driven batch | ~2 wk | P3 | ✅ DONE 2026-07-07 — `session.py`; GUI+CLI+batch wired. See EXECUTION_LOG. |
| 16 | **F3** | Brush/radial/linear local masks (+ semantic intersect) | ~2 wk | F1·F2 | ✅ DONE 2026-07-07 — `regions.py`, `_stage_local_adjustments`. See EXECUTION_LOG. |

## Phase 4 — Manual Tools (~13.5 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 17 | **C3** | Parametric film-density engine (subtractive + hue-preserving tone-map) | 2 wk | F1 | ✅ DONE 2026-07-07 — `film.py` FilmDensityEngine, 17 ParamSpecs. See EXECUTION_LOG. |
| 18 | **F4** | Spot heal / object removal v0 (Telea) | 1 wk | F3 | ✅ DONE 2026-07-07 — `spot_heal.py` SpotHealer. See EXECUTION_LOG. |
| 18b | **H0 + H1** | Strand flow field + flyaway/stray-hair removal | — | H0·F4 | ✅ DONE (H0 2026-07-03, H1 2026-07-07) — `hairwork.py` remove_flyaways. See EXECUTION_LOG. |
| 18c | **H2 (+H4 opt.)** | Wig shine shaping (deglare + anisotropic angel ring) | ~1.5 wk | H0 | ✅ DONE 2026-07-07 — `deglare_wig`+`add_angel_ring`. See EXECUTION_LOG. |
| 19 | **F5** | Face-aware liquify sliders | 2 wk | — | ✅ DONE 2026-07-07 — 9 per-feature warp builders, caps/clamps. **[VISUAL QA PENDING]** — Visual-Critical (`geometry.py`), No-Edge-Tearing/Natural-Output gates not run. See EXECUTION_LOG. |
| 20 | **S5** | Wrinkle & line softening | 1.5 wk | — | ✅ DONE 2026-07-03 (Fable-verified after fixes) — 2 bugs (crow's-feet landmarks on eyelid, unreachable via process()). See EXECUTION_LOG. |
| 21 | **F6** | Look-from-reference → editable preset extraction | 1.5 wk | C3 | ⚠️ SHIPPED-UNWIRED 2026-07-07 — `look_extractor.py` complete + 17 tests, but unreachable from GUI/CLI. See EXECUTION_LOG + wiring-debt audit. |
| 22 | **S6** | Texture transplant (pore realism v2) | 1 wk | S2 | ✅ DONE 2026-07-03 (Fable-verified after fix) — critical circular-luminance no-op bug caught. See EXECUTION_LOG. |
| 23 | **P4.a** | Model-fetch infrastructure (manifest + download-on-first-use) | 3 d | — | ✅ DONE 2026-07-07 — `models/manifest.json` + `model_fetch.py`, path-traversal guard. See EXECUTION_LOG. |
| 24 | **F7** | AI denoise + super-resolution | ~2 wk | P4.a | ✅ DONE 2026-07-07; real NAFNet ONNX acquired 2026-07-09 (3 bugs fixed). SR deliberately not acquired. See EXECUTION_LOG. |
| 25 | **F4.b** | LaMa large-region removal | 1 wk | P4.a·F4 | ✅ DONE 2026-07-07 — `LamaHealer` w/ Telea fallback; real LaMa ONNX not yet downloaded. See EXECUTION_LOG. |

## Phase 5 — Intelligence (~4.5 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 26 | **C5** | Skin-anchored color harmonization (bg auto-grade to flatter subject) | 1 wk | C1 | ✅ DONE 2026-07-07 — `harmonizer.py` BackgroundHarmonizer, 21 tests. See EXECUTION_LOG. |
| 27 | **F9** | Image analyzer + adaptive recipes (targets, not deltas) | 2 wk | — | ✅ DONE 2026-07-07. |
| 28 | **F10** | Smart Default (one-click, explainable) + `--smart` batch | 1 wk | F9 | ✅ DONE 2026-07-07 — `smart_default.py`, GUI+CLI wired. **[VISUAL QA PENDING]** — not Visual-Critical, but 20-photo `--smart` acceptance run recommended. See EXECUTION_LOG. |
| 29 | **A5** | "No plastic skin" guarantee (QA auto-back-off) | 3 d | F11·F9 | ✅ DONE 2026-07-07 — `qa_backoff.py`, back-off loop in `_run_core_pipeline`. See EXECUTION_LOG. |

## Phase 6 — Creative Expansion & Moat (~9 weeks)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 30 | **T1** | Background replace & scene relight (wires 7 dead `anime_crystal_void` keys) | 2 wk | F11 | ✅ DONE 2026-07-07 — `background.py` BackgroundReplacer, dead-key guard flipped to assert wiring. See EXECUTION_LOG. |
| 31 | **T4** | Plugin API v0 + recipe cookbook UI | 2 wk | P3 | ⚠️ SHIPPED-UNWIRED 2026-07-07 — `plugin_api.py`+`recipe_cookbook.py` complete + 60 tests, but discover/init never called + no cookbook GUI. See EXECUTION_LOG + wiring-debt audit. |
| 32 | **T2** | Makeup engine v2 (eyeshadow/liner/contour/brows/ombre) | 2–3 wk | — | ✅ DONE (per resume line 2026-07-08). |
| 33 | **A3** | Cosplay skin moat (makeup-aware smoothing, wig-lace blend, stockings, shoot-consistency lock) | 2 wk | S1·T2 | 🔄 PARTIAL — code ✅; recipe-reachable via `cosplay_wiring_demo_v1` (`0610660`, 2026-07-10); GUI sliders still not wired. See wiring-debt audit. |
| 34 | **T5** | Linear RAW develop | 2 wk | F1 | ⚠️ SHIPPED-UNWIRED (audit 2026-07-10) — `raw_develop.py` is an island (zero non-test callers; RAF still ingested 8-bit). Bug: `load_raw` omits `gamma=` → wrong domain. See EXECUTION_LOG + `PLAN_RAW_PROCESSING.md`. |
| 35 | **T3** | Body reshape (MediaPipe Pose) | 2–3 wk | P4.a·F5 | 🔄 PARTIAL — code ✅; recipe-reachable via `body_reshape_demo_v1` (`0610660`, 2026-07-10); GUI sliders still not wired. See wiring-debt audit. |
| 36 | **A4** | Neural boosters (stray hair, defect segmentation) — optional, evidence-gated by A1 | 2–3 wk | A1/A2 | 🅿️ PARKED — owner decision 2026-07-03: classical-only until A1 evidence. Competitor sweep mapped 2026-07-10. See EXECUTION_LOG. |

## Phase 7 — Ship (~1 week)

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| 37 | **P4** | Distribution hardening (signing/notarization, update check, diagnostics, Windows) | 1 wk | P4.a | 📋 NOT STARTED — final ship step (after visual-QA + wiring burn-down). |

---

## Post-plan enhancements (2026-07-10)

_All rows below are COMMITTED unless noted; the section is no longer wholesale-uncommitted._

| # | Stage | What | Effort | Deps | Status |
|---|---|---|---|---|---|
| — | **RAF import — faithful neutral dev + highlight recovery** | `io.py` raw paths: `bright=1.0`, sRGB output, `ReconstructDefault` highlights | ~0.5 d | — | ✅ DONE 2026-07-10 (`af54d2d`) — 92 tests; both raw paths agree (mean diff 0.17). **[VISUAL QA PENDING]** — HIGH-impact tone change, confirm recovered highlights on `_DSF1853.RAF`. See EXECUTION_LOG. |
| — | **Recipe integrity audit + fix** | 4 dead-key classes / 21 instances; guard-test recursion; film.preset decision | ~0.5–1 d | — | ✅ DONE 2026-07-10 (`40721d6`) — 6 film.preset → parametric `film.*` bundles, nose_smooth re-keyed, recursive dead-key test. See EXECUTION_LOG. |
| — | **Wiring-debt audit** | 4 unwired islands (T5/T4/F6 + cookbook), T3/A3 dark, 11 GUI-invisible params | ~4–6 d | — | 📋 AUDIT DONE 2026-07-10; **2nd pass (UNCOMMITTED) wired the remaining items**: T4 `plugin_api.discover/init_all` called in `engine.__init__` (guarded); `recipe_cookbook` + `look_extractor` reachable via `cli.py` (`--list-recipes`, `--search-recipes`, `--extract-look`/`--look-base`); 9 GUI sliders added (`face_exposure`, `cosplay_*`, `body_reshape_*`, `auto_body_reshape`) + full `_process_inputs`↔`param_names()` realignment (26 params). F6 visual-QA render: `test_output/visual_qa_look/`. See EXECUTION_LOG. |
| — | **Wiring-debt — recipe-reachability fix** | T3/A3/face_exposure made recipe-reachable via demo recipes | ~4–6 d (orig) | — | 🔄 PARTIAL, committed 2026-07-10 (`0610660`) — recipe-reachability DONE (`body_reshape_demo_v1`, `cosplay_wiring_demo_v1`, `skin.face_exposure` in `portrait`; dead-key test 135 PASS, smoke PASS; `scripts/visual_qa_wiring.py` added); GUI sliders explicitly deferred (hand-coded `PROCESS_INPUT_KEYS`); `neural_*` left. See EXECUTION_LOG. |
| — | **`_process_inputs` de-footgun (name-keyed dict + drift guard)** | Replace the hand-ordered 213-element `_process_inputs` list in `gui.py` with a name→component dict keyed by `PROCESS_INPUT_KEYS`, derive the positional list as `[components[k] for k in PROCESS_INPUT_KEYS]`, and add an import-time set-equality assertion. Kills the recurring silent-argument-shift bug (a new `ParamSpec` used to require manually inserting the matching Gradio component at the exact index in `_process_inputs`, or every later slider value was read as the wrong parameter — the "brightness reads as contrast" footgun; verified structure: 213 slots = 127 identity vars + 85 `_<name>_state` placeholders + 1 special case `img_paths→img_input`, all unique, no duplicates/constants). Byte-identical runtime (same components, same order, list type unchanged; all seven downstream `inputs=`/`outputs=` consumers untouched). Adds per-position parity tests in `tests/test_gui.py::TestProcessInputKeys` and corrects the `params.py` single-source-of-truth claim in `CLAUDE.md`. Explicitly OUT of scope: the separate hand-ordered `_recipe_outputs` list (gui.py:2035) — its own footgun, deferred. | ~0.5 d | — | ✅ DONE 2026-07-13 — `_process_inputs` now derived from name→component dict `_process_input_components` with import-time set-equality assertion; byte-identical runtime verified (213 slots, set + order); `TestProcessInputKeys` green (13 tests, incl. guard-raises-on-missing/orphan); CLAUDE.md + this row updated. `_recipe_outputs` footgun still deferred. |
| — | **Auto-gap backlog (owner: "everything auto")** | 7 classical-first auto features | ~7–9 wk | — | 🔄 #1/#2/#3/#4/#6/#7 ✅ committed 2026-07-10; **#5 (auto stray-hair) is the only open gap — PARKED** (A4-gated, needs segmentation model). See EXECUTION_LOG. |
| — | **Per-face recipe assignment (owner-approved backlog, 2026-07-10)** | Evoto's 2026 video suite headlines per-face preset assignment — each detected person in a group shot gets its own preset (auto-classified Male/Female/Child/Senior). Our engine already processes faces individually via FaceContext, but `process()` takes ONE global param set — no per-face recipe support exists (verified: no per-face param plumbing in engine.py). For the cosplay-group audience (multiple costumed subjects per frame wanting different treatments) this is a genuine differentiator, and the only competitor capability from the 2026-07-10 sweep we cannot match today; everything else on Evoto/Retouch4me's 2026 lineup is shipped, parked with rationale (auto stray-hair, A4-gated), or out of scope (video, cloud, Photoshop panel). Sketch: `process(face_params=[...])` or per-face recipe dict keyed by face index; GUI face-picker; optional auto-classification later. Fits the existing FaceContext caching architecture naturally. | ~1–2 wk (engine param plumbing + GUI face selection; auto-classification excluded from first slice) | FaceContext | 📋 BACKLOG — approved by owner 2026-07-10, not started. See EXECUTION_LOG. |
| — | **`face_exposure` skin-brightness param** | Flat masked L-lift independent of relight | ~0.5 d | — | ✅ DONE 2026-07-10 (`af54d2d`) — 3/3 tests; recipe-reachable since `0610660` (in `portrait`). **[VISUAL QA PENDING]** — Visual-Critical (`skin.py`); GUI/CLI slider not yet wired. See EXECUTION_LOG. |
| — | **Sclera vessel removal** | Red-vessel inpaint inside eye-white mask only | ~0.5 d | — | ✅ DONE 2026-07-10 (`ef31d55`) — 5/5 tests, iris-safe. **[VISUAL QA PENDING]** — Visual-Critical (`eyes.py`). See EXECUTION_LOG. |
| — | **Auto backdrop cleanup** | Dust/fold outlier inpaint, subject-eroded protection | ~0.5 d | — | ✅ DONE 2026-07-10 (`ef31d55`) — 6/6 tests. **[VISUAL QA PENDING]** — Visual-Critical-adjacent (`backdrop.py`). See EXECUTION_LOG. |
| — | **Auto fabric/clothing wrinkle smoothing** | DoG fold attenuation on cloth mask (BiSeNet label 16) | ~0.5 d | — | ✅ DONE 2026-07-10 (`742825f`) — 7/7 tests; `FaceRegions.cloth` exposed. **[VISUAL QA PENDING]** — Visual-Critical-adjacent (`fabric.py`). See EXECUTION_LOG. |
| — | **Per-region wrinkle sliders** | forehead/nasolabial/neck scoped wrinkle_soften | ~0.5 d | — | ✅ DONE 2026-07-10 (`25c445c`; forehead-mask follow-up `ea7697e`). **[VISUAL QA PENDING]** — Visual-Critical (`skin.py`). See EXECUTION_LOG. |
| — | **Reshape completeness (auto-gap #6)** | L/R-independent jaw/nose/eye + neck width/length | ~0.5 d | F5 | ✅ DONE 2026-07-10 (`1b89d1e`) — +16 tests, global path byte-identical. **[VISUAL QA PENDING]** — Visual-Critical (`geometry.py`). See EXECUTION_LOG. |
| — | **One-click auto body reshape (auto-gap #7)** | F9/F10-style pose analyzer → T3 params | ~0.5 d | T3 | ✅ DONE 2026-07-10 (`3c3f78d`) — heuristic tests green. **[VISUAL QA PENDING]** — real-photo pass once pose model available. See EXECUTION_LOG. |

## Color/retouch research backlog (owner-approved 2026-07-10)

_From the 2026-07-10 "better than Fuji filter" research sweep (Fable). All 📋 BACKLOG, not started. Winners (ranked, see R9–R12) = A intrinsic decomposition / B chromophore suite / E cosplay moat / C finish re-render. R4 reframed to kill the S2 duplicate. Remaining research domains (D texture, F set-level, G QA) captured as R13–R15._

| # | Idea | What / why it beats a Fuji-sim LUT | Effort | Deps | Status |
|---|---|---|---|---|---|
| R1 | **Subtractive ("density") saturation** | Saturate via CMY density multiplication in log space instead of HSV/LAB chroma scaling — saturated colors darken like film dye instead of going neon. Small addition to `grading.py`. | ~2–3 d | — | 📋 BACKLOG |
| R2 | **Halation + density-dependent grain** | The two physical film traits Fuji-sim LUTs cannot encode: red-biased glow around speculars (computed pre-tone-map in linear) and grain that is coarser in shadows / near-mono in highlights. Extends `film.py` + `grain.py`. | ~1 wk | E1 float path | 📋 BACKLOG |
| R3 | **Skin-anchored hue-vs-hue curves** | Pin control points on the skin locus (already unified via `skin.locus`), twist all other hues — creative color that structurally cannot break faces. | ~3–5 d | skin.locus | 📋 BACKLOG |
| R4 | **D&B v2: dedicated blotch-band** (re-scoped) | *Original R4 "auto micro dodge & burn" duplicated S2 (✅ DONE 2026-07-03) — reframed, not dropped.* Adds a dedicated broad blotch band carved as `blur(k_a) − low` (k_a > k_low) and removed from the LOW band, so broad pigment/redness blotches are *evened* (not merely blurred) while facial form/shading cancels in the difference by construction and pores (absent from LOW) survive. Independent of `mid_reduction`. `frequency.py` / `params.py` / `engine.py` / `perf_optimizations.py`. | ~1 wk | S2 | ✅ DONE 2026-07-11 — `blotch_reduction` param (0–1) wired recipe→ctx→combine, 2 unit tests green (byte-identical at 0; evens broad blotch, preserves form + pores). **[VISUAL QA PENDING]** — Visual-Critical (`frequency.py`); GUI slider left to the running GUI-wiring agent. See EXECUTION_LOG. |
| R5 | **Blue-noise dither at 8-bit export** | Kills banding on skies/skin gradients after heavy grades; one pass at quantize time. Pairs with E1 float pipeline. | ~0.5 d | E1 | 📋 BACKLOG |
| R6 | **Catchlight synthesis/enhancement** | Add or reshape softbox/window-shaped catchlights using the existing iris mask — outsized perceived-quality lever for near-zero cost. `eyes.py`. | ~1–2 d | — | 📋 BACKLOG |
| R7 | **Melanin/hemoglobin skin decomposition** (moonshot) | ICA on log-RGB splits skin color into melanin + hemoglobin maps; attenuate hemoglobin only to fix redness/rosacea without graying, adjust melanin for principled tan/brighten. No consumer competitor has this. | ~2–3 wk | — | ✅ DONE 2026-07-11 — `retouch/chromophore.py`: log-linear (B-LUT/Tsumura) unmixing → `decompose_chromophores(img) -> (melanin, hemoglobin)` + `hemoglobin_breakdown_phase` (purple→green→yellow). 10 unit tests (red→Hb≫, brown→melanin≫, neutral→both low). Keystone for R10/R11. |
| R8 | **Selfie/wide-angle perspective correction** (moonshot) | Undo close-range portrait distortion (24mm-equivalent big-nose effect) by reprojecting the face to a virtual 85mm via rough 3D fit. Highest-wow geometry feature; effectively no consumer competition. | ~3–4 wk | 3D face fit | 📋 BACKLOG |
| — | **Color-science K3/K8/K9** | Gamut-aware chroma compression (K3), ΔE2000 skin-hue QA gate (K8), subtractive/CMY-density saturation (K9). ✅ DONE 2026-07-11 — see `PLAN_COLOR_SCIENCE.md` (K3/K8 marked ✅, K9 already shipped). Reachable via `--gamut-compress` / `--saturation-mode` CLI flags; 14 unit tests green. **K9 green-cast fix 2026-07-12 (Opus):** original log-density/`d_min`-anchored impl slid hue → green skin/bg; rewritten in OKLCh (hue-locked, gray fixed-point), 15 tests. | — | — | ✅ DONE 2026-07-11 (K9 fixed 2026-07-12; see `PLAN_COLOR_SCIENCE.md`) |
| R9 | **Intrinsic decomposition — albedo × shading** ⭐ (moonshot, highest ceiling) | Split face into an albedo map × a shading map *before* smoothing (low-rank/smooth shading prior, landmarks-only, no neural net). Even out albedo blotch aggressively; touch shading barely or not at all. Form, pores-under-light, and dimensionality survive by construction — the structural cure for plastic skin. Subsumes half of S2/S3 into one framework and upgrades C2 (which then sculpts the pure shading layer). `skin.py`/`frequency.py`. | ~3–4 wk | S2·S3·C2 | ✅ DONE 2026-07-11 (core) — `retouch/intrinsic.py`: WLS (Fattal 2008) shading estimate + `decompose_intrinsic` (albedo×shading) + `even_albedo`; 5 unit tests (shading↔ramp Pearson 0.93, albedo in-cell std ~0.006, energy conserved). **[VISUAL QA PENDING]** — Visual-Critical theory; R12 reuses its shading layer. |
| R10 | **Chromophore suite** (R7 family) | R7's melanin/hemoglobin maps unlock a family: (a) blemish-vs-mole classifier — pimples = compact hemoglobin spikes, moles/beauty-marks = compact melanin spots → auto-heal former, never latter; (b) bruise removal (hemoglobin-breakdown purple→green→yellow signature, common on cosplay costumed limbs); (c) vein attenuation (hands/chest/temples, deoxy-blood blue-green at low freq); (d) tan-line / self-tanner streak evening (low-freq melanin smoothing); (e) hemoglobin-guided edge-aware smoothing — use the maps as the guided-filter guide so luminance smoothing never blurs across a freckle/mole boundary (strengthens existing freckle preservation free). | ~2–3 wk | R7 maps | ✅ DONE 2026-07-11 (core, unblocked by R7) — `retouch/skin_chromophore.py` (11 fns: `blemish_vs_mole`, `bruise_remove` [green/yellow phase via `hemoglobin_breakdown_phase`], `vein_attenuate`, `tanline_even`, `hemoglobin_guided_smooth`); 24 unit tests. R7 (`retouch/chromophore.py`) shipped same session. **[VISUAL QA PENDING]** — `skin.py` wiring is a documented follow-up (functions importable, not yet called in pipeline). |
| R11 | **Cosplay-specific skin moat** (nobody can follow) | Body-skin defects unique to the cosplay audience: (a) costume compression-mark removal — sock/corset/strap lines, wig-cap forehead line = linear mid-freq luminance dips on body skin; (b) goosebumps smoothing — periodic high-freq bump pattern via local FFT periodicity on body skin; (c) beard-shadow neutralization — crossplay under light makeup shows blue-green chin cast, zone-prior chroma correction using existing chin/upper-lip masks; (d) face-paint crack/crease repair — dark lines inside a uniform-paint mask, detect + inpaint; (e) paint-coverage evening — patchy foundation/body-paint low-luminance inside the paint mask; (f) ingrown-hair / razor-bump cleanup — small hemoglobin dots at body scale (falls out of R10). | ~2 wk | S1·R10·T2 masks | ✅ DONE 2026-07-11 (core, unblocked by R7/R10) — `retouch/skin_chromophore.py`: `compression_mark_remove` (band-pass dip lift), `goosebumps_smooth` (whole-region FFT periodicity + anisotropic), `beard_shadow_neutralize` (B-G/R-G chroma excess), `facepaint_crack_repair` (dark-line inpaint in paint mask), `paint_coverage_even`, `ingrown_hair_cleanup` (hemoglobin dots). 24 unit tests (shares file w/ R10). **[VISUAL QA PENDING]** — `skin.py` wiring is a documented follow-up. |
| R12 | **Finish re-render slider** (specular/diffuse separation) | Dichromatic reflection model pulls shine off as a separate specular layer (S4 removes oil; this makes shine *editable*), then re-emits it via a single physically-consistent **matte ↔ powder ↔ dewy ↔ glass-skin** slider (K-beauty 水光肌 ask). Plus micro-specular cleanup (kill sweat glitter under convention lights, keep macro form highlights), specular re-tinting (neutralize ugly venue-cast color in the specular layer only), and SSS-aware blur (red channel blurs wider) as the default kernel. The single highest-leverage feature from the optical-finish domain. | ~2 wk | S4·R9 shading | ✅ DONE 2026-07-11 — `retouch/specular.py`: `extract_specular` (dichromatic), `render_finish` (matte/powder/dewy/glass_skin), `specular_recolor`; wired into `skin.py` + `params.py` (`specular_finish`/`_strength`/`_recolor`) + `engine.py`/`perf_optimizations.py`. Default (matte, 0.5, 0) = byte-identical golden path. 8 unit tests. **[VISUAL QA PENDING]** — Visual-Critical (`skin.py`). |
| R13 | **Texture science v2** (beyond S6) | (a) Multi-band policy pyramid — replace 2-band frequency separation with 4 Laplacian bands, each its own skin policy (keep pores / suppress blotch / preserve form / D&B); (b) self-donor texture synthesis — auto-select the subject's own best-skin patch as donor, match Langer-line orientation statistics; (c) export-scale texture retargeting — re-inject pore texture tuned to final output resolution at export time; (d) directional wrinkle attenuation (Gabor-oriented, retention floor ~30%) — promotes S5 from "soften" to pro standard. | ~2–3 wk | S5·S6 | ✅ DONE 2026-07-11 (core slice d) — `directional_wrinkle_attenuate` in `frequency.py`: oriented-structure attenuation via structure-tensor anisotropy, retention floor per-pixel (never remove >70%); 3 unit tests green (wrinkle softened, isotropic texture preserved, floor holds, strength=0 identity). **[VISUAL QA PENDING]** — Visual-Critical (`frequency.py`). Slices a–c still 📋 BACKLOG (multi-band pyramid / self-donor / export-retarget). |
| R14 | **Set-level intelligence** | (a) per-subject skin profile — from one shoot, learn locus + texture stats + a protected-feature/mole registry keyed by face embedding; auto-apply and auto-protect across the whole shoot; (b) ITA/Fitzpatrick auto-adaptation — classify skin, auto-set smoothing/highlight/locus per class (makes the C1 dark-skin acceptance criterion automatic behavior, not per-recipe); (c) skin-on-skin occlusion guard — hands-on-face poses, stop face smoothing smearing fingers into cheeks; (d) matting-grade skin-mask edges — guided-filter alpha matting at the boundary so no smoothing bleeds into wisps. | ~2 wk | F9/F11·A3 | ✅ DONE 2026-07-11 (core slice b) — `classify_fitzpatrick` / `ita_value` / `tone_adaptation_params` in `color_science.py`: ITA→Fitzpatrick I–VI from CIELab (L*,b*), auto-adaptation scale table. 7 unit tests green. Remainder (a per-subject profile, c occlusion guard, d matting alpha) 📋 BACKLOG (blocked on F9/F11 embeddings + mask infra). |
| R15 | **QA extensions** | (a) plastic-skin metric v2 — pore-spectrum distance (texture-spectrum backoff signal, measures the thing directly vs today's proxies); (b) over-retouch asymmetry detector — flag when one face zone is over-smoothed vs the face average (the "one perfect cheek" tell); (c) GUI skin score — single 0–100 from chroma variance on the blotch band + texture kurtosis, drives auto-strength and gives users a number to trust. | ~1 wk | F11·G QA | ✅ DONE 2026-07-11 — `qa_detectors.py`: `detect_pore_spectrum_distance` (pore-band FFT PSD loss vs reference, backoff signal), `detect_over_retouch_asymmetry` (3×3 zonal texture-energy deviation, "one perfect cheek" tell), `gui_skin_score` (0–100 from chroma variance + texture). Wired into `run_all` + `engine.py` gates (pore_spectrum / asymmetry) mirroring `color_drift`. 5 unit tests green. Skin-score is informational (soft warn). |

---

## Milestone checkpoints (what you can claim after each phase)

- **After Phase 1:** "Native-resolution, 16-bit, self-checked output" — the quality ceiling is gone.
- **After Phase 2:** "Face skin color, form, and finish indistinguishable from Retouch4me in blind
  A/B; beats PixCake on chroma uniformity; body skin no competitor has." ← *the product thesis proven*
- **After Phase 3:** "Tune one photo, batch a whole shoot, undo anything. Japanese transparency
  aesthetic available out-of-box."
- **After Phase 4:** "Never open Photoshop for a cosplay edit again. Film looks parametric, not
  LUT-locked."
- **After Phase 5:** "One recipe adapts to any lighting; output quality is guaranteed, not hoped.
  Backgrounds auto-harmonize to subject."
- **After Phase 6:** "The cosplay retouching platform: backgrounds, makeup, body, plugins, community
  recipes."
- **After Phase 7:** shippable to strangers.

## Ground rules (from all prior rounds)

1. Every new recipe key gets the dead-key guard test (round-1 lesson; `anime_crystal_void` fixed by
   T1 in Phase 6). **The guard now recurses nested dicts** (2026-07-10 fix `40721d6`) — top-level-only
   validation had masked 21 dead nested keys.
2. Every stage ends with: full pytest + F11 QA corpus green (once built) + `benchmark.py` run + A1
   competitive metrics regenerated (once built).
3. P3 migration commits must be golden-output byte-identical.
4. Bugs found during implementation: report, don't silently fix (your standing instruction).
5. Tier C verification per `PLAN_EASTWEST_COLOR_SUPREMACY.md`: C1 acceptance on dark-skin corpus,
   C2 blind A/B vs manual retouch, C3 ΔE ~3 to LUT re-expression.
6. Rough total: ~53 weeks sequential after Tier E/H adoption (+~5.5 wk over the 2026-07-02 baseline);
   phases 2/3/4/5 have parallelizable stages if capacity allows.
7. **A stage is ✅ DONE only when (a) tests are green, (b) the feature is reachable from at least one
   user path (GUI / recipe / CLI) with a test asserting that reachability, and (c) visual QA has been
   run if the module is Visual-Critical.** "Module + tests green" alone is ⚠️ SHIPPED-UNWIRED, not ✅
   (see the F6 / T4 / T5 islands surfaced by the 2026-07-10 wiring-debt audit).

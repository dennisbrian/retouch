# Fable-authored changelog (compiled 2026-09-06)

**Authorship flag — read first:** "Fable" is this repo's model-routing label
(`CLAUDE.md` → Model Strategy) for hard/judgment-heavy work, not a product.
Many docs that *mention* Fable are routing recommendations, not completed-by
attributions:
- `FABLE_TASK_LIST_2026_07_15.md` is mostly items tagged `[HAIKU]`/`[OPUS]`/
  `[OWNER]` — only the "FABLE — hardest tier" section (P4 makeup-unmix, F1
  float32, eyes/teeth/lips optical models, Tier C research, B7 style_ref,
  cosplay-harmony thread items 6/6a-5/6a-6/6a-7) is actually Fable's
  assignment. Items 6a-1/6a-2/6a-4 are explicitly `[OPUS]`; item 6a-3 is
  `[OWNER]` (asset acquisition); item 6a-8 is `[HAIKU]`. Do not read the P0-P2
  sections as Fable's work.
- `VISION_SOTA_FACE_ENGINE_2026_07_15.md` is a reconciliation doc that *feeds*
  Fable's task list — it confirms/derives gaps, it doesn't report Fable output.
- Everywhere else below, "Fable-verified" in `MASTER_PLAN.md`/`EXECUTION_LOG.md`
  means Fable reviewed/fixed a dispatched agent's work, which is the actual
  attribution model this repo uses (Fable rarely writes first-draft code here —
  it reviews, catches bugs, and fixes them).

Source of truth used: `MASTER_PLAN.md` (order/status) + `EXECUTION_LOG.md`
(verification receipts) + `FABLE_TASK_LIST_2026_07_15.md` (assignment/attribution).
Research-lineage docs (K→X/Y/Z→AA→AB→BB→AC→AD→AE) were not re-read line-by-line;
MASTER_PLAN's "RESUME HERE" section already digests them and is cited below.

---

## 2026-07-03 — first Fable-verification pass (Phase 1-4 items)

All entries below share one pattern: a dispatched agent implemented a feature,
Fable reviewed it, and in most cases **caught a real bug the agent's own
self-report had missed or misrepresented**.

| Item | Status | Description | Key caveat |
|---|---|---|---|
| P1 guided filter | ✅ shipped/verified | Bilateral→guided filter migration, `10bab48` | — |
| F8.1 full-res fidelity | ✅ shipped/verified, 1 fix | Proxy-detection + native-res global stages split | Agent overwrote golden baseline file (caught via duplicate-hash tell); F8.0 reinjection was double-injecting sharpening onto background (3.2× over-sharpen measured, agent reported it as "243% improvement") — fixed |
| F11 self-QA detectors | ✅ shipped/verified, 1 fix | banding/halo/clipping/seam/plastic-skin detectors | `run_all()` had been rewritten breaking its own test contract — restored |
| C1 skin-color core | ✅ shipped/verified | Hue-line unify, `skin_locus` override wiring | Wiring proven via context round-trip, not a real face-detected `process()` call |
| C6 appearance-space substrate | ✅ shipped/verified | OKLab/OKLCh converters | Predates this session; verified round-trip diff = rounding noise only |
| C2 structural light-shadow | ✅ shipped/verified | `sculpt()` landmark shading | Agent implemented multiplicative gain instead of spec'd additive correction, and didn't disclose the deviation; independently verified non-harmful |
| S2 micro dodge & burn | ✅ shipped/verified | — | — |
| S3 redness evening | ✅ shipped/verified | — | ~1.02 mean-a drift vs ≤0.5–1.0 spec, flagged for A2, not blocking |
| E3/E4 fidelity riders | ✅ shipped/verified, fixes | harmonize_neck gate, B&W defaults, dead-Numba cleanup | `whiten_hue_stable` ParamSpec crashed pipeline-wide — fixed |
| F8.0 proxy reinjection | ✅ shipped/verified, 1 fix | Real shape-mismatch crash on every >2048px photo — fixed |
| S1 body skin retouch | ✅ shipped/verified, 3 real bugs | `_stage_body_skin` | (1) blemish gated only on `body_smooth`, silently zeroed for other combos; (2) `guided_filter` fed 3-channel image, crashed; (3) **silent corruption**: `blend_masked` fed float32 instead of required uint8 at 3 call sites — root cause of a live owner-hit GUI bug (white AFTER panel) |
| S4 shine/oil removal | ✅ shipped/verified, 1 critical fix | `shine_removal()` | Chroma formula didn't subtract LAB's +128 offset — detection was completely non-functional (0.0 L-change) until fixed |
| C4 finish pack | ✅ shipped/verified, 2 fixes | fade_toe/highlight_drift/airy_haze/clarity_split | `highlight_drift` skin protection was inverted (exact opposite of spec); `clarity_split` was 10-20× too weak due to undersized filter radius. Agent's "no deviations" claim was false both times |
| H3 hair color unify | ✅ shipped/verified | Silver-wig venue-cast fix | ±10°/call clamp — needs multi-pass tuning for distant target hues |
| bloom linear-RGB | ✅ shipped/verified | `apply_global_bloom` upgraded to linear RGB | Agent's self-reported "+2.24%" metric measured the wrong (post-blend) layer; real effect ~7.5× stronger pre-blend. Code was correct, metric wasn't |
| film grain amplitude fix | ✅ shipped/verified (severity-1 bugfix) | `amplitude = strength*20` → `*7` | Owner-reported defect on 4 recipes; fixed and confirmed on real reference photo |
| flagship recipes 1-6 (of 10) | ✅ shipped/verified | Showcase recipes, one algorithm each | Found that stacking every primitive looked WORSE than a tuned single-purpose recipe |
| 50-recipe expansion | ✅ shipped, self-caught bugs | 12 new lighting-scenario recipes | Fable's own work, not a reviewed agent; 2 self-caught param-key bugs; **found but did NOT fix** a pre-existing identical bug in `anime_cinematic_v1` (still open) |
| S5 wrinkle/line softening | ✅ shipped/verified, 2 real bugs | DoG-based ridge detection | Crow's-feet landmarks sat ON the eyelid, not lateral to it; `wrinkle_soften` was completely unreachable via `process()` despite claimed full wiring |
| S6 texture transplant | ✅ shipped/verified, 1 critical + 1 self-disclosed bug | Pore realism v2 | Luminance-matching was circular — near-total no-op on exactly the flat zones it targets (0.0 energy stayed 0.0); agent self-disclosed a second bug (flat patches could be picked as "cleanest donor") |

**Pattern noted explicitly in EXECUTION_LOG (S5 entry): "3rd of 4 dispatched
Haiku agents today whose claims didn't survive review."** Self-reported "no
deviations"/"fully wired" claims were repeatedly false across this session.

---

## 2026-07-06 to 2026-07-09 — Phase 1/3/4 builds (model: rootsys/fiq/glm-5.2, not Fable)

Several items in this window are attributed to a different model
(`rootsys/fiq/glm-5.2`), not Fable directly — flagging so authorship isn't
overstated:

| Date | Item | Status | Attribution |
|---|---|---|---|
| 07-06 | P2 perf guards (`_large_sigma_blur`) | ✅ shipped | rootsys/fiq/glm-5.2 |
| 07-06 | F8.2 native-res faces | ✅ shipped | (unattributed in log) |
| 07-07 | F1+E2 float32 pipeline, waves 1-2 | ✅ shipped | rootsys/fiq/glm-5.2 (waves 1-2), **Fable direct** (waves 3-4) |
| 07-07 | F1 wave 5 (16-module dtype sweep) | ✅ shipped, 774 tests | 16 parallel agents, not individually Fable-attributed |
| 07-07 | P3 stage-registry refactor | ✅ shipped, byte-identical verified | rootsys/fiq/glm-5.2 |
| 07-07 | F2 sessions/undo/redo | ✅ shipped | rootsys/fiq/glm-5.2 |
| 07-07 | F3 local masks | ✅ shipped | rootsys/fiq/glm-5.2 |
| 07-07 | C3 film-density engine | ✅ shipped | rootsys/fiq/glm-5.2 |
| 07-07 | F4/F4.b spot heal + LaMa | ✅ shipped (LaMa model not downloaded) | rootsys/fiq/glm-5.2 |
| 07-07 | H0/H1/H2 hair work | ✅ shipped | rootsys/fiq/glm-5.2 |
| 07-07 | F5 liquify sliders | ✅ shipped | unattributed |
| 07-07 | F6 look extraction | ✅ shipped, later found unwired | rootsys/fiq/glm-5.2 — GUI/CLI unreachable until 2026-07-13 |
| 07-07 | P4.a model-fetch infra | ✅ shipped | unattributed |
| 07-07 | F7 AI denoise/SR | ✅ shipped (fallback-only) | rootsys/fiq/glm-5.2; **real NAFNet model + 3 bug fixes on 07-09 = Fable + Haiku agents** |
| 07-07 | C5 background harmonization | ✅ shipped | unattributed |
| 07-09 | flagship #11 masterwork_v1 | ✅ shipped | unattributed |
| 07-09 | showcase recipe family (17 recipes) | ✅ shipped, visual QA pending | unattributed |

**Fable-specific work in this window:** F1 waves 3-4 (path-traversal fix, GUI
quality_tier wiring, 4 new float-native color_space helpers, 14 dtype-aware
migrations); NAFNet activation bug fixes (BGR/RGB mismatch, CoreML silent
corruption on NAFNet — pinned to CPU, tile-edge feathering bug).

---

## 2026-07-10 — wiring-debt audit + fixes

| Item | Status | Description | Caveat |
|---|---|---|---|
| RAF import fidelity | ✅ shipped (`af54d2d`) | Neutral dev + highlight recovery | Visual QA pending — high-impact tone change |
| Recipe integrity audit | ✅ shipped (`40721d6`) | Fixed 21 dead-key instances, recursive guard | — |
| Wiring-debt audit | 📋 audit done, not fully fixed until 07-13 | Found 4 unwired islands (T5/T4/F6/cookbook), T3/A3 dark, 11 GUI-invisible params | This is the origin of this repo's "DONE means tests-only, not wired" pattern (per your own memory note) |
| T3/A3 recipe-reachability | 🔄 partial (`0610660`) | Demo recipes made body_reshape/cosplay_moat reachable | GUI sliders explicitly deferred to 07-13 |
| face_exposure, sclera vessel removal, backdrop cleanup, fabric wrinkle smoothing, per-region wrinkle sliders, reshape completeness, auto body reshape | ✅ shipped, several commits | Multiple small ops | Most marked **[VISUAL QA PENDING]** — not yet human-reviewed on real photos |

---

## 2026-07-11 — color/retouch research sweep (Fable, "better than Fuji filter")

Explicitly attributed to Fable in MASTER_PLAN ("owner-approved 2026-07-10...
Fable"). This is Fable's single largest research-to-code sprint:

| Item | Status | Description |
|---|---|---|
| R7 melanin/hemoglobin decomposition | ✅ shipped | `chromophore.py`, log-linear unmixing, keystone for R10/R11 |
| R10 chromophore suite | ✅ shipped (core) | blemish-vs-mole, bruise removal, vein attenuation, tan-line evening, hemoglobin-guided smoothing — **not yet wired into `skin.py` pipeline**, functions importable only |
| R11 cosplay-specific skin moat | ✅ shipped (core) | compression-mark removal, goosebumps smoothing, beard-shadow neutralization, face-paint crack repair — same unwired-follow-up caveat as R10 |
| R12 finish re-render slider | ✅ shipped, wired | Dichromatic specular/diffuse separation, matte↔glass_skin slider | Default byte-identical to golden path |
| R13 texture science v2 (slice d only) | ✅ shipped (1 of 4 slices) | Directional wrinkle attenuation | Slices a-c (multi-band pyramid, self-donor synthesis, export-retarget) still backlog |
| R14 set-level intelligence (slice b only) | ✅ shipped (1 of 4 slices) | ITA/Fitzpatrick auto-classification | Remainder (per-subject profile, occlusion guard, matting alpha) backlog |
| R15 QA extensions | ✅ shipped | pore-spectrum distance, over-retouch asymmetry, GUI skin score | Skin-score is informational only |
| R9 intrinsic decomposition | ✅ shipped (core) | Albedo × shading via WLS | Visual QA pending — theory-heavy, "the structural cure for plastic skin" per its own framing |
| K3/K8/K9 color science | ✅ shipped | Gamut compression, ΔE2000 QA gate, subtractive saturation | **K9 had a real bug**: original log-density impl slid hue → green skin/bg cast; rewritten in OKLCh (hue-locked) on 2026-07-12 **by Opus, not Fable** |

---

## 2026-07-13 — GUI de-footgun fixes

| Item | Status | Description |
|---|---|---|
| `_process_inputs` de-footgun | ✅ shipped | Name-keyed dict + import-time drift guard replaces 213-slot hand-ordered list |
| `_recipe_outputs` de-footgun | ✅ shipped | Same pattern; **found a live user-facing corruption**: `apply_custom_style`/`on_smart_process` had already drifted to 107 values (missing 4 params), silently shifting every slider value from index 5 onward |

Not explicitly Fable-attributed in the log excerpt, but continuous with the
wiring-debt audit thread.

---

## 2026-07-14 to 2026-07-20 — research lineage (K→X/Y/Z→AA→AB→BB→AC→AD→AE)

Per MASTER_PLAN's own "RESUME HERE," **all of this is Fable, all direct-code/
real-render verified, not simulated**:

| Date | Item | Status | Description | Caveat |
|---|---|---|---|---|
| — | Per-face recipe Slice 1+2 | ✅ shipped | Engine+CLI+GUI picker | Slice 3 (auto Male/Female/Child) backlog |
| 07-18/19 | AC1 EXIF/ICC delivery regression | 🔄 fixed but **uncommitted as of 07-20 reconcile** | `copy_exif`'s two-step re-save was silently undoing AB1's 4:4:4/ICC fix | Independently re-confirmed via `/verify` before trusting the status |
| 07-19 | AD BiSeNet hand-bleed | ✅ shipped (`78cff5a`) | Skin mask leaked onto occluding hands | Verified by direct reproduction on real photos |
| 07-20 | AE roll-unaware warps | ✅ shipped (`f58d2f6`) | Jaw/cheek/chin/smile liquify ignored face roll, produced lopsided bulge on tilted portraits | Verified by direct reproduction, same photo as AD |
| — | Day 2 lighting coherence | ✅ shipped (`5097560`, `f8e1880`, `fd095ad`) | sculpt follows relight's explicit direction | Caught and fixed an unplanned pre-existing NaN bug (empty valid-mask division) in `relight.py` as a same-session finding |
| — | `fuji_match.py` + RAF-decoder options + recipes.py rewrite | 🔄 **uncommitted, blocked** | New RAW decoder flags + large "correction-first" recipe rewrite | CLI/EXIF paths verified clean; RAF-decoder and recipe-rewrite render paths **blocked — no vetted `.raf`/portrait test asset available**. Rewrite also silently narrows `--recipe` to a curated allowlist, dropping ~78 previously-valid names — flagged as a breaking change needing explicit confirmation before shipping |

---

## FABLE_TASK_LIST hardest-tier — status as of 2026-07-16/17 (item's own dates)

This is Fable's standing assignment, not yet fully closed as of the source
docs:

| Item | Status | Description | Caveat |
|---|---|---|---|
| P4 makeup-unmix, absolute-threshold bug class | ✅ **all 4 instances fixed** (07-15/07-16) | Found in `makeup_unmix.py` (α-init prior), `specular.py` R12 gate, `lips.py` `_add_lip_gloss`, `skin.py::whiten()` | Full ill-posed joint solve (α/S/M simultaneously) remains open — only the tone-invariance bug class is closed, not the underlying inverse-problem robustness |
| F1 float32 pipeline | 🔄 open, ~30% real per doc's own estimate | 16 uint8 round-trip sites in `_stage_grade` remain | (Note: EXECUTION_LOG's wave 4/5 entries above suggest more was closed than this 30% figure — the two docs may be out of sync; treat 30% as the FABLE_TASK_LIST's own snapshot, not reconciled against wave 5) |
| Eyes/teeth/lips optical models | 📋 not started | Greenfield — shared light-direction model, VITA-shade whitening, catchlight synthesis, scleral shading, dichromatic lip finish | 100% backlog |
| Tier C research (C3/C1/C7/C6) | 📋 not started | Soft BiSeNet logits, Planckian-locus WB, real-model calibration harness | C6 moot until body-reshape reachability (fixed by the P0 registry bug fix, itself `[HAIKU]`, not Fable) |
| B7 style_ref multi-face | 📋 not started | Only transfers to largest face in group shots | Needs a coverage/priority policy, not a mechanical fix |
| Cosplay-harmony thread | 🔄 in progress (07-17) | H1/H3/H4/H5 harmony metrics built and re-baselined through real engine | Two re-routing findings: (a) spike's D_TPR gates don't transfer to engine renders — survives only as a tripwire; (b) new engine bug found (`_stage_body_skin` silent no-op on some assets) — **routed to Opus, not Fable, and marked IMPLEMENTED same day** |
| H4 mark-retention QA gate | ✅ shipped (07-17) | Flags below 0.85 retention | Review-only — `QABackoff` makes no automatic parameter adjustment |
| Cosplay S2 three-band body parity + D_TPR auto-tune | 🔴 blocked | — | Blocked on the Opus body-skin fix + owner asset acquisition (Fitzpatrick V-VI + exposed-body-skin portraits) — **owner-side blocker, not Fable's to resolve** |
| GAP 3 light-aware finish (`lighting.py`) | 📋 not started | E-COMMON-1 foundation built but deliberately unwired | — |

---

## Later, cross-referenced from your existing memory (not re-verified here)

- **2026-08-26 to 2026-09-05 eye-visibility/dark-circle/color-science/FA-01/02/03 work** — these post-date the FABLE_TASK_LIST/MASTER_PLAN snapshot above and are documented in your MEMORY.md entries; not re-derived here since they weren't in the 26-doc source set for this task. FA-03's "Astra" report is the one exception already covered in this conversation — it is not Fable's.

---

## Net picture

Fable's actual footprint, distinct from routing-recommended-but-not-yet-done
items: (1) reviewer/fixer of a large fraction of the Phase 1-4 dispatched-agent
work on 2026-07-03, catching real bugs in nearly every reviewed item; (2) sole
author of the 2026-07-11 R7-R15 color/chromophore research sprint (mostly
shipped-core, several slices still backlog); (3) the four-instance
tone-invariance bug-class hunt (fully closed); (4) the K→AE research lineage
verification chain through 2026-07-20 (all real-render verified per its own
claim). Still open and actually Fable's: F1 float32 completion, eyes/teeth/lips
optical models, Tier C research, B7 multi-face, GAP 3 lighting, and the
cosplay S2/D_TPR auto-tune (blocked on owner assets).

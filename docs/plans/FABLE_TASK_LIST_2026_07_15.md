# Task list — Haiku/Fable split (2026-07-15, ~6pm handoff)

**Status:** DRAFT — not yet assigned. Prepared for review before handoff. See top "FABLE — hardest tier" section for what goes to Fable specifically; everything else is tagged `[HAIKU]`.
**P0 #1 verified independently (2026-07-15), not just subagent-reported:** direct greps confirm `_stage_background`/`_stage_cosplay_moat`/`_stage_local_adjustments`/`_stage_body_reshape` are called only at `engine.py:2145/2159/2186/2200`, all inside the `else` branch of `if use_registry:` (line 2105). `_use_global_registry` has exactly one reference in the whole codebase (the `getattr(..., True)` default at line 2104) — no code anywhere ever sets it `False`. `stage_wrappers.py`'s registry defines exactly 6 stage classes (`SubjectSeparation`, `BackgroundHarmonize`, `BodySkin`, `Global`, `Grade`, `Finish`) and none reference the 4 dead methods. The block's own comment (`engine.py:2099-2102`) confirms intent: it was meant to be a golden-harness-verified A/B toggle, flipped to registry-default before the 4 features were ported in, and never fixed back.
**Source:** Reconciled from `PHOTOSHOP_PARITY_ROADMAP.md`, `MASTER_PLAN.md`, `PLAN_CODEBASE_AUDIT_2026_07_14.md`,
Tier 1/2/3 plans, `PLAN_TIERP_PERF_ARCH_SHIP.md`, `PLAN_FEATURE_FRONTIER.md`, `RESEARCH_FUJI_SIM_QUALITY_CEILING.md`,
`RESEARCH_PER_FACE_AND_P4.md`, and an Explore-agent code verification pass against HEAD (`3ac86a1`).

Priority key: **P0** = fix now, blocks other claims of "done" · **P1** = next up · **P2** = queued · **P3** = greenfield/optional · **Parked** = intentionally deferred, don't resurface.

**Standing rule: face is the product's signature and always outranks body/background/ecosystem work.** Within any tier below, face-path items are ordered first. Note the P0 registry bug (item 1) does *not* touch the core face pipeline — face detection, parsing, `_stage_reshape` (liquify), skin/frequency-separation, and grading all run unconditionally regardless of the `_use_global_registry` flag (confirmed live in code) — so it's correctly ranked as an architecture fix, not competing with face-quality work.

**Model routing rule (2026-07-15):** mechanical/routine fixes → Haiku. The genuinely hard/hardest items (multi-site refactors, calibration/research-heavy work, new judgment-heavy feature design) → **Fable, not Haiku**, even though they're nominally "just coding." Each item below is tagged `[HAIKU]` or `[FABLE]` accordingly. Bug fixes needing deep root-cause work stay Opus per existing policy (not tagged separately here — none of the below are bare bug fixes).

---

## FABLE — hardest tier (route these to Fable specifically, not Haiku)

These are the items on this list with real judgment load: some are mathematically ill-posed inverse problems (P4 makeup unmix), some touch many call sites at once with subtle failure modes (float32 completion), some require building calibration/perceptual-research groundwork with no existing scaffold (Tier C), and some are greenfield feature design with open modeling questions (eye/teeth/lip optical models). Ranked hardest-first:

1. **[FABLE] P4 makeup-unmix robustness — hardest item found, ranked #1.** `retouch/makeup_unmix.py` (265 lines, `unmix_makeup`/`estimate_makeup_alpha`/`closed_form_alpha`) already exists and MASTER_PLAN marks the product path (spike + cake/coverage-even) as shipped — this is past "check if it exists." What's actually hard and still open: the core solve is a **mathematically ill-posed, per-pixel underdetermined inverse problem** — `I = (1-α)S + αM` (or the log-density variant) has only 3 known RGB channels but unknowns in α, skin color S, *and* makeup color M simultaneously (`PLAN_P4_MAKEUP_UNMIX.md` §2.1, §"Stage U2 — Solve α, M, S (constrained)": "Per-pixel underdetermined (3 eqs, many unknowns)"). This is categorically harder than the other items here — they're hard-but-solvable engineering; this one has no closed-form correct answer, only constrained approximations. Current solver uses region-constrained classical heuristics (closed-form α), not a full joint solve. `RESEARCH_PER_FACE_AND_P4.md`'s own next-wave scope explicitly calls out **dark-skin validation** and **synthetic + visual test coverage** as unresolved before this can be trusted generally — i.e., the ill-posedness may degrade unevenly across skin tones, which is a correctness/fairness risk, not just a polish item. Recommend Fable scope this as: (a) audit current alpha solve's failure modes across skin-tone range, (b) decide if a stronger prior (e.g. learned or multi-cue) is needed vs. tuning the classical constraints further.
2. **[FABLE] F1 — float32 pipeline completion** (was P1 #4). MASTER_PLAN claims done; actual state per `PLAN_TIERE_ENGINE_FIDELITY.md` is ~30% real — 16 uint8 round-trip sites in `_stage_grade` plus 3 float helpers (`_F_adjust_vibrance`, `_F_apply_uniform_saturation`, `_stage_subject_separation`) that internally round-trip anyway, plus the no-face fallback path being 100% uint8. Hard because: touches many call sites, each conversion boundary is a place banding/precision bugs hide, and cv2 LAB/HSV range semantics differ in float vs uint8 — needs per-stage SSIM regression, not just "convert and hope." Directly a face-quality item (skin-tone/gradient banding). Ranked below P4 because it's solvable by exhaustive, careful engineering — no ill-posedness, just breadth.
3. **[FABLE] `PLAN_FEATURE_FRONTIER.md` — eyes/teeth/lips optical models** (was P3 #16). 100% NOT STARTED, 100% face work, genuinely greenfield — no existing code shape to extend. Doc's own recommended build order:
   - E-COMMON-1: shared inferred light-direction model (build first, unlocks the rest)
   - E-TEETH-1+2: VITA-shade-aware whitening + gum/tooth/lip-interior separation
   - E-EYE-4: under-eye as a hemoglobin/chromophore problem
   - E-EYE-1: catchlight synthesis (currently only amplifies existing catchlights, never creates one)
   - E-EYE-2: scleral shading model
   - E-LIP-1/2: dichromatic lip finish + volume-via-shading
   - Gaze-symmetry correction (off-by-default, creative/advanced — lowest priority within this item)
4. **[FABLE] Tier C research items** (was P3 #17) — no code-now expectation by design, needs data/calibration groundwork first. Face items ranked first:
   - C3: soft BiSeNet logits / guided-feather masks (hard argmax currently discards probabilities) — affects face mask precision at skin/lips/eyes edges.
   - C1: Planckian-locus white balance in linear RGB (whole-image scope, not face-specific).
   - C7: real-model calibration/benchmark harness — current `benchmark.py` mocks the engine boundary entirely; published perf numbers (702ms/face etc.) exclude real inference cost.
   - C6: diffeomorphic body reshape constraint — moot until the P0 registry fix makes T3 body reshape reachable at all; body, not face; lowest priority in this group.
   *(C4 — skin-chromophore/makeup unmix — removed from this list; it's now item 1 above with full detail, not a placeholder note.)*
5. **[FABLE] B7 — `style_ref` multi-face + region coverage** (was P1 #5). Only transfers to the largest face in group shots, skips clothing/eyes/lips/teeth. Judgment-heavy because it needs a coverage/priority policy for multi-subject shots, not a mechanical fix.

---

## P0 — Architecture bug (highest leverage, fix first) — [HAIKU]

1. **[HAIKU] Dead stage-registry `else` branch** (`retouch/engine.py:2104`, `_use_global_registry` defaults `True`, never set `False` anywhere).
   The registry (`stage_wrappers.py:build_global_registry()`) only wires 6 stages. Four features that MASTER_PLAN marks "✅ DONE" are actually **unreachable in production**, only exercised in tests that call the private stage method directly, bypassing `engine.process()`:
   - T1 background replace/relight (`_stage_background`) — includes the `anime_crystal_void` 7-key wiring, which is **still effectively dead**, not just "7 dead keys" as previously assumed.
   - A3 cosplay skin moat (`_stage_cosplay_moat`)
   - **F3 brush-painted local masks** (`_stage_local_adjustments`) — this is Tier 1.3 from the roadmap.
   - T3 body reshape (`_stage_body_reshape`)
   Fix: either migrate these 4 stages into the registry, or gate `_use_global_registry = False` when these features are requested. Mechanical: the fix shape is already known, just needs careful wiring + test coverage — Haiku-appropriate.

2. **[HAIKU] Related — spot heal is also split-brain (HON-4):** `retouch/spot_heal.py` (`SpotHealer`/`LamaHealer` — the more capable Tier 2.1 v1 LaMa path) is never imported/called by `engine.py`. Only the older, simpler `heal.heal_region` (Telea-only, v0) is wired at `engine.py:1439`. Pairs naturally with item 1's fix (same "code exists, unreachable" bug class — audit calls this **B3**). Wiring-only, not judgment-heavy.

3. **[HAIKU] Stale research doc — hygiene, not code:** `RESEARCH_FUJI_SIM_QUALITY_CEILING.md` (written 2026-07-14) is already resolved as of the *same day* — `FilmDensityEngine` brightness-crush fixed in `9699157`, selective-color/HSL calibration blocks added in `f3290c3`. Archive or add a status note so it doesn't cause redundant re-work. Only its item 4 (reference corpus acquisition) is still genuinely open.

---

## P1 — Next up (face-path items first, per standing priority) — [HAIKU unless noted]

4. **[HAIKU] B6: ROI-crop `FaceRegions` before IPC pickling** — currently ~335MB/face pickled; face-pipeline perf/memory issue, not a general one. Mechanical scoping fix.
5. **[HAIKU] Phase 7 ship/distribution hardening** — signing, notarization, Windows build, update-check. Confirmed **not started**, no code found. Not face-specific; kept here since it's still P1-severity for shippability. Mostly config/tooling, not judgment-heavy.
6. **[HAIKU] Remaining audit Tier B items** (from `PLAN_CODEBASE_AUDIT_2026_07_14.md`), pair naturally with the P0 fix but are body/background-scoped, not face:
   - B3: wire local-adjustments + spot-heal brush UI end-to-end (see item 1-2 above)
   - B4: LaMa tile-seam fix (`spot_heal.py:311` — dead `weight` blend variable, hard seams at 1024px)
   - B5: per-model ONNX provider denylist (CoreML known-broken for NAFNet, not currently blocked)

*(Note: F1 float32 completion and B7 style_ref multi-face were originally P1 items here — both moved to the FABLE hardest-tier section above; they're judgment-heavy, not mechanical.)*

---

## P2 — Queued (face-path items first) — [HAIKU unless noted]

7. **[HAIKU, partial FABLE] Visual QA backlog — face items first:** F5 liquify (face reshape), face_exposure, per-region wrinkle sliders, reshape completeness, R9–R13 chromophore/specular/texture work (all face skin) — these are code-DONE per MASTER_PLAN but not yet human-eyeball-reviewed, and they're all face-quality-critical. Non-face visual QA (showcase recipe family, T5 RAF highlight recovery) queue after. Reviewing renders is Haiku-doable; any fixes found from R9-R13 chromophore/specular work may need Fable if they turn out to be calibration-level, not cosmetic.
8. **[HAIKU] Body Skin / Clothes as separately maskable GUI targets** — from LR competitor research (§7 of `PHOTOSHOP_PARITY_ROADMAP.md`); "Body Skin" touches face-adjacent retouching, "Clothes" doesn't — split if scoped. GUI/param wiring, mechanical.
9. **[HAIKU] 2.3 Interactive curve editor** — confirmed NOT STARTED, deliberately deferred (needs custom JS component/canvas, not a Gradio built-in). F6 (look-extraction into editable preset JSON) already shipped as a partial substitute. General-purpose, not face-specific. Frontend widget work — mechanical once scoped, but flag to Fable first if the canvas/JS component design needs real UX judgment.
10. **[HAIKU] 2.4 Super-resolution model** — denoise (F7/NAFNet) is done; SR model deliberately not yet acquired. Benefits faces same as everything else, not face-specific work. Model acquisition + wiring, mechanical.
11. **[HAIKU] Lens Blur comparison** — LR has bokeh-shape presets, Cat Eye, Bokeh Boost, focus-by-point + "Visualize Depth" toggle. Compare against our relight/lens-effects stage; note `f3290c3` already added *some* lens blur — check current coverage before scoping. Background/DoF-scoped, not face.
12. **[HAIKU] Linear RAW develop full UX** — 16-bit ingest + `LinearGrader` exist, reachable only via CLI `--linear-raw`; GUI exposure/WB sliders for the develop step are genuine backlog (T5 Step 3). Whole-image scope, not face-specific. GUI wiring, mechanical.
13. **[HAIKU] 10 `reset_*` handlers in gui.py** — still hand-ordered positional pairs (latent, low-risk, unchanged status). Purely mechanical.

---

## Parked — do not resurface without owner request

- A1/A2/A4 competitor benchmarking track (Retouch4me tuning, neural stray-hair) — owner-declined paid trial licensing.
- Per-face auto Male/Female/Child classification Slice 3 — heuristic, low priority.
- LUT hot-reload daemon — **not actually a bug**; a manual "Reload LUTs" button already works. Only the filesystem-watcher automation is deferred as a nice-to-have.
- P6 preference learning / P5 temporal-burst coherence / P8 aging — explicitly blocked on dependencies per `RESEARCH_PER_FACE_AND_P4.md`.

---

## Notes for whoever reviews this before Fable gets it

- MASTER_PLAN is the declared authoritative doc, but items 1-2 above are proof its "✅ DONE" status can mean "code exists" rather than "reachable by a user." Worth deciding whether to correct MASTER_PLAN's rows directly as part of the P0 fix, so this discrepancy doesn't recur.
- Priority chain of supersession confirmed: `PHOTOSHOP_PARITY_ROADMAP.md` → Tier 1/2/3 plans → `PLAN_TIERP_PERF_ARCH_SHIP.md` (P1 guided-filter + F8 full-res before F1 float32) → `MASTER_PLAN.md` (current authority, dated 2026-07-14).

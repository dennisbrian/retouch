# Task list for Fable — draft (2026-07-15, ~6pm handoff)

**Status:** DRAFT — not yet assigned to Fable. Prepared for review before handoff.
**P0 #1 verified independently (2026-07-15), not just subagent-reported:** direct greps confirm `_stage_background`/`_stage_cosplay_moat`/`_stage_local_adjustments`/`_stage_body_reshape` are called only at `engine.py:2145/2159/2186/2200`, all inside the `else` branch of `if use_registry:` (line 2105). `_use_global_registry` has exactly one reference in the whole codebase (the `getattr(..., True)` default at line 2104) — no code anywhere ever sets it `False`. `stage_wrappers.py`'s registry defines exactly 6 stage classes (`SubjectSeparation`, `BackgroundHarmonize`, `BodySkin`, `Global`, `Grade`, `Finish`) and none reference the 4 dead methods. The block's own comment (`engine.py:2099-2102`) confirms intent: it was meant to be a golden-harness-verified A/B toggle, flipped to registry-default before the 4 features were ported in, and never fixed back.
**Source:** Reconciled from `PHOTOSHOP_PARITY_ROADMAP.md`, `MASTER_PLAN.md`, `PLAN_CODEBASE_AUDIT_2026_07_14.md`,
Tier 1/2/3 plans, `PLAN_TIERP_PERF_ARCH_SHIP.md`, `PLAN_FEATURE_FRONTIER.md`, `RESEARCH_FUJI_SIM_QUALITY_CEILING.md`,
`RESEARCH_PER_FACE_AND_P4.md`, and an Explore-agent code verification pass against HEAD (`3ac86a1`).

Priority key: **P0** = fix now, blocks other claims of "done" · **P1** = next up · **P2** = queued · **P3** = greenfield/optional · **Parked** = intentionally deferred, don't resurface.

---

## P0 — Architecture bug (highest leverage, fix first)

1. **Dead stage-registry `else` branch** (`retouch/engine.py:2104`, `_use_global_registry` defaults `True`, never set `False` anywhere).
   The registry (`stage_wrappers.py:build_global_registry()`) only wires 6 stages. Four features that MASTER_PLAN marks "✅ DONE" are actually **unreachable in production**, only exercised in tests that call the private stage method directly, bypassing `engine.process()`:
   - T1 background replace/relight (`_stage_background`) — includes the `anime_crystal_void` 7-key wiring, which is **still effectively dead**, not just "7 dead keys" as previously assumed.
   - A3 cosplay skin moat (`_stage_cosplay_moat`)
   - **F3 brush-painted local masks** (`_stage_local_adjustments`) — this is Tier 1.3 from the roadmap.
   - T3 body reshape (`_stage_body_reshape`)
   Fix: either migrate these 4 stages into the registry, or gate `_use_global_registry = False` when these features are requested.

2. **Related — spot heal is also split-brain (HON-4):** `retouch/spot_heal.py` (`SpotHealer`/`LamaHealer` — the more capable Tier 2.1 v1 LaMa path) is never imported/called by `engine.py`. Only the older, simpler `heal.heal_region` (Telea-only, v0) is wired at `engine.py:1439`. Pairs naturally with item 1's fix (same "code exists, unreachable" bug class — audit calls this **B3**).

3. **Stale research doc — hygiene, not code:** `RESEARCH_FUJI_SIM_QUALITY_CEILING.md` (written 2026-07-14) is already resolved as of the *same day* — `FilmDensityEngine` brightness-crush fixed in `9699157`, selective-color/HSL calibration blocks added in `f3290c3`. Archive or add a status note so it doesn't cause redundant re-work. Only its item 4 (reference corpus acquisition) is still genuinely open.

---

## P1 — Next up

4. **Phase 7 ship/distribution hardening** — signing, notarization, Windows build, update-check. Confirmed **not started**, no code found.
5. **Audit Tier B items** (from `PLAN_CODEBASE_AUDIT_2026_07_14.md`), several pair naturally with the P0 fix:
   - B3: wire local-adjustments + spot-heal brush UI end-to-end (see item 1-2 above)
   - B4: LaMa tile-seam fix (`spot_heal.py:311` — dead `weight` blend variable, hard seams at 1024px)
   - B5: per-model ONNX provider denylist (CoreML known-broken for NAFNet, not currently blocked)
   - B6: ROI-crop `FaceRegions` before IPC pickling (currently ~335MB/face pickled)
   - B7: `style_ref` multi-face + region coverage (only transfers to largest face in group shots; skips clothing/eyes/lips/teeth)
6. **F1 float32 pipeline completion** — MASTER_PLAN says done; `PLAN_TIERE_ENGINE_FIDELITY.md`'s own audit says ~30% real. 16 uint8 round-trip sites remain in `_stage_grade` + 3 float helpers that internally round-trip. Real, code-verified gap.

---

## P2 — Queued

7. **Visual QA backlog** — items MASTER_PLAN marks code-DONE but not yet human-eyeball-reviewed: F5 liquify, showcase recipe family, face_exposure, per-region wrinkle sliders, reshape completeness, R9–R13 chromophore/specular/texture work, T5 RAF highlight recovery.
8. **2.3 Interactive curve editor** — confirmed NOT STARTED, deliberately deferred (needs custom JS component/canvas, not a Gradio built-in). F6 (look-extraction into editable preset JSON) already shipped as a partial substitute.
9. **2.4 Super-resolution model** — denoise (F7/NAFNet) is done; SR model deliberately not yet acquired.
10. **Body Skin / Clothes as separately maskable GUI targets** — from LR competitor research (§7 of `PHOTOSHOP_PARITY_ROADMAP.md`); real small gap, UI/UX only, no missing segmentation capability.
11. **Lens Blur comparison** — LR has bokeh-shape presets, Cat Eye, Bokeh Boost, focus-by-point + "Visualize Depth" toggle. Compare against our relight/lens-effects stage; note `f3290c3` already added *some* lens blur — check current coverage before scoping.
12. **Linear RAW develop full UX** — 16-bit ingest + `LinearGrader` exist, reachable only via CLI `--linear-raw`; GUI exposure/WB sliders for the develop step are genuine backlog (T5 Step 3).
13. **10 `reset_*` handlers in gui.py** — still hand-ordered positional pairs (latent, low-risk, unchanged status).

---

## P3 — Greenfield / research (pick at most one to scope, not urgent)

14. **`PLAN_FEATURE_FRONTIER.md`** (eyes/teeth/lips optical models) — 100% NOT STARTED, ranked by the doc itself:
    - E-COMMON-1: shared inferred light-direction model (doc recommends building this first — unlocks 4 other items)
    - E-TEETH-1+2: VITA-shade-aware whitening + gum/tooth/lip-interior separation ("kills the #1 amateur tell")
    - E-EYE-4: under-eye as a hemoglobin/chromophore problem
    - E-EYE-1: catchlight synthesis (currently only amplifies existing catchlights)
    - E-EYE-2: scleral shading model
    - E-LIP-1/2: dichromatic lip finish + volume-via-shading
    - Gaze-symmetry correction (explicitly off-by-default, creative/advanced)
15. **Audit Tier C items** (research-only, need data/calibration, not code-now by design):
    - C1: Planckian-locus white balance in linear RGB (current WB uses hardcoded fudge factors in gamma-encoded LAB)
    - C3: soft BiSeNet logits / guided-feather masks (current hard argmax discards probabilities)
    - C4: skin-chromophore invertibility / makeup unmix — check `retouch/makeup_unmix.py`, already exists ahead of the "spike" scoping in `RESEARCH_PER_FACE_AND_P4.md`; look at `git log` on that file before assuming un-started
    - C6: diffeomorphic body reshape constraint — moot until item 1's fix makes T3 body reshape reachable at all
    - C7: real-model calibration/benchmark harness — current `benchmark.py` mocks the engine boundary entirely; **published perf numbers (702ms/face etc.) exclude real inference cost, treat as lower bounds not real measurements**

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

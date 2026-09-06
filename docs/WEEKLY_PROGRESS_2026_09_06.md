# Weekly progress update — 2026-09-06

**Scope:** What's shipped/verified vs. what still needs research, across two
lineages: (1) the Fable-authored changelog (2026-07-03 → 07-20, compiled in
`FABLE_CHANGELOG_2026_09_06.md`), and (2) the more recent face-op audit work
(2026-08-26 → 09-05, per `MEMORY.md`/commit history). This doc is a status
snapshot, not a research report — see the linked docs for detail/evidence.

---

## 1. Shipped this cycle (verified, not just claimed)

### Face-op correctness (2026-08-26 → 09-05)
- **Eye handedness fix (P0/P1)** — BiSeNet vs. parsing.py convention were
  opposite; sclera/catchlight ops were silently painting the wrong eye.
- **Eye-visibility gate** shipped, then **recalibrated** (`_MIN_EAR=0.285`,
  `_MIN_CONTRAST=0.55`) after the original thresholds missed 84% of closed
  eyes on the 83-image DSCF corpus.
- **Composite-mask epsilon bug (P0)** — `max()>1.0` mis-detected float masks
  overshooting 1.0 by one ulp as already-0–255, dividing by 255 and
  silently discarding skin/hair/neck ops on ~24% of real portraits. Fixed
  at 21 sites (`f69ab1e`). **Any pre-fix "op X does nothing" conclusion is
  now suspect** — see `mask-epsilon-normaliser-bug` memory.
- **Yaw-gate recalibration + inversion fix** — bands remapped to actual
  head-turn angles; the shared ramp helper was later found returning the
  smoothstep inverted (near-zero strength past the start of the ramp,
  full strength near the end) — fixed same day.
- **Neck depth gate removed** — was a guaranteed no-op on all 15/15 frontal
  corpus faces (XY-only plane measurement made the geometry vacuous).
- **Dark-circle op v1→v2** — v1 was inert on every face (lifted the lash
  line, not the shadow, 0.03 L median improvement on skin); v2 rebuilt
  around a tear-trough support and cheek-ring-relative darkness measure.
- **Freckle tie-break fix (710cb5b)** — the one production change to ship
  from the FA-03 mark-classification research: exact/near ties in
  `_classify_anomaly` now resolve to `ambiguous` → `unknown` instead of
  silently picking `beauty_mark` by dict order (fixes eyeliner-as-mole on
  DSCF2310). 4,777/4,777 suite passed.

### Fable-lineage (2026-07-03 → 07-20 — see `FABLE_CHANGELOG_2026_09_06.md` for full detail)
- F8.1/F8.2 full-res fidelity, F11 self-QA detectors, GUI wiring
  de-footgun (`_process_inputs`/`_recipe_outputs`) — closed two real
  silent-corruption bugs along the way.
- R7 melanin/hemoglobin decomposition, R12 finish slider (wired), R15 QA
  extensions, R9 intrinsic decomposition (core).
- **Tone-invariance bug-class hunt — all 4 instances closed**:
  `makeup_unmix.py`, `specular.py`, `lips.py`, `skin.py::whiten()`.
- C1/C2 skin-color + structural light-shadow, S1–S6 body/shine/finish/
  hair/wrinkle/texture ops — all shipped, nearly every one had a real bug
  caught during Fable's review pass (silent corruption, inverted logic,
  10-20× weak effect, non-functional detector).
- AD BiSeNet hand-bleed fix, AE roll-unaware warp fix, Day 2 lighting
  coherence.

---

## 2. Needs more research (open, not just "not yet scheduled")

Ranked by how unresolved the underlying problem is, not by priority:

1. **P4 makeup-unmix — the ill-posed inverse problem itself.** Only the
   tone-invariance *bug* was fixed. The core solve
   (`I = (1-α)S + αM`, 3 equations, more unknowns) has no closed-form
   answer — still classical-heuristic, not a real joint solve. Ranked
   hardest-of-all in `FABLE_TASK_LIST_2026_07_15.md`.
2. **FA-03 mark classification — geometry/semantic reranker.** Research
   (5-arm experiment) shows geometry+eye-context features correctly
   distinguish eyeliner from moles, but thresholds are hand-tuned on 3
   real faces (not the report's own 30-portrait/192-synthetic floor), zero
   confirmed natural moles exist in any corpus image, and `ambiguous` has
   no home in the production taxonomy yet. **Not production-ready.**
3. **F1 float32 pipeline completion — status unclear, needs re-audit.**
   `FABLE_TASK_LIST`'s own snapshot says ~30% done (16 uint8 round-trips
   remain in `_stage_grade`); `EXECUTION_LOG`'s wave 4/5 entries suggest
   more was closed than that. The two docs are out of sync — resolve this
   before trusting either number.
4. **Eyes/teeth/lips optical models** — 100% greenfield: shared
   light-direction model, VITA-shade whitening, catchlight synthesis,
   scleral shading. Not started.
5. **Tier C research** — soft BiSeNet logits (vs. hard-argmax), Planckian-
   locus white balance, real-model calibration harness. Not started.
6. **B7 multi-face style_ref** — only transfers to the largest face in
   group shots; needs a coverage/priority policy, not a mechanical fix.
7. **R10/R11/R13/R14 unwired backlog** — chromophore-suite functions
   (bruise removal, vein attenuation, tan-line evening, cosplay skin moat)
   exist but aren't wired into `skin.py`; several R13/R14 sub-slices
   (multi-band pyramid, self-donor synthesis, per-subject profile,
   occlusion guard) never built.
8. **Eye-v0 uint8 ROI roundtrip** — flagged open since the 2026-08-31
   audit closeout, still not resolved.
9. **Near-eye genuine-mark safety (FA-03)** — established only on
   synthetic marks; the report's own corpus has zero confirmed real moles
   near an eye/brow, so this safety claim cannot be made on real photos
   yet.
10. **Dark-circle v2 makeup-vs-shadow ambiguity** — aegyo-sal (under-eye
    contour makeup) reads as shadow at strength ≥0.45; no darker-skin
    sample exists in corpus to check the same failure mode there.

---

## 3. Stalled (not a research gap — blocked on external input)

- **Cosplay S2 three-band body parity + D_TPR auto-tune** — blocked on
  owner asset acquisition (Fitzpatrick V–VI + exposed-body-skin
  portraits).
- **RAF-decoder rewrite + recipe.py "correction-first" rewrite** —
  uncommitted; blocked on a missing vetted `.raf` test asset. Also
  silently narrows `--recipe` to a curated allowlist, dropping ~78
  previously-valid names — needs explicit sign-off before shipping, not
  more research.
- **AC1 EXIF/ICC regression fix** — fixed but was still uncommitted as of
  the 07-20 reconcile; verify current status before assuming it shipped.

---

## 4. Cross-cutting caveats worth remembering

- **"Done" has repeatedly meant "tests-only, not wired"** in this repo's
  history (origin: the 2026-07-10 wiring-debt audit). Verify GUI/recipe/
  caller reachability before trusting a ✅, not just test-suite green.
- **The mask-epsilon bug (fixed 2026-09-02) invalidates any pre-fix claim**
  that a given skin op "does nothing" on ~24% of portraits — re-verify
  before citing old audit conclusions.
- Golden face-hash snapshots are **interpreter-specific** (bare `python3`
  cv2 4.13 vs. `.venv/bin/python` cv2 4.11) — always test with
  `.venv/bin/python` + `RETOUCH_MEDIAPIPE_BACKEND=legacy`.

---

**Sources:** `docs/FABLE_CHANGELOG_2026_09_06.md`, `MASTER_PLAN.md`,
`EXECUTION_LOG.md`, `FABLE_TASK_LIST_2026_07_15.md`,
`docs/plans/RESEARCH_FA03_CLASSIFICATION_EXPERIMENT_2026_09_05.md`,
`docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md`, and this
session's `MEMORY.md` entries (2026-08-26 → 09-05 CLAUDE.md "Outstanding
Fixes" log).

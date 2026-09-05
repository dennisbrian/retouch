# Research — FA-01 protection-consistency and abstention-observability audit

**Date:** 2026-09-05

**Status:** Audit complete; no code changes in this task. Continues
[FA-01](PLAN_FACE_RETOUCH_ALGORITHM_RESEARCH_EXECUTION_2026_09_05.md) after the
boundary-confidence field (`6b1e72d`) and mark-policy wiring for blemish +
9 evening ops (`f3b1008`, `154d854`).

**Scope:** FA-01's two remaining bullets:

1. Are hair, makeup, accessories, lashes, facial hair, and stable marks
   protected consistently across operations?
2. Do unknown or insufficiently supported regions produce a safe abstention?

Not in scope: the SegFace comparison and the plan's ontology/corpus/
decision-record documents (labels, evidence-field spec, corpus protocol,
provenance register). Neither was produced by this task; FA-01 is not
closed by this note.

**Evidence levels** follow the parent report's convention: current-code
(read directly), and real-render (an actual `RetouchEngine.process()` call
compared pixel-for-pixel). No new prior-project or published-result claims
are made here.

---

## 1. Protection consistency across operations

### 1.1 Headline finding: base smoothing is unprotected against marks

**Real-render evidence.** On DSCF2310 (meitu bake-off corpus, real BiSeNet
parse), `RetouchEngine.process(..., smooth=90, equalize=0, blemish=0,
mark_policy='preserve_all')` produced a **byte-identical** output to the
same call with `mark_policy=None`. Max diff 0, 0 nonzero pixels.

**Current code.** `frequency.combine` (the frequency-separation smoothing
stage, gated on `ctx.smooth`) always receives `smooth_mask`
(`perf_optimizations.py:469-493`), which is built entirely independently
of the `skin_n_marks_protected` mask introduced in `154d854`. Smoothing
runs before all 9 wired evening ops and before `blemish.remove`.

**Scope of the gap — precise, not the five bullet-1 materials.**
`smooth_mask` is derived from `skin_n`, which Tier 1 (§1.2 below) already
strips of eyes/brows/lips. So this gap is specifically: **smoothing has no
awareness of detected marks (moles, freckles, stable cosmetic marks)** — a
mole can be smoothed away (or its edge softened) before mark detection
even runs against fresh pixels, regardless of `mark_policy`. It is not a
gap in hair/makeup/accessory/lash protection; those are covered under §1.2.

**Why this isn't a simple "wire it like the other 9" fix.** The 9-op
split in `154d854` used one rule: does the op re-render the *surrounding*
pixels (shading/tone op → excluding a mark creates an island artifact) or
only *evenly reduce variance within* the mask (evening op → exclusion is
safe)? Frequency-separation smoothing re-renders surrounding pixels too —
by that rule it looks like a shading op. But its *intent* is variance
reduction (the textbook target of mark protection), which argues for
treating it like the 9. The classification rule shipped in `154d854` does
not resolve this; smoothing is a third case the rule wasn't built to
handle. Whether excluding a mark from smoothing produces a visible island
(as feathering plus a per-op re-render is more likely to, since
frequency separation operates at multiple spatial scales rather than one
statistical pull) is an open, testable question — not yet tested. This is
a recommendation for the next authorization decision, not something this
task decided to implement.

### 1.2 Protection mechanism inventory (current code)

| Tier | Mechanism | Materials covered | Consistency |
|---|---|---|---|
| 1 | Explicit subtraction from `regions.skin` in `_masks_from_label_map` (`parsing.py:240-243`, post-feather) | left/right eyebrow, left/right eye, lips, mouth_interior | Every op that reads `regions.skin` or a mask derived from it inherits this. Verified no skin-editing op reads `regions.face_oval` (which re-includes eyes/brows/lips/nose) as a skin proxy instead — its one consumer, `hair.enhance()`, uses it for hair-region geometry, not as an edit-eligibility mask. |
| 2 | Separate BiSeNet class, excluded from skin by argmax exclusivity | `hair` (17), `neck` (14), `cloth` (16) | Implicit: a pixel labeled 17 can't also be labeled 1 (skin). `acc_skin_hair` (skin ∪ hair ∪ neck, `perf_optimizations.py:325-338`) is written into the returned `_FaceResult` for background/matting use and never read back into any skin-editing op inside `_process_face_core` — confirmed by grep, not just the module comment. |
| 3 | Class exists in the 19-label space but no dedicated mask is extracted | Labels 0, 6, 7, 8, 9, 15, 18 (9 of 19 CelebAMask-HQ classes) | `_masks_from_label_map` extracts only 10 of 19 labels (1,2,3,4,5,10\*,11,12,13,14,16,17 — 10 counted, `10` only inside the `face_oval` union, never as its own mask). The other labels get no dedicated protection: they're excluded from `skin` only by argmax exclusivity (whatever a pixel is labeled, it isn't labeled `skin`), with no independent mask, no test, and no fallback if BiSeNet mislabels e.g. a thin accessory frame as skin. This repo's own known-limitations note (CLAUDE.md) states the corpus has zero glasses/wig cases, so this row is a code-reading conclusion, not a measured one — it cannot be rendered against a real accessory photo here. |
| 4 | No class exists at all | Facial hair (beard/mustache) | CelebAMask-HQ's 19-class taxonomy has no facial-hair label. `beard_shadow_neutralize` (`skin_chromophore.py:410`) is the only facial-hair-adjacent function in the codebase, and it does the opposite of protection: it *treats* beard-shadow pixels as skin to neutralize a color cast, using whatever mask the caller supplies (presumably `regions.skin`). There is no cheap fix here — it needs either a model class BiSeNet doesn't have, or a dedicated detector; out of scope for a diagnostic tranche, consistent with the plan's deferral of parser changes. |
| — | Lashes | Covered, not a gap | The only "lash" concept anywhere in the codebase is `regions.left_eye`/`right_eye` (lash+lid together, via BiSeNet or landmarks) plus a dedicated landmark lash-margin exclusion inside `undereye.py` v2 for the one op (darken/puffiness) that works right at that boundary. Every skin-masked op already excludes `left_eye`/`right_eye` via Tier 1, so lashes are protected wherever skin protection reaches — this is a "covered" row, not a finding. |

### 1.3 Summary for bullet 1

Protection is consistent for the five materials named in the bullet
**among ops that read `regions.skin`**: eyes/brows/lips (Tier 1) and hair
(Tier 2) are reliably excluded everywhere a skin mask is used, and lashes
ride along with the eye exclusion. Accessories (Tier 3) and facial hair
(Tier 4) have no dedicated protection anywhere in the pipeline — this is a
known, pre-existing gap (the parent report's §4.6/§7.1 already names
accessories as a SegFace-motivated future comparison) rather than a new
finding, and it cannot be measured against a real photo here (zero
accessory/facial-hair cases in the corpus per CLAUDE.md). The one new,
measured finding is §1.1: stable marks specifically are unprotected
against the base smoothing stage, which runs before every op that does
protect them.

---

## 2. Abstention observability

Five mechanisms back off or skip when evidence is missing or unreliable.
Ranked by how observable the decision is, most to least:

| Mechanism | Backs off when | Observable how |
|---|---|---|
| `safe_auto` (`decide`/`decide_mask_stage`, `perf_optimizations.py:1230`) | Confidence evidence falls below apply/dampen/review thresholds | Structured: appended to `_FaceResult.safe_auto_decisions`, collected into `ProcessingResult` via `_collect_safe_auto_decisions` (`engine.py:3117`). Available to any caller that inspects the result object; not logged. |
| `eye_visibility_gate` (`eye_visibility.py:401`) | EAR/contrast indicate a closed or low-contrast eye | Logged: `logger.info` with the gated side(s) and the EAR/contrast numbers that triggered it, only when something is actually gated. |
| `_landmark_fallback_only` (`parsing.py`) | BiSeNet unavailable or returns empty skin | Explicit sentinel: `parse_confidence = {}` / `parse_boundary_confidence = {}` (distinct from `None`, meaning "checked, nothing available" per its own comment) — the good pattern, already praised in this codebase's own conventions. |
| `assess_eye_artifact_scales` (`eye_artifact_safety.py:94`) | Iris mask absent or under 5 support pixels | Reason computed but discarded: returns `_neutral_evidence("iris_mask_unavailable")` / `_neutral_evidence("iris_support_unavailable")`, a named-reason dict — but nothing between this function and its call site (`perf_optimizations.py:303`) or its two consumers (`resolve_eye_scale`, lines 1150/1156) logs it. The scale factor is used; the reason string is dropped on the floor. |
| `utils.yaw_gate_factor` (`utils.py:199`) | Face ratio (yaw proxy) passes `YAW_GATE_START`, ramping strength to 0 by `YAW_GATE_END` | Nothing. No log line anywhere near the function or its call sites. |

**The discriminating finding.** This is a plumbing gap, not a
missing-mechanism gap — every mechanism already computes the evidence
needed to explain a decision (`assess_eye_artifact_scales` builds a named
reason string and then throws it away). The concrete cost case already in
this repo's history: the yaw-ramp inversion (`ce56ba2`, corrected
2026-09-02) shipped and went undetected because nothing logged what
strength `yaw_gate_factor` was actually returning per face — only a
midpoint-only test suite and, eventually, a corpus re-audit caught it.
`assess_eye_artifact_scales` and `yaw_gate_factor` are the two mechanisms
positioned to fail the same way today.

A plausible next diagnostic tranche (not authorized or implemented here):
a single per-face debug log line reporting `yaw_gate_factor`'s ratio and
resulting scale, and `assess_eye_artifact_scales`'s per-side reason string
— mirroring the eye-visibility gate's existing pattern and the
boundary-confidence field's observation-only contract (`6b1e72d`).

---

## 3. What remains open in FA-01

- The ontology/corpus/evidence-field/decision-record/provenance documents
  the execution plan's exit criteria call for were not produced by this
  task or the two prior implementation tranches.
- The SegFace-vs-BiSeNet boundary-error comparison (plan §1, last bullet)
  is untouched — it depends on the corpus/metrics work above.
- Accessory and facial-hair protection (§1.2, Tiers 3/4) are confirmed
  gaps but unmeasured against a real photo; the corpus has zero such
  cases per CLAUDE.md's known-limitations note.
- Whether protecting marks from the base smoothing stage is safe (vs.
  producing a new artifact class the way a naive shading-op exclusion
  would) is an open, testable question this audit surfaces but does not
  answer.
- Abstention observability for `assess_eye_artifact_scales` and
  `yaw_gate_factor` is a named, scoped-but-unimplemented diagnostic
  candidate (§2).

No engine, test, recipe, model, or image changes were made by this task.

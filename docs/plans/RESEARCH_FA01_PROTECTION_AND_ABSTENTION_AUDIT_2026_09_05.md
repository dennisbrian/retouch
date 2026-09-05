# Research — FA-01 protection-consistency and abstention-observability audit

**Date:** 2026-09-05

**Status:** Audit complete. Two follow-ups landed after the original
commit (`5bac5ac`): abstention logging (`8a2869e`, §2) and a smoothing-
protection mechanism experiment (§1.1a) that answers this document's own
open question with a recommendation but no production change. Continues
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

### 1.1a Follow-up experiment: which smoothing-protection mechanism, if any

The open question from §1.1 — whether protecting marks from smoothing is
safe, and by what mechanism — was investigated with a dedicated experiment:
`scripts/qa/smoothing_mark_protection_experiment.py`. No production code
was changed; every mechanism below is implemented locally in the script
against `retouch.frequency`'s internals (its own copy of `separate()`, its
own guided-filter variants), never against the shipped
`FrequencySeparator.combine`.

**Five candidates reduced to three mechanisms, by source-reading.** The
task named five candidates to compare: no protection, output-mask
exclusion, feathered/soft attenuation, exclusion from filter statistics,
and post-smoothing restoration. Reading `frequency.combine` shows three of
these are the same mechanism:

- `combine()`'s `skin_mask` argument only ever controls the **final blend
  alpha** — `blend_masked(orig_crop, processed_crop, m_2d)` at the end of
  the function. It is never passed into the smoothing filter itself.
- Consequently, "hard output-mask exclusion," "feathered/soft attenuation"
  (same exclusion, wider Gaussian on the mask before blending), and
  "post-smoothing restoration of source detail" (arithmetically: blending
  the *original* crop back in at the mark, after the fact) are one
  mechanism at different feather widths / timing, not three. This is
  reported as a finding, not a shortcut.
- The one mechanistically distinct candidate is **excluding protected
  pixels from the filter's own reference statistics**: `_guided_smooth` →
  `utils.guided_filter`'s self-guided branch computes `mean_I =
  cv2.blur(src)` — a plain box filter over the *entire* crop with no mask
  parameter at all. A mark's pixel values leak into neighboring pixels'
  local mean/variance during this box-filter pass regardless of what the
  caller excludes from the final blend. This confirms criterion 4's
  premise (contamination is real) but also means the first two candidates
  cannot address it by construction — only a filter-input change can.

The three arms actually compared:

| Arm | Mechanism |
|---|---|
| `baseline` | Current shipped behavior — no exclusion anywhere (the `mark_policy=None`/no-protection case). |
| `blend_alpha` | Marks excluded from the final blend only, at a swept feather width (0/9/25px) — collapses candidates 2/3/5. |
| `filter_input` | Marks additionally excluded from the guided filter's own local statistics via normalized convolution (mask-weighted moments — the same technique `retouch/undereye.py`'s analyzer already uses, not a new primitive) — candidate 4, layered on top of the same final-blend exclusion. |

**Scenes.** Two, both with pore-scale texture strong enough to keep
`_texture_adaptation_factor`'s `adapt` at 1.0 (a first attempt with lighter
noise silently floored `adapt` to 0.55, meaning the requested "0.9" and
"0.5" smoothing strengths were actually running at ~0.5 and ~0.28 — a
scene-construction error caught and fixed before the numbers below).

1. **Synthetic**: flat skin-tone canvas, band-limited noise texture, two
   marks sized on either side of `k_mid` (`face_width=400` → `k_mid=17px`,
   `k_low=49px`): a 4px-radius mark (lives mostly in the `high` band) and a
   15px-radius mark (lives mostly in `low`/`mid`).
2. **Real face, semi-synthetic mark**: a real corpus photo crop (DSCF2306,
   genuine skin texture/lighting/sensor noise) with a synthetic 6px-radius
   mark composited onto clean cheek skin at a **hand-picked coordinate**,
   away from makeup/eyeliner. `detect_marks` is never called in this arm —
   ground truth is the chosen coordinate, not a detector output. This
   avoids the eyeliner-misclassification confound already documented for
   this corpus family (§1 above) but means the arm tests real skin
   *statistics* (texture, lighting, noise), not a real mole's morphology —
   the available corpus has no clearly-visible, unambiguous natural mole to
   hand-label instead.

Both scenes use `mid_reduction=0.35` (the registered `params.py` default,
not a guess) and `smooth_engine="guided"` only — `bilateral` and
`anisotropic` are untested and are a precondition on any future
implementation, not covered by this result.

**Results, smooth_strength=0.9 (high-strength criterion), mark contrast
= surrounding-ring level minus mark level, source-relative):**

| Scene / mark | Source (ceiling) | Baseline | `blend_alpha` (f=9) | `filter_input` (f=9) |
|---|---|---|---|---|
| Synthetic, small (r=4, high-band) | 83.2 | 37.1 (55% lost) | 69.1 | 69.2 |
| Synthetic, large (r=15, low/mid-band) | 81.7 | 8.5 (90% lost) | 74.5 | 78.3 |
| Real face, mark (r=6) | 95.9 | 15.8 (84% lost) | 74.9 | 75.3 |

Baseline destroys the large majority of every mark's contrast at high
smoothing strength, on both scenes and both frequency bands. Both
protection mechanisms recover contrast to a similar degree (`filter_input`
0.1–3.9 points higher than `blend_alpha`, depending on scene/size).

**Halo/contamination signature (criterion 2 and empirical answer to
criterion 4).** Measured as the maximum deviation, arm minus baseline, in
a radial profile ring just outside the mark (`outer_delta`; a genuine halo
shows as a non-monotonic bump here, distinct from the expected big jump
right at the mark's own edge):

| Scene / mark | `blend_alpha` outer_delta | `filter_input` outer_delta |
|---|---|---|
| Synthetic, small | 0.04 | 0.26 |
| Synthetic, large | 0.35 | 4.17 |
| Real face | 0.05 | 0.54 |

`blend_alpha` stays near zero in every case. `filter_input` is
consistently higher — by roughly an order of magnitude on the synthetic
large mark, more modestly on the other two — confirming the contamination
`_guided_smooth` reads from an unmasked box filter is real, and that
*removing* it (by excluding the mark from the filter's own statistics)
creates its own visible ring artifact, visible directly in a
diff-against-baseline image (`filter_input` shows a soft grey halo
extending well past the mark; `blend_alpha`'s diff is a clean, contained
disk — see `scripts/qa/smoothing_mark_protection_out/_diff_vs_baseline_*.png`).

**Feather-width rule (a parameter finding, not just an observation).**
Widening the final-blend feather past roughly the mark's own radius
degrades contrast recovery for `blend_alpha` — because the Gaussian
feather kernel starts re-including the mark's own pixels into the blend,
not because of any interaction with the filter:

| Mark radius | Feather 9px | Feather 25px |
|---|---|---|
| 4px (synthetic small) | 69.1 | 57.3 (−12) |
| 6px (real face) | 74.9 | 65.4 (−9.5) |
| 15px (synthetic large) | 74.5 | 74.9 (≈0) |

Degradation appears exactly when feather width exceeds the mark's own
radius, and is absent when it doesn't (15px mark, 25px feather). The rule
for any future implementation is **feather width bounded by mark radius**,
not a fixed pixel constant.

**Neighboring skin continuity (criterion 3).** The `neighbor_continuity`
metric (std-dev in a ring well outside the mark, `r ∈
[radius+15,radius+25]`) was flat to six decimal places across every arm on
the synthetic large mark (5.686884 baseline = 5.686884 `blend_alpha`),
but this reflects the probe sitting outside where any halo actually lives
— it is not itself evidence against a halo. The `outer_delta` metric above
(sampled from `radius+6` outward) is the instrument that actually detected
`filter_input`'s contamination signature; read criterion 3 from that
table, not from `neighbor_continuity`.

**Zero/legacy behavior (criterion 6).** The `baseline` arm is the
`mark_policy=None` case by construction (empty protect mask); this
reproduces the byte-identical real-render result already established in
§1.1 rather than re-deriving it synthetically.

**Recommendation.** If mark protection is extended to base smoothing:
**blend-alpha exclusion (final-blend mask only), feather width bounded by
the mark's own radius (0–9px in this experiment's mark-size range), not
`filter_input`.** `filter_input` (excluding marks from the guided filter's
own statistics) buys a small, inconsistent contrast improvement (+0.1 to
+3.9 points) at the cost of a consistently worse, sometimes much worse,
halo signature — a clear rejection on the cost/benefit the task's own
criteria ask for, not a close call. This recommendation is scene-limited
(one mark shape, two sizes, one skin tone, `guided` engine only,
semi-synthetic real-face evidence rather than a genuine natural mole) and
is a recommendation for the next authorization decision — no production
code was changed to implement it.

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
protect them. §1.1a's follow-up experiment answers the open "would
protecting it be safe" question this note originally left unresolved:
blend-alpha exclusion (feather ≤ mark radius) recovers 74–78% of the
mark's source contrast at high smoothing strength with a near-zero halo
signature, on both a controlled synthetic scene and a real-face
composite — implementing it is a plausible next tranche, not decided
here.

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
- Whether protecting marks from the base smoothing stage is safe is now
  answered by experiment (§1.1a): blend-alpha exclusion, feathered ≤ the
  mark's own radius, recovers most of a mark's contrast with a near-zero
  halo signature on both a synthetic and a real-face scene.
  **Implementing this in `frequency.combine`/`perf_optimizations.py`
  remains a separate, unauthorized next step** — this tranche is
  evidence and a recommendation, not the change itself, and the result is
  scene-limited (one mark shape, two sizes, `guided` engine only —
  `bilateral`/`anisotropic` untested).
- Abstention observability for `assess_eye_artifact_scales` and
  `yaw_gate_factor` was implemented in a follow-up tranche (`8a2869e`,
  log-only, golden hashes unchanged) after this document was first
  committed (`5bac5ac`) — no longer open.
- Accessory and facial-hair protection (§1.2, Tiers 3/4) remain open:
  confirmed gaps, unmeasured against a real photo, no cheap fix (needs a
  parser class BiSeNet doesn't have).

No engine, test, recipe, model, or image changes were made by this task's
original audit. The §1.1a follow-up added a new scratch script
(`scripts/qa/smoothing_mark_protection_experiment.py`) that exercises
copies of `retouch.frequency`'s algorithms for comparison purposes only —
it does not import or modify `FrequencySeparator`, and no file under
`retouch/` changed as a result of this experiment.

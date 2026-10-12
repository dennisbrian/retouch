# P8 — Aging-vector control

Date: 2026-09-09. Code baseline inspected: `ee88d3d`.
Scope: literature review, code inspection and proposed experiments only.
No production edits, new models, image processing or test execution.
No renders were produced, so no claim below is a measured result on this corpus.

Scope assumption: P8 means the aging-vector proposal in
[Skin ProMax](PLAN_SKIN_PROMAX.md#p8--aging-vector-control-principled-age-updown-not-wrinkle-soften--novel),
following the P4–P7 research of 2026-09-06..09. `PLAN_SKIN_PROMAX.md` ranks it
8/8 and marks it `blocked | needs M/I/P1`; `RESEARCH_PER_FACE_AND_P4.md` §C
repeats `P8 aging | blocked | needs M/I/P1`. This report tests whether that
"blocked" verdict still holds against the code as it exists today, and what is
researchable with P1 absent.

Requested as "P9". There is no P9: `grep -rn "\bP9\b"` over all `*.md`/`*.py`
and `git log --grep="P9"` both return empty, and the canonical series in
`PLAN_SKIN_PROMAX.md` is P1–P8. P8 is the only undone numbered item. Owner
confirmed P8 was intended.

## 1. Recommendation

**GO for a measurement-only study: build and validate an aging-cue *readout*
before any slider exists. NO-GO for shipping a user-facing "age" slider this
cycle, and NO-GO for the plan's premise that P8 is mostly a recombination of
metrics "we already compute."**

That premise is the finding of this report, and it is wrong as stated. P8's
value proposition is a *coordinated* vector: move six cues together and
proportionally. Coordination requires that all six be **measurable**, because a
vector needs a current position before it can have a direction. Of P8's six
named cues, this repo can measure **three**, can only *apply* a fourth, and has
**no measurement path at all** for the two the plan leans on hardest.

A slider built today would move the three cues it can see and leave the other
three at whatever the recipe happened to set — which reproduces exactly the
failure mode P8 was written to avoid ("smoothed but still old"), while carrying
a control labelled "age" that implies otherwise. The honest sequencing is:
readout first, vector second.

The plan's blocked-on-P1 verdict is **directionally right but mis-attributed**.
P8 is not blocked on the full layered-SSS forward model. It is blocked on two
specific missing *measurements* — translucency and volume/AO — one of which
(translucency) is a genuinely hard inverse problem that P1 would solve, and one
of which (AO) is a much cheaper shape-estimation task that does not need P1 at
all. That distinction is worth ~4–6 weeks of moonshot and is developed in §4.

## 2. Cue-by-cue inventory (code, not plan)

P8 names six cues. Verified against `ee88d3d` by inspection:

| Cue | Measurable today? | Evidence |
|---|---|---|
| Chroma uniformity | ✅ yes | `perceptual_metrics.py::skin_homogeneity_state` |
| Edge definition (vermilion, jaw) | ✅ yes | `perceptual_metrics.py::facial_feature_contrast` |
| Sebum / gloss | ✅ yes | `specular.py::extract_specular` (tone-adaptive, per CLAUDE.md) |
| Wrinkle depth | ⚠️ apply-only | `skin.py::wrinkle_soften` + `_wrinkle_soften_masked` detect ridges via black-hat morphology *inside the operator*; no function returns a wrinkle magnitude |
| Skin translucency | ❌ none | `grep "def .*translucen\|def .*measure_sss\|def .*estimate_sss"` over `retouch/*.py` → empty |
| Volume / AO | ❌ none | no AO estimator; `relight.py::sculpt` *applies* shading, `grep "ambient_occlusion"` → no estimator |

Three confirmations that matter more than the table:

**`skin_sss` is an effect, not a measurement.** Memory records `skin_sss` as
shipped (`5a0f4b6`), and it is: `ParamSpec` at `params.py:700`, implemented at
`skin.py:1771::apply_sss`, used by `game_character_v2` ("subsurface-scatter
skin, warm translucent shadows", `recipe_cookbook.py:131`). But it *renders* a
stylistic translucency look; it does not *estimate* the subject's translucency.
P8 needs the inverse direction. Reading the shipped `skin_sss` as "translucency
is covered" is precisely the
"DONE means tests-only, not wired" trap in a new costume — the
op exists, is wired, has recipes, and still does not supply the quantity P8
needs. (The repo has hit this pattern before: shipped-and-wired is not the same
as supplying-what-the-next-feature-needs.)

**Wrinkle depth is the cheapest gap to close and is not blocked on anything.**
The ridge detection already exists inside `_wrinkle_soften_masked`; it is
computed, consumed, and discarded. Exposing the pre-softening ridge response as
a returned magnitude is a refactor of existing arithmetic, not new science. It
does need care: black-hat response is contaminated by pores, stubble and
eyeliner, so a raw magnitude is a *proxy*, and §5's experiment E2 is about
whether that proxy tracks perceived age at all before anyone trusts it.

**The M/I series is a specified plan, mostly unbuilt.** P8's blocker is written
as `M/I/P1`. M1–M6 and I1–I5 are fully specified in
[`PLAN_SKIN_FRONTIER.md`](PLAN_SKIN_FRONTIER.md), so the label is real — but
spot-checking it against code at `ee88d3d`:

| Item | Plan | In code? |
|---|---|---|
| M1 ITA° / Fitzpatrick classification | `PLAN_SKIN_FRONTIER.md:22` | ✅ `color_science.py:763::ita_value` |
| M2 melanin / erythema index | `:30` | ❌ `grep "melanin_index\|erythema"` → empty |
| M4 pore/texture GLCM/Haralick | `:42` | ❌ `grep -i "glcm\|haralick"` → empty |
| I3 ambient-occlusion synthesis | `:92` | ❌ no estimator or synthesiser found |

So "M is done" is not a safe budgeting assumption: M1 shipped, M2/M4 did not,
and what genuinely exists alongside it is `perceptual_metrics.py`,
`face_quality.py` and `image_analyzer.py`. The surface is smaller and
differently shaped than the `M/I` label implies.

**I3 would not have unblocked P8's AO cue anyway.** The plan cites "AO from I3",
but I3 is titled *ambient-occlusion **synthesis*** — it renders dimensionality,
it does not estimate it. This is the same effect-vs-measurement inversion as
`skin_sss` above, and it recurs across P8's stated dependencies: the pieces the
plan names as measurement sources are, on inspection, mostly renderers. That
pattern is the strongest single reason to treat P8's dependency line as
unreliable and re-derive it, as §4 does.

## 3. Literature position

Three lines are relevant, and the useful reading is where they *disagree* with
the plan.

**Coordinated-cue aging is well supported.** The perception literature is
consistent that apparent age is carried jointly by surface topography, colour
heterogeneity and 3D volume, and that manipulating one in isolation reads as
artefact rather than youth — the "smoothed but still old" effect P8 names.
Chromophore heterogeneity in particular is repeatedly found to carry
apparent-age signal independent of wrinkles. This validates P8's *thesis*.

**It does not validate P8's method.** The published coordinated results
generally come from either (a) 3D-captured or multi-spectral data, or (b)
learned generative latent directions (StyleGAN-style age vectors, diffusion
edits). Neither is available here: the repo is classical-only by standing
decision (`A4 parked, classical-only until A1 evidence`), and there is no 3D or
multi-spectral capture. So the literature's coordinated results are *evidence
the axis is real*, not a recipe this pipeline can follow. Citing them as
implementation support would overstate what transfers.

**Translucency estimation from a single ordinary photograph is the known-hard
part.** This is the same inverse-problem wall P7 hit on 2026-09-09 — that
report's NO-GO was for "recovering one true skin albedo and lighting model from
an ordinary single photograph." Translucency recovery is that problem plus a
transport term. P7's conclusion should be treated as binding precedent here
rather than re-litigated: single-image translucency recovery is out of reach,
and any P8 scoping that assumes otherwise inherits a NO-GO that has already been
reached once in this repo this week.

## 4. Re-scoping the blocker

Splitting `M/I/P1` into its actual components changes the cost materially:

- **Translucency — genuinely blocked, P1-shaped.** Per §3 and P7 precedent. Do
  not attempt from a single image.
- **Volume / AO — *not* P1-shaped, but not free either.** P8 wants
  fat-pad-loss volume cues. That is coarse facial *shape*, and the repo already
  carries 478-point MediaPipe landmarks that `relight.py::sculpt` and the yaw
  gate consume geometrically, so a coarse proxy is plausible without any
  transport model. The plan's cited source (I3) would not have supplied it —
  I3 synthesises AO rather than estimating it (§2). Caveat: the one prior
  attempt in this repo to derive geometry from MediaPipe *z* was the neck depth
  gate, which was a silent no-op on all frontal faces (see E3). So this is
  unblocked from P1 but carries real technical risk, and E3 is where it gets
  tested — not assumed.
- **Wrinkle depth — unblocked**, per §2.
- **The three ✅ cues — unblocked**, available today.

So the accurate status is not "P8 blocked, needs P1." It is: **five of six cues
are reachable without P1; one (translucency) is not.** The open design question
P8's plan never asks is whether a five-cue vector with translucency *held fixed*
still reads as coherent age movement, or whether translucency is load-bearing.
That question is answerable by measurement (E4) long before any slider is built,
and its answer determines whether P8 is a ~2 week item or a post-P1 item. **This
is the highest-value thing to learn about P8, and it is cheap to learn.**

## 5. Proposed experiments

Measurement-only. None of these ships a control. All use the governed corpus
(`corpus_manifest` v3, dev/calibration splits — `locked_test` stays untouched,
and per the 2026-09-06 governance memo zero real subjects are assigned yet, which
is itself a prerequisite here).

- **E1 — cue readout harness.** Assemble the three available cues
  (`skin_homogeneity_state`, `facial_feature_contrast`, `extract_specular`) into
  one per-face readout. Deliverable: a table, not an operator. Purpose: establish
  that the three cues are stable and comparable across the corpus before any
  combination is attempted.
- **E2 — wrinkle magnitude extraction.** Expose the ridge response already
  computed in `_wrinkle_soften_masked` as a measurement; check whether it
  separates known-older from known-younger faces, and how badly pores/stubble/
  eyeliner contaminate it. The eyeliner confound is not hypothetical — FA-03
  already hit exactly that class of misfire on DSCF2310.
- **E3 — landmark AO/volume proxy.** Derive a coarse volume cue from existing
  478-point landmarks (no new model). Success = correlates with age labels;
  failure is an acceptable and informative outcome.
  **Must not be built on MediaPipe z-plane geometry.** The neck "depth gate"
  (CLAUDE.md, 2026-09-02) did exactly that — "with z scaled by face width the
  plane tilts into the image" — and was a *guaranteed no-op on frontal faces*,
  losing 100% of the mask on 15/15 frontal corpus faces, undetected because its
  unit test asserted the no-op on a uniform grey image. MediaPipe z is not metric
  depth. E3 must validate against something other than z-plane geometry (2D
  landmark ratios, shading statistics under the existing skin mask), and must be
  checked on frontal faces specifically, since that is where the prior attempt
  silently died.
- **E4 — the load-bearing question.** Given E1–E3, ask whether the five
  reachable cues co-vary with apparent age strongly enough that a fixed
  translucency term is tolerable. **This gates whether P8 is a ~2 week item or a
  post-P1 item, and it is the one experiment worth running first if only one is
  run.**

Ordering: E1 → E2/E3 (parallel) → E4. E4 is the decision point; E1–E3 exist to
make E4 answerable.

## 6. Prerequisites and honest caveats

- **Age labels do not exist.** Every experiment above needs per-subject apparent-age
  ground truth. The corpus manifest has no age field, and per the 2026-09-06
  governance memo it has no real-subject person mapping yet either. **E2/E3/E4 are
  not runnable until that exists.** This is the true first blocker on P8 — ahead of
  P1, and it is not mentioned anywhere in the P8 plan.
- **Fitzpatrick coverage is a known open gap**, flagged repeatedly (eye-gate
  study §6, dark-circle v2). Aging cues are chromophore-heavy, so a vector
  calibrated on a pale corpus is a fairness risk, not merely an accuracy one —
  and CLAUDE.md's tone-invariance rule applies directly to every threshold any
  of these cues would introduce.
- **No measured claims here.** Nothing was rendered or executed; §2 is code
  inspection, §3 is literature positioning. The cue table is a claim about what
  functions exist, not about how well they work.
- **Corpus hygiene.** Several corpus images' "largest face" is a hand or a
  background person (4589/4592/4596/4597/4599/4600/4618, per the post-epsilon
  re-audit). Any age readout must exclude those or it will measure a hand.

## 7. What I would do next

Not implementation. Resolve the age-label prerequisite in §6, then run E1 (which
needs no labels) to confirm the three available cues are stable. E4 is the
question that actually decides P8's shape and cost; everything else in this
report exists to make E4 answerable.

If the answer to E4 is "translucency is load-bearing," P8 returns to the post-P1
queue with a *measured* justification rather than an inherited one — which is
worth more than the current `blocked | needs M/I/P1` line either way.

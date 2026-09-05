# FA-03 classification experiment: results and readiness

**Date:** 2026-09-05

**Status:** Experimental/diagnostic only. No production code changed.
Continues `docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md`
("Astra's report," treated as the research baseline and not re-derived
here) with the first authorized classification experiment from that
report's §5. Candidate generation is frozen throughout — every arm below
scores the identical set of components from the real, unmodified
`FreckleRemover._detect_components`.

## 1. Exact code paths inspected

- `retouch/freckle.py:75` `FreckleRemover._detect_components` — candidate
  generation, called directly and unmodified.
- `retouch/freckle.py:126` `FreckleRemover._classify_anomaly` — the
  four-class additive-rule scorer and its `max(scores, key=scores.get)`
  winner selection (line 204), the exact site of the dict-order tie bug.
  Ported verbatim into this experiment's Arm 1 (verified byte-for-byte
  reproduction of the audit's `.45/.70/.70/0` scores, see §4).
- `retouch/marks.py:167` `_geometry_features` — confirms eccentricity is
  computed from axis-aligned bbox width/height, not true component shape
  (the report's specific warning; this experiment computes real PCA
  ratios instead, see §3).
- `retouch/parsing.py:378` `FaceRegions` — confirmed `left_eye`,
  `right_eye`, `left_eyebrow`, `right_eyebrow`, `hair`,
  `left_under_eye`/`right_under_eye` all exist and are populated on a
  real capture; used as semantic-context input.
- `retouch/perf_optimizations.py` / `retouch/engine.py` — confirmed
  `engine.py:98-103` imports `_process_face_core` by name (`from
  .perf_optimizations import (..., _process_face_core, ...)`), so
  monkeypatching `perf_optimizations._process_face_core` does **not**
  intercept `engine.process()`'s call; the module-level reference in
  `retouch.engine` must be patched instead. This tripped the first
  capture attempt in this tranche (empty capture, silently) before being
  diagnosed and fixed — noted here since it is a real trap for any future
  monkeypatch-based capture against this codebase.

No file under `retouch/` was modified. No production call site was
changed.

## 2. Experiment architecture

Three passes, in a new file (`scripts/qa/fa03_mark_classification_experiment.py`),
never imported by `retouch/`:

1. **`freeze_candidates()`** calls `FreckleRemover._detect_components`
   directly and — unlike `freckle.py`'s own `classify_anomalies`, which
   discards the true component mask after computing bbox-derived stats —
   keeps the boolean component mask itself. This is the single change
   that makes real shape features possible instead of bbox proxies.
2. **`compute_features()`** — one feature dict per frozen candidate,
   read by every arm. Combines current-rule appearance inputs
   (area/a_norm/L_norm/chroma/L_std) with new shape features (true PCA
   major/minor ratio, contour circularity, solidity, skeleton
   length-to-width) and new semantic-context features (normalized
   distance to eye/brow/hair masks, boundary tangent alignment, boundary
   crossing) and a local reference-ring appearance delta with an
   explicit `ring_quality` flag for insufficient support.
3. **Five scoring arms**, each a pure function `features -> ArmResult`
   (full class-score dict + winner + margin + reason). No arm ever
   collapses scores before an explicit, order-independent tie/near-tie
   check (`_resolve_winner`, `TIE_MARGIN=0.05`): below that margin the
   outcome is `ambiguous`, never whichever class a dict happens to
   enumerate first.

Verification that the port is faithful: Arm 1 reproduces the audit's
exact skin baseline (`l_median=185.00, a_median=142.00, l_std=34.3423,
a_std=3.7356`), exact scores (`freckle=.45/beauty_mark=.70/blemish=.70`
and `beauty_mark=.70/blemish=.70`), and exact PCA ratios (`6.89`, `8.14`)
on the two DSCF2310 target components before any other arm's numbers
were trusted.

## 3. Features implemented/exposed

| Feature | Source | Notes |
|---|---|---|
| `pca_axis_ratio` | True component covariance eigenvalues | Not bbox aspect — the report's explicit requirement. Matches audit values exactly. |
| `circularity` | `4*pi*Area/Perimeter^2` on true contour | Matches audit's diagnostic metric. |
| `solidity` | Contour area / convex-hull area | New; distinguishes a thin stroke from a compact blob of equal PCA ratio. |
| `skeleton_length_to_width` | PCA-major-axis projection length / (area/length) | Cheap proxy, not a true morphological skeleton. |
| `eye_distance_norm`, `brow_distance_norm`, `hair_distance_norm` | Min distance from component centroid to nearest mask pixel, divided by `face_width` | Face-normalized, not absolute pixels. |
| `eye_alignment`, `brow_alignment` | `|cos(angle)|` between component PCA major axis and local finite-difference tangent of the nearest eye/brow contour point | New; the report's ILoveEye-motivated feature (tangent continuation). |
| `crosses_eye_boundary`, `crosses_brow_boundary`, `crosses_hair_boundary` | Direct mask overlap | Boolean, not distance. |
| `ring_dl`, `ring_da`, `ring_db`, `ring_quality` | Local 15×15-dilated annulus, component and skin<0.8 excluded | `ring_quality=0` when fewer than 8 ring pixels are available — evidence of uncertainty, not clean skin, feeding Arm 5's trust weighting. |
| `area`, `a_norm`, `l_norm`, `chroma`, `l_component_std` | Unchanged from `freckle.py` | Retained so Arm 1 is a faithful port. |

Not implemented (explicitly out of scope per the report and this task's
bound): oriented texture/Hessian multiscale response, a fitted logistic
or boosted-tree model (Arm 5 is a hand-scored rule combination, not a
fitted classifier — insufficient labels exist for fitting, per the
report's own caution), pigment/hemoglobin margin features (available
elsewhere in the pipeline but not wired into this experiment), and
dense BiSeNet probability fields (observational only elsewhere in the
codebase, not retained per-pixel here).

## 4. Results per experimental arm

### DSCF2310 target components (the confirmed failure)

| Arm | Viewer-left (bbox 236,381,18,12) | Viewer-right (bbox 367,428,22,13) |
|---|---|---|
| 1 current baseline (ported, tie-break fixed) | `ambiguous` (.45/.70/.70/0/0) | `ambiguous` (0/.70/.70/0/0) |
| 2 geometry-only | `eye_makeup` (.70) | `eye_makeup` (.70) |
| 3 semantic-context-only | `eye_makeup` (.70) | `eye_makeup` (.70) |
| 4 geometry+semantic | `eye_makeup` (.90) | `eye_makeup` (.90) |
| 5 scored classical | `eye_makeup` (.70, beauty_mark .40, blemish .40) | `eye_makeup` (.70, beauty_mark .40, blemish .40) |

**Neither component is accepted as a mole under any of the 5 arms.**
Arm 1 alone (the minimal tie-break fix, no new features) does not
produce `eye_makeup` — it correctly stops emitting a confident `mole`
but abstains rather than identifying the makeup hypothesis, since it has
no shape/context evidence to do so. Arms 2, 3, 4, 5 all correctly
identify `eye_makeup` with clear margins.

### Preservation check: synthetic near-eye compact marks (DSCF2310 canvas)

Composited a compact, round, dark synthetic mark (radius 4, PCA ratio
≈1.01) at three distances below the same eye whose lash line produced
the false positives above (5px, 15px, 60px from the eye's lower edge —
`eye_distance_norm` 0.071, 0.114, 0.325 respectively):

| Distance | Arm 1 | Arm 2 | Arm 3 | Arm 4 | Arm 5 |
|---|---|---|---|---|---|
| 5px (near) | `beauty_mark` | `beauty_mark` | `ambiguous` | `beauty_mark` | `beauty_mark` |
| 15px | `beauty_mark` | `beauty_mark` | `ambiguous` | `beauty_mark` | `beauty_mark` |
| 60px (far) | `beauty_mark` | `beauty_mark` | `ambiguous` | `beauty_mark` | `beauty_mark` |

Arms 1, 2, 4, 5 correctly preserve the compact mark as `beauty_mark` at
every distance, including 5px from the eye — proximity alone does not
reject it. **Arm 3 (semantic-context-only) abstains at every distance**,
including far from the eye, because it has no shape/appearance evidence
at all to break its own flat `freckle=.3/beauty_mark=.3` prior; this is
not a proximity-driven rejection, it is a structural consequence of
having only one evidence category. This grid varies distance only (fixed
shape/area) so it is a **floor test that all four non-context-only arms
pass**, not a test that discriminates quality among them — a genuinely
harder discriminating case is the elongated-far-from-boundary probe
below.

### Elongated stroke far from any eye/brow boundary (new probe, real face)

A synthetic elongated stroke (PCA ratio 6.47, solidity 0.884 — high
solidity despite elongation, since it is a smooth anti-aliased line
rather than fragmented real eyeliner) placed on a verified deep-interior
skin point (379, 485; `eye_distance_norm=0.273`, `brow_distance_norm=0.512`
— far from both):

| Arm | Winner | Scores |
|---|---|---|
| 1 current baseline | `beauty_mark` | .90 beauty_mark, .50 blemish |
| 2 geometry-only | **`eye_makeup`** | .70 |
| 3 semantic-context-only | `ambiguous` | .3/.3 |
| 4 geometry+semantic | `noise` (no positive score) | — |
| 5 scored classical | `beauty_mark` | .45 beauty_mark, .35 eye_makeup, .25 blemish |

**Arm 2 (geometry-only) misclassifies this as `eye_makeup` on shape
alone**, with no boundary evidence at all — a real false-positive mode
(cosplay face paint, a stray hair mid-cheek, an elongated scar) that the
DSCF2310 case and the near-eye preservation grid do not surface, because
both of those are inherently near a real boundary. Arm 4, which requires
elongation **and** boundary proximity **and** alignment together before
committing to `eye_makeup`, correctly refuses the makeup call here — but
its non-makeup fallback only covers the same narrow area bands the
current rules use (4–25, 10–70), so a large elongated non-boundary
component (area 184 here) falls through to zero score in every class
rather than a confident identity-class call. This is a real, documented
limitation of Arm 4's current fallback, not a hidden bug: it correctly
avoids the false positive but does not yet have a positive class to
assign a large non-eyeliner elongated mark (e.g., a genuine elongated
scar, which the report's taxonomy explicitly wants preserved). Arm 5
handles this case correctly (`beauty_mark`, since its `eye_makeup` vote
requires near_boundary AND aligned together, same as arm 4, but its
appearance votes still fire independently and dominate).

### Real-corpus pilot (candidate generation frozen, per-arm class distribution)

Ran on the 3 real corpus faces where a `FaceRegions` capture succeeded
(DSCF2306, DSCF2308, DSCF2310 — see §7 for the 2 that failed to capture).
Full JSON: `scripts/qa/fa03_pilot_out/pilot_results.json`.

| Face | Candidates | Arm 1 abstain rate | Arm 2 abstain rate | Arm 3 abstain rate | Arm 4 abstain rate | Arm 5 abstain rate |
|---|---|---|---|---|---|---|
| DSCF2306 | 46 | 10.9% | 6.5% | 34.8% | 17.4% | 6.5% |
| DSCF2308 | 58 | 19.0% | 13.8% | 25.9% | 22.4% | 5.2% |
| DSCF2310 | 63 | 15.9% | 9.5% | 23.8% | 14.3% | 4.8% |

**Caution on interpreting these abstention rates** (caught before
writing this table as a ranking): Arm 3's non-adjacency fallback assigns
an exact `freckle=.3/beauty_mark=.3` tie, so it structurally cannot
produce anything but `eye_makeup`, `noise`, or `ambiguous` — it has no
`freckle`/`beauty_mark`/`blemish` column in its own class distribution at
all. Its high `eye_makeup` counts (e.g. 29/58 on DSCF2308) are therefore
**not** a directly comparable measurement of proximity-driven over-calling
against the other arms' `eye_makeup` counts (8–16 on the same face) — they
reflect that arm's total absence of an appearance fallback, not a
proven rate of false makeup calls. The synthetic near-eye grid and the
DSCF2310 result are the evidence for arm 3's weakness; this table's raw
counts are not restated as that evidence. Likewise, Arm 5's low
abstention (5–6.5%) reflects many small additive weighted votes rarely
tying exactly, not superior decision quality — abstention rate here
measures score granularity, not correctness. Arm 5 is judged correct in
this report because it got DSCF2310 right, preserved all three synthetic
near-eye compacts, and correctly handled the far-from-boundary elongated
probe — not because of this table.

## 5. DSCF2310 outcome (explicit answer)

**Both confirmed eyeliner components stop being accepted as moles under
every arm except Arm 1 alone**, and Arm 1 (the minimal fix) also stops
emitting a confident `mole` — it abstains instead, which satisfies the
report's explicit acceptance criterion ("both known liner cases cease
being accepted as moles/blemishes") without requiring any new feature.
**No arm's fix for DSCF2310 caused a synthetic near-eye compact mark to
be rejected** (§4, near-eye preservation grid) — arms 1, 2, 4, 5 all
preserved `beauty_mark` at 5px, 15px, and 60px from the same eye.

## 6. Genuine-mark regression outcome

Tested **synthetically only** — the report's own corpus-limitation note
is confirmed still true: no confirmed natural mole/freckle exists
anywhere in the available corpus (CLAUDE.md's known-limitations section,
restated by the report). The synthetic near-eye grid and the isolated
far-from-boundary compact-mark case (from the prior FA-02 session, same
canvas) are the only preservation evidence available. **Near-eye genuine-
mark safety is not established on real photographs** — this experiment
cannot close that gap without reviewed real positives, which the report
also states do not currently exist.

## 7. Tests run/results

`tests/test_fa03_mark_classification_experiment.py` — 17 tests, all
passing:

- `TestTieHandlingIsOrderIndependent` (5 tests): exact tie → `ambiguous`
  regardless of dict key order; near-tie within `TIE_MARGIN` also
  abstains; a clear winner above the margin is not abstained; the exact
  DSCF2310 audit scores (`.45/.70/.70/0`) reproduce as `ambiguous`, not
  `beauty_mark` — the direct regression test for the dict-order bug.
- `TestElongatedEyeAdjacentCandidate` (3 tests): synthetic eyeliner-like
  stroke is genuinely elongated (PCA≥3.0); arms 2/4/5 classify it
  `eye_makeup`; arm 1 never emits a confident `beauty_mark` on this
  shape.
- `TestCompactGenuineMarkNearEye` (2 tests): compact near-eye mark is not
  classified `eye_makeup` by arms 1/2/4/5; arm 3's known weakness on this
  exact case is recorded (not asserted as correct) so it is not
  rediscovered by trial and error later.
- `TestCompactMarkAwayFromBoundaries` (1 test): isolated compact mark
  classifies as an identity-like class or abstains, never spuriously
  `eye_makeup`.
- `TestAmbiguityAbstentionBehavior` (3 tests): no-evidence resolves to
  `noise` not an arbitrary class; abstain results retain the full score
  vector (never discarded); `ambiguous` is a distinct outcome from every
  real class.
- `TestDeterminism` (1 test): repeated calls on identical input produce
  byte-identical scores and winners across all 5 arms.
- `TestLegacyPathUnchangedWhenExperimentalClassifierDisabled` (2 tests):
  `FreckleRemover._detect_components` is never monkeypatched or replaced;
  importing the experiment module does not change
  `FreckleRemover.classify_anomalies`'s own output.

Full existing suite sanity check (no production file touched, so this is
confirmation, not a risk mitigation): `tests/test_fa03_mark_classification_experiment.py
tests/test_freckle.py tests/test_marks.py` — 45 passed. Golden pipeline
tests were not re-run — no file under `retouch/` changed, so there is
nothing for them to detect.

## 8. Is FA-03 ready for production integration?

**No.** Concretely:

- Thresholds (`pca_axis_ratio >= 3.5`, `solidity <= 0.55`,
  `eye_distance_norm < 0.08`, `TIE_MARGIN = 0.05`) are hand-chosen from
  one portrait's measurements (DSCF2310) plus this pilot's 3 faces — the
  report's §4.1 explicitly warns against promoting one portrait's
  measured ratios into tuned production constants, and that is exactly
  their current provenance.
- Zero confirmed natural moles/freckles exist in any available corpus
  image; near-eye genuine-mark safety is established synthetically only
  (§6).
- Arm 3 has a demonstrated structural weakness (misclassifies via a flat
  prior whenever it has no adjacency evidence) and Arm 2/5 have a
  demonstrated blind spot (an elongated far-from-boundary stroke can be
  misclassified, or in Arm 4's case, correctly rejected but left with no
  positive class) — none of the 5 arms is unconditionally safe to ship
  as-is.
- The pilot is 3 real faces (2 more failed to capture — see below), not
  the report's own named 30-portrait/192-synthetic floor; this document
  does not claim otherwise.
- `ambiguous` has no home in `retouch/freckle.py`'s `_CLASS_TYPES`
  (`{freckle, beauty_mark, blemish, noise}`) or `retouch/marks.py`'s
  `MARK_CLASSES`. Shipping any arm's abstain outcome into production
  means either adding a new class throughout that taxonomy or mapping it
  to the existing `unknown` class — which `protect_identity` already
  preserves, so the practical effect might be acceptable, but this
  decision was not made or tested here and needs its own check before
  being assumed safe.

**One exception, presented per this task's explicit clause** ("If a
minimal production-safe fix becomes obvious, present the evidence and
recommended change before broad rollout"): the tie-break fix alone
(Arm 1's `_resolve_winner`, ported as a ~15-line change to
`freckle.py:204`'s `best = max(scores, key=scores.get)`) is a narrow,
independently-tested, evidence-backed change that:
- stops the exact confirmed DSCF2310 failure (both components move from
  a silently confident `mole` to an explicit `ambiguous`),
- requires no new features, thresholds, or corpus work,
- is proven order-independent by
  `test_tie_outcome_is_identical_regardless_of_dict_key_order`,
- does not touch candidate generation, repair, or any other op.

This is **not** recommended for immediate rollout in this task (out of
scope — "do not immediately replace the production classifier unless the
experiment shows a clear improvement and regression coverage is strong,"
and the `ambiguous`-taxonomy question above is unresolved), but it is
named here as the one change with clear minimal-fix character, per the
task's own conditional clause, for a future explicitly authorized
decision.

## 9. Can FA-02 now safely consume mark classifications?

**Still no**, but the specific blocker has narrowed. FA-02's option 1
(unconditional `detect_marks`, rejected in `e072fdb`) failed because the
DSCF2310 eyeliner components were classified `mole` and therefore
excluded from `restore_micro_texture`'s dimensional-zone restoration —
exactly the makeup-suppression risk that rejection was about. Under
every arm in this experiment except Arm 1 alone, those same components
now classify as `eye_makeup` or `ambiguous`, not `mole` — so the specific
confound that killed FA-02 option 1 is, in principle, addressable by an
improved classifier. This does **not** unblock FA-02 today:
- `docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md` §6 requires
  independent mark-type evidence, retained true supports, an explicit
  ambiguous/action contract, **and** a separate successful rendering
  check before any automatic exclusion is authorized — none of the last
  three exist yet.
- This experiment's classifier is not integrated into `detect_marks()`
  or `MarkRecord`; `mark_class` values like `eye_makeup` do not exist in
  `MARK_CLASSES` today.
- No FA-02 rendering experiment (restoration on/off/excluded, matched
  source/smoothing/regions, scored for defect recovery vs. makeup
  retention) has been run with this or any classifier's output — the
  report's §6 explicitly gates that on detection being frozen first.

## 10. Remaining corpus/evidence limitations

- Pilot is 3 real faces (DSCF2306, DSCF2308, DSCF2310), not Astra's
  report's named 30-portrait/192-synthetic floor — explicitly not
  presented as that benchmark.
- 2 of the 5 named corpus faces (DSCF2362, DSCF2365) failed to produce a
  `_process_face_core` capture at all in this session (`engine.process()`
  completed with 0 calls to the patched function — most likely 0 faces
  detected on that specific crop/recipe combination; not investigated
  further, since diagnosing detection recall is explicitly out of this
  task's bound — "Do not start... broad candidate-recall work").
- Zero confirmed natural moles/freckles in any available image (restated
  from the report, not newly discovered).
- No heavy-makeup-but-non-eyeliner cases (blush, contour, colored
  lenses), no profile/occluded-eye cases, no darker-skin-tone samples —
  all named as open gaps in the report's own corpus plan (§5) and not
  addressed by this 3-face pilot.
- Thresholds are not calibrated against a held-out set; development and
  evaluation use the same 3 faces (the report's own partition
  requirement — development/calibration/locked-test — was not
  implemented, consistent with "do not pretend this is the final
  benchmark").
- The elongated-far-from-boundary false-positive mode (§4) was found on
  one synthetic probe; not swept over stroke length, thickness, or
  location.

# Fix: deterministic tie/near-tie resolution in FreckleRemover classification

**Date:** 2026-09-05

**Status:** Correctness fix, shipped. **This is not FA-03's completion.**
FA-03 (`docs/plans/RESEARCH_FA03_MARK_LOCALIZATION_2026_09_05.md`,
`docs/plans/RESEARCH_FA03_CLASSIFICATION_EXPERIMENT_2026_09_05.md`)
remains open: no shape/semantic-context classifier is production, no
corpus/threshold work has been done, and FA-02 remains blocked. This
change implements only the narrow, isolated portion of that work item
named production-ready in the classification experiment's §8 — the
tie-break resolver alone, with no new features, thresholds, or
candidate-generation changes.

## Problem

`retouch/freckle.py:204`'s `best = max(scores, key=scores.get)` silently
resolved exact or near-exact ties between competing class scores by
Python dict insertion order, not by any semantic signal. The confirmed
production case: two drawn-eyeliner components on a real photograph
(DSCF2310) scored `beauty_mark=0.70` and `blemish=0.70` — an exact tie —
and were classified `beauty_mark` (which `retouch/marks.py`'s
`_FRECKLE_CLASS_MAP` renames to `mole`) purely because `"beauty_mark"`
appears before `"blemish"` in the scores dict literal.

## Fix

`retouch/freckle.py`:

- `_CLASS_TYPES` gains a fifth member, `"ambiguous"`.
- A new named constant `_CLASS_PRIORITY = ("freckle", "beauty_mark",
  "blemish", "noise")` reproduces today's dict-insertion-order behavior
  as an explicit, testable tuple rather than an emergent property of a
  dict literal — this alone satisfies "remove dictionary-order
  dependence" without changing which class wins any existing case.
- `_classify_anomaly`'s winner selection now sorts candidates by
  `(-score, _CLASS_PRIORITY.index(name))`, then checks the top-two
  margin. **Only when the resulting top class is `"beauty_mark"` and
  the margin is below `_TIE_MARGIN = 0.05`** does the outcome become the
  explicit `"ambiguous"` class instead. Every other tie pattern (e.g.
  freckle vs. blemish, both legitimately reachable at an exact 1.00/1.00
  score sum by the table's own arithmetic) keeps resolving via
  `_CLASS_PRIORITY`, exactly as `max()` did before.

`retouch/marks.py`:

- `_FRECKLE_CLASS_MAP` gains `"ambiguous": "unknown"`. `unknown` is an
  existing `MARK_CLASSES` member; every named policy (`protect_identity`,
  `preserve_all`) already has a defined, safe action for it (preserve).
  Without this entry, the first `"ambiguous"` result reaching
  `detect_marks()` would raise `KeyError` — this was caught before
  shipping, not discovered after.

## Why the scope is this narrow, not "route every tie to ambiguous"

An earlier draft of this fix abstained on **every** class tie
unconditionally. That broke 4 existing tests
(`test_classify_freckle`, `test_fitzpatrick_i_to_vi_freckle_detection_and_classification_is_invariant`,
`test_remove_freckles_removed_beauty_preserved`,
`test_light_skin_removal_output_is_byte_identical`) because `freckle`'s
and `blemish`'s rule weights both sum to exactly `1.00`
(`0.5+0.45+0.05` and `0.2+0.3+0.5`), so a freckle that is simultaneously
small, reddish, and internally uneven legitimately ties both classes at
their maximum score — a real, reachable pattern, not a fixture defect.
Under the broad draft that tie became `ambiguous`, and since
`FreckleRemover.remove()` only heals `classification == "freckle"`
records, the freckle was never healed and the byte-identical legacy hash
moved. **The task's requirement is "must not silently resolve to mole,"
not "must not silently resolve at all"** — narrowing the abstain
condition to `top_class == "beauty_mark"` satisfies the stated
requirement with zero collateral, verified by re-running the same 4
tests after narrowing (all pass unmodified, no fixture or hash changed).

## What downstream behavior actually changes

`resolve_mark_action` (`retouch/marks.py:307`) maps `mark_class ==
"unknown"` through `policy.get(mark_class, policy.get("unknown",
{"action": "preserve"}))`. Under `protect_identity`, `unknown` is
explicitly `{"action": "preserve"}` — the same disposition `mole`
already had. **This means a component that moves from `beauty_mark` to
`ambiguous` is preserved identically under an active mark policy either
way.** What changes is that the record no longer *claims* to be a mole
— an accuracy fix to the classification, not a change in what happens to
the pixels under policy. Under `mark_policy=None` (legacy, no policy),
neither `beauty_mark` nor `ambiguous` records are read by anything except
`FreckleRemover.remove()`'s own `classification == "beauty_mark"` check
(which draws a preserve circle) — an `ambiguous` component draws no such
circle, but since it also isn't `"freckle"` it is never a healing
candidate either, so it is left untouched either way at the pixel level
in `remove()`'s own no-policy path.

## Regression coverage

`tests/test_freckle.py`, new tests (12 total, all passing):

- `TestTieBreakDoesNotSilentlyResolveToMole` (6 tests):
  - `test_dscf2310_style_elongated_stroke_is_ambiguous_not_mole` — a
    full end-to-end scene through `classify_anomalies` (not just the
    scoring function), constructed by round-tripping the audit's actual
    LAB skin baseline (L=185, a=142, b=128) and two neutral-chroma dark
    shades through `LAB2BGR`, reproducing the exact `beauty_mark=0.70 /
    blemish=0.70` tie on a thin, internally two-toned, non-round
    component.
  - `test_direct_tie_beauty_mark_and_blemish_resolves_to_ambiguous` and
    `test_ambiguous_confidence_clears_the_default_threshold` — isolate
    the tie-break logic directly via `_classify_anomaly`, confirming the
    `ambiguous` confidence (`0.7`, the contested score) clears
    `classify_anomalies`'s default `0.6` threshold rather than being
    silently dropped.
  - `test_ambiguous_is_a_valid_classification_type` and
    `test_ambiguous_maps_to_unknown_mark_class_without_crashing` — the
    taxonomy-wiring guard: asserts `_FRECKLE_CLASS_MAP` covers every
    `_CLASS_TYPES` member, the exact check that would have caught the
    `KeyError` this fix avoided.
- `TestOrdinaryNonTiedClassificationsAreUnchanged` (7 tests): clear
  freckle/beauty_mark/blemish classifications, the Fitzpatrick I-VI
  freckle-invariance sweep, freckle-healing-and-beauty-mark-preservation,
  and the pre-existing byte-identical hash test all re-verified passing
  **unmodified** (no fixture or snapshot changed to make them pass) —
  plus two new ties added specifically to lock in the narrow scope:
  - `test_freckle_blemish_exact_tie_still_resolves_to_freckle_not_ambiguous`
    — a genuine, directly-constructed `1.0`/`1.0` freckle/blemish tie,
    confirmed to still resolve to `freckle` via `_CLASS_PRIORITY`, not
    `ambiguous`.
  - `test_beauty_mark_noise_tie_on_tiny_component_also_routes_to_ambiguous`
    — a second real tie path surfaced by the production before/after
    diff below: a sub-noise-area component (area 3px) can score
    `beauty_mark=0.9` from the a_norm/L_norm bonuses alone, exactly
    tying `noise=0.9`; this also must not become a confident mole.

## Verification against real production renders

Captured real `(canvas, regions)` pairs from live `engine.process()`
calls (via monkeypatching `retouch.engine._process_face_core` — patching
`perf_optimizations`'s own module-level reference does not intercept
`engine.py`'s call, since `engine.py` imports the name directly; this
tripped the first capture attempt in the preceding FA-03 experiment
session) on three real corpus faces, and diffed
`FreckleRemover.classify_anomalies`'s output before and after this fix
(candidate generation unchanged, confirmed by identical candidate
counts before/after on every face):

| Face | Total candidates | Classification changed | All changes |
|---|---|---|---|
| DSCF2306 | 44 | 4 | `beauty_mark` → `ambiguous` |
| DSCF2308 | 48 | 4 | `beauty_mark` → `ambiguous` |
| DSCF2310 | 55 | 6 | `beauty_mark` → `ambiguous` (includes both confirmed target bboxes `(236,381,18,12)` and `(367,428,22,13)`) |

Every changed component's bbox is elongated (widths 5–24px vs. heights
5–13px — aspect ratios consistent with drawn eyeliner/lash strokes, not
round marks); the remaining 40, 44, and 49 records respectively are
byte-for-byte identical in classification and confidence. This is the
"ordinary non-tied classifications preserve legacy behavior" evidence,
measured directly rather than argued from the rule table.

## Golden hashes / unrelated recipes

`tests/test_golden_pipeline.py` and `tests/test_golden_pipeline_face.py`:
**13 passed, zero hashes changed.** `tests/test_recipes.py` and
`tests/test_recipe_integration.py`: 169 passed. Combined regression run
(`test_freckle.py`, `test_marks.py`, `test_blemish_mark_policy.py`,
`test_skin_evening_mark_policy.py`, `test_mark_policy_wiring.py`,
`test_fa03_mark_classification_experiment.py`, both golden suites,
`test_recipes.py`, `test_recipe_integration.py`): **251 passed.**

## Explicit non-goals (per task scope)

- The geometry/semantic-context classifier from
  `docs/plans/RESEARCH_FA03_CLASSIFICATION_EXPERIMENT_2026_09_05.md` is
  **not** shipped here and remains experimental-only
  (`scripts/qa/fa03_mark_classification_experiment.py`).
- Candidate generation (`FreckleRemover._detect_components`) is
  unchanged — confirmed by identical candidate counts before/after on
  all three real-face captures above.
- FA-02 remains blocked for automatic detector-derived exclusions; this
  fix does not touch `retouch/perf_optimizations.py`,
  `retouch/skin.py`'s `restore_micro_texture`, or any rendering path.
- No thresholds, area bands, or scoring weights were changed —
  `_TIE_MARGIN=0.05` is new but only gates the `ambiguous` routing
  decision, not any existing rule's score contribution.

# Audit — Eye-Occlusion Gate Working Tree (post-review status check)

**Date:** 2026-08-27
**Branch:** `feat/color-science-k9-fix-and-frontier`
**Scope:** verify what actually landed in the uncommitted working tree against
`docs/review/REVIEW_EYE_VISIBILITY_GATE_2026_08_26.md`'s findings and
`docs/plans/RESEARCH_EYE_OCCLUSION_2026_08_26.md`'s premises, using the pinned
`.venv/bin/python` (mediapipe 0.10.5, protobuf 3.20.3) per that review's own
documented trap (bare `python3` = drifted system Python, silently degrades
`person_mask` to all-ones).
**Method:** read the diff, ran the touched tests, bisected a live test
failure, ran a real mutation check (not delegated to a subagent — this is
verification, not exploration).

---

## 0. What actually shipped vs. what the docs claim

The review (2026-08-26) recommended a strict order of work (§8a). The research
doc's §0 states five "premises," presented as already resolved. Checked
against the current diff:

| Item | Review/doc says | Actual state (verified) |
|---|---|---|
| §8a.0 — per-eye catchlight fix (B6) | Fix eyes.py:119-121 | **Done.** `eyes.py` loops `for iris_m in (regions.left_iris, regions.right_iris)` instead of pooling. |
| §8a.1 — P0 sclera no-op | Fix parsing.py:753-761 | **Done**, transitively — fixed by the P1 handedness swap below; `left_sclera = clip(left_eye - left_iris)` now subtracts co-located masks. |
| §8a.2 — P1 handedness swap | Fix parsing.py:369-370 | **Done.** `_masks_from_label_map` now swaps BiSeNet labels 4↔5 into camera-viewer convention, documented inline with a comment pointing at the review. |
| §8a.3 (research doc premise 3) — EAR-based gate rewrite | "being rewritten," thresholds "PROVISIONAL" | **Not done.** `retouch/eye_visibility.py` is byte-for-byte the same area/overlap heuristic the review said "should not ship in its current form" (§0 verdict). No EAR, no contrast signal. |
| Research doc premise 4 — gate memoized on `FaceRegions` at parse time | "computed once... new `__slots__` entry" | **Not done.** No new slot in `FaceRegions.__slots__`. The gate is still called fresh, from scratch, on every consumer. |
| §8a.4 — ParamSpec + log line for the gate | recommended | **Not done.** Still unconditional and silent. |
| §8a.5 — the calibration study itself (labels, ROC, corpus) | recommended | **Not done.** The research doc is a protocol only — "Status: Protocol/spec, not yet run" is accurate and still true. |
| §8a.6 — extend tests to arm B (BiSeNet-unavailable) | recommended | **Not done.** `tests/test_eye_visibility.py` still paints hair explicitly; never forces `parser._sess = None`. |
| B5 — gate result never reaches sharpen mask | fix by hoisting to `_process_face_core` | **Done.** `perf_optimizations.py:276-278` now calls `gate_occluded_eye_regions(regions)` once at the top of `_process_face_core`, before the sharpen accumulator is built. New test `test_occluded_eye_is_removed_from_sharpen_mask` (tests/test_engine.py) verifies this on a synthetic hair-occluded right eye. |
| §6.5 — BiSeNet CPU pin justification | "attach evidence or reword" | **Not addressed** — comment still asserts the CoreML MLProgram failure without a citation. Low severity per the review itself (§6 point 3: precedent exists for unparametrized correctness pins). |

**Read this as:** the two committed-code correctness bugs (P0/P1) and the two
architectural leaks the review found most dangerous (B5 sharpen leak, B6
catchlight coupling) are fixed. The actual occlusion-detection logic — the
part the whole feature exists to deliver — is untouched. This is exactly the
review's §8a step 3 ("do not ship `eye_visibility.py` as-is... replace
`_should_gate_eye`'s evidence") still outstanding, and the research doc that's
supposed to calibrate its replacement hasn't been run.

---

## 1. New finding this session: golden face-path regression, now attributable

The review's §2 retraction stated the golden face-path suite passes 8/8 under
`.venv` "with the full uncommitted diff applied." That is **no longer true**
on the current tree:

```
.venv/bin/python -m pytest tests/test_golden_pipeline_face.py -q
# -> 3 failed, 5 passed (natural, porcelain_unified_v1, outdoor_harsh_sun_v1)
```

Confirmed **not** a runtime-drift artifact (same `.venv`, same env vars as the
review used) by a clean-HEAD comparison:

```
git stash                       # -> HEAD, no working-tree diff
.venv/bin/python -m pytest tests/test_golden_pipeline_face.py -q
# -> 8 passed
git stash pop
```

**Bisected the cause** (reverted only `retouch/eyes.py`'s diff, reran):

```
git checkout -- retouch/eyes.py   # keep everything else
.venv/bin/python -m pytest tests/test_golden_pipeline_face.py -q
# -> 8 passed
git apply <the eyes.py diff>      # restored
```

The regression traces to `eyes.py`'s per-eye catchlight loop — i.e., the B6
fix itself, not the P0/P1 handedness swap (the golden fixture forces
`parser._sess = None`, so `_masks_from_label_map` never runs on this path;
BiSeNet-derived masks can't be the cause here). This matches the review's own
§B6 measurement almost exactly: pooled-vs-per-eye catchlight compositing shifts
the visible eye by up to 30 px, max per-channel delta 19, on this exact
fixture.

**This is very likely an intentional, correct render change** — B6 documented
the pooled catchlight as a latent bug the gate exposes, and fixing it changes
output on a two-visible-eye face (no gating fired) purely because the
catchlight centroid is no longer pooled across both irises. The golden
snapshots are simply stale for these 3 recipes.

**Action needed, not yet taken:** re-bless the three snapshot hashes with a
commit message citing the per-eye catchlight fix as the reason, and — per
CLAUDE.md's verification rule — view the actual before/after renders on a real
image first, not just the hash diff, since a 30 px / delta-19 shift is small
enough to hide a real defect inside a "looks like a golden update" commit.
That render check has not been done this session (the only renders in
evidence anywhere are the review's, and those predate this exact diff).

---

## 2. Re-checked: is B3 ("vacuous integration tests") still accurate?

The review's B3 finding says replacing the gate with an identity function
leaves both `tests/test_eye_visibility.py` integration tests passing. I ran
that mutation directly against the current tree (monkeypatching the
already-imported `gate_occluded_eye_regions` name in both `eyes.py` and
`eye_enhancement.py`, which is the correct patch point since both modules
import it by name):

```
4 of 5 tests in test_eye_visibility.py FAIL under identity mutation,
including both enhancer integration tests B3 called vacuous.
```

So B3's literal claim does not reproduce as stated today. But this doesn't
close the underlying gap — narrowing, not retracting:

- All 4 mutation-sensitive tests key off `regions.right_eye.fill(0.0)` or a
  hair block large enough to trip the crude `eye.max() < 0.01` / hair-overlap
  branches. None of them exercise `eye_to_iris_area`, `iris_inside_eye`, or a
  near-threshold hair ratio — exactly the branches B0, B0a, B4, and B7 proved
  give wrong answers on real corpus data (28/36 real eyes gated, mostly wide
  open; 15/26 visible small faces gated; a genuinely-closed-eye case at EAR
  0.057 read as open).
- So: the tests are no longer *vacuous*, but they still only cover the
  gate's trivial branches, not its calibration. The review's proposed
  25-test mutation-verified suite (`/tmp/test_eye_visibility_proposed.py` +
  `/tmp/conftest.py`, still present, unmerged) remains the right fix — it's
  the only test artifact that exercises the area-ratio and hair-threshold
  branches directly.

---

## 3. Checked and downgraded: "both eyes gated → frame-wide sharpen" risk

Considered whether hoisting the gate into `_process_face_core` could, on a
face where B0a-style false-positive gating hits both eyes (documented: DSCF8007,
both eyes wide open, gated on both sides by the pre-P1 tree), empty the
sharpen accumulator enough to trigger `engine.py:4433-4434`'s
`np.ones_like` frame-wide fallback.

Traced `acc_sharpen`'s construction (`perf_optimizations.py:1016-1029`): it
accumulates eye masks **plus** eyebrow, hair-edge, and other per-face region
contributions before clipping. Both eyes gated to zero does not zero the
whole accumulator unless every other contributing region is also empty on
that face — a narrow compound case, not the single-cause "whole-frame
sharpening" scenario. **Downgraded from the initial read: worth a targeted
regression test, not an urgent fix.** No concrete repro constructed this
session; flagged as an open test-coverage gap, not a confirmed defect.

---

## 4. Test suite status (this session, `.venv`, pinned env vars)

```
tests/test_eye_visibility.py                              5 passed
tests/test_parsing_fallback.py + test_parsing_feather.py   (in the 140 total)
tests/test_engine.py                                       (in the 140 total)
  -> all four directly-touched files:                     140 passed

tests/test_golden_pipeline_face.py                          5 passed, 3 failed
  -> attributed to the B6 catchlight fix, see §1; not a new-code defect,
     but an un-re-blessed golden snapshot.

Full suite minus test_golden_pipeline_face.py and TestWithRealImage
(pre-existing MediaPipe/macOS-runtime failures per review §8, unrelated
to this diff):
  4527 passed, 11 skipped, 4 deselected, 0 unexpected failures
```

No full unfiltered `pytest tests/` run was made, per CLAUDE.md ("no full
pytest unprompted") — the filtered run above covers everything except the
two categories already known-and-attributed by the prior review.

---

## 5. What should improve — ranked

1. **The occlusion-detection logic itself is still the old, disproven
   heuristic.** This is the highest-priority gap: the review measured it
   suppressing 78% of real eyes (mostly wide open) while missing its actual
   target case (a closed eye at EAR 0.057 read as fully open). Shipping the
   P0/P1/B5/B6 fixes without replacing `_should_gate_eye`'s evidence means the
   feature now runs with *more* reach (hoisted to every consumer via
   `_process_face_core`) while still being wrong most of the time it fires.
   Do the EAR-primary / tone-adaptive-contrast-secondary rewrite the review
   specified (§8a step 3) before this reaches more call sites or ships.

2. **Re-bless the 3 golden face-path snapshots**, but only after viewing an
   actual before/after render on a real image (not just the hash), since this
   audit only bisected *which* diff caused the change, not whether the new
   pixels are correct. This is a direct instance of this project's own
   "delta metrics hide face damage" rule — a 30px catchlight shift is small
   enough that a genuine regression could hide inside what looks like an
   expected update.

3. **Merge the mutation-verified test suite out of `/tmp`.** Both files
   (`/tmp/test_eye_visibility_proposed.py`, `/tmp/conftest.py`) still exist
   but are one `/tmp` cleanup away from being lost, and the committed suite —
   while no longer literally vacuous (§2) — still only proves the gate's
   trivial branches, not the ones the real-corpus studies proved wrong.

4. **Run the calibration study.** `RESEARCH_EYE_OCCLUSION_2026_08_26.md` is a
   protocol with zero measurements. Its own §0 premises 3 and 4 are false
   against the current tree (item 1 above) — update the doc's status line so
   the next session doesn't read it as reflecting shipped code.

5. **Memoize the gate on `FaceRegions`** (parse-time, one new `__slots__`
   entry). The gate now runs at up to 3 call sites per face
   (`_process_face_core`, `EyeEnhancer.enhance` ×2) — the B5 fix was the
   right call, but it increased call count without addressing the review's
   §4 perf finding (148.8 ms worst case per call, dominated by the
   `_as_mask` normalization running before the cheap early-out). This is now
   a slightly worse perf picture than what §4 measured, not a better one.

6. **Construct the both-eyes-gated sharpen-fallback case** (§3 above) as a
   regression test. Downgraded from urgent to a coverage gap, but cheap to
   close and currently unverified either way.

7. **Minor:** attach evidence for or reword the BiSeNet CoreML-failure
   comment in `parsing.py` (review §6.1) — cosmetic, not blocking.

---

## 6. What NOT to worry about

- P0, P1, B5, B6 are genuinely fixed, tested, and match the review's
  recommended fixes almost verbatim. No rework needed there.
- The broad-suite run (4527 passed) shows no new regressions outside the
  golden face-path hashes already attributed in §1.
- Concurrency/aliasing (review §5) and conventions/fairness (review §7) were
  already clean per the prior review and nothing in this diff touches those
  code paths.

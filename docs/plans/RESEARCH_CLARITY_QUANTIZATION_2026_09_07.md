# Research Results — Clarity Dispatch Bug + LAB Quantization Artifact (2026-09-07)

**Runtime:** `.venv/bin/python` (mediapipe 0.10.5, cv2 4.11, `RETOUCH_GPU=0`,
`RETOUCH_MEDIAPIPE_BACKEND=legacy`). Bare `python3` (cv2 4.13) produces different
golden hashes and no detection — never use it for studies.
**Trigger:** a new recipe (`cosplay_kitsune_daylight_v1`, built for the
arisaff47/arisaedited/arisa-original kitsune shoot) with `clarity: 8.0` /
`contrast: 6.0` produced visible grain on sky/background regions absent from
the source. User asked for a stage-by-stage trace rather than a param
back-off.
**Shipped:** call-site fix in `engine.py::_stage_global` + float-native LAB
fix in `grading.py::_add_clarity`/`_F_add_clarity` (uncommitted at time of
writing — see git status). Regression tests added in
`tests/test_float_pipeline.py::TestFAddClarity` and
`tests/test_engine_stages.py::TestStageGlobalClarityDispatch`.

## 0. Summary

Two independent, stacked defects, found by tracing rather than assumed from
the first plausible mechanism:

1. **Call-site dispatch bug (the dominant, production-triggering defect).**
   `_stage_global`'s `if ctx.clarity:` branch called
   `self._grader._add_clarity` (uint8-only contract) unconditionally — every
   sibling op in that function (`contrast`, `brightness`,
   `highlights/shadows/whites/blacks`) branches on `is_float`, but `clarity`
   never did. Since `_stage_global`'s documented contract is "accepts uint8
   or float32 [0,1]," and the pipeline calls it with float32 [0,1] on the
   documented path, `clarity` was **never** running its intended float
   implementation (`_F_add_clarity`) in that call path. `cv2.cvtColor` does
   not raise on a float array outside its expected range, so this degraded
   silently instead of crashing — a real 24MP production render came back
   with mean pixel value ≈1.5/255 (effectively black) before this was
   traced.

2. **LAB quantization artifact inside `_add_clarity` itself (real, but
   smaller than #1).** Both `_add_clarity` (uint8) and `_F_add_clarity`
   (nominally float, but implemented via `_bgr_f_to_lab_u8_conv`/
   `_lab_u8_conv_to_bgr_f`, i.e. also uint8-quantized internally)
   round-tripped through 8-bit LAB. On a flat, low-texture region this
   round-trip alone injects high-frequency energy that isn't in the source
   — confirmed with a mathematical no-op call (`strength=1e-9`) reproducing
   the same inflated HF energy as small nonzero strengths.

Both were fixed. Root cause of the grain the user reported was **#1**, not
"clarity amplifying sensor noise" as first hypothesized — that hypothesis
was tested and falsified with direct measurement (see §3).

## 1. How the bug was found (and how the first hypothesis was wrong)

Initial instrumentation (Laplacian-variance high-frequency energy on a flat
sky crop, `arisaedited/DSCF1884.jpg`) measured:

| condition | HF energy |
|---|---|
| source | 31.7 |
| `_add_clarity(sky_uint8, strength=1e-9)` (math no-op) | 34.8 |
| `_add_clarity(sky_uint8, strength=0.04)` (recipe's actual value) | 90.0 |

The 31.7→34.8 floor at a mathematical no-op already proved the quantization
mechanism (#2) — a bare `cv2.cvtColor(BGR2LAB)` → `cv2.cvtColor(LAB2BGR)`
round trip alone reproduces the same 34.8. **But this measurement entered
through the wrong path**: it called `_add_clarity` with a **uint8** crop
directly, bypassing the engine entirely. Fixing only `grading.py` and
re-running the full pipeline produced a **near-black image** (mean
≈1.5/255) — a new, unrelated failure that falsified the assumption that
`grading.py` alone was the fix surface.

Re-measuring through the *actual* production entry point (float32 [0,1]
input, engine call site) gave the real number:

| condition | HF energy |
|---|---|
| source | 31.7 |
| old `_add_clarity` given **float32 [0,1]** input, `strength=1e-9` (the actual buggy production path) | **65.6** |

Almost double the uint8-path measurement — because the old function's
internal `l_chan / 255.0` normalization is wrong when the input LAB L
channel from a float `cv2.cvtColor` call is already `[0,100]`, not `[0,255]`,
compounding with the uint8-quantization defect. **The number that belongs in
any future citation of this incident is 65.6, not 34.8** — the smaller
number describes a code path that was never actually exercised in
production.

## 2. Fixes shipped

**`engine.py::_stage_global`** (around line 4102) — added the missing
`is_float` branch, matching every other op in the function:

```python
if ctx.clarity:
    if is_float:
        result = self._grader._F_add_clarity(result, ctx.clarity / 100.0)
    else:
        result = self._grader._add_clarity(result, ctx.clarity / 100.0)
```

**`grading.py::_add_clarity` and `_F_add_clarity`** — both now use the
existing float-native `bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32` helpers
(already used elsewhere in the codebase, e.g. the eye-visibility gate)
instead of a uint8-quantized LAB round trip. The amplification math itself
(`detail * (1.0 + strength)`) is unchanged — only the two truncating
conversion points were removed. Verified this does not suppress genuine
detail: a synthetic mid-range checkerboard (real step-edges) gains far more
absolute HF energy under clarity than a smooth gradient does, at the same
strength (see the new `test_low_texture_region_not_disproportionately_amplified`
test).

Post-fix, through the correct float32 production entry point:

| strength | flat-sky HF energy (source 31.7) |
|---|---|
| 1e-9 | 31.81 |
| 0.04 (recipe's actual value) | 31.81 |
| 0.10 | 31.84 |
| 0.50 | 70.23 |
| 1.00 | 121.67 |

Flat regions no longer gain energy at low/realistic strengths; genuine
amplification at high strengths is present and expected (clarity/local-
contrast tools legitimately amplify any residual, including noise, at
extreme settings — that is correctly-characterized behavior, not a defect,
once the artifact floor is removed).

## 3. Discriminators that mattered (methodology worth reusing)

- **Isolate before fixing.** The first "clarity amplifies noise" framing
  (from the recipe author, before tracing) was never directly measured
  against a no-amplification control. `_add_clarity(crop, 1e-9)` (a
  strength that is mathematically a no-op) vs. a bare LAB round-trip with
  no clarity call at all, run side by side, is what actually separated
  "quantization artifact" from "real multiplicative amplification of
  existing noise."
- **A fixed function that breaks the full pipeline is a call-site bug, not
  a fix bug.** When the `grading.py`-only fix produced a black image, the
  instinct to "revert and try something else" would have been wrong — the
  isolated function was correct (confirmed by re-running the file's own
  existing `test_parity_with_uint8` test, which passed). The break was
  entirely in how the caller decided which function to invoke.
- **Ratio comparisons between fixtures with very different baselines
  invert.** An early version of the regression test compared HF-energy
  *ratios* between a smooth gradient (near-zero baseline) and a real-edge
  checkerboard; the gradient's tiny absolute change produced a *larger*
  ratio than the checkerboard's much larger absolute change, making the
  assertion backwards. Switched to absolute deltas, which is what the
  acceptance criterion (`RESEARCH_CLARITY_QUANTIZATION`'s stated goal:
  "should not substantially increase high-frequency energy where the
  source contains no corresponding detail") actually asks for.
- **A pure black/white synthetic edge fixture is a false negative.** The
  clarity op clips to `[0, 255]`; a step edge already at the clip boundary
  has no headroom to amplify into, so it silently reproduces byte-identical
  output regardless of strength. The edge fixture needs mid-range values
  (used {0.3, 0.7} here) to actually exercise the amplification math.

## 4. What to research next (per owner request, open items)

1. **Audit every `_stage_*` method for the same missing-`is_float`-branch
   pattern.** This bug shipped because one op out of five in a single
   function skipped the dtype branch every sibling op had; the codebase has
   no structural guard (type checking, a shared dispatch helper, a lint
   rule) that would catch a new op added the same way. Grep candidates:
   any `if ctx.<param>:` inside a `_stage_*` method not immediately
   followed by an `is_float` check.
2. **Grep for remaining uint8 LAB round-trips.** `_bgr_f_to_lab_u8_conv`/
   `_lab_u8_conv_to_bgr_f` (both still defined in `grading.py`, still used
   by other ops — e.g. line ~421's skin warmth adjustment) have the same
   quantization-on-flat-regions property as the pre-fix `_add_clarity`.
   Not all of them will be visually significant (depends on whether the op
   commonly runs on flat/background regions vs. only inside a face mask),
   but each is a candidate for the same fix pattern
   (`bgr_f32_to_lab_f32`/`lab_f32_to_bgr_f32` swap).
3. **Golden-test coverage gap.** None of `natural`, `porcelain_unified_v1`,
   `outdoor_harsh_sun_v1` (the three parametrized golden-pipeline recipes)
   set nonzero `clarity`. Every clarity-setting recipe
   (`outdoor_overcast_v1`, `cosplay_character_showcase_v1`,
   `cosplay_kitsune_daylight_v1`, and others) is completely untested by the
   golden-hash suite. Recommend adding at least one clarity-nonzero recipe
   to the golden parametrization so a future regression in this code path
   fails a snapshot test instead of requiring a manual render to notice.
4. **`CURATED_RECIPE_NAMES` policy-vs-code mismatch (separate, smaller
   finding, noticed in passing).** `fa02_texture_experimental_v1`'s own
   code comment says "Do NOT add to CURATED_RECIPE_NAMES" but it is
   present in that list (`retouch/recipes.py`). Worth a quick audit of
   whether other recipes' inline policy comments match their actual
   registration — this is the kind of drift that's cheap to check and easy
   to miss.

## 5. Corpus / verification notes

- Test image: `arisaedited/DSCF1884.jpg` (5850×3879, kitsune cosplay,
  outdoor overcast daylight, busy convention background) — chosen because
  it was the working example for the `cosplay_kitsune_daylight_v1` recipe
  being built when the artifact was found, not because it's part of any
  standing corpus.
- Full targeted regression suite green after both fixes:
  `test_grading.py`, `test_integration_pipeline.py`,
  `test_grading_internal.py`, `test_float_pipeline.py`,
  `test_engine_stages.py`, `test_recipe_integration.py`, `test_recipes.py`,
  `test_golden_pipeline.py`, `test_golden_pipeline_face.py` — 496 passed.
  Golden-pipeline hashes did not move (expected — see gap #3 above, no
  golden recipe exercises this path).
- `tests/test_recipe_integration.py::test_curated_catalog_is_partitioned_by_safety`
  needed its hardcoded `len(CURATED_RECIPE_NAMES)` bumped 63→65 for the two
  new recipes added this session (`cosplay_color_ref_v1`,
  `cosplay_kitsune_daylight_v1`) — unrelated to the clarity bug, caught by
  running the adjacent test file per the `pre-commit-grep-adjacent-tests`
  lesson from a prior incident.

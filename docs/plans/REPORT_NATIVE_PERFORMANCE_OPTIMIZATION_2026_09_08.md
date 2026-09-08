# Native-resolution performance optimization report

Date: 2026-09-08 (Asia/Kuala_Lumpur)

Scope: implementation of the accepted native-resolution audit optimizations
from `PERF_AUDIT_NATIVE_2026_09_07.md` and
`PERF_AUDIT_ASTRA_PACKET_2026_09_07.md`.

This was an engineering optimization pass only.  No new algorithm research,
recipe migration, threshold change, P1–P6 semantic change, production API
change, or P7/P8 work was performed.  All changes remain unstaged.

## Outcome

The four requested single-render optimizations are implemented and preserve
the existing output contracts in focused exact comparisons.  The profiled
native render fell from 323.56 s in the audit profile to 164.49 s in the
optimized profile (49.16% lower cProfile wall time).  The existing five-image,
five-recipe batch fell from 2,596.612 s to 1,382.092 s (46.77% lower wall
time, 1.879x throughput).

The timings are not a promise for every machine: native OpenCV workloads vary
with face size, cache state, thermal state, and recipe.  The exact-output
checks are the acceptance gate; the speedup is measured evidence on the
existing corpus.

## Files changed

Production helpers:

* `retouch/marks.py` — hoist per-detection LAB/reference statistics and map
  statistics out of the per-component `_relative_features` loop.  The
  original private-call signature remains valid through optional keyword-only
  context arguments.
* `retouch/freckle.py` — evaluate each connected component from its
  `connectedComponentsWithStats` bounding-box slice rather than scanning the
  full canvas for `labels == i`.
* `retouch/undereye.py` — run the existing elliptical morphology, distance
  transform, feathering, and ring construction in a padded ROI and restore the
  historical full-frame `(support, ring, valid)` contract.
* `retouch/hairwork.py` — retain the original flyaway implementation as
  `_flyaway_mask_full`, add a padded ROI wrapper, and bbox-slice the internal
  connected-component cleanup.  Elliptical kernels are unchanged.

Documentation and tests:

* `CLAUDE.md` — corrected the stale native `quality="full"` runtime note using
  the current measured corpus; the note now records the observed 27–325 s
  range and 5.64 GB audit peak with an explicit measurement caveat.
* `tests/test_performance_optimizations.py` — exactness tests for freckle
  component slices, hoisted mark context, under-eye ROI geometry (mid-frame
  and edge-clamped), and flyaway ROI/full-frame equivalence.

The two audit reports already present in the worktree were not edited:

* `docs/plans/PERF_AUDIT_NATIVE_2026_09_07.md`
* `docs/plans/PERF_AUDIT_ASTRA_PACKET_2026_09_07.md`

No production recipe or configuration file was changed.

## Optimizations and correctness evidence

### 1. `_relative_features` invariant hoist

`detect_marks` now computes the LAB crop, face-reference pixel selection,
L/a median and MAD, and optional mel/hemoglobin reference statistics once per
detection pass.  Each component still uses the same median, MAD, component
crop, geometry, thresholds, and arithmetic as before.  The fallback path in
`_relative_features` preserves compatibility for callers that do not supply a
context.

Evidence:

* synthetic pre-encode records/features are exactly equal to the captured
  baseline;
* a real 832×1248 crop containing 2,000 components produced 2,000/2,000
  matching records and exact feature dictionaries against the clean-HEAD
  implementation;
* the focused regression test passes.

### 2. Bounding-box component processing

Freckle classification and the related under-eye/flyaway cleanup now use each
component's existing `CC_STAT_LEFT/TOP/WIDTH/HEIGHT` rectangle.  The label
comparison is performed only on that slice, preserving the component's exact
row-major membership.  Coordinates and component ordering remain in the
original full-frame coordinate system; caps and filtering criteria are
unchanged.

Evidence:

* direct old/full versus sliced `_classify_anomaly` results are exactly equal;
* the frozen synthetic component records and all relevant focused tests pass;
* no cap, threshold, or class mapping changed.

### 3. Under-eye ROI morphology

`build_undereye_support` computes a clamped ROI around hard under-eye pixels
and exclusions.  The pad is derived from the maximum of the extend, side,
lash, eye-exclusion, feather/halo, and ring radii, with the distance-transform
support covered by `3 * ext_px`.  The same elliptical kernels and operation
order run in the crop.  Support and ring are zero outside the ROI and `valid`
is restored to true outside the ROI, matching the previous full-frame
contract.

Evidence:

* old/full versus ROI outputs are `np.array_equal` for mid-frame and
  edge-clamped masks, with and without exclusions/skin masks and across tested
  IED values;
* the saved pre-edit support/ring/valid arrays are exactly reproduced;
* the focused ROI test passes.

### 4. Flyaway ROI morphology

The expensive flyaway detector now crops around the positive hair/exclusion
support with a pad covering the large silhouette morphology, cap/black-hat
radius, output feather radius, and dominant-flow Gaussian support.  It then
maps the result back into a full-frame float32 mask.  The original
full-canvas implementation remains available internally as
`_flyaway_mask_full`; elliptical kernel shape and all detection thresholds are
unchanged.

Evidence:

* wrapper output is exactly equal to `_flyaway_mask_full` on deterministic
  random and structured masks, including exclusion input;
* saved pre-edit flyaway masks are exactly reproduced;
* focused flyaway and hairwork suites pass.

No rectangle substitution, separable approximation, Gaussian approximation,
face-context cache, or F8.2 bypass was attempted.  These were intentionally
excluded because they would change semantics or weaken an accepted safeguard.

## Native single-render benchmark

Input: `DSCF2650.jpg` (6240×4160), recipe
`studio_headshot_protected_v1`, legacy backend, GPU disabled.  Baseline is the
audit cProfile run; optimized values are a fresh cProfile run with the four
changes above.

| Site | Audit baseline (s) | Optimized (s) | Reduction |
|---|---:|---:|---:|
| `detect_marks` | 95.290 | 2.303 | 97.58% |
| `_relative_features` | 62.716 | 1.134 | 98.19% |
| `classify_anomalies` | 32.169 | 0.565 | 98.24% |
| `build_undereye_support` | 37.948 | 3.435 | 90.95% |
| `remove_flyaways` | 63.031 | 19.661 | 68.81% |
| `dilate` (all callers) | 82.206 | 19.380 | 76.43% |
| `GaussianBlur` (all callers) | 53.520 | 51.067 | 4.58% |
| cProfile total | 323.560 | 164.488 | **49.16%** |

The optimized direct engine run with stage timings took 175.806 s wall time
(the cProfile run took 164.488 s).  A repeated optimized run produced the
same SHA-256 output hash, `7e01ad0acd921e05e9d4f827c9625c4f34a58b67214d6b79f65dd67962c2c656`.
The audit profile is the comparable baseline; a separate clean-HEAD run was
much slower (660 s) because of normal native workload variability and is not
used as a speed comparison.

### Peak RSS

* audit cProfile peak: 5.64 GB;
* optimized RSS-profile peak: 5.39 GB.

This is a 4.43% lower observed high-water mark, not a guaranteed cap.  The
measurements were made on separate runs, so the result is reported as “no
observed RSS regression,” not as a statistically controlled memory benchmark.

## Five-image/five-recipe batch

Corpus and recipes are the existing delivery corpus, not a new split:

* `/Users/dennis/Desktop/Priority for Printing`
* 5 native-resolution JPEGs × 5 existing recipes
  (`heirloom_archive_v1`, `documentary_preserve_v1`, `tired_eye_rescue_v2`,
  `cinema_grade_v1`, `studio_headshot_protected_v1`)

Optimized output and manifest:

* `/Users/dennis/Desktop/Priority for Printing_retouched_perfopt_20260908/`
* manifest SHA-256 prefix: `0a1f339da...`

| Recipe | Baseline total (s) | Optimized total (s) | Reduction |
|---|---:|---:|---:|
| heirloom | 443.574 | 146.025 | 67.08% |
| documentary | 655.297 | 275.900 | 57.90% |
| tired-eye | 407.821 | 326.775 | 19.87% |
| cinema | 235.116 | 208.995 | 11.11% |
| studio | 854.804 | 424.397 | 50.35% |
| **all 25 renders** | **2,596.612** | **1,382.092** | **46.77%** |

All 25 optimized images validated as 6240×4160 uint8 finite in-range arrays.
For the prior-versus-optimized output comparison, heirloom, documentary,
tired-eye, and studio were byte-identical for all 10 checked files per recipe
(retouched output plus compare output).  Cinema differed because that existing
recipe contains an unseeded grain/random path and its real-image detector
context can vary between runs.  With a frozen face context and
`np.random.seed(123)`, clean-HEAD and optimized cinema outputs were exact.
This is a nondeterminism caveat in the existing recipe path, not evidence of a
pixel change from the optimization helpers.

## Bounded two-worker benchmark

No new parallel implementation or default was added.  The existing
`BatchProcessor.process_folder` explicit worker argument was benchmarked with
`num_workers=1` and `num_workers=2` on two native images using isolated worker
engine state.

| Explicit workers | Wall time (s) | Parent RSS | Successful renders |
|---:|---:|---:|---:|
| 1 | 233.672 | 5.608 GB | 2/2 |
| 2 | 176.361 | 5.645 GB | 2/2 |

Two workers were 24.53% faster (1.325× throughput) on this small benchmark,
with no decoded-pixel difference (`max=0`, MAE/RMSE=0, no pixels >1 DN).
JPEG byte streams differed in worker order/metadata, so decoded arrays—not
file bytes—are the relevant comparison for this existing batch API.  The
default worker count and deterministic naming/manifests were not changed.
Four workers remain outside the accepted memory ceiling and were not run.

## Tests and validation

Focused optimization and affected-module suite:

```text
141 passed in 1.17s
```

Expanded P1–P6/engine/golden/recipe suite:

```text
431 passed, 14 warnings in 19.69s
```

Command used for the expanded suite:

```bash
PYTHONDONTWRITEBYTECODE=1 RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy \
  .venv/bin/python -m pytest -q \
  tests/test_performance_optimizations.py tests/test_freckle.py tests/test_marks.py \
  tests/test_undereye.py tests/test_flyaway_removal.py tests/test_hairwork.py \
  tests/test_harmony.py tests/test_guided_filter.py tests/test_stages.py \
  tests/test_engine_stages.py tests/test_golden_pipeline_face.py tests/test_engine.py \
  tests/test_recipe_integration.py -p no:cacheprovider
```

Python compilation and whitespace checks passed:

```text
py_compile: exit 0
git diff --check: exit 0
```

The full repository suite was also run with an isolated writable cache:

```text
5264 passed, 11 skipped, 5 failed, 62 warnings in 271.71s (0:04:31)
```

The five failures are classified below.  None is caused by the four
optimizations; the affected scoped suite is green.

| Failure | Classification | Evidence |
|---|---|---|
| `tests/test_batch_workflow.py::test_cache_and_grouping` | environment/config | The test expects the cache parent directory name `retouch`; the full run intentionally used isolated cache directory `retouch_fullsuite_perfopt_20260908`. Writable-cache targeted tests pass (`213 passed`). |
| `tests/test_documentation_coverage.py::test_parameter_registry_is_covered_by_public_references` | pre-existing documentation gap | `fa02_texture_mode` is absent from `docs/architecture/API.md` at clean HEAD; no documentation-coverage source or parameter registry was changed here. |
| `tests/test_fa02_production_eligibility.py::TestOptInOnly::test_every_recipe_resolves_to_legacy` | pre-existing fixture/eligibility expectation | The baseline `fa02_texture_experimental_v1` recipe exists at clean HEAD; P4/P5/P6 optimization files do not alter recipe eligibility. |
| `tests/test_fa02_production_eligibility.py::TestOptInOnly::test_no_recipe_source_file_sets_the_key` | pre-existing fixture/eligibility expectation | Same clean-HEAD experimental recipe is present; no recipe/config file was changed. |
| `tests/test_guided_filter.py::TestGuidedFilterVsBilateral::test_guided_and_bilateral_performance_comparison` | timing/environment dependent | The unchanged bilateral comparison exceeds its 10 s threshold on this host (about 22.7 s for the 1500×1500 case); guided filter remains about 20.9 ms. The test/source is outside the changed helpers and was also failing in the prior full run. |

No failures were left unclassified as unknown, and no unrelated failure was
fixed in this diff.

## Remaining bottlenecks

After the optimizations, the largest single profile site is the distributed
`GaussianBlur` total (51.1 s).  Other remaining costs include chromophore
variance reduction (about 28.2 s), skin wrinkle/detail processing (about
16.5 s), and QA evidence analysis (about 12.1 s).  These were not changed:
the audit did not establish byte-preserving replacements, and the request
explicitly excludes new algorithm research or quality-semantic changes.

## Compatibility and release decision

* P1 semantics: preserved; existing guided-filter implementation untouched.
* P2 guards/tooling: preserved; no new allocation strategy or memory policy.
* P3 stage registry/order: preserved.
* P4 bounded attenuation: untouched.
* P5 analytical operators: untouched.
* P6 legacy clarity and explicit opt-in behavior: untouched.
* F8.2 proxy/native face-context correctness guard: preserved.
* Defaults: **unchanged**.
* Recipes: **unchanged**.
* Thresholds and component caps: **unchanged**.
* Production API/serialization: **unchanged**.
* Elliptical morphology shape: **unchanged**.

Decision: **GO for the four accepted byte-preserving single-render
optimizations and their regression coverage.**  The measured batch result is
materially faster with no observed RSS regression, and exact helper/frozen
context evidence is green.  The existing explicit two-worker path is
promising, but remains a benchmark-only result: no default worker change or
new parallel scheduler is shipped.  No further optimization should be
promoted without a new scoped audit and native exact-output evidence.


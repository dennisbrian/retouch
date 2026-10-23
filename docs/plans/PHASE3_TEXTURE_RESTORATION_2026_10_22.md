# Phase 3: texture restoration — first safety increment

Date: 2026-10-22. Base: main adc10febdc3983b138dd0a5469a8f23afd7ae06e.
Status: experimental safety improvement implemented; quality qualification pending.

## Delivery before this phase

The natural_v2 commit was rebased onto the concurrent origin/main work, reconciled,
and pushed as adc10fe. The local/remote main hashes matched after the push.
The repository pre-push gate passed governance and 428 engine/integration/tooling
tests. The original 9434cff hash was replaced by rebase, not pushed separately.

Concurrent main already supplied Phase 2: Protect/Restore brushes, per-face
strength and skip captions. Focused validation covered protection on real public
portraits with planted marks, native/draft/fast/multi-face paths, face strength,
caption generation, GUI helpers and parameter wiring. The run returned 195 passes
and seven unrelated style-library fixture permission errors; rerunning the three
affected GUI classes with RETOUCH_CACHE_DIR in /tmp returned 14 passes (including
all seven affected tests). No browser interaction was performed in this phase.
See ../guides/PROTECT_AND_FACE_STRENGTH.md for the UI workflow and its scope:
global grade/finish still apply to protected detail by design.

## Current evidence, refreshed

Both existing readiness tools were run against the ten existing FA-02 pilot
directories and the existing frozen scoring lock. Source manifests, splits,
annotations and locks were not changed.

| Evidence | Observed |
|---|---|
| Assets / people | 10 patches / 8 recorded people |
| Splits | 8 dev people; 0 calibration; 0 locked_test |
| Reviewed pores | 0 regions |
| Reviewed fine hair | 1 region |
| Reviewed noise | 1 region |
| Reviewed corrected defects / identity marks | 0 / 0 regions |
| Frozen comparison eligibility | All 10 abstain |
| Candidate versus both baselines | Not computable for DoG or multiscale |

A region label alone does not populate a measured pore/profile target. The frozen
scorer reports pore recovery, fine-hair recovery, noise amplification and defect
reintroduction as blocked for all 60 case-arms. Makeup edges, halos and broad tone
have some computable observations; these cannot stand in for missing texture
recovery or noise/defect safety measurements. No winner is claimed.

Commands (read-only apart from the named output files):

```bash
.venv/bin/python scripts/qa/fa02_readiness_report.py test_output/fa02_pilot_* \
  --json /tmp/retouch-phase3-texture/readiness.json
.venv/bin/python scripts/qa/fa02_scoring_readiness_report.py \
  test_output/fa02_scoring_lock.json test_output/fa02_pilot_* \
  --json /tmp/retouch-phase3-texture/scoring.json
```

The offline frozen gate and the live heuristic gate are different contracts.
The offline result does not imply every live face must abstain. Live thresholds
remain explicitly uncalibrated; none was relaxed to make a candidate run.

## Reproduced problem and implemented change

The opt-in DoG/multiscale path previously computed its residual from pre-smoothing
pixels minus the canvas at restoration time. Between those points, nose shading,
under-eye smoothing, freckle healing, exposure locking and flattening can run.
That residual therefore includes changes other than frequency smoothing and can
reintroduce removed freckle detail.

A discriminating pipeline test injects a known later heal, observes the candidate
extractor's actual input, and compares it with the recorded frequency-smoothing
output. Both candidate modes failed that test before the change.

The new path:

1. Captures the smoothing-only residual immediately after frequency combination.
2. Records the actual changed footprint of intervening freckle healing, including
   feathered pixels rather than only a proposed defect center.
3. Removes that footprint and a surrounding filter-support ring from restoration
   eligibility. The square guard uses the exact largest active Gaussian radius,
   ceil(3*sigma), including diagonal kernel support. No new tuning threshold.
4. Reuses the existing eligibility gate and existing frozen DoG/multiscale arms.
5. Records residual_reference, corrected_pixels_excluded and
   correction_guard_radius_px in per-face FA-02 diagnostics.

This is a destination-support guard, not a new mark classifier or denoiser. It
prevents this candidate's additive restoration in the measured healed footprint
and its filter ring; it does not certify correct healing elsewhere. An unknown
filter scale yields no candidate support and abstains.

Legacy restoration and the raw-residual baseline retain their existing behavior.
No default mode, recipe recommendation, eligibility threshold, gain, or frozen
research arm was changed. The live candidate now has a different snapshot/support
from the historic raw baseline, so live images must not be advertised as a
matched-support ranking of representations. Default/legacy golden hashes are
checked without refreshing snapshots.

## Validation and remaining work

The final focused run returned **100 passes** across boundary, eligibility,
face-path golden snapshots, worker parity and face-strength tests. This includes
14 boundary tests. Both candidate ordering probes failed before the implementation
change and pass afterward. No golden snapshots were refreshed.

```bash
RETOUCH_CACHE_DIR=/tmp/retouch-phase3-cache MPLCONFIGDIR=/tmp/retouch-mpl \
  .venv/bin/python -m pytest tests/test_fa02_restoration_boundary.py \
  tests/test_fa02_production_eligibility.py tests/test_golden_pipeline_face.py \
  tests/test_worker_processor_parity.py tests/test_face_strength.py -q
```

Governance fast checks and git diff --check also passed. These source changes are
local and uncommitted; adc10fe is the pushed natural_v2 integration commit.

The native development comparison uses the October 17 original DSCF5119 camera
JPEG, 4160 x 6240 after EXIF orientation, natural_v2, and modes off/legacy/DoG/
multiscale. Face contexts from the first render are held fixed across arms.
This provides real pipeline outputs and eligibility evidence, not reviewed pore
or fine-hair ground truth. The sample's two overlapping detections are a known
limitation under both the baseline and integrated main. It is one photographed
subject, not two independent evaluation subjects. Concurrent checks mean observed
render times are not a controlled performance benchmark.

Next qualification gate: populate reviewed native pore/hair scoring inputs and
paired noise/corrected-defect controls, then compare on identical accepted
supports under the frozen contract. Reserve untouched people for calibration and
locked_test before tuning. Defaults cannot be promoted from synthetic controls,
unlabelled changes, successful renders, or this single development photo.

### Native comparison result

All four outputs completed and the matching face comparison sheet was viewed.
DoG and multiscale both abstained at the existing provisional texture floor:
measured high-pass standard deviations were 2.727 and 2.794 for the two overlapping
face contexts, versus 3.5. Their encoded JPEGs are byte-identical to restoration
off (SHA-256 1fe83e8caddd62bcebc8e40697e21944e193bb779649fe00320972e1a5524a2e).
Legacy differs, as expected because legacy restoration remains active there.
No useful texture recovery was demonstrated on this photo; the new experimental
composite did not execute. Tests that exercise it explicitly relax eligibility
only inside the test process, not in production or the real render.

Observed times: off 34.88 s, legacy 33.85 s, DoG 32.48 s, multiscale 38.54 s.
These overlap other local checks and are not a benchmark.

Temporary review artifacts:
- `/tmp/retouch-phase3-texture/renders.json` — per-mode execution and eligibility
- `/tmp/retouch-phase3-texture/DSCF5119_face1_modes.jpg` — original/off/legacy/DoG/multiscale
- `/tmp/retouch-phase3-texture/readiness.json` and `scoring.json` — refreshed data gaps
- `/tmp/retouch-phase3-texture/tests-final.log` — 100-pass focused run

The guard now has deterministic regression evidence. The larger Phase 3 quality
goal remains open until the reviewed native scoring inputs and qualification
protocol above are populated; this report does not call it production-ready.

## Next increment — native scoring review packet (2026-10-23)

A reusable selector now packages all ten existing accepted development patches
into one self-contained HTML page. No manifest, split, support mask, scoring lock
or accepted annotation is rewritten. The prior safety changes remain local.

```bash
.venv/bin/python scripts/qa/fa02_review_packet.py test_output/fa02_pilot_* \
  --html test_output/phase3_texture_review_20261023/review.html
```

The page shows saved X and S side by side at native crop coordinates, with integer
zoom, optional mask overlays, and read-only existing region rectangles. Display
PNGs are rounded 8-bit projections; original floating-point arrays remain the
scoring authority. Source/array checksums are verified before packaging. Locked
test cases are refused, so annotation work cannot silently consume a holdout.

A reviewer can select pore centers/radii, hair cross-sections, noise rectangles,
corrected defects, protected details and uncertainty regions. Selections stay
separate from accepted labels. Downloads require a reviewer name and are marked
`draft_not_accepted`, with source/array hashes, split and native crop coordinates.
Pending two-point selections must be completed or cancelled before export.
Switching cases preserves drafts in page memory; export before closing the page.

Validate an exported draft without importing or approving it:

```bash
.venv/bin/python scripts/qa/fa02_review_packet.py test_output/fa02_pilot_* \
  --draft /path/to/texture-scoring-draft.json
```

The validator rejects stale checksums, changed splits/crops, missing or duplicate
cases, invalid target geometry, unnamed reviewers and claimed approval status.
A valid draft is still not scored or approved. This tool does not turn noise
rectangles into a paired clean/noisy control, or reconstruct missing clean truth.

The existing DSCF1058 and DSCF1606 input crops were viewed. DSCF1058's hair label
covers strands crossing the cheek near the wig: review strand ownership and its
semantic category before calling it fine skin hair. DSCF1606 shows prominent
fine mottling; its existing noise label needs paired-control evidence before a
noise-amplification claim. No pore or hair label was invented from these views.

Validation: 12 focused Python checks passed. A JavaScript DOM harness exercised
the actual generated page code: ten cases, border-correct native coordinates,
reviewer requirement, unaccepted draft export, pending-point blocking, case-switch
persistence, undo and pore bounds all passed. This is behavior verification with
mocked canvas/image objects, not an interactive browser acceptance test.

Review artifact: `test_output/phase3_texture_review_20261023/review.html`.
The human scoring-input review is the remaining next step. No production mode,
threshold or restoration default changed in this increment.

A temporary headless-Chrome screenshot attempt could not complete: the automatic
permission approval review timed out. No browser visual acceptance is claimed.

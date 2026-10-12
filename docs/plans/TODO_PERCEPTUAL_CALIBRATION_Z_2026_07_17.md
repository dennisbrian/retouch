# Perceptual Calibration Z-Series Implementation Queue

**Source:** `RESEARCH_PERCEPTUAL_CALIBRATION_Z_2026_07_17.md`
**Status:** Proposed implementation queue. Do not expose target-seeking edits
until the measurement contract and corpus baselines are reviewed.

## Scope Corrections

Z1 and Z2 are strong directions, but the cited work establishes perceptual
relationships, not a universal camera-independent target for every person.
The initial implementation must therefore be relative to the subject and
capture, never infer demographic attributes, age, ethnicity, or a "correct"
skin tone.

- Do not store or select targets by inferred demographic group.
- Treat published findings as a metric design and direction-of-change source.
- Derive recipe bands from a consented, colour-managed validation corpus and
  keep every target expressed as a bounded delta from the input.
- Make all Z1/Z2 output observational at first. A metric is not permission to
  change a face automatically.

## Priority Order

1. **Z0: shared measurement contract** (prerequisite, 2-3 days)
2. **Z1: facial-contrast metric and reporting** (3-5 days)
3. **Z2: homogeneity state vector and QA reporting** (3-5 days)
4. **Z7: degraded-input preflight** (3-5 days)
5. **Z3: hair alpha-matting spike** (1-1.5 weeks)
6. **Z1/Z2 target-seeking integration** after corpus review and Y3
7. **Z4**, then **Z5**, then **Z6**

Z3 is the highest visible quality win, but Z0-Z2 are the highest leverage
calibration win: they give X2, X4, Y3, and harmony a common definition of
"improved without plasticity."

## Stage Z0: Measurement Contract

**New module:** `retouch/perceptual_metrics.py`

Build pure, deterministic metrics with no engine, recipe, GUI, or CLI wiring:

- `facial_feature_contrast(img, regions)`: feature-versus-surrounding-skin
  L*, a*, b* deltas for eyes, lips, and brows. Build annuli from the existing
  `FaceRegions` masks, subtract every feature mask, require minimum support,
  and report per-feature confidence rather than fabricating values.
- `skin_homogeneity_state(img, skin_mask, reference=None)`: normalized
  blotch-band chroma dispersion, retained pore-band energy, and optional X2
  hemoglobin variance. Scale bands by face width, not fixed pixels.
- `quality_preflight(img)`: noise and JPEG-grid evidence only. It must not
  mutate the image or decide that an image needs repair.

**QA gate:** synthetic masks at multiple resolutions give stable relative
metrics; empty/overlapping masks return low confidence; all analysis functions
are byte-identical/no mutation.

## Stage Z1: Facial Contrast Reporting

**Files:** `retouch/perceptual_metrics.py`, dedicated tests, then QA adapter.

- Report feature contrast as a within-face relative percentile against that
  face's own feature distribution, not an inferred demographic norm.
- Add named, opt-in recipe intent profiles only after baseline review:
  `natural`, `soft_portrait`, and `editorial`. They express small bounded
  deltas, not age or health claims.
- Publish per-feature confidence and skip regions with bad parsing, eye
  closure, heavy occlusion, or insufficient surrounding skin.

**Do not yet:** alter eye, lip, brow, or blush pixels. The first deliverable is
a QA panel and JSON report proving the metric agrees with visual review.

**QA gate:** changing feature contrast moves the matching metric monotonically;
re-measuring an unchanged render is exact; a fixed relative synthetic face
sweep remains stable across lightness levels and image sizes.

## Stage Z2: Homogeneity Reporting

**Files:** `retouch/perceptual_metrics.py`, `retouch/qa_detectors.py` adapter,
dedicated tests.

- Reuse the existing pore-spectrum detector as the texture leg.
- Measure chroma at the blotch band only; do not use global chroma variance
  as a substitute for skin condition.
- If X2 confidence is high, include its hemoglobin variance; otherwise report
  it as unavailable rather than using a noisy surrogate.
- Define a hard safety floor only for lost pore energy relative to the input.
  Other values remain informational until a corpus establishes ranges.

**QA gate:** chroma-only mottle changes the chroma leg but not the pore leg;
luminance pore blur triggers the pore guard but not false chroma improvement;
metrics are stable across proxy resolutions.

## Stage Z7: Input Rescue Preflight

**New module:** `retouch/input_rescue.py`

Build detection before repair:

- Estimate chroma-noise evidence in flat non-skin/skin regions.
- Detect 8-pixel JPEG grid periodicity with a confidence score.
- Add an opt-in chroma-only denoise/deblock path only after clean-image
identity, luma pore preservation, and downstream-metric stability pass.

**Why before auto-colorist:** Y3 must not optimize noise or JPEG blocking as if
they were skin blotchiness.

## Stage Z3: Alpha-Matting Spike

**New module:** `retouch/matting.py`; no broad engine replacement initially.

- Build automatic trimaps from current person/hair masks.
- Solve only the unknown band at proxy resolution, with guided full-resolution
  refinement and a strict runtime budget.
- Evaluate on synthetic strand composites first. Only replace the alpha used
  by background compositing after halo QA improves; leave statistical masks
  unchanged.

**QA gate:** alpha MSE and halo width beat the existing soft-mask baseline;
failure/low-confidence trimaps return the existing mask exactly.

## Deferred Items

- **Z4 distribution transfer:** wait for K12 delivery-boundary work and add a
  skin-locus hold plus banding QA in the same change.
- **Z5 group consistency:** wait for merged K4 plus an I-VI mixed-light group
  corpus; equalize illumination only, never skin tone.
- **Z6 pore synthesis:** wait until smoothing provenance is available; do not
  synthesize texture over surviving real pores.

## Current Task Assignments

1. Build Z0/Z1/Z2 metric core and synthetic tests.
2. Render an inspectable baseline sheet on the validation corpus before
   choosing any recipe bands.
3. Implement Z7 preflight only after the metric core establishes its input
   quality signals.
4. Run the Z3 matting spike separately, using halo QA as the acceptance gate.

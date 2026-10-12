# FA-02 Eligibility Metric Validity Research — 2026-09-06

**Status: RESEARCH ONLY. No production threshold changed. No candidate
metric wired into `retouch/`. No A2/A3 restoration comparison run.**

## Question

Following `d8e2171`'s native-resolution survey (11 faces, dev split; 1/11
crossed the 3.5 `highpass_std` threshold, and that face was the smallest and
grainiest in the corpus, independently annotated as ISO-grain-dominated): is
the production eligibility metric — `fa02_texture_eligibility.highpass_std`
at its fixed `HIGHPASS_SIGMA_PX = 2.0`, registered here as
**`legacy_highpass_std_px2`** so no experimental variant can be confused with
it — actually measuring recoverable facial micro-texture, or is it confounded
by capture scale and sensor noise?

## Corpus

n=9 single-face dev-split assets, extracted as the EXACT
`(pre_smooth_canvas, effective_support, face_width_px)` triples the
production gate measured during the native-resolution survey
(`scripts/qa/fa02_dump_faces.py`, verified byte-identical to the survey's
own `highpass_std` to full float precision). `nahida_DSCF6754`'s 2 faces are
**not** included — that image's faces are processed inside
`FaceProcessorPool`'s spawned worker subprocess, where the in-process
extraction hook does not apply. This is disclosed, not silently dropped.

Item 6 (semantic region check) additionally used a fresh `engine.process()`
call at `max_dim=2048` (not native) across all 10 dev assets (10 faces
found brow regions) to get `regions.left_eyebrow`/`right_eyebrow`, which the
gate-input dump does not carry.

No EXIF ISO/exposure/software tags were present on any of the 9 source
JPEGs (`exiftool` checked directly); every corpus row for item 2 records
`iso`/`exposure` as `"unavailable"` rather than omitting the field.

**This corpus detects metric pathology. It does not calibrate a production
gate.** Per this round's explicit constraint, no new threshold is derived
from it.

## Findings

### Item 2 — corpus correlations (n=9, first-pass signal only)

| pair | Spearman ρ |
|---|---|
| `highpass_std` vs. face width | **−0.85** |
| `highpass_std` vs. noise_sigma | +0.55 |
| `highpass_std` vs. Laplacian variance | +0.72 |

(`face_area_proxy_px2 = face_width_px ** 2` is also reported per-row for
readability, but its rank correlation with `highpass_std` is identical to
`highpass_std` vs. face width by construction — a monotone transform of the
same variable, not an independent measurement.)

Caveat stated explicitly in the report: `highpass_std`, `noise_sigma`, and
Laplacian variance all share the same fixed-pixel-scale convention, so these
correlations partly restate the metric's own definition rather than proving
an independent confound. They motivated, but do not by themselves establish,
the conclusion below — items 3 and 4 (controlled, single-variable,
same-content) carry the actual evidentiary weight.

### Item 3 — controlled rescaling of identical content (no sharpen/denoise)

Every one of the 9 faces was resized to 0.5×/0.75×/1.0×/1.5×/2.0× (both
`INTER_AREA` and `INTER_LINEAR` compared at each downscale factor; upscale
and identity use a single `INTER_LINEAR` measurement each, since the choice
of interpolator has no meaningful effect there) with
`legacy_highpass_std_px2` re-measured on the resampled crop and its
resampled mask. **Result: 4 of 9 faces flip their eligibility verdict from
pure resize alone, zero pixel content added or removed** (10 flip-events
total when both downscale interpolators are counted separately for the
faces that flip at 0.5×).

- `ff47_DSCF1058` (abstained at 1.0×, 2.606) → **eligible at 0.5×** (3.623).
- `ff47_DSCF2274` (abstained at 1.0×, 3.083) → **eligible at 0.5×** (3.714).
- `nahida_DSCF9559` (abstained at 1.0×, 3.047) → **eligible at 0.5×** (5.020).
- `ff47_DSCF1606` (eligible at 1.0×, 4.640, the known noise-confounded face)
  → **abstains at 1.5× and 2.0×** (3.058, 2.373).

The metric moves monotonically with scale in the same direction for every
face regardless of content — down on upscale, up on downscale — which is
exactly the fixed-pixel-sigma-vs-face-scale mechanism the corpus
correlations predicted. This is the discriminating, content-controlled
evidence: **`legacy_highpass_std_px2` is not scale invariant.**

### Item 4 — noise / denoise / blur, single-variable (n=9)

| treatment | effect on `highpass_std` | consistent across corpus? |
|---|---|---|
| A. mild bilateral denoise (structure-preserving) | −0.27 to −1.98 | yes, always negative |
| B. added synthetic Gaussian noise, σ=3.0 | +0.38 to +0.84 | yes, always positive |
| C. mild Gaussian blur, σ=1.5 | −0.89 to −3.36 | yes, always negative |

Directionally sane in isolation — but the disqualifying result is the
**noise-crossing search**: for **all 9/9 faces**, some added-noise σ pushes
`highpass_std` across the 3.5 threshold, and that crossing σ is *always*
below the gate's own `MAX_NOISE_SIGMA = 6.0` ceiling (range 2.55–4.59,
DSCF1606 crosses at σ=0.0 since it's already eligible). **Requirement 4's
own test is met: added noise alone can move a face across 3.5 without ever
tripping the metric's sibling noise gate.** Gate 2 (`highpass_std`) and gate
3 (`noise_sigma` ceiling) are mutually inconsistent as currently specified.

**Mechanism, not just observation:** at the B treatment (added σ=3.0),
`estimate_noise_sigma` (the gate's OWN noise measurement) under-reports the
injected noise on all 9/9 faces. If noise combined in quadrature with the
pre-existing base noise the way independent Gaussian sources should, the
measured value after treatment should track
`sqrt(base_noise² + 3.0²)` — instead every face's measured value falls
0.7–1.0 below that expectation (e.g. `ff47_DSCF1058`: base 1.728, expected
combined 3.462, measured 2.587; `priority_printing_DSCF2650`: base 1.415,
expected 3.317, measured 2.387). **This is why gate 3 fails to catch what
gate 2 reacts to: the noise estimator itself under-measures the noise the
texture metric is responding to**, not merely an unlucky gap between two
independently-reasonable thresholds.

### Item 5 — candidate comparison (research only, not tuned for pass rate)

Two candidates evaluated, both importing the frozen
`legacy_highpass_std_px2` rather than reimplementing it:

- **`face_scaled_highpass_std`**: sigma tied to face width
  (`2.0 * face_width_px / 500`, reusing FA-02's own frozen reference scale).
  Does **not** fix the confound — DSCF1606 is still the corpus maximum
  (4.72 vs. 4.63 for the next-highest), and it inflates values broadly.
- **`noise_floor_subtracted_detail`**: `sqrt(max(0, hp_var − noise_var))` at
  the face-scaled sigma. **Does** demote DSCF1606 off the top (2.76 vs. a
  new corpus max of 4.41, a different face) — but see item 6, this comes at
  a cost that disqualifies it as-is.

### Item 6 — semantic region separation (model-derived, n=10 faces)

Eyebrow (BiSeNet/landmark-derived, structural positive) vs. eroded
skin-interior away from any feature boundary (detail-poor negative), within
the same face. Run twice, because the first version had a confound worth
disclosing rather than hiding:

**First pass (raw brow mask):**

| metric | brow ranked higher than cheek |
|---|---|
| `legacy_highpass_std_px2` | 10/10 (1.0) |
| `face_scaled_highpass_std` | 10/10 (1.0) |
| `noise_floor_subtracted_detail` | 4/10 (0.4) |

The raw BiSeNet eyebrow mask is thin, so a large fraction of its pixels sit
ON its own hair↔skin segmentation boundary. A fixed 2px high-pass responds
to *any* edge, and items 3–4 already established this metric is largely an
edge/scale response — so a 1.0 separation score here is equally consistent
with "detects the mask boundary" and "detects hair microstructure." It does
not by itself establish structural sensitivity.

**Discriminating control (brow mask eroded 3×3, same boundary-free
construction as the cheek arm):**

| metric | eroded brow ranked higher than cheek |
|---|---|
| `legacy_highpass_std_px2` | **10/10 (1.0), unchanged** |
| `face_scaled_highpass_std` | 10/10 (1.0), unchanged |
| `noise_floor_subtracted_detail` | 4/10 (0.4), unchanged |

Separation survives erosion identically for all three metrics. This is the
finding that reframes the whole result: **the legacy metric correctly
separates genuine interior hair microstructure from smooth skin in every
face tested, not merely a segmentation boundary.** It has a real, proven
capture-scale confound (items 3–4), but it is not "measuring nothing" —
`noise_floor_subtracted_detail`, the candidate that best addressed the noise
confound in items 4–5, **fails this structural-sensitivity test outright
(worse than a coin flip) both with and without the boundary control**,
meaning its noise subtraction throws away real signal along with the
confound. Neither candidate is a net improvement on the evidence gathered
here.

## Conclusion

**`legacy_highpass_std_px2` is not a valid production eligibility signal as
currently specified**, on two independent, controlled, content-preserving
lines of evidence:

1. Pure rescaling of identical content flips the eligibility verdict for
   4/9 faces with no pixel content added or removed (item 3).
2. Added synthetic noise alone crosses the threshold for 9/9 faces at a
   noise level the metric's own sibling gate does not reject (item 4).

At the same time, it is **not simply noise** — it correctly ranks genuine
interior eyebrow-hair microstructure above smooth skin in 10/10 faces (item
6), confirmed with a boundary-erosion control so the result isn't just
picking up a segmentation edge, which is why the two hand-built candidates
evaluated here (`face_scaled`, `noise_floor_subtracted`) each fixed one
problem while failing another (scale-invariance without noise-robustness, or
noise-robustness at the cost of structural sensitivity).

**Per requirement 7: no replacement metric is selected.** n=9 detects
pathology; it does not license calibrating or picking a successor. Both
candidates remain `RESEARCH_ONLY` in `scripts/qa/fa02_texture_metric_candidates.py`
and must not be imported by any `retouch/*.py` module.

**Threshold and metric are unchanged: `MIN_HIGHPASS_STD = 3.5`,
`HIGHPASS_SIGMA_PX = 2.0`, both still live in
`retouch/fa02_texture_eligibility.py` exactly as before.** No production
restoration behavior changed. No A2_dog/A3_multiscale comparison was run —
that remains blocked on both an unconfounded eligibility signal and a
naturally eligible test case, neither of which exists yet.

## Abstention semantics (item 8)

Corrected and now documented directly at the branch point
(`retouch/perf_optimizations.py`, the comment above the FA-02 dispatch): an
abstaining `fa02_texture_experimental_v1` render is **not** equivalent to a
`natural` render. `natural` runs the legacy micro-texture restoration
(`ctx.micro_restore` defaults to 20); an abstaining non-legacy face skips
both the legacy and the candidate paths, so it is strictly less processed
than `natural`. Any future baseline comparison must keep three states
distinct: `natural`, `experimental-abstained`, `experimental-restoration-executed`.

## Artifacts

- `scripts/qa/fa02_dump_faces.py` — one-time extraction of gate-exact
  `(canvas, support, face_width_px)` triples via a scoped monkeypatch of
  `evaluate_face_eligibility` (production code untouched; verified
  byte-identical against `d8e2171`'s survey numbers).
- `scripts/qa/fa02_texture_metric_candidates.py` — `legacy_highpass_std_px2`
  (thin wrapper importing the frozen production function) +
  `face_scaled_highpass_std` + `noise_floor_subtracted_detail`. Both
  candidates RESEARCH_ONLY.
- `scripts/qa/fa02_metric_validity_experiment.py` — runs items 2–6, writes
  `test_output/fa02_metric_validity_report.json` (gitignored, same
  convention as the earlier eligibility survey).
- Extracted face crops live only under `/tmp/fa02_scale_noise/` — never
  copied into the repo or `~/Desktop`, and never treated as production
  input at any point in this research.

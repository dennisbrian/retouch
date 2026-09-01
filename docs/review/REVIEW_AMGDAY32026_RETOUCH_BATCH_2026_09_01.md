# AMG Day 3 2026 Retouch Batch — Visual and Delivery Review

- **Date:** 2026-09-01
- **Status:** COMPLETE — deep-research and documentation-only closeout; no engine,
  runner, source-photo, or output-photo changes were made.
- **Branch reviewed:** `feat/color-science-k9-fix-and-frontier`
- **Version used by the batch:** local commit `fafed66` (`fix(eyes): recalibrate
  eye-occlusion gate thresholds from study data`), committed before the batch was
  rendered.
- **Recipe:** `cosplay_portrait_polish_v1`
- **Source:** `/Users/dennis/Desktop/amgday32026`
- **Output:** `/Users/dennis/Desktop/amgday32026_retouched`

## Executive verdict

The 22-photo batch is visually good for normal viewing and sharing. The treatment is
restrained, skin texture remains visible, the blue contact lenses are controlled, and
closed or winking eyes are not repainted. Face-aware detection found exactly one face
in every source/output pair.

The batch should not be treated as the best archival or final-delivery render yet. The
one-off runner explicitly selected the preview-oriented `draft` engine path and used
raw OpenCV input/output. The latter removed the embedded ICC profile and auxiliary
metadata and changed every JPEG from 4:4:4 to 4:2:0 chroma subsampling. These are
delivery-runner limitations, not evidence that the face-retouch result itself failed.

One engine-level improvement candidate was confirmed: exposed tongues in the
tongue-out sequence are partly classified as lips, so the lip operation changes tongue
colour/texture. This needs a conservative anatomical guard and a small labelled corpus;
it should not be “fixed” with an uncalibrated pink/red colour threshold.

## Scope and evidence boundary

This was a non-destructive diagnostic review of 22 matching JPEG pairs,
`DSCF2385.jpg` through `DSCF2406.jpg`. It included:

- full-frame source/output contact sheets;
- face-crop source/output/difference sheets;
- high-resolution inspection of selected closed-eye, wink, contact-lens, skin-texture,
  and tongue-out frames;
- face-aware detection using the pinned environment with the
  `mediapipe_legacy` backend;
- image-difference and face-crop sharpness diagnostics; and
- metadata, ICC-profile, resolution, and JPEG-subsampling inspection.

Local audit artifacts are under
`test_output/amgday32026_audit_20260901/`. They are Git-ignored diagnostic evidence,
not permanent versioned certification assets. The key files are:

- `whole_pairs_01.jpg` through `whole_pairs_04.jpg`;
- `face_pairs_01.jpg` through `face_pairs_03.jpg`;
- `detail_DSCF*.jpg` selected high-resolution panels;
- `mask_DSCF2397_source_overlay_lips_mouth.jpg` and the corresponding DSCF2400 mask;
- `lip_isolation_DSCF2397_source_lip16_diff_mask.jpg` and the corresponding DSCF2400
  isolated-operation panel; and
- `metrics.json`.

This review is not a full-quality rerender, a browser/print colour certification, a
multi-labeler perceptual study, or proof that every recipe and pose is safe. The deep
research addendum below includes bounded native-quality renders and algorithm probes;
those experiments deliberately do not claim that a 24 MP full batch has completed.

## Engineering implementation specification

The documentation-only implementation design is maintained separately in
[`PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md`](../plans/PLAN_AMGDAY32026_FULL_V2_ENGINEERING_2026_09_01.md).
It defines proposed code ownership, data contracts, exact-support behavior, the
tongue-safe decision object, deterministic ICC and metadata policy, QA-vector repair,
the atomic/resumable runner state machine, manifest schema, tests, mutations, rollback,
and implementation authorization gates. Its `PROPOSED` status does not claim any code
or full_v2 render exists.

## Results

### Automated diagnostics

| Check | Result | Interpretation |
| --- | ---: | --- |
| Matching source/output pairs | 22/22 | Complete reviewed batch |
| Face-aware detections | 22 faces across 22 files | Exactly one detected face per pair |
| Image dimensions | 4160 x 6240 for source and output | Runner preserved pixel dimensions |
| Full-frame mean absolute difference | 2.295–2.700 levels | Subtle overall treatment; not a quality score |
| Full-frame 95th-percentile difference | 5–6 levels | Most changes remain modest |
| Face sharpness ratio, output/source | 1.082–1.225; mean 1.187 | No evidence of overall face softening in this batch |

The sharpness ratio only measures Laplacian energy. It does not certify natural
microtexture, absence of local oversharpening, or equivalence to a full-quality render.

### Human visual review

| Area | Verdict | Notes |
| --- | --- | --- |
| Overall look | Pass | Natural and restrained at normal viewing size |
| Closed/winking eyes | Pass | DSCF2385, DSCF2395, DSCF2396, and DSCF2397 show no painted iris or catchlight on closed lids |
| Blue contact lenses | Pass | Enhancement is conservative; no obvious neon or hard-ring artifact |
| Skin texture | Pass | Texture remains visible; no obvious plastic-skin failure |
| Halos and mask edges | Pass | No obvious face-edge or local-retouch halo was found |
| Lips on ordinary poses | Pass with caution | Colour/texture is slightly stronger but remains reasonable for cosplay |
| Tongue-out poses | Improvement required | DSCF2397–DSCF2401 show tongue pixels entering the lip mask |

The resolved recipe uses the legacy `dark_circles` control and does not also activate
the advanced `undereye.darken_removal` control. Therefore this batch does not trigger
the separate double-under-eye-processing defect recorded in the per-operation audit.

## Confirmed improvement opportunities

### 1. Use the production-quality engine path for final delivery

**Current behaviour:** [`scripts/batch_amgday32026.py`](../../scripts/batch_amgday32026.py)
passes `quality="draft"`. For these 4160 x 6240 inputs, draft mode uses a 2048-pixel
proxy for detection, reshaping, and per-face work before compositing back into the
full-resolution frame. It is useful for rapid review sheets, but it is not the preferred
final-delivery path.

**Recommended change:** build a narrow delivery runner that explicitly calls the engine
with `quality="full"` and composes the existing colour-managed ingest/export, atomic
temporary-file replacement, checkpointing, and evidence-manifest facilities. No one
current entry point supplies all of those properties together. Write the rerender to a
new sibling directory, for example `amgday32026_retouched_full_v2`; do not overwrite the
accepted batch until an A/B contact-sheet review passes.

**Acceptance evidence:**

- 22/22 files complete with zero failures;
- one expected face per image;
- full-resolution face-crop A/B review against both source and current draft output;
- closed-eye, wink, contact-lens, and tongue-out anchors explicitly checked; and
- a batch manifest recording commit, recipe, quality path, detector backend, file
  status, and output checksum.

### 2. Use the existing colour-managed batch I/O

**Current behaviour:** the runner uses `cv2.imread()` and `cv2.imwrite()`. Across all
22 files:

| Delivery property | Source | Current output |
| --- | --- | --- |
| Embedded ICC profile | 22/22 `sRGB IEC61966-2.1` | 0/22 |
| JPEG chroma subsampling | 22/22 4:4:4 | 22/22 4:2:0 |
| Pixel dimensions | 4160 x 6240 | 4160 x 6240 |

A detailed spot check of DSCF2385 also found IFD/ExifIFD resolution and colour-space
fields, thumbnails, Photoshop metadata, and XMP document identifiers in the source;
the OpenCV output retained only minimal JFIF metadata.

**Recommended change:** reuse the standard colour context and
[`write_image_with_color_context()`](../../retouch/io.py). The managed path preserves
the explicit working/output colour contract, writes JPEG at 4:4:4, normalizes
orientation, and can carry EXIF and C2PA/JUMBF passthrough data. It does not currently
promise full XMP, IPTC, or Photoshop-metadata copying, so the delivery policy must name
the metadata groups it intends to retain instead of claiming blanket preservation.

**Acceptance evidence:**

- every output has an explicit expected ICC profile;
- JPEG outputs remain 4:4:4 at the selected quality;
- orientation is normalized correctly and not double-applied;
- permitted metadata is retained intentionally, with privacy-sensitive fields handled
  according to delivery policy; and
- reopening an output through the same colour-aware ingest path stays within the
  established colour-roundtrip tolerance.

### 3. Prevent tongue pixels from receiving lip enhancement

**Confirmed cause:** the face parser has mouth-interior and upper/lower-lip classes but
no dedicated tongue class. In DSCF2397 and DSCF2400, the generated mask overlay shows
part of the protruding tongue in the lip mask. An isolated `LipEnhancer` render confirms
that the tongue changes even without the rest of the recipe, so this is not merely
whole-image colour drift.

**Recommended research/implementation shape:**

1. Build an anatomical lip-ring support from the outer-lip contour minus a feathered
   or dilated inner-mouth contour.
2. Intersect semantic lip evidence with that support rather than trusting the semantic
   lip class alone on open mouths.
3. Add a conservative open-mouth conflict signal. Reduce or skip lip enhancement when
   semantic lip support invades the inner-mouth region or falls substantially outside
   the anatomical ring.
4. Use an image-safe bypass when evidence is malformed or ambiguous: return original
   lip-region pixels and skip this optional enhancement. Do not use the ambiguous phrase
   “fail open” unless the exact pixel behaviour is defined; falling back to the same
   semantic mask on an open mouth would reintroduce the confirmed tongue defect.
5. Calibrate with hand-reviewed examples covering tongue-out poses, ordinary open-mouth
   smiles, visible teeth, lipstick outside the natural vermilion border, varied skin
   tones, coloured stage lighting, and profile faces.

An absolute “pink means tongue” rule is not acceptable: makeup, skin tone, lighting,
and costume colour make it demographically and photographically unsafe.

**Acceptance evidence:**

- tongue pixels stay unchanged in DSCF2397–DSCF2401 isolation renders;
- normal lips still enhance on closed-mouth and open-smile controls;
- lipstick boundaries are not clipped to an unnaturally narrow ring;
- teeth and mouth interior remain unchanged by the lip-only operation; and
- mutation tests prove the guard, not merely the underlying parser, controls the result.

### 4. Contain lip and teeth colour-space roundtrips to exact support

The earlier per-operation audit established that lip texture/gloss and teeth whitening
can convert and return the full face canvas through LAB even when the intended edit is
masked. The drift is small, but repeated masked operations should not change unrelated
pixels.

**Recommended change:** restore original pixels outside exact operation support after
the colour-space roundtrip, or operate on a padded local bounding box and blend only
through the intended mask. This should be implemented separately from the tongue guard
so failures remain attributable.

**Acceptance evidence:** a zero mask is byte-clean, pixels outside exact support remain
unchanged, visible lip/teeth behaviour is preserved, and golden face snapshots receive
explicit human review.

## Deep research addendum

The following work was performed after the first closeout to test the proposed future
sequence instead of leaving it at architectural inference. All new render and probe
artifacts are isolated under `/private/tmp`; no production pixels or code were changed.

### A. Current runner capability matrix

There is no existing command that should simply be renamed “full_v2.” The useful
properties are split across several entry points:

| Entry point | Engine quality | Colour-managed ingest/export | Atomic final write | Evidence strength | Blocking limitation |
| --- | --- | --- | --- | --- | --- |
| `scripts/batch_amgday32026.py` | Explicit `draft` | No; raw OpenCV | No | Console progress only | Preview path, ICC/EXIF loss, 4:2:0 JPEG |
| `cli.py` | Engine default is `full` | Yes | No; writes final path directly | QA can fail the file, but no complete per-file v2 manifest | `-q/--quality` means JPEG/WebP compression, not engine quality; default worker count is CPU/2 |
| `retouch.batch_processor.BatchProcessor` | Engine default is `full`; session overrides are possible | Yes | Yes; temporary file plus `os.replace()` | Per-file callback and warning list | No complete immutable render/evidence manifest |
| `scripts/recipes/recipe_sweep.py` | Diagnostic recipe sweep | No production delivery guarantee | Manifest checkpoint is atomic | Strong detector/parser/provider/ROI diagnostics | Single-image/multi-recipe research tool; raw OpenCV image writer |

The production runner therefore needs to compose existing components instead of
forking another raw OpenCV loop. It must expose **two separately named controls**:
`engine_quality=full|draft` and `jpeg_quality=1..100`. Reusing `quality` for both is an
operational footgun.

The standard CLI defaults to `max(1, cpu_count() // 2)` workers. That is unsafe as a
starting value for native 24 MP work because measured full-mode memory is per active
render. The first full_v2 pilot must use one worker; concurrency can increase only from
measured headroom, never from CPU count alone.

`--max-dim 2048` is not a substitute for `quality="full"`. It shrinks the entire frame
before the engine and later enlarges the result with linear interpolation. That is a
useful proofing mode, but it is not a native-resolution archival render.

### B. Bounded full-versus-draft render evidence

Two 2400 x 2600 native source crops were rendered with the exact batch recipe. At 6.24
MP, each crop remains above the engine's 2048-pixel proxy threshold while keeping the
experiment safely below a blind 24 MP native run.

| Anchor | Purpose | Draft time / peak RSS | Full time / peak RSS | Face-context edge energy relative to source | Result |
| --- | --- | ---: | ---: | ---: | --- |
| DSCF2385 | Wink, contact lens, skin and hair detail | 42.15 s / 2.22 GiB | 19.69 s / 2.75 GiB | draft 2.160x; full 1.277x | Full looks more natural and materially less over-sharpened |
| DSCF2397 | Tongue-out failure anchor | 13.84 s / 1.99 GiB | 20.12 s / 2.39 GiB | draft 1.641x; full 1.284x | Full preserves texture better, but still enhances tongue pixels |

The DSCF2385 draft/full canvas difference has 2.329 mean absolute levels and 36.22 dB
PSNR; its face-context difference is 3.842 levels and 33.42 dB. DSCF2397 is closer at
1.507 full-frame and 1.917 face-context mean absolute levels. These are A/B diagnostics,
not perceptual pass scores.

The stable finding is visual/fidelity-related: both anchors put full mode near 1.28x
the source edge energy, while draft mode reaches 1.64x–2.16x. The runtime ordering is
not stable between the two images and must not be extrapolated. A historical 24 MP
`natural` benchmark in `CLAUDE.md` reports draft at 23.5 seconds/8.1 GB and a full run at
more than 25 minutes/11.2 GB without observed completion. The bounded crops therefore
prove that full is worth piloting, not that full 24 MP batch performance is solved.

Most importantly, full mode does not repair the tongue defect. Detection and parsing
still determine the semantic lip support, so the higher-fidelity path faithfully applies
the same wrong mask. The tongue guard must land before the final full_v2 rerender.

Visual comparison artifacts:

- `/private/tmp/amg_full_draft_probe/DSCF2385_quality_face_comparison.jpg`;
- `/private/tmp/amg_full_draft_probe/DSCF2397_quality_face_comparison.jpg`; and
- the corresponding lossless draft/full PNG crops and JSON runtime records.

### C. Colour-delivery probe and reproducibility caveat

DSCF2385 was decoded once through the colour-aware path, then written without any
retouch operation using the managed writer and raw OpenCV writer:

| Property | Source | Managed JPEG Q95 | Raw OpenCV JPEG Q95 |
| --- | --- | --- | --- |
| Dimensions / bit depth | 4160 x 6240 / 8-bit | Same | Same |
| ICC | `sRGB IEC61966-2.1` | Explicit Retouch working `sRGB built-in` | None |
| EXIF | Present | Present; orientation normalized | None |
| Chroma subsampling | 4:4:4 | 4:4:4 | 4:2:0 |
| Decode MAE versus working pixels | Not applicable | 0.878 levels | 0.936 levels |
| Decode maximum channel error | Not applicable | 14 levels | 42 levels |

The working profile is correct: tagged input pixels are converted into Retouch's sRGB
working space, and output pixels receive that working profile rather than being
incorrectly relabelled with arbitrary source ICC bytes.

One reproducibility issue remains. `get_working_srgb_icc()` generates a LittleCMS sRGB
profile at process startup. Two independent processes produced different profile
SHA-256 values (`fc2c340a...` and `c4a151fa...`) because the generated profile contains
process-time data. The pixels may be equivalent, but the output file hash cannot be
byte-reproducible across processes while the embedded profile changes. A certifiable
runner should package one reviewed canonical ICC asset, hash it, and reuse those exact
bytes. Add a cross-process determinism test; the current in-process `lru_cache` test
surface cannot catch this.

Metadata policy must remain precise:

- ICC, appropriate EXIF, and JPEG C2PA/JUMBF passthrough are supported;
- orientation must be normalized to 1 after pixel rotation;
- XMP/IPTC/Photoshop auxiliary groups are not currently a blanket guarantee; and
- C2PA passthrough does not create a new valid assertion for transformed pixels.

### D. Tongue/open-mouth corpus findings

The current parser cannot directly recognize a tongue. The official
[CelebAMask-HQ label list](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/README.md)
contains mouth, upper-lip, and lower-lip classes but no tongue class. MediaPipe supplies
outer and inner lip contours in its official
[Face Landmarker connections](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py),
but no tongue landmark. Dedicated research also describes tongue geometry as absent
from traditional face models and introduces separate tongue-specific data and modelling
([Ploumpis et al., 2021](https://arxiv.org/abs/2106.12302)). Therefore the credible
near-term solution is a conservative anatomical containment gate, not an invented
“tongue confidence” from models that do not emit that evidence.

A read-only detector/parser study evaluated 105 images using the pinned MediaPipe legacy
detector and ONNX parser. It included the 22-image AMG sequence plus the existing DSCF
portrait corpus. Every record completed mechanically, but that is not a correctness
claim: the larger corpus includes blurred/profile cases and poster false positives.

For the 22 AMG files:

| Group | Count | Mouth-opening ratio | Semantic mouth/lip area ratio | Mean semantic-lip support retained by hard anatomical ring |
| --- | ---: | ---: | ---: | ---: |
| Closed/ordinary controls | 17 | 0.0004–0.0333 | 0.000 | 85.50% |
| Tongue-out DSCF2397–DSCF2401 | 5 | 0.0792–0.1502 | 0.1123–0.1836 | 76.55% |

That sequence has a clean opening-ratio gap, but the larger corpus disproves using it as
a tongue classifier. Ordinary open-mouth smiles and visible-teeth poses occupy the same
opening and mouth/lip-ratio range. Opening geometry can decide **when inner-mouth
containment is relevant**; it cannot establish that a tongue is present.

A universal hard ring is also too blunt. It removes an average 14.50% of semantic lip
support even on the 17 closed controls and 23.45% on the tongue frames. That can clip
real vermilion or overdrawn lipstick. The guard should therefore leave stable closed
mouths on the semantic mask and activate the ring only for open mouths with trustworthy
geometry.

Geometry reliability is a real blocker, not a theoretical corner case:

- 9/105 evaluated records had semantic/landmark outer-lip IoU at or below 0.10;
- 19/105 were at or below 0.50;
- DSCF4560 is the clearest profile example: the semantic lip mask is visually plausible,
  while MediaPipe lip landmarks land near the nose, yielding zero intersection; and
- a universal semantic/landmark intersection would delete the entire valid lip edit on
  that frame.

The proposed mask contract is therefore:

```text
semantic_lips = parser upper/lower lip support
outer = soft polygon from outer-lip landmarks
inner = soft polygon from inner-lip landmarks
ring = semantic_lips * outer * (1 - inner)

if mouth is closed and semantic geometry is stable:
    use semantic_lips unchanged
elif mouth is open and semantic/landmark geometry agrees:
    use calibrated feathered ring
else:
    bypass optional lip enhancement and preserve original pixels
```

The reliability test may eventually include mouth-opening ratio, semantic/outer IoU,
out-of-bounds/degenerate polygon checks, inner-inside-outer consistency, face pose, and
support-retention bounds. This research intentionally does **not** select their final
thresholds or dilation/erosion radii. Those numbers must be calibrated on labelled pilot
data and frozen before holdout evaluation.

The calibration corpus must contain pixel-level or polygon labels for visible vermilion,
overdrawn lipstick, mouth interior, visible teeth, visible tongue, uncertainty, and
occlusion. Required strata include:

- closed lips, parted lips, open smiles, teeth-only, tongue tip, broad tongue exposure,
  side tongue, and partially occluded mouth;
- frontal, three-quarter, profile, tilted, blurred, small-face, and partial-face views;
- matte/gloss/metallic/dark/pale lipstick and lipstick outside the natural border;
- varied skin tones, ages, facial hair, and lip pigmentation;
- daylight, warm indoor, low-key, high-key, coloured stage light, and mixed light;
- braces, dental colour variation, props/hands/hair crossing the mouth; and
- multi-image shoot groups so near-duplicate frames cannot leak between pilot and
  holdout.

Two labelers should independently mark the critical mouth classes. Disagreements and
uncertain boundaries require adjudication. Split by subject and shoot group, not by
individual frame. Pilot data may tune thresholds and morphology; a locked holdout may
only evaluate the frozen implementation.

### E. Exact-support lip/teeth containment proof

Synthetic textured uint8 and float32 canvases reproduced full-canvas LAB roundtrip
drift in `_reapply_texture()`, `_add_lip_gloss()`, and `TeethWhitener.whiten()`:

| Operation | dtype | Pixels changed outside exact support | Maximum outside channel delta | Mean outside channel delta |
| --- | --- | ---: | ---: | ---: |
| Lip texture reapply | uint8 | 17,553 / 23,204 | 2.000 | 0.347 |
| Lip gloss | uint8 | 18,214 / 24,096 | 2.000 | 0.347 |
| Teeth whiten | uint8 | 19,100 / 24,932 | 2.000 | 0.347 |
| Lip texture reapply | float32 | 23,204 / 23,204 | 0.512 | 0.221 |
| Lip gloss | float32 | 24,096 / 24,096 | 0.512 | 0.221 |
| Teeth whiten | float32 | 24,932 / 24,932 | 0.512 | 0.218 |

Restoring the original canvas wherever the effective operation support is exactly zero
reduced every outside-support count and delta to zero while preserving all inside-
support result pixels exactly. This makes the first implementation mechanical and
low-risk:

1. calculate the final effective support after detection, morphology, feathering, and
   strength scaling;
2. perform the current operation unchanged;
3. restore original pixels where effective support equals zero; and
4. test both uint8 and float32 paths.

For gloss, the blurred specular support must also be re-clipped to the permitted lip
support so feathering cannot grow into mouth/skin pixels. For local-ROI implementations,
use a padded box for colour conversion but still perform the same exact-support restore
inside that box. A padded box alone only makes the drift smaller; it does not prove
containment.

### F. QA evidence defect found during draft/full probing

Both draft probes logged an unavailable `perceived_retouching` detector because AA6
received a proxy-resolution reshape displacement field beside a native-resolution final
image. `perceived_retouching_vector()` correctly requires `(H, W, 2)` matching the
evaluated image and fails closed when the shapes differ. The draft path upscales image
and accumulated masks but does not provide a correspondingly native displacement field.

This does not abort the render, but it means the automatic QA vector is incomplete.
`--fail-on-qa` may reject the file through the failure warning, yet a warning-only list
still cannot prove all non-flagged metrics ran. Before certification:

- upscale displacement vectors spatially and scale their x/y magnitudes correctly, or
  recompute/omit them under an explicit unavailable state;
- serialize every metric, including passes, empty ROIs, failures, thresholds, versions,
  and mask provenance;
- test full, draft, no-reshape, active-reshape, no-face, and multi-face paths; and
- mutation-test the availability gate so dropping AA6 evidence fails certification.

This aligns with `CERTIFICATION_EVIDENCE_V2.md`: process success, non-empty files, and a
warning list are not equivalent to complete automatic quality evidence.

### G. Verification completed during deep research

- 105/105 corpus records completed mechanically with detector/parser evidence; visual
  review identified the geometry-conflict cases above.
- Two bounded full/draft anchors completed with one face detected in each mode.
- The managed delivery probe was decoded and inspected with ExifTool for ICC, EXIF,
  dimensions, bit depth, and 4:4:4 subsampling.
- 116 focused I/O, ICC, batch-processor, lip, and teeth tests passed; one test skipped.
- Three focused proxy/full-path integration tests passed.
- The exact-support probe passed its proposed restore invariant for all three operations
  and both dtypes.
- No full suite, whole 24 MP full batch, calibrated tongue gate, print proof, or blinded
  human acceptance study was completed.

## Prioritized future implementation sequence

The dependencies matter. A higher-quality rerender must not precede the mask and
containment fixes that determine which pixels are allowed to change.

### Phase 0 — Freeze the baseline and evidence contract

**Work**

1. Keep sources and the accepted draft batch read-only.
2. Hash all 22 sources and 22 current outputs.
3. Freeze repository revision, resolved recipe parameters, model hashes, detector/parser
   policy, and sanitized Runtime Doctor snapshot.
4. Declare expected face count `1` for each image and tag the five tongue anchors.
5. Define metadata policy, engine quality, JPEG quality, output naming, collision rules,
   and abort/resume behaviour before rendering.

**Complete only when** the baseline manifest validates, duplicate/collision checks pass,
and no future command can silently overwrite sources or current outputs.

### Phase 1 — Land exact-support containment as an isolated mechanical fix

**Work**

1. Fix lip texture, lip gloss, and teeth whitening independently.
2. Re-clip post-blur gloss support.
3. Add uint8/float32 zero-support and outside-support byte-clean tests.
4. Add operation-isolation renders proving lip-only and teeth-only changes are zero
   outside their effective masks.
5. Review affected golden-face snapshots and real mouth crops at 100%.

**Complete only when** outside-support deltas are exactly zero, inside-support output is
unchanged relative to the pre-fix operation, focused and golden tests pass, and human
review finds no seam or lost feathering. A reduced-but-nonzero drift is not completion.

### Phase 2 — Harden the full_v2 delivery runner and QA evidence

**Work**

1. Add explicit `engine_quality` separate from `jpeg_quality`.
2. Force one worker for the first 24 MP pilot.
3. Use colour-context ingest, a frozen canonical ICC, EXIF/C2PA policy, 4:4:4 JPEG, and
   orientation normalization.
4. Write beside the final path, fsync as appropriate, decode/validate, then atomically
   replace; remove a temp file on failure.
5. Checkpoint a schema-versioned manifest after every file with source/output hashes,
   dimensions, dtype/bit depth, requested/effective mode, recipe fingerprint,
   detector/parser/provider evidence, face count/bounds, full QA vector, timings, memory,
   metadata observations, and error state.
6. Fix the draft AA6 displacement-field mismatch and prove complete QA-vector capture.
7. Add interruption/resume, stale-temp, output-collision, corrupt-write, detector-
   unavailable, parser-unavailable, and metadata-failure tests.

**Complete only when** a killed pilot resumes without duplicate/partial final files,
cross-process canonical-profile/output determinism passes where expected, manifest hashes
verify after reopening outputs, and every required evidence field is present or an
explicit failing state. “22 files exist” is not completion.

### Phase 3 — Build and lock the mouth calibration corpus

**Work**

1. Obtain consented examples for every required stratum.
2. Create the annotation vocabulary and tool instructions.
3. Double-label the critical masks and geometry-validity state; adjudicate disagreement.
4. Detect exact and near duplicates.
5. Split by subject/shoot into pilot and locked holdout.
6. Calibrate opening, geometry-agreement, support-retention, feather, erosion, and
   dilation behaviour on pilot only.
7. Freeze thresholds, morphology, implementation revision, and evaluation metrics before
   revealing holdout results.

**Complete only when** every required stratum has evaluable samples, labels have
provenance and agreement records, group leakage is zero, and the holdout remains unseen.
The existing 105-image diagnostic set is valuable research evidence but does not satisfy
this labelled-corpus gate.

### Phase 4 — Implement the tongue-safe lip support gate

**Work**

1. Preserve the semantic mask for closed, stable mouths.
2. Use the calibrated anatomical ring only for open mouths with trustworthy geometry.
3. Bypass lip enhancement on malformed, conflicting, or uncertain evidence.
4. Record decision, raw metrics, thresholds, reason, masks, and support hashes per face.
5. Add unit tests for polygon/morphology math and malformed evidence.
6. Add real-image isolation tests for tongue, teeth-only smile, lipstick overdraw,
   profile, blur, coloured lighting, and multi-face cases.
7. Add mutation tests that disable inner exclusion, invert reliability, bypass the
   open-mouth branch, or force semantic fallback; each mutation must fail a named test.

**Complete only when** tongue pixels remain original in DSCF2397–DSCF2401 and locked
holdout tongue cases; ordinary lips still enhance; teeth/mouth interior stay untouched;
profile/overdrawn-lip cases are not clipped; ambiguous cases bypass safely; and the
mutations prove the new gate is causally responsible.

### Phase 5 — Run a three-anchor native 24 MP pilot

**Work**

1. Render one stable closed/wink anchor, one ordinary open-mouth/teeth control, and one
   tongue anchor to a new full_v2 folder.
2. Record wall time, peak RSS, provider/backend, masks, QA vector, and delivery metadata.
3. Generate randomized 100% source/draft/full comparisons and amplified difference
   panels.
4. Review eyes, contact lenses, skin texture, lips/tongue/teeth, boundaries, colour,
   highlights, dark detail, and background.
5. Stop after the pilot if one-worker resource headroom is inadequate or any evidence
   gate is missing.

**Complete only when** all three native files finish, decode, retain the required
metadata contract, satisfy face/QA gates, and receive explicit human approval. Bounded
6.24 MP crop success is not this gate.

### Phase 6 — Non-destructive 22-image full_v2 batch and human acceptance

**Work**

1. Render serially into `amgday32026_retouched_full_v2`; never overwrite source or draft.
2. Validate and checkpoint after every file.
3. Generate full-frame and 100% face/mouth A/B packages.
4. Have two independent reviewers inspect randomized source/draft/full roles; adjudicate
   disagreement and critical defects.
5. Seal an immutable final manifest referencing all artifact and review hashes.

**Complete only when** 22/22 files and evidence cells pass, expected face counts match,
zero partial/stale outputs remain, all critical anchors are accepted, independent review
is resolved, and the sealed manifest recomputes successfully. Any rejected or uncertain
cell keeps the batch uncertified and routes only that file/workstream to repair.

## Explicit completion boundaries

| Workstream | Proven now | Not yet complete | Evidence required to close |
| --- | --- | --- | --- |
| Current draft batch visual audit | 22/22 pair review; eyes/contacts/texture generally good | Tongue frames have a known lip-mask defect | Existing review remains diagnostic, not certification |
| Full-quality benefit | Two 6.24 MP anchors show more natural edge energy and visible fidelity | Native 24 MP pilot and whole batch | Three-anchor 24 MP pilot, then 22/22 batch with telemetry |
| Colour-managed export | Managed probe proves explicit working ICC, EXIF, 4:4:4, and better single-image roundtrip | Production runner integration, metadata policy, frozen ICC | Cross-process determinism plus per-output ExifTool/decode/hash validation |
| Atomic/resumable batch | BatchProcessor has temp + `os.replace()` implementation | Full_v2 runner does not yet combine it with complete evidence | Kill/resume/failure-injection tests and zero partial finals |
| Tongue-safe mask | Failure reproduced; 105-image geometry research; candidate ring characterized | No calibrated thresholds, morphology, or engine change | Labelled pilot/holdout, implementation, mutations, isolation, human review |
| Exact-support containment | Drift reproduced; restore invariant reaches exact zero outside support | No source change landed | Unit/isolation/golden/real-render acceptance after implementation |
| Automatic QA completeness | AA6 draft mismatch root cause identified | Full pass-vector persistence and draft warp contract | Full metric vector, availability/failure records, mutation gate |
| Certification | Proposed v2 evidence contract exists | Writer/validator/review/finalizer and human review are not implemented | Every render, face, QA, corpus, review, and integrity gate passes |

## Closeout decision

- The present output remains accepted as visually good for ordinary viewing, sharing,
  and proof selection, with DSCF2397–DSCF2401 explicitly held as known tongue-mask
  exceptions.
- It is not the final archival, print, colour-managed, or certified batch.
- Full mode is a worthwhile quality target, but it must follow exact-support containment
  and tongue-safe masking rather than reproduce the same semantic error at higher
  fidelity.
- The future order is now explicit: baseline freeze -> exact-support fix -> runner/QA
  hardening -> labelled corpus -> tongue gate -> native pilot -> full non-destructive
  batch -> independent acceptance and sealing.
- The only intentionally unselected values are tongue-gate thresholds and final
  morphology. Selecting them without labelled pilot evidence would weaken, not complete,
  the engineering handoff.
- Sources, existing outputs, engine code, tests, and unrelated dirty-worktree files were
  untouched by this research. Only this documentation was extended.

This concludes the AMG Day 3 2026 deep research and engineering handoff.

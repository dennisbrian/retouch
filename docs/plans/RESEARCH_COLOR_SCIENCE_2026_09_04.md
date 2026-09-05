# Retouch Color Science Research and Engineering Contract

**Date:** 2026-09-04  
**Status:** documentation-only current-tree audit and implementation proposal  
**Audited revision:** `911572c`  
**Current product decision:** keep the production engine display-referred SDR
sRGB until the correctness gates in this document pass. Do not advertise the
existing helper functions as a wide-gamut, CAM16, scene-linear, or HDR pipeline.

This document re-baselines color work against the code that exists now. It
supersedes implementation-status claims in
[`PLAN_COLOR_SCIENCE.md`](PLAN_COLOR_SCIENCE.md), but not every research idea in
that older plan. A checked box or a successful round trip is not colorimetric
conformance.

## 1. Executive conclusion

Retouch already has a useful color-management foundation:

- `ColorContext` distinguishes embedded, assumed-sRGB, and RAW-derived inputs;
- embedded RGB profiles can be transformed into the sRGB working space;
- normal export embeds the working sRGB profile, while source-profile export
  transforms pixels before attaching the profile;
- Oklab/OKLCh, CIEDE2000, CAT16 white adaptation, float-aware processing, and
  final 8-bit dither are present in the live code;
- the delivery guide is honest that a 16-bit container does not prove 16-bit
  processing.

The next high-value color work is not another creative color model. It is
making the existing SDR path unambiguous and independently testable. The audit
found four delivery-integrity failures that should be fixed first:

1. Main RAW ingest requests sRGB primaries but leaves rawpy's default BT.709
   transfer in place, then records the result as `raw-srgb` and associates it
   with an sRGB ICC profile.
2. Tagged non-RAW ingest converts the Pillow image to RGB before applying the
   source profile. A real CMYK-profiled TIFF therefore fails its transform.
3. The normal non-RAW path always produces uint8. A synthetic 16-bit grayscale
   PNG did worse than ordinary quantization: values at and above 256 saturated
   to 255 in the observed path.
4. The separate linear-RAW TIFF exporter documents RGB input but passes it
   directly to an OpenCV BGR writer. A red-dominant probe reopened blue-dominant,
   and the exported linear TIFF had no ICC profile.

Several research helpers also overstate what they implement. The current
`bgr_to_cam16_ucs()` is not CIECAM16/CAM16-UCS, the ProPhoto/Adobe RGB helpers
omit required transfer decoding (and ProPhoto white-point adaptation), the P3
and Rec.2020 gamut boundaries are fixed multipliers of the sRGB boundary, and
the PQ pair alone is not an HDR delivery system. These are dormant or lightly
used today, which makes this the right time to quarantine or replace them.

## 2. Evidence language

The following labels are used so that code observations are not confused with
scientific validation:

- **Confirmed:** directly visible in the audited source or a passing focused
  test.
- **Reproduced:** demonstrated with a disposable local fixture on the audited
  environment.
- **Inference:** a conclusion from source plus a cited specification; it still
  needs an independent reference fixture before release.
- **Proposal:** a future design, not current behavior.

Audited environment: Pillow 10.4.0, OpenCV 4.11.0, and rawpy 0.27.0.

## 3. The minimum scientific contract

An array is not fully described by `float32`, `uint16`, or “RGB.” Every color
buffer crossing a subsystem boundary needs, explicitly or by one documented
invariant:

| Property | Examples | Why it matters |
|---|---|---|
| reference | scene-referred, display-referred | decides whether exposure and a display view are meaningful |
| primaries and white | sRGB/D65, Display P3/D65, ProPhoto/D50 | defines what an RGB triplet means |
| transfer | linear, IEC sRGB, BT.709, PQ, HLG | distinguishes light from encoded code values |
| range | unit, byte-scaled float, full-range integer, extended | prevents accidental clipping or double scaling |
| channels | RGB/BGR, gray, CMYK | prevents wrong transforms and channel swaps |
| alpha | absent, straight, premultiplied, flattening policy | prevents halos and silent transparency loss |
| precision | container bits and effective processed bits | prevents “16-bit” marketing from describing an 8-bit result |
| transform policy | source/destination profile, intent, BPC, CMM | makes an ICC conversion reproducible |

Bit depth does not identify gamut. A float value above 1.0 does not imply
ProPhoto RGB. An ICC profile describes the samples it accompanies; attaching a
profile after an unrelated conversion does not repair the samples. ICC uses a
profile connection space and four distinct rendering intents, each with
different reproduction behavior [S1].

For ordinary RGB-to-RGB conversion, the required conceptual sequence is:

```text
encoded source RGB
  -> decode the source transfer function
  -> linear source RGB
  -> source RGB to XYZ
  -> chromatically adapt XYZ when white points differ
  -> linear destination RGB
  -> destination transfer function
  -> destination-encoded RGB
```

The W3C reference conversions spell out this sequence and keep linear-light
spaces distinct from encoded spaces [S2]. ICC/LittleCMS may perform the same
work through profiles, but mode, profile direction, intent, black-point
compensation, and precision still need an explicit application contract.

## 4. Current pipeline, as implemented

```text
non-RAW
  -> Pillow decode + EXIF transpose
  -> convert("RGB")
  -> optional source ICC -> generated sRGB ICC (implicit perceptual intent)
  -> uint8 BGR

RAW, main engine
  -> rawpy output_color=sRGB, output_bps=16
  -> decoder-default BT.709-style transfer
  -> float32 BGR scaled to [0, 255]
  -> ColorContext source_kind="raw-srgb"

engine
  -> hybrid float/uint8 display-referred processing
  -> multiple local Lab/Oklab/linear conversions
  -> finish-stage gamut operation after clipping/quantization in one path

export
  -> working sRGB by default, or working sRGB -> source ICC on opt-in
  -> 8-bit dither or 16-bit container conversion
  -> ICC/EXIF metadata write where supported
```

This is a practical SDR image editor, not a scene-linear renderer. That is a
valid product choice. It becomes unsafe only when a subsystem silently assumes
a different reference, transfer, channel order, or precision.

## 5. Findings and repair gates

### CS-01 — RAW transfer/profile mismatch

**Priority:** P0 delivery truth  
**Evidence:** Confirmed + specification-based inference

`retouch/io.py::read_image_16bit()` and
`imread_exif_with_context()` request `rawpy.ColorSpace.sRGB` but do not pass a
`gamma` parameter. rawpy documents its default as `(2.222, 4.5)` for BT.709
[S5]. The code then creates a `raw-srgb` context with the engine's sRGB ICC.
Selecting sRGB output primaries does not make a BT.709 transfer become the IEC
sRGB transfer.

**Gate:** either request and independently verify IEC sRGB-encoded decoder
output, or describe and tag the actual encoding. Record decoder name/version,
primaries, transfer, bit depth, white-balance choice, auto-bright behavior, and
highlight mode in render evidence. A neutral ramp and color target from the
same RAW must agree with a trusted reference development before this closes.

**Status (2026-09-05): fixed, code-level.** `read_image_16bit()` now requests
`gamma=(1, 1)` (linear decoder output) and applies `white_balance.linear_to_srgb`
explicitly, so the code path matches the IEC sRGB transfer rather than rawpy's
BT.709 default. `imread_exif_with_context()`'s RAW branch now calls
`read_image_16bit()` instead of a separate `postprocess()` call, so both entry
points share one transfer. Regression: `tests/test_color_io_integrity.py::test_raw_ingest_uses_linear_decode_then_iec_srgb`.
**Not yet independently verified against a trusted reference RAW development**
(no neutral-ramp/color-target RAW fixture in this repo) — CS-01's gate is only
half satisfied. **This is a real tonal change, not just a metadata fix**: BT.709
OETF vs IEC sRGB OETF differ by roughly 13-16 8-bit codes through shadows and
midtones (e.g. linear 0.10 -> 74 old vs 89 new), tapering to ~1 code near
white. Verified the existing DSCF face-op calibration corpus
(`scripts/qa/*_study.py`, eye-gate/yaw/dark-circle thresholds in CLAUDE.md) is
sourced from `.jpg` test-output files, not `.RAF`, so those studies are not
invalidated by this change — but any future RAW-sourced calibration must be
redone after this fix, not before it.

### CS-02 — ICC transform is applied after destructive mode conversion

**Priority:** P0 input correctness  
**Evidence:** Reproduced

`_read_non_raw_with_color_context()` calls `convert("RGB")` before
`ImageCms.profileToProfile()`. Pillow requires the input image mode to match a
mode supported by the input profile [S4]. A disposable TIFF containing CMYK
samples and macOS's Generic CMYK ICC profile failed with:

```text
RuntimeError: Failed to convert tagged input to the Retouch sRGB working space
```

The source CMYK numbers had already been converted to unprofiled RGB before the
CMYK profile was asked to interpret them.

**Gate:** transform from the original decoded mode, specifying `outputMode="RGB"`
and an explicit rendering intent/flags. Add fixtures for RGB, grayscale, CMYK,
palette, and alpha-bearing images. Unsupported profile classes or modes must
fail with a precise reason before their pixels are altered.

**Status (2026-09-05): fixed.** `_read_non_raw_with_color_context()` no longer
calls `convert("RGB")` before the profile transform; it only pre-converts to
RGB when the source ICC profile itself is RGB-class (needed for RGB-tagged
palette/gray PNGs), and calls `ImageCms.profileToProfile(..., outputMode="RGB",
renderingIntent=ImageCms.Intent.PERCEPTUAL, flags=0)` explicitly. `ColorContext`
now records `transform_intent`/`black_point_compensation` from that call.
Regression: `tests/test_color_io_integrity.py::test_native_profile_mode_is_preserved_until_transform`
(CMYK/gray/LAB fixtures) and `::test_embedded_profile_conversion_records_intent_and_bpc`.
Fixtures still do not cover palette-mode or alpha-bearing tagged images (see
CS-11 below, still open) or unsupported profile classes.

### CS-03 — high-bit non-RAW ingest is not safe

**Priority:** P0 input correctness  
**Evidence:** Reproduced

All normal non-RAW input returns uint8. A generated 16-bit grayscale PNG with
samples `[0, 1, 256, 257, 32768, 65535]` returned channel values
`[0, 1, 255, 255, 255, 255]`. This is not a valid 16-to-8 scaling and destroys
most of the range. Multi-channel high-bit inputs also cannot retain more than
8-bit precision through this path.

**Gate:** preserve native sample depth through decode and color conversion. A
no-op 16-bit ramp must remain monotonic, retain more than 256 distinct levels,
and round-trip without channel clipping. If a profile/CMM path only supports
8-bit for a given mode, reject or explicitly report the downgrade; do not
silently label the result high-bit.

**Status (2026-09-05): fixed within the stated uint8 contract.** 16-bit PNG and
grayscale TIFF now decode their true 16-bit samples via `cv2`/`PIL` and are
rint-scaled to 8-bit explicitly (`_non_raw_8bit_samples()`), instead of Pillow's
implicit `convert("RGB")` truncation; `ColorContext.source_bit_depth` /
`working_bit_depth` record the downgrade and a warning is logged whenever
`source_bits > 8`. Float32 TIFF (mode `"F"`) is explicitly rejected with
`ValueError` rather than silently clipped through `convert("RGB")` (verified:
an unhandled float TIFF previously mapped `[0.1, 0.5, 0.9, 1.0]` to
`[0, 0, 0, 1]`). Regression: `tests/test_color_io_integrity.py::test_uint16_ingest_scales_full_range_and_records_downgrade`,
`::test_uint16_png_orientation_applied_once`, `::test_float_tiff_ingest_rejected_instead_of_silently_clipped`.
Gray+alpha 16-bit PNG was checked and is not a distinct code path: `cv2.imread(IMREAD_UNCHANGED)`
always expands PNG gray+alpha to 4-channel BGRA, so it takes the existing
4-channel branch. True 16-bit *processing* (this gate's contract is only
ingest-to-8-bit-safely) remains Phase 2 scope, not done here.

**Additional bug found and fixed while verifying this gate: RGB16 TIFF
orientation double-apply.** Pillow's libtiff-backed RGB16 TIFF decode pre-applies
the orientation tag at decode time and returns mode `"RGB"` directly (empirically
confirmed: an orientation-8-tagged 3x4 RGB16 TIFF decodes already-rotated, to
shape `(4,3)`, before any Pillow-level transpose call) — the same pre-rotation
behavior the existing code already knew about for grayscale `I;16` decode
(hence the `exif[274] = 1` neutralization there). But that neutralization only
matched `opened.mode in ("I", "I;16", "I;16B", "I;16L")`; RGB16 decodes as mode
`"RGB"`, missed the check, and so kept its original orientation tag through to
`ImageOps.exif_transpose()`, which rotated an already-rotated array a second
time. Verified end to end: orientation-8 content that should read `[0,4,8]`
came out `[0,3,7]`-shaped-wrong (transposed a second time) before the fix.
Fixed by adding an `opened.mode in ("RGB", "RGBA")` branch alongside the
grayscale one. Regression: `tests/test_color_io_integrity.py::test_rgb16_tiff_orientation_not_double_applied`
(orientations 1/6/8; tolerates the pre-existing ±1-code Pillow `>>8` vs
`rint(x/257)` quantization difference already documented in the 2026-08-19-ish
16-bit ingest test, since that is a separate, known, benign discrepancy).

### CS-04 — linear RAW TIFF export has channel and interpretation errors

**Priority:** P0 output correctness  
**Evidence:** Reproduced

`RAWDeveloper.load_raw()` returns linear **RGB**, and `_export_tiff()` documents
the same input. The exporter passes that array directly to `cv2.imwrite()`,
whose color convention is BGR. A probe input `[1.0, 0.2, 0.05]` RGB reopened as
approximately `[0.05, 0.2, 1.0]` RGB. The TIFF contained no ICC profile, so a
consumer also cannot know that its numbers are linear sRGB-primary values.

**Gate:** fix channel order and select one truthful deliverable:

- linear RGB TIFF with a verified linear-sRGB profile/tagging contract; or
- a defined display transform followed by tagged display-referred sRGB.

Open the result through a second library/application and verify red, green,
blue, gray, and ramp patches. The current `.dng` path is a TIFF with a DNG
suffix and must remain explicitly described as not a DNG implementation.

**Status (2026-09-05): fixed.** `RAWDeveloper._export_tiff()` now reverses
channel order explicitly (`img[..., ::-1]`, RGB -> BGR for the shared writer)
and calls a new `_write_tiff_samples()` (generalized from the 16-bit writer to
also accept float32 IEEE samples, `SampleFormat=3`) tagged with a new
`get_linear_srgb_icc()` profile — a linear-TRC (`para`, gamma 1.0) variant of
the engine's generated sRGB primaries/white profile, not an untagged dump.
`export_linear()` also now rejects non-finite or out-of-`[0,1]` input rather
than silently wrapping/clipping. Regression:
`tests/test_color_io_integrity.py::test_linear_tiff_keeps_rgb_samples_and_has_linear_profile`
(16-bit and 32-bit float, compressed and uncompressed; reopens with `cv2` for
channel order and with `ImageCms` to verify the embedded profile actually
transforms linear gray 0.18 to the expected ~118 display-sRGB code). The
`.dng` path is unchanged and still a TIFF-with-DNG-suffix, not addressed here.

### CS-05 — technical gamut mapping is split across incompatible paths

**Priority:** P1 visible color behavior  
**Evidence:** Confirmed + reproduced

There are two implementations:

- `grading.py::_apply_gamut_compress()` is identity for float samples already
  inside `[0,1]`; for out-of-range samples it clips to `[0,1]` before Oklab, so
  the original out-of-gamut direction is already lost.
- `engine.py` converts the final float result to uint8 before calling
  `compress_chroma_gamut()`. That function begins its knee at 85% of the sRGB
  boundary and therefore changes valid saturated colors, including primaries.

In a local probe, `[1.2, .2, .2]`, `[-.1, .5, 1.1]`, and `[1.5, 0, 0]` passed
through the grading helper unchanged and still out of range. The finish helper
changed valid BGR primaries, for example `[255, 0, 0]` to `[228, 49, 0]`.

**Gate:** keep extended-range values until one destination-gamut operation,
run that operation before clipping and quantization, and make the technical map
identity for all in-gamut colors. If a near-boundary roll-off is aesthetically
desired, expose it separately as a creative chroma roll-off. W3C Color 4 makes
the same important distinction: preserve out-of-gamut intermediate values,
then map when the destination cannot represent them [S2].

**Status (2026-09-05): fixed.** `engine.py::_stage_finish()`'s second gamut
call site is removed rather than swapped to the correct mapper. Reproduced
the exact finding above (`[255,0,0]` -> `[228,49,0]` via `compress_chroma_gamut`,
vs `[254,0,0]` via `gamut_compress`), then verified the removed call site was
itself always a guaranteed no-op once fixed: every operation upstream of
`_stage_finish` (selective sharpening, impact finish, purple-fringing removal)
already clips to `[0,1]`/uint8 before returning, so nothing out-of-gamut ever
reaches this stage. Spied on `_apply_gamut_compress` across natural/cosplay/
porcelain/outdoor-harsh-sun recipes and a forced +100 global-saturation push:
zero pixels changed on any of them, in both `grade()`'s call and the
now-removed second one. `grading.ColorGrader.grade()`'s internal
`_apply_gamut_compress()` call (using `gamut_compress()` +
`find_gamut_intersection()`) is now the single gamut-mapping site, positioned
before quantization, satisfying this gate. `ctx.gamut_compress` remains live
(`settings["gamut_compress"] = ctx.gamut_compress`, two sites in `engine.py`
feeding `grade()`) — removing the second call site did not orphan the param.
Visual check on the cosplay corpus (`test_output/DSCF4576.jpg`,
`cosplay_heroic_amber_v1`): the broken mapper visibly duller/muddier on the
saturated blue wig and skirt versus the fix's cleaner, more vibrant blue.
Golden face snapshots updated (`tests/golden_pipeline_face_snapshots.json`);
the non-face golden snapshot is untouched (that fixture's no-face fallback
path never reached either gamut call site, before or after). Regression:
`tests/test_color_science_k3k9.py::test_k3_finish_stage_no_longer_calls_any_gamut_mapper`,
`::test_gamut_compress_preserves_pure_primary_unlike_compress_chroma_gamut`,
`::test_ctx_gamut_compress_still_reaches_grade_after_stage_finish_removal`.
`compress_chroma_gamut()` is left in place with no remaining callers, per
CS-06's separate deprecation-marking scope — not deleted here.

### CS-06 — wide-gamut helpers are not color-space conversions

**Priority:** P1 API truth; low immediate runtime exposure  
**Evidence:** Confirmed

`color_space.py` openly states that its Adobe RGB and ProPhoto paths perform
matrix multiplication without source/destination transfer decoding. The
ProPhoto path also crosses D65 to D50 without chromatic adaptation.
`estimate_gamut()` infers ProPhoto from uint16 dtype or values above 1.0.
`find_gamut_intersection_p3()` and `_rec2020()` multiply the sRGB chroma limit
by constants, even though an RGB gamut boundary varies with lightness and hue.

**Gate:** mark these helpers experimental/deprecated now. Replacement functions
must name encoded versus linear spaces, use published primaries/white points and
transfer functions, adapt white points where required, preserve extended
values until mapping, and validate against independent vectors [S2]. Never
infer a color space from dtype or numeric range.

### CS-07 — `bgr_to_cam16_ucs()` is not CAM16-UCS

**Priority:** P1 scientific claim  
**Evidence:** Confirmed + specification-based inference

The helper accepts no adopted white, adapting luminance, background luminance,
surround, or adaptation degree. It is a custom CAT16-LMS cube-root opponent
transform with a reversible local formula. CIE describes CIECAM16 as a
viewing-condition-specific transform from XYZ to perceptual attribute
correlates [S6]. Round-tripping a custom transform does not validate it against
CIECAM16 or CAM16-UCS.

**Gate:** rename the current helper as experimental and remove “exact” from its
docstring, or replace it with a published implementation. A replacement needs
explicit viewing conditions and published forward/inverse vectors. Until then,
do not use its distance as `Delta E CAM16-UCS` in product claims.

### CS-08 — Oklab is useful, but provenance needs pinning

**Priority:** P2 reproducibility  
**Evidence:** Confirmed

The live Oklab path correctly applies the sRGB EOTF before its matrix/cube-root
steps. Its first matrix uses an older coefficient set than the author's
2021-01-25 update [S7]. The numerical difference is small; the problem is
reproducibility, not a demonstrated visible defect.

**Gate:** pick a matrix/version, cite it beside the constants, add independent
reference vectors, and centralize the sRGB EOTF/OETF so white balance, Oklab,
RAW handoff, and export cannot drift into slightly different definitions.

For future out-of-gamut work, add an extended-linear Oklab conversion that does
not clip RGB or force negative LMS values to zero before the technical mapper.

### CS-09 — the “Abney correction” is an unvalidated product heuristic

**Priority:** P1 skin-color behavior  
**Evidence:** Confirmed

`apply_abney_hue_correction()` applies a fixed `5.2 * (L - 0.6)` skin-window
formula and attributes it to the Ebner/Fairchild data. The cited research
concerns measured constant-hue data and the IPT model [S11]; the code contains
no derivation or fit evidence connecting that dataset to this formula. The
inverse test proves only that the paired functions invert one another.

**Gate:** rename it as a heuristic and feature-gate it, or fit and validate it
against the published constant-hue dataset with held-out errors. Include skin
hue/lightness sweeps and human review across the project's tone strata.

### CS-10 — PQ helpers are not HDR support

**Priority:** P1 capability truth; low immediate runtime exposure  
**Evidence:** Confirmed + specification-based inference

`linear_to_pq()` and `pq_to_linear()` implement ST 2084-shaped scalar math over
a normalized array, but define no luminance units, primaries, reference state,
tone/view transform, container signaling, or mastering metadata. BT.2100 HDR
delivery is a system specification, not merely a transfer curve [S14]. ACES
likewise separates rendering from display encoding in its output transform
[S12].

**Gate:** label these low-level experimental transfer helpers. Do not expose an
HDR export mode until the pipeline has explicit scene/display reference,
absolute luminance semantics, a tested output transform, color-volume mapping,
appropriate ICC/cICP/container signaling, and HDR-display validation.

### CS-11 — alpha is silently discarded

**Priority:** P1 compositing correctness  
**Evidence:** Reproduced

An RGBA PNG entered the normal path as three-channel BGR; alpha was dropped
without a flattening policy or warning. PNG defines alpha as a linear fraction
of opacity and does not gamma-correct it [S3].

**Gate:** either preserve straight/premultiplied alpha with explicit semantics,
or reject/flatten against an explicit color in linear light. Add transparent
edge fixtures; an undocumented drop is not acceptable.

**Status (2026-09-05): fixed.** `_read_non_raw_with_color_context()` now
extracts straight alpha from `RGBA`/`LA`/`PA` sources *before* the ICC
transform's `outputMode="RGB"` collapse (previously alpha for a tagged RGBA
image would have been lost before any flatten decision could see it, even
though `_non_raw_8bit_samples()` correctly widened PNG gray+alpha/RGBA16 to
4-channel BGRA on the initial decode). A fully-opaque alpha channel is a no-op
(`alpha_mode="opaque"`, exact pixels preserved). A partially/fully-transparent
channel is composited against white in **linear light**
(`srgb_to_linear`/`linear_to_srgb`, not naive encoded-space blending) and
recorded as `alpha_mode="flattened-white"` on `ColorContext`, with a one-line
warning per file (not per-pixel). Regression:
`tests/test_color_io_integrity.py::test_rgba_alpha_flattened_in_linear_light_against_white`,
`::test_rgba_fully_opaque_alpha_is_not_flattened`,
`::test_tagged_rgba_alpha_survives_icc_transform_then_flattens` (the last one
specifically exercises the tagged/ICC-transform ordering fix). This closes
Phase 1's alpha-behavior bullet as "flatten against documented white," not as
"preserve straight/premultiplied alpha" — full alpha preservation through the
engine is a larger change (the engine's own array contract is 3-channel BGR)
and remains future scope if transparency-preserving output is ever required.

### CS-12 — source-profile export needs capability checks

**Priority:** P2 robustness  
**Evidence:** Confirmed + library contract

The opt-in source-profile path correctly transforms working pixels before
reattaching the source profile. That is much better than profile relabeling.
However, the transform currently passes through Pillow's 8-bit image path,
uses implicit perceptual intent, and assumes an embedded source/input profile
can also serve as a destination profile. Pillow notes that profiles may support
only particular directions or intents [S4].

**Gate:** inspect profile class/color space, query transform capability, select
and record intent/flags, preserve high precision, and fall back to tagged sRGB
with a clear warning when a valid reverse transform cannot be built.

**Status (2026-09-05): partially addressed, gate text corrected to match code.**
`prepare_color_managed_export()` now rejects a non-RGB source profile
(CMYK/gray/Lab) as an export destination with a precise `RuntimeError` instead
of attempting an invalid reverse transform. This document's gate previously
said "fall back to tagged sRGB with a clear warning" for that case; the shipped
behavior is a hard failure, not a silent fallback, because a silent fallback
would let a caller believe source-profile preservation succeeded when it did
not. This is a deliberate correctness-over-convenience choice, not an
oversight — revisit only with explicit product sign-off, since it changes
export call-site error handling. Intent/flags are recorded on ingest
(`ColorContext.transform_intent`/`black_point_compensation`) but not yet
propagated through this export path; that remains open.

## 6. Operation-domain policy

“Everything linear” is as wrong as “everything sRGB.” Each operation must state
the domain matching its meaning:

| Operation intent | Required/default domain |
|---|---|
| exposure, physical light addition, flare/bloom energy, illumination adaptation | linear light in a named RGB/XYZ space |
| image resampling or alpha compositing intended to model light | linear light; alpha stays linear and premultiplication is explicit |
| perceptually even hue/chroma/lightness movement | a pinned perceptual space such as Oklab/OKLCh or validated CAM16-UCS |
| color-difference QA | named metric, white/reference conditions, and ROI; CIEDE2000 reference pairs required |
| tone/film curve | the exact scene/display/log/density domain declared by that model |
| masks, geometry, segmentation confidence | scalar data, not a color encoding |
| frequency separation used as a retouch signal | explicitly chosen algorithmic domain; it is not automatically a physical-light blur |
| destination encoding | after view/tone decision and destination gamut mapping, before quantization |

For light mixing, linear RGB/XYZ is appropriate; for perceptually even
interpolation, Oklab is appropriate [S2]. The code should therefore carry a
domain declaration, not choose a space by habit.

## 7. Proposed `ColorContext` v2

Extend the current context rather than replacing it with a heavyweight color
framework. A frame crossing an ingest, processing, preview, or export boundary
should be able to report:

```text
source_profile_hash / source_profile_class / source_mode
reference: scene | display
primaries + white_point
transfer + nominal_luminance (when applicable)
channel_order + alpha_mode
numeric_range + container_bits + effective_bits
working_encoding
transform_engine + version + intent + black_point_compensation
gamut_map + view_transform
```

These fields can initially be manifest/context metadata plus assertions. They
do not require wrapping every NumPy array on day one. Add boundary checks at
ingest, every domain-changing helper, preview, and export.

## 8. Recommended implementation sequence

### Phase 0 — truth and quarantine

- Correct current documentation and UI capability labels.
- Mark the pseudo-CAM16, pseudo-wide-gamut, P3/2020 multiplier, gamut inference,
  and context-free PQ helpers experimental.
- Add the disposable probes from this audit as regression tests.
- Give every color conversion an explicit source/destination/domain docstring.

**Exit:** no unsupported color capability is presented as shipped or exact.

### Phase 1 — trustworthy SDR ingest and export

- Fix native-mode ICC transforms and define alpha behavior.
- Make intent/black-point compensation explicit and record them.
- Repair high-bit grayscale/RGB/TIFF/PNG ingest.
- Resolve the RAW transfer/profile mismatch.
- Repair linear-RAW TIFF channel order and tagging.
- Validate source-profile export direction and precision.

**Exit:** the input/profile/depth matrix in section 9 passes, and a second
color-managed application agrees on the exported patches.

### Phase 2 — domain and precision continuity

- Centralize sRGB transfer functions and pinned matrices.
- Inventory every `astype(np.uint8)` boundary in the engine; classify it as a
  mask-only conversion, a compatibility island, or a real precision loss.
- Keep float/high-bit color through operations that claim high-bit support.
- Record effective precision from actual processing, not output suffix.

**Exit:** a no-op 16-bit ramp remains high-bit through the real engine, and the
manifest identifies every deliberate downgrade.

### Phase 3 — one destination gamut mapper

- Preserve extended-range values through creative operations.
- Add unclipped linear-RGB/Oklab conversion for boundary work.
- Replace the two live gamut paths with one technical mapper before
  quantization.
- Preserve in-gamut colors exactly; keep creative near-boundary roll-off a
  separate, named look control.

**Exit:** no output is out of destination gamut, in-gamut identity holds, hue
and lightness error are measured, and adversarial saturated ramps contain no
NaN, discontinuity, or hard channel-clip plateau.

### Phase 4 — validated perceptual and skin-color tools

- Pin Oklab vectors and CIEDE2000's Sharma reference pairs [S9][S10].
- Replace or rename pseudo-CAM16.
- Validate or retire the Abney heuristic.
- Keep ITA/Fitzpatrick estimates as descriptive QA strata only, never an
  automatic treatment selector.

**Exit:** independent datasets, not self-roundtrips, support each scientific
name used in code or UI.

### Phase 5 — optional Display P3

Only after phases 1-4, add Display P3 as a destination first, not as a guessed
working space. Define primaries, D65, transfer, ICC/cICP signaling, gamut map,
preview behavior, and sRGB fallback. Test actual P3 media on P3 and sRGB
displays through color-managed applications.

**Exit:** P3 output has verified metadata and appearance; SDR sRGB remains a
stable reference and fallback.

### Phase 6 — scene-linear/HDR only as a separate architecture

If product requirements justify it, introduce a named scene-referred working
state, camera input transforms, view/output transforms, luminance semantics,
and display-specific exports. ACES or OpenColorIO can inform this architecture:
both distinguish scene and display reference, and ACES explicitly separates
rendering from display encoding [S12][S13]. Do not turn the current SDR engine
into “HDR” by inserting PQ at the end.

## 9. Validation matrix

### 9.1 Mathematical conformance

- sRGB EOTF/OETF breakpoint and reference values.
- Published Oklab forward/inverse values, using one pinned coefficient version.
- CIEDE2000 Sharma reference pairs.
- Published CIECAM16/CAM16-UCS examples if that model is retained.
- CAT16 neutral and chromatic patch behavior under declared viewing/white
  conditions.

Self-roundtrip tests remain useful for regression but are not conformance
evidence.

### 9.2 Ingest corpus

Cross these dimensions rather than testing one convenient JPEG:

| Dimension | Required fixtures |
|---|---|
| mode | gray, gray+alpha, RGB, RGBA, palette, CMYK |
| depth | 8-bit, 16-bit integer, float TIFF where supported |
| profile | none, sRGB, Adobe RGB, Display P3, ProPhoto/ROMM, gray, CMYK, malformed, unsupported class |
| metadata | EXIF orientations 1-8, ICC plus EXIF, conflicting PNG color chunks |
| RAW | at least two camera models, neutral ramp/gray target, saturated target, highlight reconstruction |

PNG color metadata has defined precedence (`cICP`, `iCCP`, `sRGB`, then
`cHRM`/`gAMA`) [S3]. Retouch must either implement that interpretation or state
which decoder owns it and test the result.

### 9.3 Invariants

- Neutral colors remain neutral through no-op transforms.
- Red, green, and blue never exchange channels.
- In-gamut technical mapping is identity.
- A high-bit monotonic ramp remains monotonic and has more than 256 effective
  levels after a no-op high-bit render.
- Alpha-zero color cannot leak a fringe after transform/composite.
- Source and exported profile hashes, transform intent, transfer, range, and
  effective precision are present in evidence.
- Malformed/unsupported profiles fail clearly and do not silently relabel
  pixels.

### 9.4 Independent oracles

Compare selected transforms against LittleCMS/iccDEV or another independent
color-managed implementation, not another call to the same Retouch helper.
Open exports in at least two color-managed consumers. Automated Delta E and
histogram checks are necessary but do not replace calibrated-display visual
review for intentional photographic looks.

## 10. Reproduction notes from this audit

Focused current tests:

```text
./scripts/dev/test -q \
  tests/test_io_color_context.py tests/test_io_icc.py \
  tests/test_color_space.py tests/test_color_science.py \
  tests/test_color_science_k3k9.py tests/test_white_balance.py \
  tests/test_precision_dither.py tests/test_engine_precision_contract.py

161 passed, 1 skipped, 14 warnings in 1.48s
```

This is good regression evidence. It does not close CS-01 through CS-12 because
most current color tests use internal roundtrips, metadata presence, or the
same formulas on both sides.

Disposable probes were run outside the repository and did not modify source
assets:

```text
PNG16 ingest: uint8; [0, 1, 256, 257, 32768, 65535]
              became [0, 1, 255, 255, 255, 255]
CMYK ICC ingest: RuntimeError during source-profile transform
RGBA ingest: returned three channels; alpha absent
linear RGB TIFF input [1.0, 0.2, 0.05]
              reopened RGB approximately [0.05, 0.2, 1.0]; no ICC
```

## 11. What not to claim yet

- true 16-bit end-to-end processing;
- wide-gamut working-space support;
- Display P3 or Rec.2020 delivery conformance;
- genuine CAM16/CAM16-UCS or CAM16 Delta E;
- validated Abney correction;
- HDR or ACES color management;
- gamut-safe output merely because `gamut_compress=True`;
- color space inferred from dtype, file extension, or numeric range.

The safe current statement is: **Retouch is an SDR, display-referred sRGB
editor with ICC-aware RGB ingest/export and a hybrid-precision engine; several
input modes, high-bit paths, and advanced color helpers still require
conformance work.**

## 12. Primary sources

Accessed 2026-09-04.

- **[S1]** International Color Consortium,
  [ICC.1:2022 Profile Specification](https://www.color.org/specifications/ICC.1-2022-05.pdf),
  especially section 6 on the PCS and rendering intents.
- **[S2]** W3C,
  [CSS Color Module Level 4](https://www.w3.org/TR/css-color-4/), especially
  predefined RGB conversion, interpolation spaces, extended values, and gamut
  mapping. This is used as public reference math, not as a claim that Retouch
  implements CSS.
- **[S3]** W3C,
  [Portable Network Graphics, Third Edition](https://www.w3.org/TR/png-3/),
  sections 4.3 and 6.2 on color signaling and alpha.
- **[S4]** Pillow,
  [`ImageCms` documentation](https://pillow.readthedocs.io/en/stable/reference/ImageCms.html),
  including `profileToProfile()`, supported modes, directions, and default
  perceptual intent.
- **[S5]** rawpy,
  [`Params` documentation](https://letmaik.github.io/rawpy/api/rawpy.Params.html),
  including output color, bit depth, and default gamma.
- **[S6]** CIE,
  [CIE 248:2022, CIECAM16](https://www.cie.co.at/publications/cie-2016-colour-appearance-model-colour-management-systems-ciecam16).
- **[S7]** Bjorn Ottosson,
  [A perceptual color space for image processing](https://bottosson.github.io/posts/oklab/),
  including the updated Oklab matrices and D65 assumptions.
- **[S8]** Bjorn Ottosson,
  [sRGB gamut clipping](https://bottosson.github.io/posts/gamutclipping/).
- **[S9]** CIE,
  [CIEDE2000 colour-difference formula](https://www.cie.co.at/publications/colorimetry-part-6-ciede2000-colour-difference-formula-1).
- **[S10]** Sharma, Wu, and Dalal,
  [The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations](https://onlinelibrary.wiley.com/doi/abs/10.1002/col.20070).
- **[S11]** Ebner and Fairchild,
  [Development and Testing of a Color Space (IPT) with Improved Hue Uniformity](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/6/1/art00003).
- **[S12]** Academy Software Foundation / ACES,
  [ACES Output Transforms](https://docs.acescentral.com/system-components/output-transforms/).
- **[S13]** OpenColorIO,
  [Displays and Views](https://opencolorio.readthedocs.io/en/latest/guides/authoring/displays_views.html).
- **[S14]** ITU-R,
  [Recommendation BT.2100](https://www.itu.int/rec/R-REC-BT.2100/).
- **[S15]** OpenCV,
  [Color conversions](https://docs.opencv.org/4.12.0/d8/d01/group__imgproc__color__conversions.html),
  including normalized float ranges for nonlinear transforms.


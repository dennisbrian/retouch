# Research — Face Quality Evidence v1

**Date:** 2026-08-14
**Status:** Transparent measurement tranche implemented; corpus certification
and automatic decisions are not complete
**Explicit deferrals:** Blink blendshapes, expression scoring, identity
recognition, automatic reject/select, and XMP

## Executive finding

Retouch should expose a face-quality vector, not an opaque universal quality
score. The safe first vector is:

- normalized face coverage;
- detector confidence only when the detector exposes a real score;
- face-local sharpness;
- left/right eye-local sharpness; and
- explicit uncertainty and provenance for every unavailable measurement.

NIST distinguishes a quality vector containing actionable image defects such
as focus, illumination, distortion, and noise from a scalar intended to predict
face-recognition utility. Retouch is a photographic review tool, not an
identity-verification system, so the vector model is the better boundary.

Source: [NIST FATE Quality](https://pages.nist.gov/frvt/html/frvt_quality.html),
[NISTIR 8485 — Face Image Quality Vector Assessment](https://www.nist.gov/publications/face-analysis-technology-evaluation-fate-part-11-face-image-quality-vector-assessment)

Adobe's current Assisted Culling interface independently supports this product
shape: it exposes subject focus, eye focus, eye state, per-face evidence, and a
"can't tell" category while still allowing human overrides. Retouch must keep
its own measurements and limitations visible rather than imitate Adobe's
unpublished score.

Source: [Adobe Lightroom Classic Assisted Culling](https://helpx.adobe.com/lightroom-classic/desktop/organize-photos-in-lightroom-classic/assisted-culling.html)

## Implemented contract

### Measurements

`retouch/face_quality.py` accepts an image plus existing `FaceData` detections.
It does not initialize a model or make a culling decision.

For each face it records:

- normalized `(x, y, width, height)` bounding box and area coverage;
- fixed-resolution, contrast-normalized Tenengrad focus evidence for the face;
- equivalent evidence for landmark-derived left/right eye crops;
- source pixel geometry, inter-eye distance, local contrast span, measurement
  thresholds, detector runtime/model metadata, and model hash when available;
- analyzer and sharpness-method versions; and
- explicit uncertainty reasons.

The GUI exposes this as an opt-in **Measure face/eye sharpness** scan option.
If the detector runtime is unsupported, the shoot scan and manifest still
succeed, and the unavailable reason is persisted. The existing scan-only and
human-review behavior remains unchanged.

### Detector-confidence truth

`FaceData.confidence` historically defaults to `1.0` for compatibility. That
default is not evidence. A new `confidence_source` distinguishes:

- a RetinaFace detection score, which may be persisted;
- MediaPipe landmark/presence paths where a comparable result score is not
  exposed by the current Retouch adapter; and
- old/test callers whose source is unknown.

Unknown or unavailable sources produce `detector_confidence=null` plus
`detector_confidence_unavailable`. This prevents a compatibility default from
being presented as measured certainty.

### Stable face instances

Retouch assigns persistent face-instance IDs after asset-instance resolution.
On a same-content rescan, an ID is reused only when normalized boxes form an
unambiguous, mutually unique intersection-over-union match. Detector list order
is irrelevant.

Ambiguous or unmatched geometry receives a new ID and an uncertainty reason.
A content change never carries the old face identity or measurement forward.
A metadata-only scan with face analysis disabled preserves same-content
evidence but marks it `face_quality_not_refreshed`.

This is geometry continuity inside one immutable asset version. It is not face
recognition and must not be described as subject identity.

## Sharpness-method decision

The v1 metric is a declared first-derivative focus measure:

1. convert the local crop to grayscale float data;
2. normalize its 5th–95th percentile contrast range;
3. resize to a fixed analysis geometry;
4. apply a small Gaussian prefilter to reduce isolated noise response;
5. compute horizontal and vertical Sobel derivatives; and
6. store mean squared gradient energy.

OpenCV documents Sobel as a first-order image derivative. Focus-measure
research shows that operator behavior changes with noise, contrast, saturation,
and window size; this is why Retouch fixes the analysis geometry and stores the
method/version rather than asserting a universal threshold.

Sources: [OpenCV Sobel derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html),
[Pertuz, Puig, and Garcia — Analysis of focus measure operators](https://doi.org/10.1016/j.patcog.2012.11.011)

The value is deliberately raw evidence. It can still be inflated by noise,
JPEG ringing, makeup/texture, local contrast, or prior sharpening. Eye crops
can be unreliable under glasses, occlusion, profile pose, tiny faces, or bad
landmarks. Therefore:

- no fixed pass/fail focus threshold ships in v1;
- no face/eye metric changes `select`, `reject`, rating, or label;
- comparisons should eventually be calibrated within bursts and similar face
  scale; and
- human review remains authoritative.

## Landmark and blink boundary

MediaPipe Face Landmarker returns normalized landmark coordinates and can
optionally return 52 blendshape coefficients. Blendshapes are disabled in the
current Retouch Tasks configuration, and the supported legacy backend does not
provide an equivalent result through this adapter. V1 uses only canonical eye
contour landmarks to locate eye crops.

Source: [Google Face Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[Google FaceLandmarkerResult API](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult)

`eyes_open` therefore remains `uncertain`, with
`blink_analysis_deferred`. Enabling blendshapes later is not enough to certify
blink decisions: thresholds require supported-runtime tests, glasses/profile/
occlusion strata, demographic review, and human-labelled corpus calibration.

## Fairness and privacy boundary

- No absolute skin luminance or reflectance thresholds are used.
- Local percentile normalization reduces direct brightness/skin-tone coupling,
  but it does not prove demographic parity.
- No face embedding, name, demographic label, emotion, attractiveness, or
  identity comparison is generated.
- Manifest data contains measurements and geometry, not face pixels.
- Source captures are never deleted, rejected, moved, or rewritten.

## Certification plan

The feature may be called **implemented evidence collection** after unit and
integration tests. It may not be called **calibrated face-aware culling** until
all of these gates pass:

1. **Supported runtime:** repeat the scan on the pinned MediaPipe 0.10.5 legacy
   runtime and on any future Tasks runtime explicitly certified by the project.
2. **Real-image corpus:** include varied cameras, JPEG/RAW development, indoor
   and outdoor light, exposure, skin tones, ages, glasses, occlusion, pose,
   face scale, noise, sharpening, and compression.
3. **Detection recall audit:** report missed and duplicate faces by stratum;
   never compute focus success only on faces the detector happened to find.
4. **Synthetic monotonicity:** controlled blur/noise/compression/sharpening
   sweeps must reveal where the metric is monotonic and where it fails.
5. **Human agreement:** compare pairwise within-burst sharpness ordering against
   multiple reviewers; retain disagreement and `uncertain` rather than forcing
   ground truth.
6. **Geometry stability:** measure ID retention, false merges, and false splits
   across repeated scans and supported detector backends.
7. **Performance:** benchmark opt-in scan latency and memory on representative
   shoot sizes; keep the ordinary metadata scan fast.
8. **No-decision invariant:** automated rescoring must never overwrite human
   select/reject/hold, rating, labels, notes, or override history.

## Current verification boundary

The implementation is covered by model-free regression tests for:

- sharp-versus-blurred monotonicity on controlled images;
- face and eye geometry/measurement serialization;
- detector-confidence provenance;
- tiny faces, missing landmarks, and flat local contrast;
- detector-order-independent face IDs;
- ambiguous geometry, content changes, and metadata-only rescans; and
- GUI success and unsupported-detector fallback.

The installed MediaPipe 0.10.35 runtime remains explicitly unsupported in this
checkout, so this work does not claim a current-environment real-face model
pass. Blink and XMP remain deferred.

## Recommended next work

1. Run a small real-image evidence sweep under the pinned supported detector.
2. Produce a contact sheet showing face/eye boxes and raw evidence alongside
   manifest IDs for human inspection.
3. Add corpus manifests and pairwise reviewer annotations without embedding
   identity or demographic inference in product code.
4. Only after the corpus gates, consider burst-relative ranking as a separate,
   still non-destructive recommendation layer.

Subject-linked profiles should remain manual until face-instance geometry has
passed the stability gate. Blink and XMP remain independent later tranches.

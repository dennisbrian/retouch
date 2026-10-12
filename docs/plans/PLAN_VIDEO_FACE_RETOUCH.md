# Plan — Full Video Face Retouch

**Status:** 📋 FUTURE TRACK — design only; no pipeline code is authorized by
this document.

**Scope:** Turn the Retouch image engine into a temporally stable, exportable
portrait-video retouch workflow. This plan is deliberately separate from
`retouch-lite`, which remains a small, guarded demo for short clips.

## Product Definition

The target is not a frame-by-frame filter. A full video retouch must keep every
adjustment attached to the same facial features as the subject moves, speaks,
blinks, turns, is partly occluded, or leaves and re-enters the frame.

Initial product boundary:

- Offline file processing before live camera effects.
- One-person clips first; multiple faces are a later, explicit scope.
- Natural skin refine, tone/colour finish, and existing face-aware controls
  first; no identity-changing generative edits.
- Quality target: attribute-aware adaptive-natural retouch (Tier B in
  `RESEARCH_VIDEO_RETOUCH_ALGORITHMS.md`), with the classical shader stack as
  the V1 floor. Generative face re-synthesis (Bold Glamour class, Tier C)
  stays out of scope — it is the identity-changing edit excluded above.
- Preserve source audio, timing, orientation, and a browser/player-compatible
  export.

## Non-Negotiable Quality Gates

- No visible mask edge on brows, lips, nostrils, hairline, jaw, or neck.
- No frame-to-frame flicker in skin tone, smoothing strength, or face geometry.
- No temporal lag that causes the effect to slide during head rotation.
- Never smooth eyes, lashes, brows, lips, teeth, hair, or an occluding hand as
  skin.
- Preserve texture and existing `docs/PERCEPTUAL.md` invariants for every
  processed frame.
- A lost track must fade out/reset safely rather than applying a stale face mask
  to a new position.

## Proposed Architecture

```text
decode video + retain audio/timestamps
  -> face detector + persistent track IDs
  -> dense landmarks + pose + confidence
  -> semantic skin / feature / occluder masks
  -> temporal stabilization + track-loss handling
  -> face-local retouch render (GPU preferred)
  -> temporal QA and back-off
  -> composite to full frame
  -> encode video + remux preserved audio
```

### Tracking and Geometry

Replace independent frame detection with a video-aware dense face landmarker.
The tracker must emit a stable track ID, landmarks, pose/transform, confidence,
and a timestamped result for each output frame. A first implementation should
support exactly one selected face; the UI must make that choice visible.

Landmark positions and face transforms need bounded temporal filtering:

- confidence-aware EMA or Kalman smoothing for position, scale, and rotation;
- a short prediction window for brief detection misses;
- an immediate fade/reset after the confidence budget is exhausted; and
- cut detection so frames from a new shot never inherit the prior shot's track.

### Masks and Face-Local Rendering

Build a canonical face-local working space from the tracked transform. Retouch
there, then map the result and alpha mask back to the source frame. This avoids
the perceived sliding that comes from applying a new, independent ellipse to
each frame.

Masks need distinct regions for skin, eyes, brows, lips, teeth, hairline, and
face occluders. Landmark-derived masks are acceptable for the first slice;
semantic segmentation and matting are required before claiming production-grade
edge quality.

Existing Retouch modules remain the source of image-quality behavior. A video
adapter must pass only stabilized per-frame context into them; it must not
duplicate or silently alter image algorithms.

### Temporal Retouch Policy

Do not temporally blur the whole output. Stabilize the inputs and controls
instead:

- Smooth track geometry and mask alpha in face-local coordinates.
- Smooth automatically chosen strengths and colour targets over time.
- Use optical-flow-guided reuse only after validating it against fast motion,
  motion blur, cuts, and occlusions.
- Run a temporal QA gate on texture energy, skin hue, mask boundary, and
  geometry displacement. Back off or reset an effect when a gate fails.

## Delivery Phases

### V0 — Research Harness and Corpus

**Goal:** Measure the problem before changing the engine.

**Status:** 🔄 IN PROGRESS — a local temporal-QA harness and consented-corpus
manifest template exist; no corpus clips or acceptance thresholds have been
added yet.

- Assemble consented clips covering turn, blink, speech, low light, mixed
  lighting, hair crossing the face, hand occlusion, profile, camera movement,
  jump cuts, and different skin tones.
- Define objective measurements: track loss, mask-edge error, frame-to-frame
  skin-colour delta, texture-energy delta, and effect displacement.
- Produce before/after video contact sheets plus short review loops.
- Establish explicit acceptance thresholds with human visual review.

**Current V0 harness:** `scripts/review/video_qa_report.py` compares source
and result clips in normalized tracked-face coordinates. It reports retouch-map
energy, temporal flicker, source motion, and track coverage; it does not claim
to replace human edge, texture, or fairness review. Use
`docs/review/VIDEO_CORPUS_MANIFEST_TEMPLATE.md` to register consent and local
retention without committing facial video to the repository.

**Execution detail:** `docs/plans/PLAN_V0_VIDEO_QA_EXECUTION.md` — corpus
spec, the track-extraction gap, null-render calibration, threshold-setting
protocol, and the two research bake-offs.

**Exit:** A reproducible corpus and temporal-QA report exist; no image pipeline
files are changed.

### V1 — Stable Offline Single-Face Export

**Goal:** A reliable one-person, 720p, short-clip render path.

**Execution detail:** `docs/plans/PLAN_V1_VIDEO_EXPORT_EXECUTION.md` —
slice-by-slice proposal (S0 benchmarks → media I/O → tracker → stabilization
→ selected-face adapter → CLI), grounded in the V0 measurements. Proposed,
not yet authorized.

- Add video decode/encode and audio remux behind a dedicated video adapter.
- Integrate a video-mode face landmarker and timestamped one-face tracker.
- Stabilize face transforms and landmark masks; implement track-loss fade/reset.
- Adapt existing natural skin/colour controls to consume the stabilized context.
- Add CLI job input, progress reporting, output metadata, and deterministic
  short-clip fixtures.

**Exit:** No visible flicker or sliding on the V0 acceptance corpus; original
audio duration and A/V sync remain correct; image-mode golden outputs are
byte-identical.

### V2 — Feature-Accurate Beauty Controls

**Goal:** Controlled, natural effects comparable to a consumer beauty editor.

- Region-safe skin refine, tone evening, shine control, eye/teeth whitening,
  and make-up-aware handling. All tone ops stay margin-above-baseline
  (tone-invariance rule); audit against Fitzpatrick V–VI corpus clips.
- Attribute-aware automatic strength (blemish count, texture coarseness per
  face — see research doc §2.3), routed through the temporal
  strength-smoothing layer so the adaptivity itself cannot flicker.
- Spot/blemish edits anchored in canonical face-local space with a per-spot
  lifetime and confidence decay — never per-frame detection, which flickers
  by construction.
- Reuse the existing face-aware geometry tools only after video-specific
  temporal QA proves they do not wobble; dampen reshape strength from
  expression coefficients (open jaw, wide smile) before that gate is judged.
- Introduce per-effect strength envelopes and automatic QA back-off.
- Add selected-face UI, before/after scrub, and frame-level diagnostics.

**Exit:** Every effect has a per-frame and temporal visual-QA gate, plus a
representative human-review result.

### V3 — Multi-Face and Professional Export

**Goal:** Multiple tracked subjects and robust long-form renders.

- Persistent IDs through crossings and temporary occlusions.
- Per-subject settings, masks, QA reports, and safe no-track behavior.
- Job queue, cancellation, resumable export, hardware capability selection,
  codec profiles, and durable result storage.
- Audio passthrough/remux, metadata/orientation handling, and export tests for
  MP4/H.264 as the baseline delivery format.

**Exit:** Defined multi-face corpus passes with no cross-subject mask leakage.

### V4 — Live Camera Effects

**Goal:** Low-latency preview and recording, not merely faster batch export.

- Move tracking and compositor work to platform GPU paths.
- Use asynchronous camera-frame ingestion with strict frame lifecycle/memory
  control and adaptive quality tiers.
- Keep the effect processing, preview, and recorder timestamp-aligned.
- Publish supported-device performance tiers rather than implying universal
  real-time support.

**Exit:** Measured latency and frame-rate targets pass on each supported device
tier with a graceful fallback to offline export.

## Suggested Module Boundaries

New modules should be added beside—not inside—the image pipeline until their
contracts are proven:

```text
retouch/video/
  decode.py          # frame/audio/timestamp abstraction
  tracking.py        # IDs, landmarks, pose, confidence, track lifecycle
  masks.py           # feature-safe temporal masks
  stabilize.py       # temporal filters and cut handling
  renderer.py        # face-local composite adapter for image primitives
  temporal_qa.py     # flicker, edge, texture, and displacement checks
  export.py          # encoder/remuxer and output metadata
tests/video/
  fixtures/          # tiny consented clips or generated deterministic fixtures
  test_tracking.py
  test_temporal_qa.py
  test_export.py
scripts/review/
  video_qa_report.py
```

The exact package layout is a proposal, not an implementation authorization.

## Verification Strategy

Each video change is Visual-Critical. In addition to the repository's normal
precision and image QA requirements, require:

- image golden-output regression: video work must not alter image behavior;
- deterministic short generated clips for decode/encode/timestamp tests;
- corpus render with side-by-side video review;
- track/mask overlays and temporal metric report archived with the change;
- audio duration, A/V sync, orientation, and codec playback checks; and
- performance and peak-memory measurements at 720p and 1080p.

After two failed temporal-QA attempts for the same effect, stop and escalate;
do not hide flicker with stronger global blur.

## Technology Evaluation Before V1

Evaluate candidates with the V0 corpus rather than adopting one by reputation:

- **Landmarks/tracking:** MediaPipe Face Landmarker in video mode is the first
  candidate because it provides mesh landmarks and facial transforms. Confirm
  licensing, model quality, CPU/GPU paths, and profile behavior in this repo.
- **Segmentation/matting:** benchmark landmark masks against semantic face/skin
  segmentation and alpha-matting candidates on hairline/occlusion cases.
- **Mask cadence bake-off:** per-frame BiSeNet vs every-N-frames + track-warp
  vs landmark+colour prior, scored on the V0 flicker and mask-edge metrics.
- **Smoothing formulation bake-off:** current frequency separation at proxy
  resolution vs guided-filter downsample/upsample with high-pass texture
  reinjection, scored on flicker + texture-energy metrics.
- **Render backend:** begin CPU for correctness; benchmark OpenCV versus native
  GPU paths before V2. Browser/live implementations may evaluate WebGPU and
  WebCodecs separately from desktop export.
- **Media pipeline:** select a decoder/encoder/remuxer that preserves audio and
  timestamps. OpenCV-only output is insufficient for the full product contract.

## Privacy and Safety

- Process facial video only with user authority and clear retention behavior.
- Do not create identity-changing, impersonation, or face-swap workflows in
  this track.
- Keep test clips consented, access-controlled, and removable from the corpus.
- Treat face-tracking data and saved renders as sensitive user content.

## References

- Companion algorithm research (TikTok-parity tiers, smoothing/texture/reshape
  candidates, V0 bake-off recommendations):
  `docs/plans/RESEARCH_VIDEO_RETOUCH_ALGORITHMS.md`
- MediaPipe Face Landmarker video mode and output contract:
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- MediaPipe Image Segmenter video mode:
  https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
- WebCodecs processing and container caveats:
  https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API
- Temporal-consistency research framing:
  https://arxiv.org/abs/2007.01466

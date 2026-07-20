# Plan — V1 Execution: Stable Offline Single-Face Export

**Status:** 📋 PROPOSED — planning document; implementation starts only on
explicit owner approval, slice by slice. Companion to
`PLAN_VIDEO_FACE_RETOUCH.md` (V1), `PLAN_V0_VIDEO_QA_EXECUTION.md`, and
`RESEARCH_VIDEO_RETOUCH_ALGORITHMS.md`.

## What V0 measured that this plan is built on (2026-07-18, DSCF4322)

- Codec floor: null mp4v re-encode scores `effect_flicker_to_motion` 0.182,
  p95 flicker 1.51 — thresholds are margins above a per-clip null run.
- Naive per-frame render: 0.237 ratio, p95 flicker 2.28 (+51% over floor,
  spiky) — the number V1 exists to beat.
- Engine cost: ~5.45 s/frame at 4K through the full pipeline with per-frame
  detection. Unmeasured at 720p with cached contexts — S0 fixes that.
- The engine retouches **all** detected faces, including a printed banner
  face in the background. Selected-face-only processing is a correctness
  requirement, not a nicety.
- `RetouchEngine.process(..., face_contexts=[...])` already accepts cached
  contexts and skips detection/parsing (`engine.py:1415`) — the video adapter
  injects one stabilized `FaceContext` per frame. This one mechanism delivers
  selected-face-only rendering *and* the largest speedup, with zero engine
  changes.

## Architecture (concretizing the master plan)

```text
retouch/video/
  media.py       # S1: streaming decode/encode + audio remux (no full-clip RAM)
  tracker.py     # S2: MediaPipe video-mode landmarker → track contract + landmarks
  stabilize.py   # S3: One-Euro/EMA filters, track-loss fade, cut detection
  adapter.py     # S4: stabilized landmarks+parsing → FaceContext per frame
  export.py      # S5: job runner: decode → track → stabilize → render → mux
tests/video/     # deterministic generated fixtures; no facial video committed
```

Hard rules carried over: no edits inside existing image modules; image-mode
golden outputs stay byte-identical; every slice is Visual-Critical (render +
view, not metrics alone).

## Slices

**S0 — Benchmarks before design lock (half day).** Measure 720p engine cost
with and without cached `face_contexts`; measure MediaPipe FaceLandmarker
VIDEO-mode cost per frame. Output: a one-page numbers memo. If cached-context
720p exceeds ~1 s/frame, scope a param subset for video before S4.

**S1 — Streaming media I/O + audio remux.** The repo has no system ffmpeg
(`ffprobe: command not found`), and OpenCV drops audio, so this slice selects
the media dependency: **PyAV** (bundled FFmpeg libs, frame timestamps, stream
copy for audio) vs **imageio-ffmpeg** (subprocess, simpler, less control).
Decision by test, not preference: sample-exact audio passthrough, A/V sync,
orientation metadata, H.264 output, and a lossless intermediate option for QA
(the mp4v floor showed lossy intermediates eat measurement budget).
Exit: generated A/V fixture round-trips with correct duration, sync, rotation.

**S2 — Video-mode tracker.** Wrap MediaPipe Tasks FaceLandmarker in VIDEO
running mode (monotonic timestamps, reuses the repo's existing
`face_landmarker.task` model) emitting per-frame landmarks + box + confidence
in the V0 track contract, extended with a versioned `landmarks` field
(version the contract before extending — V0 plan). Selection policy: largest
face at t=0, then nearest-box continuity; no re-selection mid-clip.
Exit: on DSCF4322, coverage ≥ the V0 extractor's 100% and lower raw landmark
jitter; harness still accepts the JSON.

**S3 — Temporal stabilization.** One-Euro filter (landmark standard) or
confidence-weighted EMA on position/scale/rotation; short prediction window
for missed frames; fade-out/reset on confidence-budget exhaustion; cut
detection via frame-difference spike + track discontinuity.
Exit: on a static-camera clip, stabilized landmark jitter measurably below
raw; no lag artifacts on the turn clip (human review).

**S4 — Selected-face render adapter.** Build one `FaceContext` per frame from
stabilized landmarks; run BiSeNet parsing at cadence N (V0 bake-off decides
N) warped between runs by the stabilized transform; call
`engine.process(frame, face_contexts=[ctx], ...)` with a fixed video-safe
param set; stream results into S1's encoder.
Exit: DSCF4322 flicker ratio at or under the null floor + margin agreed at
threshold-setting; banner face byte-identical to source; human review passes
the report's gates.

**S5 — CLI job + QA integration.** `cli.py` video subcommand (or
`scripts/video_export.py` first if we want to keep cli.py untouched until
stable): job input, progress, output metadata, automatic null+result harness
runs archived next to the render.
Exit: master plan V1 exit on every corpus clip available at that time.

## Ordering and gates

S0 → S1 → S2 → S3 → S4 → S5, each landing with its tests and a review render.
S1 and S2 are independent after S0 and can be built in either order. V1
*acceptance* still requires the finished V0 corpus and
`VIDEO_QA_THRESHOLDS.md`; corpus collection proceeds in parallel and does not
block S0–S3.

## Risks

- **PyAV/FFmpeg packaging** on this Mac (LibreSSL Python 3.9) — S1 evaluates
  early precisely because packaging is the likeliest surprise.
- **Memory:** the V0 harness loads whole clips into RAM (~5 GB per 4K clip);
  V1 code must stream end-to-end, and the harness should grow a streaming
  mode before long-clip QA.
- **Engine params that misbehave under video** (e.g., per-frame auto choices)
  — S4 starts from a deliberately small param set; growth is V2's job.
- **Two failed temporal-QA attempts on the same slice → stop and escalate**
  (repo convention, restated).

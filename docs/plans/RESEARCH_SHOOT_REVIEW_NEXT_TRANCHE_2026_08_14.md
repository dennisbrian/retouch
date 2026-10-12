# Research — Shoot Review Next Tranche

**Date:** 2026-08-14
**Status:** Research and implementation contract; no production-code changes in
this note
**Baseline:** The scan-only watch UI and persistent Shoot Review Manifest are
implemented in the current worktree

## Executive verdict

The next implementation tranche should connect stable watch-folder assets to
real preview and final jobs. It should not start with automatic face rejection
or subject profiles.

Before enabling that connection, correct three identity and storage seams:

1. separate content identity from the project-local asset instance so a rename
   does not discard human review history;
2. make the manifest's project root explicit when a custom manifest path does
   not yet exist; and
3. remove recursive same-basename collisions from batch output and cache paths.

Then add a durable watch-job contract in which `done` means that the requested
output was created and verified. Face-quality evidence, XMP interoperability,
and subject-profile application should follow in that order.

## Re-baselined status

| Area | Current truth | Next gate |
|---|---|---|
| Watch folder | Stability scan only; stable records remain pending | Durable preview/final job link and verified output |
| Shoot Review Manifest | Persistent evidence, uncertainty, human decisions, override history, JSON/CSV export | Identity continuity, job provenance, and rescan reconciliation |
| Face quality | Persistent schema and explicit unavailable reason | Automatic, versioned face/eye measurement with uncertainty |
| Ratings and labels | Stored in the Retouch manifest | Optional, merge-safe XMP sidecar adapter |
| Subject profiles | Project records exist | User-confirmed face-instance links and previewed application |

The scan-only behavior is now truthful and should remain so until the job
contract below is complete.

## A. Correctness prerequisites

### A1. Split content identity from asset-instance identity

The current `asset_id` hashes project-relative path plus content SHA-256. This
detects both path and byte changes, but a rename produces a new ID. On rescan,
the old reviewed asset becomes absent and the renamed file becomes a fresh
`hold`, losing continuity even though the bytes are identical.

Use two identities:

- `content_id`: SHA-256 of the source bytes;
- `asset_instance_id`: persistent project-local UUID; and
- `relative_path`: mutable location recorded in history.

On rescan, preserve the instance when an old missing path and a new path share
one unambiguous `content_id`. If two copies have the same content, retain two
asset instances. An ambiguous match must remain unresolved for human review
rather than merging records automatically.

Record a rename/move reconciliation event in history. A byte change at the
same path creates a new asset version under the same instance and invalidates
derived evidence without erasing human history.

### A2. Keep the shoot root independent of the manifest location

When a caller supplies a new custom manifest path, manifest initialization
must still use the selected shoot root. The manifest parent is storage
location, not necessarily project identity. Make both arguments explicit and
add a test for a non-existent manifest outside the shoot directory.

The GUI scan should return and display the resolved manifest path so the human
review and export controls work without requiring the user to retype the
default location.

### A3. Make recursive output and cache paths collision-safe

The batch path currently derives output and cache names from a source basename.
Two files such as `card-a/IMG_0001.CR3` and `card-b/IMG_0001.CR3` can target the
same output/cache entry.

Preserve the relative directory in the output tree, or include the stable
asset-instance ID in the filename. Cache keys must include normalized relative
path, content/stat signature, and processing fingerprint. Add a collision test
before watch-folder jobs are allowed to call the batch processor.

Full SHA-256 on every unchanged file may be expensive for large shoots. A
persisted `(size, mtime_ns)` fast path is acceptable for unchanged records, but
new or changed records must be content-hashed before work is considered
identical.

## B. Watch Job Contract v2

### B1. State model

Use explicit durable states:

```text
pending -> queued -> processing -> done
                       |            ^
                       v            |
                     failed -- retry
```

`pending` means stable and eligible, not processed. `queued` means a durable
job record exists. `processing` means a worker owns the attempt. `done` is only
valid after output verification. A crash may return a stale `processing`
attempt to `queued` under a new lease/attempt; it must not silently become
`done`.

Each record needs:

- asset instance and source-version/content IDs;
- job ID, attempt count, lease/worker data, and timestamps;
- job kind: `preview` or `final`;
- normalized settings and settings-source fingerprint;
- pipeline/model version fingerprints;
- intended output path and format;
- output byte size and SHA-256; and
- structured error code/message for the latest attempt.

Keep preview and final completion independent. A preview artifact must never
satisfy a final-output request.

Separate execution state from aggregate outcome. A terminal multi-file job
needs an outcome such as `succeeded`, `partial`, `failed`, or `cancelled`;
generic `done` must not imply that every file succeeded. Derive the aggregate
outcome from its file records and show the counts in the GUI.

### B2. Idempotency

Derive a logical-work key from:

```text
source content ID
+ normalized settings/session hash
+ output kind and format
+ pipeline/model version fingerprint
```

Submitting the same logical work should reuse or retry its job. A source,
settings, output-kind, or pipeline-version change creates new logical work.
Use an attempt ID separately so retry history remains observable.

The settings source must be explicit: defaults, recipe/style, last approved
image, nominated hero, or Look Board. Capture One exposes similarly distinct
sources for inherited Next Capture adjustments; Retouch should preserve this
intent in the job rather than infer it.

Source: [Capture One — Adding adjustments to captured images](https://support.captureone.com/hc/en-us/articles/360002556677-Adding-adjustments-to-captured-images)

### B3. Completion invariant

A watch record may become `done` only if all applicable checks pass:

1. the processor returned successfully;
2. the intended output exists and is a regular, non-empty file;
3. its content hash is recorded;
4. any output-open/metadata validation required by the format passes;
5. the completion callback and durable state write succeed; and
6. the output corresponds to the same idempotency key that was queued.

Write output to a temporary sibling and atomically replace the final target.
If verification or the manifest update fails, retain a retryable failure and
do not claim completion. Source captures remain immutable.

### B4. Recovery and operating limits

- Use an atomic claim/lease so two scans cannot run the same record at once.
- Reconcile `queued` and stale `processing` jobs on startup.
- Bound queue depth and worker concurrency.
- Record cancellation separately from processing failure.
- Do not retry permanent configuration/model-unavailable failures without a
  changed fingerprint or explicit human action.
- Surface the job ID and output path from each manifest asset in the GUI.

## C. Face Quality Evidence v1

The current detector already exposes face bounding boxes, landmarks,
inter-eye distance, and detector confidence. These support versioned raw
measurements for face coverage and local face/eye sharpness. They do not
currently expose a truthful eyes-open result.

MediaPipe Face Landmarker can optionally output 52 blendshape scores, and its
published blendshape set includes left and right eye-blink categories. The
current Retouch task configuration disables blendshape output, and the legacy
backend has no equivalent. Add an optional face-quality analyzer or extend the
result contract without making legacy detection pretend that blink evidence
exists.

Sources: [Google Face Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[Google Face Landmarker blendshapes](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes)

For every face measurement, persist:

- stable face-instance ID;
- raw measurement values and crop geometry;
- analyzer name, version, and verified model hash;
- threshold/calibration version;
- normalized burst-relative evidence where applicable; and
- uncertainty reasons such as tiny face, occlusion, profile pose, glasses,
  detector disagreement, or analyzer unavailable.

Do not base a face-instance ID on detector list order. Match detections across
rescans with asset instance plus geometry/landmark similarity; unresolved
matches get new IDs and an uncertainty flag.

Face/eye sharpness should be compared within a burst and similar face scale,
not treated as a universal absolute score. Blink scores should first be
recorded as evidence and validated on a representative corpus. They must not
automatically reject an image before thresholds and failure modes have passed
human review. Human select/reject/hold remains authoritative.

## D. XMP sidecar interoperability

XMP is a useful optional bridge for ratings and a primary color/text label.
Adobe defines `xmp:Rating` as `-1` for rejected, `0` for unrated, and `1`–`5`
for star ratings; `xmp:Label` is a single text value. Retouch's multiple labels
therefore cannot map losslessly to `xmp:Label`.

Source: [Adobe XMP namespace reference](https://developer.adobe.com/xmp/docs/xmp-namespaces/xmp/)

Recommended adapter contract:

- remain opt-in and keep the Retouch manifest authoritative;
- map one explicitly chosen primary label to `xmp:Label`;
- preserve the full Retouch label list in the manifest;
- only export labels as `dc:subject` keywords when the user explicitly enables
  keyword export;
- merge into existing sidecars while preserving unknown namespaces and
  properties; and
- use atomic writes plus a backup/recovery policy.

Sidecars can become separated from source assets, so their path and hash should
also be recorded in the manifest. Adobe's XMP specification documents both
embedded and sidecar storage and notes the separation tradeoff.

Source: [Adobe XMP specifications](https://developer.adobe.com/xmp/docs/xmp-specifications/)

Capture One can read and update XMP sidecars without writing metadata into the
source file, but RAW and JPEG files with the same basename can share one XMP
sidecar. Retouch must detect that pair and ask whether the files are one logical
capture or separate assets before exporting ratings.

Sources: [Capture One — Metadata in XMP sidecar files](https://support.captureone.com/hc/en-us/articles/360002544898-Metadata-in-XMP-sidecar-files),
[Capture One — Pairing RAW and JPG files](https://support.captureone.com/hc/en-us/articles/30110560619165-Pairing-RAW-and-JPG-files-in-Capture-One)

## Recommended implementation sequence

1. **Correctness fixes:** split identities, explicit shoot root, visible
   manifest path, and collision-safe output/cache naming.
2. **Watch Job Contract v2:** real preview jobs first, durable states,
   idempotency, output verification, retry/recovery, and job provenance.
3. **Final jobs:** separate output namespace and independent completion.
4. **Face Quality Evidence v1:** raw versioned measurements and uncertainty;
   corpus/human calibration before any automatic decision.
5. **XMP adapter:** opt-in, merge-safe rating and primary-label exchange.
6. **Subject profiles:** user-confirmed face links, per-target preview, and
   per-image mask recomputation.

This sequence avoids building identity-sensitive profile application on top of
unstable asset and face IDs.

## Acceptance tests for the next implementation tranche

- A stable watched file remains `pending` until a durable job is created.
- Repeated scans create one logical job, not duplicate work.
- Preview success does not mark final work complete.
- An interrupted processing attempt is recovered without a false `done`.
- A write or verification failure is retryable and records no successful
  output hash.
- Same-basename files in different subdirectories produce distinct outputs and
  cache entries.
- A rename preserves human decisions when the content match is unambiguous.
- Identical copied files remain separate asset instances.
- A content change invalidates automatic evidence but retains an auditable
  human history.
- A custom manifest path retains the selected shoot root.
- Unknown or legacy face-analysis capability produces explicit uncertainty,
  not a fabricated eyes-open result.

## Verification performed during this research

Focused current-worktree suite:

```text
35 passed, 20 warnings in 16.02s
```

Covered `test_shoot_review`, `test_shoot_intelligence`, `test_watch_folder`,
`test_shoot_intelligence_gui`, `test_jobs`, and `test_batch_processor` under
Python 3 with a writable temporary Retouch cache. The warnings were existing
LibreSSL/deprecation warnings; there were no test failures. `git diff --check`
also passed before this research note was written.

This is focused verification, not real-camera ingestion, crash-recovery,
large-shoot performance, face-quality corpus certification, or XMP round-trip
certification.

# Retouch target identity and preview/export agreement — 2026-09-23

Review began September 22 and documentation completed September 23
(Asia/Kuala_Lumpur).

Status: research and documentation only. Source and selected test assertions were
read; no production/test changes, test execution, model runs, image processing,
new annotations, or Git mutations were performed. Predicted consequences below
are source-derived, not observed photo failures or measured failure rates.

This follows the [protection lifecycle research](RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22.md)
and [QA validity study](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md). Those studies ask
what must survive an edit and how to measure it. This pass asks whether the
consumer is operating on the same face, coordinates, and base image at all.
The [proposed execution plan](PLAN_RETOUCH_TARGET_AND_PREVIEW_VALIDATION_2026_09_23.md)
specifies future checks; it is not authorization to execute them.

**Status update (2026-09-23, later same day):** the three "First" rows were checked
on a real photo (DSCF4463, 6240×4160). §3 double scaling: reproduced and fixed
(`17377c8`). §4 preview face coordinates: reproduced and fixed (`bf70524`); the
`native_one_to_one` detail label remains open. §4 "cache hit removes inspection
evidence": not reproduced on fast, small full-quality or draft paths, because
`_run_detection_and_faces` returns the supplied contexts when none are rebuilt.
Remaining rows: §7 parser-dependency cache reuse reproduced and fixed
(`3e5d071`); §5 detection-order targeting reproduced on DSCF4599 and fixed with
position anchors (`db8ae05`); §6 explicit rebase now warns (`3c09e28`) but the
rebase design choice remains open.
Detail: [TODO_WEEK_2026_09_21.md](TODO_WEEK_2026_09_21.md).

## 1. Recommendation

Before calling this research tranche ready for quality qualification, establish
coordinate and target continuity across cached renders, inspection, and explicit
manual-edit rebasing. Do not select a new smoothing or parsing algorithm using
comparisons that may inspect a different region or reuse incompatible geometry.

| Order | Source finding | First discriminating check |
| --- | --- | --- |
| First | A full-quality cache hit can treat native face boxes as proxy boxes and scale them a second time | Same large image/settings, cold then warm cache, exposure correction disabled |
| First | Fast-preview pixels are upscaled for display while returned face boxes retain processing coordinates | Known face box through preview → Face inspection |
| First | The small/fast cache-reuse path returns no newly built contexts, which GUI treats as an empty current face list | Cold → warm → warm render with Face inspection after each |
| Next | Per-face edits bind to detection order, not a source-anchored subject record | Reordered, missing, and additional detections across selector/preview/export |
| Next | Explicit manual rebase reuses an already-intersected mask on a different base | Same dimensions but changed geometry or semantic boundaries |
| Alongside cache checks | Reusable parsing context has more dependencies than the GUI cache key describes | Change one parsing/preprocessing/reshape dependency with cache warm |

These priorities reflect causal dependency and potential impact, not prevalence.
This report adds no new P/FA roadmap item and changes no recipe/default.

## 2. Baseline and safeguards already present

Reviewed dirty checkout: `feat/color-science-k9-fix-and-frontier`, HEAD
`cdfd6fe5b65794e963480117d696ff0b8aff828a`. All line references below identify
this working tree. Local `main` is
`6795b7bf2e9f5ecf9b0ea60cdde3d14cf0781fb7`; no branch switch, merge, or fetch.

Read-only comparison with local main found the same inspection handler, face
selection handlers, per-face context builder, and index-based override resolver.
The native-face function differs by main's P7 source snapshot after reshape;
its cache-to-proxy and box-scaling path is unchanged. The inspected advanced,
preview-cache, inspection, and face-parameter modules have no HEAD-to-main diff.
This is a narrow comparison, not a main-branch or release audit.

Keep these existing controls:

- Preview and full-quality export capture settings snapshots; export forces
  `fast=False` and `quality="full"` ([gui.py](../../gui.py):1310–1340).
- Preview and native cache entries have different resolution keys. This report
  does **not** claim a preview entry is directly reused as a native entry.
- Inspection rejects stale settings revisions and disables downloading
  ([gui.py](../../gui.py):578–655; [gui_inspection.py](../../retouch/gui_inspection.py)).
- Advanced sessions bind to exact base pixels, dimensions, source identity, and
  processed-render evidence. Verified replay checks the saved result digest;
  mismatches remain pending ([advanced_contract.py](../../retouch/advanced_contract.py);
  [gui.py](../../gui.py):1028–1152).
- Advanced delivery rejects preview-derived bases even when their dimensions
  equal the source. The explicit detail classification already exists
  ([advanced_contract.py](../../retouch/advanced_contract.py):142–198).
- Advanced brush records store the effective mask after semantic intersection;
  ordinary replay deliberately avoids multiplying the semantic mask again
  ([advanced_retouch.py](../../retouch/advanced_retouch.py):341–387).

Selected source SHA-256 identifiers make the dirty-tree review reproducible:

| File | SHA-256 |
| --- | --- |
| `gui.py` | `bb31f024031962f348f0af931b87e7127ad63555a6c740b9ad440fefe2b6c5f8` |
| `retouch/engine.py` | `d7a9883ce2dfdfdd78e70d24f4da71be8fc601a4f47bec180d8ed896b153645d` |
| `gui_advanced.py` | `2c8381e7832ab5cd12a7112d79edf9b2bb88fe7a8052e483a4e7d3817529a81b` |
| `retouch/advanced_retouch.py` | `2ecfba102a5f751efa9f085c7669d550229c2f1c554aae5f1161b9f6faf8ef17` |
| `retouch/gui_inspection.py` | `cbc4c050b6f7e55e6f7d2a26887cdf3b7f9425ebad4c2696108331c30401edbe` |
| `retouch/gui_preview_cache.py` | `e3353a9c71e58ccb360ab745f22239bb06cb2818fc6eca8f6c3b74cd0d683abd` |

## 3. Native cache reuse can apply the proxy scale twice

For images longer than `PROXY_MAX_DIM=2048`, full quality detects on a proxy and
processes face crops at native resolution. That is intentional and preserves a
native face-processing path; full quality is not simply an enlarged proxy.

The problematic round trip is specific:

1. `_process_native_faces()` scales proxy `bbox` and inter-eye distance into
   native pixels ([engine.py](../../retouch/engine.py):2316–2330).
2. `_stage_per_face()` builds returned `FaceContext.face_data` from those native
   faces (lines 3361–3370).
3. GUI caches these returned contexts as `runtime_face_contexts`, then supplies
   them on a matching subsequent render (lines 1516–1536 of `gui.py`). Cache
   storage deep-copies them; it performs no coordinate conversion.
4. On that subsequent native path, `faces_proxy = [fc.face_data ...]` takes the
   cached native boxes (engine lines 2253–2255). `_scale_face()` applies the
   proxy-to-native factor again.
5. The later cache discard at lines 2342–2354 clears parsing reuse only after
   `faces_native` has already been constructed. It cannot undo that scaling.

Analytical example, not an executed fixture: for a 4096-pixel longest side,
proxy scale is 0.5. A proxy box with x=300, width=100 becomes native x=600,
width=200 on the first pass. Reusing that context through the same scaler yields
x=1200, width=400. Normalized landmarks stay unchanged, so the pixel box/IED
and normalized-landmark frame become inconsistent. Exact visible consequences
depend on ROI clipping, geometry, and which operations execute.

Condition this finding correctly: a matching cache entry with face contexts,
full quality above 2048, and no corrective re-detection are required. The native
auto-exposure branch can discard cached detections when correction triggers.
Use auto exposure off for a future isolating check. No claim is made that every
second render necessarily damages a photo.

Proposed contract: cached detection geometry declares its frame, dimensions,
stage, and transform lineage. Detection and parsed ROI data have distinct
lifetimes. A consumer converts from the declared frame exactly once or rejects
the context. A comment describing cached faces as proxy-resolution is not
evidence of their actual producer frame.

## 4. Preview inspection confuses array size with detail and face coordinates

The fast path downsizes the image to at most 800 pixels before detection and
face work (engine lines 1531–1547). After processing, only `result` is enlarged
back to original dimensions (lines 1943–1945). The `built_contexts` returned with
it retain the fast-processing face coordinates.

GUI records input dimensions under `native`, stores those contexts, and keeps
the enlarged RGB image as `first_processed_rgb` (lines 1551–1606). Face inspection
passes the contexts directly to `select_face_crop()`, which interprets their
boxes as native pixels; there is no intervening fast-to-output transform
([gui.py](../../gui.py):595–635;
[gui_inspection.py](../../retouch/gui_inspection.py):313–348).

For example, a 4000-pixel-wide image reduced to 800 has scale 0.2. A processing
box at x=200 corresponds roughly to x=1000 in the enlarged output, not x=200.
The exact mapping must use the actual resized dimensions and declared coordinate
convention, including rounding. This example identifies the frame mismatch; it
is not a newly observed wrong-face screenshot.

There is a separate evidence problem even when the crop is positioned correctly:
`build_inspection_contract()` marks the array as native and labels 100% display
as `native_one_to_one` without receiving render mode/effective detail (lines
493–549). A crop can truthfully avoid additional resizing while still containing
previously enlarged 800-pixel detail. Advanced delivery already distinguishes
these cases; inspection should consume the same distinction.

Proposed research contract:

- Separate source dimensions, processing dimensions, output dimensions, display
  zoom, and effective detail. A 1:1 array view does not prove native processing.
- Transform face boxes/supports into the inspected output frame before cropping;
  reject missing/ambiguous transforms instead of treating them as identity.
- Qualify preview for composition and provisional intent. Use a full-quality
  render for texture/mark acceptance; do not demand byte equality between these
  deliberately different processing paths.
- Include effective super-resolution and reshape in transform evidence. The
  engine explicitly leaves masks at retouch resolution after optional export SR
  (lines 1990–1996); that path needs its own inspection outcome, not a fabricated
  identity transform.

### A cache hit can also remove inspection evidence

On the small/fast reuse path, `_stage_per_face()` intentionally returns
`built_contexts=None` because it reused supplied contexts (engine lines
3343–3350). That value travels through `_run_global_phases()` to the returned
`ProcessingResult.face_contexts`; the engine can still report a nonzero
`face_count`.

GUI interprets `result.face_contexts or []` as the current complete list,
overwrites the cached runtime contexts, and uses that empty list for the latest
inspection evidence (GUI lines 1524–1552). A Face inspection request can
therefore reject index 0 as outside the detected list even though face work ran.
The next render can detect/parse again because the cached context list is now
empty. This is a source-derived producer/consumer mismatch, not an observed
three-render GUI sequence.

Distinguish `no new contexts built`, `known current contexts reused`, `no faces
detected`, and `face evidence unavailable`. Do not blindly carry old evidence
forward: reuse must first satisfy the frame and dependency contracts. Inspect
all three cold/warm/warm states in the future control, not only a single cache
storage call or render.

OpenCV documents resize as resampling and distinguishes interpolation methods.
Those numerical operations do not supply missing face-coordinate provenance.
This supports specifying each transform, not blindly applying one interpolation
method to images, soft alphas, and categorical labels alike.
[OpenCV 4.11 geometric transformations](https://docs.opencv.org/4.11.0/da/d54/group__imgproc__transform.html)

## 5. Detection-list position is not a persistent target

`on_detect_faces()` detects on the decoded source, labels thumbnails `Face i`,
and stores recipe overrides keyed by that integer (GUI lines 1924–1986).
The engine subsequently detects on its own preprocessing/scale path and resolves
`ctx.face_params.get(face_index)` without matching to the thumbnail's original
box or landmarks (engine lines 3176–3184).

Detection order passes through the backend's results and augmentation/merge
paths ([detection.py](../../retouch/detection.py):360–374, 405–413, 535–551).
`FaceData` stores geometry/confidence, with no persistent subject identifier.
Changing scale, exposure, or visibility can change the detected set; the code
does not establish that the same integer still names the selected person.
This is an association gap, not a measured claim that the backend routinely
randomizes order.

Two adjacent routes deserve the same qualification:

- The GUI builds one integer-keyed `engine_kwargs["face_params"]` mapping before
  looping over selected images. It is therefore an ordinal mapping reused per
  image, not a subject mapping across a shoot (GUI lines 1432–1449).
- Advanced face choices are detected from uploaded source paths, while reshape
  and semantic intersection re-detect on the current edited canvas
  ([gui_advanced.py](../../gui_advanced.py):380–392;
  [advanced_retouch.py](../../retouch/advanced_retouch.py):212–267).

The MediaPipe result API describes per-face landmark lists in normalized image
coordinates; it does not document a persistent person ID in that result. The
local wrapper and applicable backend still need their own association contract.
[MediaPipe FaceLandmarkerResult](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult)

Proposed first design comparison: a source-bound target record with geometry
and selection provenance, matched through known transforms using one-to-one
spatial/landmark agreement. Record unmatched and ambiguous cases. Sorting faces
left-to-right is insufficient when faces appear/disappear, overlap, or cross.
Do not introduce biometric recognition or cross-shoot identity inference as an
unexamined dependency. A batch index remains an index unless the user explicitly
establishes per-image correspondence.

Small-input-shift sensitivity in CNNs is documented in primary literature. It
motivates perturbation controls here; it does not quantify Retouch's sensitivity
or justify replacing its detector with an anti-aliased model.
[Zhang, ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html)

There is also a narrow invalid-input behavior: `_face_index()` returns `None`
for a nonnumeric unrecognized value, and `None` means all faces. Out-of-range
numeric indices already raise. A future contract should reserve all-face scope
for the explicit all-face tokens and reject malformed targets. The ordinary
dropdown supplies valid choices; no accidental broad edit was reproduced.

## 6. Exact replay and explicit rebase need different promises

Do not weaken the existing exact-base checks or re-intersect stored effective
masks during ordinary replay. Repeating a soft semantic intersection changes
the edit: brush alpha `a` and semantic support `s` would become `a*s*s` instead
of the stored `a*s`. The existing feathered-mask test specifically guards this.

The separate explicit rebase path matters. Clicking to load a processed result
with current edits and no pending session sets `explicit_legacy_replay=True`
([gui_advanced.py](../../gui_advanced.py):312–329). It replays the current edit
list on the new base. This is an intentional user action, not an unnoticed
session-file bypass. It correctly records the new base contract.

However, equal `image_shape` is the replay loop's geometry check, and a stored
brush mask remains the old post-semantic mask. Same dimensions do not guarantee
that a jawline, eye boundary, or protected mark stayed at those coordinates
after a new reshape or different detection. Replay of reshape also re-detects
and resolves the stored ordinal selection again. A newly recorded base digest
establishes which pixels were used; it does not prove the transferred edit still
expresses the intended target.

Proposed rebase choices to study, without selecting a default here:

1. Retain literal image-coordinate support and require review of its overlay on
   the new base. Appropriate only when that is the user's intended operation.
2. Transport support through a verified geometric transform, then review hard
   protection and material boundaries. A generic resize cannot represent reshape.
3. Reconstruct semantic-constrained intent from separately retained brush and
   constraint evidence. This is a new edit decision, not exact replay.

Current records retain the effective mask, not the original brush and semantic
support as separately replayable inputs. Therefore option 3 is a schema/design
study, not a behavior available merely by toggling the replay function.

## 7. Cache validity includes appearance and parser dependencies

The GUI key already covers source identity, resolution tier, quality/fast,
optical correction, color/RAW description, detector description, and engine
version (GUI lines 1464–1478). Keep these; a larger undifferentiated settings
hash is not necessarily the best reuse strategy.

The same entry also caches parsed regions and light-direction estimates. Those
are produced after optional preprocessing and reshape, and parsing receives
`mask_feather_mode` (engine lines 1818–1880, 3224–3231, 3343–3370).
The inspected key and invalidation list do not include these individual
dependencies (GUI lines 5075–5092). On the small/fast path, supplied contexts
skip parsing and reuse their regions/light estimate. Thus a source-file match
alone is insufficient evidence that all cached objects remain applicable.

Proposed dependency study: separate decoded-source reuse, detection reuse, and
parsed-support reuse. For each, name the producing pixels/stage, frame, model,
and relevant options. Change one dependency at a time and compare warm versus
cold results. Treat stale regions, changed selected targets, and changed edit
decisions separately from acceptable runtime-only differences. No new cache
architecture is implemented by this document.

## 8. Evidence limits and next handoff

Selected existing assertions cover cache storage/copying, stale-revision rejection,
non-downloadable inspection, blocked preview-derived Advanced export, base
mismatch, and no double semantic intersection. They were read, not rerun. Those
assertions do not themselves establish the end-to-end scale/target cases above;
this was not an exhaustive census of all tests.

The [execution proposal](PLAN_RETOUCH_TARGET_AND_PREVIEW_VALIDATION_2026_09_23.md)
starts with deterministic coordinate/identity controls, then separately reviews
real photos and decoded delivery. Keep protection-lifecycle and measurement
validity work as prerequisites, not competing roadmaps. External documentation
and literature provide contracts/motivation; none establishes Retouch quality.

This pass can close as a documented investigation. Preview/export agreement and
photo-quality qualification remain open until the proposed evidence exists.

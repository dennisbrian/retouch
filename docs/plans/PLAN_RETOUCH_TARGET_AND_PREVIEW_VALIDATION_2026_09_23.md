# Proposed target and preview/export validation — 2026-09-23

Status: proposed in a documentation-only pass. On 2026-09-24 the user asked to
finish the work; focused contract validation and the bounded T6 behavior below
were completed. Read the
[source research](RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23.md) first.
This extends the existing [QA study](PLAN_RETOUCH_QA_VALIDATION_2026_09_21.md)
and [protection study](RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22.md);
it does not introduce a new P/FA item or change recipes.

**Execution update (2026-09-24):** missing face-frame metadata now fails closed;
mask-only rebase displays its recorded support and gates further edits, session
save, snapshot save, and export on acknowledgement while the overlay is visible
at at least 20% opacity. Exact-base mask-session loads also show their support
and require review, including v2 sessions replayed later after a source becomes
available. This covers sessions saved before review provenance existed. Replay
refuses missing, fractional, or inconsistent mask geometry instead of resizing.
Reshape rebases that depend on detection order are refused. The focused
regression selection passed 80 tests. The full public GUI-to-engine control
matrix and independent photo qualification remain unexecuted. Photo use is
deferred until corpus permissions and subject/event grouping are confirmed;
the current development files do not establish either.

## 1. Questions to resolve before quality comparison

1. Does a warm cache preserve the cold render's coordinate frames and intended
   face, or explicitly invalidate incompatible cached objects?
2. Does Face/ROI inspection show the requested region of the identified render,
   with truthful processing-detail evidence?
3. Do thumbnail selection, preview, full-quality processing, and Advanced edits
   resolve to the intended subject or report uncertainty?
4. Does manual rebase state how support moves to a new base, independently of
   whether exact-base replay succeeds?

Pixel agreement alone cannot answer all four. Two identically wrong target
assignments still agree. Conversely, preview and full quality deliberately use
different processing scales and need not be byte-identical.

## 2. Freeze the diagnostic baseline

Record revision plus dirty-file hashes, source file and decoded-pixel identity,
recipe/overrides, actual backend/model versions, working color contract, and
requested/executed operations. Keep the existing dirty worktree and source
photos intact. Reuse reviewed corpus governance; saved development images do
not become an independent qualification set by relabeling them.

For each geometric object, record:

```text
source asset + run/revision + stage
frame identifier + width/height + coordinate convention
transform parent and transform data/digest
target selection identifier + matched detection identifier + match disposition
support role + support content identifier
```

Separate normalized landmarks, pixel boxes, crop-local masks, and full-image
alphas. Record actual resized dimensions, not only the intended scale factor.
Keep source/native size distinct from effective processing detail and output
size. Existing Advanced base contracts are the starting point for manual edits.

## 3. First: deterministic contract checks

Future checks should exercise the real wiring with controlled detector/parser
results. They are proposed tests, not results from this research pass.
Use the public processing route with optional neural operations and reshape off
to isolate coordinate reuse. Record any iterative QA path separately; do not
substitute a private legacy pipeline for the actual GUI-to-engine handoff.

| Control | Route | Required observation |
| --- | --- | --- |
| 4096 longest side, one known box, full quality, exposure off | Cold render → cache store → warm render | Native box/IED transformed once; identical target and valid crop on both |
| Below/at/above 2048, including non-square inputs | Native/proxy boundary | Actual x/y scale and rounding recorded; no frame-dependent cache ambiguity |
| 4000×3000 input with known 800×600 detection | Fast preview → enlarged result → Face inspection | Crop corresponds to transformed box; effective detail remains preview |
| Same pixels/revision but missing frame metadata | Inspection/cache consumer | Explicit unavailable/recompute outcome, not assumed native geometry |
| Same image/settings, cold → warm → warm | Fast and native separately, inspecting Face after each | Stable target/context availability; reused versus newly built contexts explicit |
| Two detections with reversed result order | Selector → preview/export | Selected subject remains associated or edit abstains; index alone cannot pass |
| Selected face disappears; new face occupies its list position | Preview/export | No transfer of its override to the replacement detection |
| Two photos with different people/order | Batch per-face mapping | Ordinal versus per-image target scope explicit; no inferred shared identity |
| Malformed selection and explicit all-face token | Advanced reshape/semantic targeting | Malformed target rejected; explicit all-face behavior retained |
| Change feather mode, reshape, denoise, exposure one at a time | Warm versus cold cache | Relevant parsing/detection data invalidated or equivalence demonstrated |
| Stale settings revision | Inspection/export | Existing stale-result rejection remains intact |

Include negative controls with unchanged geometry and known correct matching.
Measure availability/recomputation as well as correctness; disabling all caching
or skipping all selected edits is not evidence of a useful final design.
No performance target or acceptable failure percentage is invented here.

## 4. Then: distinguish replay from rebase

Preserve existing exact-base, saved-result digest, preview-export block, and
post-semantic-mask replay controls. Add separate future comparisons:

| Base transition | Purpose | Evidence |
| --- | --- | --- |
| Identical verified base | Exact replay regression | Existing result identity/tolerance contract maintained |
| Same dimensions, tone-only change | Literal support transfer | Overlay, intended target, and material support still reviewed |
| Same dimensions, reshape changed | Geometry-sensitive rebase | Original support, transform/registration evidence, transported or rejected support |
| Fast preview → full quality | Resolution/decision transition | Effective-detail change, face association, reviewed support and edit disposition |
| Different image with same dimensions | Source binding | Existing session mismatch blocks implicit replay |
| New detection order during reshape replay | Target association | No change of recipient merely because list order changed |

Treat literal support reuse, transform-based support transport, and newly
computed semantic intent as separate arms. Do not restore semantic intersection
to ordinary replay. If evaluating a new rebase schema, preserve old effective
support and keep legacy records visibly limited rather than inventing missing
brush/semantic provenance.

## 5. Finally: real-photo intent and delivery review

Only after coordinate controls are sound, use separately reviewed subjects/shoots
with single/group portraits, small faces, profiles/roll, occlusions, boundary
makeup, and appearance/lighting variation. Use existing consent and review rules;
do not infer identity or demographic labels from appearance.

Compare paired routes on the same source and settings:

- Preview versus downsampled full-quality output for composition, target, and
  operation/disposition agreement at a declared common viewing scale.
- Full-quality output at native detail for marks, makeup, texture, and geometry.
- Cold versus warm full-quality render for cache effects, using fixed settings
  and backend; control any stochastic operations before asserting equality.
- Pre-encode delivery buffer versus decoded export for profile, resize, codec,
  and precision effects. Match size/color explicitly; lossy JPEG is not expected
  to preserve exact pixels.

Keep denominators explicit: reviewed selected targets, successfully associated
targets, edited targets, unmatched/ambiguous targets, and measurable supports.
Report wrong-recipient edits, missed requested edits, support violations,
disposition changes, and benefit/harm separately. Include per-face and worst-face
outcomes so a successful large face cannot hide a failed small face.

Use fixed reviewed supports for preservation scoring, with transform lineage.
Source defects, interpolation effects, detector failures, intended reshape, and
new edit harm need distinct labels. Apply the existing QA-validity controls
before using an automated score to rank candidates.

## 6. Proposed artifacts and decision gates

Proposed artifacts: a run manifest, coordinate/association records, paired
cold/warm diagnostics, support overlays/contact sheets when photo work is
authorized, decoded-delivery records, and independent review dispositions.
These artifacts were not created by this documentation pass.

Gate order:

1. **Coordinate correctness:** every consumed box/support has a valid frame;
   transforms occur once; incompatible contexts recompute or stop explicitly.
2. **Target correctness:** controlled reorder/missing-face cases never silently
   redirect a selected edit. Ambiguity is represented and useful coverage reported.
3. **Replay/rebase truth:** exact replay remains protected; intentional rebase
   has explicit support/target evidence and a reviewed outcome.
4. **Photo qualification:** independent review measures intended benefit and
   preservation harm using the earlier grouped QA protocol. Contract tests alone
   do not pass this gate.

An initial implementation, if later authorized, should address the native-cache
frame mismatch and fast-preview inspection first, with narrowly scoped checks.
Target association and rebase semantics then need their own reviewed design.
Classifier replacement, stronger smoothing, and new recipe defaults remain
outside this proposal.

# Cosplay tutorial compositing review — 2026-09-25

Documentation-only synthesis of the selected tutorial sequences and a source
inspection of Retouch's current background stage. The tutorial screenshots
show manual Photoshop workflows. They do not provide editable project files,
source-resolution outputs, or controlled comparisons. No photos were processed,
no tests were run, and no production code was changed.

## Evidence reviewed

This review focuses on images 039–066, 108–135, 208–227, and 228–248 from the
[batch index](/Users/dennis/Downloads/research_review_batches_u65zuqfu/findings_index.md).
The associated notes are batches 02, 03, 05, 06, 09, and 10. Selected full-page
screenshots were reopened for this pass, including the rain-depth summary,
fire-placement and subject-mask examples, localized reflected light, bird
depth, final grade, and environment lighting. Image descriptions below refer
to what the pages visibly demonstrate; they do not assess the final composites'
photographic quality.

## What the tutorial sequences show

| Sequence | Visible workflow | Transferable review question |
|---|---|---|
| Rain portrait, 039–052 | Rain, droplets, and splashes are split across apparent near, middle, and far distances; the subject and ground are selectively masked. The summary page says one rain overlay is insufficient. See [052](/Users/dennis/Downloads/research/MTXX_20170124141041.jpeg). | Do effect layers have the right scale, focus, opacity, and overlap for their apparent distance? Are hair and ground transitions kept clean? |
| Fire cosplay, 053–066 | Flames are attached to plausible scene structures, then transformed, masked, and placed around a person. The tutorial repairs areas where fire covers the figure, adds restrained red reflected light with masked adjustments, and places embers at different distances. See [054](/Users/dennis/Downloads/research/MTXX_20170124141147.jpeg), [060](/Users/dennis/Downloads/research/MTXX_20170124141332.jpeg), and [064](/Users/dennis/Downloads/research/MTXX_20170124141429.jpeg). | Do object masks, foreground/background order, and local light agree with the intended scene? Does an overlay hide costume or skin detail that should stay visible? |
| Pharah/Mercy scene, 108–135 | Separate figure and wing pieces are cut out and assembled, then matched to a constructed ground, skyline, and sky using scale, perspective, shadows, haze, and color/light passes. | Does the requested job require full scene and pose assembly, or only polishing a photograph that already has its intended composition? |
| Blue/white fantasy scene, 208–227 | A written sequence covers planning, subject and light analysis, assembly, hair, depth, grade, and atmosphere. Birds and particles use varied placement and scale; masked color and light adjustments finish the scene. See [221](/Users/dennis/Downloads/research/MTXX_20170124162255.jpeg) and [223](/Users/dennis/Downloads/research/MTXX_20170124162334.jpeg). | Can the editor show which elements sit in front of or behind the subject, and can the local and global adjustments be reviewed independently? |
| Bloodborne scene, 228–248 | The tutorial constructs an ornate environment, places the costumed figure and foreground objects, then integrates them with red effects, smoke, and weapon light. Costume/environment lighting is a distinct late pass. See [241](/Users/dennis/Downloads/research/MTXX_20170124163113.jpeg). | Are foreground occlusion, edge masks, and environment-colored light coherent at the final viewing size? |

Across the sequences, the common authoring order is: plan the scene and light;
place or construct assets; decide layer order and masks; fit scale and
perspective; add depth-dependent atmosphere; repair occlusion; add local
reflected light; then complete the global grade. This order is a synthesis of
the page sequences, not a claim that the tutorials use one identical method.

## Retouch code boundary

The inspected checkout is `feat/wire-under-reached-ops` at `6ef691d`, as in the
[current-code crosswalk](RESEARCH_COSPLAY_TUTORIAL_CURRENT_CODE_CROSSWALK_2026_09_25.md).
The relevant source paths are `retouch/background.py` and
`RetouchEngine._stage_background` in `retouch/engine.py`.

### Capabilities present on the engine background stage

With a person mask, the stage can grade or desaturate the existing background,
apply background blur or a radial/vertical lens-blur approximation, add a
silhouette light wrap, and sharpen the subject. The matte path can use parsed
hair evidence when it composites a blurred or graded background. The stage
does not add rain, fire, birds, smoke, weapons, or other tutorial assets as
separate foreground elements.

An adjacent [person-gate false-negative report](RESEARCH_PERSON_GATE_WIG_FALSENEG_2026_09_25.md)
for code revision `6ef691d` records 9 of 47 faces rejected in one silver-wig,
bright-window shoot. That report was not revalidated in this compositing review,
and it measures face coverage rather than the quality of background effects.
It matters to this stage because `_stage_background` returns the input when no
person mask is available. Record these skipped cases as coverage misses in any
future portrait study.

`retouch/background.py` also defines `BackgroundReplacer.replace()` and
`relight_scene()`, but this checkout's engine does not call either method.
Their presence in a module is not evidence of a working GUI or CLI scene
authoring workflow. The background parameters are in the recipe/parameter
registry; the inspected Gradio bindings store them as state, and their
`ParamSpec` entries have no CLI flags.

The face compositor merges overlapping face-edit regions. That operation
protects per-face edits in a photograph; it does not assemble full characters,
props, or costume pieces. Likewise, `lens_blur` supplies a coarse distance
heuristic for background defocus, not a scene graph with separately editable
near/far objects.

## Research implications

| Candidate direction | Assessment from this evidence | Decision boundary |
|---|---|---|
| Improve a finished cosplay portrait's existing background separation | Closest to Retouch's current background stage: validate existing blur, grade, matte, light-wrap, and subject-sharpen operations on authorized source/output pairs. | Requires visual review of hair, wig, costume, and edge spill. The tutorial screenshots do not prove the current operations are effective. |
| Add a requested reflected-light pass | The tutorials show masked, scene-colored light; Retouch has face/body relight and silhouette light-wrap concepts, but they are different supports and controls. | Keep it optional and test independently from skin smoothing and global grade. Do not infer light direction or opacity from screenshots. |
| Build a Photoshop-like fantasy compositor | Pose assembly, asset transforms, per-object depth, occlusion repair, and particle/fire authoring are broader than the current face/background stage. | Treat as a separate creative-editor product decision, not as a small face-algorithm update. First define the authoring workflow and desired output cases. |
| Make tutorial values defaults | Values appear tied to individual source scenes, screenshots, and UI versions. | Do not copy layer opacity, curves, blur, or color settings into recipes. |

The most useful near-term action is to use the existing
[transfer-study protocol](PLAN_COSPLAY_TUTORIAL_TRANSFER_STUDY_2026_09_24.md)
with authorized matched portraits, if the decision concerns portrait polish.
That study should keep background treatment, local light, hair boundaries, and
face edits in separate comparison arms. A larger asset-compositing feature
would need its own brief and representative source assets before it can be
estimated responsibly.

## Limits

The pages are photographed, compressed tutorial material with gaps between
steps. They do not include the source photos, original layers, masks, or
full-resolution exports. The visible intermediate images do not establish
realism, mask accuracy, authorship consent, or user preference. This note is a
workflow and code-scope review only; it is not a render, a usability study, or
a product proposal.

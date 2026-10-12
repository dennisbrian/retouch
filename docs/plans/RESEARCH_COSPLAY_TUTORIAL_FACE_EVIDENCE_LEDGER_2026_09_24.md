# Cosplay tutorial face evidence ledger — 2026-09-24

Documentation only. This ledger transcribes and classifies the face and
face-adjacent instructions in a focused set of photographed Photoshop tutorial
pages. It extends the [batch review](RESEARCH_COSPLAY_TUTORIAL_BATCH_REVIEW_2026_09_24.md).
It does not assess Retouch code, reproduce the tutorial edits, or validate
their visual quality.

## Source boundary and confidence key

The displayed indices map through the local
[`manifest.csv`](/Users/dennis/Downloads/research_review_batches_u65zuqfu/manifest.csv)
to the original screenshots under `~/Downloads/research/`. Each source is a
photograph of a printed tutorial page or a close crop of one. English text below
is a working translation or a visual summary, not a certified transcription.
UI values are reported only where legible; none are recommendations.

| Confidence | Meaning |
|---|---|
| High | The relevant Chinese instruction and its action are legible in the full screenshot. |
| Medium | The main action is visible, but some wording, scope, or UI context is small or cropped. |
| Low | The image supports only a visual description; the instruction cannot be reliably read. |

## Page ledger

| Index / source | Printed page | Working translation or direct observation | Intended effect in the tutorial | Review notes | Translation confidence |
|---|---:|---|---|---|---|
| [143 — MTXX_20170124150013.jpeg](/Users/dennis/Downloads/research/MTXX_20170124150013.jpeg) | 209 | In the character/effects section, restore the person layer, check it against the scene, and adjust the person's skin color. The next step says to isolate the skin area for color adjustment and protect fine areas such as the eyes. | Match the photographed cosplayer's skin to a dark cyberpunk composite. | The page displays a skin-area mask. The source does not show a controlled comparison or define an automatic mask. | Medium |
| [144 — MTXX_20170124150916.jpeg](/Users/dennis/Downloads/research/MTXX_20170124150916.jpeg) | 210 | After scene grading, add a small eye catchlight using an inverted Curves adjustment and a low-opacity white brush. The page then says the character's skin needs little further adjustment. | Add eye emphasis and finish the local skin pass. | The screenshot visibly includes scene-grade steps as well as the eye instruction; those controls should not be attributed to face retouch. | High |
| [145 — MTXX_20170124150927.jpeg](/Users/dennis/Downloads/research/MTXX_20170124150927.jpeg) | 210 | Close crop of the same page's color-grade controls, including a Hue/Saturation adjustment, a solid color layer, and Levels. | Make the cyberpunk scene darker and more cohesive. | Same printed page as 144; treat it as a detail view, not an independent method or result. Visible settings belong to this image and scene. | High for the visible controls; Medium for their full context |
| [167 — MTXX_20170124151650.jpeg](/Users/dennis/Downloads/research/MTXX_20170124151650.jpeg) | 226 | The tutorial outline includes person preparation, compositing, scene construction, effects, light mapping, and grading. Its post-processing introduction mentions basic face cleanup, skin texture adjustment, stronger eye light, and a small face-slimming edit. | Establish the desired stylized cosplay look before assembling the fantasy composite. | This is a process outline and example image; it does not give a reproducible face procedure or preservation result. | Medium |
| [168 — MTXX_20170124151713.jpeg](/Users/dennis/Downloads/research/MTXX_20170124151713.jpeg) | 227 | The visible steps refine the extracted subject and hair. The author says to follow the existing hair direction and shows additional strand work with layer-by-layer refinement. | Make the cutout and hair blend into the composite. | Face-adjacent boundary work, not evidence about skin retouch. I am not assigning a precise meaning to every small caption. | Medium |
| [219 — MTXX_20170124162225.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162225.jpeg) | 268 | Add red light to lit parts of the subject on a separate layer and use a mask/blend treatment to control it. | Add reflected environmental light to the costumed subject. | The small text supports the general local-light instruction; exact layer/blend wording is less certain. Not a face-only operation. | Medium |
| [220 — MTXX_20170124162243.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162243.jpeg) | 269 | Draw hair strands along the existing direction, then add scattered strands; use the existing hair path as a guide. | Refine the hair silhouette and make added strands fit the pose. | The sequence is clear and repeated in the diagrams. It describes manual drawing, not a validated automatic hair method. | High |
| [221 — MTXX_20170124162255.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162255.jpeg) | 270 | Add foreground and background elements. The example places white birds at different sizes and positions to suggest distance and movement. | Add depth and atmosphere around the subject. | Context for compositing only; the text itself warns that foreground elements need matching scale, angle, light, and blur. | High |
| [222 — MTXX_20170124162307.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162307.jpeg) | 271 | Add gold particles and clouds, arranging them with the scene's direction and separating effects by distance. | Build atmosphere and foreground/background depth. | No face procedure. | High |
| [223 — MTXX_20170124162334.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162334.jpeg) | 272 | Overall color and light adjustment: a warm color layer is blended into the image, followed by a selective adjustment to the subject. | Unify the composite while retaining localized control over the person. | The example exposes a visible blend/opacity value, but the full source and viewing conditions are unknown. Do not reuse the value as a recipe setting. | Medium |
| [224 — MTXX_20170124162349.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162349.jpeg) | 273 (likely) | Close crop of an adjustment panel and a soft painted mask. | Show adjustment controls; the target operation is unclear from this crop alone. | Likely companion detail capture for 225; it omits the printed context, so I do not attribute its settings specifically to the face. | Low for the operation; high for the visible panel and mask |
| [225 — MTXX_20170124162402.jpeg](/Users/dennis/Downloads/research/MTXX_20170124162402.jpeg) | 273 | The detail section says the face has a dark area that can be opened with local light. The face-adjustment step says to paint a highlight in the eye area and use Liquify to slim the face. | Brighten a dark facial area and apply an intentional beauty reshape in the stylized composite. | Same printed page as 224, captured wider. Eye highlight and face slimming are deliberate appearance changes; the page gives no natural-retouch or identity-preservation criterion. | High for the visible actions; Medium for the exact wording of the local-light rationale |

## Repeated-page groups

- **144 and 145:** printed page 210. Image 144 preserves the broader page;
  image 145 is a closer view of grading controls. They are one page of evidence.
- **224 and 225:** printed page 273. Image 225 preserves the wider context;
  image 224 is a closer view of the adjustment/mask controls. They are one page
  of evidence.

No exact repeated capture was identified among the other selected indices.
Repeated captures should not increase the count of independent techniques or
examples.

## Research interpretation

The pages support four separate hypotheses for future evaluation:

1. **Skin-color adjustment can be constrained to a selected skin region.**
   Image 143 shows that manual workflow. It does not establish face parsing
   accuracy, safe boundaries, or suitability across skin tones and lighting.
2. **Eye emphasis can be an intentional local operation.** Images 144 and 225
   show painted eye highlights. The screenshots do not establish whether the
   highlight is present in the original capture or whether adding it preserves
   the person's appearance.
3. **Hair cleanup depends on direction and boundary context.** Images 168 and
   220 show manual strands following the existing hair. That is a useful review
   cue for a future hair study, not evidence that generated strands are needed.
4. **Face reshaping is a separate beauty intent.** Images 167 and 225 mention
   face slimming. This should be evaluated separately from local skin or light
   correction, with explicit user intent and likeness review.

The images concern a photographed cosplayer incorporated into a strongly
stylized scene. They cannot establish natural face-retouch quality: the bundle
has no matched source/export pairs, no original layered files, and no independent
likeness, texture, makeup, or boundary review. Tutorial steps are observations
about the author's workflow, not quality outcomes.

## Next documentation gate

Before any Retouch experiment is proposed for these techniques, prepare a small
matched-source study record with:

- the authorized original and delivered output, paired by stable asset ID;
- the requested edit intent (color, lighting, eye emphasis, hair, or reshape);
- the source and export dimensions, color profile, and viewed scale;
- face, eye, skin, makeup, hair, costume, and mark regions that must be preserved;
- a no-op or current-workflow reference and a blinded human review for likeness,
  texture, artifacts, and whether the requested change was achieved;
- a clear outcome for abstention when the intended region or edit is ambiguous.

This is a proposed evidence record only. No source/output photo experiment has
been run for this tutorial bundle.

The tailored comparison arms and review sheet are now specified in the
[proposed transfer study](PLAN_COSPLAY_TUTORIAL_TRANSFER_STUDY_2026_09_24.md).
Photo collection and evaluation remain unperformed.

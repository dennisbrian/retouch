# Cosplay detail looks

Phase 4, 2026-10-23. These four optional presets inherit `natural_v2` and appear
as **Experimental / compare first** in the GUI. Stage lighting is the owner's
priority. They are available through CLI `--recipe` as well. None replaces the
natural default or enters the recommended list.

| Preset | Intended source | Added controls |
|---|---|---|
| `cosplay_stage_detail_v1` | Deep face shadows under venue/stage light | Face Shadow Fill 35; no added global white balance, hue neutralisation or grade |
| `cosplay_wig_contacts_v1` | Visible contacts and a detectable wig edge | Iris Pop 25, Wig Lace Blend 25; Wig Shine and extra Hair Shine off |
| `cosplay_paint_detail_v1` | Blue, green, grey or purple painted skin | Body Paint 45; smoothing 15; equalising and spot healing off; preserve-all mark policy |
| `cosplay_prosthetic_detail_v1` | Skin-toned forehead/ear appliance seams | Prosthetic Edge Blend 40; real-prosthetic acceptance still missing |

The numbers above are GUI/CLI strengths, not recipe ratios. Each look keeps the
base's no-added-brightening choice, no reshaping and full mole protection. Texture
restoration remains in its existing legacy mode; experimental texture modes are
not silently enabled by a styling choice.

```bash
./run batch /path/to/originals -o /path/to/stage-review --recipe cosplay_stage_detail_v1
```

## Stage lighting

The stage candidate fills only deep local face shadows, measured against the
face's own illumination. Ordinary modelling below the operation's knee remains.
It does not neutralise coloured lighting, relight the scene, add exposure to the
whole photo, or reconstruct clipped highlights. Intentional dramatic shadows can
still be weakened: compare to `natural_v2` and reduce Face Shadow Fill or use the
Protect brush where needed. Local protection retains detail while broad tone can
still carry through; see [Protect / Restore](PROTECT_AND_FACE_STRENGTH.md).

Two coloured-light mechanism controls check that the recipe's 35-strength fill
lifts the planted deep shadow while retaining linear-light chromaticity. Those
are simulated controls, not representative RGB-stage-light quality evidence.
The real-source review uses a matching camera RAW of a venue portrait. A venue
portrait is not a substitute for a diverse coloured-stage corpus.

## Makeup boundary

All four leave `makeup_coverage_even` and `makeup_cake_reduce` off. The underlying
makeup module offers conservative appearance edits and separately reference-
conditioned attenuation for limited translucent-colour categories. It does not
recover bare skin from an arbitrary single makeup photograph. These presets do
not remove foundation, distinguish every deliberate mark from a defect, repaint
makeup, or claim physical opacity recovery. Protect deliberate patterns yourself.

The paint preset disables spot healing and legacy blemish removal so paint
patterns are not deliberately sent through a pimple healer. Paint evenness is
still an appearance edit: deliberate patchy patterns can be evened too. Red,
orange and tan paint are not distinguishable from skin by the current detector.
Body paint touching an identically coloured costume can include that costume.

## Wig and prosthetic boundaries

Native review found an orange finger near the face when Wig Shine/deglare was
35. Isolated crop renders reproduced it with deglare alone, while the lace and
iris controls did not reproduce the recolouring. Wig Shine is therefore **off**
in this new composition. Existing recipes and the standalone slider are
unchanged. This is an unresolved material/ownership limitation, not a repaired
wig-shine algorithm.

Lace blending has evidence on planted seams/natural hairline controls. The native
cosplay sample has bangs, so it cannot demonstrate real lace removal. High bangs
inside the hairline zone may have their tips softened. Iris edits need visible
iris and eye-opening support and can abstain under heavy lash shadows.

Prosthetic blending was measured on simulated appliances only. Short seams,
perfectly colour-matched ridges and seams outside the forehead/ear zones can be
missed. No success on a real appliance is claimed by making this preset selectable.

## Validation record

- 207 recipe/cookbook/parameter checks passed, including both coloured-light
  controls. 95 existing operation tests passed for shadow fill, iris, wig shine,
  lace, body paint and prosthetic blending.
- A HEAD/current comparison found no changed GUI defaults or engine contexts
  among all 151 pre-existing recipes.
- The initial 5-recipe x 6-case sweep (base plus four candidates) passed all
  30 leak/object/face-count checks. Identity flags are reported separately and
  do not fail that harness. No candidate added a per-case identity flag over
  the base in that initial sweep.
- All five initially kept 55/56 planted marks. The white-wig case lost one mark
  under every recipe, including natural_v2. Several lightness flags and one base
  geometry flag were also present. This prevents an identity-safe certification.
- After removing deglare, the final wig composition passed its 6-case rerun.
  It added no identity flag over the same natural_v2 cases and retained the
  same 55/56 planted-mark result. The final native comparison was rendered and
  viewed; the previously orange finger no longer shows that recolouring.

The small hard cases failed to expose the native finger recolouring. Passing
that sweep is not sufficient for a recommendation. No reference snapshots were
approved or refreshed, and no preset is promoted from these observations.

## Native stage review and artifacts

The priority stage comparison uses the matching camera original
`/Users/dennis/Pictures/2026/2026-10-04/DSCF5053.RAF`. Its embedded thumbnail
was checked against the desktop JPEG before selecting it. The JPEG has no camera
metadata and is therefore only an auxiliary diagnostic; the RAW is the source
for this acceptance-oriented visual review. Both recipes processed the same
native 4170 x 6246 decoded frame, with one detected face. The matching original /
natural_v2 / stage-preset face crop was viewed. The stage look shows a restrained
lift in the darker face areas; the coloured contacts, glitter and pink makeup
remain visibly present. This is one venue portrait, not an RGB-stage benchmark
or an owner preference verdict. Times observed alongside other local checks:
natural_v2 73.45 s, stage preset 62.80 s; these are not performance comparisons.

The separate warm/cool synthetic controls each lifted their deep-shadow support
by about 0.276 stop at strength 35. Maximum linear-light chromaticity changes
were below 1.8e-7. This verifies the gain mechanism on those unclipped simulated
colours, not universal hue fidelity under clipping or arbitrary stage lighting.

Temporary local review artifacts:
- `/tmp/retouch-phase4-review/stage-raw/DSCF5053_face0_comparison.jpg`
- `/tmp/retouch-phase4-review/stage-raw/manifest.json`
- `/tmp/retouch-phase4-review/stage-controls.json`
- `/tmp/retouch-phase4-review/wig-final/DSCF5119_face1_comparison.jpg`
- `/tmp/retouch-phase4-review/isolation/isolate.jpg` — reproduced rejected deglare combination
- `/tmp/retouch-phase4-review/golden.json` — initial 30 runs
- `/tmp/retouch-phase4-review/golden-wig-final.json` — final wig rerun

Final status: four selectable experimental compositions, with stage-light
priority, documented makeup limits, and an excluded unsafe wig-shine combination.
No production recommendation or identity-safe certification. These source changes
remain local and uncommitted, alongside the earlier Phase 3 work.

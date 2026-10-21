# Face retouch quality roadmap

Date: 2026-10-22. Source inspected: `main` @ `291ea14`; recipe counts rechecked at `f26e30e`.
Status: plan and documentation only. No code, recipe, test or photo changes.
Everything under "where we stand" is read from source, recipe tables, commit
history and saved study documents. Nothing here is a new render or measurement.

## 1. Goal

Make Retouch the best tool for face retouching, judged by one thing: **the
person still looks like themselves, only better.**

Owner's choice (2026-10-22): **natural skin and preserved identity are the
default; cosplay styling is something the user turns on.** The other two
directions, polished beauty/cosplay looks and shoot-wide consistency, are
covered as later tracks (Phases 4 and 5) and are not the default.

"Best" uses the bar the owner already set on 2026-09-06 (graduation items
G1-G10, listed in the appendix). We win only if all three hold against strong
commercial tools:

1. Equal or higher blind preference.
2. Fewer identity, makeup and texture failures.
3. Less manual fixing.

Each phase below names the graduation items (G1-G10) it moves.

## 2. Where we stand

### Already strong (do not rebuild)

- **Many careful face ops.** About 18 opt-in face ops shipped 2026-09-25 to
  10-17 (spot heal, face shadow fill, powder finish, iris pop, teeth
  brightening, highlight repair, stray hair, bloodshot eyes and others). Each
  one measures against the face's own skin rather than fixed brightness
  levels, and was checked on simulated darker and lighter exposure.
- **Fairness pass done.** The 10-21 survey's whole tone-invariance table was
  fixed on 10-21: `02d8b6c` skin-protect chroma gate, `6b4dbd5` blemish,
  `d49860f` eye/hair specular, `a22ab2b` face paint, `430aa7a` chromophore.
- **Protection plumbing.** The repair mask is now clipped to eligible skin
  after morphology (`e66cf1e`). PatchMatch fallback stays inside its donor
  mask (`ffe5a4c`). Freckles are no longer healed out of preserved marks
  (`077b72b`). Mark policy and mole protect exist as controls.
- **Targeting.** The per-face recipe picker is anchored to face position, not
  detection order (`db8ae05`). Preview/export face-coordinate bugs were fixed
  2026-09-23 (`17377c8`, `bf70524`, `3e5d071`).
- **QA you can trust more.** QA flags were recalibrated against the input
  photo (09-25). Patch-scale waxy-cheek and colour-cast checks were added
  (09-27). The golden hard-case harness (`tests/golden/`, 10-19) checks leaks
  onto clips, props, wigs and poster faces across recipes.
- **Manual finishing.** The layer editor has spot healing, clone, texture-aware
  healing and editable recipe layers (10-14 to 10-21).

### Gaps that block "natural and still them"

| # | Gap | Evidence (source-observed) |
|---|---|---|
| 1 | **The default never picked up the newer ops.** Of the ~18 new face ops, only one is set by any of the 150 recipes: Skin Warmth, in two conditional cosplay looks added 2026-10-22 (`47322e5`). None of the 18 recommended recipes sets any of them, and neither does `natural`. Keeping existing looks unchanged was deliberate. But it means a default run uses none of the new ops. | Recipe table scan after rebase on `f26e30e`, 2026-10-22 |
| 2 | **Every recipe runs the legacy blemish remover at its smoothing strength.** `blemish` has no recipe key; it mirrors `frequency.smooth` × 100. That is 30 in `natural` and 55 in `cosplay_clear_v1`. In the 09-27 planted test (4 real faces plus simulated darker skin, strength 60), `spot_heal` fixed 31/60 pimples, kept 22/22 moles and changed 0.09% of other skin. Legacy blemish at the same strength scored 0/60, 1/22 and 1.77% on the same three measures. **Those numbers predate `6b4dbd5` (10-21), which changed legacy blemish. Re-measure before acting.** Nothing was measured at 30. | `params.py:451`, `params.py:3144`; commit `69b2dbf` |
| 3 | **Mark protection is off by default.** Defaults are `mark_policy='legacy'` and `mole_protect=0`. After inheritance, 10 recipes set a mark policy and 11 set mole protect; `natural` sets neither. | `params.py`; resolved recipe scan |
| 4 | **Identity is measured in pieces, never as one verdict.** `qa_detectors.run_all` already computes three things. `perceived_retouching_vector` (AA6 Kee-Farid) gives geometry stats from the engine's own reshape warp field, plus face SSIM and blur/sharpen. `detect_color_drift` gives skin hue and ΔE against the input. `detect_cam16_delta_e` gives CAM16 colour difference. AA6 is informational only (score fixed at 0, never flags). It sees only the warps the engine reports, so it cannot catch features moved or softened by other ops. What is missing: a landmark check on the output itself, mark survival, per-face lightness shift, and one per-face identity result in golden QA. | `qa_detectors.py:259`, `:1181`, `:1959`; `tests/golden/README.md` |
| 5 | **`natural` lightens skin by default** (`skin.rosy` 0.1, so `whiten` = 10). That may be the intended look, but it changes skin tone on every photo, which is part of identity. | `recipes.py` `natural` |
| 6 | **The user can't paint "keep this".** The engine accepts `freckle_preserve_mask`, but only from Python. It has no CLI flag, `--set` can't carry an array, and it is left out of `PROCESS_INPUT_KEYS`. Even when it is set, only freckle healing reads it (`perf_optimizations.py:696`). Capture One ships this as a Blemish Protection brush. | `params.py:332`, `params.py:2897`, `cli.py:1355` |
| 7 | **Texture restoration is still provisional.** `fa02_texture_mode` defaults to `legacy`. The eligibility thresholds are labelled placeholders, and FA-02 is blocked on mark semantics (G3). | 09-15 gap report §2 |
| 8 | **No evaluation set.** Corpus schema v3 exists, but no real subject is assigned to calibration or locked_test. The ten pilot manifests are all `dev`, with 8 people and only light/very-light tones. | 09-15 gap report §2.4; corpus memory |

Gaps 1-3 and 5 are not bugs. Changing them changes every user's output, so
each needs owner approval (section 5).

## 3. Roadmap

Order matters. **Measure first, then change the default, then add control.**
Adding more sliders before Phase 0 would repeat the September pattern: good
ops that nobody's default uses.

### Phase 0: Identity scorecard and evaluation set (G5, G9, G10)

Build the ruler before moving anything.

**0a. Identity scorecard.** Reuse what `run_all` already computes; add only
what is missing. No new model dependency.

| Measure | How | Why |
|---|---|---|
| Face geometry drift | **New.** Re-run MediaPipe landmarks on the output and compare scale-normalised distances (eye spacing, nose width, jaw, lip shape) with the input. When all reshape params are 0, drift should be about 0. Report it next to AA6's warp-field statistics, which only cover intended reshaping. | Catches features moved by any op, not just reshape |
| Skin tone shift | **Extend.** `detect_color_drift` already gives hue and ΔE per region. Add per-face lightness change in stops. | Separates "evened" from "lightened or re-coloured" |
| Detail and blur | **Reuse.** AA6 face SSIM and the signed high-pass change D. Calibrate them so they can flag, not just inform. | Already computed; nothing acts on it yet |
| Mark survival | Labelled or planted moles/freckles still present after the render | Natural marks are identity |
| Texture retention | Existing patch-scale `plastic_skin` texture ratio | Waxy skin reads as "not them" |
| Makeup fidelity | Change in a*/b* inside lips/liner/blush areas versus the requested edit | Cosplay makeup is deliberate |
| Leak and abstain | Existing golden-QA leak checks, plus a count of faces skipped | Already built; report it next to the rest |

Wire it into `scripts/qa/golden_qa.py` as extra columns, not a new framework.
A face-embedding likeness score (ArcFace-style) can come later, as a separate,
owner-approved dependency. The pinned `mediapipe==0.10.5` and `protobuf<4`
make new model runtimes risky (see the RetinaFace note in CLAUDE.md).

**0b. Evaluation set.** Populate corpus v3 `dev` and `calibration` from real
shoots, with every person kept to one split. Fill the missing strata first:
darker skin, glasses, facial hair, group shots, colour stage light, high ISO.
This needs the owner's person mapping (section 5).

Exit: one table per recipe showing every scorecard measure on the golden cases plus
the dev set, with today's `natural` and `cosplay_clear_v1` as the baseline.

### Phase 1: Natural default v2 (G1, G2, G9)

Build a candidate `natural_v2` from ops that already exist. Do not change
`natural` until the owner picks.

1. Re-measure legacy blemish (post-`6b4dbd5`) against `spot_heal` at the
   strengths recipes actually use (30, 55), on planted spots and real dev
   faces.
2. If `spot_heal` still wins, give `natural_v2` `spot_heal` and set legacy
   blemish to 0. This needs a recipe path for `blemish`, which today only
   mirrors smooth.
3. Turn on mark protection (`mark_policy` + `mole_protect`) in `natural_v2`.
4. Decide skin lightening for `natural_v2` (owner call): keep 0.1, reduce, or
   leave it to an explicit "brighten" choice.
5. Consider the gentle, identity-safe ops: Face Shadow Fill (it lifts only
   deep shade and leaves modelling alone), Keep Nose Shape, low Powder Finish.
   Add each one only if the Phase 0 scorecard does not get worse.
6. Owner blind A/B on dev faces: `natural` vs `natural_v2`, full resolution,
   with face crops viewed. Do not judge from deltas alone.

Exit: the owner approves `natural_v2` as the default, or records why not.

### Phase 2: Control the user can trust (G1, G2, G9)

Fix the mistakes the automation will still make, quickly.

- **Protect brush.** Paint over a mark or a makeup detail and it survives the
  whole render. Wire the existing `freckle_preserve_mask` through the GUI. Then
  make it a source-anchored protection that every stage honours, not just
  freckle healing. This is the protection-lifecycle "Next" item.
- **Restore brush.** Paint back the original in one area, like Lightroom's
  Restore. First check whether a masked original layer in the layer editor
  already does this. If it does, the gap is a one-step path from the main
  GUI result.
- **One strength per face.** A single "how much" control per face on top of
  its recipe, like Capture One's Impact slider. The position-anchored per-face
  recipe picker and per-face JSON overrides already exist (`gui.py:1724`).
  The gap is a friendly control, not the plumbing.
- **Say what was skipped.** When a face or op abstains (eye gate, painted
  face, low confidence), show why on the face. The engine records Safe Auto
  and P7 decisions. As of the 09-15 report (§2.5), the preview payload carried
  Safe Auto decisions but not P7 diagnostics; re-check this before building.

### Phase 3: Real texture (G3)

Only after Phase 2's protections exist and the dev set has reviewed mark
labels. With supports held fixed, compare restoration off, `legacy` and the
existing FA-02 arms. Score them with the Phase 0 texture and mark-survival
columns. The success test: pores and fine hair come back, while removed
spots, noise and makeup edges do not. The 09-15 plan already specifies this
study; run it then.

### Phase 4: Cosplay styling, as an explicit choice (G4, G6)

Cosplay looks become a layer on top of `natural_v2`, not a separate world.

- Compose the new ops into a small set of cosplay looks: painted face, wig
  plus contacts, prosthetics, stage light. These ops are wig shine, wig
  hairline, iris pop, bloodshot eyes, teeth brightening, nose highlight, nose
  tip blush, powder finish, costume shine/clarity, lint cleanup and prosthetic
  blend. Today they are only reachable one slider at a time.
- Makeup handling (G4): either make `makeup_unmix` useful, or write down what
  the system will not attempt.
- Each look must pass the Phase 0 scorecard for the parts it doesn't mean to
  change. Example: a "porcelain" look may lighten skin, but must not move
  features or remove marks.

### Phase 5: Same person, same look, whole shoot

This is a real advantage over Capture One. Its per-face edits are not saved
in presets, and copy/apply loses them when face order changes. Retouch
anchors per-face choices by position within one photo.

- Carry a person's per-face settings and protections to their other frames in
  a shoot. Match the person across frames, then reuse their settings.
- `set_match` already measures face skin (`face_skin_mask` from landmarks)
  and matches every frame to one hero frame. Extend it to match **per
  person**, so each person keeps their own skin tone across a set instead of
  everyone being pulled toward the hero frame's subject.
- Phase 0's skin-tone shift measure doubles as a shoot-consistency check.

### Phase 6: Final exam (G8, G10)

Use 100-200 never-seen portraits on locked_test, covering every stratum.
Process them with one click. Run a blind A/B against Meitu, Lightroom or
Capture One assisted output, and a human retoucher. Report preference, the
Phase 0 failure counts and manual-fix time. Nothing is "done" before this.

## 4. Not doing (guardrails)

- **No generative face restoration by default.** Models like CodeFormer
  rebuild facial detail from a learned codebook, which works against
  "still them". CodeFormer is also under the NTU S-Lab License 1.0.
  At most it could be an opt-in repair for badly blurred faces.
- **No reshaping in the natural path.** Reshape defaults are 0. The 11 recipes
  that reshape are style recipes and stay explicit.
- **No absolute brightness thresholds** on skin-scaled signals (house rule).
- **No more single-purpose sliders until Phase 1 lands.** The bottleneck is
  defaults and composition, not the number of ops.

## 5. Decisions needed from the owner

1. **Changing the default look.** Should `natural_v2` replace `natural`, or
   ship next to it? Either way, every user's output changes.
2. **Skin lightening in the natural default:** keep `rosy` 0.1, reduce it, or
   make it opt-in.
3. **Evaluation people.** Which shoots are the same person, which people are
   still untouched (for locked_test), and whether outside or public photos
   are allowed for the missing strata.
4. **Face-likeness model.** Is a later embedding dependency acceptable, given
   the mediapipe/protobuf pins?
5. **Cosplay looks.** Which three or four looks matter most for Phase 4?

## 6. Sources

Internal:
- [Remaining gaps, 2026-09-15](RESEARCH_RETOUCH_REMAINING_GAPS_2026_09_15.md),
  [next-study plan](PLAN_RETOUCH_GAP_RESEARCH_2026_09_15.md)
- [Protection lifecycle, 2026-09-22](RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22.md)
- [Target and preview parity, 2026-09-23](RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23.md)
- [Codebase survey, 2026-10-21](../SURVEY_MASTER_REPORT_2026_10_21.md). Its
  fairness table and most of its face items were fixed on 10-21; see §2 above.
- `tests/golden/README.md`; CLAUDE.md "Outstanding Fixes" (spot heal and other
  op entries)

External (vendor docs, read 2026-10-22; vendor features, not measured quality):
- Capture One, Retouch Face Skin tool: per-face or all-faces edits, an Impact
  master slider, a Blemish Protection brush, a neck toggle. Per-face edits
  are not saved in presets and are lost when face order changes.
  https://support.captureone.com/hc/en-us/articles/27336176639133-Retouch-Face-Skin-tool
- Lightroom (web), Quick Actions Skin: per-person selection, Amount and Fade,
  Remove and Restore brushes.
  https://helpx.adobe.com/lightroom/web/edit-photos/apply-actions/remove-skin-blemishes-using-quick-actions.html
- Lightroom masking, Select People: per-person skin, hair, teeth and other
  masks. https://helpx.adobe.com/lightroom/desktop/edit-photos/masking.html
- Evoto, skin retouching: Even and Sculpt dodge & burn, textured smoothing,
  frequency separation. https://support.evoto.ai/portrait-retouching-module-skin-retouching/
- Retouch4me Dodge&Burn: automatic dodge & burn, exported as a Soft Light
  layer for non-destructive editing. https://tinacms.retouch4.me/dodgeburn
- CodeFormer: NTU S-Lab License 1.0. https://github.com/sczhou/codeformer

## Appendix: graduation items (owner, 2026-09-06)

Short form. The full wording is in the owner's notes.

| # | Item |
|---|---|
| G1 | FA-01 closure: corpus, ontology and decision record finished, or explicitly deferred; SegFace-vs-BiSeNet comparison done or deferred |
| G2 | FA-03 production mark classifier, beyond the tie-break fix, validated on a real reviewed corpus (near-eye marks, heavy makeup, face paint, facial hair, profiles, small faces) |
| G3 | FA-02 texture restoration that keeps pores and fine hair without bringing back defects, noise or makeup edges (after G2) |
| G4 | Makeup handling: make makeup unmix useful, or document what the system won't attempt |
| G5 | Hard cases: groups, occlusion, glasses, wigs, facial hair, accessories, colour light, full skin-tone range, high ISO, soft focus |
| G6 | Eyes, teeth and lips finished, if the claim is "full face engine" |
| G7 | Float32 pipeline: reconcile the conflicting documentation on remaining uint8 round-trips |
| G8 | Blind benchmark against Meitu, Lightroom/Photoshop-assisted output and a human retoucher, on the same sources with matched intent |
| G9 | Measure failure, not just beauty: mark removal, makeup damage, halos, plastic skin, geometry drift, abstention, texture loss, false blemishes, manual fixes |
| G10 | Locked real test set with subject-separated dev/calibration/test splits and no tuning on test |

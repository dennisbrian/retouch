# Over-smoothing check: patch-scale QA (2026-09-27)

**Question.** The 2026-09-25 recalibration
(`RESEARCH_QA_FLAG_CALIBRATION_2026_09_25.md`) left two gaps: one
over-smoothed cheek was caught on 4 of 7 photos, and a colour cast on part
of a small face on 5 of 7. Why do local defects slip through, and what
catches them without flagging clean renders?

**Answer.** Both checks averaged over too much. `plastic_skin` measured
texture retention over the whole face skin, so one waxy cheek moved the
average by a few points. `color_drift` measured hue over the whole person
mask, so a cast on the chin was a few percent of its pixels. Both now also
scan cheek-sized patches of each face and report the worst patch, where it
is, and a plain-language message. Nothing in the render changes: these are
review-page checks only (`--fail-on-qa` still gates on them if you opt in).

## What changed

| Check | New patch measurement | Flags when |
|---|---|---|
| `plastic_skin` (review page: "Waxy skin") | For each face of 200 px or more: fine-texture energy (L − G_σ2(L), squared) in a Gaussian window of σ = 5% of the face box, output / input. Windows mostly outside face skin, with too little input texture (< 0.35 × the face's median), or centred on the nose are skipped. | worst patch keeps < 60% (same limit as the whole face) |
| `color_drift` (review page: "Skin colour shift") | For each face in the face-skin mask: the output − input shift in Lab a/b, minus that face's median shift, averaged in a window of σ = 8% of the face box. An even grade scores ~0. | worst patch > 6 ΔE |

New evidence keys: `patch_retention` / `patch_delta_ab`, `patch_zone`
("cheek on the left of the photo", "forehead", "chin", ...), `patch_point`,
`patch_face_bbox`, and `finding`, which becomes the QA message on the review
page (for example "Waxy patch on the cheek on the right of the photo: it
keeps 43% of its skin texture, the rest of the face keeps 94%"). The review
page also shows plain names for the four skin checks.

Why the exclusions:
- **Faces under 200 px.** On a 146-px face (kit_harington) the σ2 band holds
  the nose wings and eyelid creases, not pores; ordinary smoothing read
  0.37–0.53 on 6 of 8 recipes. Such faces keep the whole-face check.
- **Nose.** Shine removal flattens the nose tip on purpose.
- **Colour limit 6 ΔE.** A recipe blush is a deliberate pink cheek patch;
  the strongest (blush 40, `zzz_anime_v2`) reads up to 5.6.

## Corpus

9 photos: the 7 of the 2026-09-25 study (`obama`, `biden`, `two_people`,
three `knn_examples/train` portraits from ageitgey/face_recognition, the
`chang_e_cosplay_tamed_shine` tile) plus Alex's two bunny-suit photos
(DSCF3503, DSCF3518: white face paint, peach nose, long wig with bangs),
rendered at 2600 px long side so their faces are 250–330 px. 18 recipes:
the 5 of the last study, 3 porcelain looks (`meitu_porcelain_v1`,
`cosplay_powder_v1`, `korean_glass_clear_v1`), 4 blush-25/30 looks
(`pink_dream`, `xiaohongshu`, `idol`, `wedding`) and 6 blush-30/40 or anime
looks (`zzz_anime_v1`, `zzz_anime_v2`, `scifi_cosplay`, `fantasy_goddess`,
`xhs_soft_glow`, `anime_cinematic_v1`). 127 renders through
`RetouchEngine.process`, with `run_qa_with_evidence`'s inputs captured.
Darker skin: `obama` only.

## Clean renders

| | Patch check result |
|---|---|
| Texture, 8 everyday/cosplay/porcelain recipes | worst patch ≥ 0.62 (`meitu_porcelain_v1` on rose); 0 new flags |
| Texture, heavy looks | 9 new flags: `zzz_anime_v1`/`v2` (alex, obama, 0.56–0.59), `xhs_soft_glow` (obama forehead, 0.42), `fantasy_goddess` (rose, 0.56), and 3 renders the colour or whole-face check already flagged (`scifi_cosplay` ×2, `pink_dream` on obama). Viewed: the obama foreheads are whitened flat against the hairline. |
| Colour, all 127 | 0 new flags. Worst patch outside renders already flagged: 5.6 (`zzz_anime_v2`, blush 40); blush 25 reads ≤ 3.7; everyday recipes ≤ 3.3 |

## Planted defects

Planted on each photo's `natural` render. "Old" = the checks before this
change (whole-face `plastic_skin` or `asymmetry` for texture; person-mask
`color_drift` for colour).

| Defect | New | Old |
|---|---|---|
| One cheek blurred σ=1.5 (either cheek) | 12/12 faces scanned (0.30–0.54) | 0/18 |
| One cheek blurred σ=3 | 12/12 | 0/18 |
| One cheek blurred σ=1 | 9/12 | 0/18 |
| Lower half of the face, hue +25° | 5/9 | 3/9 |
| One cheek, hue +25° | 5/9 | 1/9 |
| Lower half, −8 a (green) | 9/9 | 1/9 |
| One cheek, +8 a (pink) | 1/9 | 1/9 |
| Lower half, +6 b (yellow) | 1/9 | 1/9 |

Cheek blurs on the three faces under 200 px (`kit_harington`,
`chang_e` tile, `two_people`) are not scanned. The 25° hue casts are missed
on Alex's white-painted faces (a hue turn on near-white paint moves the
colour by only 2.4–3.5 ΔE, barely visible) and on the two small faces.

## Limits

- A pink or yellow cast of about 6–8 ΔE on one cheek looks the same as a
  blush and is not flagged; the limit is set by the strongest recipe blush.
- The face-skin mask on main still includes full bangs on the fallback path
  (fixed by PR #32); a smoothed fringe can read as a waxy forehead there.
- Real defects in the corpus are planted, not found; no real batch with a
  known waxy cheek has been checked.

Code: `retouch/qa_detectors.py` (`_patch_texture_retention`,
`_patch_chroma_cast`, `face_zone_name`); tests:
`tests/test_over_smoothing_check.py`.

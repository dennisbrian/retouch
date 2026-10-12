# Core Recipe Corpus and Human Review Template

Copy this template to the access-controlled certification output location.
Do not commit portrait sources, identity details, consent records, or private
paths here. The runner verifies only semantic case names and rendered output;
this record provides the human context the machine cannot infer.

## Corpus inventory

One consented, representative image is required for every row. Extra tags are
encouraged when a case exercises more than one risk.

| Case name used with `--case` | Required condition | Asset ID or private location | Consent/authority reference | Expected risks |
| --- | --- | --- | --- | --- |
| `skin_tone_*` | Skin-tone coverage |  |  | tone shift, masking, texture |
| `*_lighting` | Low, mixed, or difficult lighting |  |  | exposure, white balance, under-eye |
| `glasses_*` | Eyeglasses/reflections |  |  | eye mask, glare, frame edge |
| `wig_*` | Wig, hairline, or cosplay hair |  |  | wig lace, hair/skin boundary |
| `hands_on_face_*` | Hand/prop face occlusion |  |  | landmark loss, healing boundary |
| `group_portrait_*` | Multiple faces |  |  | per-face isolation, unequal exposure |

## Automatic evidence

- Matrix output directory:
- `matrix_manifest.json` automatic pass:
- Face-aware runs (`global_only: false`) verified:
- Missing or failed recipe outputs:
- Runtime/provider diagnostics retained:

## Human Natural Output review

Use the generated `human_review.md` and `human_review.json` in the matrix
output. For every case × Core recipe, inspect the contact sheet at 100% and
mark `approved`, `rejected`, or `pending` in JSON. An approval must include a
reviewer name/ID and notes for any visible trade-off.

Review specifically for:

- natural skin texture and marks; no demographic or lighting-specific washout;
- eyes, glasses, hair/wig, hands, and face boundaries;
- no unsupported optional-model claim or unreviewed Safe Auto decision;
- all faces in group portraits treated independently and plausibly.

Run the final, non-rendering gate only after completing the JSON worksheet:

```bash
python scripts/recipes/finalize_core_recipe_certification.py /private/path/to/core-cert
```

`final_certified=true` is possible only when the corpus is complete, all
face-aware automatic checks pass, and every case × recipe review is approved.

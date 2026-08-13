# Core recipe certification — 2026-08-12

The release certification scope is deliberately limited to 12 recipes from
the current catalog. The selection is correction-first and includes baseline
portrait, mature skin, mark preservation, pore/detail restraint, under-eye
correction, event light, and cosplay/wig detail. The authoritative list is in
`retouch/core_recipes.py`.

Certification has two separate gates:

1. Automatic QA: run `scripts/recipes/core_recipe_certification.py` against
   named corpus cases. The default is face-aware and every case manifest must
   contain `global_only: false`, 12 completed outputs, QA rows, and a non-empty
   contact sheet.
2. Human Natural Output review: inspect the generated contact sheets and fill
   `human_review.json`. Review cases must cover skin tones, lighting, glasses,
   wigs, hands-on-face, and group portraits. A machine pass is not a human
   approval.

Every case and matrix manifest separates `face_aware_run`, `automatic_pass`,
`human_review`, and `final_certified`. The legacy
`face_aware_certification` field is retained only as a compatibility alias
for `final_certified`; it is never inferred merely from omitting
`--global-only`.

Example:

```bash
python scripts/recipes/core_recipe_certification.py \
  --case skin_tone_dark=/path/to/skin_tone_dark.jpg \
  --case mixed_lighting=/path/to/mixed_lighting.jpg \
  --case glasses=/path/to/glasses.jpg \
  --case wig=/path/to/wig.jpg \
  --case hands_on_face=/path/to/hands_on_face.jpg \
  --case group_portrait=/path/to/group_portrait.jpg \
  --output /tmp/retouch-core-cert
```

The runner requires these six semantic strata by default. Use
`--allow-incomplete-corpus` only for a diagnostic matrix; that matrix records
`corpus_complete: false`, cannot pass the automatic gate, and cannot be final
certified.

Each matrix output now includes both the machine-readable
`human_review.json` and an inspectable `human_review.md`. Complete the JSON
from the worksheet, then run:

```bash
python scripts/recipes/finalize_core_recipe_certification.py /path/to/core-cert
```

The finaliser does not re-render or change source/output images; it only
recomputes per-case human-review state and the final certification gate.
Use [the corpus and review template](CORE_RECIPE_CORPUS_AND_HUMAN_REVIEW_TEMPLATE.md)
to record consented private assets and reviewer evidence outside the repository.

`--global-only` is retained for headless smoke testing only and marks the
result non-certifying in `matrix_manifest.json`. The aggregator rejects
missing outputs, failed cases, empty contact sheets, global-only evidence,
and pending human review.

## Current status

The certification harness and review worksheet are implemented. The
representative six-stratum corpus gate is now executable, but no release claim
is made until all six cases produce 12 face-aware outputs each and the human
natural-output worksheet is reviewed. The pinned one-case sample remains a
diagnostic reference, not a representative automatic pass. The default system
environment remains a blocked diagnostic because of `DrishtiMetalHelper`
initialization.

The pinned sample evidence is available at
`/private/tmp/retouch-core-cert-working/matrix_manifest.json`; it records one
face-aware case, 12 rendered Core recipes, incomplete corpus coverage, and
`final_certified: false` pending corpus completion and human review.

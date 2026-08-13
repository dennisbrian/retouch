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
  --case skin_tone_a=/path/to/skin_tone_a.jpg \
  --case glasses=/path/to/glasses.jpg \
  --case group=/path/to/group.jpg \
  --output /tmp/retouch-core-cert
```

`--global-only` is retained for headless smoke testing only and marks the
result non-certifying in `matrix_manifest.json`. The aggregator rejects
missing outputs, failed cases, empty contact sheets, global-only evidence,
and pending human review.

## Current status

The certification harness and review worksheet are implemented. The pinned
MediaPipe 0.10.5 CPU/XNNPACK environment produced 12 face-aware recipe
renders and an automatic matrix pass for the sample case. Final certification
remains pending until the human natural-output worksheet is reviewed. The
default system environment remains a blocked diagnostic because of
`DrishtiMetalHelper` initialization.

The pinned sample evidence is available at
`/private/tmp/retouch-core-cert-working/matrix_manifest.json`; it records one
face-aware case, 12 rendered Core recipes, `automatic_pass: true`, and
`final_certified: false` pending human review.

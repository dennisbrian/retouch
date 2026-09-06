---
name: corpus-report
description: Validate a retouch.corpus_manifest v3 manifest and report per-split subject counts and coverage. Use when the user asks to check, validate, or report on a corpus/dataset manifest, or before promoting any subject toward the locked_test split.
---

# Corpus split report

1. Ask for the manifest path if not given (a JSON file matching `retouch/corpus_manifest.py`'s v3 schema: `schema_version: 3`, `assets: [...]`, each asset with `person_id`/`person_ids`, `split` in `dev`/`calibration`/`locked_test`).
2. Run:
   ```
   .venv/bin/python scripts/qa/corpus_split_report.py <manifest.json>
   ```
3. Report the output plainly: `valid` status, per-split asset/subject counts, coverage by split, and any errors verbatim (they're structured — surface the `code`/`asset_id`/`field` for each, don't paraphrase them away).
4. If the manifest is invalid, stop there — do not attempt to auto-fix labels, invent `person_id` values, or guess strata to make it pass. Every field this schema requires (person_id, label_validation, strata) exists specifically so a human confirms it; silently patching one defeats the point.

## Hard rule: never populate `locked_test`

Do not add, move, or suggest moving any real asset into the `locked_test` split unless the user has explicitly confirmed, in this conversation, that the specific person(s) in that asset have never appeared in any prior study, render, test fixture, or screenshot. See [[graduation-criteria-2026-09-06]] and [[corpus-governance-v3-2026-09-06]] in project memory for why: this repo's existing photo corpus is a small number of real people (~5-15) reused across dozens of shoots, and 76+ individual images are already burned by prior research. A `person_id` crossing from `dev`/`calibration` into `locked_test` silently destroys the "never seen" guarantee the whole split exists to provide, and the schema's `person_crosses_split` check only catches it if the same `person_id` string is used consistently — it cannot catch a mis-identified person.

If asked to build or extend a manifest, default every new asset to `split: "dev"` unless told otherwise.

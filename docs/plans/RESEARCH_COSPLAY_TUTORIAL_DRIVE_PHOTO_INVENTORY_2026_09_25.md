# Google Drive Photography inventory for the cosplay tutorial study — 2026-09-25

**Scope:** Read-only inventory of the locally synced `Google Drive > My Drive >
Photography` folder to find possible references and matched source/output sets
for the documentation-first cosplay tutorial study. This records folder names,
image-file counts, and filename overlap only. No image pixels were opened,
rendered, processed, or scored; no production code or Drive files were changed.

## Inventory result

The locally synced `My Drive/Photography` folder contains 80 top-level folders
and 12,352 image-file entries under them, counting JPG, JPEG, PNG, HEIC, and
RAF files. These are file entries, not unique photographs:
the folder includes full-resolution copies, retouched copies, and repeated
shoot collections. The `Reference` folder contains 57 image files.

The browser sign-in flow requested a passkey, so this inventory used the
already-mounted local Drive mirror. It did not complete or change account
sign-in.

## Candidate matched sets

The following folders have enough filename overlap to shortlist for a later
source/output provenance check. A matching basename alone does not prove that
two files are the same exposure, that one is the source of the other, or which
processing stage an output represents.

| Candidate folders | File counts and overlap | Relevance and limit |
|---|---:|---|
| `duotian nikke` → `duotian_nikke_retouched` | 458 images in each; 458 basenames match after removing the `_retouched` output suffix. | Most directly aligned by folder name with cosplay portraits. The actual subjects, lighting, boundaries, and output stage have not been visually checked. |
| `2026-09-19` → `2026-09-19_retouched` | 206 images in each; 206 exact basenames match. | A large candidate pair set; folder names alone do not establish the shoot or processing method. |
| `2026-09-20` → `2026-09-20_retouched` | 233 images in each; 233 exact basenames match. | A large candidate pair set. The existing batch manifest separately refers to three inputs in this source folder, but that prior run is not evidence about the tutorial techniques. |
| `arisaff47` → `arisaff47improved` | 156 and 151 images; 149 exact basenames match. `arisaff47improved_fullres` has the same 151 basenames as `arisaff47improved`. | Candidate originals and improved exports; stage and relationship still require confirmation. |
| `arisa cosmic` → `arisa cosmic_retouched` | 55 and 31 images; 31 exact basenames match. `arisa cosmic_fullres` has 55 exact basenames matching `arisa cosmic`. | Some output coverage appears incomplete by filename; the full-resolution folder may be a duplicate rendition. |
| `ccram2026love` → `ccramfinal` | 269 images in each; 269 exact basenames match recursively, including nested folders. | Candidate pair set; source/output stage is not established by naming. |
| `original photo` → `editcc` | 699 and 592 images; 592 exact basenames match. | Potential originals and edits; the broad folder names do not identify the tutorial-study cases. |

The counts use case-insensitive filenames. For the `duotian` set, the output
suffix was stripped before matching. These are shortlist counts, not verified
pair counts. The presence of an `_improved`, `_retouched`, `_final`, or `edit`
folder is not by itself a quality label.

## Existing test artifacts

`retouch-photo-test-2026-09-24/batch_manifest.json` records three inputs from
`2026-09-20`, with each marked `passed` by that earlier run. The manifest points
to outputs under `/private/tmp/retouch-photo-test-2026-09-24`. This inventory
does not inspect those outputs or treat the run status as a visual-quality
judgment. It is a lead for provenance review only.

## Research use and next step

The folder removes the earlier blocker of having no local candidate pairs. The
best metadata-based shortlist is `duotian nikke` and
`duotian_nikke_retouched`, followed by either dated source/retouched set. Before
using any set in the proposed transfer study, confirm the pair identity, stage,
dimensions, source/output provenance, and requested edit intent from a small
number of cases. Then screen the cases against the plan's coverage criteria
(lighting, face direction, hair/costume boundaries, makeup/marks, and people in
frame), recording exclusions. Do not count filename-matched folders as
successful retouch evidence.

See the [transfer-study plan](PLAN_COSPLAY_TUTORIAL_TRANSFER_STUDY_2026_09_24.md)
for the comparison protocol and the [batch review](RESEARCH_COSPLAY_TUTORIAL_BATCH_REVIEW_2026_09_24.md)
for the original tutorial evidence. This inventory does not authorize photo
processing, new renders, or production changes.

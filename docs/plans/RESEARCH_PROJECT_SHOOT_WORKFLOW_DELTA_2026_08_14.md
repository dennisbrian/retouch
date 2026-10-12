# Research Delta — Project / Shoot Workflow

**Date:** 2026-08-14
**Status:** Research complete; implementation is not authorized by this note
**Baseline:** `RESEARCH_NEXT_FEATURE_FRONTIER_2026_08_12.md` plus the current
`feat/color-science-k9-fix-and-frontier` worktree at `383ada8`
**Method:** Current repository audit, focused Python 3.9 tests, and current
official product/standards documentation

## 1. Executive finding

The original frontier is still directionally correct, but its implementation
status is stale. Retouch now has useful foundations for burst grouping,
capture-quality ranking, watch-folder state, project-profile storage, a generic
dependency graph, and a weighted Look Board. These are primitives, not yet a
complete Project / Shoot workflow.

The next highest-value tranche is a **reviewable shoot manifest**:

1. persist burst membership, per-image evidence, and human select/reject/hold
   decisions;
2. export non-destructive ratings/labels or a selection manifest;
3. add per-face quality evidence with an explicit `uncertain` state;
4. connect the watch folder to real preview/final jobs without marking no-op
   callbacks complete; and
5. make project profiles apply only to user-confirmed subject/image links.

Professional handoff export should follow this tranche. C2PA signing should be
a separate integration because it requires a supported SDK/runtime and a real
key-management policy.

## 2. Corrected implementation status

| ID | Feature | Current repository status | Remaining product gap | Verdict |
|---|---|---|---|---|
| W1 | Assisted culling | Timestamp/camera/lens/fingerprint burst grouping and capture-only ranking exist | No face/eye evidence, uncertainty, persistent decisions, overrides, ratings, selection export, or corpus metrics | **Foundation; next priority** |
| W2 | Subject-linked profiles | Atomic profile records store recipe, style, params, protected marks, and reference paths | No detected-face-to-subject link model, confirmation UI, unlink/delete workflow, or application to per-face processing | **Data model only** |
| W3 | Watch folder | Stable-file detection, retry state, atomic persistence, recursion, and limits exist | GUI uses a no-op processor and reports records as processed; no preview/final job or output contract | **Plumbing only; fix status truth first** |
| W4 | Edit dependency/status | Serializable dependency graph and runtime statuses exist | No source/model/parameter/mask hashes, invalidation rules, session binding, or update-selected/update-all execution | **Generic graph only** |
| W5 | Handoff export | Flattened export and Advanced Retouch edit/session data exist | No mask package, neutral D&B layer, stage manifest, reconstruction test, PSD/PSB alpha channels, or ZIP handoff | **Not implemented** |
| W6 | Multi-reference Look Board | Board persistence, weighted extraction, GUI apply-to-Process, and editable engine params exist | GUI weights all references equally; no similarity weighting, group toggles, separate normalization/creative impact, or mixed-light validation | **Functional v0** |
| T1 | C2PA export provenance | JPEG APP11 detection and byte passthrough exist | No validation, parent ingredient, edit intent/actions, new claim, signing, or post-write verification | **Detection/passthrough only** |

## 3. Repository evidence and important corrections

### W1 — culling is capture-quality-only

`retouch/shoot_intelligence.py` deliberately ranks only Laplacian sharpness,
mean exposure, clipping, and resolution. It explicitly does not inspect face
identity, expression, eyes, hands, hair, or likeness. The GUI displays these
ranks as read-only rows and leaves `human_cull_review` ready in the graph.

This is a sound safe base, but it does not yet match the current review
contract users see elsewhere. Lightroom Assisted Culling now exposes subject
sharpness, eye sharpness, eyes-open state (including “can't tell”), exposure
issues, per-face evidence, manual select/reject overrides, and batch metadata
actions. Retouch should adopt the interaction contract, not copy an opaque
score.

Source: [Adobe Lightroom Assisted Culling](https://helpx.adobe.com/lightroom/desktop/organize-photos/assisted-culling.html)

### W2 — profiles are not yet subject links

`ProjectProfile.subject_key` is currently user-entered metadata. There is no
record connecting a detected face instance in an asset to that key, and the
processing path does not consume the profile. Calling this “subject-linked” in
the UI is therefore aspirational.

The first safe implementation should be manual:

- assign a stable asset ID and detected-face instance ID;
- let the user link one or more instances to a project-local subject key;
- show every proposed application before batch processing;
- recompute masks per target image;
- support unlink, delete, and rebuild; and
- keep cross-project reuse opt-in.

Evoto demonstrates the workflow value of syncing individual adjustments to the
same person across a project, but its automatic demographic tagging should not
be copied. Retouch's existing manual, project-local privacy boundary remains
the right one.

Source: [Evoto portrait retouching and individual sync](https://support.evoto.ai/portrait-retouching-feature-module/)

### W3 — current GUI success wording is inaccurate

`gui.py::on_watch_folder_process()` passes `lambda _path: None` to
`WatchFolder.process_pending()`. The callback returns successfully, so the
record becomes `done` and the UI reports it as “processed,” although no preview,
retouch, export, or job was created.

Before adding automation, the UI must either:

- label this action “ingest/stability scan” and leave stable files queued; or
- submit a real job and mark `done` only after a verifiable output exists.

Capture One's Next Capture workflow also shows why the source of inherited
settings must be explicit: defaults, last capture, primary image, clipboard,
or selected subsets are different user intents. Retouch should record that
choice in each job rather than infer it.

Sources: [Capture One Next Capture Adjustments](https://support.captureone.com/hc/en-us/articles/360002556677-Adding-adjustments-to-captured-images),
[Capture One AI masking and per-image recomputation](https://support.captureone.com/hc/en-us/articles/14055231933853-AI-Masking)

### W4 — status graph is not edit invalidation

`ProjectGraph` models dependencies and execution statuses, but not whether an
edit is current for a particular source/model/parameter state. A production
edit-status record needs, at minimum:

- source content hash and dimensions;
- pipeline and operation version;
- normalized parameter hash;
- model identifier plus verified model hash;
- mask type/version/hash;
- upstream node hashes;
- result hash and fallback/unavailable state; and
- a human-readable invalidation reason.

The user-visible states should remain compact: `current`, `needs_update`,
`unavailable`, `fallback`, and `failed`. “Update all” must execute in dependency
order. Lightroom's current AI Edit Status makes stale operations visible and
updates them top-down; this validates the original W4 direction.

Source: [Adobe Lightroom Manage AI Edits](https://helpx.adobe.com/sg/lightroom/web/edit-photos/manage-ai-edits.html)

### W5 — handoff has become more strategically important

Capture One 16.7.2 now exports AI, luma, brushed, and combined masks as
feather-preserving alpha channels in PSD/PSB. Retouch4me continues to expose a
neutral Soft Light dodge-and-burn layer. The competitive bar is no longer only
a flattened TIFF plus session JSON.

The lowest-risk Retouch v0 remains a folder/ZIP package:

- flattened 16-bit TIFF or PNG;
- versioned session/edit log;
- 16-bit grayscale semantic and manual masks;
- neutral-gray or signed-delta D&B representation with a declared blend mode;
- source reference and hashes;
- machine-readable and human-readable manifests; and
- an automated recomposition tolerance test.

PSD/PSB alpha-channel export is a later interoperability adapter. It should not
block a transparent, testable handoff package.

Sources: [Capture One mask export](https://support.captureone.com/hc/en-us/articles/32398712990237-Export-masks-from-Capture-One-to-Photoshop),
[Retouch4me Dodge & Burn](https://retouch4.me/dodgeburn)

### W6 — Look Board is now functional, but not target-aware enough

`LookExtractor.extract_board()` measures each reference independently and
combines scalar engine parameters by normalized stored weight. The GUI saves
every reference with weight `1.0`, then applies the result to the Process state.
This corrects the old statement that Look Boards are storage-only.

The next slice should add:

- editable manual weights;
- target/reference similarity evidence rather than hidden similarity claims;
- separate toggles for normalization, light/contrast, color relations, and
  finish;
- an impact control that excludes normalization by default;
- a consistency lock for batch use; and
- evaluation showing lower target-to-target variation on mixed-light sets.

Capture One's Match Look uses multiple references, target-dependent weighting,
inspectable adjustment groups, a separate impact control, and a consistency
option. Those are useful acceptance signals for Retouch's editable-parameter
approach.

Source: [Capture One Match Look](https://support.captureone.com/hc/en-us/articles/22188770298269-Match-Look-Tool)

### T1 — APP11 passthrough must not be called preservation

`retouch/io.py` clearly describes its current behavior as raw JPEG APP11
passthrough rather than creation of a new signed claim. That boundary must stay
visible. Edited pixels no longer satisfy the original asset's hard binding, and
copying a source manifest into the edited JPEG does not create a valid derived
work chain. The current helper also handles one APP11 payload, whereas real
JPEG C2PA manifest stores may span multiple APP11 segments.

The correct integration is:

1. validate the input and retain its validation result;
2. treat the source as the parent ingredient;
3. build an edit-intent claim with `c2pa.opened` plus truthful edit actions;
4. declare which deterministic and model-backed operations actually ran;
5. sign with protected key material;
6. embed or deliberately emit a sidecar/remote manifest; and
7. validate the output independently after signing.

C2PA 2.3 is the current published specification. The CAI Python SDK can create,
sign, read, and validate manifests, but it requires Python 3.10+, while this
project's verified runtime is Python 3.9. That makes an optional `c2patool`
subprocess or separately packaged signing service a better first spike than a
hard in-process dependency. Production signing also needs a KMS/HSM or
subprocess signer policy; private keys should not live in a preset or manifest.

Sources: [C2PA 2.3 specification](https://spec.c2pa.org/specifications/specifications/2.3/index.html),
[CAI Python SDK](https://opensource.contentauthenticity.org/docs/c2pa-python/),
[CAI edit intent](https://opensource.contentauthenticity.org/docs/c2pa-python/docs/intents/),
[c2patool usage](https://github.com/contentauth/c2pa-rs/blob/main/cli/docs/usage.md)

## 4. Recommended next tranche

### P0 — correct workflow truth

1. Rename the current watch action to a stability/ingest scan or stop marking
   the no-op callback `done`.
2. Change “subject-linked project profile” wording until actual face-instance
   links exist.
3. Change C2PA-facing wording to “detected/copied raw manifest bytes; output
   claim not validated” wherever the feature is surfaced.

### P1 — Shoot Review Manifest v1

Add a versioned project artifact with:

- stable asset IDs based on project-relative path plus content hash;
- burst groups and grouping evidence;
- versioned per-image and per-face quality measurements;
- `select`, `reject`, and `hold` decisions with `automatic` or `human` origin;
- manual override history that survives rescoring;
- explicit `uncertain` values instead of forced pass/fail;
- ratings/labels and a JSON/CSV selection export; and
- no deletion operation.

Initial per-face evidence may include face coverage, detector confidence,
face-local sharpness, left/right eye sharpness, and eyes-open
`yes/no/uncertain`. Expression quality, attractiveness, and identity scoring
remain out of scope.

### P2 — connect ingest to jobs

- Choose the source of settings explicitly: defaults, selected recipe/style,
  last approved image, nominated hero, or Look Board.
- Queue preview and final jobs separately.
- Recompute masks for every image.
- Require an output path and hash before a watch record becomes `done`.
- Keep failed jobs retryable and do not mutate source captures.

### P3 — manual subject application

- Add asset-face-instance records and user-confirmed links.
- Preview profile application per target face.
- Apply only supported per-face parameters; reject image-global keys.
- Recompute masks and protected-mark coordinates per image.
- Add unlink/delete/rebuild and cross-project export/import controls.

### P4 — handoff package v0

Implement the folder/ZIP contract and recomposition test before attempting a
PSD writer. Keep every operation that cannot be represented independently
declared as flattened-only.

### Separate spike — valid C2PA derived-work export

Prototype against `c2patool` first, using test certificates and independent
verification. Do not ship until signer custody, trust-chain behavior, offline
failure modes, and supported formats are decided.

## 5. Acceptance evidence required

### Culling

- consented real burst corpus with varied light, shallow depth of field,
  occlusions, glasses, and multiple faces;
- top-1 hero agreement and false-reject rate;
- per-signal calibration and explicit uncertainty rate;
- manual decisions preserved after rescoring; and
- no file deletion or source modification.

### Watch/project processing

- partial files, filename reuse, restart, duplicate content, moved files, and
  callback failures tested;
- every `done` record points to an existing verified output;
- target-specific masks are recomputed; and
- settings provenance is recorded per job.

### Profiles and Look Boards

- no silent cross-person merge;
- protected marks do not transfer to another person;
- user can inspect and remove every link;
- mixed-light evaluation beats a single arbitrary reference; and
- generated output remains an editable parameter set, not only a flattened LUT.

### Handoff and C2PA

- package recomposition meets a documented numeric tolerance;
- masks preserve size, bit depth, feathering, and identity labels;
- signed output validates in both the chosen SDK/tool and an independent
  verifier; and
- source provenance failures remain visible rather than being silently
  replaced by a new “valid” badge.

## 6. Verification performed for this delta

Command:

```bash
RETOUCH_CACHE_DIR=/private/tmp/retouch-research-cache \
python3 -m pytest \
  tests/test_shoot_intelligence.py \
  tests/test_watch_folder.py \
  tests/test_project_profiles.py \
  tests/test_look_extractor.py \
  tests/test_gui_wiring.py \
  -q -p no:cacheprovider
```

Result: **40 passed** on Python 3.9.6. The first collection attempt without the
cache override was blocked by sandbox write access to
`/Users/dennis/.cache/retouch/styles`; it was not a product-code failure.

This proves the focused primitives and GUI handlers under their current test
contracts. It does not prove real-shoot culling quality, face-aware profile
application, actual watch-folder processing, handoff recomposition, or valid
C2PA signing.

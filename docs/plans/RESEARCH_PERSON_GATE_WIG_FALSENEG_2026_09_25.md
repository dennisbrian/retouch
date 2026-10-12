# Person-gate false-negative on bright wig / blown-highlight backgrounds — 2026-09-25

## Status: reported, not fixed. No code changed.

## Summary

`539d3fb` (2026-09-23, "drop main-pass detections that are not on a person")
added a person-mask coverage gate (`_MAIN_PERSON_GATE_COVERAGE = 0.5`,
`retouch/detection.py::_person_gate`) to every detection path. It was
validated on the 2026-09-19/20 cosplay corpus at 0 real faces dropped out of
439 images.

On the 2026-09-13 cosplay shoot (`~/Pictures/2026/2026-09-13/15h`, 47 RAF →
JPEG, same subject/venue style, silver-white wig, backlit by a bright window),
the same gate drops **9 of 47 real, in-focus, forward-facing subject faces as
false positives (19%)** — confirmed by re-rendering the affected frames on
`539d3fb^` (immediately pre-gate), where every one of them detects and
retouches correctly.

## Root cause (visually confirmed, not just inferred from coverage numbers)

Dumped `FaceDetector.segment_person()`'s raw mask for DSCF3773 (a dropped
frame): the selfie segmenter assigns **near-zero person-probability to the
subject's head/hair region** while confidently segmenting the torso/costume
below it. See mask dump — solid black where the head/silver wig sits against
a blown-white window background, bright white for the torso against the
darker costume-store interior behind it.

Hypothesis: the segmenter's person/background boundary is confused when a
near-white subject region (platinum/silver wig, pale skin) sits against a
near-white blown-highlight background (backlit window) — insufficient
luminance contrast for the segmentation model at the head, even though a
human viewer reads the face instantly. This is a different failure mode from
the bokeh/sky/wall/costume-fabric background FPs the gate was built to catch
(RESEARCH_POSTERFP_VETO_2026_08_19, this commit's own validation set) — those
were background regions wrongly scored as person-shaped; this is the
opposite, a real person region scored as not-person.

The gate's own threshold comment (`retouch/detection.py:570-577`) records
"lowest real face 0.575" from the 2026-09-19/20 validation set — comfortably
above the 0.5 cutoff. On this shoot the dropped faces measure **coverage =
0.000–0.235**, i.e. not a borderline miscalibration, a near-total masking
failure at the face.

## Evidence

Logged gate output (`RETOUCH_MEDIAPIPE_BACKEND=legacy`, `.venv`, current
`main` @ `6ef691d`):

```
DSCF3773: bbox=(1189,1580,1066,1093) person coverage 0.000 < 0.50
DSCF3786: bbox=(1201,1611,984,1056)  person coverage 0.000 < 0.50
DSCF3809: bbox=(1030,1757,1008,1122) person coverage 0.000 < 0.50
DSCF3774: (same pattern)             person coverage 0.000 < 0.50
DSCF3799: bbox=(2255,2103,1092,1064) person coverage 0.235 < 0.50
```

Full list of affected files this shoot (`.retouch-review` records, 47-file
scan, `--max-dim 2048`, `cosplay_clear_protected_v1`): DSCF3773, 3774, 3783,
3786, 3788, 3792, 3794, 3799, 3809 — 9/47.

Discriminating test: DSCF3773, DSCF3786, DSCF3809 each re-rendered on
`539d3fb^` (`ba64bae`) — all three produce a visibly retouched face (smoothed
skin, brightened catchlights), confirming the face IS detectable pre-gate and
the gate is what drops it, not a raw MediaPipe miss. DSCF3809 (camera
rotated ~90°) also recovers pre-gate, so this isn't purely an orientation
issue either.

## What this is NOT

- Not the poster-FP problem (`RESEARCH_POSTERFP_VETO_2026_08_19`) — that's
  background wrongly flagged as a person; this is a person wrongly flagged
  as background.
- Not a MediaPipe detection failure — the face detector found these faces
  fine; `_person_gate` discarded them afterward.
- Not (apparently) pure backlighting — DSCF3800, processed successfully in
  the same lighting/venue in the same session, has a normal (non-white) hair
  color in the frame; the wig's near-white tone against the blown window may
  be the specific trigger, not backlight alone. Untested — see Open
  Questions.

## Workaround used for this delivery (2026-09-25 batch)

The 9 affected files were rendered from `539d3fb^` in a separate detached
worktree; the other 38 used current `main`. This is a one-off per-photo
routing decision for this shoot's delivery, not a code fix — `main` is
unchanged by this investigation.

## Open questions / next steps (not started)

1. Does this reproduce on other bright-wig or pale-skin subjects, or is it
   specific to this subject/venue/lighting combination? Only one shoot
   tested.
2. Is the trigger "near-white subject tone vs. near-white background" or
   something else (backlight silhouetting generally, this segmenter model's
   known weak point, cropped/tight framing)? Not isolated.
3. Candidate fix directions (none evaluated yet): a secondary
   face-detector-confidence-based override when segmentation coverage is low
   but MediaPipe's own face-presence confidence is high; relaxing the
   central-region inset so hairline/wig pixels don't dominate the sampled
   box; or a person-gate exemption when the face bbox itself (not just the
   inset) has any nonzero coverage. All unverified — do not implement
   without a fresh discriminating test per CLAUDE.md's evidence-first rule.
4. Needs its own branch + `tests/test_detection_person_gate.py` additions
   (mirroring the existing mutation-checked test style) before any change
   ships — out of scope for this note.

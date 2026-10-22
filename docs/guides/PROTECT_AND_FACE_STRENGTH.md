# Protect, Restore and Per-Face Strength

Three controls for when the retouch does something to a face that you did not
want, and a caption that says when the engine itself held back. All of them
are off by default: with no protect mask and every strength at 100%, output
is bit-identical to a render without them.

## Protect brush (GUI: "🛡️ Protect / Restore (keep this)")

Paint over anything the retouch must leave alone: a tear-mole, a beauty mark,
a freckle cluster, a tattoo, face paint detail, a scar the cosplayer wants
kept.

1. Render the photo once, then open the accordion and press **Load result**.
2. Paint with the green brush on the picture.
3. Press **Protect painted area**. The painted area is added to the mask and
   shown with a light green tint and a green outline (untick **Show mask** to
   hide it).
4. Render again (Preview or Export). Painted areas keep the photo's own
   detail.

Other buttons:

- **Restore painted area (re-render)**: same as Protect, then re-renders at
  once. Use it after the retouch removed something: paint it and it comes
  back in one step. The old tools could not do this in one step (the brush
  modes only add healing, and per-face recipes change the whole face).
- **Unprotect painted area**: erases the painted part of the mask.
- **Clear mask**: removes the whole mask. Loading another photo also clears
  it; a mask belongs to one photo.

What "protected" means: every per-face and body-skin op keeps the photo's
own detail inside the mask, while the broad tone of the face around it is
carried through, so a protected mole on a brightened face is brightened like
the skin around it instead of standing out as an untouched patch. The
whole-photo grade, tone curve, film look, sharpening and finish still apply
inside the mask, so it never looks pasted in.

Older name: `freckle_preserve_mask` (Python API) is still accepted and is
merged with the new mask.

### Which ops respect the mask

| Op group | How |
| --- | --- |
| Spot Healing, Blemish Removal, freckle and mark handling, Stray Hair Cleanup, Blown Highlight Repair, guided smoothing | Painted area excluded at detection, so no spot is found there |
| Every other per-face op (smoothing, whitening, evening, Face Polish, Powder Finish, Skin Warmth, shine, under-eye, eyes, lips, teeth, nose, makeup, reshaping texture) | Photo detail restored inside the mask at the end of the per-face core, broad tone carried |
| Glasses glare, red-eye | Same restore, around each op |
| Body skin window: Body Skin Evening and body retouch, body skin match (P7), Neck Tone Match, cosplay moat ops, body paint, prosthetic edges, Lint & Dust | Same restore over the whole window; P7 also abstains inside the mask |
| Fabric wrinkle smoothing | Same restore |
| Not protected (by design): global tone, colour grade, film, LUTs, sharpening and finish, Costume Shine, Costume Clarity, local adjustments | These are the look of the whole photo |

The mask is stored at the photo's own size and resized to whatever size a
render runs at (fast preview, draft proxy, full resolution, and the
multi-face worker pool all get the right crop).

## CLI: `--protect-mask`

```bash
./run batch shoot/ -o out/ --recipe cosplay_clear_v1 --protect-mask masks/
```

`--protect-mask` takes one image (used for every photo) or a folder holding
`<photo stem>.png` (also `.tif`, `.jpg`, `.webp`; a photo with no file there
renders without protection). White = protect. Any size with the same aspect
ratio as the photo works; it is resized. A mask with a different aspect ratio
is a different framing: that photo fails with an error naming both sizes,
rather than rendering unprotected. With `--edit-report`, the report gets a `protection` section
(share of the frame protected).

## Face retouch strength

- **Face retouch strength, all faces (%)** (Skin panel, `--face-strength`,
  0-150, default 100): scales every per-face amount (smoothing, whitening,
  eye and lip edits, reshaping, spot healing...) for all faces. 0 = face ops
  off, 150 = half again as strong (each amount is capped at its own maximum).
- **Strength for face (%)** (Per-face recipes accordion): detect faces, pick
  one, set its strength and press **Apply to face**. It multiplies the
  all-faces value and is stored with the face's position, so it stays on the
  same person if detection order changes.

Choices and protections are never scaled (which lip tint, mole protection,
the eye-visibility gate). The scaled list is checked against what the
per-face code actually reads (`tests/test_face_strength.py`), so a new face
op cannot be silently left out.

## "Skipped or dampened" caption

Under the preview, a short list says per face what the engine itself held
back and why, for example:

- `Face 0: eye edits skipped, right-hand eye: eye closed`
- `Face 0: eye edits dampened, left-hand eye: small iris and already saturated (coloured contacts), to 13%`
- `Face at 85% across, 5% down: face not on a person (background or poster), not retouched`

It covers the eye-visibility gate, coloured-contact damping, painted faces
(Skin Warmth and Neck Tone Match off), body skin match abstaining, faces the
person gate dropped or rescued, and your own per-face strength. The same
notes go to the edit report (`skipped`). Preview and export report the same
skips per face; the one exception is an eye whose openness sits right at the
gate's threshold, which is marked "(borderline, preview and export can
differ)".

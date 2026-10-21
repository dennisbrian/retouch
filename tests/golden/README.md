# Golden hard cases

Small (512-768 px) photos that each catch one way the pipeline has gone wrong
on real shoots. `tests/test_golden_qa.py` checks them in CI;
`scripts/qa/golden_qa.py` runs every curated recipe on every case.

Rebuild the inputs with `scripts/qa/build_golden_cases.py` (it downloads the
pinned base photos and checks their SHA-256). Customer photos are never
committed: where the real frame is a client's, the case is a simulated
version built on a public photo.

## Checks

For each case the harness renders twice with the same detected faces, once
with the op (or recipe) and once with every skin op off, grain off and the
body-skin ops off, and compares:

| Check | Limit | Notes |
|---|---|---|
| Faces retouched = people in the photo | exact | `persons` in `cases.json` |
| Changed pixels (> 3 levels) outside the skin mask grown by 4% of face width, plus the neck | < 2% of the skin area | Shadow Lift and Face Polish may also edit hair (by design); Skin Warmth spreads to matching body skin (by design) |
| Changed pixels on the hair mask | < 2% | same exemptions |
| Changed pixels on the planted object (`masks/`), eroded 1 px | < 2% | clip, rose, frames, poster |
| SSIM (luma) against `references/` | >= 0.97 | `natural`, `cosplay_clear_v1`, `studio_porcelain_clear_v1` |

Refresh the references only after looking at the renders:
`scripts/qa/golden_qa.py --approve --recipes natural,cosplay_clear_v1,studio_porcelain_clear_v1`.

## Cases

| Case | Real or simulated | What it catches |
|---|---|---|
| portrait_studio | real | baseline; red flag and curtains in the frame |
| hair_clip_temple | simulated | pink clip half on hair, half on skin (the albedo_even hair-clip frame) |
| red_prop_jaw | simulated | red rose at the jaw plus red cheek blotches (the redness_even leak case) |
| glasses_frames | simulated | dark frames on the skin |
| background_poster_face | simulated | framed portrait on the wall (the background false-positive face) |
| pet_false_face | real | cats and dogs, no person (**known gap**, xfail) |
| darker_exposure | simulated | 0.45x linear exposure |
| brighter_exposure | simulated | 1.5x linear exposure |
| painted_pale_face | real | powder-pale painted face, white wig |
| white_wig_clip | simulated | pink clip on a white wig |
| small_face_full_body | real | full-body beach shot, small face, darker skin |
| two_people | simulated | two subjects, one darker |

"Simulated" means the object, exposure or second person was planted on a
public photo; the original client frames are not in the repo.

## Known gaps

- **pet_false_face**: FaceMesh finds the dogs' faces and no offline model here
  (selfie, multiclass, pose) rejects them. The test is a strict xfail.
- **glasses_frames**: the skin mask is cut only for coloured objects; dark
  frames read as shadowed skin by colour, so 6-16% of the rim under the eyes
  still changes for some ops. Not in the CI op matrix.

## Sources

Base photos are MediaPipe test assets
(`storage.googleapis.com/mediapipe-assets/`): `portrait.jpg` (official US
government portrait, public domain), `mozart_square.jpg` (public-domain
painting), `pose.jpg`, `cats_and_dogs.jpg` (redistributed with the
Apache-2.0 MediaPipe project).

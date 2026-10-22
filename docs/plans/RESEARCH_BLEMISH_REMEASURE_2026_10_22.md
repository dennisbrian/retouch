# Blemish re-measure, identity scorecard and natural_v2 — 2026-10-22

Phase 0a and Phase 1 of [the face quality roadmap](PLAN_FACE_QUALITY_ROADMAP_2026_10_22.md).
Code: `tests/golden_identity.py` (scorecard), `scripts/qa/golden_qa.py` (identity columns),
`scripts/qa/blemish_remeasure.py`, `scripts/qa/identity_photos.py`, recipe `natural_v2`.
Every number below comes from a render made for this note on `main` @ `f67541e` plus
this branch; nothing is quoted from older studies.

## Photos

- **Golden cases** (`tests/golden/`): public photos at 512-768 px; most are simulated
  variants (planted objects, 0.45x / 1.5x exposure, a second person). Faces are ~70-170 px wide.
- **Real photos** (owner-supplied, used locally for measurement only, never committed):
  2 bunny cosplay photos (DSCF3503, DSCF3518, white face paint, 26 MP) and 6 finished
  third-party rednote cosplay photos (web size, ~1080x1900). At the 2000 px working size
  5 of the 8 have a face at least 120 px wide (139-279 px); the 3 night shrine/lantern
  shots don't and are left out of the numbers.
- **Simulated**: every spot, mole and freckle is planted; "darker" means the photo at
  0.45x linear light. No real acne, real darker-skin face or real dev set was available.
  Corpus v3 has no populated calibration split yet (roadmap §0b).

## 1. Identity scorecard (Phase 0a)

Per face, all relative to that face:

| Column | How | Flags when |
|---|---|---|
| Geometry drift | MediaPipe landmarks re-detected on the output; 13 feature distances (eye spacing, inner eye spacing, eye widths, nose width and length, mouth width, upper/lower lip, jaw, cheek, chin, face height) divided by the face's landmark spread (RMS of the 468 mesh points) | change > max(0.04, 3 x the face's own landmark jitter). Jitter = the same input re-detected after a 2 px shift + JPEG q92, a σ0.8 blur and a light sharpen. Faces with IED < 30 px are not measured (size gate). |
| Lightness | face-skin median L* in vs out, converted to relative luminance, in stops (a ratio: the same edit reads the same on any skin tone); `detect_color_drift`'s ΔE and chroma-floored hue shift next to it | \|shift\| > 0.15 stop |
| Mark survival | 4 moles + 4 freckles planted per face in linear light (transmittance), away from eyes/brows/lips; low-pass Lab residual of planted vs unplanted render, as a share of the residual in the input | any mark < 50% |
| AA6 | `qa_detectors.perceived_retouching_vector` on this face's skin vs the input: SSIM contrast-structure term and high-pass change D | report only |
| Leak / skipped | the existing golden leak, hair and object checks; people the recipe skipped | (existing checks) |

Report only: identity columns never fail `golden_qa.py`.

Proof (`tests/test_golden_identity.py`, real portrait and simulated 0.45x capture):
no-op render quiet on every column; an 8% horizontal face stretch flags geometry;
1.3x linear gain on face skin reads +0.38 stop (same within 0.03 stop on the darker capture)
and flags lightness; an output that lost its planted moles flags marks (and exactly one
lost mole is counted as one).

Calibration on the golden cases (`natural`, reshape 0): geometry max 0.79x tolerance, 0 flags.
With `slimming=60, eye_enlarge=40`: 8 of 11 measured faces flag (eye spacing, jaw).
`cosplay_clear_v1` (slimming 50): 8 geometry flags of 14 faces, marks 37/112 kept.

### Finding: `natural` darkens face skin on the golden cases

`natural` reads **-0.09 to -0.23 stop** (median -0.14) on face skin on the golden cases;
the face crop is visibly duller (looked at portrait_studio). Ablation on portrait_studio:
`smooth=0` gives -0.015 stop; whiten, blemish, equalize, micro_restore, contrast,
shadow lift, dodge & burn each move it by ≤ 0.025. So the smoothing path, not lightening,
moves tone most. On the real photos at 2000 px the same recipe reads -0.04 to -0.09
stop, so it is smaller on larger faces. Not fixed here (it changes `natural`).
The roadmap's gap 5 ("natural lightens skin") is true of `whiten` alone (+0.025 stop)
but the net effect of the recipe is darker.

## 2. Legacy blemish vs spot_heal (Phase 1.1)

`scripts/qa/blemish_remeasure.py`, recipe `natural` with everything else held, 2000 px.
Per face: 4 soft pimples (r 0.007 fw), 4 stronger inflamed pimples (r 0.010 fw, darker
transmittance) and 4 dark moles (r 0.008 fw), on skin away from features. 5 real faces
x 2 exposures = 10 face renders per setting.

| Setting | Pimples fixed | of which strong | Moles kept | Other skin changed |
|---|---|---|---|---|
| legacy blemish 30 | **0/80** | 0/40 | 40/40 | 0.08% |
| legacy blemish 55 | **0/80** | 0/40 | 40/40 | 0.10% |
| spot_heal 30 | **59/80** | 33/40 | 40/40 | 0.10% |
| spot_heal 55 | 59/80 | 33/40 | 40/40 | 0.10% |

By exposure (spot_heal 30): as shot 27/40, darker 32/40; moles 20/20 each.

- Fixed = residual ≤ 50% of the both-off render's; kept = ≥ 50%.
  Other skin = face-skin pixels > 3 radii from any planted spot that moved > 3 levels
  vs both off, over 149,156 px.
- **Legacy blemish touched none of the 80 planted spots** (median residual 1.00 at
  both strengths). On the unplanted rednote fox photo it changed 93 px: the drawn
  forehead marking, eye corners and the nostril edge (looked at the diff overlay).
  spot_heal changed 0 px on the same unplanted photo.
- Compared with 09-27 (strength 60, before `6b4dbd5`): legacy then erased 21/22 moles;
  now it keeps 40/40. Pimples are still 0.
- 30 and 55 give identical counts for both ops here: spots are either clearly above
  spot_heal's threshold or skipped for location (next to a nostril, lip or shading edge).
- Odd number: DSCF3518 as shot fixed 4/8 (residual 0.70) but 8/8 at 0.45x. That face is
  white paint; near the top of the range the planted spot is weaker in Lab and is
  skipped more. Not investigated further.
- "Other skin" for spot_heal reads 0.80% on the rednote fox photo as shot; on the
  unplanted photo spot_heal changes 0 px, so this is the heal feather and its knock-on
  through later stages around planted spots, not new healing.

Verdict: spot_heal still wins, so Step 3 went ahead.

## 3. natural_v2 (Phase 1.2-1.5)

- `blemish` now has a recipe path: `skin.blemish` (0-1). Without it the engine and GUI
  still mirror `frequency.smooth`; all 150 recipes resolve exactly as before
  (`tests/test_natural_v2.py`). No settings count changed (`blemish` was already a key).
- `natural_v2` = `natural` + `skin.blemish 0`, `skin.spot_heal 30`,
  `mark_policy protect_identity`, `skin.mole_protect 0.80`, `skin.face_shadow_fill 0.50`.
  `skin.rosy` stays 0.10 (lightening is an owner decision). Curated, experimental,
  not recommended; `natural` is unchanged.

### Extras tried one at a time (scorecard, on top of natural_v2)

Golden cases (9 faces >= 120 px) and the 5 real photos as shot + 0.45x (10 faces):

| Variant | Real: geometry flags (max) | Real: marks kept (moles) | Real: AA6 SSIM / D | Golden: marks | Golden: lightness flags |
|---|---|---|---|---|---|
| natural | 0 (0.86x) | 63/80 (25/40) | 0.973 / +0.121 | 63/72 | 3 |
| natural_v2 base | 0 (0.88x) | 65/80 (32/40) | 0.975 / +0.104 | 61/72 | 2 |
| + face_shadow_fill 50 | 0 (0.93x) | 65/80 (32/40) | 0.972 / +0.093 | 62/72 | 2 |
| + nose_shape 100 | **1** (1.06x) | 66/80 (32/40) | 0.979 / +0.094 | 61/72 | 2 |
| + powder_finish 20 | **1** (1.14x) | **49/80** (20/40) | 0.969 / +0.234 | 61/72 | **6** |

Kept: face shadow fill (no flag count got worse; AA6 SSIM dips 0.003 because it lifts
shade, D improves). Dropped: keep nose shape (a new geometry flag on a real face) and
powder finish (lost 16 of 80 marks, more lightness flags, more high-frequency loss).

### natural vs natural_v2 (final)

- Golden sweep (`golden_qa.py --recipes natural,natural_v2`, 12 cases): both pass every
  leak/object/face-count check; identity: geometry 0 / 0, lightness flags 5 / 3, marks
  102/112 vs 100/112 (natural_v2 heals 2 planted freckles on the small, 137 px golden faces).
- Real photos at 2000 px, rednote faces (as shot + 0.45x, 6 faces): marks 35/48 → 38/48,
  moles 12/24 → 19/24, geometry 0 flags both, lightness unchanged (median -0.062 stop).
- Bunny photos at full resolution (26 MP, as shot, 2 faces): both keep 12/16 marks and
  8/8 moles; geometry 0 flags (max 0.57x / 0.73x); lightness -0.046 / -0.028 stop. On
  white face paint the two recipes look nearly the same (crops viewed at native pixels).
- Real drawn mark: on the night shrine photo (487730b4, a 100-px face, not in the
  numbers above) `natural` erases the drawn tear-mole under the eye and `natural_v2`
  keeps it (zoomed crop viewed). This gives that sheet away, so look at the sheets before
  reading this line.
- Not changed by natural_v2: the -0.14 stop darkening on small faces (smoothing), and
  freckles: spot_heal treats some planted light-brown freckles as spots.

### Blind A/B

Face crops of `natural` and `natural_v2` side by side in a per-face random order (a
coin from a hash of the photo name) are in the project files
(`retouch/face-quality/ab/`), with `key.json` naming each side. 7 faces: the 2 bunny
photos rendered at full resolution and cropped at native pixels, and 5 rednote faces
at their web size (one rednote photo has no detected face). They are
real photos, unretouched by us except for the two recipes; nothing is planted on them.

## Open

- The owner's blind pick between `natural` and `natural_v2` (roadmap §5.1), and skin
  lightening (§5.2).
- A real dev set with darker skin and real acne (roadmap §0b): every spot here is planted.
- Why legacy blemish never fires on planted spots, and the smoothing path's darkening on
  small faces.
